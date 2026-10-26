/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Final host-link device image. Rlibs retain ordinary PTX or required compiler
//! IR alongside anchored cubins; no host archive extraction or runtime JIT.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use cuda_artifact_finalizer::{DebugPolicy, FinalizationOptions, FinalizerOutput, NamedInput};
use cuda_artifact_finalizer::ptx_merge::merge_ptx_bundles;
use oxide_artifacts::{
    ArtifactBundleSpec, ArtifactCompileOptions, ArtifactDebugPolicy, ArtifactEntrySpec,
    ArtifactPayloadKind, ArtifactPayloadSpec, OwnedArtifactBundle, OwnedArtifactPayload,
};
use reserved_oxide_symbols::LINKED_CUBIN_BUNDLE_NAME;

use crate::materialize::{MaterializationRequest, checked_finalizer, checked_ptx_assembler};

type Result<T, E = Box<dyn std::error::Error>> = std::result::Result<T, E>;

const LINK_INPUT_BUNDLE_PREFIX: &str = "__cuda_oxide_link_input:";
pub(crate) const LINK_INPUT_MODULE_PREFIX: &str = "oxide_artifact_link_input_";

/// Only the cubin object has host anchors. The separate PTX/IR object is archived
/// for subsequent device links, never extracted by ordinary host linking.
pub(crate) fn build_artifact_objects<'a>(
    mut runtime: ArtifactBundleSpec<'a>,
    link_input: Option<ArtifactPayloadSpec<'a>>,
    host_target: &str,
    target_anchor: &'a str,
    legacy_anchor: &str,
) -> Result<(Vec<u8>, Option<Vec<u8>>)> {
    if link_input.is_some() {
        // This identity distinguishes a package's library, binaries and tests,
        // which intentionally share the user-facing runtime bundle name.
        runtime.payloads[0].name = target_anchor;
    }
    let runtime_blob = oxide_artifacts::build_artifact_blob(&runtime)?;
    let runtime_object = oxide_artifacts::build_host_object_for_target_with_legacy_anchor(
        &runtime_blob, host_target, target_anchor, legacy_anchor,
    )?;
    let input_object = if let Some(input) = link_input {
        let name = format!("{LINK_INPUT_BUNDLE_PREFIX}{target_anchor}");
        let mut spec = ArtifactBundleSpec::new(&name, runtime.target)
            .with_compile_options(runtime.compile_options)
            .with_payload(input);
        spec.entries = runtime.entries;
        let blob = oxide_artifacts::build_artifact_blob(&spec)?;
        use object::Object as _;
        // Match the already-validated runtime object's host format. SHF_EXCLUDE
        // also protects mixed rlib+shared-library output invocations and host
        // whole-archive links: PTX/IR survives archiving but not image linking.
        let host = object::File::parse(runtime_object.as_slice())?;
        let mut object = object::write::Object::new(
            host.format(), host.architecture(), host.endianness(),
        );
        let section = object.add_section(
            Vec::new(), oxide_artifacts::ARTIFACT_SECTION_NAME.as_bytes().to_vec(),
            object::SectionKind::Data,
        );
        object.section_mut(section).set_data(blob, 8);
        object.section_mut(section).flags = object::SectionFlags::Elf {
            sh_flags: object::elf::SHF_EXCLUDE as u64,
        };
        Some(object.write()?)
    } else {
        None
    };
    Ok((runtime_object, input_object))
}

/// Pair every runtime cubin with its compile-only input before discarding the
/// runtime bundles. Names alone are insufficient: also require identical entry
/// metadata, target and compiler policy. Stale combined-payload bundles and
/// ordinary, non-AOT inputs are rejected rather than silently linked.
fn select_link_inputs(bundles: Vec<OwnedArtifactBundle>) -> Result<Vec<OwnedArtifactBundle>> {
    let mut inputs = BTreeMap::new();
    let mut runtime = Vec::new();
    for bundle in bundles {
        if bundle.name == LINKED_CUBIN_BUNDLE_NAME {
            continue;
        }
        if let Some(identity) = bundle.name.strip_prefix(LINK_INPUT_BUNDLE_PREFIX) {
            if bundle.payloads.len() != 1
                || !is_link_input(bundle.payloads[0].kind)
            {
                return Err(format!("invalid compile-only AOT bundle {}", bundle.name).into());
            }
            let identity = identity.to_owned();
            if let Some(previous) = inputs.get(&identity) {
                if previous != &bundle {
                    return Err(format!("conflicting AOT link inputs for {identity}").into());
                }
            } else {
                inputs.insert(identity, bundle);
            }
        } else {
            if bundle.payloads.len() != 1 || bundle.payloads[0].kind != ArtifactPayloadKind::Cubin {
                return Err(format!(
                    "AOT bundle {} is not a runtime-only cubin; rebuild all device dependencies with --materialize-cubin",
                    bundle.name,
                ).into());
            }
            runtime.push(bundle);
        }
    }
    let mut paired = BTreeSet::new();
    for bundle in &runtime {
        let identity = &bundle.payloads[0].name;
        let input = inputs.get(identity).ok_or_else(|| format!(
            "AOT cubin {} has no compile-only PTX/IR input for {identity}; rebuild the dependency with --materialize-cubin",
            bundle.name,
        ))?;
        if bundle.target != input.target || bundle.compile_options != input.compile_options
            || bundle.entries != input.entries
        {
            return Err(format!("AOT cubin/input metadata mismatch for {identity}").into());
        }
        paired.insert(identity.as_str());
    }
    if let Some(identity) = inputs.keys().find(|identity| !paired.contains(identity.as_str())) {
        return Err(format!("compile-only AOT input {identity} has no matching runtime cubin").into());
    }
    Ok(inputs.into_values().collect())
}

/// Read all artifact-bearing members, not just those the host linker extracts.
/// Rust metadata and embedded LLVM bitcode are not object files and are skipped.
fn bundles_from_bytes(bytes: &[u8]) -> Result<Vec<OwnedArtifactBundle>> {
    if bytes.starts_with(b"!<arch>\n") {
        let archive = object::read::archive::ArchiveFile::parse(bytes)?;
        let mut bundles = Vec::new();
        for member in archive.members() {
            let member = member?;
            let data = member.data(bytes)?;
            if matches!(
                object::FileKind::parse(data),
                Ok(object::FileKind::Elf32 | object::FileKind::Elf64
                    | object::FileKind::MachO32 | object::FileKind::MachO64
                    | object::FileKind::Coff | object::FileKind::CoffBig)
            ) {
                bundles.extend(oxide_artifacts::read_artifact_bundles_from_object_bytes(data)?);
            }
        }
        Ok(bundles)
    } else {
        Ok(oxide_artifacts::read_artifact_bundles_from_object_bytes(bytes)?)
    }
}

pub(crate) fn gather(paths: impl IntoIterator<Item = PathBuf>) -> Result<Vec<OwnedArtifactBundle>> {
    let mut seen = BTreeSet::new();
    let mut bundles = Vec::new();
    for path in paths {
        if seen.insert(path.clone()) {
            let bytes = std::fs::read(&path)?;
            bundles.extend(bundles_from_bytes(&bytes).map_err(|error| {
                format!("cannot read CUDA artifacts from {}: {error}", path.display())
            })?);
        }
    }
    select_link_inputs(bundles)
}

fn is_link_input(kind: ArtifactPayloadKind) -> bool {
    matches!(kind, ArtifactPayloadKind::Ptx | ArtifactPayloadKind::NvvmIr | ArtifactPayloadKind::Ltoir)
}

fn link_policy(bundles: &[OwnedArtifactBundle]) -> Result<Option<(&str, ArtifactCompileOptions)>> {
    let Some(first) = bundles.first() else {
        return Ok(None);
    };
    for bundle in bundles {
        if bundle.target != first.target || bundle.compile_options != first.compile_options {
            return Err(format!(
                "incompatible AOT device link policies: {} ({}, {:?}) and {} ({}, {:?})",
                first.name, first.target, first.compile_options,
                bundle.name, bundle.target, bundle.compile_options,
            ).into());
        }
        if bundle.payloads.len() != 1 || !is_link_input(bundle.payloads[0].kind) {
            return Err(format!(
                "AOT device bundle {} must retain exactly one PTX, NVVM IR or LTOIR input; rebuild all device dependencies with --materialize-cubin",
                bundle.name,
            ).into());
        }
    }
    Ok(Some((&first.target, first.compile_options)))
}

/// Construct the one runtime-facing generic image, with a union of entry
/// metadata. Shared PTX merge semantics or NVVM weak-ODR semantics coalesce
/// duplicate definitions without dropping modules with unique specializations.
fn linked_blob(bundles: &[OwnedArtifactBundle], cubin: &[u8]) -> Result<Vec<u8>> {
    let (target, policy) = link_policy(bundles)?.ok_or("empty AOT device link")?;
    let mut entries = BTreeMap::new();
    for bundle in bundles {
        for entry in &bundle.entries {
            if let Some(previous) = entries.insert(entry.symbol.as_str(), entry)
                && previous != entry
            {
                return Err(format!("conflicting CUDA entry metadata for {}", entry.symbol).into());
            }
        }
    }
    let mut spec = ArtifactBundleSpec::new(LINKED_CUBIN_BUNDLE_NAME, target)
        .with_compile_options(policy)
        .with_payload(ArtifactPayloadSpec::new(
            ArtifactPayloadKind::Cubin,
            "__cuda_oxide_linked.cubin",
            cubin,
        ));
    for entry in entries.into_values() {
        spec = spec.with_entry(ArtifactEntrySpec {
            symbol: &entry.symbol,
            kind: entry.kind,
            metadata: entry.metadata,
        });
    }
    Ok(oxide_artifacts::build_artifact_blob(&spec)?)
}

pub(crate) fn materialize(
    request: MaterializationRequest,
    bundles: &[OwnedArtifactBundle],
) -> Result<Option<Vec<u8>>> {
    let Some((target, policy)) = link_policy(bundles)? else {
        return Ok(None);
    };
    let debug = match policy.debug_policy() {
        ArtifactDebugPolicy::None => DebugPolicy::None,
        ArtifactDebugPolicy::LineTables => DebugPolicy::LineTables,
        ArtifactDebugPolicy::Full => DebugPolicy::Full,
    };
    let options = FinalizationOptions::new(target.parse()?)
        .with_fma_contraction(policy.fma_contraction_enabled())
        .with_debug_policy(debug);
    let has_ptx = bundles.iter().any(|bundle| bundle.payload(ArtifactPayloadKind::Ptx).is_some());
    let cubin = if has_ptx {
        // The runtime merger intentionally skips non-PTX bundles. Convert every
        // required IR input explicitly before calling it, so mixed dependency
        // graphs cannot silently lose their IR-only kernels.
        let ir_ptx = if bundles.iter().any(|bundle| bundle.payload(ArtifactPayloadKind::Ptx).is_none()) {
            Some(OwnedArtifactBundle {
                name: "__cuda_oxide_linked_ir".to_owned(),
                target: target.to_owned(),
                compile_options: policy,
                entries: Vec::new(),
                payloads: vec![OwnedArtifactPayload {
                    kind: ArtifactPayloadKind::Ptx,
                    name: "__cuda_oxide_linked_ir.ptx".to_owned(),
                    bytes: link_ir(request, bundles, &options, FinalizerOutput::Ptx)?,
                }],
            })
        } else {
            None
        };
        let merged = merge_ptx_bundles(
            bundles.iter().filter(|bundle| bundle.payload(ArtifactPayloadKind::Ptx).is_some())
                .chain(ir_ptx.iter()),
        )?;
        checked_ptx_assembler(request)?.assemble_ptx(
            NamedInput::new("__cuda_oxide_linked.ptx", merged.as_bytes()), &options,
        )?
    } else {
        link_ir(request, bundles, &options, FinalizerOutput::Cubin)?
    };
    Ok(Some(linked_blob(bundles, &cubin)?))
}

/// Compile each required NVVM module separately to preserve its metadata
/// namespace, then link all IR inputs together with NVVM weak-ODR semantics.
fn link_ir(
    request: MaterializationRequest,
    bundles: &[OwnedArtifactBundle],
    options: &FinalizationOptions,
    output: FinalizerOutput,
) -> Result<Vec<u8>> {
    let finalizer = checked_finalizer(request)?;
    let compiled: Vec<_> = bundles.iter().enumerate()
        .filter(|(_, bundle)| bundle.payload(ArtifactPayloadKind::Ptx).is_none())
        .map(|(index, bundle)| -> Result<_> {
            let bytes = if let Some(ltoir) = bundle.payload(ArtifactPayloadKind::Ltoir) {
                Cow::Borrowed(ltoir)
            } else {
                Cow::Owned(finalizer.compiler().compile_nvvm_ir_to_ltoir(
                    &bundle.name,
                    bundle.payload(ArtifactPayloadKind::NvvmIr).ok_or_else(|| {
                        format!("AOT device bundle {} has no retained IR input", bundle.name)
                    })?,
                    options,
                )?)
            };
            Ok((format!("{index}-{}.ltoir", bundle.name), bytes))
        })
        .collect::<Result<_>>()?;
    let inputs: Vec<_> = compiled.iter()
        .map(|(name, bytes)| NamedInput::new(name, bytes))
        .collect();
    Ok(finalizer.link_ltoir(&inputs, options, output)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxide_artifacts::{ArtifactEntryKind, build_artifact_blob, parse_artifact_blob};

    fn bundle(name: &str, entries: &[&str]) -> OwnedArtifactBundle {
        let mut spec = ArtifactBundleSpec::new(name, "sm_80")
            .with_payload(ArtifactPayloadSpec::new(ArtifactPayloadKind::NvvmIr, "crate.ll", b"ir"));
        for entry in entries {
            spec = spec.with_entry(ArtifactEntrySpec::new(entry, ArtifactEntryKind::Kernel));
        }
        parse_artifact_blob(&build_artifact_blob(&spec).unwrap()).unwrap().into()
    }

    fn archive_member(archive: &mut Vec<u8>, name: &str, bytes: &[u8]) {
        let header = format!("{name:<16}{:<12}{:<6}{:<6}{:<8}{:<10}`\n", 0, 0, 0, "100644", bytes.len());
        assert_eq!(header.len(), 60);
        archive.extend_from_slice(header.as_bytes());
        archive.extend_from_slice(bytes);
        if bytes.len() % 2 != 0 {
            archive.push(b'\n');
        }
    }

    fn paired_objects(anchor: &str, kind: ArtifactPayloadKind) -> (Vec<u8>, Vec<u8>) {
        let spec = ArtifactBundleSpec::new("dependency", "sm_80")
            .with_payload(ArtifactPayloadSpec::new(ArtifactPayloadKind::Cubin, "crate.cubin", b"cubin"))
            .with_entry(ArtifactEntrySpec::new("generic_u32", ArtifactEntryKind::Kernel));
        let (runtime, input) = build_artifact_objects(
            spec,
            Some(ArtifactPayloadSpec::new(kind, "crate.input", b"input")),
            "x86_64-unknown-linux-gnu", anchor, "legacy",
        ).unwrap();
        (runtime, input.unwrap())
    }

    #[test]
    fn runtime_cubin_and_compile_only_inputs_are_separate_archive_members() {
        use object::{Object, ObjectSection, ObjectSymbol};
        for kind in [ArtifactPayloadKind::Ptx, ArtifactPayloadKind::NvvmIr, ArtifactPayloadKind::Ltoir] {
            let (runtime, input) = paired_objects("target_anchor", kind);
            let runtime_bundles = bundles_from_bytes(&runtime).unwrap();
            assert_eq!(runtime_bundles[0].name, "dependency");
            assert_eq!(runtime_bundles[0].payloads.len(), 1);
            assert_eq!(runtime_bundles[0].payloads[0].kind, ArtifactPayloadKind::Cubin);
            let runtime_file = object::File::parse(runtime.as_slice()).unwrap();
            assert!(runtime_file.symbols().any(|symbol| symbol.name().unwrap() == "target_anchor"));

            let input_file = object::File::parse(input.as_slice()).unwrap();
            assert!(!input_file.symbols().any(|symbol| symbol.is_global() && symbol.is_definition()));
            let section = input_file.section_by_name(oxide_artifacts::ARTIFACT_SECTION_NAME).unwrap();
            assert!(matches!(section.flags(), object::SectionFlags::Elf { sh_flags }
                if sh_flags & object::elf::SHF_EXCLUDE as u64 != 0));

            let mut archive = b"!<arch>\n".to_vec();
            archive_member(&mut archive, "lib.rmeta/", b"metadata");
            archive_member(&mut archive, "runtime.o/", &runtime);
            archive_member(&mut archive, "input.o/", &input);
            let found = select_link_inputs(bundles_from_bytes(&archive).unwrap()).unwrap();
            assert_eq!(found.len(), 1);
            assert_eq!(found[0].payloads.len(), 1);
            assert_eq!(found[0].payload(kind), Some(b"input".as_slice()));
            let bytes = linked_blob(&found, b"linked cubin").unwrap();
            let linked = parse_artifact_blob(&bytes).unwrap();
            assert_eq!(linked.name, LINKED_CUBIN_BUNDLE_NAME);
            assert!(linked.entry("generic_u32").is_some());
            assert_eq!(linked.payloads.len(), 1);
            assert_eq!(linked.payloads[0].kind, ArtifactPayloadKind::Cubin);
        }
    }

    #[test]
    fn pairing_rejects_missing_mismatched_and_unmaterialized_inputs() {
        let (runtime, input) = paired_objects("target_anchor", ArtifactPayloadKind::Ptx);
        let mut paired = bundles_from_bytes(&runtime).unwrap();
        assert!(select_link_inputs(paired.clone()).is_err());
        assert!(select_link_inputs(bundles_from_bytes(&input).unwrap()).is_err());
        paired.extend(bundles_from_bytes(&input).unwrap());
        assert!(select_link_inputs(paired.clone()).is_ok());
        paired[1].entries.clear();
        assert!(select_link_inputs(paired).is_err());
        assert!(select_link_inputs(vec![bundle("ordinary_nvvm", &[])]).is_err());

        // Same package runtime name does not pair different Cargo targets.
        let (other_runtime, _) = paired_objects("other_target", ArtifactPayloadKind::Ptx);
        let mut mismatched = bundles_from_bytes(&other_runtime).unwrap();
        mismatched.extend(bundles_from_bytes(&input).unwrap());
        assert!(select_link_inputs(mismatched).is_err());
    }

    #[test]
    fn final_bundle_unions_local_and_dependency_entries_and_contains_only_cubin() {
        let bundles = [bundle("dependency", &["shared", "generic_u32"]), bundle("consumer", &["shared", "generic_u64"])];
        let bytes = linked_blob(&bundles, b"linked cubin").unwrap();
        let linked = parse_artifact_blob(&bytes).unwrap();
        assert_eq!(linked.name, LINKED_CUBIN_BUNDLE_NAME);
        assert_eq!(linked.payloads.len(), 1);
        assert_eq!(linked.payload(ArtifactPayloadKind::Cubin), Some(b"linked cubin".as_slice()));
        assert_eq!(linked.entries.iter().map(|entry| entry.symbol).collect::<Vec<_>>(), ["generic_u32", "generic_u64", "shared"]);
    }

    #[test]
    fn mixed_retained_inputs_are_all_paired_and_included_in_final_metadata() {
        let mut artifacts = Vec::new();
        for (anchor, kind) in [
            ("ptx_target", ArtifactPayloadKind::Ptx),
            ("nvvm_target", ArtifactPayloadKind::NvvmIr),
            ("ltoir_target", ArtifactPayloadKind::Ltoir),
        ] {
            let (runtime, input) = paired_objects(anchor, kind);
            artifacts.extend(bundles_from_bytes(&runtime).unwrap());
            artifacts.extend(bundles_from_bytes(&input).unwrap());
        }
        let inputs = select_link_inputs(artifacts).unwrap();
        assert_eq!(inputs.len(), 3);
        assert!(link_policy(&inputs).unwrap().is_some());
        for kind in [ArtifactPayloadKind::Ptx, ArtifactPayloadKind::NvvmIr, ArtifactPayloadKind::Ltoir] {
            assert_eq!(inputs.iter().filter(|bundle| bundle.payload(kind).is_some()).count(), 1);
        }
        let blob = linked_blob(&inputs, b"linked cubin").unwrap();
        let linked = parse_artifact_blob(&blob).unwrap();
        assert_eq!(linked.payloads.len(), 1);
        assert_eq!(linked.payloads[0].kind, ArtifactPayloadKind::Cubin);
        assert!(linked.entry("generic_u32").is_some());
    }

    #[test]
    fn retained_ptx_uses_shared_merger_for_generic_specializations() {
        let mut bundles = [
            bundle("dependency", &["generic_u32"]),
            bundle("consumer", &["generic_u64"]),
        ];
        for (bundle, entry) in bundles.iter_mut().zip(["generic_u32", "generic_u64"]) {
            bundle.payloads[0].kind = ArtifactPayloadKind::Ptx;
            bundle.payloads[0].bytes = format!(
                ".version 8.0\n.target sm_80\n.address_size 64\n\
                 .visible .func helper() {{ ret; }}\n.visible .entry {entry}() {{ ret; }}\n"
            ).into_bytes();
        }
        assert!(link_policy(&bundles).unwrap().is_some());
        let merged = merge_ptx_bundles(&bundles).unwrap();
        assert_eq!(merged.matches(".func helper()").count(), 1);
        assert!(merged.contains(".entry generic_u32()"));
        assert!(merged.contains(".entry generic_u64()"));
        let blob = linked_blob(&bundles, b"linked cubin").unwrap();
        let linked = parse_artifact_blob(&blob).unwrap();
        assert!(linked.entry("generic_u32").is_some());
        assert!(linked.entry("generic_u64").is_some());
        assert!(linked.payload(ArtifactPayloadKind::Ptx).is_none());
    }

    #[test]
    fn ambiguous_and_unsupported_retained_payloads_are_rejected() {
        let mut input = bundle("ambiguous", &[]);
        input.payloads.push(OwnedArtifactPayload {
            kind: ArtifactPayloadKind::Ptx,
            name: "crate.ptx".into(),
            bytes: b"ptx".to_vec(),
        });
        assert!(link_policy(std::slice::from_ref(&input)).is_err());
        input.name = format!("{LINK_INPUT_BUNDLE_PREFIX}target");
        assert!(select_link_inputs(vec![input]).is_err());
        let (runtime, input) = paired_objects("target", ArtifactPayloadKind::Cubin);
        let mut artifacts = bundles_from_bytes(&runtime).unwrap();
        artifacts.extend(bundles_from_bytes(&input).unwrap());
        assert!(select_link_inputs(artifacts).is_err());
    }

    #[test]
    fn link_policy_is_empty_or_strict_before_loading_cuda_tools() {
        assert!(link_policy(&[]).unwrap().is_none());
        let mut bundles = [bundle("dependency", &[]), bundle("consumer", &[])];
        assert!(link_policy(&bundles).unwrap().is_some());
        bundles[1].target = "sm_90".into();
        assert!(link_policy(&bundles).is_err());
        bundles[1].target = "sm_80".into();
        bundles[1].compile_options = ArtifactCompileOptions::new().with_fma_contraction(false);
        assert!(link_policy(&bundles).is_err());
        bundles[1].compile_options = ArtifactCompileOptions::new();
        bundles[1].payloads.retain(|payload| payload.kind == ArtifactPayloadKind::Cubin);
        assert!(link_policy(&bundles).is_err());
    }
}
