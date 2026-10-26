/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Final host-link device image. Rlibs retain compiler IR alongside their
//! anchored cubins; no dependency on host archive extraction or runtime JIT.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use cuda_artifact_finalizer::{DebugPolicy, FinalizationOptions, FinalizerOutput, NamedInput};
use oxide_artifacts::{
    ArtifactBundleSpec, ArtifactCompileOptions, ArtifactDebugPolicy, ArtifactEntrySpec,
    ArtifactPayloadKind, ArtifactPayloadSpec, OwnedArtifactBundle,
};
use reserved_oxide_symbols::LINKED_CUBIN_BUNDLE_NAME;

use crate::materialize::{MaterializationRequest, checked_finalizer};

type Result<T, E = Box<dyn std::error::Error>> = std::result::Result<T, E>;

const LINK_INPUT_BUNDLE_PREFIX: &str = "__cuda_oxide_link_input:";
pub(crate) const LINK_INPUT_MODULE_PREFIX: &str = "oxide_artifact_link_input_";

/// Only the cubin object has host anchors. The separate IR object is archived
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
        // whole-archive links: IR survives archiving but not image linking.
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
/// ordinary, non-AOT IR are rejected rather than silently linked.
fn select_link_inputs(bundles: Vec<OwnedArtifactBundle>) -> Result<Vec<OwnedArtifactBundle>> {
    let mut inputs = BTreeMap::new();
    let mut runtime = Vec::new();
    for bundle in bundles {
        if bundle.name == LINKED_CUBIN_BUNDLE_NAME {
            continue;
        }
        if let Some(identity) = bundle.name.strip_prefix(LINK_INPUT_BUNDLE_PREFIX) {
            if bundle.payloads.len() != 1
                || !matches!(bundle.payloads[0].kind, ArtifactPayloadKind::NvvmIr | ArtifactPayloadKind::Ltoir)
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
            "AOT cubin {} has no compile-only IR input for {identity}; rebuild the dependency with --materialize-cubin",
            bundle.name,
        ))?;
        if bundle.target != input.target || bundle.compile_options != input.compile_options
            || bundle.entries != input.entries
        {
            return Err(format!("AOT cubin/IR metadata mismatch for {identity}").into());
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
        if bundle.payload(ArtifactPayloadKind::Ltoir).is_none()
            && bundle.payload(ArtifactPayloadKind::NvvmIr).is_none()
        {
            return Err(format!(
                "AOT device bundle {} has no retained NVVM IR or LTOIR; rebuild all device dependencies with --materialize-cubin",
                bundle.name,
            ).into());
        }
    }
    Ok(Some((&first.target, first.compile_options)))
}

/// Construct the one runtime-facing generic image, with a union of entry
/// metadata. Duplicate definitions are coalesced by NVVM weak-ODR semantics,
/// not by dropping entire modules (which could lose unique specializations).
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
    let finalizer = checked_finalizer(request)?;
    // Compile each retained NVVM module separately, preserving its metadata
    // namespace; nvJitLink, rather than textual concatenation, links the result.
    let mut compiled = Vec::with_capacity(bundles.len());
    for bundle in bundles {
        compiled.push(if bundle.payload(ArtifactPayloadKind::Ltoir).is_some() {
            None
        } else {
            Some(finalizer.compiler().compile_nvvm_ir_to_ltoir(
                &bundle.name,
                bundle.payload(ArtifactPayloadKind::NvvmIr).expect("validated IR input"),
                &options,
            )?)
        });
    }
    let names: Vec<_> = bundles.iter().enumerate()
        .map(|(index, bundle)| format!("{index}-{}.ltoir", bundle.name))
        .collect();
    let inputs: Vec<_> = bundles.iter().zip(&compiled).zip(&names)
        .map(|((bundle, compiled), name)| NamedInput::new(
            name,
            compiled.as_deref().unwrap_or_else(|| {
                bundle.payload(ArtifactPayloadKind::Ltoir).expect("validated IR input")
            }),
        ))
        .collect();
    let cubin = finalizer.link_ltoir(&inputs, &options, FinalizerOutput::Cubin)?;
    Ok(Some(linked_blob(bundles, &cubin)?))
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

    fn paired_objects(anchor: &str) -> (Vec<u8>, Vec<u8>) {
        let spec = ArtifactBundleSpec::new("dependency", "sm_80")
            .with_payload(ArtifactPayloadSpec::new(ArtifactPayloadKind::Cubin, "crate.cubin", b"cubin"))
            .with_entry(ArtifactEntrySpec::new("generic_u32", ArtifactEntryKind::Kernel));
        let (runtime, input) = build_artifact_objects(
            spec,
            Some(ArtifactPayloadSpec::new(ArtifactPayloadKind::NvvmIr, "crate.ll", b"ir")),
            "x86_64-unknown-linux-gnu", anchor, "legacy",
        ).unwrap();
        (runtime, input.unwrap())
    }

    #[test]
    fn runtime_cubin_and_compile_only_ir_are_separate_archive_members() {
        use object::{Object, ObjectSection, ObjectSymbol};
        let (runtime, input) = paired_objects("target_anchor");
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
        assert_eq!(found[0].payload(ArtifactPayloadKind::NvvmIr), Some(b"ir".as_slice()));
        let bytes = linked_blob(&found, b"linked cubin").unwrap();
        let linked = parse_artifact_blob(&bytes).unwrap();
        assert_eq!(linked.name, LINKED_CUBIN_BUNDLE_NAME);
        assert!(linked.entry("generic_u32").is_some());
        assert!(linked.payload(ArtifactPayloadKind::NvvmIr).is_none());
    }

    #[test]
    fn pairing_rejects_missing_mismatched_and_unmaterialized_inputs() {
        let (runtime, input) = paired_objects("target_anchor");
        let mut paired = bundles_from_bytes(&runtime).unwrap();
        assert!(select_link_inputs(paired.clone()).is_err());
        assert!(select_link_inputs(bundles_from_bytes(&input).unwrap()).is_err());
        paired.extend(bundles_from_bytes(&input).unwrap());
        assert!(select_link_inputs(paired.clone()).is_ok());
        paired[1].entries.clear();
        assert!(select_link_inputs(paired).is_err());
        assert!(select_link_inputs(vec![bundle("ordinary_nvvm", &[])]).is_err());

        // Same package runtime name does not pair different Cargo targets.
        let (other_runtime, _) = paired_objects("other_target");
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
