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
    // A shared-library input can contain its own completed image. Relink its
    // original per-crate inputs instead, never nest a previously linked cubin.
    bundles.retain(|bundle| bundle.name != LINKED_CUBIN_BUNDLE_NAME);
    Ok(bundles)
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
            .with_payload(ArtifactPayloadSpec::new(ArtifactPayloadKind::Cubin, "crate.cubin", b"cubin"))
            .with_payload(ArtifactPayloadSpec::new(ArtifactPayloadKind::NvvmIr, "crate.ll", b"ir"));
        for entry in entries {
            spec = spec.with_entry(ArtifactEntrySpec::new(entry, ArtifactEntryKind::Kernel));
        }
        parse_artifact_blob(&build_artifact_blob(&spec).unwrap()).unwrap().into()
    }

    fn blob(bundle: &OwnedArtifactBundle) -> Vec<u8> {
        let mut spec = ArtifactBundleSpec::new(&bundle.name, &bundle.target);
        for payload in &bundle.payloads {
            spec = spec.with_payload(ArtifactPayloadSpec::new(payload.kind, &payload.name, &payload.bytes));
        }
        build_artifact_blob(&spec).unwrap()
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

    #[test]
    fn gathers_artifact_objects_inside_rlibs_without_host_anchor_extraction() {
        let bundle = bundle("dependency", &["generic_u32"]);
        let object = oxide_artifacts::build_host_object_for_target(
            &blob(&bundle), "x86_64-unknown-linux-gnu", None,
        ).unwrap();
        let mut archive = b"!<arch>\n".to_vec();
        archive_member(&mut archive, "lib.rmeta/", b"metadata");
        archive_member(&mut archive, "device.o/", &object);
        let found = bundles_from_bytes(&archive).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name, "dependency");
        assert_eq!(found[0].payload(ArtifactPayloadKind::NvvmIr), Some(b"ir".as_slice()));
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
