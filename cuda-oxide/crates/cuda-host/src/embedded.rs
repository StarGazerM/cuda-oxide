/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Host-side loading for embedded device artifact bundles.

use crate::ltoir;
pub use cuda_core::embedded::{
    ArtifactCompileOptions, ArtifactPayloadKind, EmbeddedModule, OwnedArtifactBundle,
    artifact_bundles_from_binary_path, artifact_bundles_from_current_exe,
    embedded_modules_from_current_exe,
};
use cuda_core::{CudaContext, CudaModule, DriverError};
use cuda_artifact_finalizer::ptx_merge::{PtxMergeError, merge_ptx_bundles};
use std::sync::Arc;
use thiserror::Error;

#[cfg(any(target_os = "linux", target_os = "android"))]
mod mapped_image;

/// Errors while discovering, building, or loading an embedded CUDA module.
#[derive(Debug, Error)]
pub enum EmbeddedModuleError {
    /// The artifact anchor could not be associated with its original mapped file.
    #[error("cannot read the binary containing the CUDA artifact anchor: {0}")]
    MappedImage(#[source] std::io::Error),

    /// Reading the embedded artifact section failed.
    #[error(transparent)]
    Core(#[from] cuda_core::EmbeddedModuleError),

    /// The named bundle was not present in the binary that was read.
    #[error("embedded CUDA module '{name}' was not found")]
    ModuleNotFound { name: String },

    /// No embedded bundles with loadable payloads were found.
    #[error("no embedded CUDA modules were found")]
    NoModules,

    /// A bundle existed, but it contained no supported payload.
    #[error("embedded CUDA module '{name}' has no supported payload")]
    UnsupportedPayload { name: String },

    /// Retained PTX bundles could not be merged safely.
    #[error(transparent)]
    PtxMerge(#[from] PtxMergeError),

    /// NVVM IR or LTOIR payload compilation failed.
    #[error("failed to build embedded CUDA module: {0}")]
    Ltoir(#[from] ltoir::LtoirError),

    /// The CUDA driver rejected the selected module image.
    #[error("failed to load embedded CUDA module: {0}")]
    Driver(#[from] DriverError),
}

/// Load a named embedded artifact bundle from the current executable.
///
/// Cubin and PTX payloads are loaded directly with the CUDA driver. NVVM IR and
/// LTOIR payloads are linked to an in-memory cubin for their original target.
/// A payload built for a standard pre-Blackwell target, such as `sm_86`, may
/// instead be converted to PTX and JIT-compiled by the driver on Blackwell.
pub fn load_embedded_module(
    ctx: &Arc<CudaContext>,
    name: &str,
) -> Result<Arc<CudaModule>, EmbeddedModuleError> {
    let bundle = artifact_bundles_from_current_exe()?
        .into_iter()
        .find(|bundle| bundle.name == name)
        .ok_or_else(|| EmbeddedModuleError::ModuleNotFound {
            name: name.to_string(),
        })?;
    load_bundle(ctx, &bundle)
}

/// Load a named artifact bundle from the binary containing `anchor`.
///
/// Generated non-generic module loaders borrow their artifact anchor, which
/// the linker places in the same image as the bundle. This works for both an
/// executable and a shared library, including a library opened by a relative
/// path before the working directory changes.
///
/// On Linux and Android, this requires readable `/proc/self/maps` and
/// `/proc/self/map_files`. The original mapped file must still be accessible.
/// A missing, replaced, deleted or unreadable image is an error; it never
/// causes a search in another binary. Other operating systems are unsupported.
/// Payload selection and compilation are identical to [`load_embedded_module`].
pub fn load_embedded_module_from_anchor(
    ctx: &Arc<CudaContext>,
    name: &str,
    anchor: &u8,
) -> Result<Arc<CudaModule>, EmbeddedModuleError> {
    let bundle = artifact_bundles_containing(anchor)?
        .into_iter()
        .find(|bundle| bundle.name == name)
        .ok_or_else(|| EmbeddedModuleError::ModuleNotFound {
            name: name.to_string(),
        })?;
    load_bundle(ctx, &bundle)
}

fn artifact_bundles_containing(
    anchor: &u8,
) -> Result<Vec<OwnedArtifactBundle>, EmbeddedModuleError> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let bytes = mapped_image::read(anchor).map_err(EmbeddedModuleError::MappedImage)?;
        oxide_artifacts::read_artifact_bundles_from_object_bytes(&bytes)
            .map_err(cuda_core::EmbeddedModuleError::Artifacts)
            .map_err(Into::into)
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        let _ = anchor;
        Err(EmbeddedModuleError::MappedImage(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "artifact anchor discovery requires Linux or Android procfs",
        )))
    }
}

/// Load the cross-crate kernel set, preferring its ahead-of-time linked cubin.
///
/// With build-time cubin materialization, the backend links all local and
/// dependency device artifacts into one native cubin at the final host link.
/// Generic kernel specializations are therefore available even when emitted in
/// a consuming crate. Loading this image does not invoke the driver PTX JIT.
///
/// Without materialization, this merges every PTX bundle in the executable:
/// repeated headers and identical device-function definitions are removed;
/// conflicting same-named definitions are rejected. Other payloads are skipped.
/// The historical function name describes this non-materialized path.
pub fn load_all_ptx_bundles_merged(
    ctx: &Arc<CudaContext>,
) -> Result<Arc<CudaModule>, EmbeddedModuleError> {
    // Embedded bundles are immutable for the life of the process. Share the
    // linked or merged module per context instead of loading it per program.
    thread_local! {
        static MERGED: std::cell::RefCell<Vec<(usize, Arc<CudaModule>)>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }
    let key = ctx.cu_ctx() as usize;
    if let Some(module) = MERGED.with(|cache| {
        cache.borrow().iter().find(|(k, _)| *k == key).map(|(_, m)| m.clone())
    }) {
        return Ok(module);
    }
    let module = load_all_ptx_bundles_merged_uncached(ctx)?;
    MERGED.with(|cache| cache.borrow_mut().push((key, module.clone())));
    Ok(module)
}

fn load_all_ptx_bundles_merged_uncached(
    ctx: &Arc<CudaContext>,
) -> Result<Arc<CudaModule>, EmbeddedModuleError> {
    let bundles = artifact_bundles_from_current_exe()?;
    if let Some(cubin) = linked_cubin(&bundles)? {
        return Ok(ctx.load_module_from_image(cubin)?);
    }
    let merged = merge_ptx_bundles(&bundles)?;

    let module = ctx.load_module_from_image(merged.as_bytes())?;
    // Retain the merged module's `.entry` names (a few dozen bytes per
    // kernel). If a later `_TID_` generic-kernel lookup misses while a
    // same-base entry exists under a different hash, the launch paths can
    // then report a host/device type-identity naming divergence instead of an
    // opaque "named symbol not found". See `crate::entry_registry`.
    crate::entry_registry::register_merged_module_entries(&module, &merged);
    Ok(module)
}

fn linked_cubin(bundles: &[OwnedArtifactBundle]) -> Result<Option<&[u8]>, EmbeddedModuleError> {
    let Some(bundle) = bundles
        .iter()
        .find(|bundle| bundle.name == reserved_oxide_symbols::LINKED_CUBIN_BUNDLE_NAME)
    else {
        return Ok(None);
    };
    bundle
        .payload(ArtifactPayloadKind::Cubin)
        .map(Some)
        .ok_or_else(|| EmbeddedModuleError::UnsupportedPayload {
            name: bundle.name.clone(),
        })
}

/// Load the first embedded artifact bundle with a supported payload.
pub fn load_first_embedded_module(
    ctx: &Arc<CudaContext>,
) -> Result<Arc<CudaModule>, EmbeddedModuleError> {
    for bundle in artifact_bundles_from_current_exe()? {
        match load_bundle(ctx, &bundle) {
            Ok(module) => return Ok(module),
            Err(EmbeddedModuleError::UnsupportedPayload { .. }) => continue,
            Err(error) => return Err(error),
        }
    }
    Err(EmbeddedModuleError::NoModules)
}

fn load_bundle(
    ctx: &Arc<CudaContext>,
    bundle: &OwnedArtifactBundle,
) -> Result<Arc<CudaModule>, EmbeddedModuleError> {
    if let Some(cubin) = bundle.payload(ArtifactPayloadKind::Cubin) {
        return Ok(ctx.load_module_from_image(cubin)?);
    }

    if let Some(ptx) = bundle.payload(ArtifactPayloadKind::Ptx) {
        return Ok(ctx.load_module_from_image(ptx)?);
    }

    if let Some(nvvm_ir) = bundle.payload(ArtifactPayloadKind::NvvmIr) {
        let emitted = target_arch_for_bundle(bundle)?;
        let execution = ltoir::execution_arch_for_context(ctx)?;
        let image = match ltoir::execution_route(&emitted, &execution)? {
            ltoir::ExecutionRoute::Cubin => ltoir::build_cubin_from_nvvm_ir_with_compile_options(
                nvvm_ir,
                &bundle.name,
                &emitted.sm(),
                bundle.compile_options,
            )?,
            ltoir::ExecutionRoute::PtxBridge => ltoir::build_ptx_from_nvvm_ir_with_compile_options(
                nvvm_ir,
                &bundle.name,
                &emitted.sm(),
                bundle.compile_options,
            )?,
        };
        return Ok(ctx.load_module_from_image(&image)?);
    }

    if let Some(ltoir) = bundle.payload(ArtifactPayloadKind::Ltoir) {
        let emitted = target_arch_for_bundle(bundle)?;
        let execution = ltoir::execution_arch_for_context(ctx)?;
        let image = match ltoir::execution_route(&emitted, &execution)? {
            ltoir::ExecutionRoute::Cubin => ltoir::link_ltoir_to_cubin_with_compile_options(
                ltoir,
                &bundle.name,
                &emitted.sm(),
                bundle.compile_options,
            )?,
            ltoir::ExecutionRoute::PtxBridge => ltoir::link_ltoir_to_ptx_with_compile_options(
                ltoir,
                &bundle.name,
                &emitted.sm(),
                bundle.compile_options,
            )?,
        };
        return Ok(ctx.load_module_from_image(&image)?);
    }

    Err(EmbeddedModuleError::UnsupportedPayload {
        name: bundle.name.clone(),
    })
}

fn target_arch_for_bundle(
    bundle: &OwnedArtifactBundle,
) -> Result<cuda_artifact_finalizer::CudaArch, ltoir::LtoirError> {
    let explicit = std::env::var("CUDA_OXIDE_TARGET").ok();
    target_arch_for_bundle_with_explicit(bundle, explicit.as_deref())
}

fn target_arch_for_bundle_with_explicit(
    bundle: &OwnedArtifactBundle,
    explicit_target: Option<&str>,
) -> Result<cuda_artifact_finalizer::CudaArch, ltoir::LtoirError> {
    ltoir::resolve_source_target(concrete_bundle_target(&bundle.target)?, explicit_target)
}

fn concrete_bundle_target(
    target: &str,
) -> Result<Option<cuda_artifact_finalizer::CudaArch>, ltoir::LtoirError> {
    match target {
        // Compatibility for artifacts emitted before concrete NVVM targets
        // were recorded. New bundles must never use these sentinels.
        "libdevice" | "nvvm-ir" => Ok(None),
        target => Ok(Some(target.parse()?)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bundle_with_target(target: &str) -> OwnedArtifactBundle {
        OwnedArtifactBundle {
            name: "demo".to_string(),
            target: target.to_string(),
            compile_options: ArtifactCompileOptions::new(),
            payloads: Vec::new(),
            entries: Vec::new(),
        }
    }

    #[test]
    fn target_arch_uses_bundle_sm_target() {
        assert_eq!(
            concrete_bundle_target("sm_90")
                .unwrap()
                .map(|target| target.sm()),
            Some("sm_90".to_string())
        );
    }

    fn ptx_bundle(name: &str, ptx: &str) -> OwnedArtifactBundle {
        use oxide_artifacts::OwnedArtifactPayload;
        OwnedArtifactBundle {
            name: name.to_string(),
            target: "sm_80".to_string(),
            compile_options: ArtifactCompileOptions::new(),
            payloads: vec![OwnedArtifactPayload {
                kind: ArtifactPayloadKind::Ptx,
                name: name.to_string(),
                bytes: ptx.as_bytes().to_vec(),
            }],
            entries: Vec::new(),
        }
    }

    #[test]
    fn linked_cubin_wins_over_ptx_without_runtime_compilation() {
        use oxide_artifacts::OwnedArtifactPayload;
        let ptx = ptx_bundle("ordinary", "not valid PTX");
        let mut linked = bundle_with_target("sm_89");
        linked.name = reserved_oxide_symbols::LINKED_CUBIN_BUNDLE_NAME.to_string();
        linked.payloads.push(OwnedArtifactPayload {
            kind: ArtifactPayloadKind::Cubin,
            name: "linked.cubin".to_string(),
            bytes: b"\x7fELF".to_vec(),
        });
        assert_eq!(
            linked_cubin(&[ptx, linked]).unwrap(),
            Some(b"\x7fELF".as_slice())
        );
    }

    #[test]
    fn malformed_linked_bundle_never_falls_back_to_ptx() {
        let ptx = ptx_bundle("ordinary", ".version 8.9\n");
        assert!(linked_cubin(std::slice::from_ref(&ptx)).unwrap().is_none());
        let malformed = ptx_bundle(reserved_oxide_symbols::LINKED_CUBIN_BUNDLE_NAME, ".version 8.9\n");
        assert!(matches!(
            linked_cubin(&[ptx, malformed]),
            Err(EmbeddedModuleError::UnsupportedPayload { .. })
        ));
    }

    #[test]
    fn target_arch_uses_bundle_compute_target() {
        assert_eq!(
            concrete_bundle_target("compute_90")
                .unwrap()
                .map(|target| target.sm()),
            Some("sm_90".to_string())
        );
    }

    #[test]
    fn target_arch_falls_back_for_non_arch_target() {
        // Older bundles used these names instead of recording a concrete
        // architecture. New bundles always record a validated `sm_*` target.
        for legacy in ["libdevice", "nvvm-ir"] {
            assert_eq!(concrete_bundle_target(legacy).unwrap(), None);
        }
    }

    #[test]
    fn legacy_bundle_without_recorded_target_requires_explicit_target() {
        for sentinel in ["libdevice", "nvvm-ir"] {
            let bundle = bundle_with_target(sentinel);
            let error = target_arch_for_bundle_with_explicit(&bundle, None)
                .expect_err("the original build target is required");
            assert!(matches!(error, ltoir::LtoirError::TargetNotFound));

            let asserted =
                target_arch_for_bundle_with_explicit(&bundle, Some("compute_86")).unwrap();
            assert_eq!(asserted.sm(), "sm_86");
        }
    }

    #[test]
    fn recorded_bundle_target_overrides_explicit_environment_target() {
        let bundle = bundle_with_target("sm_90");
        let selected = target_arch_for_bundle_with_explicit(&bundle, Some("sm_120")).unwrap();
        assert_eq!(selected.sm(), "sm_90");
    }

    #[test]
    fn target_arch_rejects_malformed_bundle_target() {
        assert!(concrete_bundle_target("sm_90x").is_err());
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn bundles_of_an_executable_anchor_match_the_current_exe() {
        static ANCHOR: u8 = 0;
        let by_anchor = artifact_bundles_containing(&ANCHOR).unwrap();
        let by_current_exe = artifact_bundles_from_current_exe().unwrap();
        assert_eq!(by_anchor, by_current_exe);
    }
}
