# cuda-artifact-finalizer

Driver-independent finalization of NVVM IR, LTOIR, and PTX into loadable device
code.

This crate is the single owner of cuda-oxide's libNVVM, nvJitLink, and `ptxas`
compilation policy. It deliberately does **not** link the CUDA Driver, so the
same typed target, FMA, debug, input-order, validation, and provenance rules
apply whether an artifact is materialized at build time (`cargo oxide build
--materialize-cubin`) or finalized at run time as a fallback.

`PtxAssembler` is discovered separately from the ordinary `Finalizer`. It
finds the toolkit `ptxas` executable, pins and fingerprints it, and assembles
already-linked PTX without loading libNVVM, nvJitLink, or the CUDA Driver.
Set `CUDA_OXIDE_PTXAS` to select an explicit executable; otherwise toolkit
roots and `PATH` are searched.

`ptx_merge::merge_ptx_bundles` owns the PTX merge implementation used both by
`cuda-host`'s runtime generic loader and the backend's final AOT device link.
It preserves bundle order, namespaces anonymous shared allocations, coalesces
identical callable definitions, rejects conflicting definitions, and remaps
debug file indices while removing colliding module-local debug sections.
The pure merger skips non-PTX bundles; callers that require every input must
convert or reject them explicitly.

AOT retains the ordinary codegen output as a separate, unanchored compile-only
archive member marked `SHF_EXCLUDE`; runtime-facing bundles contain only cubin.
At final host link, PTX-only inputs use the shared merger and provenance-checked
`PtxAssembler`, retaining the recorded FMA and debug policy. All-IR input sets
keep the NVVM/LTOIR link route. Mixed sets link their required IR inputs to PTX
first, then merge those definitions with the retained ordinary PTX and assemble
one cubin. Unsupported or ambiguous retained payloads are rejected rather than
silently omitted. No PTX payload is loaded at runtime in the AOT path.

Keeping that policy in one driverless crate is what lets the two paths agree.
A rule that lived in the runtime loader alone could not be applied during a
build, and one duplicated across both would drift.

Build-time materialization passes a versioned `MaterializerHandshakeV1` from
`cargo-oxide` to the codegen backend. Its named fields bind each content digest
to a retained-file identity, so child processes can avoid rereading large CUDA
DSOs while the content-derived combined digest remains Cargo's semantic
fingerprint. Identity mismatches fall back to hashing the newly opened file.

`cargo-oxide` caches the handshake it discovers at
`.oxide-artifacts/materializer-handshake/v1.json` under the workspace root, so
subsequent builds skip rehashing the CUDA DSOs. The cache is self-validating:
a stale or corrupt file is ignored and rediscovered, and deleting it simply
forces a full rehash on the next build.

Consumers:

- `cargo-oxide` and `rustc-codegen-cuda`, for build-time materialization;
- `cuda-host`, for the runtime path.
