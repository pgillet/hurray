# Hurray

**Hurray** is a language-agnostic, zero-copy runtime interchange format for multi-dimensional tensor data, optimized for the memory layout diversity, quantization schemes, and access patterns of modern AI/ML inference pipelines.

Think Apache Arrow, but for tensors.

## Current Phase

The project is in the **specification and requirements phase**. Work is focused on `docs/spec/`, `docs/impl/`, and `docs/prior-art.md`. Rust implementation work is deferred until the spec is stable. Do not produce implementation code unless explicitly asked.

## Project Structure

```
hurray/
├── CLAUDE.md                   # This file
├── TODO.md                     # Running list of ideas and future tasks (reviewed periodically)
├── Cargo.toml                  # Workspace root
├── docs/
│   ├── prior-art.md            # Research snapshot: formats, protocols, libraries
│   ├── spec/                   # Format specification (source of truth)
│   │   ├── README.md           # Scope, goals, RFC 2119 notice, versioning
│   │   ├── data-model.md       # Shape/dimension model
│   │   ├── element-types.md    # Element type system (int, float, quantized, custom)
│   │   ├── quantization.md     # Quantization schemes: per-tensor, per-channel, per-block
│   │   ├── memory-layout.md    # Layout index and overview
│   │   ├── layouts/            # Per-layout spec files
│   │   │   ├── row-major.md
│   │   │   ├── column-major.md
│   │   │   ├── strided.md
│   │   │   ├── tiled.md
│   │   │   ├── morton.md
│   │   │   ├── hilbert.md
│   │   │   ├── subpaving.md
│   │   │   ├── coo.md          # Sparse: Coordinate list
│   │   │   ├── csr.md          # Sparse: Compressed Sparse Row
│   │   │   └── csc.md          # Sparse: Compressed Sparse Column
│   │   ├── buffer-protocol.md  # Zero-copy semantics, alignment, device memory
│   │   ├── metadata.md         # Tensor descriptor binary encoding
│   │   ├── interchange.md      # Runtime interchange: in-process, IPC, cross-machine
│   │   ├── versioning.md       # Format version field, compatibility policy
│   │   └── references.md       # Normative references
│   ├── impl/                   # Implementation requirements (not the spec itself)
│   │   ├── README.md           # Overview of implementation requirement docs
│   │   ├── compliance.md       # Compliance checklist for implementors
│   │   ├── rust-reference.md   # Rust reference implementation guide
│   │   ├── c-ffi.md            # C FFI implementation guide
│   │   └── python-bindings.md  # Python bindings implementation guide
│   └── adr/                    # Architecture Decision Records
│       └── ADR-NNN-*.md
├── hurray-core/                # Core types, no I/O, no async
├── hurray-io/                  # Async I/O: streaming + file format (tokio)
├── hurray-ffi/                 # C ABI layer for language bindings
├── hurray-python/              # Python bindings (PyO3)
└── hurray-inspect/             # CLI hex viewer for Hurray descriptor files
```

## Guiding Principles

- **Spec is the source of truth.** The Rust implementation follows the spec. When they conflict, fix the implementation, not the spec.
- **Zero-copy first.** Data must be shareable across runtimes without copying whenever possible.
- **Streamable.** Both readers and writers must be able to process tensor data incrementally. A reader must be able to start processing without buffering the entire input; a writer must be able to emit tensors one at a time without buffering the entire output. Tensor descriptors always precede their data buffers; the format is self-delimiting; back-references and end-of-file indexes are forbidden.
- **Language-agnostic.** No Rust-isms leak into the format design or the C FFI boundary.
- **Correctness before performance.** This is a reference implementation. Optimize only when explicitly asked.
- **Inference-optimized.** Layout diversity, quantization, and device memory are first-class concerns.

## Crate Responsibilities

| Crate | Responsibility | Key dependencies |
|-------|---------------|-----------------|
| `hurray-core` | Format types, tensor descriptor, buffer handle, quantization descriptors. No I/O, no async. | `thiserror`, `serde` (feature-gated), `half`, `rayon` |
| `hurray-io` | Async streaming and file format read/write. | `hurray-core`, `tokio`, `bytes` |
| `hurray-ffi` | C ABI: opaque handles, function table, buffer release callbacks. No panics across FFI. | `hurray-core` |
| `hurray-python` | Python bindings with NumPy/PyTorch zero-copy interop via `__dlpack__`. | `hurray-ffi` or `hurray-core`, `pyo3` |

## Key Technical Decisions

- **Endianness**: little-endian throughout (all multi-byte fields)
- **Alignment**: minimum 64-byte buffer alignment (SIMD); page-aligned for GPU/IPC
- **Strides**: expressed in logical elements, not bytes; negative and zero strides are valid
- **Sub-byte packing**: `int4`/`bool` packing order defined in spec (see `memory-layout.md`)
- **Error handling**: crate-level `Error` enum via `thiserror`; no `unwrap()`/`expect()` in library code
- **Unsafe**: isolated to dedicated modules with `// SAFETY:` comments on every block

## Agent Roles

| Agent | Owns |
|-------|------|
| `researcher` | State-of-the-art surveys, prior art analysis, hardware constraint research. Runs before major design decisions. |
| `architect` | Design decisions, trade-off analysis, ADRs. Consumes researcher output. |
| `format-spec-writer` | All files under `docs/spec/`. Resolves ambiguities reported by implementation agents. |
| `rust-developer` | All files under `hurray-*/src/`. Implements what the spec defines. Does not write tests. |
| `rust-test-writer` | All files under `hurray-*/tests/` and `#[cfg(test)]` modules. Tests the public API, not internals. |
| `rust-reviewer` | Reviews implementation and tests for correctness, idioms, and spec fidelity. |
| `rust-build-resolver` | Resolves `cargo check` / `cargo build` failures. |
| `performance-optimizer` | Profiling and optimization passes. Only invoked explicitly. |
| `refactor-cleaner` | Code cleanup and refactoring. Only invoked explicitly. |
| `doc-updater` | Keeps `///` doc comments and `docs/` in sync with implementation changes. |
| `planner` | Breaks features into concrete, phased implementation steps. |

## Development Workflow

```
researcher          (surveys prior art)
    ↓
architect           (makes design decision, writes ADR)
    ↓
format-spec-writer  (writes normative spec section)
    ↓
rust-developer      (implements against the spec)
    ↓
rust-test-writer    (writes tests against the public API)
    ↓
rust-reviewer       (reviews both)
    ↓
rust-build-resolver (fixes any compile errors)
```

## Spec Writing Conventions

- Normative language follows RFC 2119: `MUST`, `MUST NOT`, `SHOULD`, `SHOULD NOT`, `MAY`
- Non-normative content is prefixed: `> **Note (non-normative):**`
- Open questions are marked inline: `> **[OQ-N]:** ...`
- All types use language-agnostic names: `int32`, `uint64`, `utf8 string` — never `i32`, `usize`, `String`
- All byte examples use hex literals: `0x00`, `0xFF`

## Rust Conventions

- No `unwrap()` or `expect()` in library code — propagate with `?`
- All public items have `///` doc comments with at least one example
- `unsafe` code is isolated in dedicated modules; every block has a `// SAFETY:` comment
- `cargo clippy -- -D warnings` must pass before any code is considered complete
- Feature flags: `serde` for serialization support, `tokio` for async I/O, `python` for bindings
- Do not mix `rayon` thread pool calls directly in async contexts — use `tokio::task::spawn_blocking`

## Prior Art

Key references the research and architecture agents are aware of:

| Format / Protocol | Relevance |
|-------------------|-----------|
| DLPack | Closest existing tensor ABI; no quantization, limited layout metadata |
| Apache Arrow | Buffer protocol and IPC framing inspiration; columnar, not tensor-focused |
| Apache Arrow Flight | Streaming RPC model reference; gRPC prevents true zero-copy at scale |
| SafeTensors | Simple safe serialization; not a zero-copy runtime protocol |
| GGUF | Block quantization encoding reference (Q4_K, Q8_0, etc.) |
| ONNX TensorProto | Type system breadth reference |
| Zarr v3 | Chunk/shard layout and codec pipeline reference |
| NetCDF | Widely adopted scientific N-D array file format; no zero-copy, no quantization |
| OPeNDAP | De facto array data transport protocol in Earth Sciences; HTTP-based, not zero-copy |
| NIXL | NVIDIA tensor transfer library; solves RDMA transport but has no tensor metadata vocabulary |
| NCCL + GPUDirect | GPU collective communications; raw buffer transfers, no layout or quantization metadata |
