# Rust Reference Implementation Requirements — Hurray Implementation Requirements

## Overview

The Rust reference implementation is the canonical implementation of the Hurray
format. It is authoritative: when the spec is ambiguous, the reference implementation
defines the correct behaviour. When the implementation deviates from the spec, the
implementation is wrong.

The implementation is split across three crates:

| Crate | Responsibility |
|-------|---------------|
| `hurray-core` | Format types, tensor descriptor, buffer handle, quantization descriptors. No I/O, no async. |
| `hurray-io` | Async streaming read/write, file format support. Depends on `hurray-core` and `tokio`. |
| `hurray-ffi` | C ABI layer. See `c-ffi.md`. |

## hurray-core

### Type System

- MUST define a `TensorDescriptor` type that encodes all fields from `docs/spec/metadata.md`.
- MUST define an `ElementType` enum covering all Tier 1 and Tier 2 type tags.
- MUST define a `LayoutTag` enum covering all named layout tags.
- MUST define layout-specific descriptor types for each named layout (e.g., `StridedLayout`, `TiledLayout`).
- MUST define a `BufferHandle` type carrying `byte_size`, `alignment`, `device_tag`, and a release callback (see `c-ffi.md`).
- MUST define an `Error` enum via `thiserror`. No `unwrap()` or `expect()` in library code.

### Serialization

- MUST implement binary serialization of `TensorDescriptor` to the wire format defined in `docs/spec/metadata.md`.
- MUST implement binary deserialization with full validation (magic, version, flag bits, bounds checks, sparse invariants).
- Serialization MUST be `no_std`-compatible when the `alloc` feature is enabled.
- A `serde` feature gate MUST provide `serde::Serialize` / `serde::Deserialize` for `TensorDescriptor` (JSON/CBOR interchange for tooling, not the wire format).

### Buffer Safety

- `unsafe` code MUST be isolated in dedicated modules.
- Every `unsafe` block MUST have a `// SAFETY:` comment explaining the invariant that makes the code sound.
- Buffer aliasing across runtimes MUST be mediated through the `BufferHandle` reference count and release callback.

### Correctness

- `cargo clippy -- -D warnings` MUST pass.
- All public items MUST have `///` doc comments with at least one example.
- Test coverage for the public API MUST be ≥ 80%.

## hurray-io

### Streaming Read

- MUST implement an async tensor descriptor reader that reads exactly `descriptor_length` bytes before emitting a parsed `TensorDescriptor`.
- MUST implement an async data frame reader that yields data in chunks without buffering the entire tensor.
- A reader MUST be able to start processing tensor data without buffering the entire input (streamable principle).

### Streaming Write

- MUST implement an async tensor descriptor writer that emits the descriptor before any data bytes.
- MUST implement an async data frame writer that emits data incrementally.
- A writer MUST be able to emit tensors one at a time without buffering the entire output.

### Async Runtime

- MUST use `tokio` as the async runtime.
- MUST NOT mix `rayon` thread pool calls directly in async contexts. CPU-bound operations MUST use `tokio::task::spawn_blocking`.
- All async functions MUST be `Send + 'static` to support multi-threaded tokio runtimes.

### Streaming Format

- MUST support reading and writing the streaming IPC format defined in `docs/spec/interchange.md`: a sequence of zero or more tensor descriptors + data buffers, terminated by an end-of-stream marker.
- The streaming format MUST be self-delimiting: `descriptor_length` allows a reader to advance past any descriptor without full parsing.
- Back-references and end-of-file indexes are forbidden in the streaming format (streamable principle).

### File Format

- MUST support reading and writing the Hurray file format defined in `docs/spec/file-format.md`. The file format is a single-pass writable, random-access readable container for one or more named tensors.
- A writer MUST emit the file in a single forward pass (no seek-back), producing the structure: `HRRYFILE` magic + 64-byte file header + zero or more tensor regions (each tensor descriptor followed by its data buffer(s) with appropriate padding) + optional KV metadata section + index section + 40-byte trailer at `file_size - 40`.
- The writer MUST track per-tensor `(name, descriptor_offset, descriptor_length, data_offset, data_length)` tuples in memory and emit them in the index section after all tensor regions are written.
- A random-access reader MUST locate the trailer by seeking to `file_size - 40`, verify `trailer_magic` (ASCII `HRRY`), and use `index_offset` / `index_length` from the trailer to locate the index section.
- When the `HAS_INDEX_CRC32C` file flag is set, the reader MUST verify the CRC-32C of the index section and reject the file on mismatch.
- The implementation MUST support mmap-based zero-copy loading of tensor data buffers when the file's `data_buffer_alignment` matches or exceeds the host page size.

## Code Quality

- No `unwrap()` or `expect()` in library code. All errors propagate with `?`.
- Feature flags: `serde` (serialization support), `tokio` (async I/O, enabled in `hurray-io`).
- MSRV (minimum supported Rust version): tracked in `Cargo.toml` and enforced in CI.
- All changes to `hurray-core` and `hurray-io` MUST be reviewed by the `rust-reviewer` agent before merge.
