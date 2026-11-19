# Rust Reference Implementation Requirements — Hurray Implementation Requirements

## Overview

The Rust reference implementation is the canonical implementation of the Hurray
format. It is authoritative: when the spec is ambiguous, the reference implementation
defines the correct behaviour. When the implementation deviates from the spec, the
implementation is wrong.

The implementation is split across four crates:

| Crate | Responsibility |
|-------|---------------|
| `hurray-core` | Format types, tensor descriptor, buffer handle, quantization descriptors. No I/O, no async. |
| `hurray-io` | Async streaming read/write, file format support. Depends on `hurray-core` and `tokio`. |
| `hurray-transcode` | Representation conversion: layout, element type, quantization scheme, placement. Depends on `hurray-core` only. |
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

### Core and Drivers

The streaming and file logic MUST be split into a **sans-io core** — a state machine over
bytes, holding no file, socket, or runtime — and thin **drivers** that move bytes between
a transport and that core. The async API below is one such driver; a blocking C ABI and a
Python event loop are others, and all of them MUST share the one core rather than each
growing its own bridge (ADR-041 § 1).

The following are requirements on any implementation of the core, not internal details:

- **The core supplies the read destination** and MUST NOT accept payload bytes a driver
  placed elsewhere. The guarantee is that no copy of a payload happens after the transport
  writes into that destination (§ 2). This is not end-to-end zero copy, which only mmap or
  shared memory gives; the defect it rules out is a second, tensor-sized copy.
- **The write path is not the same requirement mirrored.** The core MUST accept borrowed,
  non-contiguous payload slices and MUST NOT assemble them. The caller keeps each slice
  valid until the core reports *that item written*, and the core MUST expose that point and
  MUST NOT retain the pointer afterwards (§ 3).
- **Progress is explicit and never blocks.** One request is outstanding at a time, and
  every transfer reports a byte count. A short read is not end of input — only a zero-byte
  read is, and then only at an item boundary; mid-item it is an error. A zero-length write
  is never progress: a closed sink is a distinct signal (§ 4).
- **Alignment is the alignment the buffer handle declares**, not the 64-byte minimum and
  not the allocator's default, and MUST hold across multi-buffer and composite paths
  (§ 5).
- **Resumability requires a known position.** A transport operation and its byte-count
  report MUST form one uninterruptible step, so cancellation is observable only at a step
  boundary. If a count is ever lost the core MUST enter an explicitly named unusable state
  rather than diverge silently, and a partially filled destination MUST never be
  observable (§ 6).
- **A writer core MUST commit at completion.** A submitted item MUST NOT mutate any state
  the core carries past the operation — the name it consumes, the index entry it earns, the
  bytes it stages — until the core reports that item written. An operation abandoned before
  its first byte therefore leaves the core unchanged, and resubmitting the same item under
  the same name MUST succeed. Once the first byte is accepted the core MUST become unusable
  and MUST retire that state only when the item completes, and it MUST NOT depend on a
  driver reporting the abandonment, because a driver that errors and drops the operation
  reports nothing (§ 6b).
- **Composite nesting MUST be an explicit stack**, not call recursion, so a composite can
  survive being suspended mid-member (§ 10).

The two access patterns the format has get two request vocabularies, and a core MUST admit
both (§ 7):

- A **sequential** core is fed forwards: it asks for *the next* `n` bytes. This serves
  stream read and write, and the file writer's single forward pass.
- A **random-access** core is asked by range: it names the bytes `offset .. offset + len`
  it needs and is told how many arrived. This serves the file reader, which seeks — the
  trailer at `file_size - 40`, the index located from it, tensors found by name — and
  therefore cannot be driven by a byte feed at all.

A random-access core MUST NOT need the unusable state of § 6, and MUST NOT be given one:
because every request is absolute and the read position advances only when a count is
reported, a transfer whose count is lost is recovered by re-issuing the same range, and
re-filling the destination from the start of that range is idempotent. For the same
reason, a *job* that fails (a name not in the index, a descriptor that will not decode, a
torn composite) MUST leave the core usable — only a failure during the open sequence, where
there is no index to start again from, is terminal.

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
- A reader MUST reject a `data_buffer_alignment` that is not a power of two, rather than carrying it into the padding arithmetic: `0` is a divisor there, and a crafted header MUST NOT be able to panic a parser.
- The reader MUST NOT read a padding byte it was not asked for. Each request covers exactly one section, descriptor, or buffer; inter-buffer padding is skipped by the next request's offset, never consumed.

## hurray-transcode

`hurray-core` **describes** a tensor; `hurray-transcode` is the first crate in the stack
that **reads payload bytes as values**. That is the whole boundary, and it is where the
types already draw it: a `BufferHandle` carries no pointer, so everything `hurray-core` can
check is descriptor-internal, and everything a conversion does needs the bytes.

Nothing about the format moves into it. Index-to-address computation, the NF4 lookup table,
the MXFP scale encoding, buffer sizing, packing factors and every descriptor validity
constraint stay in `hurray-core`; when a conversion needs a format fact `hurray-core` does
not expose, the fix is to expose it there rather than re-derive it — a second
implementation of the format's arithmetic living beside the first is the specific failure
this rule prevents. Output descriptors are built through `hurray-core`'s own constructors,
so no converted tensor can be more permissive than a decoded one. What *does* live here is
validation that needs data — sorted and in-bounds sparse coordinates, a monotonic `indptr`,
a block table within its pool, an address that actually meets its declared alignment —
because a descriptor-only crate cannot perform it.

The crate is **not** a dependency of `hurray-io`, and the direction is deliberate: the
first consumer is a Level 3 server, which is `hurray-io` code that needs to *call*
transcoding to honour `ALLOW_TRANSCODE`. Note that the crate's scope is wider than the
protocol's: `interchange.md`'s `TRANSCODING` capability covers layout transcoding only, and
a server MUST NOT read `ALLOW_TRANSCODE` as permission to cast or requantize on a client's
behalf.

Requirements on any implementation of it:

- Conversion MUST be expressible as **plan, survey, run**: a plan reads descriptors only
  and publishes its output descriptor and output buffer sizes before any byte moves; a
  survey reads bytes and produces facts, never a tensor; a run reads bytes and writes
  bytes. A conversion whose admissibility or output size depends on the data MUST be
  reachable only through a survey.
- The caller MUST allocate every destination. The crate MUST NOT allocate an output buffer,
  so that a conversion can write straight into a consumer's own memory.
- A run MUST refuse a destination that is shorter than the plan published, or whose base
  address does not meet the alignment it published.
- A plan MUST publish each output buffer as **produced** or **aliased**, so that a
  conversion which needs no byte moved is a conversion with an empty body rather than a
  copy. A request whose every output buffer is an alias MUST succeed.
- A plan MUST partition its work into items that partition the **output**: running every
  item exactly once, in any order, on any number of threads, MUST produce byte-identical
  output to one serial run.
- The crate MUST be synchronous, and a default build MUST NOT be able to spawn a thread, so
  that it stays callable from inside an async task. Parallelism is an opt-in feature at the
  edge.
- A refusal MUST name what was refused. An operation outside the membership rule — a
  reshape, a lossy requantization, an operation that changes the logical value — MUST be
  refused rather than approximated.

Which conversions are implemented, and which are permanently out of scope, is the
[transcode matrix](transcode-matrix.md). The rule that decides what may ever be added is
ADR-044 § 1.

## Code Quality

- No `unwrap()` or `expect()` in library code. All errors propagate with `?`.
- Feature flags: `serde` (serialization support), `tokio` (async I/O, enabled in `hurray-io`).
- MSRV (minimum supported Rust version): tracked in `Cargo.toml` and enforced in CI.
- All changes to `hurray-core` and `hurray-io` MUST be reviewed by the `rust-reviewer` agent before merge.
