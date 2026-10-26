# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html). All crates in the
workspace share one version and are released together.

> **Pre-1.0 note:** while the version is `0.x`, the format and APIs may change between
> releases without a compatibility guarantee. A breaking change bumps the **minor** version
> (`0.x` convention). The `1.x` backward/forward-compatibility contract begins at `1.0.0`
> (see [`versioning`](docs/spec/versioning.md)).

## [Unreleased]

### Added

- **`hurray-io`** — a **random-access core for the file container**, so the `HRRYFILE`
  format can be read and written with no runtime linked (ADR-041 §§ 1, 7). Where the
  stream core is fed forwards, `FileReaderCore` is asked by *range*: it names the bytes
  `[offset, offset + len)` it needs — the trailer at `file_size - 40`, the index located
  from it, then one descriptor or buffer at a time — and the driver seeks there and reports
  how many arrived. `FileWriterCore` emits one forward pass, accumulating the footer index
  as tensors go out, and offers each payload **in place**: borrowed, one stretch at a time,
  never assembled, and never retained once the item is written. Padding is emitted from a
  shared zero page rather than materialised, so a 2 MiB `data_buffer_alignment` costs no
  2 MiB buffer. Each buffer a read returns is allocated at the alignment its handle
  *declares*, and the memory handed out is the memory the transport filled — the same
  `DestinationAlloc` seam the stream core uses, so a binding can still supply its own
  allocation. `flatten_composite` validates a composite tree and returns the order its
  tensors go on disk, which is what replaces the writer's recursion with an explicit stack.
  Runnable as `cargo run --example sans_io_file`; see
  [Random Access Without a Runtime](docs/cookbook/random-access-without-a-runtime.md).
- **`hurray-io`** — `hurray_io::file` is no longer gated on the `tokio` feature. Only
  `file::reader` and `file::writer` are, mirroring `stream`, so the container format,
  `KvValue`, `IndexEntry` and both cores are available to a consumer with no async runtime
  — which is what the C file surface will be built on.
- **`hurray-ffi`** — blocking stream read and write over a caller-supplied **transport
  vtable**, so a C consumer can finally obtain a tensor rather than only decode one handed
  to it (ADR-041 § 8). `HurrayReader` and `HurrayWriter` stop being empty forward
  declarations and become real handles: `hurray_reader_new` / `hurray_reader_next_tensor`
  / `hurray_reader_destroy` and `hurray_writer_new` /
  `hurray_writer_write_tensor` / `hurray_writer_destroy`, each over a `HurrayTransport`
  of caller callbacks — `read`, `write`, and an optional `release` for the context pointer
  Hurray never frees. Not a path and not a file descriptor: a vtable composes with a
  socket, a pipe, a memory span, or the caller's own object. The callbacks block; nothing
  inside Hurray does, because the decoding is `hurray-io`'s sans-io core and the C handles
  are a second driver for it. **No async runtime is linked into a C consumer's binary.**
  Payloads are passed as borrowed `HurrayIoSlice` values and are not retained after the
  call; each buffer a reader returns is allocated at the alignment its buffer handle
  *declares* and written into directly by the transport, so no copy of a payload happens
  after that write. Flat tensors only — composites and file containers are not reachable
  from this surface yet.
- **`hurray-ffi`** — blocking **`HRRYFILE` read and write** over the same transport
  vtable, so a C consumer can load one tensor out of a model file by name with no runtime
  linked (ADR-041 §§ 7, 8). `HurrayFileReader` opens a container — header, trailer, footer
  index, KV section — and then answers `hurray_file_reader_tensor_count` /
  `..._tensor_name` from the index without reading tensor data at all;
  `hurray_file_reader_read_descriptor` decodes one descriptor so a loader can decide
  whether it wants the tensor, and `hurray_file_reader_read_tensor` reads exactly the
  ranges that one tensor occupies. `HurrayFileWriter` emits one forward pass:
  `hurray_file_writer_new` (which writes the 64-byte header),
  `hurray_file_writer_write_tensor` per tensor, then `hurray_file_writer_finish` for the
  KV section, index and trailer. The vtable grew two callbacks for the read side —
  **`read_at(ctx, offset, dst, len, &filled)` and `size(ctx, &size)`** — because a
  container is *addressed* rather than consumed: its trailer sits at `size - 40`, and
  every request the core makes is absolute. That is also what makes a failed range read
  retryable in a way a stream read is not: re-issuing it reads *the same bytes*. A file
  writer needs only `write`. Flat tensors and the KV section; composites are not reachable
  from this surface yet. Runnable as
  `cargo run --example file_vtable -p hurray-ffi`, with C tabs on
  [Random Access Without a Runtime](docs/cookbook/random-access-without-a-runtime.md).
- **`hurray-ffi`** — the **KV metadata section in both directions**, through one tagged
  `HurrayKvValue` (and `HurrayKvEntry`) used for reads and writes alike, with the
  `HURRAY_KV_*` kinds being the section's own wire tags. An entry a reader hands back can
  be passed straight to another file's `hurray_file_writer_finish`, so copying a model's
  metadata from one container to another needs no transcription. All seven value kinds,
  arrays included.
- **`hurray-ffi`** — two status codes for the file surface: `HURRAY_ERR_NOT_FOUND` for a
  name that is not in the index — the one file-read failure that is a question rather than
  a defect — and `HURRAY_ERR_INVALID_FILE` for the container-format family. Bad magic and
  an unsupported container version keep reporting `HURRAY_ERR_INVALID_MAGIC` and
  `HURRAY_ERR_VERSION_MISMATCH`, which say more.
- **`hurray-ffi`** — `HurrayTensor`, a handle owning a descriptor **and** its buffers
  ([#174](https://github.com/pgillet/hurray/issues/174), the half of it a stream read
  forces). A tensor is that pair, and a consumer holding only one half has either bytes it
  cannot interpret or a description of bytes it does not have.
  `hurray_tensor_descriptor` and `hurray_tensor_buffers` hand back borrowed handles, and
  `hurray_tensor_destroy` releases both in one call whose order cannot be got wrong. The
  native-protocol capsule still carries a `HurrayBufferList` (ADR-030 § 2), so #174 stays
  open for that half.
- **`hurray-ffi`** — seven status codes for the stream surface:
  `HURRAY_ERR_TRANSPORT`, `HURRAY_ERR_SINK_CLOSED`, `HURRAY_ERR_UNEXPECTED_EOF`,
  `HURRAY_ERR_CORE_UNUSABLE`, `HURRAY_ERR_LIMIT_EXCEEDED`, `HURRAY_ERR_INVALID_STREAM`,
  and `HURRAY_END_OF_STREAM`. The last is the **first positive** status: a stream ending
  is not a failure, so consumers test for errors with `status < 0` rather than
  `status != HURRAY_OK`.

- **`hurray-ffi`** — descriptor field reads, completing Compliance Level 1 for the C
  surface and giving it the five optional sections. A consumer handed a CSR descriptor
  could previously read `layout_tag == 0x07` and nothing further — not `nnz`, not which
  buffer held `row_ptr` — and one handed a quantized tensor learned nothing about the
  scheme at all, so the C surface was effectively dense and unquantized. A field is now
  named by a `HurrayDescriptorField` identifier whose high half-word says which
  sub-object it addresses — layout payload, quantization, statistics, shard, extension
  type, composite member — and read through five typed accessors:
  `hurray_descriptor_field_u64`, `..._i64`, `..._f64`, `..._u64_array` and
  `..._i64_array`. That covers every layout-specific field of all twelve layouts and
  every field of all five sections. Two new status codes,
  `HURRAY_ERR_FIELD_NOT_PRESENT` and `HURRAY_ERR_FIELD_NOT_APPLICABLE`, keep an absent
  optional field distinguishable from a field belonging to another sub-object; a
  statistic whose `computed_mask` guard bit is clear reads as absent, so a C caller
  cannot read an uncomputed statistic by omitting the mask check. See
  [ADR-039](docs/adr/ADR-039-c-abi-structured-descriptor-fields.md) and
  [ADR-040](docs/adr/ADR-040-c-abi-section-field-identifiers.md).

### Changed

- `FileReader` and `FileWriter` are **drivers** over the new cores, with unchanged public
  signatures and byte-for-byte identical output. Between them they lost 600 lines of
  format logic: no descriptor decode, no offset arithmetic, no padding, no alignment, no
  index or KV parsing remains in either. A corpus of four `.hrry` files written by the
  pre-refactor writer is checked in and only ever read, so the compatibility test cannot
  become an assertion that the code agrees with itself.
- A file write that the sink refuses mid-item is now `Error::SinkClosed` rather than an
  `io::Error` of kind `WriteZero`, matching what the stream writer began reporting in the
  previous pass: a zero-length write is never progress, and a closed sink is a distinct
  signal rather than an error to retry. Both file drivers now drive the transport with
  `read`/`write` rather than `read_exact`/`write_all`, because a bulk helper that fails
  cannot report how many bytes it moved — and that count is the position.
- `FileReader::read_composite` and `FileWriter::write_composite` walk nesting with an
  explicit stack instead of `Box::pin` recursion (ADR-041 § 10). A non-composite name
  passed to `read_composite` is now refused as soon as its descriptor decodes, before its
  buffers are read; the error is unchanged.
- The workspace version is **0.2.0**. `HurrayReader` and `HurrayWriter` were publicly
  constructible unit structs in 0.1.0 and are now opaque handles, which the `0.x`
  convention makes a minor bump.
- `HURRAY_C_ABI_VERSION` is `8`: the header no longer describes the same library. Two
  forward declarations became real handles, `HurrayTensor` joined them, and
  `HurrayTransport` grew `read_at` and `size` — appended rather than grouped beside
  `read`, so a caller using positional initialisation gets null for the new fields instead
  of a release callback where a positional read belongs. A v6 or v7 consumer must relink.
- **A failed file write ends the file**, on every surface. By the time bytes move, a
  tensor has consumed its name in the footer index and staged its descriptor, so a retry
  under the same name would describe a file other than the one on the sink. The C writer
  therefore poisons itself whether the failure landed before or after the first byte; the
  status still distinguishes the two (`HURRAY_ERR_SINK_CLOSED` after, the transport's own
  status before) but the recovery is the same: write the file again from the start. This
  is a documented contract rather than a behaviour change — the underlying sharp edge is
  the Rust driver's too, and is reported as
  [#248](https://github.com/pgillet/hurray/issues/248).
- `hurray_io::Error::InvalidHeader` now maps to `HURRAY_ERR_INVALID_FILE` rather than
  `HURRAY_ERR_INVALID_STREAM` across the C surface. Every site that produces it is in the
  file container, and no C entry point could reach it before this release.
- `hurray-ffi` depends on `hurray-io` with `default-features = false`. The workspace
  dependency carries that default and the consumers that want the async drivers
  (`hurray-python`, the conformance harness) opt in with `features = ["tokio"]` — a
  workspace default of `true` silently overrides a member asking for `false`, which would
  have put tokio in the C library after all.
- The descriptor field identifier type and its accessors are named for the descriptor
  rather than for the layout, now that one identifier space covers both —
  `HurrayLayoutField` is `HurrayDescriptorField`, and the
  `hurray_descriptor_layout_field_*` functions dropped their `layout_` infix. The
  native-protocol capsule check is exact equality, so a capsule produced by an earlier
  build is refused by this one and vice versa.
- `website/check-coverage-matrix.py` now counts a `#define`d `HURRAY_*` constant as
  exported C surface, not only a declared function. The five section rows share their
  accessors with every other sub-object, so their identifier constants are the only
  header evidence specific to each one.

### Fixed

- `FileReader` no longer panics on a file whose header declares a `data_buffer_alignment`
  of `0` — or any non-power-of-two. The value reached the padding arithmetic as a divisor,
  so a crafted multi-buffer file could divide by zero inside a parser; it is now rejected
  as `Error::InvalidHeader`, which `file-format.md` already required of the field.
- `FileReader::open` no longer trusts an index or KV section's own entry count when
  reserving memory for it, and checks the trailer's `index_offset + index_length` for
  overflow. A crafted footer claiming 2^40 entries in a 1 KiB file asked for an allocation
  that size, and the sum could wrap.
- `FileWriter` no longer pads a composite head's descriptor to `data_buffer_alignment`,
  which `file-format.md § Composite Tensors` forbids: a head declares no data buffer, so
  there is nothing to align. The writer padded before it looked at the buffer table, so
  every head cost a full alignment block — 4 KiB by default, and a model with one
  composite per layer paid that per layer. A head's index entry now records
  `data_offset = descriptor_offset + descriptor_length` with `data_length = 0`, and its
  first member follows at the ordinary 8-byte descriptor boundary. An *empty* tensor still
  declares one buffer of `byte_size = 0` and is still aligned as a buffer. Files written
  before the fix are unaffected on read — every offset comes from the footer index — and
  `tests/fixtures/file/nested_composite_padded_head.hrry` is kept as the 0.1.0 output that
  proves it.
- ADR-038 was on disk but absent from the documentation navigation, so it never appeared
  on the published site.

## [0.1.0] - 2026-09-30

First release. Everything below is new, because nothing preceded it.

### Added

- **`hurray-core`** — tensor descriptor with binary encode/decode; element-type system
  (Tier 1 + Tier 2, sub-byte and private extension types); buffer handle with device and
  memory-class tags and sync modes; quantization descriptors (per-tensor / per-channel /
  per-block affine, NF4, MXFP); and the twelve-layout memory vocabulary (row-major,
  column-major, strided, tiled, Morton, Hilbert, sparse COO/CSR/CSC/CSF, block-paged, and
  composite) with element-address computation.
- **`hurray-io`** — async streaming interchange and the `HRRYFILE` container (named tensors,
  footer index, typed key-value metadata), each with composite-tensor support.
- **`hurray-ffi`** — a stable C ABI over the core types (opaque handles, function table,
  release callbacks).
- **`hurray-python`** — Python bindings (PyO3) with NumPy/DLPack zero-copy interop, sparse
  and SciPy interop, and file I/O.
- **`hurray-inspect`** — a CLI to inspect descriptor files as an annotated hex table.
- A language-neutral **conformance corpus** (`conformance/`) validated by both the Rust and
  Python test suites.
- The full **format specification**, implementation requirements, cookbook, and ADRs,
  published as a versioned documentation site.

### Not implemented

0.1.0 does not implement everything the specification defines. The
[implementation status page](docs/impl/implementation-status.md) is generated from the code
and is the authoritative account; the gaps worth knowing before you depend on this release:

- **Level 3 network transport** — specified in
  [`interchange`](docs/spec/interchange.md), implemented by no crate. None of the message
  types, capability flags, or the RDMA data plane exist anywhere. `hurray-io` implements
  the stream framing and the file container — Levels 1 and 2 — and stops there.
- **`hurray-ffi` reads a descriptor but not every layout.** The C surface exposes the type
  tag, layout tag, rank, shape, byte offset and buffer table; layout-specific fields (CSR
  `nnz`, tile shape, the block table) have no accessor yet, so a C caller cannot interpret
  every layout it can decode. Level 2 (writing) is not exposed at all.
- **Streaming over a file descriptor in `hurray-python` is Unix-only.** Paths, `bytes` and
  the in-memory writer work everywhere; passing an object with `fileno()` raises
  `hurray.UnsupportedError` on Windows.

The specification remains **Draft** at this release, which is what a `0.x` ships against:
the `1.x` compatibility contract has not been opened, and tag assignments are not yet
frozen.

[Unreleased]: https://github.com/pgillet/hurray/compare/0.1.0...HEAD
[0.1.0]: https://github.com/pgillet/hurray/releases/tag/0.1.0
