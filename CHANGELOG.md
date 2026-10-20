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

- The workspace version is **0.2.0**. `HurrayReader` and `HurrayWriter` were publicly
  constructible unit structs in 0.1.0 and are now opaque handles, which the `0.x`
  convention makes a minor bump.
- `HURRAY_C_ABI_VERSION` is `7`: the header no longer describes the same library, since
  two forward declarations became real handles and `HurrayTensor` joined them. A v6
  consumer must relink.
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
