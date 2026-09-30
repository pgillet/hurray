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
