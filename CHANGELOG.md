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

[Unreleased]: https://github.com/pgillet/hurray/commits/main
