# Hurray Format Specification

**Version:** 0.1.0-draft

Hurray is a language-agnostic, zero-copy runtime interchange format for
multi-dimensional tensor data, optimized for the memory layout diversity,
quantization schemes, and access patterns of modern AI/ML inference pipelines.

## Scope and Goals

- Define a binary tensor descriptor encoding that is language- and runtime-agnostic.
- Enable zero-copy buffer sharing across runtimes, processes, and devices.
- Support the full range of quantization schemes used in modern inference.
- Be streamable (streaming format): a reader MUST be able to start processing tensor data without buffering the entire input, and a writer MUST be able to emit tensor data incrementally without buffering the entire output. Tensor descriptors always precede their data buffers; the format is self-delimiting; back-references are not permitted. File format writers operate in a single forward pass and append a footer index at the end.
- Be extensible without breaking existing readers.
- Align the Tier 1 element type vocabulary with the Python Array API Standard, enabling zero-copy interoperability without dtype translation for standard numeric types. See [`docs/impl/python-bindings.md`](../impl/python-bindings.md) for binding-level requirements.
- Serve as the storage foundation of an array database engine: the file format, tiled/blocked layout, Morton and Hilbert curve layouts, and footer index are designed to be compatible with sub-array queries, tile-skipping, and range-based retrieval. Spec decisions that would foreclose chunk-based access, spatial locality, or dimension-range indexing MUST be evaluated against this use case before being adopted.

## RFC 2119 Notice

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD",
"SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in these documents are to be
interpreted as described in [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

## Versioning

This specification follows semantic versioning. A reader MUST reject a tensor
descriptor whose major version field exceeds the reader's supported major version.

## Table of Contents

| Section | Description |
|---------|-------------|
| [element-types.md](element-types.md) | Numeric element type system: type tags, bit widths, encoding, sub-byte packing |
| [data-model.md](data-model.md) | Element type system, shape and dimension model |
| [quantization.md](quantization.md) | Per-tensor, per-channel, and per-block quantization |
| [memory-layout.md](memory-layout.md) | Layout taxonomy, common fields, element address computation, alignment, sharding, buffer table |
| [layouts/row-major.md](layouts/row-major.md) | Row-major (C order) layout — tag `0x01` |
| [layouts/column-major.md](layouts/column-major.md) | Column-major (Fortran order) layout — tag `0x02` |
| [layouts/strided.md](layouts/strided.md) | Strided layout with negative/zero stride support — tag `0x03` |
| [layouts/tiled.md](layouts/tiled.md) | Tiled / blocked layout with recursive nesting — tag `0x04` |
| [layouts/morton.md](layouts/morton.md) | Morton (Z-order curve) layout — tag `0x05` |
| [layouts/subpaving.md](layouts/subpaving.md) | General subpaving (irregular regions) layout — tag `0x06` |
| [layouts/coo.md](layouts/coo.md) | COO (Coordinate) sparse layout — tag `0x07` |
| [layouts/csr.md](layouts/csr.md) | CSR (Compressed Sparse Row) sparse layout — tag `0x08` |
| [layouts/csc.md](layouts/csc.md) | CSC (Compressed Sparse Column) sparse layout — tag `0x09` (also known as CCS) |
| [layouts/hilbert.md](layouts/hilbert.md) | Hilbert curve layout — tag `0x40` |
| [buffer-protocol.md](buffer-protocol.md) | Zero-copy semantics, alignment, device memory |
| [metadata.md](metadata.md) | Tensor descriptor binary encoding |
| [interchange.md](interchange.md) | Streaming IPC format: in-process, IPC, cross-machine network transport |
| [file-format.md](file-format.md) | File format: random-access container with named tensors, footer index, KV metadata |
| [versioning.md](versioning.md) | Format version field and compatibility policy |
| [references.md](references.md) | Normative references |
