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
- **Extensible and evolvable:** extension points are stable across 1.x; new named values go through the spec amendment process; backward and forward-additive compatibility is guaranteed within major version `1.x`. See [`versioning`](versioning.md) § Evolvability Contract.
- Align the Tier 1 element type vocabulary with the Python Array API Standard, enabling zero-copy interoperability without dtype translation for standard numeric types. See [`docs/impl/python-bindings`](../impl/python-bindings.md) for binding-level requirements.
- Serve as the storage foundation of an array database engine: the file format, tiled/blocked layout, Morton and Hilbert curve layouts, and footer index are designed to be compatible with sub-array queries, tile-skipping, and range-based retrieval. A concrete target use case is an embeddable SQL/MDA query engine (ISO 9075-15) backed by Hurray buffers, with zero-copy handoff from query results to an ML inference pipeline. Spec decisions that would foreclose chunk-based access, spatial locality, dimension-range indexing, or SQL/MDA interoperability MUST be evaluated against this use case before being adopted.

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
| [element-types](element-types.md) | Numeric element type system: type tags, bit widths, encoding, sub-byte packing |
| [data-model](data-model.md) | Element type system, shape and dimension model |
| [quantization](quantization.md) | Quantization scheme registry: descriptor header, scheme tag space, partial-block policy, buffer placement rules |
| [quantization/per-tensor-affine](quantization/per-tensor-affine.md) | Per-tensor affine quantization — scheme tag `0x01` (Tier 1) |
| [quantization/per-channel-affine](quantization/per-channel-affine.md) | Per-channel (per-axis) affine quantization — scheme tag `0x02` (Tier 1) |
| [quantization/per-block-affine](quantization/per-block-affine.md) | Per-block affine quantization — scheme tag `0x03` (Tier 1) |
| [quantization/nf4](quantization/nf4.md) | NF4 (NormalFloat4) block quantization — scheme tag `0x04` (Tier 2) |
| [quantization/mxfp](quantization/mxfp.md) | MXFP (OCP Microscaling) block quantization — scheme tag `0x05` (Tier 2) |
| [memory-layout](memory-layout.md) | Layout taxonomy, common fields, element address computation, alignment, sharding, buffer table |
| [layouts/row-major](layouts/row-major.md) | Row-major (C order) layout — tag `0x01` |
| [layouts/column-major](layouts/column-major.md) | Column-major (Fortran order) layout — tag `0x02` |
| [layouts/strided](layouts/strided.md) | Strided layout with negative/zero stride support — tag `0x03` |
| [layouts/tiled](layouts/tiled.md) | Tiled / blocked layout with recursive nesting — tag `0x04` |
| [layouts/morton](layouts/morton.md) | Morton (Z-order curve) layout — tag `0x05` |
| [layouts/coo](layouts/coo.md) | COO (Coordinate) sparse layout — tag `0x06` |
| [layouts/csr](layouts/csr.md) | CSR (Compressed Sparse Row) sparse layout — tag `0x07` |
| [layouts/csc](layouts/csc.md) | CSC (Compressed Sparse Column) sparse layout — tag `0x08` (also known as CCS) |
| [layouts/csf](layouts/csf.md) | CSF (Compressed Sparse Fiber) sparse layout — tag `0x09` |
| [layouts/block-paged](layouts/block-paged.md) | Block-paged (PagedAttention KV cache) layout — tag `0x0A` |
| [layouts/composite](layouts/composite.md) | Composite / Virtual tensor (head + members) — tag `0x0B` |
| [layouts/hilbert](layouts/hilbert.md) | Hilbert curve layout — tag `0x40` |
| [buffer-protocol](buffer-protocol.md) | Zero-copy semantics, alignment, device memory |
| [metadata](metadata.md) | Tensor descriptor binary encoding |
| [interchange](interchange.md) | Streaming IPC format: in-process, IPC, cross-machine network transport |
| [file-format](file-format.md) | File format: random-access container with named tensors, footer index, KV metadata |
| [versioning](versioning.md) | Format version field and compatibility policy |
| [references](references.md) | Normative references |
