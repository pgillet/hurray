# Hurray Format Specification

**Version:** 0.1.0-draft

Hurray is a language-agnostic, zero-copy runtime interchange format for
multi-dimensional tensor data, optimized for the memory layout diversity,
quantization schemes, and access patterns of modern AI/ML inference pipelines.

## Scope and Goals

- Define a binary tensor descriptor encoding that is language- and runtime-agnostic.
- Enable zero-copy buffer sharing across runtimes, processes, and devices.
- Support the full range of quantization schemes used in modern inference.
- Be streamable: a reader MUST be able to start processing tensor data without buffering the entire input, and a writer MUST be able to emit tensor data incrementally without buffering the entire output. Tensor descriptors always precede their data buffers; the format is self-delimiting; back-references and end-of-file indexes are not permitted.
- Be extensible without breaking existing readers.

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
| [memory-layout.md](memory-layout.md) | Strides, contiguous, tiled, and packed (sub-byte) layouts |
| [buffer-protocol.md](buffer-protocol.md) | Zero-copy semantics, alignment, device memory |
| [metadata.md](metadata.md) | Tensor descriptor binary encoding |
| [interchange.md](interchange.md) | Runtime interchange: in-process, IPC, cross-machine |
| [versioning.md](versioning.md) | Format version field and compatibility policy |
| [references.md](references.md) | Normative references |
