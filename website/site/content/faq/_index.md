+++
title = "FAQ"
sort_by = "none"
template = "section.html"
+++

## What is Hurray?

A language-agnostic, zero-copy binary format for multi-dimensional tensor data, optimized
for AI/ML inference. It provides two formats — a streaming format for runtime interchange
and a file format for on-disk model storage — that share a single tensor descriptor
encoding.

## How is it different from SafeTensors or GGUF?

SafeTensors and GGUF are file formats. Hurray covers **both** on-disk storage *and* runtime
interchange with the same descriptor, and adds a rich layout vocabulary (twelve layouts,
including sparse and block-paged) and five normative quantization schemes with defined
dequantization formulas. GGUF's block-quantization and typed key-value metadata were direct
references; Hurray generalizes them.

## How is it different from Apache Arrow?

Arrow is columnar and record-batch oriented; Hurray is tensor oriented. Hurray borrows
Arrow's buffer-protocol and IPC-framing ideas but targets N-dimensional tensors, quantized
element types, and inference-specific layouts rather than columnar analytics.

## How is it different from DLPack?

DLPack is the closest existing tensor ABI and inspired Hurray's zero-copy, in-process
interchange. Hurray adds quantization, a much broader layout vocabulary, IPC and
cross-machine transports, and an on-disk file format — none of which DLPack addresses.

## Is it zero-copy?

Yes. Buffers are shared by reference through a stable C ABI: 64-byte SIMD alignment in the
streaming format and 4 KiB page alignment in the file format for mmap-to-GPU loading.
Quantization parameters live in separate buffer-table entries so both data and parameters
stay zero-copy.

## Which languages are supported?

The format and its C FFI boundary are deliberately language-agnostic — any language that
can read a struct from a buffer can implement Hurray. The reference implementation is in
Rust (core types plus streaming and file I/O) and includes a C ABI layer and Python
bindings. The Python bindings offer zero-copy interop with NumPy and PyTorch — via
`__dlpack__` and a native Hurray buffer protocol — plus `save`/`load` for the file format.
Bindings for other languages can build directly on the same C ABI. There is also a
`hurray-inspect` command-line tool for examining Hurray descriptors byte by byte.

## Is the format stable yet?

The specification is at `0.1.0-draft`. The format is designed to be evolvable —
backward-compatible and forward-additive across the whole `1.x` line, with public tags
never rebound — but it is pre-1.0 and may still change. See the
[versioning policy](/docs/stable/spec/versioning.html) in the docs.

## Where do I report issues or ask questions?

On [GitHub](https://github.com/pgillet/hurray). See the [Community](@/community/_index.md)
page for contribution guidelines and governance.
