+++
title = "Hurray"
+++

Hurray is a **language-agnostic, zero-copy tensor format** for multi-dimensional data,
built for the memory-layout diversity, quantization schemes, and access patterns of modern
AI/ML inference pipelines.

It defines two binary formats that share **one tensor descriptor encoding**:

- **Streaming format** — runtime interchange: in-process pointer passing, IPC, and
  cross-machine streaming. Self-delimiting, no seek required. *Think Apache Arrow, but for
  tensors.*
- **File format** — on-disk model storage and distribution: named tensors, a footer index
  for random access, and mmap-based zero-copy loading. *Think SafeTensors or GGUF, but with
  rich layout and quantization metadata.*

## Why Hurray

- **Zero-copy first** — buffers are shared by reference across runtimes and languages
  through a stable C ABI, with 64-byte SIMD alignment (streaming) and 4 KiB page alignment
  for mmap-to-GPU loading (file).
- **Rich layout vocabulary** — twelve layouts: row-major, column-major, strided,
  tiled/blocked, Morton, Hilbert, sparse COO/CSR/CSC/CSF, block-paged (PagedAttention KV
  cache), and composite.
- **First-class quantization** — per-tensor, per-channel, and per-block affine, NF4
  (QLoRA), and MXFP (OCP Microscaling / Blackwell), with normative dequantization formulas.
- **Streamable by design** — descriptors precede their data, no back-references, no
  end-of-file index; readers and writers both operate incrementally.
- **Evolvable** — backward-compatible and forward-additive across the whole `1.x` line;
  public tags are never rebound.

## Get started

- Use the **Read the docs** button above, or the **Docs** link in the navigation, for the
  full specification, implementation requirements, cookbook, and tutorials.
- Browse the [source on GitHub](https://github.com/pgillet/hurray).
