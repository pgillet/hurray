# Hurray

**Hurray** is a language-agnostic, zero-copy tensor format for multi-dimensional tensor data, optimized for the memory layout diversity, quantization schemes, and access patterns of modern AI/ML inference pipelines.

Hurray defines two binary formats that share the same tensor descriptor encoding:

- **Streaming format** — for runtime interchange: in-process pointer passing, IPC, and cross-machine streaming. Self-delimiting, no seek required. Think Apache Arrow, but for tensors.
- **File format** — for on-disk model storage and distribution: named tensors, footer index for random access, mmap-based zero-copy loading. Think SafeTensors or GGUF, but with rich layout and quantization metadata.

---

## Core Properties

These properties are the contract of the Hurray format. Every section of the specification MUST be consistent with all of them.

### 1. Zero-Copy First

Tensor data is shared across runtimes and languages via buffer handles, not copies. In the streaming format, buffers are passed by reference (64-byte minimum alignment for SIMD, page-aligned for GPU/IPC). In the file format, tensor data buffers are aligned to page boundaries within the file (4 KiB minimum) to enable zero-copy mmap-to-GPU loading. The format defines a stable C ABI so any language can participate without going through Rust. Quantization parameter arrays (scales, zero-points) live in separate buffer table entries — not interleaved with data — preserving zero-copy access to both.

### 2. Two Formats, One Descriptor

Both formats share the same tensor descriptor encoding. A tensor descriptor parser written once works for both the streaming IPC format and the file format. The file format adds a container layer (magic `HRRYFILE`, tensor names, footer index, optional typed key-value metadata) without modifying the descriptor.

### 3. Streamable by Design

**Streaming format:** The tensor descriptor always precedes its data buffer. A reader can start processing before the full payload arrives. No back-references, no end-of-file indexes, no buffering of the full input required. Both readers and writers operate incrementally. The format is self-delimiting: a receiver can determine the descriptor's total byte length from the first 10 bytes.

**File format:** Writers operate in a single forward pass — tensor descriptors and data are written sequentially, byte offsets are tracked in memory, and the footer index and 32-byte trailer are written last. No backward seek is required to write a file. Readers can use the footer index for random access (seek to any tensor by name) or read the file sequentially without seeking.

### 4. Rich Memory Layout Vocabulary

Ten layout types are defined: row-major, column-major, strided, tiled/blocked (for GEMM), Morton Z-order, Hilbert curve, general subpaving, and sparse COO/CSR/CSC. Strides are expressed in logical elements, not bytes. Negative and zero strides are valid. Sub-byte packing (int4, bool) is first-class.

### 5. First-Class Quantization

Five normative quantization schemes are defined: per-tensor affine, per-channel affine, per-block affine (GGUF family), NF4 (QLoRA), and MXFP (OCP Microscaling / NVIDIA Blackwell). The storage type (`type_tag`) is orthogonal to the quantization scheme (`scheme_tag`) — a tensor is quantized if and only if the `HAS_QUANTIZATION` flag is set. Dequantization formulas are normative.

### 6. Language-Agnostic

No Rust-isms leak into the format or the C FFI boundary. The binary spec uses generic type names (`int32`, `float16`, `uint8`). Any language that can read a struct from a buffer can implement it. The spec deliberately avoids Rust, Python, or C++ idioms.

### 7. Self-Describing and Self-Delimiting

Every tensor descriptor carries its own length in the first 10 bytes (readable before parsing the rest). Magic bytes (`HRRY`) and version fields allow format evolution. Optional sections (quantization, shard, statistics, extension type) are gated by flag bits and length-prefixed so readers can skip what they don't understand without rejecting the tensor.

### 8. Extensible by Design

Extension points — element type tags, layout tags, device tags, quantization scheme tags, flag bits, and optional sections — are stable across the lifetime of major version `1.x`. Implementation-private ranges (`0xF0`–`0xFE`) are reserved permanently; no named public value will ever be allocated into them. New public values go through the spec amendment process, not through implementation-defined registration. Older readers can skip unknown optional sections via length prefixes and parse a descriptor's shape and buffer table even when they cannot interpret an unknown layout or quantization scheme. See [`versioning.md`](versioning.md) § Extensibility Contract for the full normative definition.

### 9. Inference-Optimized Type System

Tier 1 types cover `float16`, `bfloat16`, `float32`, `float64`, signed/unsigned integers from `int4` to `int64`, `bool`. Tier 2 adds `float8_e4m3` and `float8_e5m2` for Blackwell/MI300X inference. Private extension tags (`0xF0`–`0xFE`) allow vendor-specific types. Each private extension tag MUST carry an inline descriptor encoding at minimum: bit width, packing, and floating-point parameters.

### 10. Multi-Transport Interchange

The interchange protocol covers in-process (pointer passing), IPC (shared memory), and cross-machine (streaming framing + optional RDMA data plane via GPUDirect). Layout negotiation is built into the protocol. Sender and receiver agree on the tensor format before data moves.

### 11. Array Database Foundation

Hurray is designed to serve as the storage layer of an array database engine — covering the full tensor supply chain: capture, storage, retrieval, and sharing. The tiled/blocked layout enables chunk-based access and tile-skipping for sub-array queries. Morton Z-order and Hilbert curve layouts preserve spatial locality across dimensions, improving range query cache performance. The file format footer index supports O(1) tensor lookup by name and is designed to be extensible with spatial or dimension-range indexes. Spec decisions that would foreclose chunk-based storage, spatial locality, or dimension-range indexing MUST be evaluated against this use case before being adopted.

### 12. Format Evolvability (Operational Rules for Extensibility)

Hurray defines normative rules for how the format changes over time, formally specifying how the extension surface from Core Property #11 is used in practice. Within major version `1.x`, the format is BACKWARD-compatible (a reader at minor `M` parses any minor `N ≤ M` correctly) and FORWARD_ADDITIVE (a reader at minor `M` correctly parses every part of newer-minor data that its own minor defines, ignores additive trailing bytes inside known length-prefixed sections, and rejects newer-minor data that uses an unknown flag bit or unknown public tag value). Public tag values are never rebound once allocated; deprecated tags retain their original semantics forever within `1.x`. A future major version is accompanied by a normative migration specification. Per-field tagging (Protobuf) and vtables (FlatBuffers) are explicitly rejected as incompatible with Hurray's zero-copy fixed-offset access model. See [`versioning.md`](versioning.md) § Evolvability Contract for the full normative definition.

---

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for the full annotated file tree.

---

## Key Design Invariants

These invariants apply to both formats unless noted:

- Descriptor always precedes its data buffer (both formats)
- No back-references within a tensor (both formats)
- **Streaming format only:** no end-of-file indexes; self-delimiting, no seek required
- **File format only:** footer index for random access; 4 KiB-aligned data buffers for mmap
- Strides expressed in logical elements, not bytes
- Little-endian throughout — no endianness negotiation
- 64-byte minimum buffer alignment for SIMD (streaming); 4 KiB minimum for mmap (file)
- `type_tag` (storage type) is orthogonal to `scheme_tag` (quantization scheme)

---

## Finding Open Questions and Decisions

- **Open questions** are marked inline in spec files as `> **[OQ-N]:**`. To find all open questions: `grep -rn "OQ-" docs/spec/`
- **Architectural decisions** are recorded in `docs/adr/`. Each ADR has a status (Draft / Accepted / Superseded).
- **Pending ideas and tasks** are in `TODO.md`.
- **Prior art** is surveyed in `docs/prior-art.md`.

---

## Prior Art

See `docs/prior-art.md` for the full research snapshot. Key references:

| Format / Protocol | Format type | Relevance |
|-------------------|-------------|-----------|
| DLPack | Streaming (in-process) | Closest existing tensor ABI; no quantization, limited layout metadata |
| Apache Arrow | Both (stream + file) | Buffer protocol and IPC framing inspiration; columnar, not tensor-focused |
| Apache Arrow Flight | Streaming (network) | Streaming RPC model reference; gRPC prevents true zero-copy at scale |
| SafeTensors | File | Simple safe serialization; footer index, mmap-friendly; no layout diversity, no quantization |
| GGUF | File | Block quantization encoding reference; typed KV metadata model; binary footer index |
| ONNX TensorProto | File | Type system breadth reference |
| Zarr v3 | File (chunked) | Chunk/shard layout and codec pipeline reference |
| NetCDF | File | Widely adopted scientific N-D array file format; no zero-copy, no quantization |
| OPeNDAP | Streaming (network) | De facto array data transport protocol in Earth Sciences; HTTP-based, not zero-copy |
| NIXL | Streaming (network) | NVIDIA tensor transfer library; RDMA transport but no tensor metadata vocabulary |
| NCCL + GPUDirect | Streaming (network) | GPU collective communications; raw buffer transfers, no layout or quantization metadata |
