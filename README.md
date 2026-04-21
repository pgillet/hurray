# Hurray

**Hurray** is a language-agnostic, zero-copy runtime interchange format for multi-dimensional tensor data, optimized for the memory layout diversity, quantization schemes, and access patterns of modern AI/ML inference pipelines.

Think Apache Arrow, but for tensors.

---

## Core Properties

These properties are the contract of the Hurray format. Every section of the specification MUST be consistent with all of them.

### 1. Zero-Copy First

Tensor data is shared across runtimes and languages via buffer handles, not copies. Buffers are aligned to 64 bytes minimum (SIMD), page-aligned for GPU/IPC. The format defines a stable C ABI so any language can participate without going through Rust. Quantization parameter arrays (scales, zero-points) live in separate buffer table entries — not interleaved with data — preserving zero-copy access to both.

### 2. Streamable by Design

The tensor descriptor always precedes its data buffer. A reader can start processing before the full payload arrives. No back-references, no end-of-file indexes, no buffering of the full input required. Both readers and writers operate incrementally. The format is self-delimiting: a receiver can determine the descriptor's total byte length from the first 10 bytes.

### 3. Rich Memory Layout Vocabulary

Ten layout types are defined: row-major, column-major, strided, tiled/blocked (for GEMM), Morton Z-order, Hilbert curve, general subpaving, and sparse COO/CSR/CSC. Strides are expressed in logical elements, not bytes. Negative and zero strides are valid. Sub-byte packing (int4, bool) is first-class.

### 4. First-Class Quantization

Five normative quantization schemes are defined: per-tensor affine, per-channel affine, per-block affine (GGUF family), NF4 (QLoRA), and MXFP (OCP Microscaling / NVIDIA Blackwell). The storage type (`type_tag`) is orthogonal to the quantization scheme (`scheme_tag`) — a tensor is quantized if and only if the `HAS_QUANTIZATION` flag is set. Dequantization formulas are normative.

### 5. Language-Agnostic

No Rust-isms leak into the format or the C FFI boundary. The binary spec uses generic type names (`int32`, `float16`, `uint8`). Any language that can read a struct from a buffer can implement it. The spec deliberately avoids Rust, Python, or C++ idioms.

### 6. Self-Describing and Self-Delimiting

Every tensor descriptor carries its own length in the first 10 bytes (readable before parsing the rest). Magic bytes (`HRRY`) and version fields allow format evolution. Optional sections (quantization, shard, statistics, extension type) are gated by flag bits and length-prefixed so readers can skip what they don't understand without rejecting the tensor.

### 7. Inference-Optimized Type System

Tier 1 types cover `float16`, `bfloat16`, `float32`, `float64`, signed/unsigned integers from `int4` to `int64`, `bool`. Tier 2 adds `float8_e4m3` and `float8_e5m2` for Blackwell/MI300X inference. Private extension tags (`0xF0`–`0xFE`) allow vendor-specific types. Each private extension tag MUST carry an inline descriptor encoding at minimum: bit width, packing, and floating-point parameters.

### 8. Multi-Transport Interchange

The interchange protocol covers in-process (pointer passing), IPC (shared memory), and cross-machine (streaming framing + optional RDMA data plane via GPUDirect). Layout negotiation is built into the protocol. Sender and receiver agree on the tensor format before data moves.

---

## Project Structure

```
hurray/
├── README.md                   # This file — project overview and format contract
├── CLAUDE.md                   # AI agent configuration and conventions
├── TODO.md                     # Running list of ideas and future tasks
├── Cargo.toml                  # Workspace root
├── docs/
│   ├── prior-art.md            # Research snapshot: formats, protocols, libraries
│   ├── spec/                   # Format specification (source of truth)
│   │   ├── README.md           # Scope, goals, RFC 2119 notice, versioning
│   │   ├── data-model.md       # Shape/dimension model
│   │   ├── element-types.md    # Element type system (int, float, quantized, custom)
│   │   ├── quantization.md     # Quantization schemes: per-tensor, per-channel, per-block, NF4, MXFP
│   │   ├── memory-layout.md    # Layout index and overview
│   │   ├── layouts/            # Per-layout spec files (10 layouts)
│   │   ├── buffer-protocol.md  # Zero-copy semantics, alignment, device memory
│   │   ├── metadata.md         # Tensor descriptor binary encoding
│   │   ├── interchange.md      # Runtime interchange: in-process, IPC, cross-machine
│   │   ├── versioning.md       # Format version field, compatibility policy
│   │   └── references.md       # Normative references
│   ├── impl/                   # Implementation requirements
│   │   ├── compliance.md       # Compliance checklist for implementors
│   │   ├── rust-reference.md   # Rust reference implementation guide
│   │   ├── c-ffi.md            # C FFI implementation guide
│   │   └── python-bindings.md  # Python bindings guide
│   └── adr/                    # Architecture Decision Records
│       └── ADR-NNN-*.md
├── hurray-core/                # Core types, no I/O, no async
├── hurray-io/                  # Async I/O: streaming + file format
├── hurray-ffi/                 # C ABI layer for language bindings
├── hurray-python/              # Python bindings (PyO3)
└── hurray-inspect/             # CLI hex viewer for Hurray descriptor files
```

---

## Key Design Invariants

These invariants must never be violated by any spec section or implementation:

- Descriptor always precedes its data buffer
- No back-references, no end-of-file indexes
- Strides expressed in logical elements, not bytes
- Little-endian throughout — no endianness negotiation
- 64-byte minimum buffer alignment; page-aligned for GPU/IPC
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

| Format / Protocol | Relevance |
|-------------------|-----------|
| DLPack | Closest existing tensor ABI; no quantization, limited layout metadata |
| Apache Arrow | Buffer protocol and IPC framing inspiration; columnar, not tensor-focused |
| Apache Arrow Flight | Streaming RPC model reference; gRPC prevents true zero-copy at scale |
| SafeTensors | Simple safe serialization; not a zero-copy runtime protocol |
| GGUF | Block quantization encoding reference (Q4_K, Q8_0, etc.) |
| ONNX TensorProto | Type system breadth reference |
| Zarr v3 | Chunk/shard layout and codec pipeline reference |
| NetCDF | Widely adopted scientific N-D array file format; no zero-copy, no quantization |
| OPeNDAP | De facto array data transport protocol in Earth Sciences; HTTP-based, not zero-copy |
| NIXL | NVIDIA tensor transfer library; RDMA transport but no tensor metadata vocabulary |
| NCCL + GPUDirect | GPU collective communications; raw buffer transfers, no layout or quantization metadata |
