# Prior Art: Tensor Memory Formats and Interchange Frameworks

**Status:** Research snapshot — April 2026
**Scope:** Formats, libraries, and protocols relevant to zero-copy, multi-layout tensor interchange for AI/ML inference pipelines.

---

## 1. Background: Why Tensor Interchange Is Hard

The fundamental mathematical operation in AI/ML workloads — the dot product, generalized to matrix multiplication — is sensitive to memory layout. The same data stored differently can be orders of magnitude faster or slower to process, depending on the hardware, the operation, and the cache hierarchy.

Efficient matrix multiplication requires:
1. **Tiling** the computation to fit the CPU cache hierarchy (L1 ~32–128 KB, L2 ~256 KB–2 MB, L3 ~8–128 MB per chip).
2. **Repacking** data into contiguous panel formats to eliminate stride penalties inside SIMD/Tensor Core kernels.
3. **Hardware-specific micro-layouts** for innermost kernels (NVIDIA Tensor Core fragment layouts, AMX tile registers, etc.).

No single layout is universally optimal. The right layout depends on the operation, the hardware, and which level of the memory hierarchy is the bottleneck. A general-purpose interchange format must therefore carry rich layout metadata, not mandate a single layout.

---

## 2. Existing Formats and Libraries

### 2.1 DLPack

**What it is:** A minimal open standard (originally from MXNet, now part of the Python Array API) for sharing tensor memory between frameworks without copying.

**How it works:** A `DLManagedTensor` struct carries a data pointer, shape, strides, device info, and dtype. A managed tensor struct wraps this with a destructor callback for lifetime management.

**Layout model:** Strides only. Can express row-major, column-major, and non-contiguous slices. Cannot express:
- Tiled/blocked layouts
- Morton/Z-order (bit-interleaved addressing)
- Panel-packed formats
- Sparse layouts (CSR, BSR, etc.)
- Sub-byte quantization block layouts

**Quantization:** None.

**Interchange:** In-process only (pointer passing). No IPC or streaming.

**Adoption:** Widely adopted — PyTorch, TensorFlow, JAX, CuPy, NumPy all support `__dlpack__`.

**Assessment:** The closest existing zero-copy tensor ABI. The layout model is the clearest gap relative to Hurray's goals.

---

### 2.2 Apache Arrow

**What it is:** A language-agnostic columnar memory format and IPC protocol, originally designed for tabular data.

**How it works:** A `RecordBatch` is a table of typed, named columns. Each column is backed by one or more flat buffers with defined alignment (64-byte minimum). The IPC format allows zero-copy reads via memory mapping.

**Tensor support:** `FixedShapeTensorArray` — an Arrow extension type wrapping tensors as fixed-shape arrays within an Arrow column. Supports row-major and column-major.

**Layout model:** Row-major or column-major. No tiled, blocked, or packed layouts. Consumers must repack before computing, which breaks the zero-copy promise.

**Quantization:** None in core. Extension types exist but are not standardized.

**Interchange:** Excellent — IPC, Flight (gRPC-based streaming), C Data Interface (ABI-stable FFI), shared memory.

**Assessment:** Excellent buffer model and IPC framing; fundamentally tabular. The right precedent for buffer management and IPC protocol design, not for tensor data modeling.

---

### 2.3 SafeTensors

**What it is:** A simple, safe serialization format for model weights, developed by Hugging Face.

**How it works:** A JSON header describing tensor metadata (dtype, shape, byte offsets) precedes raw binary tensor data. Memory-mappable: the header is small, and tensors can be read without deserializing the whole file.

**Layout model:** Row-major only. No strides, no tiling.

**Quantization:** None. Weights are stored in their native float16/bfloat16/float32.

**Interchange:** File-based only. No IPC, no streaming, no language-agnostic ABI.

**Safety:** Deliberately safe — cannot execute code on load (unlike PyTorch pickle).

**Adoption:** The dominant open-weights distribution format on Hugging Face Hub (LLaMA, Mistral, Falcon, etc.).

**Assessment:** Excellent for model distribution. Not designed for runtime interchange, quantized inference, or multi-layout pipelines.

---

### 2.4 GGUF

**What it is:** A self-contained model file format developed by the llama.cpp ecosystem, optimized for local CPU inference.

**How it works:** A single binary file containing model weights, tokenizer data, and hyperparameter metadata. Rich key-value metadata header. Supports memory-mapping for zero-copy reads.

**Layout model:** Row-major for unquantized tensors. Block-quantized layouts for quantized types (Q4_K, Q8_0, etc.) are stored as packed byte sequences with interleaved scale factors.

**Quantization:** Rich and practical — Q4_K_M, Q5_K_S, Q8_0, IQ4_XS, and more. Quantization schemes are llama.cpp-specific, not formally standardized.

**Interchange:** File-based only. Single-consumer oriented (the llama.cpp runtime or compatible loaders).

**Ecosystem:** Ollama, LM Studio, GPT4All, and most local inference tools use GGUF.

**Assessment:** Best-in-class for single-user local inference. Quantization schemes are practically excellent but informally specified. Not suited for multi-process or multi-language runtime interchange.

---

### 2.5 ONNX

**What it is:** A computation graph interchange format — a portable IR for neural network models.

**How it works:** A `.onnx` file is a Protocol Buffer containing a DAG of operator nodes, initializer tensors (weights), and input/output schemas. ONNX Runtime parses the graph and dispatches operators to pluggable execution provider backends (CPU, CUDA/TensorRT, CoreML, DirectML, OpenVINO).

**Layout model:** No layout control — the runtime decides. Quantization is supported via `QLinearMatMul`, `QLinearConv`, etc., but is second-class.

**Tensor interchange:** Partial. Tensors exist as graph edges and initializers, not as a standalone interchange primitive.

**Limitations:** Operator coverage gaps for new architectures; dynamic shapes are painful; Protobuf doesn't scale to large models; graph optimization is runtime-specific.

**Assessment:** For computation graph portability, not tensor data interchange. Not relevant to Hurray's runtime interchange goals.

---

### 2.6 Zarr v3

**What it is:** A format and library for chunked, compressed, cloud-friendly N-dimensional array storage.

**How it works:** Arrays are divided into fixed-size chunks stored as independent binary blobs, each compressed independently (Blosc, Zstd, LZ4, Gzip). Metadata is JSON. Chunks can be stored on a local filesystem, in a zip archive, or in a cloud object store.

**Layout model:** Chunk shape (tile shape) plus C-order or F-order within chunks. Rich codec pipeline for compression and transformations.

**Quantization:** None natively. Can be approximated via codecs.

**Interchange:** Storage-oriented. No IPC protocol, no shared-memory semantics. Compression is fundamental — incompatible with zero-copy runtime access.

**Relevance:** The conceptual complement to Hurray: Zarr for on-disk/cloud tensor storage; Hurray for in-process/IPC runtime interchange. Analogous to Parquet (storage) + Arrow (runtime) in the data engineering world.

**Assessment:** Best-in-class for tensor dataset storage. Not designed for runtime interchange.

---

### 2.7 NumPy (ndarray)

**What it is:** The de facto N-dimensional array standard in Python.

**How it works:** An `ndarray` carries a data pointer, shape, strides, dtype, and flags. Supports arbitrary strides — transpose, slice, and broadcast are all zero-copy view operations.

**Layout model:** Arbitrary strides (dense only). No tiled, packed, or sparse layouts.

**Interchange:** Python-ecosystem only via `__array_interface__` and `__dlpack__`. No language-agnostic ABI.

**Quantization:** None.

**Assessment:** The gold standard for strided dense arrays in Python. Its stride model is a useful reference; its Python coupling is a limiting factor for language-agnostic interchange.

---

### 2.8 Eigen (C++)

**What it is:** A high-performance C++ linear algebra library using expression templates.

**How it works:** Matrix storage order (row-major vs column-major) is a compile-time template parameter. Supports maps over external memory buffers with arbitrary strides. Lazy evaluation via expression templates avoids temporaries.

**Layout model:** Row-major or column-major at compile time. Arbitrary strides via `Map`. No tiled or packed layouts.

**Interchange:** None — purely in-process. No serialization or IPC.

**Assessment:** Excellent for in-process C++ computation. Not an interchange format.

---

### 2.9 xtensor (C++)

**What it is:** A C++ N-dimensional array library inspired by NumPy, with Python bindings via xtensor-python.

**How it works:** Supports row-major and column-major, arbitrary strides, lazy evaluation, and mapping over external buffers. Designed to be more Arrow-like than Eigen.

**Layout model:** Strided dense only.

**Interchange:** Partial — external buffer support enables some interop, but no formal protocol.

**Assessment:** More interop-friendly than Eigen, but still no formal interchange protocol.

---

### 2.10 HPC Libraries: PLASMA and SLATE

**PLASMA** — A parallel linear algebra library that natively uses tiled matrix layouts. Tiles are stored as individually allocated blocks, enabling asynchronous task-parallel execution on multicore CPUs. Not an interchange format.

**SLATE** — A modern redesign for distributed-memory linear algebra, supporting multiple layouts (column-major, tiled, band), mixed precision, and GPU offload. Closest to a multi-layout-aware library, but again not an interchange format.

**Assessment:** These demonstrate that tiled layouts are practically essential for high-performance GEMM. Their layout models are a useful reference for Hurray's memory-layout spec.

---

## 3. Comparative Summary

| | DLPack | Arrow | SafeTensors | GGUF | ONNX | Zarr | Hurray (goal) |
|---|---|---|---|---|---|---|---|
| **Zero-copy runtime** | ✅ | ✅ | Partial | Partial | ❌ | ❌ | ✅ |
| **IPC / streaming** | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Language-agnostic ABI** | ✅ | ✅ | ❌ | ❌ | Partial | ❌ | ✅ |
| **Tiled/blocked layouts** | ❌ | ❌ | ❌ | ❌ | ❌ | Partial | ✅ |
| **Strides** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Quantization metadata** | ❌ | ❌ | ❌ | ✅ (informal) | Partial | ❌ | ✅ |
| **Sparsity descriptors** | ❌ | ❌ | ❌ | ❌ | Partial | ❌ | ✅ |
| **Sub-byte packing** | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ |
| **Variable-shape (ragged)** | ❌ | Partial | ❌ | ❌ | Partial | ✅ | ✅ |
| **On-disk storage** | ❌ | Partial | ✅ | ✅ | ✅ | ✅ | Partial |
| **Adoption** | High | High | High | High | High | Medium | — |

---

## 4. The Gap Hurray Fills

No existing format combines:

1. **Standardized runtime interchange** (zero-copy, IPC, streaming) — Arrow has IPC but no tensor model; DLPack has zero-copy but no IPC.
2. **Rich layout metadata** — strides, tiled, panel-packed, sub-byte packed — sufficient for a consumer to know exactly what conversion it needs.
3. **Co-located quantization descriptors** — scale factors, zero points, block sizes, and scheme identifiers as first-class metadata, not afterthoughts.
4. **Language-agnostic ABI** — a stable C FFI boundary usable from any language.
5. **Alignment guarantees** — 64-byte minimum for SIMD; page-aligned for GPU/IPC — expressed in the spec, not left to convention.

The closest analogy: Parquet is to Zarr as Arrow is to Hurray. Existing formats cover storage well. The runtime interchange layer for tensor data remains genuinely open.

---

## 5. Relevant Tensor Shape Reference

Understanding typical shapes informs which layouts and alignment requirements matter most.

### LLM Weights (LLaMA-2 70B, float16)

| Tensor | Shape | Size |
|---|---|---|
| Token embedding | [128256, 8192] | ~2 GB |
| Attention Q/K/V projection | [8192, 8192] each | ~128 MB |
| FFN gate/up projection | [8192, 28672] | ~448 MB |

### LLM Activations (Dynamic)

| Tensor | Typical shape |
|---|---|
| Input token embeddings | [B=32, S=2048, D=8192] |
| Attention scores | [B=32, H=64, S=2048, S=2048] |
| KV cache (all layers) | [2, 80, 32, 64, 2048, 128] |

### Vision — CNN Feature Maps (NCHW)

| Layer | Shape |
|---|---|
| Input batch | [32, 3, 224, 224] |
| Early conv output | [32, 64, 112, 112] |
| Late conv output | [32, 2048, 7, 7] |

**Key thresholds:**
- Fits in L2/L3: `[64, 64]` — 8 KB float16
- Fits in GPU SRAM: `[2048, 2048]` — 8 MB float16
- Weight matrix: `[8192, 28672]` — 448 MB float16
- Pathological (quadratic attention): `[32, 64, 32768, 32768]` — terabyte scale

---

## 6. Memory Layout Quick Reference

The following layouts appear in production AI/ML inference pipelines:

| Layout | Best for | Notes |
|---|---|---|
| Row-major (C order) | GEMM A matrix, activations, attention scores | Default in most frameworks |
| Column-major (F order) | GEMM B matrix, some BLAS conventions | Eigen default |
| Tiled / blocked | High-intensity GEMM, convolutions | Requires repacking; cache-optimal |
| Panel-packed | Innermost GEMM kernel | Ephemeral; amortized repacking cost |
| NHWC | Inference convolutions | Channel-contiguous; SIMD-friendly |
| NCHW | Training convolutions (NVIDIA) | Historical cuDNN default |
| Paged (vLLM-style) | KV cache in autoregressive LLM serving | Variable-length sequence support |
| CSR / BSR | Sparse weight matrices | Post-pruning inference |
| Structured 2:4 | NVIDIA Sparse Tensor Cores | Requires metadata mask |
| Sub-byte packed (int4) | Quantized weights | Block structure with interleaved scales |

---

*This document summarizes the state of the art as of April 2026, based on the foundational research conversation that preceded the Hurray project. It is intended to inform the spec, architecture decisions, and implementation priorities.*
