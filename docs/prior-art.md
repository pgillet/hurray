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

### 2.10 NetCDF

**What it is:** Network Common Data Form — a widely adopted open standard for array-oriented scientific data (climate, oceanography, geophysics).

**How it works:** Files store N-dimensional variables with named dimensions, attributes, and a small set of primitive types (float32, float64, int16, int32, etc.). CDL (Common Data Language) provides a text representation. The classic format is based on XDR; NetCDF-4 uses HDF5 as the storage layer.

**Layout model:** Dense arrays only. Row-major (C order). No strides, no tiling, no sparse layouts.

**Quantization:** None. Scaling conventions (add_offset, scale_factor attributes) exist but are not standardized as first-class metadata.

**Interchange:** File-based. No in-process ABI, no IPC protocol, no zero-copy semantics.

**Adoption:** Very high in Earth Sciences, geospatial, and computational fluid dynamics communities. The Python ecosystem (xarray, netCDF4-python) relies on it heavily.

**Assessment:** A practical reference for N-dimensional array file formats with named dimensions and rich metadata conventions. Not designed for runtime interchange, quantized inference, or multi-layout pipelines.

---

### 2.11 OPeNDAP

**What it is:** Open-source Project for a Network Data Access Protocol — a de facto standard in the Earth Sciences community for remote access to scientific array data.

**How it works:** OPeNDAP defines a data model (based on NetCDF/DODS), a constraint expression language for server-side sub-setting and projection, and an HTTP-based transport protocol (DAP2 / DAP4). A client sends a constrained request (e.g., "variable X, indices [0:10, 50:100]"); the server computes the sub-set and streams the result as binary + metadata.

**Layout model:** Dense arrays, row-major. No tiling, no sparsity, no sub-byte packing.

**Quantization:** None.

**Interchange:** Network-only, request/response model. No zero-copy, no RDMA, no in-process ABI.

**Adoption:** High in Earth Sciences (NASA, NOAA, CMIP climate archives). Implemented by Hyrax (OPeNDAP server) and THREDDS Data Server.

**Assessment:** Prior art for server-side array sub-setting and streaming over HTTP. Demonstrates demand for a protocol that understands array structure (shapes, slices, variable names), not just raw bytes. Hurray's streaming and interchange goals operate in a similar problem space but target in-process / IPC / RDMA use cases rather than HTTP-based remote access.

---

### 2.12 HPC Libraries: PLASMA and SLATE

**PLASMA** — A parallel linear algebra library that natively uses tiled matrix layouts. Tiles are stored as individually allocated blocks, enabling asynchronous task-parallel execution on multicore CPUs. Not an interchange format.

**SLATE** — A modern redesign for distributed-memory linear algebra, supporting multiple layouts (column-major, tiled, band), mixed precision, and GPU offload. Closest to a multi-layout-aware library, but again not an interchange format.

**Assessment:** These demonstrate that tiled layouts are practically essential for high-performance GEMM. Their layout models are a useful reference for Hurray's memory-layout spec.

---

### 2.13 NIXL (NVIDIA Inference Xfer Library)

**What it is:** An open-source tensor transfer library from NVIDIA, designed specifically
for high-throughput tensor exchange in LLM inference pipelines (announced at GTC 2025).

**Primary use case:** KV cache migration in **disaggregated prefill/decode** inference
architectures, where the prefill (prompt processing) and decode (token generation) stages
run on different GPU nodes. The KV cache — a tensor of shape `[layers, 2, heads, seq_len,
head_dim]` — must be transferred between nodes at high speed for each request.

**Transport model:**
- Sender registers a GPU memory region (CUDA buffer) with the RDMA NIC using GPUDirect
  RDMA. The NIC can DMA directly from GPU memory without a GPU→CPU copy.
- The receiver pre-allocates an aligned GPU buffer and shares its RDMA memory key and
  remote address with the sender.
- The sender issues an RDMA Write (or Read) operation. The NIC moves data directly from
  source GPU memory to destination GPU memory over the network, with zero CPU involvement.
- Result: no CPU copies, no host-memory staging, no gRPC framing overhead.

**Transport backends:** UCX (InfiniBand, RoCE), NVIDIA GDS (GPU Direct Storage), NVMe-oF.

**What it does not define:**
- Any format for tensor descriptor, layout, or quantization metadata.
- Layout negotiation — it assumes both sides agree on the tensor format out-of-band.
- Multi-layout support — it transfers raw byte buffers, not structured tensor objects.

**Adoption:** Used by vLLM (disaggregated prefill/decode), NVIDIA TensorRT-LLM, Dynamo
(NVIDIA's inference scheduler).

**Assessment:** The closest existing prior art to what Hurray's RDMA data plane (OQ-2)
would implement at the transport layer. NIXL solves the "move bytes fast" problem but
provides no tensor metadata, layout, or quantization vocabulary — which is exactly what
Hurray's descriptor layer adds on top.

---

### 2.14 NCCL + GPUDirect RDMA

**What it is:** NVIDIA Collective Communications Library — the standard library for
GPU-to-GPU communication in distributed ML workloads.

**How it works:** NCCL implements collective operations (AllReduce, AllGather,
ReduceScatter, Broadcast, Send/Recv) over GPU tensors. When both GPUs are on different
nodes connected via InfiniBand or RoCE, NCCL uses GPUDirect RDMA: the NIC reads from
source GPU memory and writes to destination GPU memory without host CPU involvement.

**Tensor model:** None. NCCL operates on flat GPU buffers — a pointer, a count, and a
dtype. All layout semantics are handled by the caller.

**Use in inference:** Tensor parallelism (splitting a model's weight matrices across GPUs)
and pipeline parallelism (splitting model layers across nodes) use NCCL point-to-point
(Send/Recv) for activation tensors. These are increasingly used in large-model inference
(not just training).

**Assessment:** NCCL provides the RDMA transport primitive; it does not define any tensor
interchange protocol. It is the incumbent for GPU collective communications, but it
provides no layout negotiation, no streaming framing, and no quantization support.

---

### 2.15 UCX (Unified Communication X)

**What it is:** An open-source communication framework that abstracts over multiple
high-performance transport layers: InfiniBand (verbs), RoCE, TCP/IP, shared memory,
and CUDA IPC.

**How it works:** UCX provides a unified API (`ucp_put_nb`, `ucp_get_nb`, `ucp_send_nb`,
etc.) for RDMA put/get, atomic, and stream operations, dispatching to the best available
transport on a per-connection basis. Fallback to TCP is automatic when RDMA is unavailable.

**Role in the ecosystem:** UCX is not a user-facing protocol. It is the transport layer
used by:
- OpenMPI / MVAPICH (HPC MPI)
- NCCL (via its verbs backend)
- NIXL (primary transport backend)
- Ray (distributed object store transfers)

**Relevance to Hurray:** Hurray's OQ-2 (RDMA data plane) would likely sit above UCX in
the software stack. UCX handles the actual RDMA operation; the Hurray protocol handles
memory registration handshake, descriptor exchange, and session management.

**Assessment:** UCX is the practical implementation substrate for any RDMA-based Hurray
transport, not a format or protocol to evaluate as prior art per se. Worth knowing as
the layer Hurray's RDMA extension must integrate with.

---

### 2.16 Apache Arrow Flight

**What it is:** A gRPC-based RPC framework for high-performance Arrow data exchange,
built on top of the Arrow IPC format.

**How it works:** Arrow Flight defines a set of RPCs — `DoGet` (server streams data to
client), `DoPut` (client streams data to server), `DoExchange` (bidirectional), and
metadata calls (`GetFlightInfo`, `ListFlights`). Data travels as `FlightData` messages,
each containing an Arrow IPC message header and a raw data body.

**Transport:** gRPC over HTTP/2. All data — control and payload — goes through gRPC
streaming. No RDMA support. Implementations can achieve ~2–3 GB/s on fast LAN.

**Layout model:** Inherits Arrow's limitations: row-major or column-major only, no
tiled or packed layouts, no quantization.

**Strengths:** Excellent for columnar record batch interchange. Simple RPC model. Good
ecosystem (Java, C++, Python, Go, Rust clients).

**Weaknesses for tensor workloads:**
- gRPC serialization requires at least one CPU copy per message — incompatible with
  true zero-copy for GB-scale tensor buffers.
- Buffer alignment is not preserved through gRPC: receivers must copy to aligned memory
  before passing to GPU or BLAS kernels.
- No layout negotiation, no quantization metadata, no device memory support.

**Assessment:** The primary design inspiration for Hurray's network transport protocol.
Hurray adopts Arrow Flight's streaming RPC model (descriptor before data, typed messages,
bidirectional exchange) and extends it with layout negotiation, extension layout entry
encoding, parallel shard transfers, and a hook for an RDMA data plane.

---

## 3. Comparative Summary

| | DLPack | Arrow | SafeTensors | GGUF | ONNX | Zarr | NetCDF | OPeNDAP | NIXL | NCCL | Arrow Flight | Hurray (goal) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Zero-copy runtime** | ✅ | ✅ | Partial | Partial | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ |
| **RDMA / GPU-direct** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | Optional |
| **Network streaming** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | Partial | ❌ | ✅ | ✅ |
| **Layout negotiation** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Language-agnostic ABI** | ✅ | ✅ | ❌ | ❌ | Partial | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ |
| **Tiled/blocked layouts** | ❌ | ❌ | ❌ | ❌ | ❌ | Partial | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Strides** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Quantization metadata** | ❌ | ❌ | ❌ | ✅ (informal) | Partial | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Sparsity descriptors** | ❌ | ❌ | ❌ | ❌ | Partial | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Sub-byte packing** | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **On-disk storage** | ❌ | Partial | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | Partial |
| **Adoption** | High | High | High | High | High | Medium | High | Medium | Emerging | High | Medium | — |

---

## 4. The Gap Hurray Fills

No existing format combines:

1. **Standardized runtime interchange** (zero-copy, IPC, streaming) — Arrow has IPC but no tensor model; DLPack has zero-copy but no IPC.
2. **Rich layout metadata** — strides, tiled, panel-packed, sub-byte packed — sufficient for a consumer to know exactly what conversion it needs.
3. **Co-located quantization descriptors** — scale factors, zero points, block sizes, and scheme identifiers as first-class metadata, not afterthoughts.
4. **Language-agnostic ABI** — a stable C FFI boundary usable from any language.
5. **Alignment guarantees** — 64-byte minimum for SIMD; page-aligned for GPU/IPC — expressed in the spec, not left to convention.

The closest analogy: Parquet is to Zarr as Arrow is to Hurray. Existing formats cover storage well. The runtime interchange layer for tensor data remains genuinely open.

On the transport side, NIXL and NCCL solve the "move bytes fast" problem for GPU tensors using RDMA, but provide no tensor metadata vocabulary — no layout, no quantization, no descriptor. Arrow Flight provides the right streaming RPC model but uses gRPC throughout, which prevents zero-copy and alignment preservation for GB-scale tensor buffers. Hurray combines the descriptor layer missing from NIXL/NCCL with the streaming framing of Arrow Flight and an optional RDMA data plane hook.

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
