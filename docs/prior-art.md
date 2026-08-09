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

### 2.17 MLX (Apple)

**What it is:** A NumPy-like array framework for Apple Silicon, designed for ML research and inference on CPU and GPU via Apple's unified memory architecture.

**How it works:** Arrays live in shared memory accessible by both the CPU and GPU without any buffer copies or explicit device placement. Operations carry a device tag; buffers do not. Computation is lazy — arrays materialize on demand, enabling graph-level optimization before execution. Interop with NumPy and PyTorch is via the Python buffer protocol and `__dlpack__`; most dtypes are truly zero-copy (bf16 and f64 require conversion for NumPy).

**Layout model:** Standard strided dense arrays. No tiled, blocked, or sparse layouts.

**Quantization:** Quantization is an operation, not a storage dtype. `mlx.core.quantize` produces packed `uint32` weight tensors plus separate scale and bias arrays. Affine scheme: group sizes 32/64/128, 2–8 bits per element, LSB-first packing (element 0 occupies bits 0–(bits-1) of the first word). Block floating-point modes: MXFP4/MXFP8 (shared E8M0 exponent, group=32) and NVFP4 (group=16, E4M3 per-group scales, no bias). No native int4/fp8 dtypes — quantized tensors are always stored in packed `uint32` arrays.

**On-disk format:** None native. MLX saves/loads `.npy`, `.npz`, `.safetensors`, and `.gguf`. This is a deliberate design choice, not an oversight.

**Interchange:** In-process only. No IPC, no streaming, no cross-language ABI beyond `mlx-c` (an unofficial C wrapper).

**Design insight — device is per-operation, not per-buffer:** Unlike CUDA or Metal, MLX's unified memory model eliminates the host/device buffer duality. A buffer has no device affinity; only the kernel that reads it does. This is a data point for Hurray's device-tag ADR: device ownership may be an attribute of *access* rather than of the buffer itself.

**Sub-byte packing cross-reference:** MLX's documented int4 LSB-first layout (element 0 in bits 0–3 of a `uint32`) should be checked against Hurray's sub-byte packing spec in `memory-layout.md` for ecosystem compatibility.

**Adoption:** Growing in the Apple Silicon developer community. Used by Hugging Face `mlx-lm`, `whisper.mlx`, and a range of on-device LLM inference tools.

**Assessment:** Not an interchange format — MLX deliberately reuses existing formats and focuses on the runtime compute layer. Its key contributions to Hurray's prior art are: (1) the unified-memory "device is per-operation" model as input to the device-tag design, and (2) its multi-mode quantization packing (LSB-first uint32, affine + MXFP + NVFP4, variable group sizes) as a cross-reference for Hurray's sub-byte packing and quantization specs.

---

### 2.18 Apache TVM (and MLC-LLM)

**What it is:** An open-source machine-learning *compiler* stack. It ingests models (PyTorch, ONNX, …), lowers them through a graph IR (Relay, now Relax) to a tensor IR (TIR), auto-tunes kernels (AutoTVM / Ansor / MetaSchedule), and emits optimized code for a wide range of backends (x86/ARM CPUs, CUDA/ROCm/Metal/Vulkan/WebGPU GPUs, microcontrollers). **MLC-LLM** is a TVM/Relax-based project that compiles and runs LLMs across heterogeneous consumer hardware (phones, laptops, browsers).

**Relationship to DLPack:** TVM comes from the same DMLC lineage that produced **DLPack** (see §2.1); its runtime `NDArray` is DLPack-native. The zero-copy ABI Hurray targets for in-process interop is, in effect, TVM's in-memory tensor format — so TVM validates that choice rather than competing with it.

**Layout model:** Not a storage format — TVM *transforms* layouts as a compilation concern, aggressively rewriting to hardware-preferred forms (tiling, packed `NCHWc`, tensor-core fragments) for kernel performance. It has no on-disk layout vocabulary of its own.

**Quantization:** Compilation-time passes (int8, and grouped/int4 weight quantization in MLC-LLM), not a normative on-disk representation. Quantized weights are TVM's problem to *generate and execute*, not to *interchange*.

**Interchange:** In-process via DLPack (NDArray). Model artifacts are compiled modules (`.so`/`.tar`) plus a parameter blob; there is no framework-agnostic, quantization/layout-aware interchange or streaming format — parameters are typically carried as NumPy-derived or ad-hoc bundles.

**Adoption:** Mature and influential; one of the pioneers of "compile a model to any hardware." MLC-LLM is widely used for on-device LLM inference.

**Assessment:** Orthogonal and complementary — TVM is *compute* (codegen, tuning, execution); Hurray is *interchange* (moving, storing, and describing tensor data). In the tensor supply chain, TVM is an execution stage and Hurray is transport + storage + description around it. Two concrete integration paths: (1) hand dense Tier-1 tensors to/from a TVM `NDArray` zero-copy via DLPack, exactly as with NumPy/PyTorch; (2) use Hurray's file format as the quantization- and layout-aware on-disk container for TVM/MLC-LLM weights (the role GGUF plays for llama.cpp) — MLC-LLM's shuttling of grouped-int4 weights across heterogeneous hardware sits squarely in Hurray's target domain. Hurray adds none of TVM's core value (no compiler, no autotuning); it fills the gap TVM leaves — a framework-agnostic, zero-copy interchange + storage format. TVM's layout rewrites are also a reference for which packed/tiled layouts a producer might hand over pre-optimized (Hurray's tiled/blocked layouts) to avoid a re-layout copy.

---

## 3. KV Cache Transfer in Disaggregated LLM Inference

The transport entries above (§§2.13–2.16: NIXL, NCCL, UCX, Arrow Flight) are
*primitives* — they move bytes. This section covers the *systems* built on top of
them: the disaggregated-inference architectures that move the **KV cache** between
machines as their central data-plane problem. They are surveyed separately because
they are the primary real-world consumers of a tensor-transfer layer, and because
they collectively expose the exact gap Hurray targets — they move KV cache buffers
with no self-describing tensor descriptor attached.

**Background — why the KV cache moves.** Autoregressive LLM inference has two
phases with opposite hardware profiles: **prefill** (process the whole prompt,
compute-bound, fills the KV cache) and **decode** (generate tokens one at a time,
memory-bandwidth-bound, reads and extends the KV cache). Co-locating them on the
same GPU couples their latency targets (time-to-first-token vs. time-per-output-token)
and wastes resources. **Disaggregated prefill/decode** runs the two phases on
separate GPUs or nodes — which means the KV cache produced by prefill, a tensor of
logical shape `[layers, 2, heads, seq_len, head_dim]` (the `2` is key + value), must
be transferred to the decode worker for every request. For long-context models this
is gigabytes per request, so the transfer is the defining engineering constraint.
The KV cache is almost always stored **paged** (vLLM PagedAttention style): a flat
pool of fixed-size blocks plus a per-sequence block table, so transfers move
non-contiguous block lists, not a single contiguous tensor.

---

### 3.1 DistServe

**What it is:** The OSDI 2024 research system (Zhong et al.) that introduced
disaggregating prefill and decoding onto separate GPUs to optimize *goodput*
(requests served within both TTFT and TPOT SLOs).

**KV cache transfer:** Layer-by-layer transfer of the KV cache from prefill to
decode instances. DistServe's key move is **bandwidth-aware placement**: it
co-locates the prefill and decode segments of a request so that the KV transfer
rides intra-node **NVLink** (≈600 GB/s peak between A100s), which renders the
transfer overhead "negligible" relative to recompute. Placement is chosen by a
search over parallelism configurations using simulation.

**Metadata model:** None beyond layer/block references. Both phases run the
*identical* model build with an identical KV layout that is assumed out-of-band;
the transfer carries numerical KV blocks only.

**Results:** Up to 7.4× more requests or 12.6× tighter SLO vs. co-located baselines.

**Assessment:** The foundational disaggregation paper. It established the
prefill/decode split and the insight that KV-transfer cost dominates placement —
but it sidesteps the metadata problem entirely by assuming homogeneous instances on
a high-bandwidth fabric. It is the academic root the production systems below build on.

---

### 3.2 Mooncake

**What it is:** The KVCache-centric disaggregated serving platform behind Kimi
(Moonshot AI), published at FAST 2025 / ACM ToS 2025 (arXiv 2407.00079) and
open-sourced as `kvcache-ai/Mooncake`. Runs across thousands of nodes serving
>100 billion tokens/day.

**Architecture:** A **disaggregated KVCache pool** that harvests the otherwise
idle CPU, DRAM, and SSD of the GPU cluster into a shared, tiered cache, fronted by a
**KVCache-centric scheduler** that maximizes effective throughput under SLOs and
maximizes prefix-cache reuse.

**Transfer Engine:** A standalone, reusable, zero-copy RDMA transfer library —
arguably Mooncake's most influential artifact. It aggregates **multiple RDMA NICs**
per host and uses **topology-aware path selection**: each server broadcasts a
topology matrix classifying NICs into preferred/secondary lists per memory region
(set at registration time), preferring local-NUMA / local-PCIe-switch GPUDirect
paths and failing over to alternate paths on error. Reported 87 GB/s (4×200 Gbps
RoCE) and 190 GB/s (8×400 Gbps), ~2.4–4.6× faster than TCP.

**Metadata model:** Transfers are keyed by KVCache block identifiers / offsets into
registered memory regions. The tensor's shape, dtype, and paged layout are properties
of the engine configuration, agreed out-of-band — the Transfer Engine moves
registered byte ranges, not described tensors.

**Assessment:** The most complete production realization of "KV cache as a
first-class, poolable, transferable object." Its Transfer Engine is direct prior art
for Hurray's RDMA data plane (OQ-2), and its KVCache pool is the strongest evidence
that the ecosystem wants to treat KV cache as durable, shareable tensor data — yet
it still carries no portable descriptor with the bytes.

---

### 3.3 vLLM disaggregated prefilling & the KVConnector API

**What it is:** vLLM's connector framework for disaggregated prefill and KV
offloading — the de facto integration point the rest of the ecosystem plugs into. All
implementations live under `vllm/distributed/kv_transfer`.

**Interface (`KVConnectorBase_V1`):** A role-split API. Scheduler-side methods
(`get_num_new_matched_tokens`, `update_state_after_alloc`, `build_connector_meta`,
`request_finished`) track which blocks to load/save and assemble per-step metadata;
worker-side methods (`register_kv_caches`, `start_load_kv`, `wait_for_layer_load`,
`save_kv_layer`, `wait_for_save`, `get_finished`) register GPU memory and run async
block transfers. This cleanly decouples transport from model logic.

**NixlConnector:** The RDMA implementation. A lazy **ZMQ side-channel handshake**
exchanges NIXL *agent identity* and *memory descriptors*; workers compute NIXL
descriptor IDs for block arrays, and it handles the messy case where prefiller and
decoder use **different tensor-parallelism degrees** (block-mapping reshuffle).
Other connectors: `MooncakeConnector`/`MooncakeStoreConnector`, `LMCacheMPConnector`,
`OffloadingConnector`, `MultiConnector`.

**Metadata model — the crux:** Shape, dtype, layout, and quantization are
established **once, out-of-band, at `register_kv_caches()` during startup**, and
assumed constant per layer. Per-transfer, only **raw block buffers + block/request
IDs** cross the wire. The handshake negotiates *addresses and agent identity*, never
a tensor descriptor.

**Assessment:** The single most important integration surface in the survey — and
the clearest statement of the gap. vLLM has already generalized "transfer the KV
cache" into a pluggable connector, but the thing flowing through that connector is an
opaque block buffer plus an ID. The connector boundary is exactly where a
self-describing Hurray descriptor would slot in.

---

### 3.4 NVIDIA Dynamo & TensorRT-LLM disaggregated serving

**What it is:** NVIDIA's production disaggregated-inference stack. **Dynamo** is the
datacenter-scale serving framework (disaggregated prefill/decode, GPU autoscaling,
KV-aware request routing); **TensorRT-LLM** is one of its backends; **NIXL** is the
transfer library (see §2.13); **KVBM** (KV Block Manager) is a framework-agnostic
unified memory layer usable standalone (`pip install kvbm`) or within Dynamo.

**KV Cache Manager:** Cost-aware, framework-agnostic (vLLM, TRT-LLM, SGLang,
PyTorch), tiering KV cache across GPU / CPU / SSD / filesystem / cloud via NIXL's
plugin backends to free GPU memory while preserving hit rates.

**Transfer & layout conversion:** TRT-LLM supports MPI, UCX, and NIXL backends over
RDMA/NVLink. Critically, when context and generation phases use **different
parallelism** (e.g. TP2 prefill vs. PP2 decode), TRT-LLM performs **"cache layout
conversion during transmission"** in a dedicated KV-cache-exchange module that is
"modularly decoupled from the KV cache manager and communication libraries." The
accompanying metadata, `ctx_params`, carries prompt tokens, the first generated
token, and communication parameters — i.e. *how to connect and what request this is*,
not *what shape/layout the tensor has*.

**Assessment:** The most operationally mature stack, and the most revealing on the
gap: NVIDIA had to hand-write a layout-conversion module to bridge mismatched
parallelism because there is no portable descriptor to drive a general conversion.
That bespoke reshuffle logic is precisely what a layout-aware interchange descriptor
exists to generalize.

---

### 3.5 llm-d

**What it is:** A Kubernetes-native distributed-inference framework (backed by Red
Hat, Google, IBM, and others) that treats prefill/decode disaggregation as a
first-class orchestration primitive, built on vLLM and an inference-aware gateway.

**KV cache transfer:** **NIXL-powered GPU-to-GPU** KV cache transfer over RDMA
(InfiniBand / RoCE), with cache-aware routing (route a request to the worker that
already holds its prefix) and a tiered storage hierarchy. v0.5 integrated the **UCCL**
backend into the NIXL networking layer to unify over vendor collectives (NCCL/RCCL/MCCL).
Reported ~70% higher throughput and ~88% faster TTFT vs. monolithic deployments.

**Metadata model:** Inherits vLLM's connector model (§3.3) for the actual KV
movement and adds Kubernetes-level routing/placement metadata. The KV-cache payload
itself is still opaque vLLM blocks.

**Assessment:** The Kubernetes-native packaging of everything above. Its existence
as a community standard (and NVIDIA Dynamo's stated cooperation with it) signals that
disaggregated KV transfer is consolidating into shared infrastructure — which is
exactly when a common descriptor format becomes valuable rather than premature.

---

### 3.6 LMCache

**What it is:** An open-source KV cache *layer* for vLLM and SGLang (arXiv
2510.09665) that extracts KV caches out of GPU memory and shares them across engines
and queries.

**Architecture:** A multi-tier store spanning GPU memory, CPU DRAM (pinned "hot
cache"), local disk, and remote backends (e.g. Redis), exposed through a modular KV
connector and a control API (pin, lookup, cleanup, move, compress). Performance comes
from batched data movement and compute/IO pipelining.

**Distinctive optimizations:** **CacheGen** (KV cache *compression* into a compact
bitstream for storage/transfer) and **CacheBlend** (reuse of *non-prefix* KV chunks).
Reports up to 15× throughput improvement with vLLM on multi-round QA and document
analysis. Both prefix-reuse offloading and PD-disaggregation transfer are supported.

**Metadata model:** Keyed KV chunks; with CacheGen the payload is a *compressed,
codec-specific* bitstream — so even the "bytes" are no longer a plain tensor buffer,
and the decoder must know the codec out-of-band.

**Assessment:** Important because it pushes furthest past raw-buffer transfer: it
compresses and reshapes KV cache for storage and reuse, which makes the *absence* of a
self-describing format most acute — a CacheGen blob is meaningless without out-of-band
knowledge of its shape, dtype, layout, and codec. A descriptor layer that can name a
quantized/compressed KV cache is directly relevant here.

---

### 3.7 The metadata gap

Across every system above, the same pattern holds: **the KV cache moves as opaque,
engine-private bytes, and everything needed to interpret those bytes is agreed
out-of-band.** Concretely:

- The logical tensor (`[layers, 2, heads, seq_len, head_dim]`), its dtype, its paged
  block layout, and its quantization scheme are fixed by the **model build /
  engine configuration**, not transmitted.
- What actually crosses the wire is **block buffers + block IDs** (vLLM, Mooncake,
  llm-d), **connection/request params** (`ctx_params` in TRT-LLM), or a
  **codec-specific compressed blob** (LMCache/CacheGen).
- The handshakes negotiate *addresses, agent identity, and TP mapping* — never a
  portable tensor descriptor.

The consequences are exactly the symptoms a descriptor format removes:

1. **Point-to-point coupling.** Every connector pair assumes identical engine builds.
   Cross-engine (vLLM ↔ TRT-LLM), cross-version, or cross-quantization KV transfer
   needs a bespoke adapter — there is no neutral interchange representation.
2. **Hand-written layout conversion.** Mismatched parallelism forces ad-hoc reshuffle
   code (TRT-LLM's exchange module, NixlConnector's TP mapping) instead of a
   descriptor-driven, general transform.
3. **Opaque storage.** A KV cache spilled to a pool (Mooncake) or compressed to disk
   (LMCache) carries no standardized self-description, so it is only reusable by the
   exact engine that wrote it.

| System | Primary role | KV transport | Travels *with* the bytes | Assumed out-of-band |
|---|---|---|---|---|
| DistServe | P/D disaggregation + placement | NVLink (intra-node) | layer/block refs | identical model + layout |
| Mooncake | KVCache pool + scheduler | Transfer Engine (multi-NIC RDMA) | block keys / offsets | shape, dtype, paged layout |
| vLLM KVConnector | engine transfer abstraction | NIXL / Mooncake / LMCache | raw blocks + block IDs | format fixed at `register_kv_caches()` |
| Dynamo / TRT-LLM | framework-agnostic KV mgr + serving | NIXL / UCX / MPI | `ctx_params` (tokens, conn params) | layout; cross-TP reshuffled by bespoke module |
| llm-d | K8s-native P/D orchestration | NIXL (+ UCCL backend) | block + routing hints | model identity, layout |
| LMCache | multi-tier KV cache layer | connector + batched movement | compressed KV chunks + keys | engine KV format, CacheGen codec |

This is the gap Hurray fills: a **self-describing tensor descriptor — shape, dtype,
layout, quantization — carried with the buffer**, so a KV cache becomes portable
across engines, versions, parallelism strategies, and storage tiers.

---

### 3.8 Design implications for Hurray

- **Block-paged layout descriptor (specified, Draft).** The PagedAttention KV cache is
  the motivating case for the Hurray `block-paged` layout tag (`0x0B`), now specified in
  [ADR-024](adr/ADR-024-block-paged-indirect-layout.md) and
  [Block-Paged](spec/layouts/block-paged.md) (Draft). It encodes
  fixed page size, the block table (logical sequence position → physical page ID),
  per-sequence page lists (addressed CSR-style by a `seq_ptr` offset array), and
  prefix sharing across sequences (expressed as aliased page IDs in the block table).
  With it, a paged KV cache becomes a *first-class Hurray tensor*: a consumer reads the
  descriptor and knows the page size, head/layer organization, dtype, and quantization
  without out-of-band agreement — turning TRT-LLM's bespoke "layout conversion during
  transmission" (§3.4) into a general, descriptor-driven transform.

- **Native buffer protocol (ADR-023).** Hurray's `__hurray_buffer__` /
  `from_hurray_buffer` zero-copy handoff lets these engines wrap their existing paged
  GPU buffers as Hurray tensors without a copy — the producer exposes the page buffers
  plus a descriptor; the consumer imports them. This is the in-process counterpart to
  the wire-format descriptor and the natural API for a vLLM `KVConnector`-style
  integration.

- **Relationship to NIXL / Transfer Engine (composition, not replacement).** NIXL
  (§2.13) and Mooncake's Transfer Engine (§3.2) are the **data plane** — they move
  registered buffers over RDMA. Hurray is the **metadata plane** — it describes what
  those buffers *are*. They compose: a Hurray descriptor names the KV cache; NIXL
  moves it. Hurray is explicitly **not** a serving system, scheduler, or cache pool —
  it is the interchange vocabulary those systems currently improvise per-connector.

---

### 3.9 Sources

KV-cache-transfer section compiled from primary sources (accessed June 2026):

- DistServe — [arXiv:2401.09670](https://arxiv.org/abs/2401.09670), [OSDI '24](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
- Mooncake — [arXiv:2407.00079](https://arxiv.org/abs/2407.00079), [USENIX FAST '25](https://www.usenix.org/conference/fast25/presentation/qin), [Transfer Engine design docs](https://kvcache-ai.github.io/Mooncake/design/transfer-engine/index.html), [github.com/kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake)
- vLLM disaggregated prefilling — [docs.vllm.ai/.../disagg_prefill](https://docs.vllm.ai/en/stable/features/disagg_prefill/), [KV cache transfer & connectors (DeepWiki)](https://deepwiki.com/vllm-project/vllm/9.4-kv-cache-transfer-and-disaggregated-serving)
- NVIDIA Dynamo / TensorRT-LLM — [Dynamo intro](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/), [Disaggregated Serving in TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html), [Reduce KV cache bottlenecks with Dynamo](https://developer.nvidia.com/blog/how-to-reduce-kv-cache-bottlenecks-with-nvidia-dynamo/)
- llm-d — [llm-d.ai P/D disaggregation guide](https://llm-d.ai/docs/guide/Installation/pd-disaggregation), [llm-d v0.5 blog](https://llm-d.ai/blog/llm-d-v0.5-sustaining-performance-at-scale), [NVIDIA Dynamo × llm-d](https://developer.nvidia.com/blog/nvidia-dynamo-accelerates-llm-d-community-initiatives-for-advancing-large-scale-distributed-inference/)
- LMCache — [arXiv:2510.09665](https://arxiv.org/html/2510.09665v2), [github.com/LMCache/LMCache](https://github.com/LMCache/LMCache), [architecture docs](https://docs.lmcache.ai/developer_guide/architecture.html)

---

## 4. Comparative Summary

| | DLPack | Arrow | SafeTensors | GGUF | ONNX | Zarr | NetCDF | OPeNDAP | NIXL | NCCL | Arrow Flight | MLX | Hurray (goal) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Zero-copy runtime** | ✅ | ✅ | Partial | Partial | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |
| **RDMA / GPU-direct** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | Optional |
| **Network streaming** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | Partial | ❌ | ✅ | ❌ | ✅ |
| **Layout negotiation** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Language-agnostic ABI** | ✅ | ✅ | ❌ | ❌ | Partial | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | Partial | ✅ |
| **Tiled/blocked layouts** | ❌ | ❌ | ❌ | ❌ | ❌ | Partial | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Strides** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| **Quantization metadata** | ❌ | ❌ | ❌ | ✅ (informal) | Partial | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | Partial | ✅ |
| **Sparsity descriptors** | ❌ | ❌ | ❌ | ❌ | Partial | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Sub-byte packing** | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| **On-disk storage** | ❌ | Partial | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | Partial |
| **Adoption** | High | High | High | High | High | Medium | High | Medium | Emerging | High | Medium | Medium | — |

---

## 5. The Gap Hurray Fills

No existing format combines:

1. **Standardized runtime interchange** (zero-copy, IPC, streaming) — Arrow has IPC but no tensor model; DLPack has zero-copy but no IPC.
2. **Rich layout metadata** — strides, tiled, panel-packed, sub-byte packed — sufficient for a consumer to know exactly what conversion it needs.
3. **Co-located quantization descriptors** — scale factors, zero points, block sizes, and scheme identifiers as first-class metadata, not afterthoughts.
4. **Language-agnostic ABI** — a stable C FFI boundary usable from any language.
5. **Alignment guarantees** — 64-byte minimum for SIMD; page-aligned for GPU/IPC — expressed in the spec, not left to convention.

The closest analogy: Parquet is to Zarr as Arrow is to Hurray. Existing formats cover storage well. The runtime interchange layer for tensor data remains genuinely open.

On the transport side, NIXL and NCCL solve the "move bytes fast" problem for GPU tensors using RDMA, but provide no tensor metadata vocabulary — no layout, no quantization, no descriptor. Arrow Flight provides the right streaming RPC model but uses gRPC throughout, which prevents zero-copy and alignment preservation for GB-scale tensor buffers. Hurray combines the descriptor layer missing from NIXL/NCCL with the streaming framing of Arrow Flight and an optional RDMA data plane hook.

The disaggregated-inference systems surveyed in §3 (DistServe, Mooncake, vLLM's KVConnector, NVIDIA Dynamo/TensorRT-LLM, llm-d, LMCache) make this gap concrete: they move KV cache buffers between prefill and decode workers continuously, yet every one of them agrees shape, dtype, paged layout, and quantization *out-of-band* and ships only opaque blocks plus IDs. That is the descriptor-layer gap Hurray fills — see §3.7 and §3.8.

---

## 6. Relevant Tensor Shape Reference

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

## 7. Memory Layout Quick Reference

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

## 8. Region-Heterogeneous Tensor Structures

The layouts surveyed in §2 each describe a *homogeneous* array: one element type, one
layout, one (optional) quantization scheme spanning the whole index space. A separate class
of prior art describes a *single logical array whose index space is partitioned into
regions that differ in structure* — some dense, some sparse, some constant, each with its
own backing storage and sometimes its own compression or precision. This is the class
Hurray's General Subpaving layout (tag 0x06, ADR-026) targets. It matters directly to the
array-database vision, where one very large tensor with structurally heterogeneous regions
is a first-class use case rather than an edge case.

Two distinct composition models appear in the literature, and they are not interchangeable:

- **Partition (exact-cover, non-overlapping):** the index space is tiled by disjoint boxes
  that together cover it exactly; each element belongs to exactly one region. This is what
  Hurray subpaving implements. Prior art: AMReX/Chombo `DisjointBoxLayout`, OpenVDB tiles,
  HDF5 Virtual Datasets (relaxed to allow gaps).
- **Overlay (overlapping composition):** a base spanning the whole index space plus one or
  more sparse corrections at scattered positions that share indices with the base. Prior
  art: SpQR / KVQuant outlier quantization, TileDB timestamped fragments. This model
  *cannot* be expressed by a non-overlapping partition and is out of scope for subpaving.

### 8.1 Comparison

| Structure | Segment | Partition shape | Per-region inner layout | Per-region buffers | Per-region quant/precision | Composition model | Maturity |
|---|---|---|---|---|---|---|---|
| AMReX / Chombo `BoxArray` / `DisjointBoxLayout` | HPC AMR | Irregular boxes | Uniform (dense FAB) | ✅ independent | ❌ | Partition (exact-cover per level) | Production (DOE Exascale) |
| OpenVDB / NanoVDB | VFX / graphics | Hierarchical tiles + 8³ leaves | Heterogeneous (constant tile vs dense leaf) | ✅ (linearized in NanoVDB) | Partial (per-node value quant) | Partition (hierarchical) | Production (ASWF standard) |
| HDF5 Virtual Dataset (VDS) | Scientific storage | Arbitrary rectangular selections | Heterogeneous (per source dataset) | ✅ per source | Via per-source compression | Partition, but permits gaps/overlap | Standard since HDF5 1.10 |
| TileDB dense array w/ sparse fragments | Array DB | N/A (temporal) | Dense + sparse fragments | ✅ per fragment | ❌ | Overlay (timestamped, last-writer-wins) | Production |
| Zarr v3 ZEP0003 variable chunks | Scientific storage | Rectilinear variable grid | Uniform | ✅ per chunk | ❌ (array-level codec) | Partition (rectilinear only) | Emerging (behind flag, Zarr-Python 3.2) |
| MLIR `sparse_tensor` encoding | ML compiler | Per-dimension level, not per-region | Per-*level* type only | N/A | ❌ | Neither (whole-tensor encoding) | Production (LLVM) |
| SpQR / KVQuant outliers | ML quant | Scattered points (not rectangular) | Dense low-bit + CSR outliers | ✅ (base + CSR) | ✅ (base vs outlier precision) | **Overlay** | Research → adoption |
| KIVI / HF residual KV cache | ML inference | Regular seq-axis split (recent/old) | Uniform (fp16 vs quantized) | ✅ | ✅ (per-region precision) | Partition (regular, 2 regions) | Production |
| MoE per-expert quantization | ML inference | Regular expert blocks | Uniform | ✅ per expert | ✅ (per-expert bit-width) | Partition (regular) | Research → adoption |
| Block-sparse attention (BigBird, FlexAttention) | ML inference | Regular block grid | Uniform dense + mask | ❌ | ❌ | Partition (regular) + mask | Production |
| ASTC texture blocks | GPU graphics | Regular block grid | Per-block mode/partition | ❌ (packed) | ✅ per-block | Partition (regular) | Production (hardware) |
| **Hurray General Subpaving (0x06)** | Interchange | **Irregular boxes** | **Any tag: dense/sparse/paged/nested** | **✅ per-region sub-table** | **✅ per-region (ADR-026 D5)** | **Partition (exact-cover, non-overlap)** | Draft |

### 8.2 Findings

- Irregular exact-cover partitioning with independent per-region buffers is a *mature*,
  production-proven pattern in HPC adaptive-mesh refinement (AMReX/Chombo) and VFX volume
  storage (OpenVDB/NanoVDB). NanoVDB's pointerless linearization of a heterogeneous-region
  tree is a direct precedent for encoding such a tensor as a zero-copy, GPU-friendly byte
  image — matching Hurray's streamability and zero-copy constraints.
- HDF5 Virtual Datasets are the closest standardized analog to Hurray subpaving: a logical
  N-D dataset defined as per-region mappings to heterogeneous backing storage. Notably, VDS
  chose *permissive* coverage (gaps and overlap allowed) where Hurray mandates exact-cover
  and non-overlap.
- Mainstream ML mostly wants per-region *quantization/precision* on *regular* partitions
  (KIVI recent/old KV split; per-expert MoE bit-widths), not irregular partitioning. This
  supports Hurray's per-region quantization mechanism more than its irregular-box
  generality.
- The dominant ML within-tensor heterogeneity pattern — outlier quantization (SpQR,
  KVQuant) — is an **overlay** (dense base + scattered sparse high-precision residual over
  shared indices). It is architecturally incompatible with subpaving's non-overlap
  constraint and is therefore explicitly out of scope for tag 0x06.

**Relevance to Hurray:** region-heterogeneous *partition* tensors are demanded and proven
in the HPC/scientific/graphics segments that the array-database vision targets, justifying
General Subpaving as a first-class layout; per-region quantization is independently demanded
in ML inference; but the largest ML heterogeneity pattern (outlier overlays) needs a
distinct overlapping-composition primitive that subpaving must not absorb.

---

*This document summarizes the state of the art as of April 2026 (KV cache transfer section, §3, added June 2026; region-heterogeneous tensor structures section, §8, added July 2026), based on the foundational research conversation that preceded the Hurray project and follow-up surveys. It is intended to inform the spec, architecture decisions, and implementation priorities.*
