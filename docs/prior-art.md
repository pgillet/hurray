# Prior Art: Tensor Data Interchange Solutions and Compute Frameworks

**Status:** Research snapshot — restructured September 2026
**Scope:** The formats, protocols, libraries, and serving systems that move, store, or
consume tensor data in AI/ML inference pipelines, and what each one leaves undone.

---

## 1. Executive Summary

This survey exists to inform the design of **Hurray**, a zero-copy tensor interchange
protocol for AI/ML inference. It maps the landscape into two roles that the literature
usually blurs: solutions that **move or store** tensor data, and frameworks that
**compute on** it. The distinction matters because the second group depends on the first,
and every gap in the first becomes bespoke glue code in the second.

The survey finds that no existing solution carries a complete tensor description with the
bytes. DLPack shares pointers in-process but models only strides — no tiling, no
quantization, no packing. Apache Arrow contributes an excellent buffer model and IPC
framing, but its data model is tabular and its Flight RPC copies through gRPC, which
forfeits zero-copy and alignment at exactly the sizes that matter. GGUF specifies
quantization well but only for one runtime and only on disk. NIXL, NCCL, and UCX move GPU
memory at line rate over RDMA yet transfer anonymous byte ranges — they define no
descriptor at all. The consequence appears most sharply in disaggregated LLM inference,
where the KV cache moves between machines continuously: every production system surveyed
here (vLLM, Mooncake, TensorRT-LLM/Dynamo, llm-d, LMCache) agrees shape, dtype, layout,
and quantization **out-of-band** and ships opaque blocks plus IDs, which forces
hand-written layout conversion whenever the two sides disagree.

Hurray's value proposition follows directly: combine **Arrow Flight's streaming RPC
model** (descriptor before data, typed messages, bidirectional exchange) with an
**RDMA transport** in NIXL's mould, and put a **rich layout and quantization descriptor**
on the wire between them. Compute frameworks stay the clients they already are — Hurray
adds no kernels, no scheduler, and no cache pool — but they get a portable tensor
description instead of a per-connector private agreement.

---

## 2. Two Roles: Interchange versus Compute

The survey classifies every entry into exactly one of two roles, and states that role on
each entry.

**Data interchange solutions** move or store tensor data. They define a byte layout, a
descriptor, a wire protocol, a file container, or a transport — never arithmetic.
Examples: DLPack, Apache Arrow, Arrow Flight, SafeTensors, GGUF, Zarr, NetCDF, OPeNDAP,
NIXL, NCCL, UCX. Hurray belongs here.

**Compute frameworks and libraries** perform computation on tensors: kernels, graph
execution, autotuning, request scheduling. Examples: PyTorch, TensorFlow, JAX, NumPy,
Eigen, xtensor, PLASMA, SLATE, TVM, MLC-LLM, MLX, vLLM, NVIDIA Dynamo.

**Compute frameworks are the clients of interchange solutions.** PyTorch computes; DLPack
carries a tensor into and out of it. vLLM schedules and generates tokens; NIXL moves its
KV cache. When an interchange solution cannot express something — a tiled layout, a
quantization scheme, a paged KV cache — the client absorbs the cost, usually as a copy, a
repack, or a private side-channel. Reading the survey through that dependency is what
turns a list of formats into a list of design requirements.

One entry sits on the boundary and is labelled where it does the most work: NumPy is an
array library — a compute client — whose stride model is nonetheless a reference for
interchange design (§ 5.1.1).

---

## 3. Background: Why Tensor Interchange Is Hard

**No single layout is optimal.** Matrix multiplication is sensitive to memory layout:
the same numbers stored differently run orders of magnitude apart. Fast kernels tile the
computation to fit the cache hierarchy, repack operands into contiguous panels to remove
stride penalties inside SIMD and Tensor Core inner loops, and use hardware-specific
micro-layouts at the innermost level. The right choice depends on the operation, the
hardware, and which level of the memory hierarchy is saturated. An interchange format
therefore cannot mandate one layout; it must **describe** whichever layout the producer
already has, so the consumer knows precisely what conversion — if any — it needs.

**Zero-copy is not a micro-optimization at inference scale.** A single LLaMA-2 70B weight
matrix is 448 MB; a long-context KV cache runs to gigabytes per request. Copying such a
buffer to satisfy an alignment rule or to reshape it into the consumer's expected layout
costs memory bandwidth that the actual computation needs, and it doubles peak residency at
the moment GPU memory is scarcest.

**Disaggregated inference makes this the dominant cost.** Autoregressive generation has
two phases with opposite hardware profiles: *prefill* processes the whole prompt
(compute-bound, fills the KV cache) and *decode* emits tokens one at a time
(memory-bandwidth-bound, reads and extends the cache). Running them on the same GPU
couples two latency targets that want different hardware, so production systems split them
across GPUs or nodes — and then the KV cache produced by prefill must reach the decode
worker for every request. That transfer, not the arithmetic, becomes the latency budget.
It is also where an inexpressive interchange layer hurts most: if the descriptor cannot
say "paged, 16 tokens per page, int8 per-channel quantized", the two sides must agree
out-of-band and someone hand-writes the conversion when they disagree. § 6 documents that
this is exactly what every production system does today.

---

## 4. Data Interchange Solutions

Grouped by primary use case: in-process ABI, IPC/streaming, file-based, RDMA/transport.

### 4.1 In-Process ABI

#### 4.1.1 DLPack

| | |
|---|---|
| **Role** | Data interchange solution |
| **Type** | In-process ABI |
| **Use case** | Zero-copy tensor sharing between frameworks in one process |
| **Layout model** | Strided (dense only) |
| **Quantization** | None |
| **Interchange method** | Pointer passing |
| **Adoption** | Very high — PyTorch, TensorFlow, JAX, CuPy, NumPy, TVM, MLX |

A minimal open standard, originally from MXNet and now part of the Python Array API, for
sharing tensor memory between frameworks without copying. A `DLManagedTensor` carries a
data pointer, shape, strides, device, and dtype; a managed wrapper adds a destructor
callback for lifetime handoff. Strides express row-major, column-major, and non-contiguous
slices — and nothing else. It cannot describe tiled or blocked layouts, Morton order,
panel-packed formats, sparse structures, or sub-byte quantization blocks.

- ✅ Best-in-class for in-process tensor sharing; near-universal adoption
- ❌ Strides only — no tiled, packed, sparse, or sub-byte layouts
- ❌ No quantization vocabulary, no IPC, no streaming, no file form
- 🔹 Hurray keeps the same handoff shape and extends the layout and quantization vocabulary

### 4.2 IPC and Streaming

#### 4.2.1 Apache Arrow

| | |
|---|---|
| **Role** | Data interchange solution |
| **Type** | IPC / streaming (columnar) |
| **Use case** | Language-agnostic in-memory analytics, zero-copy IPC |
| **Layout model** | Row-major or column-major, tabular |
| **Quantization** | None in core |
| **Interchange method** | IPC, shared memory, C Data Interface (ABI-stable FFI) |
| **Adoption** | Very high across data engineering |

A language-agnostic columnar memory format and IPC protocol built for tabular data. A
`RecordBatch` is a table of typed, named columns, each backed by flat buffers with a
64-byte minimum alignment; the IPC format supports zero-copy reads by memory mapping.
Tensors exist through the `FixedShapeTensorArray` extension type, which wraps fixed-shape
tensors inside a column and supports row-major and column-major only. A consumer that
needs a tiled or packed layout must repack, which breaks the zero-copy promise.

- ✅ The reference buffer model and IPC framing; alignment is specified, not assumed
- ❌ Fundamentally tabular — tensors are an extension, not the data model
- ❌ No tiled, blocked, or packed layouts; no standardized quantization
- 🔹 Hurray borrows the buffer discipline and IPC design, applied to a tensor data model

#### 4.2.2 Apache Arrow Flight

| | |
|---|---|
| **Role** | Data interchange solution |
| **Type** | IPC / streaming RPC |
| **Use case** | High-throughput record-batch transfer over a network |
| **Layout model** | Inherits Arrow's — row-major or column-major |
| **Quantization** | None |
| **Interchange method** | gRPC over HTTP/2 (`DoGet`, `DoPut`, `DoExchange`) |
| **Adoption** | Medium — Java, C++, Python, Go, Rust clients |

A gRPC-based RPC framework layered on Arrow IPC. `DoGet` streams server→client, `DoPut`
client→server, `DoExchange` runs bidirectionally, and metadata calls (`GetFlightInfo`,
`ListFlights`) describe what is available. Each `FlightData` message pairs an Arrow IPC
header with a raw body. Implementations reach roughly 2–3 GB/s on a fast LAN.

For tensor workloads the transport is the limit: gRPC forces at least one CPU copy per
message, and it does not preserve buffer alignment, so receivers copy again before handing
memory to a GPU or a BLAS kernel. At GB scale that is the whole cost.

- ✅ The right streaming RPC shape: typed messages, descriptor before data, bidirectional
- ❌ gRPC framing forecloses zero-copy and destroys alignment for large buffers
- ❌ No layout negotiation, no quantization metadata, no device memory, no RDMA
- 🔹 Hurray adopts the RPC model and replaces the transport assumptions with an
  alignment-preserving framing plus an RDMA data plane

#### 4.2.3 OPeNDAP

| | |
|---|---|
| **Role** | Data interchange solution |
| **Type** | IPC / streaming (network request–response) |
| **Use case** | Remote sub-setting of scientific array data over HTTP |
| **Layout model** | Dense, row-major |
| **Quantization** | None |
| **Interchange method** | HTTP (DAP2 / DAP4) with a constraint expression language |
| **Adoption** | High in Earth Sciences — NASA, NOAA, CMIP archives |

A data model derived from NetCDF, a constraint language for server-side projection and
sub-setting, and an HTTP transport. A client asks for "variable X, indices [0:10,
50:100]"; the server computes the slice and streams binary data plus metadata. There is no
zero-copy path, no RDMA, and no in-process ABI.

- ✅ Proves demand for a protocol that understands array structure, not just bytes
- ✅ Server-side sub-setting is a genuine precedent for shard/slice requests
- ❌ HTTP request–response only; no zero-copy, no device memory, no quantization
- 🔹 Hurray targets the same "ask for part of an array" need over in-process, IPC, and
  RDMA paths instead of HTTP

### 4.3 File-Based

#### 4.3.1 SafeTensors

| | |
|---|---|
| **Role** | Data interchange solution |
| **Type** | File-based |
| **Use case** | Safe distribution of model weights |
| **Layout model** | Row-major only |
| **Quantization** | None — native float16/bfloat16/float32 |
| **Interchange method** | File, memory-mappable |
| **Adoption** | High — the default on Hugging Face Hub |

A JSON header listing dtype, shape, and byte offsets, followed by raw tensor bytes. The
header is small, so individual tensors can be read without deserializing the file, and
loading cannot execute code — the deliberate answer to PyTorch pickle.

- ✅ Excellent, safe, memory-mappable weight distribution
- ❌ Row-major only; no strides, no tiling, no quantization descriptors
- ❌ File-only: no IPC, no streaming, no language-agnostic ABI
- 🔹 Hurray covers the same on-disk case while also serving runtime interchange

#### 4.3.2 GGUF

| | |
|---|---|
| **Role** | Data interchange solution |
| **Type** | File-based |
| **Use case** | Self-contained local LLM inference artifacts |
| **Layout model** | Row-major; block-quantized packed sequences for quantized types |
| **Quantization** | Rich but informally specified (Q4_K_M, Q5_K_S, Q8_0, IQ4_XS, …) |
| **Interchange method** | File, memory-mappable |
| **Adoption** | High — llama.cpp, Ollama, LM Studio, GPT4All |

One binary file carrying weights, tokenizer, and hyperparameters behind a rich key-value
metadata header. Quantized tensors are packed byte sequences with interleaved scale
factors. The schemes are practically excellent and llama.cpp-specific: they are defined by
the implementation rather than by a portable specification.

- ✅ The strongest working proof that quantization metadata belongs in the format
- ✅ Best-in-class for single-user local inference
- ❌ Schemes are runtime-specific and informally specified
- ❌ File-only and effectively single-consumer; no multi-process runtime interchange
- 🔹 Hurray specifies equivalent block-quantization schemes normatively and carries them
  through IPC and RDMA as well as on disk

#### 4.3.3 Zarr v3

| | |
|---|---|
| **Role** | Data interchange solution |
| **Type** | File-based / object storage |
| **Use case** | Chunked, compressed, cloud-native N-dimensional array storage |
| **Layout model** | Chunk (tile) shape plus C or F order within chunks |
| **Quantization** | None natively; approximated with codecs |
| **Interchange method** | File / object store, JSON metadata |
| **Adoption** | Medium–high in scientific computing |

Arrays split into fixed-size chunks, each an independently compressed blob (Blosc, Zstd,
LZ4, Gzip), stored on a filesystem, in a zip, or in an object store. Compression is
central to the design, which puts it at odds with zero-copy runtime access.

- ✅ The reference model for chunked, cloud-native array storage
- ✅ Chunk grids are a useful precedent for tiled layout description
- ❌ Compression-first; no IPC protocol, no shared-memory semantics
- 🔹 Complementary, not competing: Zarr is to storage what Hurray is to runtime
  interchange — the Parquet/Arrow relationship, one level up in dimensionality

#### 4.3.4 NetCDF

| | |
|---|---|
| **Role** | Data interchange solution |
| **Type** | File-based |
| **Use case** | Array-oriented scientific data (climate, oceanography, geophysics) |
| **Layout model** | Dense, row-major |
| **Quantization** | None (`scale_factor`/`add_offset` are conventions, not metadata) |
| **Interchange method** | File (classic XDR-based; NetCDF-4 over HDF5) |
| **Adoption** | Very high in Earth Sciences; xarray and netCDF4-python depend on it |

N-dimensional variables with named dimensions, attributes, and a small primitive type set,
with a text representation (CDL) alongside the binary form.

- ✅ Named dimensions and rich attribute conventions are worth borrowing
- ❌ Dense row-major only; no strides, tiling, or sparsity
- ❌ No in-process ABI, no IPC, no zero-copy semantics
- 🔹 Hurray adopts the descriptive-metadata instinct without the file-only constraint

### 4.4 RDMA and Transport

#### 4.4.1 NIXL (NVIDIA Inference Xfer Library)

| | |
|---|---|
| **Role** | Data interchange solution |
| **Type** | RDMA / transport |
| **Use case** | KV cache migration between prefill and decode workers |
| **Layout model** | None — moves registered byte ranges |
| **Quantization** | None |
| **Interchange method** | RDMA (GPUDirect), via UCX / GDS / NVMe-oF backends |
| **Adoption** | Emerging but fast — vLLM, TensorRT-LLM, Dynamo, llm-d |

An open-source tensor transfer library from NVIDIA (announced at GTC 2025), built for
high-throughput exchange in LLM inference. The sender registers a GPU memory region with
the RDMA NIC using GPUDirect RDMA; the receiver pre-allocates an aligned GPU buffer and
shares its memory key and remote address; the sender issues an RDMA Write or Read and the
NIC moves data GPU-to-GPU across the network with no CPU involvement, no host staging, and
no gRPC framing.

What it deliberately does not define: any tensor descriptor, any layout or quantization
metadata, and any layout negotiation. Both sides must already agree on the format.

- ✅ The best available data plane — genuine GPU-to-GPU zero-copy at line rate
- ❌ Transfers anonymous byte ranges; no tensor vocabulary whatsoever
- ❌ Format agreement is assumed out-of-band, per connector pair
- 🔹 Hurray composes with it rather than replacing it: Hurray names the tensor, NIXL moves
  it

#### 4.4.2 NCCL + GPUDirect RDMA

| | |
|---|---|
| **Role** | Data interchange solution |
| **Type** | RDMA / transport (collectives) |
| **Use case** | GPU-to-GPU collective and point-to-point communication |
| **Layout model** | None — flat buffers (pointer, count, dtype) |
| **Quantization** | None |
| **Interchange method** | RDMA / NVLink collectives |
| **Adoption** | Very high — the incumbent for distributed training and inference |

NVIDIA's collective communications library: AllReduce, AllGather, ReduceScatter,
Broadcast, and Send/Recv over GPU tensors, using GPUDirect RDMA across InfiniBand or RoCE.
Tensor-parallel weight splits and pipeline-parallel activation handoffs both ride NCCL
point-to-point, increasingly in inference and not only training.

- ✅ The default, hardware-tuned transport primitive for GPU clusters
- ❌ No tensor model at all — layout semantics belong entirely to the caller
- ❌ No streaming framing, no negotiation, no quantization
- 🔹 Hurray supplies the descriptor layer that callers currently keep in their own heads

#### 4.4.3 UCX (Unified Communication X)

| | |
|---|---|
| **Role** | Data interchange solution |
| **Type** | RDMA / transport abstraction |
| **Use case** | Transport-agnostic RDMA put/get, atomics, and streams |
| **Layout model** | None |
| **Quantization** | None |
| **Interchange method** | InfiniBand verbs, RoCE, TCP, shared memory, CUDA IPC |
| **Adoption** | High as infrastructure — OpenMPI, MVAPICH, NCCL, NIXL, Ray |

A unified API (`ucp_put_nb`, `ucp_get_nb`, `ucp_send_nb`) that dispatches to the best
available transport per connection and falls back to TCP when RDMA is unavailable. UCX is
not a user-facing protocol; it is the substrate other systems build on.

- ✅ The practical implementation target for any RDMA-based transport
- ✅ Automatic transport selection and TCP fallback are a solved problem worth reusing
- ❌ Not a format: nothing to describe a tensor
- 🔹 Hurray's RDMA data plane sits above UCX — UCX performs the transfer; Hurray handles
  registration handshake, descriptor exchange, and session state

### 4.5 Comparison: Data Interchange Solutions

| Solution | Type | Layout model | Quantization | Interchange method | RDMA | Adoption | Role |
|---|---|---|---|---|---|---|---|
| **DLPack** | In-process ABI | Strided dense | ❌ | Pointer passing | ❌ | Very high | Data interchange |
| **Apache Arrow** | IPC / streaming | Row/column-major, tabular | ❌ | IPC, shared memory, C Data Interface | ❌ | Very high | Data interchange |
| **Arrow Flight** | IPC / streaming RPC | Row/column-major (inherited) | ❌ | gRPC over HTTP/2 | ❌ | Medium | Data interchange |
| **OPeNDAP** | IPC / streaming (HTTP) | Dense row-major | ❌ | HTTP DAP2/DAP4 + constraints | ❌ | Medium | Data interchange |
| **SafeTensors** | File-based | Row-major | ❌ | File (mmap) | ❌ | High | Data interchange |
| **GGUF** | File-based | Row-major + packed quant blocks | ✅ informal | File (mmap) | ❌ | High | Data interchange |
| **Zarr v3** | File / object store | Chunked + C/F order | ❌ (codecs only) | File / object store | ❌ | Medium | Data interchange |
| **NetCDF** | File-based | Dense row-major | ❌ | File | ❌ | High | Data interchange |
| **NIXL** | RDMA / transport | None (byte ranges) | ❌ | RDMA (UCX, GDS, NVMe-oF) | ✅ | Emerging | Data interchange |
| **NCCL** | RDMA / transport | None (flat buffers) | ❌ | RDMA / NVLink collectives | ✅ | Very high | Data interchange |
| **UCX** | RDMA / transport | None | ❌ | IB verbs, RoCE, TCP, CUDA IPC | ✅ | High (infra) | Data interchange |
| **Hurray** *(goal)* | ABI + IPC/streaming + file + RDMA | Strided, tiled, Morton, sparse (COO/CSR/CSC/CSF), block-paged, composite, extension tags | ✅ first-class (per-tensor / per-channel / per-block affine, NF4, MXFP) | `__hurray__` pointer handoff, IPC stream, file container, RDMA data plane | ✅ specified | Pre-release | Data interchange |

---

## 5. Compute Frameworks and Libraries

These are the **clients**. Each entry records which interchange solutions it already
speaks, because that is where Hurray would attach.

### 5.1 General-Purpose ML

#### 5.1.1 NumPy

| | |
|---|---|
| **Role** | Compute framework/library |
| **Domain** | General-purpose array computing |
| **Primary use case** | N-dimensional array manipulation in Python |
| **Interchange supported** | DLPack, buffer protocol, `__array_interface__`, `.npy`/`.npz` |
| **Internal layout** | Arbitrary strides, dense |

The de facto array standard in Python. An `ndarray` carries a pointer, shape, strides,
dtype, and flags; transpose, slice, and broadcast are all zero-copy views. No tiled,
packed, or sparse layouts, and no quantization.

- ✅ The reference stride model, and the on-ramp every Python tool expects
- ❌ Python-coupled; no language-agnostic ABI of its own beyond DLPack
- 🔹 Hurray's Tier 1 dtype vocabulary matches NumPy's so tensors cross without translation

#### 5.1.2 PyTorch

| | |
|---|---|
| **Role** | Compute framework/library |
| **Domain** | General-purpose ML |
| **Primary use case** | Training and inference; the substrate of most serving systems |
| **Interchange supported** | DLPack (`__dlpack__`), SafeTensors, GGUF (via ecosystem), NCCL, NIXL (through vLLM) |
| **Internal layout** | Strided dense, plus memory formats (contiguous, `channels_last`) |

Quantization is real but fragmented: native `qint8`/`quint8` tensors for the legacy
quantization stack, and packed int4/fp8 representations in `torchao` and third-party
kernels, each with its own convention for scales and zero points. None of that survives a
DLPack handoff, because DLPack has no place to put it.

- ✅ Adopts DLPack as its interchange path; ecosystem gravity is enormous
- ❌ No native RDMA tensor protocol and no multi-layout descriptor — both are bolted on
  per-project
- ❌ Quantization parameters travel beside the tensor, never with it
- 🔹 Hurray can carry layout and quantization across the same handoff PyTorch already makes

#### 5.1.3 TensorFlow

| | |
|---|---|
| **Role** | Compute framework/library |
| **Domain** | General-purpose ML |
| **Primary use case** | Training and production serving |
| **Interchange supported** | DLPack (`tf.experimental.dlpack`), SavedModel/Protobuf, NCCL |
| **Internal layout** | Strided dense, row-major; XLA chooses layouts under compilation |

Quantization lives mainly in TFLite as affine per-tensor and per-axis schemes attached to
the model artifact rather than to a transferable tensor.

- ✅ Speaks DLPack; mature serving story
- ❌ Layout decisions are internal to XLA and not expressible on the wire
- ❌ Quantization is a property of the saved model, not of an interchangeable tensor
- 🔹 Hurray gives the quantized tensor an identity independent of the model artifact

#### 5.1.4 JAX

| | |
|---|---|
| **Role** | Compute framework/library |
| **Domain** | General-purpose ML |
| **Primary use case** | Research and large-scale training/inference on TPU and GPU |
| **Interchange supported** | DLPack (`jax.dlpack`), Orbax checkpoints, NCCL |
| **Internal layout** | XLA-managed (minor-to-major dimension order, internal tiling), sharding-aware |

JAX arrays are sharded across devices through `jax.sharding`, and XLA picks physical
layouts — including tiled ones — during compilation. Neither the sharding nor the physical
layout is expressible in the DLPack handoff, so cross-framework transfers degrade to a
dense, single-device view.

- ✅ Speaks DLPack; the strongest first-class sharding model in the group
- ❌ Sharding and physical layout are invisible at the interchange boundary
- 🔹 Hurray's shard descriptor and tiled layouts are the missing vocabulary for exactly
  this handoff

### 5.2 High-Performance Linear Algebra

#### 5.2.1 Eigen (C++)

| | |
|---|---|
| **Role** | Compute framework/library |
| **Domain** | HPC / dense linear algebra |
| **Primary use case** | In-process C++ matrix computation |
| **Interchange supported** | None (raw pointer `Map` over external memory) |
| **Internal layout** | Row- or column-major as a compile-time parameter; arbitrary strides via `Map` |

Expression templates give lazy evaluation without temporaries. Storage order is a template
argument, so it is fixed at compile time, not negotiated.

- ✅ `Map` over external buffers proves that foreign-memory adoption is a solved pattern
- ❌ No serialization, no IPC, no descriptor
- 🔹 Hurray's C FFI is the boundary a library like Eigen would map from

#### 5.2.2 xtensor (C++)

| | |
|---|---|
| **Role** | Compute framework/library |
| **Domain** | HPC / general array computing |
| **Primary use case** | NumPy-like C++ arrays with Python bindings |
| **Interchange supported** | NumPy buffer protocol via xtensor-python; external buffer adaptors |
| **Internal layout** | Strided dense, row- or column-major |

Deliberately more interop-friendly than Eigen — lazy evaluation plus adaptors over
externally owned memory — but with no formal protocol of its own.

- ✅ External buffer adaptors make it an easy consumer of a foreign descriptor
- ❌ Interop stops at the Python buffer protocol; nothing cross-language or on the wire
- 🔹 Hurray provides the formal protocol its adaptors would target

#### 5.2.3 PLASMA and SLATE

| | |
|---|---|
| **Role** | Compute framework/library |
| **Domain** | HPC / dense linear algebra |
| **Primary use case** | Multicore (PLASMA) and distributed-memory (SLATE) LAPACK-class solvers |
| **Interchange supported** | None |
| **Internal layout** | PLASMA: tiled (independently allocated tiles). SLATE: column-major, tiled, band; mixed precision |

PLASMA stores matrices as individually allocated tiles so tasks can run asynchronously
across cores. SLATE is the distributed-memory redesign, adding GPU offload and multiple
coexisting layouts.

- ✅ Direct evidence that tiled layouts are mandatory, not exotic, for fast GEMM
- ✅ SLATE's multi-layout model is a reference for Hurray's layout taxonomy
- ❌ Neither defines an interchange format; layouts are private to the library
- 🔹 Hurray makes those same tile parameters describable across a process boundary

### 5.3 Compilers and Runtimes

#### 5.3.1 Apache TVM (and MLC-LLM)

| | |
|---|---|
| **Role** | Compute framework/library |
| **Domain** | ML compiler / runtime |
| **Primary use case** | Compile and autotune models for heterogeneous hardware |
| **Interchange supported** | DLPack-native (`NDArray`); compiled module + parameter blob artifacts |
| **Internal layout** | Aggressively rewritten during compilation — tiling, packed `NCHWc`, Tensor Core fragments |

TVM ingests models, lowers them through a graph IR (Relax) to a tensor IR, autotunes
kernels, and emits code for CPUs, GPUs, and microcontrollers. **MLC-LLM** builds on it to
run LLMs on phones, laptops, and browsers. TVM comes from the same DMLC lineage that
produced DLPack, so its runtime tensor *is* the in-process ABI Hurray targets — it
validates that choice rather than competing with it. Quantization (int8, grouped int4 in
MLC-LLM) is a compilation pass, not a portable on-disk representation; parameters ship as
ad-hoc bundles.

- ✅ Confirms the DLPack ABI choice from the compiler side
- ✅ Its layout rewrites name exactly the packed/tiled forms a producer might hand over
  pre-optimized, saving a re-layout copy
- ❌ No framework-agnostic, quantization-aware container for its parameter blobs
- 🔹 Two concrete integrations: zero-copy `NDArray` handoff via the shared ABI, and Hurray
  as the on-disk container for MLC-LLM's grouped-int4 weights — the role GGUF plays for
  llama.cpp

### 5.4 Specialized Hardware

#### 5.4.1 MLX (Apple)

| | |
|---|---|
| **Role** | Compute framework/library |
| **Domain** | Specialized hardware — Apple Silicon |
| **Primary use case** | On-device ML research and inference over unified memory |
| **Interchange supported** | DLPack, Python buffer protocol; loads `.npy`, `.npz`, `.safetensors`, `.gguf` |
| **Internal layout** | Strided dense |

Arrays live in memory both CPU and GPU can read with no copy and no explicit device
placement; computation is lazy, so graphs optimize before execution. Quantization is an
*operation*, not a storage dtype: `mlx.core.quantize` produces packed `uint32` weights
plus separate scale and bias arrays. The affine scheme uses group sizes 32/64/128 at 2–8
bits, packed LSB-first (element 0 occupies the low bits of the first word); block
floating-point modes cover MXFP4/MXFP8 (shared E8M0 exponent, group 32) and NVFP4 (group
16, E4M3 per-group scales, no bias). MLX defines no file format of its own — a deliberate
choice, not an omission.

Two findings carry into Hurray's design. **Device is per-operation, not per-buffer:**
unified memory removes the host/device buffer duality entirely, so device affinity belongs
to the kernel that reads a buffer rather than to the buffer — a direct input to the
device-tag design. And MLX's documented LSB-first `int4` packing is the compatibility
reference to check Hurray's sub-byte packing against.

- ✅ Multi-mode quantization packing (affine + MXFP + NVFP4, variable group sizes) is a
  useful cross-check for the quantization spec
- ✅ Reuses existing interchange formats rather than inventing one — the intended posture
  for a compute framework
- ❌ In-process only; no IPC, no streaming, no cross-language ABI beyond an unofficial C
  wrapper
- 🔹 Hurray can be the container MLX loads from and the protocol it exports through

### 5.5 Serving Systems

#### 5.5.1 vLLM

| | |
|---|---|
| **Role** | Compute framework/library |
| **Domain** | Serving |
| **Primary use case** | High-throughput LLM inference with PagedAttention |
| **Interchange supported** | NIXL, Mooncake Transfer Engine, LMCache, NCCL; SafeTensors and GGUF for weights |
| **Internal layout** | Paged KV cache (fixed-size blocks + per-sequence block table); strided dense activations |

The most important integration surface in this survey. Its `KVConnector` API has already
generalized "transfer the KV cache" into a pluggable interface — but what flows through it
is an opaque block buffer plus an ID. See § 6.3 for the interface in detail.

- ✅ Already has the right extension point: a transport-agnostic connector boundary
- ❌ Shape, dtype, layout, and quantization are fixed once at startup and assumed thereafter
- ❌ Cross-engine or cross-quantization transfer needs a bespoke adapter
- 🔹 The connector boundary is precisely where a self-describing Hurray descriptor slots in

#### 5.5.2 NVIDIA Dynamo (with TensorRT-LLM)

| | |
|---|---|
| **Role** | Compute framework/library |
| **Domain** | Serving |
| **Primary use case** | Datacenter-scale disaggregated inference with KV-aware routing |
| **Interchange supported** | NIXL, UCX, MPI, NCCL |
| **Internal layout** | Paged KV cache tiered across GPU/CPU/SSD/object storage (KVBM) |

Dynamo schedules and autoscales; TensorRT-LLM executes; NIXL moves the bytes; KVBM manages
the memory tiers. When prefill and decode run at different parallelism degrees, TensorRT-LLM
performs layout conversion *during transmission* in a dedicated module — hand-written
because no portable descriptor exists to drive a general transform. See § 6.4.

- ✅ The most operationally mature disaggregated stack
- ❌ Its bespoke cross-parallelism reshuffle module exists only because the wire carries no
  layout description
- 🔹 A layout-aware descriptor is what generalizes that module out of existence

### 5.6 Comparison: Compute Frameworks and Libraries

| Framework / Library | Domain | Primary use case | Interchange solutions supported | Internal layout model | Role |
|---|---|---|---|---|---|
| **NumPy** | General-purpose array | Python array manipulation | DLPack, buffer protocol, `.npy` | Arbitrary strides, dense | Compute framework/library |
| **PyTorch** | General-purpose ML | Training + inference substrate | DLPack, SafeTensors, NCCL, NIXL (via vLLM) | Strided dense + memory formats | Compute framework/library |
| **TensorFlow** | General-purpose ML | Training + production serving | DLPack, SavedModel, NCCL | Strided dense; XLA-chosen internally | Compute framework/library |
| **JAX** | General-purpose ML | Research + large-scale TPU/GPU | DLPack, Orbax, NCCL | XLA layouts (minor-to-major, tiled), sharded | Compute framework/library |
| **Eigen** | HPC linear algebra | In-process C++ matrix math | None (`Map` over raw memory) | Row/column-major at compile time; strided `Map` | Compute framework/library |
| **xtensor** | HPC / arrays | NumPy-like C++ with Python bindings | NumPy buffer protocol | Strided dense | Compute framework/library |
| **PLASMA** | HPC linear algebra | Multicore tiled LAPACK-class solvers | None | Tiled (independent tile allocations) | Compute framework/library |
| **SLATE** | HPC linear algebra | Distributed-memory + GPU solvers | None | Column-major, tiled, band; mixed precision | Compute framework/library |
| **Apache TVM** | Compiler / runtime | Compile + autotune for any hardware | DLPack (`NDArray`) | Rewritten: tiled, `NCHWc`, TC fragments | Compute framework/library |
| **MLC-LLM** | Compiler / runtime | LLMs on heterogeneous consumer hardware | DLPack; ad-hoc parameter bundles | TVM-generated; grouped int4 | Compute framework/library |
| **MLX** | Specialized hardware | Apple Silicon on-device ML | DLPack, buffer protocol, SafeTensors, GGUF, `.npy` | Strided dense; packed `uint32` quantized | Compute framework/library |
| **vLLM** | Serving | High-throughput LLM inference | NIXL, Mooncake, LMCache, NCCL, SafeTensors, GGUF | Paged KV cache + strided activations | Compute framework/library |
| **NVIDIA Dynamo** | Serving | Datacenter disaggregated inference | NIXL, UCX, MPI, NCCL | Paged KV cache tiered across GPU/CPU/SSD | Compute framework/library |

---

## 6. KV Cache Transfer in Disaggregated Inference

**Why this section exists.** § 4.4 covered the transport primitives; this section covers
the systems built on them. Every system here is a **compute framework** that depends on a
**data interchange solution** to move its KV cache — vLLM on NIXL, Mooncake on its own
Transfer Engine, Dynamo on NIXL and UCX. They are surveyed together because they are the
largest real-world consumers of a tensor transfer layer, and because they expose the gap
Hurray targets more clearly than any format comparison can: **they all move opaque byte
buffers, and every compute framework involved must reconstruct the meaning out-of-band.**

**What moves and why.** As § 3 described, disaggregated serving runs prefill and decode on
separate GPUs, so the KV cache — logically `[layers, 2, heads, seq_len, head_dim]`, where
the `2` is key and value — must cross the network for every request. For long contexts
that is gigabytes per request. It is almost always stored **paged** (vLLM PagedAttention
style): a flat pool of fixed-size blocks plus a per-sequence block table, so a transfer
moves a list of non-contiguous blocks rather than one contiguous tensor. Both facts —
enormous size and non-contiguous structure — are why the interchange layer's expressiveness
decides the system's performance.

### 6.1 DistServe

**Role:** compute framework (research serving system). **Interchange used:** NVLink.

The OSDI 2024 system (Zhong et al.) that introduced splitting prefill and decode across
GPUs to optimize *goodput* — requests served within both time-to-first-token and
time-per-output-token targets. It transfers the KV cache layer by layer, and its key move
is **bandwidth-aware placement**: co-locate a request's prefill and decode segments so the
transfer rides intra-node NVLink (≈600 GB/s between A100s), which makes the transfer cost
negligible against recompute. Placement comes from a simulation-driven search over
parallelism configurations. Reported up to 7.4× more requests, or 12.6× tighter SLO, versus
co-located baselines.

**Metadata model:** none beyond layer and block references. Both phases run the identical
model build with an identical KV layout, assumed out-of-band; only numerical blocks move.

- ✅ Established the prefill/decode split and the finding that transfer cost dominates
  placement
- ❌ Sidesteps metadata entirely by assuming homogeneous instances on a fast fabric
- 🔹 The academic root of everything below; the assumption it makes is the one production
  systems cannot

### 6.2 Mooncake

**Role:** compute framework (serving platform). **Interchange used:** its own Transfer
Engine (multi-NIC RDMA).

The KVCache-centric platform behind Kimi (Moonshot AI), published at FAST 2025 and
open-sourced, running across thousands of nodes and serving over 100 billion tokens a day.
It harvests otherwise idle CPU, DRAM, and SSD across the GPU cluster into a **disaggregated
KVCache pool**, fronted by a scheduler that maximizes throughput under SLOs and maximizes
prefix-cache reuse.

Its **Transfer Engine** is arguably the most influential artifact: a standalone, reusable,
zero-copy RDMA library that aggregates multiple RDMA NICs per host and picks paths from a
topology matrix broadcast by each server, which classifies NICs into preferred and
secondary lists per memory region at registration time. It prefers local-NUMA and
local-PCIe-switch GPUDirect paths and fails over on error. Reported 87 GB/s over 4×200 Gbps
RoCE and 190 GB/s over 8×400 Gbps — roughly 2.4–4.6× faster than TCP.

**Metadata model:** transfers are keyed by block identifiers and offsets into registered
memory regions. Shape, dtype, and paged layout are engine configuration, agreed
out-of-band.

- ✅ The most complete production proof that KV cache is worth treating as poolable,
  transferable, durable data
- ✅ Its Transfer Engine is direct prior art for Hurray's RDMA data plane
- ❌ Still moves registered byte ranges, not described tensors
- 🔹 A pooled KV cache with no self-description is reusable only by its writer

### 6.3 vLLM and the KVConnector API

**Role:** compute framework (inference engine). **Interchange used:** NIXL, Mooncake,
LMCache, and others through one interface.

vLLM's connector framework for disaggregated prefill and KV offloading is the de facto
integration point for the ecosystem; implementations live under
`vllm/distributed/kv_transfer`.

`KVConnectorBase_V1` splits by role. Scheduler-side methods
(`get_num_new_matched_tokens`, `update_state_after_alloc`, `build_connector_meta`,
`request_finished`) decide which blocks to load or save and assemble per-step metadata.
Worker-side methods (`register_kv_caches`, `start_load_kv`, `wait_for_layer_load`,
`save_kv_layer`, `wait_for_save`, `get_finished`) register GPU memory and run asynchronous
block transfers. Transport is cleanly decoupled from model logic.

The RDMA implementation, **NixlConnector**, performs a lazy ZMQ side-channel handshake that
exchanges NIXL agent identity and memory descriptors; workers then compute NIXL descriptor
IDs for block arrays. It also handles the messy case where prefiller and decoder run at
**different tensor-parallelism degrees**, which requires a block-mapping reshuffle. Sibling
connectors include `MooncakeConnector`/`MooncakeStoreConnector`, `LMCacheMPConnector`,
`OffloadingConnector`, and `MultiConnector`.

**Metadata model — the crux:** shape, dtype, layout, and quantization are established
**once, out-of-band, at `register_kv_caches()` during startup**, then assumed constant per
layer. Per transfer, only raw block buffers plus block and request IDs cross the wire. The
handshake negotiates addresses and agent identity — never a tensor descriptor.

- ✅ The abstraction is already right: one connector boundary, many transports
- ❌ What crosses that boundary is an opaque buffer plus an ID
- ❌ Every connector pair therefore assumes identical engine builds
- 🔹 The clearest statement of the gap, and the most natural place to close it

### 6.4 NVIDIA Dynamo and TensorRT-LLM

**Role:** compute frameworks (serving framework + inference engine). **Interchange used:**
NIXL, UCX, MPI.

**Dynamo** is the datacenter-scale serving framework — disaggregated prefill/decode, GPU
autoscaling, KV-aware routing. **TensorRT-LLM** is one of its backends. **NIXL** (§ 4.4.1)
is the transfer library. **KVBM**, the KV Block Manager, is a framework-agnostic unified
memory layer usable standalone or inside Dynamo; it tiers the KV cache across GPU, CPU,
SSD, filesystem, and cloud through NIXL's plugin backends to free GPU memory while
preserving hit rates, and it works across vLLM, TensorRT-LLM, SGLang, and PyTorch.

The revealing detail is **layout conversion during transmission**. When context and
generation phases use different parallelism — TP2 prefill against PP2 decode, say —
TensorRT-LLM converts the cache layout inside a dedicated KV-cache-exchange module,
"modularly decoupled from the KV cache manager and communication libraries." The metadata
that accompanies a transfer, `ctx_params`, carries prompt tokens, the first generated
token, and connection parameters — how to connect and which request this is, never what
shape or layout the tensor has.

- ✅ The most mature production stack, and framework-agnostic at the memory-management layer
- ❌ NVIDIA had to hand-write layout conversion because no portable descriptor exists to
  drive a general one
- 🔹 That bespoke reshuffle is precisely what a layout-aware descriptor generalizes

### 6.5 llm-d

**Role:** compute framework (Kubernetes-native serving). **Interchange used:** NIXL (with
a UCCL backend).

A Kubernetes-native distributed inference framework — backed by Red Hat, Google, and IBM
among others — that treats prefill/decode disaggregation as a first-class orchestration
primitive on top of vLLM and an inference-aware gateway. KV cache moves GPU-to-GPU over
RDMA through NIXL, with cache-aware routing (send a request to the worker that already
holds its prefix) and tiered storage. Version 0.5 integrated the **UCCL** backend into the
NIXL networking layer to unify over vendor collectives (NCCL/RCCL/MCCL). Reported roughly
70% higher throughput and 88% faster time-to-first-token versus monolithic deployments.

**Metadata model:** inherits vLLM's connector model (§ 6.3) and adds Kubernetes-level
routing and placement metadata. The payload is still opaque vLLM blocks.

- ✅ Signals that disaggregated KV transfer is consolidating into shared infrastructure
- ❌ Adds orchestration metadata, not tensor metadata
- 🔹 Consolidation is exactly when a common descriptor becomes valuable rather than
  premature

### 6.6 LMCache

**Role:** compute framework (KV cache layer). **Interchange used:** vLLM/SGLang connectors,
plus its own multi-tier store.

An open-source KV cache layer for vLLM and SGLang that lifts KV caches out of GPU memory
and shares them across engines and queries. The store spans GPU memory, pinned CPU DRAM,
local disk, and remote backends such as Redis, behind a modular connector and a control API
(pin, lookup, cleanup, move, compress). Performance comes from batched movement and
compute/IO pipelining.

Two distinctive optimizations go further than transport: **CacheGen** compresses the KV
cache into a compact bitstream for storage and transfer, and **CacheBlend** reuses
*non-prefix* KV chunks. Reported up to 15× throughput improvement with vLLM on multi-round
QA and document analysis.

**Metadata model:** keyed KV chunks — and with CacheGen the payload is a codec-specific
compressed bitstream, so even the bytes are no longer a plain tensor buffer.

- ✅ Pushes furthest past raw-buffer transfer: compresses and reshapes KV cache for reuse
- ❌ A CacheGen blob is meaningless without out-of-band knowledge of shape, dtype, layout,
  *and* codec
- 🔹 Makes the absence of a self-describing format most acute — a descriptor that can name
  a quantized or compressed KV cache is directly applicable

### 6.7 The Metadata Gap

Across every system above, one pattern holds: **the KV cache moves as opaque,
engine-private bytes, and everything needed to interpret those bytes is agreed
out-of-band.**

- The logical tensor, its dtype, its paged block layout, and its quantization scheme are
  fixed by the model build and engine configuration — not transmitted.
- What crosses the wire is block buffers plus block IDs (vLLM, Mooncake, llm-d),
  connection and request parameters (`ctx_params` in TensorRT-LLM), or a codec-specific
  compressed blob (LMCache/CacheGen).
- The handshakes negotiate addresses, agent identity, and TP mapping — never a portable
  tensor descriptor.

| System | Interchange solution used | Travels *with* the bytes | Assumed out-of-band |
|---|---|---|---|
| DistServe | NVLink (intra-node) | Layer/block references | Identical model and layout |
| Mooncake | Transfer Engine (multi-NIC RDMA) | Block keys / offsets | Shape, dtype, paged layout |
| vLLM KVConnector | NIXL / Mooncake / LMCache | Raw blocks + block IDs | Format fixed at `register_kv_caches()` |
| Dynamo / TensorRT-LLM | NIXL / UCX / MPI | `ctx_params` (tokens, connection params) | Layout; cross-TP reshuffled by a bespoke module |
| llm-d | NIXL (+ UCCL backend) | Blocks + routing hints | Model identity, layout |
| LMCache | Connector + batched movement | Compressed KV chunks + keys | Engine KV format, CacheGen codec |

Three consequences follow, and they are the symptoms a descriptor removes:

1. **Point-to-point coupling.** Every connector pair assumes identical engine builds.
   Cross-engine (vLLM ↔ TensorRT-LLM), cross-version, or cross-quantization transfer needs
   a bespoke adapter, because there is no neutral representation to meet in.
2. **Hand-written layout conversion.** Mismatched parallelism forces ad-hoc reshuffle code
   — TensorRT-LLM's exchange module, NixlConnector's TP mapping — instead of a
   descriptor-driven transform.
3. **Opaque storage.** A KV cache spilled to a pool (Mooncake) or compressed to disk
   (LMCache) carries no standardized self-description, so only the engine that wrote it can
   read it back.

### 6.8 What This Implies for Hurray

- **The block-paged layout is specified.** The PagedAttention KV cache motivated Hurray's
  `block-paged` layout tag (`0x0A`), specified in
  [ADR-024](adr/ADR-024-block-paged-indirect-layout.md) and
  [Block-Paged](spec/layouts/block-paged.md). It encodes the fixed page size, the block
  table mapping logical sequence position to physical page ID, per-sequence page lists
  (addressed CSR-style through a `seq_ptr` offset array), and prefix sharing across
  sequences as aliased page IDs. A paged KV cache becomes a first-class Hurray tensor: the
  consumer reads page size, head and layer organization, dtype, and quantization from the
  descriptor, with no out-of-band agreement — which turns TensorRT-LLM's bespoke conversion
  (§ 6.4) into a general, descriptor-driven transform.

- **The in-process handoff already matches the connector shape.** Hurray's `__hurray__` /
  `from_hurray` zero-copy protocol ([ADR-023](adr/ADR-023-hurray-python-native-buffer-protocol.md),
  renamed in [ADR-033](adr/ADR-033-native-protocol-rename-hurray.md)) lets an engine wrap
  its existing paged GPU buffers as Hurray tensors without a copy: the producer exposes the
  page buffers plus a descriptor, the consumer imports them. This is the natural API for a
  vLLM `KVConnector`-style integration.

- **Composition with the data plane, not replacement of it.** NIXL (§ 4.4.1) and Mooncake's
  Transfer Engine (§ 6.2) are the **data plane** — they move registered buffers over RDMA.
  Hurray is the **metadata plane** — it says what those buffers are. A Hurray descriptor
  names the KV cache; NIXL moves it. Hurray is explicitly not a serving system, a
  scheduler, or a cache pool. It is the interchange vocabulary those systems currently
  improvise per connector.

---

## 7. Region-Heterogeneous Tensor Structures

Every layout in § 4 describes a *homogeneous* array: one element type, one layout, one
optional quantization scheme across the whole index space. A separate body of prior art
describes a *single logical array whose index space is partitioned into regions that differ
in structure* — some dense, some sparse, some constant, each with its own storage and
sometimes its own precision. This is the class Hurray's **composite tensor** layout (tag
`0x0B`, [ADR-027](adr/ADR-027-composite-tensors-head-members-composition-rule.md)) targets.
It matters to the array-database direction, where one very large tensor with structurally
heterogeneous regions is a first-class case rather than an edge case.

Two composition models appear in the literature, and they are not interchangeable:

- **Partition (exact-cover, non-overlapping):** disjoint boxes tile the index space
  exactly; each element belongs to one region. Prior art: AMReX/Chombo
  `DisjointBoxLayout`, OpenVDB tiles, HDF5 Virtual Datasets (which relax the rule to allow
  gaps).
- **Overlay (overlapping composition):** a base spanning the whole index space plus sparse
  corrections at scattered positions sharing indices with the base. Prior art: SpQR and
  KVQuant outlier quantization, TileDB timestamped fragments.

A single-partition layout cannot express both, which is why Hurray unified them under the
composite primitive (head + members + composition rule) instead of extending the retired
subpaving layout. Composite supports partition, group, and **sealed overlay** in v1.0;
versioned/open overlay is deferred.

### 7.1 Comparison

| Structure | Segment | Partition shape | Per-region inner layout | Per-region buffers | Per-region quant/precision | Composition model | Maturity |
|---|---|---|---|---|---|---|---|
| AMReX / Chombo `BoxArray` / `DisjointBoxLayout` | HPC AMR | Irregular boxes | Uniform (dense FAB) | ✅ independent | ❌ | Partition (exact-cover per level) | Production (DOE Exascale) |
| OpenVDB / NanoVDB | VFX / graphics | Hierarchical tiles + 8³ leaves | Heterogeneous (constant tile vs dense leaf) | ✅ (linearized in NanoVDB) | Partial (per-node value quant) | Partition (hierarchical) | Production (ASWF standard) |
| HDF5 Virtual Dataset (VDS) | Scientific storage | Arbitrary rectangular selections | Heterogeneous (per source dataset) | ✅ per source | Via per-source compression | Partition, but permits gaps/overlap | Standard since HDF5 1.10 |
| TileDB dense array w/ sparse fragments | Array DB | N/A (temporal) | Dense + sparse fragments | ✅ per fragment | ❌ | Overlay (timestamped, last-writer-wins) | Production |
| Zarr v3 ZEP0003 variable chunks | Scientific storage | Rectilinear variable grid | Uniform | ✅ per chunk | ❌ (array-level codec) | Partition (rectilinear only) | Emerging (behind flag, Zarr-Python 3.2) |
| MLIR `sparse_tensor` encoding | ML compiler | Per-dimension level, not per-region | Per-*level* type only | N/A | ❌ | Neither (whole-tensor encoding) | Production (LLVM) |
| SpQR / KVQuant outliers | ML quantization | Scattered points (not rectangular) | Dense low-bit + CSR outliers | ✅ (base + CSR) | ✅ (base vs outlier precision) | **Overlay** | Research → adoption |
| KIVI / HF residual KV cache | ML inference | Regular sequence-axis split (recent/old) | Uniform (fp16 vs quantized) | ✅ | ✅ (per-region precision) | Partition (regular, 2 regions) | Production |
| MoE per-expert quantization | ML inference | Regular expert blocks | Uniform | ✅ per expert | ✅ (per-expert bit width) | Partition (regular) | Research → adoption |
| Block-sparse attention (BigBird, FlexAttention) | ML inference | Regular block grid | Uniform dense + mask | ❌ | ❌ | Partition (regular) + mask | Production |
| ASTC texture blocks | GPU graphics | Regular block grid | Per-block mode/partition | ❌ (packed) | ✅ per block | Partition (regular) | Production (hardware) |
| **Hurray composite (`0x0B`)** | Interchange | **Irregular boxes** | **Any tag — members are full descriptors** | **✅ per member** | **✅ per member** | **Partition, group, and sealed overlay** | Draft |

### 7.2 Findings

- Irregular exact-cover partitioning with independent per-region buffers is a *mature*,
  production-proven pattern in HPC adaptive mesh refinement (AMReX/Chombo) and VFX volume
  storage (OpenVDB/NanoVDB). NanoVDB's pointerless linearization of a heterogeneous-region
  tree is a direct precedent for encoding such a tensor as a zero-copy, GPU-friendly byte
  image — matching Hurray's streamability and zero-copy constraints.
- HDF5 Virtual Datasets are the closest standardized analog: a logical N-D dataset defined
  as per-region mappings onto heterogeneous backing storage. Notably, VDS chose *permissive*
  coverage (gaps and overlap allowed) where Hurray's partition rule mandates exact cover
  with no overlap.
- Mainstream ML mostly wants per-region *precision* on *regular* partitions — KIVI's
  recent/old KV split, per-expert MoE bit widths — rather than irregular partitioning. That
  supports the per-member quantization mechanism more than the irregular-box generality.
- The dominant ML heterogeneity pattern, outlier quantization (SpQR, KVQuant), is an
  **overlay**: a dense base plus a scattered high-precision residual over shared indices. It
  is incompatible with a non-overlap partition rule, which is exactly why the composite
  primitive treats overlay as its own composition rule rather than forcing it into a
  partition.

**Relevance:** region-heterogeneous partition tensors are demanded and proven in the HPC,
scientific, and graphics segments the array-database direction targets; per-region
quantization is independently demanded in ML inference; and the largest ML pattern
(outlier overlays) needs an overlapping composition rule that a partition must not absorb.

---

## 8. Gaps and Design Goals

Four gaps recur across § 4. Each one is absorbed today by a compute framework, as private
glue code, a copy, or a side channel.

### 8.1 No unified layout model

**The gap.** DLPack stops at strides. Arrow and SafeTensors stop at row/column-major. GGUF
packs quantization blocks but only in its own dialect. NIXL, NCCL, and UCX describe nothing
at all. So a producer holding a tiled or packed tensor — the form every fast GEMM kernel
actually wants (§ 5.2.3) — has no way to say so, and the consumer either repacks or the two
sides agree privately.

**Hurray's answer.** A layout tag on every descriptor, drawn from a specified taxonomy:
dense (row-major, column-major, strided, tiled/blocked, Morton), sparse (COO, CSR, CSC,
CSF), indirect (block-paged), and virtual (composite). Core layouts are **Tier 1** and must
be supported by conforming readers; **Tier 2** extension tags cover hardware-specific
packed and panel forms without forcing every implementation to carry them
([ADR-003](adr/ADR-003-panel-pack-via-extension-layout-and-content-negotiation.md)).
A consumer reads the tag and knows exactly which conversion, if any, it owes.

### 8.2 No quantization metadata in descriptors

**The gap.** Quantized inference is the norm, yet scales, zero points, and group sizes
travel beside the tensor rather than with it — in a checkpoint's config, in a framework's
private wrapper, in a codec's assumptions. DLPack has no field for them. Arrow has none in
core. GGUF has excellent ones that only llama.cpp reads.

**Hurray's answer.** A quantization descriptor as a first-class section of the tensor
descriptor: scheme tag, scale and zero-point buffers, block or group size, and the buffer
table entries that hold the parameters. Tier 1 covers per-tensor, per-channel, and
per-block affine; Tier 2 covers NF4 and MXFP microscaling. A quantized tensor stays
interpretable after it leaves the framework that produced it.

### 8.3 No RDMA-native tensor protocol

**The gap.** The fast transports carry anonymous bytes; the protocol that carries structure
(Arrow Flight) runs on gRPC, which copies and loses alignment. A framework that wants both
speed and meaning must build the meaning itself — which is what every KV connector in § 6
does.

**Hurray's answer.** An RDMA data plane in the interchange protocol (originally OQ-2, now
resolved): the buffer owner registers its memory region and sends `RDMA_REGISTER` with the
remote key, address, and length over the control plane; the peer answers `RDMA_READY`; the
transfer executes out-of-band over UCX or NIXL; `TENSOR_DATA_END` on the control plane is
the authoritative completion signal. The descriptor rides the control plane, the bytes ride
the NIC.

### 8.4 No streaming with layout negotiation

**The gap.** Nothing lets two parties *agree* on a layout. Formats fix one; transports
assume one. So a producer cannot offer "I have this tiled, or I can give you row-major",
and a consumer cannot state what it can ingest without a copy.

**Hurray's answer.** An Arrow Flight-inspired streaming model — descriptor before data,
typed messages, bidirectional exchange, self-delimiting frames with no back-references —
plus a capability handshake in which each side declares the layouts, element types, and
quantization schemes it supports, including opaque extension tags. Conversion happens once,
on the side that can do it best, instead of unconditionally on the receiver.

| Gap | Absorbed today by | Hurray's answer |
|---|---|---|
| No unified layout model | Repacking copies; private agreements | Tagged layout taxonomy, Tier 1 core + Tier 2 extensions |
| No quantization metadata | Config files, framework-private wrappers | Quantization descriptor (scheme, scales, zero points, block size) |
| No RDMA-native tensor protocol | Per-connector handshakes over NIXL/UCX | Descriptor on the control plane, RDMA data plane beneath it |
| No streaming + negotiation | Startup-time out-of-band agreement | Streaming RPC + capability/layout handshake |

---

## 9. Conclusion and Next Steps

**Three takeaways.**

1. **Interchange solutions provide the foundation but not the vocabulary.** DLPack and
   Arrow Flight got the important parts right — a zero-copy in-process ABI and a streaming
   RPC shape — and stopped short of the layout, quantization, and device description that
   inference workloads need. NIXL and UCX move bytes at hardware speed and describe nothing.
2. **Compute frameworks pay for the shortfall.** PyTorch keeps quantization parameters
   outside the tensor. JAX cannot express sharding across a handoff. vLLM fixes its KV
   format at startup and ships opaque blocks. TensorRT-LLM hand-writes a layout converter.
   None of these is a defect in those systems; each is the workaround an inexpressive
   interchange layer forces.
3. **The gap is a descriptor, not a transport.** The bytes already move fast enough. What
   is missing is a portable statement of what they are, carried with them across
   in-process, IPC, file, and RDMA paths alike. That is the whole of Hurray's claim.

**Next steps.**

1. **Finish the layout descriptor surface.** The Tier 1 layouts are specified; complete the
   implementation across all crates and keep the extension-tag path (Tier 2) usable end to
   end, so a producer can hand over a hardware-packed tensor without losing its description.
2. **Integrate with NIXL and UCX for RDMA.** The data plane is specified; the implementation
   is not. A working `KVConnector`-shaped integration with vLLM or Dynamo is the proof that
   the metadata plane composes with the transports these systems already run.
3. **Benchmark against the incumbents.** Measure a Hurray-described KV cache transfer
   against vLLM's existing NixlConnector path — throughput, latency, and the copies avoided
   when the two sides disagree on layout or parallelism. The claim in § 8 is testable, and
   it should be tested.

---

## Appendix A — Glossary

**Data interchange solution.** A format, protocol, or library whose job is to move or store
tensor data. It defines bytes, descriptors, or transports — never arithmetic.

**Compute framework/library.** A system that performs computation on tensors: kernels,
graph execution, autotuning, or request scheduling. The client of an interchange solution.

**Zero-copy.** Handing a consumer the producer's existing memory rather than a duplicate.
Requires agreement on alignment, ownership, and lifetime, which is why it is a protocol
question and not just an implementation trick.

**KV cache.** The stored keys and values from previous tokens that let an LLM generate the
next token without recomputing attention over the whole prompt. Logical shape
`[layers, 2, heads, seq_len, head_dim]`.

**Prefill / decode.** The two phases of autoregressive generation. Prefill processes the
whole prompt at once and is compute-bound; decode emits one token at a time and is
memory-bandwidth-bound.

**Disaggregated inference.** Running prefill and decode on separate GPUs or nodes so each
can be scaled and tuned independently — which requires transferring the KV cache between
them.

**PagedAttention.** vLLM's KV cache scheme: instead of one contiguous buffer per sequence,
a flat pool of fixed-size blocks plus a per-sequence block table mapping logical positions
to physical blocks. Eliminates fragmentation and makes prefix sharing a matter of pointing
two sequences at the same block.

**GPUDirect RDMA.** A network card reading from, or writing to, GPU memory directly over
the network, with no staging copy through host memory and no CPU involvement.

**RDMA.** Remote Direct Memory Access — one machine's NIC reading or writing another
machine's registered memory without involving the remote CPU. Requires exchanging a remote
key and address first.

**SIMD.** Single Instruction, Multiple Data — CPU instructions that apply one operation to
several values at once (AVX-512, NEON). SIMD kernels want operands contiguous and aligned,
which is why alignment is a format concern.

**Strided layout.** A layout described by one step size per dimension. Expresses row-major,
column-major, transposes, and slices; cannot express tiling or packing.

**Tiled (blocked) layout.** A layout that stores small rectangular blocks contiguously so
each block fits in cache. Standard in high-performance GEMM.

**Affine quantization.** Storing a value as a low-precision integer plus a scale and
(optionally) a zero point, so that `value ≈ scale × (q − zero_point)`. Parameters may apply
per tensor, per channel, or per block of elements.

**Group / block size.** How many consecutive elements share one set of quantization
parameters. Smaller groups cost more metadata and lose less accuracy.

**Tier 1 / Tier 2.** Hurray's conformance split. Tier 1 element types, layouts, and
quantization schemes must be supported by every conforming implementation; Tier 2 is
optional, for hardware-specific or specialized cases.

**Tensor parallelism (TP) / pipeline parallelism (PP).** Splitting a model across devices
by cutting individual weight matrices (TP) or by assigning whole layers to different
devices (PP). When two disaggregated stages use different degrees, their KV caches are laid
out differently — the mismatch § 6.4 describes.

---

## Appendix B — Representative Tensor Shapes

Which layouts and alignment rules matter follows from the sizes that actually occur.

**LLM weights (LLaMA-2 70B, float16)**

| Tensor | Shape | Size |
|---|---|---|
| Token embedding | [128256, 8192] | ~2 GB |
| Attention Q/K/V projection | [8192, 8192] each | ~128 MB |
| FFN gate/up projection | [8192, 28672] | ~448 MB |

**LLM activations (dynamic)**

| Tensor | Typical shape |
|---|---|
| Input token embeddings | [B=32, S=2048, D=8192] |
| Attention scores | [B=32, H=64, S=2048, S=2048] |
| KV cache (all layers) | [2, 80, 32, 64, 2048, 128] |

**Vision — CNN feature maps (NCHW)**

| Layer | Shape |
|---|---|
| Input batch | [32, 3, 224, 224] |
| Early conv output | [32, 64, 112, 112] |
| Late conv output | [32, 2048, 7, 7] |

**Thresholds worth remembering**

- Fits in L2/L3: `[64, 64]` — 8 KB float16
- Fits in GPU SRAM: `[2048, 2048]` — 8 MB float16
- A single weight matrix: `[8192, 28672]` — 448 MB float16
- Pathological (quadratic attention): `[32, 64, 32768, 32768]` — terabyte scale

---

## Appendix C — Memory Layouts in Production

| Layout | Best for | Notes |
|---|---|---|
| Row-major (C order) | GEMM A matrix, activations, attention scores | Default in most frameworks |
| Column-major (F order) | GEMM B matrix, BLAS conventions | Eigen default |
| Tiled / blocked | High-intensity GEMM, convolutions | Cache-optimal; requires repacking |
| Panel-packed | Innermost GEMM kernel | Ephemeral; repacking cost amortized |
| NHWC | Inference convolutions | Channel-contiguous, SIMD-friendly |
| NCHW | Training convolutions (NVIDIA) | Historical cuDNN default |
| Paged (vLLM-style) | KV cache in autoregressive serving | Variable-length sequences |
| CSR / BSR | Sparse weight matrices | Post-pruning inference |
| Structured 2:4 | NVIDIA Sparse Tensor Cores | Requires a metadata mask |
| Sub-byte packed (int4) | Quantized weights | Block structure with interleaved scales |

---

## Appendix D — References

**Data interchange solutions**

- DLPack — <https://github.com/dmlc/dlpack>
- Apache Arrow — <https://arrow.apache.org>
- Apache Arrow Flight — <https://arrow.apache.org/docs/format/Flight.html>
- SafeTensors — <https://github.com/huggingface/safetensors>
- GGUF — <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>
- Zarr — <https://zarr.dev>
- NetCDF — <https://www.unidata.ucar.edu/software/netcdf/>
- OPeNDAP — <https://www.opendap.org>
- NIXL — <https://github.com/ai-dynamo/nixl>
- NCCL — <https://developer.nvidia.com/nccl>
- UCX — <https://openucx.org>

**Compute frameworks and libraries**

- PyTorch — <https://pytorch.org>
- TensorFlow — <https://www.tensorflow.org>
- JAX — <https://docs.jax.dev>
- NumPy — <https://numpy.org>
- Eigen — <https://eigen.tuxfamily.org>
- xtensor — <https://xtensor.readthedocs.io>
- PLASMA / SLATE — <https://icl.utk.edu/plasma/>, <https://icl.utk.edu/slate/>
- Apache TVM — <https://tvm.apache.org>; MLC-LLM — <https://llm.mlc.ai>
- MLX — <https://ml-explore.github.io/mlx/>

**Disaggregated inference and KV cache transfer** (accessed June 2026)

- DistServe — [arXiv:2401.09670](https://arxiv.org/abs/2401.09670), [OSDI '24](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
- Mooncake — [arXiv:2407.00079](https://arxiv.org/abs/2407.00079), [USENIX FAST '25](https://www.usenix.org/conference/fast25/presentation/qin), [Transfer Engine design docs](https://kvcache-ai.github.io/Mooncake/design/transfer-engine/index.html), [github.com/kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake)
- vLLM disaggregated prefilling — [docs.vllm.ai](https://docs.vllm.ai/en/stable/features/disagg_prefill/), [KV cache transfer and connectors](https://deepwiki.com/vllm-project/vllm/9.4-kv-cache-transfer-and-disaggregated-serving)
- NVIDIA Dynamo / TensorRT-LLM — [Dynamo introduction](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/), [Disaggregated serving in TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html), [Reducing KV cache bottlenecks](https://developer.nvidia.com/blog/how-to-reduce-kv-cache-bottlenecks-with-nvidia-dynamo/)
- llm-d — [P/D disaggregation guide](https://llm-d.ai/docs/guide/Installation/pd-disaggregation), [llm-d v0.5](https://llm-d.ai/blog/llm-d-v0.5-sustaining-performance-at-scale), [NVIDIA Dynamo × llm-d](https://developer.nvidia.com/blog/nvidia-dynamo-accelerates-llm-d-community-initiatives-for-advancing-large-scale-distributed-inference/)
- LMCache — [arXiv:2510.09665](https://arxiv.org/html/2510.09665v2), [github.com/LMCache/LMCache](https://github.com/LMCache/LMCache), [architecture docs](https://docs.lmcache.ai/developer_guide/architecture.html)

**Region-heterogeneous structures**

- AMReX — <https://amrex-codes.github.io>; Chombo — <https://commons.lbl.gov/display/chombo/>
- OpenVDB / NanoVDB — <https://www.openvdb.org>
- HDF5 Virtual Datasets — <https://docs.hdfgroup.org/hdf5/develop/_v_d_s.html>
- TileDB — <https://tiledb.com>
- MLIR `sparse_tensor` — <https://mlir.llvm.org/docs/Dialects/SparseTensorOps/>
- SpQR — [arXiv:2306.03078](https://arxiv.org/abs/2306.03078); KVQuant — [arXiv:2401.18079](https://arxiv.org/abs/2401.18079)
- KIVI — [arXiv:2402.02750](https://arxiv.org/abs/2402.02750)

---

*Snapshot of the state of the art as of September 2026. Base survey April 2026; KV cache
transfer added June 2026; region-heterogeneous structures added July 2026; restructured
around the interchange/compute distinction September 2026. It informs the specification,
the architecture decisions, and the implementation priorities — see `docs/spec/` and
`docs/adr/`.*
