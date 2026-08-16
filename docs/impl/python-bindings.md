# Python Bindings Requirements — Hurray Implementation Requirements

## Overview

The `hurray-python` package is the **Python face of the Hurray interchange format**:
the library a Python program uses to *produce* Hurray tensors, *consume* them, and
*hand them off* to the surrounding array ecosystem without copying. It is built on
PyO3 and is a codec and zero-copy bridge — **not** a numerical or compute library, and
not an implementation of the Python Array API Standard (see ADR-029). Its two goals
are:

1. **Produce / consume Hurray tensors** — construct tensors and serialize them to the
   Hurray format, and parse Hurray data into usable Python objects, including the types
   the ecosystem cannot represent natively (Tier 2, quantized, sparse, composite).
2. **Zero-copy interop** — share buffers with NumPy, PyTorch, JAX, and CuPy without
   copying, via DLPack, the NumPy array protocols, and the native Hurray buffer
   protocol.

These interop protocols are **standalone** — DLPack in particular is an independent
specification, not part of the Array API (a consumer's `from_dlpack` works on any
object exposing `__dlpack__`, with no Array API namespace involved). `hurray-python`
therefore does not claim, and MUST NOT advertise, Array API conformance; it is
Array-API-*interoperable* (a Tier 1 tensor handed to NumPy/PyTorch becomes a real
array those ecosystems can compute on), not Array-API-*implementing*.

## Rationale

> **Note (non-normative):** This section explains *why* `hurray-python` exists and the
> incentives that shape its surface. It is motivation, not a requirement.

`hurray-python` is the **Python face of the Hurray format** — the library a Python
program uses to *produce* Hurray tensors (serialize its data into the format),
*consume* them (load Hurray data into usable Python objects), and *hand them off*
to the surrounding array ecosystem without copying. It is a codec and an
interchange bridge, in the same spirit as the Python packages for other data
formats: not a numerical or compute library, and not a place where array math
lives. Its reason to exist is simply that Python is where the machine-learning
ecosystem lives, and a format with no ergonomic Python entry point would never be
adopted there.

Three incentives drive its design:

- **Reach the ecosystem on day one.** The array ecosystem already shares tensors
  through a widely implemented zero-copy handoff. By speaking that same handoff,
  `hurray-python` interoperates with NumPy, PyTorch, JAX, and CuPy immediately, for
  the common case (ordinary dense tensors of the standard element types), with no
  per-library adapters to write. This is why Hurray tensors are made to feel like
  ordinary arrays in that ecosystem rather than opaque blobs.

- **Preserve full fidelity between Hurray-aware components.** The common ecosystem
  handoff can only describe the ordinary dense case. Everything that makes Hurray
  worth having — compressed and quantized data, sparse and other specialized
  layouts, richer element types, and the metadata that travels with a tensor —
  falls outside what that handoff can carry. So `hurray-python` also offers a
  **native Hurray interchange path** that preserves the tensor in full, for sharing
  between components that both understand Hurray.

- **Bridge the gap while adoption grows.** In an ideal end state, producer and
  consumer libraries would understand Hurray natively, the same way they understand
  the common handoff today. Until then, `hurray-python` is the adapter that speaks
  *both* sides: it ingests tensors from the existing ecosystem and emits Hurray, and
  vice versa. This bridging role is deliberately temporary in ambition — it recedes
  naturally as more of the ecosystem adopts Hurray directly.

Finally, `hurray-python` offers a handful of **direct convenience methods** for the
most common partners (NumPy, PyTorch) even though the generic zero-copy handoff
already exists. This is a deliberate, adoption-minded choice: a new format is
accepted or rejected on how much friction it adds, so a one-call, discoverable path
matters. These methods also cover the cases the generic handoff cannot express, so
that no data is silently left behind.

## Tensor Surface

`hurray-python` MUST expose a `hurray.Tensor` class with an inspection and interop
surface. This surface describes the tensor and hands it off zero-copy; it does **not**
include array computation (elementwise math, reductions, linear algebra, indexing, or
operators) — those belong to the framework the buffer is handed to. `hurray.Tensor`
MUST NOT implement `__array_namespace__`, and the `hurray` module is not an Array API
namespace (see ADR-029).

### Inspection and interop methods

| Method | Requirement |
|---|---|
| `__dlpack__(stream=None)` | MUST return a DLPack capsule for zero-copy buffer sharing. See [DLPack Interoperability](#dlpack-interoperability) for stream semantics. |
| `__dlpack_device__()` | MUST return a `(DLDeviceType, device_id)` tuple. `device_id` is passed as runtime metadata at `Tensor` construction time and defaults to `0`. See [DLPack Interoperability](#dlpack-interoperability). |
| `__hurray_buffer__(stream=None)` | MUST return a native Hurray buffer capsule (full-fidelity, all dtypes). See [Native Buffer Interchange Protocol](#native-buffer-interchange-protocol). |
| `dtype` | MUST return the tensor's `hurray.dtype.*` object. |
| `shape` | MUST return a `Tuple[Optional[int], ...]`. Each element is an `int` for a known dimension, or `None` for a dynamic (unknown) dimension. |
| `ndim` | MUST return the number of dimensions. |
| `size` | MUST return `Optional[int]`: the total number of elements, or `None` if one or more dimensions are dynamic (unknown). |
| `device` | MUST return a device object consistent with `__dlpack_device__`. |
| `T` | MUST return a transposed view without copying for rank-2 tensors. MUST raise `ValueError` if the tensor is not rank-2. |

### Tier 1 dtype interop correspondence

> **Note (non-normative):** Tier 1 element types use the standard numeric vocabulary,
> so a Tier 1 `hurray.Tensor` maps to a NumPy dtype without translation when handed to
> the ecosystem. This correspondence is an *interop* detail, not an Array API claim.

| Hurray type | NumPy dtype (for interop) |
|---|---|
| `bool` | `numpy.bool_` (via `__array__`; not representable over DLPack — see below) |
| `int8` | `numpy.int8` |
| `uint8` | `numpy.uint8` |
| `int16` | `numpy.int16` |
| `uint16` | `numpy.uint16` |
| `int32` | `numpy.int32` |
| `uint32` | `numpy.uint32` |
| `int64` | `numpy.int64` |
| `uint64` | `numpy.uint64` |
| `float16` | `numpy.float16` |
| `bfloat16` | no native NumPy dtype (e.g. `ml_dtypes.bfloat16`); crosses via DLPack to PyTorch/JAX |
| `float32` | `numpy.float32` |
| `float64` | `numpy.float64` |
| `complex64` | `numpy.complex64` |
| `complex128` | `numpy.complex128` |

### Tier 2 and quantized types

For **Tier 2 element types** (sub-byte integers, float8 variants) and **quantized
types**, `hurray-python` MUST expose a `hurray`-namespaced dtype object (e.g.,
`hurray.dtype.int4`, `hurray.dtype.float8_e4m3`). These have no standard NumPy dtype
and cross between Hurray-aware components via the native buffer protocol or `save`/`load`.

Tensors with Tier 2 / quantized dtypes MAY still implement `__dlpack__` where DLPack
supports the element type. For element types outside the DLPack type enum, `__dlpack__`
MUST raise the Python built-in `BufferError`.

## DLPack Interoperability

`hurray-python` MUST support DLPack for all element types that DLPack defines:
`float16`, `bfloat16`, `float32`, `float64`, `int8`, `uint8`, `int16`, `uint16`,
`int32`, `uint32`, `int64`, `uint64`, `complex64`, `complex128`, `bool`.

- `__dlpack__()` MUST return a PyCapsule named `"dltensor_versioned"` conforming to the DLPack
  specification (v1.0 or later).
- `__dlpack__()` MUST raise the Python built-in `BufferError` for any element type
  not in the DLPack type enum (e.g., `int4`, `float8` variants, quantized types).
  `BufferError` is the conventional signal used across the ecosystem (NumPy, PyTorch)
  for a tensor that cannot be represented in DLPack.
- The DLPack capsule MUST reference the original buffer without copying. The buffer's
  reference count MUST be incremented when the capsule is created and decremented when
  the capsule is consumed or deleted.

### Stream parameter semantics

The `stream` parameter of `__dlpack__(stream=None)` maps to the tensor's `SyncMode`:

| `stream` value | Requirement |
|---|---|
| `None` | The tensor MUST have `SyncMode::ProducerSynced`. The buffer is already fully written; no synchronisation is required by the consumer. |
| `-1` | The binding layer MUST perform a device-level synchronisation (equivalent to `cudaDeviceSynchronize` on CUDA) before returning the capsule. |
| Positive integer (stream handle) | If the tensor is `ProducerSynced`, the buffer is already ready; the stream argument MUST be ignored. Tensors with `SyncMode::Event` or `SyncMode::ConsumerStream` are out of scope for the initial Layer 8a implementation; the binding MUST raise `BufferError` for these modes. |

### Device ID

DLPack requires a `device_id` integer (e.g., GPU index) that is not stored in the
Hurray wire format. The `device_id` MUST be passed as runtime metadata when
constructing a `hurray.Tensor` and defaults to `0` (the first device of that type).
`__dlpack_device__()` MUST return this runtime `device_id`.

- `__dlpack_device__()` MUST return the correct `(DLDeviceType, device_id)` pair
  according to the [Device Tag Mapping (Hurray ↔ DLPack)](#device-tag-mapping-hurray--dlpack)
  table below.

### Device and Memory Class Mapping (Hurray → DLPack)

The correct DLPack `DLDeviceType` for a Hurray buffer is determined by the
**combination** of `device_tag` and `memory_class`. DLPack encodes what Hurray
separates into two orthogonal fields as a single flat enum (e.g., `kDLCUDAHost`,
`kDLCUDAManaged`). The binding layer is responsible for the translation; neither
raw Hurray values NOR raw DLPack integers are stored in the other system's fields.

See ADR-020 for the rationale behind the two-field design.

#### Full mapping table

| Hurray `device_tag` | Hurray `memory_class` | DLPack `DLDeviceType` | DLPack int |
|---|---|---|---|
| `0x00` CPU | `STANDARD` | `kDLCPU` | 1 |
| `0x00` CPU | `HOST_PINNED` | `kDLCPU` | 1 |
| `0x00` CPU | `UNIFIED` | `kDLCPU` | 1 |
| `0x01` CUDA | `STANDARD` | `kDLCUDA` | 2 |
| `0x01` CUDA | `HOST_PINNED` | `kDLCUDAHost` | 3 |
| `0x01` CUDA | `UNIFIED` | `kDLCUDAManaged` | 13 |
| `0x01` CUDA | `PEER` | — | raise `hurray.UnsupportedError` |
| `0x02` ROCm | `STANDARD` | `kDLROCM` | 10 |
| `0x02` ROCm | `HOST_PINNED` | `kDLROCMHost` | 11 |
| `0x02` ROCm | `UNIFIED` | — | raise `hurray.UnsupportedError` |
| `0x02` ROCm | `PEER` | — | raise `hurray.UnsupportedError` |
| `0x03` Metal | `STANDARD` | `kDLMetal` | 8 |
| `0x03` Metal | `HOST_PINNED` | `kDLMetal` | 8 |
| `0x03` Metal | `UNIFIED` | `kDLMetal` | 8 |
| `0x04` Vulkan | `STANDARD` | `kDLVulkan` | 7 |
| `0x04` Vulkan | `HOST_PINNED` | `kDLVulkan` | 7 |
| `0x04` Vulkan | `UNIFIED` | `kDLVulkan` | 7 |
| `0x04` Vulkan | `PEER` | — | raise `hurray.UnsupportedError` |
| `0x05` WebGPU | `STANDARD` | `kDLWebGPU` | 15 |
| `0x06` Hexagon | `STANDARD` | `kDLHexagon` | 16 |
| `0x06` Hexagon | `HOST_PINNED` | `kDLHexagon` | 16 |
| `0x06` Hexagon | `UNIFIED` | `kDLHexagon` | 16 |
| `0x07` Level Zero | `STANDARD` | `kDLOneAPI` | 14 |
| `0x07` Level Zero | `HOST_PINNED` | `kDLOneAPI` | 14 |
| `0x07` Level Zero | `UNIFIED` | `kDLOneAPI` | 14 |
| `0x07` Level Zero | `PEER` | — | raise `hurray.UnsupportedError` |
| `0x08` OpenCL | `STANDARD` | `kDLOpenCL` | 4 |
| `0x08` OpenCL | `HOST_PINNED` | `kDLOpenCL` | 4 |
| `0x08` OpenCL | `UNIFIED` | `kDLOpenCL` | 4 |
| `0xF0`–`0xFE` | any | — | raise `hurray.UnsupportedError` |

Combinations not listed above (e.g., WebGPU + HOST_PINNED, which is invalid per
the per-device validity table in ADR-020) MUST NOT occur in a conforming
descriptor; if encountered, the binding layer SHOULD raise `hurray.UnsupportedError`.

Notes on the mapping:

1. Hurray's `device_tag` values are intentionally distinct from DLPack's
   `DLDeviceType` integers (ADR-016). Translation is the binding layer's
   responsibility — Hurray tag values MUST NOT be passed directly to DLPack
   consumers, and DLPack integers MUST NOT be stored in a Hurray buffer handle.
2. When implementing `__dlpack_device__`, the binding layer MUST derive the
   `DLDeviceType` from the `(device_tag, memory_class)` pair using the table
   above. It MUST NOT return the raw Hurray `device_tag` value.
3. CPU `HOST_PINNED` and `UNIFIED` both map to `kDLCPU` because DLPack does not
   distinguish host-pinned from ordinary host memory from the CPU's perspective.
   The `memory_class` field carries this information for consumers that need it.
4. Metal `STANDARD`, `HOST_PINNED`, and `UNIFIED` all map to `kDLMetal` (8)
   because DLPack does not distinguish Metal storage modes. Consumers that need
   to distinguish them MUST read the Hurray `memory_class` field directly. Note:
   `HOST_PINNED` (`StorageManaged`) is deprecated on Apple Silicon; see ADR-020.
5. ROCm `UNIFIED` has no DLPack equivalent (`kDLROCMManaged` does not exist in
   DLPack v1.0). The binding MUST raise `hurray.UnsupportedError`.
6. `PEER` memory has no DLPack equivalent for any device type. The binding MUST
   raise `hurray.UnsupportedError` for any `PEER` buffer exposed via DLPack.
7. For implementation-private device tags (`0xF0`–`0xFE`), the binding layer
   MUST NOT fabricate a DLPack mapping. It MUST raise `hurray.UnsupportedError`
   unless the consumer has explicitly agreed on a private mapping out of band.
8. For combinations in notes 5, 6, and 7 where `hurray.UnsupportedError` is
   raised, Hurray-aware consumers SHOULD use the native buffer interchange
   protocol (`__hurray_buffer__`) instead. See
   [Native Buffer Interchange Protocol](#native-buffer-interchange-protocol).

## Native Buffer Interchange Protocol

`hurray-python` MUST expose a native buffer interchange protocol for
hurray-to-hurray zero-copy transfers covering `(device_tag, memory_class)`
combinations that DLPack v1.0 cannot represent (see ADR-023).

- `hurray.Tensor.__hurray_buffer__(stream=None) -> PyCapsule` — MUST return a
  PyCapsule named `"hurray_buffer"` wrapping a `HurrayBufferList` pointer from
  `hurray-ffi`. Available for **all** dtypes (Tier 1, Tier 2, quantized).
- `hurray.from_hurray_buffer(obj, /) -> hurray.Tensor` — MUST accept any object
  whose `__hurray_buffer__` returns a valid capsule and reconstruct a
  `hurray.Tensor` that owns the transferred buffers.

### Multi-buffer tensors

A tensor whose descriptor references more than one buffer — per-channel / NF4 /
MXFP quantization, sparse layouts, block-paged, composite — MUST carry **every**
buffer in a single capsule (ADR-030).

- The capsule pointer MUST be a `HurrayBufferList` owning one `HurrayBuffer` per
  buffer. A single-buffer tensor is the `N = 1` case, not a separate path.
- Element `i` of the list MUST be the buffer at index `i` of the descriptor's
  buffer table. Buffer indices appearing in quantization descriptors
  (`scale_buffer_index`, `zero_point_buffer_index`), layout descriptors, and
  composite members index the list directly.
- A producer MUST NOT emit a capsule whose list length differs from the
  descriptor's buffer count, and a consumer MUST reject such a capsule rather
  than construct a tensor whose buffer indices do not resolve.
- `hurray.Tensor` MUST expose the descriptor's optional sections for reading:
  `quantization` (returning the scheme class, or `None`), `statistics`, `shard`,
  and `buffer_count`. The quantization getter MUST return an object of the same
  class the constructor accepts, so an inspected scheme can be reused to build
  another tensor without conversion.
- A sparse-layout `hurray.Tensor` MUST carry its component buffers through this protocol,
  with its values and index buffers in descriptor order. A separate
  `__hurray_sparse_buffer__` protocol MUST NOT be introduced: sparse is the
  multi-buffer case, not a distinct one.
- `hurray.save()` MUST write every buffer of a tensor, and `hurray.load()` MUST
  accept multi-buffer tensors, rejecting any whose buffer count disagrees with
  its descriptor.

### Capsule lifetime

The PyCapsule lifetime rules MUST match DLPack discipline:

- **Capsule name on creation:** `"hurray_buffer"`.
- **Capsule name after consumption:** `"used_hurray_buffer"`. The consumer MUST
  rename the capsule before taking ownership, exactly as DLPack consumers rename
  `"dltensor"` to `"used_dltensor"`.
- **Capsule destructor:** if the capsule is destroyed while still named
  `"hurray_buffer"`, the destructor MUST call `hurray_buffer_list_destroy` on the
  wrapped pointer, which destroys every handle the list owns. If the capsule has
  been consumed (renamed), the consumer MUST call `hurray_buffer_list_destroy`
  exactly once. Handles obtained from `hurray_buffer_list_get` are **borrowed**
  and MUST NOT be destroyed individually.
- The source `Tensor`'s Python reference count MUST be incremented when the
  capsule is created and decremented in both the destructor and consume paths.
  See [Buffer Lifetime and Ownership](#buffer-lifetime-and-ownership).

### `stream` parameter

`__hurray_buffer__(stream=None)` uses the same `stream` semantics as `__dlpack__`:
see [Stream parameter semantics](#stream-parameter-semantics).

### ABI versioning

The capsule context MUST include the `HURRAY_C_ABI_VERSION` constant from the
producing `hurray-ffi` build. `hurray.from_hurray_buffer` MUST verify the version
before dereferencing the handle; a mismatch MUST raise `hurray.UnsupportedError`.

### Discovery

Consumers MUST discover support by probing
`hasattr(obj, '__hurray_buffer__')`. There is no separate capability flag on the
`hurray` namespace.

## Buffer Lifetime and Ownership

Zero-copy interop requires that the source object's buffer remains valid for the
entire lifetime of any object that holds a pointer to it. The binding layer MUST
enforce the following rules:

- **`hurray.from_numpy(array)` → `hurray.Tensor`**: The `Tensor` MUST hold a
  strong Python reference to the source `ndarray` for its own lifetime. The
  `ndarray` MUST NOT be garbage-collected while the `Tensor` is alive.

- **`hurray.Tensor.__array__()` → `ndarray`**: The returned `ndarray` MUST
  reference the source `Tensor` as its `base` object (via NumPy's `base`
  attribute or an equivalent mechanism), so that the `Tensor` is kept alive for
  as long as the `ndarray` holds the buffer.

- **`hurray.Tensor.__dlpack__()` → capsule**: The DLPack capsule destructor MUST
  decrement the `Tensor`'s Python reference count when the capsule is consumed or
  deleted. The reference count MUST be incremented when the capsule is created.
  This ensures the `Tensor` (and therefore its buffer) is not freed while a
  DLPack consumer holds the capsule.

The same rules apply to the component `Tensor` views a sparse-layout tensor hands out.

## NumPy Interoperability

For CPU tensors with Tier 1 element types, `hurray-python` MUST support:

- `hurray.Tensor.__array__()` — MUST return a NumPy `ndarray` backed by the same
  buffer (zero-copy for C-contiguous / row-major tensors; a copy is acceptable for
  other layouts if NumPy cannot represent them natively). See
  [Buffer Lifetime and Ownership](#buffer-lifetime-and-ownership).
- `hurray.from_numpy(array)` — MUST create a `hurray.Tensor` that shares the NumPy
  array's buffer without copying, for C-contiguous arrays. See
  [Buffer Lifetime and Ownership](#buffer-lifetime-and-ownership).

For tensors with Tier 2 / quantized element types, `hurray.Tensor.__array__()` MUST
raise `hurray.UnsupportedError`. NumPy has no dtype for `int4`, `float8` variants, or
quantized/scaled types; returning an `ndarray` is structurally impossible. Callers that
need the raw bytes SHOULD use the native buffer protocol or DLPack directly.

## PyTorch Interoperability

For CPU and CUDA tensors, `hurray-python` MUST support zero-copy conversion via
DLPack:

- `hurray.from_torch(tensor)` — MUST call `tensor.__dlpack__()` and wrap the result
  in a `hurray.Tensor` without copying.
- `hurray.Tensor.to_torch()` — MUST call `self.__dlpack__()` and construct a
  `torch.Tensor` via `torch.utils.dlpack.from_dlpack` without copying.

## Layouts and Sparse Tensor Support

`hurray-python` MUST expose exactly **one** tensor class, `hurray.Tensor`, for every
layout (ADR-031). There MUST NOT be a separate class per layout family: a sparse
tensor is a `hurray.Tensor` whose layout happens to be COO, CSR, or CSC, matching the
format, where sparse is a `layout_tag` inside the ordinary tensor descriptor.

- `.layout` — MUST report the descriptor's layout as a string: `"row_major"`,
  `"col_major"`, `"strided"`, `"tiled"`, `"morton"`, `"hilbert"`, `"coo"`, `"csr"`,
  `"csc"`, `"csf"`, `"block_paged"`, `"composite"`, or `"extension"` for a private or
  unrecognised tag.
- `.values` — a `hurray.Tensor` view over the values buffer.
- `.indices` (COO) or `.col_indices` / `.row_ptr` (CSR) or `.row_indices` / `.col_ptr`
  (CSC) — `hurray.Tensor` views over the index buffers.
- `.nnz` — the stored non-zero count, for layouts that track one.
- `.to_scipy()` — MUST convert a CSR or CSC tensor to the corresponding
  `scipy.sparse` matrix type without copying where scipy's memory layout is
  compatible.
- `hurray.from_scipy(matrix)` — MUST wrap a `scipy.sparse` matrix as a
  `hurray.Tensor` without copying.

Accessors that do not apply to a tensor's layout MUST raise `AttributeError`, so that
`hasattr` reports whether a tensor actually supports them (ADR-031 § 2, extending
design decision D10). They MUST NOT raise `hurray.UnsupportedError`, which would make
`hasattr` return `True` for every accessor on every tensor.

Protocols that require a densely addressable element buffer — `__dlpack__`,
`__array__`, `__array_interface__`, `to_torch` — MUST reject non-dense layouts, naming
the layout in the error message.

## Error Handling

All errors from the Rust core MUST be surfaced as Python exceptions:

| Rust error | Python exception |
|---|---|
| Parse / validation errors | `hurray.InvalidDescriptorError` (subclass of `ValueError`) |
| Buffer size / alignment errors | `hurray.BufferError` (subclass of `ValueError`) |
| Unsupported type or layout | `hurray.UnsupportedError` (subclass of `NotImplementedError`) |
| File I/O errors (Layer 8b) | `hurray.FileError` (subclass of `OSError`) |
| Stream I/O errors (Layer 8b) | `hurray.StreamError` (subclass of `OSError`) |

`hurray.FileError` is raised by file-level operations (`hurray.load()`,
`hurray.save()`): file not found, permission denied, corrupt HRRYFILE container,
unexpected EOF.

`hurray.StreamError` is raised by the streaming reader/writer: frame corruption
mid-stream, unexpected stream termination, framing errors on a pipe or socket.

Both `FileError` and `StreamError` are subclasses of `OSError`; callers that do
not need to distinguish between the two MAY catch `OSError` directly.

`FileError` and `StreamError` are introduced in Layer 8b (file I/O bridge) and
are not present in Layer 8a (core types + DLPack).

Panics from the Rust core MUST NOT propagate as Python crashes. The PyO3 binding
layer MUST catch panics and convert them to `hurray.InternalError` (subclass of
`RuntimeError`).

`hurray.from_hurray_buffer` (Layer 8c) MUST raise:
- `hurray.BufferError` if the capsule is null, already consumed (named
  `"used_hurray_buffer"`), or otherwise invalid.
- `hurray.UnsupportedError` if the `HURRAY_C_ABI_VERSION` embedded in the capsule
  does not match the consumer's linked version.

## Conformance and Validation

`hurray-python` is validated against the shared **golden test-vector corpus**
(`conformance/vectors/`), the same corpus the Rust implementation is checked against,
plus the binding's own unit and integration tests.

- The Python binding MUST decode every descriptor and buffer in the golden corpus to
  the same logical values as the Rust reference, and MUST re-encode round-trippable
  vectors to byte-identical output. This is exercised in CI (see the
  `python-conformance` job).
- The `array-api-tests` suite is **not** used: it targets a whole conforming Array API
  namespace and presupposes the compute core, which `hurray-python` deliberately does
  not implement (see ADR-029). Cross-checking against the golden corpus maps directly
  onto the parts of the Hurray specification the binding actually covers.
- New behaviour MUST ship with tests that exercise the public interop surface (DLPack,
  NumPy/PyTorch bridges, the native buffer protocol, `save`/`load`).

## Benchmark Suite

`hurray-python` SHOULD maintain a benchmark suite that measures performance across the
Hurray interchange and interop paths — the operations that define the binding's
purpose: constructing tensors, moving buffers zero-copy, and serializing/parsing the
Hurray format.

### Benchmark categories

The suite SHOULD cover the following categories:

| Category | Representative benchmarks |
|---|---|
| **DLPack capsule** | `__dlpack__()` round-trip (create + consume); capsule destructor overhead; `from_dlpack()` from NumPy and PyTorch. |
| **NumPy interop** | `from_numpy()` (zero-copy); `__array__()` (zero-copy); dtype coverage (all Tier 1 types). |
| **PyTorch interop** | `from_torch()` and `to_torch()` round-trip on CPU and CUDA. |
| **Native buffer** | `__hurray_buffer__()` / `from_hurray_buffer()` round-trip (full-fidelity, all dtypes). |
| **Construction** | `zeros`, `ones`, `full`, `arange`, `linspace` for representative shapes and dtypes. |
| **Serialization** | `save`/`load` and streaming read/write throughput (GiB/s) for representative tensors. |
| **Memory lifecycle** | `Tensor` allocation + deallocation throughput; large-tensor zero-copy overhead (GiB-scale). |
| **Sparse layouts** | COO/CSR/CSC construction; SciPy round-trip; file round-trip. |

### Tooling

- Benchmarks MUST be runnable via a standard Python benchmarking tool (e.g.,
  `pytest-benchmark` or `airspeed-velocity (asv)`).
- Benchmarks SHOULD report: mean, standard deviation, and minimum latency; throughput
  in GiB/s for memory-bound operations.
- Regression tracking (detecting performance regressions across commits) SHOULD be
  automated in CI. Benchmarks that regress by more than 10% relative to the baseline
  SHOULD trigger a warning in the pull request.
- The benchmark suite MUST be runnable independently from the validation tests.

## Compatibility Matrix

The `hurray-python` package MUST maintain a compatibility matrix document at
`hurray-python/COMPAT-MATRIX.md`. This document lives alongside the code — not in
the spec — because it changes with every release.

The compatibility matrix MUST record, for each `hurray-python` release series:

- The [DLPack](https://dmlc.github.io/dlpack/latest/) specification version(s)
  supported as producer and as consumer.
- The supported CPython version range.
- The Hurray format/descriptor version(s) the binding produces and consumes.

The matrix MUST be updated whenever a new `hurray-python` release changes any of these.

## Packaging

- The package MUST be installable via `pip install hurray`.
- Wheels MUST be provided for CPython ≥ 3.10 on Linux (x86_64, aarch64),
  macOS (x86_64, arm64), and Windows (x86_64).
- The package MUST NOT require a Rust toolchain at install time (pre-built wheels
  are mandatory for distribution).
- Optional dependencies: `numpy`, `torch`, `scipy` (none required at import time;
  interop functions raise `ImportError` if the target library is not installed).
