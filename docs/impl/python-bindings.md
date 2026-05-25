# Python Bindings Requirements — Hurray Implementation Requirements

## Overview

The `hurray-python` package exposes Hurray tensors to the Python ecosystem. It is
built on PyO3 and targets two primary interoperability goals:

1. **Python Array API compliance** — Hurray tensors with Tier 1 element types can be
   used as drop-in inputs to any library that consumes the Python Array API Standard.
2. **Zero-copy interop** — buffers are shared with NumPy, PyTorch, JAX, and CuPy
   without copying, via DLPack and the buffer protocol.

`hurray-python` MUST be a **strict reference implementation** of the
[Python Array API Standard](https://data-apis.org/array-api/) for all Tier 1
element types. Where the Array API standard specifies a behaviour (return value,
exception type, error condition), `hurray-python` MUST follow it exactly. Hurray
extensions (Tier 2 types, quantized types, device-specific behaviours) MUST NOT
contradict the Array API standard; they operate in the space the standard explicitly
leaves to implementations.

## Python Array API Compliance

`hurray-python` MUST expose a `hurray.Tensor` class that implements the
[Python Array API Standard](https://data-apis.org/array-api/) for tensors with
**Tier 1 element types** (see `docs/spec/element-types.md`).

### Required dunder methods

| Method | Requirement |
|---|---|
| `__array_namespace__(api_version=None)` | MUST return a `hurray` namespace object that implements the Array API function set. |
| `__dlpack__(stream=None)` | MUST return a DLPack capsule for zero-copy buffer sharing. See [DLPack Interoperability](#dlpack-interoperability) for stream semantics. |
| `__dlpack_device__()` | MUST return a `(DLDeviceType, device_id)` tuple. `device_id` is passed as runtime metadata at `Tensor` construction time and defaults to `0`. See [DLPack Interoperability](#dlpack-interoperability). |
| `dtype` | MUST return an Array API-compatible dtype object for Tier 1 types. |
| `shape` | MUST return a `Tuple[Optional[int], ...]`. Each element is an `int` for a known dimension, or `None` for a dynamic (unknown) dimension. |
| `ndim` | MUST return the number of dimensions. |
| `size` | MUST return `Optional[int]`: the total number of elements, or `None` if one or more dimensions are dynamic (unknown). |
| `device` | MUST return a device object consistent with `__dlpack_device__`. |
| `T` | MUST return a transposed view without copying for rank-2 tensors. MUST raise `ValueError` if the array is not rank-2. |

### Tier 1 dtype mapping

| Hurray type | Array API dtype |
|---|---|
| `bool` | `xp.bool` |
| `int8` | `xp.int8` |
| `uint8` | `xp.uint8` |
| `int16` | `xp.int16` |
| `uint16` | `xp.uint16` |
| `int32` | `xp.int32` |
| `uint32` | `xp.uint32` |
| `int64` | `xp.int64` |
| `uint64` | `xp.uint64` |
| `float16` | `xp.float16` |
| `bfloat16` | `xp.bfloat16` (Array API 2023.12+) |
| `float32` | `xp.float32` |
| `float64` | `xp.float64` |
| `complex64` | `xp.complex64` |
| `complex128` | `xp.complex128` |

### Tier 2 and quantized types

For **Tier 2 element types** (sub-byte integers, float8 variants) and **quantized
types**, `hurray-python` MUST expose a `hurray`-namespaced dtype object (e.g.,
`hurray.dtype.int4`, `hurray.dtype.float8_e4m3`). These MUST NOT be mapped to a
standard Array API dtype.

Tensors with non-Array-API dtypes MUST NOT implement `__array_namespace__`. They MAY
still implement `__dlpack__` where DLPack supports the type.

## DLPack Interoperability

`hurray-python` MUST support DLPack for all element types that DLPack defines:
`float16`, `bfloat16`, `float32`, `float64`, `int8`, `uint8`, `int16`, `uint16`,
`int32`, `uint32`, `int64`, `uint64`, `complex64`, `complex128`, `bool`.

- `__dlpack__()` MUST return a PyCapsule named `"dltensor"` conforming to the DLPack
  specification (v0.8 or later).
- `__dlpack__()` MUST raise the Python built-in `BufferError` for any element type
  not in the DLPack type enum (e.g., `int4`, `float8` variants, quantized types).
  This follows the Python Array API Standard, which specifies `BufferError` for
  tensors that cannot be represented in DLPack.
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

The same rules apply to `hurray.SparseTensor` and its component `Tensor` views.

## NumPy Interoperability

For CPU tensors with Tier 1 element types, `hurray-python` MUST support:

- `hurray.Tensor.__array__()` — MUST return a NumPy `ndarray` backed by the same
  buffer (zero-copy for C-contiguous / row-major tensors; a copy is acceptable for
  other layouts if NumPy cannot represent them natively). See
  [Buffer Lifetime and Ownership](#buffer-lifetime-and-ownership).
- `hurray.from_numpy(array)` — MUST create a `hurray.Tensor` that shares the NumPy
  array's buffer without copying, for C-contiguous arrays. See
  [Buffer Lifetime and Ownership](#buffer-lifetime-and-ownership).

## PyTorch Interoperability

For CPU and CUDA tensors, `hurray-python` MUST support zero-copy conversion via
DLPack:

- `hurray.from_torch(tensor)` — MUST call `tensor.__dlpack__()` and wrap the result
  in a `hurray.Tensor` without copying.
- `hurray.Tensor.to_torch()` — MUST call `self.__dlpack__()` and construct a
  `torch.Tensor` via `torch.utils.dlpack.from_dlpack` without copying.

## Sparse Tensor Support

`hurray-python` MUST expose COO, CSR, and CSC sparse tensors as `hurray.SparseTensor`
objects with:

- `.values` — a `hurray.Tensor` view over the values buffer.
- `.indices` (COO) or `.col_indices` / `.row_ptr` (CSR) or `.row_indices` / `.col_ptr`
  (CSC) — `hurray.Tensor` views over the index buffers.
- `.to_scipy()` — MUST convert to the corresponding `scipy.sparse` matrix type
  (`coo_matrix`, `csr_matrix`, `csc_matrix`) without copying where scipy's memory
  layout is compatible.
- `.from_scipy(matrix)` — MUST wrap a `scipy.sparse` matrix as a `hurray.SparseTensor`
  without copying.

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

## Packaging

- The package MUST be installable via `pip install hurray`.
- Wheels MUST be provided for CPython ≥ 3.10 on Linux (x86_64, aarch64),
  macOS (x86_64, arm64), and Windows (x86_64).
- The package MUST NOT require a Rust toolchain at install time (pre-built wheels
  are mandatory for distribution).
- Optional dependencies: `numpy`, `torch`, `scipy` (none required at import time;
  interop functions raise `ImportError` if the target library is not installed).
