# Python Bindings Requirements — Hurray Implementation Requirements

## Overview

The `hurray-python` package exposes Hurray tensors to the Python ecosystem. It is
built on PyO3 and targets two primary interoperability goals:

1. **Python Array API compliance** — Hurray tensors with Tier 1 element types can be
   used as drop-in inputs to any library that consumes the Python Array API Standard.
2. **Zero-copy interop** — buffers are shared with NumPy, PyTorch, JAX, and CuPy
   without copying, via DLPack and the buffer protocol.

## Python Array API Compliance

`hurray-python` MUST expose a `hurray.Tensor` class that implements the
[Python Array API Standard](https://data-apis.org/array-api/) for tensors with
**Tier 1 element types** (see `docs/spec/element-types.md`).

### Required dunder methods

| Method | Requirement |
|---|---|
| `__array_namespace__(api_version=None)` | MUST return a `hurray` namespace object that implements the Array API function set. |
| `__dlpack__(stream=None)` | MUST return a DLPack capsule for zero-copy buffer sharing. |
| `__dlpack_device__()` | MUST return a `(device_type, device_id)` tuple consistent with the tensor's `device_tag`. |
| `dtype` | MUST return an Array API-compatible dtype object for Tier 1 types. |
| `shape` | MUST return a tuple of ints (or `None` for dynamic dimensions). |
| `ndim` | MUST return the number of dimensions. |
| `size` | MUST return the total number of elements. |
| `device` | MUST return a device object consistent with `__dlpack_device__`. |
| `T` | MUST return a transposed view (rank-2 tensors) without copying. |

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
- The DLPack capsule MUST reference the original buffer without copying. The buffer's
  reference count MUST be incremented when the capsule is created and decremented when
  the capsule is consumed or deleted.
- `__dlpack_device__()` MUST return the correct `(DLDeviceType, device_id)` pair:
  - CPU: `(1, 0)`
  - CUDA: `(2, device_id)`
  - ROCm: `(10, device_id)`
  - Metal: `(8, device_id)`

## NumPy Interoperability

For CPU tensors with Tier 1 element types, `hurray-python` MUST support:

- `hurray.Tensor.__array__()` — MUST return a NumPy `ndarray` backed by the same
  buffer (zero-copy for C-contiguous / row-major tensors; a copy is acceptable for
  other layouts if NumPy cannot represent them natively).
- `hurray.from_numpy(array)` — MUST create a `hurray.Tensor` that shares the NumPy
  array's buffer without copying, for C-contiguous arrays.

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
| I/O errors | `hurray.IOError` (subclass of `OSError`) |

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
