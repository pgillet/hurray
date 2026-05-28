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

In **strict mode** (the default), tensors with Tier 2 / quantized dtypes MUST NOT
expose `__array_namespace__` — `hasattr(tensor, '__array_namespace__')` MUST return
`False`. In **relaxed mode**, `__array_namespace__` is present on all tensors and
returns the `hurray` namespace. See ADR-022 and [Runtime modes](#runtime-modes) below.

Tensors with Tier 2 / quantized dtypes MAY still implement `__dlpack__` where DLPack
supports the element type. For element types outside the DLPack type enum, `__dlpack__`
MUST raise the Python built-in `BufferError` (in both modes).

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
8. For combinations in notes 5, 6, and 7 where `hurray.UnsupportedError` is
   raised, Hurray-aware consumers SHOULD use the native buffer interchange
   protocol (`__hurray_buffer__`) instead. See
   [Native Buffer Interchange Protocol](#native-buffer-interchange-protocol).

## Native Buffer Interchange Protocol

`hurray-python` MUST expose a native buffer interchange protocol for
hurray-to-hurray zero-copy transfers covering `(device_tag, memory_class)`
combinations that DLPack v1.0 cannot represent (see ADR-023).

- `hurray.Tensor.__hurray_buffer__(stream=None) -> PyCapsule` — MUST return a
  PyCapsule named `"hurray_buffer"` wrapping a `HurrayBuffer` pointer from
  `hurray-ffi`. Available for **all** dtypes (Tier 1, Tier 2, quantized) in
  **both** strict and relaxed modes.
- `hurray.from_hurray_buffer(obj, /) -> hurray.Tensor` — MUST accept any object
  whose `__hurray_buffer__` returns a valid capsule and reconstruct a
  `hurray.Tensor` that owns the transferred buffer.

### Capsule lifetime

The PyCapsule lifetime rules MUST match DLPack discipline:

- **Capsule name on creation:** `"hurray_buffer"`.
- **Capsule name after consumption:** `"used_hurray_buffer"`. The consumer MUST
  rename the capsule before taking ownership, exactly as DLPack consumers rename
  `"dltensor"` to `"used_dltensor"`.
- **Capsule destructor:** if the capsule is destroyed while still named
  `"hurray_buffer"`, the destructor MUST call `hurray_buffer_destroy` on the
  wrapped pointer. If the capsule has been consumed (renamed), the consuming
  `Tensor` MUST call `hurray_buffer_destroy` exactly once at its own finalisation.
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

### Layer 8a / 8b status

`__hurray_buffer__` and `hurray.from_hurray_buffer` are **reserved** in Layers 8a
and 8b but NOT implemented. `hasattr(tensor, '__hurray_buffer__')` MUST return
`False` until Layer 8c ships.

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

For tensors with Tier 2 / quantized element types, `hurray.Tensor.__array__()` MUST
raise `hurray.UnsupportedError` in **both** strict and relaxed modes. NumPy has no
dtype for `int4`, `float8` variants, or quantized/scaled types; returning an `ndarray`
is structurally impossible regardless of compliance mode. Relaxed mode makes
`__array_namespace__` accessible on Tier 2 tensors but does not grant NumPy
representability. Callers that need the raw bytes SHOULD use the buffer protocol or
DLPack directly.

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

`hurray.from_hurray_buffer` (Layer 8c) MUST raise:
- `hurray.BufferError` if the capsule is null, already consumed (named
  `"used_hurray_buffer"`), or otherwise invalid.
- `hurray.UnsupportedError` if the `HURRAY_C_ABI_VERSION` embedded in the capsule
  does not match the consumer's linked version.

## Runtime Modes

`hurray-python` supports two runtime compliance modes (see ADR-022 for the full
rationale and architecture):

### Strict mode (default)

Strict mode enforces full compliance with the Python Array API Standard for all
operations involving Tier 1 element types. This is the default mode.

- `hurray.Tensor` instances with Tier 2 / quantized dtypes MUST NOT expose
  `__array_namespace__` — `hasattr(tensor, '__array_namespace__')` returns `False`.
- Array-API-shaped construction APIs (`hurray.zeros`, `hurray.ones`, `hurray.asarray`,
  etc.) MUST raise `hurray.UnsupportedError` for Tier 2 / quantized dtypes.
- All other Array API invariants (`size`, `T`, `shape`, DLPack semantics) are
  enforced in both modes.

### Relaxed mode

Relaxed mode allows the full Hurray feature set, including Tier 2 / quantized types,
through the standard API surface. The user opts out of Array API conformance
guarantees for the duration of the scope.

- `hurray.Tensor` instances with Tier 2 / quantized dtypes expose `__array_namespace__`
  and return the `hurray` namespace object.
- Array-API-shaped construction APIs accept Tier 2 / quantized dtypes.
- The `hurray` namespace object returned by `__array_namespace__` is the same module
  in both modes; it is not extended or restricted based on mode.

### Public API

| Name | Signature | Description |
|---|---|---|
| `hurray.set_strict(strict)` | `(bool) -> None` | Set the process-wide default mode (ContextVar default). |
| `hurray.is_strict()` | `() -> bool` | Query the current mode in the calling context. |
| `hurray.strict()` | context manager | Enter strict mode for the duration of the `with` block. |
| `hurray.relaxed()` | context manager | Enter relaxed mode for the duration of the `with` block. |

The mode is stored in a `contextvars.ContextVar` (thread-safe and coroutine-safe).
Each OS thread and each asyncio `Task` inherits a copy of the context on spawn;
changes in one thread do not affect concurrent threads.

> **Note (non-normative):** Threads created with the raw `threading.Thread` API
> inherit the ContextVar *default value* (strict), not the spawning thread's current
> mode. This is standard Python `contextvars` behaviour.

### Layer 8a status

Layer 8a (core types + DLPack) implements **strict mode only**. The relaxed path is
reserved but not yet active:

- `hurray.set_strict(True)` is a no-op.
- `hurray.set_strict(False)`, `hurray.relaxed()`, and entering a relaxed scope MUST
  raise `NotImplementedError` with a message indicating the feature is reserved for a
  future release.
- `hurray.is_strict()` always returns `True` in Layer 8a.
- `__hurray_buffer__` is reserved but absent from `hurray.Tensor`;
  `hasattr(tensor, '__hurray_buffer__')` MUST return `False`.
- `hurray.from_hurray_buffer` is reserved but not present as a public name.

Both native protocol names are implemented in Layer 8c (see ADR-023).

## Array API Conformance Testing

`hurray-python` MUST pass the
[Python Array API Standard conformance test suite](https://github.com/data-apis/array-api-tests)
for all Tier 1 element types in strict mode.

- The conformance suite MUST be executed as part of the CI pipeline via GitHub
  Actions on every pull request that touches `hurray-python/`.
- The suite MUST be run against the Array API version(s) declared in the
  [Compatibility Matrix](#compatibility-matrix).
- All tests that are not explicitly skipped with documented justification MUST pass.
  Skips MUST be declared in a `conftest.py` or equivalent file alongside the reason
  (e.g., a feature deferred to a later layer, a known upstream test-suite bug with a
  link to the upstream issue).
- Conformance is tested in strict mode only. Relaxed mode makes no Array API
  conformance claims and MUST NOT be used when running the conformance suite.

> **Note (non-normative):** The array-api-tests suite is run via:
> ```bash
> pip install array-api-tests
> pytest array_api_tests/ --array-module=hurray
> ```
> The `--array-module` flag points the suite at the `hurray` package as the
> Array API namespace under test.

## Benchmark Suite

`hurray-python` SHOULD maintain a benchmark suite that measures performance across
the Array API surface and Hurray-specific interop paths. The long-term goal is to
contribute this suite to the upstream
[Python Array API Standard benchmark project](https://data-apis.org/array-api/latest/benchmark_suite.html),
providing a reference implementation for other Array API consumers to compare against.

### Benchmark categories

The suite SHOULD cover the following categories:

| Category | Representative benchmarks |
|---|---|
| **DLPack capsule** | `__dlpack__()` round-trip (create + consume); capsule destructor overhead; `from_dlpack()` from NumPy and PyTorch. |
| **NumPy interop** | `from_numpy()` (zero-copy); `__array__()` (zero-copy); dtype coverage (all Tier 1 types). |
| **PyTorch interop** | `from_torch()` and `to_torch()` round-trip on CPU and CUDA. |
| **Array API construction** | `zeros`, `ones`, `full`, `arange`, `linspace` for representative shapes and dtypes. |
| **Array API operations** | Elementwise ops (`add`, `multiply`, `exp`); reductions (`sum`, `max`) for representative shapes. |
| **Memory lifecycle** | `Tensor` allocation + deallocation throughput; large-tensor zero-copy overhead (GiB-scale). |
| **SparseTensor** | COO/CSR/CSC construction; SciPy round-trip. |

### Tooling

- Benchmarks MUST be runnable via a standard Python benchmarking tool (e.g.,
  `pytest-benchmark` or `airspeed-velocity (asv)`).
- Benchmarks SHOULD report: mean, standard deviation, and minimum latency; throughput
  in GiB/s for memory-bound operations.
- Regression tracking (detecting performance regressions across commits) SHOULD be
  automated in CI. Benchmarks that regress by more than 10% relative to the baseline
  SHOULD trigger a warning in the pull request.
- The benchmark suite MUST be runnable independently from the conformance test suite.

### Contribution to the Array API Standard

When the upstream
[Python Array API Standard benchmark suite](https://data-apis.org/array-api/latest/benchmark_suite.html)
reaches a stable format, `hurray-python` benchmarks that cover the standard's function
set SHOULD be submitted as contributions. Benchmark results from `hurray-python`
SHOULD be published alongside results from NumPy, PyTorch, JAX, and CuPy to provide
a cross-implementation performance reference.

> **Note (non-normative):** The upstream benchmark suite is in early development as
> of this writing. The `hurray-python` suite is designed to be structurally compatible
> with it from the start (function naming, shape/dtype parametrisation), so that
> upstreaming requires minimal adaptation.

## Compatibility Matrix

The `hurray-python` package MUST maintain a compatibility matrix document at
`hurray-python/COMPAT-MATRIX.md`. This document lives alongside the code — not in
the spec — because it changes with every release.

The compatibility matrix MUST record, for each `hurray-python` release series:

- The supported [Python Array API Standard](https://data-apis.org/array-api/)
  version(s) (e.g., 2022.12, 2023.12, 2025.12).
- The [DLPack](https://dmlc.github.io/dlpack/latest/) specification version(s)
  supported as producer and as consumer.
- The supported CPython version range.

The matrix MUST be updated whenever a new `hurray-python` release adds or drops
support for a standard version. The conformance test suite (see
[Array API Conformance Testing](#array-api-conformance-testing)) MUST be run against
every Array API version listed in the matrix.

## Packaging

- The package MUST be installable via `pip install hurray`.
- Wheels MUST be provided for CPython ≥ 3.10 on Linux (x86_64, aarch64),
  macOS (x86_64, arm64), and Windows (x86_64).
- The package MUST NOT require a Rust toolchain at install time (pre-built wheels
  are mandatory for distribution).
- Optional dependencies: `numpy`, `torch`, `scipy` (none required at import time;
  interop functions raise `ImportError` if the target library is not installed).
