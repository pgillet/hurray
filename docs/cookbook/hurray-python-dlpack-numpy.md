# DLPack and NumPy Zero-Copy Interop

This guide covers zero-copy data exchange between `hurray.Tensor` and NumPy / PyTorch
via the DLPack v1.0 protocol.

## DLPack export

`hurray.Tensor` implements `__dlpack__()` and `__dlpack_device__()`, so any DLPack v1.0
consumer (NumPy 2.0, PyTorch 2.1+, JAX, CuPy) can consume it without copying:

```python
import numpy as np, struct, hurray

buf = struct.pack("6f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
t = hurray.Tensor(buf, hurray.float32, [2, 3])

# Emit a DLPack v1.0 capsule ("dltensor_versioned").
# The capsule holds a strong reference to `t`, keeping the buffer alive.
arr = np.from_dlpack(t)
assert arr.shape == (2, 3)
assert arr.dtype == np.float32
```

`__dlpack_device__()` returns a `(DLDeviceType, device_id)` tuple:

```python
device_type, device_id = t.__dlpack_device__()
assert device_type == 1   # kDLCPU
assert device_id == 0
```

### Supported dtypes

All Tier 1 Hurray types map to DLPack: `int8/16/32/64`, `uint8/16/32/64`,
`float16`, `bfloat16`, `float32`, `float64`, `complex64`, `complex128`.

`bool` raises `builtins.BufferError` — Hurray packs 8 booleans per byte while DLPack
uses 1 byte per element; there is no lossless zero-copy mapping.

Tier 2 / quantized types (`int4`, `float8` variants) also raise `builtins.BufferError`.

### Forward-compatibility kwargs

`__dlpack__` accepts `stream`, `max_version`, `dl_device`, and `copy` keyword arguments
for compatibility with DLPack v1.0 consumers. Only `stream` has defined semantics (all
tensors are `ProducerSynced`; GPU stream handling is deferred to a future pass).

## NumPy interop

### Zero-copy import: `hurray.from_numpy`

```python
import numpy as np, hurray

arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
t = hurray.from_numpy(arr)
assert t.shape == (2, 3)
assert t.dtype == hurray.float32
```

`from_numpy` stores a raw pointer into NumPy's buffer and holds a strong Python
reference to `arr` — no copy is made. The NumPy array must be **C-contiguous**
(row-major). For Fortran-order or strided arrays, call `numpy.ascontiguousarray` first:

```python
f_arr = np.asfortranarray(np.zeros((3, 4), dtype=np.float32))
t = hurray.from_numpy(np.ascontiguousarray(f_arr))
```

Attempting to pass a non-C-contiguous array raises `hurray.UnsupportedError`.

### Zero-copy export: `Tensor.__array__`

```python
t = hurray.Tensor(bytes(24), hurray.float32, [2, 3])
arr = t.__array__()
assert arr.shape == (2, 3)
assert arr.dtype == np.float32
```

Pass a target dtype to cast (a copy is made when casting):

```python
arr_f64 = t.__array__(dtype=np.float64)
assert arr_f64.dtype == np.float64
```

Pass `copy=False` if you want to assert that no copy will occur — raises
`hurray.CopyRequiredError` if a dtype cast is needed:

```python
import hurray

try:
    arr = t.__array__(dtype=np.float64, copy=False)
except hurray.CopyRequiredError:
    print("A copy would be required for the dtype cast")
```

`__array__` is only supported for **CPU tensors** with **Tier 1 element types**.
Non-CPU tensors and Tier 2 / quantized dtypes raise `hurray.UnsupportedError`.

## PyTorch interop

### Zero-copy export: `Tensor.to_torch`

```python
import hurray

t = hurray.Tensor(bytes(16), hurray.float32, [4])
torch_t = t.to_torch()  # raises ImportError if torch is not installed
```

### Zero-copy import: `hurray.from_torch`

```python
import torch, hurray

torch_t = torch.zeros(2, 3, dtype=torch.float32)
t = hurray.from_torch(torch_t)
assert t.shape == (2, 3)
assert t.dtype == hurray.float32
```

`torch` is imported at call time — `import hurray` does not require PyTorch to be
installed.

## Error reference

| Error | When raised |
|---|---|
| `builtins.BufferError` | `__dlpack__` on bool, int4, float8, or quantized types |
| `hurray.UnsupportedError` | non-CPU `__array__`, Tier 2 dtype `__array__`, non-C-contiguous `from_numpy` |
| `hurray.CopyRequiredError` | `__array__(copy=False)` when a dtype cast is needed |
| `hurray.UnsupportedError` | DLPack device/layout not representable (PEER memory, tiled layout) |
| `ImportError` | `to_torch` or `from_torch` when PyTorch is not installed |
