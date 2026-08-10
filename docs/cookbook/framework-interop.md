# Framework Interop

`hurray-python` is a codec and **zero-copy bridge**, not a compute library: you move a
tensor into NumPy, PyTorch, JAX, or CuPy and do the math there. For dense Tier 1 tensors
the hand-off is **DLPack**, which every major array framework speaks.

## The universal path: DLPack

Any framework whose `from_dlpack` accepts an object exposing `__dlpack__` can consume a
Hurray tensor with no copy:

```python
import numpy as np
import hurray

t = hurray.from_numpy(np.arange(6, dtype=np.float32).reshape(2, 3))

arr = np.from_dlpack(t)          # NumPy, zero-copy — shares t's buffer
arr[0, 0] = 42.0
assert hurray.from_numpy(arr)    # the write is visible through the shared buffer
```

DLPack also carries the tensor's **device**: `t.__dlpack_device__()` returns the
`(DLDeviceType, device_id)` pair, so a consumer sends data to the right place.

## NumPy

```python
# Ingest a NumPy array zero-copy (C-contiguous):
t = hurray.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

# Export back to NumPy — via DLPack, or the __array__ protocol (with optional cast):
a = np.from_dlpack(t)
a64 = t.__array__(dtype=np.float64)   # __array__ may copy when casting
```

## PyTorch

DLPack works directly (`torch.from_dlpack(t)`), and `hurray` ships one-call conveniences:

```python
import torch

torch_t = t.to_torch()               # hurray.Tensor → torch.Tensor, zero-copy
back = hurray.from_torch(torch_t)    # torch.Tensor → hurray.Tensor, zero-copy
```

## JAX

```python
import jax.numpy as jnp

x = jnp.from_dlpack(t)               # zero-copy on the same device
```

## CuPy (GPU)

For a Hurray tensor already on a CUDA device (`device_tag = CUDA`), CuPy shares the
device buffer — no host round-trip:

```python
import cupy as cp

g = cp.from_dlpack(t)                # device-to-device, zero-copy
```

## What DLPack cannot carry

DLPack only describes **dense, strided, standard-dtype** tensors. Some things fall
outside it:

- **`bool`** — Hurray packs it 1 bit per element; DLPack's `bool` is 1 byte, so there is
  no zero-copy mapping. Use `__array__` / `from_numpy` instead.
- **`bfloat16`** — no native NumPy dtype; it crosses to PyTorch/JAX via DLPack but not to
  plain NumPy.
- **Everything beyond dense Tier 1** — sparse layouts, quantized and sub-byte element
  types, tiled/Morton/Hilbert/composite layouts. DLPack has no vocabulary for these.

For those, use Hurray's own full-fidelity protocol.

## The native buffer protocol

`__hurray_buffer__` / `hurray.from_hurray_buffer` exchange the **entire** tensor
descriptor — quantization, sparse and exotic layouts, sub-byte types, device and sync
metadata — between Hurray-aware components, zero-copy:

```python
capsule = t.__hurray_buffer__()      # full-fidelity, all dtypes
u = hurray.from_hurray_buffer(t)     # reconstruct from any object exposing it
```

> **What adoption would unlock (non-normative).** Today only `hurray-python` implements
> `__hurray_buffer__`, so full-fidelity exchange is Hurray-to-Hurray. If a framework
> adopted the protocol, the copies that live at the *edges* today would disappear. For
> example, `hurray.from_scipy` / `hurray.sparse_coo` currently repack SciPy's separate
> `row`/`col` arrays into Hurray's packed `[nnz, rank]` layout (one interleave copy); a
> SciPy that spoke `__hurray_buffer__` could hand its sparse structure across without that
> copy. The same applies to quantized and sub-byte tensors, which have no DLPack
> representation at all — a consumer implementing the native protocol could receive them
> directly instead of falling back to `save`/`load`.

## See also

- [Quickstart](quickstart.md) — the shortest path in and out of a tensor.
- [Python: DLPack and NumPy Interop](hurray-python-dlpack-numpy.md) — more on the DLPack
  capsule and `__array__` details.
- [Python: Native Buffer Protocol](hurray-python-native-buffer.md) — the capsule lifetime
  and ABI-version rules behind `__hurray_buffer__`.
- [Python: Sparse Tensors and SciPy](hurray-python-sparse-scipy.md) — CSR/CSC/COO
  construction, including where the edge copies occur.
