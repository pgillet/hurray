# Tensor Construction Functions

`hurray-python` provides a set of functions for building `hurray.Tensor` objects in
Python: `zeros`, `ones`, `full`, `empty`, their `*_like` variants, `arange`,
`linspace`, `eye`, `asarray`, and `from_dlpack`. These constructors produce **Tier 1**
(standard numeric) tensors, which you then serialize with `save` or hand off zero-copy
to NumPy/PyTorch. Tier 2 / quantized / sparse tensors are not built here — they arrive
via the decode and interop paths.

`hurray.Tensor` is an interchange object, not an Array API array: it exposes an
inspection and interop surface (`shape`, `dtype`, `device`, `__dlpack__`,
`__hurray__`, …), not array computation. See ADR-029.

## Creation functions

All creation functions default to `dtype=float64` when a dtype is not specified.

### `zeros` and `ones`

```python
import hurray

z = hurray.zeros([3, 4])
assert z.shape == (3, 4)
assert z.dtype == hurray.float64

o = hurray.ones([2, 3], dtype=hurray.float32)
assert o.shape == (2, 3)
assert o.dtype == hurray.float32
```

### `full` and `empty`

`full` infers the dtype from the fill value when `dtype` is omitted:

```python
f = hurray.full([4], 7.0)          # float64 inferred
fi = hurray.full([4], 7, dtype=hurray.int32)  # explicit int32

e = hurray.empty([5, 5], dtype=hurray.float64)
```

`empty` zero-initialises the buffer; values must not be relied upon.

### `*_like` variants

Each creation function has a `_like` counterpart that inherits shape and dtype
from a source tensor:

```python
src = hurray.ones([3, 3], dtype=hurray.float32)

z = hurray.zeros_like(src)      # shape=(3,3), dtype=float32
o = hurray.ones_like(src)
f = hurray.full_like(src, -1.0)
e = hurray.empty_like(src)

# Override dtype or device:
z64 = hurray.zeros_like(src, dtype=hurray.float64)
```

### `arange`

Generates integer or float sequences. Dtype is inferred as `int64` when all
arguments are Python integers, `float64` otherwise:

```python
t = hurray.arange(5)               # [0, 1, 2, 3, 4], int64
t2 = hurray.arange(0, 10, 2)      # [0, 2, 4, 6, 8], int64
t3 = hurray.arange(0.0, 1.0, 0.25) # [0.0, 0.25, 0.5, 0.75], float64
```

### `linspace`

Generates `num` evenly spaced values in `[start, stop]`:

```python
t = hurray.linspace(0.0, 1.0, 5)
# [0.0, 0.25, 0.5, 0.75, 1.0]

# Exclude stop:
t2 = hurray.linspace(0.0, 1.0, 4, endpoint=False)
# [0.0, 0.25, 0.5, 0.75]
```

### `eye`

Creates a 2-D identity matrix. `k` offsets the diagonal:

```python
identity = hurray.eye(3)                # 3×3 float64 identity
rect     = hurray.eye(2, 4)             # 2×4 float64 with 1s on main diagonal
upper    = hurray.eye(3, k=1, dtype=hurray.int32)  # k=1 super-diagonal
lower    = hurray.eye(4, k=-1)          # k=-1 sub-diagonal
```

## `asarray` — generic conversion

`asarray` converts Python lists, NumPy arrays, and other array objects to
`hurray.Tensor`. For NumPy arrays and `hurray.Tensor` inputs the data buffer
is shared zero-copy where possible.

```python
import numpy as np

# From a Python list
t = hurray.asarray([1.0, 2.0, 3.0])
assert t.dtype == hurray.float64

# With explicit dtype
t2 = hurray.asarray([[1, 2], [3, 4]], dtype=hurray.int32)
assert t2.shape == (2, 2)

# From NumPy (zero-copy)
np_arr = np.array([10.0, 20.0], dtype=np.float32)
t3 = hurray.asarray(np_arr)
assert t3.dtype == hurray.float32

# From another hurray tensor (zero-copy via DLPack)
src = hurray.zeros([4])
t4 = hurray.asarray(src)
```

**bfloat16 limitation:** NumPy has no native `bfloat16` dtype. Passing
`dtype=hurray.bfloat16` to `asarray` raises `UnsupportedError`. Use
`hurray.from_numpy` on a bfloat16 array from PyTorch or a custom converter
instead.

## `from_dlpack` — DLPack zero-copy

`from_dlpack` accepts any object with `__dlpack__()` and wraps it zero-copy:

```python
import numpy as np

arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
t = hurray.from_dlpack(arr)
assert t.shape == (3,)
assert t.dtype == hurray.float64
```

DLPack is an independent zero-copy interchange protocol (not the Array API); see
[Python: DLPack and NumPy Interop](hurray-python-dlpack-numpy.md). For NumPy arrays
you can also use `hurray.from_numpy`, which takes the same zero-copy path.

## Tier 2 types are not constructible here

The construction functions are Tier 1 only. Passing a Tier 2 dtype (e.g.
`hurray.dtype.int4`) raises `UnsupportedError` — there are no meaningful fill/step
semantics for sub-byte or micro-float types in these helpers:

```python
try:
    t = hurray.zeros([4], dtype=hurray.dtype.int4)
except hurray.UnsupportedError as e:
    print(f"Tier 2 dtype rejected: {e}")
```

Tier 2 / quantized tensors are produced by decoding Hurray data (`hurray.load`) or by
the interop paths, not by these constructors.

## Runnable example

```bash
# From the repo root:
cd hurray-python
maturin develop          # build the extension
python examples/construction.py
```
