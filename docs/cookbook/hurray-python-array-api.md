# Array API Creation Functions and Namespace

`hurray.Tensor` implements the [Python Array API Standard 2025.12](https://data-apis.org/array-api/latest/)
for Tier 1 element types. This page covers the creation functions and the
`__array_namespace__` discovery mechanism.

## Array API namespace discovery

Array API consumers discover the namespace through `__array_namespace__()`.
For Tier 1 tensors this returns the `hurray` module itself:

```python
import hurray

t = hurray.zeros([3, 3])
ns = t.__array_namespace__()
assert ns is hurray        # same object

# Explicit version check
ns2 = t.__array_namespace__(api_version="2025.12")
assert ns2 is hurray
```

Tier 2 tensors (e.g. `int4`, `float8_e4m3`) raise `AttributeError` from
`__array_namespace__()` because they do not conform to the Array API. Code that
gates on Array API compliance should catch this:

```python
try:
    ns = tensor.__array_namespace__()
except AttributeError:
    ns = None   # Tier 2 or non-compliant tensor
```

## Creation functions

All creation functions default to `dtype=float64` when dtype is not specified,
matching the Array API Standard.

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

`asarray` converts Python lists, NumPy arrays, and other Array API objects to
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

This is the Array API entry point for DLPack interop. For NumPy arrays you can
also use `hurray.from_numpy`, which takes the same zero-copy path.

## Strict mode and Tier 2 types

All creation functions operate in **strict mode** only. Passing a Tier 2 dtype
(e.g. `hurray.dtype.int4`) raises `UnsupportedError`:

```python
try:
    t = hurray.zeros([4], dtype=hurray.dtype.int4)
except hurray.UnsupportedError as e:
    print(f"Tier 2 dtype rejected: {e}")
```

## Runnable example

```bash
# From the repo root:
cd hurray-python
maturin develop          # build the extension
python examples/array_api.py
```
