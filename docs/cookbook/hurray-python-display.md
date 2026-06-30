# Tensor Display: `__repr__` and `__str__`

`hurray.Tensor` and `hurray.SparseTensor` implement `__repr__` and `__str__`
following NumPy/PyTorch display conventions.

## `hurray.Tensor`

### `__repr__`

For **Tier 1 CPU tensors** (when NumPy is installed), `repr()` shows the data
values formatted by `numpy.array2string`, plus the dtype:

```python
import hurray

t = hurray.ones([2, 3], dtype=hurray.float32)
repr(t)
# hurray.Tensor([[1. 1. 1.]
#  [1. 1. 1.]], dtype=float32)

t2 = hurray.arange(5)
repr(t2)
# hurray.Tensor([0 1 2 3 4], dtype=int64)
```

Large tensors are truncated automatically (NumPy threshold, default 1000 elements):

```python
t = hurray.zeros([1000], dtype=hurray.float64)
repr(t)
# hurray.Tensor([0. 0. 0. ... 0. 0. 0.], dtype=float64)
```

**Fallback** (Tier 2 types, non-CPU devices, or NumPy not installed):

```python
# Tier 2 — no NumPy equivalent
t = hurray.Tensor(b'\x21', hurray.dtype.int4, [2])
repr(t)
# hurray.Tensor(shape=(2,), dtype=int4, device=cpu)
```

### `__str__`

`str()` returns the bare NumPy-style array string without the `hurray.Tensor(...)`
wrapper — suitable for `print()`:

```python
t = hurray.linspace(0.0, 1.0, 5)
print(t)
# [0.   0.25 0.5  0.75 1.  ]

t2 = hurray.full([3, 3], 7.0, dtype=hurray.float32)
print(t2)
# [[7. 7. 7.]
#  [7. 7. 7.]
#  [7. 7. 7.]]
```

Falls back to `repr()` when NumPy is unavailable or for Tier 2 types.

## `hurray.SparseTensor`

Both `repr()` and `str()` show format, shape, nnz, and dtype:

```python
import scipy.sparse as sp
import hurray

m = sp.csr_matrix(([1.0, 2.0], ([0, 1], [1, 0])), shape=(2, 2))
t = hurray.from_scipy(m)

repr(t)
# hurray.SparseTensor(format='csr', shape=(2, 2), nnz=2, dtype=float64)

print(t)
# hurray.SparseTensor(format='csr', shape=(2, 2), nnz=2, dtype=float64)
```

`str()` is identical to `repr()` for sparse tensors. By default the display is
**metadata only** (SciPy-style).

### Display options: metadata vs. content

Switch `SparseTensor` display to a **PyTorch-style content** form that also shows
the per-format buffer arrays. Use `hurray.set_print_options` to set it globally, or
`hurray.print_options(...)` as a context manager for a scoped change (auto-reverts
on exit). The default is `"metadata"`, so existing behavior is unchanged.

```python
import scipy.sparse as sp
import hurray

m = sp.csr_matrix(([1.0, 2.0, 3.0, 4.0], ([0, 0, 1, 2], [0, 2, 1, 0])), shape=(3, 3))
t = hurray.from_scipy(m)

# Default — metadata only:
repr(t)
# hurray.SparseTensor(format='csr', shape=(3, 3), nnz=4, dtype=float64)

# Global switch to content:
hurray.set_print_options(sparse_display="content")
repr(t)
# hurray.SparseTensor(format='csr', shape=(3, 3), nnz=4, dtype=float64,
#   values=[1. 2. 3. 4.], col_indices=[0 2 1 0], row_ptr=[0 2 3 4])
hurray.get_print_options()
# {'sparse_display': 'content'}

# Or scope it to a block (reverts automatically):
hurray.set_print_options(sparse_display="metadata")
with hurray.print_options(sparse_display="content"):
    print(repr(t))   # content form
print(repr(t))       # back to metadata
```

The per-format arrays shown in content mode are:

| Format | Arrays |
|--------|--------|
| COO | `indices`, `values` |
| CSR | `values`, `col_indices`, `row_ptr` |
| CSC | `values`, `row_indices`, `col_ptr` |

> **Note:** `hurray.SparseTensor` supports only the rank-2, SciPy-interop formats COO,
> CSR, and CSC. The CSF (Compressed Sparse Fiber) layout exists in `hurray-core`
> (`docs/spec/layouts/csf.md`) but is **not** exposed as a `hurray.SparseTensor`, so it
> has no Python display form. Exposing rank-N CSF in the Python bindings is future work.

Content mode formats the arrays via NumPy (honoring your active `numpy` print
options); if NumPy is not installed it falls back to the metadata string.
`set_print_options` and `print_options` are backed by a `contextvars.ContextVar`,
so the setting is isolated per asyncio task / thread context (like the strict/relaxed
mode config). An invalid `sparse_display` value raises `ValueError`.

## Runnable example

```bash
cd hurray-python
maturin develop
python examples/display.py
```
