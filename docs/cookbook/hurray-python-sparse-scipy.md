# Sparse Tensors and SciPy Interop

Hurray exposes COO, CSR, and CSC sparse tensors via `hurray.SparseTensor`.
For CSR and CSC, buffers are shared zero-copy with SciPy sparse matrices via
`hurray.from_scipy` and `SparseTensor.to_scipy()`.

## Constructing a CSR SparseTensor from SciPy

SciPy's `csr_matrix` stores three NumPy arrays: `.data` (values), `.indices`
(column indices), and `.indptr` (row pointers). `hurray.from_scipy` wraps all
three without copying — the resulting `SparseTensor` holds a strong reference
to the original SciPy matrix so the buffers remain valid.

**Index dtype requirement:** Hurray's wire format requires `uint64` index
arrays. SciPy defaults to `int32`. Cast before calling `from_scipy`:

```python
import numpy as np
import scipy.sparse as sp
import hurray

dense = np.array(
    [[1.0, 0.0, 2.0],
     [0.0, 3.0, 0.0],
     [4.0, 0.0, 5.0]],
    dtype=np.float32,
)
m = sp.csr_matrix(dense)

# Cast index arrays to uint64 (required by Hurray's spec).
m.indices = m.indices.astype(np.uint64)
m.indptr  = m.indptr.astype(np.uint64)

sparse = hurray.from_scipy(m)
print(sparse)
# hurray.SparseTensor(format='csr', shape=(3, 3), nnz=4, dtype=hurray.Dtype('float32'))
```

## Accessing component views

Each component buffer is accessible as a zero-copy `hurray.Tensor` view.
The view borrows the `SparseTensor`'s buffer — the parent is kept alive for
as long as any view is alive.

| Format | Attribute | Shape | dtype |
|--------|-----------|-------|-------|
| CSR | `.values` | `(nnz,)` | values dtype |
| CSR | `.col_indices` | `(nnz,)` | `uint64` |
| CSR | `.row_ptr` | `(nrows+1,)` | `uint64` |
| CSC | `.values` | `(nnz,)` | values dtype |
| CSC | `.row_indices` | `(nnz,)` | `uint64` |
| CSC | `.col_ptr` | `(ncols+1,)` | `uint64` |
| COO | `.values` | `(nnz,)` | values dtype |
| COO | `.indices` | `(nnz, rank)` | `uint64` |

Accessing a format-specific attribute on the wrong format raises `AttributeError`:

```python
sparse.indices   # AttributeError: 'SparseTensor' object has no attribute 'indices';
                 # this is a csr tensor
```

To read values into a NumPy array (zero-copy for Tier 1 types):

```python
vals_np = np.array(sparse.values)   # zero-copy via DLPack
col_idx_np = np.array(sparse.col_indices)
row_ptr_np = np.array(sparse.row_ptr)
```

## SciPy zero-copy export

`SparseTensor.to_scipy()` returns the matching `scipy.sparse` matrix type.
`copy=False` is passed to the SciPy constructor; SciPy may copy internally if
it cannot accept `uint64` index arrays (version-dependent).

```python
m2 = sparse.to_scipy()
assert isinstance(m2, sp.csr_matrix)
assert (m2.toarray() == dense).all()
```

CSC tensors return `csc_matrix`. COO tensors raise `hurray.UnsupportedError`
(see below).

## COO format caveats

`hurray.from_scipy` does **not** support COO format zero-copy. SciPy stores
COO row/col coordinates as two separate arrays, while Hurray's spec requires a
single packed `[nnz, rank]` `uint64` buffer. Passing a `coo_matrix` raises
`hurray.UnsupportedError` with instructions.

**Workaround — convert to CSR first (zero-copy from Hurray's perspective):**

```python
m_coo = sp.coo_matrix(np.eye(5, dtype=np.float32))
m_csr = m_coo.tocsr()   # SciPy makes one copy here
m_csr.indices = m_csr.indices.astype(np.uint64)
m_csr.indptr  = m_csr.indptr.astype(np.uint64)

sparse = hurray.from_scipy(m_csr)
```

**Workaround — construct directly from a packed index buffer:**

```python
row = m_coo.row.astype(np.uint64)
col = m_coo.col.astype(np.uint64)
indices = np.stack([row, col], axis=1)   # shape [nnz, 2], one copy

# (Direct COO SparseTensor construction from raw buffers is in a future pass.)
```

`SparseTensor.to_scipy()` on a COO tensor also raises `hurray.UnsupportedError`.
Access `.values` and `.indices` directly and construct `scipy.sparse.coo_matrix`
manually if needed.

## Strict mode and Tier 2 / quantized types

`to_scipy()` raises `hurray.UnsupportedError` for tensors with Tier 2 or
quantized values dtypes (`int4`, `float8` variants, etc.) because SciPy has no
equivalent dtype. The index arrays are always `uint64` and are unaffected.

## SciPy as an optional dependency

`import hurray` does not require SciPy. `from_scipy` and `to_scipy` import
`scipy.sparse` lazily at call time and raise `ImportError` if it is not
installed:

```python
try:
    sparse = hurray.from_scipy(m)
except ImportError:
    print("scipy not installed")
```

## Runnable example

```bash
# From the repo root:
cd hurray-python
maturin develop          # build the extension
python examples/sparse_scipy.py
```
