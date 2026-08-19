"""
Sparse tensor interop with SciPy.

Demonstrates:
- Constructing a CSR tensor from SciPy without copying
- Accessing component buffer views (.values, .col_indices, .row_ptr)
- Converting back to a SciPy matrix via .to_scipy()
- The uint64 index requirement and how to satisfy it
- Why from_scipy rejects COO (and what to do instead)
"""

import numpy as np

try:
    import scipy.sparse as sp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

import hurray


def demo_csr_round_trip():
    """Zero-copy CSR round-trip: SciPy → Hurray → SciPy."""
    # Build a small 4×4 CSR matrix (identity-ish)
    dense = np.array(
        [[1.0, 0.0, 0.0, 2.0],
         [0.0, 3.0, 0.0, 0.0],
         [0.0, 0.0, 4.0, 0.0],
         [5.0, 0.0, 0.0, 6.0]],
        dtype=np.float32,
    )
    m = sp.csr_matrix(dense)

    # SciPy defaults to int32 indices. Cast to uint64 as Hurray requires.
    m.indices = m.indices.astype(np.uint64)
    m.indptr = m.indptr.astype(np.uint64)

    # Zero-copy wrap.
    sparse = hurray.from_scipy(m)
    print(f"layout : {sparse.layout}")
    print(f"shape  : {sparse.shape}")
    print(f"nnz    : {sparse.nnz}")
    print(f"dtype  : {sparse.dtype}")

    # Access component views — zero-copy, each view keeps sparse alive.
    vals = sparse.values
    col_idx = sparse.col_indices
    row_ptr = sparse.row_ptr

    print(f"values shape     : {vals.shape}")
    print(f"col_indices shape: {col_idx.shape}")
    print(f"row_ptr shape    : {row_ptr.shape}")

    # Read values via NumPy bridge.
    np_vals = np.array(vals)
    print(f"values (numpy)   : {np_vals}")

    # Convert back to SciPy.
    m2 = sparse.to_scipy()
    assert isinstance(m2, sp.csr_matrix)
    assert (m2.toarray() == dense).all(), "round-trip data mismatch"
    print("CSR round-trip OK")


def demo_csc():
    """CSC zero-copy wrap."""
    dense = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    m = sp.csc_matrix(dense)
    m.indices = m.indices.astype(np.uint64)
    m.indptr = m.indptr.astype(np.uint64)

    sparse = hurray.from_scipy(m)
    assert sparse.layout.name == "csc"
    assert sparse.nnz == 2

    row_idx = sparse.row_indices
    col_ptr = sparse.col_ptr
    print(f"CSC row_indices shape: {row_idx.shape}")
    print(f"CSC col_ptr shape    : {col_ptr.shape}")
    print("CSC wrap OK")


def demo_coo_rejection():
    """
    from_scipy rejects COO because SciPy stores row/col as two separate arrays
    while Hurray requires a single packed [nnz, rank] uint64 buffer (D19).

    Construct COO directly with hurray.sparse_coo (zero-copy from packed arrays),
    or convert to CSR/CSC first for from_scipy.
    """
    m_coo = sp.coo_matrix(np.eye(3, dtype=np.float32))
    try:
        hurray.from_scipy(m_coo)
        print("ERROR: expected UnsupportedError for COO")
    except hurray.UnsupportedError as e:
        print(f"from_scipy correctly rejects COO: {e}")

    # Preferred: repack row/col into [nnz, rank] uint64 and use hurray.sparse_coo.
    indices = np.stack([m_coo.row, m_coo.col], axis=1).astype(np.uint64)
    t = hurray.sparse_coo(m_coo.data, indices, m_coo.shape)
    assert t.layout.name == "coo"
    assert t.nnz == 3
    print(f"sparse_coo OK: layout={t.layout}, nnz={t.nnz}, shape={t.shape}")

    # Alternative: convert to CSR first, then from_scipy.
    m_csr = m_coo.tocsr()
    m_csr.indices = m_csr.indices.astype(np.uint64)
    m_csr.indptr = m_csr.indptr.astype(np.uint64)
    sparse = hurray.from_scipy(m_csr)
    assert sparse.layout.name == "csr"
    print("COO→CSR alternative OK")


if __name__ == "__main__":
    if not HAS_SCIPY:
        print("scipy not installed — install with: pip install scipy")
    else:
        demo_csr_round_trip()
        print()
        demo_csc()
        print()
        demo_coo_rejection()
