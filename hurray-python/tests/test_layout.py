"""One tensor class for every layout (ADR-031).

Sparse is a layout, not a different kind of object. These tests pin the three
consequences: `layout` reports it, inapplicable accessors are genuinely absent so
`hasattr` tells the truth, and dense-only protocols reject non-dense layouts by
naming the layout rather than by the type not having the method.
"""

import numpy as np
import pytest

import hurray


def _coo():
    values = np.array([5.0, 7.0], dtype=np.float32)
    indices = np.array([[0, 0], [1, 1]], dtype=np.uint64)
    return hurray.sparse_coo(values, indices, [2, 2])


def _csr():
    """A CSR tensor. Requires scipy — hurray has no from-scratch CSR constructor."""
    scipy_sparse = pytest.importorskip("scipy.sparse")

    m = scipy_sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32))
    m.indices = m.indices.astype(np.uint64)
    m.indptr = m.indptr.astype(np.uint64)
    return hurray.from_scipy(m)


# ── One class ─────────────────────────────────────────────────────────────────


def test_there_is_no_separate_sparse_class():
    assert not hasattr(hurray, "SparseTensor")


def test_a_sparse_tensor_is_a_tensor():
    assert isinstance(_coo(), hurray.Tensor)


def test_layout_reports_the_descriptor_layout():
    assert hurray.Tensor(bytes(16), hurray.float32, [4]).layout == "row_major"
    assert _coo().layout == "coo"
    assert _csr().layout == "csr"


# ── Inapplicable accessors are absent, not merely failing ─────────────────────


def test_hasattr_tells_the_truth_for_a_dense_tensor():
    dense = hurray.Tensor(bytes(16), hurray.float32, [4])
    for attr in ("nnz", "values", "indices", "row_ptr", "col_indices", "col_ptr"):
        assert not hasattr(dense, attr), attr


def test_hasattr_tells_the_truth_across_sparse_layouts():
    coo, csr = _coo(), _csr()

    assert hasattr(coo, "indices")
    assert not hasattr(coo, "row_ptr")

    assert hasattr(csr, "row_ptr")
    assert hasattr(csr, "col_indices")
    assert not hasattr(csr, "indices")


def test_inapplicable_accessor_raises_attribute_error():
    """AttributeError, not UnsupportedError — that is what makes hasattr work."""
    with pytest.raises(AttributeError):
        _coo().row_ptr
    with pytest.raises(AttributeError):
        hurray.Tensor(bytes(16), hurray.float32, [4]).nnz


def test_applicable_accessors_return_views():
    coo = _coo()
    assert coo.nnz == 2
    assert coo.values.shape == (2,)
    assert coo.indices.shape == (2, 2)

    csr = _csr()
    assert csr.values.shape == (csr.nnz,)
    assert csr.col_indices.shape == (csr.nnz,)
    assert csr.row_ptr.shape == (csr.shape[0] + 1,)


# ── Dense-only protocols reject by layout ─────────────────────────────────────


def test_dlpack_rejects_a_sparse_layout():
    with pytest.raises(BufferError) as exc:
        _coo().__dlpack__()
    assert "coo" in str(exc.value)


def test_array_rejects_a_sparse_layout():
    with pytest.raises(hurray.UnsupportedError) as exc:
        np.asarray(_coo())
    assert "coo" in str(exc.value)


def test_native_protocol_accepts_every_layout():
    """__hurray_buffer__ is the full-fidelity path, so it must not discriminate."""
    for t in (hurray.Tensor(bytes(16), hurray.float32, [4]), _coo()):
        assert t.__hurray_buffer__() is not None


# ── File I/O covers every layout (#156) ───────────────────────────────────────


def test_sparse_tensor_round_trips_through_a_file(tmp_path):
    path = tmp_path / "m.hrry"
    original = _coo()
    hurray.save(str(path), {"m": original})

    back = hurray.load(str(path))["m"]
    assert back.layout == "coo"
    assert back.nnz == original.nnz
    assert back.buffer_count == 2


def test_csr_tensor_round_trips_through_a_file(tmp_path):
    path = tmp_path / "csr.hrry"
    original = _csr()
    hurray.save(str(path), {"m": original})

    back = hurray.load(str(path))["m"]
    assert back.layout == "csr"
    assert back.nnz == original.nnz
    assert back.buffer_count == 3


def test_a_file_can_mix_dense_and_sparse(tmp_path):
    path = tmp_path / "mixed.hrry"
    hurray.save(
        str(path),
        {"dense": hurray.Tensor(bytes(16), hurray.float32, [4]), "sparse": _coo()},
    )

    loaded = hurray.load(str(path))
    assert loaded["dense"].layout == "row_major"
    assert loaded["sparse"].layout == "coo"


# ── Repr ──────────────────────────────────────────────────────────────────────


def test_sparse_repr_names_the_layout():
    r = repr(_coo())
    assert r.startswith("hurray.Tensor(")
    assert "layout='coo'" in r
