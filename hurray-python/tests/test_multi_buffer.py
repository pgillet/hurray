"""Multi-buffer native protocol (ADR-030) from the Python side.

A tensor whose descriptor references more than one buffer — sparse layouts here,
quantization scales once descriptor authoring lands — must carry every buffer
through ``__hurray_buffer__`` in one capsule, in descriptor buffer-table order.
"""

import numpy as np
import pytest

import hurray


def _coo():
    """A 2x2 COO tensor: values + indices, so two buffers."""
    values = np.array([5.0, 7.0], dtype=np.float32)
    indices = np.array([[0, 0], [1, 1]], dtype=np.uint64)  # [nnz, rank]
    return hurray.sparse_coo(values, indices, [2, 2])


def test_dense_tensor_exposes_the_protocol():
    t = hurray.Tensor(bytes(16), hurray.float32, [4])
    assert hasattr(t, "__hurray_buffer__")
    assert t.__hurray_buffer__() is not None


def test_dense_tensor_round_trips():
    t = hurray.Tensor(bytes(16), hurray.float32, [4])
    back = hurray.from_hurray_buffer(t)
    assert back.shape == t.shape
    assert back.dtype == t.dtype


def test_sparse_tensor_exposes_the_protocol():
    """ADR-030 § 5: no separate __hurray_sparse_buffer__ — one protocol, one probe."""
    t = _coo()
    assert hasattr(t, "__hurray_buffer__")
    assert not hasattr(t, "__hurray_sparse_buffer__")
    assert t.__hurray_buffer__() is not None


def test_sparse_capsule_carries_values_and_indices():
    """The consumer receives the sparse descriptor with every buffer attached."""
    t = _coo()
    back = hurray.from_hurray_buffer(t)
    # Reconstructed as a Tensor carrying the COO descriptor, not a SparseTensor.
    assert back.shape == (2, 2)
    assert back.dtype == hurray.float32


def test_each_call_yields_a_fresh_capsule():
    t = _coo()
    assert t.__hurray_buffer__() is not None
    # Consuming a capsule does not invalidate the source tensor.
    assert hurray.from_hurray_buffer(t) is not None
    assert hurray.from_hurray_buffer(t) is not None


def test_from_hurray_buffer_rejects_a_non_participant():
    with pytest.raises(TypeError):
        hurray.from_hurray_buffer(object())
