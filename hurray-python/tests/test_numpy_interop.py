"""NumPy conversion actually converts.

``Tensor.__array__`` bridges to NumPy through DLPack. It used to hand
``numpy.from_dlpack`` a raw capsule, which NumPy 2.1 stopped accepting — it wants an
object implementing ``__dlpack__``. Nothing caught it, because no test converted a
tensor NumPy would accept: the existing tests only assert the *rejection* paths,
which return before the NumPy call.

So these tests convert.
"""

import struct

import numpy as np
import pytest

import hurray


def _f32():
    return hurray.Tensor(struct.pack("6f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0), hurray.float32, [2, 3])


# ── The conversion itself ─────────────────────────────────────────────────────


def test_asarray_produces_an_array_with_the_right_shape_and_dtype():
    arr = np.asarray(_f32())
    assert arr.shape == (2, 3)
    assert arr.dtype == np.float32
    np.testing.assert_array_equal(arr, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_dunder_array_is_callable_directly():
    arr = _f32().__array__()
    assert arr.shape == (2, 3)


def test_the_array_shares_the_tensors_buffer():
    """Zero-copy is the point; a silent copy would still pass a shape assertion."""
    t = _f32()
    arr = np.asarray(t)
    arr[0, 0] = 99.0
    assert np.asarray(t)[0, 0] == 99.0


def test_the_tensor_outlives_the_array_it_backs():
    """The ndarray must keep the buffer alive after the Tensor goes out of scope."""
    arr = np.asarray(_f32())
    import gc

    gc.collect()
    np.testing.assert_array_equal(arr[1], [4.0, 5.0, 6.0])


@pytest.mark.parametrize(
    ("dtype", "pack", "expected"),
    [
        (hurray.float32, ("2f", 1.0, 2.0), np.float32),
        (hurray.float64, ("2d", 1.0, 2.0), np.float64),
        (hurray.int32, ("2i", -1, 2), np.int32),
        (hurray.uint64, ("2Q", 1, 2), np.uint64),
        (hurray.int8, ("2b", -1, 2), np.int8),
    ],
)
def test_tier1_dtypes_round_trip(dtype, pack, expected):
    t = hurray.Tensor(struct.pack(*pack), dtype, [2])
    arr = np.asarray(t)
    assert arr.dtype == expected
    assert arr.shape == (2,)


# ── The dtype= / copy= contract (D4) ──────────────────────────────────────────


def test_a_dtype_cast_is_honoured():
    arr = np.asarray(_f32(), dtype=np.float64)
    assert arr.dtype == np.float64


def test_copy_false_with_a_cast_raises():
    with pytest.raises(hurray.CopyRequiredError):
        _f32().__array__(dtype=np.float64, copy=False)


def test_copy_false_without_a_cast_is_fine():
    arr = _f32().__array__(dtype=np.float32, copy=False)
    assert arr.dtype == np.float32


# ── Ingest: the same exchange in the other direction ──────────────────────────


def test_from_dlpack_accepts_a_numpy_array():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    t = hurray.from_dlpack(arr)
    assert t.shape == (3,)
    assert t.dtype == hurray.float64


def test_from_dlpack_accepts_a_hurray_tensor():
    t = hurray.from_dlpack(_f32())
    assert t.shape == (2, 3)
    assert t.dtype == hurray.float32


def test_asarray_on_a_tensor_shares_its_buffer():
    """The zero-copy fast path: no dtype change, no copy requested."""
    source = _f32()
    wrapped = hurray.asarray(source)
    np.asarray(wrapped)[0, 0] = 42.0
    assert np.asarray(source)[0, 0] == 42.0


def test_asarray_accepts_a_plain_list():
    t = hurray.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=hurray.float32)
    assert t.shape == (2, 2)
    assert t.dtype == hurray.float32


# ── Rejections still name what is wrong ───────────────────────────────────────


def test_a_sparse_layout_is_rejected_by_name():
    values = np.array([5.0, 7.0], dtype=np.float32)
    indices = np.array([[0, 0], [1, 1]], dtype=np.uint64)
    with pytest.raises(hurray.UnsupportedError) as exc:
        np.asarray(hurray.sparse_coo(values, indices, [2, 2]))
    assert "coo" in str(exc.value)


def test_bool_is_rejected_as_not_representable_in_dlpack():
    """Hurray packs bool to 1 bit; DLPack has no such type. The error must say so,
    rather than surfacing from somewhere inside NumPy."""
    with pytest.raises(BufferError) as exc:
        np.asarray(hurray.Tensor(bytes(4), hurray.bool, [4]))
    assert "bool" in str(exc.value)
    assert "DLPack" in str(exc.value)
