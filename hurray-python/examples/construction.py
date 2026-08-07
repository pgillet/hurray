"""
Tensor construction functions.

Demonstrates:
- hurray.zeros, ones, full, empty and their *_like variants
- hurray.arange, linspace, eye
- hurray.asarray (from Python lists and NumPy arrays)
- hurray.from_dlpack (from NumPy arrays via DLPack)
- Tier 2 dtypes are rejected by the constructors (Tier 1 only)
"""

import numpy as np

import hurray


def demo_basic_creation():
    """zeros / ones / full / empty."""
    z = hurray.zeros([3, 4])
    assert z.shape == (3, 4)
    assert z.dtype == hurray.float64
    print(f"zeros: shape={z.shape}, dtype={z.dtype}")

    o = hurray.ones([2, 3], dtype=hurray.float32)
    assert o.shape == (2, 3)
    assert o.dtype == hurray.float32
    print(f"ones : shape={o.shape}, dtype={o.dtype}")

    f = hurray.full([4], 7.0, dtype=hurray.float32)
    assert f.shape == (4,)
    assert f.dtype == hurray.float32
    print(f"full : shape={f.shape}, dtype={f.dtype}")

    e = hurray.empty([5, 5], dtype=hurray.int32)
    assert e.shape == (5, 5)
    assert e.dtype == hurray.int32
    print(f"empty: shape={e.shape}, dtype={e.dtype}")


def demo_like_variants():
    """*_like creation functions inherit shape and dtype from a source tensor."""
    src = hurray.ones([3, 3], dtype=hurray.float32)

    z = hurray.zeros_like(src)
    assert z.shape == src.shape and z.dtype == src.dtype

    o = hurray.ones_like(src)
    assert o.shape == src.shape

    f = hurray.full_like(src, -1.0)
    assert f.shape == src.shape

    e = hurray.empty_like(src)
    assert e.shape == src.shape

    print("*_like variants OK")


def demo_ranges():
    """arange and linspace."""
    # Integer range
    t = hurray.arange(5)
    assert t.shape == (5,)
    assert t.dtype == hurray.int64
    arr = np.array(t)
    assert list(arr) == [0, 1, 2, 3, 4]
    print(f"arange(5)          : {arr}")

    # Float range
    t2 = hurray.arange(0.0, 1.0, 0.25)
    assert t2.shape == (4,)
    assert t2.dtype == hurray.float64
    arr2 = np.array(t2)
    print(f"arange(0, 1, 0.25) : {arr2}")

    # linspace
    t3 = hurray.linspace(0.0, 1.0, 5)
    assert t3.shape == (5,)
    arr3 = np.array(t3)
    assert abs(arr3[0] - 0.0) < 1e-10
    assert abs(arr3[-1] - 1.0) < 1e-10
    print(f"linspace(0, 1, 5)  : {arr3}")

    # linspace without endpoint
    t4 = hurray.linspace(0.0, 1.0, 4, endpoint=False)
    arr4 = np.array(t4)
    assert abs(arr4[-1] - 0.75) < 1e-10
    print(f"linspace no ep     : {arr4}")


def demo_eye():
    """eye: 2-D identity and off-diagonal variants."""
    t = hurray.eye(3)
    assert t.shape == (3, 3)
    arr = np.array(t)
    assert np.allclose(arr, np.eye(3))
    print(f"eye(3):\n{arr}")

    # Non-square
    t2 = hurray.eye(2, 4)
    assert t2.shape == (2, 4)
    arr2 = np.array(t2)
    print(f"eye(2,4):\n{arr2}")

    # k=1 superdiagonal
    t3 = hurray.eye(3, k=1, dtype=hurray.int32)
    arr3 = np.array(t3)
    print(f"eye(3, k=1):\n{arr3}")


def demo_asarray():
    """asarray converts Python lists, numpy arrays, and hurray tensors."""
    # From Python list
    t = hurray.asarray([1.0, 2.0, 3.0])
    assert t.shape == (3,)
    assert t.dtype == hurray.float64
    print(f"asarray(list): shape={t.shape}, dtype={t.dtype}")

    # From nested list with explicit dtype
    t2 = hurray.asarray([[1, 2], [3, 4]], dtype=hurray.int32)
    assert t2.shape == (2, 2)
    assert t2.dtype == hurray.int32
    print(f"asarray(nested, int32): shape={t2.shape}")

    # From numpy array (zero-copy)
    np_arr = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    t3 = hurray.asarray(np_arr)
    assert t3.shape == (3,)
    assert t3.dtype == hurray.float32
    print(f"asarray(numpy): shape={t3.shape}, dtype={t3.dtype}")

    # From hurray tensor (zero-copy via DLPack)
    src = hurray.zeros([2, 2])
    t4 = hurray.asarray(src)
    assert t4.shape == (2, 2)
    print("asarray(hurray.Tensor): OK")


def demo_from_dlpack():
    """from_dlpack wraps any object with __dlpack__ zero-copy."""
    np_arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    t = hurray.from_dlpack(np_arr)
    assert t.shape == (3,)
    assert t.dtype == hurray.float64
    arr = np.array(t)
    assert np.allclose(arr, np_arr)
    print(f"from_dlpack(numpy): shape={t.shape}")


def demo_tier2_rejected():
    """Construction functions are Tier 1 only; Tier 2 dtypes raise UnsupportedError."""
    try:
        hurray.zeros([4], dtype=hurray.dtype.int4)
        print("ERROR: expected UnsupportedError")
    except hurray.UnsupportedError as e:
        print(f"Tier 2 dtype correctly rejected: {e}")


if __name__ == "__main__":
    print("=== Basic creation ===")
    demo_basic_creation()
    print()
    print("=== *_like variants ===")
    demo_like_variants()
    print()
    print("=== Ranges ===")
    demo_ranges()
    print()
    print("=== eye ===")
    demo_eye()
    print()
    print("=== asarray ===")
    demo_asarray()
    print()
    print("=== from_dlpack ===")
    demo_from_dlpack()
    print()
    print("=== Tier 2 rejected ===")
    demo_tier2_rejected()
