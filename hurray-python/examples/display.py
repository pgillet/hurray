"""
Tensor display: __repr__ and __str__.

Demonstrates:
- hurray.Tensor repr with data values (Tier 1 CPU tensors)
- hurray.Tensor repr fallback for Tier 2 types
- hurray.Tensor __str__ (bare NumPy-style array string)
- hurray.SparseTensor repr and str (metadata by default)
- switching SparseTensor display to PyTorch-style content via print options
"""

import scipy.sparse as sp

import hurray


def demo_tensor_repr():
    """__repr__ shows data for small Tier 1 CPU tensors."""
    t = hurray.ones([2, 3], dtype=hurray.float32)
    print("repr(ones [2,3] float32):")
    print(repr(t))
    print()

    t2 = hurray.arange(5)
    print("repr(arange(5) int64):")
    print(repr(t2))
    print()

    t3 = hurray.eye(4, dtype=hurray.float64)
    print("repr(eye(4)):")
    print(repr(t3))
    print()


def demo_tensor_str():
    """__str__ returns a bare NumPy-style array string (no hurray.Tensor wrapper)."""
    t = hurray.linspace(0.0, 1.0, 5)
    print("str(linspace(0, 1, 5)):")
    print(str(t))
    print()

    t2 = hurray.full([3, 3], 7.0, dtype=hurray.float32)
    print("str(full [3,3] 7.0):")
    print(str(t2))
    print()


def demo_large_tensor():
    """Large tensors are truncated by numpy.array2string."""
    t = hurray.zeros([1000], dtype=hurray.float64)
    print("repr(zeros [1000]) — truncated:")
    print(repr(t))
    print()


def demo_sparse_repr():
    """SparseTensor __repr__ / __str__ show format, shape, nnz, dtype."""
    m = sp.csr_matrix(
        ([1.0, 2.0, 3.0, 4.0], ([0, 0, 1, 2], [0, 2, 1, 0])), shape=(3, 3)
    )
    t = hurray.from_scipy(m)
    print("repr(SparseTensor csr):")
    print(repr(t))
    print()
    print("str(SparseTensor csr)  [same as repr]:")
    print(str(t))
    print()


def demo_sparse_print_options():
    """Switch SparseTensor display between metadata (default) and PyTorch-style content."""
    m = sp.csr_matrix(
        ([1.0, 2.0, 3.0, 4.0], ([0, 0, 1, 2], [0, 2, 1, 0])), shape=(3, 3)
    )
    t = hurray.from_scipy(m)

    # Default: metadata only (SciPy-style).
    print("default (metadata):")
    print(repr(t))
    print()

    # Opt in to content display globally.
    hurray.set_print_options(sparse_display="content")
    print('after set_print_options(sparse_display="content"):')
    print(repr(t))
    print("get_print_options():", hurray.get_print_options())
    print()
    hurray.set_print_options(sparse_display="metadata")  # restore

    # Scoped to a block via the context manager (auto-reverts on exit).
    with hurray.print_options(sparse_display="content"):
        print("inside print_options(sparse_display='content'):")
        print(repr(t))
    print("after the with-block (reverted to metadata):")
    print(repr(t))
    print()


if __name__ == "__main__":
    print("=== Tensor __repr__ ===")
    demo_tensor_repr()
    print("=== Tensor __str__ ===")
    demo_tensor_str()
    print("=== Large tensor ===")
    demo_large_tensor()
    print("=== SparseTensor ===")
    demo_sparse_repr()
    print("=== SparseTensor print options ===")
    demo_sparse_print_options()
