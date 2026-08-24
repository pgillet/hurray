"""Allocating NumPy arrays Hurray can share without copying (ADR-037 § 6a).

`from_numpy` copies most arrays, because NumPy does not promise the 64-byte alignment the
format requires and a large array served by a fresh `mmap` never has it. NEP 49 is
the sanctioned way out: install a data-memory handler for the duration of a block and the
arrays allocated inside it clear the bar.

These tests care about two things — that the arrays really are aligned, and that installing
a handler is *safe*: it restores on the way out (including through an exception), it nests,
it does not touch arrays allocated outside, and an array outlives the block that allocated
it without corrupting the heap on free.
"""

import gc

import numpy as np
import pytest

import hurray

multiarray = pytest.importorskip("numpy._core.multiarray")

ALIGN = hurray.MIN_BUFFER_ALIGNMENT

# Large enough to exercise the mmap path rather than a small-bin allocation.
#
# Note what is deliberately NOT asserted anywhere below: that an array allocated
# *outside* the block is misaligned. A fresh mmap never reaches 64, but glibc recycles
# freed chunks, and a recycled one can land anywhere — so the contrast is drawn with the
# handler NumPy records per array, which is exact, rather than with an address that is a
# matter of heap history. An earlier version asserted the address and failed on CI.
BIG = 1 << 20


def _address(arr: np.ndarray) -> int:
    return arr.__array_interface__["data"][0]


# ── Installing and restoring ──────────────────────────────────────────────────


def test_the_handler_is_installed_inside_and_restored_after():
    before = multiarray.get_handler_name()
    with hurray.aligned_allocator():
        assert multiarray.get_handler_name() == "hurray_aligned"
    assert multiarray.get_handler_name() == before


def test_the_handler_is_restored_even_when_the_block_raises():
    before = multiarray.get_handler_name()
    with pytest.raises(ZeroDivisionError):
        with hurray.aligned_allocator():
            1 / 0
    assert multiarray.get_handler_name() == before


def test_blocks_nest_and_each_restores_what_it_displaced():
    before = multiarray.get_handler_name()
    with hurray.aligned_allocator():
        with hurray.aligned_allocator():
            assert multiarray.get_handler_name() == "hurray_aligned"
        assert multiarray.get_handler_name() == "hurray_aligned"
    assert multiarray.get_handler_name() == before


def test_re_entering_the_same_object_is_refused():
    """One object holds one displaced handler; entering twice would lose the first."""
    ctx = hurray.aligned_allocator()
    with ctx:
        with pytest.raises(RuntimeError) as exc:
            with ctx:
                pass
    assert "already active" in str(exc.value)
    assert multiarray.get_handler_name() == "default_allocator"


# ── The alignment itself ──────────────────────────────────────────────────────


def test_arrays_allocated_inside_are_aligned():
    with hurray.aligned_allocator():
        for count in (1, 16, 1024, BIG):
            arr = np.zeros(count, dtype=np.float32)
            assert _address(arr) % ALIGN == 0, f"{arr.nbytes} bytes"


def test_each_array_records_the_handler_it_was_born_under():
    with hurray.aligned_allocator():
        inside = np.zeros(BIG, dtype=np.float32)
    outside = np.zeros(BIG, dtype=np.float32)

    assert multiarray.get_handler_name(inside) == "hurray_aligned"
    assert multiarray.get_handler_name(outside) == "default_allocator"


def test_the_data_is_actually_usable():
    """An allocator that returns a wrong pointer would still look aligned."""
    with hurray.aligned_allocator():
        arr = np.arange(1024, dtype=np.float64)
    assert arr[0] == 0 and arr[-1] == 1023
    assert arr.sum() == float(1023 * 1024 // 2)


def test_zeros_really_are_zero():
    """np.zeros goes through calloc, not malloc — a handler that forgot to zero would
    hand back whatever the heap last held."""
    with hurray.aligned_allocator():
        for _ in range(4):
            assert not np.zeros(4096, dtype=np.uint8).any()


def test_resize_keeps_the_alignment_and_the_contents():
    """ndarray.resize goes through realloc, which NumPy hands only the *new* size."""
    with hurray.aligned_allocator():
        arr = np.full(64, 7, dtype=np.int32)
        arr.resize(8192, refcheck=False)

    assert _address(arr) % ALIGN == 0
    assert (arr[:64] == 7).all()
    assert (arr[64:] == 0).all()


# ── Lifetime ──────────────────────────────────────────────────────────────────


def test_an_array_outlives_the_block_that_allocated_it():
    """The handler is stored per array, so the matching free runs long after exit —
    this test is really asserting that freeing does not corrupt the heap."""
    arrays = []
    with hurray.aligned_allocator():
        arrays = [np.zeros(BIG, dtype=np.float32) for _ in range(4)]

    for arr in arrays:
        arr[0] = 1.0
    del arrays
    gc.collect()

    # Anything at all after the frees: if the deallocation were mismatched, the process
    # would not get here.
    assert np.zeros(1024, dtype=np.float32).sum() == 0.0


# ── The payoff ────────────────────────────────────────────────────────────────


def test_from_numpy_shares_an_array_allocated_inside_the_block():
    with hurray.aligned_allocator():
        arr = np.zeros(BIG, dtype=np.float32)

    tensor = hurray.from_numpy(arr, copy=False)
    assert tensor.buffer_handles[0].alignment >= ALIGN

    arr[0] = 42.0
    assert np.asarray(tensor)[0] == 42.0, "the buffer was copied, not shared"


def test_an_array_the_allocator_did_not_touch_can_still_be_refused():
    """The contrast the feature exists for. Uses an offset slice rather than a fresh
    array, because a fresh one's address depends on the heap's history — under the
    allocator it does not depend on anything."""
    outside = np.zeros(BIG, dtype=np.float32)[1:]  # 4-byte aligned, guaranteed
    with pytest.raises(hurray.CopyRequiredError):
        hurray.from_numpy(outside, copy=False)

    with hurray.aligned_allocator():
        inside = np.zeros(BIG, dtype=np.float32)
    assert hurray.from_numpy(inside, copy=False).buffer_handles[0].alignment >= ALIGN


def test_the_sparse_constructors_benefit_too():
    with hurray.aligned_allocator():
        values = np.array([5.0, 7.0], dtype=np.float32)
        indices = np.array([[0, 0], [1, 1]], dtype=np.uint64)

    tensor = hurray.sparse_coo(values, indices, [2, 2], copy=False)
    assert all(h.alignment >= ALIGN for h in tensor.buffer_handles)


# ── The documented gotcha ─────────────────────────────────────────────────────


def test_a_thread_started_inside_the_block_does_not_inherit_the_handler():
    """NumPy's policy is thread-local. This is the one sharp edge, and it is better
    asserted here than discovered in a data loader."""
    import threading

    seen = {}

    def allocate():
        seen["name"] = multiarray.get_handler_name(np.zeros(BIG, dtype=np.float32))

    with hurray.aligned_allocator():
        worker = threading.Thread(target=allocate)
        worker.start()
        worker.join()

    assert seen["name"] == "default_allocator"
