"""What a buffer declares about itself, and whether the declaration is true (ADR-037).

Before this, `hurray-python` had no way to ask a tensor about its buffers at all, and the
alignment it wrote into every descriptor was a hardcoded 64 that its own allocations did
not satisfy. These tests cover the accessor and — more importantly — the property the
accessor would otherwise be reporting falsely: an address really is aligned the way the
descriptor says it is.
"""

import struct
import sys

import numpy as np
import pytest

import hurray


def _f32(n: int = 4) -> hurray.Tensor:
    return hurray.Tensor(bytes(4 * n), hurray.float32, [n])


def _address(arr: np.ndarray) -> int:
    return arr.__array_interface__["data"][0]


def _aligned_array(count: int, alignment: int = 64) -> np.ndarray:
    """A float32 array whose base address is a multiple of `alignment`.

    NumPy will not do this for you — which is the whole reason `copy` exists — so
    over-allocate and slice to the boundary.
    """
    raw = np.zeros(count + alignment // 4, dtype=np.float32)
    offset = (-_address(raw) % alignment) // 4
    arr = raw[offset : offset + count]
    assert _address(arr) % alignment == 0
    return arr


# ── The class ─────────────────────────────────────────────────────────────────


def test_a_handle_reports_its_row_of_the_buffer_table():
    handle = _f32().buffer_handles[0]
    assert handle.byte_size == 16
    assert handle.alignment >= hurray.MIN_BUFFER_ALIGNMENT
    assert handle.sync_mode == "producer_synced"
    assert not handle.is_empty


def test_the_device_is_the_tensors_own_object():
    """Device Colocation: one device per descriptor, so there is one object to hand back."""
    t = _f32()
    assert t.buffer_handles[0].device is t.device


def test_handles_compare_and_hash_by_value():
    a, b = _f32().buffer_handles[0], _f32().buffer_handles[0]
    assert a == b
    assert len({a, b}) == 1
    assert a != _f32(8).buffer_handles[0]
    assert a != "not a handle"


def test_repr_names_the_fields():
    text = repr(_f32().buffer_handles[0])
    assert text.startswith("BufferHandle(byte_size=16")
    assert "sync_mode='producer_synced'" in text


def test_a_handle_is_not_constructible_from_python():
    """Nothing a caller could supply that the buffers do not already settle."""
    with pytest.raises(TypeError):
        hurray.BufferHandle()


def test_a_handle_is_frozen():
    handle = _f32().buffer_handles[0]
    for field in ("byte_size", "alignment", "sync_mode", "device"):
        with pytest.raises(AttributeError):
            setattr(handle, field, 1)


def test_an_empty_buffer_declares_alignment_one():
    """Matching BufferHandle::empty: there is no byte to load, so nothing to align."""
    handle = hurray.Tensor(b"", hurray.float32, [0]).buffer_handles[0]
    assert handle.is_empty
    assert handle.byte_size == 0
    assert handle.alignment == 1


# ── The tuple ─────────────────────────────────────────────────────────────────


def test_buffer_handles_is_a_tuple_with_one_entry_per_buffer():
    t = _f32()
    assert isinstance(t.buffer_handles, tuple)
    assert len(t.buffer_handles) == t.buffer_count == 1


def test_a_sparse_tensor_reports_every_buffer():
    csr = hurray.Tensor(
        struct.pack("2f", 5.0, 7.0),
        hurray.float32,
        [2, 2],
        aux_buffers=[struct.pack("2Q", 0, 1), struct.pack("3Q", 0, 1, 2)],
        layout=hurray.CsrLayout(nnz=2),
    )
    handles = csr.buffer_handles
    assert len(handles) == 3
    assert [h.byte_size for h in handles] == [8, 16, 24]
    assert all(h.device is csr.device for h in handles)


def test_a_handle_pins_nothing():
    """A metadata accessor that extends buffer lifetime is a defect in a zero-copy
    format — so a handle holds no reference to its tensor and none to any buffer."""
    t = _f32()
    before = sys.getrefcount(t)
    handles = t.buffer_handles

    assert sys.getrefcount(t) == before, "the handles took a reference to the tensor"
    assert handles[0].byte_size == 16  # still answers, having copied the row out


# ── Alignment is measured, not asserted (§ 5) ─────────────────────────────────


def test_a_declared_alignment_is_one_the_address_actually_has():
    """The bug this whole pass exists for: the binding declared 64 over allocations that
    had 1, inviting a consumer's aligned SIMD load to fault."""
    for count in (1, 16, 1024, 1 << 18):
        t = hurray.from_numpy(np.zeros(count, dtype=np.float32))
        address = np.asarray(t).__array_interface__["data"][0]
        alignment = t.buffer_handles[0].alignment
        assert alignment >= hurray.MIN_BUFFER_ALIGNMENT
        assert address % alignment == 0, f"declared {alignment}, address {address}"


def test_an_owned_buffer_is_aligned_because_it_was_allocated_that_way():
    for size in (1, 63, 64, 4096):
        t = hurray.Tensor(bytes(size), hurray.uint8, [size])
        address = np.asarray(t).__array_interface__["data"][0]
        assert address % t.buffer_handles[0].alignment == 0


def test_alignment_is_exempt_from_the_round_trip_obligation():
    """§ 9: alignment describes an address, and a rebuild has a different address. A
    tensor that arrived declaring 4096 honestly declares 64 after a rebuild through
    Python bytes — forcing equality would restore the fiction § 5 removed."""
    original = hurray.from_numpy(_aligned_array(1024, alignment=4096))
    assert original.buffer_handles[0].alignment == hurray.PAGE_ALIGNMENT

    rebuilt = hurray.Tensor(np.asarray(original).tobytes(), hurray.float32, [1024])
    assert rebuilt.buffer_handles[0].alignment >= hurray.MIN_BUFFER_ALIGNMENT
    assert rebuilt.dtype == original.dtype and rebuilt.shape == original.shape


def test_the_constants_mirror_the_format():
    assert hurray.MIN_BUFFER_ALIGNMENT == 64
    assert hurray.PAGE_ALIGNMENT == 4096


# ── copy: an under-aligned source is copied (§ 6) ─────────────────────────────


def test_an_aligned_source_is_shared():
    arr = _aligned_array(1024)
    t = hurray.from_numpy(arr, copy=False)

    arr[0] = 42.0
    assert np.asarray(t)[0] == 42.0, "the buffer was copied, not shared"


def test_an_under_aligned_source_is_copied_by_default():
    arr = _aligned_array(1024)[1:]  # 4-byte aligned
    assert _address(arr) % hurray.MIN_BUFFER_ALIGNMENT != 0

    t = hurray.from_numpy(arr)
    assert t.buffer_handles[0].alignment >= hurray.MIN_BUFFER_ALIGNMENT

    arr[0] = 42.0
    assert np.asarray(t)[0] == 0.0, "the under-aligned buffer was shared"


def test_copy_false_refuses_an_under_aligned_source_and_says_why():
    arr = _aligned_array(1024)[1:]
    with pytest.raises(hurray.CopyRequiredError) as exc:
        hurray.from_numpy(arr, copy=False)

    message = str(exc.value)
    assert "4-byte aligned" in message
    assert "64" in message


def test_copy_true_always_copies():
    arr = _aligned_array(1024)
    t = hurray.from_numpy(arr, copy=True)

    arr[0] = 42.0
    assert np.asarray(t)[0] == 0.0


def test_a_large_array_is_handled_honestly_whichever_way_it_lands():
    """A large array's alignment is not something a producer can arrange: glibc puts a
    16-byte chunk header before every mmap-served block, so a fresh one never reaches 64,
    but a recycled chunk inherits whatever the heap's history gives it.

    So this asserts the invariant rather than the outcome — whatever the address is, the
    declaration matches it and `copy=False` either shares or refuses, never lies."""
    arr = np.zeros(1 << 20, dtype=np.float32)  # 4 MiB
    qualifies = _address(arr) % hurray.MIN_BUFFER_ALIGNMENT == 0

    if qualifies:
        shared = hurray.from_numpy(arr, copy=False)
        assert shared.buffer_handles[0].alignment >= hurray.MIN_BUFFER_ALIGNMENT
    else:
        with pytest.raises(hurray.CopyRequiredError):
            hurray.from_numpy(arr, copy=False)

    # Either way the default produces a tensor that declares the truth.
    tensor = hurray.from_numpy(arr)
    address = np.asarray(tensor).__array_interface__["data"][0]
    assert address % tensor.buffer_handles[0].alignment == 0


def test_an_empty_array_needs_no_copy():
    t = hurray.from_numpy(np.zeros(0, dtype=np.float32), copy=False)
    assert t.buffer_handles[0].alignment == 1


def test_copy_reaches_the_sparse_constructors():
    values = _aligned_array(2)[1:]
    indices = np.array([[1, 1]], dtype=np.uint64)
    with pytest.raises(hurray.CopyRequiredError) as exc:
        hurray.sparse_coo(values, indices, [2, 2], copy=False)
    assert "values" in str(exc.value)

    t = hurray.sparse_coo(values, indices, [2, 2])
    assert all(h.alignment >= hurray.MIN_BUFFER_ALIGNMENT for h in t.buffer_handles)


def test_copy_reaches_from_dlpack():
    arr = _aligned_array(1024)[1:]
    with pytest.raises(hurray.CopyRequiredError):
        hurray.from_dlpack(arr, copy=False)


def test_from_scipy_decides_per_component():
    scipy_sparse = pytest.importorskip("scipy.sparse")
    matrix = scipy_sparse.csr_matrix(
        np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    )
    matrix.indices = matrix.indices.astype(np.uint64)
    matrix.indptr = matrix.indptr.astype(np.uint64)

    t = hurray.from_scipy(matrix)
    assert len(t.buffer_handles) == 3
    assert all(h.alignment >= hurray.MIN_BUFFER_ALIGNMENT for h in t.buffer_handles)


# ── sync_mode (§ 7) ───────────────────────────────────────────────────────────


def test_everything_this_binding_builds_is_producer_synced():
    """Not a default but a consequence: the interpreter cannot enqueue device work
    through this API, so it cannot promise anything else."""
    tensors = [
        _f32(),
        hurray.from_numpy(np.zeros(8, dtype=np.float32)),
        hurray.sparse_coo(
            np.array([5.0], dtype=np.float32),
            np.array([[0, 0]], dtype=np.uint64),
            [2, 2],
        ),
    ]
    for t in tensors:
        assert all(h.sync_mode == "producer_synced" for h in t.buffer_handles)


def test_sync_mode_cannot_be_set_at_construction():
    """A settable field could only author a contract nothing could honour: no Python API
    supplies the device event that 'event' promises a consumer can wait on."""
    with pytest.raises(TypeError):
        hurray.Tensor(bytes(16), hurray.float32, [4], sync_mode="event")


def test_sync_mode_survives_a_stream_round_trip():
    with hurray.StreamWriter() as writer:
        writer.write(_f32())
    (back,) = list(hurray.StreamReader(writer.getvalue()))
    assert back.buffer_handles[0].sync_mode == "producer_synced"
