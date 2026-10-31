"""The frozen wire corpus, read through the Python surface.

``hurray-io/tests/fixtures/`` holds streams and files emitted by writers that predate
the sans-io cores, with a README forbidding their regeneration. Rust reads them in
``wire_compat.rs`` and ``file_wire_compat.rs``; C reads them in ``stream_ffi.rs`` and
``file_ffi.rs``. This is the third reader, and the point is the same one: the Python
binding drives those cores itself now (ADR-041 § 9), so the bytes it accepts must still
be the bytes 0.1.0 wrote.

Buffer addresses are checked against their *declared* alignment, not against 64 — the
guarantee ADR-041 § 5 makes and #235 fixed. The binding adopts the core's destination
rather than copying out of it (#253), so those addresses are the ones the core chose and
the declaration stays true of the bytes it describes.
"""

import pathlib

import numpy as np
import pytest

import hurray

FIXTURES = pathlib.Path(__file__).resolve().parents[2] / "hurray-io" / "tests" / "fixtures"


def payload(length: int, salt: int) -> bytes:
    """The corpus generator's deterministic filler, as ``wire_compat.rs`` spells it."""
    return bytes((i % 251) ^ salt for i in range(length))


def raw(tensor: hurray.Tensor, index: int = 0) -> bytes:
    """Every byte of one buffer, whatever its element type presents as."""
    return np.asarray(tensor.buffer(index)).view(np.uint8).tobytes()


def address(tensor: hurray.Tensor, index: int = 0) -> int:
    return np.asarray(tensor.buffer(index)).__array_interface__["data"][0]


# ── Streams ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,expected_items",
    [
        ("row_major.bin", 1),
        ("coo.bin", 1),
        ("quantized.bin", 1),
        # One top-level composite, whatever its nesting.
        ("nested_composite.bin", 1),
        ("cross_machine.bin", 2),
    ],
)
def test_every_frozen_stream_decodes(name, expected_items):
    assert len(list(hurray.StreamReader(str(FIXTURES / name)))) == expected_items


def test_frozen_stream_read_from_bytes_matches_read_from_path():
    """The two source shapes are different code paths to the same driver."""
    data = (FIXTURES / "row_major.bin").read_bytes()
    from_bytes = list(hurray.StreamReader(data))
    from_path = list(hurray.StreamReader(str(FIXTURES / "row_major.bin")))
    assert len(from_bytes) == len(from_path) == 1
    assert raw(from_bytes[0]) == raw(from_path[0])


def test_frozen_row_major_stream_round_trips_its_payload():
    (tensor,) = list(hurray.StreamReader(str(FIXTURES / "row_major.bin")))
    assert tensor.shape == (8, 8)
    assert tensor.dtype == hurray.float32
    assert raw(tensor) == payload(256, 0x00)


def test_frozen_coo_stream_round_trips_both_payloads():
    """Two buffers, so the multi-buffer read path (ADR-030 § 3) in descriptor order."""
    (tensor,) = list(hurray.StreamReader(str(FIXTURES / "coo.bin")))
    assert tensor.buffer_count == 2
    assert raw(tensor, 0) == payload(32, 0x11)
    assert raw(tensor, 1) == payload(128, 0x22)


def test_frozen_coo_stream_keeps_its_declared_alignments():
    """Two buffers declaring 128 and 4096, which the destination must honour (§ 5)."""
    (tensor,) = list(hurray.StreamReader(str(FIXTURES / "coo.bin")))
    for index in (0, 1):
        declared = tensor.buffer_handles[index].alignment
        assert address(tensor, index) % declared == 0, f"buffer {index} under-aligned"


def test_frozen_nested_composite_stream_reassembles():
    """One item, two members, the first of them a composite of two — depth 2."""
    (outer,) = list(hurray.StreamReader(str(FIXTURES / "nested_composite.bin")))
    assert isinstance(outer, hurray.Composite)
    assert outer.member_count == 2

    inner = outer.members[0]
    assert isinstance(inner, hurray.Composite)
    assert inner.member_count == 2
    assert raw(inner.members[0]) == payload(128, 0x55)
    assert raw(inner.members[1]) == payload(128, 0x66)
    assert raw(outer.members[1]) == payload(16, 0x77)


def test_frozen_stream_survives_a_one_byte_at_a_time_feed():
    """A pipe that delivers one byte per read is every short read at once.

    The driver reports each count and resumes, so the item is identical to the one a
    single bulk read produces (ADR-041 § 4).
    """
    import os
    import threading

    data = (FIXTURES / "coo.bin").read_bytes()
    read_fd, write_fd = os.pipe()

    def dribble():
        with os.fdopen(write_fd, "wb", buffering=0) as sink:
            for byte in data:
                sink.write(bytes([byte]))

    feeder = threading.Thread(target=dribble)
    feeder.start()
    try:
        with os.fdopen(read_fd, "rb", buffering=0) as source:
            (tensor,) = list(hurray.StreamReader(source))
    finally:
        feeder.join()

    assert raw(tensor, 0) == payload(32, 0x11)
    assert raw(tensor, 1) == payload(128, 0x22)


# ── Files ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,expected_entries",
    [
        ("row_major.hrry", 1),
        ("multi_buffer.hrry", 1),
        ("kv_sorted.hrry", 2),
        # Five index entries, one composite: the members do not also come back alone.
        ("nested_composite.hrry", 1),
        ("nested_composite_padded_head.hrry", 1),
    ],
)
def test_every_frozen_file_loads(name, expected_entries):
    assert len(hurray.load(str(FIXTURES / "file" / name))) == expected_entries


def test_frozen_file_multi_buffer_carries_both_buffers():
    """The data region has inter-buffer padding, which the core must skip, not deliver."""
    (tensor,) = hurray.load(str(FIXTURES / "file" / "multi_buffer.hrry")).values()
    assert tensor.buffer_count == 2
    sizes = [handle.byte_size for handle in tensor.buffer_handles]
    assert [len(raw(tensor, i)) for i in (0, 1)] == sizes


def test_frozen_file_multi_buffer_keeps_its_declared_alignments():
    (tensor,) = hurray.load(str(FIXTURES / "file" / "multi_buffer.hrry")).values()
    for index in (0, 1):
        declared = tensor.buffer_handles[index].alignment
        assert address(tensor, index) % declared == 0, f"buffer {index} under-aligned"


def test_frozen_file_kv_section_reads_every_tag():
    """``kv_sorted.hrry`` covers all seven wire tags, the array among them."""
    kv = hurray.load_kv(str(FIXTURES / "file" / "kv_sorted.hrry"))
    assert kv, "the file declares a KV section"
    kinds = {type(value) for value in kv.values()}
    assert str in kinds and list in kinds


def test_frozen_file_composite_reassembles():
    """The pre-#243 and post-#243 files hold the same five tensors."""
    fixed = hurray.load(str(FIXTURES / "file" / "nested_composite.hrry"))
    padded = hurray.load(str(FIXTURES / "file" / "nested_composite_padded_head.hrry"))
    assert set(fixed) == set(padded)

    for loaded in (fixed, padded):
        (outer,) = loaded.values()
        assert isinstance(outer, hurray.Composite)
        assert outer.member_count == 2
        assert isinstance(outer.members[0], hurray.Composite)
        assert outer.members[0].member_count == 2


def test_frozen_file_named_subset_matches_the_whole_load():
    """``names=`` takes a different path through the core's index lookups."""
    path = str(FIXTURES / "file" / "kv_sorted.hrry")
    everything = hurray.load(path)
    name = sorted(everything)[0]
    subset = hurray.load(path, names=[name])
    assert list(subset) == [name]
    assert raw(subset[name]) == raw(everything[name])
