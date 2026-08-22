"""The streaming interchange format from Python (ADR-035, issue #157).

The format's defining property is that neither side buffers the whole sequence: a
reader can start before the input finishes arriving, and a writer can emit tensors one
at a time. These tests exercise that, not just the round trip — a `list(reader)` would
pass even if the implementation read everything up front.
"""

import socket
import struct

import pytest

import hurray


def _tensor(seed: float):
    return hurray.Tensor(
        struct.pack("4f", seed, seed + 1, seed + 2, seed + 3), hurray.float32, [4]
    )


def _encoded(count: int) -> bytes:
    """A stream holding `count` tensors."""
    with hurray.StreamWriter() as writer:
        for i in range(count):
            writer.write(_tensor(float(i)))
    return writer.getvalue()


# ── The acceptance criterion from the issue ───────────────────────────────────


def test_a_producer_writes_many_and_a_consumer_reads_them_back():
    tensors = list(hurray.StreamReader(_encoded(5)))
    assert len(tensors) == 5
    assert all(t.shape == (4,) for t in tensors)
    assert all(t.dtype == hurray.float32 for t in tensors)


def test_the_reader_yields_before_the_stream_is_exhausted():
    """The point of streaming: a tensor is available without reading to the end."""
    reader = hurray.StreamReader(_encoded(4))
    first = next(reader)
    assert first.shape == (4,)
    # Three remain, so the first was produced without consuming the whole input.
    assert len(list(reader)) == 3


def test_tensor_order_and_contents_survive():
    import numpy as np

    tensors = list(hurray.StreamReader(_encoded(3)))
    for index, tensor in enumerate(tensors):
        np.testing.assert_array_equal(
            np.asarray(tensor), [index, index + 1, index + 2, index + 3]
        )


def test_an_empty_stream_yields_nothing():
    with hurray.StreamWriter() as writer:
        pass
    assert list(hurray.StreamReader(writer.getvalue())) == []


# ── Transports ────────────────────────────────────────────────────────────────


def test_round_trip_through_a_path(tmp_path):
    path = str(tmp_path / "tensors.hrry")
    with hurray.StreamWriter(path) as writer:
        writer.write(_tensor(1.0))
        writer.write(_tensor(2.0))

    assert len(list(hurray.StreamReader(path))) == 2


def test_round_trip_through_a_socket():
    """The case a path-only test would miss, and the reason the format exists."""
    producer, consumer = socket.socketpair()
    try:
        with hurray.StreamWriter(producer) as writer:
            writer.write(_tensor(7.0))
            writer.write(_tensor(8.0))
        producer.shutdown(socket.SHUT_WR)

        tensors = list(hurray.StreamReader(consumer))
        assert len(tensors) == 2
    finally:
        producer.close()
        consumer.close()


def test_the_caller_keeps_its_own_descriptor():
    """The stream dups the fd, so finishing must not close the caller's socket."""
    producer, consumer = socket.socketpair()
    try:
        with hurray.StreamWriter(producer) as writer:
            writer.write(_tensor(1.0))

        # If the writer had closed the caller's fd, this would raise OSError.
        producer.send(b"still open")
        assert consumer.recv(1024).endswith(b"still open")
    finally:
        producer.close()
        consumer.close()


def test_an_object_without_a_descriptor_is_rejected_with_advice():
    import io

    with pytest.raises(TypeError) as exc:
        hurray.StreamReader(io.BytesIO(b""))
    assert "getvalue" in str(exc.value)


# ── Writer lifecycle ──────────────────────────────────────────────────────────


def test_writing_after_finish_raises_stream_error():
    writer = hurray.StreamWriter()
    writer.finish()
    with pytest.raises(hurray.StreamError):
        writer.write(_tensor(1.0))


def test_finish_is_idempotent():
    """The explicit call and the context manager must compose."""
    writer = hurray.StreamWriter()
    writer.finish()
    writer.finish()  # no error


def test_the_context_manager_finishes_the_stream():
    writer = hurray.StreamWriter()
    with writer:
        writer.write(_tensor(1.0))
    # Finished on exit: writing again is refused.
    with pytest.raises(hurray.StreamError):
        writer.write(_tensor(2.0))


def test_getvalue_on_a_directed_writer_explains_itself(tmp_path):
    path = str(tmp_path / "s.hrry")
    with hurray.StreamWriter(path) as writer:
        writer.write(_tensor(1.0))
    with pytest.raises(hurray.StreamError) as exc:
        writer.getvalue()
    assert "destination" in str(exc.value)


def test_an_exception_inside_the_block_is_not_swallowed():
    with pytest.raises(ValueError):
        with hurray.StreamWriter() as writer:
            writer.write(_tensor(1.0))
            raise ValueError("boom")


# ── Reader lifecycle and framing errors ───────────────────────────────────────


def test_the_reader_is_its_own_iterator():
    reader = hurray.StreamReader(_encoded(1))
    assert iter(reader) is reader


def test_the_reader_is_a_context_manager():
    with hurray.StreamReader(_encoded(2)) as reader:
        assert len(list(reader)) == 2


def test_a_closed_reader_is_exhausted_not_broken():
    reader = hurray.StreamReader(_encoded(2))
    reader.close()
    assert list(reader) == []


def test_a_stream_truncated_mid_frame_raises_stream_error():
    """Framing errors are what hurray.StreamError is for — it had no other use."""
    wire = _encoded(2)
    with pytest.raises(hurray.StreamError):
        list(hurray.StreamReader(wire[:-10]))


def test_a_stream_truncated_on_a_frame_boundary_is_simply_short():
    """Surprising, and a property of the format rather than of this binding.

    Frames are self-delimiting and the stream has no end marker — it ends at EOF, the
    same property that forbids end-of-file indexes. So a cut that lands exactly between
    two tensors is indistinguishable from a producer that wrote fewer of them.
    """
    one, two = _encoded(1), _encoded(2)
    assert two[: len(one)] == one, "frames are concatenated, so this cut is a boundary"
    assert len(list(hurray.StreamReader(two[: len(one)]))) == 1


def test_garbage_is_rejected_rather_than_decoded():
    with pytest.raises((hurray.StreamError, hurray.InvalidDescriptorError)):
        list(hurray.StreamReader(b"not a hurray stream at all, not even close"))


# ── Multi-buffer tensors travel whole ─────────────────────────────────────────


def test_a_multi_buffer_tensor_survives_the_stream():
    """A sparse tensor's buffers must all arrive, in descriptor order."""
    csr = hurray.Tensor(
        struct.pack("2f", 5.0, 7.0),
        hurray.float32,
        [2, 2],
        aux_buffers=[struct.pack("2Q", 0, 1), struct.pack("3Q", 0, 1, 2)],
        layout=hurray.CsrLayout(nnz=2),
    )
    with hurray.StreamWriter() as writer:
        writer.write(csr)

    (back,) = list(hurray.StreamReader(writer.getvalue()))
    assert back.layout == hurray.CsrLayout(nnz=2)
    assert back.buffer_count == 3
    assert back.nnz == 2
