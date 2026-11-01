"""The asyncio surface (ADR-041 § 9, ADR-042).

Two of these tests are the pass, and ADR-041 § 11 step 5 names them: a read cancelled
mid-tensor **resumes**, and a write cancelled mid-item **poisons** rather than silently
truncating. The rest hold the surface to the sync one — same bytes out, same items in,
composites included.

Coroutines run under ``asyncio.run`` inside ordinary test functions, so this file adds no
``pytest-asyncio`` dependency to CI.
"""

import asyncio
import os
import pathlib
import socket
import threading

import numpy as np
import pytest

import hurray
import hurray.aio

FIXTURES = pathlib.Path(__file__).resolve().parents[2] / "hurray-io" / "tests" / "fixtures"

pytestmark = pytest.mark.skipif(
    not hasattr(os, "set_blocking") or os.name != "posix",
    reason="hurray.aio is descriptor-driven, and so Unix-only (issue #215)",
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def tensor(n: int = 16, fill: int = 0xCD) -> hurray.Tensor:
    return hurray.Tensor(bytes([fill]) * (n * 4), hurray.float32, [n])


def big_tensor(n: int = 1 << 20) -> hurray.Tensor:
    """Large enough that no socket send buffer takes it in one write."""
    return hurray.Tensor(bytes(n * 4), hurray.float32, [n])


def composite() -> hurray.Composite:
    """A partition of an 8x8 matrix into two halves, as test_composites.py builds them."""
    halves = [
        hurray.Tensor(
            bytes([0x50 + i]) * 128,
            hurray.float32,
            [8, 4],
            shard=hurray.Shard([8, 8], [0, i * 4]),
        )
        for i in range(2)
    ]
    return hurray.Composite("partition", shape=[8, 8], dtype=hurray.float32, members=halves)


def nested() -> hurray.Composite:
    """A composite whose member is itself a composite — depth 2, the case § 11 step 1
    singles out as where suspension gets interesting."""
    return hurray.Composite(
        "group", shape=[8, 8], dtype=hurray.float32, members=[composite(), composite()]
    )


def sync_bytes(items) -> bytes:
    """The same items through the synchronous writer, which is the reference encoding."""
    writer = hurray.StreamWriter()
    for item in items:
        writer.write(item)
    writer.finish()
    return writer.getvalue()


def raw(t: hurray.Tensor, index: int = 0) -> bytes:
    return np.asarray(t.buffer(index)).view(np.uint8).tobytes()


async def write_all(sock, items, **kwargs) -> None:
    async with hurray.aio.StreamWriter(sock, **kwargs) as writer:
        for item in items:
            await writer.write(item)
    sock.shutdown(socket.SHUT_WR)


async def read_all(sock, **kwargs) -> list:
    return [item async for item in hurray.aio.StreamReader(sock, **kwargs)]


def round_trip(items, **kwargs) -> list:
    """Write `items` through `aio` and read them back through `aio`, over a socketpair."""

    async def both():
        a, b = socket.socketpair()
        with a, b:
            _, received = await asyncio.gather(
                write_all(a, items, **kwargs), read_all(b, **kwargs)
            )
            return received

    return asyncio.run(both())


def feed_pipe(data: bytes, chunk: int | None = None) -> int:
    """A read descriptor fed `data` from a thread, one `chunk` at a time.

    Returns the read end; the writer thread closes its own end, which is the clean
    end-of-stream the reader needs.
    """
    read_fd, write_fd = os.pipe()

    def dribble():
        with os.fdopen(write_fd, "wb", buffering=0) as sink:
            if chunk is None:
                sink.write(data)
            else:
                for start in range(0, len(data), chunk):
                    sink.write(data[start : start + chunk])

    threading.Thread(target=dribble, daemon=True).start()
    return read_fd


class Fd:
    """The minimal `fileno()` object: `hurray.aio` wants a descriptor, not a file."""

    def __init__(self, fd: int) -> None:
        self._fd = fd

    def fileno(self) -> int:
        return self._fd

    def close(self) -> None:
        os.close(self._fd)


# ── The surface ───────────────────────────────────────────────────────────────


def test_a_tensor_survives_the_async_round_trip():
    (received,) = round_trip([tensor()])
    assert received.shape == (16,)
    assert received.dtype == hurray.float32
    assert raw(received) == bytes([0xCD]) * 64


def test_every_item_arrives_in_order():
    items = [tensor(4, 0x01), tensor(8, 0x02), tensor(16, 0x03)]
    received = round_trip(items)
    assert [t.shape for t in received] == [(4,), (8,), (16,)]
    assert [raw(t)[0] for t in received] == [0x01, 0x02, 0x03]


def test_an_empty_stream_yields_nothing():
    async def nothing():
        a, b = socket.socketpair()
        with a, b:
            a.shutdown(socket.SHUT_WR)
            return await read_all(b)

    assert asyncio.run(nothing()) == []


def test_the_async_writer_emits_exactly_the_sync_writer_bytes():
    """The acceptance that matters for compatibility: same core, same bytes, new driver."""
    items = [tensor(4, 0x11), composite(), tensor(16, 0x22)]

    async def collect() -> bytes:
        read_fd, write_fd = os.pipe()
        sink = Fd(write_fd)
        collected = bytearray()

        def drain():
            with os.fdopen(read_fd, "rb", buffering=0) as source:
                while chunk := source.read(1 << 16):
                    collected.extend(chunk)

        reader = threading.Thread(target=drain)
        reader.start()
        async with hurray.aio.StreamWriter(sink) as writer:
            for item in items:
                await writer.write(item)
        sink.close()
        reader.join()
        return bytes(collected)

    assert asyncio.run(collect()) == sync_bytes(items)


def test_a_composite_arrives_as_one_item():
    (received,) = round_trip([composite()])
    assert isinstance(received, hurray.Composite)
    assert received.member_count == 2
    assert raw(received.members[0]) == bytes([0x50]) * 128


def test_a_nested_composite_reassembles_one_byte_at_a_time():
    """Every short read at once, with composite state suspended mid-member."""
    wire = sync_bytes([nested()])
    source = Fd(feed_pipe(wire, chunk=1))

    async def read():
        return await read_all(source)

    (outer,) = asyncio.run(read())
    assert isinstance(outer, hurray.Composite)
    assert outer.member_count == 2
    inner = outer.members[0]
    assert isinstance(inner, hurray.Composite)
    assert inner.member_count == 2
    assert raw(inner.members[1]) == bytes([0x51]) * 128
    assert outer == nested(), "the tree that arrived is the tree that was written"


@pytest.mark.parametrize(
    "name,expected_items",
    [
        ("row_major.bin", 1),
        ("coo.bin", 1),
        ("quantized.bin", 1),
        ("nested_composite.bin", 1),
        ("cross_machine.bin", 2),
    ],
)
def test_the_frozen_corpus_reads_through_aio(name, expected_items):
    """The fourth reader of the frozen corpus, after Rust, C and the sync Python one.

    Through a pipe rather than the file: an event loop cannot poll a regular file, which
    is why `hurray.aio` refuses one.
    """
    wire = (FIXTURES / name).read_bytes()
    source = Fd(feed_pipe(wire))

    async def read():
        return await read_all(source)

    assert len(asyncio.run(read())) == expected_items


def test_a_declared_alignment_survives_the_async_read():
    """ADR-041 § 5 through this driver too — the destination is the core's (#253)."""
    wire = (FIXTURES / "coo.bin").read_bytes()
    source = Fd(feed_pipe(wire))

    async def read():
        return await read_all(source)

    (received,) = asyncio.run(read())
    for index in (0, 1):
        declared = received.buffer_handles[index].alignment
        address = np.asarray(received.buffer(index)).__array_interface__["data"][0]
        assert address % declared == 0, f"buffer {index} under-aligned"


# ── Cancellation: the two tests ADR-041 § 11 step 5 names ─────────────────────


def test_a_read_cancelled_mid_tensor_resumes_rather_than_poisoning():
    """The readiness await is the only cancellation point, so no byte has moved.

    The same reader reads the tensor through on the next call — which is ADR-041 § 6's
    resumable branch being the normal path rather than a theoretical one.
    """
    payload = bytes([0xAB]) * 64
    wire = sync_bytes([hurray.Tensor(payload, hurray.float32, [16])])
    split = len(wire) // 2

    async def cancel_then_resume():
        a, b = socket.socketpair()
        with a, b:
            reader = hurray.aio.StreamReader(b)
            a.sendall(wire[:split])

            task = asyncio.create_task(reader.next_item())
            # Let the task consume what has arrived and block on readiness for the rest.
            await asyncio.sleep(0.05)
            assert not task.done(), "the tensor was complete; nothing to cancel"
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            a.sendall(wire[split:])
            a.shutdown(socket.SHUT_WR)
            return await reader.next_item()

    received = asyncio.run(cancel_then_resume())
    assert received is not None, "the reader did not resume"
    assert raw(received) == payload


def test_a_write_cancelled_mid_item_poisons_rather_than_truncating_silently():
    """A stream cannot be un-sent, so a half-emitted item is terminal (§ 6a, § 6b).

    The assertion is that the writer refuses further writes — *not* that anything
    recovers. A fresh writer over the same connection is not a recovery, which is why the
    test does not try one.
    """

    async def cancel_mid_write():
        a, b = socket.socketpair()
        with a, b:
            # A small send buffer and an unread peer: the write blocks part-way through.
            a.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
            writer = hurray.aio.StreamWriter(a)

            task = asyncio.create_task(writer.write(big_tensor()))
            await asyncio.sleep(0.05)
            assert not task.done(), "the whole tensor fit in the send buffer"
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            assert "poisoned" in repr(writer)
            with pytest.raises(hurray.StreamError) as caught:
                await writer.write(tensor())
            return str(caught.value)

    message = asyncio.run(cancel_mid_write())
    assert "poisoned" in message
    assert "not a recovery" in message, "the message must refuse the obvious retry"


def test_a_write_cancelled_before_its_first_byte_leaves_the_writer_usable():
    """§ 6b's other half: an item that emitted nothing never happened.

    The send buffer is filled *outside* the writer, so the item's very first pump blocks
    with nothing emitted. Cancelling there leaves the core exactly as it was, and the same
    tensor goes out whole afterwards — which is the branch that makes the poisoning in the
    previous test a statement about torn items rather than about cancellation.
    """

    async def cancel_then_rewrite():
        a, b = socket.socketpair()
        with a, b:
            a.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
            a.setblocking(False)
            stuffed = 0
            try:
                while stuffed < (1 << 22):
                    stuffed += a.send(bytes(1 << 16))
            except BlockingIOError:
                pass
            assert stuffed, "the send buffer never filled; the test proves nothing"

            writer = hurray.aio.StreamWriter(a)
            task = asyncio.create_task(writer.write(tensor(4, 0x7F)))
            await asyncio.sleep(0.05)
            assert not task.done(), "a byte went out despite a full send buffer"
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert "open" in repr(writer), "an item that emitted nothing must not poison"

            # Drain the stuffing so the writer can make progress, then write it again.
            b.setblocking(False)
            while True:
                try:
                    if not b.recv(1 << 16):
                        break
                except BlockingIOError:
                    break
            await writer.write(tensor(4, 0x7F))
            return True

    assert asyncio.run(cancel_then_rewrite())


# ── Refusals ──────────────────────────────────────────────────────────────────


def test_a_path_is_refused_with_the_sync_reader_named():
    with pytest.raises(TypeError, match="hurray.StreamReader"):
        hurray.aio.StreamReader(str(FIXTURES / "row_major.bin"))


def test_bytes_are_refused():
    with pytest.raises(TypeError):
        hurray.aio.StreamWriter(b"")


def test_a_regular_file_is_refused_because_a_loop_cannot_poll_one(tmp_path):
    path = tmp_path / "stream.bin"
    path.write_bytes(b"")
    with open(path, "rb") as handle:
        with pytest.raises(hurray.UnsupportedError, match="regular file"):
            hurray.aio.StreamReader(handle)


def test_a_finished_writer_refuses_further_writes():
    async def write_after_finish():
        a, b = socket.socketpair()
        with a, b:
            writer = hurray.aio.StreamWriter(a)
            await writer.finish()
            await writer.finish()  # idempotent
            with pytest.raises(hurray.StreamError, match="finished"):
                await writer.write(tensor())

    asyncio.run(write_after_finish())


def test_the_limit_keywords_reach_the_core():
    """`max_descriptor_bytes=8` is below any real descriptor, so a stream that decodes
    without limits must now be refused — which is the keyword arriving where it matters."""
    wire = sync_bytes([tensor()])
    source = Fd(feed_pipe(wire))

    async def read():
        return await read_all(source, max_descriptor_bytes=8)

    with pytest.raises(hurray.StreamError):
        asyncio.run(read())


# ── Reprs ─────────────────────────────────────────────────────────────────────


def test_the_async_reprs_say_what_the_object_holds():
    a, b = socket.socketpair()
    with a, b:
        reader = hurray.aio.StreamReader(b)
        assert repr(reader).startswith("<hurray.aio.StreamReader fd=")
        reader.close()
        assert repr(reader) == "<hurray.aio.StreamReader closed>"

        writer = hurray.aio.StreamWriter(a)
        assert "open" in repr(writer)
