"""
Streaming under asyncio: one event loop, many peers (ADR-041 § 9, ADR-042).

`hurray.StreamReader` and `hurray.StreamWriter` block the calling thread. That is
the right shape for a script and the wrong one for a server that is serving
several peers at once: one slow consumer stalls everything behind it.

`hurray.aio` is the same wire format, the same state machines, driven by Python's
own event loop instead. No Rust runtime is involved — there is none in the
extension at all.

The surface is descriptor-driven, so it wants a socket or a pipe. A path has no
readiness to await; use the synchronous classes for those.

Run with:

    python hurray-python/examples/async_streaming.py
"""

import asyncio
import socket
import struct

import numpy as np

import hurray
import hurray.aio


def tensor(seed: float, n: int = 4) -> hurray.Tensor:
    values = struct.pack(f"{n}f", *(seed + i for i in range(n)))
    return hurray.Tensor(values, hurray.float32, [n])


# ── One peer: write and read concurrently over a socket pair ──────────────────


async def produce(sock: socket.socket, count: int) -> None:
    async with hurray.aio.StreamWriter(sock) as writer:
        for i in range(count):
            await writer.write(tensor(float(i * 10)))
    # The reader stops at a clean end of stream, which is the peer hanging up.
    sock.shutdown(socket.SHUT_WR)


async def consume(sock: socket.socket, label: str) -> int:
    seen = 0
    async for item in hurray.aio.StreamReader(sock):
        seen += 1
        print(f"  {label}: tensor {seen} shape={item.shape} dtype={item.dtype}")
    return seen


async def one_peer() -> None:
    print("=== One peer ===")
    a, b = socket.socketpair()
    with a, b:
        _, seen = await asyncio.gather(produce(a, 3), consume(b, "peer"))
    print(f"  read {seen} tensors\n")


# ── Many peers: the reason this surface exists ────────────────────────────────


async def many_peers(peers: int = 4) -> None:
    """Four consumers on one loop, in one thread, making progress together.

    With the blocking surface this needs four threads, and each one is parked in a
    `read(2)` for as long as its peer is slow.
    """
    print(f"=== {peers} peers on one event loop ===")
    pairs = [socket.socketpair() for _ in range(peers)]
    try:
        work = []
        for index, (writer_sock, reader_sock) in enumerate(pairs):
            work.append(produce(writer_sock, index + 1))
            work.append(consume(reader_sock, f"peer {index}"))
        results = await asyncio.gather(*work)
        print(f"  tensors read per peer: {results[1::2]}\n")
    finally:
        for a, b in pairs:
            a.close()
            b.close()


# ── A composite crosses the async surface whole ───────────────────────────────


async def composites() -> None:
    print("=== A composite is one item ===")
    halves = [
        hurray.Tensor(
            bytes([0x40 + i]) * 128,
            hurray.float32,
            [8, 4],
            shard=hurray.Shard([8, 8], [0, i * 4]),
        )
        for i in range(2)
    ]
    whole = hurray.Composite(
        "partition", shape=[8, 8], dtype=hurray.float32, members=halves
    )

    a, b = socket.socketpair()
    with a, b:

        async def send() -> None:
            async with hurray.aio.StreamWriter(a) as writer:
                await writer.write(whole)
            a.shutdown(socket.SHUT_WR)

        async def receive() -> hurray.Composite:
            (item,) = [x async for x in hurray.aio.StreamReader(b)]
            return item

        _, received = await asyncio.gather(send(), receive())

    print(f"  received {type(received).__name__} of {received.member_count} members")
    assert received == whole
    print("  the tree that arrived is the tree that was written\n")


# ── Cancelling a read is safe; cancelling a write is not free ─────────────────


async def cancellation() -> None:
    """The guarantee, demonstrated rather than asserted (ADR-041 §§ 6, 6b).

    A cancelled read has moved no byte, because the only cancellation point is the
    wait for readiness — so the same reader picks the tensor up again. A cancelled
    *write* is different in kind: bytes already on the wire cannot be un-sent, so
    the writer refuses to pretend otherwise.
    """
    print("=== Cancellation ===")
    wire_source, wire_sink = socket.socketpair()
    with wire_source, wire_sink:
        payload = tensor(7.0, n=16)
        with hurray.StreamWriter() as encoder:
            encoder.write(payload)
        wire = encoder.getvalue()

        reader = hurray.aio.StreamReader(wire_sink)
        wire_source.sendall(wire[: len(wire) // 2])

        task = asyncio.create_task(reader.next_item())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            print("  read cancelled half-way through the tensor")

        wire_source.sendall(wire[len(wire) // 2 :])
        wire_source.shutdown(socket.SHUT_WR)
        resumed = await reader.next_item()
        print(f"  the same reader resumed and read shape={resumed.shape}")
        assert np.array_equal(np.asarray(resumed), np.asarray(payload))

    print("  a cancelled read costs nothing; a cancelled write poisons (see the docs)\n")


async def main() -> None:
    await one_peer()
    await many_peers()
    await composites()
    await cancellation()
    print("All of it on one thread, with no runtime in the extension.")


asyncio.run(main())
