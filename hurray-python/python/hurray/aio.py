"""The asyncio surface: Hurray streams driven by Python's own event loop.

```python
import asyncio, hurray, hurray.aio

async def serve(sock):
    async with hurray.aio.StreamWriter(sock) as writer:
        for tensor in tensors:
            await writer.write(tensor)

async def consume(sock):
    async for item in hurray.aio.StreamReader(sock):
        ...                                     # a Tensor or a Composite
```

No Rust runtime is involved — there is none in the extension at all. The state machines
that hold the wire format live in `hurray-io` and are driven from here (ADR-041 § 9).

## Readiness, not transfers

Every ``await`` in this module waits for the *descriptor* to become readable or writable.
It never awaits a transfer. Each transfer is a single synchronous call into the extension
that moves the bytes **and** reports the count to the core before returning, so there is
no point at which bytes have moved and the core does not know it.

That shape is required rather than preferred (ADR-041 § 6a). Awaiting a transfer — with
``loop.sock_recv_into``, say — makes a cancellation that lands between the ``recv`` and
the future's resolution lose the byte count, and a lost count is unrecoverable: ``0x00``
is valid tensor data, so nothing can be inspected to find the position again. The
apparent escape, ``asyncio.StreamReader.read``, keeps cancelled data in its own buffer and
hands back a ``bytes`` that must then be copied into the core's destination — which is the
second, tensor-sized copy the whole design exists to avoid.

So: cancelling an ``await`` here cannot lose a count. A cancelled read resumes on the next
call; a cancelled write **poisons** the writer if the item had already put bytes on the
wire, because a stream cannot be un-sent (§ 6b).

## What this surface does not do

- **Descriptors only, and Unix only.** The source or destination must be an object with
  ``fileno()`` whose descriptor an event loop can poll — a socket or a pipe. A path or
  ``bytes`` has nothing to await; use the synchronous `hurray.StreamReader` and
  `hurray.StreamWriter` for those. A regular file is refused for the same reason: `epoll`
  cannot poll one, and a file read never blocks anyway.
- **No async file API.** `hurray.load` and `hurray.save` have no counterpart here. asyncio
  has no asynchronous random access, so a coroutine over them would be the synchronous
  API with a thread behind it, which callers can write themselves with
  ``asyncio.to_thread`` and see the cost of.
- **The descriptor is put in non-blocking mode.** ``O_NONBLOCK`` belongs to the open file
  description, which ``dup`` shares, so this is visible through the object you passed in.
  Any descriptor an event loop drives has to be in that mode regardless.
"""

from __future__ import annotations

import asyncio
import os
import stat

from .hurray import Composite, StreamError, Tensor, UnsupportedError

try:
    from .hurray import (
        _AIO_BLOCKED,
        _AIO_FINISHED,
        _AIO_READY,
        _AsyncStreamReader,
        _AsyncStreamWriter,
    )
except ImportError as exc:  # pragma: no cover - only reachable off Unix
    # The seam is compiled only on Unix, so this is what a Windows build raises. Said
    # plainly here rather than as a bare ImportError on a private name.
    raise UnsupportedError(
        "hurray.aio is Unix-only: it drives file descriptors through an event loop, and "
        "a Windows fileno() is a CRT descriptor for files but a raw SOCKET for sockets "
        "(issue #215). hurray.StreamReader and hurray.StreamWriter work everywhere"
    ) from exc

__all__ = ["StreamReader", "StreamWriter"]


def _pollable_fd(obj: object, what: str) -> int:
    """The descriptor behind `obj`, refused unless an event loop can poll it."""
    fileno = getattr(obj, "fileno", None)
    if fileno is None:
        raise TypeError(
            f"hurray.aio.{what} needs an object with fileno() — a socket or a pipe; "
            f"a path or bytes has no readiness to await, so use hurray.{what} for those"
        )
    fd = fileno()
    if not isinstance(fd, int) or fd < 0:
        raise OSError(f"fileno() returned {fd!r}")
    if stat.S_ISREG(os.fstat(fd).st_mode):
        raise UnsupportedError(
            f"hurray.aio.{what} cannot drive a regular file: an event loop cannot poll "
            f"one, and a file transfer never blocks — use hurray.{what} instead"
        )
    return fd


class _Readiness:
    """One descriptor's readiness, as awaitable futures.

    ``add_reader``/``add_writer`` rather than ``loop.sock_*``: the callback only resolves a
    future, and the transfer happens in the caller's synchronous step afterwards. The
    cancellation path is what the registration is for — ``remove_reader`` on the way out,
    so a cancelled task leaves no callback behind pointing at a future nobody awaits.
    """

    def __init__(self, fd: int, writable: bool) -> None:
        self._fd = fd
        self._writable = writable

    async def wait(self) -> None:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        add = loop.add_writer if self._writable else loop.add_reader
        remove = loop.remove_writer if self._writable else loop.remove_reader

        def ready() -> None:
            if not future.done():
                future.set_result(None)

        add(self._fd, ready)
        try:
            await future
        finally:
            # Runs on the cancellation path too, which is the point: the loop must not
            # keep calling `ready` for a future this coroutine has abandoned.
            remove(self._fd)


class StreamReader:
    """Reads items from a Hurray stream on a socket or pipe, one at a time.

    Iterate it with ``async for``, or call `next_item` directly. Each item is a
    `hurray.Tensor`, or a `hurray.Composite` when the stream carries one — one ``__anext__``
    is one item, whether that item is a tensor or a whole composite tree.

    The keyword arguments are `hurray.StreamReader`'s, with the same defaults and the same
    meaning: they bound what a single frame may claim, which matters when the peer is not
    one you control.

    ## Examples

    ```python
    import asyncio, hurray.aio

    async def consume(sock):
        async for item in hurray.aio.StreamReader(sock):
            print(item.shape if isinstance(item, hurray.Tensor) else item.member_count)
    ```
    """

    def __init__(
        self,
        source: object,
        *,
        max_descriptor_bytes: int | None = None,
        max_buffer_bytes: int | None = None,
        max_composite_depth: int | None = None,
        cross_machine: bool = False,
    ) -> None:
        fd = _pollable_fd(source, "StreamReader")
        # `None` travels through to the core, whose own options type holds the defaults —
        # so an unset limit is whatever the format considers sensible, not a number copied
        # into this file.
        core = _AsyncStreamReader(
            fd,
            max_descriptor_bytes=max_descriptor_bytes,
            max_buffer_bytes=max_buffer_bytes,
            max_composite_depth=max_composite_depth,
            cross_machine=cross_machine,
        )
        self._core: _AsyncStreamReader | None = core
        os.set_blocking(core.fileno(), False)
        self._readable = _Readiness(core.fileno(), writable=False)

    async def next_item(self) -> Tensor | Composite | None:
        """The next item, or ``None`` at a clean end of stream.

        Cancelling this leaves the reader usable: no byte has moved at the only
        cancellation point it has, so the next call reads the same item through.
        """
        core = self._core
        if core is None:
            return None
        while True:
            status = core.step()
            if status == _AIO_READY:
                return core.take_item()
            if status == _AIO_FINISHED:
                return None
            # BLOCKED: the core wants bytes. Wait for readiness, then transfer and
            # report in one synchronous call. `fill` is False on a spurious wakeup —
            # nothing moved, and nothing was reported, which matters: a reported count of
            # zero would mean end of input. So wait again rather than re-driving the core
            # for bytes that did not arrive.
            await self._readable.wait()
            while not core.fill():
                await self._readable.wait()

    def __aiter__(self) -> StreamReader:
        return self

    async def __anext__(self) -> Tensor | Composite:
        item = await self.next_item()
        if item is None:
            raise StopAsyncIteration
        return item

    async def __aenter__(self) -> StreamReader:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        self.close()
        return False  # never suppress an exception

    def close(self) -> None:
        """Release the descriptor now rather than at collection."""
        self._core = None

    def __repr__(self) -> str:
        state = "closed" if self._core is None else f"fd={self._core.fileno()}"
        return f"<hurray.aio.StreamReader {state}>"


class StreamWriter:
    """Writes tensors and composites to a Hurray stream on a socket or pipe.

    The bytes are identical to `hurray.StreamWriter`'s — same core, same wire format, a
    different driver.

    ## Cancellation poisons a torn item

    A cancelled `write` that had already put bytes on the wire leaves the stream truncated
    at the peer, and nothing can un-send them. The writer therefore refuses every later
    write, and a fresh writer over the same connection is **not** a recovery (ADR-041
    § 6b). A `write` cancelled before its first byte leaves the writer intact, and the same
    tensor can be written again.

    ## Examples

    ```python
    import asyncio, hurray, hurray.aio

    async def serve(sock, tensors):
        async with hurray.aio.StreamWriter(sock) as writer:
            for tensor in tensors:
                await writer.write(tensor)
    ```
    """

    def __init__(self, destination: object, *, cross_machine: bool = False) -> None:
        fd = _pollable_fd(destination, "StreamWriter")
        core = _AsyncStreamWriter(fd, cross_machine)
        self._core: _AsyncStreamWriter | None = core
        os.set_blocking(core.fileno(), False)
        self._writable = _Readiness(core.fileno(), writable=True)

    async def write(self, item: Tensor | Composite) -> None:
        """Write one tensor or one composite, with every buffer it references.

        Raises `hurray.StreamError` if the writer was poisoned by an earlier torn item, or
        if the stream has been finished.
        """
        core = self._core
        if core is None:
            raise StreamError(
                "this stream has been finished; create a new StreamWriter to write more"
            )
        if core.poisoned():
            # The core would refuse the submission on its own — it keeps the `torn` flag
            # an abandoned item set. This check exists for the message: the core knows
            # only that its own state is unusable, and the thing worth saying is about the
            # *sink*, which is truncated at the peer.
            raise StreamError(
                "this writer is poisoned: an earlier item put bytes on the wire and did "
                "not finish, so the stream is truncated at the peer. A new StreamWriter "
                "over the same connection is not a recovery — the peer must be told, or "
                "the connection replaced"
            )
        if isinstance(item, Composite):
            core.begin_composite(item)
        elif isinstance(item, Tensor):
            core.begin_tensor(item)
        else:
            raise TypeError("expected a hurray.Tensor or a hurray.Composite")

        try:
            while core.pump() == _AIO_BLOCKED:
                await self._writable.wait()
        except asyncio.CancelledError:
            # The only cancellation point is the readiness await, so no byte count was
            # lost. Whether the *item* survives is § 6b's question, and the core has
            # already answered it: dropping the operation leaves the `torn` flag set if a
            # byte went out, and clear if none did. So there is nothing to record here —
            # `poisoned()` reports it on the next write.
            core.abandon()
            raise

    async def finish(self) -> None:
        """Release the descriptor. Idempotent.

        Nothing is flushed: every byte went out through a `write(2)` on the descriptor as
        it was produced, so there is no buffer of ours left holding any.
        """
        self._core = None

    async def __aenter__(self) -> StreamWriter:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        await self.finish()
        return False  # never suppress an exception

    def __repr__(self) -> str:
        if self._core is None:
            return "<hurray.aio.StreamWriter finished>"
        state = "poisoned" if self._core.poisoned() else "open"
        return f"<hurray.aio.StreamWriter fd={self._core.fileno()} {state}>"
