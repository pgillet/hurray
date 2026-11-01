# Streaming Without Blocking

A blocking stream reader parks its thread until bytes arrive. That is the right shape for
a script and the wrong one for a process serving several peers at once: one slow consumer
stalls everything behind it, and a thread per peer is the workaround that stops scaling
first.

Both languages have an answer, and they are the same state machine underneath. Rust drives
it with `tokio`; Python drives it with its own event loop and no Rust runtime at all —
there is none in the extension (ADR-041 § 9).

<div class="lang-tabs">

```rust
use hurray_core::{
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode,
    TensorDescriptor, MIN_BUFFER_ALIGNMENT,
};
use hurray_io::stream::{StreamReader, StreamWriter};

fn tensor(elems: u64) -> Result<(TensorDescriptor, Vec<u8>), Box<dyn std::error::Error>> {
    let handle = BufferHandle::new(
        elems, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced,
    )?;
    let descriptor = TensorDescriptor::new(
        1, 0, ElementType::Uint8, Shape::new(vec![elems])?, 0,
        LayoutDescriptor::RowMajor, vec![handle], None, None, None, None,
    )?;
    Ok((descriptor, (0..elems as u8).collect()))
}

// Requires hurray-io's `tokio` feature (on by default).
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A duplex pair stands in for a socket: both halves are AsyncRead + AsyncWrite.
    let (producer, consumer) = tokio::io::duplex(64 * 1024);

    // Both halves run concurrently in one task — `join!` polls them together, so this
    // needs no second thread and no `spawn`.
    let writing = async {
        let mut writer = StreamWriter::new(producer);
        for elems in [8u64, 4, 16] {
            let (descriptor, data) = tensor(elems)?;
            writer.write_tensor(&descriptor, &[&data]).await?;
        }
        writer.finish().await?;
        Ok::<_, Box<dyn std::error::Error>>(())
    };

    let reading = async {
        let mut reader = StreamReader::new(consumer);
        let mut shapes = Vec::new();
        // One `next_item` is one item: a tensor, or a whole composite tree.
        while let Some(item) = reader.next_item().await? {
            shapes.push(item.descriptor().shape.dims().to_vec());
        }
        Ok::<_, Box<dyn std::error::Error>>(shapes)
    };

    let (written, shapes) = tokio::join!(writing, reading);
    written?;
    assert_eq!(shapes?, vec![vec![8u64], vec![4], vec![16]]);
    Ok(())
}
```

```python
import asyncio
import socket

import hurray
import hurray.aio


def tensor(elems):
    return hurray.Tensor(bytes(elems), hurray.uint8, [elems])


async def produce(sock):
    async with hurray.aio.StreamWriter(sock) as writer:
        for elems in (8, 4, 16):
            await writer.write(tensor(elems))
    sock.shutdown(socket.SHUT_WR)          # a clean end of stream for the peer


async def consume(sock):
    return [item.shape async for item in hurray.aio.StreamReader(sock)]


async def main():
    producer, consumer = socket.socketpair()
    with producer, consumer:
        # Both coroutines on one loop, in one thread.
        _, shapes = await asyncio.gather(produce(producer), consume(consumer))
    return shapes


assert asyncio.run(main()) == [(8,), (4,), (16,)]
```

</div>

The reader is an async iterator and the writer an async context manager. One `__anext__`
is one **item**: a `hurray.Tensor`, or a whole `hurray.Composite` when the stream carries
one.

## What `await` waits for

Every `await` in `hurray.aio` waits for the descriptor to become readable or writable. It
never awaits a transfer. Each transfer is a single synchronous call that moves the bytes
**and** reports the count to the state machine before it returns.

That is a requirement, not a preference (ADR-041 § 6a). Await a transfer instead — with
`loop.sock_recv_into`, say — and a cancellation landing between the `recv` and the
future's resolution loses the byte count. A lost count cannot be recovered by inspection,
because `0x00` is valid tensor data. The apparent way out, `asyncio.StreamReader.read`,
keeps cancelled data in its own buffer and hands back a `bytes` that must then be copied
into the reader's destination — which is the second, tensor-sized copy the format exists
to avoid.

## Cancellation

The two directions are not symmetrical, and the asymmetry is in the format rather than in
the binding.

| You cancel | What happened | What you get |
|---|---|---|
| a read, mid-tensor | no byte moved: the only cancellation point is the wait for readiness | the same reader resumes and reads the tensor through |
| a write, before its first byte | nothing was emitted, so the item never happened | the writer is intact; write the same tensor again |
| a write, mid-item | bytes are on the wire and cannot be un-sent | the writer is **poisoned** and refuses every later write |

```python
async def cancel_a_read(reader):
    task = asyncio.create_task(reader.next_item())
    await asyncio.sleep(0)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    return await reader.next_item()        # the same tensor, read through
```

A poisoned writer means the peer is holding a truncated item, so a fresh
`hurray.aio.StreamWriter` over the same connection is **not** a recovery — the peer has to
be told, or the connection replaced. `hurray.StreamError` says so when you try.

## What this surface does not do

- **Descriptors only, and Unix only.** The argument must be an object whose `fileno()` an
  event loop can poll: a socket or a pipe. A path or `bytes` has no readiness to await,
  and a regular file cannot be polled at all (`epoll` refuses one, and a file read does
  not block) — use the synchronous [`hurray.StreamReader`](hurray-python-streaming.md) for
  those. Same boundary as [issue #215](https://github.com/pgillet/hurray/issues/215).
- **No async file API.** `hurray.load` and `hurray.save` have no `hurray.aio` counterpart.
  asyncio has no asynchronous random access, so a coroutine over them would be the
  synchronous call with a thread behind it — which `asyncio.to_thread` already expresses,
  at a cost the caller can see.
- **The descriptor is put in non-blocking mode.** `O_NONBLOCK` belongs to the open file
  description, which `dup` shares, so this is visible through the object you passed in.
  Any descriptor an event loop drives has to be in that mode regardless.

## The bytes are the same bytes

`hurray.aio.StreamWriter` and `hurray.StreamWriter` produce byte-identical streams, and
either reader reads either writer's output — including through the C surface and the
frozen corpus in `hurray-io/tests/fixtures/`. The driver is what changed; the format did
not.

```python
import io

items = [tensor(8), tensor(4)]

blocking = hurray.StreamWriter()
for item in items:
    blocking.write(item)
blocking.finish()

# The same items through hurray.aio, over a pipe this time.
import os

read_fd, write_fd = os.pipe()


class Fd:
    def __init__(self, fd):
        self._fd = fd

    def fileno(self):
        return self._fd


async def write_through_aio():
    async with hurray.aio.StreamWriter(Fd(write_fd)) as writer:
        for item in items:
            await writer.write(item)
    os.close(write_fd)


async def both():
    collected = io.BytesIO()

    async def drain():
        loop = asyncio.get_running_loop()
        source = Fd(read_fd)
        os.set_blocking(read_fd, False)
        while True:
            future = loop.create_future()
            loop.add_reader(read_fd, lambda: future.done() or future.set_result(None))
            try:
                await future
            finally:
                loop.remove_reader(read_fd)
            chunk = os.read(read_fd, 1 << 16)
            if not chunk:
                return
            collected.write(chunk)

    await asyncio.gather(write_through_aio(), drain())
    os.close(read_fd)
    return collected.getvalue()


assert asyncio.run(both()) == blocking.getvalue()
```

## See also

- [Python: Streaming](hurray-python-streaming.md) — the blocking surface, paths and
  `bytes` included.
- [Streaming Composite Tensors](composite-streaming.md) — what a composite is on the wire.
- [Streaming from C](streaming-from-c.md) — the same cores over a transport vtable.
