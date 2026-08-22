# Python: Streaming

A file is written once and read back whole. A **stream** is different: the writer emits
tensors one at a time without buffering the output, and the reader gets each tensor as
soon as its bytes arrive. That is what the streaming format is for, and ADR-035 brings
it to Python.

```python
import hurray

with hurray.StreamWriter("tensors.hrry") as writer:
    for tensor in tensors:
        writer.write(tensor)

for tensor in hurray.StreamReader("tensors.hrry"):
    print(tensor.shape, tensor.dtype)
```

The reader is an iterator and the writer is a context manager. Both are blocking, and
both release the GIL while they wait on their transport, so other Python threads keep
running.

## The property that matters

Nothing buffers the whole sequence. A tensor is available before the rest of the stream
has been read:

```python
reader = hurray.StreamReader(source)
first = next(reader)        # available as soon as its bytes arrived
...                         # do work while the rest is still in flight
rest = list(reader)
```

If you only ever want everything at once, use `hurray.load` — a file gives you random
access by name, which a stream deliberately does not.

## Transports

| You have | Pass |
|---|---|
| a path | `hurray.StreamReader("t.hrry")` |
| a socket, pipe, or open file | the object itself — anything with `fileno()` |
| bytes in hand | `hurray.StreamReader(data)` |
| nowhere to put it | `hurray.StreamWriter()` and then `getvalue()` |

```python
import socket

producer, consumer = socket.socketpair()

with hurray.StreamWriter(producer) as writer:
    writer.write(tensor)
producer.shutdown(socket.SHUT_WR)       # tell the peer there is no more

for received in hurray.StreamReader(consumer):
    ...
```

The stream **duplicates** the descriptor, so finishing it does not close your socket:

```python
with hurray.StreamWriter(sock) as writer:
    writer.write(tensor)

sock.send(b"something else")     # still yours
```

`io.BytesIO` has no descriptor, so pass its contents instead:

```python
hurray.StreamReader(buffer.getvalue())
```

## Finishing

`finish()` flushes. A writer that is never finished may leave the tail of the stream
sitting in a buffer, which is why the writer is a context manager — the `with` block
cannot forget. If you cannot use `with`, call it yourself; it is idempotent, so the two
compose:

```python
writer = hurray.StreamWriter(path)
try:
    writer.write(tensor)
finally:
    writer.finish()
```

Writing after finishing raises `hurray.StreamError`.

## Multi-buffer tensors travel whole

Every buffer a descriptor references crosses the stream, in descriptor order — sparse
index arrays, quantization scales, page tables:

```python
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
back.layout        # CsrLayout(nnz=2)
back.buffer_count  # 3
```

## What can go wrong

| Failure | Exception |
|---|---|
| a truncated or malformed frame | `hurray.StreamError` |
| a descriptor that will not decode | `hurray.InvalidDescriptorError` |
| the transport failed | `hurray.FileError` |
| the stream contains a composite | `hurray.UnsupportedError` |

Composites are **refused**, not skipped. A composite head owns no buffers and
`hurray.Tensor` cannot represent one, so the reader raises rather than hand back a
stream that decoded "successfully" having lost the composition. Read those with the
Rust API until composite support reaches Python.

> **Truncation is only half-detectable.** A stream has no end marker — frames are
> self-delimiting and it ends at EOF, which is the same property that forbids
> end-of-file indexes. A cut **mid-frame** raises `hurray.StreamError`; a cut exactly
> **on a frame boundary** is indistinguishable from a producer that wrote fewer
> tensors. If you need to know a stream was complete, say so above this layer.

## Runnable example

```
python hurray-python/examples/streaming.py
```

## See also

- [IPC and Streaming](ipc-streaming.md) — the Rust side of the same format
- [Streaming Interchange](layer-5-streaming-interchange.md) — the wire framing
- [Python: File I/O](hurray-python-file-io.md) — when random access beats incrementality
