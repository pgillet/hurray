# Layer 5 — Streaming Interchange

This cookbook shows how to stream tensors between producers and consumers using
`hurray-io`'s `StreamWriter` and `StreamReader`.

## Feature flag

The streaming API requires the `tokio` feature:

```toml
# Cargo.toml
hurray-io = { path = "…/hurray-io", features = ["tokio"] }
```

## Wire format

Tensors are written as bare concatenation — no outer framing, no padding:

```
[encoded TensorDescriptor][buffer 0 bytes][buffer 1 bytes]…
[encoded TensorDescriptor][buffer 0 bytes]…
…
EOF
```

The descriptor is self-delimiting: bytes 6–9 of every descriptor hold a
little-endian `uint32` `descriptor_length` that tells the reader exactly how
many bytes to consume. Readers detect a clean EOF when zero bytes are available
before the first byte of a descriptor.

## Writing a stream

<div class="lang-tabs">

```rust
use hurray_core::{
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape,
    SyncMode, TensorDescriptor, MIN_BUFFER_ALIGNMENT,
};
use hurray_io::stream::StreamWriter;

async fn write_tensors(sink: impl tokio::io::AsyncWrite + Unpin)
    -> hurray_io::Result<()>
{
    let handle = BufferHandle::new(
        192, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced,
    )?;
    let shape = Shape::new(vec![4u64, 6, 8]).unwrap();
    let desc = TensorDescriptor::new(
        1, 0,
        ElementType::Float32,
        shape,
        0,
        LayoutDescriptor::RowMajor,
        vec![handle],
        None, None, None, None,
    )?;
    let data = vec![0u8; 192];

    let mut writer = StreamWriter::new(sink);
    writer.write_tensor(&desc, &[&data]).await?;
    writer.finish().await?;          // flushes and returns the sink
    Ok(())
}
```

```python
import hurray

tensor = hurray.Tensor(bytes(192), hurray.float32, [4, 6, 8])

# A path, an object with fileno(), or nothing at all for an in-memory buffer.
with hurray.StreamWriter("tensors.hrry") as writer:
    writer.write(tensor)
# Leaving the block finishes the stream: the sink is flushed and closed.
```

</div>

The Python writer takes a `hurray.Tensor`, which already carries its descriptor and
its buffers, so the two cannot disagree — the length checks the Rust API performs have
nothing to check.

`StreamWriter::write_tensor` validates:
- `buffers.len()` equals `desc.buffers.len()` — [`Error::MultiBufferLengthMismatch`]
- each buffer's byte length equals its handle's `byte_size` — [`Error::BufferSizeMismatch`]

## Reading a stream

<div class="lang-tabs">

```rust
use hurray_io::stream::StreamReader;

async fn read_tensors(source: impl tokio::io::AsyncRead + Unpin)
    -> hurray_io::Result<()>
{
    let mut reader = StreamReader::new(source);
    while let Some(tensor) = reader.next_tensor().await? {
        println!(
            "element_type={:?} buffers={}",
            tensor.descriptor.element_type,
            tensor.buffers.len(),
        );
        // tensor.buffers[i] is a `bytes::Bytes` — refcounted, zero-copy view
    }
    Ok(())
}
```

```python
import hurray

for tensor in hurray.StreamReader("tensors.hrry"):
    print(tensor.dtype, tensor.buffer_count)

# Nothing buffers the whole input: a tensor is available as soon as its
# descriptor and buffers have arrived. A clean end of stream ends the loop;
# a truncated one raises hurray.StreamError.
```

</div>

`next_tensor()` returns:
- `Ok(Some(StreamTensor))` — a decoded tensor
- `Ok(None)` — clean EOF (stream ended on a descriptor boundary)
- `Err(Error::UnexpectedEof)` — stream truncated mid-descriptor or mid-buffer

## Cross-machine transport

When a stream crosses machine boundaries, GPU and semaphore sync primitives are
meaningless. Use `cross_machine` constructors to enforce `ProducerSynced` on
every buffer handle:

<div class="lang-tabs">

```rust
use hurray_io::stream::{StreamReader, StreamWriter};

// Writer rejects any buffer whose sync_mode != ProducerSynced.
let mut writer = StreamWriter::cross_machine(&mut wire);

// Reader rejects any decoded buffer whose sync_mode != ProducerSynced.
let mut reader = StreamReader::cross_machine(wire.as_slice());
```

```python
import hurray

# Writer refuses any buffer whose sync_mode is not "producer_synced".
writer = hurray.StreamWriter("tensors.hrry", cross_machine=True)

# Reader refuses any decoded buffer whose sync_mode is not "producer_synced".
reader = hurray.StreamReader("tensors.hrry", cross_machine=True)
```

</div>

Everything `hurray-python` constructs is `producer_synced` already (ADR-037 § 7), so
`cross_machine=True` costs a producer nothing — its value is on the reading end, and in
saying at the call site what the transport assumes.

Both constructors are equivalent to constructing with `StreamReaderOptions` /
checking manually, but they make the intent visible at the call site.

## Configuring limits

Use `StreamReaderOptions` to protect against adversarial streams:

<div class="lang-tabs">

```rust
use hurray_io::stream::{StreamReader, StreamReaderOptions};

let options = StreamReaderOptions {
    max_descriptor_bytes: 1 * 1024 * 1024, // 1 MiB
    max_buffer_bytes: 512 * 1024 * 1024,   // 512 MiB
    enforce_cross_machine_sync: true,
};
let mut reader = StreamReader::with_options(source, options);
```

```python
import hurray

reader = hurray.StreamReader(
    source,
    max_descriptor_bytes=1 << 20,     # 1 MiB
    max_buffer_bytes=512 << 20,       # 512 MiB
    cross_machine=True,
)
```

</div>

Keyword arguments rather than an options object. `max_descriptor_bytes` defaults to
16 MiB, `max_buffer_bytes` is unbounded by default because a legitimate tensor can be
enormous, and `max_composite_depth` defaults to 64. Exceeding any of them raises
`hurray.StreamError`.

These matter because a descriptor's length field is read *before* its contents: a hostile
stream can ask a reader to allocate whatever it likes, and a Python service reading from
a socket is exposed exactly as a Rust one is.

The default `max_descriptor_bytes` is 16 MiB. `max_buffer_bytes` defaults to
`u64::MAX` (unbounded). Violations produce [`Error::FrameTooLarge`].

## In-process pipe example

<div class="lang-tabs">

```rust
use tokio::io::duplex;
use hurray_io::stream::{StreamReader, StreamWriter};

let (mut client, mut server) = duplex(64 * 1024);

// Producer task
let producer = tokio::spawn(async move {
    let mut writer = StreamWriter::new(&mut client);
    // … write tensors …
    writer.finish().await
});

// Consumer task
let consumer = tokio::spawn(async move {
    let mut reader = StreamReader::new(&mut server);
    while let Some(tensor) = reader.next_tensor().await? {
        // … process tensor …
    }
    hurray_io::Result::Ok(())
});

producer.await.unwrap().unwrap();
consumer.await.unwrap().unwrap();
```

```python
import os
import threading

import hurray

read_fd, write_fd = os.pipe()

def produce():
    with os.fdopen(write_fd, "wb") as sink:
        with hurray.StreamWriter(sink) as writer:
            writer.write(hurray.Tensor(bytes(16), hurray.float32, [4]))

producer = threading.Thread(target=produce)
producer.start()

with os.fdopen(read_fd, "rb") as source:
    for tensor in hurray.StreamReader(source):
        print(tensor.shape)

producer.join()
```

</div>

The reader releases the GIL while it waits on the pipe, so the producer thread runs.

## Runnable example

<div class="lang-tabs">

```bash
cargo run --example stream_roundtrip --features tokio -p hurray-io
```

```bash
python hurray-python/examples/streaming.py
```

</div>

Sources: `hurray-io/examples/stream_roundtrip.rs` and
`hurray-python/examples/streaming.py`.

## Error reference

| Error | Cause |
|-------|-------|
| `UnexpectedEof` | Stream ended mid-descriptor or mid-buffer |
| `InvalidHeader` | Descriptor prefix is malformed (e.g. `descriptor_length < 10`) |
| `FrameTooLarge` | Descriptor or buffer exceeded configured limit |
| `MultiBufferLengthMismatch` | `buffers` slice length ≠ `desc.buffers.len()` |
| `BufferSizeMismatch` | A buffer's byte length ≠ its handle's `byte_size` |
| `InvalidCrossMachineSyncMode` | Cross-machine mode + non-`ProducerSynced` buffer |
| `Core(…)` | Descriptor encode/decode failed |
| `Io(…)` | Underlying async I/O error |
