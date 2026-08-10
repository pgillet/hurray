# IPC and Streaming Interchange

Hurray's streaming format moves tensors between a **producer** and a **consumer** — in
one process, across a pipe or socket (IPC), or between machines. It is self-delimiting and
descriptor-before-data, so a reader can start work before the whole payload has arrived
and a writer can emit tensors one at a time without buffering the output.

In Rust this is the `hurray-io` streaming reader/writer over any async byte stream. In
Python today the equivalent producer→consumer hand-off is the file format (`save` in one
process, `load` in another); the incremental streaming API is Rust / C-FFI.

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
    let desc = TensorDescriptor::new(
        1, 0, ElementType::Uint8, Shape::new(vec![elems])?, 0,
        LayoutDescriptor::RowMajor, vec![handle], None, None, None, None,
    )?;
    Ok((desc, (0..elems as u8).collect()))
}

// Requires hurray-io's `tokio` feature.
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Producer — `wire` stands in for any async writer (a TcpStream, a pipe, …).
    let mut wire = Vec::<u8>::new();
    let mut writer = StreamWriter::new(&mut wire);
    let (d0, data0) = tensor(8)?;
    let (d1, data1) = tensor(4)?;
    writer.write_tensor(&d0, &[&data0]).await?; // descriptor then data buffers
    writer.write_tensor(&d1, &[&data1]).await?;
    writer.finish().await?;

    // Consumer — reads incrementally; no back-references, no seeking.
    let mut reader = StreamReader::new(wire.as_slice());
    while let Some(t) = reader.next_tensor().await? {
        let bytes: usize = t.buffers.iter().map(|b| b.len()).sum();
        println!("got {:?} tensor, {bytes} bytes", t.descriptor.element_type);
    }
    Ok(())
}
```

```python
import os
import tempfile

import numpy as np
import hurray

path = os.path.join(tempfile.gettempdir(), "handoff.hrry")

# Producer: write named tensors to the file container.
hurray.save(path, {
    "a": hurray.from_numpy(np.arange(8, dtype=np.uint8)),
    "b": hurray.from_numpy(np.arange(4, dtype=np.uint8)),
})

# Consumer (possibly another process): read them back.
loaded = hurray.load(path)
for name, t in loaded.items():
    print(name, t.shape, t.dtype)

os.unlink(path)
```

</div>

## Transports

The Rust `StreamWriter` / `StreamReader` work over anything implementing the async
read/write traits, so the same code drives:

- **In-process** hand-off (an in-memory buffer, as above).
- **IPC** over a Unix pipe or socket.
- **Cross-machine** streaming over TCP — use `StreamWriter::cross_machine` /
  `StreamReader::cross_machine`, which add the length-prefixed framing needed when the
  transport does not preserve message boundaries.

Because the format is self-delimiting and forbids back-references and end-of-file
indexes, the consumer never needs to seek — it can process each tensor as its bytes land.

## See also

- [Layer 5: Streaming Interchange](layer-5-streaming-interchange.md) — the wire framing in
  depth.
- [Layer 6: File Format](layer-6-file-format.md) and
  [Python: File I/O](hurray-python-file-io.md) — the HRRYFILE container behind `save` /
  `load`.
- [Streaming Composite Tensors](composite-streaming.md) — streaming a composite (head +
  members) as a unit.
