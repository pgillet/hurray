# Quickstart

Build a tensor, encode it to the Hurray wire format, and read it back — in Rust
with `hurray-core`, or in Python with the `hurray` package. Use the tabs to switch
languages; your choice is remembered across the book.

<div class="lang-tabs">

```rust
use hurray_core::{
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode,
    TensorDescriptor, MIN_BUFFER_ALIGNMENT,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Describe a float32 [2, 3] row-major tensor.
    let shape = Shape::new(vec![2u64, 3])?;
    let buffer = BufferHandle::new(
        24,                     // 6 elements × 4 bytes
        MIN_BUFFER_ALIGNMENT,   // 64-byte SIMD alignment
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )?;
    let desc = TensorDescriptor::new(
        1, 0,                   // descriptor format version 1.0
        ElementType::Float32,
        shape,
        0,                      // byte_offset to element [0, 0]
        LayoutDescriptor::RowMajor,
        vec![buffer],
        None, None, None, None, // no quantization / shard / statistics / extension-type
    )?;

    // Encode to the self-delimiting wire format …
    let bytes = desc.encode()?;
    println!("descriptor: {} bytes", bytes.len());

    // … and decode it straight back.
    let decoded = TensorDescriptor::decode(&bytes)?;
    assert_eq!(decoded, desc);
    println!(
        "shape = {:?}, dtype = {:?}",
        decoded.shape.dims(),
        decoded.element_type,
    );
    Ok(())
}
```

```python
import os
import tempfile

import numpy as np
import hurray

# Build a float32 [2, 3] tensor, zero-copy from a NumPy array.
arr = np.arange(6, dtype=np.float32).reshape(2, 3)
t = hurray.from_numpy(arr)
print("shape =", t.shape, "dtype =", t.dtype, "device =", t.device)

# Hand it back to NumPy zero-copy via DLPack (dense Tier-1 tensors share the buffer).
view = np.from_dlpack(t)
assert np.array_equal(view, arr)

# Round-trip through the Hurray file format.
path = os.path.join(tempfile.gettempdir(), "quickstart.hrry")
hurray.save(path, {"x": t})
loaded = hurray.load(path)
print("loaded:", list(loaded.keys()), "→", loaded["x"].shape)
os.unlink(path)
```

</div>

## What just happened

- A **tensor descriptor** carries everything needed to interpret a buffer: element
  type, shape, byte offset, memory layout, buffer handles, and optional sections
  (quantization, shard, statistics, extension type). The four trailing `None`s in the
  Rust call are those optional sections.
- `encode` produces the **self-delimiting** binary descriptor — the first 10 bytes give
  its total length, so a reader can consume it without any external framing. `decode`
  reverses it exactly (`decoded == desc`).
- On the Python side, `from_numpy` and `np.from_dlpack` are **zero-copy**: the tensor and
  the array share one buffer. `save`/`load` use the on-disk **HRRYFILE** container (named
  tensors, footer index, mmap-friendly alignment).

## Where to next

- [Framework Interop](framework-interop.md) — zero-copy hand-off to NumPy, PyTorch, JAX,
  and CuPy.
- [Quantized Inference](quantized-inference.md) — attach and round-trip quantization
  descriptors.
- [IPC and Streaming Interchange](ipc-streaming.md) — move tensors between a producer and
  a consumer.
- The **Layer** walkthroughs ([Layer 0](layer-0-element-types-and-shape.md) onward) cover
  each part of the format in depth.
