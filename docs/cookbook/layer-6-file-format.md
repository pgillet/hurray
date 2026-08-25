# Cookbook: Layer 6 — HRRYFILE Container Format

This guide shows how to write and read tensors using the `hurray-io` file format (`HRRYFILE`). The file format adds random-access lookup, optional KV metadata, and CRC-32C index integrity on top of the raw tensor stream.

## Prerequisites

```toml
[dependencies]
hurray-core = { path = "../hurray-core" }
hurray-io   = { path = "../hurray-io", features = ["tokio"] }
tokio       = { version = "1", features = ["full"] }
```

## Writing a file

`FileWriter` writes tensors in a single forward pass with no seeks. KV metadata and the footer index are flushed when you call `finish`.

<div class="lang-tabs">

```rust
use hurray_core::{
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor,
    Shape, SyncMode, TensorDescriptor, MIN_BUFFER_ALIGNMENT,
};
use hurray_io::file::{FileWriter, FileWriterOptions, KvValue};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build a descriptor for a 4×4 float32 tensor (64 bytes)
    let handle = BufferHandle::new(
        64, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced,
    )?;
    let desc = TensorDescriptor::new(
        1, 0, ElementType::Float32, Shape::new(vec![4u64, 4])?,
        0, LayoutDescriptor::RowMajor, vec![handle],
        None, None, None, None,
    )?;
    let data: Vec<u8> = (0u8..64).collect();

    // Write to a file; sorted_index enables binary search by readers
    let file = tokio::fs::File::create("model.hrry").await?;
    let opts = FileWriterOptions { sorted_index: true, ..Default::default() };
    let mut writer = FileWriter::with_options(file, opts).await?;

    writer.write_tensor("layer0.weight", &desc, &[&data]).await?;

    writer.finish(vec![
        ("model".to_string(),  KvValue::String("demo-v1".to_string())),
        ("layers".to_string(), KvValue::Uint64(1)),
    ]).await?;

    println!("Wrote model.hrry");
    Ok(())
}
```

```python
import hurray

tensor = hurray.Tensor(bytes(range(64)), hurray.float32, [4, 4])

hurray.save(
    "model.hrry",
    {"layer0.weight": tensor},
    kv={"model": "demo-v1", "layers": 1},
)
print("Wrote model.hrry")
```

</div>

One call rather than a writer object: `save` opens, writes every tensor in the dict,
flushes the KV section and the index, and closes. The forward-pass, no-seek property is
the writer's, not the caller's, so there is nothing to hold open.

### Multi-buffer tensors

If a `TensorDescriptor` has multiple `BufferHandle`s (e.g. quantized weight + scale), pass one `&[u8]` per buffer:

<div class="lang-tabs">

```rust
writer.write_tensor("q_layer", &desc, &[&weight_data, &scale_data]).await?;
```

```python
import hurray

# A tensor carries its own buffers, so a multi-buffer one saves like any other.
quantized = hurray.Tensor(
    weight_data,
    hurray.dtype.int8,
    [4, 4],
    aux_buffers=[scale_data],
    quantization=hurray.PerChannelAffine(axis=0, scale_buffer_index=1),
)
hurray.save("model.hrry", {"q_layer": quantized})
```

</div>

## Reading a file

`FileReader` requires a seekable source (`AsyncRead + AsyncSeek`). It reads the trailer on `open`, then seeks directly to each tensor on demand — no sequential scan.

<div class="lang-tabs">

```rust
use hurray_io::file::{FileReader, KvValue};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = tokio::fs::File::open("model.hrry").await?;
    let mut reader = FileReader::open(file).await?;

    // List all tensors (in index order — sorted if SORTED_INDEX was set)
    println!("tensors: {:?}", reader.tensor_names().collect::<Vec<_>>());

    // Read KV metadata
    for (key, value) in reader.kv() {
        println!("{key} = {value:?}");
    }

    // Load one tensor by name — seeks directly, skips others
    let tensor = reader.read_tensor("layer0.weight").await?;
    println!("shape: {:?}", tensor.descriptor.shape);
    println!("buffer: {} bytes", tensor.buffers[0].len());

    Ok(())
}
```

```python
import hurray

# Every tensor in the file, by name.
tensors = hurray.load("model.hrry")
print("tensors:", sorted(tensors))

# The KV metadata section.
for key, value in hurray.load_kv("model.hrry").items():
    print(f"{key} = {value!r}")

# One tensor by name — seeks directly, skips the others.
weight = hurray.load("model.hrry", names=["layer0.weight"])["layer0.weight"]
print("shape:", weight.shape)
print("buffer:", weight.buffer_handles[0].byte_size, "bytes")
```

</div>

`load_kv` is a separate call rather than an argument to `load` because it answers a
different question and returns a different thing. It costs a second open, which is a
footer read rather than a scan.

### Descriptor-only reads

When you only need metadata (shape, element type) without loading the buffer bytes:

<div class="lang-tabs">

```rust
let desc = reader.read_descriptor("layer0.weight").await?;
println!("element type: {:?}", desc.element_type);
```

```python
import hurray

# Python has no descriptor-only read: `load` returns tensors, whose metadata is
# already on the object. Naming one tensor is how you avoid paying for the rest.
weight = hurray.load("model.hrry", names=["layer0.weight"])["layer0.weight"]
print("element type:", weight.dtype.name)
```

</div>

> **Note (non-normative):** a descriptor-only read has no Python counterpart yet. Reading
> one tensor's metadata still transfers its buffers.

## KV value types

| Variant | Wire tag | Rust type | Python type |
|---------|----------|-----------|-------------|
| `KvValue::String(s)` | `0x01` | UTF-8 string | `str` |
| `KvValue::Int64(v)` | `0x02` | `i64` | `int` |
| `KvValue::Uint64(v)` | `0x03` | `u64` | `int` (read only — see below) |
| `KvValue::Float64(v)` | `0x04` | `f64` | `float` |
| `KvValue::Bool(v)` | `0x05` | `bool` | `bool` |
| `KvValue::Bytes(v)` | `0x06` | raw bytes | `bytes` |
| `KvValue::Array(elems)` | `0x07` | homogeneous non-empty array of the above | `list` |

Array elements must all share the same type and cannot be nested arrays.

In Python the mapping runs both ways — a dict passed to `save(kv=...)` comes back equal
from `load_kv` — with one asymmetry: Python's `int` writes as `int64`, so a value written
from Python never uses the `uint64` tag, while one written by a Rust producer reads back
as an ordinary `int`. `bool` is checked before `int`, since Python's `bool` is a subclass
of it.

## File layout overview

```
[ 64-byte file header  ]  magic "HRRYFILE", version, flags, alignment
[ Tensor region        ]  per tensor: descriptor → pad → buffer(s) → pad
[ KV section           ]  optional; count + (key, value) pairs
[ Index section        ]  count + (name, offsets, lengths, flags) entries
[ 40-byte trailer      ]  index_offset, index_length, kv_offset, kv_length,
                          index_crc32c, _reserved, magic "HRRY"
```

The reader locates the trailer at `file_size - 40`, reads offsets, verifies the CRC-32C of the index, then seeks to individual tensors. No full-file scan is ever needed.

## Error handling

All errors are variants of `hurray_io::Error`:

| Error | Cause |
|-------|-------|
| `InvalidFileMagic` | First 8 bytes are not `HRRYFILE` |
| `InvalidTrailerMagic` | Last 4 bytes are not `HRRY` |
| `IndexCrc32cMismatch { stored, computed }` | Index data is corrupt |
| `UnsupportedContainerVersion { major }` | Future format version |
| `TensorNotFound(name)` | No tensor with that name in the index |
| `DuplicateTensorName(name)` | Writer received the same name twice |
| `TensorNameEmpty` | Writer received an empty name string |
| `DuplicateKvKey(key)` | `finish()` received duplicate KV keys |

## Running the example

```bash
cargo run --example file_roundtrip -p hurray-io
```
