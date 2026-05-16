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

### Multi-buffer tensors

If a `TensorDescriptor` has multiple `BufferHandle`s (e.g. quantized weight + scale), pass one `&[u8]` per buffer:

```rust
writer.write_tensor("q_layer", &desc, &[&weight_data, &scale_data]).await?;
```

## Reading a file

`FileReader` requires a seekable source (`AsyncRead + AsyncSeek`). It reads the trailer on `open`, then seeks directly to each tensor on demand — no sequential scan.

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

### Descriptor-only reads

When you only need metadata (shape, element type) without loading the buffer bytes:

```rust
let desc = reader.read_descriptor("layer0.weight").await?;
println!("element type: {:?}", desc.element_type);
```

## KV value types

| Variant | Wire tag | Rust type |
|---------|----------|-----------|
| `KvValue::String(s)` | `0x01` | UTF-8 string |
| `KvValue::Int64(v)` | `0x02` | `i64` |
| `KvValue::Uint64(v)` | `0x03` | `u64` |
| `KvValue::Float64(v)` | `0x04` | `f64` |
| `KvValue::Bool(v)` | `0x05` | `bool` |
| `KvValue::Bytes(v)` | `0x06` | raw bytes |
| `KvValue::Array(elems)` | `0x07` | homogeneous non-empty array of the above |

Array elements must all share the same type and cannot be nested arrays.

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
