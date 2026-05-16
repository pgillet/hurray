//! Demonstrates writing and reading a Hurray file in memory.
//!
//! Run with: `cargo run --example file_roundtrip -p hurray-io`

use std::io::Cursor;

use hurray_core::{
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode, TensorDescriptor,
    MIN_BUFFER_ALIGNMENT,
};
use hurray_io::file::{FileReader, FileWriter, FileWriterOptions, KvValue};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── Build a small model tensor ────────────────────────────────────────────

    let handle = BufferHandle::new(
        64,
        MIN_BUFFER_ALIGNMENT,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )?;
    let shape = Shape::new(vec![4u64, 4])?;
    let desc = TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        shape,
        0,
        LayoutDescriptor::RowMajor,
        vec![handle],
        None,
        None,
        None,
        None,
    )?;
    let data: Vec<u8> = (0u8..64).collect();

    // ── Write to an in-memory buffer ─────────────────────────────────────────

    let opts = FileWriterOptions {
        sorted_index: true,
        ..Default::default()
    };
    let mut writer = FileWriter::with_options(Vec::<u8>::new(), opts).await?;
    writer.write_tensor("embeddings", &desc, &[&data]).await?;

    let wire = writer
        .finish(vec![
            ("model".to_string(), KvValue::String("demo".to_string())),
            ("layers".to_string(), KvValue::Uint64(1)),
        ])
        .await?;

    println!("File size: {} bytes", wire.len());

    // ── Read it back ─────────────────────────────────────────────────────────

    let mut reader = FileReader::open(Cursor::new(wire)).await?;

    println!("Tensors: {:?}", reader.tensor_names().collect::<Vec<_>>());

    let kv = reader.kv();
    println!("KV entries: {}", kv.len());
    for (k, v) in kv {
        println!("  {k} = {v:?}");
    }

    let tensor = reader.read_tensor("embeddings").await?;
    println!(
        "embeddings: element_type={:?}, shape={:?}, {} bytes",
        tensor.descriptor.element_type,
        tensor.descriptor.shape,
        tensor.buffers[0].len()
    );
    assert_eq!(&tensor.buffers[0][..], &data[..]);
    println!("Round-trip OK.");

    Ok(())
}
