//! Writes a composite tensor to a Hurray file, then reads it back reassembled.
//!
//! Builds a partition composite — a `[8, 8]` float32 logical tensor split into two `[8, 4]`
//! tiles — writes it with [`FileWriter::write_composite`] (head + members, each with its
//! own footer-index entry), then recovers the whole group with
//! [`FileReader::read_composite`].
//!
//! Run with:
//! ```text
//! cargo run --example composite_file --features tokio -p hurray-io
//! ```

use std::io::Cursor;

use hurray_core::{
    buffer_size_bytes,
    layout::{CompositeLayout, CompositionRule},
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, ShardDescriptor, SyncMode,
    TensorDescriptor, MIN_BUFFER_ALIGNMENT,
};
use hurray_io::file::{FileCompositeNode, FileItem, FileReader, FileWriter};

fn buf(byte_size: u64) -> Result<BufferHandle, Box<dyn std::error::Error>> {
    Ok(BufferHandle::new(
        byte_size,
        MIN_BUFFER_ALIGNMENT,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )?)
}

fn tile(offset: u64) -> Result<TensorDescriptor, Box<dyn std::error::Error>> {
    Ok(TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(vec![8u64, 4])?,
        0,
        LayoutDescriptor::RowMajor,
        vec![buf(buffer_size_bytes(ElementType::Float32, 8 * 4))?],
        None,
        Some(ShardDescriptor::new(vec![8, 8], vec![0, offset])?),
        None,
        None,
    )?)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let head = TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(vec![8u64, 8])?,
        0,
        LayoutDescriptor::Composite(CompositeLayout::new(CompositionRule::Partition, 2)?),
        vec![],
        None,
        None,
        None,
        None,
    )?;

    let (left, right) = (tile(0)?, tile(4)?);
    let left_data = vec![0xA5u8; 128];
    let right_data = vec![0x5Au8; 128];
    let left_buffers: [&[u8]; 1] = [left_data.as_slice()];
    let right_buffers: [&[u8]; 1] = [right_data.as_slice()];

    let members = vec![
        FileCompositeNode::Tensor {
            name: "weight.left",
            descriptor: &left,
            buffers: &left_buffers,
        },
        FileCompositeNode::Tensor {
            name: "weight.right",
            descriptor: &right,
            buffers: &right_buffers,
        },
    ];

    // Write to an in-memory buffer (a real program would pass a tokio::fs::File).
    let out = Vec::<u8>::new();
    let mut writer = FileWriter::new(out).await?;
    writer.write_composite("weight", &head, &members).await?;
    let bytes = writer.finish(vec![]).await?;
    println!("Wrote a {}-byte Hurray file.", bytes.len());

    // Read the whole composite back, reassembled and validated.
    let mut reader = FileReader::open(Cursor::new(bytes)).await?;
    println!(
        "Tensors in file: {:?}",
        reader.tensor_names().collect::<Vec<_>>()
    );

    let composite = reader.read_composite("weight").await?;
    println!(
        "Composite '{}': head shape {:?}, {} member(s)",
        composite.name,
        composite.head.shape.dims(),
        composite.members.len()
    );
    for member in &composite.members {
        if let FileItem::Tensor(t) = member {
            println!("  member '{}': {} data byte(s)", t.name, t.buffers[0].len());
        }
    }
    assert_eq!(composite.members.len(), 2);
    println!("Composite file round-trip OK.");
    Ok(())
}
