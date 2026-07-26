//! Streams a composite tensor (head + members) and reads it back assembled.
//!
//! Builds a partition composite — a `[8, 8]` float32 logical tensor split into two `[8, 4]`
//! tiles that exactly cover the index space — writes it with
//! [`StreamWriter::write_composite`], then reads it back with
//! [`StreamReader::next_item`], which reassembles and validates the group.
//!
//! Run with:
//! ```text
//! cargo run --example composite_stream --features tokio -p hurray-io
//! ```

use hurray_core::{
    buffer_size_bytes,
    layout::{CompositeLayout, CompositionRule},
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, ShardDescriptor, SyncMode,
    TensorDescriptor, MIN_BUFFER_ALIGNMENT,
};
use hurray_io::stream::{CompositeNode, StreamItem, StreamReader, StreamWriter};

fn buf(byte_size: u64) -> Result<BufferHandle, Box<dyn std::error::Error>> {
    Ok(BufferHandle::new(
        byte_size,
        MIN_BUFFER_ALIGNMENT,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )?)
}

/// An `[8, 4]` float32 tile occupying columns `offset..offset+4` of the `[8, 8]` parent.
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
    // The data-less head presents one logical [8, 8] float32 view over two members.
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

    let left = tile(0)?;
    let right = tile(4)?;
    let left_data = vec![0xA5u8; 128]; // 8 * 4 * 4 bytes
    let right_data = vec![0x5Au8; 128];
    let left_buffers: [&[u8]; 1] = [left_data.as_slice()];
    let right_buffers: [&[u8]; 1] = [right_data.as_slice()];

    let members = vec![
        CompositeNode::Tensor {
            descriptor: &left,
            buffers: &left_buffers,
        },
        CompositeNode::Tensor {
            descriptor: &right,
            buffers: &right_buffers,
        },
    ];

    // Write: the head is validated (exact cover, no overlap) before any byte is emitted.
    let mut wire = Vec::<u8>::new();
    let mut writer = StreamWriter::new(&mut wire);
    writer.write_composite(&head, &members).await?;
    writer.finish().await?;
    println!("Wrote a composite to {} bytes of wire.", wire.len());

    // Read: next_item reassembles + validates the head and its members.
    let mut reader = StreamReader::new(wire.as_slice());
    match reader.next_item().await? {
        Some(StreamItem::Composite(c)) => {
            println!(
                "Read composite: head shape {:?}, {} member(s)",
                c.head.shape.dims(),
                c.members.len()
            );
            for (i, member) in c.members.iter().enumerate() {
                if let StreamItem::Tensor(t) = member {
                    println!(
                        "  member {i}: shape {:?}, {} data byte(s)",
                        t.descriptor.shape.dims(),
                        t.buffers[0].len()
                    );
                }
            }
            assert_eq!(c.members.len(), 2);
        }
        other => panic!("expected a composite, got {other:?}"),
    }

    assert!(reader.next_item().await?.is_none(), "expected clean EOF");
    println!("Composite round-trip OK.");
    Ok(())
}
