//! Demonstrates writing then reading two tensors through the streaming wire format.
//!
//! Run with:
//! ```text
//! cargo run --example stream_roundtrip --features tokio -p hurray-io
//! ```

use hurray_core::{
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode, TensorDescriptor,
    MIN_BUFFER_ALIGNMENT,
};
use hurray_io::stream::{StreamReader, StreamWriter};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build a 4×4 f32 tensor descriptor (64 bytes of data).
    let handle_a = BufferHandle::new(
        64,
        MIN_BUFFER_ALIGNMENT,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )?;
    let shape_a = Shape::new(vec![4u64, 4]).unwrap();
    let desc_a = TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        shape_a,
        0,
        LayoutDescriptor::RowMajor,
        vec![handle_a],
        None,
        None,
        None,
        None,
    )?;
    let data_a: Vec<u8> = (0u8..64).collect();

    // Build a 2×3 f16 tensor descriptor (12 bytes of data).
    let handle_b = BufferHandle::new(
        12,
        MIN_BUFFER_ALIGNMENT,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )?;
    let shape_b = Shape::new(vec![2u64, 3]).unwrap();
    let desc_b = TensorDescriptor::new(
        1,
        0,
        ElementType::Float16,
        shape_b,
        0,
        LayoutDescriptor::RowMajor,
        vec![handle_b],
        None,
        None,
        None,
        None,
    )?;
    let data_b: Vec<u8> = vec![0u8; 12];

    // Write both tensors to an in-memory buffer.
    let mut wire = Vec::<u8>::new();
    let mut writer = StreamWriter::new(&mut wire);
    writer.write_tensor(&desc_a, &[&data_a]).await?;
    writer.write_tensor(&desc_b, &[&data_b]).await?;
    writer.finish().await?;

    println!("Wrote {} bytes to wire.", wire.len());

    // Read them back.
    let mut reader = StreamReader::new(wire.as_slice());
    let mut count = 0usize;
    while let Some(tensor) = reader.next_tensor().await? {
        count += 1;
        let total_bytes: usize = tensor.buffers.iter().map(|b| b.len()).sum();
        println!(
            "Tensor {count}: element_type={:?}, buffers={}, total_bytes={total_bytes}",
            tensor.descriptor.element_type,
            tensor.buffers.len(),
        );
        // Verify the first tensor's payload matches what we wrote.
        if count == 1 {
            assert_eq!(&tensor.buffers[0][..], &data_a[..], "payload mismatch");
        }
    }

    assert_eq!(count, 2, "expected 2 tensors");
    println!("Round-trip OK.");
    Ok(())
}
