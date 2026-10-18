//! Driving the sans-io stream core from blocking `std::io`, with no runtime at all.
//!
//! The point is what is absent: no Tokio, no executor, no `async` — and no feature of
//! `hurray-io` beyond the default. The two drivers below are the whole bridge, and the
//! same shape works over a socket, a pipe, or a C caller's callbacks.
//!
//! Run with:
//!
//! ```text
//! cargo run --example sans_io_blocking
//! ```

use std::io::{Read, Write};

use hurray_core::{
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode, TensorDescriptor,
    MIN_BUFFER_ALIGNMENT, PAGE_ALIGNMENT,
};
use hurray_io::stream::{ReadStep, StreamItem, StreamReaderCore, StreamWriterCore, WriteStep};
use hurray_io::Result;

/// Writes one tensor to any blocking sink. This is the entire write driver.
fn write_tensor<W: Write>(
    sink: &mut W,
    core: &mut StreamWriterCore,
    descriptor: &TensorDescriptor,
    payloads: &[&[u8]],
) -> Result<()> {
    let mut op = core.tensor(descriptor, payloads)?;
    loop {
        match op.step()? {
            WriteStep::Output { .. } => {
                // The sink may take fewer bytes than offered; the core re-offers the rest.
                match sink.write(op.output_src())? {
                    0 => return Err(op.report_sink_closed()),
                    n => op.report_written(n)?,
                }
            }
            WriteStep::ItemWritten => return Ok(()),
        }
    }
}

/// Reads every item from any blocking source. This is the entire read driver.
fn read_items<R: Read>(source: &mut R, core: &mut StreamReaderCore) -> Result<Vec<StreamItem>> {
    let mut items = Vec::new();
    loop {
        match core.step()? {
            ReadStep::Input { .. } => {
                // The core supplies the destination, already aligned to what the buffer
                // handle declares; this read is the only copy the payload ever takes.
                let n = source.read(core.input_dst())?;
                core.report_filled(n)?;
            }
            ReadStep::Item => items.extend(core.take_item()),
            ReadStep::Finished => return Ok(items),
        }
    }
}

fn descriptor(byte_size: u64, alignment: u32, dims: Vec<u64>) -> Result<TensorDescriptor> {
    let handle = BufferHandle::new(
        byte_size,
        alignment,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )?;
    Ok(TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(dims)?,
        0,
        LayoutDescriptor::RowMajor,
        vec![handle],
        None,
        None,
        None,
        None,
    )?)
}

fn main() -> Result<()> {
    // A [16, 16] tensor at the 64-byte floor, and a [8, 8] one declaring a page — the
    // alignment a GPU/IPC producer would set.
    let big = descriptor(1024, MIN_BUFFER_ALIGNMENT, vec![16, 16])?;
    let small = descriptor(256, PAGE_ALIGNMENT, vec![8, 8])?;
    let big_data: Vec<u8> = (0..1024).map(|i| i as u8).collect();
    let small_data = vec![0x5Au8; 256];

    let mut wire = Vec::new();
    let mut writer = StreamWriterCore::new();
    write_tensor(&mut wire, &mut writer, &big, &[&big_data])?;
    write_tensor(&mut wire, &mut writer, &small, &[&small_data])?;
    println!("wrote {} bytes for 2 tensors", wire.len());

    let mut reader = StreamReaderCore::new();
    let items = read_items(&mut wire.as_slice(), &mut reader)?;
    println!("read {} item(s) back", items.len());

    for item in &items {
        let StreamItem::Tensor(tensor) = item else {
            continue;
        };
        let declared = tensor.descriptor.buffers[0].alignment();
        let address = tensor.buffers[0].as_ptr() as usize;
        println!(
            "  {:?} {:?}: {} bytes, declared alignment {}, address {}",
            tensor.descriptor.element_type,
            tensor.descriptor.shape.dims(),
            tensor.buffers[0].len(),
            declared,
            if address.is_multiple_of(declared as usize) {
                "honours it"
            } else {
                "DOES NOT honour it"
            },
        );
    }

    assert_eq!(items.len(), 2);
    Ok(())
}
