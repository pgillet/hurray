//! Driving the sans-io **file** core from blocking `std::io`, with no runtime at all.
//!
//! The stream core is fed forwards; this one is asked by range. It says *which* bytes of
//! the file it wants — the trailer at `size - 40`, the index located from it, then one
//! descriptor and one buffer at a time — and the driver's whole job is to seek there and
//! say how many bytes arrived.
//!
//! The point is what is absent: no Tokio, no executor, no `async`, and no feature of
//! `hurray-io` beyond the default. The same two drivers work over a file, a memory span,
//! an HTTP range request, or a C caller's callbacks.
//!
//! Run with:
//!
//! ```text
//! cargo run --example sans_io_file
//! ```

use std::io::{Cursor, Read, Seek, SeekFrom, Write};

use hurray_core::{
    layout::{CompositeLayout, CompositionRule},
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, ShardDescriptor, SyncMode,
    TensorDescriptor, MIN_BUFFER_ALIGNMENT, PAGE_ALIGNMENT,
};
use hurray_io::file::{
    flatten_composite, FileCompositeNode, FileItem, FileReaderCore, FileStep, FileTensor,
    FileWriteOp, FileWriterCore, KvValue,
};
use hurray_io::stream::WriteStep;
use hurray_io::Result;

/// Writes every byte of one submitted item to a blocking sink. The entire write driver.
fn drain<W: Write>(mut op: FileWriteOp<'_>, sink: &mut W) -> Result<()> {
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

/// Answers every range the core asks for, and returns the step that ended the job. The
/// entire read driver.
fn drive<R: Read + Seek>(core: &mut FileReaderCore, source: &mut R) -> Result<FileStep> {
    loop {
        match core.step()? {
            FileStep::Range { offset, len } => {
                source.seek(SeekFrom::Start(offset))?;
                // `read`, not `read_exact`: the count is the position, and a bulk helper
                // that fails cannot report one. A short read is normal here.
                let n = source.read(&mut core.input_dst()[..len])?;
                core.report_filled(n)?;
            }
            other => return Ok(other),
        }
    }
}

fn tensor(byte_size: u64, alignment: u32, dims: Vec<u64>) -> Result<TensorDescriptor> {
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
    // ── Write a file: two tensors and a partition composite, plus KV metadata ──
    let embeddings = tensor(256, MIN_BUFFER_ALIGNMENT, vec![8, 8])?;
    let embeddings_data: Vec<u8> = (0..256).map(|i| (i % 251) as u8).collect();

    // A page-aligned buffer, the alignment an mmap-ing consumer is entitled to.
    let bias = tensor(64, PAGE_ALIGNMENT, vec![16])?;
    let bias_data = vec![0x7Fu8; 64];

    let head = TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(vec![8, 8])?,
        0,
        LayoutDescriptor::Composite(CompositeLayout::new(CompositionRule::Partition, 2)?),
        vec![],
        None,
        None,
        None,
        None,
    )?;
    let left = TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(vec![8, 4])?,
        0,
        LayoutDescriptor::RowMajor,
        vec![BufferHandle::new(
            128,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )?],
        None,
        Some(ShardDescriptor::new(vec![8, 8], vec![0, 0])?),
        None,
        None,
    )?;
    let right = TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(vec![8, 4])?,
        0,
        LayoutDescriptor::RowMajor,
        vec![BufferHandle::new(
            128,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )?],
        None,
        Some(ShardDescriptor::new(vec![8, 8], vec![0, 4])?),
        None,
        None,
    )?;
    let left_data = vec![0x11u8; 128];
    let right_data = vec![0x22u8; 128];
    let left_buffers: [&[u8]; 1] = [&left_data];
    let right_buffers: [&[u8]; 1] = [&right_data];
    let members = [
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

    let mut file = Vec::new();
    let mut writer = FileWriterCore::new();
    drain(writer.header(), &mut file)?;

    let embeddings_buffers: [&[u8]; 1] = [&embeddings_data];
    drain(
        writer.tensor("embeddings", &embeddings, &embeddings_buffers)?,
        &mut file,
    )?;
    let bias_buffers: [&[u8]; 1] = [&bias_data];
    drain(writer.tensor("bias", &bias, &bias_buffers)?, &mut file)?;

    // The composite is validated and flattened into write order first — head, then each
    // member — so a torn group never reaches the sink, and the nesting is walked with an
    // explicit stack rather than by recursing.
    for node in flatten_composite("weight", &head, &members)? {
        drain(
            writer.tensor(node.name, node.descriptor, node.buffers)?,
            &mut file,
        )?;
    }

    drain(
        writer.finish(vec![
            ("producer".into(), KvValue::String("sans_io_file".into())),
            ("tensor_count".into(), KvValue::Uint64(5)),
        ])?,
        &mut file,
    )?;
    println!("wrote {} bytes", file.len());

    // ── Read it back ───────────────────────────────────────────────────────────
    let mut source = Cursor::new(file);
    let size = source.seek(SeekFrom::End(0))?;

    let mut core = FileReaderCore::new(size);
    // Opening is four ranges: the header, the trailer, the index, the KV section.
    match drive(&mut core, &mut source)? {
        FileStep::Ready => {}
        other => panic!("unexpected {other:?} while opening"),
    }

    println!(
        "index: {:?}",
        core.index().iter().map(|e| &e.name).collect::<Vec<_>>()
    );
    for (key, value) in core.kv() {
        println!("  kv {key} = {value:?}");
    }

    // A descriptor on its own: one range, no payload read at all.
    core.begin_descriptor("bias")?;
    let descriptor = match drive(&mut core, &mut source)? {
        FileStep::Descriptor => core.take_descriptor().expect("a descriptor is ready"),
        other => panic!("unexpected {other:?}"),
    };
    println!(
        "bias: {:?} {:?}, {} byte(s), declares {}-byte alignment",
        descriptor.element_type,
        descriptor.shape.dims(),
        descriptor.buffers[0].byte_size(),
        descriptor.buffers[0].alignment(),
    );

    // A whole tensor: the descriptor, then one range per buffer.
    core.begin_tensor("embeddings")?;
    let tensor: FileTensor = match drive(&mut core, &mut source)? {
        FileStep::Tensor => core.take_tensor().expect("a tensor is ready"),
        other => panic!("unexpected {other:?}"),
    };
    assert_eq!(tensor.buffers[0].as_ref(), &embeddings_data[..]);
    // The buffer sits at the alignment its handle declared, so an aligned SIMD load on it
    // is sound — which is the whole promise of § 5.
    let declared = tensor.descriptor.buffers[0].alignment() as usize;
    assert_eq!(tensor.buffers[0].as_ptr() as usize % declared, 0);
    println!(
        "embeddings: {} byte(s) at {}-byte alignment, no copy after the read",
        tensor.buffers[0].len(),
        declared
    );

    // A composite: the core walks the index in file order and rebuilds the tree.
    core.begin_composite("weight")?;
    let composite = match drive(&mut core, &mut source)? {
        FileStep::Composite => core.take_composite().expect("a composite is ready"),
        other => panic!("unexpected {other:?}"),
    };
    let rule = match &composite.head.layout {
        LayoutDescriptor::Composite(layout) => &layout.rule,
        _ => unreachable!("a composite only ever comes back with a composite head"),
    };
    println!(
        "weight: {:?} of {} member(s)",
        rule,
        composite.members.len()
    );
    for member in &composite.members {
        match member {
            FileItem::Tensor(t) => println!("  {} — {} byte(s)", t.name, t.buffers[0].len()),
            FileItem::Composite(c) => {
                println!("  {} — nested, {} member(s)", c.name, c.members.len())
            }
        }
    }

    Ok(())
}
