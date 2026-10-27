//! The sans-io file core, driven with no runtime anywhere (ADR-041 § 7, step 4).
//!
//! Everything here runs on `std::io` over a `Vec<u8>`, which is the point: if these tests
//! needed Tokio, the core would not be sans-io.

use std::io::{Read, Seek, SeekFrom};
use std::sync::{Arc, Mutex};

use hurray_core::{
    buffer_size_bytes,
    layout::{CompositeLayout, CompositionRule},
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, ShardDescriptor, SyncMode,
    TensorDescriptor,
};
use hurray_io::file::{
    flatten_composite, FileCompositeNode, FileItem, FileReaderCore, FileStep, FileWriteOp,
    FileWriterCore, FileWriterOptions, KvValue,
};
use hurray_io::stream::{
    AlignedDestination, Destination, DestinationAlloc, DestinationRequest, SystemDestinations,
    Unusable, WriteStep,
};
use hurray_io::{Error, Result};

// ── descriptors ───────────────────────────────────────────────────────────────

fn buf(byte_size: u64, alignment: u32) -> BufferHandle {
    BufferHandle::new(
        byte_size,
        alignment,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )
    .unwrap()
}

fn row_major(byte_size: u64, alignment: u32) -> TensorDescriptor {
    TensorDescriptor::new(
        1,
        0,
        ElementType::Uint8,
        Shape::new(vec![byte_size.max(1)]).unwrap(),
        0,
        LayoutDescriptor::RowMajor,
        vec![buf(byte_size, alignment)],
        None,
        None,
        None,
        None,
    )
    .unwrap()
}

/// Three buffers declaring three different alignments: 64, 128 and 4096 (§ 5).
fn three_alignments() -> TensorDescriptor {
    TensorDescriptor::new(
        1,
        0,
        ElementType::Uint8,
        Shape::new(vec![192u64]).unwrap(),
        0,
        LayoutDescriptor::Coo(hurray_core::layout::CooLayout::new(8, false)),
        vec![buf(64, 64), buf(128, 128), buf(4096, 4096)],
        None,
        None,
        None,
        None,
    )
    .unwrap()
}

fn composite_head(rule: CompositionRule, member_count: u32) -> TensorDescriptor {
    TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(vec![8u64, 8]).unwrap(),
        0,
        LayoutDescriptor::Composite(CompositeLayout::new(rule, member_count).unwrap()),
        vec![],
        None,
        None,
        None,
        None,
    )
    .unwrap()
}

fn partition_member(offset: u64) -> TensorDescriptor {
    TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(vec![8u64, 4]).unwrap(),
        0,
        LayoutDescriptor::RowMajor,
        vec![buf(
            buffer_size_bytes(ElementType::Float32, 8 * 4),
            hurray_core::MIN_BUFFER_ALIGNMENT,
        )],
        None,
        Some(ShardDescriptor::new(vec![8, 8], vec![0, offset]).unwrap()),
        None,
        None,
    )
    .unwrap()
}

fn payload(len: usize, salt: u8) -> Vec<u8> {
    (0..len).map(|i| (i % 251) as u8 ^ salt).collect()
}

// ── drivers, written out so the protocol is visible ───────────────────────────

/// Writes every byte of one operation into `sink`, one offered stretch at a time.
fn drain(mut op: FileWriteOp<'_>, sink: &mut Vec<u8>) -> Result<()> {
    while let WriteStep::Output { .. } = op.step()? {
        let src = op.output_src();
        sink.extend_from_slice(src);
        let n = src.len();
        op.report_written(n)?;
    }
    Ok(())
}

/// Drives an operation, failing the write that would carry the `fail_at`-th byte of the
/// *item* onward. `fail_at == 0` fails before any byte of the item reaches the sink.
fn drain_failing(mut op: FileWriteOp<'_>, sink: &mut Vec<u8>, fail_at: u64) -> Result<()> {
    while let WriteStep::Output { .. } = op.step()? {
        if op.emitted() >= fail_at {
            return Err(Error::Io(std::io::Error::other("simulated")));
        }
        let src = op.output_src();
        let n = src.len().min((fail_at - op.emitted()) as usize);
        sink.extend_from_slice(&src[..n]);
        op.report_written(n)?;
    }
    Ok(())
}

/// Answers every range the core asks for, at most `chunk` bytes at a time.
fn drive<A: DestinationAlloc>(
    core: &mut FileReaderCore<A>,
    file: &mut std::io::Cursor<Vec<u8>>,
    chunk: usize,
) -> Result<FileStep> {
    loop {
        match core.step()? {
            FileStep::Range { offset, len } => {
                file.seek(SeekFrom::Start(offset)).map_err(Error::Io)?;
                let take = len.min(chunk);
                let n = file
                    .read(&mut core.input_dst()[..take])
                    .map_err(Error::Io)?;
                core.report_filled(n)?;
            }
            other => return Ok(other),
        }
    }
}

/// A one-tensor file, built with the write core.
fn one_tensor_file(name: &str, descriptor: &TensorDescriptor, buffers: &[&[u8]]) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut core = FileWriterCore::new();
    drain(core.header().unwrap(), &mut bytes).unwrap();
    drain(core.tensor(name, descriptor, buffers).unwrap(), &mut bytes).unwrap();
    drain(core.finish(vec![]).unwrap(), &mut bytes).unwrap();
    bytes
}

/// A group of two whose first member is a partition of two: nesting depth 2.
fn nested_composite_file() -> (Vec<u8>, Vec<Vec<u8>>) {
    let group = composite_head(CompositionRule::Group, 2);
    let part = composite_head(CompositionRule::Partition, 2);
    let tile_0 = partition_member(0);
    let tile_4 = partition_member(4);
    let tail = row_major(16, 64);

    let data = vec![payload(128, 0x11), payload(128, 0x22), payload(16, 0x33)];
    let tile_0_buffers: [&[u8]; 1] = [&data[0]];
    let tile_4_buffers: [&[u8]; 1] = [&data[1]];
    let tail_buffers: [&[u8]; 1] = [&data[2]];

    let inner = [
        FileCompositeNode::Tensor {
            name: "part.0",
            descriptor: &tile_0,
            buffers: &tile_0_buffers,
        },
        FileCompositeNode::Tensor {
            name: "part.1",
            descriptor: &tile_4,
            buffers: &tile_4_buffers,
        },
    ];
    let members = [
        FileCompositeNode::Composite {
            name: "part",
            head: &part,
            members: &inner,
        },
        FileCompositeNode::Tensor {
            name: "group.1",
            descriptor: &tail,
            buffers: &tail_buffers,
        },
    ];

    let mut bytes = Vec::new();
    let mut core = FileWriterCore::new();
    drain(core.header().unwrap(), &mut bytes).unwrap();
    for node in flatten_composite("group", &group, &members).unwrap() {
        drain(
            core.tensor(node.name, node.descriptor, node.buffers)
                .unwrap(),
            &mut bytes,
        )
        .unwrap();
    }
    drain(core.finish(vec![]).unwrap(), &mut bytes).unwrap();
    (bytes, data)
}

/// Records the address of every destination it hands out, so § 2 can be checked by
/// identity rather than by equality of contents.
#[derive(Clone)]
struct Recording {
    seen: Arc<Mutex<Vec<(usize, u64, u32)>>>,
}

impl DestinationAlloc for Recording {
    type Dest = AlignedDestination;

    fn allocate(&mut self, request: DestinationRequest) -> Result<Self::Dest> {
        let mut dest = SystemDestinations.allocate(request)?;
        let address = dest.as_mut_slice().as_ptr() as usize;
        self.seen
            .lock()
            .unwrap()
            .push((address, request.byte_size, request.alignment));
        Ok(dest)
    }
}

// ── § 2: the core owns the destination, and hands that same memory out ────────

#[test]
fn the_buffer_handed_out_is_the_memory_the_driver_filled() {
    let data = payload(256, 0x5A);
    let descriptor = row_major(256, 64);
    let bytes = one_tensor_file("t", &descriptor, &[&data]);

    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut core =
        FileReaderCore::with_destinations(bytes.len() as u64, Recording { seen: seen.clone() });
    let mut file = std::io::Cursor::new(bytes);
    assert_eq!(
        drive(&mut core, &mut file, usize::MAX).unwrap(),
        FileStep::Ready
    );

    core.begin_tensor("t").unwrap();
    assert_eq!(
        drive(&mut core, &mut file, usize::MAX).unwrap(),
        FileStep::Tensor
    );
    let tensor = core.take_tensor().unwrap();

    let recorded = seen.lock().unwrap().clone();
    assert_eq!(recorded.len(), 1, "one destination for one buffer");
    assert_eq!(
        tensor.buffers[0].as_ptr() as usize,
        recorded[0].0,
        "the tensor's bytes are the destination the driver filled, not a copy of it"
    );
    assert_eq!(tensor.buffers[0].as_ref(), &data[..]);
}

#[test]
fn a_binding_may_supply_the_destinations() {
    let data = payload(64, 0x01);
    let descriptor = row_major(64, 64);
    let bytes = one_tensor_file("t", &descriptor, &[&data]);

    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut core =
        FileReaderCore::with_destinations(bytes.len() as u64, Recording { seen: seen.clone() });
    let mut file = std::io::Cursor::new(bytes);
    drive(&mut core, &mut file, usize::MAX).unwrap();
    core.begin_tensor("t").unwrap();
    drive(&mut core, &mut file, usize::MAX).unwrap();

    // The allocator saw the declared size and alignment, not the allocator's own defaults.
    let recorded = seen.lock().unwrap().clone();
    assert_eq!(recorded.len(), 1);
    assert_eq!((recorded[0].1, recorded[0].2), (64u64, 64u32));
}

// ── § 5: alignment is the declared one ───────────────────────────────────────

#[test]
fn every_buffer_lands_at_its_declared_alignment() {
    let buffers = [payload(64, 0x11), payload(128, 0x22), payload(4096, 0x33)];
    let slices: [&[u8]; 3] = [&buffers[0], &buffers[1], &buffers[2]];
    let descriptor = three_alignments();
    let bytes = one_tensor_file("sparse", &descriptor, &slices);

    let mut core = FileReaderCore::new(bytes.len() as u64);
    let mut file = std::io::Cursor::new(bytes);
    drive(&mut core, &mut file, usize::MAX).unwrap();
    core.begin_tensor("sparse").unwrap();
    drive(&mut core, &mut file, usize::MAX).unwrap();
    let tensor = core.take_tensor().unwrap();

    for (index, (handle, buffer)) in descriptor.buffers.iter().zip(&tensor.buffers).enumerate() {
        let declared = handle.alignment() as usize;
        assert_eq!(
            buffer.as_ptr() as usize % declared,
            0,
            "buffer {index} is not at its declared {declared}-byte alignment"
        );
        assert_eq!(buffer.as_ref(), &buffers[index][..]);
    }
}

#[test]
fn composite_members_land_at_their_declared_alignment() {
    let (bytes, data) = nested_composite_file();
    let mut core = FileReaderCore::new(bytes.len() as u64);
    let mut file = std::io::Cursor::new(bytes);
    drive(&mut core, &mut file, usize::MAX).unwrap();

    core.begin_composite("group").unwrap();
    assert_eq!(
        drive(&mut core, &mut file, usize::MAX).unwrap(),
        FileStep::Composite
    );
    let composite = core.take_composite().unwrap();

    let FileItem::Composite(inner) = &composite.members[0] else {
        panic!("expected a nested composite")
    };
    let FileItem::Tensor(tile) = &inner.members[0] else {
        panic!("expected a tensor")
    };
    assert_eq!(tile.buffers[0].as_ptr() as usize % 64, 0);
    assert_eq!(tile.buffers[0].as_ref(), &data[0][..]);
}

// ── a composite head owns no data region (#243) ───────────────────────────────

/// The next descriptor-aligned offset, which is where a head's successor must begin.
fn next_descriptor_offset(end: u64) -> u64 {
    (end + 7) & !7
}

#[test]
fn a_composite_head_gets_no_data_buffer_alignment_padding() {
    let (bytes, _) = nested_composite_file();
    let mut core = FileReaderCore::new(bytes.len() as u64);
    let mut file = std::io::Cursor::new(bytes);
    drive(&mut core, &mut file, usize::MAX).unwrap();

    // File order, which is what adjacency is defined over.
    let mut index = core.index().to_vec();
    index.sort_unstable_by_key(|e| e.descriptor_offset);

    for (position, entry) in index.iter().enumerate() {
        // The heads of `nested_composite_file`, and the only `buffer_count = 0` tensors a
        // descriptor can express (ADR-027 D-1).
        if entry.name != "group" && entry.name != "part" {
            continue;
        }
        let end = entry.descriptor_offset + u64::from(entry.descriptor_length);
        assert_eq!(
            (entry.data_offset, entry.data_length),
            (end, 0),
            "{}: a head's empty data region begins where its descriptor ends",
            entry.name
        );
        let next = index
            .get(position + 1)
            .unwrap_or_else(|| panic!("{} has no member after it", entry.name));
        assert_eq!(
            next.descriptor_offset,
            next_descriptor_offset(end),
            "{}: its first member must follow at the 8-byte boundary, not past 4 KiB of \
             forbidden padding",
            entry.name
        );
    }
    // Both heads were actually reached, so the loop above cannot have passed vacuously.
    assert_eq!(index.len(), 5);
}

#[test]
fn an_empty_tensor_still_gets_its_data_buffer_alignment_padding() {
    // The boundary the fix must not cross: § Empty Tensors keeps one buffer of
    // `byte_size = 0`, so it is aligned as a buffer. Only `buffer_count = 0` is exempt.
    let descriptor = TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(vec![0u64]).unwrap(),
        0,
        LayoutDescriptor::RowMajor,
        vec![BufferHandle::empty(DeviceTag::Cpu)],
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let bytes = one_tensor_file("empty", &descriptor, &[&[]]);

    let mut core = FileReaderCore::new(bytes.len() as u64);
    let mut file = std::io::Cursor::new(bytes);
    drive(&mut core, &mut file, usize::MAX).unwrap();

    let entry = &core.index()[0];
    assert_eq!(entry.data_length, 0);
    assert_eq!(
        entry.data_offset % 4096,
        0,
        "an empty tensor's buffer is still placed at data_buffer_alignment"
    );
    assert!(entry.data_offset > entry.descriptor_offset + u64::from(entry.descriptor_length));
}

// ── § 4 and § 6: short reads, abandoned reads, and failures that do not stick ─

#[test]
fn a_file_is_readable_one_byte_at_a_time() {
    let data = payload(300, 0x7E);
    let descriptor = row_major(300, 64);
    let bytes = one_tensor_file("t", &descriptor, &[&data]);

    let mut core = FileReaderCore::new(bytes.len() as u64);
    let mut file = std::io::Cursor::new(bytes);
    assert_eq!(drive(&mut core, &mut file, 1).unwrap(), FileStep::Ready);
    core.begin_tensor("t").unwrap();
    assert_eq!(drive(&mut core, &mut file, 1).unwrap(), FileStep::Tensor);
    assert_eq!(core.take_tensor().unwrap().buffers[0].as_ref(), &data[..]);
}

#[test]
fn nested_composites_survive_a_one_byte_feed() {
    let (bytes, data) = nested_composite_file();
    let mut core = FileReaderCore::new(bytes.len() as u64);
    let mut file = std::io::Cursor::new(bytes);
    drive(&mut core, &mut file, 1).unwrap();

    core.begin_composite("group").unwrap();
    assert_eq!(drive(&mut core, &mut file, 1).unwrap(), FileStep::Composite);
    let composite = core.take_composite().unwrap();

    assert_eq!(composite.name, "group");
    assert_eq!(composite.members.len(), 2);
    let FileItem::Composite(inner) = &composite.members[0] else {
        panic!("expected a nested composite")
    };
    assert_eq!(inner.name, "part");
    assert_eq!(inner.members.len(), 2);
    let FileItem::Tensor(tail) = &composite.members[1] else {
        panic!("expected a tensor")
    };
    assert_eq!(tail.buffers[0].as_ref(), &data[2][..]);
}

/// The property a random-access core has and the stream core cannot: a transfer whose
/// byte count was never reported costs nothing, because the next `step` asks for the same
/// absolute range again.
#[test]
fn a_range_whose_count_was_lost_is_simply_reissued() {
    let data = payload(256, 0x3C);
    let descriptor = row_major(256, 64);
    let bytes = one_tensor_file("t", &descriptor, &[&data]);

    let mut core = FileReaderCore::new(bytes.len() as u64);
    let mut file = std::io::Cursor::new(bytes);
    drive(&mut core, &mut file, usize::MAX).unwrap();
    core.begin_tensor("t").unwrap();

    // Walk to the buffer request, then move bytes into the destination and "lose" the
    // count — exactly what a cancelled future would do.
    let mut abandoned = 0;
    loop {
        match core.step().unwrap() {
            FileStep::Range { offset, len } => {
                file.seek(SeekFrom::Start(offset)).unwrap();
                if len == 256 && abandoned == 0 {
                    // Fill half of it and report nothing at all.
                    let dst = core.input_dst();
                    file.read_exact(&mut dst[..128]).unwrap();
                    abandoned = 1;
                    let FileStep::Range {
                        offset: again,
                        len: still,
                    } = core.step().unwrap()
                    else {
                        panic!("the core should re-ask for the range it never heard about")
                    };
                    assert_eq!((again, still), (offset, len), "same range, unchanged");
                    continue;
                }
                let n = file.read(&mut core.input_dst()[..len]).unwrap();
                core.report_filled(n).unwrap();
            }
            FileStep::Tensor => break,
            other => panic!("unexpected {other:?}"),
        }
    }

    assert_eq!(abandoned, 1, "the abandoned read must have happened");
    let tensor = core.take_tensor().unwrap();
    assert_eq!(
        tensor.buffers[0].as_ref(),
        &data[..],
        "re-filling the destination from the start of the range is idempotent"
    );
    assert!(core.is_open(), "nothing was poisoned");
}

#[test]
fn a_failed_job_leaves_the_core_usable() {
    let data = payload(32, 0x09);
    let descriptor = row_major(32, 64);
    let bytes = one_tensor_file("good", &descriptor, &[&data]);

    let mut core = FileReaderCore::new(bytes.len() as u64);
    let mut file = std::io::Cursor::new(bytes);
    drive(&mut core, &mut file, usize::MAX).unwrap();

    assert!(matches!(
        core.begin_tensor("missing"),
        Err(Error::TensorNotFound(name)) if name == "missing"
    ));
    assert_eq!(core.step().unwrap(), FileStep::Ready);

    core.begin_tensor("good").unwrap();
    drive(&mut core, &mut file, usize::MAX).unwrap();
    assert_eq!(core.take_tensor().unwrap().buffers[0].as_ref(), &data[..]);
}

#[test]
fn a_zero_byte_read_is_a_truncated_file() {
    let data = payload(64, 0x00);
    let descriptor = row_major(64, 64);
    let bytes = one_tensor_file("t", &descriptor, &[&data]);

    // Claim the file is longer than it is, so the trailer range reads past the end.
    let mut core = FileReaderCore::new(bytes.len() as u64 + 64);
    let mut file = std::io::Cursor::new(bytes);
    assert!(matches!(
        drive(&mut core, &mut file, usize::MAX),
        Err(Error::UnexpectedEof)
    ));
    // The open sequence failing is terminal: there is no index to start again from.
    assert!(!core.is_open());
    assert!(matches!(core.step(), Err(Error::CoreFailed)));
}

#[test]
fn a_header_alignment_of_zero_is_rejected_not_divided_by() {
    let data = payload(64, 0x00);
    let descriptor = row_major(64, 64);
    let mut bytes = one_tensor_file("t", &descriptor, &[&data]);
    // `data_buffer_alignment` at offset 16 of the header, zeroed.
    bytes[16..20].copy_from_slice(&0u32.to_le_bytes());

    let mut core = FileReaderCore::new(bytes.len() as u64);
    let mut file = std::io::Cursor::new(bytes);
    assert!(matches!(
        drive(&mut core, &mut file, usize::MAX),
        Err(Error::InvalidHeader(_))
    ));
}

#[test]
fn a_torn_composite_is_reported_as_torn() {
    // A head that claims two members, with only one tensor after it.
    let head = composite_head(CompositionRule::Group, 2);
    let member = partition_member(0);
    let data = payload(128, 0x44);

    let mut bytes = Vec::new();
    let mut core = FileWriterCore::new();
    drain(core.header().unwrap(), &mut bytes).unwrap();
    drain(core.tensor("g", &head, &[]).unwrap(), &mut bytes).unwrap();
    drain(core.tensor("g.0", &member, &[&data]).unwrap(), &mut bytes).unwrap();
    drain(core.finish(vec![]).unwrap(), &mut bytes).unwrap();

    let mut reader = FileReaderCore::new(bytes.len() as u64);
    let mut file = std::io::Cursor::new(bytes);
    drive(&mut reader, &mut file, usize::MAX).unwrap();
    reader.begin_composite("g").unwrap();
    assert!(matches!(
        drive(&mut reader, &mut file, usize::MAX),
        Err(Error::TornComposite {
            declared: 2,
            actual: 1
        })
    ));
}

#[test]
fn a_plain_tensor_is_not_a_composite() {
    let data = payload(32, 0x00);
    let descriptor = row_major(32, 64);
    let bytes = one_tensor_file("t", &descriptor, &[&data]);

    let mut core = FileReaderCore::new(bytes.len() as u64);
    let mut file = std::io::Cursor::new(bytes);
    drive(&mut core, &mut file, usize::MAX).unwrap();
    core.begin_composite("t").unwrap();
    assert!(matches!(
        drive(&mut core, &mut file, usize::MAX),
        Err(Error::NotAComposite(name)) if name == "t"
    ));
}

#[test]
fn nesting_deeper_than_the_limit_is_refused() {
    let (bytes, _) = nested_composite_file();
    let mut core = FileReaderCore::new(bytes.len() as u64).with_max_composite_depth(1);
    let mut file = std::io::Cursor::new(bytes);
    drive(&mut core, &mut file, usize::MAX).unwrap();
    core.begin_composite("group").unwrap();
    assert!(matches!(
        drive(&mut core, &mut file, usize::MAX),
        Err(Error::CompositeNestingTooDeep { limit: 1 })
    ));
}

// ── § 3: the write core borrows payloads and never assembles them ────────────

#[test]
fn a_payload_is_offered_from_the_callers_own_memory() {
    let data = payload(4096, 0x6B);
    let descriptor = row_major(4096, 64);
    let buffers: [&[u8]; 1] = [&data];

    let mut core = FileWriterCore::new();
    let mut sink = Vec::new();
    drain(core.header().unwrap(), &mut sink).unwrap();

    let mut op = core.tensor("t", &descriptor, &buffers).unwrap();
    let mut offered_from_the_payload = false;
    while let WriteStep::Output { .. } = op.step().unwrap() {
        let src = op.output_src();
        if src.as_ptr() == data.as_ptr() {
            assert_eq!(src.len(), data.len(), "the whole payload in one stretch");
            offered_from_the_payload = true;
        }
        let n = src.len();
        sink.extend_from_slice(src);
        op.report_written(n).unwrap();
    }
    assert!(
        offered_from_the_payload,
        "the payload must be offered in place, never copied into the core first"
    );
}

#[test]
fn a_zero_length_write_is_not_progress() {
    let data = payload(64, 0x00);
    let descriptor = row_major(64, 64);
    let buffers: [&[u8]; 1] = [&data];

    let mut core = FileWriterCore::new();
    // The header is incidental setup: § 6b refuses a tensor before it.
    drain(core.header().unwrap(), &mut Vec::new()).unwrap();
    let mut op = core.tensor("t", &descriptor, &buffers).unwrap();
    assert!(matches!(op.step().unwrap(), WriteStep::Output { .. }));
    assert!(matches!(op.report_written(0), Err(Error::ZeroLengthWrite)));
    // Position is still known, so the operation may be retried.
    assert!(matches!(op.step().unwrap(), WriteStep::Output { .. }));

    // Declaring the sink closed is the other answer, and it poisons the core.
    assert!(matches!(op.report_sink_closed(), Error::SinkClosed));
    drop(op);
    assert!(core.is_unusable());
    assert!(matches!(
        core.tensor("u", &descriptor, &buffers),
        Err(Error::CoreUnusable(_))
    ));
}

#[test]
fn over_reporting_a_write_poisons_the_core() {
    let data = payload(64, 0x00);
    let descriptor = row_major(64, 64);
    let buffers: [&[u8]; 1] = [&data];

    let mut core = FileWriterCore::new();
    // The header is incidental setup: § 6b refuses a tensor before it.
    drain(core.header().unwrap(), &mut Vec::new()).unwrap();
    let mut op = core.tensor("t", &descriptor, &buffers).unwrap();
    let WriteStep::Output { len } = op.step().unwrap() else {
        panic!("expected output")
    };
    assert!(matches!(
        op.report_written(len + 1),
        Err(Error::InvalidProgressReport { .. })
    ));
    drop(op);
    assert!(core.is_unusable());
}

#[test]
fn a_short_write_re_offers_the_remainder() {
    let data = payload(1024, 0x2D);
    let descriptor = row_major(1024, 64);
    let buffers: [&[u8]; 1] = [&data];

    let mut sink = Vec::new();
    let mut core = FileWriterCore::new();
    // One byte per call, the whole way through.
    let mut op = core.header().unwrap();
    while let WriteStep::Output { .. } = op.step().unwrap() {
        let src = op.output_src();
        sink.push(src[0]);
        op.report_written(1).unwrap();
    }
    drop(op);
    let mut op = core.tensor("t", &descriptor, &buffers).unwrap();
    while let WriteStep::Output { .. } = op.step().unwrap() {
        let src = op.output_src();
        sink.push(src[0]);
        op.report_written(1).unwrap();
    }
    drop(op);
    let mut op = core.finish(vec![]).unwrap();
    while let WriteStep::Output { .. } = op.step().unwrap() {
        let src = op.output_src();
        sink.push(src[0]);
        op.report_written(1).unwrap();
    }
    drop(op);

    // The file that comes out of one-byte writes is the same file.
    assert_eq!(sink, one_tensor_file("t", &descriptor, &buffers));
}

// ── the round trip, and the composite write order ────────────────────────────

#[test]
fn the_cores_round_trip_a_file_with_kv_and_a_sorted_index() {
    let zeta = row_major(256, 64);
    let alpha = row_major(16, 64);
    let zeta_data = payload(256, 0x55);
    let alpha_data = payload(16, 0x66);
    let kv = vec![
        ("model".to_string(), KvValue::String("hurray".into())),
        ("layers".to_string(), KvValue::Uint64(32)),
        (
            "sizes".to_string(),
            KvValue::Array(vec![KvValue::Int64(1), KvValue::Int64(2)]),
        ),
    ];

    let mut bytes = Vec::new();
    let mut writer = FileWriterCore::with_options(FileWriterOptions {
        data_buffer_alignment: 8192,
        sorted_index: true,
    })
    .unwrap();
    drain(writer.header().unwrap(), &mut bytes).unwrap();
    drain(
        writer.tensor("zeta", &zeta, &[&zeta_data]).unwrap(),
        &mut bytes,
    )
    .unwrap();
    drain(
        writer.tensor("alpha", &alpha, &[&alpha_data]).unwrap(),
        &mut bytes,
    )
    .unwrap();
    drain(writer.finish(kv.clone()).unwrap(), &mut bytes).unwrap();

    let mut core = FileReaderCore::new(bytes.len() as u64);
    let mut file = std::io::Cursor::new(bytes);
    drive(&mut core, &mut file, 7).unwrap();

    // `sorted_index` orders the index by name, not by write order.
    let names: Vec<&str> = core.index().iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, ["alpha", "zeta"]);
    assert_eq!(core.kv(), &kv[..]);

    core.begin_descriptor("zeta").unwrap();
    assert_eq!(
        drive(&mut core, &mut file, 7).unwrap(),
        FileStep::Descriptor
    );
    assert_eq!(core.take_descriptor().unwrap(), zeta);

    core.begin_tensor("alpha").unwrap();
    drive(&mut core, &mut file, 7).unwrap();
    assert_eq!(
        core.take_tensor().unwrap().buffers[0].as_ref(),
        &alpha_data[..]
    );
}

#[test]
fn flatten_composite_puts_every_head_before_its_members() {
    let group = composite_head(CompositionRule::Group, 2);
    let part = composite_head(CompositionRule::Partition, 2);
    let tile_0 = partition_member(0);
    let tile_4 = partition_member(4);
    let tail = row_major(16, 64);
    let data = payload(128, 0x00);
    let tail_data = payload(16, 0x00);
    let tile_buffers: [&[u8]; 1] = [&data];
    let tail_buffers: [&[u8]; 1] = [&tail_data];

    let inner = [
        FileCompositeNode::Tensor {
            name: "part.0",
            descriptor: &tile_0,
            buffers: &tile_buffers,
        },
        FileCompositeNode::Tensor {
            name: "part.1",
            descriptor: &tile_4,
            buffers: &tile_buffers,
        },
    ];
    let members = [
        FileCompositeNode::Composite {
            name: "part",
            head: &part,
            members: &inner,
        },
        FileCompositeNode::Tensor {
            name: "group.1",
            descriptor: &tail,
            buffers: &tail_buffers,
        },
    ];

    let order = flatten_composite("group", &group, &members).unwrap();
    let names: Vec<&str> = order.iter().map(|w| w.name).collect();
    assert_eq!(names, ["group", "part", "part.0", "part.1", "group.1"]);
    assert!(order[0].buffers.is_empty(), "a head declares no buffers");
    assert!(order[1].buffers.is_empty());
}

#[test]
fn flatten_composite_rejects_a_tree_that_does_not_validate() {
    // A partition of two whose members do not cover the parent shape.
    let head = composite_head(CompositionRule::Partition, 2);
    let tile_0 = partition_member(0);
    let data = payload(128, 0x00);
    let buffers: [&[u8]; 1] = [&data];
    let members = [
        FileCompositeNode::Tensor {
            name: "p.0",
            descriptor: &tile_0,
            buffers: &buffers,
        },
        FileCompositeNode::Tensor {
            name: "p.1",
            descriptor: &tile_0,
            buffers: &buffers,
        },
    ];

    assert!(flatten_composite("p", &head, &members).is_err());
}

#[test]
fn a_file_too_short_for_a_header_is_truncated_not_malformed() {
    // Shorter than the 64-byte header: the header range comes back short, then empty.
    let mut file = std::io::Cursor::new(b"HRRYFILE".to_vec());
    let mut core = FileReaderCore::new(8);
    assert!(matches!(
        drive(&mut core, &mut file, usize::MAX),
        Err(Error::UnexpectedEof)
    ));
}

#[test]
fn a_file_too_short_for_a_trailer_is_malformed() {
    // Long enough for the header, too short for the 40-byte trailer after it.
    let data = payload(64, 0x00);
    let descriptor = row_major(64, 64);
    let bytes = one_tensor_file("t", &descriptor, &[&data])[..70].to_vec();
    let mut file = std::io::Cursor::new(bytes);
    let mut core = FileReaderCore::new(70);
    assert!(matches!(
        drive(&mut core, &mut file, usize::MAX),
        Err(Error::InvalidHeader(_))
    ));
}

#[test]
fn nothing_may_follow_the_footer() {
    let data = payload(64, 0x00);
    let descriptor = row_major(64, 64);
    let buffers: [&[u8]; 1] = [&data];

    let mut sink = Vec::new();
    let mut core = FileWriterCore::new();
    drain(core.header().unwrap(), &mut sink).unwrap();
    drain(core.tensor("t", &descriptor, &buffers).unwrap(), &mut sink).unwrap();
    drain(core.finish(vec![]).unwrap(), &mut sink).unwrap();

    // The trailer is the last thing in the file; a tensor after it would corrupt it.
    assert!(matches!(
        core.tensor("u", &descriptor, &buffers),
        Err(Error::CoreFailed)
    ));
    assert!(matches!(core.finish(vec![]), Err(Error::CoreFailed)));
}

/// Issue #248's own repro, driven to completion. A write that fails before its first byte
/// consumed neither the name nor a byte of the file, so the same tensor may be submitted
/// again — and the file that results is *byte-identical* to one written with no failure at
/// all. That equality is the whole claim: no burnt name, and no orphan descriptor ahead of
/// the next item's bytes.
#[test]
fn a_write_that_fails_before_its_first_byte_is_retryable_under_the_same_name() {
    let data = payload(256, 0x11);
    let descriptor = row_major(256, 64);
    let buffers: [&[u8]; 1] = [&data];

    let mut sink = Vec::new();
    let mut core = FileWriterCore::new();
    drain(core.header().unwrap(), &mut sink).unwrap();

    // Fails before any byte of the tensor goes out.
    assert!(matches!(
        drain_failing(
            core.tensor("weight", &descriptor, &buffers).unwrap(),
            &mut sink,
            0
        ),
        Err(Error::Io(_))
    ));
    assert!(!core.is_unusable());

    // The name is free, which `DuplicateTensorName` used to deny.
    drain(
        core.tensor("weight", &descriptor, &buffers).unwrap(),
        &mut sink,
    )
    .unwrap();
    drain(core.finish(Vec::new()).unwrap(), &mut sink).unwrap();

    assert_eq!(sink, one_tensor_file("weight", &descriptor, &buffers));
}

/// The other half of § 6b: once a byte is out, no forward-only writer can make the file
/// conforming, so the core is terminal — and `finish` must refuse rather than write a
/// clean index over a torn region.
#[test]
fn a_write_that_fails_after_its_first_byte_poisons_the_core() {
    let data = payload(256, 0x11);
    let descriptor = row_major(256, 64);
    let buffers: [&[u8]; 1] = [&data];

    let mut sink = Vec::new();
    let mut core = FileWriterCore::new();
    drain(core.header().unwrap(), &mut sink).unwrap();

    assert!(matches!(
        drain_failing(
            core.tensor("weight", &descriptor, &buffers).unwrap(),
            &mut sink,
            8
        ),
        Err(Error::Io(_))
    ));
    assert!(core.is_unusable());
    assert!(matches!(
        core.tensor("other", &descriptor, &buffers),
        Err(Error::CoreUnusable(Unusable::IncompleteOutput))
    ));
    assert!(matches!(
        core.finish(Vec::new()),
        Err(Error::CoreUnusable(Unusable::IncompleteOutput))
    ));
}

/// An operation simply dropped — no error reported, which is what a driver that returns
/// early does — must leave the core exactly as it was. The poison cannot depend on being
/// told, and neither can the release of the name.
#[test]
fn an_op_dropped_before_its_first_byte_leaves_the_name_free() {
    let data = payload(256, 0x11);
    let descriptor = row_major(256, 64);
    let buffers: [&[u8]; 1] = [&data];

    let mut sink = Vec::new();
    let mut core = FileWriterCore::new();
    drain(core.header().unwrap(), &mut sink).unwrap();

    let mut op = core.tensor("weight", &descriptor, &buffers).unwrap();
    assert!(matches!(op.step().unwrap(), WriteStep::Output { .. }));
    assert_eq!(op.emitted(), 0);
    drop(op);

    assert!(!core.is_unusable());
    assert_eq!(sink.len(), 64, "nothing of the item reached the sink");
    drain(
        core.tensor("weight", &descriptor, &buffers).unwrap(),
        &mut sink,
    )
    .unwrap();
    drain(core.finish(Vec::new()).unwrap(), &mut sink).unwrap();
    assert_eq!(sink, one_tensor_file("weight", &descriptor, &buffers));
}

/// ADR-041's struck open question. `finish` is an item under § 6b, so a footer that emitted
/// nothing is retryable — which turns a wasted multi-gigabyte data write into a complete
/// file, and is the retry most worth having in the whole path.
#[test]
fn a_footer_that_fails_before_its_first_byte_may_be_written_again() {
    let data = payload(256, 0x11);
    let descriptor = row_major(256, 64);
    let buffers: [&[u8]; 1] = [&data];

    let mut sink = Vec::new();
    let mut core = FileWriterCore::new();
    drain(core.header().unwrap(), &mut sink).unwrap();
    drain(
        core.tensor("weight", &descriptor, &buffers).unwrap(),
        &mut sink,
    )
    .unwrap();

    assert!(matches!(
        drain_failing(core.finish(Vec::new()).unwrap(), &mut sink, 0),
        Err(Error::Io(_))
    ));
    assert!(!core.is_unusable());

    drain(core.finish(Vec::new()).unwrap(), &mut sink).unwrap();
    assert_eq!(sink, one_tensor_file("weight", &descriptor, &buffers));
}

/// The `sorted_index` path mutates `entries` while building the footer, which is the one
/// place § 6b's rule is bent. A footer retried *after another tensor was written* is what
/// would catch an in-place sort leaking out of an abandoned operation.
#[test]
fn a_sorted_index_survives_a_footer_retry_with_a_tensor_in_between() {
    let data = payload(256, 0x11);
    let descriptor = row_major(256, 64);
    let buffers: [&[u8]; 1] = [&data];

    let sorted = || FileWriterOptions {
        sorted_index: true,
        ..Default::default()
    };
    let mut sink = Vec::new();
    let mut core = FileWriterCore::with_options(sorted()).unwrap();
    drain(core.header().unwrap(), &mut sink).unwrap();
    drain(core.tensor("b", &descriptor, &buffers).unwrap(), &mut sink).unwrap();

    // A footer that emitted nothing, then more work, then the real footer.
    assert!(drain_failing(core.finish(Vec::new()).unwrap(), &mut sink, 0).is_err());
    drain(core.tensor("a", &descriptor, &buffers).unwrap(), &mut sink).unwrap();
    drain(core.finish(Vec::new()).unwrap(), &mut sink).unwrap();

    // Identical to writing both tensors in the same order with no failure at all.
    let mut clean = Vec::new();
    let mut c2 = FileWriterCore::with_options(sorted()).unwrap();
    drain(c2.header().unwrap(), &mut clean).unwrap();
    drain(c2.tensor("b", &descriptor, &buffers).unwrap(), &mut clean).unwrap();
    drain(c2.tensor("a", &descriptor, &buffers).unwrap(), &mut clean).unwrap();
    drain(c2.finish(Vec::new()).unwrap(), &mut clean).unwrap();
    assert_eq!(sink, clean);
}

/// `emitted` means bytes of the item on the sink, so alignment padding counts. A reader of
/// the signature would otherwise assume it meant payload bytes.
#[test]
fn emitted_counts_padding_as_well_as_descriptor_and_payload() {
    let data = payload(256, 0x11);
    // A 4096-byte alignment forces real padding between descriptor and payload.
    let descriptor = row_major(256, 4096);
    let buffers: [&[u8]; 1] = [&data];

    let options = FileWriterOptions {
        data_buffer_alignment: 4096,
        ..Default::default()
    };
    let mut sink = Vec::new();
    let mut core = FileWriterCore::with_options(options).unwrap();
    drain(core.header().unwrap(), &mut sink).unwrap();

    let mut op = core.tensor("weight", &descriptor, &buffers).unwrap();
    let before = sink.len();
    while let WriteStep::Output { .. } = op.step().unwrap() {
        let src = op.output_src();
        sink.extend_from_slice(src);
        let n = src.len();
        op.report_written(n).unwrap();
    }
    let item_bytes = (sink.len() - before) as u64;
    assert_eq!(op.emitted(), item_bytes);
    // Strictly more than descriptor + payload: the gap between them was padded.
    assert!(item_bytes > 256);
}

/// § 6b applies to the header too: a tensor submitted before one is refused, and so is a
/// second header.
#[test]
fn the_header_must_be_written_before_anything_else() {
    let data = payload(256, 0x11);
    let descriptor = row_major(256, 64);
    let buffers: [&[u8]; 1] = [&data];

    let mut core = FileWriterCore::new();
    assert!(matches!(
        core.tensor("weight", &descriptor, &buffers),
        Err(Error::HeaderNotWritten)
    ));
    assert!(matches!(
        core.finish(Vec::new()),
        Err(Error::HeaderNotWritten)
    ));

    let mut sink = Vec::new();
    drain(core.header().unwrap(), &mut sink).unwrap();
    // A second header would describe a file inside a file.
    assert!(matches!(core.header(), Err(Error::CoreFailed)));
}

/// A header abandoned mid-emission is a torn item like any other, even though nothing it
/// would have committed was a tensor.
#[test]
fn a_header_abandoned_mid_emission_poisons_the_core() {
    let mut sink = Vec::new();
    let mut core = FileWriterCore::new();
    assert!(matches!(
        drain_failing(core.header().unwrap(), &mut sink, 16),
        Err(Error::Io(_))
    ));
    assert!(core.is_unusable());
    assert!(matches!(
        core.finish(Vec::new()),
        Err(Error::CoreUnusable(Unusable::IncompleteOutput))
    ));
}
