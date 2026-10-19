//! The contracts ADR-041 §§ 2–6 put on the sans-io stream core, as tests.
//!
//! Everything here drives the core by hand — `step`, transfer, report — with no runtime
//! involved, which is the property under test as much as the results are.

use bytes::Bytes;
use hurray_core::{
    buffer_size_bytes,
    layout::{CompositeLayout, CompositionRule, CooLayout},
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, ShardDescriptor, SyncMode,
    TensorDescriptor, MIN_BUFFER_ALIGNMENT,
};
use hurray_io::stream::{
    CompositeAssembler, CompositeNode, Destination, DestinationAlloc, DestinationRequest, ReadStep,
    StreamItem, StreamReaderCore, StreamReaderOptions, StreamTensor, StreamWriterCore,
    SystemDestinations, Unusable, WriteStep, DEFAULT_MAX_COMPOSITE_DEPTH,
};
use hurray_io::Error;

// ── fixtures ──────────────────────────────────────────────────────────────────

fn buf(byte_size: u64, alignment: u32) -> BufferHandle {
    BufferHandle::new(
        byte_size,
        alignment,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )
    .unwrap()
}

/// A `[8, 8]` float32 row-major tensor: one 256-byte buffer at `alignment`.
fn row_major(alignment: u32) -> TensorDescriptor {
    TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(vec![8u64, 8]).unwrap(),
        0,
        LayoutDescriptor::RowMajor,
        vec![buf(256, alignment)],
        None,
        None,
        None,
        None,
    )
    .unwrap()
}

/// A COO tensor with two buffers declaring different alignments.
fn coo(values_alignment: u32, indices_alignment: u32) -> TensorDescriptor {
    let nnz = 8u64;
    TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(vec![8u64, 8]).unwrap(),
        0,
        LayoutDescriptor::Coo(CooLayout::new(nnz, false)),
        vec![
            buf(32, values_alignment),
            buf(nnz * 2 * 8, indices_alignment),
        ],
        None,
        None,
        None,
        None,
    )
    .unwrap()
}

fn composite_head(rule: CompositionRule, member_count: u32, shape: Vec<u64>) -> TensorDescriptor {
    TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(shape).unwrap(),
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

/// An `[8, 4]` partition member at column `offset` of an `[8, 8]` head.
fn partition_member(offset: u64, alignment: u32) -> TensorDescriptor {
    TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(vec![8u64, 4]).unwrap(),
        0,
        LayoutDescriptor::RowMajor,
        vec![buf(buffer_size_bytes(ElementType::Float32, 32), alignment)],
        None,
        Some(ShardDescriptor::new(vec![8, 8], vec![0, offset]).unwrap()),
        None,
        None,
    )
    .unwrap()
}

// ── drivers (the whole point: these are 20 lines, and synchronous) ────────────

/// Writes one item through the core, handing the sink at most `chunk` bytes at a time.
fn write_tensor(
    descriptor: &TensorDescriptor,
    payloads: &[&[u8]],
    chunk: usize,
) -> Result<Vec<u8>, Error> {
    let mut core = StreamWriterCore::new();
    let mut op = core.tensor(descriptor, payloads)?;
    let mut sink = Vec::new();
    loop {
        match op.step()? {
            WriteStep::Output { .. } => {
                let src = op.output_src();
                let n = src.len().min(chunk);
                sink.extend_from_slice(&src[..n]);
                op.report_written(n)?;
            }
            WriteStep::ItemWritten => return Ok(sink),
        }
    }
}

/// Writes a composite through the core, one `chunk` at a time.
fn write_composite(
    head: &TensorDescriptor,
    members: &[CompositeNode<'_>],
    chunk: usize,
) -> Result<Vec<u8>, Error> {
    let mut core = StreamWriterCore::new();
    let mut op = core.composite(head, members)?;
    let mut sink = Vec::new();
    loop {
        match op.step()? {
            WriteStep::Output { .. } => {
                let src = op.output_src();
                let n = src.len().min(chunk);
                sink.extend_from_slice(&src[..n]);
                op.report_written(n)?;
            }
            WriteStep::ItemWritten => return Ok(sink),
        }
    }
}

/// Reads every item from `wire`, handing the core at most `chunk` bytes per request.
///
/// The core yields flat tensors; [`CompositeAssembler`] turns them back into items. The
/// two are separate on purpose — reassembly needs no bytes — so a harness that wants
/// structure composes them, exactly as the drivers do.
fn read_all_with(
    core: &mut StreamReaderCore,
    max_depth: usize,
    wire: &[u8],
    chunk: usize,
) -> Result<Vec<StreamItem>, Error> {
    let mut assembler = CompositeAssembler::new(max_depth);
    let mut rest = wire;
    let mut items = Vec::new();
    loop {
        match core.step()? {
            ReadStep::Input { len } => {
                let n = len.min(chunk).min(rest.len());
                core.input_dst()[..n].copy_from_slice(&rest[..n]);
                rest = &rest[n..];
                core.report_filled(n)?;
            }
            ReadStep::Tensor => {
                if let Some(item) = assembler.push(take(core))? {
                    items.push(item);
                }
            }
            ReadStep::Finished => {
                assembler.finish()?;
                return Ok(items);
            }
        }
    }
}

fn read_all(wire: &[u8], chunk: usize) -> Result<Vec<StreamItem>, Error> {
    read_all_with(
        &mut StreamReaderCore::new(),
        DEFAULT_MAX_COMPOSITE_DEPTH,
        wire,
        chunk,
    )
}

#[track_caller]
fn take(core: &mut StreamReaderCore) -> StreamTensor {
    core.take_tensor().expect("the core announced a tensor")
}

#[track_caller]
fn expect_tensor(item: &StreamItem) -> (&TensorDescriptor, &[Bytes]) {
    match item {
        StreamItem::Tensor(t) => (&t.descriptor, &t.buffers),
        StreamItem::Composite(_) => panic!("expected a tensor, got a composite"),
    }
}

// ── § 4: partial progress never loses position ────────────────────────────────

#[test]
fn a_short_read_is_not_eof() {
    let descriptor = row_major(MIN_BUFFER_ALIGNMENT);
    let payload: Vec<u8> = (0..256).map(|i| i as u8).collect();
    let buffers: [&[u8]; 1] = [&payload];
    let wire = write_tensor(&descriptor, &buffers, usize::MAX).unwrap();

    // One byte per request is the worst case the contract has to survive.
    for chunk in [1usize, 3, 7, 64, usize::MAX] {
        let items = read_all(&wire, chunk).unwrap();
        assert_eq!(items.len(), 1, "chunk {chunk}");
        let (_, read) = expect_tensor(&items[0]);
        assert_eq!(read[0].as_ref(), &payload[..], "chunk {chunk}");
    }
}

#[test]
fn zero_read_at_an_item_boundary_is_finished() {
    let items = read_all(&[], 1).unwrap();
    assert!(items.is_empty(), "an empty stream ends cleanly");

    let descriptor = row_major(MIN_BUFFER_ALIGNMENT);
    let payload = vec![0x11u8; 256];
    let buffers: [&[u8]; 1] = [&payload];
    let wire = write_tensor(&descriptor, &buffers, usize::MAX).unwrap();
    assert_eq!(read_all(&wire, 1).unwrap().len(), 1);
}

#[test]
fn zero_read_mid_item_is_an_error() {
    let descriptor = row_major(MIN_BUFFER_ALIGNMENT);
    let payload = vec![0x22u8; 256];
    let buffers: [&[u8]; 1] = [&payload];
    let wire = write_tensor(&descriptor, &buffers, usize::MAX).unwrap();

    // Mid-descriptor, mid-buffer, and one byte short of complete.
    for cut in [1usize, 5, wire.len() - 200, wire.len() - 1] {
        let err = read_all(&wire[..cut], 8).expect_err("truncated stream must not succeed");
        assert!(
            matches!(err, Error::UnexpectedEof),
            "cut {cut}: expected UnexpectedEof, got {err:?}"
        );
    }
}

#[test]
fn an_error_leaves_the_core_failed_by_name() {
    let descriptor = row_major(MIN_BUFFER_ALIGNMENT);
    let payload = vec![0x33u8; 256];
    let buffers: [&[u8]; 1] = [&payload];
    let wire = write_tensor(&descriptor, &buffers, usize::MAX).unwrap();

    let mut core = StreamReaderCore::new();
    assert!(read_all_with(&mut core, DEFAULT_MAX_COMPOSITE_DEPTH, &wire[..20], 8).is_err());
    // Not CoreUnusable: the position was known, the stream was simply truncated.
    assert!(!core.is_unusable());
    assert!(matches!(core.step(), Err(Error::CoreFailed)));
    assert!(matches!(core.report_filled(1), Err(Error::CoreFailed)));
}

#[test]
fn over_reporting_a_read_makes_the_core_unusable() {
    let mut core = StreamReaderCore::new();
    let ReadStep::Input { len } = core.step().unwrap() else {
        panic!("a fresh core wants input")
    };

    let err = core
        .report_filled(len + 1)
        .expect_err("over-report must fail");
    assert!(matches!(
        err,
        Error::InvalidProgressReport { reported, offered } if reported == len + 1 && offered == len
    ));
    // § 6: bytes moved that the core cannot place, so the position is gone — by name.
    assert!(core.is_unusable());
    assert!(matches!(
        core.step(),
        Err(Error::CoreUnusable(Unusable::LostReadCount))
    ));
}

#[test]
fn a_short_write_re_offers_the_remainder() {
    let descriptor = row_major(MIN_BUFFER_ALIGNMENT);
    let payload: Vec<u8> = (0..256).map(|i| i as u8).collect();
    let buffers: [&[u8]; 1] = [&payload];

    let whole = write_tensor(&descriptor, &buffers, usize::MAX).unwrap();
    for chunk in [1usize, 3, 7, 64] {
        assert_eq!(
            write_tensor(&descriptor, &buffers, chunk).unwrap(),
            whole,
            "chunk {chunk} produced different bytes"
        );
    }
    assert_eq!(whole.len(), descriptor.encode().unwrap().len() + 256);
}

#[test]
fn a_zero_length_write_is_not_progress() {
    let descriptor = row_major(MIN_BUFFER_ALIGNMENT);
    let payload = vec![0x44u8; 256];
    let buffers: [&[u8]; 1] = [&payload];

    let mut core = StreamWriterCore::new();
    let mut op = core.tensor(&descriptor, &buffers).unwrap();
    let WriteStep::Output { len } = op.step().unwrap() else {
        panic!("a fresh operation has bytes to write")
    };

    assert!(matches!(op.report_written(0), Err(Error::ZeroLengthWrite)));
    // Nothing moved, so the position is intact and the same bytes are re-offered.
    assert_eq!(op.step().unwrap(), WriteStep::Output { len });
    assert!(!core.is_unusable());
}

#[test]
fn a_closed_sink_mid_item_poisons_the_core() {
    let descriptor = row_major(MIN_BUFFER_ALIGNMENT);
    let payload = vec![0x55u8; 256];
    let buffers: [&[u8]; 1] = [&payload];

    let mut core = StreamWriterCore::new();
    let mut op = core.tensor(&descriptor, &buffers).unwrap();
    op.step().unwrap();
    assert!(matches!(op.report_sink_closed(), Error::SinkClosed));
    drop(op);

    // § 6a: a truncated item is on the wire and cannot be un-sent, so a fresh writer over
    // the same sink is not a recovery.
    assert!(core.is_unusable());
    assert!(matches!(
        core.tensor(&descriptor, &buffers),
        Err(Error::CoreUnusable(Unusable::LostWriteCount))
    ));
}

#[test]
fn over_reporting_a_write_poisons_the_core() {
    let descriptor = row_major(MIN_BUFFER_ALIGNMENT);
    let payload = vec![0x66u8; 256];
    let buffers: [&[u8]; 1] = [&payload];

    let mut core = StreamWriterCore::new();
    let mut op = core.tensor(&descriptor, &buffers).unwrap();
    let WriteStep::Output { len } = op.step().unwrap() else {
        panic!("a fresh operation has bytes to write")
    };
    assert!(op.report_written(len + 1).is_err());
    drop(op);
    assert!(core.is_unusable());
}

// ── § 2: the core owns the destination, and it is the only copy ───────────────

#[test]
fn payload_bytes_are_never_copied_after_the_transport_writes_them() {
    let descriptor = row_major(MIN_BUFFER_ALIGNMENT);
    let payload = vec![0x77u8; 256];
    let buffers: [&[u8]; 1] = [&payload];
    let wire = write_tensor(&descriptor, &buffers, usize::MAX).unwrap();

    let mut core = StreamReaderCore::new();
    let mut rest = &wire[..];
    let mut payload_dst: Option<usize> = None;
    let mut tensor = None;
    loop {
        match core.step().unwrap() {
            ReadStep::Input { len } => {
                let dst = core.input_dst();
                // The 256-byte request is the payload's: remember where the core put it.
                if len == 256 {
                    payload_dst = Some(dst.as_ptr() as usize);
                }
                let n = len.min(rest.len());
                dst[..n].copy_from_slice(&rest[..n]);
                rest = &rest[n..];
                core.report_filled(n).unwrap();
            }
            ReadStep::Tensor => tensor = core.take_tensor(),
            ReadStep::Finished => break,
        }
    }

    let read = tensor.expect("one tensor").buffers;
    assert_eq!(
        read[0].as_ptr() as usize,
        payload_dst.expect("the core offered a 256-byte destination"),
        "the buffer handed out is the destination the transport filled, not a copy of it"
    );
}

#[test]
fn a_partially_filled_destination_is_never_observable() {
    let descriptor = row_major(MIN_BUFFER_ALIGNMENT);
    let payload = vec![0x88u8; 256];
    let buffers: [&[u8]; 1] = [&payload];
    let wire = write_tensor(&descriptor, &buffers, usize::MAX).unwrap();

    let mut core = StreamReaderCore::new();
    let mut rest = &wire[..];
    // Stop one byte short of the item.
    while rest.len() > 1 {
        match core.step().unwrap() {
            ReadStep::Input { len } => {
                let n = len.min(rest.len() - 1);
                core.input_dst()[..n].copy_from_slice(&rest[..n]);
                rest = &rest[n..];
                core.report_filled(n).unwrap();
            }
            step => panic!("expected input, got {step:?}"),
        }
    }
    assert!(
        core.take_tensor().is_none(),
        "no tensor is observable until it is complete"
    );
}

// ── § 5: the destination carries the declared alignment ───────────────────────

#[test]
fn destinations_satisfy_the_declared_alignment() {
    for alignment in [64u32, 128, 4096] {
        let descriptor = row_major(alignment);
        let payload = vec![0x99u8; 256];
        let buffers: [&[u8]; 1] = [&payload];
        let wire = write_tensor(&descriptor, &buffers, usize::MAX).unwrap();

        let items = read_all(&wire, 17).unwrap();
        let (_, read) = expect_tensor(&items[0]);
        assert_eq!(
            read[0].as_ptr() as usize % alignment as usize,
            0,
            "declared {alignment} not honoured"
        );
    }
}

#[test]
fn multi_buffer_destinations_are_aligned_per_buffer() {
    let descriptor = coo(4096, 128);
    let values = vec![0xAAu8; 32];
    let indices = vec![0xBBu8; 128];
    let buffers: [&[u8]; 2] = [&values, &indices];
    let wire = write_tensor(&descriptor, &buffers, 5).unwrap();

    let items = read_all(&wire, 13).unwrap();
    let (desc, read) = expect_tensor(&items[0]);
    for (handle, data) in desc.buffers.iter().zip(read) {
        assert_eq!(data.as_ptr() as usize % handle.alignment() as usize, 0);
    }
    assert_eq!(read[0].as_ref(), &values[..]);
    assert_eq!(read[1].as_ref(), &indices[..]);
}

/// Records every request, then defers to the system allocator.
struct Recording {
    seen: std::sync::Arc<std::sync::Mutex<Vec<DestinationRequest>>>,
}

impl DestinationAlloc for Recording {
    fn allocate(&mut self, request: DestinationRequest) -> hurray_io::Result<Box<dyn Destination>> {
        if let Ok(mut seen) = self.seen.lock() {
            seen.push(request);
        }
        SystemDestinations.allocate(request)
    }
}

#[test]
fn a_binding_may_supply_the_destinations() {
    let descriptor = coo(4096, 128);
    let values = vec![0xCCu8; 32];
    let indices = vec![0xDDu8; 128];
    let buffers: [&[u8]; 2] = [&values, &indices];
    let wire = write_tensor(&descriptor, &buffers, usize::MAX).unwrap();

    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut core = StreamReaderCore::with_destinations(
        StreamReaderOptions::default(),
        Box::new(Recording {
            seen: std::sync::Arc::clone(&seen),
        }),
    );
    let items = read_all_with(&mut core, DEFAULT_MAX_COMPOSITE_DEPTH, &wire, usize::MAX).unwrap();
    assert_eq!(items.len(), 1);

    // The core asked for exactly what the buffer table declares — which is what makes a
    // NumPy-backed destination possible without the state machine knowing (§ 5).
    let requests = seen.lock().unwrap().clone();
    assert_eq!(
        requests,
        vec![
            DestinationRequest {
                index: 0,
                byte_size: 32,
                alignment: 4096
            },
            DestinationRequest {
                index: 1,
                byte_size: 128,
                alignment: 128
            },
        ]
    );
}

// ── § 6: resumability, and composites across a suspension ─────────────────────

#[test]
fn the_core_resumes_where_a_stopped_driver_left_it() {
    let descriptor = row_major(MIN_BUFFER_ALIGNMENT);
    let payload: Vec<u8> = (0..256).map(|i| i as u8).collect();
    let buffers: [&[u8]; 1] = [&payload];
    let wire = write_tensor(&descriptor, &buffers, usize::MAX).unwrap();

    // Drive with one loop, abandon it mid-tensor, then finish with a *different* loop
    // over the same core: a cancelled await is exactly this.
    let mut core = StreamReaderCore::new();
    let cut = wire.len() / 2;
    let mut rest = &wire[..cut];
    while !rest.is_empty() {
        match core.step().unwrap() {
            ReadStep::Input { len } => {
                let n = len.min(rest.len());
                core.input_dst()[..n].copy_from_slice(&rest[..n]);
                rest = &rest[n..];
                core.report_filled(n).unwrap();
            }
            step => panic!("expected input, got {step:?}"),
        }
    }

    let items = read_all_with(&mut core, DEFAULT_MAX_COMPOSITE_DEPTH, &wire[cut..], 9).unwrap();
    assert_eq!(items.len(), 1);
    let (_, read) = expect_tensor(&items[0]);
    assert_eq!(read[0].as_ref(), &payload[..]);
}

#[test]
fn a_nested_composite_survives_being_suspended_mid_member() {
    // A group whose second member is a partition composite: two levels, assembled from a
    // one-byte-at-a-time feed, with the depth checked as it goes.
    let inner_head = composite_head(CompositionRule::Partition, 2, vec![8, 8]);
    let (m0, m1) = (partition_member(0, 4096), partition_member(4, 128));
    let d0 = vec![0xA1u8; 128];
    let d1 = vec![0xB2u8; 128];
    let b0: [&[u8]; 1] = [&d0];
    let b1: [&[u8]; 1] = [&d1];
    let inner_members = [
        CompositeNode::Tensor {
            descriptor: &m0,
            buffers: &b0,
        },
        CompositeNode::Tensor {
            descriptor: &m1,
            buffers: &b1,
        },
    ];

    let plain = row_major(4096);
    let plain_data = vec![0xC3u8; 256];
    let plain_buffers: [&[u8]; 1] = [&plain_data];

    let outer_head = composite_head(CompositionRule::Group, 2, vec![8, 8]);
    let outer_members = [
        CompositeNode::Tensor {
            descriptor: &plain,
            buffers: &plain_buffers,
        },
        CompositeNode::Composite {
            head: &inner_head,
            members: &inner_members,
        },
    ];

    let wire = write_composite(&outer_head, &outer_members, 1).unwrap();

    let mut core = StreamReaderCore::new();
    let mut assembler = CompositeAssembler::new(DEFAULT_MAX_COMPOSITE_DEPTH);
    let mut rest = &wire[..];
    let mut depths = Vec::new();
    let mut items = Vec::new();
    loop {
        match core.step().unwrap() {
            ReadStep::Input { len } => {
                let n = len.min(1).min(rest.len());
                core.input_dst()[..n].copy_from_slice(&rest[..n]);
                rest = &rest[n..];
                core.report_filled(n).unwrap();
                depths.push(assembler.depth());
            }
            ReadStep::Tensor => {
                if let Some(item) = assembler.push(take(&mut core)).unwrap() {
                    items.push(item);
                }
            }
            ReadStep::Finished => {
                assembler.finish().unwrap();
                break;
            }
        }
    }

    assert_eq!(items.len(), 1);
    let StreamItem::Composite(outer) = &items[0] else {
        panic!("expected a composite")
    };
    assert_eq!(outer.members.len(), 2);
    let StreamItem::Composite(inner) = &outer.members[1] else {
        panic!("expected a nested composite")
    };
    assert_eq!(inner.members.len(), 2);
    for (member, expected) in inner.members.iter().zip([&d0, &d1]) {
        let (desc, read) = expect_tensor(member);
        assert_eq!(read[0].as_ref(), &expected[..]);
        assert_eq!(
            read[0].as_ptr() as usize % desc.buffers[0].alignment() as usize,
            0,
            "a member's destination keeps its own declared alignment"
        );
    }

    // The stack was two deep while the nested composite's members were arriving, and
    // unwound to nothing at the end (§ 10).
    assert_eq!(depths.iter().copied().max(), Some(2));
    assert_eq!(depths.last().copied(), Some(0));
}

#[test]
fn a_composite_cut_short_is_torn_not_finished() {
    let head = composite_head(CompositionRule::Partition, 2, vec![8, 8]);
    let (m0, m1) = (partition_member(0, 64), partition_member(4, 64));
    let d0 = vec![0xE1u8; 128];
    let d1 = vec![0xE2u8; 128];
    let b0: [&[u8]; 1] = [&d0];
    let b1: [&[u8]; 1] = [&d1];
    let members = [
        CompositeNode::Tensor {
            descriptor: &m0,
            buffers: &b0,
        },
        CompositeNode::Tensor {
            descriptor: &m1,
            buffers: &b1,
        },
    ];
    let wire = write_composite(&head, &members, usize::MAX).unwrap();

    // Cut after the head and its first member: end of input at a *member* boundary, which
    // is an item boundary for the frame but not for the composite.
    let first_member_end = head.encode().unwrap().len() + m0.encode().unwrap().len() + 128;
    let err = read_all(&wire[..first_member_end], usize::MAX).expect_err("must not succeed");
    assert!(
        matches!(
            err,
            Error::TornComposite {
                declared: 2,
                actual: 1
            }
        ),
        "got {err:?}"
    );
}

// ── limits and modes the core now owns ───────────────────────────────────────

/// A CUDA buffer synchronised by an event: valid locally, meaningless across machines.
fn device_tensor() -> TensorDescriptor {
    let handle = BufferHandle::new(256, 256, DeviceTag::Cuda, SyncMode::Event).unwrap();
    TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(vec![8u64, 8]).unwrap(),
        0,
        LayoutDescriptor::RowMajor,
        vec![handle],
        None,
        None,
        None,
        None,
    )
    .unwrap()
}

#[test]
fn a_cross_machine_reader_rejects_a_device_synced_buffer() {
    let descriptor = device_tensor();
    let payload = vec![0x5Cu8; 256];
    let buffers: [&[u8]; 1] = [&payload];
    let wire = write_tensor(&descriptor, &buffers, usize::MAX).unwrap();

    // The same bytes read in local mode are fine.
    assert_eq!(read_all(&wire, usize::MAX).unwrap().len(), 1);

    let mut core = StreamReaderCore::with_options(StreamReaderOptions {
        enforce_cross_machine_sync: true,
        ..Default::default()
    });
    let err = read_all_with(&mut core, DEFAULT_MAX_COMPOSITE_DEPTH, &wire, usize::MAX)
        .expect_err("must be rejected");
    assert!(
        matches!(err, Error::InvalidCrossMachineSyncMode { index: 0, .. }),
        "got {err:?}"
    );
}

#[test]
fn a_cross_machine_writer_refuses_before_writing_a_byte() {
    let descriptor = device_tensor();
    let payload = vec![0x5Du8; 256];
    let buffers: [&[u8]; 1] = [&payload];

    let mut core = StreamWriterCore::cross_machine();
    let err = core
        .tensor(&descriptor, &buffers)
        .err()
        .expect("must be rejected");
    assert!(
        matches!(err, Error::InvalidCrossMachineSyncMode { index: 0, .. }),
        "got {err:?}"
    );
    // Rejected at submission, so the core is still good for a valid tensor.
    assert!(!core.is_unusable());
}

#[test]
fn the_configured_limits_are_enforced() {
    let descriptor = row_major(MIN_BUFFER_ALIGNMENT);
    let payload = vec![0x5Eu8; 256];
    let buffers: [&[u8]; 1] = [&payload];
    let wire = write_tensor(&descriptor, &buffers, usize::MAX).unwrap();

    let mut tight_buffers = StreamReaderCore::with_options(StreamReaderOptions {
        max_buffer_bytes: 255,
        ..Default::default()
    });
    assert!(matches!(
        read_all_with(
            &mut tight_buffers,
            DEFAULT_MAX_COMPOSITE_DEPTH,
            &wire,
            usize::MAX
        ),
        Err(Error::FrameTooLarge { kind: "buffer", .. })
    ));

    let mut tight_descriptors = StreamReaderCore::with_options(StreamReaderOptions {
        max_descriptor_bytes: 16,
        ..Default::default()
    });
    assert!(matches!(
        read_all_with(
            &mut tight_descriptors,
            DEFAULT_MAX_COMPOSITE_DEPTH,
            &wire,
            usize::MAX
        ),
        Err(Error::FrameTooLarge {
            kind: "descriptor",
            ..
        })
    ));
}

#[test]
fn composite_nesting_is_bounded() {
    let inner_head = composite_head(CompositionRule::Group, 1, vec![8, 8]);
    let plain = row_major(MIN_BUFFER_ALIGNMENT);
    let data = vec![0x5Fu8; 256];
    let plain_buffers: [&[u8]; 1] = [&data];
    let inner_members = [CompositeNode::Tensor {
        descriptor: &plain,
        buffers: &plain_buffers,
    }];
    let outer_head = composite_head(CompositionRule::Group, 1, vec![8, 8]);
    let outer_members = [CompositeNode::Composite {
        head: &inner_head,
        members: &inner_members,
    }];
    let wire = write_composite(&outer_head, &outer_members, usize::MAX).unwrap();

    // Depth 1 admits the outer head and rejects the nested one: the same bound the
    // recursive reader enforced against its call depth (§ 10).
    let mut core = StreamReaderCore::new();
    assert!(matches!(
        read_all_with(&mut core, 1, &wire, usize::MAX),
        Err(Error::CompositeNestingTooDeep { limit: 1 })
    ));

    let mut deeper = StreamReaderCore::new();
    assert_eq!(
        read_all_with(&mut deeper, 2, &wire, usize::MAX)
            .unwrap()
            .len(),
        1
    );
}

// ── the core agrees with the shipped driver, byte for byte ────────────────────
//
// There is nothing left to assert here: the tokio drivers *are* this core now, so a
// comparison between them would pass by construction. The wire-compatibility check moved
// to `wire_compat.rs`, against bytes frozen before the refactor.
