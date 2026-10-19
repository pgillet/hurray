#![cfg(feature = "tokio")]

//! The stream wire format, frozen.
//!
//! ADR-041 § 11 step 2 asks for byte-for-byte wire compatibility across the driver
//! refactor. A test that compares the core against the tokio driver cannot show that once
//! the driver *is* the core — both sides become the same code and the assertion passes by
//! construction. So the bytes here are frozen artifacts: produced by the 0.1.0 drivers
//! before the refactor, checked in, and only ever read.
//!
//! See `fixtures/README.md` for their provenance.

use hurray_core::{
    buffer_size_bytes,
    layout::{CompositeLayout, CompositionRule, CooLayout},
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, PerBlockAffine, QuantizationDescriptor,
    Shape, ShardDescriptor, SyncMode, TensorDescriptor, MIN_BUFFER_ALIGNMENT,
};
use hurray_io::stream::{CompositeNode, StreamItem, StreamReader, StreamWriter};

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

/// Deterministic payload: the corpus must not shift when the test is re-run.
fn payload(len: usize, salt: u8) -> Vec<u8> {
    (0..len).map(|i| (i % 251) as u8 ^ salt).collect()
}

/// An `[8, 8]` float32 row-major tensor: one 256-byte buffer.
fn row_major() -> TensorDescriptor {
    TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(vec![8u64, 8]).unwrap(),
        0,
        LayoutDescriptor::RowMajor,
        vec![buf(256, MIN_BUFFER_ALIGNMENT)],
        None,
        None,
        None,
        None,
    )
    .unwrap()
}

/// A COO tensor whose two buffers declare different alignments (128 and 4096).
fn coo() -> TensorDescriptor {
    let nnz = 8u64;
    TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(vec![8u64, 8]).unwrap(),
        0,
        LayoutDescriptor::Coo(CooLayout::new(nnz, false)),
        vec![buf(32, 128), buf(nnz * 2 * 8, 4096)],
        None,
        None,
        None,
        None,
    )
    .unwrap()
}

/// An int4 tensor carrying a per-block-affine quantization section and a shard section,
/// so the corpus covers a descriptor with optional sections present.
fn quantized() -> TensorDescriptor {
    let shape = vec![256u64, 256];
    let pba = PerBlockAffine::new_symmetric(1, 128, 1, ElementType::Float16).unwrap();
    let num_blocks = pba.num_blocks_per_axis(256) * 256;
    let quant = QuantizationDescriptor::PerBlockAffine(pba);
    TensorDescriptor::new(
        1,
        0,
        ElementType::Int4,
        Shape::new(shape.clone()).unwrap(),
        0,
        LayoutDescriptor::RowMajor,
        vec![
            buf(
                buffer_size_bytes(ElementType::Int4, 256 * 256),
                MIN_BUFFER_ALIGNMENT,
            ),
            buf(
                buffer_size_bytes(ElementType::Float16, num_blocks),
                MIN_BUFFER_ALIGNMENT,
            ),
        ],
        Some(quant.encode_to_vec()),
        Some(ShardDescriptor::new(shape, vec![0, 0]).unwrap()),
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

/// A partition member: an `[8, 4]` float32 tile at column `offset` of the `[8, 8]` parent.
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
            MIN_BUFFER_ALIGNMENT,
        )],
        None,
        Some(ShardDescriptor::new(vec![8, 8], vec![0, offset]).unwrap()),
        None,
        None,
    )
    .unwrap()
}

/// A plain `[2, 2]` float32 tensor; groups impose no shard requirement.
fn plain_2x2() -> TensorDescriptor {
    TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(vec![2u64, 2]).unwrap(),
        0,
        LayoutDescriptor::RowMajor,
        vec![buf(
            buffer_size_bytes(ElementType::Float32, 4),
            MIN_BUFFER_ALIGNMENT,
        )],
        None,
        None,
        None,
        None,
    )
    .unwrap()
}

// ── the five streams, each built only from the public writer API ──────────────

async fn write_row_major() -> Vec<u8> {
    let desc = row_major();
    let data = payload(256, 0x00);
    let buffers: [&[u8]; 1] = [&data];

    let mut wire = Vec::new();
    let mut writer = StreamWriter::new(&mut wire);
    writer.write_tensor(&desc, &buffers).await.unwrap();
    writer.finish().await.unwrap();
    wire
}

async fn write_coo() -> Vec<u8> {
    let desc = coo();
    let values = payload(32, 0x11);
    let indices = payload(128, 0x22);
    let buffers: [&[u8]; 2] = [&values, &indices];

    let mut wire = Vec::new();
    let mut writer = StreamWriter::new(&mut wire);
    writer.write_tensor(&desc, &buffers).await.unwrap();
    writer.finish().await.unwrap();
    wire
}

async fn write_quantized() -> Vec<u8> {
    let desc = quantized();
    let weights = payload(desc.buffers[0].byte_size() as usize, 0x33);
    let scales = payload(desc.buffers[1].byte_size() as usize, 0x44);
    let buffers: [&[u8]; 2] = [&weights, &scales];

    let mut wire = Vec::new();
    let mut writer = StreamWriter::new(&mut wire);
    writer.write_tensor(&desc, &buffers).await.unwrap();
    writer.finish().await.unwrap();
    wire
}

/// A group of two, whose first member is itself a partition of two: nesting depth 2.
async fn write_nested_composite() -> Vec<u8> {
    let outer = composite_head(CompositionRule::Group, 2, vec![8, 8]);
    let inner = composite_head(CompositionRule::Partition, 2, vec![8, 8]);
    let m0 = partition_member(0);
    let m1 = partition_member(4);
    let tail = plain_2x2();

    let d0 = payload(128, 0x55);
    let d1 = payload(128, 0x66);
    let dt = payload(16, 0x77);
    let b0: [&[u8]; 1] = [&d0];
    let b1: [&[u8]; 1] = [&d1];
    let bt: [&[u8]; 1] = [&dt];

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
    let outer_members = [
        CompositeNode::Composite {
            head: &inner,
            members: &inner_members,
        },
        CompositeNode::Tensor {
            descriptor: &tail,
            buffers: &bt,
        },
    ];

    let mut wire = Vec::new();
    let mut writer = StreamWriter::new(&mut wire);
    writer
        .write_composite(&outer, &outer_members)
        .await
        .unwrap();
    writer.finish().await.unwrap();
    wire
}

/// Two tensors back to back through the cross-machine writer: the wire must be identical
/// to the default writer's, since the mode only *rejects* buffers, it never re-encodes.
async fn write_cross_machine() -> Vec<u8> {
    let desc = row_major();
    let first = payload(256, 0x88);
    let second = payload(256, 0x99);
    let b1: [&[u8]; 1] = [&first];
    let b2: [&[u8]; 1] = [&second];

    let mut wire = Vec::new();
    let mut writer = StreamWriter::cross_machine(&mut wire);
    writer.write_tensor(&desc, &b1).await.unwrap();
    writer.write_tensor(&desc, &b2).await.unwrap();
    writer.finish().await.unwrap();
    wire
}

// ── the frozen bytes ────────────────────────────────────────────

#[tokio::test]
async fn row_major_matches_frozen_bytes() {
    let frozen: &[u8] = include_bytes!("fixtures/row_major.bin");
    assert_eq!(write_row_major().await, frozen);
}

#[tokio::test]
async fn coo_matches_frozen_bytes() {
    let frozen: &[u8] = include_bytes!("fixtures/coo.bin");
    assert_eq!(write_coo().await, frozen);
}

#[tokio::test]
async fn quantized_matches_frozen_bytes() {
    let frozen: &[u8] = include_bytes!("fixtures/quantized.bin");
    assert_eq!(write_quantized().await, frozen);
}

#[tokio::test]
async fn nested_composite_matches_frozen_bytes() {
    let frozen: &[u8] = include_bytes!("fixtures/nested_composite.bin");
    assert_eq!(write_nested_composite().await, frozen);
}

#[tokio::test]
async fn cross_machine_matches_frozen_bytes() {
    let frozen: &[u8] = include_bytes!("fixtures/cross_machine.bin");
    assert_eq!(write_cross_machine().await, frozen);
}

// ── the frozen bytes still decode to the same tensors ─────────────────────────

/// Compatibility runs both ways: the reader must still decode the frozen corpus, not just
/// the writer reproduce it. A reader that drifted in step with the writer would pass every
/// assertion above.
#[tokio::test]
async fn the_reader_decodes_every_frozen_stream() {
    let cases: [(&[u8], usize); 5] = [
        (include_bytes!("fixtures/row_major.bin"), 1),
        (include_bytes!("fixtures/coo.bin"), 1),
        (include_bytes!("fixtures/quantized.bin"), 1),
        // One top-level composite, whatever its nesting.
        (include_bytes!("fixtures/nested_composite.bin"), 1),
        (include_bytes!("fixtures/cross_machine.bin"), 2),
    ];

    for (wire, expected_items) in cases {
        let mut reader = StreamReader::new(wire);
        let mut count = 0;
        while reader.next_item().await.unwrap().is_some() {
            count += 1;
        }
        assert_eq!(count, expected_items, "item count from a frozen stream");
    }
}

#[tokio::test]
async fn the_frozen_coo_round_trips_its_payloads() {
    let wire: &[u8] = include_bytes!("fixtures/coo.bin");
    let mut reader = StreamReader::new(wire);
    let tensor = reader.next_tensor().await.unwrap().expect("one tensor");

    assert_eq!(tensor.descriptor, coo());
    assert_eq!(tensor.buffers.len(), 2);
    assert_eq!(tensor.buffers[0].as_ref(), payload(32, 0x11).as_slice());
    assert_eq!(tensor.buffers[1].as_ref(), payload(128, 0x22).as_slice());
    // The declared alignments (128 and 4096) are honoured on the way back in (#235).
    assert_eq!(tensor.buffers[0].as_ptr() as usize % 128, 0);
    assert_eq!(tensor.buffers[1].as_ptr() as usize % 4096, 0);
}

/// `next_tensor` is oblivious to composition: it yields the head with no buffers, then
/// each descriptor in wire order, at every nesting level.
#[tokio::test]
async fn the_frozen_nested_composite_reads_flat() {
    let wire: &[u8] = include_bytes!("fixtures/nested_composite.bin");
    let mut reader = StreamReader::new(wire);

    let mut flat = Vec::new();
    while let Some(tensor) = reader.next_tensor().await.unwrap() {
        flat.push(tensor);
    }

    // outer head, inner head, two partition members, the trailing [2, 2].
    assert_eq!(flat.len(), 5);
    assert!(flat[0].buffers.is_empty(), "outer head owns no data");
    assert!(flat[1].buffers.is_empty(), "inner head owns no data");
    assert_eq!(flat[2].buffers[0].as_ref(), payload(128, 0x55).as_slice());
    assert_eq!(flat[3].buffers[0].as_ref(), payload(128, 0x66).as_slice());
    assert_eq!(flat[4].buffers[0].as_ref(), payload(16, 0x77).as_slice());
}

/// The nested shape survives: a group whose first member is a partition of two.
#[tokio::test]
async fn the_frozen_nested_composite_reassembles() {
    let wire: &[u8] = include_bytes!("fixtures/nested_composite.bin");
    let mut reader = StreamReader::new(wire);

    let item = reader.next_item().await.unwrap().expect("one item");
    let StreamItem::Composite(outer) = item else {
        panic!("expected a composite");
    };
    assert_eq!(outer.members.len(), 2);

    let StreamItem::Composite(inner) = &outer.members[0] else {
        panic!("expected a nested composite as the first member");
    };
    assert_eq!(inner.members.len(), 2);
    assert!(matches!(&outer.members[1], StreamItem::Tensor(_)));
    assert!(reader.next_item().await.unwrap().is_none(), "clean EOF");
}
