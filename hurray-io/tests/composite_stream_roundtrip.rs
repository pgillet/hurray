#![cfg(feature = "tokio")]

//! Streaming round-trip tests for composite tensors (ADR-027 § Binding, Layer 5).
//!
//! Everything is driven through the public API: [`StreamWriter::write_composite`] and
//! [`StreamReader::next_item`], plus the low-level [`StreamWriter::write_tensor`] /
//! [`StreamReader::next_tensor`] to fabricate and observe raw wire layouts.

use hurray_core::{
    buffer_size_bytes,
    layout::{CompositeLayout, CompositionRule},
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, ShardDescriptor, SyncMode,
    TensorDescriptor, MIN_BUFFER_ALIGNMENT,
};
use hurray_io::stream::{
    CompositeNode, StreamItem, StreamReader, StreamReaderOptions, StreamWriter,
};

// ── helpers ──────────────────────────────────────────────────────────────────

fn buf(byte_size: u64) -> BufferHandle {
    BufferHandle::new(
        byte_size,
        MIN_BUFFER_ALIGNMENT,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
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
        vec![], // a composite head owns no data buffers
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
        vec![buf(buffer_size_bytes(ElementType::Float32, 8 * 4))],
        None,
        Some(ShardDescriptor::new(vec![8, 8], vec![0, offset]).unwrap()),
        None,
        None,
    )
    .unwrap()
}

/// A plain `[2, 2]` float32 tensor (used as a group member; groups need no shards).
fn plain_2x2() -> TensorDescriptor {
    TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(vec![2u64, 2]).unwrap(),
        0,
        LayoutDescriptor::RowMajor,
        vec![buf(buffer_size_bytes(ElementType::Float32, 4))],
        None,
        None,
        None,
        None,
    )
    .unwrap()
}

// ── partition round-trip ───────────────────────────────────────────────────────

#[tokio::test]
async fn roundtrip_partition_composite() {
    let head = composite_head(CompositionRule::Partition, 2, vec![8, 8]);
    let m0 = partition_member(0);
    let m1 = partition_member(4);
    let d0 = vec![0xAAu8; 128];
    let d1 = vec![0xBBu8; 128];
    let b0: [&[u8]; 1] = [d0.as_slice()];
    let b1: [&[u8]; 1] = [d1.as_slice()];

    let members = vec![
        CompositeNode::Tensor {
            descriptor: &m0,
            buffers: &b0,
        },
        CompositeNode::Tensor {
            descriptor: &m1,
            buffers: &b1,
        },
    ];

    let mut wire = Vec::<u8>::new();
    let mut writer = StreamWriter::new(&mut wire);
    writer.write_composite(&head, &members).await.unwrap();
    writer.finish().await.unwrap();

    let mut reader = StreamReader::new(wire.as_slice());
    let item = reader.next_item().await.unwrap().expect("expected an item");
    let StreamItem::Composite(c) = item else {
        panic!("expected a composite, got a plain tensor");
    };
    assert_eq!(c.head.shape.dims(), &[8, 8]);
    assert_eq!(c.members.len(), 2);
    for (member, expected) in c.members.iter().zip([&d0, &d1]) {
        let StreamItem::Tensor(t) = member else {
            panic!("partition member should be a plain tensor");
        };
        assert_eq!(t.buffers[0].as_ref(), expected.as_slice());
    }
    assert!(reader.next_item().await.unwrap().is_none(), "clean EOF");
}

// ── back-compat: next_tensor still yields head + members flat ────────────────────

#[tokio::test]
async fn next_tensor_reads_composite_as_flat_tensors() {
    let head = composite_head(CompositionRule::Partition, 2, vec![8, 8]);
    let m0 = partition_member(0);
    let m1 = partition_member(4);
    let d0 = vec![1u8; 128];
    let d1 = vec![2u8; 128];
    let b0: [&[u8]; 1] = [d0.as_slice()];
    let b1: [&[u8]; 1] = [d1.as_slice()];

    let members = vec![
        CompositeNode::Tensor {
            descriptor: &m0,
            buffers: &b0,
        },
        CompositeNode::Tensor {
            descriptor: &m1,
            buffers: &b1,
        },
    ];

    let mut wire = Vec::<u8>::new();
    StreamWriter::new(&mut wire)
        .write_composite(&head, &members)
        .await
        .unwrap();

    // The low-level reader is oblivious to composition: it returns the head (no buffers)
    // then each member, in wire order.
    let mut reader = StreamReader::new(wire.as_slice());
    let head_t = reader.next_tensor().await.unwrap().unwrap();
    assert!(head_t.buffers.is_empty(), "head owns no data");
    assert!(matches!(
        head_t.descriptor.layout,
        LayoutDescriptor::Composite(_)
    ));
    let t0 = reader.next_tensor().await.unwrap().unwrap();
    assert_eq!(t0.buffers[0].as_ref(), d0.as_slice());
    let t1 = reader.next_tensor().await.unwrap().unwrap();
    assert_eq!(t1.buffers[0].as_ref(), d1.as_slice());
    assert!(reader.next_tensor().await.unwrap().is_none());
}

// ── nested composite (recursion) ─────────────────────────────────────────────────

#[tokio::test]
async fn roundtrip_nested_group_composite() {
    // Outer group of 2: [ inner group of 2 tensors, one plain tensor ].
    let outer = composite_head(CompositionRule::Group, 2, vec![2, 2]);
    let inner = composite_head(CompositionRule::Group, 2, vec![2, 2]);

    let (a, b, c) = (plain_2x2(), plain_2x2(), plain_2x2());
    let (da, db, dc) = (vec![0x11u8; 16], vec![0x22u8; 16], vec![0x33u8; 16]);
    let ba: [&[u8]; 1] = [da.as_slice()];
    let bb: [&[u8]; 1] = [db.as_slice()];
    let bc: [&[u8]; 1] = [dc.as_slice()];

    let inner_members = vec![
        CompositeNode::Tensor {
            descriptor: &a,
            buffers: &ba,
        },
        CompositeNode::Tensor {
            descriptor: &b,
            buffers: &bb,
        },
    ];
    let outer_members = vec![
        CompositeNode::Composite {
            head: &inner,
            members: &inner_members,
        },
        CompositeNode::Tensor {
            descriptor: &c,
            buffers: &bc,
        },
    ];

    let mut wire = Vec::<u8>::new();
    let mut writer = StreamWriter::new(&mut wire);
    writer
        .write_composite(&outer, &outer_members)
        .await
        .unwrap();
    writer.finish().await.unwrap();

    let mut reader = StreamReader::new(wire.as_slice());
    let StreamItem::Composite(top) = reader.next_item().await.unwrap().unwrap() else {
        panic!("expected outer composite");
    };
    assert_eq!(top.members.len(), 2);

    let StreamItem::Composite(nested) = &top.members[0] else {
        panic!("first member should itself be a composite");
    };
    assert_eq!(nested.members.len(), 2);
    let StreamItem::Tensor(inner_a) = &nested.members[0] else {
        panic!("nested member 0 should be a tensor");
    };
    assert_eq!(inner_a.buffers[0].as_ref(), da.as_slice());

    let StreamItem::Tensor(outer_c) = &top.members[1] else {
        panic!("second outer member should be a tensor");
    };
    assert_eq!(outer_c.buffers[0].as_ref(), dc.as_slice());
    assert!(reader.next_item().await.unwrap().is_none());
}

// ── torn composite ───────────────────────────────────────────────────────────────

#[tokio::test]
async fn next_item_errors_on_torn_composite() {
    // Fabricate a torn composite with the low-level writer: a head declaring 3 members but
    // only 2 actually written. write_composite would reject this, so we bypass it.
    let head = composite_head(CompositionRule::Group, 3, vec![2, 2]);
    let (a, b) = (plain_2x2(), plain_2x2());
    let data = vec![0u8; 16];

    let mut wire = Vec::<u8>::new();
    let mut writer = StreamWriter::new(&mut wire);
    writer.write_tensor(&head, &[]).await.unwrap();
    writer.write_tensor(&a, &[data.as_slice()]).await.unwrap();
    writer.write_tensor(&b, &[data.as_slice()]).await.unwrap();
    writer.finish().await.unwrap();

    let mut reader = StreamReader::new(wire.as_slice());
    let err = reader.next_item().await.unwrap_err();
    assert!(
        matches!(
            err,
            hurray_io::Error::TornComposite {
                declared: 3,
                actual: 2
            }
        ),
        "unexpected error: {err}"
    );
}

// ── writer validation ────────────────────────────────────────────────────────────

#[tokio::test]
async fn write_composite_rejects_member_count_mismatch() {
    // Head declares 2 members, 3 supplied → validation fails before any byte is written.
    let head = composite_head(CompositionRule::Group, 2, vec![2, 2]);
    let (a, b, c) = (plain_2x2(), plain_2x2(), plain_2x2());
    let data = vec![0u8; 16];
    let bd: [&[u8]; 1] = [data.as_slice()];
    let members = vec![
        CompositeNode::Tensor {
            descriptor: &a,
            buffers: &bd,
        },
        CompositeNode::Tensor {
            descriptor: &b,
            buffers: &bd,
        },
        CompositeNode::Tensor {
            descriptor: &c,
            buffers: &bd,
        },
    ];

    let mut wire = Vec::<u8>::new();
    let mut writer = StreamWriter::new(&mut wire);
    let err = writer.write_composite(&head, &members).await.unwrap_err();
    assert!(
        matches!(err, hurray_io::Error::Core(_)),
        "unexpected: {err}"
    );
    assert!(
        wire.is_empty(),
        "nothing should be written on validation failure"
    );
}

// ── nesting depth guard ──────────────────────────────────────────────────────────

#[tokio::test]
async fn next_item_enforces_max_composite_depth() {
    let outer = composite_head(CompositionRule::Group, 1, vec![2, 2]);
    let inner = composite_head(CompositionRule::Group, 1, vec![2, 2]);
    let a = plain_2x2();
    let data = vec![0u8; 16];
    let bd: [&[u8]; 1] = [data.as_slice()];

    let inner_members = vec![CompositeNode::Tensor {
        descriptor: &a,
        buffers: &bd,
    }];
    let outer_members = vec![CompositeNode::Composite {
        head: &inner,
        members: &inner_members,
    }];

    let mut wire = Vec::<u8>::new();
    let mut writer = StreamWriter::new(&mut wire);
    writer
        .write_composite(&outer, &outer_members)
        .await
        .unwrap();
    writer.finish().await.unwrap();

    // Limit 1: the outer composite (depth 0) is fine, but its nested member (depth 1) trips.
    let options = StreamReaderOptions {
        max_composite_depth: 1,
        ..Default::default()
    };
    let mut reader = StreamReader::with_options(wire.as_slice(), options);
    let err = reader.next_item().await.unwrap_err();
    assert!(
        matches!(err, hurray_io::Error::CompositeNestingTooDeep { limit: 1 }),
        "unexpected error: {err}"
    );
}
