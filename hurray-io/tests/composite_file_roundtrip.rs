#![cfg(feature = "tokio")]

//! File-format round-trip tests for composite tensors (ADR-027 § Binding, Layer 6).
//!
//! Driven through the public API: [`FileWriter::write_composite`] and
//! [`FileReader::read_composite`], plus [`FileWriter::write_tensor`] to fabricate a torn
//! composite the safe API would reject.

use std::io::Cursor;

use hurray_core::{
    buffer_size_bytes,
    layout::{CompositeLayout, CompositionRule},
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, ShardDescriptor, SyncMode,
    TensorDescriptor, MIN_BUFFER_ALIGNMENT,
};
use hurray_io::file::{FileCompositeNode, FileItem, FileReader, FileWriter, FileWriterOptions};

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
        vec![buf(buffer_size_bytes(ElementType::Float32, 8 * 4))],
        None,
        Some(ShardDescriptor::new(vec![8, 8], vec![0, offset]).unwrap()),
        None,
        None,
    )
    .unwrap()
}

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
async fn roundtrip_partition_composite_file() {
    let head = composite_head(CompositionRule::Partition, 2, vec![8, 8]);
    let (m0, m1) = (partition_member(0), partition_member(4));
    let d0 = vec![0xAAu8; 128];
    let d1 = vec![0xBBu8; 128];
    let b0: [&[u8]; 1] = [d0.as_slice()];
    let b1: [&[u8]; 1] = [d1.as_slice()];

    let members = vec![
        FileCompositeNode::Tensor {
            name: "weight.tile0",
            descriptor: &m0,
            buffers: &b0,
        },
        FileCompositeNode::Tensor {
            name: "weight.tile1",
            descriptor: &m1,
            buffers: &b1,
        },
    ];

    let out = Vec::<u8>::new();
    let mut writer = FileWriter::new(out).await.unwrap();
    writer
        .write_composite("weight", &head, &members)
        .await
        .unwrap();
    let wire = writer.finish(vec![]).await.unwrap();

    let mut reader = FileReader::open(Cursor::new(wire)).await.unwrap();

    // Every tensor is individually addressable by name.
    assert_eq!(
        reader.tensor_names().collect::<Vec<_>>(),
        ["weight", "weight.tile0", "weight.tile1"]
    );
    let head_only = reader.read_tensor("weight").await.unwrap();
    assert!(head_only.buffers.is_empty(), "head owns no data");
    let tile0 = reader.read_tensor("weight.tile0").await.unwrap();
    assert_eq!(tile0.buffers[0].as_ref(), d0.as_slice());

    // And the whole composite reassembles.
    let composite = reader.read_composite("weight").await.unwrap();
    assert_eq!(composite.name, "weight");
    assert_eq!(composite.head.shape.dims(), &[8, 8]);
    assert_eq!(composite.members.len(), 2);
    for (member, expected) in composite.members.iter().zip([&d0, &d1]) {
        let FileItem::Tensor(t) = member else {
            panic!("partition member should be a plain tensor");
        };
        assert_eq!(t.buffers[0].as_ref(), expected.as_slice());
    }
}

// ── recovery is independent of index sort order ──────────────────────────────────

#[tokio::test]
async fn read_composite_recovers_under_sorted_index() {
    // Head name sorts *after* its members, so a sorted index reorders the array — yet
    // membership recovery keys on file offset, not array position.
    let head = composite_head(CompositionRule::Partition, 2, vec![8, 8]);
    let (m0, m1) = (partition_member(0), partition_member(4));
    let d0 = vec![1u8; 128];
    let d1 = vec![2u8; 128];
    let b0: [&[u8]; 1] = [d0.as_slice()];
    let b1: [&[u8]; 1] = [d1.as_slice()];

    let members = vec![
        FileCompositeNode::Tensor {
            name: "aaa",
            descriptor: &m0,
            buffers: &b0,
        },
        FileCompositeNode::Tensor {
            name: "bbb",
            descriptor: &m1,
            buffers: &b1,
        },
    ];

    let options = FileWriterOptions {
        sorted_index: true,
        ..Default::default()
    };
    let out = Vec::<u8>::new();
    let mut writer = FileWriter::with_options(out, options).await.unwrap();
    writer
        .write_composite("zzz", &head, &members)
        .await
        .unwrap();
    let wire = writer.finish(vec![]).await.unwrap();

    let mut reader = FileReader::open(Cursor::new(wire)).await.unwrap();
    // Index array is alphabetical: aaa, bbb, zzz.
    assert_eq!(
        reader.tensor_names().collect::<Vec<_>>(),
        ["aaa", "bbb", "zzz"]
    );

    let composite = reader.read_composite("zzz").await.unwrap();
    assert_eq!(composite.members.len(), 2);
    let FileItem::Tensor(first) = &composite.members[0] else {
        panic!("expected a tensor");
    };
    // First member by *file order* is aaa, regardless of the sorted index.
    assert_eq!(first.name, "aaa");
    assert_eq!(first.buffers[0].as_ref(), d0.as_slice());
}

// ── nested composite (recursion) ─────────────────────────────────────────────────

#[tokio::test]
async fn roundtrip_nested_group_composite_file() {
    let outer = composite_head(CompositionRule::Group, 2, vec![2, 2]);
    let inner = composite_head(CompositionRule::Group, 2, vec![2, 2]);
    let (a, b, c) = (plain_2x2(), plain_2x2(), plain_2x2());
    let (da, db, dc) = (vec![0x11u8; 16], vec![0x22u8; 16], vec![0x33u8; 16]);
    let ba: [&[u8]; 1] = [da.as_slice()];
    let bb: [&[u8]; 1] = [db.as_slice()];
    let bc: [&[u8]; 1] = [dc.as_slice()];

    let inner_members = vec![
        FileCompositeNode::Tensor {
            name: "g.i.a",
            descriptor: &a,
            buffers: &ba,
        },
        FileCompositeNode::Tensor {
            name: "g.i.b",
            descriptor: &b,
            buffers: &bb,
        },
    ];
    let outer_members = vec![
        FileCompositeNode::Composite {
            name: "g.i",
            head: &inner,
            members: &inner_members,
        },
        FileCompositeNode::Tensor {
            name: "g.c",
            descriptor: &c,
            buffers: &bc,
        },
    ];

    let out = Vec::<u8>::new();
    let mut writer = FileWriter::new(out).await.unwrap();
    writer
        .write_composite("g", &outer, &outer_members)
        .await
        .unwrap();
    let wire = writer.finish(vec![]).await.unwrap();

    let mut reader = FileReader::open(Cursor::new(wire)).await.unwrap();
    let top = reader.read_composite("g").await.unwrap();
    assert_eq!(top.members.len(), 2);

    let FileItem::Composite(nested) = &top.members[0] else {
        panic!("first member should be a nested composite");
    };
    assert_eq!(nested.name, "g.i");
    assert_eq!(nested.members.len(), 2);
    let FileItem::Tensor(inner_a) = &nested.members[0] else {
        panic!("nested member 0 should be a tensor");
    };
    assert_eq!(inner_a.buffers[0].as_ref(), da.as_slice());

    let FileItem::Tensor(outer_c) = &top.members[1] else {
        panic!("second outer member should be a tensor");
    };
    assert_eq!(outer_c.buffers[0].as_ref(), dc.as_slice());
}

// ── error cases ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn read_composite_errors_on_non_composite() {
    let desc = plain_2x2();
    let data = vec![0u8; 16];
    let out = Vec::<u8>::new();
    let mut writer = FileWriter::new(out).await.unwrap();
    writer
        .write_tensor("p", &desc, &[data.as_slice()])
        .await
        .unwrap();
    let wire = writer.finish(vec![]).await.unwrap();

    let mut reader = FileReader::open(Cursor::new(wire)).await.unwrap();
    let err = reader.read_composite("p").await.unwrap_err();
    assert!(
        matches!(err, hurray_io::Error::NotAComposite(ref n) if n == "p"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn read_composite_errors_on_missing_name() {
    let desc = plain_2x2();
    let data = vec![0u8; 16];
    let out = Vec::<u8>::new();
    let mut writer = FileWriter::new(out).await.unwrap();
    writer
        .write_tensor("p", &desc, &[data.as_slice()])
        .await
        .unwrap();
    let wire = writer.finish(vec![]).await.unwrap();

    let mut reader = FileReader::open(Cursor::new(wire)).await.unwrap();
    let err = reader.read_composite("nope").await.unwrap_err();
    assert!(
        matches!(err, hurray_io::Error::TensorNotFound(ref n) if n == "nope"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn read_composite_errors_on_torn_composite() {
    // Fabricate a head declaring 3 members but only 2 written (write_composite would reject).
    let head = composite_head(CompositionRule::Group, 3, vec![2, 2]);
    let (a, b) = (plain_2x2(), plain_2x2());
    let data = vec![0u8; 16];

    let out = Vec::<u8>::new();
    let mut writer = FileWriter::new(out).await.unwrap();
    writer.write_tensor("t", &head, &[]).await.unwrap();
    writer
        .write_tensor("t.a", &a, &[data.as_slice()])
        .await
        .unwrap();
    writer
        .write_tensor("t.b", &b, &[data.as_slice()])
        .await
        .unwrap();
    let wire = writer.finish(vec![]).await.unwrap();

    let mut reader = FileReader::open(Cursor::new(wire)).await.unwrap();
    let err = reader.read_composite("t").await.unwrap_err();
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

#[tokio::test]
async fn read_composite_enforces_max_depth() {
    let outer = composite_head(CompositionRule::Group, 1, vec![2, 2]);
    let inner = composite_head(CompositionRule::Group, 1, vec![2, 2]);
    let a = plain_2x2();
    let data = vec![0u8; 16];
    let ba: [&[u8]; 1] = [data.as_slice()];

    let inner_members = vec![FileCompositeNode::Tensor {
        name: "o.i.a",
        descriptor: &a,
        buffers: &ba,
    }];
    let outer_members = vec![FileCompositeNode::Composite {
        name: "o.i",
        head: &inner,
        members: &inner_members,
    }];

    let out = Vec::<u8>::new();
    let mut writer = FileWriter::new(out).await.unwrap();
    writer
        .write_composite("o", &outer, &outer_members)
        .await
        .unwrap();
    let wire = writer.finish(vec![]).await.unwrap();

    let mut reader = FileReader::open(Cursor::new(wire))
        .await
        .unwrap()
        .with_max_composite_depth(1);
    let err = reader.read_composite("o").await.unwrap_err();
    assert!(
        matches!(err, hurray_io::Error::CompositeNestingTooDeep { limit: 1 }),
        "unexpected error: {err}"
    );
}
