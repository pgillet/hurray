#![cfg(feature = "tokio")]

//! A reader MUST hand out every non-empty buffer at the alignment its handle declares.
//!
//! `docs/spec/buffer-protocol.md § Minimum Alignment` makes the declaration binding on
//! both sides: the base address of a non-empty buffer MUST be at least 64-byte aligned,
//! and *a reader MAY rely on the declared alignment for SIMD loads*. A consumer is
//! therefore entitled to issue an aligned load against `handle.alignment()`, so these
//! tests assert the **address** of what the readers return — which nothing else in the
//! suite does (issue #235).
//!
//! Each case reads several tensors per stream, and includes 4096, so a run cannot pass by
//! an allocator coincidentally returning an over-aligned block.

use std::io::Cursor;

use hurray_core::{
    buffer_size_bytes,
    layout::{CompositeLayout, CompositionRule, CooLayout},
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, ShardDescriptor, SyncMode,
    TensorDescriptor,
};
use hurray_io::file::{FileCompositeNode, FileItem, FileReader, FileWriter, FileWriterOptions};
use hurray_io::stream::{CompositeNode, StreamItem, StreamReader, StreamWriter};

/// The declared alignments every path is checked against: the format's floor, a
/// power-of-two above it, and the page alignment a GPU/IPC/RDMA writer must use.
const DECLARED: [u32; 3] = [64, 128, 4096];

/// Tensors per stream, so a single lucky allocation cannot carry a case.
const PER_STREAM: usize = 4;

fn buf(byte_size: u64, alignment: u32) -> BufferHandle {
    BufferHandle::new(
        byte_size,
        alignment,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )
    .unwrap()
}

/// A `[8, 8]` float32 row-major tensor whose single buffer declares `alignment`.
fn row_major(alignment: u32) -> TensorDescriptor {
    TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(vec![8u64, 8]).unwrap(),
        0,
        LayoutDescriptor::RowMajor,
        vec![buf(buffer_size_bytes(ElementType::Float32, 64), alignment)],
        None,
        None,
        None,
        None,
    )
    .unwrap()
}

/// A sparse COO tensor: two buffers (values, indices) declaring *different* alignments,
/// so a per-buffer declaration cannot be satisfied by one allocation policy.
fn coo_two_buffers(values_alignment: u32, indices_alignment: u32) -> TensorDescriptor {
    let nnz = 8u64;
    TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(vec![8u64, 8]).unwrap(),
        0,
        LayoutDescriptor::Coo(CooLayout::new(nnz, false)),
        vec![
            buf(
                buffer_size_bytes(ElementType::Float32, nnz),
                values_alignment,
            ),
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

#[track_caller]
fn assert_aligned(what: &str, bytes: &[u8], declared: u32) {
    let addr = bytes.as_ptr() as usize;
    assert_eq!(
        addr % declared as usize,
        0,
        "{what}: buffer at {addr:#x} does not satisfy its declared {declared}-byte alignment"
    );
}

/// Every buffer of `desc`/`bytes`, checked against its own handle's declaration.
#[track_caller]
fn assert_all_aligned(what: &str, desc: &TensorDescriptor, buffers: &[bytes::Bytes]) {
    assert_eq!(buffers.len(), desc.buffers.len(), "{what}: buffer count");
    for (i, (handle, data)) in desc.buffers.iter().zip(buffers).enumerate() {
        if handle.byte_size() == 0 {
            continue; // An empty buffer's base address is not dereferenceable.
        }
        assert_aligned(&format!("{what}/buffer[{i}]"), data, handle.alignment());
    }
}

// ── stream reader ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn stream_tensor_honours_declared_alignment() {
    for declared in DECLARED {
        let desc = row_major(declared);
        let payload = vec![0x5Au8; 256];

        let mut wire = Vec::<u8>::new();
        let mut writer = StreamWriter::new(&mut wire);
        for _ in 0..PER_STREAM {
            writer.write_tensor(&desc, &[&payload]).await.unwrap();
        }
        writer.finish().await.unwrap();

        let mut reader = StreamReader::new(wire.as_slice());
        for i in 0..PER_STREAM {
            let tensor = reader.next_tensor().await.unwrap().expect("a tensor");
            assert_all_aligned(
                &format!("stream[{i}] declaring {declared}"),
                &tensor.descriptor,
                &tensor.buffers,
            );
            assert_eq!(tensor.buffers[0].as_ref(), &payload[..], "payload intact");
        }
    }
}

#[tokio::test]
async fn stream_multi_buffer_honours_per_buffer_alignment() {
    let desc = coo_two_buffers(128, 4096);
    let values = vec![0x11u8; 32];
    let indices = vec![0x22u8; 128];

    let mut wire = Vec::<u8>::new();
    let mut writer = StreamWriter::new(&mut wire);
    for _ in 0..PER_STREAM {
        writer
            .write_tensor(&desc, &[&values, &indices])
            .await
            .unwrap();
    }
    writer.finish().await.unwrap();

    let mut reader = StreamReader::new(wire.as_slice());
    for i in 0..PER_STREAM {
        let tensor = reader.next_tensor().await.unwrap().expect("a tensor");
        assert_all_aligned(
            &format!("stream coo[{i}]"),
            &tensor.descriptor,
            &tensor.buffers,
        );
        assert_eq!(tensor.buffers[0].as_ref(), &values[..]);
        assert_eq!(tensor.buffers[1].as_ref(), &indices[..]);
    }
}

#[tokio::test]
async fn stream_composite_honours_declared_alignment() {
    // A partition whose two members declare different alignments, with the whole
    // composite nested one level deep inside a group: the recursive read path must
    // allocate per member, not once per item.
    let inner_head = composite_head(CompositionRule::Partition, 2, vec![8, 8]);
    let (m0, m1) = (partition_member(0, 4096), partition_member(4, 128));
    let d0 = vec![0xAAu8; 128];
    let d1 = vec![0xBBu8; 128];
    let b0: [&[u8]; 1] = [d0.as_slice()];
    let b1: [&[u8]; 1] = [d1.as_slice()];
    let inner_members = vec![
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
    let plain_data = vec![0xCCu8; 256];
    let plain_buffers: [&[u8]; 1] = [plain_data.as_slice()];

    let outer_head = composite_head(CompositionRule::Group, 2, vec![8, 8]);
    let outer_members = vec![
        CompositeNode::Tensor {
            descriptor: &plain,
            buffers: &plain_buffers,
        },
        CompositeNode::Composite {
            head: &inner_head,
            members: &inner_members,
        },
    ];

    let mut wire = Vec::<u8>::new();
    let mut writer = StreamWriter::new(&mut wire);
    writer
        .write_composite(&outer_head, &outer_members)
        .await
        .unwrap();
    writer.finish().await.unwrap();

    let mut reader = StreamReader::new(wire.as_slice());
    let item = reader.next_item().await.unwrap().expect("an item");
    assert_stream_item_aligned("outer", &item);
}

/// Walks a [`StreamItem`] tree, asserting every tensor's buffers against their handles.
#[track_caller]
fn assert_stream_item_aligned(path: &str, item: &StreamItem) {
    match item {
        StreamItem::Tensor(t) => assert_all_aligned(path, &t.descriptor, &t.buffers),
        StreamItem::Composite(c) => {
            assert!(c.head.buffers.is_empty(), "{path}: head owns no buffers");
            for (i, member) in c.members.iter().enumerate() {
                assert_stream_item_aligned(&format!("{path}/member[{i}]"), member);
            }
        }
    }
}

// ── file reader ───────────────────────────────────────────────────────────────

#[tokio::test]
async fn file_tensor_honours_declared_alignment() {
    for declared in DECLARED {
        let desc = row_major(declared);
        let payload = vec![0x5Au8; 256];

        let mut writer = FileWriter::new(Vec::<u8>::new()).await.unwrap();
        for i in 0..PER_STREAM {
            writer
                .write_tensor(&format!("t{i}"), &desc, &[&payload])
                .await
                .unwrap();
        }
        let wire = writer.finish(vec![]).await.unwrap();

        let mut reader = FileReader::open(Cursor::new(wire)).await.unwrap();
        for i in 0..PER_STREAM {
            let tensor = reader.read_tensor(&format!("t{i}")).await.unwrap();
            assert_all_aligned(
                &format!("file[{i}] declaring {declared}"),
                &tensor.descriptor,
                &tensor.buffers,
            );
            assert_eq!(tensor.buffers[0].as_ref(), &payload[..], "payload intact");
        }
    }
}

#[tokio::test]
async fn file_multi_buffer_honours_per_buffer_alignment() {
    let desc = coo_two_buffers(4096, 128);
    let values = vec![0x11u8; 32];
    let indices = vec![0x22u8; 128];

    let mut writer = FileWriter::new(Vec::<u8>::new()).await.unwrap();
    writer
        .write_tensor("sparse", &desc, &[&values, &indices])
        .await
        .unwrap();
    let wire = writer.finish(vec![]).await.unwrap();

    let mut reader = FileReader::open(Cursor::new(wire)).await.unwrap();
    let tensor = reader.read_tensor("sparse").await.unwrap();
    assert_all_aligned("file coo", &tensor.descriptor, &tensor.buffers);
    assert_eq!(tensor.buffers[0].as_ref(), &values[..]);
    assert_eq!(tensor.buffers[1].as_ref(), &indices[..]);
}

#[tokio::test]
async fn file_composite_honours_declared_alignment() {
    let head = composite_head(CompositionRule::Partition, 2, vec![8, 8]);
    let (m0, m1) = (partition_member(0, 4096), partition_member(4, 128));
    let d0 = vec![0xAAu8; 128];
    let d1 = vec![0xBBu8; 128];
    let b0: [&[u8]; 1] = [d0.as_slice()];
    let b1: [&[u8]; 1] = [d1.as_slice()];
    let members = vec![
        FileCompositeNode::Tensor {
            name: "w.tile0",
            descriptor: &m0,
            buffers: &b0,
        },
        FileCompositeNode::Tensor {
            name: "w.tile1",
            descriptor: &m1,
            buffers: &b1,
        },
    ];

    let mut writer = FileWriter::new(Vec::<u8>::new()).await.unwrap();
    writer.write_composite("w", &head, &members).await.unwrap();
    let wire = writer.finish(vec![]).await.unwrap();

    let mut reader = FileReader::open(Cursor::new(wire)).await.unwrap();
    let composite = reader.read_composite("w").await.unwrap();
    for (i, member) in composite.members.iter().enumerate() {
        assert_file_item_aligned(&format!("file composite/member[{i}]"), member);
    }
}

/// Walks a [`FileItem`] tree, asserting every tensor's buffers against their handles.
#[track_caller]
fn assert_file_item_aligned(path: &str, item: &FileItem) {
    match item {
        FileItem::Tensor(t) => assert_all_aligned(path, &t.descriptor, &t.buffers),
        FileItem::Composite(c) => {
            assert!(c.head.buffers.is_empty(), "{path}: head owns no buffers");
            for (i, member) in c.members.iter().enumerate() {
                assert_file_item_aligned(&format!("{path}/member[{i}]"), member);
            }
        }
    }
}

// ── empty buffers ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn empty_buffer_is_legal_at_any_declared_power_of_two() {
    // `byte_size == 0` dereferences nothing, so the spec keeps it legal at any
    // power-of-two alignment — including 1, which BufferHandle::empty declares.
    let desc = TensorDescriptor::new(
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

    let mut wire = Vec::<u8>::new();
    let mut writer = StreamWriter::new(&mut wire);
    writer.write_tensor(&desc, &[&[]]).await.unwrap();
    writer.finish().await.unwrap();

    let mut reader = StreamReader::new(wire.as_slice());
    let tensor = reader.next_tensor().await.unwrap().expect("a tensor");
    assert!(tensor.buffers[0].is_empty());

    let mut writer = FileWriter::with_options(Vec::<u8>::new(), FileWriterOptions::default())
        .await
        .unwrap();
    writer.write_tensor("empty", &desc, &[&[]]).await.unwrap();
    let file = writer.finish(vec![]).await.unwrap();
    let mut reader = FileReader::open(Cursor::new(file)).await.unwrap();
    assert!(reader.read_tensor("empty").await.unwrap().buffers[0].is_empty());
}
