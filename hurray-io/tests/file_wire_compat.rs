#![cfg(feature = "tokio")]

//! The file container format, frozen.
//!
//! ADR-041 § 11 asks for byte-for-byte wire compatibility across the driver refactor, and
//! a test that compares the core against the driver cannot show that once the driver *is*
//! the core: both sides become the same code and the assertion passes by construction. So
//! the bytes here are frozen artifacts, emitted by the pre-refactor `FileWriter` and only
//! ever read.
//!
//! The descriptors below are the specification of what each fixture contains — the
//! corpus is self-contained for that reason, rather than sharing builders with
//! `wire_compat.rs`: a frozen-corpus test that reaches into another test's helpers can
//! have its meaning changed from outside.
//!
//! See `fixtures/file/README.md` for provenance.

use std::io::Cursor;

use hurray_core::{
    buffer_size_bytes,
    layout::{CompositeLayout, CompositionRule, CooLayout},
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, ShardDescriptor, SyncMode,
    TensorDescriptor, MIN_BUFFER_ALIGNMENT,
};
use hurray_io::file::{
    FileCompositeNode, FileItem, FileReader, FileWriter, FileWriterOptions, KvValue,
};

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

/// A COO tensor whose two buffers declare different alignments (128 and 4096), so the
/// corpus covers inter-buffer padding inside one tensor's data region.
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

/// Every KV wire tag, including an array of scalars (tag `0x07`).
fn kv_section() -> Vec<(String, KvValue)> {
    vec![
        ("model".into(), KvValue::String("hurray-test".into())),
        ("epoch".into(), KvValue::Int64(-7)),
        ("tokens".into(), KvValue::Uint64(1 << 40)),
        ("temperature".into(), KvValue::Float64(0.8125)),
        ("quantized".into(), KvValue::Bool(true)),
        (
            "digest".into(),
            KvValue::Bytes(vec![0xDE, 0xAD, 0xBE, 0xEF]),
        ),
        (
            "layer_sizes".into(),
            KvValue::Array(vec![
                KvValue::Int64(4096),
                KvValue::Int64(11008),
                KvValue::Int64(4096),
            ]),
        ),
    ]
}

// ── the four files, each built only from the public writer API ────────────────

/// One tensor, default options (4096-byte data alignment, unsorted index), no KV.
async fn write_row_major() -> Vec<u8> {
    let desc = row_major();
    let data = payload(256, 0x00);

    let mut writer = FileWriter::new(Vec::<u8>::new()).await.unwrap();
    writer
        .write_tensor("embeddings", &desc, &[&data])
        .await
        .unwrap();
    writer.finish(vec![]).await.unwrap()
}

/// A two-buffer tensor, so the data region carries padding between buffers.
async fn write_multi_buffer() -> Vec<u8> {
    let desc = coo();
    let values = payload(32, 0x11);
    let indices = payload(128, 0x22);

    let mut writer = FileWriter::new(Vec::<u8>::new()).await.unwrap();
    writer
        .write_tensor("sparse", &desc, &[&values, &indices])
        .await
        .unwrap();
    writer.finish(vec![]).await.unwrap()
}

/// Two tensors written out of byte order with `sorted_index`, a non-default data
/// alignment, and a KV section covering every tag.
async fn write_kv_sorted() -> Vec<u8> {
    let zeta = row_major();
    let alpha = plain_2x2();
    let zeta_data = payload(256, 0x55);
    let alpha_data = payload(16, 0x66);

    let options = FileWriterOptions {
        data_buffer_alignment: 8192,
        sorted_index: true,
    };
    let mut writer = FileWriter::with_options(Vec::<u8>::new(), options)
        .await
        .unwrap();
    writer
        .write_tensor("zeta", &zeta, &[&zeta_data])
        .await
        .unwrap();
    writer
        .write_tensor("alpha", &alpha, &[&alpha_data])
        .await
        .unwrap();
    writer.finish(kv_section()).await.unwrap()
}

/// A group of two whose first member is a partition of two: nesting depth 2.
async fn write_nested_composite() -> Vec<u8> {
    let group = composite_head(CompositionRule::Group, 2, vec![8, 8]);
    let part = composite_head(CompositionRule::Partition, 2, vec![8, 8]);
    let tile_0 = partition_member(0);
    let tile_4 = partition_member(4);
    let tail = plain_2x2();

    let tile_0_data = payload(128, 0x77);
    let tile_4_data = payload(128, 0x88);
    let tail_data = payload(16, 0x99);

    let inner = [
        FileCompositeNode::Tensor {
            name: "part.0",
            descriptor: &tile_0,
            buffers: &[&tile_0_data],
        },
        FileCompositeNode::Tensor {
            name: "part.1",
            descriptor: &tile_4,
            buffers: &[&tile_4_data],
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
            buffers: &[&tail_data],
        },
    ];

    let mut writer = FileWriter::new(Vec::<u8>::new()).await.unwrap();
    writer
        .write_composite("group", &group, &members)
        .await
        .unwrap();
    writer.finish(vec![]).await.unwrap()
}

// ── the corpus ────────────────────────────────────────────────────────────────

fn fixture(name: &str) -> Vec<u8> {
    let path = format!(
        "{}/tests/fixtures/file/{}",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
    std::fs::read(&path).unwrap_or_else(|e| panic!("reading fixture {path}: {e}"))
}

async fn corpus() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        ("row_major.hrry", write_row_major().await),
        ("multi_buffer.hrry", write_multi_buffer().await),
        ("kv_sorted.hrry", write_kv_sorted().await),
        ("nested_composite.hrry", write_nested_composite().await),
    ]
}

// ── the tests ─────────────────────────────────────────────────────────────────

/// The writer still emits the frozen bytes, exactly.
#[tokio::test]
async fn the_writer_still_emits_the_frozen_bytes() {
    for (name, produced) in corpus().await {
        let frozen = fixture(name);
        assert_eq!(
            produced.len(),
            frozen.len(),
            "{name}: length changed ({} produced vs {} frozen)",
            produced.len(),
            frozen.len()
        );
        if let Some(at) = produced.iter().zip(&frozen).position(|(a, b)| a != b) {
            panic!(
                "{name}: byte {at} differs (produced 0x{:02X}, frozen 0x{:02X})",
                produced[at], frozen[at]
            );
        }
    }
}

/// The reader still reads the frozen bytes, with every buffer at its declared alignment.
#[tokio::test]
async fn the_reader_still_reads_the_frozen_bytes() {
    // row_major.hrry
    let mut reader = FileReader::open(Cursor::new(fixture("row_major.hrry")))
        .await
        .unwrap();
    assert_eq!(reader.tensor_names().collect::<Vec<_>>(), ["embeddings"]);
    let tensor = reader.read_tensor("embeddings").await.unwrap();
    assert_eq!(tensor.descriptor, row_major());
    assert_eq!(tensor.buffers[0].as_ref(), &payload(256, 0x00)[..]);

    // multi_buffer.hrry — two buffers, declared alignments 128 and 4096
    let mut reader = FileReader::open(Cursor::new(fixture("multi_buffer.hrry")))
        .await
        .unwrap();
    let tensor = reader.read_tensor("sparse").await.unwrap();
    assert_eq!(tensor.buffers[0].as_ref(), &payload(32, 0x11)[..]);
    assert_eq!(tensor.buffers[1].as_ref(), &payload(128, 0x22)[..]);
    for (i, (handle, buffer)) in tensor
        .descriptor
        .buffers
        .iter()
        .zip(&tensor.buffers)
        .enumerate()
    {
        let declared = handle.alignment() as usize;
        assert_eq!(
            buffer.as_ptr() as usize % declared,
            0,
            "buffer {i} is not at its declared {declared}-byte alignment"
        );
    }

    // kv_sorted.hrry — sorted index, full KV section
    let mut reader = FileReader::open(Cursor::new(fixture("kv_sorted.hrry")))
        .await
        .unwrap();
    assert_eq!(reader.tensor_names().collect::<Vec<_>>(), ["alpha", "zeta"]);
    assert_eq!(reader.kv(), &kv_section()[..]);
    assert_eq!(
        reader.read_tensor("zeta").await.unwrap().buffers[0].as_ref(),
        &payload(256, 0x55)[..]
    );
    assert_eq!(
        reader.read_tensor("alpha").await.unwrap().buffers[0].as_ref(),
        &payload(16, 0x66)[..]
    );

    // nested_composite.hrry — depth 2, five index entries
    let mut reader = FileReader::open(Cursor::new(fixture("nested_composite.hrry")))
        .await
        .unwrap();
    assert_eq!(
        reader.tensor_names().collect::<Vec<_>>(),
        ["group", "part", "part.0", "part.1", "group.1"]
    );
    let composite = reader.read_composite("group").await.unwrap();
    assert_eq!(composite.name, "group");
    assert_eq!(composite.members.len(), 2);
    let FileItem::Composite(inner) = &composite.members[0] else {
        panic!("expected the first member to be a nested composite")
    };
    assert_eq!(inner.name, "part");
    assert_eq!(inner.members.len(), 2);
    let FileItem::Tensor(tile_0) = &inner.members[0] else {
        panic!("expected a plain tensor")
    };
    assert_eq!(tile_0.buffers[0].as_ref(), &payload(128, 0x77)[..]);
    let FileItem::Tensor(tail) = &composite.members[1] else {
        panic!("expected a plain tensor")
    };
    assert_eq!(tail.name, "group.1");
    assert_eq!(tail.buffers[0].as_ref(), &payload(16, 0x99)[..]);
}
