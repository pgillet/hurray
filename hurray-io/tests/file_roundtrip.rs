#![cfg(feature = "tokio")]

use std::io::Cursor;

use hurray_core::{
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode, TensorDescriptor,
    MIN_BUFFER_ALIGNMENT,
};
use hurray_io::file::{FileReader, FileWriter, FileWriterOptions, KvValue};

// ── helpers ───────────────────────────────────────────────────────────────────

fn make_desc(byte_size: u64, element_type: ElementType, dims: Vec<u64>) -> TensorDescriptor {
    let handle = BufferHandle::new(
        byte_size,
        MIN_BUFFER_ALIGNMENT,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )
    .unwrap();
    let shape = Shape::new(dims).unwrap();
    TensorDescriptor::new(
        1,
        0,
        element_type,
        shape,
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

/// Convenience: write tensors + kv to a Vec<u8> and return it.
async fn build_file(
    tensors: &[(&str, TensorDescriptor, Vec<u8>)],
    kv: Vec<(String, KvValue)>,
    options: FileWriterOptions,
) -> Vec<u8> {
    let buf = Vec::<u8>::new();
    let mut writer = FileWriter::with_options(buf, options).await.unwrap();
    for (name, desc, data) in tensors {
        writer.write_tensor(name, desc, &[data]).await.unwrap();
    }
    writer.finish(kv).await.unwrap()
}

// ── round-trip ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn roundtrip_single_tensor() {
    let desc = make_desc(64, ElementType::Float32, vec![4, 4]);
    let payload: Vec<u8> = (0u8..64).collect();

    let wire = build_file(
        &[("weight", desc, payload.clone())],
        vec![],
        FileWriterOptions::default(),
    )
    .await;

    let mut reader = FileReader::open(Cursor::new(wire)).await.unwrap();
    assert_eq!(reader.tensor_names().collect::<Vec<_>>(), ["weight"]);

    let tensor = reader.read_tensor("weight").await.unwrap();
    assert_eq!(tensor.name, "weight");
    assert_eq!(tensor.buffers[0].as_ref(), &payload[..]);
}

#[tokio::test]
async fn roundtrip_two_tensors() {
    let desc_a = make_desc(64, ElementType::Float32, vec![4, 4]);
    let desc_b = make_desc(64, ElementType::Float16, vec![4, 8]);
    let data_a: Vec<u8> = (0u8..64).collect();
    let data_b: Vec<u8> = (64u8..128).collect();

    let wire = build_file(
        &[
            ("layer0.weight", desc_a, data_a.clone()),
            ("layer1.weight", desc_b, data_b.clone()),
        ],
        vec![],
        FileWriterOptions::default(),
    )
    .await;

    let mut reader = FileReader::open(Cursor::new(wire)).await.unwrap();
    assert_eq!(
        reader.tensor_names().collect::<Vec<_>>(),
        ["layer0.weight", "layer1.weight"]
    );

    let t0 = reader.read_tensor("layer0.weight").await.unwrap();
    assert_eq!(t0.buffers[0].as_ref(), &data_a[..]);

    let t1 = reader.read_tensor("layer1.weight").await.unwrap();
    assert_eq!(t1.buffers[0].as_ref(), &data_b[..]);
}

#[tokio::test]
async fn random_access_skips_preceding_tensor() {
    let desc_a = make_desc(64, ElementType::Float32, vec![4, 4]);
    let desc_b = make_desc(128, ElementType::Uint8, vec![128]);
    let data_a = vec![0xAAu8; 64];
    let data_b: Vec<u8> = (0u8..128).collect();

    let wire = build_file(
        &[
            ("first", desc_a, data_a),
            ("second", desc_b, data_b.clone()),
        ],
        vec![],
        FileWriterOptions::default(),
    )
    .await;

    let mut reader = FileReader::open(Cursor::new(wire)).await.unwrap();
    // Read the second tensor directly — the reader must seek, not scan forward.
    let t = reader.read_tensor("second").await.unwrap();
    assert_eq!(t.buffers[0].as_ref(), &data_b[..]);
}

// ── KV metadata ───────────────────────────────────────────────────────────────

#[tokio::test]
async fn kv_roundtrip() {
    let desc = make_desc(64, ElementType::Float32, vec![4, 4]);
    let kv = vec![
        ("model".to_string(), KvValue::String("llama-3".to_string())),
        ("layers".to_string(), KvValue::Uint64(32)),
        ("quantized".to_string(), KvValue::Bool(false)),
        ("temperature".to_string(), KvValue::Float64(0.7)),
    ];

    let wire = build_file(
        &[("w", desc, vec![0u8; 64])],
        kv.clone(),
        FileWriterOptions::default(),
    )
    .await;

    let reader = FileReader::open(Cursor::new(wire)).await.unwrap();
    let got: std::collections::HashMap<_, _> =
        reader.kv().iter().map(|(k, v)| (k.as_str(), v)).collect();

    assert_eq!(got["model"], &KvValue::String("llama-3".to_string()));
    assert_eq!(got["layers"], &KvValue::Uint64(32));
    assert_eq!(got["quantized"], &KvValue::Bool(false));
    assert_eq!(got["temperature"], &KvValue::Float64(0.7));
}

#[tokio::test]
async fn kv_array_roundtrip() {
    let desc = make_desc(64, ElementType::Float32, vec![4, 4]);
    let kv = vec![(
        "vocab_sizes".to_string(),
        KvValue::Array(vec![
            KvValue::Uint64(1000),
            KvValue::Uint64(2000),
            KvValue::Uint64(4000),
        ]),
    )];

    let wire = build_file(
        &[("w", desc, vec![0u8; 64])],
        kv,
        FileWriterOptions::default(),
    )
    .await;
    let reader = FileReader::open(Cursor::new(wire)).await.unwrap();
    let kv_map: std::collections::HashMap<_, _> =
        reader.kv().iter().map(|(k, v)| (k.as_str(), v)).collect();

    assert!(matches!(kv_map["vocab_sizes"], KvValue::Array(_)));
    if let KvValue::Array(elems) = kv_map["vocab_sizes"] {
        assert_eq!(elems.len(), 3);
        assert_eq!(elems[1], KvValue::Uint64(2000));
    }
}

#[tokio::test]
async fn empty_kv_produces_no_kv_section() {
    let desc = make_desc(64, ElementType::Float32, vec![4, 4]);
    let wire = build_file(
        &[("w", desc, vec![0u8; 64])],
        vec![],
        FileWriterOptions::default(),
    )
    .await;
    let reader = FileReader::open(Cursor::new(wire)).await.unwrap();
    assert!(reader.kv().is_empty());
}

// ── sorted index ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn sorted_index_flag() {
    let opts = FileWriterOptions {
        sorted_index: true,
        ..Default::default()
    };
    let desc = make_desc(64, ElementType::Float32, vec![4, 4]);
    let wire = build_file(
        &[
            ("z_weight", desc.clone(), vec![0u8; 64]),
            ("a_weight", desc, vec![1u8; 64]),
        ],
        vec![],
        opts,
    )
    .await;

    let reader = FileReader::open(Cursor::new(wire)).await.unwrap();
    let names: Vec<_> = reader.tensor_names().collect();
    // With SORTED_INDEX the index is ordered by name bytes.
    assert_eq!(names, ["a_weight", "z_weight"]);
}

// ── read_descriptor ───────────────────────────────────────────────────────────

#[tokio::test]
async fn read_descriptor_only() {
    let desc = make_desc(64, ElementType::Float32, vec![4, 4]);
    let wire = build_file(
        &[("w", desc.clone(), vec![0u8; 64])],
        vec![],
        FileWriterOptions::default(),
    )
    .await;

    let mut reader = FileReader::open(Cursor::new(wire)).await.unwrap();
    let got = reader.read_descriptor("w").await.unwrap();
    assert_eq!(got.element_type, ElementType::Float32);
    assert_eq!(got.shape, desc.shape);
}

// ── writer validation ─────────────────────────────────────────────────────────

#[tokio::test]
async fn writer_rejects_empty_name() {
    let desc = make_desc(64, ElementType::Float32, vec![4, 4]);
    let mut writer = FileWriter::new(Vec::<u8>::new()).await.unwrap();
    let err = writer
        .write_tensor("", &desc, &[&vec![0u8; 64][..]])
        .await
        .unwrap_err();
    assert!(matches!(err, hurray_io::Error::TensorNameEmpty));
}

#[tokio::test]
async fn writer_rejects_duplicate_name() {
    let desc = make_desc(64, ElementType::Float32, vec![4, 4]);
    let mut writer = FileWriter::new(Vec::<u8>::new()).await.unwrap();
    writer
        .write_tensor("w", &desc, &[&vec![0u8; 64]])
        .await
        .unwrap();
    let err = writer
        .write_tensor("w", &desc, &[&vec![0u8; 64]])
        .await
        .unwrap_err();
    assert!(matches!(err, hurray_io::Error::DuplicateTensorName(_)));
}

#[tokio::test]
async fn finish_rejects_duplicate_kv_key() {
    let writer = FileWriter::new(Vec::<u8>::new()).await.unwrap();
    let kv = vec![
        ("key".to_string(), KvValue::Uint64(1)),
        ("key".to_string(), KvValue::Uint64(2)),
    ];
    let err = writer.finish(kv).await.unwrap_err();
    assert!(matches!(err, hurray_io::Error::DuplicateKvKey(_)));
}

// ── reader validation ─────────────────────────────────────────────────────────

#[tokio::test]
async fn reader_rejects_bad_magic() {
    let mut bytes = vec![0u8; 128];
    bytes[0..8].copy_from_slice(b"BADMAGIC");
    let err = FileReader::open(Cursor::new(bytes)).await.unwrap_err();
    assert!(matches!(err, hurray_io::Error::InvalidFileMagic));
}

#[tokio::test]
async fn reader_rejects_bad_trailer_magic() {
    let desc = make_desc(64, ElementType::Float32, vec![4, 4]);
    let mut wire = build_file(
        &[("w", desc, vec![0u8; 64])],
        vec![],
        FileWriterOptions::default(),
    )
    .await;
    // Corrupt the trailer magic
    let len = wire.len();
    wire[len - 4..].copy_from_slice(b"NOPE");
    let err = FileReader::open(Cursor::new(wire)).await.unwrap_err();
    assert!(matches!(err, hurray_io::Error::InvalidTrailerMagic));
}

#[tokio::test]
async fn reader_rejects_crc_mismatch() {
    let desc = make_desc(64, ElementType::Float32, vec![4, 4]);
    let mut wire = build_file(
        &[("w", desc, vec![0u8; 64])],
        vec![],
        FileWriterOptions::default(),
    )
    .await;
    // Corrupt the stored CRC in the trailer (bytes at offset len-12..len-8)
    let len = wire.len();
    wire[len - 12..len - 8].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
    let err = FileReader::open(Cursor::new(wire)).await.unwrap_err();
    assert!(matches!(err, hurray_io::Error::IndexCrc32cMismatch { .. }));
}

#[tokio::test]
async fn reader_returns_error_for_unknown_tensor() {
    let desc = make_desc(64, ElementType::Float32, vec![4, 4]);
    let wire = build_file(
        &[("w", desc, vec![0u8; 64])],
        vec![],
        FileWriterOptions::default(),
    )
    .await;
    let mut reader = FileReader::open(Cursor::new(wire)).await.unwrap();
    let err = reader.read_tensor("missing").await.unwrap_err();
    assert!(matches!(err, hurray_io::Error::TensorNotFound(_)));
}
