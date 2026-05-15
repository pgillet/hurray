#![cfg(feature = "tokio")]

use hurray_core::{
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode, TensorDescriptor,
    MIN_BUFFER_ALIGNMENT,
};
use hurray_io::stream::{StreamReader, StreamReaderOptions, StreamWriter};

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

// ── round-trip ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn roundtrip_single_tensor() {
    let desc = make_desc(64, ElementType::Float32, vec![4, 4]);
    let payload: Vec<u8> = (0u8..64).collect();

    let mut wire = Vec::<u8>::new();
    StreamWriter::new(&mut wire)
        .write_tensor(&desc, &[&payload])
        .await
        .unwrap();

    let mut reader = StreamReader::new(wire.as_slice());
    let tensor = reader
        .next_tensor()
        .await
        .unwrap()
        .expect("expected a tensor");
    assert_eq!(tensor.buffers[0].as_ref(), &payload[..]);
    assert!(
        reader.next_tensor().await.unwrap().is_none(),
        "expected clean EOF"
    );
}

#[tokio::test]
async fn roundtrip_two_tensors() {
    let desc_a = make_desc(64, ElementType::Float32, vec![4, 4]);
    let desc_b = make_desc(64, ElementType::Float16, vec![4, 8]);
    let data_a: Vec<u8> = (0u8..64).collect();
    let data_b: Vec<u8> = (64u8..128).collect();

    let mut wire = Vec::<u8>::new();
    let mut writer = StreamWriter::new(&mut wire);
    writer.write_tensor(&desc_a, &[&data_a]).await.unwrap();
    writer.write_tensor(&desc_b, &[&data_b]).await.unwrap();
    writer.finish().await.unwrap();

    let mut reader = StreamReader::new(wire.as_slice());
    let t1 = reader.next_tensor().await.unwrap().unwrap();
    assert_eq!(t1.buffers[0].as_ref(), &data_a[..]);
    let t2 = reader.next_tensor().await.unwrap().unwrap();
    assert_eq!(t2.buffers[0].as_ref(), &data_b[..]);
    assert!(reader.next_tensor().await.unwrap().is_none());
}

#[tokio::test]
async fn roundtrip_empty_stream() {
    let wire: &[u8] = &[];
    let mut reader = StreamReader::new(wire);
    assert!(reader.next_tensor().await.unwrap().is_none());
}

// ── writer validation ──────────────────────────────────────────────────────────

#[tokio::test]
async fn writer_rejects_wrong_buffer_count() {
    let desc = make_desc(64, ElementType::Float32, vec![4, 4]);
    let payload = vec![0u8; 64];
    let mut wire = Vec::<u8>::new();
    let mut writer = StreamWriter::new(&mut wire);
    let err = writer
        .write_tensor(&desc, &[&payload, &payload])
        .await
        .unwrap_err();
    assert!(
        matches!(
            err,
            hurray_io::Error::MultiBufferLengthMismatch {
                declared: 1,
                actual: 2
            }
        ),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn writer_rejects_buffer_size_mismatch() {
    let desc = make_desc(64, ElementType::Float32, vec![4, 4]);
    let short = vec![0u8; 32]; // declared 64, actual 32
    let mut wire = Vec::<u8>::new();
    let mut writer = StreamWriter::new(&mut wire);
    let err = writer.write_tensor(&desc, &[&short]).await.unwrap_err();
    assert!(
        matches!(
            err,
            hurray_io::Error::BufferSizeMismatch {
                index: 0,
                declared: 64,
                actual: 32
            }
        ),
        "unexpected error: {err}"
    );
}

// ── cross-machine enforcement ──────────────────────────────────────────────────

#[tokio::test]
async fn cross_machine_writer_accepts_producer_synced() {
    let desc = make_desc(64, ElementType::Float32, vec![4, 4]);
    let payload = vec![0u8; 64];
    let mut wire = Vec::<u8>::new();
    StreamWriter::cross_machine(&mut wire)
        .write_tensor(&desc, &[&payload])
        .await
        .expect("ProducerSynced must be accepted by cross-machine writer");
}

#[tokio::test]
async fn cross_machine_roundtrip() {
    let desc = make_desc(64, ElementType::Float32, vec![8, 2]);
    let payload: Vec<u8> = (0u8..64).collect();

    let mut wire = Vec::<u8>::new();
    StreamWriter::cross_machine(&mut wire)
        .write_tensor(&desc, &[&payload])
        .await
        .unwrap();

    let mut reader = StreamReader::cross_machine(wire.as_slice());
    let tensor = reader.next_tensor().await.unwrap().unwrap();
    assert_eq!(tensor.buffers[0].as_ref(), &payload[..]);
}

// ── size limits ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn reader_enforces_max_descriptor_bytes() {
    let desc = make_desc(64, ElementType::Float32, vec![4, 4]);
    let payload = vec![0u8; 64];

    let mut wire = Vec::<u8>::new();
    StreamWriter::new(&mut wire)
        .write_tensor(&desc, &[&payload])
        .await
        .unwrap();

    // Set limit below the descriptor's actual encoded length.
    let options = StreamReaderOptions {
        max_descriptor_bytes: 5,
        ..Default::default()
    };
    let mut reader = StreamReader::with_options(wire.as_slice(), options);
    let err = reader.next_tensor().await.unwrap_err();
    assert!(
        matches!(
            err,
            hurray_io::Error::FrameTooLarge {
                kind: "descriptor",
                ..
            }
        ),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn reader_enforces_max_buffer_bytes() {
    let desc = make_desc(128, ElementType::Float32, vec![4, 8]);
    let payload = vec![0u8; 128];

    let mut wire = Vec::<u8>::new();
    StreamWriter::new(&mut wire)
        .write_tensor(&desc, &[&payload])
        .await
        .unwrap();

    let options = StreamReaderOptions {
        max_buffer_bytes: 64, // descriptor declares 128
        ..Default::default()
    };
    let mut reader = StreamReader::with_options(wire.as_slice(), options);
    let err = reader.next_tensor().await.unwrap_err();
    assert!(
        matches!(err, hurray_io::Error::FrameTooLarge { kind: "buffer", .. }),
        "unexpected error: {err}"
    );
}

// ── truncated stream ───────────────────────────────────────────────────────────

#[tokio::test]
async fn reader_returns_unexpected_eof_on_truncated_descriptor() {
    let desc = make_desc(64, ElementType::Float32, vec![4, 4]);
    let payload = vec![0u8; 64];

    let mut wire = Vec::<u8>::new();
    StreamWriter::new(&mut wire)
        .write_tensor(&desc, &[&payload])
        .await
        .unwrap();

    let truncated = &wire[..5];
    let mut reader = StreamReader::new(truncated);
    let err = reader.next_tensor().await.unwrap_err();
    assert!(
        matches!(err, hurray_io::Error::UnexpectedEof),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn reader_returns_unexpected_eof_on_truncated_buffer() {
    let desc = make_desc(64, ElementType::Float32, vec![4, 4]);
    let payload = vec![0u8; 64];

    let mut wire = Vec::<u8>::new();
    StreamWriter::new(&mut wire)
        .write_tensor(&desc, &[&payload])
        .await
        .unwrap();

    // Keep the full descriptor but only half the buffer payload.
    let desc_len = wire.len() - 64;
    let truncated = &wire[..desc_len + 32];
    let mut reader = StreamReader::new(truncated);
    let err = reader.next_tensor().await.unwrap_err();
    assert!(
        matches!(err, hurray_io::Error::UnexpectedEof),
        "unexpected error: {err}"
    );
}
