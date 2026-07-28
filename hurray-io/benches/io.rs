//! Microbenchmarks for the `hurray-io` hot paths: streaming and file read/write of a
//! tensor (descriptor framing + buffer transfer).
//!
//! Run with: `cargo bench -p hurray-io`

use std::hint::black_box;
use std::io::Cursor;

use criterion::{criterion_group, criterion_main, Criterion};
use hurray_core::{
    buffer_size_bytes, BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode,
    TensorDescriptor, MIN_BUFFER_ALIGNMENT,
};
use hurray_io::file::{FileReader, FileWriter};
use hurray_io::stream::{StreamReader, StreamWriter};

// A float32 [128, 128] tensor: 64 KiB of data plus a dense descriptor.
const N: u64 = 128 * 128;

fn descriptor() -> TensorDescriptor {
    let handle = BufferHandle::new(
        buffer_size_bytes(ElementType::Float32, N),
        MIN_BUFFER_ALIGNMENT,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )
    .unwrap();
    TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(vec![128, 128]).unwrap(),
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

fn bench_stream(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let desc = descriptor();
    let data = vec![0u8; buffer_size_bytes(ElementType::Float32, N) as usize];

    let mut g = c.benchmark_group("stream");

    g.bench_function("write_tensor", |b| {
        b.to_async(&rt).iter(|| async {
            let mut wire = Vec::<u8>::new();
            let mut w = StreamWriter::new(&mut wire);
            w.write_tensor(black_box(&desc), &[&data]).await.unwrap();
            wire
        });
    });

    // Pre-build the wire once; each iteration decodes a fresh reader over the same bytes.
    let wire = rt.block_on(async {
        let mut wire = Vec::<u8>::new();
        let mut w = StreamWriter::new(&mut wire);
        w.write_tensor(&desc, &[&data]).await.unwrap();
        wire
    });
    g.bench_function("read_tensor", |b| {
        b.to_async(&rt).iter(|| async {
            let mut r = StreamReader::new(black_box(wire.as_slice()));
            r.next_tensor().await.unwrap().unwrap()
        });
    });

    g.finish();
}

fn bench_file(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let desc = descriptor();
    let data = vec![0u8; buffer_size_bytes(ElementType::Float32, N) as usize];

    let mut g = c.benchmark_group("file");

    g.bench_function("write", |b| {
        b.to_async(&rt).iter(|| async {
            let mut w = FileWriter::new(Vec::<u8>::new()).await.unwrap();
            w.write_tensor("t", black_box(&desc), &[&data])
                .await
                .unwrap();
            w.finish(vec![]).await.unwrap()
        });
    });

    let file_bytes = rt.block_on(async {
        let mut w = FileWriter::new(Vec::<u8>::new()).await.unwrap();
        w.write_tensor("t", &desc, &[&data]).await.unwrap();
        w.finish(vec![]).await.unwrap()
    });
    g.bench_function("open_and_read", |b| {
        b.to_async(&rt).iter(|| async {
            let mut r = FileReader::open(Cursor::new(black_box(file_bytes.clone())))
                .await
                .unwrap();
            r.read_tensor("t").await.unwrap()
        });
    });

    g.finish();
}

criterion_group!(benches, bench_stream, bench_file);
criterion_main!(benches);
