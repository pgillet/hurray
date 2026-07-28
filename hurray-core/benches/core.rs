//! Microbenchmarks for the `hurray-core` hot paths: tensor descriptor encode/decode and
//! element-address computation.
//!
//! Run with: `cargo bench -p hurray-core`

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use hurray_core::{
    buffer_size_bytes,
    layout::{CsrLayout, StridedLayout},
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, PerBlockAffine, QuantizationDescriptor,
    Shape, SyncMode, TensorDescriptor, MIN_BUFFER_ALIGNMENT,
};

fn buf(byte_size: u64) -> BufferHandle {
    BufferHandle::new(
        byte_size,
        MIN_BUFFER_ALIGNMENT,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )
    .unwrap()
}

/// A plain dense row-major float32 [256, 256] descriptor.
fn dense_descriptor() -> TensorDescriptor {
    TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(vec![256, 256]).unwrap(),
        0,
        LayoutDescriptor::RowMajor,
        vec![buf(buffer_size_bytes(ElementType::Float32, 256 * 256))],
        None,
        None,
        None,
        None,
    )
    .unwrap()
}

/// A quantized int4 [256, 256] per-block-affine descriptor (heavier: quant section + 2 buffers).
fn quantized_descriptor() -> TensorDescriptor {
    let pba = PerBlockAffine::new_symmetric(1, 64, 1, ElementType::Float16).unwrap();
    let num_blocks = pba.num_blocks_per_axis(256) * 256;
    let quant = QuantizationDescriptor::PerBlockAffine(pba);
    TensorDescriptor::new(
        1,
        0,
        ElementType::Int4,
        Shape::new(vec![256, 256]).unwrap(),
        0,
        LayoutDescriptor::RowMajor,
        vec![
            buf(buffer_size_bytes(ElementType::Int4, 256 * 256)),
            buf(buffer_size_bytes(ElementType::Float16, num_blocks)),
        ],
        Some(quant.encode_to_vec()),
        None,
        None,
        None,
    )
    .unwrap()
}

fn bench_descriptor_encode(c: &mut Criterion) {
    let mut g = c.benchmark_group("descriptor_encode");
    let dense = dense_descriptor();
    g.bench_function("dense_row_major_f32", |b| {
        b.iter(|| black_box(&dense).encode().unwrap())
    });
    let quant = quantized_descriptor();
    g.bench_function("quantized_int4_per_block", |b| {
        b.iter(|| black_box(&quant).encode().unwrap())
    });
    g.finish();
}

fn bench_descriptor_decode(c: &mut Criterion) {
    let mut g = c.benchmark_group("descriptor_decode");
    let dense = dense_descriptor().encode().unwrap();
    g.bench_function("dense_row_major_f32", |b| {
        b.iter(|| TensorDescriptor::decode(black_box(&dense)).unwrap())
    });
    let quant = quantized_descriptor().encode().unwrap();
    g.bench_function("quantized_int4_per_block", |b| {
        b.iter(|| TensorDescriptor::decode(black_box(&quant)).unwrap())
    });
    g.finish();
}

fn bench_element_offset(c: &mut Criterion) {
    let mut g = c.benchmark_group("element_offset");
    let shape = Shape::new(vec![256, 256]).unwrap();

    let row_major = LayoutDescriptor::RowMajor;
    g.bench_function("row_major", |b| {
        b.iter(|| {
            row_major
                .element_offset(black_box(&[123, 45]), &shape)
                .unwrap()
        })
    });

    let strided = LayoutDescriptor::Strided(StridedLayout::new(vec![256, 1]));
    g.bench_function("strided", |b| {
        b.iter(|| {
            strided
                .element_offset(black_box(&[123, 45]), &shape)
                .unwrap()
        })
    });

    g.finish();
}

fn bench_sparse_lookup(c: &mut Criterion) {
    use hurray_core::layout::addressing::csr;

    // 1024×1024 CSR with 4 non-zeros per row (sorted columns), for a realistic per-row search.
    let nrows: u64 = 1024;
    let per_row: u64 = 4;
    let mut col_indices = Vec::new();
    let mut row_ptr = vec![0u64];
    for _ in 0..nrows {
        for k in 0..per_row {
            col_indices.push(k * 256); // columns 0, 256, 512, 768
        }
        row_ptr.push(col_indices.len() as u64);
    }
    let _ = CsrLayout::new(col_indices.len() as u64);

    let mut g = c.benchmark_group("sparse_lookup");
    g.bench_function("csr_hit", |b| {
        // Look up an existing non-zero (row 500, column 512).
        b.iter(|| csr::element_offset(black_box(&[500, 512]), &col_indices, &row_ptr).unwrap())
    });
    g.bench_function("csr_miss", |b| {
        // Look up a structural zero (row 500, column 100).
        b.iter(|| csr::element_offset(black_box(&[500, 100]), &col_indices, &row_ptr).unwrap())
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_descriptor_encode,
    bench_descriptor_decode,
    bench_element_offset,
    bench_sparse_lookup
);
criterion_main!(benches);
