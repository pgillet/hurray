//! Converting an externally quantized tensor into a Hurray descriptor.
//!
//! An external toolchain hands you quantized weights, per-block scales, and a
//! zero-point plane. The scales transfer as-is. The zero-point plane does not:
//! Hurray dequantizes as `scale * (q - zero_point)`, subtracting the stored
//! value with no implicit bias, while toolchains in the GPTQ family have
//! conventionally stored `zero_point - 1` and removed the bias in the loader.
//!
//! The plane is rebuilt during conversion regardless — the foreign side packs
//! two 4-bit values per byte, Hurray's per-block affine scheme takes one `int32`
//! per block (`docs/spec/quantization/per-block-affine.md` § Referenced Buffers)
//! — so the normalization is one line inside a transform you are already
//! writing, not a step to remember afterwards.
//!
//! Run with: `cargo run --example convert_external_quantized -p hurray-core`

use hurray_core::{
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, PerBlockAffine, QuantizationDescriptor,
    Shape, SyncMode, TensorDescriptor, DESCRIPTOR_VERSION_MAJOR, DESCRIPTOR_VERSION_MINOR,
    MIN_BUFFER_ALIGNMENT,
};

const ROWS: usize = 2;
const COLS: usize = 16;
const BLOCK: usize = 8;

/// Blocks along the quantized axis, per row: `ceil(COLS / BLOCK)`.
const BLOCKS_PER_ROW: usize = COLS / BLOCK;
/// One scale and one zero point per block, across the whole tensor.
const NUM_BLOCKS: usize = BLOCKS_PER_ROW * ROWS;

/// Quantized weights, row-major, `[ROWS, COLS]`.
#[rustfmt::skip]
const WEIGHTS: [i8; ROWS * COLS] = [
    9, 8, 7, 10, 8, 6, 12, 8,   3, 9, 8, 11, 7, 8,  5, 10,
    8, 13, 8, 4, 9, 8,  7, 8,  10, 8, 6,  8, 9, 2,  8, 14,
];

/// Per-block scales. These transfer unchanged — only the zero points carry a
/// convention.
const SCALES: [f32; NUM_BLOCKS] = [0.02, 0.015, 0.025, 0.01];

/// The foreign zero-point plane: 4-bit values packed two per byte, low nibble
/// first, each holding `zero_point - 1`. Decodes to 7, 6, 8, 7.
const FOREIGN_ZERO_POINTS_PACKED: [u8; NUM_BLOCKS / 2] = [0x67, 0x78];

/// Rebuild the foreign 4-bit plane as Hurray's `int32` plane.
///
/// Two things happen here, and only one of them is visible in the output: the
/// unpacking, which fails loudly if you get it wrong, and the `+ 1`, which does
/// not fail at all.
fn normalize_zero_points(packed: &[u8], count: usize) -> Vec<i32> {
    (0..count)
        .map(|i| {
            let byte = packed[i / 2];
            let nibble = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            // The convention normalization. Hurray subtracts the stored zero
            // point as-is, so the toolchain's -1 bias is removed here or never.
            i32::from(nibble) + 1
        })
        .collect()
}

/// Block index for logical position `[row, col]`, per the spec:
/// `b = outer * blocks_per_row + floor(col / BLOCK)`, where `outer` is `row`
/// for a 2-D tensor quantized along axis 1.
fn block_of(row: usize, col: usize) -> usize {
    row * BLOCKS_PER_ROW + col / BLOCK
}

/// Apply the spec's dequantization formula: `x = scale * (q - zero_point)`.
///
/// Hurray describes tensors, it does not compute on them — there is no
/// dequantize in `hurray-core` and there should not be. This is example code
/// reading the formula out of the spec, not a library function.
fn dequantize(weights: &[i8], scales: &[f32], zero_points: &[i32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(weights.len());
    for row in 0..ROWS {
        for col in 0..COLS {
            let b = block_of(row, col);
            let q = i32::from(weights[row * COLS + col]);
            out.push(scales[b] * (q - zero_points[b]) as f32);
        }
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Converting an externally quantized tensor\n");

    // ── 1. Rebuild the zero-point plane ─────────────────────────────────────

    let zero_points = normalize_zero_points(&FOREIGN_ZERO_POINTS_PACKED, NUM_BLOCKS);
    assert_eq!(zero_points, vec![8, 7, 9, 8], "normalized zero points");

    println!("  foreign plane : {FOREIGN_ZERO_POINTS_PACKED:02X?} (4-bit, biased)");
    println!("  hurray plane  : {zero_points:?} (int32, as subtracted)\n");

    // ── 2. Build the descriptor over the rebuilt buffers ────────────────────

    let handle = |len: u64| {
        BufferHandle::new(
            len,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )
    };

    // Buffer 0: the weights. Buffer 1: float32 scales. Buffer 2: int32 zero points.
    let buffers = vec![
        handle((ROWS * COLS) as u64)?,
        handle((NUM_BLOCKS * 4) as u64)?,
        handle((NUM_BLOCKS * 4) as u64)?,
    ];

    let quant = QuantizationDescriptor::PerBlockAffine(PerBlockAffine::new_asymmetric(
        1,                    // axis: blocks run along the columns
        BLOCK as u32,         // block_size
        1,                    // scale_buffer_index
        2,                    // zero_point_buffer_index
        ElementType::Float32, // scale_type
    )?);

    let desc = TensorDescriptor::new(
        DESCRIPTOR_VERSION_MAJOR,
        DESCRIPTOR_VERSION_MINOR,
        ElementType::Int8,
        Shape::new(vec![ROWS as u64, COLS as u64])?,
        0,
        LayoutDescriptor::RowMajor,
        buffers,
        Some(quant.encode_to_vec()),
        None, // no shard
        None, // no statistics
        None, // no extension type
    )?;

    println!("  descriptor    : {} bytes\n", desc.encode()?.len());

    // ── 3. What skipping the normalization costs ────────────────────────────

    let correct = dequantize(&WEIGHTS, &SCALES, &zero_points);

    // The same conversion with the foreign values copied straight through.
    let unnormalized: Vec<i32> = zero_points.iter().map(|z| z - 1).collect();
    let wrong = dequantize(&WEIGHTS, &SCALES, &unnormalized);

    // Every element is off by exactly one scale step of its own block — a valid
    // descriptor, a clean decode, and the wrong numbers.
    for row in 0..ROWS {
        for col in 0..COLS {
            let i = row * COLS + col;
            let drift = wrong[i] - correct[i];
            let step = SCALES[block_of(row, col)];
            assert!(
                (drift - step).abs() < 1e-6,
                "element [{row}, {col}]: drift {drift} is not one scale step {step}"
            );
        }
    }

    println!(
        "  element [0, 0]: correct {:.4}, un-normalized {:.4}",
        correct[0], wrong[0]
    );
    println!("  every element off by exactly one scale step of its block\n");
    println!("Nothing rejects the second descriptor. Only the arithmetic tells you.");

    Ok(())
}
