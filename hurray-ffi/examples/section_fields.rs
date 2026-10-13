//! Reading the optional descriptor sections from the C ABI (ADR-040).
//!
//! `hurray_descriptor_decode` parses and validates all five optional sections,
//! but before these identifiers existed the result was unreachable from C: a
//! consumer handed a quantized tensor could not learn the scheme, let alone the
//! block size or which buffer held the scales.
//!
//! The pattern is the one the layout payload already uses — a field is named by
//! an identifier whose high half-word says which sub-object it addresses, and
//! read through a typed accessor. Two of them are new here: `..._i64` for a
//! signed zero point, `..._f64` for scales and statistics.
//!
//! This walks what a quantized-inference consumer actually meets: find the
//! scheme, read its parameters, then check the advisory statistics and the
//! shard's place in its parent. Written in Rust only because that is what this
//! repository builds; the sequence is what a Go or Julia binding on the C ABI
//! would follow.
//!
//! Run with:
//!
//!     cargo run -p hurray-ffi --example section_fields

use hurray_core::descriptor::{ShardDescriptor, Statistics, StatisticsMask};
use hurray_core::{
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, PerBlockAffine, PerTensorAffine,
    QuantizationDescriptor, Shape, SyncMode, TensorDescriptor, MIN_BUFFER_ALIGNMENT,
};
use hurray_ffi::{
    descriptor::hurray_descriptor_destroy,
    field::{
        hurray_descriptor_field_f64, hurray_descriptor_field_i64, hurray_descriptor_field_u64,
        hurray_descriptor_field_u64_array, HurrayDescriptorField,
        HURRAY_QUANT_FIELD_PER_BLOCK_BLOCK_SIZE, HURRAY_QUANT_FIELD_PER_BLOCK_SCALE_BUFFER,
        HURRAY_QUANT_FIELD_PER_BLOCK_SCALE_TYPE, HURRAY_QUANT_FIELD_PER_BLOCK_ZERO_POINT_BUFFER,
        HURRAY_QUANT_FIELD_PER_TENSOR_SCALE, HURRAY_QUANT_FIELD_PER_TENSOR_ZERO_POINT,
        HURRAY_QUANT_FIELD_SCHEME_TAG, HURRAY_SHARD_FIELD_PARENT_SHAPE,
        HURRAY_SHARD_FIELD_SHARD_OFFSET, HURRAY_STATS_FIELD_NNZ, HURRAY_STATS_FIELD_VALUE_ABS_MAX,
        HURRAY_STATS_FIELD_VALUE_MEAN,
    },
    hurray_descriptor_decode, HurrayDescriptor, HURRAY_ERR_BUFFER_TOO_SMALL,
    HURRAY_ERR_FIELD_NOT_PRESENT, HURRAY_OK,
};

/// Encodes a descriptor so this example has something to decode.
///
/// A real consumer receives these bytes over a stream, a file, or a capsule; it
/// never builds them.
fn encode(
    quantization: Option<QuantizationDescriptor>,
    shard: Option<ShardDescriptor>,
    statistics: Option<Statistics>,
) -> Vec<u8> {
    let buffers = (0..3)
        .map(|_| {
            BufferHandle::new(
                4096,
                MIN_BUFFER_ALIGNMENT,
                DeviceTag::Cpu,
                SyncMode::ProducerSynced,
            )
            .expect("buffer handle")
        })
        .collect();

    TensorDescriptor::new(
        1,
        0,
        ElementType::Int8,
        Shape::new(vec![512, 256]).expect("shape"),
        0,
        LayoutDescriptor::RowMajor,
        buffers,
        quantization.map(|q| q.encode_to_vec()),
        shard,
        statistics,
        None,
    )
    .expect("descriptor")
    .encode()
    .expect("encode")
}

/// Decodes bytes into a handle. The caller must destroy it.
fn decode(bytes: &[u8]) -> *mut HurrayDescriptor {
    let mut handle: *mut HurrayDescriptor = std::ptr::null_mut();
    let status = unsafe { hurray_descriptor_decode(bytes.as_ptr(), bytes.len(), &mut handle) };
    assert_eq!(status, HURRAY_OK, "decode failed with status {status}");
    handle
}

/// Reads an unsigned scalar, or `None` if this descriptor does not carry it.
fn u64_field(handle: *const HurrayDescriptor, field: HurrayDescriptorField) -> Option<u64> {
    let mut value = 0u64;
    match unsafe { hurray_descriptor_field_u64(handle, field, 0, &mut value) } {
        HURRAY_OK => Some(value),
        HURRAY_ERR_FIELD_NOT_PRESENT => None,
        status => panic!("field {field:#x} failed with {status}"),
    }
}

/// Reads a real scalar, or `None` if this descriptor does not carry it.
fn f64_field(handle: *const HurrayDescriptor, field: HurrayDescriptorField) -> Option<f64> {
    let mut value = 0f64;
    match unsafe { hurray_descriptor_field_f64(handle, field, 0, &mut value) } {
        HURRAY_OK => Some(value),
        HURRAY_ERR_FIELD_NOT_PRESENT => None,
        status => panic!("field {field:#x} failed with {status}"),
    }
}

/// Reads an unsigned array field with the two-step capacity query.
///
/// Step one passes a null buffer to learn the length; step two allocates and
/// reads. The same protocol `hurray_descriptor_shape` uses.
fn u64_array(handle: *const HurrayDescriptor, field: HurrayDescriptorField) -> Vec<u64> {
    let mut len = 0usize;
    let status = unsafe {
        hurray_descriptor_field_u64_array(handle, field, 0, std::ptr::null_mut(), &mut len)
    };
    assert_eq!(status, HURRAY_ERR_BUFFER_TOO_SMALL);

    let mut out = vec![0u64; len];
    let status =
        unsafe { hurray_descriptor_field_u64_array(handle, field, 0, out.as_mut_ptr(), &mut len) };
    assert_eq!(status, HURRAY_OK, "array field {field:#x} failed");
    out
}

/// Prints whichever quantization scheme a descriptor carries.
///
/// The scheme tag comes first: it decides which of the scheme-specific
/// identifiers apply, exactly as the layout tag does for the layout payload.
fn report_quantization(handle: *const HurrayDescriptor) {
    let Some(tag) = u64_field(handle, HURRAY_QUANT_FIELD_SCHEME_TAG) else {
        println!("  quantization: none — the tensor stores its values directly");
        return;
    };

    match tag {
        0x01 => {
            let scale = f64_field(handle, HURRAY_QUANT_FIELD_PER_TENSOR_SCALE)
                .expect("per-tensor affine always carries a scale");
            let mut zero_point = 0i64;
            let status = unsafe {
                hurray_descriptor_field_i64(
                    handle,
                    HURRAY_QUANT_FIELD_PER_TENSOR_ZERO_POINT,
                    0,
                    &mut zero_point,
                )
            };
            assert_eq!(status, HURRAY_OK);
            // The signed accessor earns its place here: a zero point is routinely
            // negative, and an unsigned read would report -128 as 2^64 - 128.
            println!("  quantization: per-tensor affine (0x01)");
            println!("    scale        = {scale}");
            println!("    zero point   = {zero_point}");
            println!("    value        = (stored - zero point) * scale");
        }
        0x03 => {
            let block_size = u64_field(handle, HURRAY_QUANT_FIELD_PER_BLOCK_BLOCK_SIZE)
                .expect("per-block affine always carries a block size");
            let scale_buffer = u64_field(handle, HURRAY_QUANT_FIELD_PER_BLOCK_SCALE_BUFFER)
                .expect("per-block affine always carries a scale buffer");
            let scale_type = u64_field(handle, HURRAY_QUANT_FIELD_PER_BLOCK_SCALE_TYPE)
                .expect("per-block affine always carries a scale type");
            println!("  quantization: per-block affine (0x03)");
            println!("    block size   = {block_size} elements");
            println!("    scales       = buffer {scale_buffer}, element type {scale_type:#04x}");
            // Absence is the answer, not an error: a symmetric scheme has no
            // zero point at all, which is why this is not a sentinel value.
            match u64_field(handle, HURRAY_QUANT_FIELD_PER_BLOCK_ZERO_POINT_BUFFER) {
                Some(i) => println!("    zero points  = buffer {i} (asymmetric)"),
                None => println!("    zero points  = none (symmetric)"),
            }
        }
        other => println!("  quantization: scheme {other:#04x}, not handled by this example"),
    }
}

/// Prints the advisory statistics a descriptor carries, if any.
fn report_statistics(handle: *const HurrayDescriptor) {
    // Each statistic is guarded by a bit in `computed_mask`, and the accessor
    // checks it: an uncomputed field reads as absent rather than as whatever
    // placeholder the writer left behind.
    let fields = [
        (
            "nnz",
            u64_field(handle, HURRAY_STATS_FIELD_NNZ).map(|v| v as f64),
        ),
        (
            "abs max",
            f64_field(handle, HURRAY_STATS_FIELD_VALUE_ABS_MAX),
        ),
        ("mean", f64_field(handle, HURRAY_STATS_FIELD_VALUE_MEAN)),
    ];
    for (name, value) in fields {
        match value {
            Some(v) => println!("  statistics: {name} = {v}"),
            None => println!("  statistics: {name} not computed"),
        }
    }
}

fn main() {
    // ── A per-tensor quantized tensor ────────────────────────────────────────
    let per_tensor = QuantizationDescriptor::PerTensorAffine(
        PerTensorAffine::new(0.0078125, -128).expect("per-tensor"),
    );
    let bytes = encode(Some(per_tensor), None, None);
    let handle = decode(&bytes);
    println!("per-tensor quantized int8 tensor:");
    report_quantization(handle);
    report_statistics(handle);
    assert_eq!(unsafe { hurray_descriptor_destroy(handle) }, HURRAY_OK);

    // ── A per-block quantized shard, with statistics ─────────────────────────
    let per_block = QuantizationDescriptor::PerBlockAffine(
        PerBlockAffine::new_symmetric(0, 64, 1, ElementType::Float16).expect("per-block"),
    );
    // This tensor is rows 512..1024 of a 2048 x 256 parent.
    let shard = ShardDescriptor::new(vec![2048, 256], vec![512, 0]).expect("shard");
    let statistics = Statistics {
        computed_mask: StatisticsMask(
            StatisticsMask::NNZ_VALID | StatisticsMask::VALUE_RANGE_VALID,
        ),
        nnz: 130_816,
        sparsity_ratio: 0.0,
        value_min: -0.98,
        value_max: 0.97,
        value_abs_max: 0.98,
        value_mean: 0.0,
        value_stddev: 0.0,
        nm_n: 0,
        nm_m: 0,
        has_nan: false,
        has_inf: false,
    };
    let bytes = encode(Some(per_block), Some(shard), Some(statistics));
    let handle = decode(&bytes);

    println!("\nper-block quantized shard:");
    report_quantization(handle);
    report_statistics(handle);

    let parent = u64_array(handle, HURRAY_SHARD_FIELD_PARENT_SHAPE);
    let offset = u64_array(handle, HURRAY_SHARD_FIELD_SHARD_OFFSET);
    println!("  shard: offset {offset:?} within parent {parent:?}");

    assert_eq!(unsafe { hurray_descriptor_destroy(handle) }, HURRAY_OK);

    // ── A plain tensor: every section absent ─────────────────────────────────
    let bytes = encode(None, None, None);
    let handle = decode(&bytes);
    println!("\nunquantized tensor with no sections:");
    report_quantization(handle);
    report_statistics(handle);
    assert_eq!(unsafe { hurray_descriptor_destroy(handle) }, HURRAY_OK);
}
