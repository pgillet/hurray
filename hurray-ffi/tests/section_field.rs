//! Integration tests for the optional-section field accessors (ADR-040).
//!
//! Every test drives the public C ABI: encode a descriptor carrying a section
//! with `hurray-core`, decode it through `hurray_descriptor_decode`, then read
//! fields back the way a C consumer would.

use hurray_core::descriptor::{
    CompositeMemberDescriptor, ExtensionTypeDescriptor, MemberRole, ShardDescriptor, Statistics,
    StatisticsMask,
};
use hurray_core::{
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Mxfp, Nf4, PerBlockAffine,
    PerChannelAffine, PerTensorAffine, QuantizationDescriptor, Shape, SyncMode, TensorDescriptor,
    MIN_BUFFER_ALIGNMENT,
};
use hurray_ffi::{
    descriptor::hurray_descriptor_destroy, field::*, hurray_descriptor_decode, HurrayDescriptor,
    HURRAY_ERR_BUFFER_TOO_SMALL, HURRAY_ERR_FIELD_NOT_APPLICABLE, HURRAY_ERR_FIELD_NOT_PRESENT,
    HURRAY_ERR_INDEX_OUT_OF_BOUNDS, HURRAY_OK,
};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn buffers(n: usize) -> Vec<BufferHandle> {
    (0..n)
        .map(|_| {
            BufferHandle::new(
                256,
                MIN_BUFFER_ALIGNMENT,
                DeviceTag::Cpu,
                SyncMode::ProducerSynced,
            )
            .expect("buffer handle")
        })
        .collect()
}

/// A plain row-major descriptor, optionally carrying sections.
struct Sections {
    quantization: Option<Vec<u8>>,
    shard: Option<ShardDescriptor>,
    statistics: Option<Statistics>,
    extension_type: Option<ExtensionTypeDescriptor>,
    composite_member: Option<CompositeMemberDescriptor>,
}

impl Sections {
    fn none() -> Self {
        Self {
            quantization: None,
            shard: None,
            statistics: None,
            extension_type: None,
            composite_member: None,
        }
    }

    fn quantization(q: QuantizationDescriptor) -> Self {
        Self {
            quantization: Some(q.encode_to_vec()),
            ..Self::none()
        }
    }

    fn statistics(s: Statistics) -> Self {
        Self {
            statistics: Some(s),
            ..Self::none()
        }
    }
}

/// Encodes a descriptor carrying `sections`, with enough buffers for any scale
/// vector the quantization scheme may reference.
fn encode(sections: Sections) -> Vec<u8> {
    let element_type = match &sections.extension_type {
        Some(_) => ElementType::Extension(0xF1),
        None => ElementType::Float32,
    };
    let desc = TensorDescriptor::new(
        1,
        0,
        element_type,
        Shape::new(vec![4, 8]).expect("shape"),
        0,
        LayoutDescriptor::RowMajor,
        buffers(3),
        sections.quantization,
        sections.shard,
        sections.statistics,
        sections.extension_type,
    )
    .expect("descriptor");
    let desc = match sections.composite_member {
        Some(cm) => desc.with_composite_member(cm),
        None => desc,
    };
    desc.encode().expect("encode")
}

/// Decodes bytes into a handle. The caller must destroy it.
fn decode(bytes: &[u8]) -> *mut HurrayDescriptor {
    let mut handle: *mut HurrayDescriptor = std::ptr::null_mut();
    let status = unsafe { hurray_descriptor_decode(bytes.as_ptr(), bytes.len(), &mut handle) };
    assert_eq!(status, HURRAY_OK, "decode failed with status {status}");
    handle
}

fn destroy(handle: *mut HurrayDescriptor) {
    assert_eq!(unsafe { hurray_descriptor_destroy(handle) }, HURRAY_OK);
}

/// Reads an unsigned scalar, asserting success.
fn u64_field(handle: *mut HurrayDescriptor, field: HurrayDescriptorField) -> u64 {
    let mut value = 0u64;
    let status = unsafe { hurray_descriptor_field_u64(handle, field, 0, &mut value) };
    assert_eq!(status, HURRAY_OK, "field {field:#x} failed with {status}");
    value
}

/// Reads a real scalar, asserting success.
fn f64_field(handle: *mut HurrayDescriptor, field: HurrayDescriptorField) -> f64 {
    let mut value = 0f64;
    let status = unsafe { hurray_descriptor_field_f64(handle, field, 0, &mut value) };
    assert_eq!(status, HURRAY_OK, "field {field:#x} failed with {status}");
    value
}

/// Reads an unsigned array field with a two-step capacity query, as a C caller would.
fn u64_array(handle: *mut HurrayDescriptor, field: HurrayDescriptorField) -> Vec<u64> {
    let mut len = 0usize;
    let status = unsafe {
        hurray_descriptor_field_u64_array(handle, field, 0, std::ptr::null_mut(), &mut len)
    };
    assert_eq!(status, HURRAY_ERR_BUFFER_TOO_SMALL);

    let mut out = vec![0u64; len];
    let status =
        unsafe { hurray_descriptor_field_u64_array(handle, field, 0, out.as_mut_ptr(), &mut len) };
    assert_eq!(status, HURRAY_OK);
    assert_eq!(len, out.len());
    out
}

/// Asserts a field read fails with `expected`, leaving the output untouched.
fn u64_field_err(handle: *mut HurrayDescriptor, field: HurrayDescriptorField, expected: i32) {
    let mut value = u64::MAX;
    let status = unsafe { hurray_descriptor_field_u64(handle, field, 0, &mut value) };
    assert_eq!(status, expected, "field {field:#x}");
    assert_eq!(value, u64::MAX, "output must be untouched on failure");
}

// ── Quantization ──────────────────────────────────────────────────────────────

#[test]
fn reads_per_tensor_affine_scale_and_zero_point() {
    let q = QuantizationDescriptor::PerTensorAffine(
        PerTensorAffine::new(0.125, -128).expect("per-tensor"),
    );
    let bytes = encode(Sections::quantization(q));
    let h = decode(&bytes);

    assert_eq!(u64_field(h, HURRAY_QUANT_FIELD_SCHEME_TAG), 0x01);
    assert_eq!(f64_field(h, HURRAY_QUANT_FIELD_PER_TENSOR_SCALE), 0.125);

    let mut zp = 0i64;
    let status = unsafe {
        hurray_descriptor_field_i64(h, HURRAY_QUANT_FIELD_PER_TENSOR_ZERO_POINT, 0, &mut zp)
    };
    assert_eq!(status, HURRAY_OK);
    assert_eq!(zp, -128, "a negative zero point survives the read");

    destroy(h);
}

#[test]
fn a_negative_zero_point_is_not_readable_as_unsigned() {
    let q =
        QuantizationDescriptor::PerTensorAffine(PerTensorAffine::new(1.0, -5).expect("per-tensor"));
    let bytes = encode(Sections::quantization(q));
    let h = decode(&bytes);

    // The signed field has no unsigned reading: the accessor rejects it rather
    // than reporting -5 as a huge positive number.
    u64_field_err(
        h,
        HURRAY_QUANT_FIELD_PER_TENSOR_ZERO_POINT,
        HURRAY_ERR_FIELD_NOT_APPLICABLE,
    );
    destroy(h);
}

#[test]
fn reads_per_channel_affine_asymmetric() {
    let q = QuantizationDescriptor::PerChannelAffine(
        PerChannelAffine::new_asymmetric(1, 1, 2).expect("per-channel"),
    );
    let bytes = encode(Sections::quantization(q));
    let h = decode(&bytes);

    assert_eq!(u64_field(h, HURRAY_QUANT_FIELD_SCHEME_TAG), 0x02);
    assert_eq!(u64_field(h, HURRAY_QUANT_FIELD_PER_CHANNEL_AXIS), 1);
    assert_eq!(u64_field(h, HURRAY_QUANT_FIELD_PER_CHANNEL_SCALE_BUFFER), 1);
    assert_eq!(
        u64_field(h, HURRAY_QUANT_FIELD_PER_CHANNEL_ZERO_POINT_BUFFER),
        2
    );
    destroy(h);
}

#[test]
fn a_symmetric_scheme_reports_no_zero_point_buffer() {
    let q = QuantizationDescriptor::PerChannelAffine(
        PerChannelAffine::new_symmetric(0, 1).expect("per-channel"),
    );
    let bytes = encode(Sections::quantization(q));
    let h = decode(&bytes);

    assert_eq!(
        u64_field(h, HURRAY_QUANT_FIELD_PER_CHANNEL_SCALE_TYPE),
        ElementType::Float32.tag() as u64
    );
    // Symmetric is the absence of a zero point, not a zero one.
    u64_field_err(
        h,
        HURRAY_QUANT_FIELD_PER_CHANNEL_ZERO_POINT_BUFFER,
        HURRAY_ERR_FIELD_NOT_PRESENT,
    );
    destroy(h);
}

#[test]
fn reads_per_block_affine() {
    let q = QuantizationDescriptor::PerBlockAffine(
        PerBlockAffine::new_asymmetric(0, 64, 1, 2, ElementType::Float16).expect("per-block"),
    );
    let bytes = encode(Sections::quantization(q));
    let h = decode(&bytes);

    assert_eq!(u64_field(h, HURRAY_QUANT_FIELD_SCHEME_TAG), 0x03);
    assert_eq!(u64_field(h, HURRAY_QUANT_FIELD_PER_BLOCK_AXIS), 0);
    assert_eq!(u64_field(h, HURRAY_QUANT_FIELD_PER_BLOCK_BLOCK_SIZE), 64);
    assert_eq!(u64_field(h, HURRAY_QUANT_FIELD_PER_BLOCK_SCALE_BUFFER), 1);
    assert_eq!(
        u64_field(h, HURRAY_QUANT_FIELD_PER_BLOCK_ZERO_POINT_BUFFER),
        2
    );
    assert_eq!(
        u64_field(h, HURRAY_QUANT_FIELD_PER_BLOCK_SCALE_TYPE),
        ElementType::Float16.tag() as u64
    );
    destroy(h);
}

#[test]
fn reads_nf4() {
    let q = QuantizationDescriptor::Nf4(Nf4::new(0, 64, 1).expect("nf4"));
    let bytes = encode(Sections::quantization(q));
    let h = decode(&bytes);

    assert_eq!(u64_field(h, HURRAY_QUANT_FIELD_SCHEME_TAG), 0x04);
    assert_eq!(u64_field(h, HURRAY_QUANT_FIELD_NF4_AXIS), 0);
    assert_eq!(u64_field(h, HURRAY_QUANT_FIELD_NF4_BLOCK_SIZE), 64);
    assert_eq!(u64_field(h, HURRAY_QUANT_FIELD_NF4_SCALE_BUFFER), 1);
    destroy(h);
}

#[test]
fn reads_mxfp() {
    let q = QuantizationDescriptor::Mxfp(Mxfp::new(1, 32, 2).expect("mxfp"));
    let bytes = encode(Sections::quantization(q));
    let h = decode(&bytes);

    assert_eq!(u64_field(h, HURRAY_QUANT_FIELD_SCHEME_TAG), 0x05);
    assert_eq!(u64_field(h, HURRAY_QUANT_FIELD_MXFP_AXIS), 1);
    assert_eq!(u64_field(h, HURRAY_QUANT_FIELD_MXFP_BLOCK_SIZE), 32);
    assert_eq!(u64_field(h, HURRAY_QUANT_FIELD_MXFP_SCALE_BUFFER), 2);
    destroy(h);
}

#[test]
fn a_field_from_another_scheme_is_not_applicable() {
    let q = QuantizationDescriptor::Nf4(Nf4::new(0, 64, 1).expect("nf4"));
    let bytes = encode(Sections::quantization(q));
    let h = decode(&bytes);

    u64_field_err(
        h,
        HURRAY_QUANT_FIELD_MXFP_BLOCK_SIZE,
        HURRAY_ERR_FIELD_NOT_APPLICABLE,
    );
    u64_field_err(
        h,
        HURRAY_QUANT_FIELD_PER_BLOCK_AXIS,
        HURRAY_ERR_FIELD_NOT_APPLICABLE,
    );
    destroy(h);
}

// ── Statistics ────────────────────────────────────────────────────────────────

/// Statistics with every group marked valid.
fn full_statistics() -> Statistics {
    Statistics {
        computed_mask: StatisticsMask(
            StatisticsMask::NNZ_VALID
                | StatisticsMask::SPARSITY_VALID
                | StatisticsMask::VALUE_RANGE_VALID
                | StatisticsMask::VALUE_STATS_VALID
                | StatisticsMask::NM_SPARSITY_VALID
                | StatisticsMask::NAN_INF_VALID,
        ),
        nnz: 1234,
        sparsity_ratio: 0.25,
        value_min: -3.5,
        value_max: 7.25,
        value_abs_max: 7.25,
        value_mean: 0.5,
        value_stddev: 1.5,
        nm_n: 2,
        nm_m: 4,
        has_nan: false,
        has_inf: true,
    }
}

#[test]
fn reads_every_statistic_when_the_mask_allows() {
    let bytes = encode(Sections::statistics(full_statistics()));
    let h = decode(&bytes);

    assert_eq!(u64_field(h, HURRAY_STATS_FIELD_COMPUTED_MASK), 0x3F);
    assert_eq!(u64_field(h, HURRAY_STATS_FIELD_NNZ), 1234);
    assert_eq!(f64_field(h, HURRAY_STATS_FIELD_SPARSITY_RATIO), 0.25);
    assert_eq!(f64_field(h, HURRAY_STATS_FIELD_VALUE_MIN), -3.5);
    assert_eq!(f64_field(h, HURRAY_STATS_FIELD_VALUE_MAX), 7.25);
    assert_eq!(f64_field(h, HURRAY_STATS_FIELD_VALUE_ABS_MAX), 7.25);
    assert_eq!(f64_field(h, HURRAY_STATS_FIELD_VALUE_MEAN), 0.5);
    assert_eq!(f64_field(h, HURRAY_STATS_FIELD_VALUE_STDDEV), 1.5);
    assert_eq!(u64_field(h, HURRAY_STATS_FIELD_NM_N), 2);
    assert_eq!(u64_field(h, HURRAY_STATS_FIELD_NM_M), 4);
    assert_eq!(u64_field(h, HURRAY_STATS_FIELD_HAS_NAN), 0);
    assert_eq!(u64_field(h, HURRAY_STATS_FIELD_HAS_INF), 1);
    destroy(h);
}

#[test]
fn an_unguarded_statistic_reports_not_present() {
    // Only NNZ is computed; every other field holds a placeholder the caller
    // must not read.
    let stats = Statistics {
        computed_mask: StatisticsMask(StatisticsMask::NNZ_VALID),
        ..full_statistics()
    };
    let bytes = encode(Sections::statistics(stats));
    let h = decode(&bytes);

    assert_eq!(u64_field(h, HURRAY_STATS_FIELD_NNZ), 1234);
    for field in [
        HURRAY_STATS_FIELD_NM_N,
        HURRAY_STATS_FIELD_NM_M,
        HURRAY_STATS_FIELD_HAS_NAN,
        HURRAY_STATS_FIELD_HAS_INF,
    ] {
        u64_field_err(h, field, HURRAY_ERR_FIELD_NOT_PRESENT);
    }
    for field in [
        HURRAY_STATS_FIELD_SPARSITY_RATIO,
        HURRAY_STATS_FIELD_VALUE_MIN,
        HURRAY_STATS_FIELD_VALUE_MAX,
        HURRAY_STATS_FIELD_VALUE_ABS_MAX,
        HURRAY_STATS_FIELD_VALUE_MEAN,
        HURRAY_STATS_FIELD_VALUE_STDDEV,
    ] {
        let mut value = f64::NAN;
        let status = unsafe { hurray_descriptor_field_f64(h, field, 0, &mut value) };
        assert_eq!(status, HURRAY_ERR_FIELD_NOT_PRESENT, "field {field:#x}");
        assert!(value.is_nan(), "output must be untouched on failure");
    }

    // The raw mask is always readable — a caller may prefer to branch on it.
    assert_eq!(
        u64_field(h, HURRAY_STATS_FIELD_COMPUTED_MASK),
        StatisticsMask::NNZ_VALID as u64
    );
    destroy(h);
}

// ── Shard ─────────────────────────────────────────────────────────────────────

#[test]
fn reads_shard_parent_shape_and_offset() {
    let shard = ShardDescriptor::new(vec![8, 8], vec![4, 0]).expect("shard");
    let bytes = encode(Sections {
        shard: Some(shard),
        ..Sections::none()
    });
    let h = decode(&bytes);

    assert_eq!(u64_array(h, HURRAY_SHARD_FIELD_PARENT_SHAPE), [8, 8]);
    assert_eq!(u64_array(h, HURRAY_SHARD_FIELD_SHARD_OFFSET), [4, 0]);
    destroy(h);
}

#[test]
fn insufficient_shard_capacity_reports_the_true_length() {
    let shard = ShardDescriptor::new(vec![8, 8], vec![4, 0]).expect("shard");
    let bytes = encode(Sections {
        shard: Some(shard),
        ..Sections::none()
    });
    let h = decode(&bytes);

    let mut out = [0u64; 1];
    let mut len = 1usize; // capacity of 1, true length is 2
    let status = unsafe {
        hurray_descriptor_field_u64_array(
            h,
            HURRAY_SHARD_FIELD_PARENT_SHAPE,
            0,
            out.as_mut_ptr(),
            &mut len,
        )
    };
    assert_eq!(status, HURRAY_ERR_BUFFER_TOO_SMALL);
    assert_eq!(len, 2, "the true length is written so the caller can retry");
    destroy(h);
}

// ── Extension type ────────────────────────────────────────────────────────────

#[test]
fn reads_every_extension_type_field() {
    // An FP4 E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit, two per byte.
    // `is_signed` is false even though the type has a sign bit — core reserves
    // that flag for integers and carries a float's sign in `sign_bits`.
    let ext = ExtensionTypeDescriptor::new(4, 2, true, false, 1, 2, 1, 1, true, false)
        .expect("extension type");
    let bytes = encode(Sections {
        extension_type: Some(ext),
        ..Sections::none()
    });
    let h = decode(&bytes);

    assert_eq!(u64_field(h, HURRAY_EXT_TYPE_FIELD_BIT_WIDTH), 4);
    assert_eq!(u64_field(h, HURRAY_EXT_TYPE_FIELD_PACKING_FACTOR), 2);
    assert_eq!(u64_field(h, HURRAY_EXT_TYPE_FIELD_IS_FLOAT), 1);
    assert_eq!(u64_field(h, HURRAY_EXT_TYPE_FIELD_IS_SIGNED), 0);
    assert_eq!(u64_field(h, HURRAY_EXT_TYPE_FIELD_SIGN_BITS), 1);
    assert_eq!(u64_field(h, HURRAY_EXT_TYPE_FIELD_EXPONENT_BITS), 2);
    assert_eq!(u64_field(h, HURRAY_EXT_TYPE_FIELD_MANTISSA_BITS), 1);
    assert_eq!(u64_field(h, HURRAY_EXT_TYPE_FIELD_EXPONENT_BIAS), 1);
    assert_eq!(u64_field(h, HURRAY_EXT_TYPE_FIELD_HAS_NAN), 1);
    assert_eq!(u64_field(h, HURRAY_EXT_TYPE_FIELD_HAS_INF), 0);
    destroy(h);
}

// ── Composite member ──────────────────────────────────────────────────────────

#[test]
fn reads_composite_member_role() {
    for (role, wire) in [(MemberRole::Correction, 0), (MemberRole::Base, 1)] {
        let bytes = encode(Sections {
            composite_member: Some(CompositeMemberDescriptor::new(role)),
            ..Sections::none()
        });
        let h = decode(&bytes);
        assert_eq!(u64_field(h, HURRAY_COMPOSITE_MEMBER_FIELD_ROLE), wire);
        destroy(h);
    }
}

// ── Absence and misuse ────────────────────────────────────────────────────────

#[test]
fn every_section_reports_not_present_when_absent() {
    let bytes = encode(Sections::none());
    let h = decode(&bytes);

    for field in [
        HURRAY_QUANT_FIELD_SCHEME_TAG,
        HURRAY_QUANT_FIELD_NF4_BLOCK_SIZE,
        HURRAY_STATS_FIELD_COMPUTED_MASK,
        HURRAY_STATS_FIELD_NNZ,
        HURRAY_EXT_TYPE_FIELD_BIT_WIDTH,
        HURRAY_COMPOSITE_MEMBER_FIELD_ROLE,
    ] {
        u64_field_err(h, field, HURRAY_ERR_FIELD_NOT_PRESENT);
    }

    let mut len = 0usize;
    let status = unsafe {
        hurray_descriptor_field_u64_array(
            h,
            HURRAY_SHARD_FIELD_PARENT_SHAPE,
            0,
            std::ptr::null_mut(),
            &mut len,
        )
    };
    assert_eq!(status, HURRAY_ERR_FIELD_NOT_PRESENT);
    destroy(h);
}

#[test]
fn a_layout_field_has_no_signed_or_real_reading() {
    let bytes = encode(Sections::none());
    let h = decode(&bytes);

    // Every layout-payload field is integral and unsigned, so the signed and
    // real accessors reject the whole 0x0001 range.
    let mut signed = 0i64;
    assert_eq!(
        unsafe { hurray_descriptor_field_i64(h, HURRAY_LAYOUT_FIELD_CSR_NNZ, 0, &mut signed) },
        HURRAY_ERR_FIELD_NOT_APPLICABLE
    );
    let mut real = 0f64;
    assert_eq!(
        unsafe { hurray_descriptor_field_f64(h, HURRAY_LAYOUT_FIELD_CSR_NNZ, 0, &mut real) },
        HURRAY_ERR_FIELD_NOT_APPLICABLE
    );
    destroy(h);
}

#[test]
fn a_section_has_no_signed_array_reading() {
    let shard = ShardDescriptor::new(vec![8, 8], vec![4, 0]).expect("shard");
    let bytes = encode(Sections {
        shard: Some(shard),
        ..Sections::none()
    });
    let h = decode(&bytes);

    // No section carries a signed array; the i64 array accessor serves the
    // layout range only.
    let mut len = 0usize;
    let status = unsafe {
        hurray_descriptor_field_i64_array(
            h,
            HURRAY_SHARD_FIELD_PARENT_SHAPE,
            0,
            std::ptr::null_mut(),
            &mut len,
        )
    };
    assert_eq!(status, HURRAY_ERR_FIELD_NOT_APPLICABLE);
    destroy(h);
}

#[test]
fn a_nonzero_level_is_rejected_for_every_section_field() {
    let bytes = encode(Sections::statistics(full_statistics()));
    let h = decode(&bytes);

    // `level` addresses tiled nesting; no section nests, so a non-zero level is
    // a caller mistake rather than a field that happens not to be there.
    let mut value = 0u64;
    assert_eq!(
        unsafe { hurray_descriptor_field_u64(h, HURRAY_STATS_FIELD_NNZ, 1, &mut value) },
        HURRAY_ERR_INDEX_OUT_OF_BOUNDS
    );
    let mut real = 0f64;
    assert_eq!(
        unsafe { hurray_descriptor_field_f64(h, HURRAY_STATS_FIELD_VALUE_MIN, 1, &mut real) },
        HURRAY_ERR_INDEX_OUT_OF_BOUNDS
    );
    destroy(h);
}

#[test]
fn an_unallocated_section_identifier_is_not_applicable() {
    let bytes = encode(Sections::statistics(full_statistics()));
    let h = decode(&bytes);

    // 0x0003 is the statistics range; 0x00FF is past its last field index.
    // 0x0007 is an unallocated sub-object range.
    for field in [0x0003_00FFu32, 0x0003_0100, 0x0007_0000] {
        u64_field_err(h, field, HURRAY_ERR_FIELD_NOT_APPLICABLE);
    }
    destroy(h);
}
