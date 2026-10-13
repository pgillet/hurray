//! Optional-section fields: identifier ranges `0x0002_xxxx` through `0x0006_xxxx`.
//!
//! Each of the five optional descriptor sections takes one range (ADR-040).
//! Quantization is the only section with an inner sum type, so it is the only
//! one that uses the discriminant byte — for the scheme tag, exactly as the
//! layout range uses the layout tag. The others carry `0x00` there.
//!
//! None of these sections is required at any conformance level, so an absent
//! section is a normal answer: every read returns
//! [`HURRAY_ERR_FIELD_NOT_PRESENT`] rather than an error when its section is
//! missing.

use hurray_core::{
    descriptor::{ExtensionTypeDescriptor, ShardDescriptor, Statistics, TensorDescriptor},
    QuantizationDescriptor,
};

use super::{HurrayDescriptorField, U64Source};
use crate::status::{
    status_from_core_error, HurrayStatus, HURRAY_ERR_FIELD_NOT_APPLICABLE,
    HURRAY_ERR_FIELD_NOT_PRESENT, HURRAY_ERR_INDEX_OUT_OF_BOUNDS,
};

// ── Quantization (0x0002) ─────────────────────────────────────────────────────

/// Quantization scheme tag of the section this descriptor carries: per-tensor
/// affine `0x01`, per-channel affine `0x02`, per-block affine `0x03`, NF4
/// `0x04`, MXFP `0x05`.
///
/// Read this first: it tells a caller which of the scheme-specific identifiers
/// below apply. The discriminant byte is `0x00`, which is not a valid scheme
/// tag, so the slot addresses the section rather than one scheme within it.
pub const HURRAY_QUANT_FIELD_SCHEME_TAG: HurrayDescriptorField = 0x0002_0000;

/// Per-tensor affine (`0x01`) scale. Real scalar, `float32` on the wire.
pub const HURRAY_QUANT_FIELD_PER_TENSOR_SCALE: HurrayDescriptorField = 0x0002_0100;

/// Per-tensor affine (`0x01`) zero point. Signed scalar, `int32` on the wire.
pub const HURRAY_QUANT_FIELD_PER_TENSOR_ZERO_POINT: HurrayDescriptorField = 0x0002_0101;

/// Per-channel affine (`0x02`) quantization axis.
pub const HURRAY_QUANT_FIELD_PER_CHANNEL_AXIS: HurrayDescriptorField = 0x0002_0200;

/// Per-channel affine (`0x02`) buffer index holding the scale vector.
pub const HURRAY_QUANT_FIELD_PER_CHANNEL_SCALE_BUFFER: HurrayDescriptorField = 0x0002_0201;

/// Per-channel affine (`0x02`) buffer index holding the zero-point vector.
/// Absent for a symmetric scheme.
pub const HURRAY_QUANT_FIELD_PER_CHANNEL_ZERO_POINT_BUFFER: HurrayDescriptorField = 0x0002_0202;

/// Per-channel affine (`0x02`) element type tag of the scale vector.
pub const HURRAY_QUANT_FIELD_PER_CHANNEL_SCALE_TYPE: HurrayDescriptorField = 0x0002_0203;

/// Per-block affine (`0x03`) quantization axis.
pub const HURRAY_QUANT_FIELD_PER_BLOCK_AXIS: HurrayDescriptorField = 0x0002_0300;

/// Per-block affine (`0x03`) elements per quantization block.
pub const HURRAY_QUANT_FIELD_PER_BLOCK_BLOCK_SIZE: HurrayDescriptorField = 0x0002_0301;

/// Per-block affine (`0x03`) buffer index holding the scale vector.
pub const HURRAY_QUANT_FIELD_PER_BLOCK_SCALE_BUFFER: HurrayDescriptorField = 0x0002_0302;

/// Per-block affine (`0x03`) buffer index holding the zero-point vector.
/// Absent for a symmetric scheme.
pub const HURRAY_QUANT_FIELD_PER_BLOCK_ZERO_POINT_BUFFER: HurrayDescriptorField = 0x0002_0303;

/// Per-block affine (`0x03`) element type tag of the scale vector.
pub const HURRAY_QUANT_FIELD_PER_BLOCK_SCALE_TYPE: HurrayDescriptorField = 0x0002_0304;

/// NF4 (`0x04`) quantization axis.
pub const HURRAY_QUANT_FIELD_NF4_AXIS: HurrayDescriptorField = 0x0002_0400;

/// NF4 (`0x04`) elements per quantization block.
pub const HURRAY_QUANT_FIELD_NF4_BLOCK_SIZE: HurrayDescriptorField = 0x0002_0401;

/// NF4 (`0x04`) buffer index holding the scale vector.
pub const HURRAY_QUANT_FIELD_NF4_SCALE_BUFFER: HurrayDescriptorField = 0x0002_0402;

/// MXFP (`0x05`) quantization axis.
pub const HURRAY_QUANT_FIELD_MXFP_AXIS: HurrayDescriptorField = 0x0002_0500;

/// MXFP (`0x05`) elements per quantization block.
pub const HURRAY_QUANT_FIELD_MXFP_BLOCK_SIZE: HurrayDescriptorField = 0x0002_0501;

/// MXFP (`0x05`) buffer index holding the shared-exponent vector.
pub const HURRAY_QUANT_FIELD_MXFP_SCALE_BUFFER: HurrayDescriptorField = 0x0002_0502;

// ── Statistics (0x0003) ───────────────────────────────────────────────────────

/// Raw `computed_mask` bitfield. Always present when the section is.
pub const HURRAY_STATS_FIELD_COMPUTED_MASK: HurrayDescriptorField = 0x0003_0000;

/// Non-zero element count. Guarded by `NNZ_VALID` (bit 0).
pub const HURRAY_STATS_FIELD_NNZ: HurrayDescriptorField = 0x0003_0001;

/// Fraction of elements that are zero. Real scalar, guarded by `SPARSITY_VALID` (bit 1).
pub const HURRAY_STATS_FIELD_SPARSITY_RATIO: HurrayDescriptorField = 0x0003_0002;

/// Smallest stored value. Real scalar, guarded by `VALUE_RANGE_VALID` (bit 2).
pub const HURRAY_STATS_FIELD_VALUE_MIN: HurrayDescriptorField = 0x0003_0003;

/// Largest stored value. Real scalar, guarded by `VALUE_RANGE_VALID` (bit 2).
pub const HURRAY_STATS_FIELD_VALUE_MAX: HurrayDescriptorField = 0x0003_0004;

/// Largest absolute stored value. Real scalar, guarded by `VALUE_RANGE_VALID` (bit 2).
pub const HURRAY_STATS_FIELD_VALUE_ABS_MAX: HurrayDescriptorField = 0x0003_0005;

/// Arithmetic mean. Real scalar, guarded by `VALUE_STATS_VALID` (bit 3).
pub const HURRAY_STATS_FIELD_VALUE_MEAN: HurrayDescriptorField = 0x0003_0006;

/// Standard deviation. Real scalar, guarded by `VALUE_STATS_VALID` (bit 3).
pub const HURRAY_STATS_FIELD_VALUE_STDDEV: HurrayDescriptorField = 0x0003_0007;

/// `N` of N:M structured sparsity. Guarded by `NM_SPARSITY_VALID` (bit 4).
pub const HURRAY_STATS_FIELD_NM_N: HurrayDescriptorField = 0x0003_0008;

/// `M` of N:M structured sparsity. Guarded by `NM_SPARSITY_VALID` (bit 4).
pub const HURRAY_STATS_FIELD_NM_M: HurrayDescriptorField = 0x0003_0009;

/// Whether any element is NaN. `0` or `1`, guarded by `NAN_INF_VALID` (bit 5).
pub const HURRAY_STATS_FIELD_HAS_NAN: HurrayDescriptorField = 0x0003_000A;

/// Whether any element is infinite. `0` or `1`, guarded by `NAN_INF_VALID` (bit 5).
pub const HURRAY_STATS_FIELD_HAS_INF: HurrayDescriptorField = 0x0003_000B;

// ── Shard (0x0004) ────────────────────────────────────────────────────────────

/// Shape of the logical parent tensor. Unsigned array, length equal to the rank.
pub const HURRAY_SHARD_FIELD_PARENT_SHAPE: HurrayDescriptorField = 0x0004_0000;

/// This shard's starting index within the parent. Unsigned array, length equal
/// to the rank.
pub const HURRAY_SHARD_FIELD_SHARD_OFFSET: HurrayDescriptorField = 0x0004_0001;

// ── Extension type (0x0005) ───────────────────────────────────────────────────

/// Width of one element in bits.
pub const HURRAY_EXT_TYPE_FIELD_BIT_WIDTH: HurrayDescriptorField = 0x0005_0000;

/// Elements packed per storage byte; `1` when unpacked.
pub const HURRAY_EXT_TYPE_FIELD_PACKING_FACTOR: HurrayDescriptorField = 0x0005_0001;

/// Whether the type is floating-point. `0` or `1`.
pub const HURRAY_EXT_TYPE_FIELD_IS_FLOAT: HurrayDescriptorField = 0x0005_0002;

/// Whether the type is signed. `0` or `1`.
pub const HURRAY_EXT_TYPE_FIELD_IS_SIGNED: HurrayDescriptorField = 0x0005_0003;

/// Number of sign bits.
pub const HURRAY_EXT_TYPE_FIELD_SIGN_BITS: HurrayDescriptorField = 0x0005_0004;

/// Number of exponent bits; `0` for integer types.
pub const HURRAY_EXT_TYPE_FIELD_EXPONENT_BITS: HurrayDescriptorField = 0x0005_0005;

/// Number of mantissa bits.
pub const HURRAY_EXT_TYPE_FIELD_MANTISSA_BITS: HurrayDescriptorField = 0x0005_0006;

/// Exponent bias; `0` for integer types.
pub const HURRAY_EXT_TYPE_FIELD_EXPONENT_BIAS: HurrayDescriptorField = 0x0005_0007;

/// Whether the type encodes NaN. `0` or `1`.
pub const HURRAY_EXT_TYPE_FIELD_HAS_NAN: HurrayDescriptorField = 0x0005_0008;

/// Whether the type encodes infinity. `0` or `1`.
pub const HURRAY_EXT_TYPE_FIELD_HAS_INF: HurrayDescriptorField = 0x0005_0009;

// ── Composite member (0x0006) ─────────────────────────────────────────────────

/// This tensor's role within its enclosing composite: correction `0x00`,
/// base `0x01`.
pub const HURRAY_COMPOSITE_MEMBER_FIELD_ROLE: HurrayDescriptorField = 0x0006_0000;

// ── Internal resolution ───────────────────────────────────────────────────────

/// Rejects a non-zero `level`.
///
/// Nesting is a layout concern; no section nests. Silently ignoring `level` here
/// would let a caller believe it had read a nested level that cannot exist.
fn check_level(level: u32) -> Result<(), HurrayStatus> {
    if level != 0 {
        return Err(HURRAY_ERR_INDEX_OUT_OF_BOUNDS);
    }
    Ok(())
}

/// Decodes the quantization section, or reports its absence.
///
/// Decoded per call rather than cached in the handle: `hurray-core` stores the
/// section verbatim because typed decode belongs to a layer with schema
/// context, so caching here would move a possible failure into
/// `hurray_descriptor_decode`, which today accepts a descriptor whose
/// quantization bytes it never typed. The payload is at most 24 bytes of
/// fixed-width fields, so the decode is arithmetic, not allocation.
fn quantization(desc: &TensorDescriptor) -> Result<QuantizationDescriptor, HurrayStatus> {
    let bytes = desc
        .quantization
        .as_deref()
        .ok_or(HURRAY_ERR_FIELD_NOT_PRESENT)?;
    QuantizationDescriptor::decode(bytes)
        .map(|(q, _)| q)
        .map_err(|e| status_from_core_error(&e))
}

/// Borrows the statistics section, or reports its absence.
fn statistics(desc: &TensorDescriptor) -> Result<&Statistics, HurrayStatus> {
    desc.statistics.as_ref().ok_or(HURRAY_ERR_FIELD_NOT_PRESENT)
}

/// Borrows the shard section, or reports its absence.
fn shard(desc: &TensorDescriptor) -> Result<&ShardDescriptor, HurrayStatus> {
    desc.shard.as_ref().ok_or(HURRAY_ERR_FIELD_NOT_PRESENT)
}

/// Borrows the extension type section, or reports its absence.
fn ext_type(desc: &TensorDescriptor) -> Result<&ExtensionTypeDescriptor, HurrayStatus> {
    desc.extension_type
        .as_ref()
        .ok_or(HURRAY_ERR_FIELD_NOT_PRESENT)
}

/// Gates a statistic on its `computed_mask` bit.
///
/// The spec requires a reader to check the guard bit before using a statistic.
/// Returning [`HURRAY_ERR_FIELD_NOT_PRESENT`] for a clear bit enforces that
/// instead of advertising it: a C caller cannot read an uncomputed statistic by
/// forgetting to test the mask, which is the whole failure mode of an advisory
/// section whose fields are always physically present.
fn guarded<T>(valid: bool, value: T) -> Result<T, HurrayStatus> {
    if valid {
        Ok(value)
    } else {
        Err(HURRAY_ERR_FIELD_NOT_PRESENT)
    }
}

/// Resolves an unsigned scalar field against a section.
pub(super) fn u64_scalar(
    desc: &TensorDescriptor,
    field: HurrayDescriptorField,
    level: u32,
) -> Result<u64, HurrayStatus> {
    check_level(level)?;

    match field {
        // Quantization
        HURRAY_QUANT_FIELD_SCHEME_TAG => Ok(u64::from(quantization(desc)?.scheme_tag().tag())),
        HURRAY_QUANT_FIELD_PER_CHANNEL_AXIS
        | HURRAY_QUANT_FIELD_PER_CHANNEL_SCALE_BUFFER
        | HURRAY_QUANT_FIELD_PER_CHANNEL_ZERO_POINT_BUFFER
        | HURRAY_QUANT_FIELD_PER_CHANNEL_SCALE_TYPE
        | HURRAY_QUANT_FIELD_PER_BLOCK_AXIS
        | HURRAY_QUANT_FIELD_PER_BLOCK_BLOCK_SIZE
        | HURRAY_QUANT_FIELD_PER_BLOCK_SCALE_BUFFER
        | HURRAY_QUANT_FIELD_PER_BLOCK_ZERO_POINT_BUFFER
        | HURRAY_QUANT_FIELD_PER_BLOCK_SCALE_TYPE
        | HURRAY_QUANT_FIELD_NF4_AXIS
        | HURRAY_QUANT_FIELD_NF4_BLOCK_SIZE
        | HURRAY_QUANT_FIELD_NF4_SCALE_BUFFER
        | HURRAY_QUANT_FIELD_MXFP_AXIS
        | HURRAY_QUANT_FIELD_MXFP_BLOCK_SIZE
        | HURRAY_QUANT_FIELD_MXFP_SCALE_BUFFER => quant_u64(&quantization(desc)?, field),

        // Statistics
        HURRAY_STATS_FIELD_COMPUTED_MASK => Ok(u64::from(statistics(desc)?.computed_mask.0)),
        HURRAY_STATS_FIELD_NNZ => {
            let s = statistics(desc)?;
            guarded(s.computed_mask.nnz_valid(), s.nnz)
        }
        HURRAY_STATS_FIELD_NM_N => {
            let s = statistics(desc)?;
            guarded(s.computed_mask.nm_sparsity_valid(), u64::from(s.nm_n))
        }
        HURRAY_STATS_FIELD_NM_M => {
            let s = statistics(desc)?;
            guarded(s.computed_mask.nm_sparsity_valid(), u64::from(s.nm_m))
        }
        HURRAY_STATS_FIELD_HAS_NAN => {
            let s = statistics(desc)?;
            guarded(s.computed_mask.nan_inf_valid(), u64::from(s.has_nan))
        }
        HURRAY_STATS_FIELD_HAS_INF => {
            let s = statistics(desc)?;
            guarded(s.computed_mask.nan_inf_valid(), u64::from(s.has_inf))
        }

        // Extension type
        HURRAY_EXT_TYPE_FIELD_BIT_WIDTH => Ok(u64::from(ext_type(desc)?.bit_width)),
        HURRAY_EXT_TYPE_FIELD_PACKING_FACTOR => Ok(u64::from(ext_type(desc)?.packing_factor)),
        HURRAY_EXT_TYPE_FIELD_IS_FLOAT => Ok(u64::from(ext_type(desc)?.is_float)),
        HURRAY_EXT_TYPE_FIELD_IS_SIGNED => Ok(u64::from(ext_type(desc)?.is_signed)),
        HURRAY_EXT_TYPE_FIELD_SIGN_BITS => Ok(u64::from(ext_type(desc)?.sign_bits)),
        HURRAY_EXT_TYPE_FIELD_EXPONENT_BITS => Ok(u64::from(ext_type(desc)?.exponent_bits)),
        HURRAY_EXT_TYPE_FIELD_MANTISSA_BITS => Ok(u64::from(ext_type(desc)?.mantissa_bits)),
        HURRAY_EXT_TYPE_FIELD_EXPONENT_BIAS => Ok(u64::from(ext_type(desc)?.exponent_bias)),
        HURRAY_EXT_TYPE_FIELD_HAS_NAN => Ok(u64::from(ext_type(desc)?.has_nan)),
        HURRAY_EXT_TYPE_FIELD_HAS_INF => Ok(u64::from(ext_type(desc)?.has_inf)),

        // Composite member
        HURRAY_COMPOSITE_MEMBER_FIELD_ROLE => desc
            .composite_member
            .as_ref()
            .map(|cm| u64::from(cm.member_role.wire_byte()))
            .ok_or(HURRAY_ERR_FIELD_NOT_PRESENT),

        _ => Err(HURRAY_ERR_FIELD_NOT_APPLICABLE),
    }
}

/// Resolves a scheme-specific unsigned quantization field.
///
/// Split out so the `(field, scheme)` pairing stays one match: the scheme is the
/// identifier's discriminant byte, so a mismatch falls through to
/// [`HURRAY_ERR_FIELD_NOT_APPLICABLE`] with no separate tag comparison.
fn quant_u64(
    q: &QuantizationDescriptor,
    field: HurrayDescriptorField,
) -> Result<u64, HurrayStatus> {
    match (field, q) {
        // Per-channel affine
        (HURRAY_QUANT_FIELD_PER_CHANNEL_AXIS, QuantizationDescriptor::PerChannelAffine(s)) => {
            Ok(u64::from(s.axis()))
        }
        (
            HURRAY_QUANT_FIELD_PER_CHANNEL_SCALE_BUFFER,
            QuantizationDescriptor::PerChannelAffine(s),
        ) => Ok(u64::from(s.scale_buffer_index())),
        (
            HURRAY_QUANT_FIELD_PER_CHANNEL_ZERO_POINT_BUFFER,
            QuantizationDescriptor::PerChannelAffine(s),
        ) => s
            .zero_point_buffer_index()
            .map(u64::from)
            .ok_or(HURRAY_ERR_FIELD_NOT_PRESENT),
        (
            HURRAY_QUANT_FIELD_PER_CHANNEL_SCALE_TYPE,
            QuantizationDescriptor::PerChannelAffine(s),
        ) => Ok(u64::from(s.scale_type().tag())),

        // Per-block affine
        (HURRAY_QUANT_FIELD_PER_BLOCK_AXIS, QuantizationDescriptor::PerBlockAffine(s)) => {
            Ok(u64::from(s.axis()))
        }
        (HURRAY_QUANT_FIELD_PER_BLOCK_BLOCK_SIZE, QuantizationDescriptor::PerBlockAffine(s)) => {
            Ok(u64::from(s.block_size()))
        }
        (HURRAY_QUANT_FIELD_PER_BLOCK_SCALE_BUFFER, QuantizationDescriptor::PerBlockAffine(s)) => {
            Ok(u64::from(s.scale_buffer_index()))
        }
        (
            HURRAY_QUANT_FIELD_PER_BLOCK_ZERO_POINT_BUFFER,
            QuantizationDescriptor::PerBlockAffine(s),
        ) => s
            .zero_point_buffer_index()
            .map(u64::from)
            .ok_or(HURRAY_ERR_FIELD_NOT_PRESENT),
        (HURRAY_QUANT_FIELD_PER_BLOCK_SCALE_TYPE, QuantizationDescriptor::PerBlockAffine(s)) => {
            Ok(u64::from(s.scale_type().tag()))
        }

        // NF4
        (HURRAY_QUANT_FIELD_NF4_AXIS, QuantizationDescriptor::Nf4(s)) => Ok(u64::from(s.axis())),
        (HURRAY_QUANT_FIELD_NF4_BLOCK_SIZE, QuantizationDescriptor::Nf4(s)) => {
            Ok(u64::from(s.block_size()))
        }
        (HURRAY_QUANT_FIELD_NF4_SCALE_BUFFER, QuantizationDescriptor::Nf4(s)) => {
            Ok(u64::from(s.scale_buffer_index()))
        }

        // MXFP
        (HURRAY_QUANT_FIELD_MXFP_AXIS, QuantizationDescriptor::Mxfp(s)) => Ok(u64::from(s.axis())),
        (HURRAY_QUANT_FIELD_MXFP_BLOCK_SIZE, QuantizationDescriptor::Mxfp(s)) => {
            Ok(u64::from(s.block_size()))
        }
        (HURRAY_QUANT_FIELD_MXFP_SCALE_BUFFER, QuantizationDescriptor::Mxfp(s)) => {
            Ok(u64::from(s.scale_buffer_index()))
        }

        _ => Err(HURRAY_ERR_FIELD_NOT_APPLICABLE),
    }
}

/// Resolves a signed scalar field against a section.
pub(super) fn i64_scalar(
    desc: &TensorDescriptor,
    field: HurrayDescriptorField,
    level: u32,
) -> Result<i64, HurrayStatus> {
    check_level(level)?;

    match field {
        HURRAY_QUANT_FIELD_PER_TENSOR_ZERO_POINT => match quantization(desc)? {
            QuantizationDescriptor::PerTensorAffine(s) => Ok(i64::from(s.zero_point())),
            _ => Err(HURRAY_ERR_FIELD_NOT_APPLICABLE),
        },
        _ => Err(HURRAY_ERR_FIELD_NOT_APPLICABLE),
    }
}

/// Resolves a real scalar field against a section.
pub(super) fn f64_scalar(
    desc: &TensorDescriptor,
    field: HurrayDescriptorField,
    level: u32,
) -> Result<f64, HurrayStatus> {
    check_level(level)?;

    match field {
        HURRAY_QUANT_FIELD_PER_TENSOR_SCALE => match quantization(desc)? {
            QuantizationDescriptor::PerTensorAffine(s) => Ok(f64::from(s.scale())),
            _ => Err(HURRAY_ERR_FIELD_NOT_APPLICABLE),
        },

        HURRAY_STATS_FIELD_SPARSITY_RATIO => {
            let s = statistics(desc)?;
            guarded(s.computed_mask.sparsity_valid(), s.sparsity_ratio)
        }
        HURRAY_STATS_FIELD_VALUE_MIN => {
            let s = statistics(desc)?;
            guarded(s.computed_mask.value_range_valid(), s.value_min)
        }
        HURRAY_STATS_FIELD_VALUE_MAX => {
            let s = statistics(desc)?;
            guarded(s.computed_mask.value_range_valid(), s.value_max)
        }
        HURRAY_STATS_FIELD_VALUE_ABS_MAX => {
            let s = statistics(desc)?;
            guarded(s.computed_mask.value_range_valid(), s.value_abs_max)
        }
        HURRAY_STATS_FIELD_VALUE_MEAN => {
            let s = statistics(desc)?;
            guarded(s.computed_mask.value_stats_valid(), s.value_mean)
        }
        HURRAY_STATS_FIELD_VALUE_STDDEV => {
            let s = statistics(desc)?;
            guarded(s.computed_mask.value_stats_valid(), s.value_stddev)
        }

        _ => Err(HURRAY_ERR_FIELD_NOT_APPLICABLE),
    }
}

/// Resolves an unsigned array field against a section.
pub(super) fn u64_array<'a>(
    desc: &'a TensorDescriptor,
    field: HurrayDescriptorField,
    level: u32,
) -> Result<U64Source<'a>, HurrayStatus> {
    check_level(level)?;

    match field {
        HURRAY_SHARD_FIELD_PARENT_SHAPE => Ok(U64Source::Direct(&shard(desc)?.parent_shape)),
        HURRAY_SHARD_FIELD_SHARD_OFFSET => Ok(U64Source::Direct(&shard(desc)?.shard_offset)),
        _ => Err(HURRAY_ERR_FIELD_NOT_APPLICABLE),
    }
}
