//! Descriptor field reads for the Hurray C ABI (ADR-039, ADR-040).
//!
//! [`hurray_descriptor_decode`](crate::hurray_descriptor_decode) validates every
//! layout and every optional section `hurray-core` supports, but the scalar
//! accessors beside it stop at the fixed header. A C caller handed a CSR
//! descriptor would learn `layout_tag == 0x07` and nothing more — not `nnz`, not
//! which buffer is `row_ptr` — and one handed a per-block-quantized tensor would
//! learn nothing at all about the scheme. This module is what closes that gap.
//!
//! ## Five functions, not seventy
//!
//! The layout payloads and the five optional sections hold about seventy fields
//! between them. Rather than one accessor each, a field is named by a
//! [`HurrayDescriptorField`] identifier and read through one of five typed
//! functions:
//!
//! | Function | Reads |
//! |---|---|
//! | [`hurray_descriptor_field_u64`] | every unsigned or boolean scalar |
//! | [`hurray_descriptor_field_i64`] | signed scalars (`zero_point`) |
//! | [`hurray_descriptor_field_f64`] | real scalars (`scale`, the statistics) |
//! | [`hurray_descriptor_field_u64_array`] | `tile_shape`, `morton_bits`, `mode_order`, `parent_shape`, `shard_offset` |
//! | [`hurray_descriptor_field_i64_array`] | `strides`, `outer_strides`, `inner_strides` |
//!
//! A new layout or section adds identifier constants, not symbols — which is how
//! the wire format itself evolves (ADR-019), applied to the ABI. The trade is
//! that a C compiler cannot check a field against the sub-object in hand; a
//! mismatch is [`HURRAY_ERR_FIELD_NOT_APPLICABLE`] at runtime instead.
//!
//! ## Reading one
//!
//! Scalars are a single call. Arrays use the capacity/length in-out pattern
//! [`hurray_descriptor_shape`](crate::descriptor::hurray_descriptor_shape)
//! established: pass the capacity in `*out_len`, read the true length back out,
//! and retry after [`HURRAY_ERR_BUFFER_TOO_SMALL`].

mod layout;
mod section;

use std::ptr;

use hurray_core::descriptor::TensorDescriptor;

pub use layout::{
    HURRAY_LAYOUT_FIELD_BLOCK_PAGED_INDEX_TYPE, HURRAY_LAYOUT_FIELD_BLOCK_PAGED_KV_ROLE,
    HURRAY_LAYOUT_FIELD_BLOCK_PAGED_LAYER_INDEX, HURRAY_LAYOUT_FIELD_BLOCK_PAGED_NUM_PAGES,
    HURRAY_LAYOUT_FIELD_BLOCK_PAGED_NUM_SEQS, HURRAY_LAYOUT_FIELD_BLOCK_PAGED_PAGED_AXIS,
    HURRAY_LAYOUT_FIELD_BLOCK_PAGED_PAGE_SIZE, HURRAY_LAYOUT_FIELD_COMPOSITE_COMBINE_OP,
    HURRAY_LAYOUT_FIELD_COMPOSITE_MEMBER_COUNT, HURRAY_LAYOUT_FIELD_COMPOSITE_RULE,
    HURRAY_LAYOUT_FIELD_COO_IS_SORTED, HURRAY_LAYOUT_FIELD_COO_NNZ, HURRAY_LAYOUT_FIELD_CSC_NNZ,
    HURRAY_LAYOUT_FIELD_CSF_MODE_ORDER, HURRAY_LAYOUT_FIELD_CSF_NNZ, HURRAY_LAYOUT_FIELD_CSR_NNZ,
    HURRAY_LAYOUT_FIELD_HILBERT_ORDER, HURRAY_LAYOUT_FIELD_HILBERT_RANK,
    HURRAY_LAYOUT_FIELD_MORTON_BITS, HURRAY_LAYOUT_FIELD_STRIDED_STRIDES,
    HURRAY_LAYOUT_FIELD_TILED_DEPTH, HURRAY_LAYOUT_FIELD_TILED_INNER_LAYOUT,
    HURRAY_LAYOUT_FIELD_TILED_INNER_STRIDES, HURRAY_LAYOUT_FIELD_TILED_OUTER_LAYOUT,
    HURRAY_LAYOUT_FIELD_TILED_OUTER_STRIDES, HURRAY_LAYOUT_FIELD_TILED_TILE_SHAPE,
};
pub use section::{
    HURRAY_COMPOSITE_MEMBER_FIELD_ROLE, HURRAY_EXT_TYPE_FIELD_BIT_WIDTH,
    HURRAY_EXT_TYPE_FIELD_EXPONENT_BIAS, HURRAY_EXT_TYPE_FIELD_EXPONENT_BITS,
    HURRAY_EXT_TYPE_FIELD_HAS_INF, HURRAY_EXT_TYPE_FIELD_HAS_NAN, HURRAY_EXT_TYPE_FIELD_IS_FLOAT,
    HURRAY_EXT_TYPE_FIELD_IS_SIGNED, HURRAY_EXT_TYPE_FIELD_MANTISSA_BITS,
    HURRAY_EXT_TYPE_FIELD_PACKING_FACTOR, HURRAY_EXT_TYPE_FIELD_SIGN_BITS,
    HURRAY_QUANT_FIELD_MXFP_AXIS, HURRAY_QUANT_FIELD_MXFP_BLOCK_SIZE,
    HURRAY_QUANT_FIELD_MXFP_SCALE_BUFFER, HURRAY_QUANT_FIELD_NF4_AXIS,
    HURRAY_QUANT_FIELD_NF4_BLOCK_SIZE, HURRAY_QUANT_FIELD_NF4_SCALE_BUFFER,
    HURRAY_QUANT_FIELD_PER_BLOCK_AXIS, HURRAY_QUANT_FIELD_PER_BLOCK_BLOCK_SIZE,
    HURRAY_QUANT_FIELD_PER_BLOCK_SCALE_BUFFER, HURRAY_QUANT_FIELD_PER_BLOCK_SCALE_TYPE,
    HURRAY_QUANT_FIELD_PER_BLOCK_ZERO_POINT_BUFFER, HURRAY_QUANT_FIELD_PER_CHANNEL_AXIS,
    HURRAY_QUANT_FIELD_PER_CHANNEL_SCALE_BUFFER, HURRAY_QUANT_FIELD_PER_CHANNEL_SCALE_TYPE,
    HURRAY_QUANT_FIELD_PER_CHANNEL_ZERO_POINT_BUFFER, HURRAY_QUANT_FIELD_PER_TENSOR_SCALE,
    HURRAY_QUANT_FIELD_PER_TENSOR_ZERO_POINT, HURRAY_QUANT_FIELD_SCHEME_TAG,
    HURRAY_SHARD_FIELD_PARENT_SHAPE, HURRAY_SHARD_FIELD_SHARD_OFFSET,
    HURRAY_STATS_FIELD_COMPUTED_MASK, HURRAY_STATS_FIELD_HAS_INF, HURRAY_STATS_FIELD_HAS_NAN,
    HURRAY_STATS_FIELD_NM_M, HURRAY_STATS_FIELD_NM_N, HURRAY_STATS_FIELD_NNZ,
    HURRAY_STATS_FIELD_SPARSITY_RATIO, HURRAY_STATS_FIELD_VALUE_ABS_MAX,
    HURRAY_STATS_FIELD_VALUE_MAX, HURRAY_STATS_FIELD_VALUE_MEAN, HURRAY_STATS_FIELD_VALUE_MIN,
    HURRAY_STATS_FIELD_VALUE_STDDEV,
};

use crate::{
    descriptor::HurrayDescriptor,
    panic::{catch, null_check},
    status::{
        HurrayStatus, HURRAY_ERR_BUFFER_TOO_SMALL, HURRAY_ERR_FIELD_NOT_APPLICABLE, HURRAY_OK,
    },
};

// ── Field identifier ──────────────────────────────────────────────────────────

/// Identifies one field of a tensor descriptor's layout payload or of one of its
/// optional sections.
///
/// A plain `uint32_t` rather than a `#[repr(C)]` enum: a C caller can pass any
/// integer, and an out-of-range value materialised as a Rust enum is undefined
/// behaviour. Validation happens in Rust, which is sound for every input.
///
/// The encoding is
///
/// ```text
/// field_id = (range << 16) | (discriminant << 8) | field_index
/// ```
///
/// so an identifier carries the sub-object it belongs to and the applicability
/// check is a comparison rather than a table.
///
/// | Range | Sub-object | Discriminant byte |
/// |---|---|---|
/// | `0x0001` | layout payload | layout tag |
/// | `0x0002` | quantization | scheme tag; `0x00` addresses the section itself |
/// | `0x0003` | statistics | `0x00` |
/// | `0x0004` | shard | `0x00` |
/// | `0x0005` | extension type | `0x00` |
/// | `0x0006` | composite member | `0x00` |
///
/// Only quantization has an inner sum type, so it is the only section that uses
/// the discriminant byte; [`HURRAY_QUANT_FIELD_SCHEME_TAG`] reads the scheme a
/// descriptor carries, which a caller needs before it can pick applicable
/// identifiers. `0x0000_0000` is never a valid identifier, so a zeroed variable
/// is caught rather than silently read as the first field.
///
/// Generated headers render these in decimal, so to decode one by hand: CSR
/// `nnz` is `0x0001_0700` = 67328 — range `0x0001`, layout tag `0x07`, field
/// index `0x00`.
pub type HurrayDescriptorField = u32;

/// Layout-payload range: `0x0001_xxxx`.
const RANGE_LAYOUT: u32 = 0x0001;

/// Section ranges, contiguous: quantization `0x0002` through composite member
/// `0x0006`. Dispatch only needs the bounds; each section's own range is the
/// high half-word of its constants in [`section`].
const RANGE_SECTION_FIRST: u32 = 0x0002;
const RANGE_SECTION_LAST: u32 = 0x0006;

/// Extracts the sub-object range from a field identifier.
#[inline]
fn range_of(field: HurrayDescriptorField) -> u32 {
    field >> 16
}

// ── Shared array source ───────────────────────────────────────────────────────

/// An unsigned array field's source, borrowed from the descriptor.
///
/// `morton_bits` and `mode_order` are `u32` on the wire while `tile_shape` and
/// the shard vectors are `u64`, so the widening cannot be a `memcpy` for all of
/// them. Keeping the source borrowed and widening on write avoids allocating a
/// temporary for a field a caller may only be measuring.
pub(crate) enum U64Source<'a> {
    Direct(&'a [u64]),
    Widened(&'a [u32]),
}

impl U64Source<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Direct(s) => s.len(),
            Self::Widened(s) => s.len(),
        }
    }

    fn write_to(&self, dst: &mut [u64]) {
        match self {
            Self::Direct(s) => dst.copy_from_slice(s),
            Self::Widened(s) => {
                for (d, v) in dst.iter_mut().zip(s.iter()) {
                    *d = u64::from(*v);
                }
            }
        }
    }
}

// ── Dispatch ──────────────────────────────────────────────────────────────────

/// Routes an unsigned scalar read to the sub-object its identifier names.
fn u64_scalar(
    desc: &TensorDescriptor,
    field: HurrayDescriptorField,
    level: u32,
) -> Result<u64, HurrayStatus> {
    match range_of(field) {
        RANGE_LAYOUT => layout::u64_scalar(&desc.layout, field, level),
        RANGE_SECTION_FIRST..=RANGE_SECTION_LAST => section::u64_scalar(desc, field, level),
        _ => Err(HURRAY_ERR_FIELD_NOT_APPLICABLE),
    }
}

/// Routes an unsigned array read to the sub-object its identifier names.
fn u64_array<'a>(
    desc: &'a TensorDescriptor,
    field: HurrayDescriptorField,
    level: u32,
) -> Result<U64Source<'a>, HurrayStatus> {
    match range_of(field) {
        RANGE_LAYOUT => layout::u64_array(&desc.layout, field, level),
        RANGE_SECTION_FIRST..=RANGE_SECTION_LAST => section::u64_array(desc, field, level),
        _ => Err(HURRAY_ERR_FIELD_NOT_APPLICABLE),
    }
}

// ── Scalar accessors ──────────────────────────────────────────────────────────

/// Reads an unsigned or boolean scalar field of the descriptor.
///
/// Covers every scalar field that is unsigned or boolean, which is all of them
/// but two: a boolean reads as `0` or `1`, and a wire tag byte — `kv_role`,
/// `block_table_index_type`, `scale_type`, `member_role`, the quantization
/// scheme tag — reads as its documented value. The two exceptions are
/// [`hurray_descriptor_field_i64`] and [`hurray_descriptor_field_f64`].
///
/// # Arguments
///
/// - `handle` — Descriptor created by
///   [`hurray_descriptor_decode`](crate::hurray_descriptor_decode).
/// - `field` — One of the `HURRAY_*_FIELD_*` unsigned scalar constants.
/// - `level` — Tiled nesting level to read; MUST be `0` for every other layout
///   and for every section field.
/// - `out_value` — Non-null pointer; written only on [`HURRAY_OK`].
///
/// # Errors
///
/// - [`HURRAY_ERR_FIELD_NOT_APPLICABLE`] — `field` does not belong to this
///   descriptor's layout or quantization scheme, or is not an unsigned scalar.
/// - [`HURRAY_ERR_FIELD_NOT_PRESENT`](crate::status::HURRAY_ERR_FIELD_NOT_PRESENT) — the field is optional and this
///   descriptor does not carry it: an absent section, an unset optional field,
///   or a statistic whose `computed_mask` guard bit is clear.
/// - [`HURRAY_ERR_INDEX_OUT_OF_BOUNDS`](crate::status::HURRAY_ERR_INDEX_OUT_OF_BOUNDS) — `level` exceeds the nesting depth, or
///   is non-zero for a field that does not nest.
///
/// # Safety
///
/// `handle` and `out_value` MUST be valid, non-null pointers, and `handle` MUST
/// NOT have been destroyed.
#[no_mangle]
pub unsafe extern "C" fn hurray_descriptor_field_u64(
    handle: *const HurrayDescriptor,
    field: HurrayDescriptorField,
    level: u32,
    out_value: *mut u64,
) -> HurrayStatus {
    catch(|| {
        null_check!(handle, out_value);

        // SAFETY: handle is non-null and points to a live HurrayDescriptor.
        match u64_scalar((*handle).inner(), field, level) {
            Ok(value) => {
                // SAFETY: out_value is non-null and writable for one u64.
                *out_value = value;
                HURRAY_OK
            }
            Err(status) => status,
        }
    })
}

/// Reads a signed scalar field of the descriptor.
///
/// Only per-tensor affine `zero_point` is signed; it is an `int32` on the wire
/// and widens losslessly. Separate from [`hurray_descriptor_field_u64`] because
/// an unsigned accessor would report a negative zero point as a huge positive
/// number, which is exactly the kind of silent corruption a quantized read
/// cannot absorb.
///
/// # Errors
///
/// As [`hurray_descriptor_field_u64`], with
/// [`HURRAY_ERR_FIELD_NOT_APPLICABLE`] for any field that is not a signed
/// scalar.
///
/// # Safety
///
/// `handle` and `out_value` MUST be valid, non-null pointers, and `handle` MUST
/// NOT have been destroyed.
#[no_mangle]
pub unsafe extern "C" fn hurray_descriptor_field_i64(
    handle: *const HurrayDescriptor,
    field: HurrayDescriptorField,
    level: u32,
    out_value: *mut i64,
) -> HurrayStatus {
    catch(|| {
        null_check!(handle, out_value);

        // Signed scalars exist only in the sections; no layout field is signed.
        if range_of(field) == RANGE_LAYOUT {
            return HURRAY_ERR_FIELD_NOT_APPLICABLE;
        }

        // SAFETY: handle is non-null and points to a live HurrayDescriptor.
        match section::i64_scalar((*handle).inner(), field, level) {
            Ok(value) => {
                // SAFETY: out_value is non-null and writable for one i64.
                *out_value = value;
                HURRAY_OK
            }
            Err(status) => status,
        }
    })
}

/// Reads a real scalar field of the descriptor.
///
/// Covers per-tensor affine `scale` — a `float32` on the wire, widened — and the
/// six `float64` statistics. A statistic whose `computed_mask` guard bit is
/// clear returns [`HURRAY_ERR_FIELD_NOT_PRESENT`](crate::status::HURRAY_ERR_FIELD_NOT_PRESENT) rather than the stored
/// placeholder, so a caller cannot read an uncomputed value by forgetting to
/// test the mask.
///
/// # Errors
///
/// As [`hurray_descriptor_field_u64`], with
/// [`HURRAY_ERR_FIELD_NOT_APPLICABLE`] for any field that is not a real scalar.
///
/// # Safety
///
/// `handle` and `out_value` MUST be valid, non-null pointers, and `handle` MUST
/// NOT have been destroyed.
#[no_mangle]
pub unsafe extern "C" fn hurray_descriptor_field_f64(
    handle: *const HurrayDescriptor,
    field: HurrayDescriptorField,
    level: u32,
    out_value: *mut f64,
) -> HurrayStatus {
    catch(|| {
        null_check!(handle, out_value);

        // Real scalars exist only in the sections; every layout field is integral.
        if range_of(field) == RANGE_LAYOUT {
            return HURRAY_ERR_FIELD_NOT_APPLICABLE;
        }

        // SAFETY: handle is non-null and points to a live HurrayDescriptor.
        match section::f64_scalar((*handle).inner(), field, level) {
            Ok(value) => {
                // SAFETY: out_value is non-null and writable for one f64.
                *out_value = value;
                HURRAY_OK
            }
            Err(status) => status,
        }
    })
}

// ── Array accessors ───────────────────────────────────────────────────────────

/// Reads an unsigned array field of the descriptor.
///
/// Uses the capacity/length in-out pattern
/// [`hurray_descriptor_shape`](crate::descriptor::hurray_descriptor_shape) uses:
///
/// 1. Set `*out_len` to the capacity of `out_values` in elements, or `0` if
///    `out_values` is null.
/// 2. On return `*out_len` always holds the field's true length.
/// 3. A null `out_values` or an insufficient capacity returns
///    [`HURRAY_ERR_BUFFER_TOO_SMALL`], so the caller can allocate and retry.
///
/// # Errors
///
/// As [`hurray_descriptor_field_u64`], plus [`HURRAY_ERR_BUFFER_TOO_SMALL`] per
/// the protocol above.
///
/// # Safety
///
/// - `handle` and `out_len` MUST be valid, non-null pointers.
/// - If `out_values` is non-null it MUST be writable for `capacity` × 8 bytes,
///   where `capacity` is the value at `*out_len` on entry.
#[no_mangle]
pub unsafe extern "C" fn hurray_descriptor_field_u64_array(
    handle: *const HurrayDescriptor,
    field: HurrayDescriptorField,
    level: u32,
    out_values: *mut u64,
    out_len: *mut usize,
) -> HurrayStatus {
    catch(|| {
        // out_values may be null (capacity-query mode); out_len is always required.
        null_check!(handle, out_len);

        // SAFETY: handle is non-null and points to a live HurrayDescriptor.
        let source = match u64_array((*handle).inner(), field, level) {
            Ok(s) => s,
            Err(status) => return status,
        };

        let true_len = source.len();
        // SAFETY: out_len is non-null and writable; read the caller's capacity
        // before overwriting it with the true length.
        let capacity = *out_len;
        *out_len = true_len;

        if out_values.is_null() || capacity < true_len {
            return HURRAY_ERR_BUFFER_TOO_SMALL;
        }

        // SAFETY: out_values is non-null and writable for at least true_len
        // elements, checked immediately above.
        source.write_to(std::slice::from_raw_parts_mut(out_values, true_len));
        HURRAY_OK
    })
}

/// Reads a signed array field of the descriptor.
///
/// Signed because strides may be negative (reversed) or zero (broadcast), which
/// an unsigned accessor would misreport. The capacity/length protocol is the one
/// [`hurray_descriptor_field_u64_array`] documents.
///
/// # Errors
///
/// As [`hurray_descriptor_field_u64_array`].
///
/// # Safety
///
/// - `handle` and `out_len` MUST be valid, non-null pointers.
/// - If `out_values` is non-null it MUST be writable for `capacity` × 8 bytes,
///   where `capacity` is the value at `*out_len` on entry.
#[no_mangle]
pub unsafe extern "C" fn hurray_descriptor_field_i64_array(
    handle: *const HurrayDescriptor,
    field: HurrayDescriptorField,
    level: u32,
    out_values: *mut i64,
    out_len: *mut usize,
) -> HurrayStatus {
    catch(|| {
        // out_values may be null (capacity-query mode); out_len is always required.
        null_check!(handle, out_len);

        // Signed arrays exist only in the layout payload; no section has one.
        if range_of(field) != RANGE_LAYOUT {
            return HURRAY_ERR_FIELD_NOT_APPLICABLE;
        }

        // SAFETY: handle is non-null and points to a live HurrayDescriptor.
        let values = match layout::i64_array(&(*handle).inner().layout, field, level) {
            Ok(v) => v,
            Err(status) => return status,
        };

        let true_len = values.len();
        // SAFETY: out_len is non-null and writable; read the caller's capacity
        // before overwriting it with the true length.
        let capacity = *out_len;
        *out_len = true_len;

        if out_values.is_null() || capacity < true_len {
            return HURRAY_ERR_BUFFER_TOO_SMALL;
        }

        // SAFETY: out_values is non-null and writable for at least true_len
        // elements, checked immediately above.
        ptr::copy_nonoverlapping(values.as_ptr(), out_values, true_len);
        HURRAY_OK
    })
}
