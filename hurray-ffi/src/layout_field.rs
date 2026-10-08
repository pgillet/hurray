//! Layout-payload field reads for the Hurray C ABI (ADR-039).
//!
//! [`hurray_descriptor_decode`](crate::hurray_descriptor_decode) validates every
//! layout `hurray-core` supports, but the scalar accessors beside it stop at the
//! layout *tag*. A C caller handed a CSR descriptor would learn `layout_tag ==
//! 0x07` and nothing more — not `nnz`, not which buffer is `row_ptr` — so it
//! could not address a single element. This module is what closes that gap, and
//! with it Compliance Level 1 for `hurray-ffi`.
//!
//! ## Three functions, not twenty-six
//!
//! The layout payloads hold about twenty-six fields between them. Rather than one
//! accessor each, a field is named by a [`HurrayLayoutField`] identifier and read
//! through one of three typed functions:
//!
//! | Function | Reads |
//! |---|---|
//! | [`hurray_descriptor_layout_field_u64`] | every scalar field |
//! | [`hurray_descriptor_layout_field_u64_array`] | `tile_shape`, `morton_bits`, `mode_order` |
//! | [`hurray_descriptor_layout_field_i64_array`] | `strides`, `outer_strides`, `inner_strides` |
//!
//! A new layout adds identifier constants, not symbols — which is how the wire
//! format itself evolves (ADR-019), applied to the ABI. The trade is that a C
//! compiler cannot check a field against the layout in hand; a mismatch is
//! [`HURRAY_ERR_FIELD_NOT_APPLICABLE`] at runtime instead.
//!
//! ## Reading one
//!
//! Scalars are a single call. Arrays use the capacity/length in-out pattern
//! [`hurray_descriptor_shape`](crate::descriptor::hurray_descriptor_shape)
//! established: pass the capacity in `*out_len`, read the true length back out,
//! and retry after [`HURRAY_ERR_BUFFER_TOO_SMALL`].

use std::ptr;

use hurray_core::layout::{LayoutDescriptor, TiledLayout};

use crate::{
    descriptor::HurrayDescriptor,
    panic::{catch, null_check},
    status::{
        HurrayStatus, HURRAY_ERR_BUFFER_TOO_SMALL, HURRAY_ERR_FIELD_NOT_APPLICABLE,
        HURRAY_ERR_FIELD_NOT_PRESENT, HURRAY_ERR_INDEX_OUT_OF_BOUNDS, HURRAY_OK,
    },
};

// ── Field identifier ──────────────────────────────────────────────────────────

/// Identifies one field of a tensor descriptor's layout payload.
///
/// A plain `uint32_t` rather than a `#[repr(C)]` enum: a C caller can pass any
/// integer, and an out-of-range value materialised as a Rust enum is undefined
/// behaviour. Validation happens in Rust, which is sound for every input.
///
/// The encoding is `0x0001_0000 | (layout_tag << 8) | field_index`, so an
/// identifier carries the layout it belongs to and the applicability check is a
/// comparison rather than a table. `0x0001_xxxx` is the layout-field range;
/// each optional descriptor section takes its own range when it is implemented.
///
/// Generated headers render these in decimal, so to decode one by hand: CSR
/// `nnz` is `0x0001_0700` = 67328 — range `0x0001`, layout tag `0x07`, field
/// index `0x00`.
pub type HurrayLayoutField = u32;

/// Strided (`0x03`) per-dimension strides, in logical elements. Signed array.
pub const HURRAY_LAYOUT_FIELD_STRIDED_STRIDES: HurrayLayoutField = 0x0001_0300;

/// Tiled (`0x04`) tile extent per dimension. Unsigned array.
pub const HURRAY_LAYOUT_FIELD_TILED_TILE_SHAPE: HurrayLayoutField = 0x0001_0400;

/// Tiled (`0x04`) layout tag governing the order of tiles.
pub const HURRAY_LAYOUT_FIELD_TILED_OUTER_LAYOUT: HurrayLayoutField = 0x0001_0401;

/// Tiled (`0x04`) layout tag governing the order of elements within a tile.
pub const HURRAY_LAYOUT_FIELD_TILED_INNER_LAYOUT: HurrayLayoutField = 0x0001_0402;

/// Tiled (`0x04`) explicit strides between tiles. Signed array, optional.
pub const HURRAY_LAYOUT_FIELD_TILED_OUTER_STRIDES: HurrayLayoutField = 0x0001_0403;

/// Tiled (`0x04`) explicit strides within a tile. Signed array, optional.
pub const HURRAY_LAYOUT_FIELD_TILED_INNER_STRIDES: HurrayLayoutField = 0x0001_0404;

/// Tiled (`0x04`) number of nesting levels from `level` downward.
pub const HURRAY_LAYOUT_FIELD_TILED_DEPTH: HurrayLayoutField = 0x0001_0405;

/// Morton (`0x05`) per-dimension interleave bit counts. Unsigned array.
pub const HURRAY_LAYOUT_FIELD_MORTON_BITS: HurrayLayoutField = 0x0001_0500;

/// COO (`0x06`) stored non-zero count.
pub const HURRAY_LAYOUT_FIELD_COO_NNZ: HurrayLayoutField = 0x0001_0600;

/// COO (`0x06`) whether coordinates are in lexicographic order. `0` or `1`.
pub const HURRAY_LAYOUT_FIELD_COO_IS_SORTED: HurrayLayoutField = 0x0001_0601;

/// CSR (`0x07`) stored non-zero count.
pub const HURRAY_LAYOUT_FIELD_CSR_NNZ: HurrayLayoutField = 0x0001_0700;

/// CSC (`0x08`) stored non-zero count.
pub const HURRAY_LAYOUT_FIELD_CSC_NNZ: HurrayLayoutField = 0x0001_0800;

/// CSF (`0x09`) stored non-zero count.
pub const HURRAY_LAYOUT_FIELD_CSF_NNZ: HurrayLayoutField = 0x0001_0900;

/// CSF (`0x09`) mode storage order. Unsigned array.
pub const HURRAY_LAYOUT_FIELD_CSF_MODE_ORDER: HurrayLayoutField = 0x0001_0901;

/// Block-paged (`0x0A`) elements per page along the paged axis.
pub const HURRAY_LAYOUT_FIELD_BLOCK_PAGED_PAGE_SIZE: HurrayLayoutField = 0x0001_0A00;

/// Block-paged (`0x0A`) total pages in the shared pool.
pub const HURRAY_LAYOUT_FIELD_BLOCK_PAGED_NUM_PAGES: HurrayLayoutField = 0x0001_0A01;

/// Block-paged (`0x0A`) index of the paged axis.
pub const HURRAY_LAYOUT_FIELD_BLOCK_PAGED_PAGED_AXIS: HurrayLayoutField = 0x0001_0A02;

/// Block-paged (`0x0A`) number of sequences sharing the pool.
pub const HURRAY_LAYOUT_FIELD_BLOCK_PAGED_NUM_SEQS: HurrayLayoutField = 0x0001_0A03;

/// Block-paged (`0x0A`) KV role wire byte: key `0x00`, value `0x01`, fused `0x02`.
pub const HURRAY_LAYOUT_FIELD_BLOCK_PAGED_KV_ROLE: HurrayLayoutField = 0x0001_0A04;

/// Block-paged (`0x0A`) transformer layer index. Optional.
pub const HURRAY_LAYOUT_FIELD_BLOCK_PAGED_LAYER_INDEX: HurrayLayoutField = 0x0001_0A05;

/// Block-paged (`0x0A`) block-table element width: `u32` `0x00`, `u64` `0x01`.
pub const HURRAY_LAYOUT_FIELD_BLOCK_PAGED_INDEX_TYPE: HurrayLayoutField = 0x0001_0A06;

/// Composite (`0x0B`) composition rule wire byte.
pub const HURRAY_LAYOUT_FIELD_COMPOSITE_RULE: HurrayLayoutField = 0x0001_0B00;

/// Composite (`0x0B`) combine-op wire byte; `0x00` unless the rule is overlay.
pub const HURRAY_LAYOUT_FIELD_COMPOSITE_COMBINE_OP: HurrayLayoutField = 0x0001_0B01;

/// Composite (`0x0B`) number of member descriptors following the head.
pub const HURRAY_LAYOUT_FIELD_COMPOSITE_MEMBER_COUNT: HurrayLayoutField = 0x0001_0B02;

/// Hilbert (`0x40`) curve order; every dimension equals `2^order`.
pub const HURRAY_LAYOUT_FIELD_HILBERT_ORDER: HurrayLayoutField = 0x0001_4000;

/// Hilbert (`0x40`) curve rank.
pub const HURRAY_LAYOUT_FIELD_HILBERT_RANK: HurrayLayoutField = 0x0001_4001;

// ── Internal resolution ───────────────────────────────────────────────────────

/// Walks `level` steps down a tiled layout's nesting chain.
fn tiled_at_level(root: &TiledLayout, level: u32) -> Result<&TiledLayout, HurrayStatus> {
    let mut current = root;
    for _ in 0..level {
        current = match current.inner_tiled.as_deref() {
            Some(inner) => inner,
            None => return Err(HURRAY_ERR_INDEX_OUT_OF_BOUNDS),
        };
    }
    Ok(current)
}

/// Counts nesting levels from `tile` downward, including `tile` itself.
fn tiled_depth(tile: &TiledLayout) -> u64 {
    let mut depth = 1;
    let mut current = tile;
    while let Some(inner) = current.inner_tiled.as_deref() {
        depth += 1;
        current = inner;
    }
    depth
}

/// Rejects a non-zero `level` against a layout that does not nest.
///
/// Checked once up front rather than in every arm: `level` is meaningful only
/// for tiled layouts, and silently ignoring it elsewhere would let a caller
/// believe it had read a nested level that does not exist.
fn check_level(layout: &LayoutDescriptor, level: u32) -> Result<(), HurrayStatus> {
    if level != 0 && !matches!(layout, LayoutDescriptor::Tiled(_)) {
        return Err(HURRAY_ERR_INDEX_OUT_OF_BOUNDS);
    }
    Ok(())
}

/// Resolves a scalar field against a layout.
///
/// Matching on the `(field, layout)` pair makes the match arms themselves the
/// applicability check — no separate tag comparison, and no unreachable arm.
fn scalar(
    layout: &LayoutDescriptor,
    field: HurrayLayoutField,
    level: u32,
) -> Result<u64, HurrayStatus> {
    check_level(layout, level)?;

    match (field, layout) {
        // Tiled
        (HURRAY_LAYOUT_FIELD_TILED_OUTER_LAYOUT, LayoutDescriptor::Tiled(root)) => {
            Ok(tiled_at_level(root, level)?.outer_layout as u64)
        }
        (HURRAY_LAYOUT_FIELD_TILED_INNER_LAYOUT, LayoutDescriptor::Tiled(root)) => {
            Ok(tiled_at_level(root, level)?.inner_layout as u64)
        }
        (HURRAY_LAYOUT_FIELD_TILED_DEPTH, LayoutDescriptor::Tiled(root)) => {
            Ok(tiled_depth(tiled_at_level(root, level)?))
        }

        // Sparse
        (HURRAY_LAYOUT_FIELD_COO_NNZ, LayoutDescriptor::Coo(l)) => Ok(l.nnz),
        (HURRAY_LAYOUT_FIELD_COO_IS_SORTED, LayoutDescriptor::Coo(l)) => Ok(l.is_sorted as u64),
        (HURRAY_LAYOUT_FIELD_CSR_NNZ, LayoutDescriptor::Csr(l)) => Ok(l.nnz),
        (HURRAY_LAYOUT_FIELD_CSC_NNZ, LayoutDescriptor::Csc(l)) => Ok(l.nnz),
        (HURRAY_LAYOUT_FIELD_CSF_NNZ, LayoutDescriptor::Csf(l)) => Ok(l.nnz),

        // Block-paged
        (HURRAY_LAYOUT_FIELD_BLOCK_PAGED_PAGE_SIZE, LayoutDescriptor::BlockPaged(l)) => {
            Ok(l.page_size as u64)
        }
        (HURRAY_LAYOUT_FIELD_BLOCK_PAGED_NUM_PAGES, LayoutDescriptor::BlockPaged(l)) => {
            Ok(l.num_pages)
        }
        (HURRAY_LAYOUT_FIELD_BLOCK_PAGED_PAGED_AXIS, LayoutDescriptor::BlockPaged(l)) => {
            Ok(l.paged_axis as u64)
        }
        (HURRAY_LAYOUT_FIELD_BLOCK_PAGED_NUM_SEQS, LayoutDescriptor::BlockPaged(l)) => {
            Ok(l.num_seqs as u64)
        }
        (HURRAY_LAYOUT_FIELD_BLOCK_PAGED_KV_ROLE, LayoutDescriptor::BlockPaged(l)) => {
            Ok(l.kv_role.wire_byte() as u64)
        }
        (HURRAY_LAYOUT_FIELD_BLOCK_PAGED_LAYER_INDEX, LayoutDescriptor::BlockPaged(l)) => l
            .layer_index
            .map(u64::from)
            .ok_or(HURRAY_ERR_FIELD_NOT_PRESENT),
        (HURRAY_LAYOUT_FIELD_BLOCK_PAGED_INDEX_TYPE, LayoutDescriptor::BlockPaged(l)) => {
            Ok(l.block_table_index_type.wire_byte() as u64)
        }

        // Composite
        (HURRAY_LAYOUT_FIELD_COMPOSITE_RULE, LayoutDescriptor::Composite(l)) => {
            Ok(l.rule.rule_byte() as u64)
        }
        (HURRAY_LAYOUT_FIELD_COMPOSITE_COMBINE_OP, LayoutDescriptor::Composite(l)) => {
            Ok(l.rule.combine_op_byte() as u64)
        }
        (HURRAY_LAYOUT_FIELD_COMPOSITE_MEMBER_COUNT, LayoutDescriptor::Composite(l)) => {
            Ok(l.member_count as u64)
        }

        // Hilbert
        (HURRAY_LAYOUT_FIELD_HILBERT_ORDER, LayoutDescriptor::Hilbert(l)) => {
            Ok(l.hilbert_order as u64)
        }
        (HURRAY_LAYOUT_FIELD_HILBERT_RANK, LayoutDescriptor::Hilbert(l)) => {
            Ok(l.hilbert_rank as u64)
        }

        _ => Err(HURRAY_ERR_FIELD_NOT_APPLICABLE),
    }
}

/// An unsigned array field's source, borrowed from the descriptor.
///
/// `morton_bits` and `mode_order` are `u32` on the wire while `tile_shape` is
/// `u64`, so the widening cannot be a `memcpy` for all three. Keeping the source
/// borrowed and widening on write avoids allocating a temporary for a field a
/// caller may only be measuring.
enum U64Field<'a> {
    Direct(&'a [u64]),
    Widened(&'a [u32]),
}

impl U64Field<'_> {
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

/// Resolves an unsigned array field against a layout.
fn u64_array<'a>(
    layout: &'a LayoutDescriptor,
    field: HurrayLayoutField,
    level: u32,
) -> Result<U64Field<'a>, HurrayStatus> {
    check_level(layout, level)?;

    match (field, layout) {
        (HURRAY_LAYOUT_FIELD_TILED_TILE_SHAPE, LayoutDescriptor::Tiled(root)) => {
            Ok(U64Field::Direct(&tiled_at_level(root, level)?.tile_shape))
        }
        (HURRAY_LAYOUT_FIELD_MORTON_BITS, LayoutDescriptor::Morton(l)) => {
            Ok(U64Field::Widened(&l.morton_bits))
        }
        (HURRAY_LAYOUT_FIELD_CSF_MODE_ORDER, LayoutDescriptor::Csf(l)) => {
            Ok(U64Field::Widened(&l.mode_order))
        }
        _ => Err(HURRAY_ERR_FIELD_NOT_APPLICABLE),
    }
}

/// Resolves a signed array field against a layout.
fn i64_array(
    layout: &LayoutDescriptor,
    field: HurrayLayoutField,
    level: u32,
) -> Result<&[i64], HurrayStatus> {
    check_level(layout, level)?;

    match (field, layout) {
        (HURRAY_LAYOUT_FIELD_STRIDED_STRIDES, LayoutDescriptor::Strided(l)) => Ok(&l.strides),
        (HURRAY_LAYOUT_FIELD_TILED_OUTER_STRIDES, LayoutDescriptor::Tiled(root)) => {
            match tiled_at_level(root, level)?.outer_strides.as_ref() {
                Some(s) => Ok(&s.strides),
                None => Err(HURRAY_ERR_FIELD_NOT_PRESENT),
            }
        }
        (HURRAY_LAYOUT_FIELD_TILED_INNER_STRIDES, LayoutDescriptor::Tiled(root)) => {
            match tiled_at_level(root, level)?.inner_strides.as_ref() {
                Some(s) => Ok(&s.strides),
                None => Err(HURRAY_ERR_FIELD_NOT_PRESENT),
            }
        }
        _ => Err(HURRAY_ERR_FIELD_NOT_APPLICABLE),
    }
}

// ── Scalar accessor ───────────────────────────────────────────────────────────

/// Reads a scalar field of the descriptor's layout payload.
///
/// Every scalar payload field in the format is unsigned or boolean, so all of
/// them widen losslessly into `uint64_t`: a boolean reads as `0` or `1`, and a
/// wire tag byte reads as its documented value.
///
/// # Arguments
///
/// - `handle` — Descriptor created by
///   [`hurray_descriptor_decode`](crate::hurray_descriptor_decode).
/// - `field` — One of the `HURRAY_LAYOUT_FIELD_*` scalar constants.
/// - `level` — Tiled nesting level to read; MUST be `0` for every other layout.
/// - `out_value` — Non-null pointer; written only on [`HURRAY_OK`].
///
/// # Errors
///
/// - [`HURRAY_ERR_FIELD_NOT_APPLICABLE`] — `field` does not belong to this
///   descriptor's layout, or is not a scalar field.
/// - [`HURRAY_ERR_FIELD_NOT_PRESENT`] — the field is optional and this
///   descriptor does not carry it.
/// - [`HURRAY_ERR_INDEX_OUT_OF_BOUNDS`] — `level` exceeds the nesting depth, or
///   is non-zero for a layout that does not nest.
///
/// # Safety
///
/// `handle` and `out_value` MUST be valid, non-null pointers, and `handle` MUST
/// NOT have been destroyed.
#[no_mangle]
pub unsafe extern "C" fn hurray_descriptor_layout_field_u64(
    handle: *const HurrayDescriptor,
    field: HurrayLayoutField,
    level: u32,
    out_value: *mut u64,
) -> HurrayStatus {
    catch(|| {
        null_check!(handle, out_value);

        // SAFETY: handle is non-null and points to a live HurrayDescriptor.
        match scalar(&(*handle).inner().layout, field, level) {
            Ok(value) => {
                // SAFETY: out_value is non-null and writable for one u64.
                *out_value = value;
                HURRAY_OK
            }
            Err(status) => status,
        }
    })
}

// ── Array accessors ───────────────────────────────────────────────────────────

/// Reads an unsigned array field of the descriptor's layout payload.
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
/// As [`hurray_descriptor_layout_field_u64`], plus
/// [`HURRAY_ERR_BUFFER_TOO_SMALL`] per the protocol above.
///
/// # Safety
///
/// - `handle` and `out_len` MUST be valid, non-null pointers.
/// - If `out_values` is non-null it MUST be writable for `capacity` × 8 bytes,
///   where `capacity` is the value at `*out_len` on entry.
#[no_mangle]
pub unsafe extern "C" fn hurray_descriptor_layout_field_u64_array(
    handle: *const HurrayDescriptor,
    field: HurrayLayoutField,
    level: u32,
    out_values: *mut u64,
    out_len: *mut usize,
) -> HurrayStatus {
    catch(|| {
        // out_values may be null (capacity-query mode); out_len is always required.
        null_check!(handle, out_len);

        // SAFETY: handle is non-null and points to a live HurrayDescriptor.
        let source = match u64_array(&(*handle).inner().layout, field, level) {
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

/// Reads a signed array field of the descriptor's layout payload.
///
/// Signed because strides may be negative (reversed) or zero (broadcast), which
/// an unsigned accessor would misreport. The capacity/length protocol is the one
/// [`hurray_descriptor_layout_field_u64_array`] documents.
///
/// # Errors
///
/// As [`hurray_descriptor_layout_field_u64_array`].
///
/// # Safety
///
/// - `handle` and `out_len` MUST be valid, non-null pointers.
/// - If `out_values` is non-null it MUST be writable for `capacity` × 8 bytes,
///   where `capacity` is the value at `*out_len` on entry.
#[no_mangle]
pub unsafe extern "C" fn hurray_descriptor_layout_field_i64_array(
    handle: *const HurrayDescriptor,
    field: HurrayLayoutField,
    level: u32,
    out_values: *mut i64,
    out_len: *mut usize,
) -> HurrayStatus {
    catch(|| {
        // out_values may be null (capacity-query mode); out_len is always required.
        null_check!(handle, out_len);

        // SAFETY: handle is non-null and points to a live HurrayDescriptor.
        let values = match i64_array(&(*handle).inner().layout, field, level) {
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
