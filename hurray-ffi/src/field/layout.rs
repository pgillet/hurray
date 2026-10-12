//! Layout-payload fields: identifier range `0x0001_xxxx`.
//!
//! The discriminant byte is the layout tag, so an identifier's applicability is
//! a comparison against the descriptor's own tag and the field space inherits
//! the layout tag space's reserved and private ranges for free (ADR-039 § 3).

use hurray_core::layout::{LayoutDescriptor, TiledLayout};

use super::{HurrayDescriptorField, U64Source};
use crate::status::{
    HurrayStatus, HURRAY_ERR_FIELD_NOT_APPLICABLE, HURRAY_ERR_FIELD_NOT_PRESENT,
    HURRAY_ERR_INDEX_OUT_OF_BOUNDS,
};

// ── Field identifiers ─────────────────────────────────────────────────────────

/// Strided (`0x03`) per-dimension strides, in logical elements. Signed array.
pub const HURRAY_LAYOUT_FIELD_STRIDED_STRIDES: HurrayDescriptorField = 0x0001_0300;

/// Tiled (`0x04`) tile extent per dimension. Unsigned array.
pub const HURRAY_LAYOUT_FIELD_TILED_TILE_SHAPE: HurrayDescriptorField = 0x0001_0400;

/// Tiled (`0x04`) layout tag governing the order of tiles.
pub const HURRAY_LAYOUT_FIELD_TILED_OUTER_LAYOUT: HurrayDescriptorField = 0x0001_0401;

/// Tiled (`0x04`) layout tag governing the order of elements within a tile.
pub const HURRAY_LAYOUT_FIELD_TILED_INNER_LAYOUT: HurrayDescriptorField = 0x0001_0402;

/// Tiled (`0x04`) explicit strides between tiles. Signed array, optional.
pub const HURRAY_LAYOUT_FIELD_TILED_OUTER_STRIDES: HurrayDescriptorField = 0x0001_0403;

/// Tiled (`0x04`) explicit strides within a tile. Signed array, optional.
pub const HURRAY_LAYOUT_FIELD_TILED_INNER_STRIDES: HurrayDescriptorField = 0x0001_0404;

/// Tiled (`0x04`) number of nesting levels from `level` downward.
pub const HURRAY_LAYOUT_FIELD_TILED_DEPTH: HurrayDescriptorField = 0x0001_0405;

/// Morton (`0x05`) per-dimension interleave bit counts. Unsigned array.
pub const HURRAY_LAYOUT_FIELD_MORTON_BITS: HurrayDescriptorField = 0x0001_0500;

/// COO (`0x06`) stored non-zero count.
pub const HURRAY_LAYOUT_FIELD_COO_NNZ: HurrayDescriptorField = 0x0001_0600;

/// COO (`0x06`) whether coordinates are in lexicographic order. `0` or `1`.
pub const HURRAY_LAYOUT_FIELD_COO_IS_SORTED: HurrayDescriptorField = 0x0001_0601;

/// CSR (`0x07`) stored non-zero count.
pub const HURRAY_LAYOUT_FIELD_CSR_NNZ: HurrayDescriptorField = 0x0001_0700;

/// CSC (`0x08`) stored non-zero count.
pub const HURRAY_LAYOUT_FIELD_CSC_NNZ: HurrayDescriptorField = 0x0001_0800;

/// CSF (`0x09`) stored non-zero count.
pub const HURRAY_LAYOUT_FIELD_CSF_NNZ: HurrayDescriptorField = 0x0001_0900;

/// CSF (`0x09`) mode storage order. Unsigned array.
pub const HURRAY_LAYOUT_FIELD_CSF_MODE_ORDER: HurrayDescriptorField = 0x0001_0901;

/// Block-paged (`0x0A`) elements per page along the paged axis.
pub const HURRAY_LAYOUT_FIELD_BLOCK_PAGED_PAGE_SIZE: HurrayDescriptorField = 0x0001_0A00;

/// Block-paged (`0x0A`) total pages in the shared pool.
pub const HURRAY_LAYOUT_FIELD_BLOCK_PAGED_NUM_PAGES: HurrayDescriptorField = 0x0001_0A01;

/// Block-paged (`0x0A`) index of the paged axis.
pub const HURRAY_LAYOUT_FIELD_BLOCK_PAGED_PAGED_AXIS: HurrayDescriptorField = 0x0001_0A02;

/// Block-paged (`0x0A`) number of sequences sharing the pool.
pub const HURRAY_LAYOUT_FIELD_BLOCK_PAGED_NUM_SEQS: HurrayDescriptorField = 0x0001_0A03;

/// Block-paged (`0x0A`) KV role wire byte: key `0x00`, value `0x01`, fused `0x02`.
pub const HURRAY_LAYOUT_FIELD_BLOCK_PAGED_KV_ROLE: HurrayDescriptorField = 0x0001_0A04;

/// Block-paged (`0x0A`) transformer layer index. Optional.
pub const HURRAY_LAYOUT_FIELD_BLOCK_PAGED_LAYER_INDEX: HurrayDescriptorField = 0x0001_0A05;

/// Block-paged (`0x0A`) block-table element width: `u32` `0x00`, `u64` `0x01`.
pub const HURRAY_LAYOUT_FIELD_BLOCK_PAGED_INDEX_TYPE: HurrayDescriptorField = 0x0001_0A06;

/// Composite (`0x0B`) composition rule wire byte.
pub const HURRAY_LAYOUT_FIELD_COMPOSITE_RULE: HurrayDescriptorField = 0x0001_0B00;

/// Composite (`0x0B`) combine-op wire byte; `0x00` unless the rule is overlay.
pub const HURRAY_LAYOUT_FIELD_COMPOSITE_COMBINE_OP: HurrayDescriptorField = 0x0001_0B01;

/// Composite (`0x0B`) number of member descriptors following the head.
pub const HURRAY_LAYOUT_FIELD_COMPOSITE_MEMBER_COUNT: HurrayDescriptorField = 0x0001_0B02;

/// Hilbert (`0x40`) curve order; every dimension equals `2^order`.
pub const HURRAY_LAYOUT_FIELD_HILBERT_ORDER: HurrayDescriptorField = 0x0001_4000;

/// Hilbert (`0x40`) curve rank.
pub const HURRAY_LAYOUT_FIELD_HILBERT_RANK: HurrayDescriptorField = 0x0001_4001;

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

/// Resolves an unsigned scalar field against a layout.
///
/// Matching on the `(field, layout)` pair makes the match arms themselves the
/// applicability check — no separate tag comparison, and no unreachable arm.
pub(super) fn u64_scalar(
    layout: &LayoutDescriptor,
    field: HurrayDescriptorField,
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

/// Resolves an unsigned array field against a layout.
pub(super) fn u64_array<'a>(
    layout: &'a LayoutDescriptor,
    field: HurrayDescriptorField,
    level: u32,
) -> Result<U64Source<'a>, HurrayStatus> {
    check_level(layout, level)?;

    match (field, layout) {
        (HURRAY_LAYOUT_FIELD_TILED_TILE_SHAPE, LayoutDescriptor::Tiled(root)) => {
            Ok(U64Source::Direct(&tiled_at_level(root, level)?.tile_shape))
        }
        (HURRAY_LAYOUT_FIELD_MORTON_BITS, LayoutDescriptor::Morton(l)) => {
            Ok(U64Source::Widened(&l.morton_bits))
        }
        (HURRAY_LAYOUT_FIELD_CSF_MODE_ORDER, LayoutDescriptor::Csf(l)) => {
            Ok(U64Source::Widened(&l.mode_order))
        }
        _ => Err(HURRAY_ERR_FIELD_NOT_APPLICABLE),
    }
}

/// Resolves a signed array field against a layout.
pub(super) fn i64_array(
    layout: &LayoutDescriptor,
    field: HurrayDescriptorField,
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
