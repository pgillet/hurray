//! Per-layout-tag encode/decode dispatch for the layout-specific fields section.
//!
//! Wire formats per layout tag (spec § Layout-Specific Fields):
//!
//! - `0x01` (RowMajor), `0x02` (ColMajor): no payload
//! - `0x03` (Strided): `strides int64[rank]`
//! - `0x04` (Tiled): `tile_shape u64[rank]`, `outer_layout u8`, `inner_layout u8`,
//!   `_reserved u8[2]`, then conditionally outer_strides / inner_strides / inner_tiled
//! - `0x05` (Morton): `morton_bits u32[rank]`
//! - `0x06` (Subpaving): `region_count u32`, then per-region fields
//! - `0x07` (COO): `nnz u64`, `is_sorted u8`, `_reserved u8[7]`
//! - `0x08` (CSR): `nnz u64`, `_reserved u8[8]`
//! - `0x09` (CSC): `nnz u64`, `_reserved u8[8]`
//! - `0x0A` (CSF): `nnz u64`, `mode_order u32[rank]`, `_reserved u8[8]`
//! - `0x40` (Hilbert): `hilbert_order u32`, `hilbert_rank u32`
//! - `0xF0`–`0xFE` (PrivateExtension): `extension_layout_id u64`, `extension_data_length u32`,
//!   `extension_data bytes[len]`

use crate::descriptor::cursor::{ByteCursor, ByteWriter};
use crate::layout::{
    BlockPagedLayout, BlockTableIndexType, CooLayout, CscLayout, CsfLayout, CsrLayout,
    HilbertLayout, InnerStrides, KvRole, LayoutDescriptor, MortonLayout, OuterStrides,
    PrivateExtensionLayout, RegionDescriptor, StridedLayout, SubpavingLayout, TiledLayout,
    MAX_TILED_DEPTH, TAG_BLOCK_PAGED, TAG_COL_MAJOR, TAG_COO, TAG_CSC, TAG_CSF, TAG_CSR,
    TAG_HILBERT, TAG_MORTON, TAG_ROW_MAJOR, TAG_STRIDED, TAG_SUBPAVING, TAG_TILED,
};
use crate::{Error, Result};

// ── Public entry points ───────────────────────────────────────────────────────

/// Encodes the layout-specific payload for `layout` into `w`.
///
/// The layout tag byte itself is NOT written here — it is part of the fixed
/// header written by the caller (`encode.rs`).
pub(crate) fn encode_layout_payload(
    layout: &LayoutDescriptor,
    rank: u32,
    w: &mut ByteWriter,
) -> Result<()> {
    // Top-level callers always start at depth 0; recursion uses the internal
    // depth-aware variant so the MAX_TILED_DEPTH guard fires on the encode path.
    encode_layout_payload_at_depth(layout, rank, w, 0)
}

/// Internal depth-aware dispatcher.  `depth` mirrors the same counter used in
/// `decode_layout_payload` so the recursion guard applies symmetrically to
/// both encode and decode.
fn encode_layout_payload_at_depth(
    layout: &LayoutDescriptor,
    rank: u32,
    w: &mut ByteWriter,
    depth: u8,
) -> Result<()> {
    match layout {
        LayoutDescriptor::RowMajor | LayoutDescriptor::ColMajor => {
            // No additional payload.
            Ok(())
        }
        LayoutDescriptor::Strided(s) => encode_strided(s, w),
        LayoutDescriptor::Tiled(t) => encode_tiled(t, rank, w, depth),
        LayoutDescriptor::Morton(m) => encode_morton(m, w),
        LayoutDescriptor::Subpaving(sp) => encode_subpaving(sp, rank, w, depth),
        LayoutDescriptor::Coo(c) => encode_coo(c, w),
        LayoutDescriptor::Csr(c) => encode_csr(c, w),
        LayoutDescriptor::Csc(c) => encode_csc(c, w),
        LayoutDescriptor::Csf(c) => encode_csf(c, rank, w),
        LayoutDescriptor::BlockPaged(bp) => encode_block_paged(bp, w),
        LayoutDescriptor::Hilbert(h) => encode_hilbert(h, w),
        LayoutDescriptor::PrivateExtension(p) => encode_private_extension(p, w),
        LayoutDescriptor::Unknown(_) => {
            // Unknown layouts cannot be re-encoded in strict mode.
            Err(Error::UnknownLayoutTag(layout.tag()))
        }
    }
}

/// Decodes the layout-specific payload for `tag` from `cursor`, returning a
/// fully constructed [`LayoutDescriptor`].
///
/// `depth` tracks recursion level; callers at the top level pass `0`.
pub(crate) fn decode_layout_payload(
    tag: u8,
    rank: u32,
    cursor: &mut ByteCursor<'_>,
    depth: u8,
) -> Result<LayoutDescriptor> {
    if depth >= MAX_TILED_DEPTH as u8 {
        return Err(Error::SubpavingNestingTooDeep);
    }
    match tag {
        TAG_ROW_MAJOR => Ok(LayoutDescriptor::RowMajor),
        TAG_COL_MAJOR => Ok(LayoutDescriptor::ColMajor),
        TAG_STRIDED => decode_strided(cursor, rank),
        TAG_TILED => Ok(LayoutDescriptor::Tiled(Box::new(decode_tiled(
            cursor, rank, depth,
        )?))),
        TAG_MORTON => decode_morton(cursor, rank),
        TAG_SUBPAVING => decode_subpaving(cursor, rank, depth),
        TAG_COO => decode_coo(cursor),
        TAG_CSR => decode_csr(cursor),
        TAG_CSC => decode_csc(cursor),
        TAG_CSF => decode_csf(cursor, rank),
        TAG_BLOCK_PAGED => decode_block_paged(cursor),
        TAG_HILBERT => decode_hilbert(cursor),
        0xF0..=0xFE => decode_private_extension(tag, cursor),
        0x00 | 0xFF => Err(Error::InvalidLayoutTag(tag)),
        t if crate::layout::is_reserved_tag(t) => Err(Error::ReservedLayoutTag(tag)),
        _ => Err(Error::UnknownLayoutTag(tag)),
    }
}

// ── Strided ───────────────────────────────────────────────────────────────────

fn encode_strided(layout: &StridedLayout, w: &mut ByteWriter) -> Result<()> {
    for &stride in &layout.strides {
        w.write_i64_le(stride);
    }
    Ok(())
}

fn decode_strided(cursor: &mut ByteCursor<'_>, rank: u32) -> Result<LayoutDescriptor> {
    let mut strides = Vec::with_capacity(rank as usize);
    for _ in 0..rank {
        strides.push(cursor.read_i64_le()?);
    }
    Ok(LayoutDescriptor::Strided(StridedLayout::new(strides)))
}

// ── Tiled ─────────────────────────────────────────────────────────────────────

// `rank` is passed through to recursive calls of encode_tiled; not used directly in this level.
#[allow(clippy::only_used_in_recursion)]
fn encode_tiled(layout: &TiledLayout, rank: u32, w: &mut ByteWriter, depth: u8) -> Result<()> {
    if depth >= MAX_TILED_DEPTH as u8 {
        return Err(Error::SubpavingNestingTooDeep);
    }
    // tile_shape: uint64[rank]
    for &dim in &layout.tile_shape {
        w.write_u64_le(dim);
    }
    w.write_u8(layout.outer_layout);
    w.write_u8(layout.inner_layout);
    w.write_zeros(2); // _reserved

    // Conditional: outer_strides if outer_layout == 0x03.
    // The stride-length invariant (strides.len() == rank) is maintained by TiledLayout::new,
    // so no padding is needed here.
    if layout.outer_layout == TAG_STRIDED {
        if let Some(os) = &layout.outer_strides {
            for &s in &os.strides {
                w.write_i64_le(s);
            }
        }
    }

    // Conditional: inner_strides if inner_layout == 0x03
    if layout.inner_layout == TAG_STRIDED {
        if let Some(is) = &layout.inner_strides {
            for &s in &is.strides {
                w.write_i64_le(s);
            }
        }
    }

    // Conditional: recurse if inner_layout == 0x04
    if layout.inner_layout == TAG_TILED {
        if let Some(inner) = &layout.inner_tiled {
            encode_tiled(inner, rank, w, depth + 1)?;
        }
    }

    Ok(())
}

// Returns TiledLayout directly so call sites can wrap it without an unreachable!()
// match arm on the LayoutDescriptor::Tiled variant.
fn decode_tiled(cursor: &mut ByteCursor<'_>, rank: u32, depth: u8) -> Result<TiledLayout> {
    if depth >= MAX_TILED_DEPTH as u8 {
        return Err(Error::SubpavingNestingTooDeep);
    }

    // tile_shape: uint64[rank]
    let mut tile_shape = Vec::with_capacity(rank as usize);
    for _ in 0..rank {
        tile_shape.push(cursor.read_u64_le()?);
    }

    let outer_layout = cursor.read_u8()?;
    let inner_layout = cursor.read_u8()?;
    let reserved = cursor.read_bytes(2)?;
    if reserved != [0u8, 0] {
        return Err(Error::ReservedBytesNonZero {
            field: "tiled._reserved",
        });
    }

    // Conditional: outer_strides if outer_layout == strided
    let outer_strides = if outer_layout == TAG_STRIDED {
        let mut strides = Vec::with_capacity(rank as usize);
        for _ in 0..rank {
            strides.push(cursor.read_i64_le()?);
        }
        Some(OuterStrides::new(strides))
    } else {
        None
    };

    // Conditional: inner_strides if inner_layout == strided
    let inner_strides = if inner_layout == TAG_STRIDED {
        let mut strides = Vec::with_capacity(rank as usize);
        for _ in 0..rank {
            strides.push(cursor.read_i64_le()?);
        }
        Some(InnerStrides::new(strides))
    } else {
        None
    };

    // Conditional: recursive inner tiled if inner_layout == tiled
    let inner_tiled: Option<Box<TiledLayout>> = if inner_layout == TAG_TILED {
        Some(Box::new(decode_tiled(cursor, rank, depth + 1)?))
    } else {
        None
    };

    TiledLayout::new(
        tile_shape,
        outer_layout,
        inner_layout,
        outer_strides,
        inner_strides,
        inner_tiled,
    )
}

// ── Morton ────────────────────────────────────────────────────────────────────

fn encode_morton(layout: &MortonLayout, w: &mut ByteWriter) -> Result<()> {
    for &bits in &layout.morton_bits {
        w.write_u32_le(bits);
    }
    Ok(())
}

fn decode_morton(cursor: &mut ByteCursor<'_>, rank: u32) -> Result<LayoutDescriptor> {
    let mut bits = Vec::with_capacity(rank as usize);
    for _ in 0..rank {
        bits.push(cursor.read_u32_le()?);
    }
    let layout = MortonLayout::new(bits)?;
    Ok(LayoutDescriptor::Morton(layout))
}

// ── Subpaving ─────────────────────────────────────────────────────────────────

fn encode_subpaving(
    layout: &SubpavingLayout,
    rank: u32,
    w: &mut ByteWriter,
    depth: u8,
) -> Result<()> {
    if depth >= MAX_TILED_DEPTH as u8 {
        return Err(Error::SubpavingNestingTooDeep);
    }
    w.write_u32_le(layout.regions.len() as u32);
    for region in &layout.regions {
        encode_region(region, rank, w, depth)?;
    }
    Ok(())
}

fn encode_region(
    region: &RegionDescriptor,
    rank: u32,
    w: &mut ByteWriter,
    depth: u8,
) -> Result<()> {
    // origin uint64[rank]
    for &v in &region.origin {
        w.write_u64_le(v);
    }
    // region_shape uint64[rank]
    for &v in &region.region_shape {
        w.write_u64_le(v);
    }
    w.write_u8(region.region_layout_tag);
    w.write_zeros(3); // _reserved
    w.write_u32_le(region.buffer_index);
    w.write_u64_le(region.region_byte_offset);

    // Encode region_layout_length and region_layout_payload per ADR-015.
    // Row-major (0x01) and col-major (0x02) have no additional fields; all other
    // tags carry their layout-specific fields in inner_layout.
    match region.region_layout_tag {
        TAG_ROW_MAJOR | TAG_COL_MAJOR => {
            // No inner layout payload for these two tags.
            w.write_u32_le(0u32);
        }
        _ => {
            // Every other tag requires an inner_layout payload. A missing inner_layout
            // here is a producer bug — the RegionDescriptor was constructed without
            // calling with_inner_layout() for a tag that mandates it.
            let inner = region.inner_layout.as_deref().ok_or_else(|| {
                Error::InvalidLayout(format!(
                    "region_layout_tag 0x{:02X} requires an inner_layout payload, but none is present",
                    region.region_layout_tag
                ))
            })?;

            // Subpaving regions that are themselves subpavings increment depth to match
            // the depth accounting in decode_region.  The incremented depth is passed
            // through encode_layout_payload_at_depth so that encode_subpaving's
            // MAX_TILED_DEPTH guard fires on the encode path, not only on decode.
            let inner_depth = if region.region_layout_tag == TAG_SUBPAVING {
                depth + 1
            } else {
                depth
            };

            // Encode the inner layout's payload into a temporary buffer, then write
            // the length-prefixed bytes. A temporary Vec avoids seek-back on the
            // main writer, which ByteWriter does not support.
            let mut payload_w = ByteWriter::new();
            encode_layout_payload_at_depth(inner, rank, &mut payload_w, inner_depth)?;
            let payload = payload_w.into_vec();

            let payload_len = u32::try_from(payload.len()).map_err(|_| {
                Error::InvalidLayout(format!(
                    "inner layout payload for region_layout_tag 0x{:02X} exceeds u32::MAX bytes ({})",
                    region.region_layout_tag,
                    payload.len()
                ))
            })?;
            w.write_u32_le(payload_len);
            w.write_bytes(&payload);
        }
    }

    Ok(())
}

fn decode_subpaving(cursor: &mut ByteCursor<'_>, rank: u32, depth: u8) -> Result<LayoutDescriptor> {
    if depth >= MAX_TILED_DEPTH as u8 {
        return Err(Error::SubpavingNestingTooDeep);
    }
    let region_count = cursor.read_u32_le()?;
    if region_count == 0 {
        return Err(Error::InvalidLayout(
            "subpaving region_count must be > 0".to_string(),
        ));
    }
    let mut regions = Vec::with_capacity(region_count as usize);
    for _ in 0..region_count {
        regions.push(decode_region(cursor, rank, depth)?);
    }
    let layout = SubpavingLayout::new(regions)?;
    Ok(LayoutDescriptor::Subpaving(layout))
}

fn decode_region(cursor: &mut ByteCursor<'_>, rank: u32, depth: u8) -> Result<RegionDescriptor> {
    let rank_us = rank as usize;

    // origin uint64[rank]
    let mut origin = Vec::with_capacity(rank_us);
    for _ in 0..rank {
        origin.push(cursor.read_u64_le()?);
    }

    // region_shape uint64[rank]
    let mut region_shape = Vec::with_capacity(rank_us);
    for _ in 0..rank {
        region_shape.push(cursor.read_u64_le()?);
    }

    let region_layout_tag = cursor.read_u8()?;
    let reserved = cursor.read_bytes(3)?;
    if reserved != [0u8, 0, 0] {
        return Err(Error::ReservedBytesNonZero {
            field: "subpaving_region._reserved",
        });
    }
    let buffer_index = cursor.read_u32_le()?;
    let region_byte_offset = cursor.read_u64_le()?;
    let region_layout_length = cursor.read_u32_le()?;

    // Decode the inner layout payload if present.
    // Row-major (0x01) and col-major (0x02) have no additional fields; all other
    // tags require recursive decode. Recursive subpaving (0x06) increments depth
    // so the MAX_TILED_DEPTH guard in decode_layout_payload fires correctly.
    let base = RegionDescriptor::new(
        origin,
        region_shape,
        region_layout_tag,
        buffer_index,
        region_byte_offset,
    )?;

    if region_layout_length == 0 || matches!(region_layout_tag, TAG_ROW_MAJOR | TAG_COL_MAJOR) {
        // No inner layout payload to decode.
        Ok(base)
    } else {
        let payload = cursor.read_bytes(region_layout_length as usize)?.to_vec();
        let mut sub = ByteCursor::new(&payload, payload.len());
        // Subpaving regions that are themselves subpavings increment depth.
        let inner_depth = if region_layout_tag == TAG_SUBPAVING {
            depth + 1
        } else {
            depth
        };
        let inner_layout = decode_layout_payload(region_layout_tag, rank, &mut sub, inner_depth)?;
        base.with_inner_layout(inner_layout)
    }
}

// ── COO ───────────────────────────────────────────────────────────────────────

fn encode_coo(layout: &CooLayout, w: &mut ByteWriter) -> Result<()> {
    w.write_u64_le(layout.nnz);
    w.write_u8(u8::from(layout.is_sorted));
    w.write_zeros(7); // _reserved
    Ok(())
}

fn decode_coo(cursor: &mut ByteCursor<'_>) -> Result<LayoutDescriptor> {
    let nnz = cursor.read_u64_le()?;
    let is_sorted = cursor.read_u8()? != 0;
    let reserved = cursor.read_bytes(7)?;
    if reserved.iter().any(|&b| b != 0) {
        return Err(Error::ReservedBytesNonZero {
            field: "coo._reserved",
        });
    }
    Ok(LayoutDescriptor::Coo(CooLayout::new(nnz, is_sorted)))
}

// ── CSR ───────────────────────────────────────────────────────────────────────

fn encode_csr(layout: &CsrLayout, w: &mut ByteWriter) -> Result<()> {
    w.write_u64_le(layout.nnz);
    w.write_zeros(8); // _reserved
    Ok(())
}

fn decode_csr(cursor: &mut ByteCursor<'_>) -> Result<LayoutDescriptor> {
    let nnz = cursor.read_u64_le()?;
    let reserved = cursor.read_bytes(8)?;
    if reserved.iter().any(|&b| b != 0) {
        return Err(Error::ReservedBytesNonZero {
            field: "csr._reserved",
        });
    }
    Ok(LayoutDescriptor::Csr(CsrLayout::new(nnz)))
}

// ── CSC ───────────────────────────────────────────────────────────────────────

fn encode_csc(layout: &CscLayout, w: &mut ByteWriter) -> Result<()> {
    w.write_u64_le(layout.nnz);
    w.write_zeros(8); // _reserved
    Ok(())
}

fn decode_csc(cursor: &mut ByteCursor<'_>) -> Result<LayoutDescriptor> {
    let nnz = cursor.read_u64_le()?;
    let reserved = cursor.read_bytes(8)?;
    if reserved.iter().any(|&b| b != 0) {
        return Err(Error::ReservedBytesNonZero {
            field: "csc._reserved",
        });
    }
    Ok(LayoutDescriptor::Csc(CscLayout::new(nnz)))
}

// ── CSF ───────────────────────────────────────────────────────────────────────

/// Field order for CSF (spec `docs/spec/layouts/csf.md § Additional Descriptor Fields`,
/// all little-endian):
///
/// | Field         | Wire type       | Bytes          |
/// |---------------|-----------------|----------------|
/// | `nnz`         | `uint64`        | 8              |
/// | `mode_order`  | `uint32[rank]`  | `4 * rank`     |
/// | `_reserved`   | `uint8[8]`      | 8              |
fn encode_csf(layout: &CsfLayout, rank: u32, w: &mut ByteWriter) -> Result<()> {
    w.write_u64_le(layout.nnz);
    // mode_order.len() IS the authoritative rank; it MUST agree with the caller's
    // `rank` (derived from shape.rank()). Fail fast on a mismatch rather than padding
    // or over-writing — either would emit a wire payload the decoder mis-frames.
    let mo_len = layout.mode_order.len() as u32;
    if mo_len != rank {
        return Err(Error::InvalidLayout(format!(
            "csf encode: mode_order.len() ({mo_len}) != rank ({rank}); \
             call validate_against_shape before encoding"
        )));
    }
    for &dim in &layout.mode_order {
        w.write_u32_le(dim);
    }
    w.write_zeros(8); // _reserved — MUST be 0x00
    Ok(())
}

fn decode_csf(cursor: &mut ByteCursor<'_>, rank: u32) -> Result<LayoutDescriptor> {
    let nnz = cursor.read_u64_le()?;

    // Read mode_order[rank]: the permutation of logical dimensions.
    let mut mode_order = Vec::with_capacity(rank as usize);
    for _ in 0..rank {
        mode_order.push(cursor.read_u32_le()?);
    }

    // Spec: _reserved MUST be 0x00; readers MUST reject non-zero reserved bytes.
    let reserved = cursor.read_bytes(8)?;
    if reserved.iter().any(|&b| b != 0) {
        return Err(Error::ReservedBytesNonZero {
            field: "csf._reserved",
        });
    }

    // Validate mode_order is a permutation of 0..rank.
    // Structural validation here matches what validate_against_shape also checks,
    // but the codec validates eagerly so a malformed wire descriptor is rejected
    // before the caller can observe a partially-constructed CsfLayout.
    let rank_us = rank as usize;
    let mut seen = 0u64;
    for (level, &dim) in mode_order.iter().enumerate() {
        if dim as usize >= rank_us {
            return Err(Error::InvalidLayout(format!(
                "csf: mode_order[{level}]={dim} out of range [0, {rank})"
            )));
        }
        let bit = 1u64 << dim;
        if seen & bit != 0 {
            return Err(Error::InvalidLayout(format!(
                "csf: mode_order contains duplicate value {dim}"
            )));
        }
        seen |= bit;
    }

    Ok(LayoutDescriptor::Csf(CsfLayout::new(nnz, mode_order)))
}

// ── BlockPaged ────────────────────────────────────────────────────────────────

/// Field order for block-paged (spec § Additional Descriptor Fields, all little-endian):
///
/// | Field                  | Wire type | Bytes |
/// |------------------------|-----------|-------|
/// | `page_size`            | uint32    | 4     |
/// | `num_pages`            | uint64    | 8     |
/// | `paged_axis`           | uint32    | 4     |
/// | `num_seqs`             | uint32    | 4     |
/// | `kv_role`              | uint8     | 1     |
/// | `layer_index`          | uint32    | 4     |
/// | `block_table_index_type` | uint8   | 1     |
/// | `_reserved`            | uint8[6]  | 6     |
///                                        Total 32 bytes
fn encode_block_paged(layout: &BlockPagedLayout, w: &mut ByteWriter) -> crate::Result<()> {
    w.write_u32_le(layout.page_size);
    w.write_u64_le(layout.num_pages);
    w.write_u32_le(layout.paged_axis);
    w.write_u32_le(layout.num_seqs);
    w.write_u8(layout.kv_role.wire_byte());
    // layer_index: None → 0xFFFFFFFF sentinel; Some(n) → n.
    let layer_index_wire = layout
        .layer_index
        .unwrap_or(crate::layout::block_paged::LAYER_INDEX_NONE);
    w.write_u32_le(layer_index_wire);
    w.write_u8(layout.block_table_index_type.wire_byte());
    w.write_zeros(6); // _reserved — MUST be 0x00
    Ok(())
}

fn decode_block_paged(cursor: &mut ByteCursor<'_>) -> crate::Result<LayoutDescriptor> {
    let page_size = cursor.read_u32_le()?;
    let num_pages = cursor.read_u64_le()?;
    let paged_axis = cursor.read_u32_le()?;
    let num_seqs = cursor.read_u32_le()?;

    let kv_role_byte = cursor.read_u8()?;
    let kv_role = KvRole::from_wire(kv_role_byte).ok_or_else(|| {
        crate::Error::InvalidLayout(format!(
            "block-paged: unknown kv_role byte 0x{kv_role_byte:02X}"
        ))
    })?;

    let layer_index_wire = cursor.read_u32_le()?;
    let layer_index = if layer_index_wire == crate::layout::block_paged::LAYER_INDEX_NONE {
        None
    } else {
        Some(layer_index_wire)
    };

    let index_type_byte = cursor.read_u8()?;
    let block_table_index_type =
        BlockTableIndexType::from_wire(index_type_byte).ok_or_else(|| {
            crate::Error::InvalidLayout(format!(
                "block-paged: unknown block_table_index_type byte 0x{index_type_byte:02X}"
            ))
        })?;

    // Spec: readers MUST reject a descriptor with any non-zero reserved byte.
    let reserved = cursor.read_bytes(6)?;
    if reserved.iter().any(|&b| b != 0) {
        return Err(crate::Error::ReservedBytesNonZero {
            field: "block_paged._reserved",
        });
    }

    // TODO(spec): block-paged quantization compatibility (block-paged.md §
    // Quantization Compatibility) requires cross-validating the layout's
    // page_size against the quantization descriptor's axis and block_size.
    // The quantization bytes are stored raw at this layer (TensorDescriptor.quantization
    // is Option<Vec<u8>>); the check is deferred to the typed quantization layer
    // (Layer 4 / higher) via BlockPagedLayout::validate_quantization_compatibility.
    // Note: shard rejection (block-paged.md § Sharding) is enforced in
    // TensorDescriptor::new, the cross-section seam where layout and shard are both
    // known — see descriptor/mod.rs. It is intentionally not checked here.

    Ok(LayoutDescriptor::BlockPaged(BlockPagedLayout::new(
        page_size,
        num_pages,
        paged_axis,
        num_seqs,
        kv_role,
        layer_index,
        block_table_index_type,
    )))
}

// ── Hilbert ───────────────────────────────────────────────────────────────────

fn encode_hilbert(layout: &HilbertLayout, w: &mut ByteWriter) -> Result<()> {
    w.write_u32_le(layout.hilbert_order);
    w.write_u32_le(layout.hilbert_rank);
    Ok(())
}

fn decode_hilbert(cursor: &mut ByteCursor<'_>) -> Result<LayoutDescriptor> {
    let hilbert_order = cursor.read_u32_le()?;
    let hilbert_rank = cursor.read_u32_le()?;
    let layout = HilbertLayout::new(hilbert_order, hilbert_rank)?;
    Ok(LayoutDescriptor::Hilbert(layout))
}

// ── PrivateExtension ──────────────────────────────────────────────────────────

fn encode_private_extension(layout: &PrivateExtensionLayout, w: &mut ByteWriter) -> Result<()> {
    w.write_u64_le(layout.extension_layout_id);
    w.write_u32_le(layout.extension_data.len() as u32);
    w.write_bytes(&layout.extension_data);
    Ok(())
}

fn decode_private_extension(tag: u8, cursor: &mut ByteCursor<'_>) -> Result<LayoutDescriptor> {
    let extension_layout_id = cursor.read_u64_le()?;
    let extension_data_length = cursor.read_u32_le()?;
    let data = cursor.read_bytes(extension_data_length as usize)?.to_vec();
    let layout = PrivateExtensionLayout::new(tag, extension_layout_id, data)?;
    Ok(LayoutDescriptor::PrivateExtension(layout))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::cursor::{ByteCursor, ByteWriter};

    fn round_trip(layout: &LayoutDescriptor, rank: u32) -> LayoutDescriptor {
        let mut w = ByteWriter::new();
        encode_layout_payload(layout, rank, &mut w).unwrap();
        let bytes = w.into_vec();
        let mut c = ByteCursor::new(&bytes, bytes.len());
        decode_layout_payload(layout.tag(), rank, &mut c, 0).unwrap()
    }

    #[test]
    fn row_major_round_trip() {
        let layout = LayoutDescriptor::RowMajor;
        assert_eq!(round_trip(&layout, 2), layout);
    }

    #[test]
    fn col_major_round_trip() {
        let layout = LayoutDescriptor::ColMajor;
        assert_eq!(round_trip(&layout, 2), layout);
    }

    #[test]
    fn strided_round_trip() {
        let layout = LayoutDescriptor::Strided(StridedLayout::new(vec![4, 1]));
        assert_eq!(round_trip(&layout, 2), layout);
    }

    #[test]
    fn morton_round_trip() {
        let layout = LayoutDescriptor::Morton(MortonLayout::new(vec![4, 4]).unwrap());
        assert_eq!(round_trip(&layout, 2), layout);
    }

    #[test]
    fn coo_round_trip() {
        let layout = LayoutDescriptor::Coo(CooLayout::new(42, true));
        assert_eq!(round_trip(&layout, 2), layout);
    }

    #[test]
    fn csr_round_trip() {
        let layout = LayoutDescriptor::Csr(CsrLayout::new(100));
        assert_eq!(round_trip(&layout, 2), layout);
    }

    #[test]
    fn csc_round_trip() {
        let layout = LayoutDescriptor::Csc(CscLayout::new(50));
        assert_eq!(round_trip(&layout, 2), layout);
    }

    #[test]
    fn hilbert_round_trip() {
        let layout = LayoutDescriptor::Hilbert(HilbertLayout::new(3, 2).unwrap());
        assert_eq!(round_trip(&layout, 2), layout);
    }

    #[test]
    fn private_extension_round_trip() {
        let layout = LayoutDescriptor::PrivateExtension(
            PrivateExtensionLayout::new(0xF0, 0xDEAD_BEEF, vec![1, 2, 3]).unwrap(),
        );
        assert_eq!(round_trip(&layout, 0), layout);
    }

    #[test]
    fn tiled_row_row_round_trip() {
        let t = TiledLayout::new(vec![4, 4], 0x01, 0x01, None, None, None).unwrap();
        let layout = LayoutDescriptor::Tiled(Box::new(t));
        assert_eq!(round_trip(&layout, 2), layout);
    }

    #[test]
    fn tiled_strided_outer_round_trip() {
        let os = OuterStrides::new(vec![2, 1]);
        let t = TiledLayout::new(vec![8, 8], 0x03, 0x01, Some(os), None, None).unwrap();
        let layout = LayoutDescriptor::Tiled(Box::new(t));
        assert_eq!(round_trip(&layout, 2), layout);
    }

    #[test]
    fn tiled_nested_round_trip() {
        let inner = TiledLayout::new(vec![2, 2], 0x01, 0x02, None, None, None).unwrap();
        let outer =
            TiledLayout::new(vec![4, 4], 0x01, 0x04, None, None, Some(Box::new(inner))).unwrap();
        let layout = LayoutDescriptor::Tiled(Box::new(outer));
        assert_eq!(round_trip(&layout, 2), layout);
    }

    #[test]
    fn subpaving_simple_round_trip() {
        use crate::layout::RegionDescriptor;
        let r = RegionDescriptor::new(vec![0, 0], vec![4, 4], 0x01, 0, 0).unwrap();
        let layout = LayoutDescriptor::Subpaving(SubpavingLayout::new(vec![r]).unwrap());
        assert_eq!(round_trip(&layout, 2), layout);
    }

    #[test]
    fn coo_reserved_bytes_rejected() {
        let mut w = ByteWriter::new();
        w.write_u64_le(0u64); // nnz
        w.write_u8(0u8); // is_sorted
        w.write_u8(0xFFu8); // _reserved[0] non-zero
        w.write_zeros(6);
        let bytes = w.into_vec();
        let mut c = ByteCursor::new(&bytes, bytes.len());
        let err = decode_layout_payload(TAG_COO, 0, &mut c, 0).unwrap_err();
        assert!(matches!(err, Error::ReservedBytesNonZero { .. }));
    }

    #[test]
    fn invalid_tag_rejected() {
        let bytes: &[u8] = &[];
        let mut c = ByteCursor::new(bytes, 0);
        let err = decode_layout_payload(0x00, 0, &mut c, 0).unwrap_err();
        assert!(matches!(err, Error::InvalidLayoutTag(0x00)));
    }

    #[test]
    fn reserved_tag_rejected() {
        // 0x0A is now TAG_CSF (a named layout), not reserved. Use 0x0C instead.
        let bytes: &[u8] = &[];
        let mut c = ByteCursor::new(bytes, 0);
        let err = decode_layout_payload(0x0C, 0, &mut c, 0).unwrap_err();
        assert!(matches!(err, Error::ReservedLayoutTag(0x0C)));
    }

    // ── Group A: subpaving with non-trivial inner layout round-trips ──────────

    mod subpaving_inner_layout {
        use crate::descriptor::TensorDescriptor;
        use crate::layout::{
            LayoutDescriptor, MortonLayout, RegionDescriptor, StridedLayout, SubpavingLayout,
            TiledLayout,
        };
        use crate::{BufferHandle, DeviceTag, ElementType, Shape, SyncMode, MIN_BUFFER_ALIGNMENT};

        /// Helper: a minimal float32 [8,8] TensorDescriptor with the given subpaving layout.
        fn descriptor_8x8(layout: LayoutDescriptor) -> TensorDescriptor {
            let shape = Shape::new(vec![8u64, 8]).unwrap();
            // Subpaving uses 1 buffer (dense).
            let buf = BufferHandle::new(
                512,
                MIN_BUFFER_ALIGNMENT,
                DeviceTag::Cpu,
                SyncMode::ProducerSynced,
            )
            .unwrap();
            TensorDescriptor::new(
                1,
                0,
                ElementType::Float32,
                shape,
                0,
                layout,
                vec![buf],
                None,
                None,
                None,
                None,
            )
            .unwrap()
        }

        /// Region 0: [0,0]→[4,8], Region 1: [4,0]→[4,8], both strided.
        ///
        /// Strided inner layouts carry an `int64[rank]` payload in the region's
        /// `region_layout_length` / `region_layout_payload` fields.  This is the
        /// simplest non-trivial inner layout: both regions share the same strides
        /// but each carries its own encoded payload.
        #[test]
        fn region_with_strided_inner_layout() {
            let strided = LayoutDescriptor::Strided(StridedLayout::new(vec![8i64, 1]));
            let r0 = RegionDescriptor::new(vec![0u64, 0], vec![4, 8], 0x03, 0, 0)
                .unwrap()
                .with_inner_layout(strided.clone())
                .unwrap();
            let r1 = RegionDescriptor::new(vec![4u64, 0], vec![4, 8], 0x03, 0, 256)
                .unwrap()
                .with_inner_layout(strided)
                .unwrap();
            let layout = LayoutDescriptor::Subpaving(SubpavingLayout::new(vec![r0, r1]).unwrap());

            let desc = descriptor_8x8(layout);
            let encoded = desc.encode().unwrap();
            let decoded = TensorDescriptor::decode(&encoded).unwrap();
            assert_eq!(decoded, desc);
        }

        /// One region covering the whole [8,8] tensor with a tiled inner layout.
        ///
        /// Uses a 2-level tiling (outer row-major, inner col-major, 4×4 tiles).
        /// The tiled payload includes `tile_shape u64[rank]` + tag bytes, which
        /// must survive the length-prefixed region_layout_payload encoding path.
        #[test]
        fn region_with_tiled_inner_layout() {
            let tiled = LayoutDescriptor::Tiled(Box::new(
                TiledLayout::new(vec![4u64, 4], 0x01, 0x02, None, None, None).unwrap(),
            ));
            let r = RegionDescriptor::new(vec![0u64, 0], vec![8, 8], 0x04, 0, 0)
                .unwrap()
                .with_inner_layout(tiled)
                .unwrap();
            let layout = LayoutDescriptor::Subpaving(SubpavingLayout::new(vec![r]).unwrap());

            let desc = descriptor_8x8(layout);
            let encoded = desc.encode().unwrap();
            let decoded = TensorDescriptor::decode(&encoded).unwrap();
            assert_eq!(decoded, desc);
        }

        /// One region with a Morton (Z-order) inner layout.
        ///
        /// Morton payload is `uint32[rank]` (bits_per_dim), so rank=2 gives 8 bytes.
        /// 3 bits per dimension can address 2^3 = 8 elements, fitting the [8,8] region.
        #[test]
        fn region_with_morton_inner_layout() {
            let morton = LayoutDescriptor::Morton(MortonLayout::new(vec![3u32, 3]).unwrap());
            let r = RegionDescriptor::new(vec![0u64, 0], vec![8, 8], 0x05, 0, 0)
                .unwrap()
                .with_inner_layout(morton)
                .unwrap();
            let layout = LayoutDescriptor::Subpaving(SubpavingLayout::new(vec![r]).unwrap());

            let desc = descriptor_8x8(layout);
            let encoded = desc.encode().unwrap();
            let decoded = TensorDescriptor::decode(&encoded).unwrap();
            assert_eq!(decoded, desc);
        }

        /// Outer subpaving with two regions; Region 0 itself contains an inner
        /// subpaving, exercising the recursive depth-increment path.
        ///
        /// Outer:
        ///   Region 0: origin [0,0], extent [4,8] → inner subpaving (depth+1)
        ///   Region 1: origin [4,0], extent [4,8] → row-major (no payload)
        ///
        /// Inner subpaving (nested inside Region 0):
        ///   Inner Region A: origin [0,0], extent [2,8] → row-major
        ///   Inner Region B: origin [2,0], extent [2,8] → row-major
        ///
        /// The inner regions are within the [8,8] tensor shape so
        /// validate_against_shape passes.  depth increments from 0→1 in
        /// encode_region / decode_region when region_layout_tag == TAG_SUBPAVING.
        #[test]
        fn recursive_subpaving_round_trip() {
            // Inner subpaving (lives inside the [0,0]→[4,8] outer region).
            let inner_r_a = RegionDescriptor::new(vec![0u64, 0], vec![2, 8], 0x01, 0, 0).unwrap();
            let inner_r_b = RegionDescriptor::new(vec![2u64, 0], vec![2, 8], 0x01, 0, 128).unwrap();
            let inner_sp = LayoutDescriptor::Subpaving(
                SubpavingLayout::new(vec![inner_r_a, inner_r_b]).unwrap(),
            );

            // Outer Region 0: tag 0x06 (subpaving), carries inner_sp as payload.
            let outer_r0 = RegionDescriptor::new(vec![0u64, 0], vec![4, 8], 0x06, 0, 0)
                .unwrap()
                .with_inner_layout(inner_sp)
                .unwrap();

            // Outer Region 1: row-major, no payload.
            let outer_r1 = RegionDescriptor::new(vec![4u64, 0], vec![4, 8], 0x01, 0, 256).unwrap();

            let layout = LayoutDescriptor::Subpaving(
                SubpavingLayout::new(vec![outer_r0, outer_r1]).unwrap(),
            );

            let desc = descriptor_8x8(layout);
            let encoded = desc.encode().unwrap();
            let decoded = TensorDescriptor::decode(&encoded).unwrap();
            assert_eq!(decoded, desc);
        }
    }

    // ── Group B: Unknown layout encode rejection ──────────────────────────────

    mod unknown_encode {
        use crate::descriptor::TensorDescriptor;
        use crate::layout::{LayoutDescriptor, UnknownLayout};
        use crate::{
            BufferHandle, DeviceTag, ElementType, Error, Shape, SyncMode, MIN_BUFFER_ALIGNMENT,
        };

        /// `encode_layout_payload` MUST return `Err(UnknownLayoutTag)` when the
        /// layout descriptor is `LayoutDescriptor::Unknown`.
        ///
        /// The spec (§ strict mode) prohibits re-encoding unrecognized layouts;
        /// the implementation enforces this by rejecting `Unknown` on the encode
        /// path, not only on the decode path.
        #[test]
        fn encode_unknown_layout_returns_error() {
            // Use a reserved-range tag (0x0C) for the Unknown variant, since 0x0A is now TAG_CSF.
            let unknown_layout =
                LayoutDescriptor::Unknown(UnknownLayout::new(0x0C, vec![0x01, 0x02]).unwrap());
            let shape = Shape::new(vec![4u64]).unwrap();
            let buf = BufferHandle::new(
                64,
                MIN_BUFFER_ALIGNMENT,
                DeviceTag::Cpu,
                SyncMode::ProducerSynced,
            )
            .unwrap();
            // TensorDescriptor::new accepts Unknown (it does not validate the layout tag).
            let desc = TensorDescriptor::new(
                1,
                0,
                ElementType::Float32,
                shape,
                0,
                unknown_layout,
                vec![buf],
                None,
                None,
                None,
                None,
            )
            .unwrap();
            let err = desc.encode().unwrap_err();
            assert!(
                matches!(err, Error::UnknownLayoutTag(_)),
                "expected UnknownLayoutTag, got {err:?}"
            );
        }
    }
}
