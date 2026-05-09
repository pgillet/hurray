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
//! - `0x40` (Hilbert): `hilbert_order u32`, `hilbert_rank u32`
//! - `0xF0`–`0xFE` (PrivateExtension): `extension_layout_id u64`, `extension_data_length u32`,
//!   `extension_data bytes[len]`

use crate::descriptor::cursor::{ByteCursor, ByteWriter};
use crate::layout::{
    CooLayout, CscLayout, CsrLayout, HilbertLayout, InnerStrides, LayoutDescriptor, MortonLayout,
    OuterStrides, PrivateExtensionLayout, RegionDescriptor, StridedLayout, SubpavingLayout,
    TiledLayout, TAG_COL_MAJOR, TAG_COO, TAG_CSC, TAG_CSR, TAG_HILBERT, TAG_MORTON, TAG_ROW_MAJOR,
    TAG_STRIDED, TAG_SUBPAVING, TAG_TILED,
};
use crate::{Error, Result};

/// Maximum recursion depth for nested tiled / subpaving descriptors.
const MAX_RECURSION_DEPTH: u8 = 8;

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
    match layout {
        LayoutDescriptor::RowMajor | LayoutDescriptor::ColMajor => {
            // No additional payload.
            Ok(())
        }
        LayoutDescriptor::Strided(s) => encode_strided(s, w),
        LayoutDescriptor::Tiled(t) => encode_tiled(t, rank, w, 0),
        LayoutDescriptor::Morton(m) => encode_morton(m, w),
        LayoutDescriptor::Subpaving(sp) => encode_subpaving(sp, rank, w, 0),
        LayoutDescriptor::Coo(c) => encode_coo(c, w),
        LayoutDescriptor::Csr(c) => encode_csr(c, w),
        LayoutDescriptor::Csc(c) => encode_csc(c, w),
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
    if depth >= MAX_RECURSION_DEPTH {
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

fn encode_tiled(layout: &TiledLayout, rank: u32, w: &mut ByteWriter, depth: u8) -> Result<()> {
    if depth >= MAX_RECURSION_DEPTH {
        return Err(Error::SubpavingNestingTooDeep);
    }
    // tile_shape: uint64[rank]
    for &dim in &layout.tile_shape {
        w.write_u64_le(dim);
    }
    w.write_u8(layout.outer_layout);
    w.write_u8(layout.inner_layout);
    w.write_zeros(2); // _reserved

    // Conditional: outer_strides if outer_layout == 0x03
    if layout.outer_layout == TAG_STRIDED {
        if let Some(os) = &layout.outer_strides {
            for _ in 0..rank as usize - os.strides.len() {
                // Pad if needed — though lengths should match rank.
            }
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
    if depth >= MAX_RECURSION_DEPTH {
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
    if depth >= MAX_RECURSION_DEPTH {
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
    _rank: u32,
    w: &mut ByteWriter,
    _depth: u8,
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
    // region_layout_length = 0: this worktree's RegionDescriptor does not store an
    // inner layout payload — it holds only the tag, which is sufficient for RowMajor
    // and ColMajor regions. Tags that require a payload (strided, tiled, …) are not
    // supported in this struct version; the wire length is written as 0.
    w.write_u32_le(0u32);
    Ok(())
}

fn decode_subpaving(cursor: &mut ByteCursor<'_>, rank: u32, depth: u8) -> Result<LayoutDescriptor> {
    if depth >= MAX_RECURSION_DEPTH {
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
    // so the MAX_RECURSION_DEPTH guard in decode_layout_payload fires correctly.
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
        let bytes: &[u8] = &[];
        let mut c = ByteCursor::new(bytes, 0);
        let err = decode_layout_payload(0x0A, 0, &mut c, 0).unwrap_err();
        assert!(matches!(err, Error::ReservedLayoutTag(0x0A)));
    }
}
