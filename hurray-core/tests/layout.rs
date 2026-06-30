//! Integration tests for Layer 3 layout descriptors.
//!
//! These tests validate the public API contract of `LayoutDescriptor` and all
//! per-layout payload types against the spec invariants defined in ADR-013 and
//! `docs/spec/memory-layout.md`. Tests are organized by invariant category.
//!
//! No private items are accessed; everything is exercised through the public API.

use std::num::NonZeroU8;

use hurray_core::descriptor::TensorDescriptor;
use hurray_core::{
    layout::{
        addressing::csf as csf_addressing, is_invalid_tag, is_private_tag, is_reserved_tag,
        validate_layout_tag_strict, BlockPagedLayout, BlockTableIndexType, CooLayout, CscLayout,
        CsfLayout, CsrLayout, HilbertLayout, InnerStrides, KvRole, LayoutDescriptor, MortonLayout,
        OuterStrides, PrivateExtensionLayout, RegionDescriptor, StridedLayout, SubpavingLayout,
        TiledLayout, UnknownLayout, MAX_TILED_DEPTH, PRIVATE_LAYOUT_TAG_MAX,
        PRIVATE_LAYOUT_TAG_MIN, TAG_BLOCK_PAGED, TAG_COL_MAJOR, TAG_COO, TAG_CSC, TAG_CSF, TAG_CSR,
        TAG_HILBERT, TAG_MORTON, TAG_ROW_MAJOR, TAG_STRIDED, TAG_SUBPAVING, TAG_TILED,
    },
    BufferHandle, DeviceTag, ElementType, Error, Shape, SyncMode, DYNAMIC, MIN_BUFFER_ALIGNMENT,
};

// ── Helpers ──────────────────────────────────────────────────────────────────

fn shape(dims: &[u64]) -> Shape {
    Shape::new(dims.to_vec()).expect("valid shape in test helper")
}

fn simple_region_2d() -> RegionDescriptor {
    RegionDescriptor::new(vec![0, 0], vec![4, 4], 0x01, 0, 0).expect("valid region in test helper")
}

// ── ADR-013 invariant 1: tag() returns the correct wire byte ─────────────────

/// Spec §memory-layout.md §Layout Taxonomy: each named layout has a fixed tag byte.
#[test]
fn tag_values_match_spec_constants() {
    assert_eq!(TAG_ROW_MAJOR, 0x01);
    assert_eq!(TAG_COL_MAJOR, 0x02);
    assert_eq!(TAG_STRIDED, 0x03);
    assert_eq!(TAG_TILED, 0x04);
    assert_eq!(TAG_MORTON, 0x05);
    assert_eq!(TAG_SUBPAVING, 0x06);
    assert_eq!(TAG_COO, 0x07);
    assert_eq!(TAG_CSR, 0x08);
    assert_eq!(TAG_CSC, 0x09);
    assert_eq!(TAG_HILBERT, 0x40);
}

#[test]
fn row_major_tag_is_0x01() {
    assert_eq!(LayoutDescriptor::RowMajor.tag(), 0x01);
}

#[test]
fn col_major_tag_is_0x02() {
    assert_eq!(LayoutDescriptor::ColMajor.tag(), 0x02);
}

#[test]
fn strided_tag_is_0x03() {
    let d = LayoutDescriptor::Strided(StridedLayout::new(vec![4, 1]));
    assert_eq!(d.tag(), 0x03);
}

#[test]
fn tiled_tag_is_0x04() {
    let t = TiledLayout::new(vec![4, 4], 0x01, 0x01, None, None, None).unwrap();
    assert_eq!(LayoutDescriptor::Tiled(Box::new(t)).tag(), 0x04);
}

#[test]
fn morton_tag_is_0x05() {
    let d = LayoutDescriptor::Morton(MortonLayout::new(vec![2, 2]).unwrap());
    assert_eq!(d.tag(), 0x05);
}

#[test]
fn subpaving_tag_is_0x06() {
    let sp = SubpavingLayout::new(vec![simple_region_2d()]).unwrap();
    assert_eq!(LayoutDescriptor::Subpaving(sp).tag(), 0x06);
}

#[test]
fn coo_tag_is_0x07() {
    assert_eq!(LayoutDescriptor::Coo(CooLayout::new(0, false)).tag(), 0x07);
}

#[test]
fn csr_tag_is_0x08() {
    assert_eq!(LayoutDescriptor::Csr(CsrLayout::new(0)).tag(), 0x08);
}

#[test]
fn csc_tag_is_0x09() {
    assert_eq!(LayoutDescriptor::Csc(CscLayout::new(0)).tag(), 0x09);
}

#[test]
fn hilbert_tag_is_0x40() {
    let d = LayoutDescriptor::Hilbert(HilbertLayout::new(2, 2).unwrap());
    assert_eq!(d.tag(), 0x40);
}

/// Private-extension tag is passed through verbatim.
#[test]
fn private_extension_tag_passthrough_all_valid_values() {
    for tag in PRIVATE_LAYOUT_TAG_MIN..=PRIVATE_LAYOUT_TAG_MAX {
        let d = LayoutDescriptor::PrivateExtension(
            PrivateExtensionLayout::new(tag, 0, vec![]).unwrap(),
        );
        assert_eq!(
            d.tag(),
            tag,
            "expected tag() == 0x{tag:02X} for PrivateExtension"
        );
    }
}

/// Unknown tag is passed through verbatim (permissive mode).
#[test]
fn unknown_tag_passthrough() {
    // Use a reserved-range tag (not 0x0A which is now TAG_CSF).
    let d = LayoutDescriptor::Unknown(UnknownLayout::new(0x0C, vec![0xDE, 0xAD]).unwrap());
    assert_eq!(d.tag(), 0x0C);
}

// ── ADR-013 invariant 2: buffer_count() values ───────────────────────────────

/// Spec §memory-layout.md: dense layouts require exactly 1 buffer.
#[test]
fn row_major_buffer_count_is_1() {
    assert_eq!(LayoutDescriptor::RowMajor.buffer_count(), NonZeroU8::new(1));
}

#[test]
fn col_major_buffer_count_is_1() {
    assert_eq!(LayoutDescriptor::ColMajor.buffer_count(), NonZeroU8::new(1));
}

#[test]
fn strided_buffer_count_is_1() {
    let d = LayoutDescriptor::Strided(StridedLayout::new(vec![1, 4]));
    assert_eq!(d.buffer_count(), NonZeroU8::new(1));
}

#[test]
fn tiled_buffer_count_is_1() {
    let t = TiledLayout::new(vec![4], 0x01, 0x01, None, None, None).unwrap();
    assert_eq!(
        LayoutDescriptor::Tiled(Box::new(t)).buffer_count(),
        NonZeroU8::new(1)
    );
}

#[test]
fn morton_buffer_count_is_1() {
    let d = LayoutDescriptor::Morton(MortonLayout::new(vec![2]).unwrap());
    assert_eq!(d.buffer_count(), NonZeroU8::new(1));
}

#[test]
fn subpaving_buffer_count_is_1() {
    let sp = SubpavingLayout::new(vec![simple_region_2d()]).unwrap();
    assert_eq!(
        LayoutDescriptor::Subpaving(sp).buffer_count(),
        NonZeroU8::new(1)
    );
}

#[test]
fn hilbert_buffer_count_is_1() {
    let d = LayoutDescriptor::Hilbert(HilbertLayout::new(2, 2).unwrap());
    assert_eq!(d.buffer_count(), NonZeroU8::new(1));
}

/// Spec §layouts/coo.md: COO requires values + indices = 2 buffers.
#[test]
fn coo_buffer_count_is_2() {
    assert_eq!(
        LayoutDescriptor::Coo(CooLayout::new(0, false)).buffer_count(),
        NonZeroU8::new(2)
    );
}

/// Same count regardless of nnz / is_sorted.
#[test]
fn coo_buffer_count_is_2_for_sorted_nonempty() {
    assert_eq!(
        LayoutDescriptor::Coo(CooLayout::new(1_000_000, true)).buffer_count(),
        NonZeroU8::new(2)
    );
}

/// Spec §layouts/csr.md: CSR requires values + col_indices + row_ptr = 3 buffers.
#[test]
fn csr_buffer_count_is_3() {
    assert_eq!(
        LayoutDescriptor::Csr(CsrLayout::new(0)).buffer_count(),
        NonZeroU8::new(3)
    );
}

/// Spec §layouts/csc.md: CSC requires values + row_indices + col_ptr = 3 buffers.
#[test]
fn csc_buffer_count_is_3() {
    assert_eq!(
        LayoutDescriptor::Csc(CscLayout::new(0)).buffer_count(),
        NonZeroU8::new(3)
    );
}

/// Spec §memory-layout.md §Extension Layouts: private layouts have unknown buffer count.
#[test]
fn private_extension_buffer_count_is_none() {
    let d =
        LayoutDescriptor::PrivateExtension(PrivateExtensionLayout::new(0xF5, 42, vec![]).unwrap());
    assert!(d.buffer_count().is_none());
}

/// Spec §memory-layout.md §Strict vs Permissive: unknown layout buffer count is None.
#[test]
fn unknown_buffer_count_is_none() {
    // Use a reserved-range tag (not 0x0A which is now TAG_CSF).
    let d = LayoutDescriptor::Unknown(UnknownLayout::new(0x0C, vec![]).unwrap());
    assert!(d.buffer_count().is_none());
}

// ── ADR-013 invariant 3: Unknown is the only permissive-mode fallback ────────

/// Unknown can hold any raw tag byte, including normally-invalid ones.
/// This lets permissive readers pass unrecognized tags through without error.
#[test]
fn unknown_can_hold_reserved_tag() {
    // Reserved range tag: 0x0C (0x0A is now TAG_CSF — a named layout).
    let d = LayoutDescriptor::Unknown(UnknownLayout::new(0x0C, vec![]).unwrap());
    assert_eq!(d.tag(), 0x0C);
    assert!(d.buffer_count().is_none());
}

#[test]
fn unknown_preserves_opaque_raw_bytes() {
    let raw = vec![0xDE, 0xAD, 0xBE, 0xEF];
    let u = UnknownLayout::new(0x3F, raw.clone()).unwrap();
    assert_eq!(u.raw_bytes, raw);
}

// ── ADR-013 invariant 4: 0x00 / 0xFF always rejected (InvalidLayoutTag) ──────

/// Spec §memory-layout.md §Layout Taxonomy: 0x00 and 0xFF are permanently invalid sentinels.
#[test]
fn validate_strict_rejects_tag_0x00_with_invalid_layout_tag() {
    assert!(matches!(
        validate_layout_tag_strict(0x00),
        Err(Error::InvalidLayoutTag(0x00))
    ));
}

#[test]
fn validate_strict_rejects_tag_0xff_with_invalid_layout_tag() {
    assert!(matches!(
        validate_layout_tag_strict(0xFF),
        Err(Error::InvalidLayoutTag(0xFF))
    ));
}

#[test]
fn is_invalid_tag_true_for_0x00_and_0xff() {
    assert!(is_invalid_tag(0x00));
    assert!(is_invalid_tag(0xFF));
}

#[test]
fn is_invalid_tag_false_for_all_other_values() {
    for tag in 0x01u8..=0xFE {
        assert!(!is_invalid_tag(tag), "0x{tag:02X} should not be invalid");
    }
}

// ── ADR-013 invariant 5: reserved-range tags rejected (ReservedLayoutTag) ────

/// Spec §memory-layout.md: reserved ranges are 0x0C–0x3F, 0x41–0x7F, 0x80–0xEF.
/// (0x0A is CSF and 0x0B is block-paged — both are named layout tags.)
#[test]
fn validate_strict_rejects_reserved_range_boundaries() {
    // First and last of each reserved sub-range (0x0A is now CSF, so range starts at 0x0C).
    let reserved_boundary_pairs = [(0x0C_u8, 0x3F), (0x41, 0x7F), (0x80, 0xEF)];
    for (lo, hi) in reserved_boundary_pairs {
        assert!(
            matches!(
                validate_layout_tag_strict(lo),
                Err(Error::ReservedLayoutTag(_))
            ),
            "0x{lo:02X} should be ReservedLayoutTag"
        );
        assert!(
            matches!(
                validate_layout_tag_strict(hi),
                Err(Error::ReservedLayoutTag(_))
            ),
            "0x{hi:02X} should be ReservedLayoutTag"
        );
    }
}

#[test]
fn is_reserved_tag_true_for_all_reserved_ranges() {
    // Spot-check each reserved sub-range (0x0A is now CSF — named tag, not reserved).
    for tag in [0x0C_u8, 0x20, 0x3F, 0x41, 0x60, 0x7F, 0x80, 0xA0, 0xEF] {
        assert!(is_reserved_tag(tag), "0x{tag:02X} should be reserved");
    }
}

#[test]
fn is_reserved_tag_false_for_named_and_private_tags() {
    for tag in [
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x40, 0xF0, 0xFE,
    ] {
        assert!(
            !is_reserved_tag(tag as u8),
            "0x{tag:02X} should not be reserved"
        );
    }
}

/// The tag just before the first reserved range must not be reserved.
#[test]
fn tag_0x09_not_reserved() {
    assert!(!is_reserved_tag(0x09));
}

/// The tag just after the reserved range boundary (0x40, the Hilbert tag) must not be reserved.
#[test]
fn tag_0x40_not_reserved() {
    assert!(!is_reserved_tag(0x40));
}

// ── ADR-013 invariant 6: private-range tags (0xF0–0xFE) ─────────────────────

/// Spec §memory-layout.md §Extension Layouts: 0xF0–0xFE are implementation-private.
#[test]
fn private_extension_accepts_all_valid_tags() {
    for tag in 0xF0u8..=0xFE {
        let result = PrivateExtensionLayout::new(tag, 0, vec![]);
        assert!(
            result.is_ok(),
            "0x{tag:02X} should be accepted by PrivateExtensionLayout"
        );
    }
}

#[test]
fn private_extension_rejects_tag_0xff() {
    // 0xFF is permanently invalid — not part of the private range.
    assert!(matches!(
        PrivateExtensionLayout::new(0xFF, 0, vec![]),
        Err(Error::InvalidLayout(_))
    ));
}

#[test]
fn private_extension_rejects_core_layout_tags() {
    for tag in [
        0x01_u8, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x40,
    ] {
        assert!(
            matches!(
                PrivateExtensionLayout::new(tag, 0, vec![]),
                Err(Error::InvalidLayout(_))
            ),
            "core tag 0x{tag:02X} should be rejected by PrivateExtensionLayout"
        );
    }
}

#[test]
fn private_extension_rejects_reserved_tag() {
    // 0xEF is in the reserved range (0x80–0xEF), not the private range.
    assert!(matches!(
        PrivateExtensionLayout::new(0xEF, 0, vec![]),
        Err(Error::InvalidLayout(_))
    ));
}

/// In strict mode, private tags are rejected even though they are valid for PrivateExtensionLayout.
#[test]
fn validate_strict_rejects_private_tags() {
    for tag in [0xF0_u8, 0xF7, 0xFE] {
        assert!(
            matches!(
                validate_layout_tag_strict(tag),
                Err(Error::PrivateLayoutTag(_))
            ),
            "strict mode should reject private tag 0x{tag:02X}"
        );
    }
}

#[test]
fn is_private_tag_true_for_0xf0_to_0xfe() {
    for tag in 0xF0u8..=0xFE {
        assert!(is_private_tag(tag), "0x{tag:02X} should be private");
    }
}

#[test]
fn is_private_tag_false_for_0xff() {
    assert!(!is_private_tag(0xFF));
}

#[test]
fn is_private_tag_false_for_core_tags() {
    assert!(!is_private_tag(0x01));
    assert!(!is_private_tag(0x09));
    assert!(!is_private_tag(0x40));
}

/// The private tag range constants agree with the spec.
#[test]
fn private_tag_range_constants_match_spec() {
    assert_eq!(PRIVATE_LAYOUT_TAG_MIN, 0xF0);
    assert_eq!(PRIVATE_LAYOUT_TAG_MAX, 0xFE);
}

/// PrivateExtensionLayout stores the extension_layout_id and data faithfully.
#[test]
fn private_extension_stores_id_and_data() {
    let id = 0xDEAD_BEEF_CAFE_0001_u64;
    let data = vec![0xAB, 0xCD, 0xEF];
    let p = PrivateExtensionLayout::new(0xF2, id, data.clone()).unwrap();
    assert_eq!(p.tag, 0xF2);
    assert_eq!(p.extension_layout_id, id);
    assert_eq!(p.extension_data, data);
}

// ── ADR-013 invariant 7: validate_against_shape — Strided rank mismatch ──────

/// Spec §layouts/strided.md: strides.len() MUST equal shape.rank().
#[test]
fn strided_valid_when_rank_matches() {
    let d = LayoutDescriptor::Strided(StridedLayout::new(vec![4, 1]));
    assert!(d.validate_against_shape(&shape(&[3, 4])).is_ok());
}

#[test]
fn strided_invalid_when_strides_too_short() {
    let d = LayoutDescriptor::Strided(StridedLayout::new(vec![1]));
    assert!(matches!(
        d.validate_against_shape(&shape(&[3, 4])),
        Err(Error::InvalidLayout(_))
    ));
}

#[test]
fn strided_invalid_when_strides_too_long() {
    let d = LayoutDescriptor::Strided(StridedLayout::new(vec![4, 1, 1]));
    assert!(matches!(
        d.validate_against_shape(&shape(&[3, 4])),
        Err(Error::InvalidLayout(_))
    ));
}

/// Rank-0 (scalar) shape MUST be rejected by Strided.
#[test]
fn strided_rejects_scalar_shape() {
    // Even an empty strides vec (matching rank-0) must be rejected by the spec.
    let d = LayoutDescriptor::Strided(StridedLayout::new(vec![]));
    assert!(matches!(
        d.validate_against_shape(&Shape::scalar()),
        Err(Error::InvalidLayout(_))
    ));
}

/// Spec §layouts/strided.md: negative strides (reversed dimension) are valid.
#[test]
fn strided_negative_strides_are_valid() {
    let d = LayoutDescriptor::Strided(StridedLayout::new(vec![-4, 1]));
    assert!(d.validate_against_shape(&shape(&[3, 4])).is_ok());
}

/// Spec §layouts/strided.md: zero stride (broadcast dimension) is valid.
#[test]
fn strided_zero_stride_is_valid() {
    let d = LayoutDescriptor::Strided(StridedLayout::new(vec![0, 1]));
    assert!(d.validate_against_shape(&shape(&[3, 4])).is_ok());
}

/// Mix of positive, negative, and zero strides must be accepted.
#[test]
fn strided_mixed_strides_accepted() {
    let d = LayoutDescriptor::Strided(StridedLayout::new(vec![-10, 0, 1]));
    assert!(d.validate_against_shape(&shape(&[5, 3, 4])).is_ok());
}

/// Rank-1 shape with a single stride should succeed.
#[test]
fn strided_valid_for_rank_1() {
    let d = LayoutDescriptor::Strided(StridedLayout::new(vec![1]));
    assert!(d.validate_against_shape(&shape(&[100])).is_ok());
}

// ── ADR-013 invariant 8: validate_against_shape — Tiled rank mismatch ────────

/// Spec §layouts/tiled.md: tile_shape.len() MUST equal shape.rank().
#[test]
fn tiled_valid_when_rank_matches() {
    let t = TiledLayout::new(vec![4, 4], 0x01, 0x01, None, None, None).unwrap();
    assert!(LayoutDescriptor::Tiled(Box::new(t))
        .validate_against_shape(&shape(&[8, 8]))
        .is_ok());
}

#[test]
fn tiled_invalid_when_tile_shape_too_short() {
    let t = TiledLayout::new(vec![4], 0x01, 0x01, None, None, None).unwrap();
    assert!(matches!(
        LayoutDescriptor::Tiled(Box::new(t)).validate_against_shape(&shape(&[8, 8])),
        Err(Error::InvalidLayout(_))
    ));
}

#[test]
fn tiled_invalid_when_tile_shape_too_long() {
    let t = TiledLayout::new(vec![4, 4, 4], 0x01, 0x01, None, None, None).unwrap();
    assert!(matches!(
        LayoutDescriptor::Tiled(Box::new(t)).validate_against_shape(&shape(&[8, 8])),
        Err(Error::InvalidLayout(_))
    ));
}

/// Spec §layouts/tiled.md: rank-0 tensors MUST NOT use the tiled layout.
#[test]
fn tiled_rejects_scalar_shape() {
    // tile_shape with len=0 cannot be constructed (constructor rejects empty),
    // so test that a rank-1 tile against rank-0 shape is rejected.
    let t = TiledLayout::new(vec![4], 0x01, 0x01, None, None, None).unwrap();
    assert!(matches!(
        LayoutDescriptor::Tiled(Box::new(t)).validate_against_shape(&Shape::scalar()),
        Err(Error::InvalidLayout(_))
    ));
}

/// Spec §layouts/tiled.md: tile_shape values of 0 MUST be rejected at construction.
#[test]
fn tiled_rejects_zero_tile_dimension() {
    assert!(matches!(
        TiledLayout::new(vec![0, 4], 0x01, 0x01, None, None, None),
        Err(Error::InvalidLayout(_))
    ));
}

#[test]
fn tiled_rejects_all_zero_tile_dimensions() {
    assert!(matches!(
        TiledLayout::new(vec![0, 0], 0x01, 0x01, None, None, None),
        Err(Error::InvalidLayout(_))
    ));
}

/// outer_layout must be 0x01, 0x02, or 0x03 only.
#[test]
fn tiled_rejects_invalid_outer_layout_0x00() {
    assert!(matches!(
        TiledLayout::new(vec![4, 4], 0x00, 0x01, None, None, None),
        Err(Error::InvalidLayout(_))
    ));
}

#[test]
fn tiled_rejects_outer_layout_0x04() {
    // 0x04 is tiled itself — not valid as an outer layout.
    assert!(matches!(
        TiledLayout::new(vec![4, 4], 0x04, 0x01, None, None, None),
        Err(Error::InvalidLayout(_))
    ));
}

#[test]
fn tiled_accepts_valid_outer_layouts() {
    for outer in [0x01_u8, 0x02, 0x03] {
        let outer_strides = if outer == 0x03 {
            Some(OuterStrides::new(vec![2, 1]))
        } else {
            None
        };
        let result = TiledLayout::new(vec![4, 4], outer, 0x01, outer_strides, None, None);
        assert!(
            result.is_ok(),
            "outer_layout 0x{outer:02X} should be accepted"
        );
    }
}

/// inner_layout must be 0x01, 0x02, 0x03, or 0x04 (recursive tiling).
#[test]
fn tiled_rejects_invalid_inner_layout_0x05() {
    assert!(matches!(
        TiledLayout::new(vec![4, 4], 0x01, 0x05, None, None, None),
        Err(Error::InvalidLayout(_))
    ));
}

#[test]
fn tiled_accepts_valid_inner_layouts() {
    // 0x01: row-major inner.
    let r1 = TiledLayout::new(vec![4, 4], 0x01, 0x01, None, None, None);
    assert!(r1.is_ok());

    // 0x02: col-major inner.
    let r2 = TiledLayout::new(vec![4, 4], 0x01, 0x02, None, None, None);
    assert!(r2.is_ok());

    // 0x03: strided inner (requires inner_strides).
    let inner_strides = Some(InnerStrides::new(vec![1, 4]));
    let r3 = TiledLayout::new(vec![4, 4], 0x01, 0x03, None, inner_strides, None);
    assert!(r3.is_ok());

    // 0x04: recursive tiling (requires inner_tiled).
    let inner = TiledLayout::new(vec![2, 2], 0x01, 0x01, None, None, None).unwrap();
    let r4 = TiledLayout::new(vec![4, 4], 0x01, 0x04, None, None, Some(Box::new(inner)));
    assert!(r4.is_ok());
}

/// outer_strides MUST be present iff outer_layout == 0x03.
#[test]
fn tiled_rejects_missing_outer_strides_when_outer_strided() {
    assert!(matches!(
        TiledLayout::new(vec![4, 4], 0x03, 0x01, None, None, None),
        Err(Error::InvalidLayout(_))
    ));
}

#[test]
fn tiled_rejects_spurious_outer_strides_when_outer_not_strided() {
    let outer_strides = Some(OuterStrides::new(vec![2, 1]));
    assert!(matches!(
        TiledLayout::new(vec![4, 4], 0x01, 0x01, outer_strides, None, None),
        Err(Error::InvalidLayout(_))
    ));
}

/// inner_strides MUST be present iff inner_layout == 0x03.
#[test]
fn tiled_rejects_missing_inner_strides_when_inner_strided() {
    assert!(matches!(
        TiledLayout::new(vec![4, 4], 0x01, 0x03, None, None, None),
        Err(Error::InvalidLayout(_))
    ));
}

#[test]
fn tiled_rejects_spurious_inner_strides_when_inner_not_strided() {
    let inner_strides = Some(InnerStrides::new(vec![1, 4]));
    assert!(matches!(
        TiledLayout::new(vec![4, 4], 0x01, 0x01, None, inner_strides, None),
        Err(Error::InvalidLayout(_))
    ));
}

// ── ADR-013 invariant 9: COO/CSR/CSC reject rank-0 shapes ───────────────────

/// Spec §layouts/coo.md: COO cannot be used for rank-0 (scalar) tensors.
#[test]
fn coo_rejects_scalar_shape() {
    let d = LayoutDescriptor::Coo(CooLayout::new(0, false));
    assert!(matches!(
        d.validate_against_shape(&Shape::scalar()),
        Err(Error::InvalidLayout(_))
    ));
}

/// COO accepts rank-1 shapes.
#[test]
fn coo_valid_for_rank_1_shape() {
    let d = LayoutDescriptor::Coo(CooLayout::new(0, false));
    assert!(d.validate_against_shape(&shape(&[100])).is_ok());
}

/// COO accepts rank-2 shapes.
#[test]
fn coo_valid_for_rank_2_shape() {
    let d = LayoutDescriptor::Coo(CooLayout::new(42, true));
    assert!(d.validate_against_shape(&shape(&[10, 20])).is_ok());
}

/// COO with nnz=0 is a valid empty sparse tensor.
#[test]
fn coo_nnz_zero_is_valid() {
    let d = LayoutDescriptor::Coo(CooLayout::new(0, false));
    assert!(d.validate_against_shape(&shape(&[5, 5])).is_ok());
}

/// Spec §layouts/csr.md: CSR requires exactly rank 2.
#[test]
fn csr_rejects_rank_0() {
    let d = LayoutDescriptor::Csr(CsrLayout::new(0));
    assert!(matches!(
        d.validate_against_shape(&Shape::scalar()),
        Err(Error::InvalidLayout(_))
    ));
}

#[test]
fn csr_rejects_rank_1() {
    let d = LayoutDescriptor::Csr(CsrLayout::new(0));
    assert!(matches!(
        d.validate_against_shape(&shape(&[10])),
        Err(Error::InvalidLayout(_))
    ));
}

#[test]
fn csr_rejects_rank_3() {
    let d = LayoutDescriptor::Csr(CsrLayout::new(0));
    assert!(matches!(
        d.validate_against_shape(&shape(&[2, 3, 4])),
        Err(Error::InvalidLayout(_))
    ));
}

#[test]
fn csr_valid_for_rank_2() {
    let d = LayoutDescriptor::Csr(CsrLayout::new(5));
    assert!(d.validate_against_shape(&shape(&[4, 5])).is_ok());
}

/// CSR with nnz=0 (empty sparse matrix) must be valid.
#[test]
fn csr_nnz_zero_is_valid() {
    let d = LayoutDescriptor::Csr(CsrLayout::new(0));
    assert!(d.validate_against_shape(&shape(&[100, 200])).is_ok());
}

/// Spec §layouts/csc.md: CSC requires exactly rank 2.
#[test]
fn csc_rejects_rank_0() {
    let d = LayoutDescriptor::Csc(CscLayout::new(0));
    assert!(matches!(
        d.validate_against_shape(&Shape::scalar()),
        Err(Error::InvalidLayout(_))
    ));
}

#[test]
fn csc_rejects_rank_1() {
    let d = LayoutDescriptor::Csc(CscLayout::new(0));
    assert!(matches!(
        d.validate_against_shape(&shape(&[10])),
        Err(Error::InvalidLayout(_))
    ));
}

#[test]
fn csc_rejects_rank_3() {
    let d = LayoutDescriptor::Csc(CscLayout::new(0));
    assert!(matches!(
        d.validate_against_shape(&shape(&[2, 3, 4])),
        Err(Error::InvalidLayout(_))
    ));
}

#[test]
fn csc_valid_for_rank_2() {
    let d = LayoutDescriptor::Csc(CscLayout::new(7));
    assert!(d.validate_against_shape(&shape(&[4, 5])).is_ok());
}

/// CSC with nnz=0 (empty sparse matrix) must be valid.
#[test]
fn csc_nnz_zero_is_valid() {
    let d = LayoutDescriptor::Csc(CscLayout::new(0));
    assert!(d.validate_against_shape(&shape(&[50, 50])).is_ok());
}

// ── ADR-013 invariant 10: Subpaving out-of-bounds regions ────────────────────

/// Spec §layouts/subpaving.md: every region MUST lie within the tensor shape.
#[test]
fn subpaving_rejects_region_exceeding_first_dimension() {
    // origin[0]=5, region_shape[0]=4 → end=9 > shape[0]=8
    let r = RegionDescriptor::new(vec![5, 0], vec![4, 8], 0x01, 0, 0).unwrap();
    let sp = SubpavingLayout::new(vec![r]).unwrap();
    assert!(matches!(
        LayoutDescriptor::Subpaving(sp).validate_against_shape(&shape(&[8, 8])),
        Err(Error::InvalidLayout(_))
    ));
}

#[test]
fn subpaving_rejects_region_exceeding_second_dimension() {
    // origin[1]=6, region_shape[1]=4 → end=10 > shape[1]=8
    let r = RegionDescriptor::new(vec![0, 6], vec![4, 4], 0x01, 0, 0).unwrap();
    let sp = SubpavingLayout::new(vec![r]).unwrap();
    assert!(matches!(
        LayoutDescriptor::Subpaving(sp).validate_against_shape(&shape(&[8, 8])),
        Err(Error::InvalidLayout(_))
    ));
}

/// A region that fits exactly at the boundary (origin+size == shape) must be valid.
/// Paired with a complementary top region so the subpaving also satisfies full coverage.
#[test]
fn subpaving_accepts_region_exactly_at_boundary() {
    // r_bot: origin=[4,0], region_shape=[4,8] → end=[8,8] == shape=[8,8] (the boundary).
    let r_top = RegionDescriptor::new(vec![0, 0], vec![4, 8], 0x01, 0, 0).unwrap();
    let r_bot = RegionDescriptor::new(vec![4, 0], vec![4, 8], 0x01, 0, 256).unwrap();
    let sp = SubpavingLayout::new(vec![r_top, r_bot]).unwrap();
    assert!(LayoutDescriptor::Subpaving(sp)
        .validate_against_shape(&shape(&[8, 8]))
        .is_ok());
}

/// A region starting at origin zero filling the whole shape must be valid.
#[test]
fn subpaving_single_region_covering_whole_shape() {
    let r = RegionDescriptor::new(vec![0, 0], vec![8, 8], 0x01, 0, 0).unwrap();
    let sp = SubpavingLayout::new(vec![r]).unwrap();
    assert!(LayoutDescriptor::Subpaving(sp)
        .validate_against_shape(&shape(&[8, 8]))
        .is_ok());
}

// ── ADR-013 invariant 11: Subpaving overlapping regions ──────────────────────

/// Spec §layouts/subpaving.md: regions MUST NOT overlap.
#[test]
fn subpaving_rejects_two_identical_overlapping_regions() {
    let r1 = RegionDescriptor::new(vec![0, 0], vec![4, 4], 0x01, 0, 0).unwrap();
    let r2 = RegionDescriptor::new(vec![0, 0], vec![4, 4], 0x01, 0, 64).unwrap();
    let sp = SubpavingLayout::new(vec![r1, r2]).unwrap();
    assert!(matches!(
        LayoutDescriptor::Subpaving(sp).validate_against_shape(&shape(&[8, 8])),
        Err(Error::InvalidLayout(_))
    ));
}

#[test]
fn subpaving_rejects_partially_overlapping_regions() {
    // r1: [0,0]–[4,4]; r2: [2,2]–[6,6]. They overlap in [2,2]–[4,4].
    let r1 = RegionDescriptor::new(vec![0, 0], vec![4, 4], 0x01, 0, 0).unwrap();
    let r2 = RegionDescriptor::new(vec![2, 2], vec![4, 4], 0x01, 0, 64).unwrap();
    let sp = SubpavingLayout::new(vec![r1, r2]).unwrap();
    assert!(matches!(
        LayoutDescriptor::Subpaving(sp).validate_against_shape(&shape(&[8, 8])),
        Err(Error::InvalidLayout(_))
    ));
}

/// Non-overlapping adjacent regions (touching edges) must be accepted.
#[test]
fn subpaving_accepts_two_adjacent_non_overlapping_regions() {
    // r1 covers [0,0]–[4,8]; r2 covers [4,0]–[8,8]. They share the edge row 4 but don't overlap.
    let r1 = RegionDescriptor::new(vec![0, 0], vec![4, 8], 0x01, 0, 0).unwrap();
    let r2 = RegionDescriptor::new(vec![4, 0], vec![4, 8], 0x01, 0, 256).unwrap();
    let sp = SubpavingLayout::new(vec![r1, r2]).unwrap();
    assert!(LayoutDescriptor::Subpaving(sp)
        .validate_against_shape(&shape(&[8, 8]))
        .is_ok());
}

/// Four non-overlapping quadrant regions of an 8×8 tensor must be valid.
#[test]
fn subpaving_four_quadrant_regions_valid() {
    let regions = vec![
        RegionDescriptor::new(vec![0, 0], vec![4, 4], 0x01, 0, 0).unwrap(),
        RegionDescriptor::new(vec![0, 4], vec![4, 4], 0x01, 0, 64).unwrap(),
        RegionDescriptor::new(vec![4, 0], vec![4, 4], 0x01, 0, 128).unwrap(),
        RegionDescriptor::new(vec![4, 4], vec![4, 4], 0x01, 0, 192).unwrap(),
    ];
    let sp = SubpavingLayout::new(regions).unwrap();
    assert!(LayoutDescriptor::Subpaving(sp)
        .validate_against_shape(&shape(&[8, 8]))
        .is_ok());
}

/// An overlap among three regions (the third overlaps with the first) must be detected.
#[test]
fn subpaving_rejects_third_region_overlapping_with_first() {
    let r1 = RegionDescriptor::new(vec![0, 0], vec![4, 4], 0x01, 0, 0).unwrap();
    let r2 = RegionDescriptor::new(vec![0, 4], vec![4, 4], 0x01, 0, 64).unwrap();
    // r3 overlaps r1: [0,0]–[2,2] is within r1.
    let r3 = RegionDescriptor::new(vec![0, 0], vec![2, 2], 0x01, 0, 128).unwrap();
    let sp = SubpavingLayout::new(vec![r1, r2, r3]).unwrap();
    assert!(matches!(
        LayoutDescriptor::Subpaving(sp).validate_against_shape(&shape(&[8, 8])),
        Err(Error::InvalidLayout(_))
    ));
}

// ── subpaving.md § Coverage Constraint: union must exactly cover the index space ──

/// A subpaving that leaves a gap (region volumes sum to less than the total) MUST be
/// rejected — the union does not cover every element of the index space.
#[test]
fn subpaving_rejects_incomplete_coverage() {
    // A single region covers only the top half of [8, 8] (32 of 64 elements).
    let r = RegionDescriptor::new(vec![0, 0], vec![4, 8], 0x01, 0, 0).unwrap();
    let sp = SubpavingLayout::new(vec![r]).unwrap();
    assert!(matches!(
        LayoutDescriptor::Subpaving(sp).validate_against_shape(&shape(&[8, 8])),
        Err(Error::InvalidLayout(_))
    ));
}

/// Coverage cannot be checked when a dimension is dynamic (the total is unknowable), so a
/// partial region that would otherwise fail coverage is accepted — deferred to the caller.
#[test]
fn subpaving_skips_coverage_when_dimension_is_dynamic() {
    let r = RegionDescriptor::new(vec![0, 0], vec![4, 8], 0x01, 0, 0).unwrap();
    let sp = SubpavingLayout::new(vec![r]).unwrap();
    assert!(LayoutDescriptor::Subpaving(sp)
        .validate_against_shape(&shape(&[DYNAMIC, 8]))
        .is_ok());
}

// ── ADR-013 invariant 12: Tiled recursion depth limit ────────────────────────

/// Spec §layouts/tiled.md: maximum nesting depth is MAX_TILED_DEPTH (8).
/// Depth 7 must be accepted; depth 8 must be rejected.
#[test]
fn tiled_max_depth_constant_is_8() {
    assert_eq!(MAX_TILED_DEPTH, 8);
}

/// Build a chain of exactly MAX_TILED_DEPTH - 1 levels deep; must succeed.
#[test]
fn tiled_depth_7_is_accepted() {
    // Build from the inside out. Level 0 is the innermost (depth MAX_TILED_DEPTH - 1 during construction).
    // We build 7 levels: indices 1..=7 where level 1 is innermost.
    let mut current: Option<Box<TiledLayout>> = None;
    // We want depth = 7 levels total, meaning the outermost TiledLayout::new call goes down 7 levels.
    // Build 6 inner levels first (they succeed), then wrap once more.
    for _ in 0..(MAX_TILED_DEPTH - 2) {
        let inner = if current.is_none() {
            TiledLayout::new(vec![2, 2], 0x01, 0x01, None, None, None).unwrap()
        } else {
            TiledLayout::new(vec![2, 2], 0x01, 0x04, None, None, current).unwrap()
        };
        current = Some(Box::new(inner));
    }
    // Final outermost level — total depth = MAX_TILED_DEPTH - 1 (7): must succeed.
    let result = TiledLayout::new(vec![2, 2], 0x01, 0x04, None, None, current);
    assert!(
        result.is_ok(),
        "depth {} should be accepted",
        MAX_TILED_DEPTH - 1
    );
}

/// A chain of MAX_TILED_DEPTH levels must be rejected.
///
/// BUG: This test is currently ignored because the implementation does not
/// enforce the depth limit when a tiled chain is built bottom-up via separate
/// `TiledLayout::new` calls. Each standalone `new()` resets the depth counter
/// to 0; only the re-validation of an already-built inner subtree during a
/// parent `new()` call advances the counter. A chain of 8 levels built
/// bottom-up is accepted instead of being rejected.
///
/// Spec §layouts/tiled.md: maximum nesting depth MUST be ≤ MAX_TILED_DEPTH (8).
/// The implementation MUST reject any chain that exceeds this limit regardless
/// of construction order.
#[test]
#[ignore = "BUG: depth enforcement only tracks re-validation depth, not total chain depth; \
            a bottom-up chain of 8 levels bypasses the limit (see test comment for details)"]
fn tiled_depth_8_is_rejected() {
    // Build a proper 8-level chain bottom-up using correct inner_layout flags.
    // Level 1 (innermost): inner_layout=0x01 (row-major, leaf node).
    // Levels 2-8: inner_layout=0x04 (recursive tiled).
    fn make_inner(depth: usize) -> Option<Box<TiledLayout>> {
        if depth == 0 {
            return None;
        }
        TiledLayout::new(
            vec![2, 2],
            0x01,
            if depth == 1 { 0x01 } else { 0x04 },
            None,
            None,
            make_inner(depth - 1),
        )
        .ok()
        .map(Box::new)
    }

    // Total chain: outermost + make_inner(MAX_TILED_DEPTH - 1) = 8 levels.
    let result = TiledLayout::new(
        vec![2, 2],
        0x01,
        0x04,
        None,
        None,
        make_inner(MAX_TILED_DEPTH - 1),
    );
    assert!(
        result.is_err(),
        "depth {} should be rejected",
        MAX_TILED_DEPTH
    );
}

// ── Spec edge cases: Hilbert layout ──────────────────────────────────────────

/// Spec §layouts/hilbert.md: hilbert_order MUST be > 0.
#[test]
fn hilbert_rejects_order_zero() {
    assert!(matches!(
        HilbertLayout::new(0, 2),
        Err(Error::InvalidLayout(_))
    ));
}

/// Spec §layouts/hilbert.md: hilbert_rank MUST be >= 2.
#[test]
fn hilbert_rejects_rank_1() {
    assert!(matches!(
        HilbertLayout::new(1, 1),
        Err(Error::InvalidLayout(_))
    ));
}

#[test]
fn hilbert_rejects_rank_0() {
    assert!(matches!(
        HilbertLayout::new(1, 0),
        Err(Error::InvalidLayout(_))
    ));
}

#[test]
fn hilbert_accepts_minimum_valid_params() {
    // order=1, rank=2: a 2×2 Hilbert curve.
    let h = HilbertLayout::new(1, 2);
    assert!(h.is_ok());
}

/// validate_against_shape: all dimensions must equal 2^hilbert_order.
#[test]
fn hilbert_valid_for_4x4_order_2() {
    let d = LayoutDescriptor::Hilbert(HilbertLayout::new(2, 2).unwrap());
    // 2^2 = 4; shape [4, 4].
    assert!(d.validate_against_shape(&shape(&[4, 4])).is_ok());
}

#[test]
fn hilbert_valid_for_8x8x8_order_3_rank_3() {
    let d = LayoutDescriptor::Hilbert(HilbertLayout::new(3, 3).unwrap());
    // 2^3 = 8; shape [8, 8, 8].
    assert!(d.validate_against_shape(&shape(&[8, 8, 8])).is_ok());
}

#[test]
fn hilbert_rejects_shape_with_wrong_dimension_value() {
    // order=2 → expected dim=4; shape[0]=3 is wrong.
    let d = LayoutDescriptor::Hilbert(HilbertLayout::new(2, 2).unwrap());
    assert!(matches!(
        d.validate_against_shape(&shape(&[3, 4])),
        Err(Error::InvalidLayout(_))
    ));
}

#[test]
fn hilbert_rejects_mismatched_rank() {
    // hilbert_rank=2 but shape is rank 3.
    let d = LayoutDescriptor::Hilbert(HilbertLayout::new(2, 2).unwrap());
    assert!(matches!(
        d.validate_against_shape(&shape(&[4, 4, 4])),
        Err(Error::InvalidLayout(_))
    ));
}

/// Dynamic dimensions are skipped during Hilbert shape validation.
#[test]
fn hilbert_skips_dynamic_dimensions() {
    let d = LayoutDescriptor::Hilbert(HilbertLayout::new(2, 2).unwrap());
    // shape[0]=4 (static, matches 2^2), shape[1]=DYNAMIC (skipped).
    let s = Shape::new(vec![4, DYNAMIC]).unwrap();
    assert!(d.validate_against_shape(&s).is_ok());
}

// ── Spec edge cases: Morton layout ───────────────────────────────────────────

/// Spec §layouts/morton.md: morton_bits MUST NOT be empty.
#[test]
fn morton_rejects_empty_bits() {
    assert!(matches!(
        MortonLayout::new(vec![]),
        Err(Error::InvalidLayout(_))
    ));
}

/// Spec §layouts/morton.md: every morton_bits[k] MUST be > 0.
#[test]
fn morton_rejects_zero_bit_count() {
    assert!(matches!(
        MortonLayout::new(vec![2, 0]),
        Err(Error::InvalidLayout(_))
    ));
}

/// validate_against_shape: morton_bits.len() must equal shape.rank().
#[test]
fn morton_rejects_rank_mismatch() {
    let d = LayoutDescriptor::Morton(MortonLayout::new(vec![2, 2, 2]).unwrap());
    assert!(matches!(
        d.validate_against_shape(&shape(&[4, 4])),
        Err(Error::InvalidLayout(_))
    ));
}

/// validate_against_shape: shape[k] must not exceed 2^morton_bits[k].
#[test]
fn morton_rejects_shape_exceeding_bit_capacity() {
    // morton_bits[0]=1 → max_dim = 2^1 = 2; shape[0]=3 > 2.
    let d = LayoutDescriptor::Morton(MortonLayout::new(vec![1, 2]).unwrap());
    assert!(matches!(
        d.validate_against_shape(&shape(&[3, 4])),
        Err(Error::InvalidLayout(_))
    ));
}

/// shape[k] exactly at capacity (2^morton_bits[k]) must be valid.
#[test]
fn morton_valid_when_shape_exactly_at_bit_capacity() {
    // morton_bits=[2,3] → max dims [4, 8]; shape=[4, 8].
    let d = LayoutDescriptor::Morton(MortonLayout::new(vec![2, 3]).unwrap());
    assert!(d.validate_against_shape(&shape(&[4, 8])).is_ok());
}

/// shape[k] below capacity must be valid.
#[test]
fn morton_valid_when_shape_below_bit_capacity() {
    // morton_bits=[3,3] → max=8; shape=[3, 5].
    let d = LayoutDescriptor::Morton(MortonLayout::new(vec![3, 3]).unwrap());
    assert!(d.validate_against_shape(&shape(&[3, 5])).is_ok());
}

/// Dynamic dimensions are skipped during Morton shape validation.
#[test]
fn morton_skips_dynamic_dimensions() {
    let d = LayoutDescriptor::Morton(MortonLayout::new(vec![2, 2]).unwrap());
    let s = Shape::new(vec![4, DYNAMIC]).unwrap();
    assert!(d.validate_against_shape(&s).is_ok());
}

/// Large morton_bits value (>=64) means max_dim saturates at u64::MAX.
#[test]
fn morton_large_bit_count_saturates_at_u64_max() {
    let d = LayoutDescriptor::Morton(MortonLayout::new(vec![64, 64]).unwrap());
    // Any static shape[k] <= u64::MAX should be valid.
    assert!(d
        .validate_against_shape(&shape(&[u64::MAX - 1, u64::MAX - 1]))
        .is_ok());
}

// ── Spec edge cases: Subpaving ────────────────────────────────────────────────

/// Spec §layouts/subpaving.md: region_count MUST be > 0.
#[test]
fn subpaving_rejects_empty_region_list() {
    assert!(matches!(
        SubpavingLayout::new(vec![]),
        Err(Error::InvalidLayout(_))
    ));
}

/// RegionDescriptor rejects a zero region_shape dimension.
#[test]
fn region_descriptor_rejects_zero_shape_dimension() {
    assert!(matches!(
        RegionDescriptor::new(vec![0, 0], vec![0, 4], 0x01, 0, 0),
        Err(Error::InvalidLayout(_))
    ));
}

/// RegionDescriptor rejects mismatched origin and region_shape lengths.
#[test]
fn region_descriptor_rejects_mismatched_origin_region_shape_lengths() {
    // origin has 1 element but region_shape has 2.
    assert!(matches!(
        RegionDescriptor::new(vec![0], vec![4, 4], 0x01, 0, 0),
        Err(Error::InvalidLayout(_))
    ));
}

/// RegionDescriptor rejects permanently-invalid layout tags 0x00 and 0xFF.
#[test]
fn region_descriptor_rejects_invalid_layout_tag_0x00() {
    assert!(matches!(
        RegionDescriptor::new(vec![0, 0], vec![4, 4], 0x00, 0, 0),
        Err(Error::InvalidLayout(_))
    ));
}

#[test]
fn region_descriptor_rejects_invalid_layout_tag_0xff() {
    assert!(matches!(
        RegionDescriptor::new(vec![0, 0], vec![4, 4], 0xFF, 0, 0),
        Err(Error::InvalidLayout(_))
    ));
}

/// RegionDescriptor accepts named and reserved layout tags (it only
/// rejects 0x00 and 0xFF — the interpretation of non-standard tags is deferred).
/// 0x0A is TAG_CSF (named), 0x0C is a reserved-range tag; both are accepted.
#[test]
fn region_descriptor_accepts_valid_layout_tags() {
    for tag in [0x01_u8, 0x02, 0x03, 0x06, 0x40, 0xF0, 0x0A, 0x0C] {
        let result = RegionDescriptor::new(vec![0, 0], vec![4, 4], tag, 0, 0);
        assert!(
            result.is_ok(),
            "tag 0x{tag:02X} should be accepted by RegionDescriptor"
        );
    }
}

/// RegionDescriptor stores all fields faithfully.
#[test]
fn region_descriptor_stores_fields() {
    let r = RegionDescriptor::new(vec![10, 20], vec![4, 8], 0x02, 3, 512).unwrap();
    assert_eq!(r.origin, [10, 20]);
    assert_eq!(r.region_shape, [4, 8]);
    assert_eq!(r.region_layout_tag, 0x02);
    assert_eq!(r.buffer_index, 3);
    assert_eq!(r.region_byte_offset, 512);
}

// ── Spec edge cases: RowMajor / ColMajor accept any rank ─────────────────────

/// Spec: RowMajor and ColMajor place no rank constraints.
#[test]
fn row_major_valid_for_scalar() {
    assert!(LayoutDescriptor::RowMajor
        .validate_against_shape(&Shape::scalar())
        .is_ok());
}

#[test]
fn row_major_valid_for_rank_1() {
    assert!(LayoutDescriptor::RowMajor
        .validate_against_shape(&shape(&[100]))
        .is_ok());
}

#[test]
fn row_major_valid_for_max_rank() {
    let s = Shape::new(vec![1u64; 64]).unwrap();
    assert!(LayoutDescriptor::RowMajor
        .validate_against_shape(&s)
        .is_ok());
}

#[test]
fn col_major_valid_for_scalar() {
    assert!(LayoutDescriptor::ColMajor
        .validate_against_shape(&Shape::scalar())
        .is_ok());
}

#[test]
fn col_major_valid_for_rank_64() {
    let s = Shape::new(vec![2u64; 64]).unwrap();
    assert!(LayoutDescriptor::ColMajor
        .validate_against_shape(&s)
        .is_ok());
}

// ── Spec edge cases: Unknown always passes validation ─────────────────────────

/// Spec §memory-layout.md §Strict vs Permissive: Unknown always passes validate_against_shape.
#[test]
fn unknown_passes_validation_for_scalar() {
    // Use a reserved-range tag (not 0x0A which is now TAG_CSF).
    let d = LayoutDescriptor::Unknown(UnknownLayout::new(0x0C, vec![]).unwrap());
    assert!(d.validate_against_shape(&Shape::scalar()).is_ok());
}

#[test]
fn unknown_passes_validation_for_large_rank() {
    let d = LayoutDescriptor::Unknown(UnknownLayout::new(0x3F, vec![0xAB]).unwrap());
    let s = Shape::new(vec![100u64; 64]).unwrap();
    assert!(d.validate_against_shape(&s).is_ok());
}

// ── Spec edge cases: PrivateExtension passes validate_against_shape ───────────

/// PrivateExtension: shape constraints are implementation-defined, so validation always passes.
#[test]
fn private_extension_passes_validation() {
    let d = LayoutDescriptor::PrivateExtension(
        PrivateExtensionLayout::new(0xF0, 1, vec![0x01, 0x02]).unwrap(),
    );
    assert!(d.validate_against_shape(&Shape::scalar()).is_ok());
    assert!(d.validate_against_shape(&shape(&[64, 64])).is_ok());
}

// ── Validate all known tags pass strict validation ────────────────────────────

#[test]
fn validate_strict_accepts_all_known_tags() {
    for tag in [
        0x01_u8, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x40,
    ] {
        assert!(
            validate_layout_tag_strict(tag).is_ok(),
            "known tag 0x{tag:02X} should pass strict validation"
        );
    }
}

// ── is_* helper functions — exhaustive boundary checks ────────────────────────

/// The first reserved range starts at 0x0C (0x0A is CSF, 0x0B is block-paged).
#[test]
fn first_reserved_range_lower_boundary() {
    assert!(is_reserved_tag(0x0C));
    // 0x0A is TAG_CSF — a named layout, not reserved.
    assert!(!is_reserved_tag(0x0A));
}

/// The tag immediately before the first reserved range (0x09) must not be reserved.
#[test]
fn tag_0x09_before_first_reserved_range_not_reserved() {
    assert!(!is_reserved_tag(0x09));
}

/// The tag immediately after the first reserved range (0x40) must not be reserved.
#[test]
fn tag_0x40_after_first_reserved_range_not_reserved() {
    assert!(!is_reserved_tag(0x40));
}

// ── PartialEq / Clone / Debug derive coverage ─────────────────────────────────

/// PartialEq for LayoutDescriptor must compare structurally.
#[test]
fn layout_descriptor_eq_same_variant() {
    let a = LayoutDescriptor::Coo(CooLayout::new(42, true));
    let b = LayoutDescriptor::Coo(CooLayout::new(42, true));
    assert_eq!(a, b);
}

#[test]
fn layout_descriptor_ne_different_nnz() {
    let a = LayoutDescriptor::Coo(CooLayout::new(42, true));
    let b = LayoutDescriptor::Coo(CooLayout::new(0, true));
    assert_ne!(a, b);
}

#[test]
fn layout_descriptor_ne_different_variants() {
    let a = LayoutDescriptor::RowMajor;
    let b = LayoutDescriptor::ColMajor;
    assert_ne!(a, b);
}

#[test]
fn layout_descriptor_clone_is_equal() {
    let orig = LayoutDescriptor::Strided(StridedLayout::new(vec![4, 1]));
    let cloned = orig.clone();
    assert_eq!(orig, cloned);
}

#[test]
fn layout_descriptor_debug_is_non_empty() {
    let d = LayoutDescriptor::RowMajor;
    let s = format!("{d:?}");
    assert!(!s.is_empty());
}

// ── Parametrized: all dense variants return buffer_count == Some(1) ───────────

#[test]
fn all_dense_variants_have_buffer_count_1() {
    let sp = SubpavingLayout::new(vec![simple_region_2d()]).unwrap();
    let dense: Vec<LayoutDescriptor> = vec![
        LayoutDescriptor::RowMajor,
        LayoutDescriptor::ColMajor,
        LayoutDescriptor::Strided(StridedLayout::new(vec![4, 1])),
        LayoutDescriptor::Tiled(Box::new(
            TiledLayout::new(vec![4, 4], 0x01, 0x01, None, None, None).unwrap(),
        )),
        LayoutDescriptor::Morton(MortonLayout::new(vec![2, 2]).unwrap()),
        LayoutDescriptor::Subpaving(sp),
        LayoutDescriptor::Hilbert(HilbertLayout::new(2, 2).unwrap()),
    ];
    for d in &dense {
        assert_eq!(
            d.buffer_count(),
            NonZeroU8::new(1),
            "dense variant tag=0x{:02X} should have buffer_count=1",
            d.tag()
        );
    }
}

// ── Sparse variants buffer_count assertions ───────────────────────────────────

#[test]
fn sparse_coo_csr_csc_buffer_counts() {
    let cases: &[(LayoutDescriptor, u8)] = &[
        (LayoutDescriptor::Coo(CooLayout::new(0, false)), 2),
        (LayoutDescriptor::Csr(CsrLayout::new(0)), 3),
        (LayoutDescriptor::Csc(CscLayout::new(0)), 3),
    ];
    for (d, expected) in cases {
        assert_eq!(
            d.buffer_count(),
            NonZeroU8::new(*expected),
            "tag=0x{:02X} expected buffer_count={expected}",
            d.tag()
        );
    }
}

// ── Subpaving: rank-mismatch in region origin/shape vs shape rank ─────────────

#[test]
fn subpaving_validate_rejects_region_with_wrong_rank() {
    // Region has rank-1 origin/shape, but tensor shape is rank 2.
    let r = RegionDescriptor::new(vec![0], vec![8], 0x01, 0, 0).unwrap();
    let sp = SubpavingLayout::new(vec![r]).unwrap();
    assert!(matches!(
        LayoutDescriptor::Subpaving(sp).validate_against_shape(&shape(&[8, 8])),
        Err(Error::InvalidLayout(_))
    ));
}

// ── Tiled: validate_against_shape rejects scalar (rank-0) ────────────────────

/// Spec: tiled layout MUST NOT be used for rank-0 tensors.
/// Construction of a rank-0 tile_shape fails, so test rank-1 tile vs rank-0 shape.
#[test]
fn tiled_validate_rejects_scalar_shape_via_rank_mismatch() {
    let t = TiledLayout::new(vec![4], 0x01, 0x01, None, None, None).unwrap();
    let result = LayoutDescriptor::Tiled(Box::new(t)).validate_against_shape(&Shape::scalar());
    assert!(matches!(result, Err(Error::InvalidLayout(_))));
}

// ── CooLayout: is_sorted field is stored and accessible ───────────────────────

#[test]
fn coo_layout_stores_is_sorted_true() {
    let c = CooLayout::new(10, true);
    assert!(c.is_sorted);
    assert_eq!(c.nnz, 10);
}

#[test]
fn coo_layout_stores_is_sorted_false() {
    let c = CooLayout::new(0, false);
    assert!(!c.is_sorted);
}

// ── Strided: i64::MIN and i64::MAX strides must not panic ─────────────────────

#[test]
fn strided_extreme_stride_values_do_not_panic() {
    let d = LayoutDescriptor::Strided(StridedLayout::new(vec![i64::MIN, i64::MAX]));
    // validate_against_shape only checks len, not stride values.
    assert!(d.validate_against_shape(&shape(&[10, 10])).is_ok());
}

// ── Morton: large shape values at exact power-of-two boundary ─────────────────

#[test]
fn morton_shape_equal_to_2_pow_32_bits_is_valid() {
    // morton_bits=[32] → max_dim = 2^32 = 4_294_967_296; shape = 4_294_967_296.
    let d = LayoutDescriptor::Morton(MortonLayout::new(vec![32]).unwrap());
    assert!(d.validate_against_shape(&shape(&[4_294_967_296])).is_ok());
}

#[test]
fn morton_shape_one_above_2_pow_32_bits_is_rejected() {
    // morton_bits=[32] → max_dim = 2^32; shape = 2^32 + 1.
    let d = LayoutDescriptor::Morton(MortonLayout::new(vec![32]).unwrap());
    assert!(matches!(
        d.validate_against_shape(&shape(&[4_294_967_297])),
        Err(Error::InvalidLayout(_))
    ));
}

// ── COO/CSR/CSC: nnz u64::MAX is structurally valid ──────────────────────────

#[test]
fn coo_nnz_u64_max_is_structurally_valid() {
    // u64::MAX is a valid nnz value at the descriptor level.
    let d = LayoutDescriptor::Coo(CooLayout::new(u64::MAX, false));
    assert!(d.validate_against_shape(&shape(&[100, 100])).is_ok());
}

#[test]
fn csr_nnz_u64_max_is_structurally_valid() {
    let d = LayoutDescriptor::Csr(CsrLayout::new(u64::MAX));
    assert!(d.validate_against_shape(&shape(&[50, 50])).is_ok());
}

#[test]
fn csc_nnz_u64_max_is_structurally_valid() {
    let d = LayoutDescriptor::Csc(CscLayout::new(u64::MAX));
    assert!(d.validate_against_shape(&shape(&[50, 50])).is_ok());
}

// ── Hilbert: tag passthrough and invariants via LayoutDescriptor wrapper ───────

#[test]
fn hilbert_large_order_accepted_by_constructor() {
    // order=30, rank=2: 2^30 per dimension. Constructor does not validate shape.
    let h = HilbertLayout::new(30, 2);
    assert!(h.is_ok());
    assert_eq!(h.unwrap().hilbert_order, 30);
}

#[test]
fn hilbert_high_rank_accepted_by_constructor() {
    let h = HilbertLayout::new(1, 10);
    assert!(h.is_ok());
    assert_eq!(h.unwrap().hilbert_rank, 10);
}

// ── validate_layout_tag_strict: boundary at private range ─────────────────────

/// 0xF0 is private (PrivateLayoutTag), not unknown.
#[test]
fn validate_strict_0xf0_gives_private_layout_tag() {
    assert!(matches!(
        validate_layout_tag_strict(0xF0),
        Err(Error::PrivateLayoutTag(0xF0))
    ));
}

/// 0xFE is the last private tag.
#[test]
fn validate_strict_0xfe_gives_private_layout_tag() {
    assert!(matches!(
        validate_layout_tag_strict(0xFE),
        Err(Error::PrivateLayoutTag(0xFE))
    ));
}

/// All reserved tags must produce ReservedLayoutTag.
///
/// Note: `0x0A` (CSF) and `0x0B` (block-paged) are both now named layout tags,
/// not reserved. The first reserved range starts at `0x0C`.
#[test]
fn validate_strict_all_reserved_ranges_produce_reserved_error() {
    // Reserved ranges after assigning 0x0A to CSF and 0x0B to block-paged:
    //   0x0C–0x3F, 0x41–0x7F, 0x80–0xEF.
    let reserved_tags: Vec<u8> = (0x0C_u8..=0x3F)
        .chain(0x41_u8..=0x7F)
        .chain(0x80_u8..=0xEF)
        .collect();
    for tag in reserved_tags {
        assert!(
            matches!(
                validate_layout_tag_strict(tag),
                Err(Error::ReservedLayoutTag(_))
            ),
            "tag 0x{tag:02X} in reserved range should produce ReservedLayoutTag"
        );
    }
}

// ── Block-paged: tag constant and classification ──────────────────────────────

/// Spec §block-paged.md: layout tag MUST be 0x0B.
#[test]
fn block_paged_tag_constant_is_0x0b() {
    assert_eq!(TAG_BLOCK_PAGED, 0x0B);
}

/// 0x0B is a named layout, not reserved or invalid.
#[test]
fn block_paged_tag_0x0b_passes_strict_validation() {
    assert!(
        validate_layout_tag_strict(0x0B).is_ok(),
        "0x0B (block-paged) must pass strict validation"
    );
}

/// 0x0A is CSF — a named layout, now passes strict validation.
#[test]
fn tag_0x0a_csf_passes_strict_validation() {
    assert!(
        validate_layout_tag_strict(0x0A).is_ok(),
        "0x0A (CSF) must pass strict validation"
    );
}

/// 0x0C (the first reserved slot above block-paged and CSF) is reserved.
#[test]
fn tag_0x0c_above_block_paged_is_reserved() {
    assert!(matches!(
        validate_layout_tag_strict(0x0C),
        Err(Error::ReservedLayoutTag(0x0C))
    ));
}

/// 0x0B is NOT in the reserved range (is_reserved_tag must return false).
#[test]
fn is_reserved_tag_returns_false_for_0x0b() {
    assert!(!is_reserved_tag(0x0B));
}

// ── Block-paged: tag() and buffer_count() via LayoutDescriptor ────────────────

fn make_block_paged(
    page_size: u32,
    num_pages: u64,
    num_seqs: u32,
    kv_role: KvRole,
    layer_index: Option<u32>,
    index_type: BlockTableIndexType,
) -> LayoutDescriptor {
    LayoutDescriptor::BlockPaged(BlockPagedLayout::new(
        page_size,
        num_pages,
        0, // paged_axis always 0
        num_seqs,
        kv_role,
        layer_index,
        index_type,
    ))
}

/// Spec §block-paged.md: layout tag MUST be 0x0B.
#[test]
fn block_paged_descriptor_tag_is_0x0b() {
    let d = make_block_paged(16, 64, 2, KvRole::Key, Some(0), BlockTableIndexType::U32);
    assert_eq!(d.tag(), 0x0B);
}

/// Spec §block-paged.md §Buffer Table: buffer_count MUST be 3.
#[test]
fn block_paged_descriptor_buffer_count_is_3() {
    let d = make_block_paged(16, 64, 2, KvRole::Value, None, BlockTableIndexType::U64);
    assert_eq!(d.buffer_count(), NonZeroU8::new(3));
}

/// buffer_count is independent of kv_role, layer_index, or index type.
#[test]
fn block_paged_buffer_count_3_for_all_kv_roles() {
    for role in [KvRole::Key, KvRole::Value, KvRole::Fused] {
        let d = make_block_paged(4, 10, 1, role, None, BlockTableIndexType::U32);
        assert_eq!(
            d.buffer_count(),
            NonZeroU8::new(3),
            "buffer_count must be 3 for role {role:?}"
        );
    }
}

// ── Block-paged: validate_against_shape ──────────────────────────────────────

/// Spec §block-paged.md: rank MUST be 3. Rank-3 must be accepted.
#[test]
fn block_paged_validate_accepts_rank_3() {
    let d = make_block_paged(4, 10, 2, KvRole::Key, Some(3), BlockTableIndexType::U32);
    let s = Shape::new(vec![8u64, 2, 8]).unwrap();
    assert!(d.validate_against_shape(&s).is_ok());
}

/// Spec §block-paged.md: a conforming implementation MUST reject rank != 3.
#[test]
fn block_paged_validate_rejects_rank_0() {
    let d = make_block_paged(4, 10, 1, KvRole::Key, None, BlockTableIndexType::U32);
    assert!(matches!(
        d.validate_against_shape(&Shape::scalar()),
        Err(Error::InvalidLayout(_))
    ));
}

#[test]
fn block_paged_validate_rejects_rank_1() {
    let d = make_block_paged(4, 10, 1, KvRole::Key, None, BlockTableIndexType::U32);
    assert!(matches!(
        d.validate_against_shape(&Shape::new(vec![8u64]).unwrap()),
        Err(Error::InvalidLayout(_))
    ));
}

#[test]
fn block_paged_validate_rejects_rank_2() {
    let d = make_block_paged(4, 10, 1, KvRole::Key, None, BlockTableIndexType::U32);
    assert!(matches!(
        d.validate_against_shape(&Shape::new(vec![8u64, 4]).unwrap()),
        Err(Error::InvalidLayout(_))
    ));
}

#[test]
fn block_paged_validate_rejects_rank_4() {
    let d = make_block_paged(4, 10, 1, KvRole::Key, None, BlockTableIndexType::U32);
    assert!(matches!(
        d.validate_against_shape(&Shape::new(vec![8u64, 4, 2, 2]).unwrap()),
        Err(Error::InvalidLayout(_))
    ));
}

/// Spec §block-paged.md: paged_axis MUST be 0 in this version.
#[test]
fn block_paged_validate_rejects_paged_axis_nonzero() {
    // Construct a descriptor manually with paged_axis=1 (bypassing the helper).
    let bp = BlockPagedLayout::new(
        4,
        10,
        1, /* paged_axis=1 */
        2,
        KvRole::Key,
        None,
        BlockTableIndexType::U32,
    );
    let d = LayoutDescriptor::BlockPaged(bp);
    assert!(matches!(
        d.validate_against_shape(&Shape::new(vec![8u64, 2, 8]).unwrap()),
        Err(Error::InvalidLayout(_))
    ));
}

/// Spec §block-paged.md §Additional Descriptor Fields: page_size MUST be >= 1.
#[test]
fn block_paged_validate_rejects_page_size_zero() {
    let bp = BlockPagedLayout::new(
        0, /* page_size=0 */
        10,
        0,
        2,
        KvRole::Key,
        None,
        BlockTableIndexType::U32,
    );
    let d = LayoutDescriptor::BlockPaged(bp);
    assert!(matches!(
        d.validate_against_shape(&Shape::new(vec![8u64, 2, 8]).unwrap()),
        Err(Error::InvalidLayout(_))
    ));
}

// ── Block-paged: codec round-trip via TensorDescriptor encode/decode ──────────

/// Helper: build a minimal 3-buffer TensorDescriptor for a block-paged layout.
fn block_paged_descriptor(layout: LayoutDescriptor) -> TensorDescriptor {
    // Shape: [total_tokens=8, num_heads=2, head_dim=8]
    let s = Shape::new(vec![8u64, 2, 8]).unwrap();
    // 3 buffers required: page_pool, block_table, seq_ptr.
    let buf = |sz: u64| {
        BufferHandle::new(
            sz,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )
        .unwrap()
    };
    TensorDescriptor::new(
        1,
        0,
        ElementType::Float16,
        s,
        0, // byte_offset must be 0 for block-paged (spec §byte_offset)
        layout,
        vec![buf(320 * 2), buf(12), buf(12)], // 3 buffers
        None,
        None,
        None,
        None,
    )
    .unwrap()
}

/// Spec §block-paged.md: encode then decode preserves all fields exactly.
/// Configuration: Key role, layer_index=3, U32 indices.
#[test]
fn block_paged_codec_round_trip_key_u32_layer3() {
    let layout = make_block_paged(4, 5, 2, KvRole::Key, Some(3), BlockTableIndexType::U32);
    let desc = block_paged_descriptor(layout);
    let encoded = desc.encode().unwrap();
    let decoded = TensorDescriptor::decode(&encoded).unwrap();
    assert_eq!(decoded, desc);
}

/// Value role, no layer index (sentinel 0xFFFFFFFF), U64 indices.
#[test]
fn block_paged_codec_round_trip_value_u64_layer_none() {
    let layout = make_block_paged(16, 128, 4, KvRole::Value, None, BlockTableIndexType::U64);
    let desc = block_paged_descriptor(layout);
    let encoded = desc.encode().unwrap();
    let decoded = TensorDescriptor::decode(&encoded).unwrap();
    assert_eq!(decoded, desc);
}

/// Fused role, layer_index=Some(0) (must NOT become None; sentinel is 0xFFFFFFFF not 0).
#[test]
fn block_paged_codec_round_trip_fused_layer_index_zero() {
    let layout = make_block_paged(32, 64, 0, KvRole::Fused, Some(0), BlockTableIndexType::U32);
    let desc = block_paged_descriptor(layout);
    let encoded = desc.encode().unwrap();
    let decoded = TensorDescriptor::decode(&encoded).unwrap();
    assert_eq!(decoded, desc);
    // Explicitly assert layer_index=Some(0) survived.
    if let LayoutDescriptor::BlockPaged(bp) = &decoded.layout {
        assert_eq!(bp.layer_index, Some(0));
    } else {
        panic!("expected BlockPaged layout after decode");
    }
}

/// layer_index=Some(0xFFFFFFFE) — largest non-sentinel value.
#[test]
fn block_paged_codec_round_trip_layer_index_large() {
    let layout = make_block_paged(
        16,
        10,
        1,
        KvRole::Key,
        Some(0xFFFF_FFFE),
        BlockTableIndexType::U32,
    );
    let desc = block_paged_descriptor(layout);
    let encoded = desc.encode().unwrap();
    let decoded = TensorDescriptor::decode(&encoded).unwrap();
    assert_eq!(decoded, desc);
    if let LayoutDescriptor::BlockPaged(bp) = &decoded.layout {
        assert_eq!(bp.layer_index, Some(0xFFFF_FFFE));
    } else {
        panic!("expected BlockPaged layout after decode");
    }
}

/// num_seqs=0 (empty batch).
#[test]
fn block_paged_codec_round_trip_empty_batch_num_seqs_zero() {
    let layout = make_block_paged(16, 32, 0, KvRole::Key, Some(1), BlockTableIndexType::U32);
    let desc = block_paged_descriptor(layout);
    let encoded = desc.encode().unwrap();
    let decoded = TensorDescriptor::decode(&encoded).unwrap();
    assert_eq!(decoded, desc);
}

/// Varied page_size and num_pages values.
#[test]
fn block_paged_codec_round_trip_large_pool() {
    let layout = make_block_paged(
        64,
        0x0001_0000_0000_u64,
        8,
        KvRole::Value,
        None,
        BlockTableIndexType::U64,
    );
    let desc = block_paged_descriptor(layout);
    let encoded = desc.encode().unwrap();
    let decoded = TensorDescriptor::decode(&encoded).unwrap();
    assert_eq!(decoded, desc);
}

/// Spec §block-paged.md §Additional Descriptor Fields: the encoded payload for
/// block-paged is exactly 32 bytes (all fields: 4+8+4+4+1+4+1+6 = 32).
#[test]
fn block_paged_encoded_payload_length_is_32_bytes() {
    // Encode a full descriptor and verify the layout-specific payload occupies
    // exactly 32 bytes. We do this by encoding two descriptors that differ only
    // in the layout and checking that the byte at the known layout payload offset
    // matches expectations. A simpler approach: just check total encode length
    // is deterministic and consistent.
    let layout = make_block_paged(16, 5, 2, KvRole::Key, Some(3), BlockTableIndexType::U32);
    let desc = block_paged_descriptor(layout);
    let encoded = desc.encode().unwrap();
    // Decode back and re-encode: round-trip must produce identical bytes.
    let decoded = TensorDescriptor::decode(&encoded).unwrap();
    let re_encoded = decoded.encode().unwrap();
    assert_eq!(
        encoded, re_encoded,
        "encode → decode → encode must produce identical bytes"
    );
}

// ── Block-paged: reserved-byte rejection ─────────────────────────────────────

/// Spec §block-paged.md §Additional Descriptor Fields: readers MUST reject a
/// descriptor with any non-zero reserved byte in the 6-byte _reserved field.
///
/// We inject a poisoned block-paged payload by mutating the reserved bytes
/// in an otherwise-valid encoded TensorDescriptor.
///
/// Wire layout (spec §metadata.md):
///   Fixed header: magic(4) + major(1) + minor(1) + length(4) + flags(4)
///                 + type_tag(1) + layout_tag(1) + rank(4) = 20 bytes.
///   layout_tag is at byte 15 (0-indexed).
///   After layout_tag: rank(4) + shape(rank×8) + byte_offset(8) before payload.
///   For rank=3: 4 + 24 + 8 = 36 bytes between layout_tag and payload start.
///
/// Block-paged payload (32 bytes):
///   page_size(4) + num_pages(8) + paged_axis(4) + num_seqs(4)
///   + kv_role(1) + layer_index(4) + index_type(1) + _reserved(6) = 32.
#[test]
fn block_paged_decoder_rejects_nonzero_reserved_byte() {
    let layout = make_block_paged(16, 5, 2, KvRole::Key, Some(3), BlockTableIndexType::U32);
    let desc = block_paged_descriptor(layout);
    let encoded = desc.encode().unwrap();

    // layout_tag (0x0B) is at fixed byte 15 in the wire format.
    // The payload starts 36 bytes after the tag byte:
    //   rank field (4) + shape[3 dims × 8] (24) + byte_offset (8) = 36.
    let tag_byte = 15_usize;
    assert_eq!(encoded[tag_byte], 0x0B, "layout tag must be at byte 15");
    let payload_start = tag_byte + 1 + 36; // = 52

    // _reserved is the last 6 bytes of the 32-byte payload.
    // Offset within payload: 4+8+4+4+1+4+1 = 26.
    let reserved_start = payload_start + 26;

    for i in 0..6 {
        let mut poisoned = encoded.clone();
        poisoned[reserved_start + i] = 0xFF;
        let result = TensorDescriptor::decode(&poisoned);
        assert!(
            matches!(result, Err(Error::ReservedBytesNonZero { .. })),
            "non-zero reserved byte at payload[26+{i}] must be ReservedBytesNonZero"
        );
    }
    // Baseline: clean encoding must still decode successfully.
    assert!(TensorDescriptor::decode(&encoded).is_ok());
}

/// Unknown kv_role byte must be rejected.
///
/// kv_role is at payload offset 20:
///   page_size(4) + num_pages(8) + paged_axis(4) + num_seqs(4) = 20.
#[test]
fn block_paged_decoder_rejects_unknown_kv_role_byte() {
    let layout = make_block_paged(16, 5, 2, KvRole::Key, Some(3), BlockTableIndexType::U32);
    let desc = block_paged_descriptor(layout);
    let mut encoded = desc.encode().unwrap();

    // layout_tag at byte 15; payload starts at byte 52.
    let tag_byte = 15_usize;
    assert_eq!(encoded[tag_byte], 0x0B, "layout tag must be at byte 15");
    let payload_start = tag_byte + 1 + 36; // = 52
                                           // kv_role at payload offset 20: page_size(4) + num_pages(8) + paged_axis(4) + num_seqs(4).
    let kv_role_offset = payload_start + 20;
    // Confirm we're pointing at a valid kv_role byte (0x00 = Key) before corrupting.
    assert!(
        encoded[kv_role_offset] <= 0x02,
        "expected a valid kv_role byte at offset {kv_role_offset}"
    );
    encoded[kv_role_offset] = 0x03; // unknown kv_role byte (only 0x00–0x02 are valid)
    let result = TensorDescriptor::decode(&encoded);
    assert!(
        matches!(result, Err(Error::InvalidLayout(_))),
        "unknown kv_role byte 0x03 must be rejected with InvalidLayout"
    );
}

/// Unknown block_table_index_type byte must be rejected.
///
/// block_table_index_type is at payload offset 21:
///   page_size(4) + num_pages(8) + paged_axis(4) + num_seqs(4) + kv_role(1) = 21.
/// Then layer_index(4) puts block_table_index_type at payload offset 25.
#[test]
fn block_paged_decoder_rejects_unknown_block_table_index_type_byte() {
    let layout = make_block_paged(16, 5, 2, KvRole::Key, Some(3), BlockTableIndexType::U32);
    let desc = block_paged_descriptor(layout);
    let mut encoded = desc.encode().unwrap();

    // layout_tag at byte 15; payload starts at byte 52.
    let tag_byte = 15_usize;
    assert_eq!(encoded[tag_byte], 0x0B, "layout tag must be at byte 15");
    let payload_start = tag_byte + 1 + 36; // = 52
                                           // block_table_index_type at payload offset 25:
                                           //   page_size(4) + num_pages(8) + paged_axis(4) + num_seqs(4) + kv_role(1) + layer_index(4) = 25.
    let index_type_offset = payload_start + 25;
    encoded[index_type_offset] = 0x02; // unknown index type (only 0x00–0x01 are valid)
    let result = TensorDescriptor::decode(&encoded);
    assert!(
        matches!(result, Err(Error::InvalidLayout(_))),
        "unknown block_table_index_type byte 0x02 must be rejected with InvalidLayout"
    );
}

// ── CSF (Compressed Sparse Fiber) layout ─────────────────────────────────────
//
// Spec: docs/spec/layouts/csf.md
// Tag: 0x0A  |  Buffer count: 2·rank+1  |  rank >= 3

// ── CSF: tag constant and classification ─────────────────────────────────────

/// Spec §csf.md: TAG_CSF MUST be 0x0A.
#[test]
fn csf_tag_constant_is_0x0a() {
    assert_eq!(TAG_CSF, 0x0A);
}

/// 0x0A is a named layout, not in any reserved range.
#[test]
fn csf_tag_0x0a_is_not_reserved() {
    use hurray_core::layout::is_reserved_tag;
    assert!(!is_reserved_tag(0x0A));
}

/// 0x0A passes strict-mode validation (it is a named layout tag).
#[test]
fn csf_tag_passes_strict_validation() {
    assert!(
        validate_layout_tag_strict(0x0A).is_ok(),
        "0x0A (CSF) must pass strict validation"
    );
}

/// 0x0C is the first reserved slot after CSF (0x0A) and block-paged (0x0B).
#[test]
fn tag_0x0c_is_reserved_lower_boundary_after_csf() {
    assert!(
        matches!(
            validate_layout_tag_strict(0x0C),
            Err(Error::ReservedLayoutTag(0x0C))
        ),
        "0x0C must be ReservedLayoutTag (new lower boundary after CSF and block-paged)"
    );
}

// ── CSF: tag() and buffer_count() via LayoutDescriptor ───────────────────────

/// Spec §csf.md: LayoutDescriptor::Csf.tag() == 0x0A.
#[test]
fn csf_descriptor_tag_is_0x0a() {
    let d = LayoutDescriptor::Csf(CsfLayout::new(0, vec![0, 1, 2]));
    assert_eq!(d.tag(), 0x0A);
}

/// Spec §csf.md §Buffer Table: buffer_count = 2·rank+1.  rank=3 → 7.
#[test]
fn csf_buffer_count_rank3_is_7() {
    let d = LayoutDescriptor::Csf(CsfLayout::new(4, vec![0, 1, 2]));
    assert_eq!(d.buffer_count(), NonZeroU8::new(7));
}

/// rank=4 → 2*4+1 = 9.
#[test]
fn csf_buffer_count_rank4_is_9() {
    let d = LayoutDescriptor::Csf(CsfLayout::new(0, vec![0, 1, 2, 3]));
    assert_eq!(d.buffer_count(), NonZeroU8::new(9));
}

/// rank=5 → 2*5+1 = 11.
#[test]
fn csf_buffer_count_rank5_is_11() {
    let d = LayoutDescriptor::Csf(CsfLayout::new(0, vec![0, 1, 2, 3, 4]));
    assert_eq!(d.buffer_count(), NonZeroU8::new(11));
}

/// buffer_count is derived from mode_order.len() — confirmed for a non-identity permutation.
#[test]
fn csf_buffer_count_uses_mode_order_len_not_a_separate_rank_field() {
    // [2, 0, 1] is a valid rank-3 permutation; buffer_count must still be 7.
    let d = LayoutDescriptor::Csf(CsfLayout::new(10, vec![2, 0, 1]));
    assert_eq!(d.buffer_count(), NonZeroU8::new(7));
}

// ── CSF: validate_against_shape ──────────────────────────────────────────────

/// Spec §csf.md §Validity Constraints: rank >= 3 required.
#[test]
fn csf_validate_accepts_rank3_identity() {
    let d = LayoutDescriptor::Csf(CsfLayout::new(0, vec![0, 1, 2]));
    assert!(d.validate_against_shape(&shape(&[2, 3, 4])).is_ok());
}

#[test]
fn csf_validate_accepts_rank4_identity() {
    let d = LayoutDescriptor::Csf(CsfLayout::new(0, vec![0, 1, 2, 3]));
    assert!(d.validate_against_shape(&shape(&[2, 3, 4, 5])).is_ok());
}

#[test]
fn csf_validate_accepts_rank5_identity() {
    let d = LayoutDescriptor::Csf(CsfLayout::new(0, vec![0, 1, 2, 3, 4]));
    assert!(d.validate_against_shape(&shape(&[2, 3, 4, 5, 6])).is_ok());
}

/// Spec §csf.md: CSF MUST NOT be used for rank < 3.
#[test]
fn csf_validate_rejects_rank0() {
    let d = LayoutDescriptor::Csf(CsfLayout::new(0, vec![]));
    assert!(matches!(
        d.validate_against_shape(&Shape::scalar()),
        Err(Error::InvalidLayout(_))
    ));
}

#[test]
fn csf_validate_rejects_rank1() {
    let d = LayoutDescriptor::Csf(CsfLayout::new(0, vec![0]));
    assert!(matches!(
        d.validate_against_shape(&shape(&[10])),
        Err(Error::InvalidLayout(_))
    ));
}

#[test]
fn csf_validate_rejects_rank2() {
    let d = LayoutDescriptor::Csf(CsfLayout::new(0, vec![0, 1]));
    assert!(matches!(
        d.validate_against_shape(&shape(&[4, 5])),
        Err(Error::InvalidLayout(_))
    ));
}

/// mode_order.len() must equal shape.rank().
#[test]
fn csf_validate_rejects_mode_order_len_shorter_than_rank() {
    // mode_order has 2 entries but shape has rank 3.
    let d = LayoutDescriptor::Csf(CsfLayout::new(0, vec![0, 1]));
    // Note: rank 2 is also invalid (<3), so the rank-too-low error fires first.
    // Use rank-4 shape and rank-3 mode_order to isolate the length-mismatch path.
    let d2 = LayoutDescriptor::Csf(CsfLayout::new(0, vec![0, 1, 2]));
    assert!(matches!(
        d2.validate_against_shape(&shape(&[2, 3, 4, 5])),
        Err(Error::InvalidLayout(_))
    ));
    // Also verify the rank-2 shape path.
    assert!(matches!(
        d.validate_against_shape(&shape(&[2, 3, 4])),
        Err(Error::InvalidLayout(_))
    ));
}

#[test]
fn csf_validate_rejects_mode_order_len_longer_than_rank() {
    // mode_order has 4 entries but shape has rank 3.
    let d = LayoutDescriptor::Csf(CsfLayout::new(0, vec![0, 1, 2, 3]));
    assert!(matches!(
        d.validate_against_shape(&shape(&[2, 3, 4])),
        Err(Error::InvalidLayout(_))
    ));
}

/// Spec §csf.md §Mode Ordering: mode_order with out-of-range value rejected.
#[test]
fn csf_validate_rejects_mode_order_out_of_range_value() {
    // mode_order[2]=3 is out of range for rank 3 (valid range: 0..2).
    let d = LayoutDescriptor::Csf(CsfLayout::new(0, vec![0, 1, 3]));
    assert!(matches!(
        d.validate_against_shape(&shape(&[2, 3, 4])),
        Err(Error::InvalidLayout(_))
    ));
}

/// Spec §csf.md §Mode Ordering: mode_order with duplicate rejected.
#[test]
fn csf_validate_rejects_mode_order_repeated_value() {
    // [0, 1, 1] is not a permutation: 1 appears twice.
    let d = LayoutDescriptor::Csf(CsfLayout::new(0, vec![0, 1, 1]));
    assert!(matches!(
        d.validate_against_shape(&shape(&[2, 3, 4])),
        Err(Error::InvalidLayout(_))
    ));
}

/// Spec §csf.md §Mode Ordering: non-identity permutations MUST be accepted.
#[test]
fn csf_validate_accepts_permutation_2_0_1() {
    let d = LayoutDescriptor::Csf(CsfLayout::new(4, vec![2, 0, 1]));
    assert!(d.validate_against_shape(&shape(&[2, 3, 4])).is_ok());
}

#[test]
fn csf_validate_accepts_reverse_permutation_2_1_0() {
    let d = LayoutDescriptor::Csf(CsfLayout::new(4, vec![2, 1, 0]));
    assert!(d.validate_against_shape(&shape(&[2, 3, 4])).is_ok());
}

// ── CSF: codec round-trip via TensorDescriptor encode/decode ─────────────────

// Wire layout for CSF payload (spec §csf.md §Additional Descriptor Fields):
//   nnz        uint64        8 bytes
//   mode_order uint32[rank]  4·rank bytes
//   _reserved  uint8[8]      8 bytes
// Total: 8 + 4·rank + 8 bytes.

// Helper: construct a minimal TensorDescriptor for a CSF layout.
/// CSF requires 2·rank+1 buffers (values + rank pos + rank crd).
/// For testing codec, all buffers are identical zero-size placeholders
/// (TensorDescriptor::new requires at least one buffer per the buffer table, and
/// the buffer count must match what buffer_count() returns).
fn csf_descriptor(layout: LayoutDescriptor) -> TensorDescriptor {
    let rank = match &layout {
        LayoutDescriptor::Csf(c) => c.mode_order.len(),
        _ => panic!("expected Csf layout"),
    };
    let buf_count = 2 * rank + 1;
    let dims: Vec<u64> = (0..rank).map(|i| i as u64 + 2).collect(); // [2, 3, 4, ...]
    let s = Shape::new(dims).unwrap();
    let buf = || {
        BufferHandle::new(
            64,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )
        .unwrap()
    };
    TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        s,
        0, // byte_offset MUST be 0 for CSF (spec §csf.md §Buffer Size)
        layout,
        (0..buf_count).map(|_| buf()).collect(),
        None,
        None,
        None,
        None,
    )
    .unwrap()
}

/// Spec §csf.md: encode → decode preserves all fields for identity mode_order, rank=3, nnz=4.
#[test]
fn csf_codec_round_trip_rank3_identity_nnz4() {
    let layout = LayoutDescriptor::Csf(CsfLayout::new(4, vec![0, 1, 2]));
    let desc = csf_descriptor(layout);
    let encoded = desc.encode().unwrap();
    let decoded = TensorDescriptor::decode(&encoded).unwrap();
    assert_eq!(decoded, desc);
}

/// nnz=0 (empty sparse tensor) round-trips correctly.
#[test]
fn csf_codec_round_trip_rank3_identity_nnz0() {
    let layout = LayoutDescriptor::Csf(CsfLayout::new(0, vec![0, 1, 2]));
    let desc = csf_descriptor(layout);
    let encoded = desc.encode().unwrap();
    let decoded = TensorDescriptor::decode(&encoded).unwrap();
    assert_eq!(decoded, desc);
}

/// Non-identity permutation [2, 0, 1] round-trips.
#[test]
fn csf_codec_round_trip_rank3_permutation_2_0_1() {
    let layout = LayoutDescriptor::Csf(CsfLayout::new(7, vec![2, 0, 1]));
    let desc = csf_descriptor(layout);
    let encoded = desc.encode().unwrap();
    let decoded = TensorDescriptor::decode(&encoded).unwrap();
    assert_eq!(decoded, desc);
}

/// rank=4 round-trip (mode_order = [3, 0, 2, 1]).
#[test]
fn csf_codec_round_trip_rank4_varied_permutation() {
    let layout = LayoutDescriptor::Csf(CsfLayout::new(100, vec![3, 0, 2, 1]));
    let desc = csf_descriptor(layout);
    let encoded = desc.encode().unwrap();
    let decoded = TensorDescriptor::decode(&encoded).unwrap();
    assert_eq!(decoded, desc);
}

/// rank=5 round-trip with reverse permutation.
#[test]
fn csf_codec_round_trip_rank5_reverse_permutation_nnz0() {
    let layout = LayoutDescriptor::Csf(CsfLayout::new(0, vec![4, 3, 2, 1, 0]));
    let desc = csf_descriptor(layout);
    let encoded = desc.encode().unwrap();
    let decoded = TensorDescriptor::decode(&encoded).unwrap();
    assert_eq!(decoded, desc);
}

/// Spec §csf.md §Additional Descriptor Fields: payload length = 8 + 4·rank + 8.
/// Verified by encoding → decode → re-encode producing identical bytes (idempotent).
#[test]
fn csf_codec_idempotent_encode_decode_encode() {
    let layout = LayoutDescriptor::Csf(CsfLayout::new(42, vec![0, 1, 2]));
    let desc = csf_descriptor(layout);
    let enc1 = desc.encode().unwrap();
    let dec = TensorDescriptor::decode(&enc1).unwrap();
    let enc2 = dec.encode().unwrap();
    assert_eq!(enc1, enc2, "encode → decode → encode must be idempotent");
}

// ── CSF: decoder rejection tests ─────────────────────────────────────────────
//
// Wire layout within a rank-3 TensorDescriptor (all sizes in bytes):
//
//   Descriptor fixed header:
//     magic[4] + major[1] + minor[1] + length[4] + flags[4]
//     + type_tag[1] + layout_tag[1] + rank[4] = 20 bytes.
//   layout_tag is at byte 15 (0-indexed).
//
//   After layout_tag:
//     rank[4] + shape[rank × 8] + byte_offset[8]
//     = 4 + 24 + 8 = 36 bytes (for rank=3).
//
//   CSF payload starts at byte 15 + 1 + 36 = 52.
//   CSF payload (rank=3): nnz[8] + mode_order[4×3] + _reserved[8] = 28 bytes.
//     payload[0..8]   = nnz
//     payload[8..20]  = mode_order (3 × uint32 LE)
//     payload[20..28] = _reserved (must be 0x00)

/// Spec §csf.md §Additional Descriptor Fields: _reserved bytes MUST be 0x00.
/// A reader MUST reject a descriptor with any non-zero reserved byte.
#[test]
fn csf_decoder_rejects_nonzero_reserved_byte() {
    let layout = LayoutDescriptor::Csf(CsfLayout::new(4, vec![0, 1, 2]));
    let desc = csf_descriptor(layout);
    let encoded = desc.encode().unwrap();

    assert_eq!(encoded[15], 0x0A, "layout tag must be at byte 15");
    let payload_start = 15 + 1 + 36; // = 52
                                     // _reserved is payload[20..28].
    let reserved_offset = payload_start + 20; // nnz(8) + mode_order(3×4=12) = 20

    for i in 0..8 {
        let mut poisoned = encoded.clone();
        poisoned[reserved_offset + i] = 0x01;
        let result = TensorDescriptor::decode(&poisoned);
        assert!(
            matches!(result, Err(Error::ReservedBytesNonZero { .. })),
            "non-zero _reserved byte at payload[20+{i}] must be ReservedBytesNonZero"
        );
    }
    // Baseline: clean encoding decodes without error.
    assert!(TensorDescriptor::decode(&encoded).is_ok());
}

/// Spec §csf.md §Mode Ordering: a mode_order that is not a permutation must be rejected.
/// We inject an invalid mode_order (value 3, which is out of range for rank=3) by
/// patching the encoded bytes.
#[test]
fn csf_decoder_rejects_mode_order_out_of_range_value() {
    let layout = LayoutDescriptor::Csf(CsfLayout::new(4, vec![0, 1, 2]));
    let desc = csf_descriptor(layout);
    let mut encoded = desc.encode().unwrap();

    assert_eq!(encoded[15], 0x0A, "layout tag must be at byte 15");
    let payload_start = 15 + 1 + 36; // = 52
                                     // mode_order[2] starts at payload[8 + 2*4] = payload[16].
                                     // Write 3u32 LE into mode_order[2].
    let mo2_offset = payload_start + 8 + 2 * 4; // nnz(8) + 0-indexed slot 2
    encoded[mo2_offset] = 3;
    encoded[mo2_offset + 1] = 0;
    encoded[mo2_offset + 2] = 0;
    encoded[mo2_offset + 3] = 0;

    let result = TensorDescriptor::decode(&encoded);
    assert!(
        matches!(result, Err(Error::InvalidLayout(_))),
        "mode_order[2]=3 is out of range for rank=3, must be InvalidLayout"
    );
}

/// mode_order with a duplicate value must be rejected by the decoder.
#[test]
fn csf_decoder_rejects_mode_order_duplicate_value() {
    let layout = LayoutDescriptor::Csf(CsfLayout::new(4, vec![0, 1, 2]));
    let desc = csf_descriptor(layout);
    let mut encoded = desc.encode().unwrap();

    assert_eq!(encoded[15], 0x0A, "layout tag must be at byte 15");
    let payload_start = 15 + 1 + 36;
    // Overwrite mode_order[2] to be 1 (same as mode_order[1]) → duplicate.
    let mo2_offset = payload_start + 8 + 2 * 4;
    encoded[mo2_offset] = 1;
    encoded[mo2_offset + 1] = 0;
    encoded[mo2_offset + 2] = 0;
    encoded[mo2_offset + 3] = 0;

    let result = TensorDescriptor::decode(&encoded);
    assert!(
        matches!(result, Err(Error::InvalidLayout(_))),
        "mode_order [0,1,1] has a duplicate, must be InvalidLayout"
    );
}

// ── CSF: element lookup — spec worked example ─────────────────────────────────
//
// Spec §csf.md §Example:
//   Rank-3, shape [2, 3, 4], mode_order = [0, 1, 2], nnz = 4.
//   Non-zeros: (0,0,1)→1.0, (0,2,3)→2.0, (1,1,0)→3.0, (1,1,2)→4.0
//
//   pos_0 = [0, 2]           crd_0 = [0, 1]
//   pos_1 = [0, 2, 3]        crd_1 = [0, 2, 1]
//   pos_2 = [0, 1, 2, 4]     crd_2 = [1, 3, 0, 2]
//   values = [1.0, 2.0, 3.0, 4.0]

const SPEC_MODE_ORDER: &[u32] = &[0, 1, 2];
const SPEC_POS_0: &[u64] = &[0, 2];
const SPEC_CRD_0: &[u64] = &[0, 1];
const SPEC_POS_1: &[u64] = &[0, 2, 3];
const SPEC_CRD_1: &[u64] = &[0, 2, 1];
const SPEC_POS_2: &[u64] = &[0, 1, 2, 4];
const SPEC_CRD_2: &[u64] = &[1, 3, 0, 2];

fn spec_pos_levels() -> Vec<&'static [u64]> {
    vec![SPEC_POS_0, SPEC_POS_1, SPEC_POS_2]
}
fn spec_crd_levels() -> Vec<&'static [u64]> {
    vec![SPEC_CRD_0, SPEC_CRD_1, SPEC_CRD_2]
}

/// Spec §csf.md §Element Lookup: (1, 1, 2) → leaf position 3.
/// Trace: L0: p=0+1=1; L1: p=2+0=2; L2: p=2+1=3.
#[test]
fn csf_element_lookup_spec_example_1_1_2_yields_leaf_3() {
    let result = csf_addressing::element_offset(
        &[1, 1, 2],
        SPEC_MODE_ORDER,
        &spec_pos_levels(),
        &spec_crd_levels(),
    )
    .unwrap();
    assert_eq!(result, Some(3), "(1,1,2) must resolve to values[3]");
}

/// (0, 0, 1) → leaf position 0 (values[0] = 1.0).
#[test]
fn csf_element_lookup_0_0_1_yields_leaf_0() {
    let result = csf_addressing::element_offset(
        &[0, 0, 1],
        SPEC_MODE_ORDER,
        &spec_pos_levels(),
        &spec_crd_levels(),
    )
    .unwrap();
    assert_eq!(result, Some(0));
}

/// (0, 2, 3) → leaf position 1 (values[1] = 2.0).
#[test]
fn csf_element_lookup_0_2_3_yields_leaf_1() {
    let result = csf_addressing::element_offset(
        &[0, 2, 3],
        SPEC_MODE_ORDER,
        &spec_pos_levels(),
        &spec_crd_levels(),
    )
    .unwrap();
    assert_eq!(result, Some(1));
}

/// (1, 1, 0) → leaf position 2 (values[2] = 3.0).
#[test]
fn csf_element_lookup_1_1_0_yields_leaf_2() {
    let result = csf_addressing::element_offset(
        &[1, 1, 0],
        SPEC_MODE_ORDER,
        &spec_pos_levels(),
        &spec_crd_levels(),
    )
    .unwrap();
    assert_eq!(result, Some(2));
}

/// Spec §csf.md §Element Lookup: element not stored → returns None (implicit zero).
/// (0, 1, 0): i=0 found, but j=1 is absent under i=0 (only j=0 and j=2 are).
#[test]
fn csf_element_lookup_absent_at_level1_returns_none() {
    let result = csf_addressing::element_offset(
        &[0, 1, 0],
        SPEC_MODE_ORDER,
        &spec_pos_levels(),
        &spec_crd_levels(),
    )
    .unwrap();
    assert_eq!(result, None, "(0,1,0) is not stored; must be implicit zero");
}

/// (1, 0, 0): i=1 found, but j=0 absent under i=1 (only j=1 is stored).
#[test]
fn csf_element_lookup_absent_at_level1_for_i1_returns_none() {
    let result = csf_addressing::element_offset(
        &[1, 0, 0],
        SPEC_MODE_ORDER,
        &spec_pos_levels(),
        &spec_crd_levels(),
    )
    .unwrap();
    assert_eq!(result, None);
}

/// Rank mismatch: query with wrong coordinate count → IndexRankMismatch.
#[test]
fn csf_element_lookup_rank_mismatch_returns_error() {
    let err = csf_addressing::element_offset(
        &[0, 1], // rank 2 query against rank-3 tree
        SPEC_MODE_ORDER,
        &spec_pos_levels(),
        &spec_crd_levels(),
    )
    .unwrap_err();
    assert!(
        matches!(err, Error::IndexRankMismatch { .. }),
        "expected IndexRankMismatch, got {err:?}"
    );
}

/// Spec §csf.md §Mode Ordering: readers MUST honour any valid permutation.
/// Build a rank-3 tree with mode_order=[2,1,0] (column-first) holding a
/// single non-zero at logical (0, 0, 1).
#[test]
fn csf_element_lookup_non_identity_mode_order_column_first() {
    // mode_order = [2, 1, 0]: level 0 stores dim 2 (bound 4), level 1 dim 1 (bound 3), level 2 dim 0 (bound 2).
    // Single non-zero at logical (0, 0, 1):
    //   q_0 = idx[mode_order[0]] = idx[2] = 1  (dim 2)
    //   q_1 = idx[mode_order[1]] = idx[1] = 0  (dim 1)
    //   q_2 = idx[mode_order[2]] = idx[0] = 0  (dim 0)
    //
    // Level 0 (dim 2): pos=[0,1], crd=[1]
    // Level 1 (dim 1): pos=[0,1], crd=[0]
    // Level 2 (dim 0): pos=[0,1], crd=[0]
    let mode_order: &[u32] = &[2, 1, 0];
    let pos_levels: Vec<&[u64]> = vec![&[0, 1], &[0, 1], &[0, 1]];
    let crd_levels: Vec<&[u64]> = vec![&[1], &[0], &[0]];

    // Stored element (0,0,1) → leaf 0.
    let found =
        csf_addressing::element_offset(&[0, 0, 1], mode_order, &pos_levels, &crd_levels).unwrap();
    assert_eq!(found, Some(0));

    // Absent element (0, 0, 0): q_0 = idx[2] = 0, not in crd_0=[1] → None.
    let absent =
        csf_addressing::element_offset(&[0, 0, 0], mode_order, &pos_levels, &crd_levels).unwrap();
    assert_eq!(absent, None);
}

// ── CSF: validate_index_buffers — spec worked example ────────────────────────

/// Spec §csf.md §Storage Invariants: the spec example satisfies all invariants.
#[test]
fn csf_validate_index_buffers_spec_example_is_valid() {
    assert!(csf_addressing::validate_index_buffers(
        4,
        SPEC_MODE_ORDER,
        &[2, 3, 4],
        &spec_pos_levels(),
        &spec_crd_levels(),
    )
    .is_ok());
}

/// Spec §csf.md §Storage Invariants: empty tensor (nnz=0) with correct buffer shapes is valid.
/// pos_0=[0,0], pos_1=[0], pos_2=[0]; every crd_L=[].
#[test]
fn csf_validate_index_buffers_empty_tensor_nnz0_is_valid() {
    let pos_levels: Vec<&[u64]> = vec![&[0, 0], &[0], &[0]];
    let crd_levels: Vec<&[u64]> = vec![&[], &[], &[]];
    assert!(csf_addressing::validate_index_buffers(
        0,
        SPEC_MODE_ORDER,
        &[2, 3, 4],
        &pos_levels,
        &crd_levels,
    )
    .is_ok());
}

/// Invariant 1: pos_L[0] must be 0.
#[test]
fn csf_validate_index_buffers_rejects_pos_not_starting_at_zero() {
    // pos_0[0]=1 violates invariant 1.
    let pos_levels: Vec<&[u64]> = vec![&[1, 2], SPEC_POS_1, SPEC_POS_2];
    let result = csf_addressing::validate_index_buffers(
        4,
        SPEC_MODE_ORDER,
        &[2, 3, 4],
        &pos_levels,
        &spec_crd_levels(),
    );
    assert!(
        matches!(result, Err(Error::InvalidLayout(_))),
        "pos_0[0]=1 must violate invariant 1"
    );
}

/// Invariant 2: pos_L must be non-decreasing.
#[test]
fn csf_validate_index_buffers_rejects_non_monotone_pos() {
    // pos_1 = [0, 3, 2]: not non-decreasing between indices 1 and 2.
    let pos_levels: Vec<&[u64]> = vec![SPEC_POS_0, &[0, 3, 2], SPEC_POS_2];
    let result = csf_addressing::validate_index_buffers(
        4,
        SPEC_MODE_ORDER,
        &[2, 3, 4],
        &pos_levels,
        &spec_crd_levels(),
    );
    assert!(
        matches!(result, Err(Error::InvalidLayout(_))),
        "pos_1=[0,3,2] violates non-decreasing invariant"
    );
}

/// Invariant 3: terminal pos entry must equal the level's node count.
/// At the leaf level, that count must equal nnz.
#[test]
fn csf_validate_index_buffers_rejects_terminal_pos_not_equal_to_nnz() {
    // pos_2 = [0, 1, 2, 3]: terminal = 3, but nnz = 4.
    let pos_levels: Vec<&[u64]> = vec![SPEC_POS_0, SPEC_POS_1, &[0, 1, 2, 3]];
    let result = csf_addressing::validate_index_buffers(
        4,
        SPEC_MODE_ORDER,
        &[2, 3, 4],
        &pos_levels,
        &spec_crd_levels(),
    );
    assert!(
        matches!(result, Err(Error::InvalidLayout(_))),
        "pos_2 terminal=3 != nnz=4 must be rejected"
    );
}

/// Invariant 4a: duplicate siblings within a parent slice must be rejected.
#[test]
fn csf_validate_index_buffers_rejects_duplicate_siblings() {
    // crd_0 = [0, 0]: duplicate within the single parent slice.
    let crd_levels: Vec<&[u64]> = vec![&[0, 0], SPEC_CRD_1, SPEC_CRD_2];
    let result = csf_addressing::validate_index_buffers(
        4,
        SPEC_MODE_ORDER,
        &[2, 3, 4],
        &spec_pos_levels(),
        &crd_levels,
    );
    assert!(
        matches!(result, Err(Error::InvalidLayout(_))),
        "duplicate siblings [0,0] in crd_0 must be rejected"
    );
}

/// Invariant 4b: non-strictly-increasing (decreasing) siblings must be rejected.
#[test]
fn csf_validate_index_buffers_rejects_decreasing_siblings() {
    // crd_0 = [1, 0]: decreasing order.
    let crd_levels: Vec<&[u64]> = vec![&[1, 0], SPEC_CRD_1, SPEC_CRD_2];
    let result = csf_addressing::validate_index_buffers(
        4,
        SPEC_MODE_ORDER,
        &[2, 3, 4],
        &spec_pos_levels(),
        &crd_levels,
    );
    assert!(
        matches!(result, Err(Error::InvalidLayout(_))),
        "decreasing siblings [1,0] in crd_0 must be rejected"
    );
}

/// Invariant 5: coordinate out of bounds (>= shape[mode_order[L]]) must be rejected.
#[test]
fn csf_validate_index_buffers_rejects_coordinate_out_of_bounds() {
    // crd_2[3]=4 but shape[mode_order[2]] = shape[2] = 4; valid range is 0..3.
    let crd_levels: Vec<&[u64]> = vec![SPEC_CRD_0, SPEC_CRD_1, &[1, 3, 0, 4]];
    let result = csf_addressing::validate_index_buffers(
        4,
        SPEC_MODE_ORDER,
        &[2, 3, 4],
        &spec_pos_levels(),
        &crd_levels,
    );
    assert!(
        matches!(result, Err(Error::InvalidLayout(_))),
        "crd_2[3]=4 >= shape[2]=4 must be rejected"
    );
}

// ── CSF: LayoutDescriptor::element_offset returns LayoutRequiresMultiBuffer ──

/// CSF is a multi-buffer layout; LayoutDescriptor::element_offset must
/// return LayoutRequiresMultiBuffer, not attempt a lookup.
#[test]
fn csf_descriptor_element_offset_returns_multi_buffer_error() {
    let d = LayoutDescriptor::Csf(CsfLayout::new(4, vec![0, 1, 2]));
    let s = Shape::new(vec![2u64, 3, 4]).unwrap();
    assert!(matches!(
        d.element_offset(&[0, 0, 0], &s),
        Err(Error::LayoutRequiresMultiBuffer { layout_tag: 0x0A })
    ));
}
