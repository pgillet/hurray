//! Layout descriptors for the Hurray tensor interchange format.
//!
//! Every tensor descriptor includes a **layout tag** (`uint8`) and a
//! layout-specific payload that together describe how elements are arranged in
//! memory. This module provides the [`LayoutDescriptor`] enum and all per-layout
//! payload structs.
//!
//! ## Layout tag space
//!
//! | Range | Allocation |
//! |-------|-----------|
//! | `0x00` | Reserved (permanently invalid) |
//! | `0x01`–`0x09` | Core named layouts (Tier 1) |
//! | `0x0A` | Reserved for future specification versions |
//! | `0x0B` | Block-paged indirect layout (Tier 1) |
//! | `0x0C`–`0x3F` | Reserved for future specification versions |
//! | `0x40` | Hilbert curve layout (Tier 2) |
//! | `0x41`–`0x7F` | Reserved for future specification versions |
//! | `0x80`–`0xEF` | Reserved for future specification versions |
//! | `0xF0`–`0xFE` | Implementation-private extension layouts |
//! | `0xFF` | Reserved (permanently invalid) |
//!
//! ## Strict vs permissive mode
//!
//! The [`LayoutDescriptor`] enum's named variants (everything except
//! [`LayoutDescriptor::Unknown`]) correspond to tags this implementation
//! understands. A decoder operating in **strict mode** must reject any tag not
//! covered by a named variant; a decoder operating in **permissive mode** may
//! wrap unrecognised tags in [`LayoutDescriptor::Unknown`].
//!
//! [`LayoutDescriptor::tag`] and [`LayoutDescriptor::buffer_count`] work on all
//! variants including `Unknown`.
//!
//! See `docs/spec/memory-layout.md` for the normative definition.

pub mod addressing;
pub mod block_paged;
pub mod col_major;
pub mod coo;
pub mod csc;
pub mod csr;
pub mod hilbert;
pub mod morton;
pub mod private_extension;
pub mod row_major;
pub mod strided;
pub mod subpaving;
pub mod tiled;
pub mod unknown;

pub use addressing::{byte_address_from_element_offset, ElementAddress, SubpavingLocation};
pub use block_paged::{BlockPagedLayout, BlockTableIndexType, KvRole};
pub use coo::CooLayout;
pub use csc::CscLayout;
pub use csr::CsrLayout;
pub use hilbert::HilbertLayout;
pub use morton::MortonLayout;
pub use private_extension::{
    PrivateExtensionLayout, PRIVATE_LAYOUT_TAG_MAX, PRIVATE_LAYOUT_TAG_MIN,
};
pub use strided::StridedLayout;
pub use subpaving::{RegionDescriptor, SubpavingLayout};
pub use tiled::{InnerStrides, OuterStrides, TiledLayout, MAX_TILED_DEPTH};
pub use unknown::UnknownLayout;

use std::num::NonZeroU8;

use crate::{Error, Result, Shape};

// ── Tag constants (public, for use in Layer 4 encoding) ──────────────────────

/// Layout tag for row-major (C-order) layout.
pub const TAG_ROW_MAJOR: u8 = 0x01;
/// Layout tag for column-major (Fortran-order) layout.
pub const TAG_COL_MAJOR: u8 = 0x02;
/// Layout tag for strided layout.
pub const TAG_STRIDED: u8 = 0x03;
/// Layout tag for tiled / blocked layout.
pub const TAG_TILED: u8 = 0x04;
/// Layout tag for Morton (Z-order) layout.
pub const TAG_MORTON: u8 = 0x05;
/// Layout tag for general subpaving layout.
pub const TAG_SUBPAVING: u8 = 0x06;
/// Layout tag for COO (Coordinate) sparse layout.
pub const TAG_COO: u8 = 0x07;
/// Layout tag for CSR (Compressed Sparse Row) layout.
pub const TAG_CSR: u8 = 0x08;
/// Layout tag for CSC (Compressed Sparse Column) layout.
pub const TAG_CSC: u8 = 0x09;
/// Layout tag for block-paged indirect layout.
pub const TAG_BLOCK_PAGED: u8 = 0x0B;
/// Layout tag for Hilbert curve layout.
pub const TAG_HILBERT: u8 = 0x40;

// ── Layout tag classification helpers ────────────────────────────────────────

/// Returns `true` if `tag` is permanently invalid (`0x00` or `0xFF`).
///
/// # Examples
///
/// ```
/// use hurray_core::layout::is_invalid_tag;
///
/// assert!(is_invalid_tag(0x00));
/// assert!(is_invalid_tag(0xFF));
/// assert!(!is_invalid_tag(0x01));
/// ```
#[inline]
pub fn is_invalid_tag(tag: u8) -> bool {
    tag == 0x00 || tag == 0xFF
}

/// Returns `true` if `tag` falls in a range reserved for future specification
/// versions (`0x0A`, `0x0C`–`0x3F`, `0x41`–`0x7F`, `0x80`–`0xEF`).
///
/// Tag `0x0B` (block-paged) is NOT reserved — it is a named layout tag.
///
/// # Examples
///
/// ```
/// use hurray_core::layout::is_reserved_tag;
///
/// assert!(is_reserved_tag(0x0A));
/// assert!(!is_reserved_tag(0x0B)); // block-paged — named tag
/// assert!(is_reserved_tag(0x0C));
/// assert!(is_reserved_tag(0x3F));
/// assert!(is_reserved_tag(0x41));
/// assert!(is_reserved_tag(0xEF));
/// assert!(!is_reserved_tag(0x01));
/// assert!(!is_reserved_tag(0x40));
/// ```
#[inline]
pub fn is_reserved_tag(tag: u8) -> bool {
    // 0x0B is TAG_BLOCK_PAGED — a named layout, not reserved.
    matches!(tag, 0x0A | 0x0C..=0x3F | 0x41..=0x7F | 0x80..=0xEF)
}

/// Returns `true` if `tag` is in the private-extension range (`0xF0`–`0xFE`).
///
/// # Examples
///
/// ```
/// use hurray_core::layout::is_private_tag;
///
/// assert!(is_private_tag(0xF0));
/// assert!(is_private_tag(0xFE));
/// assert!(!is_private_tag(0xFF));
/// assert!(!is_private_tag(0x01));
/// ```
#[inline]
pub fn is_private_tag(tag: u8) -> bool {
    matches!(tag, PRIVATE_LAYOUT_TAG_MIN..=PRIVATE_LAYOUT_TAG_MAX)
}

/// Validates a layout tag in **strict mode**, returning the appropriate error
/// for tags that are invalid, reserved, or private.
///
/// Named (known) tags return `Ok(())`. Unknown tags in allocated-but-unassigned
/// ranges return [`Error::UnknownLayoutTag`].
///
/// # Errors
///
/// | Tag range | Error |
/// |-----------|-------|
/// | `0x00`, `0xFF` | [`Error::InvalidLayoutTag`] |
/// | `0x0A`–`0x3F`, `0x41`–`0x7F`, `0x80`–`0xEF` | [`Error::ReservedLayoutTag`] |
/// | `0xF0`–`0xFE` | [`Error::PrivateLayoutTag`] |
/// | Any other unrecognised value | [`Error::UnknownLayoutTag`] |
///
/// # Examples
///
/// ```
/// use hurray_core::{Error, layout::validate_layout_tag_strict};
///
/// assert!(validate_layout_tag_strict(0x01).is_ok());
/// assert!(matches!(validate_layout_tag_strict(0x00), Err(Error::InvalidLayoutTag(0x00))));
/// assert!(matches!(validate_layout_tag_strict(0xFF), Err(Error::InvalidLayoutTag(0xFF))));
/// assert!(matches!(validate_layout_tag_strict(0x0A), Err(Error::ReservedLayoutTag(0x0A))));
/// assert!(matches!(validate_layout_tag_strict(0xF0), Err(Error::PrivateLayoutTag(0xF0))));
/// ```
pub fn validate_layout_tag_strict(tag: u8) -> Result<()> {
    match tag {
        0x00 | 0xFF => Err(Error::InvalidLayoutTag(tag)),
        t if is_reserved_tag(t) => Err(Error::ReservedLayoutTag(tag)),
        t if is_private_tag(t) => Err(Error::PrivateLayoutTag(tag)),
        TAG_ROW_MAJOR | TAG_COL_MAJOR | TAG_STRIDED | TAG_TILED | TAG_MORTON | TAG_SUBPAVING
        | TAG_COO | TAG_CSR | TAG_CSC | TAG_BLOCK_PAGED | TAG_HILBERT => Ok(()),
        _ => Err(Error::UnknownLayoutTag(tag)),
    }
}

// ── LayoutDescriptor ─────────────────────────────────────────────────────────

/// The memory layout of a tensor's data buffer.
///
/// Every variant corresponds to a layout tag value defined in the Hurray spec
/// (`docs/spec/memory-layout.md`). The discriminant IS the tag byte; call
/// [`LayoutDescriptor::tag`] to retrieve it.
///
/// The enum is `#[non_exhaustive]` so that future spec versions can add new
/// named layouts without breaking existing `match` arms in downstream crates.
///
/// ## Construction
///
/// - `RowMajor`, `ColMajor`, and `Morton` are unit variants — construct them
///   directly (`LayoutDescriptor::RowMajor`). No separate constructor is needed
///   because they carry no configurable fields.
/// - All other variants wrap a payload struct. Construct the struct via its own
///   `new()` method (which validates invariants), then wrap it.
/// - `Unknown` is the permissive-mode fallback and is not produced by any
///   named-variant path.
///
/// ## Examples
///
/// ```
/// use hurray_core::layout::{LayoutDescriptor, StridedLayout, CooLayout};
///
/// // Unit variant — no constructor needed.
/// let rm = LayoutDescriptor::RowMajor;
/// assert_eq!(rm.tag(), 0x01);
///
/// // Strided layout with explicit strides.
/// let strided = LayoutDescriptor::Strided(StridedLayout::new(vec![4, 1]));
/// assert_eq!(strided.tag(), 0x03);
///
/// // Sparse COO layout.
/// let coo = LayoutDescriptor::Coo(CooLayout::new(42, true));
/// assert_eq!(coo.tag(), 0x07);
/// assert_eq!(coo.buffer_count().map(|n| n.get()), Some(2));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LayoutDescriptor {
    /// Row-major (C-order) layout. Tag `0x01`.
    RowMajor,

    /// Column-major (Fortran-order) layout. Tag `0x02`.
    ColMajor,

    /// Strided layout with explicit per-dimension strides. Tag `0x03`.
    Strided(StridedLayout),

    /// Tiled / blocked layout. Tag `0x04`.
    ///
    /// Boxed because [`TiledLayout`] is recursive (inner tiles) and would
    /// otherwise make the enum size unbounded.
    Tiled(Box<TiledLayout>),

    /// Morton (Z-order curve) layout. Tag `0x05`.
    Morton(MortonLayout),

    /// General subpaving layout (irregular rectangular regions). Tag `0x06`.
    Subpaving(SubpavingLayout),

    /// COO (Coordinate) sparse layout. Tag `0x07`. Buffer count = 2.
    Coo(CooLayout),

    /// CSR (Compressed Sparse Row) layout. Tag `0x08`. Buffer count = 3.
    Csr(CsrLayout),

    /// CSC (Compressed Sparse Column) layout. Tag `0x09`. Buffer count = 3.
    Csc(CscLayout),

    /// Block-paged indirect layout. Tag `0x0B`. Buffer count = 3 (+ optional quant buffers).
    ///
    /// Stores a KV-cache tensor whose paged axis is divided into fixed-size pages
    /// drawn from a shared page pool, with a block table mapping logical page
    /// positions to physical page IDs. Designed for PagedAttention KV caches.
    ///
    /// See `docs/spec/layouts/block-paged.md`.
    BlockPaged(BlockPagedLayout),

    /// Hilbert curve layout. Tag `0x40`.
    Hilbert(HilbertLayout),

    /// Implementation-private extension layout. Tags `0xF0`–`0xFE`.
    PrivateExtension(PrivateExtensionLayout),

    /// Unrecognized layout accepted in permissive mode.
    ///
    /// A strict-mode reader MUST NOT produce this variant. A permissive-mode
    /// reader MAY produce it for tags not covered by any named variant above,
    /// but MUST NOT dereference or interpret the associated tensor data buffer.
    Unknown(UnknownLayout),
}

impl LayoutDescriptor {
    /// Returns the wire tag byte for this layout descriptor.
    ///
    /// The tag is the discriminant used in the binary encoding. It is always a
    /// single `uint8` value.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::layout::{LayoutDescriptor, CooLayout, UnknownLayout};
    ///
    /// assert_eq!(LayoutDescriptor::RowMajor.tag(), 0x01);
    /// assert_eq!(LayoutDescriptor::ColMajor.tag(), 0x02);
    /// assert_eq!(LayoutDescriptor::Coo(CooLayout::new(0, false)).tag(), 0x07);
    ///
    /// // Unknown passthrough.
    /// let u = LayoutDescriptor::Unknown(UnknownLayout::new(0x0A, vec![]).unwrap());
    /// assert_eq!(u.tag(), 0x0A);
    /// ```
    #[inline]
    pub fn tag(&self) -> u8 {
        match self {
            Self::RowMajor => TAG_ROW_MAJOR,
            Self::ColMajor => TAG_COL_MAJOR,
            Self::Strided(_) => TAG_STRIDED,
            Self::Tiled(_) => TAG_TILED,
            Self::Morton(_) => TAG_MORTON,
            Self::Subpaving(_) => TAG_SUBPAVING,
            Self::Coo(_) => TAG_COO,
            Self::Csr(_) => TAG_CSR,
            Self::Csc(_) => TAG_CSC,
            Self::BlockPaged(_) => TAG_BLOCK_PAGED,
            Self::Hilbert(_) => TAG_HILBERT,
            Self::PrivateExtension(p) => p.tag,
            Self::Unknown(u) => u.tag,
        }
    }

    /// Returns the number of data buffers this layout requires, or `None` if
    /// the count is not statically known (only for [`LayoutDescriptor::Unknown`]
    /// and [`LayoutDescriptor::PrivateExtension`]).
    ///
    /// Dense layouts always require 1 buffer. Sparse layouts require a fixed
    /// number of component buffers as defined per format:
    ///
    /// | Layout | Buffer count |
    /// |--------|-------------|
    /// | RowMajor, ColMajor, Strided, Tiled, Morton, Subpaving, Hilbert | 1 |
    /// | COO | 2 |
    /// | CSR, CSC, BlockPaged | 3 |
    /// | PrivateExtension, Unknown | `None` |
    ///
    /// > **Note:** This is the layout's **minimum** buffer count, not the total size
    /// > of the tensor descriptor's buffer table. Quantization-parameter buffers are
    /// > counted separately and appended after the layout's own buffers (so a quantized
    /// > block-paged tensor has `3 + n` buffers, matching the spec's `buffer_count >= 3`).
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroU8;
    /// use hurray_core::layout::{CooLayout, CsrLayout, LayoutDescriptor, UnknownLayout};
    ///
    /// assert_eq!(LayoutDescriptor::RowMajor.buffer_count(), NonZeroU8::new(1));
    /// assert_eq!(
    ///     LayoutDescriptor::Coo(CooLayout::new(0, false)).buffer_count(),
    ///     NonZeroU8::new(2),
    /// );
    /// assert_eq!(
    ///     LayoutDescriptor::Csr(CsrLayout::new(0)).buffer_count(),
    ///     NonZeroU8::new(3),
    /// );
    /// let u = LayoutDescriptor::Unknown(UnknownLayout::new(0x0A, vec![]).unwrap());
    /// assert_eq!(u.buffer_count(), None);
    /// ```
    #[inline]
    pub fn buffer_count(&self) -> Option<NonZeroU8> {
        match self {
            // Dense layouts: exactly one data buffer.
            Self::RowMajor
            | Self::ColMajor
            | Self::Strided(_)
            | Self::Tiled(_)
            | Self::Morton(_)
            | Self::Subpaving(_)
            | Self::Hilbert(_) => NonZeroU8::new(1),

            // COO: values + indices.
            Self::Coo(_) => NonZeroU8::new(2),

            // CSR / CSC / BlockPaged: values + index array + pointer array.
            Self::Csr(_) | Self::Csc(_) | Self::BlockPaged(_) => NonZeroU8::new(3),

            // Private and unknown: buffer requirements are not known statically.
            Self::PrivateExtension(_) | Self::Unknown(_) => None,
        }
    }

    /// Validates layout-specific constraints against the provided tensor shape.
    ///
    /// This method is called by Layer 4 (tensor descriptor) to check that the
    /// layout descriptor is consistent with the tensor's rank and dimension
    /// sizes. Constructors do not call this method because shape is not
    /// available at layout-descriptor construction time.
    ///
    /// ## Checked constraints
    ///
    /// | Layout | Constraint |
    /// |--------|-----------|
    /// | `Strided` | `strides.len() == shape.rank()` |
    /// | `Tiled` | `tile_shape.len() == shape.rank()`; each `tile_shape[k] > 0` |
    /// | `Morton` | `morton_bits.len() == shape.rank()`; `shape[k] <= 2^morton_bits[k]` for all static dims |
    /// | `Hilbert` | `hilbert_rank == shape.rank()`; `hilbert_rank >= 2`; each static dim == `2^hilbert_order` |
    /// | `Subpaving` | each region: `origin.len() == shape.rank()`, `region_shape.len() == shape.rank()`; all regions within bounds; no overlaps |
    /// | `Coo` | `shape.rank() >= 1` (no scalar sparse tensors) |
    /// | `Csr` | `shape.rank() == 2` |
    /// | `Csc` | `shape.rank() == 2` |
    /// | `Unknown` | always `Ok(())` |
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidLayout`] if any constraint is violated.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{Shape, layout::{LayoutDescriptor, StridedLayout}};
    ///
    /// let shape = Shape::new(vec![3, 4]).unwrap();
    ///
    /// // Correct rank.
    /// let ok = LayoutDescriptor::Strided(StridedLayout::new(vec![4, 1]));
    /// assert!(ok.validate_against_shape(&shape).is_ok());
    ///
    /// // Wrong rank.
    /// let bad = LayoutDescriptor::Strided(StridedLayout::new(vec![1]));
    /// assert!(bad.validate_against_shape(&shape).is_err());
    /// ```
    pub fn validate_against_shape(&self, shape: &Shape) -> Result<()> {
        match self {
            Self::RowMajor | Self::ColMajor => {
                // No layout-specific constraints beyond what the tensor descriptor checks.
                Ok(())
            }

            Self::Strided(s) => {
                if s.strides.len() != shape.rank() {
                    return Err(Error::InvalidLayout(format!(
                        "strided layout: strides.len() ({}) != shape.rank() ({})",
                        s.strides.len(),
                        shape.rank()
                    )));
                }
                // Spec: MUST NOT be used for rank-0 (scalar) tensors.
                if shape.rank() == 0 {
                    return Err(Error::InvalidLayout(
                        "strided layout cannot be used for rank-0 (scalar) tensors".to_string(),
                    ));
                }
                Ok(())
            }

            Self::Tiled(t) => {
                if t.tile_shape.len() != shape.rank() {
                    return Err(Error::InvalidLayout(format!(
                        "tiled layout: tile_shape.len() ({}) != shape.rank() ({})",
                        t.tile_shape.len(),
                        shape.rank()
                    )));
                }
                // Spec: MUST NOT be used for rank-0 tensors.
                if shape.rank() == 0 {
                    return Err(Error::InvalidLayout(
                        "tiled layout cannot be used for rank-0 (scalar) tensors".to_string(),
                    ));
                }
                // tile_shape values > 0 already enforced by TiledLayout::new.
                Ok(())
            }

            Self::Morton(m) => {
                if m.morton_bits.len() != shape.rank() {
                    return Err(Error::InvalidLayout(format!(
                        "Morton layout: morton_bits.len() ({}) != shape.rank() ({})",
                        m.morton_bits.len(),
                        shape.rank()
                    )));
                }
                // Validate shape[k] <= 2^morton_bits[k] for all static (non-DYNAMIC) dims.
                for (k, (&bits, &dim)) in m.morton_bits.iter().zip(shape.dims().iter()).enumerate()
                {
                    if dim == crate::shape::DYNAMIC {
                        continue;
                    }
                    // 2^bits saturates at u64::MAX if bits >= 64.
                    let max_dim = if bits >= 64 { u64::MAX } else { 1u64 << bits };
                    if dim > max_dim {
                        return Err(Error::InvalidLayout(format!(
                            "Morton layout: shape[{k}] ({dim}) > 2^morton_bits[{k}] ({max_dim})"
                        )));
                    }
                }
                Ok(())
            }

            Self::Subpaving(sp) => {
                let rank = shape.rank();
                for (i, region) in sp.regions.iter().enumerate() {
                    if region.origin.len() != rank {
                        return Err(Error::InvalidLayout(format!(
                            "subpaving region {i}: origin.len() ({}) != shape.rank() ({rank})",
                            region.origin.len()
                        )));
                    }
                    if region.region_shape.len() != rank {
                        return Err(Error::InvalidLayout(format!(
                            "subpaving region {i}: region_shape.len() ({}) != shape.rank() ({rank})",
                            region.region_shape.len()
                        )));
                    }
                    // Validate each region is within shape bounds (for static dims).
                    for k in 0..rank {
                        let dim = shape.dims()[k];
                        if dim == crate::shape::DYNAMIC {
                            continue;
                        }
                        let end = region.origin[k].saturating_add(region.region_shape[k]);
                        if end > dim {
                            return Err(Error::InvalidLayout(format!(
                                "subpaving region {i}: dimension {k}: \
                                 origin[{k}] ({}) + region_shape[{k}] ({}) = {end} > shape[{k}] ({dim})",
                                region.origin[k], region.region_shape[k]
                            )));
                        }
                    }
                }
                // Non-overlap check (SHOULD validate per spec — we enforce it in strict mode).
                // O(n^2) over regions; acceptable for descriptor validation (not a hot path).
                // Full coverage (union == tensor index space) is not validated here: the
                // general case is O(∏ shape) and deferred to Layer 4 / application layer.
                // The volume-sum heuristic (∑ region volumes == total elements) is a
                // necessary but not sufficient check and is also deferred. TODO: add it.
                let n = sp.regions.len();
                for i in 0..n {
                    for j in (i + 1)..n {
                        if regions_overlap(&sp.regions[i], &sp.regions[j], shape) {
                            return Err(Error::InvalidLayout(format!(
                                "subpaving regions {i} and {j} overlap"
                            )));
                        }
                    }
                }
                Ok(())
            }

            Self::Coo(_) => {
                if shape.rank() == 0 {
                    return Err(Error::InvalidLayout(
                        "COO layout cannot be used for rank-0 (scalar) tensors".to_string(),
                    ));
                }
                Ok(())
            }

            Self::Csr(_) => {
                if shape.rank() != 2 {
                    return Err(Error::InvalidLayout(format!(
                        "CSR layout requires rank 2, got rank {}",
                        shape.rank()
                    )));
                }
                Ok(())
            }

            Self::Csc(_) => {
                if shape.rank() != 2 {
                    return Err(Error::InvalidLayout(format!(
                        "CSC layout requires rank 2, got rank {}",
                        shape.rank()
                    )));
                }
                Ok(())
            }

            Self::BlockPaged(bp) => {
                // Spec: block-paged is defined only for rank-3 tensors.
                if shape.rank() != 3 {
                    return Err(Error::InvalidLayout(format!(
                        "block-paged layout requires rank 3, got rank {}",
                        shape.rank()
                    )));
                }
                // Spec: paged_axis MUST be 0 in this version.
                if bp.paged_axis != 0 {
                    return Err(Error::InvalidLayout(format!(
                        "block-paged layout: paged_axis must be 0, got {}",
                        bp.paged_axis
                    )));
                }
                // Spec: page_size MUST be >= 1.
                if bp.page_size < 1 {
                    return Err(Error::InvalidLayout(
                        "block-paged layout: page_size must be >= 1".to_string(),
                    ));
                }
                // kv_role and block_table_index_type are strongly typed enums:
                // all wire-representable values are valid here. No additional
                // check needed — invalid wire bytes are rejected at decode time.
                Ok(())
            }

            Self::Hilbert(h) => {
                if h.hilbert_rank as usize != shape.rank() {
                    return Err(Error::InvalidLayout(format!(
                        "Hilbert layout: hilbert_rank ({}) != shape.rank() ({})",
                        h.hilbert_rank,
                        shape.rank()
                    )));
                }
                // hilbert_rank >= 2 already enforced by HilbertLayout::new.
                // Each static dim must equal 2^hilbert_order.
                let expected_dim = 1u64.checked_shl(h.hilbert_order).ok_or_else(|| {
                    Error::InvalidLayout(format!(
                        "Hilbert layout: hilbert_order ({}) would require dimension > u64::MAX",
                        h.hilbert_order
                    ))
                })?;
                for (k, &dim) in shape.dims().iter().enumerate() {
                    if dim == crate::shape::DYNAMIC {
                        continue;
                    }
                    if dim != expected_dim {
                        return Err(Error::InvalidLayout(format!(
                            "Hilbert layout: shape[{k}] ({dim}) != 2^hilbert_order ({expected_dim})"
                        )));
                    }
                }
                Ok(())
            }

            // Private-extension: shape constraints are implementation-defined.
            Self::PrivateExtension(_) => Ok(()),

            // Unknown: permissive mode — accept without validation.
            Self::Unknown(_) => Ok(()),
        }
    }

    /// Returns the linear element offset for a multi-dimensional logical index.
    ///
    /// Dispatches to the layout-specific [`addressing::ElementAddress`] implementation.
    /// For multi-buffer layouts (sparse formats, general subpaving), returns
    /// [`Error::LayoutRequiresMultiBuffer`] — use the layout-specific method instead
    /// (e.g., [`SubpavingLocation`] via [`SubpavingLayout::locate_element`]).
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::layout::LayoutDescriptor;
    /// use hurray_core::Shape;
    ///
    /// let shape = Shape::new(vec![3, 4]).unwrap();
    ///
    /// // Row-major: element [1, 2] → offset 1*4 + 2 = 6.
    /// assert_eq!(LayoutDescriptor::RowMajor.element_offset(&[1, 2], &shape).unwrap(), 6);
    ///
    /// // Column-major: element [1, 2] → offset 1*1 + 2*3 = 7.
    /// assert_eq!(LayoutDescriptor::ColMajor.element_offset(&[1, 2], &shape).unwrap(), 7);
    /// ```
    pub fn element_offset(&self, index: &[u64], shape: &Shape) -> crate::Result<u64> {
        use addressing::ElementAddress;
        match self {
            Self::RowMajor => addressing::row_major::element_offset(index, shape),
            Self::ColMajor => addressing::col_major::element_offset(index, shape),
            Self::Strided(s) => s.element_offset(index, shape),
            Self::Tiled(t) => t.element_offset(index, shape),
            Self::Morton(m) => m.element_offset(index, shape),
            Self::Hilbert(h) => h.element_offset(index, shape),
            // Subpaving is multi-buffer; use SubpavingLayout::locate_element.
            Self::Subpaving(_) => Err(crate::Error::LayoutRequiresMultiBuffer {
                layout_tag: TAG_SUBPAVING,
            }),
            Self::Coo(_) => Err(crate::Error::LayoutRequiresMultiBuffer {
                layout_tag: TAG_COO,
            }),
            Self::Csr(_) => Err(crate::Error::LayoutRequiresMultiBuffer {
                layout_tag: TAG_CSR,
            }),
            Self::Csc(_) => Err(crate::Error::LayoutRequiresMultiBuffer {
                layout_tag: TAG_CSC,
            }),
            // BlockPaged is indirect (multi-buffer); use addressing::block_paged directly.
            Self::BlockPaged(_) => Err(crate::Error::LayoutRequiresMultiBuffer {
                layout_tag: TAG_BLOCK_PAGED,
            }),
            Self::PrivateExtension(p) => Err(crate::Error::PrivateLayoutTag(p.tag)),
            Self::Unknown(u) => Err(crate::Error::UnknownLayoutTag(u.tag)),
        }
    }
}

// ── Internal helpers ─────────────────────────────────────────────────────────

/// Returns `true` if two subpaving regions overlap along all dimensions.
///
/// Overlap along dimension `k` iff:
/// `A.origin[k] < B.origin[k] + B.region_shape[k]`
/// AND `B.origin[k] < A.origin[k] + A.region_shape[k]`
///
/// DYNAMIC dimensions are treated conservatively: if either region's bound
/// is dynamic we skip the check for that dimension (cannot prove no overlap).
fn regions_overlap(a: &RegionDescriptor, b: &RegionDescriptor, shape: &Shape) -> bool {
    for k in 0..a.origin.len() {
        let dim = if k < shape.dims().len() {
            shape.dims()[k]
        } else {
            // If shape has fewer dims than regions (shouldn't happen post earlier check),
            // treat as dynamic.
            crate::shape::DYNAMIC
        };
        if dim == crate::shape::DYNAMIC {
            // Cannot determine overlap for dynamic dims; skip conservatively.
            continue;
        }
        let a_end = a.origin[k].saturating_add(a.region_shape[k]);
        let b_end = b.origin[k].saturating_add(b.region_shape[k]);
        // No overlap on this dimension — regions are disjoint.
        if a.origin[k] >= b_end || b.origin[k] >= a_end {
            return false;
        }
    }
    true
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── tag() ─────────────────────────────────────────────────────────────────

    #[test]
    fn tag_values_match_spec() {
        assert_eq!(LayoutDescriptor::RowMajor.tag(), 0x01);
        assert_eq!(LayoutDescriptor::ColMajor.tag(), 0x02);
        assert_eq!(
            LayoutDescriptor::Strided(StridedLayout::new(vec![1])).tag(),
            0x03
        );
        assert_eq!(
            LayoutDescriptor::Tiled(Box::new(
                TiledLayout::new(vec![4], 0x01, 0x01, None, None, None).unwrap()
            ))
            .tag(),
            0x04
        );
        assert_eq!(
            LayoutDescriptor::Morton(MortonLayout::new(vec![2]).unwrap()).tag(),
            0x05
        );
        assert_eq!(
            LayoutDescriptor::Subpaving(
                SubpavingLayout::new(vec![
                    RegionDescriptor::new(vec![0], vec![4], 0x01, 0, 0).unwrap()
                ])
                .unwrap()
            )
            .tag(),
            0x06
        );
        assert_eq!(
            LayoutDescriptor::Coo(CooLayout {
                nnz: 0,
                is_sorted: false
            })
            .tag(),
            0x07
        );
        assert_eq!(LayoutDescriptor::Csr(CsrLayout { nnz: 0 }).tag(), 0x08);
        assert_eq!(LayoutDescriptor::Csc(CscLayout { nnz: 0 }).tag(), 0x09);
        assert_eq!(
            LayoutDescriptor::Hilbert(HilbertLayout::new(2, 2).unwrap()).tag(),
            0x40
        );
    }

    // ── buffer_count() ────────────────────────────────────────────────────────

    #[test]
    fn dense_layouts_have_buffer_count_1() {
        let dense = [
            LayoutDescriptor::RowMajor,
            LayoutDescriptor::ColMajor,
            LayoutDescriptor::Strided(StridedLayout::new(vec![4, 1])),
            LayoutDescriptor::Tiled(Box::new(
                TiledLayout::new(vec![4, 4], 0x01, 0x01, None, None, None).unwrap(),
            )),
            LayoutDescriptor::Morton(MortonLayout::new(vec![2, 2]).unwrap()),
            LayoutDescriptor::Subpaving(
                SubpavingLayout::new(vec![RegionDescriptor::new(
                    vec![0, 0],
                    vec![4, 4],
                    0x01,
                    0,
                    0,
                )
                .unwrap()])
                .unwrap(),
            ),
            LayoutDescriptor::Hilbert(HilbertLayout::new(2, 2).unwrap()),
        ];
        for layout in &dense {
            assert_eq!(
                layout.buffer_count(),
                NonZeroU8::new(1),
                "expected buffer_count=1 for {:?}",
                layout.tag()
            );
        }
    }

    #[test]
    fn coo_has_buffer_count_2() {
        assert_eq!(
            LayoutDescriptor::Coo(CooLayout {
                nnz: 0,
                is_sorted: false
            })
            .buffer_count(),
            NonZeroU8::new(2)
        );
    }

    #[test]
    fn csr_has_buffer_count_3() {
        assert_eq!(
            LayoutDescriptor::Csr(CsrLayout { nnz: 0 }).buffer_count(),
            NonZeroU8::new(3)
        );
    }

    #[test]
    fn csc_has_buffer_count_3() {
        assert_eq!(
            LayoutDescriptor::Csc(CscLayout { nnz: 0 }).buffer_count(),
            NonZeroU8::new(3)
        );
    }

    #[test]
    fn private_extension_buffer_count_is_none() {
        let layout = LayoutDescriptor::PrivateExtension(
            PrivateExtensionLayout::new(0xF0, 0, vec![]).unwrap(),
        );
        assert!(layout.buffer_count().is_none());
    }

    #[test]
    fn unknown_buffer_count_is_none() {
        let layout = LayoutDescriptor::Unknown(UnknownLayout {
            tag: 0x0A,
            raw_bytes: vec![],
        });
        assert!(layout.buffer_count().is_none());
    }

    // ── validate_layout_tag_strict ────────────────────────────────────────────

    #[test]
    fn known_tags_pass_strict_validation() {
        for tag in [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0B, 0x40,
        ] {
            assert!(
                validate_layout_tag_strict(tag).is_ok(),
                "tag 0x{tag:02X} should be valid"
            );
        }
    }

    #[test]
    fn invalid_sentinels_rejected() {
        assert!(matches!(
            validate_layout_tag_strict(0x00),
            Err(Error::InvalidLayoutTag(0x00))
        ));
        assert!(matches!(
            validate_layout_tag_strict(0xFF),
            Err(Error::InvalidLayoutTag(0xFF))
        ));
    }

    #[test]
    fn reserved_tags_rejected() {
        for tag in [0x0A_u8, 0x3F, 0x41, 0x7F, 0x80, 0xEF] {
            assert!(
                matches!(
                    validate_layout_tag_strict(tag),
                    Err(Error::ReservedLayoutTag(_))
                ),
                "tag 0x{tag:02X} should be ReservedLayoutTag"
            );
        }
    }

    #[test]
    fn private_tags_rejected_in_strict_mode() {
        for tag in [0xF0_u8, 0xF5, 0xFE] {
            assert!(
                matches!(
                    validate_layout_tag_strict(tag),
                    Err(Error::PrivateLayoutTag(_))
                ),
                "tag 0x{tag:02X} should be PrivateLayoutTag"
            );
        }
    }

    // ── validate_against_shape ────────────────────────────────────────────────

    #[test]
    fn row_major_valid_for_any_rank() {
        let shape = Shape::new(vec![3, 4]).unwrap();
        assert!(LayoutDescriptor::RowMajor
            .validate_against_shape(&shape)
            .is_ok());
        assert!(LayoutDescriptor::ColMajor
            .validate_against_shape(&Shape::scalar())
            .is_ok());
    }

    #[test]
    fn strided_valid_when_strides_len_matches_rank() {
        let shape = Shape::new(vec![3, 4]).unwrap();
        let layout = LayoutDescriptor::Strided(StridedLayout::new(vec![4, 1]));
        assert!(layout.validate_against_shape(&shape).is_ok());
    }

    #[test]
    fn strided_invalid_when_strides_len_mismatches_rank() {
        let shape = Shape::new(vec![3, 4]).unwrap();
        let layout = LayoutDescriptor::Strided(StridedLayout::new(vec![1]));
        assert!(matches!(
            layout.validate_against_shape(&shape),
            Err(Error::InvalidLayout(_))
        ));
    }

    #[test]
    fn strided_invalid_for_scalar() {
        let layout = LayoutDescriptor::Strided(StridedLayout::new(vec![]));
        assert!(matches!(
            layout.validate_against_shape(&Shape::scalar()),
            Err(Error::InvalidLayout(_))
        ));
    }

    #[test]
    fn csr_valid_for_rank_2() {
        let shape = Shape::new(vec![4, 5]).unwrap();
        let layout = LayoutDescriptor::Csr(CsrLayout { nnz: 0 });
        assert!(layout.validate_against_shape(&shape).is_ok());
    }

    #[test]
    fn csr_invalid_for_rank_3() {
        let shape = Shape::new(vec![2, 3, 4]).unwrap();
        let layout = LayoutDescriptor::Csr(CsrLayout { nnz: 0 });
        assert!(matches!(
            layout.validate_against_shape(&shape),
            Err(Error::InvalidLayout(_))
        ));
    }

    #[test]
    fn coo_invalid_for_scalar() {
        let layout = LayoutDescriptor::Coo(CooLayout {
            nnz: 0,
            is_sorted: false,
        });
        assert!(matches!(
            layout.validate_against_shape(&Shape::scalar()),
            Err(Error::InvalidLayout(_))
        ));
    }

    #[test]
    fn coo_valid_for_rank_1() {
        let shape = Shape::new(vec![10]).unwrap();
        let layout = LayoutDescriptor::Coo(CooLayout {
            nnz: 3,
            is_sorted: true,
        });
        assert!(layout.validate_against_shape(&shape).is_ok());
    }

    #[test]
    fn hilbert_valid_for_4x4_order2() {
        let shape = Shape::new(vec![4, 4]).unwrap();
        let layout = LayoutDescriptor::Hilbert(HilbertLayout::new(2, 2).unwrap());
        assert!(layout.validate_against_shape(&shape).is_ok());
    }

    #[test]
    fn hilbert_invalid_when_dim_not_power_of_two_order() {
        let shape = Shape::new(vec![3, 4]).unwrap(); // 3 != 2^2
        let layout = LayoutDescriptor::Hilbert(HilbertLayout::new(2, 2).unwrap());
        assert!(matches!(
            layout.validate_against_shape(&shape),
            Err(Error::InvalidLayout(_))
        ));
    }

    #[test]
    fn morton_invalid_when_shape_exceeds_bits() {
        // morton_bits=[1,1] → max dim = 2^1 = 2; shape[0]=3 > 2.
        let shape = Shape::new(vec![3, 2]).unwrap();
        let layout = LayoutDescriptor::Morton(MortonLayout::new(vec![1, 1]).unwrap());
        assert!(matches!(
            layout.validate_against_shape(&shape),
            Err(Error::InvalidLayout(_))
        ));
    }

    #[test]
    fn subpaving_rejects_overlapping_regions() {
        // Two identical overlapping regions.
        let r1 = RegionDescriptor::new(vec![0, 0], vec![4, 4], 0x01, 0, 0).unwrap();
        let r2 = RegionDescriptor::new(vec![2, 2], vec![4, 4], 0x01, 0, 64).unwrap();
        let sp = SubpavingLayout::new(vec![r1, r2]).unwrap();
        let shape = Shape::new(vec![8, 8]).unwrap();
        let layout = LayoutDescriptor::Subpaving(sp);
        assert!(matches!(
            layout.validate_against_shape(&shape),
            Err(Error::InvalidLayout(_))
        ));
    }

    #[test]
    fn subpaving_rejects_region_out_of_shape_bounds() {
        let r = RegionDescriptor::new(vec![5, 0], vec![4, 4], 0x01, 0, 0).unwrap(); // 5+4=9 > 8
        let sp = SubpavingLayout::new(vec![r]).unwrap();
        let shape = Shape::new(vec![8, 8]).unwrap();
        let layout = LayoutDescriptor::Subpaving(sp);
        assert!(matches!(
            layout.validate_against_shape(&shape),
            Err(Error::InvalidLayout(_))
        ));
    }

    #[test]
    fn unknown_always_passes_validation() {
        let layout = LayoutDescriptor::Unknown(UnknownLayout {
            tag: 0x0A,
            raw_bytes: vec![],
        });
        assert!(layout.validate_against_shape(&Shape::scalar()).is_ok());
        assert!(layout
            .validate_against_shape(&Shape::new(vec![100, 200]).unwrap())
            .is_ok());
    }
}
