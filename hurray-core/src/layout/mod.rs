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
//! | `0x01`–`0x0B` | Core named layouts (Tier 1), including the Composite / Virtual head (`0x0B`, ADR-027) |
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
pub mod composite;
pub mod coo;
pub mod csc;
pub mod csf;
pub mod csr;
pub mod hilbert;
pub mod morton;
pub mod private_extension;
pub mod row_major;
pub mod strided;
pub mod tiled;
pub mod unknown;

pub use addressing::{byte_address_from_element_offset, ElementAddress};
pub use block_paged::{BlockPagedLayout, BlockTableIndexType, KvRole};
pub use composite::{CombineOp, CompositeLayout, CompositionRule};
pub use coo::CooLayout;
pub use csc::CscLayout;
pub use csf::CsfLayout;
pub use csr::CsrLayout;
pub use hilbert::HilbertLayout;
pub use morton::MortonLayout;
pub use private_extension::{
    PrivateExtensionLayout, PRIVATE_LAYOUT_TAG_MAX, PRIVATE_LAYOUT_TAG_MIN,
};
pub use strided::StridedLayout;
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
/// Layout tag for COO (Coordinate) sparse layout.
pub const TAG_COO: u8 = 0x06;
/// Layout tag for CSR (Compressed Sparse Row) layout.
pub const TAG_CSR: u8 = 0x07;
/// Layout tag for CSC (Compressed Sparse Column) layout.
pub const TAG_CSC: u8 = 0x08;
/// Layout tag for CSF (Compressed Sparse Fiber) layout.
pub const TAG_CSF: u8 = 0x09;
/// Layout tag for block-paged indirect layout.
pub const TAG_BLOCK_PAGED: u8 = 0x0A;
/// Layout tag for the composite / virtual head (ADR-027). See `docs/spec/layouts/composite.md`.
pub const TAG_COMPOSITE: u8 = 0x0B;
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
/// versions (`0x0C`–`0x3F`, `0x41`–`0x7F`, `0x80`–`0xEF`).
///
/// Tag `0x0B` (composite / virtual head, ADR-027) is now a named Tier-1 tag and is
/// therefore no longer reserved — it moved out of this range when Composite was
/// implemented.
///
/// # Examples
///
/// ```
/// use hurray_core::layout::is_reserved_tag;
///
/// assert!(!is_reserved_tag(0x0A)); // block-paged — named tag
/// assert!(!is_reserved_tag(0x0B)); // composite / virtual head — named tag
/// assert!(is_reserved_tag(0x0C));
/// assert!(is_reserved_tag(0x3F));
/// assert!(is_reserved_tag(0x41));
/// assert!(is_reserved_tag(0xEF));
/// assert!(!is_reserved_tag(0x01));
/// assert!(!is_reserved_tag(0x40));
/// ```
#[inline]
pub fn is_reserved_tag(tag: u8) -> bool {
    matches!(tag, 0x0C..=0x3F | 0x41..=0x7F | 0x80..=0xEF)
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

/// Returns `true` if `tag` has a named [`LayoutDescriptor`] variant in this crate.
///
/// The complement of "unknown": a named tag carries structure this implementation
/// knows how to check, so it MUST NOT be wrapped in
/// [`UnknownLayout`], which has neither a buffer count nor shape
/// constraints.
///
/// # Examples
///
/// ```
/// use hurray_core::layout::is_named_tag;
///
/// assert!(is_named_tag(0x01));  // row-major
/// assert!(is_named_tag(0x0B));  // composite / virtual head
/// assert!(is_named_tag(0x40));  // Hilbert
/// assert!(!is_named_tag(0x0C)); // reserved for a future version
/// assert!(!is_named_tag(0xF0)); // private extension
/// ```
#[inline]
pub fn is_named_tag(tag: u8) -> bool {
    matches!(
        tag,
        TAG_ROW_MAJOR
            | TAG_COL_MAJOR
            | TAG_STRIDED
            | TAG_TILED
            | TAG_MORTON
            | TAG_COO
            | TAG_CSR
            | TAG_CSC
            | TAG_CSF
            | TAG_BLOCK_PAGED
            | TAG_COMPOSITE
            | TAG_HILBERT
    )
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
/// | `0x0C`–`0x3F`, `0x41`–`0x7F`, `0x80`–`0xEF` | [`Error::ReservedLayoutTag`] |
/// | `0xF0`–`0xFE` | [`Error::PrivateLayoutTag`] |
/// | Any other unrecognised value | [`Error::UnknownLayoutTag`] |
///
/// # Examples
///
/// ```
/// use hurray_core::{Error, layout::validate_layout_tag_strict};
///
/// assert!(validate_layout_tag_strict(0x01).is_ok());
/// assert!(validate_layout_tag_strict(0x09).is_ok()); // CSF
/// assert!(validate_layout_tag_strict(0x0B).is_ok()); // composite / virtual head
/// assert!(matches!(validate_layout_tag_strict(0x00), Err(Error::InvalidLayoutTag(0x00))));
/// assert!(matches!(validate_layout_tag_strict(0xFF), Err(Error::InvalidLayoutTag(0xFF))));
/// assert!(matches!(validate_layout_tag_strict(0x0C), Err(Error::ReservedLayoutTag(0x0C))));
/// assert!(matches!(validate_layout_tag_strict(0xF0), Err(Error::PrivateLayoutTag(0xF0))));
/// ```
pub fn validate_layout_tag_strict(tag: u8) -> Result<()> {
    match tag {
        0x00 | 0xFF => Err(Error::InvalidLayoutTag(tag)),
        t if is_reserved_tag(t) => Err(Error::ReservedLayoutTag(tag)),
        t if is_private_tag(t) => Err(Error::PrivateLayoutTag(tag)),
        // One list of named tags, in is_named_tag, so strict validation and the
        // unknown-layout constructor cannot come to disagree about what is named.
        t if is_named_tag(t) => Ok(()),
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
/// assert_eq!(coo.tag(), 0x06);
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

    /// COO (Coordinate) sparse layout. Tag `0x06`. Buffer count = 2.
    Coo(CooLayout),

    /// CSR (Compressed Sparse Row) layout. Tag `0x07`. Buffer count = 3.
    Csr(CsrLayout),

    /// CSC (Compressed Sparse Column) layout. Tag `0x08`. Buffer count = 3.
    Csc(CscLayout),

    /// CSF (Compressed Sparse Fiber) layout. Tag `0x09`. Buffer count = `2·rank+1`.
    ///
    /// The rank-N generalisation of CSR/CSC, storing a sparse tensor as a tree of
    /// `rank` levels. Only valid for rank ≥ 3. Buffer count is rank-dependent because
    /// each level contributes one `pos` and one `crd` buffer plus a single `values` buffer.
    ///
    /// See `docs/spec/layouts/csf.md`.
    Csf(CsfLayout),

    /// Block-paged indirect layout. Tag `0x0A`. Buffer count = 3 (+ optional quant buffers).
    ///
    /// Stores a KV-cache tensor whose paged axis is divided into fixed-size pages
    /// drawn from a shared page pool, with a block table mapping logical page
    /// positions to physical page IDs. Designed for PagedAttention KV caches.
    ///
    /// See `docs/spec/layouts/block-paged.md`.
    BlockPaged(BlockPagedLayout),

    /// Composite / Virtual head. Tag `0x0B`. Owns no data (`buffer_count = 0`).
    ///
    /// The head presents a single logical `shape`/`type_tag` view over an ordered
    /// set of **member** tensor descriptors bound by forward stream adjacency (see
    /// [`crate::composite::CompositeTensor`]). This is a new addressing category,
    /// **Virtual**, alongside Dense, Sparse, and Indirect.
    ///
    /// See `docs/spec/layouts/composite.md`.
    Composite(CompositeLayout),

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
    /// use hurray_core::layout::{CooLayout, CsfLayout, LayoutDescriptor, UnknownLayout};
    ///
    /// assert_eq!(LayoutDescriptor::RowMajor.tag(), 0x01);
    /// assert_eq!(LayoutDescriptor::ColMajor.tag(), 0x02);
    /// assert_eq!(LayoutDescriptor::Coo(CooLayout::new(0, false)).tag(), 0x06);
    /// assert_eq!(LayoutDescriptor::Csf(CsfLayout::new(0, vec![0, 1, 2])).tag(), 0x09);
    ///
    /// // Unknown passthrough (with a reserved tag outside the named set).
    /// let u = LayoutDescriptor::Unknown(UnknownLayout::new(0x0C, vec![]).unwrap());
    /// assert_eq!(u.tag(), 0x0C);
    /// ```
    #[inline]
    pub fn tag(&self) -> u8 {
        match self {
            Self::RowMajor => TAG_ROW_MAJOR,
            Self::ColMajor => TAG_COL_MAJOR,
            Self::Strided(_) => TAG_STRIDED,
            Self::Tiled(_) => TAG_TILED,
            Self::Morton(_) => TAG_MORTON,
            Self::Coo(_) => TAG_COO,
            Self::Csr(_) => TAG_CSR,
            Self::Csc(_) => TAG_CSC,
            Self::Csf(_) => TAG_CSF,
            Self::BlockPaged(_) => TAG_BLOCK_PAGED,
            Self::Composite(_) => TAG_COMPOSITE,
            Self::Hilbert(_) => TAG_HILBERT,
            Self::PrivateExtension(p) => p.tag,
            Self::Unknown(u) => u.tag,
        }
    }

    /// Returns the number of data buffers this layout requires, or `None` if the
    /// count is either not statically known ([`LayoutDescriptor::Unknown`],
    /// [`LayoutDescriptor::PrivateExtension`]) or is a virtual "known zero"
    /// ([`LayoutDescriptor::Composite`] — use [`LayoutDescriptor::is_virtual`] to
    /// tell the two apart).
    ///
    /// Dense layouts always require 1 buffer. Sparse layouts require a fixed
    /// number of component buffers as defined per format:
    ///
    /// | Layout | Buffer count |
    /// |--------|-------------|
    /// | RowMajor, ColMajor, Strided, Tiled, Morton, Hilbert | 1 |
    /// | COO | 2 |
    /// | CSR, CSC, BlockPaged | 3 |
    /// | CSF | `2·rank+1` (rank-dependent; embedded in `mode_order.len()`) |
    /// | Composite | `0`, not `NonZeroU8`-representable — reported as `None` |
    /// | PrivateExtension, Unknown | `None` (genuinely unknown) |
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
    /// use hurray_core::layout::{CooLayout, CsrLayout, CsfLayout, LayoutDescriptor, UnknownLayout};
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
    /// // CSF rank-3: 2*3+1 = 7 buffers.
    /// assert_eq!(
    ///     LayoutDescriptor::Csf(CsfLayout::new(0, vec![0, 1, 2])).buffer_count(),
    ///     NonZeroU8::new(7),
    /// );
    /// let u = LayoutDescriptor::Unknown(UnknownLayout::new(0x0C, vec![]).unwrap());
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
            | Self::Hilbert(_) => NonZeroU8::new(1),

            // COO: values + indices.
            Self::Coo(_) => NonZeroU8::new(2),

            // CSR / CSC / BlockPaged: values + index array + pointer array.
            Self::Csr(_) | Self::Csc(_) | Self::BlockPaged(_) => NonZeroU8::new(3),

            // CSF: 2·rank+1 buffers (values + rank pos arrays + rank crd arrays).
            // rank ≤ 64 (data-model.md), so 2·64+1 = 129 fits in u8 (max 255).
            // Saturate at u8::MAX if mode_order is somehow empty (would be caught
            // by validate_against_shape before any buffer-table use).
            Self::Csf(c) => {
                // 2·rank+1 with rank = mode_order.len(); rank ≥ 3 after validation.
                // Saturate rather than truncate so a malformed (pre-validation) rank
                // can never silently wrap to a small count.
                let count = (2 * c.mode_order.len() + 1).min(u8::MAX as usize) as u8;
                NonZeroU8::new(count)
            }

            // Composite: a virtual head owns exactly zero buffers, but "known zero" isn't
            // representable via NonZeroU8 — callers MUST use `is_virtual()` to distinguish
            // this case from PrivateExtension/Unknown's genuinely-unknown count.
            Self::Composite(_) => None,

            // Private and unknown: buffer requirements are not known statically.
            Self::PrivateExtension(_) | Self::Unknown(_) => None,
        }
    }

    /// Returns `true` if this layout is **virtual** (owns no data buffer).
    ///
    /// Only [`LayoutDescriptor::Composite`] is virtual. This lets callers
    /// distinguish "buffer count is `None` because the layout is virtual (a
    /// known zero, per spec)" from "buffer count is `None` because it is not
    /// statically known" ([`LayoutDescriptor::Unknown`],
    /// [`LayoutDescriptor::PrivateExtension`]) — both report `None` from
    /// [`LayoutDescriptor::buffer_count`], but only the former is genuinely zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::layout::{CompositeLayout, CompositionRule, LayoutDescriptor};
    ///
    /// let composite = LayoutDescriptor::Composite(
    ///     CompositeLayout::new(CompositionRule::Group, 0).unwrap(),
    /// );
    /// assert!(composite.is_virtual());
    /// assert!(!LayoutDescriptor::RowMajor.is_virtual());
    /// ```
    #[inline]
    pub fn is_virtual(&self) -> bool {
        matches!(self, Self::Composite(_))
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
    /// | `Coo` | `shape.rank() >= 1` (no scalar sparse tensors) |
    /// | `Csr` | `shape.rank() == 2` |
    /// | `Csc` | `shape.rank() == 2` |
    /// | `Csf` | `shape.rank() >= 3`; `mode_order.len() == shape.rank()`; `mode_order` is a permutation of `0..rank` |
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

            Self::Csf(c) => {
                let rank = shape.rank();
                // Spec csf.md §Validity Constraints: rank MUST be >= 3.
                if rank < 3 {
                    return Err(Error::InvalidLayout(format!(
                        "CSF layout requires rank >= 3, got rank {rank}"
                    )));
                }
                // mode_order length must equal rank.
                if c.mode_order.len() != rank {
                    return Err(Error::InvalidLayout(format!(
                        "CSF layout: mode_order.len() ({}) != shape.rank() ({rank})",
                        c.mode_order.len()
                    )));
                }
                // mode_order must be a permutation of 0..rank.
                // Use a presence-bitmap approach: O(rank), no allocation for rank ≤ 64.
                let mut seen = 0u64;
                for (level, &dim) in c.mode_order.iter().enumerate() {
                    if dim as usize >= rank {
                        return Err(Error::InvalidLayout(format!(
                            "CSF layout: mode_order[{level}]={dim} out of range [0, {rank})"
                        )));
                    }
                    let bit = 1u64 << dim;
                    if seen & bit != 0 {
                        return Err(Error::InvalidLayout(format!(
                            "CSF layout: mode_order contains duplicate value {dim}"
                        )));
                    }
                    seen |= bit;
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

            Self::Composite(c) => {
                // Spec (composite.md § Head Descriptor): partition and overlay
                // heads present a coverage/box-addressed index space, so a DYNAMIC
                // dimension (unresolved size) would make coverage math ill-defined.
                // Group has no spatial semantics, so it is unaffected. This check has
                // the composition rule in hand here (unlike a bare shape check), so it
                // belongs at this layer rather than being deferred to the cross-member
                // validator in `crate::composite`, which only sees whole descriptors.
                let needs_static_shape = !matches!(c.rule, CompositionRule::Group);
                if needs_static_shape && shape.has_dynamic() {
                    return Err(Error::InvalidLayout(
                        "composite head: partition/overlay composition requires a fully \
                         static shape (no DYNAMIC dimension)"
                            .to_string(),
                    ));
                }
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
    /// For multi-buffer (sparse) layouts, returns [`Error::LayoutRequiresMultiBuffer`] —
    /// use the layout-specific addressing method instead.
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
            Self::Coo(_) => Err(crate::Error::LayoutRequiresMultiBuffer {
                layout_tag: TAG_COO,
            }),
            Self::Csr(_) => Err(crate::Error::LayoutRequiresMultiBuffer {
                layout_tag: TAG_CSR,
            }),
            Self::Csc(_) => Err(crate::Error::LayoutRequiresMultiBuffer {
                layout_tag: TAG_CSC,
            }),
            // CSF is sparse multi-buffer; use addressing::csf::element_offset with index buffers.
            Self::Csf(_) => Err(crate::Error::LayoutRequiresMultiBuffer {
                layout_tag: TAG_CSF,
            }),
            // BlockPaged is indirect (multi-buffer); use addressing::block_paged directly.
            Self::BlockPaged(_) => Err(crate::Error::LayoutRequiresMultiBuffer {
                layout_tag: TAG_BLOCK_PAGED,
            }),
            // Composite is virtual (zero buffers, not "many buffers") — a distinct failure
            // mode from LayoutRequiresMultiBuffer, so it gets its own error rather than
            // overloading that variant's "use the layout-specific method instead" framing.
            // Addressing a composite means resolving to a member first; see
            // `crate::composite::CompositeTensor`.
            Self::Composite(_) => Err(crate::Error::LayoutIsVirtual {
                layout_tag: TAG_COMPOSITE,
            }),
            Self::PrivateExtension(p) => Err(crate::Error::PrivateLayoutTag(p.tag)),
            Self::Unknown(u) => Err(crate::Error::UnknownLayoutTag(u.tag)),
        }
    }
    /// Returns the per-dimension element strides of a layout whose addressing is an
    /// affine function of the logical index, or `None` for a layout that has no such
    /// form.
    ///
    /// Three layouts qualify: row-major and column-major, whose strides the
    /// specification derives from the shape, and strided, which declares them
    /// explicitly. Tiled, Morton, Hilbert, the sparse layouts, block-paged and
    /// composite address their storage by a rule that no stride vector expresses, and
    /// return `None`.
    ///
    /// Strides are in **logical elements** and signed, per
    /// `docs/spec/layouts/strided.md § Stride Semantics`: positive advances forward,
    /// negative reverses that dimension, zero broadcasts it.
    ///
    /// # Errors
    ///
    /// - [`Error::DynamicDimInIndexing`] — a dimension is [`crate::DYNAMIC`]. Every
    ///   caller of a stride vector needs it to size or address a buffer, which an
    ///   unresolved extent cannot do, so this is rejected for *all* three layouts
    ///   rather than only for the two that compute strides from the dims.
    /// - [`Error::IndexRankMismatch`] — a strided layout's `strides.len()` does not
    ///   equal the shape rank.
    /// - [`Error::AddressOverflow`] — the canonical stride product overflows `int64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::layout::{LayoutDescriptor, StridedLayout, MortonLayout};
    /// use hurray_core::Shape;
    ///
    /// let shape = Shape::new(vec![3, 4]).unwrap();
    ///
    /// // Row-major strides for a 3×4 tensor: [4, 1].
    /// assert_eq!(
    ///     LayoutDescriptor::RowMajor.dense_element_strides(&shape).unwrap(),
    ///     Some(vec![4, 1])
    /// );
    ///
    /// // Column-major strides for the same shape: [1, 3].
    /// assert_eq!(
    ///     LayoutDescriptor::ColMajor.dense_element_strides(&shape).unwrap(),
    ///     Some(vec![1, 3])
    /// );
    ///
    /// // A strided layout reports what it declares, reversal and broadcast included.
    /// let strided = LayoutDescriptor::Strided(StridedLayout::new(vec![-4, 0]));
    /// assert_eq!(strided.dense_element_strides(&shape).unwrap(), Some(vec![-4, 0]));
    ///
    /// // Morton addressing is not affine in the index: no stride vector exists.
    /// let morton = LayoutDescriptor::Morton(MortonLayout::new(vec![2, 2]).unwrap());
    /// assert_eq!(morton.dense_element_strides(&shape).unwrap(), None);
    /// ```
    pub fn dense_element_strides(&self, shape: &Shape) -> crate::Result<Option<Vec<i64>>> {
        if let Some(dim) = shape.dims().iter().position(|&d| d == crate::DYNAMIC) {
            return Err(Error::DynamicDimInIndexing { dim: dim as u32 });
        }
        let dims = shape.dims();
        match self {
            // Row-major: stride[k] = product of dims after k (spec § Element Address).
            Self::RowMajor => {
                let mut strides = vec![0i64; dims.len()];
                let mut acc: i64 = 1;
                for k in (0..dims.len()).rev() {
                    strides[k] = acc;
                    acc = acc
                        .checked_mul(i64::try_from(dims[k]).map_err(|_| Error::AddressOverflow)?)
                        .ok_or(Error::AddressOverflow)?;
                }
                Ok(Some(strides))
            }
            // Column-major: stride[k] = product of dims before k.
            Self::ColMajor => {
                let mut strides = vec![0i64; dims.len()];
                let mut acc: i64 = 1;
                for k in 0..dims.len() {
                    strides[k] = acc;
                    acc = acc
                        .checked_mul(i64::try_from(dims[k]).map_err(|_| Error::AddressOverflow)?)
                        .ok_or(Error::AddressOverflow)?;
                }
                Ok(Some(strides))
            }
            Self::Strided(s) => {
                if s.strides.len() != dims.len() {
                    return Err(Error::IndexRankMismatch {
                        index_rank: s.strides.len(),
                        shape_rank: dims.len(),
                    });
                }
                Ok(Some(s.strides.clone()))
            }
            _ => Ok(None),
        }
    }

    /// Returns the lowest and highest element offset a layout addresses for a shape,
    /// or `None` for a layout with no affine stride form.
    ///
    /// Implements `docs/spec/layouts/strided.md § Buffer Size`:
    ///
    /// ```text
    /// max_offset = sum(max(0, strides[k] * (shape[k] - 1)))
    /// min_offset = sum(min(0, strides[k] * (shape[k] - 1)))
    /// ```
    ///
    /// The pair is inclusive, so a buffer covers the layout iff it holds
    /// `max_offset - min_offset + 1` elements, and `byte_offset` must be at least
    /// `-min_offset` elements into it — `min_offset` is non-zero exactly when some
    /// stride is negative, which is the case where the base address is not the lowest
    /// address the tensor reads.
    ///
    /// An **empty** tensor (any dimension zero) addresses nothing, and is reported as
    /// `Some((0, 0))`: the formula above has no meaning for it, and a caller sizing a
    /// buffer must use the element count, which is zero.
    ///
    /// # Errors
    ///
    /// The errors of [`LayoutDescriptor::dense_element_strides`], plus
    /// [`Error::AddressOverflow`] if the span itself overflows `int64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::layout::{LayoutDescriptor, StridedLayout};
    /// use hurray_core::Shape;
    ///
    /// let shape = Shape::new(vec![3, 4]).unwrap();
    ///
    /// // Row-major 3×4: offsets 0..=11, so the buffer needs 12 elements.
    /// assert_eq!(
    ///     LayoutDescriptor::RowMajor.element_offset_span(&shape).unwrap(),
    ///     Some((0, 11))
    /// );
    ///
    /// // The spec's own example: strides [-4, 1] reach -8 through 3, nine elements,
    /// // with element [0, 0] sitting eight elements into the buffer.
    /// let reversed = LayoutDescriptor::Strided(StridedLayout::new(vec![-4, 1]));
    /// assert_eq!(reversed.element_offset_span(&shape).unwrap(), Some((-8, 3)));
    ///
    /// // A broadcast dimension contributes nothing: four elements, read three times.
    /// let broadcast = LayoutDescriptor::Strided(StridedLayout::new(vec![0, 1]));
    /// assert_eq!(broadcast.element_offset_span(&shape).unwrap(), Some((0, 3)));
    /// ```
    pub fn element_offset_span(&self, shape: &Shape) -> crate::Result<Option<(i64, i64)>> {
        let Some(strides) = self.dense_element_strides(shape)? else {
            return Ok(None);
        };
        let dims = shape.dims();
        if dims.contains(&0) {
            return Ok(Some((0, 0)));
        }
        let mut min_offset: i64 = 0;
        let mut max_offset: i64 = 0;
        for (&stride, &dim) in strides.iter().zip(dims.iter()) {
            let extent = i64::try_from(dim - 1).map_err(|_| Error::AddressOverflow)?;
            let term = stride.checked_mul(extent).ok_or(Error::AddressOverflow)?;
            if term < 0 {
                min_offset = min_offset.checked_add(term).ok_or(Error::AddressOverflow)?;
            } else {
                max_offset = max_offset.checked_add(term).ok_or(Error::AddressOverflow)?;
            }
        }
        Ok(Some((min_offset, max_offset)))
    }
}

/// Returns `true` when a stride vector **provably** maps every index of `dims` to a
/// distinct element offset.
///
/// A layout whose strides alias — two logical indices sharing one stored element — is
/// legal in this format, and sometimes the whole point: a zero stride *is* a broadcast
/// (`docs/spec/layouts/strided.md § Stride Semantics`). It is nonetheless a property a
/// consumer needs to know about, because a tensor cannot be *written* through an aliasing
/// layout without losing values, and a parallel writer cannot treat distinct indices as
/// distinct destinations.
///
/// Zero strides are not the only way to alias, which is the trap this function exists to
/// close: strides `[1, 1]` over shape `[2, 2]` send both `[0, 1]` and `[1, 0]` to offset
/// `1`, with no zero stride anywhere.
///
/// # The test, and what `false` means
///
/// The answer is **conservative**: `true` is a proof of distinctness, while `false` means
/// *not provably distinct* — either genuinely aliasing, or one of the exotic interleavings
/// this test does not recognise (strides `[2, 3]` over shape `[3, 2]` is injective, and
/// reported `false`). Deciding injectivity exactly is a bounded subset-sum problem; the
/// sufficient condition below is the one that admits every layout the format's own
/// § Contiguity definition calls dense, every permutation of it, and every padded variant.
///
/// Three reductions make it a two-line check:
///
/// - **Sign does not matter.** Replacing `i_k` by `d_k - 1 - i_k` on a negative-stride axis
///   is a bijection of the index box onto itself and turns that axis's contribution into
///   `const + i_k · |s_k|`, so the map is injective exactly when the all-positive version
///   is.
/// - **An axis of extent 1 contributes nothing**, and a zero stride on an axis of extent
///   above 1 aliases by definition.
/// - **Sorted by `|stride|` ascending, the strides must be strictly nested**: each axis's
///   stride must exceed the total span of every smaller axis, so that no combination of the
///   smaller axes can reach it.
///
/// # Errors
///
/// - [`Error::IndexRankMismatch`] — `strides.len() != dims.len()`.
/// - [`Error::AddressOverflow`] — the span arithmetic overflows `uint64`.
///
/// # Examples
///
/// ```
/// use hurray_core::layout::strides_provably_distinct;
///
/// // Row-major and column-major strides for a 3×4 tensor, and a padded variant.
/// assert!(strides_provably_distinct(&[4, 1], &[3, 4]).unwrap());
/// assert!(strides_provably_distinct(&[1, 3], &[3, 4]).unwrap());
/// assert!(strides_provably_distinct(&[8, 1], &[3, 4]).unwrap());
///
/// // A reversed axis is still distinct: the sign does not matter.
/// assert!(strides_provably_distinct(&[-4, 1], &[3, 4]).unwrap());
///
/// // Aliasing with no zero stride in sight: [0,1] and [1,0] both land on offset 1.
/// assert!(!strides_provably_distinct(&[1, 1], &[2, 2]).unwrap());
///
/// // A broadcast axis aliases by definition — unless its extent is 1.
/// assert!(!strides_provably_distinct(&[0, 1], &[3, 4]).unwrap());
/// assert!(strides_provably_distinct(&[0, 1], &[1, 4]).unwrap());
///
/// // An empty tensor has no two indices to collide.
/// assert!(strides_provably_distinct(&[1, 1], &[0, 4]).unwrap());
/// ```
pub fn strides_provably_distinct(strides: &[i64], dims: &[u64]) -> crate::Result<bool> {
    if strides.len() != dims.len() {
        return Err(Error::IndexRankMismatch {
            index_rank: strides.len(),
            shape_rank: dims.len(),
        });
    }
    // An empty index space contains no pair of indices to collide.
    if dims.contains(&0) {
        return Ok(true);
    }
    // (|stride|, extent - 1) per axis that can move, which is where the sign reduction and
    // the extent-1 reduction in the doc comment are applied.
    let mut axes: Vec<(u64, u64)> = Vec::with_capacity(dims.len());
    for (&stride, &dim) in strides.iter().zip(dims.iter()) {
        if dim == 1 {
            continue;
        }
        let magnitude = stride.unsigned_abs();
        if magnitude == 0 {
            return Ok(false);
        }
        axes.push((magnitude, dim - 1));
    }
    axes.sort_unstable();
    let mut reach: u64 = 0;
    for (magnitude, extent) in axes {
        // Strictly nested: anything this axis can reach must lie beyond everything the
        // smaller axes can reach together. Equality is a collision, not a near miss — two
        // axes with the same stride put index (1, 0) and (0, 1) on one element.
        if magnitude <= reach {
            return Ok(false);
        }
        reach = reach
            .checked_add(
                magnitude
                    .checked_mul(extent)
                    .ok_or(Error::AddressOverflow)?,
            )
            .ok_or(Error::AddressOverflow)?;
    }
    Ok(true)
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
            LayoutDescriptor::Coo(CooLayout {
                nnz: 0,
                is_sorted: false
            })
            .tag(),
            0x06
        );
        assert_eq!(LayoutDescriptor::Csr(CsrLayout { nnz: 0 }).tag(), 0x07);
        assert_eq!(LayoutDescriptor::Csc(CscLayout { nnz: 0 }).tag(), 0x08);
        assert_eq!(
            LayoutDescriptor::Csf(CsfLayout::new(0, vec![0, 1, 2])).tag(),
            0x09
        );
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
        // Use a tag in the reserved range (0x0A is now the last named tag, block-paged).
        let layout = LayoutDescriptor::Unknown(UnknownLayout {
            tag: 0x0C,
            raw_bytes: vec![],
        });
        assert!(layout.buffer_count().is_none());
    }

    // ── validate_layout_tag_strict ────────────────────────────────────────────

    #[test]
    fn known_tags_pass_strict_validation() {
        for tag in [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x40,
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
        // 0x0A is block-paged and 0x0B is the composite / virtual head (ADR-027)
        // — both named tags, not reserved. 0x0C is now the lower boundary.
        for tag in [0x0C_u8, 0x3F, 0x41, 0x7F, 0x80, 0xEF] {
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
    fn unknown_always_passes_validation() {
        // 0x0A is now block-paged; use a reserved-range tag for Unknown.
        let layout = LayoutDescriptor::Unknown(UnknownLayout {
            tag: 0x0C,
            raw_bytes: vec![],
        });
        assert!(layout.validate_against_shape(&Shape::scalar()).is_ok());
        assert!(layout
            .validate_against_shape(&Shape::new(vec![100, 200]).unwrap())
            .is_ok());
    }
    // ── dense_element_strides / element_offset_span ───────────────────────────

    // Spec docs/spec/layouts/row-major.md § Element Address: stride[k] is the
    // product of the dimensions after k.
    #[test]
    fn row_major_strides_rank3() {
        let shape = Shape::new(vec![2, 3, 4]).unwrap();
        assert_eq!(
            LayoutDescriptor::RowMajor
                .dense_element_strides(&shape)
                .unwrap(),
            Some(vec![12, 4, 1])
        );
    }

    // Spec docs/spec/layouts/column-major.md § Element Address: stride[k] is the
    // product of the dimensions before k.
    #[test]
    fn col_major_strides_rank3() {
        let shape = Shape::new(vec![2, 3, 4]).unwrap();
        assert_eq!(
            LayoutDescriptor::ColMajor
                .dense_element_strides(&shape)
                .unwrap(),
            Some(vec![1, 2, 6])
        );
    }

    // The canonical strides must reproduce the dispatched element offsets exactly:
    // the two are the same spec formula, and a divergence would be a second
    // implementation of it.
    #[test]
    fn canonical_strides_agree_with_element_offset() {
        let shape = Shape::new(vec![3, 4, 5]).unwrap();
        for layout in [LayoutDescriptor::RowMajor, LayoutDescriptor::ColMajor] {
            let strides = layout.dense_element_strides(&shape).unwrap().unwrap();
            for i in 0..3u64 {
                for j in 0..4u64 {
                    for k in 0..5u64 {
                        let dispatched = layout.element_offset(&[i, j, k], &shape).unwrap();
                        let affine = (i as i64) * strides[0]
                            + (j as i64) * strides[1]
                            + (k as i64) * strides[2];
                        assert_eq!(dispatched as i64, affine, "{layout:?} at [{i},{j},{k}]");
                    }
                }
            }
        }
    }

    // A rank-0 tensor has no dimensions, so it has no strides and addresses offset 0.
    #[test]
    fn scalar_strides_are_empty() {
        let shape = Shape::scalar();
        assert_eq!(
            LayoutDescriptor::RowMajor
                .dense_element_strides(&shape)
                .unwrap(),
            Some(vec![])
        );
        assert_eq!(
            LayoutDescriptor::RowMajor
                .element_offset_span(&shape)
                .unwrap(),
            Some((0, 0))
        );
    }

    // A strided layout reports its declared strides verbatim.
    #[test]
    fn strided_reports_declared_strides() {
        let shape = Shape::new(vec![3, 4]).unwrap();
        let layout = LayoutDescriptor::Strided(StridedLayout::new(vec![1, 3]));
        assert_eq!(
            layout.dense_element_strides(&shape).unwrap(),
            Some(vec![1, 3])
        );
    }

    // A strided stride vector of the wrong rank is a rank mismatch, not a panic.
    #[test]
    fn strided_rank_mismatch_is_rejected() {
        let shape = Shape::new(vec![3, 4]).unwrap();
        let layout = LayoutDescriptor::Strided(StridedLayout::new(vec![1]));
        assert!(matches!(
            layout.dense_element_strides(&shape),
            Err(Error::IndexRankMismatch {
                index_rank: 1,
                shape_rank: 2
            })
        ));
    }

    // Every layout without an affine stride form reports None rather than erroring:
    // "no stride vector exists" is an answer, not a failure.
    #[test]
    fn non_affine_layouts_have_no_strides() {
        let shape = Shape::new(vec![4, 4]).unwrap();
        let layouts = [
            LayoutDescriptor::Tiled(Box::new(
                TiledLayout::new(vec![2, 2], 0x01, 0x01, None, None, None).unwrap(),
            )),
            LayoutDescriptor::Morton(MortonLayout::new(vec![2, 2]).unwrap()),
            LayoutDescriptor::Hilbert(HilbertLayout::new(2, 2).unwrap()),
            LayoutDescriptor::Coo(CooLayout::new(4, false)),
            LayoutDescriptor::Csr(CsrLayout::new(4)),
        ];
        for layout in layouts {
            assert_eq!(
                layout.dense_element_strides(&shape).unwrap(),
                None,
                "{layout:?}"
            );
            assert_eq!(
                layout.element_offset_span(&shape).unwrap(),
                None,
                "{layout:?}"
            );
        }
    }

    // A DYNAMIC dimension makes the stride vector unsizeable, for every affine
    // layout — including strided, whose strides are declared rather than derived.
    #[test]
    fn dynamic_dim_is_rejected_for_every_affine_layout() {
        let shape = Shape::new(vec![3, crate::DYNAMIC]).unwrap();
        for layout in [
            LayoutDescriptor::RowMajor,
            LayoutDescriptor::ColMajor,
            LayoutDescriptor::Strided(StridedLayout::new(vec![4, 1])),
        ] {
            assert!(matches!(
                layout.dense_element_strides(&shape),
                Err(Error::DynamicDimInIndexing { dim: 1 })
            ));
        }
    }

    // Spec docs/spec/layouts/strided.md § Buffer Size, on that section's own example:
    // shape [3,4] with strides [-4,1] reaches -8 (element [2,0]) through 3 (element
    // [0,3]), which the formula makes 12 elements — the whole reversed matrix.
    //
    // The § Example prose says "from offset -8 to 0 (9 elements)", which contradicts
    // the formula directly above it; the formula and `element_offset` agree with each
    // other and with this test (see `span_bounds_every_addressed_offset`).
    #[test]
    fn span_matches_spec_negative_stride_example() {
        let shape = Shape::new(vec![3, 4]).unwrap();
        let layout = LayoutDescriptor::Strided(StridedLayout::new(vec![-4, 1]));
        let (min, max) = layout.element_offset_span(&shape).unwrap().unwrap();
        assert_eq!((min, max), (-8, 3));
        assert_eq!(max - min + 1, 12);
    }

    // The span must bound every offset the layout actually addresses — asserted
    // against the dispatched offsets, which are the authority.
    #[test]
    fn span_bounds_every_addressed_offset() {
        let shape = Shape::new(vec![3, 4]).unwrap();
        for layout in [
            LayoutDescriptor::RowMajor,
            LayoutDescriptor::ColMajor,
            LayoutDescriptor::Strided(StridedLayout::new(vec![-4, 1])),
            LayoutDescriptor::Strided(StridedLayout::new(vec![1, -3])),
            LayoutDescriptor::Strided(StridedLayout::new(vec![0, 1])),
            LayoutDescriptor::Strided(StridedLayout::new(vec![8, 1])),
        ] {
            let (min, max) = layout.element_offset_span(&shape).unwrap().unwrap();
            let mut seen_min = i64::MAX;
            let mut seen_max = i64::MIN;
            for i in 0..3u64 {
                for j in 0..4u64 {
                    let off = layout.element_offset(&[i, j], &shape).unwrap() as i64;
                    seen_min = seen_min.min(off);
                    seen_max = seen_max.max(off);
                }
            }
            assert_eq!((min, max), (seen_min, seen_max), "{layout:?}");
        }
    }

    // A padded strided layout spans more elements than the tensor holds: the gap is
    // part of the buffer the layout requires.
    #[test]
    fn span_includes_stride_padding() {
        let shape = Shape::new(vec![3, 4]).unwrap();
        // Row stride 8 for 4-element rows: 4 elements of padding per row.
        let layout = LayoutDescriptor::Strided(StridedLayout::new(vec![8, 1]));
        assert_eq!(layout.element_offset_span(&shape).unwrap(), Some((0, 19)));
    }

    // An empty tensor addresses nothing; the spec's (shape[k] - 1) formula has no
    // meaning for it, so the span is (0, 0) and the element count is what sizes a
    // buffer.
    #[test]
    fn empty_tensor_span_is_zero() {
        let shape = Shape::new(vec![0, 4]).unwrap();
        assert_eq!(
            LayoutDescriptor::RowMajor
                .element_offset_span(&shape)
                .unwrap(),
            Some((0, 0))
        );
        assert_eq!(shape.element_count(), Some(0));
    }
    // ── strides_provably_distinct ─────────────────────────────────────────────

    /// Brute force over a small box: the exact answer, computed independently of the
    /// function under test.
    fn actually_distinct(strides: &[i64], dims: &[u64]) -> bool {
        let total: u64 = dims.iter().product();
        let mut seen = std::collections::HashSet::new();
        for flat in 0..total {
            let mut rest = flat;
            let mut offset: i64 = 0;
            for k in (0..dims.len()).rev() {
                let idx = (rest % dims[k]) as i64;
                rest /= dims[k];
                offset += idx * strides[k];
            }
            if !seen.insert(offset) {
                return false;
            }
        }
        true
    }

    /// The one direction that must never fail: a `true` answer is a *proof*. A false
    /// positive here is the memory-safety bug the predicate exists to prevent, so this
    /// sweeps every small stride vector rather than a chosen few.
    #[test]
    fn claimed_distinctness_is_always_real() {
        let dim_sets: [&[u64]; 6] = [&[4], &[2, 2], &[2, 3], &[3, 2], &[2, 2, 2], &[1, 3, 2]];
        for dims in dim_sets {
            let rank = dims.len();
            let candidates: Vec<i64> = (-4..=4).collect();
            // Every stride vector in [-4, 4]^rank.
            let mut index = vec![0usize; rank];
            loop {
                let strides: Vec<i64> = index.iter().map(|&i| candidates[i]).collect();
                if strides_provably_distinct(&strides, dims).unwrap() {
                    assert!(
                        actually_distinct(&strides, dims),
                        "claimed distinct but is not: strides {strides:?} dims {dims:?}"
                    );
                }
                // Odometer over the candidate grid.
                let mut k = rank;
                loop {
                    if k == 0 {
                        break;
                    }
                    k -= 1;
                    index[k] += 1;
                    if index[k] < candidates.len() {
                        break;
                    }
                    index[k] = 0;
                    if k == 0 {
                        break;
                    }
                }
                if index.iter().all(|&i| i == 0) {
                    break;
                }
            }
        }
    }

    /// The other direction is deliberately incomplete, and this records which cases the
    /// conservative test gives up on, so a later pass can see the cost of tightening it.
    #[test]
    fn conservatism_is_confined_to_interleavings() {
        // Injective, and reported false: axis 0 strides by 2 over 3 positions (0,2,4) and
        // axis 1 by 3 over 2 (0,3) — the sums 0,2,4,3,5,7 happen not to collide.
        assert!(actually_distinct(&[2, 3], &[3, 2]));
        assert!(!strides_provably_distinct(&[2, 3], &[3, 2]).unwrap());
    }

    #[test]
    fn the_review_counterexample_is_rejected() {
        // The case that made a parallel run race: no zero stride, and [0,1] and [1,0]
        // both address element 1.
        assert!(!strides_provably_distinct(&[1, 1], &[2, 2]).unwrap());
        assert!(!actually_distinct(&[1, 1], &[2, 2]));
        // And at the scale where the collision crosses a work-item boundary.
        assert!(!strides_provably_distinct(&[1, 1], &[2, 300_000]).unwrap());
    }

    #[test]
    fn canonical_strides_of_every_rank_are_distinct() {
        for dims in [vec![7u64], vec![3, 4], vec![2, 3, 4], vec![2, 2, 2, 2]] {
            let shape = Shape::new(dims.clone()).unwrap();
            for layout in [LayoutDescriptor::RowMajor, LayoutDescriptor::ColMajor] {
                let strides = layout.dense_element_strides(&shape).unwrap().unwrap();
                assert!(
                    strides_provably_distinct(&strides, &dims).unwrap(),
                    "{layout:?} over {dims:?}"
                );
            }
        }
    }

    #[test]
    fn a_rank_zero_tensor_is_distinct() {
        assert!(strides_provably_distinct(&[], &[]).unwrap());
    }

    #[test]
    fn extent_one_axes_are_ignored_whatever_their_stride() {
        // Three axes of extent 1 and one that moves: no two indices exist to collide.
        assert!(strides_provably_distinct(&[0, 0, 0, 1], &[1, 1, 1, 5]).unwrap());
        assert!(strides_provably_distinct(&[7, 7, 7, 1], &[1, 1, 1, 5]).unwrap());
    }

    #[test]
    fn a_rank_mismatch_is_an_error_not_a_verdict() {
        assert!(matches!(
            strides_provably_distinct(&[1], &[2, 2]),
            Err(Error::IndexRankMismatch {
                index_rank: 1,
                shape_rank: 2
            })
        ));
    }

    #[test]
    fn a_span_that_overflows_is_an_error_not_a_verdict() {
        // |stride| × (dim - 1) beyond uint64: no answer rather than a wrapped one.
        assert!(matches!(
            strides_provably_distinct(&[i64::MAX, 1], &[u64::MAX / 2, 2]),
            Err(Error::AddressOverflow)
        ));
    }
}
