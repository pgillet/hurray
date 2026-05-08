//! General subpaving layout descriptor.
//!
//! Tag `0x06`. The most general dense layout: an irregular set of rectangular
//! regions each with its own inner layout.
//! See `docs/spec/layouts/subpaving.md`.

use crate::layout::LayoutDescriptor;
use crate::{Error, Result};

/// A single rectangular region within a general subpaving layout.
///
/// Each region has its own origin, shape, layout tag, buffer reference, byte
/// offset, and (for tags with additional descriptor fields) an `inner_layout`
/// descriptor. Row-major (`0x01`) and column-major (`0x02`) have no additional
/// fields; all other tags carry their layout-specific fields in `inner_layout`.
///
/// # Examples
///
/// ```
/// use hurray_core::layout::RegionDescriptor;
///
/// let region = RegionDescriptor::new(vec![0, 0], vec![4, 4], 0x01, 0, 0).unwrap();
/// assert_eq!(region.region_layout_tag, 0x01);
/// assert!(region.inner_layout.is_none());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct RegionDescriptor {
    /// Starting index of the region along each dimension (inclusive).
    pub origin: Vec<u64>,

    /// Size of the region along each dimension. Every value MUST be > 0.
    pub region_shape: Vec<u64>,

    /// Layout tag for elements within this region.
    /// MUST NOT be `0x00` or `0xFF`.
    pub region_layout_tag: u8,

    /// Index into the tensor's buffer table for this region's data.
    pub buffer_index: u32,

    /// Byte offset within the referenced buffer to the start of this region's data.
    pub region_byte_offset: u64,

    /// Inner layout descriptor for tags with additional descriptor fields.
    ///
    /// `None` for `region_layout_tag` values `0x01` (row-major) and `0x02`
    /// (column-major), which carry no additional fields. `Some` for all other
    /// tags; the contained `LayoutDescriptor` variant MUST satisfy
    /// `inner.tag() == region_layout_tag`.
    ///
    /// See ADR-015 and `docs/spec/layouts/subpaving.md` § Additional Descriptor Fields.
    pub inner_layout: Option<Box<LayoutDescriptor>>,
}

impl RegionDescriptor {
    /// Creates a new [`RegionDescriptor`] with no inner layout (for row-major
    /// `0x01` and column-major `0x02` regions).
    ///
    /// Validates that `region_layout_tag` is not `0x00` or `0xFF`, all
    /// `region_shape` values are > 0, and `origin` and `region_shape` have the
    /// same length.
    ///
    /// For tags with additional descriptor fields (strided, tiled, Morton,
    /// Hilbert, or recursive subpaving), use [`RegionDescriptor::with_inner_layout`]
    /// to attach the inner [`LayoutDescriptor`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidLayout`] on any violation.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::layout::RegionDescriptor;
    ///
    /// let r = RegionDescriptor::new(vec![0, 4], vec![4, 4], 0x01, 0, 0).unwrap();
    /// assert_eq!(r.origin, [0, 4]);
    /// assert!(r.inner_layout.is_none());
    /// ```
    pub fn new(
        origin: Vec<u64>,
        region_shape: Vec<u64>,
        region_layout_tag: u8,
        buffer_index: u32,
        region_byte_offset: u64,
    ) -> Result<Self> {
        if region_layout_tag == 0x00 || region_layout_tag == 0xFF {
            return Err(Error::InvalidLayout(format!(
                "region_layout_tag 0x{region_layout_tag:02X} is permanently reserved"
            )));
        }
        if origin.len() != region_shape.len() {
            return Err(Error::InvalidLayout(format!(
                "origin.len() ({}) != region_shape.len() ({})",
                origin.len(),
                region_shape.len()
            )));
        }
        for (k, &dim) in region_shape.iter().enumerate() {
            if dim == 0 {
                return Err(Error::InvalidLayout(format!(
                    "region_shape[{k}] must be > 0, got 0"
                )));
            }
        }
        Ok(Self {
            origin,
            region_shape,
            region_layout_tag,
            buffer_index,
            region_byte_offset,
            inner_layout: None,
        })
    }

    /// Attaches an inner [`LayoutDescriptor`] to this region.
    ///
    /// The inner layout's tag MUST match `self.region_layout_tag`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidLayout`] if `inner.tag() != self.region_layout_tag`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::layout::{LayoutDescriptor, MortonLayout, RegionDescriptor};
    ///
    /// let morton = LayoutDescriptor::Morton(MortonLayout::new(vec![4, 4]).unwrap());
    /// let r = RegionDescriptor::new(vec![0, 0], vec![4, 4], 0x05, 0, 0)
    ///     .unwrap()
    ///     .with_inner_layout(morton)
    ///     .unwrap();
    /// assert!(r.inner_layout.is_some());
    /// ```
    pub fn with_inner_layout(mut self, inner: LayoutDescriptor) -> Result<Self> {
        if inner.tag() != self.region_layout_tag {
            return Err(Error::InvalidLayout(format!(
                "inner_layout tag 0x{:02X} does not match region_layout_tag 0x{:02X}",
                inner.tag(),
                self.region_layout_tag
            )));
        }
        self.inner_layout = Some(Box::new(inner));
        Ok(self)
    }
}

/// Descriptor for the general subpaving layout.
///
/// The tensor's index space is partitioned into a set of non-overlapping
/// rectangular regions, each with its own layout descriptor. This is the most
/// general Hurray layout and accommodates tensors with structurally
/// heterogeneous regions (mixed dense/sparse, different tiling strategies,
/// independently-produced shards).
///
/// Coverage and non-overlap constraints are SHOULD-level for readers in strict
/// mode and are checked by [`LayoutDescriptor::validate_against_shape`].
///
/// # Examples
///
/// ```
/// use hurray_core::layout::{LayoutDescriptor, RegionDescriptor, SubpavingLayout};
///
/// // 8×8 tensor split into four 4×4 row-major quadrants.
/// let regions = vec![
///     RegionDescriptor::new(vec![0, 0], vec![4, 4], 0x01, 0, 0).unwrap(),
///     RegionDescriptor::new(vec![0, 4], vec![4, 4], 0x01, 0, 64).unwrap(),
///     RegionDescriptor::new(vec![4, 0], vec![4, 4], 0x01, 0, 128).unwrap(),
///     RegionDescriptor::new(vec![4, 4], vec![4, 4], 0x01, 0, 192).unwrap(),
/// ];
/// let layout = LayoutDescriptor::Subpaving(SubpavingLayout::new(regions).unwrap());
/// assert_eq!(layout.tag(), 0x06);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct SubpavingLayout {
    /// The ordered list of region descriptors. MUST be non-empty.
    pub regions: Vec<RegionDescriptor>,
}

impl SubpavingLayout {
    /// Creates a new [`SubpavingLayout`], validating that `regions` is non-empty.
    ///
    /// Coverage and non-overlap validation are deferred to
    /// [`LayoutDescriptor::validate_against_shape`] because they require the
    /// tensor shape, which is not available at construction time.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidLayout`] if `regions` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::layout::{RegionDescriptor, SubpavingLayout};
    ///
    /// let r = RegionDescriptor::new(vec![0, 0], vec![8, 8], 0x01, 0, 0).unwrap();
    /// let sp = SubpavingLayout::new(vec![r]).unwrap();
    /// assert_eq!(sp.regions.len(), 1);
    /// ```
    pub fn new(regions: Vec<RegionDescriptor>) -> Result<Self> {
        if regions.is_empty() {
            return Err(Error::InvalidLayout(
                "subpaving must have at least one region (region_count must be > 0)".to_string(),
            ));
        }
        Ok(Self { regions })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{LayoutDescriptor, MortonLayout};

    fn simple_region() -> RegionDescriptor {
        RegionDescriptor::new(vec![0, 0], vec![4, 4], 0x01, 0, 0).unwrap()
    }

    // with_inner_layout: Morton layout attached to a 0x05 region.
    #[test]
    fn region_with_inner_layout_morton() {
        let morton = LayoutDescriptor::Morton(MortonLayout::new(vec![4, 4]).unwrap());
        let r = RegionDescriptor::new(vec![0, 0], vec![4, 4], 0x05, 0, 0)
            .unwrap()
            .with_inner_layout(morton)
            .unwrap();
        assert!(r.inner_layout.is_some());
        assert_eq!(r.inner_layout.as_deref().unwrap().tag(), 0x05);
    }

    // with_inner_layout: tag mismatch returns error.
    #[test]
    fn region_with_inner_layout_tag_mismatch() {
        let morton = LayoutDescriptor::Morton(MortonLayout::new(vec![4, 4]).unwrap());
        // tag 0x01 in new(), 0x05 in with_inner_layout → mismatch
        let err = RegionDescriptor::new(vec![0, 0], vec![4, 4], 0x01, 0, 0)
            .unwrap()
            .with_inner_layout(morton);
        assert!(matches!(err, Err(Error::InvalidLayout(_))));
    }

    #[test]
    fn subpaving_tag_is_0x06() {
        let layout =
            LayoutDescriptor::Subpaving(SubpavingLayout::new(vec![simple_region()]).unwrap());
        assert_eq!(layout.tag(), 0x06);
    }

    #[test]
    fn subpaving_buffer_count_is_1() {
        use std::num::NonZeroU8;
        let layout =
            LayoutDescriptor::Subpaving(SubpavingLayout::new(vec![simple_region()]).unwrap());
        assert_eq!(layout.buffer_count(), Some(NonZeroU8::new(1).unwrap()));
    }

    #[test]
    fn rejects_empty_regions() {
        assert!(matches!(
            SubpavingLayout::new(vec![]),
            Err(Error::InvalidLayout(_))
        ));
    }

    #[test]
    fn rejects_reserved_region_layout_tag_zero() {
        assert!(matches!(
            RegionDescriptor::new(vec![0, 0], vec![4, 4], 0x00, 0, 0),
            Err(Error::InvalidLayout(_))
        ));
    }

    #[test]
    fn rejects_reserved_region_layout_tag_ff() {
        assert!(matches!(
            RegionDescriptor::new(vec![0, 0], vec![4, 4], 0xFF, 0, 0),
            Err(Error::InvalidLayout(_))
        ));
    }

    #[test]
    fn rejects_zero_region_shape_dim() {
        assert!(matches!(
            RegionDescriptor::new(vec![0, 0], vec![0, 4], 0x01, 0, 0),
            Err(Error::InvalidLayout(_))
        ));
    }

    #[test]
    fn rejects_mismatched_origin_and_shape_len() {
        assert!(matches!(
            RegionDescriptor::new(vec![0], vec![4, 4], 0x01, 0, 0),
            Err(Error::InvalidLayout(_))
        ));
    }
}
