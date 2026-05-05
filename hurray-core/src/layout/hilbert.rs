//! Hilbert curve layout descriptor.
//!
//! Tag `0x40`. Tier 2. Carries curve order and rank.
//! See `docs/spec/layouts/hilbert.md`.

use crate::{Error, Result};

/// Descriptor for the Hilbert space-filling curve layout.
///
/// Elements are ordered according to a Hilbert curve, which provides better
/// spatial locality than Morton (Z-order): consecutive Hilbert indices always
/// differ by exactly 1 in exactly one coordinate (L∞ distance = 1).
///
/// Constraints (enforced by [`HilbertLayout::new`]):
/// - `hilbert_order > 0`
/// - `hilbert_rank >= 2`
/// - Every tensor dimension must equal `2^hilbert_order`
///   (validated at shape-binding time via
///   [`LayoutDescriptor::validate_against_shape`])
///
/// The normative index mapping is the Skilling (2004) algorithm; see
/// `docs/spec/layouts/hilbert.md § Normative Index Mapping`.
///
/// # Examples
///
/// ```
/// use hurray_core::layout::{LayoutDescriptor, HilbertLayout};
///
/// // 2-D Hilbert curve, order 2 (4×4 tensor).
/// let layout = LayoutDescriptor::Hilbert(HilbertLayout::new(2, 2).unwrap());
/// assert_eq!(layout.tag(), 0x40);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct HilbertLayout {
    /// Order of the Hilbert curve. MUST be > 0.
    /// Each tensor dimension must equal `2^hilbert_order`.
    pub hilbert_order: u32,

    /// Number of curve dimensions. MUST equal the tensor rank. MUST be >= 2.
    pub hilbert_rank: u32,
}

impl HilbertLayout {
    /// Creates a new [`HilbertLayout`], validating that `hilbert_order > 0`
    /// and `hilbert_rank >= 2`.
    ///
    /// Shape consistency (`shape[k] == 2^hilbert_order` for all `k`) is
    /// deferred to [`LayoutDescriptor::validate_against_shape`] because
    /// the layout descriptor does not carry the shape.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidLayout`] if `hilbert_order == 0` or
    /// `hilbert_rank < 2`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::layout::HilbertLayout;
    ///
    /// let h = HilbertLayout::new(3, 2).unwrap(); // 8×8 tensor
    /// assert_eq!(h.hilbert_order, 3);
    /// assert_eq!(h.hilbert_rank, 2);
    ///
    /// assert!(HilbertLayout::new(0, 2).is_err()); // order must be > 0
    /// assert!(HilbertLayout::new(2, 1).is_err()); // rank must be >= 2
    /// ```
    pub fn new(hilbert_order: u32, hilbert_rank: u32) -> Result<Self> {
        if hilbert_order == 0 {
            return Err(Error::InvalidLayout(
                "hilbert_order must be > 0".to_string(),
            ));
        }
        if hilbert_rank < 2 {
            return Err(Error::InvalidLayout(format!(
                "hilbert_rank must be >= 2, got {hilbert_rank}"
            )));
        }
        Ok(Self {
            hilbert_order,
            hilbert_rank,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::LayoutDescriptor;

    #[test]
    fn hilbert_tag_is_0x40() {
        let layout = LayoutDescriptor::Hilbert(HilbertLayout::new(2, 2).unwrap());
        assert_eq!(layout.tag(), 0x40);
    }

    #[test]
    fn hilbert_buffer_count_is_1() {
        use std::num::NonZeroU8;
        let layout = LayoutDescriptor::Hilbert(HilbertLayout::new(2, 2).unwrap());
        assert_eq!(layout.buffer_count(), Some(NonZeroU8::new(1).unwrap()));
    }

    #[test]
    fn rejects_zero_order() {
        assert!(matches!(
            HilbertLayout::new(0, 2),
            Err(Error::InvalidLayout(_))
        ));
    }

    #[test]
    fn rejects_rank_one() {
        assert!(matches!(
            HilbertLayout::new(2, 1),
            Err(Error::InvalidLayout(_))
        ));
    }

    #[test]
    fn rejects_rank_zero() {
        assert!(matches!(
            HilbertLayout::new(2, 0),
            Err(Error::InvalidLayout(_))
        ));
    }
}
