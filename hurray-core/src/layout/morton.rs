//! Morton (Z-order curve) layout descriptor.
//!
//! Tag `0x05`. Carries per-dimension bit counts for the Morton encoding.
//! See `docs/spec/layouts/morton.md`.

use crate::{Error, Result};

/// Descriptor for the Morton (Z-order curve) layout.
///
/// Elements are stored by interleaving the bits of their dimension indices in
/// round-robin order (LSB of dimension 0 first), producing a linear order with
/// good spatial locality for multi-dimensional access patterns.
///
/// For each dimension `k`, `shape[k]` MUST satisfy `shape[k] <= 2^morton_bits[k]`.
/// The buffer must hold exactly `2^(sum(morton_bits))` elements.
///
/// # Examples
///
/// ```
/// use hurray_core::layout::{LayoutDescriptor, MortonLayout};
///
/// // 4×4 tensor: 2 bits per dimension.
/// let layout = LayoutDescriptor::Morton(MortonLayout::new(vec![2, 2]).unwrap());
/// assert_eq!(layout.tag(), 0x05);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct MortonLayout {
    /// Number of bits used per dimension in the Morton encoding.
    /// Every value MUST be > 0.
    pub morton_bits: Vec<u32>,
}

impl MortonLayout {
    /// Creates a new [`MortonLayout`], validating that every entry in
    /// `morton_bits` is greater than 0.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidLayout`] if any `morton_bits[k]` is 0, or if
    /// `morton_bits` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::layout::MortonLayout;
    ///
    /// let m = MortonLayout::new(vec![2, 2]).unwrap();
    /// assert_eq!(m.morton_bits, [2, 2]);
    ///
    /// // Zero bits is invalid.
    /// assert!(MortonLayout::new(vec![2, 0]).is_err());
    /// ```
    pub fn new(morton_bits: Vec<u32>) -> Result<Self> {
        if morton_bits.is_empty() {
            return Err(Error::InvalidLayout(
                "morton_bits must not be empty".to_string(),
            ));
        }
        for (k, &bits) in morton_bits.iter().enumerate() {
            if bits == 0 {
                return Err(Error::InvalidLayout(format!(
                    "morton_bits[{k}] must be > 0, got 0"
                )));
            }
        }
        Ok(Self { morton_bits })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::LayoutDescriptor;

    #[test]
    fn morton_tag_is_0x05() {
        let layout = LayoutDescriptor::Morton(MortonLayout::new(vec![2, 2]).unwrap());
        assert_eq!(layout.tag(), 0x05);
    }

    #[test]
    fn morton_buffer_count_is_1() {
        use std::num::NonZeroU8;
        let layout = LayoutDescriptor::Morton(MortonLayout::new(vec![2, 2]).unwrap());
        assert_eq!(layout.buffer_count(), Some(NonZeroU8::new(1).unwrap()));
    }

    #[test]
    fn rejects_zero_bits() {
        assert!(matches!(
            MortonLayout::new(vec![2, 0]),
            Err(Error::InvalidLayout(_))
        ));
    }

    #[test]
    fn rejects_empty_morton_bits() {
        assert!(matches!(
            MortonLayout::new(vec![]),
            Err(Error::InvalidLayout(_))
        ));
    }
}
