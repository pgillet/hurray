//! Strided layout descriptor.
//!
//! Tag `0x03`. Carries an explicit per-dimension stride vector.
//! See `docs/spec/layouts/strided.md`.

/// Descriptor for the strided layout.
///
/// The strided layout generalises row-major and column-major by allowing an
/// arbitrary stride per dimension. Negative strides (reversed dimension) and
/// zero strides (broadcast dimension) are both valid.
///
/// `strides` are in **logical elements**, not bytes.
///
/// # Examples
///
/// ```
/// use hurray_core::layout::{LayoutDescriptor, StridedLayout};
///
/// // Row-major strides for a 3×4 tensor: [4, 1]
/// let layout = LayoutDescriptor::Strided(StridedLayout::new(vec![4, 1]));
/// assert_eq!(layout.tag(), 0x03);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct StridedLayout {
    /// Per-dimension strides in **logical elements**.
    ///
    /// Positive: forward; negative: reversed; zero: broadcast (virtual) dimension.
    /// `strides.len()` MUST equal the tensor rank (validated by
    /// [`LayoutDescriptor::validate_against_shape`]).
    pub strides: Vec<i64>,
}

impl StridedLayout {
    /// Creates a new [`StridedLayout`] with the given per-dimension strides.
    ///
    /// No rank validation is performed here — call
    /// [`LayoutDescriptor::validate_against_shape`] to check that
    /// `strides.len() == shape.rank()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::layout::StridedLayout;
    ///
    /// // Reversed first dimension of a 3×4 tensor (row-major with rows reversed).
    /// let s = StridedLayout::new(vec![-4, 1]);
    /// assert_eq!(s.strides, &[-4, 1]);
    /// ```
    pub fn new(strides: Vec<i64>) -> Self {
        Self { strides }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::LayoutDescriptor;

    #[test]
    fn strided_tag_is_0x03() {
        let layout = LayoutDescriptor::Strided(StridedLayout::new(vec![4, 1]));
        assert_eq!(layout.tag(), 0x03);
    }

    #[test]
    fn strided_buffer_count_is_1() {
        use std::num::NonZeroU8;
        let layout = LayoutDescriptor::Strided(StridedLayout::new(vec![4, 1]));
        assert_eq!(layout.buffer_count(), Some(NonZeroU8::new(1).unwrap()));
    }

    #[test]
    fn strided_allows_negative_and_zero_strides() {
        // Negative strides (reversed dim) and zero strides (broadcast) are valid.
        let s = StridedLayout::new(vec![-4, 0, 1]);
        assert_eq!(s.strides, [-4, 0, 1]);
    }
}
