//! COO (Coordinate) sparse layout descriptor.
//!
//! Tag `0x06`. Buffer count = 2 (values + indices).
//! See `docs/spec/layouts/coo.md`.

/// Descriptor for the COO (Coordinate list) sparse layout.
///
/// A COO tensor stores non-zero elements as `(index_tuple, value)` pairs.
/// It requires two buffers:
///
/// | Buffer | Contents |
/// |--------|----------|
/// | 0 — `values` | Non-zero element values; `nnz` elements of the tensor element type. |
/// | 1 — `indices` | Index tuples as `uint64`; `nnz × rank` elements, stored row-major: `indices[i * rank + d]` is coordinate `d` of non-zero `i`. |
///
/// The `byte_offset` field in the common descriptor header MUST be `0` for
/// COO tensors (no meaningful "first element at a fixed offset").
///
/// # Examples
///
/// ```
/// use hurray_core::layout::{CooLayout, LayoutDescriptor};
///
/// let layout = LayoutDescriptor::Coo(CooLayout::new(3, true));
/// assert_eq!(layout.tag(), 0x06);
/// assert_eq!(layout.buffer_count().map(|n| n.get()), Some(2));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct CooLayout {
    /// Number of stored (non-zero) elements. MAY be 0 for an empty sparse tensor.
    pub nnz: u64,

    /// `true` if non-zeros are stored in lexicographic index order
    /// (dimension 0 major); `false` if no ordering guarantee is made.
    pub is_sorted: bool,
}

impl CooLayout {
    /// Creates a new [`CooLayout`].
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::layout::CooLayout;
    ///
    /// let c = CooLayout::new(42, true);
    /// assert_eq!(c.nnz, 42);
    /// assert!(c.is_sorted);
    /// ```
    pub fn new(nnz: u64, is_sorted: bool) -> Self {
        Self { nnz, is_sorted }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::LayoutDescriptor;

    #[test]
    fn coo_tag_is_0x06() {
        let layout = LayoutDescriptor::Coo(CooLayout::new(0, false));
        assert_eq!(layout.tag(), 0x06);
    }

    #[test]
    fn coo_buffer_count_is_2() {
        use std::num::NonZeroU8;
        let layout = LayoutDescriptor::Coo(CooLayout::new(100, true));
        assert_eq!(layout.buffer_count(), Some(NonZeroU8::new(2).unwrap()));
    }
}
