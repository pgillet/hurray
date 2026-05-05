//! CSC (Compressed Sparse Column) layout descriptor.
//!
//! Tag `0x09`. Rank MUST be 2. Buffer count = 3 (values + row_indices + col_ptr).
//! See `docs/spec/layouts/csc.md`.

/// Descriptor for the CSC (Compressed Sparse Column) sparse layout.
///
/// CSC is the column analog of CSR and is defined **only for rank-2 tensors**.
/// A conforming implementation MUST reject a CSC descriptor whose tensor rank
/// is not 2 (checked by [`LayoutDescriptor::validate_against_shape`]).
///
/// Three buffers are required:
///
/// | Buffer | Contents |
/// |--------|----------|
/// | 0 — `values` | Non-zero values in column-major order; `nnz` elements. |
/// | 1 — `row_indices` | Row index of each non-zero; `nnz` `uint64` elements. |
/// | 2 — `col_ptr` | Column pointer array; `ncols + 1` `uint64` elements where `ncols = shape[1]`. |
///
/// The `byte_offset` field MUST be `0` for CSC tensors.
///
/// > Also known as **CCS (Compressed Column Storage)**.
///
/// # Examples
///
/// ```
/// use hurray_core::layout::{CscLayout, LayoutDescriptor};
///
/// let layout = LayoutDescriptor::Csc(CscLayout::new(5));
/// assert_eq!(layout.tag(), 0x09);
/// assert_eq!(layout.buffer_count().map(|n| n.get()), Some(3));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct CscLayout {
    /// Number of stored (non-zero) elements. MAY be 0 for an empty sparse matrix.
    pub nnz: u64,
}

impl CscLayout {
    /// Creates a new [`CscLayout`].
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::layout::CscLayout;
    ///
    /// let c = CscLayout::new(5);
    /// assert_eq!(c.nnz, 5);
    /// ```
    pub fn new(nnz: u64) -> Self {
        Self { nnz }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::LayoutDescriptor;

    #[test]
    fn csc_tag_is_0x09() {
        let layout = LayoutDescriptor::Csc(CscLayout::new(0));
        assert_eq!(layout.tag(), 0x09);
    }

    #[test]
    fn csc_buffer_count_is_3() {
        use std::num::NonZeroU8;
        let layout = LayoutDescriptor::Csc(CscLayout::new(5));
        assert_eq!(layout.buffer_count(), Some(NonZeroU8::new(3).unwrap()));
    }
}
