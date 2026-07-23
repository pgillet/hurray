//! CSR (Compressed Sparse Row) layout descriptor.
//!
//! Tag `0x07`. Rank MUST be 2. Buffer count = 3 (values + col_indices + row_ptr).
//! See `docs/spec/layouts/csr.md`.

/// Descriptor for the CSR (Compressed Sparse Row) sparse layout.
///
/// CSR is defined **only for rank-2 tensors**. A conforming implementation MUST
/// reject a CSR descriptor whose tensor rank is not 2 (checked by
/// [`LayoutDescriptor::validate_against_shape`]).
///
/// Three buffers are required:
///
/// | Buffer | Contents |
/// |--------|----------|
/// | 0 — `values` | Non-zero values in row-major order; `nnz` elements. |
/// | 1 — `col_indices` | Column index of each non-zero; `nnz` `uint64` elements. |
/// | 2 — `row_ptr` | Row pointer array; `nrows + 1` `uint64` elements where `nrows = shape[0]`. |
///
/// The `byte_offset` field MUST be `0` for CSR tensors.
///
/// # Examples
///
/// ```
/// use hurray_core::layout::{CsrLayout, LayoutDescriptor};
///
/// let layout = LayoutDescriptor::Csr(CsrLayout::new(5));
/// assert_eq!(layout.tag(), 0x07);
/// assert_eq!(layout.buffer_count().map(|n| n.get()), Some(3));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct CsrLayout {
    /// Number of stored (non-zero) elements. MAY be 0 for an empty sparse matrix.
    pub nnz: u64,
}

impl CsrLayout {
    /// Creates a new [`CsrLayout`].
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::layout::CsrLayout;
    ///
    /// let c = CsrLayout::new(5);
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
    fn csr_tag_is_0x07() {
        let layout = LayoutDescriptor::Csr(CsrLayout::new(0));
        assert_eq!(layout.tag(), 0x07);
    }

    #[test]
    fn csr_buffer_count_is_3() {
        use std::num::NonZeroU8;
        let layout = LayoutDescriptor::Csr(CsrLayout::new(5));
        assert_eq!(layout.buffer_count(), Some(NonZeroU8::new(3).unwrap()));
    }
}
