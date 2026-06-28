//! CSF (Compressed Sparse Fiber) layout descriptor.
//!
//! Tag `0x0A`. Rank MUST be ≥ 3. Buffer count = 2·rank+1 (values + rank pos arrays + rank crd arrays).
//! See `docs/spec/layouts/csf.md`.

/// Descriptor for the CSF (Compressed Sparse Fiber) sparse layout.
///
/// CSF is the rank-N generalisation of CSR/CSC, storing a sparse tensor as a
/// tree of `rank` levels. Each level compresses one mode with a `(pos, crd)` pair;
/// the `values` buffer holds non-zero values at the leaves.
///
/// CSF is **defined only for rank ≥ 3 tensors** in this version of the specification.
/// Rank-2 sparse matrices are served by CSR/CSC; see [`crate::layout::CsrLayout`].
///
/// `buffer_count = 2·rank+1` comes from the layout descriptor alone because
/// `mode_order.len()` equals the tensor rank — the rank is embedded in the field,
/// not repeated separately.
///
/// # Buffer table
///
/// | Buffer index | Name | Element type | Length |
/// |---|---|---|---|
/// | `0` | `values` | tensor element type | `nnz` elements |
/// | `2L + 1` | `pos_L` | `uint64` | `n_{L-1} + 1` elements (`2` for L=0) |
/// | `2L + 2` | `crd_L` | `uint64` | `n_L` elements (`nnz` for the leaf level) |
///
/// where `n_L` is the count of tree nodes at level `L`, `n_{-1} = 1` (virtual root),
/// and `n_{rank-1} = nnz`.
///
/// The `byte_offset` field MUST be `0` for CSF tensors — the first element is
/// reached by descending the tree, not at a fixed offset.
///
/// # Examples
///
/// ```
/// use hurray_core::layout::{CsfLayout, LayoutDescriptor};
///
/// // Rank-3 sparse tensor with 4 non-zeros, identity mode_order.
/// let layout = CsfLayout::new(4, vec![0, 1, 2]);
/// let desc = LayoutDescriptor::Csf(layout);
/// assert_eq!(desc.tag(), 0x0A);
/// // 2*rank+1 = 2*3+1 = 7 buffers.
/// assert_eq!(desc.buffer_count().map(|n| n.get()), Some(7));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct CsfLayout {
    /// Number of stored (non-zero) elements. MAY be 0 for an empty sparse tensor.
    pub nnz: u64,

    /// Permutation of `0..rank-1`; `mode_order[L]` is the logical dimension stored at
    /// tree level `L`. The tensor rank equals `mode_order.len()`.
    ///
    /// `mode_order` is stored here (rather than just `rank`) because buffer_count()
    /// and validate_against_shape() both need it, and carrying the full permutation
    /// avoids a separate rank field while making the accessor surface explicit.
    // WHY mode_order carries rank: buffer_count() = 2·rank+1 needs rank, and the
    // permutation is always validated to have length == rank anyway — no duplication.
    pub mode_order: Vec<u32>,
}

impl CsfLayout {
    /// Creates a new [`CsfLayout`] with the given number of non-zeros and mode order.
    ///
    /// This constructor does not validate that `mode_order` is a valid permutation
    /// of `0..mode_order.len()` or that `mode_order.len() >= 3`; call
    /// [`crate::layout::LayoutDescriptor::validate_against_shape`] with the tensor
    /// shape to perform those checks.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::layout::CsfLayout;
    ///
    /// // Rank-3 tensor, identity mode order, 10 non-zeros.
    /// let c = CsfLayout::new(10, vec![0, 1, 2]);
    /// assert_eq!(c.nnz, 10);
    /// assert_eq!(c.mode_order, [0, 1, 2]);
    ///
    /// // Rank-4 tensor with reordered modes.
    /// let c4 = CsfLayout::new(0, vec![3, 0, 1, 2]);
    /// assert_eq!(c4.mode_order.len(), 4);
    /// ```
    pub fn new(nnz: u64, mode_order: Vec<u32>) -> Self {
        Self { nnz, mode_order }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::LayoutDescriptor;
    use std::num::NonZeroU8;

    // ── Tag ──────────────────────────────────────────────────────────────────

    /// Spec §csf.md: layout tag MUST be 0x0A.
    #[test]
    fn csf_tag_is_0x0a() {
        let layout = LayoutDescriptor::Csf(CsfLayout::new(0, vec![0, 1, 2]));
        assert_eq!(layout.tag(), 0x0A);
    }

    /// Tag is independent of nnz and mode_order content.
    #[test]
    fn csf_tag_is_0x0a_for_varying_nnz_and_permutation() {
        let layout = LayoutDescriptor::Csf(CsfLayout::new(999, vec![2, 0, 1, 3]));
        assert_eq!(layout.tag(), 0x0A);
    }

    // ── Buffer count ─────────────────────────────────────────────────────────

    /// Spec §csf.md §Buffer Table: buffer_count = 2·rank+1.
    /// rank=3 → 2*3+1 = 7.
    #[test]
    fn csf_buffer_count_rank3_is_7() {
        let layout = LayoutDescriptor::Csf(CsfLayout::new(5, vec![0, 1, 2]));
        assert_eq!(layout.buffer_count(), Some(NonZeroU8::new(7).unwrap()));
    }

    /// rank=4 → 2*4+1 = 9.
    #[test]
    fn csf_buffer_count_rank4_is_9() {
        let layout = LayoutDescriptor::Csf(CsfLayout::new(0, vec![0, 1, 2, 3]));
        assert_eq!(layout.buffer_count(), Some(NonZeroU8::new(9).unwrap()));
    }

    /// rank=5 → 2*5+1 = 11.
    #[test]
    fn csf_buffer_count_rank5_is_11() {
        let layout = LayoutDescriptor::Csf(CsfLayout::new(0, vec![0, 1, 2, 3, 4]));
        assert_eq!(layout.buffer_count(), Some(NonZeroU8::new(11).unwrap()));
    }

    /// buffer_count is derived from mode_order.len(), not a separate rank field.
    #[test]
    fn csf_buffer_count_derived_from_mode_order_len() {
        // Non-identity permutation: rank still equals mode_order.len().
        let layout = LayoutDescriptor::Csf(CsfLayout::new(10, vec![2, 0, 1]));
        // 2*3+1 = 7
        assert_eq!(layout.buffer_count(), Some(NonZeroU8::new(7).unwrap()));
    }

    // ── CsfLayout::new stores fields faithfully ───────────────────────────────

    #[test]
    fn csf_layout_new_stores_nnz() {
        let c = CsfLayout::new(42, vec![0, 1, 2]);
        assert_eq!(c.nnz, 42);
    }

    #[test]
    fn csf_layout_new_stores_mode_order() {
        let mo = vec![2u32, 0, 1];
        let c = CsfLayout::new(0, mo.clone());
        assert_eq!(c.mode_order, mo);
    }

    #[test]
    fn csf_layout_new_nnz_zero_is_valid() {
        let c = CsfLayout::new(0, vec![0, 1, 2]);
        assert_eq!(c.nnz, 0);
    }

    // ── validate_against_shape ───────────────────────────────────────────────

    /// Spec §csf.md §Validity Constraints: rank MUST be >= 3.
    #[test]
    fn csf_validate_accepts_rank3() {
        let layout = LayoutDescriptor::Csf(CsfLayout::new(0, vec![0, 1, 2]));
        let shape = crate::Shape::new(vec![2u64, 3, 4]).unwrap();
        assert!(layout.validate_against_shape(&shape).is_ok());
    }

    #[test]
    fn csf_validate_accepts_rank4() {
        let layout = LayoutDescriptor::Csf(CsfLayout::new(0, vec![0, 1, 2, 3]));
        let shape = crate::Shape::new(vec![2u64, 3, 4, 5]).unwrap();
        assert!(layout.validate_against_shape(&shape).is_ok());
    }

    #[test]
    fn csf_validate_accepts_rank5() {
        let layout = LayoutDescriptor::Csf(CsfLayout::new(0, vec![0, 1, 2, 3, 4]));
        let shape = crate::Shape::new(vec![2u64, 3, 4, 5, 6]).unwrap();
        assert!(layout.validate_against_shape(&shape).is_ok());
    }

    /// Spec §csf.md: CSF MUST NOT be used below rank 3.
    #[test]
    fn csf_validate_rejects_rank0() {
        let layout = LayoutDescriptor::Csf(CsfLayout::new(0, vec![]));
        assert!(matches!(
            layout.validate_against_shape(&crate::Shape::scalar()),
            Err(crate::Error::InvalidLayout(_))
        ));
    }

    #[test]
    fn csf_validate_rejects_rank1() {
        let layout = LayoutDescriptor::Csf(CsfLayout::new(0, vec![0]));
        let shape = crate::Shape::new(vec![10u64]).unwrap();
        assert!(matches!(
            layout.validate_against_shape(&shape),
            Err(crate::Error::InvalidLayout(_))
        ));
    }

    #[test]
    fn csf_validate_rejects_rank2() {
        let layout = LayoutDescriptor::Csf(CsfLayout::new(0, vec![0, 1]));
        let shape = crate::Shape::new(vec![4u64, 5]).unwrap();
        assert!(matches!(
            layout.validate_against_shape(&shape),
            Err(crate::Error::InvalidLayout(_))
        ));
    }

    /// mode_order.len() must equal shape.rank().
    #[test]
    fn csf_validate_rejects_mode_order_len_not_equal_to_rank() {
        // mode_order has 4 entries but shape has rank 3.
        let layout = LayoutDescriptor::Csf(CsfLayout::new(0, vec![0, 1, 2, 3]));
        let shape = crate::Shape::new(vec![2u64, 3, 4]).unwrap();
        assert!(matches!(
            layout.validate_against_shape(&shape),
            Err(crate::Error::InvalidLayout(_))
        ));
    }

    /// mode_order out-of-range value (>= rank) must be rejected.
    #[test]
    fn csf_validate_rejects_mode_order_out_of_range_value() {
        // mode_order[2]=3 is out of range for rank=3.
        let layout = LayoutDescriptor::Csf(CsfLayout::new(0, vec![0, 1, 3]));
        let shape = crate::Shape::new(vec![2u64, 3, 4]).unwrap();
        assert!(matches!(
            layout.validate_against_shape(&shape),
            Err(crate::Error::InvalidLayout(_))
        ));
    }

    /// mode_order with a repeated value (not a permutation) must be rejected.
    #[test]
    fn csf_validate_rejects_mode_order_repeated_value() {
        // [0, 1, 1] repeats dimension 1.
        let layout = LayoutDescriptor::Csf(CsfLayout::new(0, vec![0, 1, 1]));
        let shape = crate::Shape::new(vec![2u64, 3, 4]).unwrap();
        assert!(matches!(
            layout.validate_against_shape(&shape),
            Err(crate::Error::InvalidLayout(_))
        ));
    }

    /// Non-identity permutation [2, 0, 1] must be accepted.
    #[test]
    fn csf_validate_accepts_non_identity_permutation() {
        let layout = LayoutDescriptor::Csf(CsfLayout::new(4, vec![2, 0, 1]));
        let shape = crate::Shape::new(vec![2u64, 3, 4]).unwrap();
        assert!(layout.validate_against_shape(&shape).is_ok());
    }

    /// Reverse permutation [2, 1, 0] must be accepted.
    #[test]
    fn csf_validate_accepts_reverse_permutation() {
        let layout = LayoutDescriptor::Csf(CsfLayout::new(0, vec![2, 1, 0]));
        let shape = crate::Shape::new(vec![2u64, 3, 4]).unwrap();
        assert!(layout.validate_against_shape(&shape).is_ok());
    }

    // ── PartialEq / Clone / Debug derives ────────────────────────────────────

    #[test]
    fn csf_layout_partial_eq() {
        let a = CsfLayout::new(4, vec![0, 1, 2]);
        let b = CsfLayout::new(4, vec![0, 1, 2]);
        assert_eq!(a, b);
    }

    #[test]
    fn csf_layout_ne_different_nnz() {
        let a = CsfLayout::new(4, vec![0, 1, 2]);
        let b = CsfLayout::new(0, vec![0, 1, 2]);
        assert_ne!(a, b);
    }

    #[test]
    fn csf_layout_clone_is_equal() {
        let orig = CsfLayout::new(10, vec![2, 0, 1]);
        assert_eq!(orig.clone(), orig);
    }

    #[test]
    fn csf_layout_debug_is_non_empty() {
        let c = CsfLayout::new(0, vec![0, 1, 2]);
        assert!(!format!("{c:?}").is_empty());
    }
}
