//! Tensor shape — rank and dimension sizes.
//!
//! A [`Shape`] is an ordered sequence of dimension sizes. Each size is a
//! `u64`; the sentinel value [`DYNAMIC`] (`u64::MAX`) marks a dimension whose
//! concrete size is not known at descriptor-write time.
//!
//! See `docs/spec/data-model.md` for the normative definition.

use std::fmt;

use crate::Error;

/// Sentinel value for a **dynamic dimension** whose concrete size is not known
/// at descriptor-write time.
///
/// In the wire encoding this is `0xFFFF_FFFF_FFFF_FFFF` (`UINT64_MAX`).
/// A reader MUST NOT compute buffer sizes, strides, or element counts for a
/// tensor containing a dynamic dimension without first resolving it to a
/// concrete value.
///
/// # Examples
///
/// ```
/// use hurray_core::{Shape, DYNAMIC};
///
/// let shape = Shape::new(vec![1, DYNAMIC, 768]).expect("valid shape");
/// assert!(shape.has_dynamic());
/// assert!(shape.element_count().is_none());
/// ```
pub const DYNAMIC: u64 = u64::MAX;

/// Maximum permitted tensor rank.
///
/// A writer MUST NOT emit a descriptor with `rank > MAX_RANK`. A reader MUST
/// reject any descriptor whose rank field exceeds this value.
///
/// The cap matches PyTorch's `MAX_DIMS = 64` and bounds the shape-array size
/// at 512 bytes (64 × 8 bytes per dimension), enabling stack allocation on the
/// descriptor-parsing hot path.
///
/// # Examples
///
/// ```
/// use hurray_core::MAX_RANK;
///
/// assert_eq!(MAX_RANK, 64);
/// ```
pub const MAX_RANK: usize = 64;

/// The shape of a tensor: an ordered sequence of dimension sizes.
///
/// Each dimension size is a [`u64`]. The special value [`DYNAMIC`] marks a
/// dimension whose concrete size is unknown at descriptor-write time. A size
/// of `0` denotes an empty dimension (the tensor contains no elements along
/// that axis).
///
/// A [`Shape`] with no dimensions (`rank == 0`) represents a **scalar tensor**
/// containing exactly one element.
///
/// # Examples
///
/// ```
/// use hurray_core::{Shape, DYNAMIC};
///
/// // 3-D tensor with all static dimensions.
/// let s = Shape::new(vec![3, 4, 5]).expect("valid");
/// assert_eq!(s.rank(), 3);
/// assert_eq!(s.element_count(), Some(60));
///
/// // Scalar tensor.
/// let scalar = Shape::scalar();
/// assert_eq!(scalar.rank(), 0);
/// assert_eq!(scalar.element_count(), Some(1));
///
/// // Shape with a dynamic dimension.
/// let dyn_shape = Shape::new(vec![1, DYNAMIC, 768]).expect("valid");
/// assert!(dyn_shape.has_dynamic());
/// assert!(dyn_shape.element_count().is_none());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(Vec<u64>);

impl Shape {
    /// Creates a new [`Shape`] from the given dimension sizes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::RankExceedsMaximum`] if `dims.len() > MAX_RANK` (64).
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{Shape, Error};
    ///
    /// let s = Shape::new(vec![3, 4, 5]).expect("valid");
    /// assert_eq!(s.rank(), 3);
    ///
    /// // Rank 65 must be rejected.
    /// let too_many = Shape::new(vec![1u64; 65]);
    /// assert!(matches!(too_many, Err(Error::RankExceedsMaximum { rank: 65, max: 64 })));
    /// ```
    pub fn new(dims: impl Into<Vec<u64>>) -> crate::Result<Self> {
        let dims = dims.into();
        if dims.len() > MAX_RANK {
            return Err(Error::RankExceedsMaximum {
                rank: dims.len() as u32,
                max: MAX_RANK as u32,
            });
        }
        Ok(Self(dims))
    }

    /// Returns a [`Shape`] representing a **scalar tensor** (rank 0, one element).
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::Shape;
    ///
    /// let s = Shape::scalar();
    /// assert_eq!(s.rank(), 0);
    /// assert_eq!(s.dims(), &[]);
    /// assert_eq!(s.element_count(), Some(1));
    /// ```
    #[inline]
    pub fn scalar() -> Self {
        Self(Vec::new())
    }

    /// Returns the number of dimensions (the rank) of this shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::Shape;
    ///
    /// assert_eq!(Shape::scalar().rank(), 0);
    /// assert_eq!(Shape::new(vec![3, 4]).unwrap().rank(), 2);
    /// ```
    #[inline]
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Returns the dimension sizes as a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::Shape;
    ///
    /// let s = Shape::new(vec![2, 3]).unwrap();
    /// assert_eq!(s.dims(), &[2, 3]);
    /// ```
    #[inline]
    pub fn dims(&self) -> &[u64] {
        &self.0
    }

    /// Returns the total number of logical elements, or `None` if any
    /// dimension is [`DYNAMIC`] or if the product overflows `u64`.
    ///
    /// The product of an empty sequence (scalar) is `1`. A tensor with any
    /// zero-size dimension has `element_count == Some(0)`.
    ///
    /// A reader MUST NOT use this value to compute buffer sizes or strides
    /// until all dynamic dimensions have been resolved.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{Shape, DYNAMIC};
    ///
    /// // Static shape.
    /// assert_eq!(Shape::new(vec![3, 4, 5]).unwrap().element_count(), Some(60));
    ///
    /// // Scalar: product of empty sequence is 1.
    /// assert_eq!(Shape::scalar().element_count(), Some(1));
    ///
    /// // Empty tensor: any zero dimension makes the count 0.
    /// assert_eq!(Shape::new(vec![3, 0, 5]).unwrap().element_count(), Some(0));
    ///
    /// // Dynamic dimension: result is unknown.
    /// assert_eq!(Shape::new(vec![1, DYNAMIC, 768]).unwrap().element_count(), None);
    /// ```
    pub fn element_count(&self) -> Option<u64> {
        let mut product: u64 = 1;
        for &dim in &self.0 {
            if dim == DYNAMIC {
                return None;
            }
            product = product.checked_mul(dim)?;
        }
        Some(product)
    }

    /// Returns `true` if any dimension has size `0`.
    ///
    /// An empty tensor is valid; its data buffer has size 0 bytes. Note that
    /// a [`DYNAMIC`] dimension does not make a tensor empty — its size is
    /// unknown, not zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{Shape, DYNAMIC};
    ///
    /// assert!(Shape::new(vec![3, 0, 5]).unwrap().is_empty_tensor());
    /// assert!(!Shape::new(vec![3, 4, 5]).unwrap().is_empty_tensor());
    /// // Dynamic is not empty.
    /// assert!(!Shape::new(vec![1, DYNAMIC, 768]).unwrap().is_empty_tensor());
    /// ```
    #[inline]
    pub fn is_empty_tensor(&self) -> bool {
        self.0.contains(&0)
    }

    /// Returns `true` if any dimension equals [`DYNAMIC`].
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{Shape, DYNAMIC};
    ///
    /// assert!(Shape::new(vec![1, DYNAMIC, 768]).unwrap().has_dynamic());
    /// assert!(!Shape::new(vec![1, 128, 768]).unwrap().has_dynamic());
    /// ```
    #[inline]
    pub fn has_dynamic(&self) -> bool {
        self.0.contains(&DYNAMIC)
    }
}

impl fmt::Display for Shape {
    /// Formats the shape as a bracket-enclosed, comma-separated list of sizes.
    ///
    /// Dynamic dimensions are shown as `?`. An empty (scalar) shape is `[]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{Shape, DYNAMIC};
    ///
    /// assert_eq!(Shape::scalar().to_string(), "[]");
    /// assert_eq!(Shape::new(vec![3, 4, 5]).unwrap().to_string(), "[3, 4, 5]");
    /// assert_eq!(Shape::new(vec![1, DYNAMIC, 768]).unwrap().to_string(), "[1, ?, 768]");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[")?;
        for (i, &dim) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            if dim == DYNAMIC {
                f.write_str("?")?;
            } else {
                write!(f, "{dim}")?;
            }
        }
        f.write_str("]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Shape::new ───────────────────────────────────────────────────────────

    #[test]
    fn new_rank_0_succeeds() {
        let s = Shape::new(vec![]).expect("rank 0 is valid");
        assert_eq!(s.rank(), 0);
    }

    #[test]
    fn new_rank_1_succeeds() {
        let s = Shape::new(vec![42]).expect("rank 1 is valid");
        assert_eq!(s.rank(), 1);
        assert_eq!(s.dims(), &[42]);
    }

    #[test]
    fn new_rank_64_succeeds() {
        let s = Shape::new(vec![1u64; 64]).expect("rank 64 is valid (MAX_RANK)");
        assert_eq!(s.rank(), 64);
    }

    /// Spec § data-model: a writer MUST NOT emit a descriptor with rank > MAX_RANK;
    /// a reader MUST reject any descriptor whose rank field exceeds this value.
    #[test]
    fn new_rank_65_returns_rank_exceeds_maximum() {
        let result = Shape::new(vec![1u64; 65]);
        assert!(
            matches!(result, Err(Error::RankExceedsMaximum { rank: 65, max: 64 })),
            "expected RankExceedsMaximum for rank 65, got {result:?}"
        );
    }

    #[test]
    fn new_rank_100_returns_rank_exceeds_maximum() {
        let result = Shape::new(vec![1u64; 100]);
        assert!(matches!(
            result,
            Err(Error::RankExceedsMaximum { rank: 100, max: 64 })
        ));
    }

    // ── Shape::scalar ────────────────────────────────────────────────────────

    #[test]
    fn scalar_has_rank_0() {
        assert_eq!(Shape::scalar().rank(), 0);
    }

    #[test]
    fn scalar_dims_is_empty() {
        assert_eq!(Shape::scalar().dims(), &[] as &[u64]);
    }

    #[test]
    fn scalar_element_count_is_one() {
        assert_eq!(Shape::scalar().element_count(), Some(1));
    }

    #[test]
    fn scalar_is_not_empty_tensor() {
        assert!(!Shape::scalar().is_empty_tensor());
    }

    #[test]
    fn scalar_has_no_dynamic() {
        assert!(!Shape::scalar().has_dynamic());
    }

    // ── rank ─────────────────────────────────────────────────────────────────

    #[test]
    fn rank_matches_dims_len() {
        for n in [0, 1, 2, 10, 64] {
            let s = Shape::new(vec![3u64; n]).expect("valid");
            assert_eq!(s.rank(), n, "rank mismatch for n={n}");
        }
    }

    // ── dims ─────────────────────────────────────────────────────────────────

    #[test]
    fn dims_returns_original_slice() {
        let input = vec![3u64, 4, 5];
        let s = Shape::new(input.clone()).expect("valid");
        assert_eq!(s.dims(), input.as_slice());
    }

    #[test]
    fn dims_is_empty_for_scalar() {
        assert!(Shape::scalar().dims().is_empty());
    }

    // ── element_count ────────────────────────────────────────────────────────

    #[test]
    fn element_count_static_shape_3x4x5() {
        let s = Shape::new(vec![3, 4, 5]).expect("valid");
        assert_eq!(s.element_count(), Some(60));
    }

    #[test]
    fn element_count_1d_shape() {
        assert_eq!(Shape::new(vec![7]).unwrap().element_count(), Some(7));
    }

    #[test]
    fn element_count_scalar_is_one() {
        assert_eq!(Shape::scalar().element_count(), Some(1));
    }

    #[test]
    fn element_count_with_zero_dim_is_zero() {
        assert_eq!(Shape::new(vec![3, 0, 5]).unwrap().element_count(), Some(0));
    }

    #[test]
    fn element_count_all_zero_dims_is_zero() {
        assert_eq!(Shape::new(vec![0, 0]).unwrap().element_count(), Some(0));
    }

    #[test]
    fn element_count_with_dynamic_is_none() {
        assert_eq!(
            Shape::new(vec![1, DYNAMIC, 768]).unwrap().element_count(),
            None
        );
    }

    #[test]
    fn element_count_all_dynamic_is_none() {
        assert_eq!(
            Shape::new(vec![DYNAMIC, DYNAMIC]).unwrap().element_count(),
            None
        );
    }

    /// Overflow during product computation must return None, not panic.
    #[test]
    fn element_count_overflow_returns_none() {
        // u64::MAX / 2 + 1 times two dims = overflow
        let big = u64::MAX / 2 + 1;
        let s = Shape::new(vec![big, 2]).expect("rank 2 is valid");
        assert_eq!(s.element_count(), None);
    }

    // ── is_empty_tensor ──────────────────────────────────────────────────────

    #[test]
    fn is_empty_tensor_true_when_one_dim_is_zero() {
        assert!(Shape::new(vec![3, 0, 5]).unwrap().is_empty_tensor());
    }

    #[test]
    fn is_empty_tensor_true_when_all_dims_zero() {
        assert!(Shape::new(vec![0]).unwrap().is_empty_tensor());
    }

    #[test]
    fn is_empty_tensor_false_for_static_nonzero_shape() {
        assert!(!Shape::new(vec![3, 4, 5]).unwrap().is_empty_tensor());
    }

    /// A DYNAMIC dimension is unknown, not zero — the tensor is not empty.
    #[test]
    fn is_empty_tensor_false_for_dynamic_dim() {
        assert!(!Shape::new(vec![1, DYNAMIC, 768]).unwrap().is_empty_tensor());
    }

    #[test]
    fn is_empty_tensor_false_for_scalar() {
        assert!(!Shape::scalar().is_empty_tensor());
    }

    // ── has_dynamic ──────────────────────────────────────────────────────────

    #[test]
    fn has_dynamic_true_when_one_dim_is_dynamic() {
        assert!(Shape::new(vec![1, DYNAMIC, 768]).unwrap().has_dynamic());
    }

    #[test]
    fn has_dynamic_true_when_all_dims_dynamic() {
        assert!(Shape::new(vec![DYNAMIC, DYNAMIC]).unwrap().has_dynamic());
    }

    #[test]
    fn has_dynamic_false_for_fully_static_shape() {
        assert!(!Shape::new(vec![1, 128, 768]).unwrap().has_dynamic());
    }

    #[test]
    fn has_dynamic_false_for_scalar() {
        assert!(!Shape::scalar().has_dynamic());
    }

    // ── Display ──────────────────────────────────────────────────────────────

    #[test]
    fn display_scalar_is_empty_brackets() {
        assert_eq!(Shape::scalar().to_string(), "[]");
    }

    #[test]
    fn display_1d_shape() {
        assert_eq!(Shape::new(vec![5]).unwrap().to_string(), "[5]");
    }

    #[test]
    fn display_3d_static_shape() {
        assert_eq!(Shape::new(vec![3, 4, 5]).unwrap().to_string(), "[3, 4, 5]");
    }

    #[test]
    fn display_dynamic_dimension_shown_as_question_mark() {
        assert_eq!(
            Shape::new(vec![1, DYNAMIC, 768]).unwrap().to_string(),
            "[1, ?, 768]"
        );
    }

    #[test]
    fn display_all_dynamic() {
        assert_eq!(
            Shape::new(vec![DYNAMIC, DYNAMIC]).unwrap().to_string(),
            "[?, ?]"
        );
    }

    #[test]
    fn display_zero_dimension() {
        assert_eq!(Shape::new(vec![3, 0, 5]).unwrap().to_string(), "[3, 0, 5]");
    }

    // ── DYNAMIC and MAX_RANK constants ───────────────────────────────────────

    #[test]
    fn dynamic_constant_is_u64_max() {
        assert_eq!(DYNAMIC, u64::MAX);
    }

    #[test]
    fn max_rank_constant_is_64() {
        assert_eq!(MAX_RANK, 64);
    }
}
