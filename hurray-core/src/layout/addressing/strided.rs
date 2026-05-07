//! Strided layout element offset computation.
//!
//! Spec: docs/spec/layouts/strided.md § Element Address

use crate::layout::StridedLayout;
use crate::{Error, Result, Shape};

use super::{validate_index, ElementAddress};

// Sum is computed as i64 to support negative strides; cast to u64 via
// two's-complement reinterpretation so byte_address_from_element_offset
// can reconstruct the signed value. ADR-014 Amendment.
impl ElementAddress for StridedLayout {
    fn element_offset(&self, index: &[u64], shape: &Shape) -> Result<u64> {
        validate_index(index, shape)?;
        if self.strides.len() != shape.rank() {
            return Err(Error::IndexRankMismatch {
                index_rank: self.strides.len(),
                shape_rank: shape.rank(),
            });
        }
        let mut sum: i64 = 0;
        for (&idx_val, &stride) in index.iter().zip(self.strides.iter()) {
            let term = (idx_val as i64)
                .checked_mul(stride)
                .ok_or(Error::AddressOverflow)?;
            sum = sum.checked_add(term).ok_or(Error::AddressOverflow)?;
        }
        Ok(sum as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Error, Shape, DYNAMIC};

    // Spec docs/spec/layouts/strided.md § Element Address:
    // offset = Σ index[k] × strides[k]  (strides are signed i64, in logical elements)
    //
    // Row-major strides [4,1] for shape [3,4]: element [1,2] → 1*4 + 2*1 = 6.
    #[test]
    fn strided_row_major_equivalent() {
        let shape = Shape::new(vec![3, 4]).unwrap();
        let layout = StridedLayout::new(vec![4, 1]);
        assert_eq!(layout.element_offset(&[1, 2], &shape).unwrap(), 6);
    }

    // Col-major strides [1,3] for shape [3,4]: element [1,2] → 1*1 + 2*3 = 7.
    #[test]
    fn strided_col_major_equivalent() {
        let shape = Shape::new(vec![3, 4]).unwrap();
        let layout = StridedLayout::new(vec![1, 3]);
        assert_eq!(layout.element_offset(&[1, 2], &shape).unwrap(), 7);
    }

    // Negative stride (reversed row dimension):
    // shape [3,4], strides [-4, 1].
    // element [0,0] → 0*(-4) + 0*1 = 0.
    #[test]
    fn strided_negative_stride_first_element_is_zero() {
        let shape = Shape::new(vec![3, 4]).unwrap();
        let layout = StridedLayout::new(vec![-4, 1]);
        assert_eq!(layout.element_offset(&[0, 0], &shape).unwrap(), 0);
    }

    // Negative stride (reversed rows):
    // shape [3,4], strides [-4, 1].
    // element [2,3] → 2*(-4) + 3*1 = -5 → stored as i64 → u64 two's-complement.
    #[test]
    fn strided_negative_stride_last_element() {
        let shape = Shape::new(vec![3, 4]).unwrap();
        let layout = StridedLayout::new(vec![-4, 1]);
        let offset = layout.element_offset(&[2, 3], &shape).unwrap();
        // -5 as u64 via two's-complement.
        assert_eq!(offset, (-5i64) as u64);
    }

    // Zero stride (broadcast dimension):
    // strides [0,1] for shape [5,4]: element [3,2] → 3*0 + 2*1 = 2.
    #[test]
    fn strided_zero_stride_broadcast() {
        let shape = Shape::new(vec![5, 4]).unwrap();
        let layout = StridedLayout::new(vec![0, 1]);
        assert_eq!(layout.element_offset(&[3, 2], &shape).unwrap(), 2);
        // All elements in the first dimension map to the same offset.
        assert_eq!(layout.element_offset(&[0, 2], &shape).unwrap(), 2);
        assert_eq!(layout.element_offset(&[4, 2], &shape).unwrap(), 2);
    }

    // Error: strides length does not match shape rank.
    // validate_index passes (rank matches), but then the strides-rank check fires.
    #[test]
    fn strided_strides_rank_mismatch() {
        let shape = Shape::new(vec![3, 4]).unwrap();
        // 3 strides for rank-2 shape → mismatch.
        let layout = StridedLayout::new(vec![4, 1, 0]);
        let err = layout.element_offset(&[1, 2], &shape).unwrap_err();
        assert!(
            matches!(err, Error::IndexRankMismatch { .. }),
            "expected IndexRankMismatch, got {err:?}"
        );
    }

    // Error: index out of bounds.
    #[test]
    fn strided_index_out_of_range() {
        let shape = Shape::new(vec![3, 4]).unwrap();
        let layout = StridedLayout::new(vec![4, 1]);
        let err = layout.element_offset(&[3, 0], &shape).unwrap_err();
        assert!(
            matches!(
                err,
                Error::IndexOutOfRange {
                    dim: 0,
                    index: 3,
                    size: 3
                }
            ),
            "expected IndexOutOfRange, got {err:?}"
        );
    }

    // Error: DYNAMIC dimension in shape.
    #[test]
    fn strided_dynamic_dim() {
        let shape = Shape::new(vec![3, DYNAMIC]).unwrap();
        let layout = StridedLayout::new(vec![1, 1]);
        let err = layout.element_offset(&[0, 1], &shape).unwrap_err();
        assert!(
            matches!(err, Error::DynamicDimInIndexing { dim: 1 }),
            "expected DynamicDimInIndexing, got {err:?}"
        );
    }
}
