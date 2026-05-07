//! Row-major (C-order) element offset computation.
//!
//! Spec: docs/spec/layouts/row-major.md § Element Address

use crate::{Error, Result, Shape};

use super::validate_index;

pub(crate) fn element_offset(index: &[u64], shape: &Shape) -> Result<u64> {
    validate_index(index, shape)?;
    row_major_offset(index, shape.dims())
}

/// Computes the row-major linear element offset without re-validating.
///
/// `index` and `dims` must have the same length; all indices must be in-bounds.
pub(crate) fn row_major_offset(index: &[u64], dims: &[u64]) -> Result<u64> {
    if index.is_empty() {
        return Ok(0); // scalar
    }
    let mut offset: u64 = 0;
    let mut stride: u64 = 1;
    for k in (0..index.len()).rev() {
        offset = offset
            .checked_add(index[k].checked_mul(stride).ok_or(Error::AddressOverflow)?)
            .ok_or(Error::AddressOverflow)?;
        if k > 0 {
            stride = stride.checked_mul(dims[k]).ok_or(Error::AddressOverflow)?;
        }
    }
    Ok(offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Error, Shape, DYNAMIC};

    // Spec docs/spec/layouts/row-major.md § Element Address:
    // offset(i_0, …, i_{R-1}) = Σ i_k × Π_{j=k+1}^{R-1} dim_j
    //
    // For shape [3,4], element [1,2]: offset = 1*4 + 2*1 = 6
    #[test]
    fn rm_rank2_spec_example() {
        let shape = Shape::new(vec![3, 4]).unwrap();
        assert_eq!(element_offset(&[1, 2], &shape).unwrap(), 6);
    }

    // shape [2,3,4], element [1,1,2]: 1*12 + 1*4 + 2*1 = 18
    #[test]
    fn rm_rank3() {
        let shape = Shape::new(vec![2, 3, 4]).unwrap();
        assert_eq!(element_offset(&[1, 1, 2], &shape).unwrap(), 18);
    }

    // Scalar (rank 0): offset is always 0 regardless of index (empty).
    #[test]
    fn rm_scalar() {
        let shape = Shape::scalar();
        assert_eq!(element_offset(&[], &shape).unwrap(), 0);
    }

    // Rank-1: element [5] in shape [10] → offset 5.
    #[test]
    fn rm_rank1() {
        let shape = Shape::new(vec![10]).unwrap();
        assert_eq!(element_offset(&[5], &shape).unwrap(), 5);
    }

    // First element of a 2D tensor: always offset 0.
    #[test]
    fn rm_first_element() {
        let shape = Shape::new(vec![3, 4]).unwrap();
        assert_eq!(element_offset(&[0, 0], &shape).unwrap(), 0);
    }

    // Last element of a 2D tensor [3,4]: [2,3] → 2*4 + 3 = 11.
    #[test]
    fn rm_last_element() {
        let shape = Shape::new(vec![3, 4]).unwrap();
        assert_eq!(element_offset(&[2, 3], &shape).unwrap(), 11);
    }

    // Error: index rank differs from shape rank.
    #[test]
    fn rm_rank_mismatch() {
        let shape = Shape::new(vec![3, 4]).unwrap();
        let err = element_offset(&[1], &shape).unwrap_err();
        assert!(
            matches!(
                err,
                Error::IndexRankMismatch {
                    index_rank: 1,
                    shape_rank: 2
                }
            ),
            "expected IndexRankMismatch, got {err:?}"
        );
    }

    // Error: index component >= dimension size.
    #[test]
    fn rm_index_out_of_range() {
        let shape = Shape::new(vec![3, 4]).unwrap();
        // index[1]=4 is out of range for dim size 4 (valid range is [0,3]).
        let err = element_offset(&[1, 4], &shape).unwrap_err();
        assert!(
            matches!(
                err,
                Error::IndexOutOfRange {
                    dim: 1,
                    index: 4,
                    size: 4
                }
            ),
            "expected IndexOutOfRange, got {err:?}"
        );
    }

    // Error: DYNAMIC dimension in shape cannot be used for addressing.
    #[test]
    fn rm_dynamic_dim() {
        let shape = Shape::new(vec![3, DYNAMIC]).unwrap();
        let err = element_offset(&[1, 2], &shape).unwrap_err();
        assert!(
            matches!(err, Error::DynamicDimInIndexing { dim: 1 }),
            "expected DynamicDimInIndexing, got {err:?}"
        );
    }
}
