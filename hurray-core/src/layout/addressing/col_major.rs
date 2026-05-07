//! Column-major (Fortran-order) element offset computation.
//!
//! Spec: docs/spec/layouts/column-major.md § Element Address

use crate::{Error, Result, Shape};

use super::validate_index;

pub(crate) fn element_offset(index: &[u64], shape: &Shape) -> Result<u64> {
    validate_index(index, shape)?;
    col_major_offset(index, shape.dims())
}

/// Computes the column-major linear element offset without re-validating.
pub(crate) fn col_major_offset(index: &[u64], dims: &[u64]) -> Result<u64> {
    if index.is_empty() {
        return Ok(0); // scalar
    }
    let mut offset: u64 = 0;
    let mut stride: u64 = 1;
    for k in 0..index.len() {
        offset = offset
            .checked_add(index[k].checked_mul(stride).ok_or(Error::AddressOverflow)?)
            .ok_or(Error::AddressOverflow)?;
        if k < index.len() - 1 {
            stride = stride.checked_mul(dims[k]).ok_or(Error::AddressOverflow)?;
        }
    }
    Ok(offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Error, Shape, DYNAMIC};

    // Spec docs/spec/layouts/column-major.md § Element Address:
    // offset(i_0, …, i_{R-1}) = Σ i_k × Π_{j=0}^{k-1} dim_j
    //
    // For shape [3,4], element [1,2]: offset = 1*1 + 2*3 = 7.
    // (Differs from row-major which gives 6 for the same index.)
    #[test]
    fn cm_rank2_spec_example() {
        let shape = Shape::new(vec![3, 4]).unwrap();
        assert_eq!(element_offset(&[1, 2], &shape).unwrap(), 7);
    }

    // Verify col-major ≠ row-major for the same non-trivial index.
    #[test]
    fn cm_differs_from_row_major() {
        let shape = Shape::new(vec![3, 4]).unwrap();
        let cm = element_offset(&[1, 2], &shape).unwrap();
        let rm = super::super::row_major::element_offset(&[1, 2], &shape).unwrap();
        assert_ne!(
            cm, rm,
            "col-major and row-major must differ for [1,2] in [3,4]"
        );
        assert_eq!(cm, 7);
        assert_eq!(rm, 6);
    }

    // Scalar (rank 0): offset is always 0.
    #[test]
    fn cm_scalar() {
        let shape = Shape::scalar();
        assert_eq!(element_offset(&[], &shape).unwrap(), 0);
    }

    // Rank-1: identical to row-major (only one dimension).
    #[test]
    fn cm_rank1() {
        let shape = Shape::new(vec![10]).unwrap();
        assert_eq!(element_offset(&[7], &shape).unwrap(), 7);
    }

    // shape [2,3,4], element [1,2,3]:
    // offset = 1*1 + 2*2 + 3*(2*3) = 1 + 4 + 18 = 23
    #[test]
    fn cm_rank3() {
        let shape = Shape::new(vec![2, 3, 4]).unwrap();
        assert_eq!(element_offset(&[1, 2, 3], &shape).unwrap(), 23);
    }

    // First element: always 0.
    #[test]
    fn cm_first_element() {
        let shape = Shape::new(vec![3, 4]).unwrap();
        assert_eq!(element_offset(&[0, 0], &shape).unwrap(), 0);
    }

    // Last element of [3,4]: [2,3] → 2*1 + 3*3 = 11.
    #[test]
    fn cm_last_element() {
        let shape = Shape::new(vec![3, 4]).unwrap();
        assert_eq!(element_offset(&[2, 3], &shape).unwrap(), 11);
    }

    // Error: rank mismatch.
    #[test]
    fn cm_rank_mismatch() {
        let shape = Shape::new(vec![3, 4]).unwrap();
        let err = element_offset(&[1, 2, 0], &shape).unwrap_err();
        assert!(
            matches!(
                err,
                Error::IndexRankMismatch {
                    index_rank: 3,
                    shape_rank: 2
                }
            ),
            "expected IndexRankMismatch, got {err:?}"
        );
    }

    // Error: index out of range.
    #[test]
    fn cm_index_out_of_range() {
        let shape = Shape::new(vec![3, 4]).unwrap();
        // index[0]=3 >= dim[0]=3.
        let err = element_offset(&[3, 0], &shape).unwrap_err();
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
    fn cm_dynamic_dim() {
        let shape = Shape::new(vec![DYNAMIC, 4]).unwrap();
        let err = element_offset(&[1, 2], &shape).unwrap_err();
        assert!(
            matches!(err, Error::DynamicDimInIndexing { dim: 0 }),
            "expected DynamicDimInIndexing, got {err:?}"
        );
    }
}
