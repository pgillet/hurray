//! CSC (Compressed Sparse Column) sparse layout element lookup.
//!
//! The column analog of CSR (`docs/spec/layouts/csc.md`): within a column, the stored row
//! indices are strictly increasing, so a non-zero is found by binary-searching the
//! column's slice of `row_indices`. Mirrors the standalone-function API validated by
//! [`super::csf::element_offset`].

use crate::{Error, Result};

/// Looks up the storage offset (`values` / `row_indices` buffer index) of logical index
/// `(row, col)` in a CSC matrix, or returns `None` if the element is a structural zero.
///
/// # Arguments
///
/// - `query` — logical index `[row, col]`; MUST have length 2 (CSC is rank-2).
/// - `row_indices` — the `row_indices` buffer (buffer 1) as `uint64`: the row of each
///   non-zero, in column-major storage order (`nnz` entries).
/// - `col_ptr` — the `col_ptr` buffer (buffer 2) as `uint64`: `ncols + 1` entries, where
///   `col_ptr[j]` is the first storage index of column `j` and `col_ptr[ncols] = nnz`.
///
/// # Errors
///
/// - [`Error::IndexRankMismatch`] — `query.len() != 2`.
/// - [`Error::IndexOutOfRange`] — `col` is not a valid column (`col >= ncols`).
/// - [`Error::InvalidLayout`] — `col_ptr` is empty, or a `col_ptr` entry points outside
///   `row_indices` / is non-monotone for the queried column.
///
/// The caller is responsible for validating `row` against `shape[0]`; an out-of-range row
/// simply reports a structural zero (`None`).
///
/// # Examples
///
/// ```
/// use hurray_core::layout::addressing::csc::element_offset;
///
/// // 3×3 matrix, stored column-major:
/// //   col 0: (0,0)
/// //   col 1: (2,1)
/// //   col 2: (0,2)
/// let row_indices: &[u64] = &[0, 2, 0];
/// let col_ptr: &[u64] = &[0, 1, 2, 3];
///
/// assert_eq!(element_offset(&[2, 1], row_indices, col_ptr).unwrap(), Some(1));
/// assert_eq!(element_offset(&[0, 2], row_indices, col_ptr).unwrap(), Some(2));
/// assert_eq!(element_offset(&[1, 1], row_indices, col_ptr).unwrap(), None); // structural zero
/// ```
pub fn element_offset(query: &[u64], row_indices: &[u64], col_ptr: &[u64]) -> Result<Option<u64>> {
    if query.len() != 2 {
        return Err(Error::IndexRankMismatch {
            index_rank: query.len(),
            shape_rank: 2,
        });
    }
    let (row, col) = (query[0], query[1]);

    if col_ptr.is_empty() {
        return Err(Error::InvalidLayout(
            "csc: col_ptr must have at least ncols+1 = 1 entries".into(),
        ));
    }
    let ncols = (col_ptr.len() - 1) as u64;
    if col >= ncols {
        return Err(Error::IndexOutOfRange {
            dim: 1,
            index: col,
            size: ncols,
        });
    }

    let start = col_ptr[col as usize] as usize;
    let end = col_ptr[col as usize + 1] as usize;
    if start > end || end > row_indices.len() {
        return Err(Error::InvalidLayout(format!(
            "csc: col_ptr[{col}..{}] = [{start}..{end}] out of bounds for row_indices.len()={}",
            col + 1,
            row_indices.len()
        )));
    }

    match row_indices[start..end].binary_search(&row) {
        Ok(k) => Ok(Some((start + k) as u64)),
        Err(_) => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 4×4 matrix, stored column-major:
    //   col 0: (0,0)=a, (3,0)=b
    //   col 1: (1,1)=c
    //   col 2: (3,2)=d
    //   col 3: (0,3)=e, (3,3)=f
    // values order: a,b,c,d,e,f
    const ROW_INDICES: &[u64] = &[0, 3, 1, 3, 0, 3];
    const COL_PTR: &[u64] = &[0, 2, 3, 4, 6];

    #[test]
    fn lookup_hits() {
        assert_eq!(
            element_offset(&[0, 0], ROW_INDICES, COL_PTR).unwrap(),
            Some(0)
        );
        assert_eq!(
            element_offset(&[3, 0], ROW_INDICES, COL_PTR).unwrap(),
            Some(1)
        );
        assert_eq!(
            element_offset(&[1, 1], ROW_INDICES, COL_PTR).unwrap(),
            Some(2)
        );
        assert_eq!(
            element_offset(&[3, 2], ROW_INDICES, COL_PTR).unwrap(),
            Some(3)
        );
        assert_eq!(
            element_offset(&[0, 3], ROW_INDICES, COL_PTR).unwrap(),
            Some(4)
        );
        assert_eq!(
            element_offset(&[3, 3], ROW_INDICES, COL_PTR).unwrap(),
            Some(5)
        );
    }

    #[test]
    fn structural_zeros() {
        assert_eq!(element_offset(&[1, 0], ROW_INDICES, COL_PTR).unwrap(), None);
        assert_eq!(element_offset(&[0, 1], ROW_INDICES, COL_PTR).unwrap(), None);
        assert_eq!(element_offset(&[0, 2], ROW_INDICES, COL_PTR).unwrap(), None); // col 2 has only row 3
        assert_eq!(element_offset(&[2, 3], ROW_INDICES, COL_PTR).unwrap(), None);
    }

    #[test]
    fn wrong_rank_rejected() {
        assert!(matches!(
            element_offset(&[0], ROW_INDICES, COL_PTR),
            Err(Error::IndexRankMismatch { .. })
        ));
    }

    #[test]
    fn col_out_of_range_rejected() {
        assert!(matches!(
            element_offset(&[0, 4], ROW_INDICES, COL_PTR),
            Err(Error::IndexOutOfRange { dim: 1, .. })
        ));
    }

    #[test]
    fn empty_col_ptr_rejected() {
        assert!(matches!(
            element_offset(&[0, 0], ROW_INDICES, &[]),
            Err(Error::InvalidLayout(_))
        ));
    }

    #[test]
    fn empty_matrix() {
        assert_eq!(element_offset(&[0, 0], &[], &[0, 0, 0]).unwrap(), None);
    }
}
