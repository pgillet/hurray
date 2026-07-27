//! CSR (Compressed Sparse Row) sparse layout element lookup.
//!
//! Implements element lookup from `docs/spec/layouts/csr.md`: within a row, the stored
//! column indices are strictly increasing, so a non-zero is found by binary-searching the
//! row's slice of `col_indices`. Mirrors the standalone-function API validated by
//! [`super::csf::element_offset`].

use crate::{Error, Result};

/// Looks up the storage offset (`values` / `col_indices` buffer index) of logical index
/// `(row, col)` in a CSR matrix, or returns `None` if the element is a structural zero.
///
/// # Arguments
///
/// - `query` — logical index `[row, col]`; MUST have length 2 (CSR is rank-2).
/// - `col_indices` — the `col_indices` buffer (buffer 1) as `uint64`: the column of each
///   non-zero, in row-major storage order (`nnz` entries).
/// - `row_ptr` — the `row_ptr` buffer (buffer 2) as `uint64`: `nrows + 1` entries, where
///   `row_ptr[i]` is the first storage index of row `i` and `row_ptr[nrows] = nnz`.
///
/// # Errors
///
/// - [`Error::IndexRankMismatch`] — `query.len() != 2`.
/// - [`Error::IndexOutOfRange`] — `row` is not a valid row (`row >= nrows`).
/// - [`Error::InvalidLayout`] — `row_ptr` is empty, or a `row_ptr` entry points outside
///   `col_indices` / is non-monotone for the queried row.
///
/// The caller is responsible for validating `col` against `shape[1]`; an out-of-range
/// column simply reports a structural zero (`None`).
///
/// # Examples
///
/// ```
/// use hurray_core::layout::addressing::csr::element_offset;
///
/// // 3×3 matrix:
/// //   row 0: (0,0), (0,2)
/// //   row 1: —
/// //   row 2: (2,1)
/// let col_indices: &[u64] = &[0, 2, 1];
/// let row_ptr: &[u64] = &[0, 2, 2, 3];
///
/// assert_eq!(element_offset(&[0, 2], col_indices, row_ptr).unwrap(), Some(1));
/// assert_eq!(element_offset(&[2, 1], col_indices, row_ptr).unwrap(), Some(2));
/// assert_eq!(element_offset(&[1, 0], col_indices, row_ptr).unwrap(), None); // empty row
/// ```
pub fn element_offset(query: &[u64], col_indices: &[u64], row_ptr: &[u64]) -> Result<Option<u64>> {
    if query.len() != 2 {
        return Err(Error::IndexRankMismatch {
            index_rank: query.len(),
            shape_rank: 2,
        });
    }
    let (row, col) = (query[0], query[1]);

    if row_ptr.is_empty() {
        return Err(Error::InvalidLayout(
            "csr: row_ptr must have at least nrows+1 = 1 entries".into(),
        ));
    }
    let nrows = (row_ptr.len() - 1) as u64;
    if row >= nrows {
        return Err(Error::IndexOutOfRange {
            dim: 0,
            index: row,
            size: nrows,
        });
    }

    let start = row_ptr[row as usize] as usize;
    let end = row_ptr[row as usize + 1] as usize;
    if start > end || end > col_indices.len() {
        return Err(Error::InvalidLayout(format!(
            "csr: row_ptr[{row}..{}] = [{start}..{end}] out of bounds for col_indices.len()={}",
            row + 1,
            col_indices.len()
        )));
    }

    match col_indices[start..end].binary_search(&col) {
        Ok(k) => Ok(Some((start + k) as u64)),
        Err(_) => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 4×4 matrix:
    //   row 0: (0,0)=a, (0,3)=b
    //   row 1: (1,1)=c
    //   row 2: —
    //   row 3: (3,0)=d, (3,2)=e, (3,3)=f
    // values order: a,b,c,d,e,f
    const COL_INDICES: &[u64] = &[0, 3, 1, 0, 2, 3];
    const ROW_PTR: &[u64] = &[0, 2, 3, 3, 6];

    #[test]
    fn lookup_hits() {
        assert_eq!(
            element_offset(&[0, 0], COL_INDICES, ROW_PTR).unwrap(),
            Some(0)
        );
        assert_eq!(
            element_offset(&[0, 3], COL_INDICES, ROW_PTR).unwrap(),
            Some(1)
        );
        assert_eq!(
            element_offset(&[1, 1], COL_INDICES, ROW_PTR).unwrap(),
            Some(2)
        );
        assert_eq!(
            element_offset(&[3, 0], COL_INDICES, ROW_PTR).unwrap(),
            Some(3)
        );
        assert_eq!(
            element_offset(&[3, 2], COL_INDICES, ROW_PTR).unwrap(),
            Some(4)
        );
        assert_eq!(
            element_offset(&[3, 3], COL_INDICES, ROW_PTR).unwrap(),
            Some(5)
        );
    }

    #[test]
    fn structural_zeros() {
        assert_eq!(element_offset(&[0, 1], COL_INDICES, ROW_PTR).unwrap(), None);
        assert_eq!(element_offset(&[1, 0], COL_INDICES, ROW_PTR).unwrap(), None);
        assert_eq!(element_offset(&[2, 2], COL_INDICES, ROW_PTR).unwrap(), None); // empty row
        assert_eq!(element_offset(&[3, 1], COL_INDICES, ROW_PTR).unwrap(), None);
    }

    #[test]
    fn wrong_rank_rejected() {
        assert!(matches!(
            element_offset(&[0], COL_INDICES, ROW_PTR),
            Err(Error::IndexRankMismatch { .. })
        ));
        assert!(matches!(
            element_offset(&[0, 0, 0], COL_INDICES, ROW_PTR),
            Err(Error::IndexRankMismatch { .. })
        ));
    }

    #[test]
    fn row_out_of_range_rejected() {
        assert!(matches!(
            element_offset(&[4, 0], COL_INDICES, ROW_PTR),
            Err(Error::IndexOutOfRange { dim: 0, .. })
        ));
    }

    #[test]
    fn empty_row_ptr_rejected() {
        assert!(matches!(
            element_offset(&[0, 0], COL_INDICES, &[]),
            Err(Error::InvalidLayout(_))
        ));
    }

    #[test]
    fn empty_matrix() {
        // 2×2 all-zero: row_ptr = [0,0,0], no stored columns.
        assert_eq!(element_offset(&[0, 0], &[], &[0, 0, 0]).unwrap(), None);
        assert_eq!(element_offset(&[1, 1], &[], &[0, 0, 0]).unwrap(), None);
    }
}
