//! COO (Coordinate) sparse layout element lookup.
//!
//! Implements element lookup from `docs/spec/layouts/coo.md § Storage Order`:
//! locate the stored non-zero whose coordinate tuple equals the query, using a
//! lexicographic binary search when the entries are sorted, or a linear scan
//! otherwise. Mirrors the standalone-function API validated by
//! [`super::csf::element_offset`].

use std::cmp::Ordering;

use crate::{Error, Result};

/// Looks up the storage offset (`values` buffer index) of a logical index in a COO
/// tensor, or returns `None` if the element is structurally absent (implicit zero).
///
/// # Arguments
///
/// - `query` — logical index `[i0, …, i_{rank-1}]`; its length defines the rank.
/// - `is_sorted` — the COO descriptor's `is_sorted` flag. When `true`, the stored
///   entries are in strictly increasing lexicographic order and a binary search is used;
///   when `false`, a linear scan is used.
/// - `indices` — the `indices` buffer (buffer 1) as `uint64` values: `nnz × rank`
///   coordinates in row-major order, so entry `r`'s coordinates are
///   `indices[r*rank .. r*rank + rank]`.
///
/// The returned offset indexes both the `values` buffer and the entry's row in `indices`.
///
/// # Errors
///
/// - [`Error::IndexRankMismatch`] — `query` is empty (rank 0).
/// - [`Error::InvalidLayout`] — `indices.len()` is not a multiple of the rank.
///
/// The caller is responsible for validating each query coordinate against the tensor
/// shape; an out-of-bounds coordinate simply reports a structural zero (`None`).
///
/// # Examples
///
/// ```
/// use hurray_core::layout::addressing::coo::element_offset;
///
/// // 4×4 matrix with three sorted non-zeros: (0,1), (2,0), (2,3).
/// let indices: &[u64] = &[0, 1, /* */ 2, 0, /* */ 2, 3];
///
/// assert_eq!(element_offset(&[2, 0], true, indices).unwrap(), Some(1));
/// assert_eq!(element_offset(&[2, 3], true, indices).unwrap(), Some(2));
/// assert_eq!(element_offset(&[1, 1], true, indices).unwrap(), None); // structural zero
/// ```
pub fn element_offset(query: &[u64], is_sorted: bool, indices: &[u64]) -> Result<Option<u64>> {
    let rank = query.len();
    if rank == 0 {
        return Err(Error::IndexRankMismatch {
            index_rank: 0,
            shape_rank: 0,
        });
    }
    if !indices.len().is_multiple_of(rank) {
        return Err(Error::InvalidLayout(format!(
            "coo: indices.len()={} is not a multiple of rank={rank}",
            indices.len()
        )));
    }
    let nnz = indices.len() / rank;
    let coords = |r: usize| &indices[r * rank..r * rank + rank];

    if is_sorted {
        // Lexicographic binary search: slice `Ord` compares element-wise, which is exactly
        // the spec's dimension-0-major lexicographic order.
        let (mut lo, mut hi) = (0usize, nnz);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match coords(mid).cmp(query) {
                Ordering::Less => lo = mid + 1,
                Ordering::Greater => hi = mid,
                Ordering::Equal => return Ok(Some(mid as u64)),
            }
        }
        Ok(None)
    } else {
        for r in 0..nnz {
            if coords(r) == query {
                return Ok(Some(r as u64));
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 4×4 matrix, sorted non-zeros at (0,1), (2,0), (2,3), (3,3).
    const INDICES: &[u64] = &[0, 1, 2, 0, 2, 3, 3, 3];

    #[test]
    fn sorted_lookup_hits() {
        assert_eq!(element_offset(&[0, 1], true, INDICES).unwrap(), Some(0));
        assert_eq!(element_offset(&[2, 0], true, INDICES).unwrap(), Some(1));
        assert_eq!(element_offset(&[2, 3], true, INDICES).unwrap(), Some(2));
        assert_eq!(element_offset(&[3, 3], true, INDICES).unwrap(), Some(3));
    }

    #[test]
    fn sorted_lookup_structural_zeros() {
        assert_eq!(element_offset(&[0, 0], true, INDICES).unwrap(), None);
        assert_eq!(element_offset(&[1, 1], true, INDICES).unwrap(), None);
        assert_eq!(element_offset(&[2, 2], true, INDICES).unwrap(), None);
        // Between stored coordinates of row 3.
        assert_eq!(element_offset(&[3, 0], true, INDICES).unwrap(), None);
    }

    #[test]
    fn unsorted_lookup_matches_sorted() {
        // Same non-zeros, permuted; linear scan must still find them.
        let unsorted: &[u64] = &[2, 3, 0, 1, 3, 3, 2, 0];
        assert_eq!(element_offset(&[0, 1], false, unsorted).unwrap(), Some(1));
        assert_eq!(element_offset(&[2, 3], false, unsorted).unwrap(), Some(0));
        assert_eq!(element_offset(&[3, 3], false, unsorted).unwrap(), Some(2));
        assert_eq!(element_offset(&[1, 1], false, unsorted).unwrap(), None);
    }

    #[test]
    fn rank_3_lookup() {
        // Shape [2,3,4]; non-zeros (0,0,1), (0,2,3), (1,1,0), sorted.
        let indices: &[u64] = &[0, 0, 1, 0, 2, 3, 1, 1, 0];
        assert_eq!(element_offset(&[0, 2, 3], true, indices).unwrap(), Some(1));
        assert_eq!(element_offset(&[1, 1, 0], true, indices).unwrap(), Some(2));
        assert_eq!(element_offset(&[1, 1, 2], true, indices).unwrap(), None);
    }

    #[test]
    fn empty_tensor_is_all_zeros() {
        assert_eq!(element_offset(&[0, 0], true, &[]).unwrap(), None);
        assert_eq!(element_offset(&[0, 0], false, &[]).unwrap(), None);
    }

    #[test]
    fn rank_zero_query_is_rejected() {
        assert!(matches!(
            element_offset(&[], true, &[]),
            Err(Error::IndexRankMismatch { .. })
        ));
    }

    #[test]
    fn indices_not_multiple_of_rank_is_rejected() {
        // rank 2, but 3 index values.
        assert!(matches!(
            element_offset(&[0, 0], true, &[0, 1, 2]),
            Err(Error::InvalidLayout(_))
        ));
    }
}
