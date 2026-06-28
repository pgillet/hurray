//! CSF (Compressed Sparse Fiber) element lookup and storage-invariant validation.
//!
//! Implements the per-level binary-search descent from
//! `docs/spec/layouts/csf.md § Element Lookup` and the invariant checks from
//! `docs/spec/layouts/csf.md § Storage Invariants`.
//!
//! All arithmetic uses `checked_*` to avoid silent overflow — this code can run
//! on arbitrarily large tensors.

use crate::Error;

/// Looks up the storage offset (`values` buffer index) for a logical index in a
/// CSF tensor, or returns `None` if the element is structurally absent (implicit zero).
///
/// # Arguments
///
/// - `query_coords` — logical index `[idx[0], ..., idx[rank-1]]`; length must equal `rank`.
/// - `mode_order` — the level-to-dimension permutation from the CSF descriptor.
/// - `pos_levels` — slice of `rank` pos arrays; `pos_levels[L]` is the `pos_L` buffer
///   as a slice of `uint64` values.
/// - `crd_levels` — slice of `rank` crd arrays; `crd_levels[L]` is the `crd_L` buffer
///   as a slice of `uint64` values.
///
/// # Return value
///
/// Returns `Ok(Some(p))` where `p` is the leaf position in `values`, or `Ok(None)` if
/// the element is not stored (structural zero). Returns `Err` only on structural
/// violations (malformed buffers that make the search impossible without panicking),
/// not on valid structural-zero hits.
///
/// # Errors
///
/// - [`Error::IndexRankMismatch`] — `query_coords.len() != mode_order.len()`.
/// - [`Error::IndexOutOfRange`] — any query coordinate is out of bounds for its mode.
///   (Caller must validate coordinates against the shape before calling.)
/// - [`Error::InvalidLayout`] — a pos/crd buffer is so malformed that a safe traversal
///   is impossible (e.g. `pos[p] > crd.len()`).
/// - [`Error::AddressOverflow`] — intermediate index arithmetic overflowed `u64`.
///
/// # Examples
///
/// ```
/// use hurray_core::layout::addressing::csf::element_offset;
///
/// // Rank-3 sparse tensor with shape [2, 3, 4], mode_order = [0, 1, 2], nnz = 4.
/// // Non-zeros: (0,0,1)→1.0, (0,2,3)→2.0, (1,1,0)→3.0, (1,1,2)→4.0
/// //
/// // Level 0:  pos_0 = [0, 2],     crd_0 = [0, 1]
/// // Level 1:  pos_1 = [0, 2, 3],  crd_1 = [0, 2, 1]
/// // Level 2:  pos_2 = [0, 1, 2, 4], crd_2 = [1, 3, 0, 2]
///
/// let mode_order: &[u32] = &[0, 1, 2];
/// let pos_levels: &[&[u64]] = &[&[0, 2], &[0, 2, 3], &[0, 1, 2, 4]];
/// let crd_levels: &[&[u64]] = &[&[0, 1], &[0, 2, 1], &[1, 3, 0, 2]];
///
/// // Lookup (1, 1, 2) → leaf position 3 (values[3] = 4.0).
/// assert_eq!(
///     element_offset(&[1, 1, 2], mode_order, pos_levels, crd_levels).unwrap(),
///     Some(3)
/// );
///
/// // Lookup (0, 1, 0) → not stored (structural zero).
/// assert_eq!(
///     element_offset(&[0, 1, 0], mode_order, pos_levels, crd_levels).unwrap(),
///     None
/// );
/// ```
pub fn element_offset(
    query_coords: &[u64],
    mode_order: &[u32],
    pos_levels: &[&[u64]],
    crd_levels: &[&[u64]],
) -> crate::Result<Option<u64>> {
    let rank = mode_order.len();

    if query_coords.len() != rank {
        return Err(Error::IndexRankMismatch {
            index_rank: query_coords.len(),
            shape_rank: rank,
        });
    }

    // Spec §Element Lookup step 1: permute the query into storage order.
    // q_L = query_coords[mode_order[L]] — the coordinate sought at level L.
    // We do this inline below rather than pre-allocating a Vec, because this
    // is potentially called in tight inner loops (structural zero hits are cheap).

    // Spec §Element Lookup step 2: initialise parent position p = 0 (virtual root).
    let mut p: u64 = 0;

    // Spec §Element Lookup step 3: descend level by level.
    for level in 0..rank {
        let dim_idx = mode_order[level] as usize;
        // Safety-guard: mode_order is validated by validate_against_shape before this
        // function is typically called, but we return a crate error rather than panic.
        if dim_idx >= query_coords.len() {
            return Err(Error::InvalidLayout(format!(
                "csf: mode_order[{level}]={dim_idx} out of range for rank {rank}"
            )));
        }
        let q_l: u64 = query_coords[dim_idx];

        let pos = pos_levels[level];
        let crd = crd_levels[level];

        // The children of the current parent occupy crd_L[pos_L[p]..pos_L[p+1]).
        let p_usize = p as usize;

        // Bounds-check: pos must have at least p+2 entries.
        if p_usize + 1 >= pos.len() {
            return Err(Error::InvalidLayout(format!(
                "csf: pos_{level}[{p}..{}] out of bounds (pos_{level}.len()={})",
                p_usize + 1,
                pos.len()
            )));
        }

        let slice_start = pos[p_usize] as usize;
        let slice_end = pos[p_usize + 1] as usize;

        // Guard against a malformed (non-monotone) pos array.
        if slice_start > slice_end {
            return Err(Error::InvalidLayout(format!(
                "csf: pos_{level}[{p}]={slice_start} > pos_{level}[{}]={slice_end} (not non-decreasing)",
                p_usize + 1
            )));
        }

        // Guard against pos entries that exceed the crd array length.
        if slice_end > crd.len() {
            return Err(Error::InvalidLayout(format!(
                "csf: pos_{level}[{}]={slice_end} exceeds crd_{level}.len()={}",
                p_usize + 1,
                crd.len()
            )));
        }

        let slice = &crd[slice_start..slice_end];

        // Spec §Element Lookup: binary search on the sorted sibling slice.
        match slice.binary_search(&q_l) {
            Ok(relative_offset) => {
                // Found at relative offset k; new absolute parent position = pos_L[p] + k.
                let abs = pos[p_usize]
                    .checked_add(relative_offset as u64)
                    .ok_or(Error::AddressOverflow)?;
                p = abs;
                // Continue to the next level.
            }
            Err(_) => {
                // Not found — the element is an implicit zero.
                return Ok(None);
            }
        }
    }

    // After rank levels, p is the leaf position in values[].
    Ok(Some(p))
}

/// Validates the storage invariants for all CSF index buffers.
///
/// Per `docs/spec/layouts/csf.md § Storage Invariants`, for every level `L`:
///
/// 1. `pos_L[0] == 0`.
/// 2. `pos_L` is non-decreasing.
/// 3. The terminal `pos_L` entry equals the next level's node count:
///    `pos_0[1] == n_0`, `pos_L[n_{L-1}] == n_L`, and `n_{rank-1} == nnz`.
/// 4. Within each parent slice `crd_L[pos_L[k]..pos_L[k+1])`, coordinates are
///    strictly increasing (no duplicate siblings).
/// 5. All coordinates are within bounds: `crd_L[i] < shape[mode_order[L]]`.
///
/// For the empty tensor (`nnz == 0`):
/// - `pos_0` MUST be `[0, 0]` (length 2).
/// - Every `crd_L` MUST be empty (length 0).
/// - For `L >= 1`, `pos_L` MUST be `[0]` (length 1).
///
/// Index buffers are `uint64` throughout to match CSR/COO/CSC.
///
/// # Arguments
///
/// - `nnz` — declared number of non-zeros (from the CSF descriptor).
/// - `mode_order` — the level-to-dimension permutation.
/// - `shape_dims` — the tensor's dimension sizes; `shape_dims[mode_order[L]]` is the
///   coordinate bound for level `L`.
/// - `pos_levels` — `pos_L` buffers as typed `u64` slices, one per level.
/// - `crd_levels` — `crd_L` buffers as typed `u64` slices, one per level.
///
/// # Errors
///
/// Returns [`Error::InvalidLayout`] if any invariant is violated.
///
/// # Examples
///
/// ```
/// use hurray_core::layout::addressing::csf::validate_index_buffers;
///
/// // The spec § Example: shape [2,3,4], mode_order=[0,1,2], nnz=4.
/// let mode_order: &[u32] = &[0, 1, 2];
/// let shape_dims: &[u64] = &[2, 3, 4];
/// let pos_levels: &[&[u64]] = &[&[0, 2], &[0, 2, 3], &[0, 1, 2, 4]];
/// let crd_levels: &[&[u64]] = &[&[0, 1], &[0, 2, 1], &[1, 3, 0, 2]];
///
/// assert!(validate_index_buffers(4, mode_order, shape_dims, pos_levels, crd_levels).is_ok());
///
/// // Empty tensor: nnz=0.
/// let pos_levels_empty: &[&[u64]] = &[&[0, 0], &[0], &[0]];
/// let crd_levels_empty: &[&[u64]] = &[&[], &[], &[]];
/// assert!(validate_index_buffers(0, mode_order, shape_dims, pos_levels_empty, crd_levels_empty).is_ok());
/// ```
pub fn validate_index_buffers(
    nnz: u64,
    mode_order: &[u32],
    shape_dims: &[u64],
    pos_levels: &[&[u64]],
    crd_levels: &[&[u64]],
) -> crate::Result<()> {
    let rank = mode_order.len();

    if pos_levels.len() != rank {
        return Err(Error::InvalidLayout(format!(
            "csf: pos_levels.len()={} != rank={}",
            pos_levels.len(),
            rank
        )));
    }
    if crd_levels.len() != rank {
        return Err(Error::InvalidLayout(format!(
            "csf: crd_levels.len()={} != rank={}",
            crd_levels.len(),
            rank
        )));
    }

    // n_L: node count at each level. n_{-1} = 1 (virtual root).
    let mut n_prev: u64 = 1; // n_{L-1}, starting as virtual-root count

    for level in 0..rank {
        let pos = pos_levels[level];
        let crd = crd_levels[level];
        let dim_idx = mode_order[level] as usize;

        // Coordinate bound for this level.
        let coord_bound = if dim_idx < shape_dims.len() {
            shape_dims[dim_idx]
        } else {
            return Err(Error::InvalidLayout(format!(
                "csf: mode_order[{level}]={dim_idx} out of range for shape.len()={}",
                shape_dims.len()
            )));
        };

        // Expected pos length: n_{L-1} + 1.
        // For nnz=0: level 0 pos must be [0,0] (length 2, n_{-1}=1 → 1+1=2).
        // For nnz=0, level ≥ 1: n_{L-1}=0, so pos must be length 1 → [0].
        let expected_pos_len = (n_prev as usize).saturating_add(1);
        if pos.len() != expected_pos_len {
            return Err(Error::InvalidLayout(format!(
                "csf: pos_{level}.len()={} != n_{{L-1}}+1={} (n_{{L-1}}={n_prev})",
                pos.len(),
                expected_pos_len
            )));
        }

        // Invariant 1: pos_L[0] == 0.
        if pos[0] != 0 {
            return Err(Error::InvalidLayout(format!(
                "csf: pos_{level}[0]={} must be 0",
                pos[0]
            )));
        }

        // Validate pos is non-decreasing (invariant 2) and collect n_L = pos[n_{L-1}].
        for k in 0..(pos.len() - 1) {
            if pos[k] > pos[k + 1] {
                return Err(Error::InvalidLayout(format!(
                    "csf: pos_{level}[{k}]={} > pos_{level}[{}]={} (not non-decreasing)",
                    pos[k],
                    k + 1,
                    pos[k + 1]
                )));
            }
        }

        // Invariant 3: terminal pos entry equals this level's node count (n_L).
        // For non-leaf levels n_L is enforced implicitly — it sizes `crd` (the
        // crd.len() == n_L check below) and becomes n_{L-1} for the next level's
        // pos-length check. Only the leaf level (n_{rank-1} == nnz) is checked here.
        let n_l = pos[n_prev as usize]; // pos[n_{L-1}]
        if level == rank - 1 && n_l != nnz {
            return Err(Error::InvalidLayout(format!(
                "csf: pos_{level}[n_{{rank-1}}]={n_l} must equal nnz={nnz}"
            )));
        }

        // crd length must equal n_L.
        if crd.len() as u64 != n_l {
            return Err(Error::InvalidLayout(format!(
                "csf: crd_{level}.len()={} != n_{level}={n_l}",
                crd.len()
            )));
        }

        // Invariant 4: strictly increasing within each parent slice.
        // Invariant 5: coordinates < coord_bound.
        for k in 0..(n_prev as usize) {
            let start = pos[k] as usize;
            let end = pos[k + 1] as usize;
            let sibling_slice = &crd[start..end];

            // Check bounds and strict monotonicity together in one pass.
            let mut prev_coord: Option<u64> = None;
            for (j, &coord) in sibling_slice.iter().enumerate() {
                if coord >= coord_bound {
                    return Err(Error::InvalidLayout(format!(
                        "csf: crd_{level}[{}]={coord} >= shape[mode_order[{level}]]={coord_bound}",
                        start + j
                    )));
                }
                if let Some(prev) = prev_coord {
                    if coord <= prev {
                        return Err(Error::InvalidLayout(format!(
                            "csf: crd_{level}[{start}..{end}] is not strictly increasing at index {} ({prev} >= {coord})",
                            start + j
                        )));
                    }
                }
                prev_coord = Some(coord);
            }
        }

        // Advance n_prev for the next level.
        n_prev = n_l;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Spec example from csf.md § Example ───────────────────────────────────
    //
    // Rank-3 shape [2,3,4], mode_order=[0,1,2], nnz=4:
    // Non-zeros: (0,0,1)→1.0, (0,2,3)→2.0, (1,1,0)→3.0, (1,1,2)→4.0
    //
    // pos_0 = [0, 2]        crd_0 = [0, 1]
    // pos_1 = [0, 2, 3]     crd_1 = [0, 2, 1]
    // pos_2 = [0, 1, 2, 4]  crd_2 = [1, 3, 0, 2]

    const MODE_ORDER: &[u32] = &[0, 1, 2];
    const SHAPE_DIMS: &[u64] = &[2, 3, 4];
    const POS_0: &[u64] = &[0, 2];
    const CRD_0: &[u64] = &[0, 1];
    const POS_1: &[u64] = &[0, 2, 3];
    const CRD_1: &[u64] = &[0, 2, 1];
    const POS_2: &[u64] = &[0, 1, 2, 4];
    const CRD_2: &[u64] = &[1, 3, 0, 2];

    fn spec_pos_levels() -> Vec<&'static [u64]> {
        vec![POS_0, POS_1, POS_2]
    }
    fn spec_crd_levels() -> Vec<&'static [u64]> {
        vec![CRD_0, CRD_1, CRD_2]
    }

    // ── element_offset: spec worked example ───────────────────────────────────

    /// Spec §Element Lookup: (1, 1, 2) → leaf position 3.
    #[test]
    fn element_offset_spec_lookup_1_1_2() {
        let result = element_offset(
            &[1, 1, 2],
            MODE_ORDER,
            &spec_pos_levels(),
            &spec_crd_levels(),
        )
        .unwrap();
        assert_eq!(result, Some(3));
    }

    /// (0, 0, 1) → leaf position 0.
    #[test]
    fn element_offset_spec_lookup_0_0_1() {
        let result = element_offset(
            &[0, 0, 1],
            MODE_ORDER,
            &spec_pos_levels(),
            &spec_crd_levels(),
        )
        .unwrap();
        assert_eq!(result, Some(0));
    }

    /// (0, 2, 3) → leaf position 1.
    #[test]
    fn element_offset_spec_lookup_0_2_3() {
        let result = element_offset(
            &[0, 2, 3],
            MODE_ORDER,
            &spec_pos_levels(),
            &spec_crd_levels(),
        )
        .unwrap();
        assert_eq!(result, Some(1));
    }

    /// (1, 1, 0) → leaf position 2.
    #[test]
    fn element_offset_spec_lookup_1_1_0() {
        let result = element_offset(
            &[1, 1, 0],
            MODE_ORDER,
            &spec_pos_levels(),
            &spec_crd_levels(),
        )
        .unwrap();
        assert_eq!(result, Some(2));
    }

    /// (0, 1, 0) is not stored → structural zero.
    #[test]
    fn element_offset_structural_zero_returns_none() {
        let result = element_offset(
            &[0, 1, 0],
            MODE_ORDER,
            &spec_pos_levels(),
            &spec_crd_levels(),
        )
        .unwrap();
        assert_eq!(result, None);
    }

    /// (1, 0, 0) is not stored (i=1 child list has only dim-1 coordinate 1).
    #[test]
    fn element_offset_structural_zero_at_level_1() {
        // Mode 0: i=1 found at level 0 (crd_0[1]=1).
        // Mode 1: parent p=1, crd_1[pos_1[1]..pos_1[2])=crd_1[2..3)=[1]; search for 0 → not found.
        let result = element_offset(
            &[1, 0, 0],
            MODE_ORDER,
            &spec_pos_levels(),
            &spec_crd_levels(),
        )
        .unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn element_offset_structural_zero_at_level_0() {
        // i0=2 is not stored at the root: crd_0=[0,1] has no entry 2 → miss at the
        // very first level → implicit zero, before descending any further.
        let result = element_offset(
            &[2, 0, 0],
            MODE_ORDER,
            &spec_pos_levels(),
            &spec_crd_levels(),
        )
        .unwrap();
        assert_eq!(result, None);
    }

    /// Rank mismatch returns IndexRankMismatch.
    #[test]
    fn element_offset_rank_mismatch_returns_error() {
        let err = element_offset(
            &[0, 1], // rank 2, but mode_order is rank 3
            MODE_ORDER,
            &spec_pos_levels(),
            &spec_crd_levels(),
        )
        .unwrap_err();
        assert!(matches!(err, crate::Error::IndexRankMismatch { .. }));
    }

    // ── element_offset: non-identity mode_order ───────────────────────────────

    /// mode_order = [2, 1, 0]: column-first ordering.
    /// Build a rank-3 tree [2,3,4] with a single non-zero (0,0,1).
    /// With mode_order=[2,1,0], level 0 stores dim 2 (bound 4), etc.
    #[test]
    fn element_offset_non_identity_mode_order() {
        // Single non-zero at logical (0, 0, 1).
        // mode_order=[2,1,0]: level 0 → dim 2 (coord 1), level 1 → dim 1 (coord 0), level 2 → dim 0 (coord 0).
        let mode_order: &[u32] = &[2, 1, 0];
        // Level 0 (dim 2): pos=[0,1], crd=[1]
        // Level 1 (dim 1): pos=[0,1], crd=[0]
        // Level 2 (dim 0): pos=[0,1], crd=[0]
        let pos_levels: Vec<&[u64]> = vec![&[0, 1], &[0, 1], &[0, 1]];
        let crd_levels: Vec<&[u64]> = vec![&[1], &[0], &[0]];

        // Lookup (0, 0, 1): permuted to q[2]=1, q[1]=0, q[0]=0.
        let result = element_offset(&[0, 0, 1], mode_order, &pos_levels, &crd_levels).unwrap();
        assert_eq!(result, Some(0));

        // Lookup (0, 0, 0): q[2]=0 not in crd_0=[1] → structural zero.
        let result = element_offset(&[0, 0, 0], mode_order, &pos_levels, &crd_levels).unwrap();
        assert_eq!(result, None);
    }

    // ── validate_index_buffers: valid inputs ──────────────────────────────────

    /// Spec §Storage Invariants: the spec worked example satisfies all invariants.
    #[test]
    fn validate_index_buffers_spec_example_is_valid() {
        assert!(validate_index_buffers(
            4,
            MODE_ORDER,
            SHAPE_DIMS,
            &spec_pos_levels(),
            &spec_crd_levels(),
        )
        .is_ok());
    }

    /// Empty tensor: nnz=0, correct buffer shapes.
    #[test]
    fn validate_index_buffers_empty_tensor_is_valid() {
        // pos_0=[0,0], pos_1=[0], pos_2=[0]; crd_0=[], crd_1=[], crd_2=[].
        let pos_levels: Vec<&[u64]> = vec![&[0, 0], &[0], &[0]];
        let crd_levels: Vec<&[u64]> = vec![&[], &[], &[]];
        assert!(
            validate_index_buffers(0, MODE_ORDER, SHAPE_DIMS, &pos_levels, &crd_levels).is_ok()
        );
    }

    // ── validate_index_buffers: invariant 1 — pos_L[0] != 0 ─────────────────

    #[test]
    fn validate_rejects_pos_not_starting_at_zero() {
        let pos_levels: Vec<&[u64]> = vec![&[1, 2], &[0, 2, 3], &[0, 1, 2, 4]]; // pos_0[0]=1
        let result =
            validate_index_buffers(4, MODE_ORDER, SHAPE_DIMS, &pos_levels, &spec_crd_levels());
        assert!(matches!(result, Err(crate::Error::InvalidLayout(_))));
    }

    // ── validate_index_buffers: invariant 2 — non-decreasing ─────────────────

    #[test]
    fn validate_rejects_non_monotone_pos() {
        // pos_1 = [0, 3, 2] — not non-decreasing at index 1.
        let pos_levels: Vec<&[u64]> = vec![&[0, 2], &[0, 3, 2], &[0, 1, 2, 4]];
        let result =
            validate_index_buffers(4, MODE_ORDER, SHAPE_DIMS, &pos_levels, &spec_crd_levels());
        assert!(matches!(result, Err(crate::Error::InvalidLayout(_))));
    }

    // ── validate_index_buffers: invariant 3 — terminal count ─────────────────

    #[test]
    fn validate_rejects_terminal_pos_mismatch_nnz() {
        // pos_2 = [0, 1, 2, 3] → terminal = 3, but nnz=4.
        let pos_levels: Vec<&[u64]> = vec![&[0, 2], &[0, 2, 3], &[0, 1, 2, 3]];
        let result =
            validate_index_buffers(4, MODE_ORDER, SHAPE_DIMS, &pos_levels, &spec_crd_levels());
        assert!(matches!(result, Err(crate::Error::InvalidLayout(_))));
    }

    // ── validate_index_buffers: invariant 4 — strictly increasing ────────────

    #[test]
    fn validate_rejects_duplicate_siblings() {
        // crd_0 = [0, 0] — duplicate in the only parent slice.
        let crd_levels: Vec<&[u64]> = vec![&[0, 0], CRD_1, CRD_2];
        let result =
            validate_index_buffers(4, MODE_ORDER, SHAPE_DIMS, &spec_pos_levels(), &crd_levels);
        assert!(matches!(result, Err(crate::Error::InvalidLayout(_))));
    }

    #[test]
    fn validate_rejects_decreasing_siblings() {
        // crd_0 = [1, 0] — decreasing within the only parent slice.
        let crd_levels: Vec<&[u64]> = vec![&[1, 0], CRD_1, CRD_2];
        let result =
            validate_index_buffers(4, MODE_ORDER, SHAPE_DIMS, &spec_pos_levels(), &crd_levels);
        assert!(matches!(result, Err(crate::Error::InvalidLayout(_))));
    }

    // ── validate_index_buffers: invariant 5 — coordinate bounds ──────────────

    #[test]
    fn validate_rejects_out_of_bounds_coordinate() {
        // crd_2[3] = 4 >= shape[mode_order[2]]=shape[2]=4 → out of range.
        let crd_levels: Vec<&[u64]> = vec![CRD_0, CRD_1, &[1, 3, 0, 4]];
        let result =
            validate_index_buffers(4, MODE_ORDER, SHAPE_DIMS, &spec_pos_levels(), &crd_levels);
        assert!(matches!(result, Err(crate::Error::InvalidLayout(_))));
    }
}
