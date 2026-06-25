//! Block-paged layout element lookup.
//!
//! Implements the element-address formula from
//! `docs/spec/layouts/block-paged.md § Element Lookup`:
//!
//! ```text
//! page_in_seq    = t / page_size
//! offset_in_page = t mod page_size
//! phys_page      = block_table[seq_ptr[s] + page_in_seq]
//! flat           = ((phys_page * page_size + offset_in_page) * num_heads + h) * head_dim + d
//! ```
//!
//! The index buffers are passed by the caller as typed slices to keep this
//! module free of byte-parsing — parsing belongs at the descriptor layer.

use crate::Error;

/// Computes the flat element offset into `page_pool` (buffer 0) for the
/// element at token `t` of sequence `s`, head `h`, dimension `d`.
///
/// Returns `Err` if any index is out of bounds or if the `seq_ptr` / `block_table`
/// invariants are violated. Never panics.
///
/// # Arguments
///
/// - `s` — sequence index (must be `< num_seqs`).
/// - `t` — token index within the sequence (zero-based).
/// - `h` — head index (must be `< num_heads`).
/// - `d` — head-dimension index (must be `< head_dim`).
/// - `page_size` — tokens per page (from the layout descriptor; `>= 1`).
/// - `num_pages` — number of physical pages (from the layout descriptor).
/// - `num_heads` — `shape[1]` of the tensor.
/// - `head_dim` — `shape[2]` of the tensor.
/// - `block_table` — slice of physical page IDs (buffer 1). Length must be >= `seq_ptr[num_seqs]`.
/// - `seq_ptr` — offset array (buffer 2). Length must be `num_seqs + 1`.
///
/// # Errors
///
/// - [`Error::IndexOutOfRange`] — any index component is out of bounds.
/// - [`Error::AddressOverflow`] — intermediate arithmetic overflowed `u64`.
///
/// # Examples
///
/// ```
/// use hurray_core::layout::addressing::block_paged::element_offset_u32;
///
/// // From the spec example: page_size=4, num_pages=5, num_heads=2, head_dim=8.
/// // Sequence 0 uses block_table[0..2] = [0, 1].
/// // Token 4 is in page 1 (4/4=1), offset 0 (4%4=0).
/// // phys_page = block_table[0 + 1] = 1.
/// // flat = ((1*4 + 0)*2 + 0)*8 + 0 = (4*2)*8 = 64.
/// let block_table: &[u32] = &[0, 1, 0]; // seq0: [0,1], seq1: [0]
/// let seq_ptr: &[u32] = &[0, 2, 3];     // seq0 owns bt[0..2], seq1 owns bt[2..3]
/// let flat = element_offset_u32(
///     /*s=*/0, /*t=*/4, /*h=*/0, /*d=*/0,
///     /*page_size=*/4, /*num_pages=*/5,
///     /*num_heads=*/2, /*head_dim=*/8,
///     block_table, seq_ptr,
/// ).unwrap();
/// assert_eq!(flat, 64);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn element_offset_u32(
    s: u64,
    t: u64,
    h: u64,
    d: u64,
    page_size: u32,
    num_pages: u64,
    num_heads: u64,
    head_dim: u64,
    block_table: &[u32],
    seq_ptr: &[u32],
) -> crate::Result<u64> {
    element_offset_impl(
        s,
        t,
        h,
        d,
        page_size,
        num_pages,
        num_heads,
        head_dim,
        seq_ptr,
        block_table,
    )
}

/// Computes the flat element offset into `page_pool` (buffer 0) for the
/// element at token `t` of sequence `s`, head `h`, dimension `d`,
/// using 64-bit index buffers.
///
/// See [`element_offset_u32`] for the full documentation. The only difference
/// is that `block_table` and `seq_ptr` carry `u64` entries rather than `u32`,
/// enabling pools larger than 4 billion pages.
///
/// # Examples
///
/// ```
/// use hurray_core::layout::addressing::block_paged::element_offset_u64;
///
/// let block_table: &[u64] = &[0u64, 1, 0];
/// let seq_ptr: &[u64] = &[0u64, 2, 3];
/// let flat = element_offset_u64(
///     0, 4, 0, 0, 4, 5, 2, 8, block_table, seq_ptr,
/// ).unwrap();
/// assert_eq!(flat, 64);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn element_offset_u64(
    s: u64,
    t: u64,
    h: u64,
    d: u64,
    page_size: u32,
    num_pages: u64,
    num_heads: u64,
    head_dim: u64,
    block_table: &[u64],
    seq_ptr: &[u64],
) -> crate::Result<u64> {
    element_offset_impl(
        s,
        t,
        h,
        d,
        page_size,
        num_pages,
        num_heads,
        head_dim,
        seq_ptr,
        block_table,
    )
}

/// Shared inner implementation, generic over the index width.
///
/// `seq_ptr` has `num_seqs + 1` entries. `block_table` has `seq_ptr[num_seqs]` entries.
/// Generic over `T: Copy + Into<u64>` so the `u32` and `u64` entry points share this
/// logic without widening the index buffers into a temporary allocation per call —
/// this is the KV-cache lookup hot path, so per-call heap traffic is avoided.
#[allow(clippy::too_many_arguments)]
fn element_offset_impl<T: Copy + Into<u64>>(
    s: u64,
    t: u64,
    h: u64,
    d: u64,
    page_size: u32,
    num_pages: u64,
    num_heads: u64,
    head_dim: u64,
    seq_ptr: &[T],
    block_table: &[T],
) -> crate::Result<u64> {
    // seq_ptr must have at least 1 element (even for an empty batch: [0]).
    // The last element is seq_ptr[num_seqs].
    let num_seqs = seq_ptr.len().saturating_sub(1) as u64;

    // Validate sequence index.
    if s >= num_seqs {
        return Err(Error::IndexOutOfRange {
            dim: 0,
            index: s,
            size: num_seqs,
        });
    }

    // Validate head index.
    if h >= num_heads {
        return Err(Error::IndexOutOfRange {
            dim: 1,
            index: h,
            size: num_heads,
        });
    }

    // Validate dimension index.
    if d >= head_dim {
        return Err(Error::IndexOutOfRange {
            dim: 2,
            index: d,
            size: head_dim,
        });
    }

    let page_size_u64 = page_size as u64;

    // Locate the block-table slice for sequence s.
    // seq_ptr should be non-decreasing (storage invariant), but this function does
    // not assume the caller validated it — checked_sub guards against a malformed
    // (non-monotone) seq_ptr instead of panicking, honouring the "never panics" contract.
    let bt_start: u64 = seq_ptr[s as usize].into();
    let bt_end: u64 = seq_ptr[s as usize + 1].into();

    // The sequence's logical page count for this sequence.
    let seq_page_count = bt_end.checked_sub(bt_start).ok_or_else(|| {
        Error::InvalidLayout(
            "block-paged: seq_ptr is not non-decreasing (storage invariant violated)".to_string(),
        )
    })?;

    // Token → page index within the sequence.
    let page_in_seq = t / page_size_u64;
    let offset_in_page = t % page_size_u64;

    // Validate token: must be within the sequence's allocated page slots.
    // (The spec allows trailing undefined slots in the last page; we
    //  allow t up to seq_page_count * page_size - 1 here, matching what a
    //  reader is permitted to access. The caller is responsible for not
    //  reading past the sequence's valid token count.)
    if page_in_seq >= seq_page_count {
        return Err(Error::IndexOutOfRange {
            dim: 0,
            index: t,
            // saturating_mul: this is only the reported upper bound in an error path;
            // a saturated value still conveys "out of range" without risking overflow.
            size: seq_page_count.saturating_mul(page_size_u64),
        });
    }

    // Resolve physical page via block table.
    let bt_idx = bt_start
        .checked_add(page_in_seq)
        .ok_or(Error::AddressOverflow)?;

    let bt_idx_usize = bt_idx as usize;
    if bt_idx_usize >= block_table.len() {
        // Storage invariant violation: block_table is too short.
        return Err(Error::IndexOutOfRange {
            dim: 0,
            index: bt_idx,
            size: block_table.len() as u64,
        });
    }

    let phys_page: u64 = block_table[bt_idx_usize].into();

    // Storage invariant: all physical page IDs must be < num_pages.
    if phys_page >= num_pages {
        return Err(Error::IndexOutOfRange {
            dim: 0,
            index: phys_page,
            size: num_pages,
        });
    }

    // Compute flat offset per spec:
    //   flat = ((phys_page * page_size + offset_in_page) * num_heads + h) * head_dim + d
    let flat = phys_page
        .checked_mul(page_size_u64)
        .and_then(|v| v.checked_add(offset_in_page))
        .and_then(|v| v.checked_mul(num_heads))
        .and_then(|v| v.checked_add(h))
        .and_then(|v| v.checked_mul(head_dim))
        .and_then(|v| v.checked_add(d))
        .ok_or(Error::AddressOverflow)?;

    Ok(flat)
}

/// Validates the storage invariants for a block-paged tensor's index buffers.
///
/// Per `docs/spec/layouts/block-paged.md § Storage Invariants`:
///
/// 1. `seq_ptr[0] == 0`.
/// 2. `seq_ptr` is non-decreasing.
/// 3. `seq_ptr[num_seqs] == total_logical_pages` (equals `block_table.len()`).
/// 4. Every `block_table[k] < num_pages`.
///
/// Both U32 and U64 variants are supported through the typed overloads below.
/// This function operates on already-widened slices.
///
/// # Errors
///
/// Returns [`Error::InvalidLayout`] if any invariant is violated.
///
/// # Examples
///
/// ```
/// use hurray_core::layout::addressing::block_paged::validate_index_buffers_u64;
///
/// let seq_ptr: &[u64] = &[0, 2, 3];
/// let block_table: &[u64] = &[0, 1, 0];
/// assert!(validate_index_buffers_u64(5, 2, seq_ptr, block_table).is_ok());
/// ```
pub fn validate_index_buffers_u64(
    num_pages: u64,
    num_seqs: u32,
    seq_ptr: &[u64],
    block_table: &[u64],
) -> crate::Result<()> {
    validate_index_buffers_impl(num_pages, num_seqs, seq_ptr, block_table)
}

/// Validates the storage invariants for block-paged index buffers using `u32` entries.
///
/// Widening wrapper over [`validate_index_buffers_u64`]. See that function for
/// the full invariant list.
///
/// # Errors
///
/// Returns [`Error::InvalidLayout`] if any invariant is violated.
///
/// # Examples
///
/// ```
/// use hurray_core::layout::addressing::block_paged::validate_index_buffers_u32;
///
/// let seq_ptr: &[u32] = &[0, 2, 3];
/// let block_table: &[u32] = &[0, 1, 0];
/// assert!(validate_index_buffers_u32(5, 2, seq_ptr, block_table).is_ok());
/// ```
pub fn validate_index_buffers_u32(
    num_pages: u64,
    num_seqs: u32,
    seq_ptr: &[u32],
    block_table: &[u32],
) -> crate::Result<()> {
    // Widen before delegating — the inner impl uses u64 throughout.
    let sp: Vec<u64> = seq_ptr.iter().map(|&v| v as u64).collect();
    let bt: Vec<u64> = block_table.iter().map(|&v| v as u64).collect();
    validate_index_buffers_impl(num_pages, num_seqs, &sp, &bt)
}

fn validate_index_buffers_impl(
    num_pages: u64,
    num_seqs: u32,
    seq_ptr: &[u64],
    block_table: &[u64],
) -> crate::Result<()> {
    let expected_sp_len = (num_seqs as usize).saturating_add(1);
    if seq_ptr.len() != expected_sp_len {
        return Err(Error::InvalidLayout(format!(
            "block-paged: seq_ptr length {} does not equal num_seqs + 1 = {}",
            seq_ptr.len(),
            expected_sp_len
        )));
    }

    // Invariant 1: seq_ptr[0] == 0.
    if seq_ptr[0] != 0 {
        return Err(Error::InvalidLayout(format!(
            "block-paged: seq_ptr[0] must be 0, got {}",
            seq_ptr[0]
        )));
    }

    // Invariant 2: seq_ptr is non-decreasing.
    for i in 0..(seq_ptr.len() - 1) {
        if seq_ptr[i] > seq_ptr[i + 1] {
            return Err(Error::InvalidLayout(format!(
                "block-paged: seq_ptr is not non-decreasing at index {}: seq_ptr[{}]={} > seq_ptr[{}]={}",
                i,
                i,
                seq_ptr[i],
                i + 1,
                seq_ptr[i + 1]
            )));
        }
    }

    // Invariant 3: seq_ptr[num_seqs] == total_logical_pages == block_table.len().
    let total_logical_pages = seq_ptr[num_seqs as usize];
    if total_logical_pages != block_table.len() as u64 {
        return Err(Error::InvalidLayout(format!(
            "block-paged: seq_ptr[num_seqs]={} does not equal block_table.len()={}",
            total_logical_pages,
            block_table.len()
        )));
    }

    // Invariant 4: every block_table[k] < num_pages.
    for (k, &phys_page) in block_table.iter().enumerate() {
        if phys_page >= num_pages {
            return Err(Error::InvalidLayout(format!(
                "block-paged: block_table[{k}]={phys_page} >= num_pages={num_pages}"
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// The spec worked example (§ Example):
    ///   layer 3, key role, 2 seqs, page_size=4, num_pages=5, num_heads=2, head_dim=8.
    ///   seq_ptr     = [0, 2, 3]
    ///   block_table = [0, 1, 0]   ← seq 1 aliases page 0 (prefix sharing)
    ///
    /// Flat-address formula (spec §Element Lookup):
    ///   flat = ((phys_page * page_size + offset_in_page) * num_heads + h) * head_dim + d
    fn spec_example_u32() -> (&'static [u32], &'static [u32]) {
        let block_table: &[u32] = &[0, 1, 0];
        let seq_ptr: &[u32] = &[0, 2, 3];
        (block_table, seq_ptr)
    }

    fn spec_example_u64() -> (&'static [u64], &'static [u64]) {
        let block_table: &[u64] = &[0, 1, 0];
        let seq_ptr: &[u64] = &[0, 2, 3];
        (block_table, seq_ptr)
    }

    // ── element_offset_u32: spec worked example ───────────────────────────────

    /// Spec §Example: token 0 of seq 0, head 0, dim 0.
    /// page_in_seq=0, offset=0, phys_page=block_table[0]=0.
    /// flat = ((0*4 + 0)*2 + 0)*8 + 0 = 0.
    #[test]
    fn element_offset_u32_spec_example_seq0_t0_h0_d0() {
        let (bt, sp) = spec_example_u32();
        let flat = element_offset_u32(0, 0, 0, 0, 4, 5, 2, 8, bt, sp).unwrap();
        assert_eq!(flat, 0);
    }

    /// Spec §Example: token 4 of seq 0, head 0, dim 0.
    /// page_in_seq=1, offset=0, phys_page=block_table[1]=1.
    /// flat = ((1*4 + 0)*2 + 0)*8 + 0 = 64.
    #[test]
    fn element_offset_u32_spec_example_seq0_t4_h0_d0() {
        let (bt, sp) = spec_example_u32();
        let flat = element_offset_u32(0, 4, 0, 0, 4, 5, 2, 8, bt, sp).unwrap();
        assert_eq!(flat, 64);
    }

    /// Spec §Example: token 5 of seq 0, head 1, dim 7 (last valid element).
    /// page_in_seq=1, offset=1, phys_page=1.
    /// flat = ((1*4 + 1)*2 + 1)*8 + 7 = ((5)*2 + 1)*8 + 7 = 11*8 + 7 = 95.
    #[test]
    fn element_offset_u32_spec_example_seq0_t5_h1_d7() {
        let (bt, sp) = spec_example_u32();
        let flat = element_offset_u32(0, 5, 1, 7, 4, 5, 2, 8, bt, sp).unwrap();
        assert_eq!(flat, 95);
    }

    /// Spec §Prefix Sharing: seq 1 token 0 uses the same physical page as seq 0 token 0.
    /// seq 1: block_table[sp[1]..sp[2]] = block_table[2..3] = [0].
    /// page_in_seq=0, offset=0, phys_page=block_table[2]=0 (same as seq 0 page 0).
    /// flat = ((0*4 + 0)*2 + 0)*8 + 0 = 0.
    #[test]
    fn element_offset_u32_prefix_sharing_seq1_same_physical_page_as_seq0() {
        let (bt, sp) = spec_example_u32();
        let flat_seq0 = element_offset_u32(0, 0, 0, 0, 4, 5, 2, 8, bt, sp).unwrap();
        let flat_seq1 = element_offset_u32(1, 0, 0, 0, 4, 5, 2, 8, bt, sp).unwrap();
        // Both sequences' token 0 resolve to the same physical page (page 0), same offset.
        assert_eq!(
            flat_seq0, flat_seq1,
            "aliased pages must resolve to identical flat offsets"
        );
        assert_eq!(flat_seq1, 0);
    }

    /// Seq 1 has only 1 logical page (sp[1]=2, sp[2]=3). Token 4 would need a second page.
    #[test]
    fn element_offset_u32_token_beyond_sequence_page_count_is_error() {
        let (bt, sp) = spec_example_u32();
        let result = element_offset_u32(1, 4, 0, 0, 4, 5, 2, 8, bt, sp);
        assert!(
            matches!(result, Err(Error::IndexOutOfRange { .. })),
            "token 4 in seq 1 (only 1 page) must be IndexOutOfRange"
        );
    }

    /// Head index >= num_heads must be rejected.
    #[test]
    fn element_offset_u32_head_out_of_range_is_error() {
        let (bt, sp) = spec_example_u32();
        let result = element_offset_u32(0, 0, 2, 0, 4, 5, 2, 8, bt, sp);
        assert!(matches!(result, Err(Error::IndexOutOfRange { dim: 1, .. })));
    }

    /// Dimension index >= head_dim must be rejected.
    #[test]
    fn element_offset_u32_dim_out_of_range_is_error() {
        let (bt, sp) = spec_example_u32();
        let result = element_offset_u32(0, 0, 0, 8, 4, 5, 2, 8, bt, sp);
        assert!(matches!(result, Err(Error::IndexOutOfRange { dim: 2, .. })));
    }

    /// Sequence index >= num_seqs must be rejected.
    #[test]
    fn element_offset_u32_seq_out_of_range_is_error() {
        let (bt, sp) = spec_example_u32();
        let result = element_offset_u32(2, 0, 0, 0, 4, 5, 2, 8, bt, sp);
        assert!(matches!(result, Err(Error::IndexOutOfRange { dim: 0, .. })));
    }

    /// A non-monotone `seq_ptr` (storage-invariant violation) must return an error,
    /// not panic — `element_offset` does not assume the caller pre-validated the buffers.
    #[test]
    fn element_offset_u32_non_monotone_seq_ptr_is_error_not_panic() {
        let block_table: &[u32] = &[0];
        let seq_ptr: &[u32] = &[0, 2, 1]; // seq_ptr[2] < seq_ptr[1]
        let result = element_offset_u32(1, 0, 0, 0, 4, 5, 2, 8, block_table, seq_ptr);
        assert!(matches!(
            result,
            Err(Error::InvalidLayout(_) | Error::IndexOutOfRange { .. })
        ));
    }

    // ── element_offset_u64 ────────────────────────────────────────────────────

    /// Confirm element_offset_u64 produces the same result as u32 on the spec example.
    #[test]
    fn element_offset_u64_matches_u32_on_spec_example() {
        let (bt, sp) = spec_example_u64();
        let flat = element_offset_u64(0, 4, 0, 0, 4, 5, 2, 8, bt, sp).unwrap();
        assert_eq!(flat, 64);
    }

    /// Prefix sharing is observable via u64 variant too.
    #[test]
    fn element_offset_u64_prefix_sharing_aliased_offsets_are_equal() {
        let (bt, sp) = spec_example_u64();
        let flat_seq0 = element_offset_u64(0, 0, 0, 0, 4, 5, 2, 8, bt, sp).unwrap();
        let flat_seq1 = element_offset_u64(1, 0, 0, 0, 4, 5, 2, 8, bt, sp).unwrap();
        assert_eq!(flat_seq0, flat_seq1);
    }

    // ── element_offset: last valid slot in the page pool ─────────────────────

    /// Absolute last addressable element: page 4 (last page), slot 3, head 1, dim 7.
    /// flat = ((4*4 + 3)*2 + 1)*8 + 7 = (19*2 + 1)*8 + 7 = 39*8 + 7 = 319.
    /// page_pool has 5*4*2*8 = 320 elements; index 319 is the last.
    #[test]
    fn element_offset_u32_last_element_in_pool() {
        // Use a seq that owns physical page 4 via its block table entry.
        let block_table: &[u32] = &[4]; // seq 0 has 1 logical page → physical page 4
        let seq_ptr: &[u32] = &[0, 1];
        let flat = element_offset_u32(0, 3, 1, 7, 4, 5, 2, 8, block_table, seq_ptr).unwrap();
        assert_eq!(flat, 319);
    }

    // ── validate_index_buffers_u32: valid inputs ──────────────────────────────

    /// Spec §Storage Invariants (1)–(4): the spec example is valid.
    #[test]
    fn validate_index_buffers_u32_spec_example_is_valid() {
        let seq_ptr: &[u32] = &[0, 2, 3];
        let block_table: &[u32] = &[0, 1, 0];
        assert!(
            validate_index_buffers_u32(5, 2, seq_ptr, block_table).is_ok(),
            "spec example index buffers must be valid"
        );
    }

    /// Empty batch: num_seqs=0, seq_ptr=[0], block_table=[].
    #[test]
    fn validate_index_buffers_u32_empty_batch_is_valid() {
        let seq_ptr: &[u32] = &[0];
        let block_table: &[u32] = &[];
        assert!(
            validate_index_buffers_u32(5, 0, seq_ptr, block_table).is_ok(),
            "empty batch must be valid"
        );
    }

    /// Single-sequence batch with multiple pages.
    #[test]
    fn validate_index_buffers_u32_single_seq_multiple_pages_is_valid() {
        let seq_ptr: &[u32] = &[0, 3];
        let block_table: &[u32] = &[2, 0, 4];
        assert!(validate_index_buffers_u32(5, 1, seq_ptr, block_table).is_ok());
    }

    // ── validate_index_buffers_u32: invariant 1 — seq_ptr[0] != 0 ────────────

    /// Spec §Storage Invariant 1: seq_ptr[0] MUST be 0.
    #[test]
    fn validate_index_buffers_u32_rejects_seq_ptr_not_starting_at_zero() {
        let seq_ptr: &[u32] = &[1, 3]; // seq_ptr[0] = 1, not 0
        let block_table: &[u32] = &[0, 1, 2];
        let result = validate_index_buffers_u32(5, 1, seq_ptr, block_table);
        assert!(
            matches!(result, Err(Error::InvalidLayout(_))),
            "seq_ptr[0] != 0 must be InvalidLayout"
        );
    }

    // ── validate_index_buffers_u32: invariant 2 — non-monotonic seq_ptr ──────

    /// Spec §Storage Invariant 2: seq_ptr must be non-decreasing.
    #[test]
    fn validate_index_buffers_u32_rejects_non_monotonic_seq_ptr() {
        let seq_ptr: &[u32] = &[0, 3, 2]; // seq_ptr[1] > seq_ptr[2]
        let block_table: &[u32] = &[0, 1, 0];
        let result = validate_index_buffers_u32(5, 2, seq_ptr, block_table);
        assert!(
            matches!(result, Err(Error::InvalidLayout(_))),
            "non-monotonic seq_ptr must be InvalidLayout"
        );
    }

    // ── validate_index_buffers_u32: invariant 3 — seq_ptr[num_seqs] mismatch ─

    /// Spec §Storage Invariant 3: seq_ptr[num_seqs] must equal block_table.len().
    #[test]
    fn validate_index_buffers_u32_rejects_seq_ptr_total_mismatch_block_table_len() {
        // seq_ptr says 4 total pages but block_table has 3 entries.
        let seq_ptr: &[u32] = &[0, 2, 4];
        let block_table: &[u32] = &[0, 1, 2]; // only 3 entries, not 4
        let result = validate_index_buffers_u32(5, 2, seq_ptr, block_table);
        assert!(
            matches!(result, Err(Error::InvalidLayout(_))),
            "seq_ptr[num_seqs] != block_table.len() must be InvalidLayout"
        );
    }

    // ── validate_index_buffers_u32: invariant 4 — out-of-bounds page id ──────

    /// Spec §Storage Invariant 4: every block_table[k] < num_pages.
    #[test]
    fn validate_index_buffers_u32_rejects_out_of_bounds_page_id() {
        let seq_ptr: &[u32] = &[0, 2, 3];
        let block_table: &[u32] = &[0, 5, 0]; // page 5 >= num_pages=5
        let result = validate_index_buffers_u32(5, 2, seq_ptr, block_table);
        assert!(
            matches!(result, Err(Error::InvalidLayout(_))),
            "block_table[k] >= num_pages must be InvalidLayout"
        );
    }

    /// Page id equal to num_pages (exactly at boundary) must be rejected.
    #[test]
    fn validate_index_buffers_u32_rejects_page_id_equal_to_num_pages() {
        let seq_ptr: &[u32] = &[0, 1];
        let block_table: &[u32] = &[5]; // == num_pages (0-indexed: valid range is 0..4)
        let result = validate_index_buffers_u32(5, 1, seq_ptr, block_table);
        assert!(matches!(result, Err(Error::InvalidLayout(_))));
    }

    // ── validate_index_buffers_u64: basic coverage ────────────────────────────

    #[test]
    fn validate_index_buffers_u64_spec_example_is_valid() {
        let seq_ptr: &[u64] = &[0, 2, 3];
        let block_table: &[u64] = &[0, 1, 0];
        assert!(validate_index_buffers_u64(5, 2, seq_ptr, block_table).is_ok());
    }

    #[test]
    fn validate_index_buffers_u64_empty_batch_is_valid() {
        let seq_ptr: &[u64] = &[0];
        let block_table: &[u64] = &[];
        assert!(validate_index_buffers_u64(5, 0, seq_ptr, block_table).is_ok());
    }

    #[test]
    fn validate_index_buffers_u64_rejects_out_of_bounds_page_id() {
        let seq_ptr: &[u64] = &[0, 1];
        let block_table: &[u64] = &[5]; // == num_pages
        let result = validate_index_buffers_u64(5, 1, seq_ptr, block_table);
        assert!(matches!(result, Err(Error::InvalidLayout(_))));
    }
}
