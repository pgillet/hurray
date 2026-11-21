//! Where a run writes its bytes.
//!
//! A run writes through this trait rather than directly into a slice, so that the serial
//! path and the `rayon` path share one engine. The serial path writes into a `&mut [u8]`
//! and contains no `unsafe`; the parallel path (feature `rayon`) implements the same
//! trait over a raw pointer, because handing every work item a `&mut [u8]` over the same
//! destination would create overlapping mutable references — unsound even when the bytes
//! each item writes are disjoint.

/// A destination buffer a run writes into.
///
/// Implementors may assume every `(offset, bytes)` pair is in bounds: a run validates the
/// destination against the plan's published size before it writes anything, and the plan
/// derives that size from the layout's own addressing span.
pub(crate) trait DestinationBytes {
    /// Copies `bytes` into the destination starting at `offset`.
    fn write_at(&mut self, offset: usize, bytes: &[u8]);
}

impl DestinationBytes for &mut [u8] {
    #[inline]
    fn write_at(&mut self, offset: usize, bytes: &[u8]) {
        self[offset..offset + bytes.len()].copy_from_slice(bytes);
    }
}
