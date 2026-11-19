//! The `rayon` edge: a destination several work items can write at once.
//!
//! This module exists because handing every work item a `&mut [u8]` over the same
//! destination would create overlapping mutable references, which is undefined behaviour
//! in Rust even when the bytes each item writes are disjoint. Writing through a raw
//! pointer is the formulation that matches what the plan actually guarantees: items
//! partition the *output*, so no two of them write the same byte (ADR-044 § 6).

use crate::dest::DestinationBytes;

/// A destination addressed by raw pointer, so that work items running on different
/// threads can write into disjoint parts of it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SharedDestination {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: the pointer is derived from a `&mut [u8]` that outlives every use (the borrow
// is held by the caller of `run_parallel` for the whole fan-out), and the only operation
// performed through it writes bytes that one work item owns exclusively. Sending the
// handle to another thread is therefore no more than sending a disjoint `&mut` would be.
unsafe impl Send for SharedDestination {}
// SAFETY: as above — `&SharedDestination` grants no read path and no write path other
// than `write_at`, whose disjointness argument is the one recorded on `Send`.
unsafe impl Sync for SharedDestination {}

impl SharedDestination {
    pub(crate) fn new(dst: &mut [u8]) -> Self {
        Self {
            ptr: dst.as_mut_ptr(),
            len: dst.len(),
        }
    }
}

impl DestinationBytes for SharedDestination {
    #[inline]
    fn write_at(&mut self, offset: usize, bytes: &[u8]) {
        debug_assert!(offset + bytes.len() <= self.len, "destination overrun");
        // SAFETY: `offset + bytes.len() <= self.len` — the run validated every
        // destination against the plan's published size before any item started, and the
        // plan derives that size from the layout's own addressing span, so every address
        // this engine computes lies inside the destination. The written range belongs to
        // exactly one work item (§ 6), so no other thread writes or reads it
        // concurrently, and `bytes` borrows the input, which is never written at all.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.ptr.add(offset), bytes.len());
        }
    }
}
