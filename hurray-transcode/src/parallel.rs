//! The `rayon` edge: a destination several work items can write at once.
//!
//! This module exists because handing every work item a `&mut [u8]` over the same
//! destination would create overlapping mutable references, which is undefined behaviour
//! in Rust even when the bytes each item writes are disjoint. Writing through a raw
//! pointer is the formulation that matches what the plan actually guarantees.
//!
//! # The disjointness invariant, and where each part of it is established
//!
//! Two work items never write the same byte, and that follows from four facts rather than
//! from testing:
//!
//! 1. **Items partition the output cover's flat index range.** `run_items` converts an
//!    item range into `start..end` over `0..total` and walks exactly those positions, so
//!    distinct items cover disjoint sets of cover indices.
//! 2. **Distinct cover indices address distinct elements.** The plan refuses any target
//!    whose strides are not *provably* injective over the cover
//!    (`hurray_core::strides_provably_distinct`, called in `Remap::new`), which is the
//!    check that makes this step hold for arbitrary strides — including `[1, 1]`, which has
//!    no zero stride and still aliases. A broadcast axis is clamped out of the cover
//!    before the check, so the one aliasing target this crate supports is the one whose
//!    repeats are never written twice.
//! 3. **Distinct elements occupy disjoint byte ranges.** Addressing is affine in whole
//!    elements and every element is exactly `elem_bytes` long, so two distinct element
//!    offsets are at least `elem_bytes` apart. `element_bytes` refuses sub-byte types,
//!    which is what keeps this step true.
//! 4. **Every byte written is inside the destination.** `Geometry` proves at plan time
//!    that every address lies in `[0, extent)` with `extent ≤ isize::MAX`, and a run
//!    refuses a destination shorter than `extent` before writing anything.
//!
//! Steps 1–3 give disjointness; step 4 gives in-bounds. Both are properties of the plan,
//! established before any thread starts, which is why the `unsafe` below can be audited
//! without reading the loop.

use crate::dest::DestinationBytes;

/// A destination addressed by raw pointer, so that work items running on different
/// threads can write into disjoint parts of it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SharedDestination {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: the pointer is derived from a `&mut [u8]` whose borrow is held by the caller of
// `run_parallel` for the whole fan-out, so the allocation outlives every use and no other
// reference to it exists. The only operation performed through the handle writes bytes
// that one work item owns exclusively, per the four-step argument in this module's
// documentation. Sending it to another thread is therefore no more than sending a disjoint
// `&mut [u8]` would be.
unsafe impl Send for SharedDestination {}
// SAFETY: as above — a shared reference to this handle grants no read path and no write
// path other than `write_at`, whose disjointness argument is the one recorded on `Send`.
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
        // A hard check, not a `debug_assert`: it is one branch per copied run — never per
        // byte — and it is the last thing standing between a defect in the address
        // arithmetic and an out-of-bounds write through a raw pointer. Step 4 above says
        // it cannot fire; if it ever does, a panic is the outcome to want.
        let end = offset
            .checked_add(bytes.len())
            .expect("destination offset overflows usize");
        assert!(
            end <= self.len,
            "destination overrun: {end} bytes into a {} byte destination",
            self.len
        );
        // SAFETY: `offset + bytes.len() <= self.len`, asserted immediately above, so the
        // written range lies inside the allocation `self.ptr` came from. That range belongs
        // to exactly one work item (steps 1–3 in this module's documentation), so no other
        // thread reads or writes it concurrently. `bytes` borrows the input, which is a
        // `&[u8]` the caller holds simultaneously with the destination's `&mut [u8]` — so
        // safe Rust has already established that the two do not overlap, which is what
        // `copy_nonoverlapping` requires.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.ptr.add(offset), bytes.len());
        }
    }
}
