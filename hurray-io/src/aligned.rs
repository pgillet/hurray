//! Owned allocations that satisfy a buffer handle's **declared** alignment.
//!
//! `docs/spec/buffer-protocol.md § Minimum Alignment` makes a buffer's declared alignment
//! a promise to the consumer: the base address of every non-empty buffer MUST be at least
//! 64-byte aligned, and a reader MAY rely on the declaration for SIMD loads. A reader that
//! materialises buffers therefore has to allocate at `handle.alignment()` — 4096 or more
//! for a GPU/IPC/RDMA producer — and not at whatever the allocator returns for a byte
//! array. See issue #235.

use std::alloc::{self, Layout};
use std::ptr::NonNull;

use bytes::Bytes;

/// An owned, zero-initialised allocation whose base address is aligned to a value chosen
/// at run time.
///
/// Not `BytesMut` or `Vec<u8>`: both allocate at `align_of::<u8>()`, which in practice
/// yields 8 or 16 bytes — under the format's 64-byte floor, let alone a declared 4096.
///
/// Zero-initialised rather than uninitialised because the destination is handed out as
/// `&mut [u8]` for a transport to fill, and a slice over uninitialised memory is UB. For a
/// page-aligned allocation the zeroing is the allocator's, not a memset.
pub(crate) struct AlignedBuf {
    /// Dangling when `len == 0`, which allocates nothing and is never dereferenced.
    ptr: NonNull<u8>,
    layout: Layout,
    len: usize,
}

impl AlignedBuf {
    /// Allocates `byte_size` zeroed bytes at `alignment`.
    ///
    /// Returns `None` when the two do not form a valid host layout: `byte_size` beyond
    /// `usize`/`isize` range, or an `alignment` that is not a power of two. Both are
    /// reachable from a descriptor off the wire, so neither may panic.
    pub(crate) fn zeroed(byte_size: u64, alignment: u32) -> Option<Self> {
        let len = usize::try_from(byte_size).ok()?;
        if len == 0 {
            // An empty buffer stays legal at any declared power of two
            // (`buffer-protocol.md § Empty Buffers`) and allocates nothing.
            return Some(Self {
                ptr: NonNull::dangling(),
                layout: Layout::from_size_align(0, 1).ok()?,
                len: 0,
            });
        }
        let layout = Layout::from_size_align(len, alignment as usize).ok()?;
        // SAFETY: layout has non-zero size (len > 0 checked above).
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        // Allocation failure aborts, as it does for Vec: there is no fallible path a
        // reader could usefully take, and the request came from a size we accepted.
        let ptr = NonNull::new(ptr).unwrap_or_else(|| alloc::handle_alloc_error(layout));
        Some(Self { ptr, layout, len })
    }

    /// The destination to fill, as a mutable slice of exactly the requested length.
    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.len == 0 {
            return &mut [];
        }
        // SAFETY: ptr is a live allocation of len zeroed bytes (so initialised), owned
        // exclusively by self, and &mut self prevents any aliasing slice.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Hands the allocation out as [`Bytes`] without copying or re-aligning it.
    ///
    /// `Bytes::from_owner` keeps *this* allocation and its address, which a `BytesMut`
    /// round trip would not. It costs one small owner box per buffer — paid once per
    /// buffer, never per byte — which is why the alignment is not bought with a copy.
    pub(crate) fn freeze(self) -> Bytes {
        if self.len == 0 {
            return Bytes::new();
        }
        Bytes::from_owner(self)
    }
}

impl AsRef<[u8]> for AlignedBuf {
    fn as_ref(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: as in as_mut_slice; &self grants shared read access only.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        if self.len > 0 {
            // SAFETY: ptr came from alloc_zeroed with exactly this layout, and Drop runs
            // once per allocation.
            unsafe { alloc::dealloc(self.ptr.as_ptr(), self.layout) };
        }
    }
}

// SAFETY: AlignedBuf exclusively owns its allocation and hands out a slice only through
// &self / &mut self, so moving it between threads transfers the whole allocation and
// shared access is read-only. Required because Bytes::from_owner needs a Send owner.
unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

#[cfg(test)]
mod tests {
    use super::*;
    use hurray_core::{MIN_BUFFER_ALIGNMENT, PAGE_ALIGNMENT};

    #[test]
    fn allocates_at_the_requested_alignment() {
        for alignment in [MIN_BUFFER_ALIGNMENT, 128, PAGE_ALIGNMENT, 1 << 20] {
            let buf = AlignedBuf::zeroed(1024, alignment).expect("valid layout");
            assert_eq!(
                buf.as_ref().as_ptr() as usize % alignment as usize,
                0,
                "allocation is not {alignment}-byte aligned"
            );
            assert_eq!(buf.as_ref().len(), 1024);
            assert!(buf.as_ref().iter().all(|&b| b == 0), "not zeroed");
        }
    }

    #[test]
    fn freeze_preserves_address_and_contents() {
        let mut buf = AlignedBuf::zeroed(256, PAGE_ALIGNMENT).expect("valid layout");
        buf.as_mut_slice().fill(0xAB);
        let addr = buf.as_ref().as_ptr() as usize;
        let frozen = buf.freeze();
        assert_eq!(
            frozen.as_ptr() as usize,
            addr,
            "freeze moved the allocation"
        );
        assert_eq!(frozen.len(), 256);
        assert!(frozen.iter().all(|&b| b == 0xAB));
    }

    #[test]
    fn empty_allocates_nothing_at_any_power_of_two() {
        for alignment in [1, MIN_BUFFER_ALIGNMENT, PAGE_ALIGNMENT] {
            let mut buf = AlignedBuf::zeroed(0, alignment).expect("valid layout");
            assert!(buf.as_mut_slice().is_empty());
            assert!(buf.freeze().is_empty());
        }
    }

    #[test]
    fn rejects_layouts_it_cannot_allocate() {
        // Both are reachable from a descriptor off the wire; neither may panic.
        assert!(AlignedBuf::zeroed(u64::MAX, MIN_BUFFER_ALIGNMENT).is_none());
        assert!(AlignedBuf::zeroed(1024, 96).is_none(), "not a power of two");
    }
}
