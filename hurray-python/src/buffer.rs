//! Buffer ownership model for `hurray.Tensor`.
//!
//! ## Design decision (D2)
//!
//! `BufferStore` replaces the `Vec<u8>` from Phase 8a.2. Two variants:
//!
//! - `Owned` — bytes copied at construction time (e.g. `hurray.Tensor(buf, …)`).
//! - `Borrowed` — zero-copy pointer into a source Python object's buffer
//!   (e.g. `hurray.from_numpy(arr)`). A strong Python reference (`base`) keeps
//!   the source alive for the Tensor's entire lifetime.
//!
//! The alternative (Python buffer protocol via `PyBuffer`) was considered but
//! rejected: it doesn't cover GPU tensors (CUDA buffers don't implement it).
//! The raw ptr+base pattern is what PyTorch itself uses for DLPack zero-copy.

use hurray_core::MIN_BUFFER_ALIGNMENT;
use pyo3::prelude::*;

/// Owns or borrows the element data buffer of a `hurray.Tensor`.
///
/// # Safety invariant (Borrowed variant)
///
/// `ptr` MUST remain valid for as long as the `Tensor` is alive.
/// The `base` field enforces this: it holds a strong Python reference to the
/// source object (NumPy array, torch.Tensor, …). Python's garbage collector
/// will not free the source — and therefore the underlying buffer — as long as
/// `base` exists.
#[derive(Debug)]
pub enum BufferStore {
    /// Tensor owns its data, in an allocation over-aligned to
    /// [`MIN_BUFFER_ALIGNMENT`].
    ///
    /// Not a `Box<[u8]>`: that is allocated at `align_of::<u8>() == 1`, and in
    /// practice `malloc` returns 16 bytes of alignment for these sizes. The format
    /// requires the base address of every non-empty buffer to be 64-byte aligned
    /// (`buffer-protocol.md` § Alignment), and a descriptor that declares 64 over a
    /// 16-aligned address invites a consumer's aligned SIMD load to fault.
    ///
    /// `ptr` is dangling and `len` is `0` for an empty buffer, which allocates nothing.
    Owned { ptr: *mut u8, len: usize },
    /// Zero-copy pointer into a source Python object's buffer.
    ///
    /// `base` MUST be a strong Python reference to the object that owns the
    /// allocation at `ptr`. The Tensor does NOT own the bytes.
    Borrowed {
        ptr: *mut u8,
        len: usize,
        /// Strong Python reference to the buffer owner.
        /// Kept alive as long as this `BufferStore` exists.
        base: Py<PyAny>,
    },
}

impl Drop for BufferStore {
    fn drop(&mut self) {
        if let BufferStore::Owned { ptr, len } = self {
            if *len > 0 {
                // SAFETY: ptr came from std::alloc::alloc with exactly this layout in
                // from_slice, and Drop runs once.
                unsafe { std::alloc::dealloc(*ptr, Self::owned_layout(*len)) };
            }
        }
    }
}

// SAFETY: BufferStore is only accessed while holding the GIL (all entry points
// go through PyO3 `#[pymethods]` which hold the GIL). The raw pointer in
// Borrowed is never sent across thread boundaries without GIL protection.
unsafe impl Send for BufferStore {}
unsafe impl Sync for BufferStore {}

impl BufferStore {
    /// Construct an owned buffer by copying the given slice into an allocation
    /// aligned to [`MIN_BUFFER_ALIGNMENT`].
    ///
    /// # Panics
    ///
    /// On allocation failure, via [`std::alloc::handle_alloc_error`] — the same
    /// behaviour `Vec` has, and the only option available: this is called from
    /// contexts that cannot return an error.
    pub fn from_slice(bytes: &[u8]) -> Self {
        let len = bytes.len();
        if len == 0 {
            return BufferStore::Owned {
                ptr: std::ptr::NonNull::<u8>::dangling().as_ptr(),
                len: 0,
            };
        }
        let layout = Self::owned_layout(len);
        // SAFETY: layout has non-zero size (len > 0 checked above).
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        // SAFETY: ptr is a fresh allocation of at least len bytes, and bytes is a
        // live slice of exactly len bytes; the two cannot overlap.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, len) };
        BufferStore::Owned { ptr, len }
    }

    /// The allocation layout an owned buffer of `len` bytes uses.
    ///
    /// Alignment is padded up so `size` is a multiple of it, as `Layout` requires.
    fn owned_layout(len: usize) -> std::alloc::Layout {
        std::alloc::Layout::from_size_align(len, MIN_BUFFER_ALIGNMENT as usize)
            .expect("MIN_BUFFER_ALIGNMENT is a valid power of two and len fits in isize")
            .pad_to_align()
    }

    /// Construct a borrowed buffer from a raw pointer and a Python base object.
    ///
    /// # Safety
    ///
    /// The caller MUST ensure:
    /// - `ptr` points to at least `len` bytes of valid, initialised memory.
    /// - `base` is a Python object whose lifetime controls the allocation at `ptr`
    ///   (i.e., the allocation will not be freed while `base` is alive).
    pub unsafe fn borrowed(ptr: *mut u8, len: usize, base: Py<PyAny>) -> Self {
        BufferStore::Borrowed { ptr, len, base }
    }

    /// Return a raw const pointer to the first byte of the buffer.
    pub fn as_ptr(&self) -> *const u8 {
        match self {
            BufferStore::Owned { ptr, .. } => *ptr,
            // SAFETY: ptr is valid for at least `len` bytes per the Borrowed invariant.
            BufferStore::Borrowed { ptr, .. } => *ptr as *const u8,
        }
    }

    /// Return a mutable raw pointer to the first byte of the buffer.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        match self {
            BufferStore::Owned { ptr, .. } => *ptr,
            BufferStore::Borrowed { ptr, .. } => *ptr,
        }
    }

    /// Number of bytes in the buffer.
    pub fn len(&self) -> usize {
        match self {
            BufferStore::Owned { len, .. } => *len,
            BufferStore::Borrowed { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// View the buffer as a byte slice.
    ///
    /// # Safety
    ///
    /// For `Borrowed` buffers, the caller must hold the GIL and must not
    /// mutate the underlying allocation through another Python reference while
    /// this slice is live.
    pub unsafe fn as_slice(&self) -> &[u8] {
        match self {
            // SAFETY: ptr is valid for len bytes; dangling only when len is 0,
            // which from_raw_parts permits for a correctly aligned dangling pointer.
            BufferStore::Owned { ptr, len } => std::slice::from_raw_parts(*ptr, *len),
            // SAFETY: ptr is valid for `len` bytes per the Borrowed invariant.
            BufferStore::Borrowed { ptr, len, .. } => std::slice::from_raw_parts(*ptr, *len),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bug this replaced: `Box<[u8]>` is allocated at `align_of::<u8>() == 1`, so
    /// every descriptor `hurray-python` produced declared 64-byte alignment over an
    /// address that usually had 16 — inviting a consumer's aligned SIMD load to fault.
    #[test]
    fn owned_buffers_are_actually_aligned_to_the_alignment_they_declare() {
        for len in [1usize, 16, 63, 64, 256, 1024, 4096, 4097] {
            let store = BufferStore::from_slice(&vec![0xABu8; len]);
            let address = store.as_ptr() as usize;
            assert_eq!(
                address % MIN_BUFFER_ALIGNMENT as usize,
                0,
                "a {len}-byte owned buffer landed at {address:#x}, which is not \
                 {MIN_BUFFER_ALIGNMENT}-byte aligned"
            );
            assert_eq!(store.len(), len);
        }
    }

    #[test]
    fn an_empty_owned_buffer_allocates_nothing_and_still_reads() {
        let store = BufferStore::from_slice(&[]);
        assert_eq!(store.len(), 0);
        // SAFETY: an empty slice over a dangling but aligned pointer is well-defined.
        assert!(unsafe { store.as_slice() }.is_empty());
    }

    #[test]
    fn owned_round_trip() {
        let data = vec![1u8, 2, 3, 4];
        let store = BufferStore::from_slice(&data);
        assert_eq!(store.len(), 4);
        // SAFETY: Owned variant, no aliasing.
        let slice = unsafe { store.as_slice() };
        assert_eq!(slice, &[1, 2, 3, 4]);
    }

    #[test]
    fn owned_is_empty_on_empty_slice() {
        let store = BufferStore::from_slice(&[]);
        assert!(store.is_empty());
    }
}
