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
    /// Tensor owns its data. Allocated at construction; dropped with the Tensor.
    Owned(Box<[u8]>),
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

// SAFETY: BufferStore is only accessed while holding the GIL (all entry points
// go through PyO3 `#[pymethods]` which hold the GIL). The raw pointer in
// Borrowed is never sent across thread boundaries without GIL protection.
unsafe impl Send for BufferStore {}
unsafe impl Sync for BufferStore {}

impl BufferStore {
    /// Construct an owned buffer by copying the given slice.
    pub fn from_slice(bytes: &[u8]) -> Self {
        BufferStore::Owned(bytes.to_vec().into_boxed_slice())
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
            BufferStore::Owned(b) => b.as_ptr(),
            // SAFETY: ptr is valid for at least `len` bytes per the Borrowed invariant.
            BufferStore::Borrowed { ptr, .. } => *ptr as *const u8,
        }
    }

    /// Return a mutable raw pointer to the first byte of the buffer.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        match self {
            BufferStore::Owned(b) => b.as_mut_ptr(),
            BufferStore::Borrowed { ptr, .. } => *ptr,
        }
    }

    /// Number of bytes in the buffer.
    pub fn len(&self) -> usize {
        match self {
            BufferStore::Owned(b) => b.len(),
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
            BufferStore::Owned(b) => b,
            // SAFETY: ptr is valid for `len` bytes per the Borrowed invariant.
            BufferStore::Borrowed { ptr, len, .. } => std::slice::from_raw_parts(*ptr, *len),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
