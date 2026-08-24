//! Buffer ownership model for `hurray.Tensor`.
//!
//! ## Design decision (D2)
//!
//! Two variants:
//!
//! - `Owned` — bytes copied into an allocation this module over-aligns to
//!   [`MIN_BUFFER_ALIGNMENT`] (e.g. `hurray.Tensor(buf, …)`).
//! - `Borrowed` — zero-copy pointer into a source Python object's buffer
//!   (e.g. `hurray.from_numpy(arr)`). A strong Python reference (`base`) keeps
//!   the source alive for the Tensor's entire lifetime.
//!
//! The alternative (Python buffer protocol via `PyBuffer`) was considered but
//! rejected: it doesn't cover GPU tensors (CUDA buffers don't implement it).
//! The raw ptr+base pattern is what PyTorch itself uses for DLPack zero-copy.
//!
//! Which variant a Python buffer lands in is not the caller's choice alone: [`ingest`]
//! borrows only what the format's alignment floor accepts, and copies the rest
//! (ADR-037 § 5–6). Every descriptor this crate emits therefore declares an alignment
//! its bytes actually have — see [`measured_alignment`].

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

    /// The alignment these bytes actually satisfy, as a power of two.
    ///
    /// This is what a descriptor MUST declare for the buffer (ADR-037 § 5). It is always
    /// at least [`MIN_BUFFER_ALIGNMENT`] for a non-empty store: `Owned` allocates
    /// over-aligned, and `Borrowed` is only constructed through [`ingest`] on an address
    /// that already qualifies.
    pub(crate) fn alignment(&self) -> u32 {
        measured_alignment(self.as_ptr(), self.len())
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

// ── Alignment: measured, never asserted (ADR-037 § 5) ─────────────────────────

/// The alignment a buffer's base address actually satisfies, as a power of two.
///
/// Reports the largest power of two the address is a multiple of, capped at
/// [`PAGE_ALIGNMENT`](hurray_core::PAGE_ALIGNMENT) — a stronger *true* declaration is
/// legal and useful to IPC and RDMA consumers, and costs nothing to state. Empty buffers
/// report `1`, matching `BufferHandle::empty`.
///
/// The result may be **below** [`MIN_BUFFER_ALIGNMENT`], in which case the caller must
/// copy into an aligned allocation rather than declare it: the format requires 64, and
/// `hurray-core` rejects anything less. Use [`must_copy`] to make that decision.
pub(crate) fn measured_alignment(ptr: *const u8, len: usize) -> u32 {
    if len == 0 {
        return 1;
    }
    let address = ptr as usize;
    let mut alignment: u32 = 1;
    while alignment < hurray_core::PAGE_ALIGNMENT && address.is_multiple_of(alignment as usize * 2)
    {
        alignment *= 2;
    }
    alignment
}

/// Whether a borrowed Python buffer must be copied to be declarable, honouring the
/// caller's `copy` request (ADR-037 § 6).
///
/// - `Some(true)` — always copy.
/// - `None` — copy only when the source is under-aligned. The default, because refusing
///   by default would break `from_numpy` for essentially every array (glibc puts large
///   NumPy allocations 16 bytes past a page boundary, deterministically), and copying
///   unconditionally would give up zero-copy even when the source qualified.
/// - `Some(false)` — never copy; an under-aligned source is an error naming the
///   alignment actually measured, so the cost is visible rather than silent.
///
/// `what` names the buffer in that error (`"array"`, `"values"`, `"indptr"`, …).
pub(crate) fn must_copy(
    ptr: *const u8,
    len: usize,
    copy: Option<bool>,
    what: &str,
) -> PyResult<bool> {
    if copy == Some(true) {
        return Ok(true);
    }
    let alignment = measured_alignment(ptr, len);
    // An empty buffer declares alignment 1 and is accepted as-is: there is no byte to
    // load, so there is nothing to align.
    if len == 0 || alignment >= MIN_BUFFER_ALIGNMENT {
        return Ok(false);
    }
    if copy == Some(false) {
        // CopyRequiredError, not BufferError as ADR-037 § 6 first wrote: the binding
        // already reserves this class for "copy=False but a copy is needed", and a
        // caller catching it from __array__ should catch the same thing here.
        return Err(crate::errors::CopyRequiredError::new_err(format!(
            // The message names the remedy, not just the problem: a caller who reached
            // for copy=False wants the copy gone, and hurray.aligned_allocator() is how.
            "{what} is {alignment}-byte aligned, below the {MIN_BUFFER_ALIGNMENT}-byte minimum \
             the format requires; pass copy=None (the default) to copy it into an aligned \
             allocation, or allocate the source inside a `with hurray.aligned_allocator():` \
             block so no copy is needed"
        )));
    }
    Ok(true)
}

/// Take a Python-owned buffer into a [`BufferStore`]: shared when its address already
/// satisfies the format's alignment floor, copied into an aligned allocation when it does
/// not, and refused when the caller passed `copy=False` (ADR-037 § 6).
///
/// This is the one door every borrowed buffer goes through, so no ingest path can declare
/// an alignment it did not check.
///
/// # Safety
///
/// The caller MUST ensure `ptr` points to at least `len` bytes of valid, initialised
/// memory, and that `base` keeps that allocation alive. The GIL MUST be held: on the copy
/// path the bytes are read here, and Python must not be able to free them concurrently.
pub(crate) unsafe fn ingest(
    ptr: *mut u8,
    len: usize,
    base: Py<PyAny>,
    copy: Option<bool>,
    what: &str,
) -> PyResult<BufferStore> {
    if must_copy(ptr, len, copy, what)? {
        // SAFETY: the caller guarantees ptr is valid for len bytes, and the GIL is held
        // for the duration of the copy.
        Ok(BufferStore::from_slice(unsafe {
            std::slice::from_raw_parts(ptr as *const u8, len)
        }))
    } else {
        // SAFETY: forwarded from this function's own contract.
        Ok(unsafe { BufferStore::borrowed(ptr, len, base) })
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
