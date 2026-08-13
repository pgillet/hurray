//! Opaque list of buffer handles for multi-buffer tensors (ADR-030).
//!
//! A tensor whose descriptor references more than one buffer — per-channel,
//! NF4 or MXFP quantization, sparse layouts, block-paged, composite — needs all
//! of its buffers to travel together. [`HurrayBufferList`] is that carrier: an
//! ordered, owning collection of [`HurrayBuffer`] handles.
//!
//! ## Ownership
//!
//! The list **owns** every handle pushed into it. [`hurray_buffer_list_get`]
//! returns a *borrowed* pointer — the caller MUST NOT destroy it.
//! [`hurray_buffer_list_destroy`] destroys every owned handle exactly once and
//! then frees the list.
//!
//! ## Order
//!
//! Element `i` of the list is the buffer at index `i` of the tensor
//! descriptor's buffer table (ADR-030 § 3). Buffer indices appearing in
//! quantization descriptors (`scale_buffer_index`, `zero_point_buffer_index`),
//! layout descriptors, and composite members therefore index this list
//! directly.

use crate::{
    buffer::{hurray_buffer_destroy, HurrayBuffer},
    panic::{catch, null_check},
    status::{HurrayStatus, HURRAY_ERR_INDEX_OUT_OF_BOUNDS, HURRAY_ERR_INTERNAL, HURRAY_OK},
};

// ── Debug sentinel constants ──────────────────────────────────────────────────

/// Sentinel placed in `HurrayBufferList.sentinel` on allocation (debug builds only).
#[cfg(debug_assertions)]
const SENTINEL_LIVE: u64 = 0x1157_0000_11FE_0000;

/// Sentinel written before the list allocation is freed, to detect a double
/// destroy in debug builds.
#[cfg(debug_assertions)]
const SENTINEL_DEAD: u64 = 0x1157_DEAD_1157_DEAD;

// ── HurrayBufferList ──────────────────────────────────────────────────────────

/// Opaque handle to an ordered, owning list of [`HurrayBuffer`] handles.
///
/// Like [`HurrayBuffer`], this struct is **not** `#[repr(C)]`; its layout is an
/// implementation detail and callers MUST treat it as a black-box pointer.
/// Keeping it opaque is what allows [`hurray_buffer_list_get`] to bounds-check
/// and to hand back a borrowed handle — a bare `HurrayBuffer**` array could do
/// neither.
pub struct HurrayBufferList {
    /// Double-destroy sentinel (debug builds only).
    #[cfg(debug_assertions)]
    sentinel: u64,
    /// Owned handles in descriptor-buffer-table order. A slot is nulled as soon
    /// as its handle is destroyed, so a re-entrant destroy cannot free it twice.
    buffers: Vec<*mut HurrayBuffer>,
}

// SAFETY: mirrors the `HurrayBuffer` contract — the C ABI is single-thread-at-a-time
// per handle, so `Send` only permits moving the box across threads during
// construction and destruction, never concurrent access.
unsafe impl Send for HurrayBufferList {}

// ── Constructor ───────────────────────────────────────────────────────────────

/// Creates an empty [`HurrayBufferList`].
///
/// `capacity` is a hint only; the list grows as needed. Pass the tensor's
/// buffer count to avoid reallocation.
///
/// The caller owns the returned list and MUST destroy it exactly once with
/// [`hurray_buffer_list_destroy`].
///
/// # Safety
///
/// `out_list` MUST be a valid, non-null, writable pointer.
///
/// # Examples
///
/// ```
/// use hurray_ffi::buffer_list::{hurray_buffer_list_destroy, hurray_buffer_list_new};
/// use hurray_ffi::{HurrayBufferList, HURRAY_OK};
///
/// let mut list: *mut HurrayBufferList = std::ptr::null_mut();
/// assert_eq!(unsafe { hurray_buffer_list_new(2, &mut list) }, HURRAY_OK);
/// assert!(!list.is_null());
///
/// assert_eq!(unsafe { hurray_buffer_list_destroy(&mut list) }, HURRAY_OK);
/// assert!(list.is_null()); // destroy nulls the caller's pointer
/// ```
#[no_mangle]
pub unsafe extern "C" fn hurray_buffer_list_new(
    capacity: u64,
    out_list: *mut *mut HurrayBufferList,
) -> HurrayStatus {
    catch(|| {
        null_check!(out_list);

        // Cap the pre-allocation: `capacity` is an untrusted hint, and a bogus
        // value must not let a caller request an enormous allocation up front.
        let hint = capacity.min(64) as usize;

        let boxed = Box::new(HurrayBufferList {
            #[cfg(debug_assertions)]
            sentinel: SENTINEL_LIVE,
            buffers: Vec::with_capacity(hint),
        });
        // SAFETY: just allocated; ownership transfers to the caller, who must
        // call hurray_buffer_list_destroy exactly once.
        *out_list = Box::into_raw(boxed);
        HURRAY_OK
    })
}

// ── Push ──────────────────────────────────────────────────────────────────────

/// Appends `buffer` to `list`, transferring ownership of the handle to the list.
///
/// On success the caller MUST NOT destroy `buffer` — destroying the list
/// destroys it. On failure ownership stays with the caller.
///
/// Buffers MUST be pushed in descriptor buffer-table order: the first push is
/// buffer index `0`, the second is index `1`, and so on.
///
/// # Safety
///
/// - `list` MUST be a live handle from [`hurray_buffer_list_new`].
/// - `buffer` MUST be a live handle from `hurray_buffer_from_ptr` that has not
///   been destroyed and is not already owned by another list.
///
/// # Examples
///
/// ```
/// use hurray_ffi::buffer::hurray_buffer_from_ptr;
/// use hurray_ffi::buffer_list::{
///     hurray_buffer_list_destroy, hurray_buffer_list_len, hurray_buffer_list_new,
///     hurray_buffer_list_push,
/// };
/// use hurray_ffi::{HurrayBuffer, HurrayBufferList, HURRAY_OK};
///
/// #[repr(align(64))]
/// struct Aligned([u8; 64]);
/// let mut data = Aligned([0u8; 64]);
///
/// let mut buffer: *mut HurrayBuffer = std::ptr::null_mut();
/// assert_eq!(
///     unsafe {
///         hurray_buffer_from_ptr(
///             data.0.as_mut_ptr().cast(), 64, 64, 0, 0, 0,
///             None, std::ptr::null_mut(), &mut buffer,
///         )
///     },
///     HURRAY_OK
/// );
///
/// let mut list: *mut HurrayBufferList = std::ptr::null_mut();
/// unsafe { hurray_buffer_list_new(1, &mut list) };
/// assert_eq!(unsafe { hurray_buffer_list_push(list, buffer) }, HURRAY_OK);
///
/// let mut len: u64 = 0;
/// unsafe { hurray_buffer_list_len(list, &mut len) };
/// assert_eq!(len, 1);
///
/// // Destroying the list destroys the pushed buffer too.
/// unsafe { hurray_buffer_list_destroy(&mut list) };
/// ```
#[no_mangle]
pub unsafe extern "C" fn hurray_buffer_list_push(
    list: *mut HurrayBufferList,
    buffer: *mut HurrayBuffer,
) -> HurrayStatus {
    catch(|| {
        null_check!(list, buffer);

        #[cfg(debug_assertions)]
        // SAFETY: list is non-null and points to a live HurrayBufferList.
        if (*list).sentinel == SENTINEL_DEAD {
            return HURRAY_ERR_INTERNAL;
        }

        // SAFETY: list is non-null and points to a live HurrayBufferList.
        (*list).buffers.push(buffer);
        HURRAY_OK
    })
}

// ── Accessors ─────────────────────────────────────────────────────────────────

/// Reads the number of buffers in `list`.
///
/// # Safety
///
/// - `list` MUST be a live handle from [`hurray_buffer_list_new`].
/// - `out_len` MUST be a valid, non-null, writable pointer.
///
/// # Examples
///
/// ```
/// use hurray_ffi::buffer_list::{
///     hurray_buffer_list_destroy, hurray_buffer_list_len, hurray_buffer_list_new,
/// };
/// use hurray_ffi::{HurrayBufferList, HURRAY_OK};
///
/// let mut list: *mut HurrayBufferList = std::ptr::null_mut();
/// unsafe { hurray_buffer_list_new(0, &mut list) };
///
/// let mut len: u64 = 7;
/// assert_eq!(unsafe { hurray_buffer_list_len(list, &mut len) }, HURRAY_OK);
/// assert_eq!(len, 0);
///
/// unsafe { hurray_buffer_list_destroy(&mut list) };
/// ```
#[no_mangle]
pub unsafe extern "C" fn hurray_buffer_list_len(
    list: *const HurrayBufferList,
    out_len: *mut u64,
) -> HurrayStatus {
    catch(|| {
        null_check!(list, out_len);

        // SAFETY: list is non-null and points to a live HurrayBufferList.
        *out_len = (*list).buffers.len() as u64;
        HURRAY_OK
    })
}

/// Borrows the [`HurrayBuffer`] at `index`.
///
/// The returned handle is **borrowed**: ownership stays with the list, and the
/// caller MUST NOT call `hurray_buffer_destroy` on it. It stays valid until the
/// list is destroyed.
///
/// Returns [`HURRAY_ERR_INDEX_OUT_OF_BOUNDS`] if `index` is not less than the
/// list length.
///
/// # Safety
///
/// - `list` MUST be a live handle from [`hurray_buffer_list_new`].
/// - `out_buffer` MUST be a valid, non-null, writable pointer.
///
/// # Examples
///
/// ```
/// use hurray_ffi::buffer_list::{
///     hurray_buffer_list_destroy, hurray_buffer_list_get, hurray_buffer_list_new,
/// };
/// use hurray_ffi::status::HURRAY_ERR_INDEX_OUT_OF_BOUNDS;
/// use hurray_ffi::{HurrayBuffer, HurrayBufferList};
///
/// let mut list: *mut HurrayBufferList = std::ptr::null_mut();
/// unsafe { hurray_buffer_list_new(0, &mut list) };
///
/// // An empty list has no index 0.
/// let mut got: *mut HurrayBuffer = std::ptr::null_mut();
/// assert_eq!(
///     unsafe { hurray_buffer_list_get(list, 0, &mut got) },
///     HURRAY_ERR_INDEX_OUT_OF_BOUNDS
/// );
///
/// unsafe { hurray_buffer_list_destroy(&mut list) };
/// ```
#[no_mangle]
pub unsafe extern "C" fn hurray_buffer_list_get(
    list: *const HurrayBufferList,
    index: u64,
    out_buffer: *mut *mut HurrayBuffer,
) -> HurrayStatus {
    catch(|| {
        null_check!(list, out_buffer);

        // SAFETY: list is non-null and points to a live HurrayBufferList.
        let buffers = &(*list).buffers;
        let Ok(idx) = usize::try_from(index) else {
            return HURRAY_ERR_INDEX_OUT_OF_BOUNDS;
        };
        match buffers.get(idx) {
            // A nulled slot means the list is mid-destroy; treat it as absent
            // rather than handing back a dangling handle.
            Some(&b) if !b.is_null() => {
                *out_buffer = b;
                HURRAY_OK
            }
            _ => HURRAY_ERR_INDEX_OUT_OF_BOUNDS,
        }
    })
}

// ── Destructor ────────────────────────────────────────────────────────────────

/// Destroys `*list` and every [`HurrayBuffer`] it owns, then writes null through
/// `list`.
///
/// Nulling the caller's pointer is the sound half of Arrow's "release marks the
/// structure released" discipline: the list allocation itself is freed here, so
/// a marker written inside it could not be read back, but the caller's own
/// variable can be invalidated.
///
/// Each owned slot is nulled as its handle is destroyed, so a release callback
/// that panics or re-enters cannot cause a double free. A panicking callback
/// leaks the remainder of the list rather than corrupting it.
///
/// Passing a pointer to a null pointer is a no-op and returns [`HURRAY_OK`],
/// which makes cleanup paths idempotent.
///
/// # Safety
///
/// - `list` MUST be a valid, non-null, writable pointer to a `*mut HurrayBufferList`.
/// - `*list` MUST be a live handle from [`hurray_buffer_list_new`], or null.
///
/// # Examples
///
/// ```
/// use hurray_ffi::buffer_list::{hurray_buffer_list_destroy, hurray_buffer_list_new};
/// use hurray_ffi::{HurrayBufferList, HURRAY_OK};
///
/// let mut list: *mut HurrayBufferList = std::ptr::null_mut();
/// unsafe { hurray_buffer_list_new(0, &mut list) };
///
/// assert_eq!(unsafe { hurray_buffer_list_destroy(&mut list) }, HURRAY_OK);
/// assert!(list.is_null());
///
/// // Idempotent: destroying an already-nulled pointer is a no-op.
/// assert_eq!(unsafe { hurray_buffer_list_destroy(&mut list) }, HURRAY_OK);
/// ```
#[no_mangle]
pub unsafe extern "C" fn hurray_buffer_list_destroy(
    list: *mut *mut HurrayBufferList,
) -> HurrayStatus {
    catch(|| {
        null_check!(list);

        // SAFETY: list is a valid, writable pointer to a handle slot.
        let ptr = *list;
        if ptr.is_null() {
            return HURRAY_OK;
        }

        #[cfg(debug_assertions)]
        {
            // SAFETY: ptr is non-null and points to a live HurrayBufferList.
            if (*ptr).sentinel == SENTINEL_DEAD {
                return HURRAY_ERR_INTERNAL;
            }
            // Written through the raw pointer before Box::from_raw so the freed
            // slot carries the dead marker; writing after would only update the
            // moved-out local copy.
            (*ptr).sentinel = SENTINEL_DEAD;
        }

        // Destroy each owned handle, nulling its slot first so a re-entrant or
        // panicking release callback can never see a destroyable handle twice.
        // SAFETY: ptr is non-null and points to a live HurrayBufferList.
        for slot in (*ptr).buffers.iter_mut() {
            let buffer = std::mem::replace(slot, std::ptr::null_mut());
            if !buffer.is_null() {
                // SAFETY: the list owns this handle, it is non-null, and this is
                // its first and only destroy.
                hurray_buffer_destroy(buffer);
            }
        }

        // SAFETY: created by hurray_buffer_list_new via Box::into_raw; this is
        // the first and only destroy.
        drop(Box::from_raw(ptr));

        // The caller's variable is now observably dead.
        *list = std::ptr::null_mut();
        HURRAY_OK
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{hurray_buffer_byte_size, hurray_buffer_from_ptr};
    use crate::status::HURRAY_ERR_NULL_POINTER;
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[repr(align(64))]
    struct Aligned([u8; 128]);

    /// Counts releases through the caller-supplied context pointer rather than a
    /// global: cargo runs tests in parallel, so a shared counter would be raced by
    /// any other test that destroys a buffer.
    unsafe extern "C" fn counting_release(_data: *mut c_void, ctx: *mut c_void) {
        if !ctx.is_null() {
            (*(ctx as *const AtomicUsize)).fetch_add(1, Ordering::SeqCst);
        }
    }

    fn make_buffer(data: &mut Aligned, size: u64, releases: &AtomicUsize) -> *mut HurrayBuffer {
        let mut buffer: *mut HurrayBuffer = std::ptr::null_mut();
        let status = unsafe {
            hurray_buffer_from_ptr(
                data.0.as_mut_ptr().cast(),
                size,
                64,
                0,
                0,
                0,
                Some(counting_release),
                releases as *const AtomicUsize as *mut c_void,
                &mut buffer,
            )
        };
        assert_eq!(status, HURRAY_OK);
        buffer
    }

    #[test]
    fn new_and_destroy_nulls_the_caller_pointer() {
        let mut list: *mut HurrayBufferList = std::ptr::null_mut();
        assert_eq!(unsafe { hurray_buffer_list_new(4, &mut list) }, HURRAY_OK);
        assert!(!list.is_null());
        assert_eq!(unsafe { hurray_buffer_list_destroy(&mut list) }, HURRAY_OK);
        assert!(list.is_null());
    }

    #[test]
    fn destroy_is_idempotent_on_a_null_slot() {
        let mut list: *mut HurrayBufferList = std::ptr::null_mut();
        assert_eq!(unsafe { hurray_buffer_list_destroy(&mut list) }, HURRAY_OK);
    }

    #[test]
    fn null_arguments_are_rejected() {
        assert_eq!(
            unsafe { hurray_buffer_list_new(0, std::ptr::null_mut()) },
            HURRAY_ERR_NULL_POINTER
        );
        assert_eq!(
            unsafe { hurray_buffer_list_destroy(std::ptr::null_mut()) },
            HURRAY_ERR_NULL_POINTER
        );
        let mut len = 0u64;
        assert_eq!(
            unsafe { hurray_buffer_list_len(std::ptr::null(), &mut len) },
            HURRAY_ERR_NULL_POINTER
        );
    }

    #[test]
    fn push_then_len_and_get_round_trip_in_order() {
        let mut a = Aligned([1u8; 128]);
        let mut b = Aligned([2u8; 128]);
        let releases = AtomicUsize::new(0);
        let buf_a = make_buffer(&mut a, 128, &releases);
        let buf_b = make_buffer(&mut b, 64, &releases);

        let mut list: *mut HurrayBufferList = std::ptr::null_mut();
        unsafe { hurray_buffer_list_new(2, &mut list) };
        assert_eq!(unsafe { hurray_buffer_list_push(list, buf_a) }, HURRAY_OK);
        assert_eq!(unsafe { hurray_buffer_list_push(list, buf_b) }, HURRAY_OK);

        let mut len = 0u64;
        unsafe { hurray_buffer_list_len(list, &mut len) };
        assert_eq!(len, 2);

        // Order is push order, i.e. descriptor buffer-table order: the 128-byte
        // buffer is index 0 and the 64-byte one is index 1.
        for (index, expected) in [(0u64, 128u64), (1, 64)] {
            let mut got: *mut HurrayBuffer = std::ptr::null_mut();
            assert_eq!(
                unsafe { hurray_buffer_list_get(list, index, &mut got) },
                HURRAY_OK
            );
            let mut size = 0u64;
            unsafe { hurray_buffer_byte_size(got, &mut size) };
            assert_eq!(size, expected);
        }

        unsafe { hurray_buffer_list_destroy(&mut list) };
    }

    #[test]
    fn get_rejects_an_out_of_range_index() {
        let mut a = Aligned([0u8; 128]);
        let releases = AtomicUsize::new(0);
        let buf = make_buffer(&mut a, 128, &releases);
        let mut list: *mut HurrayBufferList = std::ptr::null_mut();
        unsafe { hurray_buffer_list_new(1, &mut list) };
        unsafe { hurray_buffer_list_push(list, buf) };

        let mut got: *mut HurrayBuffer = std::ptr::null_mut();
        assert_eq!(
            unsafe { hurray_buffer_list_get(list, 1, &mut got) },
            HURRAY_ERR_INDEX_OUT_OF_BOUNDS
        );
        assert_eq!(
            unsafe { hurray_buffer_list_get(list, u64::MAX, &mut got) },
            HURRAY_ERR_INDEX_OUT_OF_BOUNDS
        );

        unsafe { hurray_buffer_list_destroy(&mut list) };
    }

    #[test]
    fn destroying_the_list_releases_every_owned_buffer_exactly_once() {
        let releases = AtomicUsize::new(0);
        let mut a = Aligned([0u8; 128]);
        let mut b = Aligned([0u8; 128]);
        let mut c = Aligned([0u8; 128]);

        let mut list: *mut HurrayBufferList = std::ptr::null_mut();
        unsafe { hurray_buffer_list_new(3, &mut list) };
        for data in [&mut a, &mut b, &mut c] {
            let buf = make_buffer(data, 128, &releases);
            unsafe { hurray_buffer_list_push(list, buf) };
        }

        unsafe { hurray_buffer_list_destroy(&mut list) };
        assert_eq!(releases.load(Ordering::SeqCst), 3);
        assert!(list.is_null());
    }
}
