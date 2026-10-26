//! A tensor handle: a descriptor and its buffers, paired structurally.
//!
//! A tensor *is* a descriptor plus its buffers, and until now the C ABI made callers keep
//! the two halves paired by convention — a consumer that fetched one without the other
//! got either bytes it could not interpret or a description of bytes it did not have
//! ([#174](https://github.com/pgillet/hurray/issues/174)). A stream read produces exactly
//! that pair, so it produces this handle.
//!
//! One destroy call, and its internal order cannot be got wrong.

use std::ffi::c_void;

use bytes::Bytes;
use hurray_core::TensorDescriptor;

use crate::{
    buffer::HurrayBuffer,
    buffer_list::{hurray_buffer_list_destroy, HurrayBufferList},
    descriptor::{hurray_descriptor_destroy, HurrayDescriptor},
    panic::{catch, null_check},
    status::{HurrayStatus, HURRAY_ERR_INTERNAL, HURRAY_OK},
};

// ── Debug sentinel constants ──────────────────────────────────────────────────

/// Sentinel placed in `HurrayTensor.sentinel` on allocation (debug builds only).
#[cfg(debug_assertions)]
const SENTINEL_LIVE: u64 = 0x7E43_0000_7E43_0000;

/// Sentinel written before the allocation is freed, to detect a double destroy in debug
/// builds.
#[cfg(debug_assertions)]
const SENTINEL_DEAD: u64 = 0x7E43_DEAD_7E43_DEAD;

// ── HurrayTensor ──────────────────────────────────────────────────────────────

/// Opaque handle to a tensor: its descriptor and its buffers.
///
/// Not `#[repr(C)]`; the layout is an implementation detail. The handle **owns** both
/// halves. [`hurray_tensor_descriptor`] and [`hurray_tensor_buffers`] hand out *borrowed*
/// pointers — valid until [`hurray_tensor_destroy`], and never to be destroyed
/// individually.
pub struct HurrayTensor {
    /// Double-destroy sentinel (debug builds only).
    #[cfg(debug_assertions)]
    sentinel: u64,
    /// Owned; nulled as soon as it is destroyed so a re-entrant destroy cannot free twice.
    descriptor: *mut HurrayDescriptor,
    /// Owned, in descriptor buffer-table order.
    buffers: *mut HurrayBufferList,
}

// SAFETY: mirrors the HurrayBuffer and HurrayBufferList contract — the C ABI is
// single-thread-at-a-time per handle, so Send only permits moving the box across threads
// during construction and destruction, never concurrent access.
unsafe impl Send for HurrayTensor {}

impl HurrayTensor {
    /// Takes ownership of a descriptor handle and a buffer list.
    pub(crate) fn from_parts(
        descriptor: *mut HurrayDescriptor,
        buffers: *mut HurrayBufferList,
    ) -> Box<Self> {
        Box::new(Self {
            #[cfg(debug_assertions)]
            sentinel: SENTINEL_LIVE,
            descriptor,
            buffers,
        })
    }
}

// ── Accessors ─────────────────────────────────────────────────────────────────

/// Borrows the tensor's descriptor handle.
///
/// The returned handle carries every accessor in [`crate::descriptor`] and
/// [`crate::field`]. It is **borrowed**: the caller MUST NOT call
/// [`hurray_descriptor_destroy`] on it.
///
/// # Safety
///
/// `handle` and `out_descriptor` MUST be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn hurray_tensor_descriptor(
    handle: *const HurrayTensor,
    out_descriptor: *mut *mut HurrayDescriptor,
) -> HurrayStatus {
    catch(|| {
        null_check!(handle, out_descriptor);
        // SAFETY: handle is non-null and points to a live HurrayTensor.
        *out_descriptor = (*handle).descriptor;
        HURRAY_OK
    })
}

/// Borrows the tensor's buffer list.
///
/// Element `i` of the list is buffer `i` of the descriptor's buffer table. The list is
/// **borrowed**: the caller MUST NOT call [`hurray_buffer_list_destroy`] on it.
///
/// # Safety
///
/// `handle` and `out_buffers` MUST be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn hurray_tensor_buffers(
    handle: *const HurrayTensor,
    out_buffers: *mut *mut HurrayBufferList,
) -> HurrayStatus {
    catch(|| {
        null_check!(handle, out_buffers);
        // SAFETY: handle is non-null and points to a live HurrayTensor.
        *out_buffers = (*handle).buffers;
        HURRAY_OK
    })
}

// ── Destructor ────────────────────────────────────────────────────────────────

/// Destroys a tensor handle, its descriptor and every buffer it owns, then writes null
/// through `handle`.
///
/// One call rather than two, so the teardown order is not the caller's problem.
///
/// # Safety
///
/// - `handle` MUST be a valid, non-null pointer to a `*mut HurrayTensor` produced by
///   [`hurray_reader_next_tensor`][crate::hurray_reader_next_tensor].
/// - That tensor MUST NOT have been destroyed already.
#[no_mangle]
pub unsafe extern "C" fn hurray_tensor_destroy(handle: *mut *mut HurrayTensor) -> HurrayStatus {
    catch(|| {
        null_check!(handle);
        let tensor = *handle;
        if tensor.is_null() {
            return HURRAY_ERR_INTERNAL;
        }

        #[cfg(debug_assertions)]
        {
            // SAFETY: tensor is non-null and points to a live HurrayTensor. Read and mark
            // through the raw pointer before Box::from_raw, so the dead marker lands in
            // the heap slot a second destroy would inspect.
            if (*tensor).sentinel == SENTINEL_DEAD {
                return HURRAY_ERR_INTERNAL;
            }
            (*tensor).sentinel = SENTINEL_DEAD;
        }

        // SAFETY: produced by HurrayTensor::from_parts via Box::into_raw; the caller
        // guarantees this is the first and only destroy call.
        let mut boxed = Box::from_raw(tensor);

        let descriptor = std::mem::replace(&mut boxed.descriptor, std::ptr::null_mut());
        if !descriptor.is_null() {
            // SAFETY: owned by this tensor and destroyed exactly once.
            hurray_descriptor_destroy(descriptor);
        }
        let mut buffers = std::mem::replace(&mut boxed.buffers, std::ptr::null_mut());
        if !buffers.is_null() {
            // SAFETY: owned by this tensor and destroyed exactly once; the list destroys
            // every handle it owns.
            hurray_buffer_list_destroy(&mut buffers);
        }

        drop(boxed);
        *handle = std::ptr::null_mut();
        HURRAY_OK
    })
}

// ── Handing a decoded tensor to C ─────────────────────────────────────────────

/// Releases one buffer's backing `Bytes` when its [`HurrayBuffer`] is destroyed.
///
/// # Safety
///
/// `context` MUST be the `Box<Bytes>` leaked by [`tensor_handle`] for this buffer, and
/// MUST NOT have been released already.
unsafe extern "C" fn release_bytes(_data: *mut c_void, context: *mut c_void) {
    if !context.is_null() {
        // SAFETY: context is the Box<Bytes> leaked for this buffer; hurray_buffer_destroy
        // invokes a release callback exactly once.
        drop(Box::from_raw(context as *mut Bytes));
    }
}

/// Wraps a decoded tensor in the handles a C caller can hold.
///
/// No payload is copied: each `HurrayBuffer` points into the destination the core
/// allocated and the transport filled, and owns a reference to it. Shared by the stream
/// and file readers — a decoded tensor is the same thing whichever core produced it.
pub(crate) fn tensor_handle(
    descriptor: TensorDescriptor,
    buffers: Vec<Bytes>,
) -> Box<HurrayTensor> {
    let mut handles: Vec<*mut HurrayBuffer> = Vec::with_capacity(buffers.len());
    for (bytes, handle) in buffers.into_iter().zip(descriptor.buffers.iter().copied()) {
        let boxed = Box::new(bytes);
        let data = boxed.as_ptr() as *mut c_void;
        let context = Box::into_raw(boxed) as *mut c_void;
        handles.push(Box::into_raw(HurrayBuffer::from_parts(
            data,
            handle,
            Some(release_bytes),
            context,
        )));
    }

    let list = Box::into_raw(HurrayBufferList::from_handles(handles));
    let descriptor = Box::into_raw(HurrayDescriptor::from_descriptor(descriptor));
    HurrayTensor::from_parts(descriptor, list)
}
