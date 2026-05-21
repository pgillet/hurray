//! Opaque tensor descriptor handle for the Hurray C ABI.
//!
//! [`HurrayDescriptor`] is an opaque heap-allocated handle wrapping a decoded
//! [`hurray_core::TensorDescriptor`]. Callers create handles by decoding raw
//! bytes via [`hurray_descriptor_decode`] and MUST release them exactly once
//! via [`hurray_descriptor_destroy`].

use std::ptr;

use hurray_core::TensorDescriptor;

use crate::{
    panic::{catch, null_check},
    status::{status_from_core_error, HurrayStatus, HURRAY_ERR_BUFFER_TOO_SMALL, HURRAY_OK},
};

// ── HurrayDescriptor ──────────────────────────────────────────────────────────

/// Opaque handle to a decoded tensor descriptor.
///
/// This struct is **not** `#[repr(C)]`; its layout is an implementation
/// detail. Callers MUST treat it as a black-box opaque pointer created by
/// [`hurray_descriptor_decode`] and destroyed by [`hurray_descriptor_destroy`].
pub struct HurrayDescriptor(Box<TensorDescriptor>);

// ── Decode / destroy ──────────────────────────────────────────────────────────

/// Decodes a binary tensor descriptor from a byte buffer.
///
/// On success, `*out_handle` is set to a newly allocated [`HurrayDescriptor`]
/// that MUST be released via [`hurray_descriptor_destroy`].
///
/// # Arguments
///
/// - `bytes` — Non-null pointer to the first byte of the encoded descriptor.
/// - `len` — Number of bytes readable at `bytes`.
/// - `out_handle` — Non-null pointer to a `*mut HurrayDescriptor`; set on
///   success.
///
/// # Safety
///
/// - `bytes` MUST be valid and readable for `len` bytes for the duration of
///   this call.
/// - `out_handle` MUST be a valid, non-null, writable pointer.
#[no_mangle]
pub unsafe extern "C" fn hurray_descriptor_decode(
    bytes: *const u8,
    len: usize,
    out_handle: *mut *mut HurrayDescriptor,
) -> HurrayStatus {
    catch(|| {
        null_check!(bytes, out_handle);

        // SAFETY: caller guarantees bytes is valid for len bytes with lifetime
        // at least as long as this call.
        let slice = std::slice::from_raw_parts(bytes, len);

        let desc = match TensorDescriptor::decode(slice) {
            Ok(d) => d,
            Err(e) => return status_from_core_error(&e),
        };

        // SAFETY: Box::into_raw transfers ownership to the caller; they must
        // call hurray_descriptor_destroy exactly once.
        *out_handle = Box::into_raw(Box::new(HurrayDescriptor(Box::new(desc))));
        HURRAY_OK
    })
}

/// Destroys a [`HurrayDescriptor`] handle and frees its memory.
///
/// After this call, `handle` is no longer valid and MUST NOT be dereferenced.
///
/// # Safety
///
/// - `handle` MUST have been created by [`hurray_descriptor_decode`].
/// - `handle` MUST NOT have been previously destroyed.
#[no_mangle]
pub unsafe extern "C" fn hurray_descriptor_destroy(handle: *mut HurrayDescriptor) -> HurrayStatus {
    catch(|| {
        null_check!(handle);
        // SAFETY: created by hurray_descriptor_decode via Box::into_raw; caller
        // guarantees this is the first and only destroy call.
        drop(Box::from_raw(handle));
        HURRAY_OK
    })
}

// ── Scalar accessors ──────────────────────────────────────────────────────────

/// Reads the rank (number of dimensions) of the tensor descriptor.
///
/// # Safety
///
/// `handle` and `out_rank` MUST be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn hurray_descriptor_rank(
    handle: *const HurrayDescriptor,
    out_rank: *mut u32,
) -> HurrayStatus {
    catch(|| {
        null_check!(handle, out_rank);
        // SAFETY: handle is non-null and points to a live HurrayDescriptor.
        *out_rank = (*handle).0.shape.rank() as u32;
        HURRAY_OK
    })
}

/// Reads the element type tag byte of the tensor descriptor.
///
/// # Safety
///
/// `handle` and `out_tag` MUST be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn hurray_descriptor_element_type_tag(
    handle: *const HurrayDescriptor,
    out_tag: *mut u8,
) -> HurrayStatus {
    catch(|| {
        null_check!(handle, out_tag);
        // SAFETY: handle is non-null and points to a live HurrayDescriptor.
        *out_tag = (*handle).0.element_type.tag();
        HURRAY_OK
    })
}

/// Reads the layout tag byte of the tensor descriptor.
///
/// # Safety
///
/// `handle` and `out_tag` MUST be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn hurray_descriptor_layout_tag(
    handle: *const HurrayDescriptor,
    out_tag: *mut u8,
) -> HurrayStatus {
    catch(|| {
        null_check!(handle, out_tag);
        // SAFETY: handle is non-null and points to a live HurrayDescriptor.
        *out_tag = (*handle).0.layout.tag();
        HURRAY_OK
    })
}

/// Reads the byte offset from the start of buffer 0 to logical element `[0,…,0]`.
///
/// # Safety
///
/// `handle` and `out_offset` MUST be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn hurray_descriptor_byte_offset(
    handle: *const HurrayDescriptor,
    out_offset: *mut u64,
) -> HurrayStatus {
    catch(|| {
        null_check!(handle, out_offset);
        // SAFETY: handle is non-null and points to a live HurrayDescriptor.
        *out_offset = (*handle).0.byte_offset;
        HURRAY_OK
    })
}

/// Reads the number of buffer handles in the tensor descriptor's buffer table.
///
/// # Safety
///
/// `handle` and `out_count` MUST be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn hurray_descriptor_buffer_count(
    handle: *const HurrayDescriptor,
    out_count: *mut u32,
) -> HurrayStatus {
    catch(|| {
        null_check!(handle, out_count);
        // SAFETY: handle is non-null and points to a live HurrayDescriptor.
        *out_count = (*handle).0.buffers.len() as u32;
        HURRAY_OK
    })
}

// ── Shape accessor ────────────────────────────────────────────────────────────

/// Reads the shape (dimension sizes) of a tensor descriptor.
///
/// This function uses a **capacity/length in-out pattern**:
///
/// 1. Set `*out_rank` to the capacity of the `out_dims` array (number of
///    `uint64_t` elements it can hold), or to `0` if `out_dims` is null.
/// 2. On return, `*out_rank` always contains the true rank of the tensor.
/// 3. If `out_dims` is null OR the capacity was less than the true rank,
///    the function returns [`HURRAY_ERR_BUFFER_TOO_SMALL`] and `*out_rank`
///    contains the true rank so the caller can allocate and retry.
/// 4. Otherwise the function writes the dimension sizes into `out_dims[0..rank]`
///    and returns [`HURRAY_OK`].
///
/// # Safety
///
/// - `handle` and `out_rank` MUST be valid, non-null pointers.
/// - If `out_dims` is non-null it MUST be writable for `capacity` × 8 bytes,
///   where `capacity` is the value at `*out_rank` on entry.
#[no_mangle]
pub unsafe extern "C" fn hurray_descriptor_shape(
    handle: *const HurrayDescriptor,
    out_dims: *mut u64,
    out_rank: *mut usize,
) -> HurrayStatus {
    catch(|| {
        // out_dims may be null (capacity-query mode); out_rank is always required.
        null_check!(handle, out_rank);

        // SAFETY: handle and out_rank are non-null and point to live memory.
        let shape = &(*handle).0.shape;
        let true_rank = shape.rank();

        // Read the caller-supplied capacity before overwriting *out_rank.
        let capacity = *out_rank;

        // Write the true rank unconditionally so the caller can use it to
        // allocate on a BUFFER_TOO_SMALL retry.
        *out_rank = true_rank;

        if out_dims.is_null() || capacity < true_rank {
            return HURRAY_ERR_BUFFER_TOO_SMALL;
        }

        // SAFETY: out_dims is non-null and writable for at least true_rank
        // elements; shape.dims() has exactly true_rank elements.
        ptr::copy_nonoverlapping(shape.dims().as_ptr(), out_dims, true_rank);
        HURRAY_OK
    })
}
