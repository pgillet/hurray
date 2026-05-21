//! Opaque buffer handle for the Hurray C ABI.
//!
//! [`HurrayBuffer`] is an opaque heap-allocated handle that wraps a
//! [`hurray_core::BufferHandle`] together with the raw data pointer and an
//! optional release callback. Callers create handles via
//! [`hurray_buffer_from_ptr`] and MUST destroy them exactly once via
//! [`hurray_buffer_destroy`].

use std::ffi::c_void;

use hurray_core::{BufferHandle, DeviceTag, MemoryClass, SyncMode};

use crate::{
    panic::{catch, null_check},
    status::{status_from_core_error, HurrayStatus, HURRAY_ERR_INTERNAL, HURRAY_OK},
};

// ── Debug sentinel constants ──────────────────────────────────────────────────

/// Sentinel placed in `HurrayBuffer.sentinel` on allocation (debug builds only).
#[cfg(debug_assertions)]
const SENTINEL_LIVE: u64 = 0xDEAD_BEEF_CAFE_BABE;

/// Sentinel placed in `HurrayBuffer.sentinel` after `hurray_buffer_destroy`
/// to detect double-free in debug builds.
#[cfg(debug_assertions)]
const SENTINEL_DEAD: u64 = 0x0000_DEAD_0000_DEAD;

// ── Release callback type ─────────────────────────────────────────────────────

/// Optional release callback invoked by [`hurray_buffer_destroy`].
///
/// The first argument is the `data` pointer passed to [`hurray_buffer_from_ptr`].
/// The second argument is the `release_context` pointer passed to the same
/// function. Both arguments carry the exact values provided at construction time.
///
/// Implementations MUST NOT call back into the Hurray C ABI from within this
/// callback; doing so may cause deadlocks or use-after-free.
pub type HurrayReleaseCallback = Option<unsafe extern "C" fn(*mut c_void, *mut c_void)>;

// ── HurrayBuffer ─────────────────────────────────────────────────────────────

/// Opaque handle to a buffer registered with the Hurray C ABI.
///
/// This struct is **not** `#[repr(C)]`; its internal layout is an
/// implementation detail. Callers MUST treat it as a black-box opaque
/// pointer. The handle is heap-allocated by [`hurray_buffer_from_ptr`] and
/// MUST be released exactly once by [`hurray_buffer_destroy`].
pub struct HurrayBuffer {
    /// Double-free sentinel (debug builds only).
    #[cfg(debug_assertions)]
    sentinel: u64,
    /// Raw pointer to the buffer's first byte.
    data: *mut c_void,
    /// Metadata about the buffer's size, alignment, device, and sync mode.
    pub(crate) handle: BufferHandle,
    /// Optional callback invoked during [`hurray_buffer_destroy`] to release
    /// the underlying memory.
    release: HurrayReleaseCallback,
    /// Caller-supplied context pointer forwarded to `release`.
    release_context: *mut c_void,
}

// SAFETY: Single-thread-at-a-time per the C ABI contract; Send allows
// Box<HurrayBuffer> to move across threads during destruction without any
// concurrent access — the C API has no shared-state concurrency model.
unsafe impl Send for HurrayBuffer {}

// ── Constructor ───────────────────────────────────────────────────────────────

/// Creates a new [`HurrayBuffer`] handle from a raw data pointer and metadata.
///
/// The caller retains ownership of `data` until the handle is destroyed via
/// [`hurray_buffer_destroy`], at which point the `release` callback (if
/// non-null) is invoked with `(data, release_context)`. If `release` is null,
/// the caller is responsible for freeing `data` after destroying the handle.
///
/// # Arguments
///
/// - `data` — Non-null pointer to the first byte of the buffer. MUST be
///   aligned to at least `alignment` bytes.
/// - `byte_size` — Length of the buffer in bytes.
/// - `alignment` — Declared alignment of `data` in bytes; MUST be a power of
///   two and at least [`hurray_core::MIN_BUFFER_ALIGNMENT`] for non-empty
///   buffers.
/// - `device_tag` — Wire byte identifying the memory space (see
///   `docs/spec/buffer-protocol.md § Device Tags`).
/// - `sync_mode` — Wire byte for producer/consumer ordering (see
///   `docs/spec/buffer-protocol.md § Synchronization Mode`).
/// - `memory_class` — Wire byte for memory accessibility class.
/// - `release` — Optional callback invoked on [`hurray_buffer_destroy`].
/// - `release_context` — Opaque context pointer forwarded to `release`; MAY
///   be null.
/// - `out_handle` — Non-null pointer to a `*mut HurrayBuffer`; set to the
///   newly allocated handle on success.
///
/// # Safety
///
/// - `data` MUST be a valid, non-null pointer to at least `byte_size` bytes.
/// - `data` MUST remain valid until the release callback is called.
/// - `out_handle` MUST be a valid, non-null, writable pointer.
/// - `release_context` lifetime is caller-managed; it MUST remain valid until
///   the release callback returns.
#[no_mangle]
pub unsafe extern "C" fn hurray_buffer_from_ptr(
    data: *mut c_void,
    byte_size: u64,
    alignment: u32,
    device_tag: u8,
    sync_mode: u8,
    memory_class: u8,
    release: HurrayReleaseCallback,
    release_context: *mut c_void,
    out_handle: *mut *mut HurrayBuffer,
) -> HurrayStatus {
    catch(|| {
        // release_context may be null — it is caller-controlled context.
        null_check!(data, out_handle);

        let device = match DeviceTag::from_byte(device_tag) {
            Ok(d) => d,
            Err(e) => return status_from_core_error(&e),
        };
        let mode = match SyncMode::from_byte(sync_mode) {
            Ok(m) => m,
            Err(e) => return status_from_core_error(&e),
        };
        let class = match MemoryClass::from_byte(memory_class) {
            Ok(c) => c,
            Err(e) => return status_from_core_error(&e),
        };
        let handle =
            match BufferHandle::with_memory_class(byte_size, alignment, device, mode, class) {
                Ok(h) => h,
                Err(e) => return status_from_core_error(&e),
            };

        let boxed = Box::new(HurrayBuffer {
            #[cfg(debug_assertions)]
            sentinel: SENTINEL_LIVE,
            data,
            handle,
            release,
            release_context,
        });
        // SAFETY: we just allocated `boxed`; Box::into_raw transfers ownership
        // to the caller, who must call hurray_buffer_destroy exactly once.
        *out_handle = Box::into_raw(boxed);
        HURRAY_OK
    })
}

// ── Destructor ────────────────────────────────────────────────────────────────

/// Destroys a [`HurrayBuffer`] handle and invokes the release callback.
///
/// After this call, `buffer` is no longer valid and MUST NOT be dereferenced.
/// In debug builds, a double-free attempt returns [`HURRAY_ERR_INTERNAL`].
///
/// # Safety
///
/// - `buffer` MUST have been created by [`hurray_buffer_from_ptr`].
/// - `buffer` MUST NOT have been previously destroyed.
#[no_mangle]
pub unsafe extern "C" fn hurray_buffer_destroy(buffer: *mut HurrayBuffer) -> HurrayStatus {
    catch(|| {
        null_check!(buffer);

        #[cfg(debug_assertions)]
        {
            // SAFETY: buffer is non-null and points to a live HurrayBuffer;
            // reading the sentinel before Box::from_raw for double-free detection.
            if (*buffer).sentinel == SENTINEL_DEAD {
                return HURRAY_ERR_INTERNAL;
            }
            // Write SENTINEL_DEAD through the raw pointer *before* Box::from_raw
            // so the freed allocation slot carries the dead marker.  Writing
            // after Box::from_raw only updates the moved-out local copy, not the
            // original heap slot that the allocator may inspect on the second call.
            (*buffer).sentinel = SENTINEL_DEAD;
        }

        // SAFETY: created by hurray_buffer_from_ptr via Box::into_raw; caller
        // guarantees this is the first and only destroy call.
        let mut b = Box::from_raw(buffer);

        // SAFETY: caller provided `release` and `release_context` with valid
        // lifetimes; we invoke the callback exactly once at destruction time.
        if let Some(f) = b.release.take() {
            f(b.data, b.release_context);
        }

        HURRAY_OK
    })
}

// ── Accessors ─────────────────────────────────────────────────────────────────

/// Reads the byte size of a buffer handle.
///
/// # Safety
///
/// `buffer` and `out_byte_size` MUST be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn hurray_buffer_byte_size(
    buffer: *const HurrayBuffer,
    out_byte_size: *mut u64,
) -> HurrayStatus {
    catch(|| {
        null_check!(buffer, out_byte_size);
        // SAFETY: buffer is non-null and points to a live HurrayBuffer.
        *out_byte_size = (*buffer).handle.byte_size();
        HURRAY_OK
    })
}

/// Reads the declared alignment of a buffer handle (in bytes).
///
/// # Safety
///
/// `buffer` and `out_alignment` MUST be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn hurray_buffer_alignment(
    buffer: *const HurrayBuffer,
    out_alignment: *mut u32,
) -> HurrayStatus {
    catch(|| {
        null_check!(buffer, out_alignment);
        // SAFETY: buffer is non-null and points to a live HurrayBuffer.
        *out_alignment = (*buffer).handle.alignment();
        HURRAY_OK
    })
}

/// Reads the device tag wire byte of a buffer handle.
///
/// # Safety
///
/// `buffer` and `out_device_tag` MUST be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn hurray_buffer_device_tag(
    buffer: *const HurrayBuffer,
    out_device_tag: *mut u8,
) -> HurrayStatus {
    catch(|| {
        null_check!(buffer, out_device_tag);
        // SAFETY: buffer is non-null and points to a live HurrayBuffer.
        *out_device_tag = (*buffer).handle.device_tag().to_byte();
        HURRAY_OK
    })
}

/// Reads the sync mode wire byte of a buffer handle.
///
/// # Safety
///
/// `buffer` and `out_sync_mode` MUST be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn hurray_buffer_sync_mode(
    buffer: *const HurrayBuffer,
    out_sync_mode: *mut u8,
) -> HurrayStatus {
    catch(|| {
        null_check!(buffer, out_sync_mode);
        // SAFETY: buffer is non-null and points to a live HurrayBuffer.
        *out_sync_mode = (*buffer).handle.sync_mode().to_byte();
        HURRAY_OK
    })
}

/// Reads the memory class wire byte of a buffer handle.
///
/// # Safety
///
/// `buffer` and `out_memory_class` MUST be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn hurray_buffer_memory_class(
    buffer: *const HurrayBuffer,
    out_memory_class: *mut u8,
) -> HurrayStatus {
    catch(|| {
        null_check!(buffer, out_memory_class);
        // SAFETY: buffer is non-null and points to a live HurrayBuffer.
        *out_memory_class = (*buffer).handle.memory_class().to_byte();
        HURRAY_OK
    })
}

/// Reads the raw data pointer stored in a buffer handle.
///
/// # Safety
///
/// `buffer` and `out_data` MUST be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn hurray_buffer_data_ptr(
    buffer: *const HurrayBuffer,
    out_data: *mut *mut c_void,
) -> HurrayStatus {
    catch(|| {
        null_check!(buffer, out_data);
        // SAFETY: buffer is non-null and points to a live HurrayBuffer.
        *out_data = (*buffer).data;
        HURRAY_OK
    })
}
