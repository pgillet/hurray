//! Blocking stream writes over a caller-supplied transport (ADR-041 §§ 3, 8).
//!
//! The write core never assembles payloads: it hands the driver the caller's own slices,
//! one at a time, and this module passes each straight to the write callback. The rule
//! that the core must not retain a payload pointer past the item costs a blocking surface
//! nothing — the whole operation lives and dies inside one call.

use hurray_io::stream::{StreamWriterCore, WriteStep};

use crate::{
    descriptor::HurrayDescriptor,
    panic::{catch, null_check},
    status::{
        status_from_io_error, HurrayStatus, HURRAY_ERR_INTERNAL, HURRAY_ERR_NULL_POINTER,
        HURRAY_ERR_SINK_CLOSED, HURRAY_OK,
    },
    stream::{transport::Transport, HurrayTransport},
};

// ── Options and payloads ──────────────────────────────────────────────────────

/// Transport assumptions for a writer.
///
/// Pass null to [`hurray_writer_new`] for the defaults.
#[repr(C)]
pub struct HurrayWriterOptions {
    /// Refuse to emit a buffer whose sync mode is not `ProducerSynced`. Set this when the
    /// stream crosses a machine boundary, where a device-local synchronization handle
    /// means nothing to the far end.
    pub enforce_cross_machine_sync: bool,
}

/// One borrowed payload: a pointer and a length.
///
/// Borrowed for the duration of the [`hurray_writer_write_tensor`] call and not retained
/// afterwards. A zero-length slice MAY carry a null `ptr`.
#[repr(C)]
pub struct HurrayIoSlice {
    /// First byte of the payload.
    pub ptr: *const u8,
    /// Payload length in bytes. MUST equal the declared `byte_size` of the buffer handle
    /// at the same index of the descriptor's buffer table.
    pub len: usize,
}

// ── HurrayWriter ──────────────────────────────────────────────────────────────

/// Opaque handle to a blocking stream writer.
///
/// Not `#[repr(C)]`; the layout is an implementation detail. Created by
/// [`hurray_writer_new`] and released exactly once by [`hurray_writer_destroy`].
///
/// One handle, one thread at a time.
pub struct HurrayWriter {
    core: StreamWriterCore,
    transport: Transport,
}

// ── Construction / destruction ────────────────────────────────────────────────

/// Creates a writer over `transport`.
///
/// The vtable is copied, so the caller MAY free the [`HurrayTransport`] struct immediately;
/// its `ctx` must stay valid until the writer is destroyed, at which point the vtable's
/// release callback is invoked.
///
/// # Arguments
///
/// - `transport` — Non-null; its `write` callback MUST be non-null.
/// - `options` — MAY be null for [`HurrayWriterOptions`]' documented defaults.
/// - `out_handle` — Non-null pointer to a `*mut HurrayWriter`; set on success.
///
/// # Safety
///
/// - `transport` MUST point to a readable [`HurrayTransport`] for the duration of the call.
/// - `options`, if non-null, MUST point to a readable [`HurrayWriterOptions`].
/// - `out_handle` MUST be a valid, non-null, writable pointer.
#[no_mangle]
pub unsafe extern "C" fn hurray_writer_new(
    transport: *const HurrayTransport,
    options: *const HurrayWriterOptions,
    out_handle: *mut *mut HurrayWriter,
) -> HurrayStatus {
    catch(|| {
        // options may be null: that is the request for defaults.
        null_check!(transport, out_handle);

        // SAFETY: caller guarantees transport is readable for this call.
        let transport = Transport::from_c(transport);
        if !transport.can_write() {
            return HURRAY_ERR_NULL_POINTER;
        }

        // SAFETY: options is either null or caller-guaranteed readable.
        let cross_machine = !options.is_null() && (*options).enforce_cross_machine_sync;
        let core = if cross_machine {
            StreamWriterCore::cross_machine()
        } else {
            StreamWriterCore::new()
        };

        // SAFETY: Box::into_raw transfers ownership to the caller, who must call
        // hurray_writer_destroy exactly once.
        *out_handle = Box::into_raw(Box::new(HurrayWriter { core, transport }));
        HURRAY_OK
    })
}

/// Destroys a writer, releases its transport context, and writes null through `handle`.
///
/// There is nothing to flush: a tensor is fully on the wire, or the call that wrote it
/// failed. The transport's own close, if it has one, belongs in its release callback.
///
/// # Safety
///
/// - `handle` MUST be a valid, non-null pointer to a `*mut HurrayWriter` from
///   [`hurray_writer_new`].
/// - That writer MUST NOT have been destroyed already.
#[no_mangle]
pub unsafe extern "C" fn hurray_writer_destroy(handle: *mut *mut HurrayWriter) -> HurrayStatus {
    catch(|| {
        null_check!(handle);
        let writer = *handle;
        if writer.is_null() {
            return HURRAY_ERR_INTERNAL;
        }
        // SAFETY: created by hurray_writer_new via Box::into_raw; the caller guarantees
        // this is the first and only destroy call. Dropping it runs the transport's
        // release callback.
        drop(Box::from_raw(writer));
        *handle = std::ptr::null_mut();
        HURRAY_OK
    })
}

// ── Writing ───────────────────────────────────────────────────────────────────

/// Writes one tensor — its descriptor then its payloads — blocking in the transport's
/// write callback as needed.
///
/// `payloads` MUST have one entry per buffer in the descriptor's buffer table, in that
/// order, each with the declared `byte_size`. The whole tensor is validated before a byte
/// reaches the sink, so a mismatch is reported without putting a torn item on the wire.
///
/// Returns:
///
/// - [`HURRAY_OK`] — the tensor is entirely written and the payload pointers are no longer
///   held.
/// - [`HURRAY_ERR_SINK_CLOSED`] — the sink stopped accepting bytes mid-tensor, so a
///   truncated item is on the wire. The writer is poisoned: every later call fails, and a
///   fresh writer over the same connection is not a recovery.
/// - Whatever status the write callback returned, if it failed **before any byte of this
///   tensor reached the sink**. Nothing was emitted, so the call MAY be retried. A failure
///   after the first byte is reported as [`HURRAY_ERR_SINK_CLOSED`] instead, because at
///   that point the damage is the same.
///
/// Composite tensors are not yet writable from C; a composite head alone is not a valid
/// tensor and is refused.
///
/// # Safety
///
/// - `handle` MUST point to a live [`HurrayWriter`] and `descriptor` to a live
///   [`HurrayDescriptor`].
/// - If `count` is non-zero, `payloads` MUST point to `count` readable [`HurrayIoSlice`]
///   values, each describing memory readable for its own `len` bytes.
/// - All of that memory MUST stay valid and unmodified until this call returns.
#[no_mangle]
pub unsafe extern "C" fn hurray_writer_write_tensor(
    handle: *mut HurrayWriter,
    descriptor: *const HurrayDescriptor,
    payloads: *const HurrayIoSlice,
    count: usize,
) -> HurrayStatus {
    catch(|| {
        null_check!(handle, descriptor);
        if count > 0 {
            null_check!(payloads);
        }

        // SAFETY: non-null and caller-guaranteed to describe `count` readable slices.
        let slices = if count == 0 {
            &[][..]
        } else {
            std::slice::from_raw_parts(payloads, count)
        };

        let mut borrowed: Vec<&[u8]> = Vec::with_capacity(count);
        for slice in slices {
            if slice.len == 0 {
                // An empty slice needs no pointer, and from_raw_parts needs a valid one.
                borrowed.push(&[]);
            } else if slice.ptr.is_null() {
                return HURRAY_ERR_NULL_POINTER;
            } else {
                // SAFETY: caller guarantees `len` readable bytes at `ptr`, unmodified for
                // the duration of this call.
                borrowed.push(std::slice::from_raw_parts(slice.ptr, slice.len));
            }
        }

        // SAFETY: both pointers are non-null and point to live handles, which the C ABI's
        // one-handle-one-thread rule says nothing else is touching.
        let HurrayWriter { core, transport } = &mut *handle;
        let descriptor = (*descriptor).inner();

        let mut op = match core.tensor(descriptor, &borrowed) {
            Ok(op) => op,
            Err(e) => return status_from_io_error(&e),
        };

        loop {
            match op.step() {
                Err(e) => return status_from_io_error(&e),
                Ok(WriteStep::ItemWritten) => return HURRAY_OK,
                Ok(WriteStep::Output { .. }) => {
                    let src = op.output_src();
                    match transport.write(src) {
                        Ok(0) => {
                            // § 4: zero accepted bytes is never progress.
                            let e = op.report_sink_closed();
                            return status_from_io_error(&e);
                        }
                        Ok(n) => {
                            if let Err(e) = op.report_written(n) {
                                return status_from_io_error(&e);
                            }
                        }
                        Err(status) => {
                            // No `report_sink_closed` here: the core poisoned itself at
                            // this item's first accepted byte (§ 6b), and calling it would
                            // overwrite that with `LostWriteCount` — claiming the count is
                            // unknown when it is known exactly.
                            if op.emitted() > 0 {
                                // A torn item: the transport's own status understates it.
                                return HURRAY_ERR_SINK_CLOSED;
                            }
                            // Nothing of this item is on the wire, so the callback's status
                            // is the more useful of the two — it says why the stream stopped.
                            return status;
                        }
                    }
                }
            }
        }
    })
}
