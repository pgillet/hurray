//! Blocking stream reads over a caller-supplied transport (ADR-041 §§ 7, 8).
//!
//! [`HurrayReader`] is a driver, not a parser: the decoding is `hurray-io`'s sans-io
//! `StreamReaderCore`, and this module only moves bytes between the caller's transport and
//! the destination the core hands it. Which is why there is no runtime here — the core's
//! progress API never blocks, so the blocking happens entirely inside the caller's own
//! read callback.

use hurray_io::stream::{ReadStep, StreamReaderCore, StreamReaderOptions};

use crate::{
    panic::{catch, null_check},
    status::{
        status_from_io_error, HurrayStatus, HURRAY_END_OF_STREAM, HURRAY_ERR_INTERNAL,
        HURRAY_ERR_NULL_POINTER, HURRAY_OK,
    },
    stream::{
        tensor::{tensor_handle, HurrayTensor},
        transport::Transport,
        HurrayTransport,
    },
};

// ── Options ───────────────────────────────────────────────────────────────────

/// Resource limits and transport assumptions for a reader.
///
/// Pass null to [`hurray_reader_new`] for the defaults. The fields map one-to-one onto
/// `hurray-io`'s `StreamReaderOptions`; a reader over an untrusted socket SHOULD set both
/// limits, because a descriptor's declared sizes are attacker-controlled until they are
/// bounded.
#[repr(C)]
pub struct HurrayReaderOptions {
    /// Maximum byte length of a single descriptor. Default: 16 MiB.
    pub max_descriptor_bytes: u64,
    /// Maximum byte length of a single buffer. Default: unbounded.
    pub max_buffer_bytes: u64,
    /// Reject buffers whose sync mode is not `ProducerSynced`. Set this when the stream
    /// crosses a machine boundary, where a device-local synchronization handle means
    /// nothing.
    pub enforce_cross_machine_sync: bool,
}

impl HurrayReaderOptions {
    /// Translates the C struct into `hurray-io`'s options.
    ///
    /// Composite nesting depth is deliberately absent: the read core assembles no
    /// composites, and the C surface does not reach them yet.
    fn to_io(&self) -> StreamReaderOptions {
        StreamReaderOptions {
            max_descriptor_bytes: self.max_descriptor_bytes,
            max_buffer_bytes: self.max_buffer_bytes,
            enforce_cross_machine_sync: self.enforce_cross_machine_sync,
            ..StreamReaderOptions::default()
        }
    }
}

// ── HurrayReader ──────────────────────────────────────────────────────────────

/// Opaque handle to a blocking stream reader.
///
/// Not `#[repr(C)]`; the layout is an implementation detail. Created by
/// [`hurray_reader_new`] and released exactly once by [`hurray_reader_destroy`].
///
/// One handle, one thread at a time.
pub struct HurrayReader {
    core: StreamReaderCore,
    transport: Transport,
}

// ── Construction / destruction ────────────────────────────────────────────────

/// Creates a reader over `transport`.
///
/// The vtable is copied, so the caller MAY free the [`HurrayTransport`] struct immediately;
/// its `ctx` must stay valid until the reader is destroyed, at which point the vtable's
/// release callback is invoked.
///
/// # Arguments
///
/// - `transport` — Non-null; its `read` callback MUST be non-null.
/// - `options` — MAY be null for [`HurrayReaderOptions`]' documented defaults.
/// - `out_handle` — Non-null pointer to a `*mut HurrayReader`; set on success.
///
/// # Safety
///
/// - `transport` MUST point to a readable [`HurrayTransport`] for the duration of the call.
/// - `options`, if non-null, MUST point to a readable [`HurrayReaderOptions`].
/// - `out_handle` MUST be a valid, non-null, writable pointer.
#[no_mangle]
pub unsafe extern "C" fn hurray_reader_new(
    transport: *const HurrayTransport,
    options: *const HurrayReaderOptions,
    out_handle: *mut *mut HurrayReader,
) -> HurrayStatus {
    catch(|| {
        // options may be null: that is the request for defaults.
        null_check!(transport, out_handle);

        // SAFETY: caller guarantees transport is readable for this call.
        let transport = Transport::from_c(transport);
        if !transport.can_read() {
            return HURRAY_ERR_NULL_POINTER;
        }

        let io_options = if options.is_null() {
            StreamReaderOptions::default()
        } else {
            // SAFETY: non-null and caller-guaranteed readable.
            (*options).to_io()
        };

        let reader = HurrayReader {
            core: StreamReaderCore::with_options(io_options),
            transport,
        };
        // SAFETY: Box::into_raw transfers ownership to the caller, who must call
        // hurray_reader_destroy exactly once.
        *out_handle = Box::into_raw(Box::new(reader));
        HURRAY_OK
    })
}

/// Destroys a reader, releases its transport context, and writes null through `handle`.
///
/// Tensors already produced by [`hurray_reader_next_tensor`] are unaffected: each owns its
/// own buffers and outlives the reader.
///
/// # Safety
///
/// - `handle` MUST be a valid, non-null pointer to a `*mut HurrayReader` from
///   [`hurray_reader_new`].
/// - That reader MUST NOT have been destroyed already.
#[no_mangle]
pub unsafe extern "C" fn hurray_reader_destroy(handle: *mut *mut HurrayReader) -> HurrayStatus {
    catch(|| {
        null_check!(handle);
        let reader = *handle;
        if reader.is_null() {
            return HURRAY_ERR_INTERNAL;
        }
        // SAFETY: created by hurray_reader_new via Box::into_raw; the caller guarantees
        // this is the first and only destroy call. Dropping it runs the transport's
        // release callback.
        drop(Box::from_raw(reader));
        *handle = std::ptr::null_mut();
        HURRAY_OK
    })
}

// ── Reading ───────────────────────────────────────────────────────────────────

/// Reads the next tensor, blocking in the transport's read callback as needed.
///
/// Returns:
///
/// - [`HURRAY_OK`] — `*out_tensor` is a new [`HurrayTensor`], to be released with
///   [`hurray_tensor_destroy`][crate::hurray_tensor_destroy].
/// - [`HURRAY_END_OF_STREAM`] — input ended cleanly at a tensor boundary. `*out_tensor` is
///   untouched. Positive, because this is how a stream ends, not a failure.
/// - [`HURRAY_ERR_UNEXPECTED_EOF`][crate::HURRAY_ERR_UNEXPECTED_EOF] — input ended inside
///   a descriptor or a buffer.
/// - Whatever status the read callback returned, if it failed. The reader's position is
///   intact in that case — the transport contract says a failing callback moved no bytes —
///   so **the same call MAY be retried** and will resume mid-tensor rather than restart it.
///
/// A composite head arrives as the buffer-less tensor it is on the wire; walking a
/// composite's members from C is not yet supported.
///
/// Every buffer in the returned tensor points at memory the read core allocated at the
/// alignment its buffer handle *declares*, and the transport wrote into it directly — so a
/// SIMD load on it is sound, and no copy of the payload happened after that write.
///
/// # Safety
///
/// - `handle` MUST point to a live [`HurrayReader`].
/// - `out_tensor` MUST be a valid, non-null, writable pointer.
#[no_mangle]
pub unsafe extern "C" fn hurray_reader_next_tensor(
    handle: *mut HurrayReader,
    out_tensor: *mut *mut HurrayTensor,
) -> HurrayStatus {
    catch(|| {
        null_check!(handle, out_tensor);
        // SAFETY: handle is non-null and points to a live HurrayReader, which the C ABI's
        // one-handle-one-thread rule says nothing else is touching.
        let reader = &mut *handle;

        loop {
            match reader.core.step() {
                Err(e) => return status_from_io_error(&e),
                Ok(ReadStep::Finished) => return HURRAY_END_OF_STREAM,
                Ok(ReadStep::Tensor) => {
                    let Some(tensor) = reader.core.take_tensor() else {
                        // step() said Tensor; the core cannot then have none.
                        return HURRAY_ERR_INTERNAL;
                    };
                    *out_tensor = Box::into_raw(tensor_handle(tensor.descriptor, tensor.buffers));
                    return HURRAY_OK;
                }
                Ok(ReadStep::Input { .. }) => {
                    // Disjoint fields: the core lends out its destination while the
                    // transport fills it.
                    let dst = reader.core.input_dst();
                    let filled = match reader.transport.read(dst) {
                        Ok(n) => n,
                        // No bytes moved, so no report is owed and the position still
                        // matches the transport's: § 6's LostReadCount is unreachable from
                        // here, and the call is retryable.
                        Err(status) => return status,
                    };
                    if let Err(e) = reader.core.report_filled(filled) {
                        return status_from_io_error(&e);
                    }
                }
            }
        }
    })
}
