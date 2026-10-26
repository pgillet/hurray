//! Blocking file writes over a caller-supplied transport (ADR-041 §§ 3, 7, 8).
//!
//! A file is written in one forward pass — header, tensors, footer — so the writer needs
//! nothing from the vtable that a stream writer does not: no seek, no size, just `write`.
//! The footer index is accumulated as tensors go out and emitted last, which is why no
//! offset is ever patched.

use hurray_io::file::{FileWriteOp, FileWriterCore, FileWriterOptions};
use hurray_io::stream::WriteStep;

use crate::{
    descriptor::HurrayDescriptor,
    file::kv::{kv_from_c, HurrayKvEntry},
    panic::{catch, null_check},
    status::{
        status_from_io_error, HurrayStatus, HURRAY_ERR_INTERNAL, HURRAY_ERR_NULL_POINTER,
        HURRAY_ERR_SINK_CLOSED, HURRAY_OK,
    },
    stream::{transport::Transport, HurrayIoSlice, HurrayTransport},
};

// ── Options ───────────────────────────────────────────────────────────────────

/// Container options for a file writer.
///
/// Pass null to [`hurray_file_writer_new`] for the defaults: 4096-byte data buffer
/// alignment, index in write order.
#[repr(C)]
pub struct HurrayFileWriterOptions {
    /// Alignment applied to every data buffer in the file. MUST be a power of two in
    /// `[4096, 2097152]`. Default: 4096, which is page-aligned on every common OS and so
    /// lets a consumer mmap a buffer directly.
    pub data_buffer_alignment: u32,
    /// Sort the footer index by tensor name instead of leaving it in write order, so a
    /// reader can binary-search it. Default: false.
    pub sorted_index: bool,
}

// ── HurrayFileWriter ──────────────────────────────────────────────────────────

/// Opaque handle to a blocking file writer.
///
/// Not `#[repr(C)]`; the layout is an implementation detail. Created by
/// [`hurray_file_writer_new`] and released exactly once by
/// [`hurray_file_writer_destroy`].
///
/// One handle, one thread at a time.
pub struct HurrayFileWriter {
    core: FileWriterCore,
    transport: Transport,
}

// ── Construction / destruction ────────────────────────────────────────────────

/// Creates a file writer over `transport` and writes the 64-byte file header.
///
/// The header goes out here, not on the first tensor, which is what makes every later
/// offset in the file knowable without seeking back.
///
/// # Arguments
///
/// - `transport` — Non-null; its `write` callback MUST be non-null. `read_at` and `size`
///   are not used: a file writer never reads what it has written.
/// - `options` — MAY be null for [`HurrayFileWriterOptions`]' documented defaults.
/// - `out_handle` — Non-null pointer to a `*mut HurrayFileWriter`; set on success.
///
/// Returns [`HURRAY_ERR_INVALID_FILE`][crate::HURRAY_ERR_INVALID_FILE] if
/// `data_buffer_alignment` is not a power of two in range.
///
/// # Safety
///
/// - `transport` MUST point to a readable [`HurrayTransport`] for the duration of the call.
/// - `options`, if non-null, MUST point to a readable [`HurrayFileWriterOptions`].
/// - `out_handle` MUST be a valid, non-null, writable pointer.
#[no_mangle]
pub unsafe extern "C" fn hurray_file_writer_new(
    transport: *const HurrayTransport,
    options: *const HurrayFileWriterOptions,
    out_handle: *mut *mut HurrayFileWriter,
) -> HurrayStatus {
    catch(|| {
        // options may be null: that is the request for defaults.
        null_check!(transport, out_handle);

        // SAFETY: caller guarantees transport is readable for this call.
        let transport = Transport::from_c(transport);
        if !transport.can_write() {
            return HURRAY_ERR_NULL_POINTER;
        }

        let io_options = if options.is_null() {
            FileWriterOptions::default()
        } else {
            // SAFETY: non-null and caller-guaranteed readable.
            FileWriterOptions {
                data_buffer_alignment: (*options).data_buffer_alignment,
                sorted_index: (*options).sorted_index,
            }
        };

        let mut core = match FileWriterCore::with_options(io_options) {
            Ok(core) => core,
            Err(e) => return status_from_io_error(&e),
        };

        let status = drain(&mut core.header(), &transport);
        if status != HURRAY_OK {
            return status;
        }

        // SAFETY: Box::into_raw transfers ownership to the caller, who must call
        // hurray_file_writer_destroy exactly once.
        *out_handle = Box::into_raw(Box::new(HurrayFileWriter { core, transport }));
        HURRAY_OK
    })
}

/// Destroys a writer, releases its transport context, and writes null through `handle`.
///
/// Destroying a writer that was never finished leaves a file with no footer, which no
/// reader can open — the bytes are not a container until
/// [`hurray_file_writer_finish`] has written its index and trailer.
///
/// # Safety
///
/// - `handle` MUST be a valid, non-null pointer to a `*mut HurrayFileWriter` from
///   [`hurray_file_writer_new`].
/// - That writer MUST NOT have been destroyed already.
#[no_mangle]
pub unsafe extern "C" fn hurray_file_writer_destroy(
    handle: *mut *mut HurrayFileWriter,
) -> HurrayStatus {
    catch(|| {
        null_check!(handle);
        let writer = *handle;
        if writer.is_null() {
            return HURRAY_ERR_INTERNAL;
        }
        // SAFETY: created by hurray_file_writer_new via Box::into_raw; the caller
        // guarantees this is the first and only destroy call. Dropping it runs the
        // transport's release callback.
        drop(Box::from_raw(writer));
        *handle = std::ptr::null_mut();
        HURRAY_OK
    })
}

// ── Writing ───────────────────────────────────────────────────────────────────

/// Writes one tensor under `name`: its descriptor, its padding, then its data buffers.
///
/// `payloads` MUST have one entry per buffer in the descriptor's buffer table, in that
/// order, each with the declared `byte_size`. The whole tensor is validated before a byte
/// reaches the sink. The name is recorded in the footer index, so it is what
/// [`hurray_file_reader_read_tensor`][crate::file::hurray_file_reader_read_tensor] will
/// ask for; it MUST be non-empty, at most 65 535 bytes, and unique within the file.
///
/// Returns:
///
/// - [`HURRAY_OK`] — the tensor is entirely written and the payload pointers are no
///   longer held.
/// - [`HURRAY_ERR_INVALID_FILE`][crate::HURRAY_ERR_INVALID_FILE] — the name is empty, too
///   long, or already used in this file. Refused before a byte goes out, so the file is
///   still completable and another name may be used.
/// - [`HURRAY_ERR_SINK_CLOSED`] — the sink stopped accepting bytes, or the write callback
///   failed, once part of this tensor was already out.
/// - Whatever status the write callback returned, if it failed before any byte of this
///   tensor reached the sink.
///
/// **A write that fails is the end of the file**, whichever of those two it reports: the
/// writer is poisoned, every later call on it fails, and the bytes on the sink are not a
/// container. Unlike a stream write, it is not retryable — the tensor's name is spent and
/// its descriptor is already staged, so a second attempt would describe a different file
/// from the one on the sink. Writing a file again from the beginning is the recovery, and
/// for a storage failure it is the only one worth having.
///
/// Composite tensors are not yet writable from C; a composite head alone is not a valid
/// tensor and is refused.
///
/// # Safety
///
/// - `handle` MUST point to a live [`HurrayFileWriter`] and `descriptor` to a live
///   [`HurrayDescriptor`].
/// - `name` MUST be readable for `name_len` bytes.
/// - If `count` is non-zero, `payloads` MUST point to `count` readable [`HurrayIoSlice`]
///   values, each describing memory readable for its own `len` bytes.
/// - All of that memory MUST stay valid and unmodified until this call returns.
#[no_mangle]
pub unsafe extern "C" fn hurray_file_writer_write_tensor(
    handle: *mut HurrayFileWriter,
    name: *const u8,
    name_len: usize,
    descriptor: *const HurrayDescriptor,
    payloads: *const HurrayIoSlice,
    count: usize,
) -> HurrayStatus {
    catch(|| {
        null_check!(handle, descriptor);
        if count > 0 {
            null_check!(payloads);
        }

        // SAFETY: caller guarantees `name_len` readable bytes at `name`.
        let name = match crate::file::kv::str_from_c(name, name_len) {
            Ok(name) => name,
            Err(status) => return status,
        };

        // SAFETY: non-null and caller-guaranteed to describe `count` readable slices.
        let slices = if count == 0 {
            &[][..]
        } else {
            std::slice::from_raw_parts(payloads, count)
        };

        let mut borrowed: Vec<&[u8]> = Vec::with_capacity(count);
        for slice in slices {
            // SAFETY: caller guarantees `len` readable bytes at `ptr`, unmodified for the
            // duration of this call; a zero length needs no pointer.
            match crate::file::kv::bytes_from_c(slice.ptr, slice.len) {
                Ok(bytes) => borrowed.push(bytes),
                Err(status) => return status,
            }
        }

        // SAFETY: both pointers are non-null and point to live handles, which the C ABI's
        // one-handle-one-thread rule says nothing else is touching.
        let HurrayFileWriter { core, transport } = &mut *handle;
        let descriptor = (*descriptor).inner();

        let mut op = match core.tensor(name, descriptor, &borrowed) {
            Ok(op) => op,
            Err(e) => return status_from_io_error(&e),
        };
        drain(&mut op, transport)
    })
}

/// Writes the KV section, the footer index and the 40-byte trailer, completing the file.
///
/// Until this returns [`HURRAY_OK`] the bytes on the sink are not a container: the index
/// a reader opens from is written here. Calling it twice fails.
///
/// `kv` MAY be null with a `kv_count` of `0` for a file with no metadata section. Keys
/// MUST be non-empty, at most 65 535 bytes, and unique. Entries read out of another file
/// with [`hurray_file_reader_kv_entry`][crate::file::hurray_file_reader_kv_entry] can be
/// passed straight in.
///
/// The writer handle stays alive and MUST still be destroyed with
/// [`hurray_file_writer_destroy`]; any flush or close belongs to the transport's release
/// callback, because only the caller knows what its sink is.
///
/// # Safety
///
/// - `handle` MUST point to a live [`HurrayFileWriter`].
/// - If `kv_count` is non-zero, `kv` MUST point to `kv_count` readable
///   [`HurrayKvEntry`] values whose own pointers are readable for their declared lengths,
///   all unmodified until this call returns.
#[no_mangle]
pub unsafe extern "C" fn hurray_file_writer_finish(
    handle: *mut HurrayFileWriter,
    kv: *const HurrayKvEntry,
    kv_count: usize,
) -> HurrayStatus {
    catch(|| {
        null_check!(handle);
        if kv_count > 0 {
            null_check!(kv);
        }

        let entries = if kv_count == 0 {
            Vec::new()
        } else {
            // SAFETY: caller guarantees `kv_count` readable entries at `kv`.
            match kv_from_c(kv, kv_count) {
                Ok(entries) => entries,
                Err(status) => return status,
            }
        };

        // SAFETY: handle is non-null and points to a live HurrayFileWriter.
        let HurrayFileWriter { core, transport } = &mut *handle;

        let mut op = match core.finish(entries) {
            Ok(op) => op,
            Err(e) => return status_from_io_error(&e),
        };
        drain(&mut op, transport)
    })
}

// ── The driver ────────────────────────────────────────────────────────────────

/// Writes every byte of one submitted item — header, tensor or footer — to the transport.
///
/// Any transport failure poisons the writer: see
/// [`hurray_file_writer_write_tensor`] for why a file write is not retryable where a
/// stream write is.
///
/// Not shared with the stream writer's loop: `FileWriteOp` and the stream's `WriteOp` are
/// distinct types with no common trait upstream, and a trait defined here to join them
/// would be more surface than the twenty lines it saves.
fn drain(op: &mut FileWriteOp<'_>, transport: &Transport) -> HurrayStatus {
    // Bytes of *this* item already on the sink. What makes a transport failure
    // recoverable or not: nothing emitted means nothing to un-send.
    let mut emitted = 0usize;
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
                        emitted += n;
                    }
                    Err(status) => {
                        // Poisoned either way, unlike the stream writer. A file's tensor
                        // has already consumed its name and staged its descriptor by the
                        // time bytes move, and the core offers no rollback for that, so a
                        // retry would write a file that does not match its own index.
                        op.report_sink_closed();
                        if emitted > 0 {
                            // A torn item: the transport's own status would understate it.
                            return HURRAY_ERR_SINK_CLOSED;
                        }
                        // Nothing of this item is on the sink, so the callback's status is
                        // the more useful of the two — it says why the file stopped.
                        return status;
                    }
                }
            }
        }
    }
}
