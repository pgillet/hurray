//! Blocking random-access file reads over a caller-supplied transport (ADR-041 §§ 7, 8).
//!
//! [`HurrayFileReader`] is a driver: `hurray-io`'s `FileReaderCore` holds the container
//! format and decides which bytes it wants, and this module asks the caller's `read_at`
//! callback for exactly that range. Every request is absolute, which is why the vtable
//! needs a positional read rather than a cursor — and why a failed call here is simply
//! retryable.

use hurray_io::file::{FileReaderCore, FileStep};

use crate::{
    descriptor::HurrayDescriptor,
    file::kv::{HurrayKvEntry, KvTable},
    panic::{catch, null_check},
    status::{
        status_from_io_error, HurrayStatus, HURRAY_ERR_INDEX_OUT_OF_BOUNDS, HURRAY_ERR_INTERNAL,
        HURRAY_ERR_NULL_POINTER, HURRAY_OK,
    },
    stream::{tensor::tensor_handle, transport::Transport, HurrayTensor, HurrayTransport},
};

// ── HurrayFileReader ──────────────────────────────────────────────────────────

/// Opaque handle to a blocking file reader.
///
/// Not `#[repr(C)]`; the layout is an implementation detail. Created by
/// [`hurray_file_reader_new`] and released exactly once by [`hurray_file_reader_destroy`].
///
/// One handle, one thread at a time.
pub struct HurrayFileReader {
    core: FileReaderCore,
    transport: Transport,
    /// Tensor names in index order, owned: a C caller holds pointers, and a pointer into
    /// the core would not survive the next read borrowing it mutably.
    names: Vec<String>,
    /// The KV section, owned for the same reason.
    kv: KvTable,
}

// ── Construction / destruction ────────────────────────────────────────────────

/// Opens a Hurray file over `transport`: header, trailer, footer index and KV section.
///
/// The transport's `read_at` and `size` callbacks are both required — the container is
/// located from its end, so its size is a precondition of opening it. The vtable is
/// copied, so the caller MAY free the [`HurrayTransport`] struct immediately; its `ctx`
/// must stay valid until the reader is destroyed, at which point the vtable's release
/// callback is invoked.
///
/// Returns:
///
/// - [`HURRAY_OK`] — `*out_handle` is a new reader.
/// - [`HURRAY_ERR_INVALID_MAGIC`][crate::HURRAY_ERR_INVALID_MAGIC] — not a `HRRYFILE`.
/// - [`HURRAY_ERR_VERSION_MISMATCH`][crate::HURRAY_ERR_VERSION_MISMATCH] — a container
///   version this build does not read.
/// - [`HURRAY_ERR_INVALID_FILE`][crate::HURRAY_ERR_INVALID_FILE] — the footer does not
///   check out: a CRC mismatch, an index overrunning the trailer, unknown header flags.
/// - [`HURRAY_ERR_UNEXPECTED_EOF`][crate::HURRAY_ERR_UNEXPECTED_EOF] — the file is
///   shorter than its own structure claims.
/// - Whatever status a callback returned, if it failed. A failed open is terminal: there
///   is no index to start from, so create a new reader rather than retrying this one.
///
/// # Safety
///
/// - `transport` MUST point to a readable [`HurrayTransport`] for the duration of the call.
/// - `out_handle` MUST be a valid, non-null, writable pointer.
#[no_mangle]
pub unsafe extern "C" fn hurray_file_reader_new(
    transport: *const HurrayTransport,
    out_handle: *mut *mut HurrayFileReader,
) -> HurrayStatus {
    catch(|| {
        null_check!(transport, out_handle);

        // SAFETY: caller guarantees transport is readable for this call.
        let transport = Transport::from_c(transport);
        if !transport.can_read_at() {
            return HURRAY_ERR_NULL_POINTER;
        }

        let size = match transport.size() {
            Ok(size) => size,
            Err(status) => return status,
        };
        let mut core = FileReaderCore::new(size);

        // Opening is a job like any other: drive it to the first `Ready`.
        match drive(&mut core, &transport) {
            Ok(FileStep::Ready) => {}
            Ok(_) => return HURRAY_ERR_INTERNAL,
            Err(status) => return status,
        }

        let reader = HurrayFileReader {
            names: core.index().iter().map(|e| e.name.clone()).collect(),
            kv: KvTable::new(core.kv()),
            core,
            transport,
        };
        // SAFETY: Box::into_raw transfers ownership to the caller, who must call
        // hurray_file_reader_destroy exactly once.
        *out_handle = Box::into_raw(Box::new(reader));
        HURRAY_OK
    })
}

/// Destroys a reader, releases its transport context, and writes null through `handle`.
///
/// Tensors already produced by [`hurray_file_reader_read_tensor`] are unaffected: each
/// owns its own buffers and outlives the reader. Name and KV pointers are *not* — they
/// borrow from the handle and die with it.
///
/// # Safety
///
/// - `handle` MUST be a valid, non-null pointer to a `*mut HurrayFileReader` from
///   [`hurray_file_reader_new`].
/// - That reader MUST NOT have been destroyed already.
#[no_mangle]
pub unsafe extern "C" fn hurray_file_reader_destroy(
    handle: *mut *mut HurrayFileReader,
) -> HurrayStatus {
    catch(|| {
        null_check!(handle);
        let reader = *handle;
        if reader.is_null() {
            return HURRAY_ERR_INTERNAL;
        }
        // SAFETY: created by hurray_file_reader_new via Box::into_raw; the caller
        // guarantees this is the first and only destroy call. Dropping it runs the
        // transport's release callback.
        drop(Box::from_raw(reader));
        *handle = std::ptr::null_mut();
        HURRAY_OK
    })
}

// ── The index ─────────────────────────────────────────────────────────────────

/// Writes the number of tensors in the file's footer index.
///
/// Every tensor is individually addressable, composite heads and members included.
///
/// # Safety
///
/// `handle` and `out_count` MUST be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn hurray_file_reader_tensor_count(
    handle: *const HurrayFileReader,
    out_count: *mut u64,
) -> HurrayStatus {
    catch(|| {
        null_check!(handle, out_count);
        // SAFETY: handle is non-null and points to a live HurrayFileReader.
        *out_count = (*handle).names.len() as u64;
        HURRAY_OK
    })
}

/// Borrows the name of index entry `index` as UTF-8 bytes.
///
/// The bytes are **not** NUL-terminated and are owned by the reader: valid until
/// [`hurray_file_reader_destroy`], never to be freed by the caller.
///
/// Returns [`HURRAY_ERR_INDEX_OUT_OF_BOUNDS`][crate::HURRAY_ERR_INDEX_OUT_OF_BOUNDS] if
/// `index` is past the end.
///
/// # Safety
///
/// `handle`, `out_name` and `out_len` MUST be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn hurray_file_reader_tensor_name(
    handle: *const HurrayFileReader,
    index: u64,
    out_name: *mut *const u8,
    out_len: *mut usize,
) -> HurrayStatus {
    catch(|| {
        null_check!(handle, out_name, out_len);
        // SAFETY: handle is non-null and points to a live HurrayFileReader.
        let reader = &*handle;
        let Some(name) = reader.names.get(index as usize) else {
            return HURRAY_ERR_INDEX_OUT_OF_BOUNDS;
        };
        *out_name = name.as_ptr();
        *out_len = name.len();
        HURRAY_OK
    })
}

// ── Reading ───────────────────────────────────────────────────────────────────

/// Reads the descriptor of the tensor named `name`, without reading a byte of its data.
///
/// What a loader inspects before deciding whether it wants the tensor at all: element
/// type, shape, layout, quantization.
///
/// `*out_descriptor` is **owned** by the caller and released with
/// [`hurray_descriptor_destroy`][crate::descriptor::hurray_descriptor_destroy].
///
/// Returns [`HURRAY_ERR_NOT_FOUND`][crate::HURRAY_ERR_NOT_FOUND] if the name is not in
/// the index. A failed read leaves the reader usable: the request was for an absolute
/// range, so nothing about the reader's position is lost.
///
/// # Safety
///
/// - `handle` MUST point to a live [`HurrayFileReader`].
/// - `name` MUST be readable for `name_len` bytes.
/// - `out_descriptor` MUST be a valid, non-null, writable pointer.
#[no_mangle]
pub unsafe extern "C" fn hurray_file_reader_read_descriptor(
    handle: *mut HurrayFileReader,
    name: *const u8,
    name_len: usize,
    out_descriptor: *mut *mut HurrayDescriptor,
) -> HurrayStatus {
    catch(|| {
        null_check!(handle, out_descriptor);
        // SAFETY: caller guarantees `name_len` readable bytes at `name`.
        let name = match crate::file::kv::str_from_c(name, name_len) {
            Ok(name) => name,
            Err(status) => return status,
        };

        // SAFETY: handle is non-null and points to a live HurrayFileReader, which the C
        // ABI's one-handle-one-thread rule says nothing else is touching.
        let HurrayFileReader {
            core, transport, ..
        } = &mut *handle;

        if let Err(e) = core.begin_descriptor(name) {
            return status_from_io_error(&e);
        }
        match drive(core, transport) {
            Ok(FileStep::Descriptor) => match core.take_descriptor() {
                Some(descriptor) => {
                    *out_descriptor = Box::into_raw(HurrayDescriptor::from_descriptor(descriptor));
                    HURRAY_OK
                }
                // step() said Descriptor; the core cannot then have none.
                None => HURRAY_ERR_INTERNAL,
            },
            Ok(_) => HURRAY_ERR_INTERNAL,
            Err(status) => status,
        }
    })
}

/// Reads the tensor named `name`: its descriptor and every data buffer.
///
/// `*out_tensor` is **owned** by the caller and released with
/// [`hurray_tensor_destroy`][crate::hurray_tensor_destroy], which frees both halves.
///
/// Every buffer points at memory the core allocated at the alignment its buffer handle
/// *declares*, which the transport wrote into directly — so a SIMD load on it is sound,
/// and no copy of the payload happened after that write.
///
/// Returns [`HURRAY_ERR_NOT_FOUND`][crate::HURRAY_ERR_NOT_FOUND] for a name that is not
/// in the index. A composite head reads back as the buffer-less tensor it is in the file;
/// walking its members from C is not supported yet.
///
/// # Safety
///
/// - `handle` MUST point to a live [`HurrayFileReader`].
/// - `name` MUST be readable for `name_len` bytes.
/// - `out_tensor` MUST be a valid, non-null, writable pointer.
#[no_mangle]
pub unsafe extern "C" fn hurray_file_reader_read_tensor(
    handle: *mut HurrayFileReader,
    name: *const u8,
    name_len: usize,
    out_tensor: *mut *mut HurrayTensor,
) -> HurrayStatus {
    catch(|| {
        null_check!(handle, out_tensor);
        // SAFETY: caller guarantees `name_len` readable bytes at `name`.
        let name = match crate::file::kv::str_from_c(name, name_len) {
            Ok(name) => name,
            Err(status) => return status,
        };

        // SAFETY: handle is non-null and points to a live HurrayFileReader.
        let HurrayFileReader {
            core, transport, ..
        } = &mut *handle;

        if let Err(e) = core.begin_tensor(name) {
            return status_from_io_error(&e);
        }
        match drive(core, transport) {
            Ok(FileStep::Tensor) => match core.take_tensor() {
                Some(tensor) => {
                    *out_tensor = Box::into_raw(tensor_handle(tensor.descriptor, tensor.buffers));
                    HURRAY_OK
                }
                // step() said Tensor; the core cannot then have none.
                None => HURRAY_ERR_INTERNAL,
            },
            Ok(_) => HURRAY_ERR_INTERNAL,
            Err(status) => status,
        }
    })
}

// ── The KV section ────────────────────────────────────────────────────────────

/// Writes the number of entries in the file's KV metadata section.
///
/// # Safety
///
/// `handle` and `out_count` MUST be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn hurray_file_reader_kv_count(
    handle: *const HurrayFileReader,
    out_count: *mut u64,
) -> HurrayStatus {
    catch(|| {
        null_check!(handle, out_count);
        // SAFETY: handle is non-null and points to a live HurrayFileReader.
        *out_count = (*handle).kv.entries().len() as u64;
        HURRAY_OK
    })
}

/// Copies KV entry `index` — its key and its typed value — into `out_entry`.
///
/// The pointers inside the entry borrow from the reader: valid until
/// [`hurray_file_reader_destroy`], never to be freed. The entry can be handed straight to
/// [`hurray_file_writer_finish`][crate::file::hurray_file_writer_finish], which is how a
/// KV section is copied from one file into another without being transcribed.
///
/// Returns [`HURRAY_ERR_INDEX_OUT_OF_BOUNDS`][crate::HURRAY_ERR_INDEX_OUT_OF_BOUNDS] if
/// `index` is past the end.
///
/// # Safety
///
/// `handle` MUST point to a live [`HurrayFileReader`], and `out_entry` MUST be a valid,
/// non-null pointer writable for one [`HurrayKvEntry`].
#[no_mangle]
pub unsafe extern "C" fn hurray_file_reader_kv_entry(
    handle: *const HurrayFileReader,
    index: u64,
    out_entry: *mut HurrayKvEntry,
) -> HurrayStatus {
    catch(|| {
        null_check!(handle, out_entry);
        // SAFETY: handle is non-null and points to a live HurrayFileReader.
        let reader = &*handle;
        let Some(entry) = reader.kv.entries().get(index as usize) else {
            return HURRAY_ERR_INDEX_OUT_OF_BOUNDS;
        };
        *out_entry = *entry;
        HURRAY_OK
    })
}

// ── The driver ────────────────────────────────────────────────────────────────

/// Answers every range the core asks for, and returns the step that ends the job.
///
/// A free function rather than a method so the core and the transport can be borrowed as
/// disjoint fields — and so the open sequence can run before the handle exists.
///
/// The transfer and its byte-count report are one step with nothing suspended between
/// them, the shape ADR-041 § 6a requires of every driver.
fn drive(core: &mut FileReaderCore, transport: &Transport) -> Result<FileStep, HurrayStatus> {
    loop {
        match core.step() {
            Err(e) => return Err(status_from_io_error(&e)),
            // `input_dst` is exactly as long as the outstanding range, so the request's
            // own `len` needs no second application here.
            Ok(FileStep::Range { offset, .. }) => {
                let filled = transport.read_at(offset, core.input_dst())?;
                // No bytes moved on an error, and the range was absolute: the core's
                // position is intact and the same call may simply be made again.
                if let Err(e) = core.report_filled(filled) {
                    return Err(status_from_io_error(&e));
                }
            }
            Ok(other) => return Ok(other),
        }
    }
}
