//! # hurray-ffi
//!
//! C ABI layer for hurray language bindings.
//!
//! This crate exposes opaque handles, a function table, and buffer release
//! callbacks via a stable C ABI. It is the foundation for all non-Rust language
//! bindings.
//!
//! ## Safety contract
//!
//! - No panics may propagate across the FFI boundary. All `extern "C"` functions
//!   wrap their bodies in `std::panic::catch_unwind`.
//! - All `unsafe` blocks carry a `// SAFETY:` comment explaining soundness.
//! - Buffer pointers MUST be aligned to at least 64 bytes.
//!
//! ## C ABI version
//!
//! The C ABI version is exposed via [`HURRAY_C_ABI_VERSION`] and the function
//! [`hurray_c_abi_version`]. Callers SHOULD check this at startup to detect
//! incompatible library versions.

pub mod buffer;
pub mod buffer_list;
pub mod descriptor;
pub mod field;
pub mod file;
pub(crate) mod panic;
pub mod status;
pub mod stream;
pub mod sync;
pub mod tensor_context;

// ── Re-exports ────────────────────────────────────────────────────────────────

pub use buffer::{hurray_buffer_from_ptr, HurrayBuffer, HurrayReleaseCallback};
pub use buffer_list::{hurray_buffer_list_new, HurrayBufferList};
pub use descriptor::{hurray_descriptor_decode, HurrayDescriptor};
pub use field::{
    hurray_descriptor_field_f64, hurray_descriptor_field_i64, hurray_descriptor_field_i64_array,
    hurray_descriptor_field_u64, hurray_descriptor_field_u64_array, HurrayDescriptorField,
};
pub use file::{
    hurray_file_reader_destroy, hurray_file_reader_kv_count, hurray_file_reader_kv_entry,
    hurray_file_reader_new, hurray_file_reader_read_descriptor, hurray_file_reader_read_tensor,
    hurray_file_reader_tensor_count, hurray_file_reader_tensor_name, hurray_file_writer_destroy,
    hurray_file_writer_finish, hurray_file_writer_new, hurray_file_writer_write_tensor,
    HurrayFileReader, HurrayFileWriter, HurrayFileWriterOptions, HurrayKvEntry, HurrayKvValue,
    HURRAY_KV_ARRAY, HURRAY_KV_BOOL, HURRAY_KV_BYTES, HURRAY_KV_FLOAT64, HURRAY_KV_INT64,
    HURRAY_KV_STRING, HURRAY_KV_UINT64,
};
pub use status::{
    HurrayStatus, HURRAY_END_OF_STREAM, HURRAY_ERR_BUFFER_TOO_SMALL, HURRAY_ERR_CORE_UNUSABLE,
    HURRAY_ERR_FIELD_NOT_APPLICABLE, HURRAY_ERR_FIELD_NOT_PRESENT, HURRAY_ERR_INDEX_OUT_OF_BOUNDS,
    HURRAY_ERR_INTERNAL, HURRAY_ERR_INTERNAL_PANIC, HURRAY_ERR_INVALID_FILE,
    HURRAY_ERR_INVALID_LAYOUT, HURRAY_ERR_INVALID_MAGIC, HURRAY_ERR_INVALID_STREAM,
    HURRAY_ERR_INVALID_SYNC_MODE, HURRAY_ERR_INVALID_TYPE, HURRAY_ERR_LIMIT_EXCEEDED,
    HURRAY_ERR_NOT_FOUND, HURRAY_ERR_NULL_POINTER, HURRAY_ERR_SINK_CLOSED,
    HURRAY_ERR_SYNC_MODE_MISMATCH, HURRAY_ERR_TRANSPORT, HURRAY_ERR_UNEXPECTED_EOF,
    HURRAY_ERR_VERSION_MISMATCH, HURRAY_OK,
};
pub use stream::{
    hurray_reader_destroy, hurray_reader_new, hurray_reader_next_tensor, hurray_tensor_buffers,
    hurray_tensor_descriptor, hurray_tensor_destroy, hurray_writer_destroy, hurray_writer_new,
    hurray_writer_write_tensor, HurrayIoSlice, HurrayReader, HurrayReaderOptions, HurrayTensor,
    HurrayTransport, HurrayTransportReadAtFn, HurrayTransportReadFn, HurrayTransportReleaseFn,
    HurrayTransportSizeFn, HurrayTransportWriteFn, HurrayWriter, HurrayWriterOptions,
};
pub use sync::{HurrayEventReleaseFn, HurraySyncConsumerStreamPayload, HurraySyncEventPayload};
pub use tensor_context::{hurray_tensor_context_new, HurrayOwnerReleaseFn, HurrayTensorContext};

// ── ABI version ───────────────────────────────────────────────────────────────

/// The integer version of the Hurray C ABI exposed by this build.
///
/// Callers SHOULD retrieve this at runtime via [`hurray_c_abi_version`] and
/// compare it to the version they were compiled against.
///
/// # Examples
///
/// ```
/// use hurray_ffi::HURRAY_C_ABI_VERSION;
///
/// assert_eq!(HURRAY_C_ABI_VERSION, 8);
/// ```
pub const HURRAY_C_ABI_VERSION: u32 = 8;

/// Returns the [`HURRAY_C_ABI_VERSION`] constant.
///
/// This function is infallible and requires no panic barrier.
///
/// # Examples
///
/// ```
/// assert_eq!(unsafe { hurray_ffi::hurray_c_abi_version() }, 8);
/// ```
#[no_mangle]
pub extern "C" fn hurray_c_abi_version() -> u32 {
    HURRAY_C_ABI_VERSION
}
