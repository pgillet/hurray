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
pub(crate) mod panic;
pub mod status;
pub mod sync;
pub mod tensor_context;

// ── Re-exports ────────────────────────────────────────────────────────────────

pub use buffer::{hurray_buffer_from_ptr, HurrayBuffer, HurrayReleaseCallback};
pub use buffer_list::{hurray_buffer_list_new, HurrayBufferList};
pub use descriptor::{hurray_descriptor_decode, HurrayDescriptor};
pub use status::{
    HurrayStatus, HURRAY_ERR_BUFFER_TOO_SMALL, HURRAY_ERR_INDEX_OUT_OF_BOUNDS, HURRAY_ERR_INTERNAL,
    HURRAY_ERR_INTERNAL_PANIC, HURRAY_ERR_INVALID_LAYOUT, HURRAY_ERR_INVALID_MAGIC,
    HURRAY_ERR_INVALID_SYNC_MODE, HURRAY_ERR_INVALID_TYPE, HURRAY_ERR_NULL_POINTER,
    HURRAY_ERR_SYNC_MODE_MISMATCH, HURRAY_ERR_VERSION_MISMATCH, HURRAY_OK,
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
/// assert_eq!(HURRAY_C_ABI_VERSION, 4);
/// ```
pub const HURRAY_C_ABI_VERSION: u32 = 4;

/// Returns the [`HURRAY_C_ABI_VERSION`] constant.
///
/// This function is infallible and requires no panic barrier.
///
/// # Examples
///
/// ```
/// assert_eq!(unsafe { hurray_ffi::hurray_c_abi_version() }, 4);
/// ```
#[no_mangle]
pub extern "C" fn hurray_c_abi_version() -> u32 {
    HURRAY_C_ABI_VERSION
}

// ── Forward declarations for future async bridge ──────────────────────────────

/// Opaque handle to an async streaming reader (full implementation deferred).
///
/// A reader bridges `hurray-io`'s async streaming layer into the C ABI.
/// The async-to-sync bridging strategy (e.g., blocking executor, channel
/// hand-off) is deferred; this stub allows cbindgen to emit the type
/// declaration in the generated header.
pub struct HurrayReader;

/// Opaque handle to an async streaming writer (full implementation deferred).
///
/// Symmetric to [`HurrayReader`]. Deferred pending the async bridge design.
pub struct HurrayWriter;
