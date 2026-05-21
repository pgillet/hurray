//! Status codes and error mapping for the Hurray C ABI.
//!
//! All exported `extern "C"` functions return a [`HurrayStatus`] integer. Zero
//! indicates success; negative values indicate a specific error condition.
//! Callers MUST check the return value of every function before using output
//! pointer arguments.

use hurray_core::Error;

// ── Status code type ─────────────────────────────────────────────────────────

/// Integer status code returned by every Hurray C ABI function.
///
/// Zero (`HURRAY_OK`) indicates success. All negative values indicate errors.
/// Positive values are reserved for future use.
///
/// # Examples
///
/// ```
/// use hurray_ffi::{HurrayStatus, HURRAY_OK};
///
/// fn check(s: HurrayStatus) {
///     assert_eq!(s, HURRAY_OK);
/// }
/// check(HURRAY_OK);
/// ```
pub type HurrayStatus = i32;

/// Operation completed successfully.
pub const HURRAY_OK: HurrayStatus = 0;

/// The descriptor magic bytes were not `"HRRY"` (`0x48 0x52 0x52 0x59`).
pub const HURRAY_ERR_INVALID_MAGIC: HurrayStatus = -1;

/// The descriptor version is not supported by this implementation.
pub const HURRAY_ERR_VERSION_MISMATCH: HurrayStatus = -2;

/// A layout descriptor field or tag is invalid.
pub const HURRAY_ERR_INVALID_LAYOUT: HurrayStatus = -3;

/// A type tag, device tag, or memory class is invalid.
pub const HURRAY_ERR_INVALID_TYPE: HurrayStatus = -4;

/// An output buffer is too small to hold the result.
pub const HURRAY_ERR_BUFFER_TOO_SMALL: HurrayStatus = -5;

/// A required pointer argument is null.
pub const HURRAY_ERR_NULL_POINTER: HurrayStatus = -6;

/// The function panicked internally; the handle MUST NOT be reused.
pub const HURRAY_ERR_INTERNAL_PANIC: HurrayStatus = -7;

/// The sync mode byte is not a recognized value.
pub const HURRAY_ERR_INVALID_SYNC_MODE: HurrayStatus = -8;

/// A sync-mode handoff payload does not match the buffer's declared sync mode.
pub const HURRAY_ERR_SYNC_MODE_MISMATCH: HurrayStatus = -9;

/// An unclassified internal error occurred.
pub const HURRAY_ERR_INTERNAL: HurrayStatus = -10;

// ── Error mapping ─────────────────────────────────────────────────────────────

/// Maps a [`hurray_core::Error`] to the closest [`HurrayStatus`] error code.
///
/// This function is exhaustive over all known variants and falls back to
/// [`HURRAY_ERR_INTERNAL`] for variants added in future crate versions
/// (the `Error` enum is `#[non_exhaustive]`).
pub(crate) fn status_from_core_error(e: &Error) -> HurrayStatus {
    match e {
        Error::InvalidMagic { .. } => HURRAY_ERR_INVALID_MAGIC,

        Error::UnsupportedDescriptorVersion { .. } => HURRAY_ERR_VERSION_MISMATCH,

        // Layout-structural errors
        Error::DescriptorTooShort { .. }
        | Error::DescriptorTruncated { .. }
        | Error::DescriptorLengthMismatch { .. }
        | Error::ReservedDescriptorFlagBitsSet { .. }
        | Error::ReservedBytesNonZero { .. }
        | Error::EmptyBufferTable
        | Error::RankExceedsMaximum { .. }
        | Error::InvalidLayout(_)
        | Error::InvalidLayoutTag(_)
        | Error::ReservedLayoutTag(_)
        | Error::PrivateLayoutTag(_)
        | Error::UnknownLayoutTag(_)
        | Error::ExtensionTypeFlagMismatch { .. }
        | Error::ExtensionTypePackingInvalid { .. }
        | Error::ShardOutOfBounds { .. }
        | Error::StatisticsReservedMaskBitsSet { .. }
        | Error::AlignmentNotPowerOfTwo { .. }
        | Error::AlignmentBelowMinimum { .. }
        | Error::AlignmentError { .. }
        | Error::InvalidShape(_) => HURRAY_ERR_INVALID_LAYOUT,

        // Type / device / memory class errors
        Error::InvalidTypeTag(_)
        | Error::ReservedTypeTag(_)
        | Error::UnknownTypeTag(_)
        | Error::UnsupportedElementType(_)
        | Error::InvalidDeviceTag(_)
        | Error::ReservedDeviceTag(_)
        | Error::InvalidMemoryClass(_)
        | Error::ReservedMemoryClass(_) => HURRAY_ERR_INVALID_TYPE,

        Error::InvalidSyncMode(_) => HURRAY_ERR_INVALID_SYNC_MODE,

        // Everything else (including future non_exhaustive variants)
        _ => HURRAY_ERR_INTERNAL,
    }
}
