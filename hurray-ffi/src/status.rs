//! Status codes and error mapping for the Hurray C ABI.
//!
//! All exported `extern "C"` functions return a [`HurrayStatus`] integer. Zero
//! indicates success; negative values indicate a specific error condition.
//! Callers MUST check the return value of every function before using output
//! pointer arguments.
//!
//! One positive value exists: [`HURRAY_END_OF_STREAM`]. It is not an error, which is why
//! it is not negative, and why `status < 0` remains the error test.

use hurray_core::Error;

// ── Status code type ─────────────────────────────────────────────────────────

/// Integer status code returned by every Hurray C ABI function.
///
/// Zero (`HURRAY_OK`) indicates success. All negative values indicate errors.
/// Positive values report a normal, non-error outcome that is not plain success;
/// [`HURRAY_END_OF_STREAM`] is the only one so far.
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

/// An index argument is outside the valid range for the collection.
pub const HURRAY_ERR_INDEX_OUT_OF_BOUNDS: HurrayStatus = -11;

/// The requested field is valid for this layout but absent from this descriptor.
///
/// Distinct from [`HURRAY_ERR_FIELD_NOT_APPLICABLE`]: absence is a state this
/// format expresses (a cleared flag bit), so it is reported rather than
/// flattened to a zero the caller could mistake for data.
pub const HURRAY_ERR_FIELD_NOT_PRESENT: HurrayStatus = -12;

/// The requested field does not belong to this descriptor's layout.
pub const HURRAY_ERR_FIELD_NOT_APPLICABLE: HurrayStatus = -13;

/// A caller-supplied transport callback reported failure.
///
/// The handle is left usable: a callback that returns a failing status MUST NOT have
/// moved any bytes (`hurray-ffi`'s transport contract, ADR-041 § 8), so the stream
/// position is intact and the same call MAY be retried.
pub const HURRAY_ERR_TRANSPORT: HurrayStatus = -14;

/// The sink accepted zero bytes while an item was in flight.
///
/// Distinct from [`HURRAY_ERR_TRANSPORT`]: a truncated item is already on the wire, so
/// the writer is poisoned and no further call on it can succeed (ADR-041 § 4).
pub const HURRAY_ERR_SINK_CLOSED: HurrayStatus = -15;

/// Input ended in the middle of a descriptor or a buffer.
///
/// Distinct from [`HURRAY_END_OF_STREAM`], which is input ending cleanly at a tensor
/// boundary.
pub const HURRAY_ERR_UNEXPECTED_EOF: HurrayStatus = -16;

/// A transport moved bytes whose count was lost, so the handle's position no longer
/// matches the transport's and nothing can recover it (ADR-041 § 6).
///
/// Unreachable from a conforming transport vtable, which reports a count for every
/// transfer; the code exists because the state does.
pub const HURRAY_ERR_CORE_UNUSABLE: HurrayStatus = -17;

/// A descriptor, buffer, or nesting depth exceeded a configured limit.
pub const HURRAY_ERR_LIMIT_EXCEEDED: HurrayStatus = -18;

/// The byte stream is not a valid Hurray stream, or a stream that already failed was
/// used again.
pub const HURRAY_ERR_INVALID_STREAM: HurrayStatus = -19;

/// No tensor of that name is in the file's footer index.
///
/// Its own code because it is the one file-read failure that is a question, not a defect:
/// the file is fine and the caller asked for something that is not in it.
pub const HURRAY_ERR_NOT_FOUND: HurrayStatus = -20;

/// The file is not a valid HRRYFILE container, or the file being written would not be one.
///
/// Covers the footer family — a CRC mismatch, an index that overruns the trailer,
/// unknown header flag bits — and the name and KV-key rules a writer must keep. A wrong
/// magic or an unsupported container version report
/// [`HURRAY_ERR_INVALID_MAGIC`] and [`HURRAY_ERR_VERSION_MISMATCH`] instead, which say
/// more.
pub const HURRAY_ERR_INVALID_FILE: HurrayStatus = -21;

// ── Non-error outcomes ────────────────────────────────────────────────────────

/// Input ended cleanly at a tensor boundary: there is no next tensor.
///
/// Positive, because end of input is the normal way a stream finishes and callers test
/// errors with `status < 0`.
pub const HURRAY_END_OF_STREAM: HurrayStatus = 1;

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

/// Maps a [`hurray_io::Error`] to the closest [`HurrayStatus`] error code.
///
/// Format errors are delegated to [`status_from_core_error`] so that a malformed
/// descriptor reports the same code whether it arrived through
/// [`hurray_descriptor_decode`][crate::hurray_descriptor_decode] or through a stream.
pub(crate) fn status_from_io_error(e: &hurray_io::Error) -> HurrayStatus {
    use hurray_io::Error as E;
    match e {
        E::Core(inner) => status_from_core_error(inner),

        E::Io(_) => HURRAY_ERR_TRANSPORT,
        E::UnexpectedEof => HURRAY_ERR_UNEXPECTED_EOF,
        E::SinkClosed => HURRAY_ERR_SINK_CLOSED,
        E::CoreUnusable(_) => HURRAY_ERR_CORE_UNUSABLE,

        E::FrameTooLarge { .. } | E::CompositeNestingTooDeep { .. } => HURRAY_ERR_LIMIT_EXCEEDED,

        E::MultiBufferLengthMismatch { .. }
        | E::BufferSizeMismatch { .. }
        | E::InvalidCrossMachineSyncMode { .. }
        | E::UnallocatableBuffer { .. }
        | E::TornComposite { .. }
        | E::NotAComposite(_)
        | E::CoreFailed => HURRAY_ERR_INVALID_STREAM,

        // A driver bug, not a caller error: this crate's driver always reports a count it
        // was offered. Reachable only if that invariant breaks, and never silently.
        E::InvalidProgressReport { .. } | E::ZeroLengthWrite => HURRAY_ERR_INTERNAL,

        // The file container. Magic and version reuse the codes that already name them,
        // rather than flattening into one "bad file" status that says less.
        E::InvalidFileMagic | E::InvalidTrailerMagic => HURRAY_ERR_INVALID_MAGIC,
        E::UnsupportedContainerVersion { .. } => HURRAY_ERR_VERSION_MISMATCH,
        E::TensorNotFound(_) => HURRAY_ERR_NOT_FOUND,
        // `InvalidHeader` is the container's own variant — every site that produces it is
        // under `file/` — so it belongs here rather than with the stream codes.
        E::InvalidHeader(_)
        | E::IndexCrc32cMismatch { .. }
        | E::IndexOverrunsTrailer
        | E::ReservedFileFlagBits { .. }
        | E::DuplicateTensorName(_)
        | E::TensorNameEmpty
        | E::TensorNameTooLong { .. }
        | E::DuplicateKvKey(_)
        | E::KvKeyEmpty
        | E::KvKeyTooLong { .. } => HURRAY_ERR_INVALID_FILE,

        // Any future variants.
        _ => HURRAY_ERR_INTERNAL,
    }
}
