/// Crate-level error type for `hurray-core`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// An element type is not supported or recognized.
    #[error("unsupported element type: {0}")]
    UnsupportedElementType(String),

    /// A shape or stride value is invalid.
    #[error("invalid shape or stride: {0}")]
    InvalidShape(String),

    /// A buffer alignment requirement was not satisfied.
    ///
    /// This variant is preserved for callers that check alignment against an
    /// externally observed value (e.g., checking the actual base-address
    /// alignment of a pointer). For descriptor-level validation, prefer
    /// [`Error::AlignmentNotPowerOfTwo`] and [`Error::AlignmentBelowMinimum`].
    #[error("buffer alignment error: expected {expected}-byte alignment, got {actual}")]
    AlignmentError { expected: usize, actual: usize },

    /// The `alignment` field is not a power of two.
    ///
    /// The spec requires that `alignment` is always a power of two
    /// (see `docs/spec/buffer-protocol.md § Alignment`).
    #[error("alignment {alignment} is not a power of two")]
    AlignmentNotPowerOfTwo { alignment: u32 },

    /// The `alignment` field is below the minimum required for a non-empty buffer.
    ///
    /// Non-empty buffers (those with `byte_size > 0`) MUST declare an alignment
    /// of at least [`crate::MIN_BUFFER_ALIGNMENT`] (64 bytes) to guarantee
    /// compatibility with all current SIMD instruction sets.
    #[error(
        "alignment {alignment} is below the minimum of {minimum} bytes required for non-empty buffers"
    )]
    AlignmentBelowMinimum { alignment: u32, minimum: u32 },

    /// The device tag byte `0xFF` is permanently invalid.
    ///
    /// This sentinel is reserved by the spec and MUST be rejected by all
    /// conforming readers (see `docs/spec/buffer-protocol.md § Device Tags`).
    #[error("invalid device tag: 0x{0:02X} is permanently reserved")]
    InvalidDeviceTag(u8),

    /// The device tag falls in the range `0x04`–`0xEF` reserved for future spec versions.
    ///
    /// Implementations MUST NOT assign semantics to tags in this range.
    #[error("reserved device tag: 0x{0:02X} is reserved for future specification versions")]
    ReservedDeviceTag(u8),

    /// All buffers in a tensor descriptor must reside on the same device.
    ///
    /// The `expected` and `found` fields are raw wire bytes so that the error
    /// message is independent of which device tags the current implementation
    /// recognises.
    #[error("device tag mismatch: expected 0x{expected:02X}, found 0x{found:02X}")]
    DeviceTagMismatch {
        /// The wire byte of the first buffer's device tag.
        expected: u8,
        /// The wire byte of the mismatching buffer's device tag.
        found: u8,
    },

    /// Buffer list is empty; at least one buffer handle is required.
    ///
    /// Returned by [`crate::validate_colocation`] when called with an empty
    /// slice — there is no device tag to validate against.
    #[error("buffer list is empty")]
    EmptyBufferList,

    /// A quantization descriptor is malformed.
    #[error("invalid quantization descriptor: {0}")]
    InvalidQuantization(String),

    /// The type tag `0x00` or `0xFF` is explicitly invalid per the spec.
    ///
    /// These two sentinels are permanently reserved and MUST be rejected by all
    /// conforming readers regardless of operating mode.
    #[error(
        "invalid type tag: 0x{0:02X} is permanently reserved and must never appear in a descriptor"
    )]
    InvalidTypeTag(u8),

    /// The type tag falls in a range reserved for future specification versions.
    ///
    /// Reserved ranges: `0x47` and `0x80`–`0xEF`. Implementations MUST NOT
    /// assign semantics to these tags.
    #[error("reserved type tag: 0x{0:02X} is reserved for future specification versions")]
    ReservedTypeTag(u8),

    /// The type tag is not recognized by this implementation.
    ///
    /// This covers the private-extension range `0xF0`–`0xFE` and any other
    /// unassigned tag value not covered by [`Error::InvalidTypeTag`] or
    /// [`Error::ReservedTypeTag`].
    #[error("unknown type tag: 0x{0:02X} is not recognized by this implementation")]
    UnknownTypeTag(u8),

    /// The tensor rank exceeds the maximum of 64 defined by the spec.
    #[error("rank {rank} exceeds the maximum permitted rank of {max}")]
    RankExceedsMaximum {
        /// The rank value that was rejected.
        rank: u32,
        /// The maximum permitted rank (`64`).
        max: u32,
    },
}

/// Convenience alias for `Result` with [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
