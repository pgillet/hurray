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
    #[error("buffer alignment error: expected {expected}-byte alignment, got {actual}")]
    AlignmentError { expected: usize, actual: usize },

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
