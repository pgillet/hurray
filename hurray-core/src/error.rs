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

    /// The quantization descriptor payload is shorter than the minimum required.
    #[error("quantization descriptor too short: have {found} bytes, need {needed}")]
    QuantizationDescriptorTooShort {
        /// Number of bytes available.
        found: usize,
        /// Number of bytes required.
        needed: usize,
    },

    /// The scheme tag `0x00` or `0xFF` is permanently reserved by the spec.
    ///
    /// A reader MUST reject any descriptor whose `scheme_tag` is `0x00` or `0xFF`.
    #[error("invalid quantization scheme tag: 0x{0:02X} is permanently reserved")]
    InvalidQuantizationSchemeTag(u8),

    /// The scheme tag falls in a range reserved for future specification versions.
    ///
    /// Ranges: `0x60`–`0xEF`. Implementations MUST NOT assign semantics to these tags.
    #[error(
        "reserved quantization scheme tag: 0x{0:02X} is reserved for future specification versions"
    )]
    ReservedQuantizationSchemeTag(u8),

    /// The scheme tag is in the implementation-private range `0xF0`–`0xFE`.
    ///
    /// Private scheme tags have unconstrained payloads beyond the 4-byte header;
    /// `hurray-core` cannot interpret them. Callers that need private scheme
    /// support must handle the raw bytes at a higher layer.
    ///
    /// WHY rejected here: the payload beyond the 4-byte header is unconstrained
    /// for private tags, giving this crate nothing useful to return. Callers that
    /// need private schemes handle the raw bytes at a higher layer (design decision #1).
    #[error(
        "private quantization scheme tag: 0x{0:02X} is implementation-private and not interpretable by this crate"
    )]
    PrivateQuantizationSchemeTag(u8),

    /// The scheme tag is in an allocated range but not assigned to any known scheme.
    #[error("unknown quantization scheme tag: 0x{0:02X}")]
    UnknownQuantizationSchemeTag(u8),

    /// The `scheme_version` field exceeds the highest version this implementation supports.
    ///
    /// Per the spec, a reader MUST reject a descriptor whose `scheme_version` exceeds
    /// the highest version defined for the given `scheme_tag`.
    #[error(
        "unsupported scheme version: scheme tag 0x{tag:02X} version {version} is not supported (highest supported: {supported})"
    )]
    UnsupportedSchemeVersion {
        /// The scheme tag being decoded.
        tag: u8,
        /// The version found on the wire.
        version: u8,
        /// The highest version this implementation supports for this tag.
        supported: u8,
    },

    /// Reserved `flags` bits are set in the quantization descriptor header.
    ///
    /// Per the spec, a reader MUST reject a descriptor with any reserved `flags` bit set.
    #[error("reserved quantization flags bits set: 0x{flags:04X} (reserved mask: 0x{mask:04X})")]
    ReservedQuantizationFlagsBits {
        /// The full flags value found on the wire.
        flags: u16,
        /// The bitmask of bits that must be zero.
        mask: u16,
    },

    /// The `block_size` field is out of range or not a power of two for the given scheme.
    #[error(
        "invalid block_size {block_size} for scheme 0x{scheme_tag:02X}: must be a power of two in [{min}, {max}]"
    )]
    InvalidBlockSize {
        /// The quantization scheme tag.
        scheme_tag: u8,
        /// The block size found on the wire.
        block_size: u32,
        /// Minimum permitted block size for this scheme.
        min: u32,
        /// Maximum permitted block size for this scheme.
        max: u32,
    },

    /// The tensor's storage `type_tag` is not valid for the given quantization scheme.
    ///
    /// Each quantization scheme defines a fixed set of permitted storage types.
    #[error(
        "element type 0x{type_tag:02X} is not a valid storage type for scheme 0x{scheme_tag:02X}"
    )]
    InvalidStorageTypeForScheme {
        /// The quantization scheme tag.
        scheme_tag: u8,
        /// The storage type tag found in the tensor descriptor.
        type_tag: u8,
    },

    /// The quantization axis index is out of bounds for the tensor's rank.
    ///
    /// Per the spec, `axis` MUST satisfy `axis < rank`.
    #[error("quantization axis {axis} out of bounds: rank is {rank}")]
    QuantizationAxisOutOfBounds {
        /// The axis value from the quantization descriptor.
        axis: u32,
        /// The rank of the tensor descriptor.
        rank: u32,
    },

    /// The tensor shape along the quantization axis is incompatible with the block size.
    ///
    /// For MXFP, `shape[axis]` must be a positive multiple of `block_size`.
    /// For per-block-affine and NF4, `block_size` must not exceed `shape[axis]`
    /// when `shape[axis] > 0`.
    #[error(
        "quantization shape mismatch on axis {axis}: shape[axis] = {shape_axis}, block_size = {block_size}: {reason}"
    )]
    QuantizationShapeMismatch {
        /// The quantization axis.
        axis: u32,
        /// The resolved size of `shape[axis]`.
        shape_axis: u64,
        /// The block size from the quantization descriptor.
        block_size: u32,
        /// A human-readable description of the constraint violated.
        reason: &'static str,
    },

    /// The `zero_point` value is outside the representable range of the storage type.
    ///
    /// Per the spec, `zero_point` MUST lie within the representable range of the
    /// storage type (e.g., `[0, 255]` for `uint8`).
    #[error(
        "zero_point {zero_point} is outside the range of storage type 0x{type_tag:02X}: [{min}, {max}]"
    )]
    ZeroPointOutOfRange {
        /// The storage type tag.
        type_tag: u8,
        /// The zero-point value that was rejected.
        zero_point: i32,
        /// Minimum representable value for the storage type.
        min: i64,
        /// Maximum representable value for the storage type.
        max: i64,
    },

    /// A quantization-parameter buffer index aliases the tensor data buffer.
    ///
    /// Per the spec, quantization-parameter buffers MUST occupy distinct indices
    /// from the tensor data buffer.
    #[error("quantization parameter buffer index {index} aliases the tensor data buffer index")]
    QuantizationBufferAliasesData {
        /// The offending quantization-parameter buffer index.
        index: u32,
    },

    /// A quantization-parameter buffer index is out of range.
    ///
    /// The index must be less than `buffer_count` in the tensor descriptor's
    /// buffer table.
    #[error(
        "quantization parameter buffer index {index} is out of range (buffer_count = {buffer_count})"
    )]
    QuantizationBufferIndexOutOfRange {
        /// The out-of-range buffer index.
        index: u32,
        /// The number of buffers in the tensor descriptor's buffer table.
        buffer_count: u32,
    },

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

    /// Layout tag `0x00` or `0xFF` is permanently invalid.
    ///
    /// These sentinels are reserved by the spec and MUST be rejected by all
    /// conforming readers regardless of operating mode
    /// (see `docs/spec/memory-layout.md § Layout Taxonomy`).
    #[error("invalid layout tag: 0x{0:02X} is permanently reserved")]
    InvalidLayoutTag(u8),

    /// Layout tag is in a range reserved for future specification versions.
    ///
    /// Reserved ranges: `0x0A`–`0x3F`, `0x41`–`0x7F`, `0x80`–`0xEF`.
    /// Implementations MUST NOT assign semantics to these tags in strict mode.
    #[error("reserved layout tag: 0x{0:02X} is reserved for future specification versions")]
    ReservedLayoutTag(u8),

    /// Layout tag is in the private-extension range `0xF0`–`0xFE`.
    ///
    /// Private extension layouts have unconstrained payloads beyond the standard
    /// header fields; strict-mode readers reject them unless both parties have
    /// agreed on semantics out of band (see `docs/spec/memory-layout.md § Extension Layouts`).
    #[error("private layout tag: 0x{0:02X} is in the private-extension range (0xF0–0xFE)")]
    PrivateLayoutTag(u8),

    /// Layout tag is not recognized by this implementation.
    ///
    /// Covers any allocated but unassigned tag not already matched by
    /// [`Error::InvalidLayoutTag`] or [`Error::ReservedLayoutTag`].
    #[error("unknown layout tag: 0x{0:02X} is not recognized by this implementation")]
    UnknownLayoutTag(u8),

    /// A layout descriptor field is invalid.
    ///
    /// The inner message describes the specific field and the constraint violated.
    #[error("invalid layout descriptor: {0}")]
    InvalidLayout(String),
}

/// Convenience alias for `Result` with [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
