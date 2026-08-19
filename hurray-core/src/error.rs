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

    /// The memory class byte is `0xFF`, which is permanently reserved by the spec.
    ///
    /// Readers MUST reject a buffer handle whose `memory_class` is `0xFF`.
    #[error("invalid memory class: 0x{0:02X} is permanently reserved")]
    InvalidMemoryClass(u8),

    /// The memory class byte falls in the range `0x04`–`0xEF` reserved for future spec versions.
    ///
    /// Readers MUST reject a buffer handle with a `memory_class` in this range.
    #[error("reserved memory class: 0x{0:02X} is reserved for future specification versions")]
    ReservedMemoryClass(u8),

    /// All buffers in a tensor descriptor must share the same memory class.
    ///
    /// The `expected` and `found` fields are raw wire bytes so that the error
    /// message is independent of which memory classes the current implementation
    /// recognises.
    #[error("memory class mismatch: expected 0x{expected:02X}, found 0x{found:02X}")]
    MemoryClassMismatch {
        /// The wire byte of the first buffer's memory class.
        expected: u8,
        /// The wire byte of the mismatching buffer's memory class.
        found: u8,
    },

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
    /// Reserved ranges: `0x0C`–`0x3F`, `0x41`–`0x7F`, `0x80`–`0xEF`.
    /// (`0x0B` is the composite / virtual head (ADR-027) and `0x40` is Hilbert;
    /// both are named tags in this crate.)
    /// Implementations MUST NOT assign semantics to these tags in strict mode.
    #[error("reserved layout tag: 0x{0:02X} is reserved for future specification versions")]
    ReservedLayoutTag(u8),

    /// A tag with a named layout descriptor was passed to the permissive
    /// [`UnknownLayout`](crate::layout::UnknownLayout) constructor.
    ///
    /// "Unknown" is what lets a permissive reader relay a tag it does not
    /// understand. Applied to a tag this implementation *does* understand, it
    /// becomes a bypass: `Unknown` has no buffer count and no shape constraints,
    /// so such a descriptor would skip every check the named variant applies and
    /// then encode to a wire tag a conforming reader parses as that named layout.
    #[error("layout tag 0x{0:02X} has a named descriptor and must not be wrapped as unknown")]
    NamedLayoutTag(u8),

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

    /// The number of index components does not match the tensor rank.
    #[error("index rank {index_rank} does not match shape rank {shape_rank}")]
    IndexRankMismatch {
        index_rank: usize,
        shape_rank: usize,
    },

    /// An index component exceeds the dimension size.
    #[error("index[{dim}] = {index} is out of range [0, {size})")]
    IndexOutOfRange { dim: u32, index: u64, size: u64 },

    /// Arithmetic overflow when computing an element or byte address.
    #[error("address arithmetic overflow")]
    AddressOverflow,

    /// The computed byte address falls outside the buffer bounds.
    #[error("byte address falls outside buffer bounds [0, {buffer_size})")]
    ByteAddressOverflow { buffer_size: u64 },

    /// This layout requires multi-buffer access via an inherent method, not the trait.
    #[error("layout tag 0x{layout_tag:02X} requires multi-buffer access; use the layout-specific method")]
    LayoutRequiresMultiBuffer { layout_tag: u8 },

    /// Recursive layout nesting depth exceeded the implementation limit of 8 levels.
    ///
    /// Despite the name, this guards recursion depth for every recursive
    /// structure in the crate: [`crate::layout::TiledLayout`]'s nested layout
    /// payload, and [`crate::composite::CompositeTensor`]'s cross-descriptor
    /// nesting (a member whose own layout is itself `Composite`). Kept as
    /// `SubpavingNestingTooDeep` for wire/API stability rather than renamed
    /// as part of the subpaving removal.
    #[error("layout nesting depth exceeds the implementation limit of 8 levels")]
    SubpavingNestingTooDeep,

    /// A DYNAMIC dimension cannot be used for element addressing.
    #[error("dimension {dim} is DYNAMIC and cannot be used for element addressing")]
    DynamicDimInIndexing { dim: u32 },

    /// Arithmetic overflow in Morton or Hilbert index computation.
    #[error("index arithmetic overflow in space-filling curve computation")]
    IndexArithmeticOverflow,

    // ── Descriptor (Layer 4) errors ───────────────────────────────────────────
    /// The sync mode byte is not a recognized value.
    ///
    /// Valid sync mode bytes are `0x00` (`ProducerSynced`), `0x01` (`Event`),
    /// and `0x02` (`ConsumerStream`). Bytes `0x03`–`0xFF` are reserved or
    /// permanently invalid; all conforming readers MUST reject them.
    ///
    /// See `docs/spec/buffer-protocol.md § Synchronization Mode` and ADR-018.
    #[error("invalid sync mode byte: 0x{0:02X} (reserved or invalid)")]
    InvalidSyncMode(u8),

    /// Magic bytes are not `"HRRY"` (`0x48 0x52 0x52 0x59`).
    #[error("invalid magic: expected 48 52 52 59, got {got:02X?}")]
    InvalidMagic {
        /// The four bytes actually found at offset 0.
        got: [u8; 4],
    },

    /// `version_major` exceeds the supported major version (`1`).
    #[error("unsupported descriptor version {major}.{minor}")]
    UnsupportedDescriptorVersion {
        /// The major version found on the wire.
        major: u8,
        /// The minor version found on the wire.
        minor: u8,
    },

    /// `descriptor_length` is less than the minimum valid size (`20`).
    #[error("descriptor_length {length} is below the minimum of 20 bytes")]
    DescriptorTooShort {
        /// The `descriptor_length` value found on the wire.
        length: u32,
    },

    /// Cursor ran out of bytes before a field could be read.
    #[error("descriptor truncated: need {needed} bytes at offset {offset}, have {available}")]
    DescriptorTruncated {
        /// Byte offset at which the read was attempted.
        offset: usize,
        /// Number of bytes needed.
        needed: usize,
        /// Number of bytes actually available.
        available: usize,
    },

    /// A reserved flag bit is set.
    #[error("reserved flag bits set: 0x{flags:08X} (reserved mask: 0x{mask:08X})")]
    ReservedDescriptorFlagBitsSet {
        /// The full flags value found on the wire.
        flags: u32,
        /// The bitmask of bits that must be zero.
        mask: u32,
    },

    /// A reserved field that MUST be `0x00` contains a non-zero byte.
    #[error("reserved field '{field}' must be 0x00")]
    ReservedBytesNonZero {
        /// Human-readable name of the reserved field.
        field: &'static str,
    },

    /// `buffer_count` is `0` (minimum is `1`).
    #[error("buffer_count is 0; at least one buffer handle is required")]
    EmptyBufferTable,

    /// `HAS_EXTENSION_TYPE` flag ↔ `type_tag` range disagree.
    #[error(
        "HAS_EXTENSION_TYPE flag is {flag_set} but type_tag 0x{type_tag:02X} is {type_tag_in_range}"
    )]
    ExtensionTypeFlagMismatch {
        /// Whether the `HAS_EXTENSION_TYPE` flag was set.
        flag_set: bool,
        /// The `type_tag` byte found on the wire.
        type_tag: u8,
        /// Human-readable range description.
        type_tag_in_range: &'static str,
    },

    /// Extension type `bit_width` / `packing_factor` is invalid.
    #[error(
        "extension type packing invalid: bit_width={bit_width}, packing_factor={packing_factor}"
    )]
    ExtensionTypePackingInvalid {
        /// The `bit_width` value from the extension type descriptor.
        bit_width: u32,
        /// The `packing_factor` value from the extension type descriptor.
        packing_factor: u8,
    },

    /// `shard_offset[k] + shape[k] > parent_shape[k]`.
    #[error("shard out of bounds on dim {dim}: offset {offset} + size {size} > parent {parent}")]
    ShardOutOfBounds {
        /// The dimension index where the constraint was violated.
        dim: usize,
        /// The `shard_offset` value for that dimension.
        offset: u64,
        /// The tensor `shape` value for that dimension.
        size: u64,
        /// The `parent_shape` value for that dimension.
        parent: u64,
    },

    /// `computed_mask` has reserved bits set (bits ≥ 6).
    #[error("statistics computed_mask 0x{mask:08X} has reserved bits set")]
    StatisticsReservedMaskBitsSet {
        /// The full `computed_mask` value found on the wire.
        mask: u32,
    },

    /// `descriptor_length` declared in the header does not match the actual encoded length.
    #[error("descriptor_length mismatch: declared {declared}, actual {actual}")]
    DescriptorLengthMismatch {
        /// The `descriptor_length` value declared in the header.
        declared: u32,
        /// The actual number of bytes consumed when parsing.
        actual: usize,
    },

    // ── Composite / Virtual (ADR-027, Layer 4) errors ─────────────────────────
    /// A composite head (`layout_tag = 0x0B`) declared a non-empty buffer table.
    #[error(
        "composite head has {count} buffer(s); a composite head (layout_tag 0x0B) MUST have buffer_count = 0"
    )]
    CompositeHeadHasBuffers {
        /// The (non-zero) buffer count found on the wire.
        count: u8,
    },

    /// A composite head declared a non-zero `byte_offset`.
    #[error(
        "composite head has byte_offset {byte_offset}; a composite head MUST have byte_offset = 0"
    )]
    CompositeHeadHasByteOffset {
        /// The offending `byte_offset` value.
        byte_offset: u64,
    },

    /// A composite head set the `HAS_QUANTIZATION` flag, but a virtual head owns no stored data.
    #[error("composite head MUST NOT set HAS_QUANTIZATION (a virtual head owns no stored data)")]
    CompositeHeadHasQuantization,

    /// The `composition_rule` byte `0xFF` is permanently invalid.
    #[error("invalid composition rule: 0x{0:02X} is permanently reserved")]
    InvalidCompositionRule(u8),

    /// The `composition_rule` byte falls in the range reserved for future specification
    /// versions (`0x00` and `0x04`–`0xEF`).
    #[error("reserved composition rule: 0x{0:02X} is reserved for future specification versions")]
    ReservedCompositionRule(u8),

    /// The `composition_rule` byte is in the implementation-private range `0xF0`–`0xFE`.
    #[error(
        "private composition rule: 0x{0:02X} is implementation-private and not interpretable by this crate"
    )]
    PrivateCompositionRule(u8),

    /// The `combine_op` byte is not legal for the given `composition_rule`.
    #[error("invalid combine_op 0x{combine_op:02X} for composition rule 0x{rule:02X}")]
    InvalidCombineOp {
        /// The wire `composition_rule` byte.
        rule: u8,
        /// The offending `combine_op` byte.
        combine_op: u8,
    },

    /// The `member_count` sentinel `0xFFFFFFFF` (open composite) is RESERVED and not usable in v1.0.
    #[error("member_count = 0xFFFFFFFF (open composite) is reserved and not usable in v1.0")]
    OpenCompositeReserved,

    /// The Composite Member section's `member_role` byte is not `0x00` or `0x01`.
    #[error("invalid composite member_role: 0x{0:02X} (MUST be 0x00 correction or 0x01 base)")]
    InvalidMemberRole(u8),

    /// A member's `HAS_COMPOSITE_MEMBER` flag disagrees with what its composition rule requires.
    ///
    /// Overlay members MUST carry a Composite Member section; partition and group members
    /// MUST NOT.
    #[error(
        "composite member HAS_COMPOSITE_MEMBER presence ({has_flag}) disagrees with composition rule 0x{rule:02X}"
    )]
    CompositeMemberFlagMismatch {
        /// The wire `composition_rule` byte of the enclosing head.
        rule: u8,
        /// Whether the member actually carried a Composite Member section.
        has_flag: bool,
    },

    /// The number of members actually supplied does not match the head's declared `member_count`.
    #[error(
        "composite member_count mismatch: head declared {declared}, but {actual} member(s) were supplied"
    )]
    CompositeMemberCountMismatch {
        /// The `member_count` declared by the head.
        declared: u32,
        /// The number of members actually pushed to the validator.
        actual: usize,
    },

    /// A partition or overlay member did not carry a shard section (`HAS_SHARD`).
    #[error(
        "composite member {index} is missing a shard section (partition/overlay members MUST carry HAS_SHARD)"
    )]
    CompositeMemberMissingShard {
        /// The 0-based index of the offending member.
        index: usize,
    },

    /// A member's shard `parent_shape` does not equal the composite head's `shape`.
    #[error("composite member {index}: shard parent_shape does not equal the head's shape")]
    CompositeMemberParentShapeMismatch {
        /// The 0-based index of the offending member.
        index: usize,
    },

    /// A member's decoded value type does not equal the composite head's `type_tag`.
    #[error(
        "composite member {index}: decoded type 0x{member:02X} does not equal head type_tag 0x{head:02X}"
    )]
    CompositeMemberTypeMismatch {
        /// The 0-based index of the offending member.
        index: usize,
        /// The member's own `type_tag` byte.
        member: u8,
        /// The head's `type_tag` byte.
        head: u8,
    },

    /// An overlay composite's first member is not the base, or a base member appeared at a
    /// position other than first.
    #[error(
        "overlay composite: the base member (member_role = 0x01) MUST be first, and only the first member may be the base"
    )]
    CompositeOverlayBaseNotFirst,

    /// An overlay composite's base member does not span the whole index space.
    #[error(
        "overlay composite: the base member MUST span the whole index space (zero shard_offset, shape == head shape)"
    )]
    CompositeOverlayBaseNotSpanning,

    /// An overlay composite head declared `member_count = 0` (no base member available).
    #[error(
        "overlay composite MUST have at least one member (the base); member_count = 0 is rejected"
    )]
    CompositeOverlayEmpty,

    /// A partition composite's members leave a gap in the head's index space.
    #[error(
        "partition composite: member boxes do not exactly cover the head's index space (gap detected)"
    )]
    CompositePartitionGap,

    /// Two partition members' boxes overlap.
    #[error("partition composite: member {a} and member {b} boxes overlap")]
    CompositePartitionOverlap {
        /// The 0-based index of the first overlapping member.
        a: usize,
        /// The 0-based index of the second overlapping member.
        b: usize,
    },

    /// A layout is virtual (owns no data buffer) and does not support element addressing.
    ///
    /// Currently only the composite head (`layout_tag = 0x0B`) is virtual; addressing a
    /// composite requires resolving to a member first (see `hurray_core::composite`).
    #[error("layout tag 0x{layout_tag:02X} is virtual (owns no data) and does not support element addressing")]
    LayoutIsVirtual {
        /// The virtual layout's wire tag.
        layout_tag: u8,
    },
}

/// Convenience alias for `Result` with [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
