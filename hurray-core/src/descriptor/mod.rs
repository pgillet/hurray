//! Tensor descriptor — binary encoding and decoding.
//!
//! The [`TensorDescriptor`] type is the top-level carrier for all metadata
//! required to interpret a tensor's data buffer: element type, rank, shape,
//! memory layout, buffer handles, and optional quantization, shard, statistics,
//! and extension-type annotations.
//!
//! # Wire format
//!
//! Defined in `docs/spec/metadata.md`. Fixed header (20 bytes) followed by
//! variable-length core fields, layout-specific payload, buffer table, and
//! up to four optional sections selected by the `flags` bitmask.
//!
//! # Examples
//!
//! ```
//! use hurray_core::{
//!     BufferHandle, DeviceTag, ElementType, Shape, SyncMode, MIN_BUFFER_ALIGNMENT,
//!     descriptor::TensorDescriptor,
//!     layout::LayoutDescriptor,
//! };
//!
//! // Build a float32 [3, 4] row-major tensor descriptor (the spec's worked example).
//! let shape = Shape::new(vec![3, 4]).unwrap();
//! let buffer = BufferHandle::new(192, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap();
//! let desc = TensorDescriptor::new(
//!     1, 0,
//!     ElementType::Float32,
//!     shape,
//!     0,
//!     LayoutDescriptor::RowMajor,
//!     vec![buffer],
//!     None, None, None, None,
//! ).unwrap();
//!
//! let encoded = desc.encode().unwrap();
//! assert_eq!(encoded.len(), 61); // spec worked example is 61 bytes
//!
//! let decoded = TensorDescriptor::decode(&encoded).unwrap();
//! assert_eq!(decoded, desc);
//! ```

// ── Sub-modules ───────────────────────────────────────────────────────────────

pub mod composite_member;
pub(crate) mod cursor;
mod decode;
mod encode;
pub mod ext_type;
pub mod layout_codec;
pub mod shard;
pub mod statistics;

// Internal type alias used by encode.rs / decode.rs to import TensorDescriptor
// without a circular path.
pub(crate) mod mod_types {
    pub(crate) use super::{DescriptorFlags, TensorDescriptor, RESERVED_FLAGS_MASK};
}

// ── Public re-exports ─────────────────────────────────────────────────────────

pub use composite_member::{CompositeMemberDescriptor, MemberRole};
pub use ext_type::ExtensionTypeDescriptor;
pub use shard::ShardDescriptor;
pub use statistics::{Statistics, StatisticsMask};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Magic bytes that open every Hurray tensor descriptor: ASCII `"HRRY"`.
///
/// ```
/// use hurray_core::descriptor::MAGIC;
/// assert_eq!(&MAGIC, b"HRRY");
/// ```
pub const MAGIC: [u8; 4] = *b"HRRY";

/// Current major version of the descriptor format.
pub const DESCRIPTOR_VERSION_MAJOR: u8 = 1;

/// Current minor version of the descriptor format.
pub const DESCRIPTOR_VERSION_MINOR: u8 = 0;

/// Minimum valid value for the `descriptor_length` field (= fixed header size).
pub(crate) const MIN_DESCRIPTOR_LEN: u32 = 20;

/// Bitmask of all reserved flag bits (bits 5–31). Readers MUST reject descriptors
/// with any reserved bit set.
pub(crate) const RESERVED_FLAGS_MASK: u32 = !0x1F;

// ── DescriptorFlags ───────────────────────────────────────────────────────────

/// Descriptor flags bitmask (wire field `flags`, offset 10 in the fixed header).
///
/// Each bit selects an optional section that follows the buffer table. Bits 4–31
/// are reserved and MUST be `0`.
///
/// # Examples
///
/// ```
/// use hurray_core::descriptor::DescriptorFlags;
///
/// let f = DescriptorFlags(DescriptorFlags::HAS_QUANTIZATION | DescriptorFlags::HAS_SHARD);
/// assert!(f.has_quantization());
/// assert!(f.has_shard());
/// assert!(!f.has_statistics());
/// assert!(!f.has_extension_type());
/// assert!(!f.has_composite_member());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorFlags(pub u32);

impl DescriptorFlags {
    /// Bit 0: quantization section is present.
    pub const HAS_QUANTIZATION: u32 = 1 << 0;
    /// Bit 1: shard section is present.
    pub const HAS_SHARD: u32 = 1 << 1;
    /// Bit 2: extension type section is present.
    pub const HAS_EXTENSION_TYPE: u32 = 1 << 2;
    /// Bit 3: statistics section is present.
    pub const HAS_STATISTICS: u32 = 1 << 3;
    /// Bit 4: Composite Member section is present. Set on the members of an
    /// overlay composite; see `docs/spec/layouts/composite.md`.
    pub const HAS_COMPOSITE_MEMBER: u32 = 1 << 4;

    /// Returns `true` if the `HAS_QUANTIZATION` bit is set.
    #[inline]
    pub fn has_quantization(self) -> bool {
        self.0 & Self::HAS_QUANTIZATION != 0
    }

    /// Returns `true` if the `HAS_SHARD` bit is set.
    #[inline]
    pub fn has_shard(self) -> bool {
        self.0 & Self::HAS_SHARD != 0
    }

    /// Returns `true` if the `HAS_EXTENSION_TYPE` bit is set.
    #[inline]
    pub fn has_extension_type(self) -> bool {
        self.0 & Self::HAS_EXTENSION_TYPE != 0
    }

    /// Returns `true` if the `HAS_STATISTICS` bit is set.
    #[inline]
    pub fn has_statistics(self) -> bool {
        self.0 & Self::HAS_STATISTICS != 0
    }

    /// Returns `true` if the `HAS_COMPOSITE_MEMBER` bit is set.
    #[inline]
    pub fn has_composite_member(self) -> bool {
        self.0 & Self::HAS_COMPOSITE_MEMBER != 0
    }
}

// ── TensorDescriptor ──────────────────────────────────────────────────────────

use crate::descriptor::composite_member::CompositeMemberDescriptor as CompositeMemberDesc;
use crate::descriptor::ext_type::ExtensionTypeDescriptor as ExtDesc;
use crate::descriptor::shard::ShardDescriptor as ShardDesc;
use crate::descriptor::statistics::Statistics as Stats;
use crate::quantization::QuantizationDescriptor;
use crate::{BufferHandle, ElementType, Error, LayoutDescriptor, Result, Shape};

/// The top-level tensor descriptor carrying all metadata required to interpret
/// a tensor's data buffer.
///
/// Construct via [`TensorDescriptor::new`], encode to bytes with
/// [`TensorDescriptor::encode`], and decode from bytes with
/// [`TensorDescriptor::decode`].
///
/// Derives `PartialEq` but NOT `Eq` — the `statistics` field may contain `f64`
/// values, and NaN semantics make `Eq` unsound for those fields.
///
/// # Examples
///
/// ```
/// use hurray_core::{
///     BufferHandle, DeviceTag, ElementType, Shape, SyncMode, MIN_BUFFER_ALIGNMENT,
///     descriptor::TensorDescriptor,
///     layout::LayoutDescriptor,
/// };
///
/// let shape  = Shape::new(vec![3u64, 4]).unwrap();
/// let buffer = BufferHandle::new(192, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap();
/// let desc   = TensorDescriptor::new(
///     1, 0,
///     ElementType::Float32,
///     shape,
///     0,
///     LayoutDescriptor::RowMajor,
///     vec![buffer],
///     None, None, None, None,
/// ).unwrap();
///
/// assert_eq!(desc.version_major, 1);
/// assert_eq!(desc.element_type, ElementType::Float32);
/// assert_eq!(desc.shape.rank(), 2);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct TensorDescriptor {
    /// Major format version stored in the wire header.
    pub version_major: u8,
    /// Minor format version stored in the wire header.
    pub version_minor: u8,
    /// Element type tag for this tensor's data.
    pub element_type: ElementType,
    /// Tensor shape (rank and dimension sizes).
    pub shape: Shape,
    /// Byte offset from the start of buffer 0 to logical element `[0,…,0]`.
    pub byte_offset: u64,
    /// Memory layout descriptor.
    pub layout: LayoutDescriptor,
    /// Buffer handles (at least one required).
    pub buffers: Vec<BufferHandle>,
    /// Raw quantization payload (opaque `quantization_descriptor` bytes).
    ///
    /// When present the bytes are stored verbatim; typed decode is not performed
    /// in this layer (design decision: typed quantization decode is deferred to
    /// a higher layer that has schema context).
    pub quantization: Option<Vec<u8>>,
    /// Shard annotation (this tensor is a sub-region of a larger parent).
    pub shard: Option<ShardDesc>,
    /// Advisory statistics about the tensor's data buffer.
    pub statistics: Option<Stats>,
    /// Extension type descriptor (present iff `type_tag` is in `0xF0`–`0xFE`).
    pub extension_type: Option<ExtDesc>,
    /// Composite Member section (this tensor's role within an enclosing overlay
    /// composite). `None` by default; set via [`TensorDescriptor::with_composite_member`].
    ///
    /// Not a [`TensorDescriptor::new`] parameter — a composite head declares
    /// `member_count` before its members exist, so member-role assignment is
    /// necessarily a post-construction, opt-in step (see ADR-027).
    pub composite_member: Option<CompositeMemberDesc>,
}

impl TensorDescriptor {
    /// Creates a new [`TensorDescriptor`], validating invariants.
    ///
    /// # Errors
    ///
    /// - [`Error::EmptyBufferTable`] — `buffers` is empty (and `layout` is not the
    ///   composite head, `layout_tag = 0x0B`, for which an empty buffer table is
    ///   the *only* valid table — see [`Error::CompositeHeadHasBuffers`]).
    /// - [`Error::CompositeHeadHasBuffers`] — `layout` is the composite head but
    ///   `buffers` is non-empty.
    /// - [`Error::CompositeHeadHasByteOffset`] — `layout` is the composite head but
    ///   `byte_offset != 0`.
    /// - [`Error::CompositeHeadHasQuantization`] — `layout` is the composite head but
    ///   `quantization` is `Some`.
    /// - [`Error::ExtensionTypeFlagMismatch`] — `extension_type` is `Some` but
    ///   `element_type.tag()` is not in `0xF0`–`0xFE`, or vice-versa.
    /// - [`Error::InvalidShape`] — `shard.parent_shape.len() != shape.rank()`.
    /// - [`Error::InvalidLayout`] — a block-paged or CSF layout carries a `shard` descriptor
    ///   (spec § Sharding).
    /// - [`Error::InvalidQuantization`] — a block-paged layout carries a `quantization`
    ///   descriptor incompatible with the paged layout (spec § Quantization Compatibility).
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{
    ///     BufferHandle, DeviceTag, ElementType, Shape, SyncMode, MIN_BUFFER_ALIGNMENT,
    ///     descriptor::TensorDescriptor,
    ///     layout::LayoutDescriptor,
    /// };
    ///
    /// let shape  = Shape::new(vec![2u64, 3]).unwrap();
    /// let buffer = BufferHandle::new(64, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap();
    /// let desc   = TensorDescriptor::new(
    ///     1, 0, ElementType::Float32, shape, 0,
    ///     LayoutDescriptor::RowMajor, vec![buffer],
    ///     None, None, None, None,
    /// ).unwrap();
    /// assert_eq!(desc.buffers.len(), 1);
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        version_major: u8,
        version_minor: u8,
        element_type: ElementType,
        shape: Shape,
        byte_offset: u64,
        layout: LayoutDescriptor,
        buffers: Vec<BufferHandle>,
        quantization: Option<Vec<u8>>,
        shard: Option<ShardDesc>,
        statistics: Option<Stats>,
        extension_type: Option<ExtDesc>,
    ) -> Result<Self> {
        // Invariant (ADR-027 D-1): buffer_count == 0 is allowed only for the composite
        // head (layout_tag 0x0B), which owns no data. A composite head additionally MUST
        // have byte_offset == 0 and MUST NOT carry a quantization section — checked here,
        // the seam where layout, buffers, byte_offset, and quantization are all in hand.
        let is_composite_head = matches!(&layout, LayoutDescriptor::Composite(_));
        if is_composite_head {
            if !buffers.is_empty() {
                return Err(Error::CompositeHeadHasBuffers {
                    count: buffers.len().min(u8::MAX as usize) as u8,
                });
            }
            if byte_offset != 0 {
                return Err(Error::CompositeHeadHasByteOffset { byte_offset });
            }
            if quantization.is_some() {
                return Err(Error::CompositeHeadHasQuantization);
            }
        } else if buffers.is_empty() {
            return Err(Error::EmptyBufferTable);
        }

        // Invariant: buffer_count fits in uint8 (wire field ceiling is 255).
        if buffers.len() > 255 {
            return Err(Error::InvalidLayout(format!(
                "buffer_count {} exceeds the maximum of 255 (uint8 ceiling)",
                buffers.len()
            )));
        }

        // Invariant: HAS_EXTENSION_TYPE ↔ type_tag in 0xF0–0xFE.
        let is_ext_tag = matches!(element_type.tag(), 0xF0..=0xFE);
        let has_ext = extension_type.is_some();
        if is_ext_tag != has_ext {
            return Err(Error::ExtensionTypeFlagMismatch {
                flag_set: has_ext,
                type_tag: element_type.tag(),
                type_tag_in_range: if is_ext_tag {
                    "in 0xF0-0xFE"
                } else {
                    "not in 0xF0-0xFE"
                },
            });
        }

        // Invariant: shard rank matches tensor rank.
        if let Some(s) = &shard {
            if s.parent_shape.len() != shape.rank() {
                return Err(Error::InvalidShape(format!(
                    "shard.parent_shape.len() ({}) != shape.rank() ({})",
                    s.parent_shape.len(),
                    shape.rank()
                )));
            }

            // Spec (block-paged.md § Sharding): a block-paged descriptor MUST NOT
            // carry a shard descriptor. Enforced here — the cross-section seam where
            // both layout and shard are decoded — rather than in the layout codec.
            if matches!(&layout, LayoutDescriptor::BlockPaged(_)) {
                return Err(Error::InvalidLayout(
                    "block-paged layout MUST NOT carry a shard descriptor (spec § Sharding)"
                        .to_string(),
                ));
            }

            // Spec (csf.md § Sharding): a CSF descriptor MUST NOT carry a shard
            // descriptor in this version. Same cross-section seam as block-paged.
            if matches!(&layout, LayoutDescriptor::Csf(_)) {
                return Err(Error::InvalidLayout(
                    "CSF layout MUST NOT carry a shard descriptor (spec § Sharding)".to_string(),
                ));
            }
        }

        // Spec (block-paged.md § Quantization Compatibility): a block-paged layout carrying a
        // quantization descriptor must use a quant axis/block_size compatible with the paged
        // layout. Enforced here — the cross-section seam where the typed layout and the raw
        // quantization bytes are both in hand — by decoding the descriptor and delegating to
        // the layout's own rule. (The bytes are stored raw on the descriptor; this is the one
        // place both are known, mirroring the shard rejection above.)
        if let (LayoutDescriptor::BlockPaged(bp), Some(q)) = (&layout, &quantization) {
            let (qd, _) = QuantizationDescriptor::decode(q)?;
            let (axis, block_size) = match &qd {
                QuantizationDescriptor::PerChannelAffine(x) => (x.axis(), 0u32),
                QuantizationDescriptor::PerBlockAffine(x) => (x.axis(), x.block_size()),
                QuantizationDescriptor::Nf4(x) => (x.axis(), x.block_size()),
                QuantizationDescriptor::Mxfp(x) => (x.axis(), x.block_size()),
                QuantizationDescriptor::PerTensorAffine(_) => (0u32, 0u32),
            };
            bp.validate_quantization_compatibility(qd.scheme_tag().tag(), axis, block_size)?;
        }

        Ok(Self {
            version_major,
            version_minor,
            element_type,
            shape,
            byte_offset,
            layout,
            buffers,
            quantization,
            shard,
            statistics,
            extension_type,
            composite_member: None,
        })
    }

    /// Attaches a [`CompositeMemberDesc`] (this tensor's role within an enclosing
    /// overlay composite), returning `self` for chaining.
    ///
    /// Opt-in and separate from [`TensorDescriptor::new`] because a composite head
    /// declares `member_count` before its members exist (see ADR-027 § D2); member
    /// role assignment is necessarily a step applied after a member descriptor is
    /// otherwise fully built.
    ///
    /// The one local cross-field invariant that applies regardless of composition
    /// context — a composite head (`layout_tag = 0x0B`) MUST NOT itself carry a
    /// Composite Member section — cannot be rejected here without breaking the
    /// infallible `Self` return type this builder is specified to have; it is
    /// re-validated at [`TensorDescriptor::encode`] / [`TensorDescriptor::decode`]
    /// time instead, so the invariant still holds for every descriptor that
    /// round-trips through the wire format.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{
    ///     BufferHandle, DeviceTag, ElementType, Shape, SyncMode, MIN_BUFFER_ALIGNMENT,
    ///     descriptor::{CompositeMemberDescriptor, MemberRole, TensorDescriptor},
    ///     layout::LayoutDescriptor,
    /// };
    ///
    /// let shape  = Shape::new(vec![4u64]).unwrap();
    /// let buffer = BufferHandle::new(64, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap();
    /// let member = TensorDescriptor::new(
    ///     1, 0, ElementType::Float32, shape, 0,
    ///     LayoutDescriptor::RowMajor, vec![buffer],
    ///     None, None, None, None,
    /// )
    /// .unwrap()
    /// .with_composite_member(CompositeMemberDescriptor::new(MemberRole::Base));
    ///
    /// assert!(member.flags().has_composite_member());
    /// ```
    pub fn with_composite_member(mut self, cm: CompositeMemberDesc) -> Self {
        self.composite_member = Some(cm);
        self
    }

    /// Derives the [`DescriptorFlags`] bitmask from which optional sections are present.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{
    ///     BufferHandle, DeviceTag, ElementType, Shape, SyncMode, MIN_BUFFER_ALIGNMENT,
    ///     descriptor::{TensorDescriptor, DescriptorFlags},
    ///     layout::LayoutDescriptor,
    /// };
    ///
    /// let shape  = Shape::new(vec![4u64]).unwrap();
    /// let buffer = BufferHandle::new(64, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap();
    /// let desc   = TensorDescriptor::new(
    ///     1, 0, ElementType::Float32, shape, 0,
    ///     LayoutDescriptor::RowMajor, vec![buffer],
    ///     None, None, None, None,
    /// ).unwrap();
    /// assert_eq!(desc.flags().0, 0);
    /// ```
    pub fn flags(&self) -> DescriptorFlags {
        let mut flags = 0u32;
        if self.quantization.is_some() {
            flags |= DescriptorFlags::HAS_QUANTIZATION;
        }
        if self.shard.is_some() {
            flags |= DescriptorFlags::HAS_SHARD;
        }
        if self.extension_type.is_some() {
            flags |= DescriptorFlags::HAS_EXTENSION_TYPE;
        }
        if self.statistics.is_some() {
            flags |= DescriptorFlags::HAS_STATISTICS;
        }
        if self.composite_member.is_some() {
            flags |= DescriptorFlags::HAS_COMPOSITE_MEMBER;
        }
        DescriptorFlags(flags)
    }

    /// Encodes this descriptor to its wire representation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DescriptorLengthMismatch`] if the encoded length exceeds
    /// `u32::MAX` (practically impossible for well-formed descriptors).
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{
    ///     BufferHandle, DeviceTag, ElementType, Shape, SyncMode, MIN_BUFFER_ALIGNMENT,
    ///     descriptor::TensorDescriptor,
    ///     layout::LayoutDescriptor,
    /// };
    ///
    /// let shape  = Shape::new(vec![3u64, 4]).unwrap();
    /// let buffer = BufferHandle::new(192, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap();
    /// let desc   = TensorDescriptor::new(
    ///     1, 0, ElementType::Float32, shape, 0,
    ///     LayoutDescriptor::RowMajor, vec![buffer],
    ///     None, None, None, None,
    /// ).unwrap();
    ///
    /// let bytes = desc.encode().unwrap();
    /// assert_eq!(bytes.len(), 61); // spec § Worked Example
    /// ```
    pub fn encode(&self) -> Result<Vec<u8>> {
        encode::encode(self)
    }

    /// Decodes a [`TensorDescriptor`] from its wire representation.
    ///
    /// # Errors
    ///
    /// Returns a variant of [`Error`] for any malformed field:
    /// - [`Error::InvalidMagic`] — magic bytes are not `"HRRY"`.
    /// - [`Error::UnsupportedDescriptorVersion`] — `version_major > 1`.
    /// - [`Error::DescriptorTooShort`] — `descriptor_length < 20`.
    /// - [`Error::DescriptorTruncated`] — byte slice ends before a field is complete.
    /// - [`Error::ReservedDescriptorFlagBitsSet`] — reserved flag bits are set.
    /// - [`Error::EmptyBufferTable`] — `buffer_count == 0`.
    /// - [`Error::DescriptorLengthMismatch`] — consumed bytes ≠ `descriptor_length`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{
    ///     BufferHandle, DeviceTag, ElementType, Shape, SyncMode, MIN_BUFFER_ALIGNMENT,
    ///     descriptor::TensorDescriptor,
    ///     layout::LayoutDescriptor,
    /// };
    ///
    /// let shape  = Shape::new(vec![3u64, 4]).unwrap();
    /// let buffer = BufferHandle::new(192, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap();
    /// let desc   = TensorDescriptor::new(
    ///     1, 0, ElementType::Float32, shape, 0,
    ///     LayoutDescriptor::RowMajor, vec![buffer],
    ///     None, None, None, None,
    /// ).unwrap();
    ///
    /// let bytes   = desc.encode().unwrap();
    /// let decoded = TensorDescriptor::decode(&bytes).unwrap();
    /// assert_eq!(decoded, desc);
    /// ```
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        decode::decode(bytes)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::LayoutDescriptor;
    use crate::{BufferHandle, DeviceTag, ElementType, Shape, SyncMode, MIN_BUFFER_ALIGNMENT};

    // Helper: build the spec's worked example (float32, [3,4], row-major, 1 buffer).
    fn worked_example() -> TensorDescriptor {
        let shape = Shape::new(vec![3u64, 4]).unwrap();
        let buffer = BufferHandle::new(
            192,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )
        .unwrap();
        TensorDescriptor::new(
            1,
            0,
            ElementType::Float32,
            shape,
            0,
            LayoutDescriptor::RowMajor,
            vec![buffer],
            None,
            None,
            None,
            None,
        )
        .unwrap()
    }

    /// Spec § Worked Example: the exact 61-byte encoding.
    #[rustfmt::skip]
    const WORKED_EXAMPLE_BYTES: [u8; 61] = [
        // magic
        0x48, 0x52, 0x52, 0x59,
        // version 1.0
        0x01, 0x00,
        // descriptor_length = 61 (0x3D)
        0x3D, 0x00, 0x00, 0x00,
        // flags = 0
        0x00, 0x00, 0x00, 0x00,
        // type_tag = 0x03 (float32)
        0x03,
        // layout_tag = 0x01 (row-major)
        0x01,
        // rank = 2
        0x02, 0x00, 0x00, 0x00,
        // shape[0] = 3
        0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        // shape[1] = 4
        0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        // byte_offset = 0
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        // buffer_count = 1
        0x01,
        // buffer[0].byte_size = 192 (0xC0)
        0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        // buffer[0].alignment = 64 (0x40)
        0x40, 0x00, 0x00, 0x00,
        // buffer[0].device_tag = 0x00 (CPU)
        0x00,
        // buffer[0].sync_mode = 0x00 (ProducerSynced, ADR-018)
        0x00,
        // buffer[0]._reserved[2]
        0x00, 0x00,
    ];

    #[test]
    fn worked_example_encode_exact_bytes() {
        let desc = worked_example();
        let encoded = desc.encode().unwrap();
        assert_eq!(
            encoded.len(),
            61,
            "expected 61 bytes per spec worked example, got {}",
            encoded.len()
        );
        assert_eq!(
            encoded.as_slice(),
            &WORKED_EXAMPLE_BYTES,
            "encoded bytes do not match spec worked example"
        );
    }

    #[test]
    fn worked_example_decode_round_trip() {
        let desc = worked_example();
        let encoded = desc.encode().unwrap();
        let decoded = TensorDescriptor::decode(&encoded).unwrap();
        assert_eq!(decoded, desc);
    }

    #[test]
    fn worked_example_decode_from_spec_bytes() {
        let decoded = TensorDescriptor::decode(&WORKED_EXAMPLE_BYTES).unwrap();
        assert_eq!(decoded, worked_example());
    }

    // ── Negative tests ────────────────────────────────────────────────────────

    #[test]
    fn decode_bad_magic() {
        let mut bytes = WORKED_EXAMPLE_BYTES;
        bytes[0] = 0xFF;
        let err = TensorDescriptor::decode(&bytes).unwrap_err();
        assert!(matches!(err, Error::InvalidMagic { .. }));
    }

    #[test]
    fn decode_version_major_too_high() {
        let mut bytes = WORKED_EXAMPLE_BYTES;
        bytes[4] = 2; // version_major = 2 > DESCRIPTOR_VERSION_MAJOR
        let err = TensorDescriptor::decode(&bytes).unwrap_err();
        assert!(matches!(
            err,
            Error::UnsupportedDescriptorVersion { major: 2, .. }
        ));
    }

    #[test]
    fn decode_descriptor_too_short() {
        let mut bytes = WORKED_EXAMPLE_BYTES;
        // descriptor_length = 10 < MIN_DESCRIPTOR_LEN (20)
        bytes[6] = 10;
        bytes[7] = 0;
        bytes[8] = 0;
        bytes[9] = 0;
        let err = TensorDescriptor::decode(&bytes).unwrap_err();
        assert!(matches!(err, Error::DescriptorTooShort { length: 10 }));
    }

    #[test]
    fn decode_reserved_flag_bit_set() {
        let mut bytes = WORKED_EXAMPLE_BYTES;
        // Bit 4 is now HAS_COMPOSITE_MEMBER (ADR-027) and no longer reserved; set
        // bit 5 (the first genuinely-still-reserved flag bit) at offset 10.
        bytes[10] = 0x20;
        let err = TensorDescriptor::decode(&bytes).unwrap_err();
        assert!(matches!(err, Error::ReservedDescriptorFlagBitsSet { .. }));
    }

    #[test]
    fn decode_empty_buffer_table() {
        // Build a valid descriptor, then patch buffer_count to 0.
        let desc = worked_example();
        let mut bytes = desc.encode().unwrap();
        // buffer_count is at offset 44 in the worked example.
        bytes[44] = 0;
        let err = TensorDescriptor::decode(&bytes).unwrap_err();
        assert!(matches!(err, Error::EmptyBufferTable));
    }

    #[test]
    fn decode_truncated() {
        // Fewer than 10 bytes — cannot even read the header.
        let bytes = &[0x48u8, 0x52, 0x52];
        let err = TensorDescriptor::decode(bytes).unwrap_err();
        assert!(matches!(err, Error::DescriptorTruncated { .. }));
    }

    // ── flags() helper ────────────────────────────────────────────────────────

    #[test]
    fn flags_none_set() {
        assert_eq!(worked_example().flags().0, 0);
    }

    #[test]
    fn flags_quantization_set() {
        let shape = Shape::new(vec![4u64]).unwrap();
        let buf = BufferHandle::new(
            64,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )
        .unwrap();
        let desc = TensorDescriptor::new(
            1,
            0,
            ElementType::Int8,
            shape,
            0,
            LayoutDescriptor::RowMajor,
            vec![buf],
            Some(vec![0x01, 0x00, 0x00, 0x00]),
            None,
            None,
            None,
        )
        .unwrap();
        assert!(desc.flags().has_quantization());
        assert!(!desc.flags().has_shard());
    }

    // ── constructor validation ────────────────────────────────────────────────

    #[test]
    fn new_rejects_empty_buffers() {
        let shape = Shape::new(vec![4u64]).unwrap();
        let err = TensorDescriptor::new(
            1,
            0,
            ElementType::Float32,
            shape,
            0,
            LayoutDescriptor::RowMajor,
            vec![],
            None,
            None,
            None,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, Error::EmptyBufferTable));
    }

    #[test]
    fn new_rejects_shard_rank_mismatch() {
        let shape = Shape::new(vec![4u64, 4]).unwrap(); // rank 2
        let buf = BufferHandle::new(
            64,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )
        .unwrap();
        // shard with rank 1 (mismatch)
        let shard = ShardDescriptor::new(vec![10u64], vec![0u64]).unwrap();
        let err = TensorDescriptor::new(
            1,
            0,
            ElementType::Float32,
            shape,
            0,
            LayoutDescriptor::RowMajor,
            vec![buf],
            None,
            Some(shard),
            None,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidShape(_)));
    }

    #[test]
    fn new_rejects_block_paged_with_shard() {
        use crate::layout::{BlockPagedLayout, BlockTableIndexType, KvRole};

        // Block-paged is rank-3; give the shard a matching rank-3 parent_shape so the
        // rank check passes and we reach the block-paged § Sharding prohibition.
        let shape = Shape::new(vec![6u64, 2, 8]).unwrap();
        let buf = BufferHandle::new(
            64,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )
        .unwrap();
        let shard = ShardDescriptor::new(vec![12u64, 2, 8], vec![0u64, 0, 0]).unwrap();
        let layout = LayoutDescriptor::BlockPaged(BlockPagedLayout::new(
            4,
            5,
            0,
            2,
            KvRole::Key,
            Some(3),
            BlockTableIndexType::U32,
        ));
        let err = TensorDescriptor::new(
            1,
            0,
            ElementType::Float16,
            shape,
            0,
            layout,
            vec![buf],
            None,
            Some(shard),
            None,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidLayout(_)));
    }

    #[test]
    fn new_rejects_block_paged_with_incompatible_quantization() {
        use crate::layout::{BlockPagedLayout, BlockTableIndexType, KvRole};
        use crate::quantization::PerChannelAffine;

        // Per-channel quantization on the paged axis (axis 0) is prohibited for a
        // block-paged layout (block-paged.md § Quantization Compatibility). This is the
        // cross-section check wired into TensorDescriptor::new.
        let quant = QuantizationDescriptor::PerChannelAffine(
            PerChannelAffine::new_symmetric(0, 1).unwrap(),
        )
        .encode_to_vec();
        let shape = Shape::new(vec![6u64, 2, 8]).unwrap();
        let buf = BufferHandle::new(
            64,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )
        .unwrap();
        let layout = LayoutDescriptor::BlockPaged(BlockPagedLayout::new(
            4,
            5,
            0,
            2,
            KvRole::Key,
            Some(3),
            BlockTableIndexType::U32,
        ));
        let err = TensorDescriptor::new(
            1,
            0,
            ElementType::Float16,
            shape,
            0,
            layout,
            vec![buf],
            Some(quant),
            None,
            None,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidQuantization(_)));
    }

    #[test]
    fn new_accepts_block_paged_with_compatible_quantization() {
        use crate::layout::{BlockPagedLayout, BlockTableIndexType, KvRole};
        use crate::quantization::PerChannelAffine;

        // Per-channel on axis 1 (num_heads) is permitted for block-paged.
        let quant = QuantizationDescriptor::PerChannelAffine(
            PerChannelAffine::new_symmetric(1, 1).unwrap(),
        )
        .encode_to_vec();
        let shape = Shape::new(vec![6u64, 2, 8]).unwrap();
        let buf = BufferHandle::new(
            64,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )
        .unwrap();
        let layout = LayoutDescriptor::BlockPaged(BlockPagedLayout::new(
            4,
            5,
            0,
            2,
            KvRole::Key,
            Some(3),
            BlockTableIndexType::U32,
        ));
        let desc = TensorDescriptor::new(
            1,
            0,
            ElementType::Float16,
            shape,
            0,
            layout,
            vec![buf],
            Some(quant),
            None,
            None,
            None,
        );
        assert!(desc.is_ok());
    }

    #[test]
    fn new_rejects_csf_with_shard() {
        use crate::layout::CsfLayout;

        // CSF is rank-3+; give the shard a matching rank-3 parent_shape so the rank
        // check passes and we reach the CSF § Sharding prohibition.
        let shape = Shape::new(vec![2u64, 3, 4]).unwrap();
        let buf = BufferHandle::new(
            64,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )
        .unwrap();
        let shard = ShardDescriptor::new(vec![4u64, 3, 4], vec![0u64, 0, 0]).unwrap();
        let layout = LayoutDescriptor::Csf(CsfLayout::new(4, vec![0, 1, 2]));
        let err = TensorDescriptor::new(
            1,
            0,
            ElementType::Float32,
            shape,
            0,
            layout,
            vec![buf],
            None,
            Some(shard),
            None,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidLayout(_)));
    }

    // ── round-trip with optional sections ────────────────────────────────────

    #[test]
    fn round_trip_with_shard() {
        use crate::layout::LayoutDescriptor;
        let shape = Shape::new(vec![4u64, 8]).unwrap();
        let buf = BufferHandle::new(
            128,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )
        .unwrap();
        let shard = ShardDescriptor::new(vec![10u64, 20], vec![0u64, 4]).unwrap();
        let desc = TensorDescriptor::new(
            1,
            0,
            ElementType::Float32,
            shape,
            0,
            LayoutDescriptor::RowMajor,
            vec![buf],
            None,
            Some(shard),
            None,
            None,
        )
        .unwrap();
        let decoded = TensorDescriptor::decode(&desc.encode().unwrap()).unwrap();
        assert_eq!(decoded, desc);
    }

    #[test]
    fn round_trip_with_quantization_payload() {
        let shape = Shape::new(vec![8u64]).unwrap();
        let buf = BufferHandle::new(
            64,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )
        .unwrap();
        let quant_bytes = vec![0x01u8, 0x00, 0x00, 0x00, 0xAB, 0xCD];
        let desc = TensorDescriptor::new(
            1,
            0,
            ElementType::Int8,
            shape,
            0,
            LayoutDescriptor::RowMajor,
            vec![buf],
            Some(quant_bytes),
            None,
            None,
            None,
        )
        .unwrap();
        let decoded = TensorDescriptor::decode(&desc.encode().unwrap()).unwrap();
        assert_eq!(decoded, desc);
    }

    #[test]
    fn round_trip_with_statistics() {
        use crate::descriptor::statistics::{Statistics, StatisticsMask};
        let shape = Shape::new(vec![16u64]).unwrap();
        let buf = BufferHandle::new(
            128,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )
        .unwrap();
        let stats = Statistics {
            computed_mask: StatisticsMask(StatisticsMask::NNZ_VALID),
            nnz: 10,
            sparsity_ratio: 0.0,
            value_min: 0.0,
            value_max: 0.0,
            value_abs_max: 0.0,
            value_mean: 0.0,
            value_stddev: 0.0,
            nm_n: 0,
            nm_m: 0,
            has_nan: false,
            has_inf: false,
        };
        let desc = TensorDescriptor::new(
            1,
            0,
            ElementType::Float32,
            shape,
            0,
            LayoutDescriptor::RowMajor,
            vec![buf],
            None,
            None,
            Some(stats),
            None,
        )
        .unwrap();
        let encoded = desc.encode().unwrap();
        let decoded = TensorDescriptor::decode(&encoded).unwrap();
        assert_eq!(decoded.statistics.as_ref().unwrap().nnz, 10);
        assert!(decoded.flags().has_statistics());
    }

    #[test]
    fn round_trip_strided_layout() {
        use crate::layout::StridedLayout;
        let shape = Shape::new(vec![3u64, 4]).unwrap();
        let buf = BufferHandle::new(
            192,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )
        .unwrap();
        let layout = LayoutDescriptor::Strided(StridedLayout::new(vec![8i64, 1]));
        let desc = TensorDescriptor::new(
            1,
            0,
            ElementType::Float32,
            shape,
            0,
            layout,
            vec![buf],
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let decoded = TensorDescriptor::decode(&desc.encode().unwrap()).unwrap();
        assert_eq!(decoded, desc);
    }

    #[test]
    fn round_trip_coo_layout() {
        use crate::layout::CooLayout;
        let shape = Shape::new(vec![10u64, 10]).unwrap();
        // COO requires 2 buffers (values + indices)
        let buf0 = BufferHandle::new(
            64,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )
        .unwrap();
        let buf1 = BufferHandle::new(
            128,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )
        .unwrap();
        let layout = LayoutDescriptor::Coo(CooLayout::new(5, true));
        let desc = TensorDescriptor::new(
            1,
            0,
            ElementType::Float32,
            shape,
            0,
            layout,
            vec![buf0, buf1],
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let decoded = TensorDescriptor::decode(&desc.encode().unwrap()).unwrap();
        assert_eq!(decoded, desc);
    }

    // ── C-1: Extension type round-trip ────────────────────────────────────────

    /// A descriptor with ElementType::Extension(0xF1) and a populated
    /// ExtensionTypeDescriptor must round-trip through encode/decode unchanged.
    #[test]
    fn round_trip_extension_element_type() {
        use crate::descriptor::ExtensionTypeDescriptor;
        let shape = Shape::new(vec![4u64]).unwrap();
        let buf = BufferHandle::new(
            64,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )
        .unwrap();
        // 8-bit integer extension type, packing_factor=1.
        let ext =
            ExtensionTypeDescriptor::new(8, 1, false, true, 0, 0, 0, 0, false, false).unwrap();
        let desc = TensorDescriptor::new(
            1,
            0,
            ElementType::Extension(0xF1),
            shape,
            0,
            LayoutDescriptor::RowMajor,
            vec![buf],
            None,
            None,
            None,
            Some(ext.clone()),
        )
        .unwrap();
        let encoded = desc.encode().unwrap();
        let decoded = TensorDescriptor::decode(&encoded).unwrap();
        assert_eq!(decoded.element_type, ElementType::Extension(0xF1));
        assert_eq!(decoded.extension_type.as_ref().unwrap().bit_width, 8);
        assert_eq!(decoded, desc);
    }

    // ── H-1: Buffer count ceiling ─────────────────────────────────────────────

    /// Constructing a TensorDescriptor with 256 buffers must be rejected at
    /// new() with InvalidLayout, not silently truncated.
    #[test]
    fn new_rejects_buffer_count_exceeding_255() {
        let shape = Shape::new(vec![4u64]).unwrap();
        let buffers: Vec<BufferHandle> = (0..256)
            .map(|_| {
                BufferHandle::new(
                    64,
                    MIN_BUFFER_ALIGNMENT,
                    DeviceTag::Cpu,
                    SyncMode::ProducerSynced,
                )
                .unwrap()
            })
            .collect();
        let err = TensorDescriptor::new(
            1,
            0,
            ElementType::Float32,
            shape,
            0,
            LayoutDescriptor::RowMajor,
            buffers,
            None,
            None,
            None,
            None,
        )
        .unwrap_err();
        assert!(
            matches!(err, Error::InvalidLayout(_)),
            "expected InvalidLayout, got {err:?}"
        );
    }

    // ── M-2: descriptor_length mismatch ───────────────────────────────────────

    /// Patching descriptor_length to declared+1 (without adding a real byte)
    /// must produce DescriptorLengthMismatch on decode.
    #[test]
    fn decode_descriptor_length_mismatch() {
        let desc = worked_example();
        let mut bytes = desc.encode().unwrap();
        // descriptor_length is a little-endian u32 at bytes[6..10].
        let declared = u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]);
        let patched = declared + 1;
        bytes[6..10].copy_from_slice(&patched.to_le_bytes());
        let err = TensorDescriptor::decode(&bytes).unwrap_err();
        assert!(
            matches!(err, Error::DescriptorLengthMismatch { .. }),
            "expected DescriptorLengthMismatch, got {err:?}"
        );
    }

    // ── M-3: tiled nesting too deep ───────────────────────────────────────────

    /// A tiled descriptor nested 9 levels deep (exceeding MAX_RECURSION_DEPTH=8)
    /// must be rejected with SubpavingNestingTooDeep on decode.
    ///
    /// The layout bytes are hand-crafted to bypass TiledLayout::new's own depth
    /// guard (which fires at construction), producing wire bytes that the decoder
    /// must reject at the depth limit.
    #[test]
    fn decode_tiled_nesting_too_deep() {
        // Hand-craft 9-level nested tiled layout bytes for rank=1.
        // Each tiled level for rank=1 with inner_layout=TAG_TILED (0x04):
        //   tile_shape u64[1] = 8 bytes
        //   outer_layout u8   = 0x01 (row-major)
        //   inner_layout u8   = 0x04 (tiled, recurse) OR 0x01 (innermost)
        //   _reserved u16     = 0x0000
        // Total per level: 12 bytes.
        // 9 levels (0..=8): 9 × 12 = 108 bytes of layout payload.
        let mut layout_bytes: Vec<u8> = Vec::new();
        for level in 0..9u32 {
            // tile_shape[0] = 2 (LE u64)
            layout_bytes.extend_from_slice(&2u64.to_le_bytes());
            // outer_layout = 0x01 (row-major)
            layout_bytes.push(0x01);
            // inner_layout: 0x04 (nested tiled) for levels 0..8, 0x01 (row-major) for level 8
            if level < 8 {
                layout_bytes.push(0x04); // TAG_TILED — recurse
            } else {
                layout_bytes.push(0x01); // innermost — row-major
            }
            // _reserved[2]
            layout_bytes.extend_from_slice(&[0x00, 0x00]);
        }

        // Build a minimal valid descriptor for a rank-1 float32 tensor with a
        // shallow (1-level) tiled layout, then splice in the 9-level bytes.
        use crate::layout::TiledLayout;
        let shape = Shape::new(vec![2u64]).unwrap();
        let buf = BufferHandle::new(
            64,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )
        .unwrap();
        let shallow = LayoutDescriptor::Tiled(Box::new(
            TiledLayout::new(vec![2u64], 0x01, 0x01, None, None, None).unwrap(),
        ));
        let desc = TensorDescriptor::new(
            1,
            0,
            ElementType::Float32,
            shape,
            0,
            shallow,
            vec![buf],
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let mut encoded = desc.encode().unwrap();

        // Splice: layout payload starts at offset 36
        // (fixed header 20 + shape u64[1] 8 + byte_offset u64 8).
        // The shallow 1-level tiled layout for rank=1: 12 bytes.
        let layout_start = 36usize;
        let shallow_layout_len = 12usize;
        let after_layout = layout_start + shallow_layout_len;

        let rest = encoded[after_layout..].to_vec();
        encoded.truncate(layout_start);
        encoded.extend_from_slice(&layout_bytes);
        encoded.extend_from_slice(&rest);

        // Back-patch descriptor_length (bytes 6..10).
        let new_len = encoded.len() as u32;
        encoded[6..10].copy_from_slice(&new_len.to_le_bytes());

        // Decoding must fail at the depth guard (MAX_RECURSION_DEPTH = 8).
        let err = TensorDescriptor::decode(&encoded).unwrap_err();
        assert!(
            matches!(err, Error::SubpavingNestingTooDeep),
            "expected SubpavingNestingTooDeep, got {err:?}"
        );
    }

    // ── ADR-027 D-1: composite head buffer-table carve-out ───────────────────

    fn composite_head(member_count: u32) -> TensorDescriptor {
        use crate::layout::{CompositeLayout, CompositionRule};
        let layout = LayoutDescriptor::Composite(
            CompositeLayout::new(CompositionRule::Group, member_count).unwrap(),
        );
        TensorDescriptor::new(
            1,
            0,
            ElementType::Float32,
            Shape::new(vec![4u64]).unwrap(),
            0,
            layout,
            vec![],
            None,
            None,
            None,
            None,
        )
        .unwrap()
    }

    /// A composite head (`layout_tag = 0x0B`) with an empty buffer table is the
    /// *only* valid buffer table for that layout, and round-trips cleanly.
    #[test]
    fn composite_head_empty_buffers_round_trips() {
        let desc = composite_head(0);
        let decoded = TensorDescriptor::decode(&desc.encode().unwrap()).unwrap();
        assert_eq!(decoded, desc);
        assert!(decoded.buffers.is_empty());
    }

    /// A non-empty buffer table on a composite head is rejected.
    #[test]
    fn composite_head_nonzero_buffer_count_rejected() {
        use crate::layout::{CompositeLayout, CompositionRule};
        let buf = BufferHandle::new(
            64,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )
        .unwrap();
        let layout =
            LayoutDescriptor::Composite(CompositeLayout::new(CompositionRule::Group, 0).unwrap());
        let err = TensorDescriptor::new(
            1,
            0,
            ElementType::Float32,
            Shape::new(vec![4u64]).unwrap(),
            0,
            layout,
            vec![buf],
            None,
            None,
            None,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, Error::CompositeHeadHasBuffers { count: 1 }));
    }

    /// A non-zero `byte_offset` on a composite head is rejected.
    #[test]
    fn composite_head_nonzero_byte_offset_rejected() {
        use crate::layout::{CompositeLayout, CompositionRule};
        let layout =
            LayoutDescriptor::Composite(CompositeLayout::new(CompositionRule::Group, 0).unwrap());
        let err = TensorDescriptor::new(
            1,
            0,
            ElementType::Float32,
            Shape::new(vec![4u64]).unwrap(),
            8, // non-zero byte_offset
            layout,
            vec![],
            None,
            None,
            None,
            None,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            Error::CompositeHeadHasByteOffset { byte_offset: 8 }
        ));
    }

    /// A composite head MUST NOT set `HAS_QUANTIZATION` (it owns no stored data).
    #[test]
    fn composite_head_with_quantization_rejected() {
        use crate::layout::{CompositeLayout, CompositionRule};
        let layout =
            LayoutDescriptor::Composite(CompositeLayout::new(CompositionRule::Group, 0).unwrap());
        let err = TensorDescriptor::new(
            1,
            0,
            ElementType::Float32,
            Shape::new(vec![4u64]).unwrap(),
            0,
            layout,
            vec![],
            Some(vec![0x01, 0x00, 0x00, 0x00]),
            None,
            None,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, Error::CompositeHeadHasQuantization));
    }

    /// A composite head carrying a Composite Member section itself (not a
    /// member of a further-enclosing composite) is rejected at encode time.
    #[test]
    fn composite_head_with_composite_member_rejected_at_encode() {
        use crate::descriptor::composite_member::CompositeMemberDescriptor;
        let desc = composite_head(0).with_composite_member(CompositeMemberDescriptor::new(
            crate::descriptor::MemberRole::Base,
        ));
        let err = desc.encode().unwrap_err();
        assert!(matches!(err, Error::InvalidLayout(_)));
    }

    /// The same invariant, enforced symmetrically on the decode path: a
    /// hand-crafted composite head with `HAS_COMPOSITE_MEMBER` set and a valid
    /// 16-byte Composite Member section appended is rejected.
    #[test]
    fn composite_head_with_composite_member_rejected_at_decode() {
        let desc = composite_head(0);
        let mut encoded = desc.encode().unwrap();

        // Set HAS_COMPOSITE_MEMBER (bit 4 = 0x10) in the flags field at offset 10.
        encoded[10] |= DescriptorFlags::HAS_COMPOSITE_MEMBER as u8;

        // Append a valid 16-byte Composite Member section (member_role=0x00, 15
        // zero reserved bytes) — this is the last section per spec wire order.
        let mut cm_bytes = vec![0x00u8]; // member_role = Correction
        cm_bytes.extend(std::iter::repeat_n(0u8, 15)); // _reserved
        encoded.extend_from_slice(&cm_bytes);

        // Back-patch descriptor_length.
        let new_len = encoded.len() as u32;
        encoded[6..10].copy_from_slice(&new_len.to_le_bytes());

        let err = TensorDescriptor::decode(&encoded).unwrap_err();
        assert!(matches!(err, Error::InvalidLayout(_)));
    }

    /// A non-composite descriptor with `with_composite_member` set round-trips
    /// through full encode/decode.
    #[test]
    fn with_composite_member_round_trips() {
        use crate::descriptor::composite_member::CompositeMemberDescriptor;
        let desc = worked_example().with_composite_member(CompositeMemberDescriptor::new(
            crate::descriptor::MemberRole::Base,
        ));
        assert!(desc.flags().has_composite_member());
        let decoded = TensorDescriptor::decode(&desc.encode().unwrap()).unwrap();
        assert_eq!(decoded, desc);
        assert_eq!(
            decoded.composite_member.unwrap().member_role,
            crate::descriptor::MemberRole::Base
        );
    }
}
