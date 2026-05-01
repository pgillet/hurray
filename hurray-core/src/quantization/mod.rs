//! Quantization descriptor types for the Hurray tensor format.
//!
//! Every quantized tensor carries a **quantization descriptor** immediately
//! following the tensor descriptor header. The descriptor starts with a 4-byte
//! header (scheme tag, scheme version, flags) followed by a scheme-specific
//! payload.
//!
//! ## Supported schemes
//!
//! | Scheme tag | Tier | Type |
//! |------------|------|------|
//! | `0x01` | 1 | [`PerTensorAffine`] |
//! | `0x02` | 1 | [`PerChannelAffine`] |
//! | `0x03` | 1 | [`PerBlockAffine`] |
//! | `0x04` | 1 | [`Nf4`] |
//! | `0x05` | 1 | [`Mxfp`] |
//!
//! ## Encoding overview
//!
//! The canonical encoding path is:
//!
//! 1. Call [`QuantizationDescriptor::encode_into`] to write into a caller-owned
//!    buffer (zero-alloc, required by Layer 5 streaming writers).
//! 2. Or call [`QuantizationDescriptor::encode_to_vec`] for a convenience
//!    `Vec<u8>`.
//!
//! The canonical decoding path is:
//!
//! - Call [`QuantizationDescriptor::decode`] which returns the descriptor and
//!   the number of bytes consumed, allowing streaming readers to advance their
//!   cursor without re-scanning.
//!
//! See `docs/spec/quantization.md` for the normative definition.

pub mod mxfp;
pub mod nf4;
pub mod per_block_affine;
pub mod per_channel_affine;
pub mod per_tensor_affine;

pub use mxfp::{Mxfp, CANONICAL_BLOCK_SIZE, MAX_BLOCK_SIZE, MIN_BLOCK_SIZE};
pub use nf4::{Nf4, NF4_LUT};
pub use per_block_affine::PerBlockAffine;
pub use per_channel_affine::PerChannelAffine;
pub use per_tensor_affine::PerTensorAffine;

use crate::{BufferHandle, ElementType, Error, Result};

// ── Scheme tag ranges ─────────────────────────────────────────────────────────

// Tier 2 assigned range: 0x40–0x5F
const TIER2_MIN: u8 = 0x40;
const TIER2_MAX: u8 = 0x5F;

// Reserved range: 0x60–0xEF
const RESERVED_MIN: u8 = 0x60;
const RESERVED_MAX: u8 = 0xEF;

// Private range: 0xF0–0xFE
const PRIVATE_MIN: u8 = 0xF0;
const PRIVATE_MAX: u8 = 0xFE;

// ── QuantizationSchemeTag ─────────────────────────────────────────────────────

/// The wire scheme tag that identifies which quantization scheme a descriptor uses.
///
/// The tag byte occupies offset `0` of the quantization descriptor. It determines
/// both the payload layout and the set of valid storage types for the tensor.
///
/// # Tag ranges
///
/// | Range | Meaning |
/// |-------|---------|
/// | `0x01`–`0x05` | Tier 1 — assigned by this spec version |
/// | `0x06`–`0x3F` | Unallocated Tier 1 — reader MUST treat as unknown |
/// | `0x40`–`0x5F` | Tier 2 — reserved for future assignment |
/// | `0x60`–`0xEF` | Reserved — reader MUST reject |
/// | `0xF0`–`0xFE` | Private — reader MUST reject (unconstrained payload) |
/// | `0x00`, `0xFF` | Permanently invalid — reader MUST reject |
///
/// # Examples
///
/// ```
/// use hurray_core::QuantizationSchemeTag;
///
/// assert_eq!(QuantizationSchemeTag::PerTensorAffine.tag(), 0x01);
/// assert_eq!(QuantizationSchemeTag::Mxfp.tag(), 0x05);
/// assert_eq!(QuantizationSchemeTag::PerTensorAffine.tier(), 1);
/// assert_eq!(QuantizationSchemeTag::Mxfp.tier(), 2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum QuantizationSchemeTag {
    /// Per-tensor affine quantization. Tag `0x01`, Tier 1.
    PerTensorAffine = 0x01,
    /// Per-channel affine quantization. Tag `0x02`, Tier 1.
    PerChannelAffine = 0x02,
    /// Per-block affine quantization. Tag `0x03`, Tier 1.
    PerBlockAffine = 0x03,
    /// NF4 (NormalFloat4) block quantization. Tag `0x04`, Tier 1.
    Nf4 = 0x04,
    /// MXFP (OCP Microscaling) block quantization. Tag `0x05`, Tier 1.
    Mxfp = 0x05,
}

impl QuantizationSchemeTag {
    /// Returns the one-byte wire representation of this scheme tag.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::QuantizationSchemeTag;
    ///
    /// assert_eq!(QuantizationSchemeTag::PerTensorAffine.tag(), 0x01);
    /// assert_eq!(QuantizationSchemeTag::Mxfp.tag(), 0x05);
    /// ```
    #[inline]
    pub fn tag(self) -> u8 {
        self as u8
    }

    /// Returns the tier of this scheme: `1` for mandatory schemes, `2` for optional schemes.
    ///
    /// Per `docs/spec/quantization.md` § Tag Assignment Table:
    /// - Tier 1 (`MUST` support): `PerTensorAffine`, `PerChannelAffine`, `PerBlockAffine`.
    /// - Tier 2 (OPTIONAL): `Nf4`, `Mxfp`.
    ///
    /// Note: NF4 (`0x04`) and MXFP (`0x05`) have wire tags in the `0x01`–`0x3F` numeric
    /// range but are designated Tier 2 by the spec scheme table. Tier is a support-obligation
    /// property, not a simple function of the tag byte.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::QuantizationSchemeTag;
    ///
    /// assert_eq!(QuantizationSchemeTag::PerTensorAffine.tier(), 1);
    /// assert_eq!(QuantizationSchemeTag::PerBlockAffine.tier(), 1);
    /// assert_eq!(QuantizationSchemeTag::Nf4.tier(), 2);
    /// assert_eq!(QuantizationSchemeTag::Mxfp.tier(), 2);
    /// ```
    #[inline]
    pub fn tier(self) -> u8 {
        // WHY explicit match: NF4/MXFP are Tier 2 per spec scheme table despite having
        // tags in the 0x01–0x3F numeric range; a simple byte threshold would be wrong.
        match self {
            Self::PerTensorAffine | Self::PerChannelAffine | Self::PerBlockAffine => 1,
            Self::Nf4 | Self::Mxfp => 2,
        }
    }

    /// Parses a [`QuantizationSchemeTag`] from its one-byte wire representation.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidQuantizationSchemeTag`] — byte is `0x00` or `0xFF`
    ///   (permanently reserved).
    /// - [`Error::ReservedQuantizationSchemeTag`] — byte is in `0x60`–`0xEF`
    ///   (reserved for future specification versions).
    /// - [`Error::PrivateQuantizationSchemeTag`] — byte is in `0xF0`–`0xFE`.
    ///   Private tags have unconstrained payloads beyond the 4-byte header;
    ///   callers that need private scheme support must handle raw bytes at a
    ///   higher layer (design decision #1).
    /// - [`Error::UnknownQuantizationSchemeTag`] — byte is in `0x06`–`0x3F`
    ///   (unallocated Tier 1 range) or in `0x40`–`0x5F` but not yet assigned.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{QuantizationSchemeTag, Error};
    ///
    /// assert_eq!(
    ///     QuantizationSchemeTag::from_byte(0x01).unwrap(),
    ///     QuantizationSchemeTag::PerTensorAffine
    /// );
    /// assert!(matches!(
    ///     QuantizationSchemeTag::from_byte(0x00),
    ///     Err(Error::InvalidQuantizationSchemeTag(0x00))
    /// ));
    /// assert!(matches!(
    ///     QuantizationSchemeTag::from_byte(0x60),
    ///     Err(Error::ReservedQuantizationSchemeTag(0x60))
    /// ));
    /// assert!(matches!(
    ///     QuantizationSchemeTag::from_byte(0xF0),
    ///     Err(Error::PrivateQuantizationSchemeTag(0xF0))
    /// ));
    /// assert!(matches!(
    ///     QuantizationSchemeTag::from_byte(0x06),
    ///     Err(Error::UnknownQuantizationSchemeTag(0x06))
    /// ));
    /// ```
    pub fn from_byte(b: u8) -> Result<Self> {
        match b {
            // Permanently invalid sentinels.
            0x00 | 0xFF => Err(Error::InvalidQuantizationSchemeTag(b)),

            // Tier 1 assigned range.
            0x01 => Ok(Self::PerTensorAffine),
            0x02 => Ok(Self::PerChannelAffine),
            0x03 => Ok(Self::PerBlockAffine),
            0x04 => Ok(Self::Nf4),
            0x05 => Ok(Self::Mxfp),

            // Unallocated range: 0x06–0x3F (arms above exhausted all assigned tags 0x01–0x05).
            b if (0x06..=0x3F).contains(&b) => Err(Error::UnknownQuantizationSchemeTag(b)),

            // Future Tier 2 numeric range: 0x40–0x5F.
            // No tags in this range are currently assigned.
            b if (TIER2_MIN..=TIER2_MAX).contains(&b) => {
                Err(Error::UnknownQuantizationSchemeTag(b))
            }

            // Reserved range: 0x60–0xEF.
            b if (RESERVED_MIN..=RESERVED_MAX).contains(&b) => {
                Err(Error::ReservedQuantizationSchemeTag(b))
            }

            // Private range: 0xF0–0xFE (reject — unconstrained wire format beyond header
            // gives callers nothing useful; design decision #1).
            b if (PRIVATE_MIN..=PRIVATE_MAX).contains(&b) => {
                Err(Error::PrivateQuantizationSchemeTag(b))
            }

            _ => Err(Error::UnknownQuantizationSchemeTag(b)),
        }
    }
}

// ── QuantizationHeader ────────────────────────────────────────────────────────

/// The 4-byte common header present at the start of every quantization descriptor.
///
/// | Offset | Field | Type |
/// |--------|-------|------|
/// | 0 | `scheme_tag` | `uint8` |
/// | 1 | `scheme_version` | `uint8` |
/// | 2 | `flags` | `uint16` LE |
///
/// All multi-byte fields are little-endian.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QuantizationHeader {
    pub scheme_tag: QuantizationSchemeTag,
    pub scheme_version: u8,
    pub flags: u16,
}

impl QuantizationHeader {
    /// Minimum number of bytes required to read a header.
    pub(crate) const LEN: usize = 4;

    /// Parses a [`QuantizationHeader`] from the first 4 bytes of `bytes`.
    ///
    /// Validates that `scheme_version` does not exceed `supported_version`.
    ///
    /// # Errors
    ///
    /// - [`Error::QuantizationDescriptorTooShort`] — fewer than 4 bytes available.
    /// - Scheme tag errors forwarded from [`QuantizationSchemeTag::from_byte`].
    /// - [`Error::UnsupportedSchemeVersion`] — `scheme_version` exceeds
    ///   `supported_version`.
    pub(crate) fn parse(bytes: &[u8], supported_version: u8) -> Result<Self> {
        if bytes.len() < Self::LEN {
            return Err(Error::QuantizationDescriptorTooShort {
                found: bytes.len(),
                needed: Self::LEN,
            });
        }
        let scheme_tag_byte = bytes[0];
        let scheme_version = bytes[1];
        let flags = u16::from_le_bytes([bytes[2], bytes[3]]);

        let scheme_tag = QuantizationSchemeTag::from_byte(scheme_tag_byte)?;

        if scheme_version > supported_version {
            return Err(Error::UnsupportedSchemeVersion {
                tag: scheme_tag_byte,
                version: scheme_version,
                supported: supported_version,
            });
        }

        Ok(Self {
            scheme_tag,
            scheme_version,
            flags,
        })
    }

    /// Writes this header into the first 4 bytes of `out`.
    ///
    /// `out` must be at least 4 bytes long.
    pub(crate) fn write(&self, out: &mut [u8]) {
        out[0] = self.scheme_tag.tag();
        out[1] = self.scheme_version;
        let flags_le = self.flags.to_le_bytes();
        out[2] = flags_le[0];
        out[3] = flags_le[1];
    }
}

// ── QuantizationDescriptor ────────────────────────────────────────────────────

/// A typed quantization descriptor carrying the parameters for one of the
/// supported quantization schemes.
///
/// # Equality and hashing
///
/// Only [`PartialEq`] is derived — **not** `Eq` or `Hash` — because the
/// [`PerTensorAffine`] variant contains `f32` fields. IEEE 754 NaN != NaN, which
/// would make `Eq` semantically unsound (design decision #2).
///
/// # Examples
///
/// ```
/// use hurray_core::{PerTensorAffine, QuantizationDescriptor, QuantizationSchemeTag};
///
/// let desc = QuantizationDescriptor::PerTensorAffine(
///     PerTensorAffine::new(0.5, 128).unwrap(),
/// );
/// assert_eq!(desc.scheme_tag(), QuantizationSchemeTag::PerTensorAffine);
/// assert_eq!(desc.encoded_len(), 16);
/// ```
// WHY PartialEq only: f32 fields make Eq/Hash unsound due to NaN semantics
// (design decision #2).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum QuantizationDescriptor {
    /// Per-tensor affine quantization (scheme tag `0x01`).
    PerTensorAffine(PerTensorAffine),
    /// Per-channel affine quantization (scheme tag `0x02`).
    PerChannelAffine(PerChannelAffine),
    /// Per-block affine quantization (scheme tag `0x03`).
    PerBlockAffine(PerBlockAffine),
    /// NF4 (NormalFloat4) block quantization (scheme tag `0x04`).
    Nf4(Nf4),
    /// MXFP (OCP Microscaling) block quantization (scheme tag `0x05`).
    Mxfp(Mxfp),
}

impl QuantizationDescriptor {
    /// Returns the [`QuantizationSchemeTag`] for this descriptor.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{Nf4, QuantizationDescriptor, QuantizationSchemeTag};
    ///
    /// let desc = QuantizationDescriptor::Nf4(Nf4::new(0, 64, 1).unwrap());
    /// assert_eq!(desc.scheme_tag(), QuantizationSchemeTag::Nf4);
    /// ```
    pub fn scheme_tag(&self) -> QuantizationSchemeTag {
        match self {
            Self::PerTensorAffine(_) => QuantizationSchemeTag::PerTensorAffine,
            Self::PerChannelAffine(_) => QuantizationSchemeTag::PerChannelAffine,
            Self::PerBlockAffine(_) => QuantizationSchemeTag::PerBlockAffine,
            Self::Nf4(_) => QuantizationSchemeTag::Nf4,
            Self::Mxfp(_) => QuantizationSchemeTag::Mxfp,
        }
    }

    /// Returns the total encoded length of this descriptor in bytes, including
    /// the 4-byte header.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{PerTensorAffine, PerBlockAffine, ElementType, QuantizationDescriptor};
    ///
    /// let pt = QuantizationDescriptor::PerTensorAffine(
    ///     PerTensorAffine::new(1.0, 0).unwrap()
    /// );
    /// assert_eq!(pt.encoded_len(), 16);
    ///
    /// let pb = QuantizationDescriptor::PerBlockAffine(
    ///     PerBlockAffine::new_symmetric(0, 32, 1, ElementType::Float32).unwrap()
    /// );
    /// assert_eq!(pb.encoded_len(), 24);
    /// ```
    pub fn encoded_len(&self) -> usize {
        match self {
            Self::PerTensorAffine(_) => per_tensor_affine::ENCODED_LEN,
            Self::PerChannelAffine(_) => per_channel_affine::ENCODED_LEN,
            Self::PerBlockAffine(_) => per_block_affine::ENCODED_LEN,
            Self::Nf4(_) => nf4::ENCODED_LEN,
            Self::Mxfp(_) => mxfp::ENCODED_LEN,
        }
    }

    /// Encodes this descriptor into `out` without any allocation.
    ///
    /// This is the zero-alloc canonical encoding path required by Layer 5
    /// streaming writers. `out` must be at least [`Self::encoded_len()`] bytes.
    ///
    /// Returns the number of bytes written.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidQuantization`] — `out` is shorter than `encoded_len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{PerTensorAffine, QuantizationDescriptor};
    ///
    /// let desc = QuantizationDescriptor::PerTensorAffine(
    ///     PerTensorAffine::new(0.5, 0).unwrap()
    /// );
    /// let mut buf = vec![0u8; desc.encoded_len()];
    /// let written = desc.encode_into(&mut buf).unwrap();
    /// assert_eq!(written, 16);
    ///
    /// // Round-trip.
    /// let (decoded, consumed) = QuantizationDescriptor::decode(&buf).unwrap();
    /// assert_eq!(consumed, 16);
    /// assert_eq!(decoded, desc);
    /// ```
    pub fn encode_into(&self, out: &mut [u8]) -> Result<usize> {
        let len = self.encoded_len();
        if out.len() < len {
            return Err(Error::InvalidQuantization(format!(
                "output buffer too short: need {len} bytes, have {}",
                out.len()
            )));
        }

        // Build and write the 4-byte header.
        let (scheme_version, flags) = self.header_version_and_flags();
        let header = QuantizationHeader {
            scheme_tag: self.scheme_tag(),
            scheme_version,
            flags,
        };
        header.write(out);

        // Write the scheme-specific payload (bytes 4 onward).
        match self {
            Self::PerTensorAffine(inner) => inner.encode_payload(out),
            Self::PerChannelAffine(inner) => inner.encode_payload(out),
            Self::PerBlockAffine(inner) => inner.encode_payload(out),
            Self::Nf4(inner) => inner.encode_payload(out),
            Self::Mxfp(inner) => inner.encode_payload(out),
        }

        Ok(len)
    }

    /// Encodes this descriptor into a freshly allocated `Vec<u8>`.
    ///
    /// This is a convenience wrapper around [`Self::encode_into`]. Prefer
    /// [`Self::encode_into`] in hot paths to avoid allocation.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{Nf4, QuantizationDescriptor};
    ///
    /// let desc = QuantizationDescriptor::Nf4(Nf4::new(0, 64, 1).unwrap());
    /// let bytes = desc.encode_to_vec();
    /// assert_eq!(bytes.len(), 16);
    ///
    /// let (decoded, _) = QuantizationDescriptor::decode(&bytes).unwrap();
    /// assert_eq!(decoded, desc);
    /// ```
    pub fn encode_to_vec(&self) -> Vec<u8> {
        let len = self.encoded_len();
        let mut buf = vec![0u8; len];
        // SAFETY (logic): encode_into only fails if buf is shorter than encoded_len(),
        // and we allocate exactly encoded_len() bytes above, so this never errors.
        self.encode_into(&mut buf)
            .expect("encode_into into exact-sized buffer is infallible");
        buf
    }

    /// Decodes a [`QuantizationDescriptor`] from the beginning of `bytes`.
    ///
    /// Returns the decoded descriptor and the number of bytes consumed.
    /// This lets streaming readers advance their cursor without re-scanning
    /// (design decision: returning consumed count avoids a second length query).
    ///
    /// # Errors
    ///
    /// - [`Error::QuantizationDescriptorTooShort`] — not enough bytes.
    /// - [`Error::InvalidQuantizationSchemeTag`] — tag is `0x00` or `0xFF`.
    /// - [`Error::ReservedQuantizationSchemeTag`] — tag is in `0x60`–`0xEF`.
    /// - [`Error::PrivateQuantizationSchemeTag`] — tag is in `0xF0`–`0xFE`.
    /// - [`Error::UnknownQuantizationSchemeTag`] — tag is unallocated.
    /// - [`Error::UnsupportedSchemeVersion`] — version exceeds supported maximum.
    /// - [`Error::ReservedQuantizationFlagsBits`] — reserved flags bits are set.
    /// - [`Error::InvalidBlockSize`] — block_size is out of range or not a power of two.
    /// - [`Error::InvalidQuantization`] — other payload validation failure.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{PerTensorAffine, QuantizationDescriptor};
    ///
    /// let desc = QuantizationDescriptor::PerTensorAffine(
    ///     PerTensorAffine::new(1.0, 0).unwrap()
    /// );
    /// let bytes = desc.encode_to_vec();
    /// let (decoded, consumed) = QuantizationDescriptor::decode(&bytes).unwrap();
    /// assert_eq!(consumed, bytes.len());
    /// assert_eq!(decoded, desc);
    /// ```
    pub fn decode(bytes: &[u8]) -> Result<(Self, usize)> {
        // Peek at the scheme tag first so we know the supported version to pass.
        if bytes.is_empty() {
            return Err(Error::QuantizationDescriptorTooShort {
                found: 0,
                needed: QuantizationHeader::LEN,
            });
        }
        let tag_byte = bytes[0];
        let tag = QuantizationSchemeTag::from_byte(tag_byte)?;
        let supported_version = supported_version_for(tag);

        let header = QuantizationHeader::parse(bytes, supported_version)?;
        let descriptor = match tag {
            QuantizationSchemeTag::PerTensorAffine => Self::PerTensorAffine(
                per_tensor_affine::PerTensorAffine::decode_payload(header.flags, bytes)?,
            ),
            QuantizationSchemeTag::PerChannelAffine => Self::PerChannelAffine(
                per_channel_affine::PerChannelAffine::decode_payload(header.flags, bytes)?,
            ),
            QuantizationSchemeTag::PerBlockAffine => Self::PerBlockAffine(
                per_block_affine::PerBlockAffine::decode_payload(header.flags, bytes)?,
            ),
            QuantizationSchemeTag::Nf4 => Self::Nf4(nf4::Nf4::decode_payload(header.flags, bytes)?),
            QuantizationSchemeTag::Mxfp => {
                Self::Mxfp(mxfp::Mxfp::decode_payload(header.flags, bytes)?)
            }
        };

        let consumed = descriptor.encoded_len();
        Ok((descriptor, consumed))
    }

    /// Returns the set of storage [`ElementType`]s that are valid for this descriptor.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{ElementType, Mxfp, QuantizationDescriptor};
    ///
    /// let desc = QuantizationDescriptor::Mxfp(Mxfp::new(0, 32, 1).unwrap());
    /// assert!(desc.valid_storage_types().contains(&ElementType::Float8E4M3));
    /// assert!(!desc.valid_storage_types().contains(&ElementType::Float32));
    /// ```
    pub fn valid_storage_types(&self) -> &'static [ElementType] {
        match self {
            Self::PerTensorAffine(_) => PerTensorAffine::valid_storage_types(),
            Self::PerChannelAffine(_) => PerChannelAffine::valid_storage_types(),
            Self::PerBlockAffine(_) => PerBlockAffine::valid_storage_types(),
            Self::Nf4(_) => Nf4::valid_storage_types(),
            Self::Mxfp(_) => Mxfp::valid_storage_types(),
        }
    }

    /// Returns `true` if `ty` is a valid storage type for this descriptor.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{ElementType, PerTensorAffine, QuantizationDescriptor};
    ///
    /// let desc = QuantizationDescriptor::PerTensorAffine(
    ///     PerTensorAffine::new(1.0, 0).unwrap()
    /// );
    /// assert!(desc.is_valid_storage_type(ElementType::Int8));
    /// assert!(!desc.is_valid_storage_type(ElementType::Float32));
    /// ```
    pub fn is_valid_storage_type(&self, ty: ElementType) -> bool {
        self.valid_storage_types().contains(&ty)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Returns the `(scheme_version, flags)` pair to write into the header.
    fn header_version_and_flags(&self) -> (u8, u16) {
        match self {
            Self::PerTensorAffine(_) => (per_tensor_affine::SUPPORTED_VERSION, 0),
            Self::PerChannelAffine(inner) => (per_channel_affine::SUPPORTED_VERSION, inner.flags()),
            Self::PerBlockAffine(inner) => (per_block_affine::SUPPORTED_VERSION, inner.flags()),
            Self::Nf4(_) => (nf4::SUPPORTED_VERSION, 0),
            Self::Mxfp(_) => (mxfp::SUPPORTED_VERSION, 0),
        }
    }
}

// ── Module-level helpers ──────────────────────────────────────────────────────

/// Returns the highest `scheme_version` this implementation supports for `tag`.
fn supported_version_for(tag: QuantizationSchemeTag) -> u8 {
    match tag {
        QuantizationSchemeTag::PerTensorAffine => per_tensor_affine::SUPPORTED_VERSION,
        QuantizationSchemeTag::PerChannelAffine => per_channel_affine::SUPPORTED_VERSION,
        QuantizationSchemeTag::PerBlockAffine => per_block_affine::SUPPORTED_VERSION,
        QuantizationSchemeTag::Nf4 => nf4::SUPPORTED_VERSION,
        QuantizationSchemeTag::Mxfp => mxfp::SUPPORTED_VERSION,
    }
}

/// Validates that `axis` is a valid axis index for a tensor of the given `rank`.
///
/// Per the spec, `axis` MUST satisfy `axis < rank`.
///
/// # Errors
///
/// - [`Error::QuantizationAxisOutOfBounds`] — `axis >= rank`.
///
/// # Examples
///
/// ```
/// use hurray_core::validate_axis;
///
/// assert!(validate_axis(0, 3).is_ok());
/// assert!(validate_axis(2, 3).is_ok());
/// assert!(validate_axis(3, 3).is_err()); // axis == rank is out of bounds
/// assert!(validate_axis(0, 0).is_err()); // rank 0 means no valid axis
/// ```
pub fn validate_axis(axis: u32, rank: u32) -> Result<()> {
    if axis >= rank {
        return Err(Error::QuantizationAxisOutOfBounds { axis, rank });
    }
    Ok(())
}

/// Validates that all quantization-parameter buffer indices in `desc` are valid
/// within the `buffers` table and do not alias the tensor data buffer.
///
/// For each parameter buffer index referenced by `desc`:
///
/// 1. The index MUST be less than `buffers.len()`.
/// 2. The index MUST NOT equal `data_buffer_index`.
/// 3. The referenced buffer's `device_tag` MUST match that of
///    `buffers[data_buffer_index]`.
///
/// # Errors
///
/// - [`Error::QuantizationBufferIndexOutOfRange`] — a parameter buffer index is
///   out of range.
/// - [`Error::QuantizationBufferAliasesData`] — a parameter buffer index equals
///   `data_buffer_index`.
/// - [`Error::DeviceTagMismatch`] — a parameter buffer is on a different device
///   than the data buffer.
///
/// # Examples
///
/// ```
/// use hurray_core::{
///     BufferHandle, DeviceTag, Nf4, QuantizationDescriptor,
///     validate_buffer_placement, MIN_BUFFER_ALIGNMENT,
/// };
///
/// let data = BufferHandle::new(1024, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu).unwrap();
/// let scale = BufferHandle::new(64, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu).unwrap();
/// let buffers = [data, scale];
///
/// let desc = QuantizationDescriptor::Nf4(Nf4::new(0, 64, 1).unwrap());
/// // scale buffer index 1 is valid: in range, != data index 0, same device.
/// assert!(validate_buffer_placement(&desc, &buffers, 0).is_ok());
/// ```
pub fn validate_buffer_placement(
    desc: &QuantizationDescriptor,
    buffers: &[BufferHandle],
    data_buffer_index: u32,
) -> Result<()> {
    let data_device = buffers
        .get(data_buffer_index as usize)
        .ok_or(Error::QuantizationBufferIndexOutOfRange {
            index: data_buffer_index,
            buffer_count: buffers.len() as u32,
        })?
        .device_tag();

    for param_index in param_buffer_indices(desc) {
        // Check index is in range.
        if param_index as usize >= buffers.len() {
            return Err(Error::QuantizationBufferIndexOutOfRange {
                index: param_index,
                buffer_count: buffers.len() as u32,
            });
        }
        // Check it does not alias the data buffer.
        if param_index == data_buffer_index {
            return Err(Error::QuantizationBufferAliasesData { index: param_index });
        }
        // Check same device as data buffer.
        let param_device = buffers[param_index as usize].device_tag();
        if param_device != data_device {
            return Err(Error::DeviceTagMismatch {
                expected: data_device.to_byte(),
                found: param_device.to_byte(),
            });
        }
    }
    Ok(())
}

/// Returns the list of parameter buffer indices referenced by `desc`.
///
/// This is an internal helper that collects the indices in a small fixed-size
/// array to avoid allocating per call.
fn param_buffer_indices(desc: &QuantizationDescriptor) -> impl Iterator<Item = u32> {
    // Max 2 parameter buffers (scale + optional zero_point for per-channel/per-block).
    let mut indices = [u32::MAX; 2];
    let mut count = 0usize;

    match desc {
        QuantizationDescriptor::PerTensorAffine(_) => {
            // All parameters are inline — no separate buffer references.
        }
        QuantizationDescriptor::PerChannelAffine(inner) => {
            indices[count] = inner.scale_buffer_index();
            count += 1;
            if let Some(zp) = inner.zero_point_buffer_index() {
                indices[count] = zp;
                count += 1;
            }
        }
        QuantizationDescriptor::PerBlockAffine(inner) => {
            indices[count] = inner.scale_buffer_index();
            count += 1;
            if let Some(zp) = inner.zero_point_buffer_index() {
                indices[count] = zp;
                count += 1;
            }
        }
        QuantizationDescriptor::Nf4(inner) => {
            indices[count] = inner.scale_buffer_index();
            count += 1;
        }
        QuantizationDescriptor::Mxfp(inner) => {
            indices[count] = inner.scale_buffer_index();
            count += 1;
        }
    }

    indices.into_iter().take(count)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BufferHandle, DeviceTag, ElementType, Error, MIN_BUFFER_ALIGNMENT};

    // ── QuantizationSchemeTag::from_byte ─────────────────────────────────────

    #[test]
    fn scheme_tag_from_byte_tier1_variants() {
        // Spec §: tag range 0x01–0x05 are Tier 1 assigned.
        assert_eq!(
            QuantizationSchemeTag::from_byte(0x01).unwrap(),
            QuantizationSchemeTag::PerTensorAffine
        );
        assert_eq!(
            QuantizationSchemeTag::from_byte(0x02).unwrap(),
            QuantizationSchemeTag::PerChannelAffine
        );
        assert_eq!(
            QuantizationSchemeTag::from_byte(0x03).unwrap(),
            QuantizationSchemeTag::PerBlockAffine
        );
        assert_eq!(
            QuantizationSchemeTag::from_byte(0x04).unwrap(),
            QuantizationSchemeTag::Nf4
        );
        assert_eq!(
            QuantizationSchemeTag::from_byte(0x05).unwrap(),
            QuantizationSchemeTag::Mxfp
        );
    }

    #[test]
    fn scheme_tag_from_byte_permanently_invalid_sentinels() {
        // Spec §: 0x00 and 0xFF are permanently invalid.
        assert!(matches!(
            QuantizationSchemeTag::from_byte(0x00),
            Err(Error::InvalidQuantizationSchemeTag(0x00))
        ));
        assert!(matches!(
            QuantizationSchemeTag::from_byte(0xFF),
            Err(Error::InvalidQuantizationSchemeTag(0xFF))
        ));
    }

    #[test]
    fn scheme_tag_from_byte_reserved_range() {
        // Spec §: 0x60–0xEF are reserved.
        assert!(matches!(
            QuantizationSchemeTag::from_byte(0x60),
            Err(Error::ReservedQuantizationSchemeTag(0x60))
        ));
        assert!(matches!(
            QuantizationSchemeTag::from_byte(0xEF),
            Err(Error::ReservedQuantizationSchemeTag(0xEF))
        ));
    }

    #[test]
    fn scheme_tag_from_byte_private_range() {
        // Spec §: 0xF0–0xFE are private — unconstrained payload, rejected by this crate.
        assert!(matches!(
            QuantizationSchemeTag::from_byte(0xF0),
            Err(Error::PrivateQuantizationSchemeTag(0xF0))
        ));
        assert!(matches!(
            QuantizationSchemeTag::from_byte(0xFE),
            Err(Error::PrivateQuantizationSchemeTag(0xFE))
        ));
    }

    #[test]
    fn scheme_tag_from_byte_unknown_unallocated_tier1() {
        // Spec §: 0x06–0x3F are unallocated Tier 1 — must treat as unknown.
        assert!(matches!(
            QuantizationSchemeTag::from_byte(0x06),
            Err(Error::UnknownQuantizationSchemeTag(0x06))
        ));
    }

    #[test]
    fn scheme_tag_tag_returns_raw_byte() {
        assert_eq!(QuantizationSchemeTag::PerTensorAffine.tag(), 0x01);
        assert_eq!(QuantizationSchemeTag::PerChannelAffine.tag(), 0x02);
        assert_eq!(QuantizationSchemeTag::PerBlockAffine.tag(), 0x03);
        assert_eq!(QuantizationSchemeTag::Nf4.tag(), 0x04);
        assert_eq!(QuantizationSchemeTag::Mxfp.tag(), 0x05);
    }

    #[test]
    fn scheme_tag_tier_correct_per_spec_table() {
        // Tier 1 (mandatory): per-tensor, per-channel, per-block.
        assert_eq!(QuantizationSchemeTag::PerTensorAffine.tier(), 1);
        assert_eq!(QuantizationSchemeTag::PerChannelAffine.tier(), 1);
        assert_eq!(QuantizationSchemeTag::PerBlockAffine.tier(), 1);
        // Tier 2 (optional): NF4 and MXFP — despite having tags in the 0x01–0x3F
        // numeric range, the spec scheme table designates them as Tier 2.
        assert_eq!(QuantizationSchemeTag::Nf4.tier(), 2);
        assert_eq!(QuantizationSchemeTag::Mxfp.tier(), 2);
    }

    // ── QuantizationDescriptor::encoded_len ──────────────────────────────────

    #[test]
    fn encoded_len_per_tensor_affine_is_16() {
        let desc = QuantizationDescriptor::PerTensorAffine(PerTensorAffine::new(1.0, 0).unwrap());
        assert_eq!(desc.encoded_len(), 16);
    }

    #[test]
    fn encoded_len_per_channel_affine_is_20() {
        let desc = QuantizationDescriptor::PerChannelAffine(
            PerChannelAffine::new_symmetric(0, 1).unwrap(),
        );
        assert_eq!(desc.encoded_len(), 20);
    }

    #[test]
    fn encoded_len_per_block_affine_is_24() {
        let desc = QuantizationDescriptor::PerBlockAffine(
            PerBlockAffine::new_symmetric(0, 32, 1, ElementType::Float32).unwrap(),
        );
        assert_eq!(desc.encoded_len(), 24);
    }

    #[test]
    fn encoded_len_nf4_is_16() {
        let desc = QuantizationDescriptor::Nf4(Nf4::new(0, 64, 1).unwrap());
        assert_eq!(desc.encoded_len(), 16);
    }

    #[test]
    fn encoded_len_mxfp_is_16() {
        let desc = QuantizationDescriptor::Mxfp(Mxfp::new(0, 32, 1).unwrap());
        assert_eq!(desc.encoded_len(), 16);
    }

    // ── Round-trips (encode_into + decode) ───────────────────────────────────

    fn round_trip(desc: &QuantizationDescriptor) {
        let len = desc.encoded_len();
        let mut buf = vec![0u8; len];
        let written = desc.encode_into(&mut buf).unwrap();
        assert_eq!(written, len, "encode_into must return encoded_len");
        let (decoded, consumed) = QuantizationDescriptor::decode(&buf).unwrap();
        assert_eq!(
            consumed, len,
            "decode must consume exactly encoded_len bytes"
        );
        assert_eq!(&decoded, desc, "decoded descriptor must equal original");
    }

    #[test]
    fn round_trip_per_tensor_affine() {
        round_trip(&QuantizationDescriptor::PerTensorAffine(
            PerTensorAffine::new(0.5, 128).unwrap(),
        ));
    }

    #[test]
    fn round_trip_per_channel_affine_symmetric() {
        round_trip(&QuantizationDescriptor::PerChannelAffine(
            PerChannelAffine::new_symmetric(0, 1).unwrap(),
        ));
    }

    #[test]
    fn round_trip_per_channel_affine_asymmetric() {
        round_trip(&QuantizationDescriptor::PerChannelAffine(
            PerChannelAffine::new_asymmetric(2, 1, 3).unwrap(),
        ));
    }

    #[test]
    fn round_trip_per_block_affine_symmetric() {
        round_trip(&QuantizationDescriptor::PerBlockAffine(
            PerBlockAffine::new_symmetric(0, 32, 1, ElementType::Float32).unwrap(),
        ));
    }

    #[test]
    fn round_trip_per_block_affine_asymmetric() {
        round_trip(&QuantizationDescriptor::PerBlockAffine(
            PerBlockAffine::new_asymmetric(1, 64, 2, 3, ElementType::Float16).unwrap(),
        ));
    }

    #[test]
    fn round_trip_nf4() {
        round_trip(&QuantizationDescriptor::Nf4(Nf4::new(0, 64, 1).unwrap()));
    }

    #[test]
    fn round_trip_mxfp() {
        round_trip(&QuantizationDescriptor::Mxfp(Mxfp::new(0, 32, 1).unwrap()));
    }

    #[test]
    fn encode_to_vec_matches_encode_into() {
        let desc = QuantizationDescriptor::PerTensorAffine(PerTensorAffine::new(0.25, -7).unwrap());
        let vec_bytes = desc.encode_to_vec();
        let mut into_bytes = vec![0u8; desc.encoded_len()];
        desc.encode_into(&mut into_bytes).unwrap();
        assert_eq!(vec_bytes, into_bytes);
    }

    // ── validate_axis ─────────────────────────────────────────────────────────

    #[test]
    fn validate_axis_within_rank_is_ok() {
        assert!(validate_axis(0, 3).is_ok());
        assert!(validate_axis(2, 3).is_ok());
    }

    #[test]
    fn validate_axis_equal_to_rank_is_out_of_bounds() {
        assert!(matches!(
            validate_axis(3, 3),
            Err(Error::QuantizationAxisOutOfBounds { axis: 3, rank: 3 })
        ));
    }

    #[test]
    fn validate_axis_greater_than_rank_is_out_of_bounds() {
        assert!(matches!(
            validate_axis(10, 3),
            Err(Error::QuantizationAxisOutOfBounds { axis: 10, rank: 3 })
        ));
    }

    #[test]
    fn validate_axis_rank_zero_always_errs() {
        // rank == 0: no valid axis exists.
        assert!(matches!(
            validate_axis(0, 0),
            Err(Error::QuantizationAxisOutOfBounds { axis: 0, rank: 0 })
        ));
    }

    // ── validate_buffer_placement ─────────────────────────────────────────────

    fn make_cpu_buffer() -> BufferHandle {
        BufferHandle::new(1024, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu).unwrap()
    }

    #[test]
    fn validate_buffer_placement_nf4_scale_only_valid() {
        // NF4 has one scale buffer.  data=index 0, scale=index 1 — valid.
        let buffers = [make_cpu_buffer(), make_cpu_buffer()];
        let desc = QuantizationDescriptor::Nf4(Nf4::new(0, 64, 1).unwrap());
        assert!(validate_buffer_placement(&desc, &buffers, 0).is_ok());
    }

    #[test]
    fn validate_buffer_placement_mxfp_scale_only_valid() {
        let buffers = [make_cpu_buffer(), make_cpu_buffer()];
        let desc = QuantizationDescriptor::Mxfp(Mxfp::new(0, 32, 1).unwrap());
        assert!(validate_buffer_placement(&desc, &buffers, 0).is_ok());
    }

    #[test]
    fn validate_buffer_placement_per_block_asymmetric_scale_and_zp_valid() {
        // PerBlockAffine asymmetric: data=0, scale=1, zp=2 — all valid, same device.
        let buffers = [make_cpu_buffer(), make_cpu_buffer(), make_cpu_buffer()];
        let desc = QuantizationDescriptor::PerBlockAffine(
            PerBlockAffine::new_asymmetric(0, 32, 1, 2, ElementType::Float32).unwrap(),
        );
        assert!(validate_buffer_placement(&desc, &buffers, 0).is_ok());
    }

    #[test]
    fn validate_buffer_placement_aliases_data_is_err() {
        // Param buffer index == data buffer index → QuantizationBufferAliasesData.
        let buffers = [make_cpu_buffer(), make_cpu_buffer()];
        // scale_buffer_index = 0, which is the data buffer index.
        let desc = QuantizationDescriptor::Nf4(Nf4::new(0, 64, 0).unwrap());
        assert!(matches!(
            validate_buffer_placement(&desc, &buffers, 0),
            Err(Error::QuantizationBufferAliasesData { index: 0 })
        ));
    }

    #[test]
    fn validate_buffer_placement_index_out_of_range_is_err() {
        // Only 1 buffer; scale_buffer_index = 5 is out of range.
        let buffers = [make_cpu_buffer()];
        let desc = QuantizationDescriptor::Nf4(Nf4::new(0, 64, 5).unwrap());
        assert!(matches!(
            validate_buffer_placement(&desc, &buffers, 0),
            Err(Error::QuantizationBufferIndexOutOfRange { .. })
        ));
    }

    #[test]
    fn validate_buffer_placement_device_tag_mismatch_is_err() {
        // data=Cpu (index 0), scale=Cuda (index 1) → DeviceTagMismatch.
        let cpu_buf = BufferHandle::new(1024, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu).unwrap();
        let cuda_buf = BufferHandle::new(1024, MIN_BUFFER_ALIGNMENT, DeviceTag::Cuda).unwrap();
        let buffers = [cpu_buf, cuda_buf];
        let desc = QuantizationDescriptor::Nf4(Nf4::new(0, 64, 1).unwrap());
        assert!(matches!(
            validate_buffer_placement(&desc, &buffers, 0),
            Err(Error::DeviceTagMismatch { .. })
        ));
    }
}
