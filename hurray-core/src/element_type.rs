//! Element type system for the Hurray tensor format.
//!
//! Every tensor element is identified by a one-byte **type tag** stored in the
//! binary tensor descriptor. This module defines the complete set of types
//! recognised by this implementation, along with their wire tags, bit widths,
//! alignment requirements, and other properties derived from the spec.
//!
//! See `docs/spec/element-types.md` for the normative definition.

use std::fmt;

use crate::Error;

/// All element types defined by the Hurray format specification.
///
/// The discriminant of each variant is the **wire tag** — the `uint8` value
/// stored in the binary tensor descriptor. Tags match those in the
/// *Type Properties Summary* table in `docs/spec/element-types.md`.
///
/// # Examples
///
/// ```
/// use hurray_core::ElementType;
///
/// let ty = ElementType::Float32;
/// assert_eq!(ty.tag(), 0x03);
/// assert_eq!(ty.bit_width(), 32);
/// assert_eq!(ty.element_alignment(), 4);
/// assert!(ty.is_float());
/// assert!(!ty.is_integer());
/// ```
// Cannot use #[repr(u8)] because Extension(u8) carries a payload — the tag()
// method encodes the discriminant explicitly for all variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ElementType {
    // ── Tier 1 ── floating-point ─────────────────────────────────────────────
    /// IEEE 754 binary16 (half precision). Tag `0x01`.
    Float16,
    /// Brain floating point (1 sign, 8 exponent, 7 mantissa bits). Tag `0x02`.
    BFloat16,
    /// IEEE 754 binary32 (single precision). Tag `0x03`.
    Float32,
    /// IEEE 754 binary64 (double precision). Tag `0x04`.
    Float64,

    // ── Tier 1 ── integer ────────────────────────────────────────────────────
    /// 8-bit signed integer (two's complement). Tag `0x10`.
    Int8,
    /// 8-bit unsigned integer. Tag `0x11`.
    Uint8,
    /// 16-bit signed integer (little-endian). Tag `0x12`.
    Int16,
    /// 16-bit unsigned integer (little-endian). Tag `0x13`.
    Uint16,
    /// 32-bit signed integer (little-endian). Tag `0x14`.
    Int32,
    /// 32-bit unsigned integer (little-endian). Tag `0x15`.
    Uint32,
    /// 64-bit signed integer (little-endian). Tag `0x16`.
    Int64,
    /// 64-bit unsigned integer (little-endian). Tag `0x17`.
    Uint64,

    // ── Tier 1 ── boolean ────────────────────────────────────────────────────
    /// Boolean, packed 8 per byte, LSB-first. Tag `0x20`.
    Bool,

    // ── Tier 2 ── float8 variants ────────────────────────────────────────────
    /// OFP8 float8 (1 sign, 4 exponent, 3 mantissa, bias 7). Tag `0x40`.
    Float8E4M3,
    /// OFP8 float8 (1 sign, 5 exponent, 2 mantissa, bias 15). Tag `0x41`.
    Float8E5M2,
    /// OCP MX exponent-only float8 (8 exponent bits, no sign, no mantissa). Tag `0x42`.
    Float8E8M0,

    // ── Tier 2 ── sub-byte floating-point ────────────────────────────────────
    /// OCP MXFP4 (1 sign, 2 exponent, 1 mantissa, bias 1). Tag `0x43`.
    Float4E2M1,
    /// OCP MX float6 (1 sign, 2 exponent, 3 mantissa, bias 1). Tag `0x44`.
    Float6E2M3,
    /// OCP MX float6 (1 sign, 3 exponent, 2 mantissa, bias 3). Tag `0x45`.
    Float6E3M2,

    // ── Tier 2 ── extended floating-point ────────────────────────────────────
    /// IEEE 754 binary128 (quad precision). Tag `0x46`.
    Float128,
    // 0x47 is reserved — no variant.

    // ── Tier 2 ── sub-byte integer ───────────────────────────────────────────
    /// 4-bit signed integer (two's complement, LSB-first). Tag `0x48`.
    Int4,
    /// 4-bit unsigned integer (LSB-first). Tag `0x49`.
    Uint4,
    /// 2-bit signed integer (two's complement, LSB-first). Tag `0x4A`.
    Int2,
    /// 2-bit unsigned integer (LSB-first). Tag `0x4B`.
    Uint2,

    // ── Tier 2 ── complex ────────────────────────────────────────────────────
    /// Two consecutive `float32` values (real, imaginary). Tag `0x50`.
    Complex64,
    /// Two consecutive `float64` values (real, imaginary). Tag `0x51`.
    Complex128,

    // ── Private-extension range ───────────────────────────────────────────────
    /// Implementation-private extension type. The inner `u8` is the raw wire
    /// tag in the range `0xF0`–`0xFE`.
    ///
    /// The actual bit-width and packing rules are carried by the
    /// [`ExtensionTypeDescriptor`][crate::descriptor::ExtensionTypeDescriptor]
    /// section in the binary descriptor (flag `HAS_EXTENSION_TYPE`).
    ///
    /// `bit_width()` returns `0` for this variant — a sentinel meaning "consult
    /// the extension type descriptor for sizing".
    Extension(u8),
}

impl ElementType {
    /// Returns the one-byte wire tag stored in the binary tensor descriptor.
    ///
    /// For [`ElementType::Extension`], returns the inner tag byte directly
    /// (always in `0xF0`–`0xFE`).
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::ElementType;
    ///
    /// assert_eq!(ElementType::Float32.tag(), 0x03);
    /// assert_eq!(ElementType::Int4.tag(), 0x48);
    /// assert_eq!(ElementType::Complex128.tag(), 0x51);
    /// assert_eq!(ElementType::Extension(0xF1).tag(), 0xF1);
    /// ```
    #[inline]
    pub fn tag(self) -> u8 {
        match self {
            Self::Float16 => 0x01,
            Self::BFloat16 => 0x02,
            Self::Float32 => 0x03,
            Self::Float64 => 0x04,
            Self::Int8 => 0x10,
            Self::Uint8 => 0x11,
            Self::Int16 => 0x12,
            Self::Uint16 => 0x13,
            Self::Int32 => 0x14,
            Self::Uint32 => 0x15,
            Self::Int64 => 0x16,
            Self::Uint64 => 0x17,
            Self::Bool => 0x20,
            Self::Float8E4M3 => 0x40,
            Self::Float8E5M2 => 0x41,
            Self::Float8E8M0 => 0x42,
            Self::Float4E2M1 => 0x43,
            Self::Float6E2M3 => 0x44,
            Self::Float6E3M2 => 0x45,
            Self::Float128 => 0x46,
            Self::Int4 => 0x48,
            Self::Uint4 => 0x49,
            Self::Int2 => 0x4A,
            Self::Uint2 => 0x4B,
            Self::Complex64 => 0x50,
            Self::Complex128 => 0x51,
            // Extension: inner byte IS the wire tag (0xF0–0xFE).
            Self::Extension(t) => t,
        }
    }

    /// Parses an [`ElementType`] from its one-byte wire tag.
    ///
    /// Tags in the private-extension range `0xF0`–`0xFE` are accepted and
    /// returned as [`ElementType::Extension`]. The actual numeric semantics
    /// are carried by the `ExtensionTypeDescriptor` section in the descriptor.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidTypeTag`] — tag is `0x00` or `0xFF` (permanently
    ///   reserved sentinels).
    /// - [`Error::ReservedTypeTag`] — tag is `0x47` or in `0x80`–`0xEF`
    ///   (reserved for future specification versions).
    /// - [`Error::UnknownTypeTag`] — tag is any other unrecognised value not
    ///   in the above categories.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{ElementType, Error};
    ///
    /// assert_eq!(ElementType::from_tag(0x03).unwrap(), ElementType::Float32);
    /// assert_eq!(ElementType::from_tag(0xF1).unwrap(), ElementType::Extension(0xF1));
    /// assert!(matches!(ElementType::from_tag(0x00), Err(Error::InvalidTypeTag(0x00))));
    /// assert!(matches!(ElementType::from_tag(0xFF), Err(Error::InvalidTypeTag(0xFF))));
    /// assert!(matches!(ElementType::from_tag(0x47), Err(Error::ReservedTypeTag(0x47))));
    /// assert!(matches!(ElementType::from_tag(0x80), Err(Error::ReservedTypeTag(0x80))));
    /// ```
    pub fn from_tag(tag: u8) -> crate::Result<Self> {
        match tag {
            // Permanently invalid sentinels.
            0x00 | 0xFF => Err(Error::InvalidTypeTag(tag)),

            // Tier 1 — floating-point.
            0x01 => Ok(Self::Float16),
            0x02 => Ok(Self::BFloat16),
            0x03 => Ok(Self::Float32),
            0x04 => Ok(Self::Float64),

            // Tier 1 — integer.
            0x10 => Ok(Self::Int8),
            0x11 => Ok(Self::Uint8),
            0x12 => Ok(Self::Int16),
            0x13 => Ok(Self::Uint16),
            0x14 => Ok(Self::Int32),
            0x15 => Ok(Self::Uint32),
            0x16 => Ok(Self::Int64),
            0x17 => Ok(Self::Uint64),

            // Tier 1 — boolean.
            0x20 => Ok(Self::Bool),

            // Tier 2 — float8.
            0x40 => Ok(Self::Float8E4M3),
            0x41 => Ok(Self::Float8E5M2),
            0x42 => Ok(Self::Float8E8M0),

            // Tier 2 — sub-byte float.
            0x43 => Ok(Self::Float4E2M1),
            0x44 => Ok(Self::Float6E2M3),
            0x45 => Ok(Self::Float6E3M2),

            // Tier 2 — extended float.
            0x46 => Ok(Self::Float128),

            // 0x47 — reserved for future assignment.
            0x47 => Err(Error::ReservedTypeTag(tag)),

            // Tier 2 — sub-byte integer.
            0x48 => Ok(Self::Int4),
            0x49 => Ok(Self::Uint4),
            0x4A => Ok(Self::Int2),
            0x4B => Ok(Self::Uint2),

            // Tier 2 — complex.
            0x50 => Ok(Self::Complex64),
            0x51 => Ok(Self::Complex128),

            // 0x80–0xEF — reserved for future spec versions.
            0x80..=0xEF => Err(Error::ReservedTypeTag(tag)),

            // 0xF0–0xFE — private-extension range. Accepted; semantics are
            // carried by the ExtensionTypeDescriptor section in the descriptor.
            0xF0..=0xFE => Ok(Self::Extension(tag)),

            // Everything else unrecognised.
            _ => Err(Error::UnknownTypeTag(tag)),
        }
    }

    /// Returns the **tier** of this type: `1` for core types, `2` for extended
    /// types.
    ///
    /// All Tier 1 types have a wire tag in `0x01`–`0x3F`. All Tier 2 types
    /// have a wire tag in `0x40`–`0x7F`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::ElementType;
    ///
    /// assert_eq!(ElementType::Float32.tier(), 1);
    /// assert_eq!(ElementType::Bool.tier(), 1);
    /// assert_eq!(ElementType::Float8E4M3.tier(), 2);
    /// assert_eq!(ElementType::Int4.tier(), 2);
    /// ```
    #[inline]
    pub fn tier(self) -> u8 {
        if self.tag() < 0x40 {
            1
        } else {
            2
        }
    }

    /// Returns the number of **bits** each element occupies in the data buffer.
    ///
    /// For sub-byte types this is less than 8; see `docs/spec/element-types.md`
    /// for the packing rules that govern how these bits are laid out within bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::ElementType;
    ///
    /// assert_eq!(ElementType::Bool.bit_width(), 1);
    /// assert_eq!(ElementType::Int4.bit_width(), 4);
    /// assert_eq!(ElementType::Float32.bit_width(), 32);
    /// assert_eq!(ElementType::Complex128.bit_width(), 128);
    /// ```
    #[inline]
    pub fn bit_width(self) -> u32 {
        match self {
            Self::Bool => 1,

            Self::Int2 | Self::Uint2 => 2,

            Self::Int4 | Self::Uint4 | Self::Float4E2M1 => 4,

            Self::Float6E2M3 | Self::Float6E3M2 => 6,

            Self::Int8 | Self::Uint8 | Self::Float8E4M3 | Self::Float8E5M2 | Self::Float8E8M0 => 8,

            Self::Float16 | Self::BFloat16 | Self::Int16 | Self::Uint16 => 16,

            Self::Float32 | Self::Int32 | Self::Uint32 => 32,

            Self::Float64 | Self::Int64 | Self::Uint64 | Self::Complex64 => 64,

            Self::Float128 | Self::Complex128 => 128,

            // Extension types declare their bit_width via ExtensionTypeDescriptor;
            // 0 is a sentinel meaning "consult the extension type descriptor".
            Self::Extension(_) => 0,
        }
    }

    /// Returns `true` if this type occupies fewer than 8 bits per element.
    ///
    /// Sub-byte types require special buffer-size arithmetic and have
    /// packing rules defined in `docs/spec/element-types.md`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::ElementType;
    ///
    /// assert!(ElementType::Bool.is_sub_byte());
    /// assert!(ElementType::Int4.is_sub_byte());
    /// assert!(ElementType::Float6E2M3.is_sub_byte());
    /// assert!(!ElementType::Float32.is_sub_byte());
    /// assert!(!ElementType::Float8E4M3.is_sub_byte());
    /// ```
    #[inline]
    pub fn is_sub_byte(self) -> bool {
        // Extension types have unknown width (bit_width() == 0); treat as
        // non-sub-byte here — callers must use ExtensionTypeDescriptor for sizing.
        if matches!(self, Self::Extension(_)) {
            return false;
        }
        self.bit_width() < 8
    }

    /// Returns `true` if this type is a floating-point or complex type.
    ///
    /// Returns `false` for all integer types and [`ElementType::Bool`].
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::ElementType;
    ///
    /// assert!(ElementType::Float32.is_float());
    /// assert!(ElementType::BFloat16.is_float());
    /// assert!(ElementType::Complex64.is_float());
    /// assert!(!ElementType::Int32.is_float());
    /// assert!(!ElementType::Bool.is_float());
    /// ```
    #[inline]
    pub fn is_float(self) -> bool {
        matches!(
            self,
            Self::Float16
                | Self::BFloat16
                | Self::Float32
                | Self::Float64
                | Self::Float8E4M3
                | Self::Float8E5M2
                | Self::Float8E8M0
                | Self::Float4E2M1
                | Self::Float6E2M3
                | Self::Float6E3M2
                | Self::Float128
                | Self::Complex64
                | Self::Complex128
        )
    }

    /// Returns `true` if this type is a (signed or unsigned) integer type.
    ///
    /// Returns `false` for all floating-point, complex, and boolean types.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::ElementType;
    ///
    /// assert!(ElementType::Int32.is_integer());
    /// assert!(ElementType::Uint4.is_integer());
    /// assert!(!ElementType::Float32.is_integer());
    /// assert!(!ElementType::Bool.is_integer());
    /// assert!(!ElementType::Complex64.is_integer());
    /// ```
    #[inline]
    pub fn is_integer(self) -> bool {
        matches!(
            self,
            Self::Int8
                | Self::Uint8
                | Self::Int16
                | Self::Uint16
                | Self::Int32
                | Self::Uint32
                | Self::Int64
                | Self::Uint64
                | Self::Int4
                | Self::Uint4
                | Self::Int2
                | Self::Uint2
        )
    }

    /// Returns `true` if this type is signed.
    ///
    /// The following types are considered **unsigned** and return `false`:
    /// [`ElementType::Uint8`], [`ElementType::Uint16`], [`ElementType::Uint32`],
    /// [`ElementType::Uint64`], [`ElementType::Uint4`], [`ElementType::Uint2`],
    /// [`ElementType::Bool`], and [`ElementType::Float8E8M0`] (which has no sign
    /// bit per the OCP MX specification).
    ///
    /// All other floating-point, complex, and signed integer types return `true`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::ElementType;
    ///
    /// assert!(ElementType::Float32.is_signed());
    /// assert!(ElementType::Int8.is_signed());
    /// assert!(ElementType::Complex64.is_signed());
    /// assert!(!ElementType::Uint32.is_signed());
    /// assert!(!ElementType::Bool.is_signed());
    /// assert!(!ElementType::Float8E8M0.is_signed());
    /// ```
    #[inline]
    pub fn is_signed(self) -> bool {
        // Extension types carry their sign semantics in ExtensionTypeDescriptor;
        // return false here to avoid false positives on unknown types.
        if matches!(self, Self::Extension(_)) {
            return false;
        }
        !matches!(
            self,
            Self::Uint8
                | Self::Uint16
                | Self::Uint32
                | Self::Uint64
                | Self::Uint4
                | Self::Uint2
                | Self::Bool
                | Self::Float8E8M0
        )
    }

    /// Returns the **element-level alignment** in bytes as specified in the
    /// *Type Properties Summary* table in `docs/spec/element-types.md`.
    ///
    /// This is the minimum alignment requirement of an individual element
    /// within a contiguous buffer. The buffer itself has a stricter alignment
    /// requirement (see `docs/spec/buffer-protocol.md`).
    ///
    /// For sub-byte types the alignment is `1` (byte granularity): packed
    /// data MUST start at a byte boundary within the buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::ElementType;
    ///
    /// assert_eq!(ElementType::Int4.element_alignment(), 1);
    /// assert_eq!(ElementType::Float16.element_alignment(), 2);
    /// assert_eq!(ElementType::Float32.element_alignment(), 4);
    /// assert_eq!(ElementType::Float64.element_alignment(), 8);
    /// assert_eq!(ElementType::Float128.element_alignment(), 16);
    /// // Complex types align to the constituent float, not the whole element.
    /// assert_eq!(ElementType::Complex64.element_alignment(), 4);
    /// assert_eq!(ElementType::Complex128.element_alignment(), 8);
    /// ```
    #[inline]
    pub fn element_alignment(self) -> usize {
        match self {
            // Sub-byte types and byte-wide types: 1-byte alignment.
            Self::Bool
            | Self::Int2
            | Self::Uint2
            | Self::Int4
            | Self::Uint4
            | Self::Float4E2M1
            | Self::Float6E2M3
            | Self::Float6E3M2
            | Self::Int8
            | Self::Uint8
            | Self::Float8E4M3
            | Self::Float8E5M2
            | Self::Float8E8M0 => 1,

            // 16-bit types.
            Self::Float16 | Self::BFloat16 | Self::Int16 | Self::Uint16 => 2,

            // 32-bit types + Complex64 (aligns to constituent float32).
            Self::Float32 | Self::Int32 | Self::Uint32 | Self::Complex64 => 4,

            // 64-bit types + Complex128 (aligns to constituent float64).
            Self::Float64 | Self::Int64 | Self::Uint64 | Self::Complex128 => 8,

            // 128-bit type.
            Self::Float128 => 16,

            // Extension types: alignment is unknown without ExtensionTypeDescriptor;
            // return 1 (byte granularity) as a safe, conservative default.
            Self::Extension(_) => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── from_tag round-trip ──────────────────────────────────────────────────

    /// Every defined variant must round-trip through its wire tag.
    #[test]
    fn from_tag_round_trip_all_variants() {
        let all = [
            ElementType::Float16,
            ElementType::BFloat16,
            ElementType::Float32,
            ElementType::Float64,
            ElementType::Int8,
            ElementType::Uint8,
            ElementType::Int16,
            ElementType::Uint16,
            ElementType::Int32,
            ElementType::Uint32,
            ElementType::Int64,
            ElementType::Uint64,
            ElementType::Bool,
            ElementType::Float8E4M3,
            ElementType::Float8E5M2,
            ElementType::Float8E8M0,
            ElementType::Float4E2M1,
            ElementType::Float6E2M3,
            ElementType::Float6E3M2,
            ElementType::Float128,
            ElementType::Int4,
            ElementType::Uint4,
            ElementType::Int2,
            ElementType::Uint2,
            ElementType::Complex64,
            ElementType::Complex128,
        ];
        for ty in all {
            let tag = ty.tag();
            let parsed = ElementType::from_tag(tag)
                .unwrap_or_else(|_| panic!("from_tag(0x{tag:02X}) failed for {ty:?}"));
            assert_eq!(
                parsed, ty,
                "round-trip mismatch: tag 0x{tag:02X} decoded as {parsed:?}, expected {ty:?}"
            );
        }
    }

    // ── from_tag error cases ─────────────────────────────────────────────────

    /// Spec § element-types: tags 0x00 and 0xFF are permanently reserved
    /// sentinels and MUST yield `InvalidTypeTag`.
    #[test]
    fn from_tag_invalid_sentinel_0x00() {
        assert!(matches!(
            ElementType::from_tag(0x00),
            Err(Error::InvalidTypeTag(0x00))
        ));
    }

    #[test]
    fn from_tag_invalid_sentinel_0xff() {
        assert!(matches!(
            ElementType::from_tag(0xFF),
            Err(Error::InvalidTypeTag(0xFF))
        ));
    }

    /// Tag 0x47 is reserved (gap between Float128 and Int4 in Tier 2).
    #[test]
    fn from_tag_reserved_0x47() {
        assert!(matches!(
            ElementType::from_tag(0x47),
            Err(Error::ReservedTypeTag(0x47))
        ));
    }

    /// Tags 0x80–0xEF are reserved for future specification versions.
    #[test]
    fn from_tag_reserved_range_0x80_to_0xef() {
        for tag in 0x80u8..=0xEF {
            assert!(
                matches!(
                    ElementType::from_tag(tag),
                    Err(Error::ReservedTypeTag(t)) if t == tag
                ),
                "expected ReservedTypeTag for 0x{tag:02X}"
            );
        }
    }

    /// Tags 0xF0–0xFE are the private-extension range and must yield
    /// `Extension(tag)` — they are now valid and round-trip through the variant.
    #[test]
    fn from_tag_extension_range_0xf0_to_0xfe() {
        for tag in 0xF0u8..=0xFE {
            let result = ElementType::from_tag(tag)
                .unwrap_or_else(|e| panic!("from_tag(0x{tag:02X}) should succeed, got {e:?}"));
            assert_eq!(
                result,
                ElementType::Extension(tag),
                "expected Extension(0x{tag:02X})"
            );
            assert_eq!(result.tag(), tag, "tag() round-trip failed for 0x{tag:02X}");
        }
    }

    /// Tags in the gaps between known Tier 1 and Tier 2 assignments that are
    /// not explicitly called out as reserved still fall through to
    /// `UnknownTypeTag` (the wildcard arm of from_tag).
    #[test]
    fn from_tag_unknown_gaps_within_assigned_ranges() {
        // 0x05–0x0F: gap between Float64 and Int8
        for tag in 0x05u8..=0x0F {
            assert!(
                matches!(ElementType::from_tag(tag), Err(Error::UnknownTypeTag(t)) if t == tag),
                "expected UnknownTypeTag for 0x{tag:02X}"
            );
        }
        // 0x18–0x1F: gap after Uint64
        for tag in 0x18u8..=0x1F {
            assert!(
                matches!(ElementType::from_tag(tag), Err(Error::UnknownTypeTag(t)) if t == tag),
                "expected UnknownTypeTag for 0x{tag:02X}"
            );
        }
        // 0x21–0x3F: gap after Bool
        for tag in 0x21u8..=0x3F {
            assert!(
                matches!(ElementType::from_tag(tag), Err(Error::UnknownTypeTag(t)) if t == tag),
                "expected UnknownTypeTag for 0x{tag:02X}"
            );
        }
        // 0x52–0x7F: gap after Complex128
        for tag in 0x52u8..=0x7F {
            assert!(
                matches!(ElementType::from_tag(tag), Err(Error::UnknownTypeTag(t)) if t == tag),
                "expected UnknownTypeTag for 0x{tag:02X}"
            );
        }
    }

    // ── tier ─────────────────────────────────────────────────────────────────

    /// All Tier 1 types have a tag < 0x40 and must return tier 1.
    #[test]
    fn tier_all_tier1_types_return_1() {
        let tier1 = [
            ElementType::Float16,
            ElementType::BFloat16,
            ElementType::Float32,
            ElementType::Float64,
            ElementType::Int8,
            ElementType::Uint8,
            ElementType::Int16,
            ElementType::Uint16,
            ElementType::Int32,
            ElementType::Uint32,
            ElementType::Int64,
            ElementType::Uint64,
            ElementType::Bool,
        ];
        for ty in tier1 {
            assert_eq!(ty.tier(), 1, "{ty:?} should be tier 1");
        }
    }

    /// All Tier 2 types have a tag >= 0x40 and must return tier 2.
    #[test]
    fn tier_all_tier2_types_return_2() {
        let tier2 = [
            ElementType::Float8E4M3,
            ElementType::Float8E5M2,
            ElementType::Float8E8M0,
            ElementType::Float4E2M1,
            ElementType::Float6E2M3,
            ElementType::Float6E3M2,
            ElementType::Float128,
            ElementType::Int4,
            ElementType::Uint4,
            ElementType::Int2,
            ElementType::Uint2,
            ElementType::Complex64,
            ElementType::Complex128,
        ];
        for ty in tier2 {
            assert_eq!(ty.tier(), 2, "{ty:?} should be tier 2");
        }
    }

    // ── bit_width ────────────────────────────────────────────────────────────

    #[test]
    fn bit_width_1_bit() {
        assert_eq!(ElementType::Bool.bit_width(), 1);
    }

    #[test]
    fn bit_width_2_bit() {
        assert_eq!(ElementType::Int2.bit_width(), 2);
        assert_eq!(ElementType::Uint2.bit_width(), 2);
    }

    #[test]
    fn bit_width_4_bit() {
        assert_eq!(ElementType::Int4.bit_width(), 4);
        assert_eq!(ElementType::Uint4.bit_width(), 4);
        assert_eq!(ElementType::Float4E2M1.bit_width(), 4);
    }

    #[test]
    fn bit_width_6_bit() {
        assert_eq!(ElementType::Float6E2M3.bit_width(), 6);
        assert_eq!(ElementType::Float6E3M2.bit_width(), 6);
    }

    #[test]
    fn bit_width_8_bit() {
        assert_eq!(ElementType::Int8.bit_width(), 8);
        assert_eq!(ElementType::Uint8.bit_width(), 8);
        assert_eq!(ElementType::Float8E4M3.bit_width(), 8);
        assert_eq!(ElementType::Float8E5M2.bit_width(), 8);
        assert_eq!(ElementType::Float8E8M0.bit_width(), 8);
    }

    #[test]
    fn bit_width_16_bit() {
        assert_eq!(ElementType::Float16.bit_width(), 16);
        assert_eq!(ElementType::BFloat16.bit_width(), 16);
        assert_eq!(ElementType::Int16.bit_width(), 16);
        assert_eq!(ElementType::Uint16.bit_width(), 16);
    }

    #[test]
    fn bit_width_32_bit() {
        assert_eq!(ElementType::Float32.bit_width(), 32);
        assert_eq!(ElementType::Int32.bit_width(), 32);
        assert_eq!(ElementType::Uint32.bit_width(), 32);
    }

    #[test]
    fn bit_width_64_bit() {
        assert_eq!(ElementType::Float64.bit_width(), 64);
        assert_eq!(ElementType::Int64.bit_width(), 64);
        assert_eq!(ElementType::Uint64.bit_width(), 64);
        assert_eq!(ElementType::Complex64.bit_width(), 64);
    }

    #[test]
    fn bit_width_128_bit() {
        assert_eq!(ElementType::Float128.bit_width(), 128);
        assert_eq!(ElementType::Complex128.bit_width(), 128);
    }

    // ── is_sub_byte ──────────────────────────────────────────────────────────

    /// All sub-byte types must return true.
    #[test]
    fn is_sub_byte_true_for_sub_byte_types() {
        let sub_byte = [
            ElementType::Bool,
            ElementType::Int2,
            ElementType::Uint2,
            ElementType::Int4,
            ElementType::Uint4,
            ElementType::Float4E2M1,
            ElementType::Float6E2M3,
            ElementType::Float6E3M2,
        ];
        for ty in sub_byte {
            assert!(ty.is_sub_byte(), "{ty:?} should be sub-byte");
        }
    }

    /// Byte-aligned and wider types must return false.
    #[test]
    fn is_sub_byte_false_for_byte_and_wider_types() {
        let not_sub_byte = [
            ElementType::Int8,
            ElementType::Uint8,
            ElementType::Float8E4M3,
            ElementType::Float8E5M2,
            ElementType::Float8E8M0,
            ElementType::Float16,
            ElementType::BFloat16,
            ElementType::Int16,
            ElementType::Uint16,
            ElementType::Float32,
            ElementType::Int32,
            ElementType::Uint32,
            ElementType::Float64,
            ElementType::Int64,
            ElementType::Uint64,
            ElementType::Float128,
            ElementType::Complex64,
            ElementType::Complex128,
        ];
        for ty in not_sub_byte {
            assert!(!ty.is_sub_byte(), "{ty:?} should NOT be sub-byte");
        }
    }

    // ── is_float ─────────────────────────────────────────────────────────────

    /// All float variants (including complex) must return true.
    #[test]
    fn is_float_true_for_all_float_variants() {
        let floats = [
            ElementType::Float16,
            ElementType::BFloat16,
            ElementType::Float32,
            ElementType::Float64,
            ElementType::Float8E4M3,
            ElementType::Float8E5M2,
            ElementType::Float8E8M0,
            ElementType::Float4E2M1,
            ElementType::Float6E2M3,
            ElementType::Float6E3M2,
            ElementType::Float128,
            ElementType::Complex64,
            ElementType::Complex128,
        ];
        for ty in floats {
            assert!(ty.is_float(), "{ty:?} should be float");
        }
    }

    /// Integer types and Bool must not be classified as float.
    #[test]
    fn is_float_false_for_integers_and_bool() {
        let non_floats = [
            ElementType::Int8,
            ElementType::Uint8,
            ElementType::Int16,
            ElementType::Uint16,
            ElementType::Int32,
            ElementType::Uint32,
            ElementType::Int64,
            ElementType::Uint64,
            ElementType::Int4,
            ElementType::Uint4,
            ElementType::Int2,
            ElementType::Uint2,
            ElementType::Bool,
        ];
        for ty in non_floats {
            assert!(!ty.is_float(), "{ty:?} should NOT be float");
        }
    }

    // ── is_integer ───────────────────────────────────────────────────────────

    /// All integer variants (signed and unsigned, all widths) must return true.
    #[test]
    fn is_integer_true_for_all_integer_variants() {
        let integers = [
            ElementType::Int8,
            ElementType::Uint8,
            ElementType::Int16,
            ElementType::Uint16,
            ElementType::Int32,
            ElementType::Uint32,
            ElementType::Int64,
            ElementType::Uint64,
            ElementType::Int4,
            ElementType::Uint4,
            ElementType::Int2,
            ElementType::Uint2,
        ];
        for ty in integers {
            assert!(ty.is_integer(), "{ty:?} should be integer");
        }
    }

    /// Float, complex, and Bool types must not be classified as integer.
    #[test]
    fn is_integer_false_for_floats_complex_and_bool() {
        let non_integers = [
            ElementType::Float16,
            ElementType::BFloat16,
            ElementType::Float32,
            ElementType::Float64,
            ElementType::Float8E4M3,
            ElementType::Float8E5M2,
            ElementType::Float8E8M0,
            ElementType::Float4E2M1,
            ElementType::Float6E2M3,
            ElementType::Float6E3M2,
            ElementType::Float128,
            ElementType::Complex64,
            ElementType::Complex128,
            ElementType::Bool,
        ];
        for ty in non_integers {
            assert!(!ty.is_integer(), "{ty:?} should NOT be integer");
        }
    }

    // ── is_signed ────────────────────────────────────────────────────────────

    /// Unsigned integer types, Bool, and Float8E8M0 (no sign bit per OCP MX
    /// spec) must return false.
    #[test]
    fn is_signed_false_for_unsigned_types() {
        let unsigned = [
            ElementType::Uint8,
            ElementType::Uint16,
            ElementType::Uint32,
            ElementType::Uint64,
            ElementType::Uint4,
            ElementType::Uint2,
            ElementType::Bool,
            ElementType::Float8E8M0,
        ];
        for ty in unsigned {
            assert!(
                !ty.is_signed(),
                "{ty:?} should be unsigned (is_signed == false)"
            );
        }
    }

    /// All other types — signed integers, floats (with sign bit), complex —
    /// must return true.
    #[test]
    fn is_signed_true_for_signed_types() {
        let signed = [
            ElementType::Float16,
            ElementType::BFloat16,
            ElementType::Float32,
            ElementType::Float64,
            ElementType::Float8E4M3,
            ElementType::Float8E5M2,
            ElementType::Float4E2M1,
            ElementType::Float6E2M3,
            ElementType::Float6E3M2,
            ElementType::Float128,
            ElementType::Int8,
            ElementType::Int16,
            ElementType::Int32,
            ElementType::Int64,
            ElementType::Int4,
            ElementType::Int2,
            ElementType::Complex64,
            ElementType::Complex128,
        ];
        for ty in signed {
            assert!(ty.is_signed(), "{ty:?} should be signed");
        }
    }

    // ── element_alignment ────────────────────────────────────────────────────

    /// Sub-byte and byte-wide types: alignment 1.
    #[test]
    fn element_alignment_1_for_byte_and_sub_byte() {
        let align1 = [
            ElementType::Bool,
            ElementType::Int2,
            ElementType::Uint2,
            ElementType::Int4,
            ElementType::Uint4,
            ElementType::Float4E2M1,
            ElementType::Float6E2M3,
            ElementType::Float6E3M2,
            ElementType::Int8,
            ElementType::Uint8,
            ElementType::Float8E4M3,
            ElementType::Float8E5M2,
            ElementType::Float8E8M0,
        ];
        for ty in align1 {
            assert_eq!(ty.element_alignment(), 1, "{ty:?} should have alignment 1");
        }
    }

    /// 16-bit types: alignment 2.
    #[test]
    fn element_alignment_2_for_16_bit_types() {
        let align2 = [
            ElementType::Float16,
            ElementType::BFloat16,
            ElementType::Int16,
            ElementType::Uint16,
        ];
        for ty in align2 {
            assert_eq!(ty.element_alignment(), 2, "{ty:?} should have alignment 2");
        }
    }

    /// 32-bit types and Complex64 (aligns to constituent float32): alignment 4.
    #[test]
    fn element_alignment_4_for_32_bit_and_complex64() {
        let align4 = [
            ElementType::Float32,
            ElementType::Int32,
            ElementType::Uint32,
            ElementType::Complex64,
        ];
        for ty in align4 {
            assert_eq!(ty.element_alignment(), 4, "{ty:?} should have alignment 4");
        }
    }

    /// 64-bit types and Complex128 (aligns to constituent float64): alignment 8.
    #[test]
    fn element_alignment_8_for_64_bit_and_complex128() {
        let align8 = [
            ElementType::Float64,
            ElementType::Int64,
            ElementType::Uint64,
            ElementType::Complex128,
        ];
        for ty in align8 {
            assert_eq!(ty.element_alignment(), 8, "{ty:?} should have alignment 8");
        }
    }

    /// Float128: alignment 16.
    #[test]
    fn element_alignment_16_for_float128() {
        assert_eq!(ElementType::Float128.element_alignment(), 16);
    }

    // ── Display ──────────────────────────────────────────────────────────────

    /// Canonical names must match the spec's Type Properties Summary table.
    #[test]
    fn display_canonical_names() {
        let cases: &[(ElementType, &str)] = &[
            (ElementType::Float16, "float16"),
            (ElementType::BFloat16, "bfloat16"),
            (ElementType::Float32, "float32"),
            (ElementType::Float64, "float64"),
            (ElementType::Int8, "int8"),
            (ElementType::Uint8, "uint8"),
            (ElementType::Int16, "int16"),
            (ElementType::Uint16, "uint16"),
            (ElementType::Int32, "int32"),
            (ElementType::Uint32, "uint32"),
            (ElementType::Int64, "int64"),
            (ElementType::Uint64, "uint64"),
            (ElementType::Bool, "bool"),
            (ElementType::Float8E4M3, "float8_e4m3"),
            (ElementType::Float8E5M2, "float8_e5m2"),
            (ElementType::Float8E8M0, "float8_e8m0"),
            (ElementType::Float4E2M1, "float4_e2m1"),
            (ElementType::Float6E2M3, "float6_e2m3"),
            (ElementType::Float6E3M2, "float6_e3m2"),
            (ElementType::Float128, "float128"),
            (ElementType::Int4, "int4"),
            (ElementType::Uint4, "uint4"),
            (ElementType::Int2, "int2"),
            (ElementType::Uint2, "uint2"),
            (ElementType::Complex64, "complex64"),
            (ElementType::Complex128, "complex128"),
        ];
        for &(ty, expected) in cases {
            assert_eq!(
                ty.to_string(),
                expected,
                "{ty:?} display name mismatch: got '{}', expected '{expected}'",
                ty.to_string()
            );
        }
    }

    /// Display output must be all-lowercase (no uppercase letters).
    #[test]
    fn display_is_lowercase() {
        let all = [
            ElementType::Float16,
            ElementType::BFloat16,
            ElementType::Float32,
            ElementType::Float64,
            ElementType::Int8,
            ElementType::Uint8,
            ElementType::Int16,
            ElementType::Uint16,
            ElementType::Int32,
            ElementType::Uint32,
            ElementType::Int64,
            ElementType::Uint64,
            ElementType::Bool,
            ElementType::Float8E4M3,
            ElementType::Float8E5M2,
            ElementType::Float8E8M0,
            ElementType::Float4E2M1,
            ElementType::Float6E2M3,
            ElementType::Float6E3M2,
            ElementType::Float128,
            ElementType::Int4,
            ElementType::Uint4,
            ElementType::Int2,
            ElementType::Uint2,
            ElementType::Complex64,
            ElementType::Complex128,
        ];
        for ty in all {
            let s = ty.to_string();
            assert_eq!(s, s.to_lowercase(), "{ty:?} display has uppercase: '{s}'");
        }
    }
}

impl fmt::Display for ElementType {
    /// Formats the type as its canonical lowercase spec name.
    ///
    /// The returned string matches the type name column in `element-types.md`
    /// exactly: lowercase, underscore-separated.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::ElementType;
    ///
    /// assert_eq!(ElementType::Float32.to_string(), "float32");
    /// assert_eq!(ElementType::BFloat16.to_string(), "bfloat16");
    /// assert_eq!(ElementType::Float8E4M3.to_string(), "float8_e4m3");
    /// assert_eq!(ElementType::Float6E2M3.to_string(), "float6_e2m3");
    /// assert_eq!(ElementType::Int4.to_string(), "int4");
    /// assert_eq!(ElementType::Bool.to_string(), "bool");
    /// assert_eq!(ElementType::Complex64.to_string(), "complex64");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Float16 => f.write_str("float16"),
            Self::BFloat16 => f.write_str("bfloat16"),
            Self::Float32 => f.write_str("float32"),
            Self::Float64 => f.write_str("float64"),
            Self::Int8 => f.write_str("int8"),
            Self::Uint8 => f.write_str("uint8"),
            Self::Int16 => f.write_str("int16"),
            Self::Uint16 => f.write_str("uint16"),
            Self::Int32 => f.write_str("int32"),
            Self::Uint32 => f.write_str("uint32"),
            Self::Int64 => f.write_str("int64"),
            Self::Uint64 => f.write_str("uint64"),
            Self::Bool => f.write_str("bool"),
            Self::Float8E4M3 => f.write_str("float8_e4m3"),
            Self::Float8E5M2 => f.write_str("float8_e5m2"),
            Self::Float8E8M0 => f.write_str("float8_e8m0"),
            Self::Float4E2M1 => f.write_str("float4_e2m1"),
            Self::Float6E2M3 => f.write_str("float6_e2m3"),
            Self::Float6E3M2 => f.write_str("float6_e3m2"),
            Self::Float128 => f.write_str("float128"),
            Self::Int4 => f.write_str("int4"),
            Self::Uint4 => f.write_str("uint4"),
            Self::Int2 => f.write_str("int2"),
            Self::Uint2 => f.write_str("uint2"),
            Self::Complex64 => f.write_str("complex64"),
            Self::Complex128 => f.write_str("complex128"),
            Self::Extension(t) => write!(f, "extension(0x{t:02x})"),
        }
    }
}
