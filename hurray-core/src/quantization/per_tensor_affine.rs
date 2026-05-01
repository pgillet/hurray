//! Per-tensor affine quantization descriptor (scheme tag `0x01`).
//!
//! A single `scale` and `zero_point` pair applies uniformly to every element of
//! the tensor. Parameters are stored inline in the quantization descriptor; no
//! additional buffer table entries are required.
//!
//! See `docs/spec/quantization/per-tensor-affine.md` for the normative definition.

use crate::{ElementType, Error, Result};

// ── Wire layout constants ─────────────────────────────────────────────────────

/// Scheme tag byte for per-tensor affine quantization.
#[allow(dead_code)]
pub(crate) const SCHEME_TAG: u8 = 0x01;

/// Total descriptor length in bytes (header 4 + scale 4 + zero_point 4 + reserved 4).
pub(crate) const ENCODED_LEN: usize = 16;

/// Version this implementation supports.
pub(crate) const SUPPORTED_VERSION: u8 = 0x01;

/// No flags are defined for this scheme; all 16 bits must be zero.
const RESERVED_FLAGS_MASK: u16 = 0xFFFF;

/// Sentinel offset of the `scale` field within the descriptor.
const OFFSET_SCALE: usize = 4;
/// Sentinel offset of the `zero_point` field within the descriptor.
const OFFSET_ZERO_POINT: usize = 8;
/// Reserved bytes at offset 12–15 must be zero.
const OFFSET_RESERVED: usize = 12;

// ── PerTensorAffine ───────────────────────────────────────────────────────────

/// Inline quantization parameters for per-tensor affine quantization.
///
/// A single `scale` and `zero_point` pair applies to every element of the tensor.
/// The dequantization formula is:
///
/// ```text
/// x_real = scale * (q - zero_point)
/// ```
///
/// where `q` is the raw storage value, the subtraction is in signed 32-bit
/// integer arithmetic, and the multiplication is in `float32`.
///
/// # Wire format
///
/// Total descriptor length: **16 bytes** (including the 4-byte header).
///
/// | Offset | Field | Type |
/// |--------|-------|------|
/// | 4 | `scale` | `float32` LE |
/// | 8 | `zero_point` | `int32` LE |
/// | 12 | `_reserved` | `uint8[4]` (must be `0x00`) |
///
/// # Design notes
///
/// `PartialEq` only (no `Eq` or `Hash`) because `f32` fields containing NaN
/// would make `Eq` semantically unsound — NaN != NaN by IEEE 754.
///
/// `Copy` because the struct is ≤ 24 bytes with no `Drop` glue.
///
/// # Examples
///
/// ```
/// use hurray_core::PerTensorAffine;
///
/// let q = PerTensorAffine::new(0.5, 128).unwrap();
/// assert_eq!(q.scale(), 0.5f32);
/// assert_eq!(q.zero_point(), 128);
/// ```
// WHY PartialEq only: f32 fields (scale) make Eq/Hash unsound due to NaN semantics
// (design decision #2).
// WHY Copy: ≤24 bytes, no Drop glue, caller gets by-value returns with no indirection
// (design decision #7).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PerTensorAffine {
    scale: f32,
    zero_point: i32,
}

impl PerTensorAffine {
    /// Creates a new [`PerTensorAffine`] descriptor with the given scale and zero point.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidQuantization`] — `scale` is zero, NaN, or infinite.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{PerTensorAffine, Error};
    ///
    /// assert!(PerTensorAffine::new(0.5, 0).is_ok());
    /// assert!(PerTensorAffine::new(-1.0, 128).is_ok());
    ///
    /// // Zero scale is forbidden by the spec.
    /// assert!(PerTensorAffine::new(0.0, 0).is_err());
    /// // NaN scale is forbidden.
    /// assert!(PerTensorAffine::new(f32::NAN, 0).is_err());
    /// // Infinite scale is forbidden.
    /// assert!(PerTensorAffine::new(f32::INFINITY, 0).is_err());
    /// ```
    pub fn new(scale: f32, zero_point: i32) -> Result<Self> {
        if !scale.is_finite() || scale == 0.0 {
            return Err(Error::InvalidQuantization(format!(
                "per-tensor affine scale must be finite and non-zero, got {scale}"
            )));
        }
        Ok(Self { scale, zero_point })
    }

    /// Returns the dequantization scale.
    ///
    /// Guaranteed to be finite and non-zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::PerTensorAffine;
    ///
    /// let q = PerTensorAffine::new(0.25, 0).unwrap();
    /// assert_eq!(q.scale(), 0.25f32);
    /// ```
    #[inline]
    pub fn scale(self) -> f32 {
        self.scale
    }

    /// Returns the quantization zero point.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::PerTensorAffine;
    ///
    /// let q = PerTensorAffine::new(1.0, -128).unwrap();
    /// assert_eq!(q.zero_point(), -128);
    /// ```
    #[inline]
    pub fn zero_point(self) -> i32 {
        self.zero_point
    }

    /// Returns the set of storage [`ElementType`]s that are valid for this scheme.
    ///
    /// Per `docs/spec/quantization/per-tensor-affine.md § Valid Storage Types`.
    ///
    /// WHY `&'static [ElementType]`: no allocation per call; `slice::contains`
    /// over ≤10 items beats any hash structure (design decision #5).
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{ElementType, PerTensorAffine};
    ///
    /// assert!(PerTensorAffine::valid_storage_types().contains(&ElementType::Int8));
    /// assert!(PerTensorAffine::valid_storage_types().contains(&ElementType::Uint4));
    /// assert!(!PerTensorAffine::valid_storage_types().contains(&ElementType::Float32));
    /// ```
    pub fn valid_storage_types() -> &'static [ElementType] {
        &[
            ElementType::Int8,
            ElementType::Uint8,
            ElementType::Int16,
            ElementType::Uint16,
            ElementType::Int32,
            ElementType::Uint32,
            ElementType::Int4,
            ElementType::Uint4,
            ElementType::Int2,
            ElementType::Uint2,
        ]
    }

    /// Validates that `zero_point` is within the representable range of `storage`.
    ///
    /// Per `docs/spec/quantization/per-tensor-affine.md § Validity Constraints`:
    /// `zero_point` MUST lie within the representable range of the storage type.
    ///
    /// # Errors
    ///
    /// - [`Error::ZeroPointOutOfRange`] — `zero_point` is outside the type's range.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{ElementType, PerTensorAffine};
    ///
    /// let q = PerTensorAffine::new(1.0, 200).unwrap();
    /// // 200 is valid for uint8 [0, 255].
    /// assert!(q.validate_zero_point_for(ElementType::Uint8).is_ok());
    /// // 200 is out of range for int8 [-128, 127].
    /// assert!(q.validate_zero_point_for(ElementType::Int8).is_err());
    /// ```
    pub fn validate_zero_point_for(&self, storage: ElementType) -> Result<()> {
        let (min, max) = integer_range_for_type(storage);
        let zp = self.zero_point as i64;
        if zp < min || zp > max {
            return Err(Error::ZeroPointOutOfRange {
                type_tag: storage.tag(),
                zero_point: self.zero_point,
                min,
                max,
            });
        }
        Ok(())
    }

    // ── Crate-internal encode/decode ──────────────────────────────────────────

    /// Decodes the scheme-specific payload from `bytes`.
    ///
    /// `bytes` is the full descriptor slice (including the 4-byte header that
    /// the caller has already validated). `flags` are the header flags (must be
    /// zero for this scheme).
    pub(crate) fn decode_payload(flags: u16, bytes: &[u8]) -> Result<Self> {
        if bytes.len() < ENCODED_LEN {
            return Err(Error::QuantizationDescriptorTooShort {
                found: bytes.len(),
                needed: ENCODED_LEN,
            });
        }
        // No flags defined for this scheme.
        if flags & RESERVED_FLAGS_MASK != 0 {
            return Err(Error::ReservedQuantizationFlagsBits {
                flags,
                mask: RESERVED_FLAGS_MASK,
            });
        }
        let scale = f32::from_le_bytes(
            bytes[OFFSET_SCALE..OFFSET_SCALE + 4]
                .try_into()
                .map_err(|_| Error::InvalidQuantization("scale slice error".into()))?,
        );
        let zero_point = i32::from_le_bytes(
            bytes[OFFSET_ZERO_POINT..OFFSET_ZERO_POINT + 4]
                .try_into()
                .map_err(|_| Error::InvalidQuantization("zero_point slice error".into()))?,
        );
        // Reserved bytes [12..16] must be 0x00.
        if bytes[OFFSET_RESERVED..OFFSET_RESERVED + 4]
            .iter()
            .any(|&b| b != 0)
        {
            return Err(Error::InvalidQuantization(
                "per-tensor affine reserved bytes [12..16] must be 0x00".into(),
            ));
        }
        Self::new(scale, zero_point)
    }

    /// Encodes the scheme-specific payload into `out`.
    ///
    /// `out` must be at least [`ENCODED_LEN`] bytes long. The 4-byte header
    /// (scheme_tag, scheme_version, flags) is written by the caller via
    /// [`super::QuantizationHeader::write`]; this method writes bytes 4–15.
    pub(crate) fn encode_payload(&self, out: &mut [u8]) {
        out[OFFSET_SCALE..OFFSET_SCALE + 4].copy_from_slice(&self.scale.to_le_bytes());
        out[OFFSET_ZERO_POINT..OFFSET_ZERO_POINT + 4]
            .copy_from_slice(&self.zero_point.to_le_bytes());
        // Bytes 12–15: reserved, zero.
        out[OFFSET_RESERVED..OFFSET_RESERVED + 4].fill(0);
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ElementType, Error};

    // ── PerTensorAffine::new ──────────────────────────────────────────────────

    #[test]
    fn new_positive_scale_is_ok() {
        assert!(PerTensorAffine::new(1.0, 0).is_ok());
    }

    #[test]
    fn new_negative_scale_is_ok() {
        assert!(PerTensorAffine::new(-0.5, 0).is_ok());
    }

    #[test]
    fn new_very_small_positive_scale_is_ok() {
        assert!(PerTensorAffine::new(f32::EPSILON, 0).is_ok());
    }

    #[test]
    fn new_large_scale_is_ok() {
        assert!(PerTensorAffine::new(1e38, 0).is_ok());
    }

    #[test]
    fn new_zero_scale_is_err() {
        assert!(matches!(
            PerTensorAffine::new(0.0, 0),
            Err(Error::InvalidQuantization(_))
        ));
    }

    #[test]
    fn new_nan_scale_is_err() {
        assert!(matches!(
            PerTensorAffine::new(f32::NAN, 0),
            Err(Error::InvalidQuantization(_))
        ));
    }

    #[test]
    fn new_positive_infinity_scale_is_err() {
        assert!(matches!(
            PerTensorAffine::new(f32::INFINITY, 0),
            Err(Error::InvalidQuantization(_))
        ));
    }

    #[test]
    fn new_negative_infinity_scale_is_err() {
        assert!(matches!(
            PerTensorAffine::new(f32::NEG_INFINITY, 0),
            Err(Error::InvalidQuantization(_))
        ));
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    #[test]
    fn scale_accessor_returns_stored_value() {
        let q = PerTensorAffine::new(0.25, 0).unwrap();
        assert_eq!(q.scale(), 0.25f32);
    }

    #[test]
    fn zero_point_accessor_returns_stored_value() {
        let q = PerTensorAffine::new(1.0, -128).unwrap();
        assert_eq!(q.zero_point(), -128);
    }

    // ── Round-trips ───────────────────────────────────────────────────────────

    fn encode_then_decode(scale: f32, zero_point: i32) -> PerTensorAffine {
        let original = PerTensorAffine::new(scale, zero_point).unwrap();
        let mut buf = vec![0u8; ENCODED_LEN];
        // Write header manually (scheme_tag=0x01, version=0x01, flags=0x0000).
        buf[0] = SCHEME_TAG;
        buf[1] = SUPPORTED_VERSION;
        buf[2] = 0;
        buf[3] = 0;
        original.encode_payload(&mut buf);
        PerTensorAffine::decode_payload(0, &buf).unwrap()
    }

    #[test]
    fn round_trip_positive_scale_zero_zp() {
        let rt = encode_then_decode(0.5, 0);
        assert_eq!(rt.scale(), 0.5f32);
        assert_eq!(rt.zero_point(), 0);
    }

    #[test]
    fn round_trip_negative_scale_nonzero_zp() {
        let rt = encode_then_decode(-1.0, 128);
        assert_eq!(rt.scale(), -1.0f32);
        assert_eq!(rt.zero_point(), 128);
    }

    #[test]
    fn round_trip_epsilon_scale() {
        let rt = encode_then_decode(f32::EPSILON, 0);
        assert_eq!(rt.scale(), f32::EPSILON);
    }

    #[test]
    fn round_trip_zero_point_i32_min() {
        let rt = encode_then_decode(1.0, i32::MIN);
        assert_eq!(rt.zero_point(), i32::MIN);
    }

    #[test]
    fn round_trip_zero_point_i32_max() {
        let rt = encode_then_decode(1.0, i32::MAX);
        assert_eq!(rt.zero_point(), i32::MAX);
    }

    // ── decode_payload error cases ────────────────────────────────────────────

    #[test]
    fn decode_payload_too_short_is_err() {
        let short = vec![0u8; ENCODED_LEN - 1];
        assert!(matches!(
            PerTensorAffine::decode_payload(0, &short),
            Err(Error::QuantizationDescriptorTooShort { .. })
        ));
    }

    #[test]
    fn decode_payload_nonzero_reserved_bytes_is_err() {
        let mut buf = vec![0u8; ENCODED_LEN];
        buf[0] = SCHEME_TAG;
        buf[1] = SUPPORTED_VERSION;
        buf[2] = 0;
        buf[3] = 0;
        // Write a valid scale.
        buf[4..8].copy_from_slice(&1.0f32.to_le_bytes());
        // zero_point = 0.
        buf[8..12].copy_from_slice(&0i32.to_le_bytes());
        // Pollute reserved bytes [12..16].
        buf[12] = 0x01;
        assert!(matches!(
            PerTensorAffine::decode_payload(0, &buf),
            Err(Error::InvalidQuantization(_))
        ));
    }

    #[test]
    fn decode_payload_nonzero_flags_is_err() {
        let mut buf = vec![0u8; ENCODED_LEN];
        buf[0] = SCHEME_TAG;
        buf[1] = SUPPORTED_VERSION;
        buf[4..8].copy_from_slice(&1.0f32.to_le_bytes());
        // flags = 0x0001 — reserved bit set.
        assert!(matches!(
            PerTensorAffine::decode_payload(0x0001, &buf),
            Err(Error::ReservedQuantizationFlagsBits { .. })
        ));
    }

    #[test]
    fn decode_payload_nan_scale_on_wire_is_err() {
        let mut buf = vec![0u8; ENCODED_LEN];
        buf[0] = SCHEME_TAG;
        buf[1] = SUPPORTED_VERSION;
        // Write a canonical quiet NaN bit pattern for f32.
        buf[4..8].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(matches!(
            PerTensorAffine::decode_payload(0, &buf),
            Err(Error::InvalidQuantization(_))
        ));
    }

    // ── validate_zero_point_for ───────────────────────────────────────────────

    #[test]
    fn validate_zero_point_200_valid_for_uint8() {
        let q = PerTensorAffine::new(1.0, 200).unwrap();
        assert!(q.validate_zero_point_for(ElementType::Uint8).is_ok());
    }

    #[test]
    fn validate_zero_point_200_invalid_for_int8() {
        let q = PerTensorAffine::new(1.0, 200).unwrap();
        assert!(matches!(
            q.validate_zero_point_for(ElementType::Int8),
            Err(Error::ZeroPointOutOfRange { .. })
        ));
    }

    #[test]
    fn validate_zero_point_neg128_valid_for_int8() {
        let q = PerTensorAffine::new(1.0, -128).unwrap();
        assert!(q.validate_zero_point_for(ElementType::Int8).is_ok());
    }

    #[test]
    fn validate_zero_point_neg128_invalid_for_uint8() {
        let q = PerTensorAffine::new(1.0, -128).unwrap();
        assert!(matches!(
            q.validate_zero_point_for(ElementType::Uint8),
            Err(Error::ZeroPointOutOfRange { .. })
        ));
    }

    #[test]
    fn validate_zero_point_15_valid_for_uint4() {
        let q = PerTensorAffine::new(1.0, 15).unwrap();
        assert!(q.validate_zero_point_for(ElementType::Uint4).is_ok());
    }

    #[test]
    fn validate_zero_point_15_invalid_for_int4() {
        // Int4 range is [-8, 7]; 15 is out of range.
        let q = PerTensorAffine::new(1.0, 15).unwrap();
        assert!(matches!(
            q.validate_zero_point_for(ElementType::Int4),
            Err(Error::ZeroPointOutOfRange { .. })
        ));
    }

    #[test]
    fn validate_zero_point_neg8_valid_for_int4() {
        let q = PerTensorAffine::new(1.0, -8).unwrap();
        assert!(q.validate_zero_point_for(ElementType::Int4).is_ok());
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Returns the `(min, max)` representable value range (as `i64`) for integer
/// storage types used as quantization storage.
///
/// Floating-point types and types not used for zero-point quantization return
/// a symmetric `(0, 0)` sentinel; callers validate storage type separately.
pub(crate) fn integer_range_for_type(ty: ElementType) -> (i64, i64) {
    match ty {
        ElementType::Int8 => (-128, 127),
        ElementType::Uint8 => (0, 255),
        ElementType::Int16 => (-32768, 32767),
        ElementType::Uint16 => (0, 65535),
        ElementType::Int32 => (i32::MIN as i64, i32::MAX as i64),
        ElementType::Uint32 => (0, u32::MAX as i64),
        ElementType::Int4 => (-8, 7),
        ElementType::Uint4 => (0, 15),
        ElementType::Int2 => (-2, 1),
        ElementType::Uint2 => (0, 3),
        // Non-integer types: zero-point is always 0, so any i32 is "in range";
        // callers should not invoke this for float storage types.
        _ => (i32::MIN as i64, i32::MAX as i64),
    }
}
