//! Extension type descriptor — binary encode/decode.
//!
//! Present when the `HAS_EXTENSION_TYPE` flag is set (i.e., `type_tag` is in
//! `0xF0`–`0xFE`). Provides the bit-width and packing parameters necessary to
//! compute buffer sizes for private extension types.
//!
//! Wire layout (all little-endian, total = 20 bytes):
//! ```text
//! offset  0: bit_width       uint32
//! offset  4: packing_factor  uint8
//! offset  5: is_float        uint8   (0x01 or 0x00)
//! offset  6: is_signed       uint8   (0x01 or 0x00)
//! offset  7: sign_bits       uint8
//! offset  8: exponent_bits   uint8
//! offset  9: mantissa_bits   uint8
//! offset 10: _reserved       uint8[2]  MUST be 0
//! offset 12: exponent_bias   uint32
//! offset 16: has_nan         uint8
//! offset 17: has_inf         uint8
//! offset 18: _reserved2      uint8[2]  MUST be 0
//! ```

use crate::descriptor::cursor::{ByteCursor, ByteWriter};
use crate::{Error, Result};

/// Total byte length of the encoded extension type section.
#[cfg(test)]
const EXT_TYPE_BYTE_LEN: usize = 20;

/// Inline descriptor for an implementation-private extension element type.
///
/// Extension type tags (`0xF0`–`0xFE`) MUST carry this descriptor, which
/// provides at minimum the `bit_width` and `packing_factor` needed to compute
/// buffer sizes without understanding the numeric semantics of the type.
///
/// # Packing rules (spec § Extension Type Section)
///
/// - If `bit_width >= 8`: `packing_factor` MUST be `1`.
/// - If `bit_width < 8`: `bit_width` MUST be `1`, `2`, or `4`; `packing_factor`
///   MUST equal `8 / bit_width` (`8`, `4`, or `2` respectively).
/// - All other sub-byte widths (3, 5, 6, 7) MUST NOT be encoded as extension types.
///
/// # Examples
///
/// ```
/// use hurray_core::descriptor::ExtensionTypeDescriptor;
///
/// // 8-bit integer extension type.
/// let ext = ExtensionTypeDescriptor::new(8, 1, false, true, 0, 0, 0, 0, false, false).unwrap();
/// assert_eq!(ext.bit_width, 8);
/// assert_eq!(ext.packing_factor, 1);
///
/// // 4-bit sub-byte extension type (packing_factor must be 2).
/// let sub = ExtensionTypeDescriptor::new(4, 2, false, false, 0, 0, 0, 0, false, false).unwrap();
/// assert_eq!(sub.packing_factor, 2);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtensionTypeDescriptor {
    /// Bit width of one element. MUST be > 0.
    pub bit_width: u32,
    /// Elements packed per byte. MUST be `1` for `bit_width >= 8`; `8/bit_width` for sub-byte.
    pub packing_factor: u8,
    /// `true` if floating-point, `false` if integer.
    pub is_float: bool,
    /// `true` if signed integer. MUST be `false` for float types.
    pub is_signed: bool,
    /// Number of sign bits (for float types). MUST be 0 or 1.
    pub sign_bits: u8,
    /// Number of exponent bits (for float types).
    pub exponent_bits: u8,
    /// Number of mantissa bits (for float types).
    pub mantissa_bits: u8,
    /// Exponent bias (for float types).
    pub exponent_bias: u32,
    /// `true` if NaN is representable (float types only).
    pub has_nan: bool,
    /// `true` if infinity is representable (float types only).
    pub has_inf: bool,
}

impl ExtensionTypeDescriptor {
    /// Creates a new [`ExtensionTypeDescriptor`], validating the packing rule.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ExtensionTypePackingInvalid`] if the `bit_width` /
    /// `packing_factor` combination violates the spec constraint.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::descriptor::ExtensionTypeDescriptor;
    /// use hurray_core::Error;
    ///
    /// // Valid whole-byte type.
    /// assert!(ExtensionTypeDescriptor::new(16, 1, true, false, 1, 5, 10, 15, true, false).is_ok());
    ///
    /// // Invalid: bit_width=3 is not 1, 2, or 4.
    /// assert!(matches!(
    ///     ExtensionTypeDescriptor::new(3, 1, false, false, 0, 0, 0, 0, false, false),
    ///     Err(Error::ExtensionTypePackingInvalid { .. })
    /// ));
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        bit_width: u32,
        packing_factor: u8,
        is_float: bool,
        is_signed: bool,
        sign_bits: u8,
        exponent_bits: u8,
        mantissa_bits: u8,
        exponent_bias: u32,
        has_nan: bool,
        has_inf: bool,
    ) -> Result<Self> {
        validate_packing(bit_width, packing_factor)?;
        Ok(Self {
            bit_width,
            packing_factor,
            is_float,
            is_signed,
            sign_bits,
            exponent_bits,
            mantissa_bits,
            exponent_bias,
            has_nan,
            has_inf,
        })
    }

    /// Encodes this descriptor into `w` as an exact 20-byte block.
    pub(crate) fn encode_into(&self, w: &mut ByteWriter) {
        w.write_u32_le(self.bit_width); // offset  0
        w.write_u8(self.packing_factor); // offset  4
        w.write_u8(u8::from(self.is_float)); // offset  5
        w.write_u8(u8::from(self.is_signed)); // offset  6
        w.write_u8(self.sign_bits); // offset  7
        w.write_u8(self.exponent_bits); // offset  8
        w.write_u8(self.mantissa_bits); // offset  9
        w.write_zeros(2); // offset 10 — _reserved
        w.write_u32_le(self.exponent_bias); // offset 12
        w.write_u8(u8::from(self.has_nan)); // offset 16
        w.write_u8(u8::from(self.has_inf)); // offset 17
        w.write_zeros(2); // offset 18 — _reserved2
    }

    /// Decodes a 20-byte extension type block from `cursor`.
    ///
    /// # Errors
    ///
    /// - [`Error::ExtensionTypePackingInvalid`] if the packing rule is violated.
    /// - [`Error::ReservedBytesNonZero`] if any `_reserved` field is non-zero.
    /// - [`Error::DescriptorTruncated`] if fewer than 20 bytes remain.
    pub(crate) fn decode_from(cursor: &mut ByteCursor<'_>) -> Result<Self> {
        let bit_width = cursor.read_u32_le()?; // offset  0
        let packing_factor = cursor.read_u8()?; // offset  4
        let is_float = cursor.read_u8()? != 0; // offset  5
        let is_signed = cursor.read_u8()? != 0; // offset  6
        let sign_bits = cursor.read_u8()?; // offset  7
        let exponent_bits = cursor.read_u8()?; // offset  8
        let mantissa_bits = cursor.read_u8()?; // offset  9
        let reserved1 = cursor.read_bytes(2)?; // offset 10
        let exponent_bias = cursor.read_u32_le()?; // offset 12
        let has_nan = cursor.read_u8()? != 0; // offset 16
        let has_inf = cursor.read_u8()? != 0; // offset 17
        let reserved2 = cursor.read_bytes(2)?; // offset 18

        validate_packing(bit_width, packing_factor)?;

        if reserved1 != [0u8, 0] {
            return Err(Error::ReservedBytesNonZero {
                field: "ext_type._reserved",
            });
        }
        if reserved2 != [0u8, 0] {
            return Err(Error::ReservedBytesNonZero {
                field: "ext_type._reserved2",
            });
        }

        Ok(Self {
            bit_width,
            packing_factor,
            is_float,
            is_signed,
            sign_bits,
            exponent_bits,
            mantissa_bits,
            exponent_bias,
            has_nan,
            has_inf,
        })
    }
}

/// Validates the `bit_width` / `packing_factor` pairing per spec rules.
fn validate_packing(bit_width: u32, packing_factor: u8) -> Result<()> {
    if bit_width == 0 {
        return Err(Error::ExtensionTypePackingInvalid {
            bit_width,
            packing_factor,
        });
    }
    if bit_width >= 8 {
        // Whole-byte type: packing_factor MUST be 1.
        if packing_factor != 1 {
            return Err(Error::ExtensionTypePackingInvalid {
                bit_width,
                packing_factor,
            });
        }
    } else {
        // Sub-byte type: bit_width MUST be 1, 2, or 4; packing_factor MUST be 8/bit_width.
        let expected_packing = match bit_width {
            1 => 8u8,
            2 => 4u8,
            4 => 2u8,
            // bit_width 3, 5, 6, 7 — not valid for extension types.
            _ => {
                return Err(Error::ExtensionTypePackingInvalid {
                    bit_width,
                    packing_factor,
                })
            }
        };
        if packing_factor != expected_packing {
            return Err(Error::ExtensionTypePackingInvalid {
                bit_width,
                packing_factor,
            });
        }
    }
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::cursor::{ByteCursor, ByteWriter};

    fn sample_ext() -> ExtensionTypeDescriptor {
        ExtensionTypeDescriptor::new(16, 1, true, false, 1, 5, 10, 15, true, false).unwrap()
    }

    #[test]
    fn ext_type_round_trip() {
        let ext = sample_ext();
        let mut w = ByteWriter::new();
        ext.encode_into(&mut w);
        let bytes = w.into_vec();
        assert_eq!(bytes.len(), EXT_TYPE_BYTE_LEN);
        let mut c = ByteCursor::new(&bytes, bytes.len());
        let decoded = ExtensionTypeDescriptor::decode_from(&mut c).unwrap();
        assert_eq!(decoded, ext);
    }

    #[test]
    fn ext_type_whole_byte_packing_ok() {
        // bit_width=32, packing_factor=1 — valid whole-byte type.
        let ext = ExtensionTypeDescriptor::new(32, 1, false, true, 0, 0, 0, 0, false, false);
        assert!(ext.is_ok());
    }

    #[test]
    fn ext_type_sub_byte_packing_ok() {
        // bit_width=4, packing_factor=2 — valid sub-byte type.
        let ext = ExtensionTypeDescriptor::new(4, 2, false, false, 0, 0, 0, 0, false, false);
        assert!(ext.is_ok());
    }

    #[test]
    fn ext_type_sub_byte_packing_1bit_ok() {
        let ext = ExtensionTypeDescriptor::new(1, 8, false, false, 0, 0, 0, 0, false, false);
        assert!(ext.is_ok());
    }

    #[test]
    fn ext_type_sub_byte_packing_2bit_ok() {
        let ext = ExtensionTypeDescriptor::new(2, 4, false, false, 0, 0, 0, 0, false, false);
        assert!(ext.is_ok());
    }

    #[test]
    fn ext_type_invalid_sub_byte_width_rejected() {
        // bit_width=3 is not 1, 2, or 4 — must be rejected.
        let err =
            ExtensionTypeDescriptor::new(3, 1, false, false, 0, 0, 0, 0, false, false).unwrap_err();
        assert!(matches!(
            err,
            Error::ExtensionTypePackingInvalid { bit_width: 3, .. }
        ));
    }

    #[test]
    fn ext_type_invalid_sub_byte_width_5_rejected() {
        let err =
            ExtensionTypeDescriptor::new(5, 1, false, false, 0, 0, 0, 0, false, false).unwrap_err();
        assert!(matches!(
            err,
            Error::ExtensionTypePackingInvalid { bit_width: 5, .. }
        ));
    }

    #[test]
    fn ext_type_wrong_packing_factor_rejected() {
        // bit_width=4 requires packing_factor=2, not 1.
        let err =
            ExtensionTypeDescriptor::new(4, 1, false, false, 0, 0, 0, 0, false, false).unwrap_err();
        assert!(matches!(
            err,
            Error::ExtensionTypePackingInvalid {
                bit_width: 4,
                packing_factor: 1
            }
        ));
    }

    #[test]
    fn ext_type_whole_byte_wrong_packing_rejected() {
        // bit_width=8 requires packing_factor=1, not 2.
        let err =
            ExtensionTypeDescriptor::new(8, 2, false, false, 0, 0, 0, 0, false, false).unwrap_err();
        assert!(matches!(
            err,
            Error::ExtensionTypePackingInvalid {
                bit_width: 8,
                packing_factor: 2
            }
        ));
    }

    #[test]
    fn ext_type_reserved_bytes_rejected() {
        let mut w = ByteWriter::new();
        w.write_u32_le(16u32); // bit_width
        w.write_u8(1u8); // packing_factor
        w.write_u8(0u8); // is_float
        w.write_u8(0u8); // is_signed
        w.write_u8(0u8); // sign_bits
        w.write_u8(0u8); // exponent_bits
        w.write_u8(0u8); // mantissa_bits
        w.write_u8(0xFFu8); // _reserved[0] — non-zero, must be rejected
        w.write_u8(0u8); // _reserved[1]
        w.write_u32_le(0u32); // exponent_bias
        w.write_u8(0u8); // has_nan
        w.write_u8(0u8); // has_inf
        w.write_zeros(2); // _reserved2
        let bytes = w.into_vec();
        let mut c = ByteCursor::new(&bytes, bytes.len());
        let err = ExtensionTypeDescriptor::decode_from(&mut c).unwrap_err();
        assert!(matches!(err, Error::ReservedBytesNonZero { .. }));
    }

    #[test]
    fn ext_type_zero_bit_width_rejected() {
        let err =
            ExtensionTypeDescriptor::new(0, 1, false, false, 0, 0, 0, 0, false, false).unwrap_err();
        assert!(matches!(
            err,
            Error::ExtensionTypePackingInvalid { bit_width: 0, .. }
        ));
    }

    #[test]
    fn encoded_length_is_20_bytes() {
        let mut w = ByteWriter::new();
        sample_ext().encode_into(&mut w);
        assert_eq!(w.len(), EXT_TYPE_BYTE_LEN);
    }
}
