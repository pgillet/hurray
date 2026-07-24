//! Composite Member section — binary encode/decode.
//!
//! A composite member descriptor identifies this tensor's role (`base` or
//! `correction`) within an **overlay** composite (`composition_rule = 0x02`; see
//! `docs/spec/layouts/composite.md`). It is present in the wire format when the
//! `HAS_COMPOSITE_MEMBER` flag is set, and appears after the Extension Type section.
//!
//! Wire layout (spec § Composite Member Section), a fixed 16-byte block:
//! ```text
//! member_role  uint8     (1 byte)
//! _reserved    uint8[15] (15 bytes) — MUST be 0x00
//! ```

use crate::descriptor::cursor::{ByteCursor, ByteWriter};
use crate::{Error, Result};

/// Total byte length of the encoded Composite Member section.
pub(crate) const COMPOSITE_MEMBER_BYTE_LEN: usize = 16;

/// The role a member plays within an overlay composite.
///
/// v1.0 defines only `Correction` and `Base`; wire values `0x02`–`0xFF` are
/// RESERVED for a future tombstone kind (see `docs/spec/layouts/composite.md`
/// § Deferred) and are rejected by [`MemberRole::from_wire`].
///
/// # Examples
///
/// ```
/// use hurray_core::descriptor::MemberRole;
///
/// assert_eq!(MemberRole::Correction.wire_byte(), 0x00);
/// assert_eq!(MemberRole::Base.wire_byte(), 0x01);
/// assert_eq!(MemberRole::from_wire(0x01).unwrap(), MemberRole::Base);
/// assert!(MemberRole::from_wire(0x02).is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MemberRole {
    /// A scattered correction applied within its box. Wire byte `0x00`.
    Correction,
    /// The base member spanning the whole index space. Wire byte `0x01`.
    Base,
}

impl MemberRole {
    /// Returns the wire byte for this role.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::descriptor::MemberRole;
    ///
    /// assert_eq!(MemberRole::Base.wire_byte(), 0x01);
    /// ```
    #[inline]
    pub fn wire_byte(self) -> u8 {
        match self {
            Self::Correction => 0x00,
            Self::Base => 0x01,
        }
    }

    /// Constructs a [`MemberRole`] from its wire byte.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidMemberRole`] for any byte other than `0x00` or `0x01`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{Error, descriptor::MemberRole};
    ///
    /// assert_eq!(MemberRole::from_wire(0x00).unwrap(), MemberRole::Correction);
    /// assert!(matches!(MemberRole::from_wire(0xFF), Err(Error::InvalidMemberRole(0xFF))));
    /// ```
    #[inline]
    pub fn from_wire(byte: u8) -> Result<Self> {
        match byte {
            0x00 => Ok(Self::Correction),
            0x01 => Ok(Self::Base),
            _ => Err(Error::InvalidMemberRole(byte)),
        }
    }
}

/// The Composite Member section: a member's role within an overlay composite.
///
/// Carried by [`crate::descriptor::TensorDescriptor::composite_member`], gated by
/// the `HAS_COMPOSITE_MEMBER` descriptor flag (bit 4). See
/// `docs/spec/layouts/composite.md` § Composite Member Section and
/// [`crate::composite::CompositeValidator`] for the cross-member rules that apply
/// this role (first member MUST be base and span the index space; subsequent
/// members MUST be corrections).
///
/// # Examples
///
/// ```
/// use hurray_core::descriptor::{CompositeMemberDescriptor, MemberRole};
///
/// let base = CompositeMemberDescriptor::new(MemberRole::Base);
/// assert_eq!(base.member_role, MemberRole::Base);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CompositeMemberDescriptor {
    /// This member's role within the enclosing overlay composite.
    pub member_role: MemberRole,
}

impl CompositeMemberDescriptor {
    /// Creates a new [`CompositeMemberDescriptor`].
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::descriptor::{CompositeMemberDescriptor, MemberRole};
    ///
    /// let cm = CompositeMemberDescriptor::new(MemberRole::Correction);
    /// assert_eq!(cm.member_role, MemberRole::Correction);
    /// ```
    pub fn new(member_role: MemberRole) -> Self {
        Self { member_role }
    }

    /// Encodes this section into `w` as an exact [`COMPOSITE_MEMBER_BYTE_LEN`]-byte block.
    pub(crate) fn encode_into(&self, w: &mut ByteWriter) {
        let start = w.len();
        w.write_u8(self.member_role.wire_byte());
        w.write_zeros(15); // _reserved — MUST be 0x00
        debug_assert_eq!(
            w.len() - start,
            COMPOSITE_MEMBER_BYTE_LEN,
            "composite member encoded size invariant violated"
        );
    }

    /// Decodes a [`COMPOSITE_MEMBER_BYTE_LEN`]-byte Composite Member section from `cursor`.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidMemberRole`] if `member_role` is not `0x00` or `0x01`.
    /// - [`Error::ReservedBytesNonZero`] if any `_reserved` byte is non-zero.
    /// - [`Error::DescriptorTruncated`] if fewer than [`COMPOSITE_MEMBER_BYTE_LEN`] bytes remain.
    pub(crate) fn decode_from(cursor: &mut ByteCursor<'_>) -> Result<Self> {
        let role_byte = cursor.read_u8()?;
        let member_role = MemberRole::from_wire(role_byte)?;

        let reserved = cursor.read_bytes(15)?;
        if reserved.iter().any(|&b| b != 0) {
            return Err(Error::ReservedBytesNonZero {
                field: "composite_member._reserved",
            });
        }

        Ok(Self { member_role })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(desc: &CompositeMemberDescriptor) -> CompositeMemberDescriptor {
        let mut w = ByteWriter::new();
        desc.encode_into(&mut w);
        let bytes = w.into_vec();
        assert_eq!(bytes.len(), COMPOSITE_MEMBER_BYTE_LEN);
        let mut c = ByteCursor::new(&bytes, bytes.len());
        CompositeMemberDescriptor::decode_from(&mut c).unwrap()
    }

    // ── MemberRole wire round trip ──────────────────────────────────────────

    #[test]
    fn member_role_wire_round_trip() {
        assert_eq!(MemberRole::Correction.wire_byte(), 0x00);
        assert_eq!(MemberRole::Base.wire_byte(), 0x01);
        assert_eq!(MemberRole::from_wire(0x00).unwrap(), MemberRole::Correction);
        assert_eq!(MemberRole::from_wire(0x01).unwrap(), MemberRole::Base);
    }

    #[test]
    fn member_role_rejects_reserved_bytes() {
        for byte in [0x02_u8, 0x50, 0xFE, 0xFF] {
            assert!(
                matches!(MemberRole::from_wire(byte), Err(Error::InvalidMemberRole(b)) if b == byte),
                "0x{byte:02X} should be rejected as InvalidMemberRole"
            );
        }
    }

    // ── CompositeMemberDescriptor encode/decode round trip ──────────────────

    #[test]
    fn composite_member_round_trip_correction() {
        let desc = CompositeMemberDescriptor::new(MemberRole::Correction);
        assert_eq!(round_trip(&desc), desc);
    }

    #[test]
    fn composite_member_round_trip_base() {
        let desc = CompositeMemberDescriptor::new(MemberRole::Base);
        assert_eq!(round_trip(&desc), desc);
    }

    #[test]
    fn composite_member_encode_reserved_bytes_are_zero() {
        let desc = CompositeMemberDescriptor::new(MemberRole::Base);
        let mut w = ByteWriter::new();
        desc.encode_into(&mut w);
        let bytes = w.into_vec();
        assert_eq!(bytes[0], 0x01); // member_role
        assert!(
            bytes[1..16].iter().all(|&b| b == 0),
            "_reserved must be all-zero"
        );
    }

    // ── Rejection when reserved bytes are non-zero ──────────────────────────

    #[test]
    fn composite_member_decode_rejects_nonzero_reserved_bytes() {
        let mut w = ByteWriter::new();
        w.write_u8(MemberRole::Base.wire_byte());
        w.write_zeros(14);
        w.write_u8(0xFF); // last reserved byte non-zero
        let bytes = w.into_vec();
        let mut c = ByteCursor::new(&bytes, bytes.len());
        let err = CompositeMemberDescriptor::decode_from(&mut c).unwrap_err();
        assert!(matches!(err, Error::ReservedBytesNonZero { .. }));
    }

    #[test]
    fn composite_member_decode_rejects_invalid_role() {
        let mut w = ByteWriter::new();
        w.write_u8(0xFF); // invalid member_role
        w.write_zeros(15);
        let bytes = w.into_vec();
        let mut c = ByteCursor::new(&bytes, bytes.len());
        let err = CompositeMemberDescriptor::decode_from(&mut c).unwrap_err();
        assert!(matches!(err, Error::InvalidMemberRole(0xFF)));
    }

    #[test]
    fn composite_member_decode_truncated() {
        // Only 5 bytes available, need 16.
        let bytes = [0x00u8, 0x00, 0x00, 0x00, 0x00];
        let mut c = ByteCursor::new(&bytes, bytes.len());
        let err = CompositeMemberDescriptor::decode_from(&mut c).unwrap_err();
        assert!(matches!(err, Error::DescriptorTruncated { .. }));
    }
}
