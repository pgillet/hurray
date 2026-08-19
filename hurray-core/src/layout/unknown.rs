//! Unknown layout descriptor — permissive-mode fallback.
//!
//! Carries a raw tag byte and opaque payload. Only reachable via the permissive
//! constructor path; the named-variant constructors always reject unrecognised tags.

use crate::{Error, Result};

/// Descriptor for an unrecognized layout, accepted only in permissive mode.
///
/// A conforming reader in **permissive mode** MAY accept tensor descriptors
/// whose layout tag is not recognized, but MUST NOT dereference or interpret
/// the tensor data buffer for such tensors. The `Unknown` variant preserves the
/// raw tag and bytes for inspection without dereferencing.
///
/// Named-variant constructors on [`LayoutDescriptor`](super::LayoutDescriptor)
/// reject tags `0x00`, `0xFF`, reserved ranges, and private-extension tags —
/// callers that want to pass unknown layouts through must construct
/// `Unknown` explicitly (or via a permissive decoder in a higher layer).
///
/// `buffer_count` is always `None` for `Unknown` because the number of
/// required buffers is not known without understanding the layout.
///
/// The constructor rejects any tag this implementation *does* understand — see
/// [`UnknownLayout::new`]. The type means "I could not parse this", and that claim
/// has to stay true, because a permissive relay downstream acts on it.
///
/// # Examples
///
/// ```
/// use hurray_core::layout::{LayoutDescriptor, UnknownLayout};
///
/// // Simulate a permissive reader accepting an unrecognised tag.
/// let layout = LayoutDescriptor::Unknown(UnknownLayout::new(0x0C, vec![0x00, 0x01]).unwrap());
/// assert_eq!(layout.tag(), 0x0C);
/// assert!(layout.buffer_count().is_none());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct UnknownLayout {
    /// The raw tag byte from the wire, exactly as received.
    pub tag: u8,

    /// The raw layout-specific bytes following the tag, as received.
    pub raw_bytes: Vec<u8>,
}

impl UnknownLayout {
    /// Creates a new [`UnknownLayout`] from a raw tag byte and opaque payload bytes.
    ///
    /// # Errors
    ///
    /// | Tag | Error |
    /// |-----|-------|
    /// | `0x00`, `0xFF` | [`Error::InvalidLayoutTag`] — permanently invalid in every mode |
    /// | a tag with a named variant | [`Error::NamedLayoutTag`] |
    /// | `0xF0`–`0xFE` | [`Error::PrivateLayoutTag`] — use [`PrivateExtensionLayout`](super::PrivateExtensionLayout) |
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::layout::UnknownLayout;
    ///
    /// let u = UnknownLayout::new(0x0C, vec![1, 2, 3]).unwrap();
    /// assert_eq!(u.tag, 0x0C);
    /// assert_eq!(u.raw_bytes, [1, 2, 3]);
    ///
    /// // Permanently-invalid sentinels are rejected even in permissive mode.
    /// assert!(UnknownLayout::new(0x00, vec![]).is_err());
    /// assert!(UnknownLayout::new(0xFF, vec![]).is_err());
    ///
    /// // So is a tag this implementation understands.
    /// assert!(UnknownLayout::new(0x07, vec![]).is_err()); // CSR
    /// assert!(UnknownLayout::new(0xF0, vec![]).is_err()); // private extension
    /// ```
    pub fn new(tag: u8, raw_bytes: Vec<u8>) -> Result<Self> {
        // Permanently-invalid sentinels must be rejected in all modes per spec.
        if super::is_invalid_tag(tag) {
            return Err(Error::InvalidLayoutTag(tag));
        }
        // "Unknown" must not claim a tag this implementation understands. Unknown has
        // no buffer count and no shape constraints, so such a descriptor would skip
        // every check the named variant applies, then encode to a wire tag a
        // conforming reader parses as that named layout.
        if super::is_named_tag(tag) {
            return Err(Error::NamedLayoutTag(tag));
        }
        // A private tag is understood too — as an extension id plus payload, which
        // PrivateExtensionLayout preserves and this type would flatten into opaque
        // bytes.
        if super::is_private_tag(tag) {
            return Err(Error::PrivateLayoutTag(tag));
        }
        Ok(Self { tag, raw_bytes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{layout::LayoutDescriptor, Error};

    #[test]
    fn unknown_tag_passthrough() {
        let layout = LayoutDescriptor::Unknown(UnknownLayout::new(0x0C, vec![]).unwrap());
        assert_eq!(layout.tag(), 0x0C);
    }

    #[test]
    fn unknown_buffer_count_is_none() {
        let layout = LayoutDescriptor::Unknown(UnknownLayout::new(0x0C, vec![1, 2, 3]).unwrap());
        assert!(layout.buffer_count().is_none());
    }

    #[test]
    fn rejects_invalid_sentinel_0x00() {
        assert!(matches!(
            UnknownLayout::new(0x00, vec![]),
            Err(Error::InvalidLayoutTag(0x00))
        ));
    }

    #[test]
    fn rejects_invalid_sentinel_0xff() {
        assert!(matches!(
            UnknownLayout::new(0xFF, vec![]),
            Err(Error::InvalidLayoutTag(0xFF))
        ));
    }

    #[test]
    fn accepts_reserved_range_tag_in_permissive_mode() {
        // Reserved tags are not permanently invalid — permissive mode may accept them.
        assert!(UnknownLayout::new(0x0C, vec![]).is_ok());
        assert!(UnknownLayout::new(0x3F, vec![]).is_ok());
        assert!(UnknownLayout::new(0xEF, vec![]).is_ok());
    }

    #[test]
    fn rejects_every_named_tag() {
        // Wrapping a tag this crate understands would skip the checks its named
        // variant applies, while still encoding to that tag on the wire.
        for tag in [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x40,
        ] {
            assert!(
                matches!(UnknownLayout::new(tag, vec![]), Err(Error::NamedLayoutTag(t)) if t == tag),
                "tag 0x{tag:02X} must be rejected as named"
            );
        }
    }

    #[test]
    fn rejects_private_tags_in_favour_of_the_extension_type() {
        // PrivateExtensionLayout keeps the extension id; Unknown would flatten it.
        assert!(matches!(
            UnknownLayout::new(0xF0, vec![]),
            Err(Error::PrivateLayoutTag(0xF0))
        ));
        assert!(matches!(
            UnknownLayout::new(0xFE, vec![]),
            Err(Error::PrivateLayoutTag(0xFE))
        ));
    }
}
