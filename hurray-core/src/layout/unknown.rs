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
/// # Examples
///
/// ```
/// use hurray_core::layout::{LayoutDescriptor, UnknownLayout};
///
/// // Simulate a permissive reader accepting an unrecognised tag.
/// let layout = LayoutDescriptor::Unknown(UnknownLayout::new(0x0B, vec![0x00, 0x01]).unwrap());
/// assert_eq!(layout.tag(), 0x0B);
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
    /// Returns [`Error::InvalidLayoutTag`] if `tag` is `0x00` or `0xFF` — these
    /// sentinels are permanently invalid and MUST be rejected even in permissive mode.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::layout::UnknownLayout;
    ///
    /// let u = UnknownLayout::new(0x0B, vec![1, 2, 3]).unwrap();
    /// assert_eq!(u.tag, 0x0B);
    /// assert_eq!(u.raw_bytes, [1, 2, 3]);
    ///
    /// // Permanently-invalid sentinels are rejected even in permissive mode.
    /// assert!(UnknownLayout::new(0x00, vec![]).is_err());
    /// assert!(UnknownLayout::new(0xFF, vec![]).is_err());
    /// ```
    pub fn new(tag: u8, raw_bytes: Vec<u8>) -> Result<Self> {
        // Permanently-invalid sentinels must be rejected in all modes per spec.
        if tag == 0x00 || tag == 0xFF {
            return Err(Error::InvalidLayoutTag(tag));
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
        let layout = LayoutDescriptor::Unknown(UnknownLayout::new(0x0B, vec![]).unwrap());
        assert_eq!(layout.tag(), 0x0B);
    }

    #[test]
    fn unknown_buffer_count_is_none() {
        let layout = LayoutDescriptor::Unknown(UnknownLayout::new(0x0B, vec![1, 2, 3]).unwrap());
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
        assert!(UnknownLayout::new(0x0B, vec![]).is_ok());
        assert!(UnknownLayout::new(0xEF, vec![]).is_ok());
    }
}
