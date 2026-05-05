//! Private extension layout descriptor.
//!
//! Tags `0xF0`–`0xFE`. Implementation-private; opaque to all conforming readers
//! without an out-of-band semantic agreement.
//! See `docs/spec/memory-layout.md § Extension Layouts`.

use crate::{Error, Result};

/// The inclusive byte range of private-extension layout tags.
pub const PRIVATE_LAYOUT_TAG_MIN: u8 = 0xF0;
/// The inclusive upper bound of the private-extension layout tag range.
pub const PRIVATE_LAYOUT_TAG_MAX: u8 = 0xFE;

/// Descriptor for an implementation-private extension layout.
///
/// Tags in `0xF0`–`0xFE` are reserved for implementation-private layouts.
/// Tensors using these tags MUST NOT be exchanged between independent
/// implementations unless both parties have agreed on the layout semantics out
/// of band.
///
/// An extension layout descriptor carries:
/// - an `extension_layout_id` (`uint64`): implementation-defined unique identifier,
/// - `extension_data` (`Vec<u8>`): opaque layout-specific metadata.
///
/// The `Unknown` variant (for unrecognized tags in permissive mode) is
/// separate; see [`UnknownLayout`].
///
/// # Examples
///
/// ```
/// use hurray_core::layout::{LayoutDescriptor, PrivateExtensionLayout};
///
/// let layout = LayoutDescriptor::PrivateExtension(
///     PrivateExtensionLayout::new(0xF0, 0xDEAD_BEEF_CAFE_0001, vec![1, 2, 3]).unwrap(),
/// );
/// assert_eq!(layout.tag(), 0xF0);
/// // buffer_count is None — unknown for private layouts.
/// assert!(layout.buffer_count().is_none());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct PrivateExtensionLayout {
    /// The layout tag byte (`0xF0`–`0xFE`).
    pub tag: u8,

    /// Implementation-defined unique identifier for this extension layout.
    pub extension_layout_id: u64,

    /// Opaque layout-specific metadata.
    pub extension_data: Vec<u8>,
}

impl PrivateExtensionLayout {
    /// Creates a new [`PrivateExtensionLayout`], validating that `tag` is in
    /// the private-extension range `0xF0`–`0xFE`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidLayout`] if `tag` is outside `0xF0`–`0xFE`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::layout::PrivateExtensionLayout;
    ///
    /// let p = PrivateExtensionLayout::new(0xF1, 42, vec![0xAB, 0xCD]).unwrap();
    /// assert_eq!(p.tag, 0xF1);
    ///
    /// // Tags outside the private range are rejected.
    /// assert!(PrivateExtensionLayout::new(0x01, 0, vec![]).is_err());
    /// assert!(PrivateExtensionLayout::new(0xFF, 0, vec![]).is_err());
    /// ```
    pub fn new(tag: u8, extension_layout_id: u64, extension_data: Vec<u8>) -> Result<Self> {
        if !(PRIVATE_LAYOUT_TAG_MIN..=PRIVATE_LAYOUT_TAG_MAX).contains(&tag) {
            return Err(Error::InvalidLayout(format!(
                "private extension layout tag 0x{tag:02X} is outside the valid range \
                 0x{PRIVATE_LAYOUT_TAG_MIN:02X}–0x{PRIVATE_LAYOUT_TAG_MAX:02X}"
            )));
        }
        Ok(Self {
            tag,
            extension_layout_id,
            extension_data,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::LayoutDescriptor;

    #[test]
    fn private_extension_tag_passthrough() {
        for tag in 0xF0u8..=0xFEu8 {
            let layout = LayoutDescriptor::PrivateExtension(
                PrivateExtensionLayout::new(tag, 0, vec![]).unwrap(),
            );
            assert_eq!(layout.tag(), tag);
        }
    }

    #[test]
    fn private_extension_buffer_count_is_none() {
        let layout = LayoutDescriptor::PrivateExtension(
            PrivateExtensionLayout::new(0xF0, 0, vec![]).unwrap(),
        );
        assert!(layout.buffer_count().is_none());
    }

    #[test]
    fn rejects_tag_below_range() {
        // 0xEF is reserved, not private.
        assert!(matches!(
            PrivateExtensionLayout::new(0xEF, 0, vec![]),
            Err(Error::InvalidLayout(_))
        ));
    }

    #[test]
    fn rejects_tag_0xff() {
        // 0xFF is permanently invalid.
        assert!(matches!(
            PrivateExtensionLayout::new(0xFF, 0, vec![]),
            Err(Error::InvalidLayout(_))
        ));
    }

    #[test]
    fn rejects_core_layout_tag() {
        assert!(matches!(
            PrivateExtensionLayout::new(0x01, 0, vec![]),
            Err(Error::InvalidLayout(_))
        ));
    }
}
