//! Composite / Virtual layout descriptor — the head payload for a composite tensor.
//!
//! Tag `0x0B`. The composite head is a **virtual** (data-less) descriptor: it owns
//! no buffers. Its layout-specific payload declares the composition rule that binds
//! the `member_count` tensor descriptors that immediately follow it in stream/file
//! order. See `docs/spec/layouts/composite.md` for the full normative definition.
//!
//! This module carries only the wire-level head payload (`CompositeLayout` and its
//! `CompositionRule`/`CombineOp` fields). The cross-descriptor aggregate — a head
//! plus its actual member `TensorDescriptor`s, and the stateful validator that
//! checks them against each other — lives in the separate top-level
//! [`crate::composite`] module (distinct module path; see that module's docs for
//! why the two are kept apart).

use crate::{Error, Result};

/// Wire sentinel for `member_count` denoting an **open composite** (unbounded,
/// appendable membership). RESERVED in v1.0 — see `docs/spec/layouts/composite.md`
/// § Head Layout-Specific Fields. [`CompositeLayout::new`] rejects it.
pub(crate) const OPEN_COMPOSITE_SENTINEL: u32 = 0xFFFF_FFFF;

/// The combine operation for an **overlay** composition (`composition_rule = 0x02`).
///
/// Selects how a correction member's value at a covered index combines with the
/// value shown by lower-precedence members (down to the base).
///
/// # Examples
///
/// ```
/// use hurray_core::layout::CombineOp;
///
/// assert_eq!(CombineOp::Replace.wire_byte(), 0x01);
/// assert_eq!(CombineOp::Add.wire_byte(), 0x02);
/// assert_eq!(CombineOp::from_wire(0x01), Some(CombineOp::Replace));
/// assert_eq!(CombineOp::from_wire(0x00), None);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CombineOp {
    /// Last-wins: the topmost (latest-emitted) member covering an index wins
    /// within its box, and the base shows through elsewhere. Wire byte `0x01`.
    Replace,
    /// The value at an index is the base value plus the sum of all covering
    /// corrections' values at that index. Wire byte `0x02`.
    Add,
}

impl CombineOp {
    /// Returns the wire byte for this combine operation.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::layout::CombineOp;
    ///
    /// assert_eq!(CombineOp::Replace.wire_byte(), 0x01);
    /// ```
    #[inline]
    pub fn wire_byte(self) -> u8 {
        match self {
            Self::Replace => 0x01,
            Self::Add => 0x02,
        }
    }

    /// Constructs a [`CombineOp`] from its wire byte. Returns `None` for any
    /// value other than `0x01` or `0x02`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::layout::CombineOp;
    ///
    /// assert_eq!(CombineOp::from_wire(0x02), Some(CombineOp::Add));
    /// assert_eq!(CombineOp::from_wire(0xFF), None);
    /// ```
    #[inline]
    pub fn from_wire(byte: u8) -> Option<Self> {
        match byte {
            0x01 => Some(Self::Replace),
            0x02 => Some(Self::Add),
            _ => None,
        }
    }
}

/// The composition rule declared by a composite head (`composition_rule` wire field).
///
/// Models `combine_op` as part of the rule (rather than a sibling field) so that
/// illegal combinations — e.g. a non-zero `combine_op` on a partition or group head —
/// are unrepresentable in this type: only [`CompositionRule::Overlay`] carries a
/// [`CombineOp`] at all.
///
/// # Examples
///
/// ```
/// use hurray_core::layout::{CombineOp, CompositionRule};
///
/// let partition = CompositionRule::Partition;
/// assert_eq!(partition.rule_byte(), 0x01);
/// assert_eq!(partition.combine_op_byte(), 0x00);
///
/// let overlay = CompositionRule::Overlay(CombineOp::Add);
/// assert_eq!(overlay.rule_byte(), 0x02);
/// assert_eq!(overlay.combine_op_byte(), 0x02);
///
/// assert_eq!(
///     CompositionRule::from_wire(0x02, 0x01).unwrap(),
///     CompositionRule::Overlay(CombineOp::Replace),
/// );
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CompositionRule {
    /// Exact-cover, non-overlapping tiling of the head's index space. Wire `0x01`.
    Partition,
    /// A base spanning the whole index space plus scattered corrections. Wire `0x02`.
    Overlay(CombineOp),
    /// Independent tensors under one head identity; no spatial semantics. Wire `0x03`.
    Group,
}

impl CompositionRule {
    /// Returns the wire `composition_rule` byte for this rule.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::layout::{CombineOp, CompositionRule};
    ///
    /// assert_eq!(CompositionRule::Group.rule_byte(), 0x03);
    /// assert_eq!(CompositionRule::Overlay(CombineOp::Replace).rule_byte(), 0x02);
    /// ```
    #[inline]
    pub fn rule_byte(&self) -> u8 {
        match self {
            Self::Partition => 0x01,
            Self::Overlay(_) => 0x02,
            Self::Group => 0x03,
        }
    }

    /// Returns the wire `combine_op` byte: `0x00` unless this is
    /// [`CompositionRule::Overlay`], in which case it is the wrapped
    /// [`CombineOp`]'s wire byte.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::layout::CompositionRule;
    ///
    /// assert_eq!(CompositionRule::Partition.combine_op_byte(), 0x00);
    /// ```
    #[inline]
    pub fn combine_op_byte(&self) -> u8 {
        match self {
            Self::Overlay(op) => op.wire_byte(),
            Self::Partition | Self::Group => 0x00,
        }
    }

    /// Constructs a [`CompositionRule`] from its wire `composition_rule` and
    /// `combine_op` bytes, validating both per spec § Head Layout-Specific Fields.
    ///
    /// The private-range rejection (`0xF0`–`0xFE`) mirrors
    /// [`crate::layout::is_private_tag`]'s layout-tag convention, for consistency
    /// across the crate's wire-byte taxonomies.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidCompositionRule`] — `rule == 0xFF` (permanently invalid).
    /// - [`Error::ReservedCompositionRule`] — `rule == 0x00` or in `0x04..=0xEF`.
    /// - [`Error::PrivateCompositionRule`] — `rule` is in `0xF0..=0xFE`.
    /// - [`Error::InvalidCombineOp`] — `combine_op` is not `0x00` for partition/group,
    ///   or not `0x01`/`0x02` for overlay.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{Error, layout::{CombineOp, CompositionRule}};
    ///
    /// assert_eq!(CompositionRule::from_wire(0x01, 0x00).unwrap(), CompositionRule::Partition);
    /// assert_eq!(CompositionRule::from_wire(0x03, 0x00).unwrap(), CompositionRule::Group);
    /// assert_eq!(
    ///     CompositionRule::from_wire(0x02, 0x02).unwrap(),
    ///     CompositionRule::Overlay(CombineOp::Add),
    /// );
    ///
    /// // 0x00 is reserved (not the permanently-invalid sentinel — that's 0xFF).
    /// assert!(matches!(
    ///     CompositionRule::from_wire(0x00, 0x00),
    ///     Err(Error::ReservedCompositionRule(0x00))
    /// ));
    /// assert!(matches!(
    ///     CompositionRule::from_wire(0xFF, 0x00),
    ///     Err(Error::InvalidCompositionRule(0xFF))
    /// ));
    /// // A non-zero combine_op on a partition head is rejected.
    /// assert!(CompositionRule::from_wire(0x01, 0x01).is_err());
    /// // combine_op outside {0x01, 0x02} on an overlay head is rejected.
    /// assert!(CompositionRule::from_wire(0x02, 0x03).is_err());
    /// ```
    pub fn from_wire(rule: u8, combine_op: u8) -> Result<Self> {
        // Every u8 value is classified by exactly one of these four arms
        // (0x00, 0x01-0x03, 0x04-0xEF, 0xF0-0xFE, 0xFF) — rustc proves this match
        // exhaustive without a wildcard, so none is added here.
        match rule {
            0xFF => return Err(Error::InvalidCompositionRule(rule)),
            0x00 | 0x04..=0xEF => return Err(Error::ReservedCompositionRule(rule)),
            0xF0..=0xFE => return Err(Error::PrivateCompositionRule(rule)),
            0x01..=0x03 => {}
        }

        match (rule, combine_op) {
            (0x01, 0x00) => Ok(Self::Partition),
            (0x03, 0x00) => Ok(Self::Group),
            (0x02, 0x01) => Ok(Self::Overlay(CombineOp::Replace)),
            (0x02, 0x02) => Ok(Self::Overlay(CombineOp::Add)),
            _ => Err(Error::InvalidCombineOp { rule, combine_op }),
        }
    }
}

/// The composite head's layout-specific payload (spec § Head Layout-Specific Fields).
///
/// Carried by [`crate::layout::LayoutDescriptor::Composite`]. See
/// `docs/spec/layouts/composite.md` and [`crate::composite`] for the full head +
/// members model and cross-member validation.
///
/// # Examples
///
/// ```
/// use hurray_core::layout::{CompositeLayout, CompositionRule, LayoutDescriptor};
///
/// let layout = CompositeLayout::new(CompositionRule::Partition, 2).unwrap();
/// let desc = LayoutDescriptor::Composite(layout);
/// assert_eq!(desc.tag(), 0x0B);
/// assert!(desc.buffer_count().is_none()); // "known zero" isn't NonZeroU8-representable
/// assert!(desc.is_virtual());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct CompositeLayout {
    /// The composition rule (`composition_rule` + `combine_op` wire fields).
    pub rule: CompositionRule,
    /// Number of member tensors that immediately follow the head. A definite count
    /// for every v1.0 composition rule; `0xFFFFFFFF` (open composite) is RESERVED.
    pub member_count: u32,
}

impl CompositeLayout {
    /// Creates a new [`CompositeLayout`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::OpenCompositeReserved`] if `member_count == 0xFFFFFFFF` (the
    /// open-composite sentinel, RESERVED and not usable in v1.0).
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::layout::{CompositeLayout, CompositionRule};
    ///
    /// let ok = CompositeLayout::new(CompositionRule::Group, 3).unwrap();
    /// assert_eq!(ok.member_count, 3);
    ///
    /// // The open-composite sentinel is reserved in v1.0.
    /// assert!(CompositeLayout::new(CompositionRule::Partition, 0xFFFF_FFFF).is_err());
    /// ```
    pub fn new(rule: CompositionRule, member_count: u32) -> Result<Self> {
        if member_count == OPEN_COMPOSITE_SENTINEL {
            return Err(Error::OpenCompositeReserved);
        }
        Ok(Self { rule, member_count })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::LayoutDescriptor;
    use crate::shape::DYNAMIC;
    use crate::Shape;

    // ── CombineOp wire round trip ───────────────────────────────────────────

    #[test]
    fn combine_op_wire_round_trip() {
        for op in [CombineOp::Replace, CombineOp::Add] {
            let byte = op.wire_byte();
            assert_eq!(CombineOp::from_wire(byte), Some(op));
        }
    }

    #[test]
    fn combine_op_wire_bytes_match_spec() {
        assert_eq!(CombineOp::Replace.wire_byte(), 0x01);
        assert_eq!(CombineOp::Add.wire_byte(), 0x02);
    }

    #[test]
    fn combine_op_from_wire_rejects_invalid_bytes() {
        for byte in [0x00_u8, 0x03, 0xFF] {
            assert!(
                CombineOp::from_wire(byte).is_none(),
                "0x{byte:02X} should not decode to a CombineOp"
            );
        }
    }

    // ── CompositionRule wire round trip ─────────────────────────────────────

    #[test]
    fn composition_rule_round_trip_partition() {
        let rule = CompositionRule::from_wire(0x01, 0x00).unwrap();
        assert_eq!(rule, CompositionRule::Partition);
        assert_eq!(rule.rule_byte(), 0x01);
        assert_eq!(rule.combine_op_byte(), 0x00);
    }

    #[test]
    fn composition_rule_round_trip_group() {
        let rule = CompositionRule::from_wire(0x03, 0x00).unwrap();
        assert_eq!(rule, CompositionRule::Group);
        assert_eq!(rule.rule_byte(), 0x03);
        assert_eq!(rule.combine_op_byte(), 0x00);
    }

    #[test]
    fn composition_rule_round_trip_overlay_replace() {
        let rule = CompositionRule::from_wire(0x02, 0x01).unwrap();
        assert_eq!(rule, CompositionRule::Overlay(CombineOp::Replace));
        assert_eq!(rule.rule_byte(), 0x02);
        assert_eq!(rule.combine_op_byte(), 0x01);
    }

    #[test]
    fn composition_rule_round_trip_overlay_add() {
        let rule = CompositionRule::from_wire(0x02, 0x02).unwrap();
        assert_eq!(rule, CompositionRule::Overlay(CombineOp::Add));
        assert_eq!(rule.rule_byte(), 0x02);
        assert_eq!(rule.combine_op_byte(), 0x02);
    }

    // ── CompositionRule::from_wire rejection ────────────────────────────────

    #[test]
    fn composition_rule_rejects_reserved_zero() {
        assert!(matches!(
            CompositionRule::from_wire(0x00, 0x00),
            Err(Error::ReservedCompositionRule(0x00))
        ));
    }

    #[test]
    fn composition_rule_rejects_invalid_0xff() {
        assert!(matches!(
            CompositionRule::from_wire(0xFF, 0x00),
            Err(Error::InvalidCompositionRule(0xFF))
        ));
    }

    #[test]
    fn composition_rule_rejects_reserved_range() {
        for rule in [0x04_u8, 0x50, 0xEF] {
            assert!(
                matches!(
                    CompositionRule::from_wire(rule, 0x00),
                    Err(Error::ReservedCompositionRule(_))
                ),
                "0x{rule:02X} should be ReservedCompositionRule"
            );
        }
    }

    #[test]
    fn composition_rule_rejects_private_range() {
        for rule in [0xF0_u8, 0xF7, 0xFE] {
            assert!(
                matches!(
                    CompositionRule::from_wire(rule, 0x00),
                    Err(Error::PrivateCompositionRule(_))
                ),
                "0x{rule:02X} should be PrivateCompositionRule"
            );
        }
    }

    #[test]
    fn composition_rule_rejects_nonzero_combine_op_for_partition() {
        assert!(matches!(
            CompositionRule::from_wire(0x01, 0x01),
            Err(Error::InvalidCombineOp {
                rule: 0x01,
                combine_op: 0x01
            })
        ));
    }

    #[test]
    fn composition_rule_rejects_nonzero_combine_op_for_group() {
        assert!(matches!(
            CompositionRule::from_wire(0x03, 0x02),
            Err(Error::InvalidCombineOp {
                rule: 0x03,
                combine_op: 0x02
            })
        ));
    }

    #[test]
    fn composition_rule_rejects_combine_op_outside_replace_add_for_overlay() {
        for combine_op in [0x00_u8, 0x03, 0xFF] {
            assert!(
                matches!(
                    CompositionRule::from_wire(0x02, combine_op),
                    Err(Error::InvalidCombineOp { rule: 0x02, .. })
                ),
                "combine_op 0x{combine_op:02X} should be rejected for overlay"
            );
        }
    }

    // ── CompositeLayout::new ────────────────────────────────────────────────

    #[test]
    fn composite_layout_new_rejects_open_composite_sentinel() {
        assert!(matches!(
            CompositeLayout::new(CompositionRule::Partition, OPEN_COMPOSITE_SENTINEL),
            Err(Error::OpenCompositeReserved)
        ));
    }

    #[test]
    fn composite_layout_new_accepts_definite_counts() {
        let layout = CompositeLayout::new(CompositionRule::Group, 0).unwrap();
        assert_eq!(layout.member_count, 0);
        let layout =
            CompositeLayout::new(CompositionRule::Partition, OPEN_COMPOSITE_SENTINEL - 1).unwrap();
        assert_eq!(layout.member_count, OPEN_COMPOSITE_SENTINEL - 1);
    }

    // ── LayoutDescriptor::Composite integration ─────────────────────────────

    #[test]
    fn composite_layout_descriptor_tag_is_0x0b() {
        let layout =
            LayoutDescriptor::Composite(CompositeLayout::new(CompositionRule::Group, 0).unwrap());
        assert_eq!(layout.tag(), 0x0B);
    }

    #[test]
    fn composite_layout_descriptor_buffer_count_is_none() {
        let layout = LayoutDescriptor::Composite(
            CompositeLayout::new(CompositionRule::Partition, 2).unwrap(),
        );
        assert!(layout.buffer_count().is_none());
    }

    #[test]
    fn composite_layout_descriptor_is_virtual() {
        let composite =
            LayoutDescriptor::Composite(CompositeLayout::new(CompositionRule::Group, 0).unwrap());
        assert!(composite.is_virtual());
        // A non-composite layout must NOT be reported virtual — proves the
        // predicate actually discriminates rather than trivially returning true.
        assert!(!LayoutDescriptor::RowMajor.is_virtual());
    }

    #[test]
    fn composite_tag_is_not_reserved() {
        assert!(!crate::layout::is_reserved_tag(0x0B));
    }

    #[test]
    fn composite_tag_passes_strict_validation() {
        assert!(crate::layout::validate_layout_tag_strict(0x0B).is_ok());
    }

    #[test]
    fn composite_element_offset_returns_layout_is_virtual() {
        let layout =
            LayoutDescriptor::Composite(CompositeLayout::new(CompositionRule::Group, 0).unwrap());
        let shape = Shape::new(vec![4u64]).unwrap();
        let err = layout.element_offset(&[0], &shape).unwrap_err();
        assert!(matches!(err, Error::LayoutIsVirtual { layout_tag: 0x0B }));
    }

    // ── validate_against_shape: DYNAMIC head shape rejection ────────────────

    #[test]
    fn composite_validate_against_shape_rejects_dynamic_for_partition() {
        let layout = LayoutDescriptor::Composite(
            CompositeLayout::new(CompositionRule::Partition, 0).unwrap(),
        );
        let shape = Shape::new(vec![DYNAMIC, 4]).unwrap();
        assert!(matches!(
            layout.validate_against_shape(&shape),
            Err(Error::InvalidLayout(_))
        ));
    }

    #[test]
    fn composite_validate_against_shape_rejects_dynamic_for_overlay() {
        let layout = LayoutDescriptor::Composite(
            CompositeLayout::new(CompositionRule::Overlay(CombineOp::Replace), 0).unwrap(),
        );
        let shape = Shape::new(vec![DYNAMIC]).unwrap();
        assert!(matches!(
            layout.validate_against_shape(&shape),
            Err(Error::InvalidLayout(_))
        ));
    }

    #[test]
    fn composite_validate_against_shape_accepts_dynamic_for_group() {
        // Group has no spatial semantics, so a DYNAMIC head dimension is fine.
        let layout =
            LayoutDescriptor::Composite(CompositeLayout::new(CompositionRule::Group, 0).unwrap());
        let shape = Shape::new(vec![DYNAMIC]).unwrap();
        assert!(layout.validate_against_shape(&shape).is_ok());
    }

    #[test]
    fn composite_validate_against_shape_accepts_static_shape_for_partition() {
        let layout = LayoutDescriptor::Composite(
            CompositeLayout::new(CompositionRule::Partition, 0).unwrap(),
        );
        let shape = Shape::new(vec![4u64, 4]).unwrap();
        assert!(layout.validate_against_shape(&shape).is_ok());
    }
}
