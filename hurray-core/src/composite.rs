//! Composite tensor: head + members, aggregated across whole `TensorDescriptor`s.
//!
//! Every other layout in this crate is validated and addressed within a single
//! [`crate::descriptor::TensorDescriptor`]. Composite is different: a composite
//! "tensor" is a **head** descriptor (`layout_tag = 0x0B`, see
//! [`crate::layout::composite`]) plus **N** ordinary tensor descriptors — its
//! **members** — bound by forward stream/file adjacency. This module is where that
//! cross-descriptor aggregate lives.
//!
//! Distinct from [`crate::layout::composite`], which defines only the wire-level
//! `CompositeLayout` payload carried *inside* the head descriptor's layout-specific
//! fields (`composition_rule`, `combine_op`, `member_count`). This module aggregates
//! a complete head plus its actual member descriptors and validates them against
//! each other per `docs/spec/layouts/composite.md` § Validation. The two modules are
//! kept at distinct paths (`hurray_core::layout::composite` vs
//! `hurray_core::composite`) precisely so their similarly-named types — `CompositeLayout`
//! / `CompositionRule` / `CombineOp` here vs [`CompositeTensor`] / [`CompositeValidator`]
//! there — never collide in any glob or re-export.

use crate::descriptor::TensorDescriptor;
use crate::descriptor::{CompositeMemberDescriptor, MemberRole};
use crate::layout::{CompositionRule, LayoutDescriptor};
use crate::{Error, Result};

/// Maximum composite nesting depth.
///
/// Mirrors the recursion-depth discipline used for nested Tiled layouts
/// ([`crate::layout::MAX_TILED_DEPTH`]), per `docs/spec/layouts/composite.md`
/// § Binding: "a reader MUST enforce a maximum composite nesting depth of 8
/// levels." A member whose own `layout_tag` is the composite tag counts as one
/// additional level.
pub const MAX_COMPOSITE_DEPTH: u8 = 8;

/// A composite tensor: a head descriptor plus its ordered member descriptors.
///
/// Construct via [`CompositeTensor::new`], which drives a [`CompositeValidator`]
/// across every member and rejects a malformed head+members set — see that type's
/// docs for the full per-member and close-time checks performed.
///
/// # Nesting depth — a note on this layer's limitation
///
/// A member here is a flat [`TensorDescriptor`], not a further-expanded
/// [`CompositeTensor`]. If a member's own `layout_tag` happens to be the composite
/// tag, [`CompositeTensor::new`] counts that as one additional nesting level for
/// the depth cap, but it cannot recurse into *that* member's own members — they
/// are not available at this layer (Layer 4 sees single descriptors, not a
/// pre-assembled tree). Full nesting-depth validation therefore cannot be
/// completed until a higher layer (Layer 5 streaming or Layer 6 file, which do
/// assemble the full nested tree as they parse it) walks the tree top to bottom.
/// [`CompositeTensor::new`] is depth `0`; a `pub(crate)` depth-aware constructor
/// is provided for that future recursive caller.
///
/// # Examples
///
/// ```
/// use hurray_core::{
///     composite::CompositeTensor,
///     descriptor::TensorDescriptor,
///     layout::{CompositeLayout, CompositionRule, LayoutDescriptor},
///     BufferHandle, DeviceTag, ElementType, Shape, ShardDescriptor, SyncMode,
///     MIN_BUFFER_ALIGNMENT,
/// };
///
/// // Head: float32 [8, 8], partition of 2 members.
/// let head = TensorDescriptor::new(
///     1, 0, ElementType::Float32, Shape::new(vec![8u64, 8]).unwrap(), 0,
///     LayoutDescriptor::Composite(CompositeLayout::new(CompositionRule::Partition, 2).unwrap()),
///     vec![],
///     None, None, None, None,
/// ).unwrap();
///
/// // Two [8, 4] members tiling the head's [8, 8] index space.
/// let member = |offset: u64| {
///     let buf = BufferHandle::new(128, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap();
///     let shard = ShardDescriptor::new(vec![8, 8], vec![0, offset]).unwrap();
///     TensorDescriptor::new(
///         1, 0, ElementType::Float32, Shape::new(vec![8u64, 4]).unwrap(), 0,
///         LayoutDescriptor::RowMajor, vec![buf],
///         None, Some(shard), None, None,
///     ).unwrap()
/// };
///
/// let composite = CompositeTensor::new(head, vec![member(0), member(4)]).unwrap();
/// assert_eq!(composite.member_count(), 2);
/// assert_eq!(*composite.rule(), CompositionRule::Partition);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CompositeTensor {
    /// The composite head descriptor (`layout_tag = 0x0B`).
    pub head: TensorDescriptor,
    /// The ordered member descriptors bound to the head.
    pub members: Vec<TensorDescriptor>,
}

impl CompositeTensor {
    /// Creates a new [`CompositeTensor`], validating `members` against `head`
    /// via [`CompositeValidator`] (per-member checks, then close-time checks).
    ///
    /// Equivalent to `CompositeTensor::new_at_depth(head, members, 0)`.
    ///
    /// # Errors
    ///
    /// Propagates any [`CompositeValidator::push_member`] / [`CompositeValidator::finish`]
    /// error, plus [`Error::InvalidLayout`] if `head.layout` is not
    /// [`LayoutDescriptor::Composite`].
    ///
    /// # Examples
    ///
    /// See the module-level and struct-level examples.
    pub fn new(head: TensorDescriptor, members: Vec<TensorDescriptor>) -> Result<Self> {
        Self::new_at_depth(head, members, 0)
    }

    /// Depth-aware constructor for recursive assembly by a higher layer.
    ///
    /// `depth` is the nesting depth of `head` itself (the top-level call from
    /// [`CompositeTensor::new`] uses `0`). A future Layer 5/6 caller that has
    /// assembled a full nested composite tree can recurse with `depth + 1` for
    /// each member that is itself a composite head — this layer cannot do that
    /// recursion itself (see the struct-level docs), but it does track and
    /// enforce the depth budget for the one level of nesting it *can* see: a
    /// member whose own `layout_tag` is the composite tag.
    pub(crate) fn new_at_depth(
        head: TensorDescriptor,
        members: Vec<TensorDescriptor>,
        depth: u8,
    ) -> Result<Self> {
        if depth >= MAX_COMPOSITE_DEPTH {
            return Err(Error::SubpavingNestingTooDeep);
        }

        let mut validator = CompositeValidator::new(&head)?;
        for member in &members {
            if matches!(member.layout, LayoutDescriptor::Composite(_))
                && depth + 1 >= MAX_COMPOSITE_DEPTH
            {
                return Err(Error::SubpavingNestingTooDeep);
            }
            validator.push_member(member)?;
        }
        validator.finish()?;

        Ok(Self { head, members })
    }

    /// Returns the composition rule declared by the head.
    ///
    /// # Examples
    ///
    /// See the struct-level example.
    pub fn rule(&self) -> &CompositionRule {
        match &self.head.layout {
            LayoutDescriptor::Composite(c) => &c.rule,
            // INVARIANT: CompositeTensor::new only returns Ok when head.layout is
            // Composite (CompositeValidator::new rejects any other layout first).
            // Struct fields are `pub` (matching TensorDescriptor's own convention
            // elsewhere in this crate), so a caller COULD hand-build an invalid
            // instance bypassing `new()`; this is the same trust boundary the rest
            // of the crate already accepts for its `pub`-field descriptor types.
            _ => unreachable!(
                "CompositeTensor invariant: head.layout is always LayoutDescriptor::Composite"
            ),
        }
    }

    /// Returns the member count declared by the head.
    ///
    /// # Examples
    ///
    /// See the struct-level example.
    pub fn member_count(&self) -> u32 {
        match &self.head.layout {
            LayoutDescriptor::Composite(c) => c.member_count,
            _ => unreachable!(
                "CompositeTensor invariant: head.layout is always LayoutDescriptor::Composite"
            ),
        }
    }
}

/// Stateful, bounded validator for a composite head's members.
///
/// Drive it with [`CompositeValidator::new`] (extracts the composition rule from
/// the head), then [`CompositeValidator::push_member`] once per member in stream
/// order, then [`CompositeValidator::finish`] to run the close-time checks. This
/// mirrors `docs/spec/layouts/composite.md` § Validation exactly: "a reader
/// accumulates state only up to N, reaches one verdict at the Nth member, and is
/// done" — every v1.0 composition rule uses a definite `member_count`.
///
/// [`CompositeTensor::new`] drives one of these internally; construct one
/// directly only if you need to validate members incrementally as they arrive
/// (e.g. a future streaming reader) rather than from an already-collected `Vec`.
///
/// # Decoded-type checking and quantized members
///
/// Per-member check 2 (member decoded type == head `type_tag`) is only fully
/// checkable for **unquantized** members at this layer: [`TensorDescriptor`]
/// stores quantization as raw `Option<Vec<u8>>` bytes with no typed schema at
/// Layer 4 (see its `quantization` field docs), so a quantized member's decoded
/// type cannot be resolved here. When a member carries a quantization section,
/// this check is skipped; verifying that its dequantized type agrees with the
/// head is deferred to a higher layer that has quantization-scheme context.
///
/// # Examples
///
/// ```
/// use hurray_core::{
///     composite::CompositeValidator,
///     descriptor::TensorDescriptor,
///     layout::{CompositeLayout, CompositionRule, LayoutDescriptor},
///     BufferHandle, DeviceTag, ElementType, Shape, SyncMode, MIN_BUFFER_ALIGNMENT,
/// };
///
/// // A group composite: members need no shard section and may differ arbitrarily.
/// let head = TensorDescriptor::new(
///     1, 0, ElementType::Float32, Shape::new(vec![1u64]).unwrap(), 0,
///     LayoutDescriptor::Composite(CompositeLayout::new(CompositionRule::Group, 1).unwrap()),
///     vec![],
///     None, None, None, None,
/// ).unwrap();
///
/// let buf = BufferHandle::new(4, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap();
/// let member = TensorDescriptor::new(
///     1, 0, ElementType::Int8, Shape::new(vec![100u64]).unwrap(), 0,
///     LayoutDescriptor::RowMajor, vec![buf],
///     None, None, None, None,
/// ).unwrap();
///
/// let mut validator = CompositeValidator::new(&head).unwrap();
/// validator.push_member(&member).unwrap();
/// assert!(validator.finish().is_ok());
/// ```
pub struct CompositeValidator {
    rule: CompositionRule,
    declared_member_count: u32,
    head_shape_dims: Vec<u64>,
    head_type_tag: u8,
    members_seen: usize,
    /// Accumulated `(shard_offset, shape)` boxes for the partition coverage check.
    partition_boxes: Vec<(Vec<u64>, Vec<u64>)>,
}

impl CompositeValidator {
    /// Creates a validator seeded from `head`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidLayout`] if `head.layout` is not
    /// [`LayoutDescriptor::Composite`], or if `head.shape` contains a `DYNAMIC`
    /// dimension while the rule is partition or overlay — coverage/volume math
    /// is undefined for a dynamic shape, so both rules require a fully static
    /// head. This is the same check [`LayoutDescriptor::validate_against_shape`]
    /// performs for a composite layout; re-checked here because that method is
    /// opt-in and this validator is the layer that actually gates composite
    /// assembly.
    ///
    /// # Examples
    ///
    /// See the type-level example.
    pub fn new(head: &TensorDescriptor) -> Result<Self> {
        let composite_layout = match &head.layout {
            LayoutDescriptor::Composite(c) => c,
            other => {
                return Err(Error::InvalidLayout(format!(
                    "composite head's layout_tag must be 0x0B (Composite), got 0x{:02X}",
                    other.tag()
                )))
            }
        };

        let rule = composite_layout.rule.clone();
        if !matches!(rule, CompositionRule::Group) && head.shape.has_dynamic() {
            return Err(Error::InvalidLayout(
                "partition/overlay composite head shape MUST NOT contain a DYNAMIC dimension \
                 (coverage/volume math requires a fully static shape)"
                    .to_string(),
            ));
        }

        Ok(Self {
            rule,
            declared_member_count: composite_layout.member_count,
            head_shape_dims: head.shape.dims().to_vec(),
            head_type_tag: head.element_type.tag(),
            members_seen: 0,
            partition_boxes: Vec::new(),
        })
    }

    /// Runs the per-member checks (spec § Validation › Per-member checks) for one
    /// member, in the order it arrives.
    ///
    /// # Errors
    ///
    /// See the crate [`Error`] variants prefixed `Composite`, plus
    /// [`Error::ShardOutOfBounds`] (reused from ADR-004 for the box-in-bounds check).
    ///
    /// # Examples
    ///
    /// See the type-level example.
    pub fn push_member(&mut self, member: &TensorDescriptor) -> Result<()> {
        let index = self.members_seen;
        let rule = self.rule.clone();

        match rule {
            CompositionRule::Partition => {
                let shard = self.shard_box(member, index)?;
                self.check_member_type(member, index)?;
                self.reject_composite_member_flag(member)?;
                self.partition_boxes
                    .push((shard.shard_offset.clone(), member.shape.dims().to_vec()));
            }
            CompositionRule::Overlay(_) => {
                self.shard_box(member, index)?;
                self.check_member_type(member, index)?;
                self.check_overlay_member_role(member, index)?;
            }
            CompositionRule::Group => {
                // Group: no shard section required, no decoded-type check (spec §
                // Composition Semantics › Group: members MAY differ arbitrarily).
                self.reject_composite_member_flag(member)?;
            }
        }

        self.members_seen += 1;
        Ok(())
    }

    /// Runs the close-time checks (spec § Validation › Close-time checks) once
    /// every member has been pushed.
    ///
    /// # Errors
    ///
    /// - [`Error::CompositeMemberCountMismatch`] — the number of members pushed
    ///   does not equal the head's declared `member_count`.
    /// - [`Error::CompositeOverlayEmpty`] — an overlay composite received zero members.
    /// - [`Error::CompositePartitionGap`] / [`Error::CompositePartitionOverlap`] —
    ///   partition members do not exactly cover the head's index space.
    ///
    /// # Examples
    ///
    /// See the type-level example.
    pub fn finish(self) -> Result<()> {
        if self.members_seen as u32 != self.declared_member_count {
            return Err(Error::CompositeMemberCountMismatch {
                declared: self.declared_member_count,
                actual: self.members_seen,
            });
        }

        match self.rule {
            CompositionRule::Partition => self.validate_partition_coverage(),
            CompositionRule::Overlay(_) => {
                // The base-span / base-first check already ran per-member (check 4)
                // for whatever members did arrive; a zero-member overlay is a
                // separate case — it never had a base to check in the first place.
                if self.members_seen == 0 {
                    return Err(Error::CompositeOverlayEmpty);
                }
                Ok(())
            }
            CompositionRule::Group => Ok(()),
        }
    }

    // ── Per-member check helpers ─────────────────────────────────────────────

    /// Check 1: `member` carries a shard section whose `parent_shape` equals the
    /// head's shape and whose box is in bounds. Returns the shard on success so
    /// callers needn't re-borrow `member.shard`.
    fn shard_box<'a>(
        &self,
        member: &'a TensorDescriptor,
        index: usize,
    ) -> Result<&'a crate::descriptor::ShardDescriptor> {
        let shard = member
            .shard
            .as_ref()
            .ok_or(Error::CompositeMemberMissingShard { index })?;
        if shard.parent_shape.as_slice() != self.head_shape_dims.as_slice() {
            return Err(Error::CompositeMemberParentShapeMismatch { index });
        }
        // Box-in-bounds: shard_offset[k] + shape[k] <= parent_shape[k]. Reused
        // verbatim from ADR-004 rather than a bespoke composite variant — same
        // constraint, same failure semantics.
        shard.validate_against_shape(&member.shape)?;
        Ok(shard)
    }

    /// Check 2: `member`'s decoded value type equals the head's `type_tag`.
    /// Skipped for quantized members — see the type-level docs' "Decoded-type
    /// checking" section for why.
    fn check_member_type(&self, member: &TensorDescriptor, index: usize) -> Result<()> {
        if member.quantization.is_some() {
            return Ok(());
        }
        let member_tag = member.element_type.tag();
        if member_tag != self.head_type_tag {
            return Err(Error::CompositeMemberTypeMismatch {
                index,
                member: member_tag,
                head: self.head_type_tag,
            });
        }
        Ok(())
    }

    /// Check 4 (overlay only): `member` carries a Composite Member section with a
    /// valid role; the first member MUST be the base and span the index space;
    /// every subsequent member MUST be a correction.
    fn check_overlay_member_role(&self, member: &TensorDescriptor, index: usize) -> Result<()> {
        let cm: &CompositeMemberDescriptor =
            member
                .composite_member
                .as_ref()
                .ok_or(Error::CompositeMemberFlagMismatch {
                    rule: self.rule.rule_byte(),
                    has_flag: false,
                })?;

        if index == 0 {
            if cm.member_role != MemberRole::Base {
                return Err(Error::CompositeOverlayBaseNotFirst);
            }
            // Presence of `shard` is already guaranteed by the `shard_box` call
            // that precedes this one in `push_member`.
            let shard = member
                .shard
                .as_ref()
                .ok_or(Error::CompositeMemberMissingShard { index })?;
            let spans_fully = shard.shard_offset.iter().all(|&o| o == 0)
                && member.shape.dims() == self.head_shape_dims.as_slice();
            if !spans_fully {
                return Err(Error::CompositeOverlayBaseNotSpanning);
            }
        } else if cm.member_role != MemberRole::Correction {
            // Only the first member may be the base; reusing the same error keeps
            // the "base MUST be first" framing symmetric for both directions of
            // the violation (missing at position 0, or appearing elsewhere).
            return Err(Error::CompositeOverlayBaseNotFirst);
        }

        Ok(())
    }

    /// Check 5 (partition/group): `member` MUST NOT carry a Composite Member section.
    fn reject_composite_member_flag(&self, member: &TensorDescriptor) -> Result<()> {
        if member.composite_member.is_some() {
            return Err(Error::CompositeMemberFlagMismatch {
                rule: self.rule.rule_byte(),
                has_flag: true,
            });
        }
        Ok(())
    }

    // ── Close-time check helpers ──────────────────────────────────────────────

    /// Partition close-time check: exact-cover + non-overlap over the accumulated
    /// member boxes (spec § Composition Semantics › Partition).
    ///
    /// Mirrors the volume-sum + pairwise-overlap algorithm this crate previously
    /// used for subpaving coverage validation (see git history:
    /// `fix(hurray-core): enforce subpaving full-coverage validation`): pairwise
    /// non-overlap is checked first (O(n²) over members; not a hot path), then —
    /// only once no overlap was found — coverage is checked via volume sum. Given
    /// every box is already confirmed in-bounds (check 1) and no two overlap, a
    /// volume-sum shortfall can only mean a gap, never an overlap masquerading as
    /// one; checking overlap first and gap second keeps the two error variants
    /// unambiguous rather than defaulting everything to "gap".
    fn validate_partition_coverage(&self) -> Result<()> {
        let n = self.partition_boxes.len();
        for i in 0..n {
            for j in (i + 1)..n {
                if boxes_overlap(&self.partition_boxes[i], &self.partition_boxes[j]) {
                    return Err(Error::CompositePartitionOverlap { a: i, b: j });
                }
            }
        }

        // Full-coverage via volume sum. Skipped (accepted without further check) if
        // any dimension is dynamic or a product overflows u64 — the total can't then
        // be computed reliably, and CompositeValidator::new already rejects DYNAMIC
        // head shapes for partition, so this branch is unreachable in practice; kept
        // as a defensive fallback matching the subpaving precedent.
        let total = self.head_shape_dims.iter().try_fold(1u64, |acc, &d| {
            if d == crate::shape::DYNAMIC {
                None
            } else {
                acc.checked_mul(d)
            }
        });
        if let Some(total) = total {
            let mut covered: Option<u64> = Some(0);
            for (_, member_shape) in &self.partition_boxes {
                let vol = member_shape.iter().try_fold(1u64, |acc, &d| {
                    if d == crate::shape::DYNAMIC {
                        None
                    } else {
                        acc.checked_mul(d)
                    }
                });
                covered = match (covered, vol) {
                    (Some(c), Some(v)) => c.checked_add(v),
                    _ => None,
                };
            }
            if let Some(covered) = covered {
                if covered != total {
                    return Err(Error::CompositePartitionGap);
                }
            }
        }

        Ok(())
    }
}

/// Returns `true` if boxes `a` and `b` (each `(shard_offset, shape)`) overlap.
///
/// Per spec § Composition Semantics › Partition: `A` and `B` overlap if, for
/// every dimension `k`, `A.offset[k] < B.offset[k] + B.shape[k]` AND
/// `B.offset[k] < A.offset[k] + A.shape[k]`. Uses `saturating_add` so a
/// pathological (already-rejected-elsewhere) huge offset/shape pair cannot wrap
/// around and produce a false negative.
fn boxes_overlap(a: &(Vec<u64>, Vec<u64>), b: &(Vec<u64>, Vec<u64>)) -> bool {
    let (a_off, a_shape) = a;
    let (b_off, b_shape) = b;
    a_off
        .iter()
        .zip(a_shape)
        .zip(b_off.iter().zip(b_shape))
        .all(|((&ao, &asz), (&bo, &bsz))| {
            ao < bo.saturating_add(bsz) && bo < ao.saturating_add(asz)
        })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{CombineOp, CompositeLayout};
    use crate::{BufferHandle, DeviceTag, Error, Shape, ShardDescriptor, SyncMode};

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn buf(byte_size: u64) -> BufferHandle {
        BufferHandle::new(
            byte_size,
            crate::MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )
        .unwrap()
    }

    fn composite_head(
        rule: CompositionRule,
        member_count: u32,
        shape: Vec<u64>,
        ty: crate::ElementType,
    ) -> TensorDescriptor {
        let layout = LayoutDescriptor::Composite(CompositeLayout::new(rule, member_count).unwrap());
        TensorDescriptor::new(
            1,
            0,
            ty,
            Shape::new(shape).unwrap(),
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

    /// A plain dense (row-major) member with an optional shard section. Uses a
    /// fixed 64-byte buffer — the byte size is irrelevant to composite
    /// validation, which only inspects shape/type/shard/flags.
    fn dense_member(
        ty: crate::ElementType,
        shape: Vec<u64>,
        shard: Option<ShardDescriptor>,
    ) -> TensorDescriptor {
        TensorDescriptor::new(
            1,
            0,
            ty,
            Shape::new(shape).unwrap(),
            0,
            LayoutDescriptor::RowMajor,
            vec![buf(64)],
            None,
            shard,
            None,
            None,
        )
        .unwrap()
    }

    fn shard(parent: Vec<u64>, offset: Vec<u64>) -> ShardDescriptor {
        ShardDescriptor::new(parent, offset).unwrap()
    }

    // ── Partition ────────────────────────────────────────────────────────────

    /// Spec worked example: [8, 8] head split into two [8, 4] members.
    #[test]
    fn partition_exact_tiling_accepted() {
        let head = composite_head(
            CompositionRule::Partition,
            2,
            vec![8, 8],
            crate::ElementType::Float32,
        );
        let m0 = dense_member(
            crate::ElementType::Float32,
            vec![8, 4],
            Some(shard(vec![8, 8], vec![0, 0])),
        );
        let m1 = dense_member(
            crate::ElementType::Float32,
            vec![8, 4],
            Some(shard(vec![8, 8], vec![0, 4])),
        );
        let composite = CompositeTensor::new(head, vec![m0, m1]).unwrap();
        assert_eq!(composite.member_count(), 2);
        assert_eq!(*composite.rule(), CompositionRule::Partition);
    }

    /// A 4-way tiling of an [8, 8] head into four [4, 4] quadrants.
    #[test]
    fn partition_four_way_tiling_accepted() {
        let head = composite_head(
            CompositionRule::Partition,
            4,
            vec![8, 8],
            crate::ElementType::Float32,
        );
        let members = [(0u64, 0u64), (0, 4), (4, 0), (4, 4)]
            .into_iter()
            .map(|(r, c)| {
                dense_member(
                    crate::ElementType::Float32,
                    vec![4, 4],
                    Some(shard(vec![8, 8], vec![r, c])),
                )
            })
            .collect::<Vec<_>>();
        let composite = CompositeTensor::new(head, members).unwrap();
        assert_eq!(composite.member_count(), 4);
    }

    /// Members that leave a gap in the head's index space are rejected.
    #[test]
    fn partition_gap_rejected() {
        let head = composite_head(
            CompositionRule::Partition,
            2,
            vec![8, 8],
            crate::ElementType::Float32,
        );
        // Non-overlapping boxes covering columns [0,3) and [3,6): columns
        // [6, 8) are left uncovered — a genuine gap with no overlap (overlap is
        // checked before coverage; see validate_partition_coverage's doc comment,
        // so a gap-only fixture is needed to isolate this error variant).
        let m0 = dense_member(
            crate::ElementType::Float32,
            vec![8, 3],
            Some(shard(vec![8, 8], vec![0, 0])),
        );
        let m1 = dense_member(
            crate::ElementType::Float32,
            vec![8, 3],
            Some(shard(vec![8, 8], vec![0, 3])),
        );
        let err = CompositeTensor::new(head, vec![m0, m1]).unwrap_err();
        assert!(matches!(err, Error::CompositePartitionGap));
    }

    /// Two overlapping member boxes are rejected with the offending indices.
    #[test]
    fn partition_overlap_rejected() {
        let head = composite_head(
            CompositionRule::Partition,
            2,
            vec![8, 8],
            crate::ElementType::Float32,
        );
        let m0 = dense_member(
            crate::ElementType::Float32,
            vec![8, 5],
            Some(shard(vec![8, 8], vec![0, 0])),
        );
        let m1 = dense_member(
            crate::ElementType::Float32,
            vec![8, 5],
            Some(shard(vec![8, 8], vec![0, 3])),
        );
        let err = CompositeTensor::new(head, vec![m0, m1]).unwrap_err();
        assert!(matches!(
            err,
            Error::CompositePartitionOverlap { a: 0, b: 1 }
        ));
    }

    /// A partition member without a shard section is rejected.
    #[test]
    fn partition_member_missing_shard_rejected() {
        let head = composite_head(
            CompositionRule::Partition,
            1,
            vec![8, 8],
            crate::ElementType::Float32,
        );
        let m0 = dense_member(crate::ElementType::Float32, vec![8, 8], None);
        let err = CompositeTensor::new(head, vec![m0]).unwrap_err();
        assert!(matches!(
            err,
            Error::CompositeMemberMissingShard { index: 0 }
        ));
    }

    /// A member's shard `parent_shape` must equal the head's shape.
    #[test]
    fn partition_member_parent_shape_mismatch_rejected() {
        let head = composite_head(
            CompositionRule::Partition,
            1,
            vec![8, 8],
            crate::ElementType::Float32,
        );
        // parent_shape [10, 10] != head shape [8, 8].
        let m0 = dense_member(
            crate::ElementType::Float32,
            vec![8, 8],
            Some(shard(vec![10, 10], vec![0, 0])),
        );
        let err = CompositeTensor::new(head, vec![m0]).unwrap_err();
        assert!(matches!(
            err,
            Error::CompositeMemberParentShapeMismatch { index: 0 }
        ));
    }

    /// A member box that extends past the head's index space is rejected with
    /// the reused ADR-004 `ShardOutOfBounds` variant (not a bespoke composite one).
    #[test]
    fn partition_member_out_of_bounds_rejected() {
        let head = composite_head(
            CompositionRule::Partition,
            1,
            vec![8, 8],
            crate::ElementType::Float32,
        );
        // shard_offset[1]=4 + shape[1]=8 = 12 > parent_shape[1]=8.
        let m0 = dense_member(
            crate::ElementType::Float32,
            vec![8, 8],
            Some(shard(vec![8, 8], vec![0, 4])),
        );
        let err = CompositeTensor::new(head, vec![m0]).unwrap_err();
        assert!(matches!(err, Error::ShardOutOfBounds { .. }));
    }

    /// An unquantized member whose decoded type disagrees with the head's
    /// `type_tag` is rejected.
    #[test]
    fn partition_member_type_mismatch_rejected() {
        let head = composite_head(
            CompositionRule::Partition,
            1,
            vec![8, 8],
            crate::ElementType::Float32,
        );
        let m0 = dense_member(
            crate::ElementType::Int8,
            vec![8, 8],
            Some(shard(vec![8, 8], vec![0, 0])),
        );
        let err = CompositeTensor::new(head, vec![m0]).unwrap_err();
        assert!(matches!(
            err,
            Error::CompositeMemberTypeMismatch { index: 0, .. }
        ));
    }

    /// A *quantized* member's "mismatched" stored type is NOT rejected on that
    /// basis alone — the decoded-type check is skipped when a quantization
    /// section is present (Layer 4 has no typed quantization schema to resolve
    /// the decoded type against; see the carve-out documented on
    /// [`CompositeValidator::push_member`]).
    #[test]
    fn partition_quantized_member_type_mismatch_not_rejected() {
        let head = composite_head(
            CompositionRule::Partition,
            1,
            vec![8, 8],
            crate::ElementType::Float16,
        );
        // Stored type int4 disagrees with the head's float16 — normally rejected,
        // but quantization is Some(_), so the decoded-type check is skipped.
        let m0 = TensorDescriptor::new(
            1,
            0,
            crate::ElementType::Int4,
            Shape::new(vec![8u64, 8]).unwrap(),
            0,
            LayoutDescriptor::RowMajor,
            vec![buf(64)],
            Some(vec![0x01, 0x00, 0x00, 0x00]),
            Some(shard(vec![8, 8], vec![0, 0])),
            None,
            None,
        )
        .unwrap();
        let composite = CompositeTensor::new(head, vec![m0]).unwrap();
        assert_eq!(composite.member_count(), 1);
    }

    /// A partition member MUST NOT carry a Composite Member section.
    #[test]
    fn partition_member_illegal_composite_member_flag_rejected() {
        let head = composite_head(
            CompositionRule::Partition,
            1,
            vec![8, 8],
            crate::ElementType::Float32,
        );
        let m0 = dense_member(
            crate::ElementType::Float32,
            vec![8, 8],
            Some(shard(vec![8, 8], vec![0, 0])),
        )
        .with_composite_member(CompositeMemberDescriptor::new(MemberRole::Base));
        let err = CompositeTensor::new(head, vec![m0]).unwrap_err();
        assert!(matches!(
            err,
            Error::CompositeMemberFlagMismatch { has_flag: true, .. }
        ));
    }

    /// Declared `member_count` not matching the number of members actually
    /// pushed is rejected at close time.
    #[test]
    fn partition_member_count_mismatch_rejected() {
        let head = composite_head(
            CompositionRule::Partition,
            2,
            vec![8, 8],
            crate::ElementType::Float32,
        );
        let m0 = dense_member(
            crate::ElementType::Float32,
            vec![8, 8],
            Some(shard(vec![8, 8], vec![0, 0])),
        );
        let err = CompositeTensor::new(head, vec![m0]).unwrap_err();
        assert!(matches!(
            err,
            Error::CompositeMemberCountMismatch {
                declared: 2,
                actual: 1,
            }
        ));
    }

    /// A DYNAMIC-shaped head is rejected end-to-end through the public entry
    /// point ([`CompositeTensor::new`] / [`CompositeValidator::new`]) — this is
    /// the layer that actually gates composite assembly; see
    /// `LayoutDescriptor::validate_against_shape` for the opt-in Layer-3-only
    /// check of the same constraint.
    #[test]
    fn partition_dynamic_head_shape_rejected() {
        let layout = LayoutDescriptor::Composite(
            CompositeLayout::new(CompositionRule::Partition, 0).unwrap(),
        );
        let head = TensorDescriptor::new(
            1,
            0,
            crate::ElementType::Float32,
            Shape::new(vec![crate::shape::DYNAMIC, 8]).unwrap(),
            0,
            layout,
            vec![],
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let err = CompositeTensor::new(head, vec![]).unwrap_err();
        assert!(matches!(err, Error::InvalidLayout(_)));
    }

    // ── Overlay ──────────────────────────────────────────────────────────────

    fn base_member(ty: crate::ElementType, head_shape: Vec<u64>) -> TensorDescriptor {
        let offset = vec![0u64; head_shape.len()];
        dense_member(ty, head_shape.clone(), Some(shard(head_shape, offset)))
            .with_composite_member(CompositeMemberDescriptor::new(MemberRole::Base))
    }

    fn correction_member(
        ty: crate::ElementType,
        head_shape: Vec<u64>,
        shape: Vec<u64>,
        offset: Vec<u64>,
    ) -> TensorDescriptor {
        dense_member(ty, shape, Some(shard(head_shape, offset)))
            .with_composite_member(CompositeMemberDescriptor::new(MemberRole::Correction))
    }

    #[test]
    fn overlay_base_and_correction_replace_accepted() {
        let head = composite_head(
            CompositionRule::Overlay(CombineOp::Replace),
            2,
            vec![4, 4],
            crate::ElementType::Float32,
        );
        let base = base_member(crate::ElementType::Float32, vec![4, 4]);
        let correction = correction_member(
            crate::ElementType::Float32,
            vec![4, 4],
            vec![2, 2],
            vec![0, 0],
        );
        let composite = CompositeTensor::new(head, vec![base, correction]).unwrap();
        assert_eq!(composite.member_count(), 2);
        assert_eq!(
            *composite.rule(),
            CompositionRule::Overlay(CombineOp::Replace)
        );
    }

    #[test]
    fn overlay_base_and_correction_add_accepted() {
        let head = composite_head(
            CompositionRule::Overlay(CombineOp::Add),
            2,
            vec![4, 4],
            crate::ElementType::Float32,
        );
        let base = base_member(crate::ElementType::Float32, vec![4, 4]);
        let correction = correction_member(
            crate::ElementType::Float32,
            vec![4, 4],
            vec![2, 2],
            vec![1, 1],
        );
        let composite = CompositeTensor::new(head, vec![base, correction]).unwrap();
        assert_eq!(*composite.rule(), CompositionRule::Overlay(CombineOp::Add));
    }

    /// The first member of an overlay MUST be the base.
    #[test]
    fn overlay_base_not_first_rejected() {
        let head = composite_head(
            CompositionRule::Overlay(CombineOp::Replace),
            1,
            vec![4, 4],
            crate::ElementType::Float32,
        );
        let correction = correction_member(
            crate::ElementType::Float32,
            vec![4, 4],
            vec![4, 4],
            vec![0, 0],
        );
        let err = CompositeTensor::new(head, vec![correction]).unwrap_err();
        assert!(matches!(err, Error::CompositeOverlayBaseNotFirst));
    }

    /// A base member appearing at a position other than first is also rejected
    /// (same error, symmetric framing).
    #[test]
    fn overlay_base_appears_second_rejected() {
        let head = composite_head(
            CompositionRule::Overlay(CombineOp::Replace),
            2,
            vec![4, 4],
            crate::ElementType::Float32,
        );
        let correction = correction_member(
            crate::ElementType::Float32,
            vec![4, 4],
            vec![4, 4],
            vec![0, 0],
        );
        let base = base_member(crate::ElementType::Float32, vec![4, 4]);
        let err = CompositeTensor::new(head, vec![correction, base]).unwrap_err();
        assert!(matches!(err, Error::CompositeOverlayBaseNotFirst));
    }

    /// The base member MUST span the whole index space.
    #[test]
    fn overlay_base_not_spanning_rejected() {
        let head = composite_head(
            CompositionRule::Overlay(CombineOp::Replace),
            1,
            vec![4, 4],
            crate::ElementType::Float32,
        );
        // Base shape [2, 2] does not equal the head's [4, 4] shape.
        let base = dense_member(
            crate::ElementType::Float32,
            vec![2, 2],
            Some(shard(vec![4, 4], vec![0, 0])),
        )
        .with_composite_member(CompositeMemberDescriptor::new(MemberRole::Base));
        let err = CompositeTensor::new(head, vec![base]).unwrap_err();
        assert!(matches!(err, Error::CompositeOverlayBaseNotSpanning));
    }

    /// An overlay head with `member_count = 0` has no base member available.
    #[test]
    fn overlay_zero_members_rejected() {
        let head = composite_head(
            CompositionRule::Overlay(CombineOp::Replace),
            0,
            vec![4, 4],
            crate::ElementType::Float32,
        );
        let err = CompositeTensor::new(head, vec![]).unwrap_err();
        assert!(matches!(err, Error::CompositeOverlayEmpty));
    }

    /// Corrections MAY legally overlap each other (unlike partition members).
    #[test]
    fn overlay_corrections_may_overlap() {
        let head = composite_head(
            CompositionRule::Overlay(CombineOp::Replace),
            3,
            vec![4, 4],
            crate::ElementType::Float32,
        );
        let base = base_member(crate::ElementType::Float32, vec![4, 4]);
        // Both corrections cover the same [0,0]..[2,2] box — legal for overlay.
        let c0 = correction_member(
            crate::ElementType::Float32,
            vec![4, 4],
            vec![2, 2],
            vec![0, 0],
        );
        let c1 = correction_member(
            crate::ElementType::Float32,
            vec![4, 4],
            vec![2, 2],
            vec![0, 0],
        );
        let composite = CompositeTensor::new(head, vec![base, c0, c1]).unwrap();
        assert_eq!(composite.member_count(), 3);
    }

    /// A correction missing the Composite Member section is rejected.
    #[test]
    fn overlay_correction_missing_composite_member_flag_rejected() {
        let head = composite_head(
            CompositionRule::Overlay(CombineOp::Replace),
            2,
            vec![4, 4],
            crate::ElementType::Float32,
        );
        let base = base_member(crate::ElementType::Float32, vec![4, 4]);
        // Missing .with_composite_member(...) — HAS_COMPOSITE_MEMBER absent.
        let bad_correction = dense_member(
            crate::ElementType::Float32,
            vec![2, 2],
            Some(shard(vec![4, 4], vec![0, 0])),
        );
        let err = CompositeTensor::new(head, vec![base, bad_correction]).unwrap_err();
        assert!(matches!(
            err,
            Error::CompositeMemberFlagMismatch {
                has_flag: false,
                ..
            }
        ));
    }

    // ── Group ────────────────────────────────────────────────────────────────

    /// Group members MAY differ arbitrarily: different shapes, types, and no
    /// shard sections at all — no coverage or type check applies.
    #[test]
    fn group_heterogeneous_members_accepted() {
        let head = composite_head(
            CompositionRule::Group,
            2,
            vec![1],
            crate::ElementType::Float32,
        );
        let m0 = dense_member(crate::ElementType::Int8, vec![100], None);
        let m1 = dense_member(crate::ElementType::Float64, vec![3, 3, 3], None);
        let composite = CompositeTensor::new(head, vec![m0, m1]).unwrap();
        assert_eq!(composite.member_count(), 2);
        assert_eq!(*composite.rule(), CompositionRule::Group);
    }

    /// A group member count mismatch is still rejected (the only failure mode
    /// group composites share with the other rules, since there is no
    /// coverage/type check to violate).
    #[test]
    fn group_member_count_mismatch_rejected() {
        let head = composite_head(
            CompositionRule::Group,
            2,
            vec![1],
            crate::ElementType::Float32,
        );
        let m0 = dense_member(crate::ElementType::Int8, vec![100], None);
        let err = CompositeTensor::new(head, vec![m0]).unwrap_err();
        assert!(matches!(
            err,
            Error::CompositeMemberCountMismatch {
                declared: 2,
                actual: 1,
            }
        ));
    }

    // ── Nesting / depth ──────────────────────────────────────────────────────

    #[test]
    fn max_composite_depth_constant_is_8() {
        assert_eq!(MAX_COMPOSITE_DEPTH, 8);
    }

    /// `new_at_depth` at depth 7 (the 8th level, 0-indexed) with no nested
    /// composite member is accepted.
    #[test]
    fn new_at_depth_7_is_accepted() {
        let head = composite_head(
            CompositionRule::Group,
            1,
            vec![1],
            crate::ElementType::Float32,
        );
        let m0 = dense_member(crate::ElementType::Int8, vec![4], None);
        let result = CompositeTensor::new_at_depth(head, vec![m0], 7);
        assert!(result.is_ok(), "depth 7 should be accepted");
    }

    /// `new_at_depth` at depth 8 (the 9th level) is rejected immediately,
    /// regardless of the members supplied.
    #[test]
    fn new_at_depth_8_is_rejected() {
        let head = composite_head(
            CompositionRule::Group,
            0,
            vec![1],
            crate::ElementType::Float32,
        );
        let err = CompositeTensor::new_at_depth(head, vec![], 8).unwrap_err();
        assert!(matches!(err, Error::SubpavingNestingTooDeep));
    }

    /// A member whose own `layout_tag` is the composite tag counts as one
    /// additional nesting level: at depth 6, a nested composite member pushes
    /// the effective depth to 7 (still within the cap) and is accepted.
    #[test]
    fn new_at_depth_6_with_nested_composite_member_accepted() {
        let head = composite_head(
            CompositionRule::Group,
            1,
            vec![1],
            crate::ElementType::Float32,
        );
        let nested_head = composite_head(
            CompositionRule::Group,
            0,
            vec![1],
            crate::ElementType::Float32,
        );
        let result = CompositeTensor::new_at_depth(head, vec![nested_head], 6);
        assert!(
            result.is_ok(),
            "a nested composite member at depth 6 (effective depth 7) should be accepted"
        );
    }

    /// The same nested-composite-member case at depth 7 pushes the effective
    /// depth to 8, exceeding the cap.
    #[test]
    fn new_at_depth_7_with_nested_composite_member_rejected() {
        let head = composite_head(
            CompositionRule::Group,
            1,
            vec![1],
            crate::ElementType::Float32,
        );
        let nested_head = composite_head(
            CompositionRule::Group,
            0,
            vec![1],
            crate::ElementType::Float32,
        );
        let err = CompositeTensor::new_at_depth(head, vec![nested_head], 7).unwrap_err();
        assert!(matches!(err, Error::SubpavingNestingTooDeep));
    }

    // ── CompositeValidator direct use ───────────────────────────────────────

    #[test]
    fn validator_rejects_non_composite_head_layout() {
        // CompositeValidator does not derive Debug (it holds no Debug-friendly
        // state worth printing), so match the Result directly rather than
        // unwrap_err() (which requires Ok's type: Debug).
        let head = dense_member(crate::ElementType::Float32, vec![4], None);
        let result = CompositeValidator::new(&head);
        assert!(matches!(result, Err(Error::InvalidLayout(_))));
    }
}
