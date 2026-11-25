//! The dense remap engine: one element mover for every layout pair with an affine
//! stride form.
//!
//! Row-major, column-major and strided all address their buffer as
//! `offset = Σ index[k] × stride[k]` (`docs/spec/layouts/strided.md § Element Address`),
//! and `hurray_core::layout::LayoutDescriptor::dense_element_strides` is what reports the
//! stride vector for each. So a conversion between any two of them is one loop over the
//! output index space, reading at the source's strides and writing at the target's —
//! there is no per-layout code here, and none should be added.
//!
//! # The arithmetic contract
//!
//! Every address this module computes is **proved representable at planning time**, so the
//! run loop can use plain `int64` arithmetic without a per-element check. [`Geometry`] is
//! where that proof is established: it computes the byte strides and the lowest and
//! highest byte a layout addresses in checked `int128`, and refuses the conversion unless
//! all of them fit in `int64` *and* inside the largest slice this host can address. The
//! argument the loop then relies on is recorded on [`Remap::byte_address`].
//!
//! # Maintenance: the proof is a standing obligation, not a one-time check
//!
//! The arithmetic above is sound only while every address the engine computes comes from
//! the same validated [`Geometry`]. An optimization that adds a faster path — a wider bulk
//! copy, a per-axis specialisation, a precomputed row table — inherits that obligation, and
//! the cost of breaking it is an out-of-bounds write through the raw pointer in
//! `parallel.rs`. Three rules keep it:
//!
//! 1. **Every address comes from [`Remap::byte_address`].** No open-coded
//!    `base + index × stride`: the bound in its doc comment is stated for that function's
//!    arithmetic, and a copy of the arithmetic elsewhere is not covered by it.
//! 2. **Every bulk copy stays inside one run of the cover.** A copy of `n` elements is in
//!    bounds because those `n` elements are consecutive cover indices along the last axis
//!    *and* both layouts stride that axis by one — which is what `contiguous_last` means. A
//!    path that copies across an axis boundary, or at a stride other than 1, needs its own
//!    bound before it can be trusted.
//! 3. **Every path is named in a test.** `path_selection_is_explicit` pins which stride
//!    pairs choose which path, so deleting or bypassing one is a test failure rather than a
//!    silent loss of coverage, and `address_safety.rs` runs each path against an independent
//!    oracle on both destination implementations.
//!
//! There are two paths today — per element, and one copy per contiguous run — and two
//! destinations, the safe slice and the shared raw pointer. A third of either is where to
//! re-read this list.

use std::ops::Range;

use hurray_core::{strides_provably_distinct, Distinctness, ElementType, Shape};

use crate::dest::DestinationBytes;
use crate::{Error, Result, RunReport};

/// Bytes of output a single work item aims to cover.
///
/// Large enough that the per-item setup (an index decode and a stride dot product) is
/// noise against the copying, small enough that a few-megabyte tensor still has items
/// left to steal on a many-core fan-out. Not a knob: a work item is an implementation
/// detail of the partition, and a caller that wants different granularity wants
/// `run_range` (ADR-044 § 6).
const TARGET_ITEM_BYTES: u64 = 256 * 1024;

/// The largest byte count that can be presented to a run as `&mut [u8]`.
///
/// Rust caps a slice at `isize::MAX` bytes, so a destination beyond this cannot be handed
/// over at all, whatever the caller allocates it from — which makes it a planning-time
/// refusal rather than a run-time surprise. It is also what keeps every address in
/// [`Geometry`] convertible to `usize` without a narrowing check in the loop.
const ADDRESSABLE_LIMIT: i128 = isize::MAX as i128;

/// The validated byte geometry of one tensor: where its elements live, relative to the
/// start of its buffer.
///
/// Constructing one is the proof obligation; using one is arithmetic that cannot fail.
#[derive(Debug, Clone)]
pub(crate) struct Geometry {
    /// Byte address of logical element `[0, …, 0]`, from the start of the buffer. Equal to
    /// the descriptor's `byte_offset` for a source, and chosen by the plan for a
    /// destination — non-zero exactly when a stride is negative.
    base: i64,
    /// Per-axis strides in **bytes**: `stride[k] × elem_bytes`.
    byte_strides: Vec<i64>,
    /// One past the highest byte the layout addresses, from the start of the buffer. Zero
    /// for an empty tensor, which addresses nothing.
    extent: u64,
    /// The lowest byte the layout addresses, from the start of the buffer. Never negative:
    /// that is checked at construction.
    lowest: u64,
}

impl Geometry {
    /// The byte address of logical element `[0, …, 0]`, which is a descriptor's
    /// `byte_offset`.
    pub(crate) fn byte_offset(&self) -> u64 {
        // Non-negative by construction (checked in `geometry`).
        self.base as u64
    }

    /// One past the highest byte addressed, from the start of the buffer: the smallest
    /// buffer that can hold this tensor.
    pub(crate) fn extent(&self) -> u64 {
        self.extent
    }
}

/// The geometry of a tensor whose base address is already fixed — a source, whose
/// `byte_offset` its descriptor declares.
pub(crate) fn source_geometry(
    dims: &[u64],
    strides: &[i64],
    byte_offset: u64,
    elem_bytes: u64,
) -> Result<Geometry> {
    geometry(dims, strides, i128::from(byte_offset), elem_bytes)
}

/// The geometry of a tensor this plan is placing — a destination, whose base is derived
/// rather than declared.
///
/// The base is the layout's own negative reach: a negative stride reverses an axis, so
/// element `[0, …, 0]` is not the lowest address the tensor touches, and the plan must put
/// it that far into the buffer. For non-negative strides the base is 0.
pub(crate) fn destination_geometry(
    dims: &[u64],
    strides: &[i64],
    elem_bytes: u64,
) -> Result<Geometry> {
    // Measured from a base of 0 first, purely to learn how far below it the layout reaches.
    let reach = negative_reach(dims, strides, elem_bytes)?;
    geometry(dims, strides, -reach, elem_bytes)
}

/// How far below its base a layout reaches, in bytes, as a non-positive `int128`.
fn negative_reach(dims: &[u64], strides: &[i64], elem_bytes: u64) -> Result<i128> {
    if dims.contains(&0) {
        return Ok(0);
    }
    let elem_bytes = i128::from(elem_bytes);
    let mut low: i128 = 0;
    for (&stride, &dim) in strides.iter().zip(dims.iter()) {
        let term = i128::from(stride)
            .checked_mul(elem_bytes)
            .and_then(|byte_stride| byte_stride.checked_mul(i128::from(dim - 1)))
            .ok_or(hurray_core::Error::AddressOverflow)?;
        if term < 0 {
            low = low
                .checked_add(term)
                .ok_or(hurray_core::Error::AddressOverflow)?;
        }
    }
    Ok(low)
}

/// Computes and validates a tensor's byte geometry.
///
/// All arithmetic is checked `int128`, and every value the run loop will use is narrowed
/// only after it has been shown to fit. The three refusals this produces are the whole
/// arithmetic surface: an overflow beyond `int128`, a value beyond `int64`, and an extent
/// beyond what this host can address.
fn geometry(dims: &[u64], strides: &[i64], base: i128, elem_bytes: u64) -> Result<Geometry> {
    debug_assert_eq!(strides.len(), dims.len());
    let eb = i128::from(elem_bytes);
    let empty = dims.contains(&0);

    let mut byte_strides = Vec::with_capacity(strides.len());
    // Sum of the extreme negative terms and of the extreme positive terms. Because every
    // negative term is at most 0 and every positive term at least 0, these two bound every
    // *partial* sum as well, which is what makes the loop's running total safe.
    let mut low_sum: i128 = 0;
    let mut high_sum: i128 = 0;
    for (&stride, &dim) in strides.iter().zip(dims.iter()) {
        let byte_stride = i128::from(stride)
            .checked_mul(eb)
            .ok_or(hurray_core::Error::AddressOverflow)?;
        byte_strides.push(fits_i64(byte_stride)?);
        if empty {
            continue;
        }
        // Every index along this axis is cast to `int64` in the address loop, so the
        // largest of them has to fit. Only a dimension above 2^63 fails, and only when a
        // zero stride would have made the cast harmless anyway — checking it outright is
        // what lets `byte_address` state its proof without a case analysis.
        let extent = i64::try_from(dim - 1).map_err(|_| Error::BeyondHostLimit {
            what: "a dimension of the shape",
            value: i128::from(dim),
            limit: i128::from(i64::MAX),
        })?;
        let term = byte_stride
            .checked_mul(i128::from(extent))
            .ok_or(hurray_core::Error::AddressOverflow)?;
        let slot = if term < 0 {
            &mut low_sum
        } else {
            &mut high_sum
        };
        *slot = slot
            .checked_add(term)
            .ok_or(hurray_core::Error::AddressOverflow)?;
    }

    // An empty tensor addresses nothing, so it has no span and needs no buffer. The
    // spec's (dim - 1) arithmetic has no meaning for it.
    if empty {
        let base = fits_i64(base)?;
        if base < 0 {
            return Err(hurray_core::Error::AddressOverflow.into());
        }
        return Ok(Geometry {
            base,
            byte_strides,
            extent: 0,
            lowest: base as u64,
        });
    }

    let lowest = base
        .checked_add(low_sum)
        .ok_or(hurray_core::Error::AddressOverflow)?;
    let highest_end = base
        .checked_add(high_sum)
        .and_then(|high| high.checked_add(eb))
        .ok_or(hurray_core::Error::AddressOverflow)?;
    if lowest < 0 {
        // For a source this is an invalid descriptor: `strided.md § Element Address`
        // requires `byte_offset` to put element [0, …, 0] far enough into the buffer that
        // every element is addressable. For a destination it cannot happen, because the
        // base is derived from the reach.
        return Err(Error::LayoutReachesBelowBuffer {
            byte_offset: u64::try_from(base).unwrap_or(u64::MAX),
            below: lowest,
        });
    }
    if highest_end > ADDRESSABLE_LIMIT {
        return Err(Error::BeyondHostLimit {
            what: "the byte extent of a tensor's layout",
            value: highest_end,
            limit: ADDRESSABLE_LIMIT,
        });
    }
    // Both endpoints are now in [0, isize::MAX], so they fit int64 and usize alike.
    Ok(Geometry {
        base: fits_i64(base)?,
        byte_strides,
        extent: highest_end as u64,
        lowest: lowest as u64,
    })
}

fn fits_i64(value: i128) -> Result<i64> {
    i64::try_from(value).map_err(|_| hurray_core::Error::AddressOverflow.into())
}

/// A planned element move between two affine-strided layouts of one tensor.
#[derive(Debug, Clone)]
pub(crate) struct Remap {
    /// Full logical dimensions — the space the source is read over.
    dims: Vec<u64>,
    /// `dims` with every target-broadcast axis clamped to 1: the *output* cover.
    ///
    /// A zero target stride means many logical indices share one output element, so
    /// iterating `dims` would write the same bytes repeatedly and break § 6's
    /// disjointness invariant. Iterating the cover writes each output element once, and
    /// the survey is what establishes that the discarded indices carried the same value.
    out_dims: Vec<u64>,
    src_strides: Vec<i64>,
    dst_strides: Vec<i64>,
    src: Geometry,
    dst: Geometry,
    /// Bytes per element. Sub-byte types are refused at plan time, so this is exact — and
    /// the disjointness argument on [`Remap::byte_address`] depends on it being exact.
    elem_bytes: u64,
    /// Elements in the output cover.
    total: u64,
    /// Elements one work item covers.
    item_elements: u64,
    item_count: usize,
    /// True when the last axis advances by one element in both layouts, which makes a
    /// whole row one memory copy instead of one per element.
    contiguous_last: bool,
}

impl Remap {
    /// Builds the recipe, refusing any conversion whose addresses cannot be computed
    /// safely, or whose destination is not *proved* to address every element distinctly.
    pub(crate) fn new(
        shape: &Shape,
        src_strides: Vec<i64>,
        dst_strides: Vec<i64>,
        elem_bytes: u64,
        src: Geometry,
        dst: Geometry,
    ) -> Result<Self> {
        let dims = shape.dims().to_vec();
        // The output cover: a broadcast axis is collapsed to one element, because the
        // conversion writes each repeat once. `d.min(1)` rather than `1`, so that an axis
        // which is *both* broadcast and empty stays empty — clamping a zero extent up to
        // one would invent an index space for a tensor that has none, and the distinctness
        // check below would then rule on indices that cannot exist.
        let out_dims: Vec<u64> = dims
            .iter()
            .zip(dst_strides.iter())
            .map(|(&d, &s)| if s == 0 { d.min(1) } else { d })
            .collect();

        // The invariant everything else here rests on: distinct indices of the cover must
        // address distinct output elements. Checked on the *cover* rather than on `dims`,
        // because a broadcast axis is deliberately collapsed to one element before this
        // point — that axis aliases by design, and the survey is what makes it sound.
        //
        // Without this check a target like `Strided([1, 1])` over shape `[2, 2]` is
        // accepted: it has no zero stride, yet [0,1] and [1,0] address the same element.
        // Serially that silently overwrites a value the conversion promised to preserve;
        // in parallel the two writes can land in different work items, which is a data
        // race through a raw pointer.
        // The two refusals are kept apart because they have different remedies: an
        // aliasing target is refused forever, an unproved one only until the proof is
        // stronger. `Distinctness` is exhaustive, so a new verdict would break this match
        // rather than fall silently into the accepting arm.
        match strides_provably_distinct(&dst_strides, &out_dims)? {
            Distinctness::Distinct => {}
            Distinctness::Aliases => {
                return Err(Error::AliasingTarget {
                    strides: dst_strides,
                    dims: out_dims,
                })
            }
            Distinctness::NotProven => {
                return Err(Error::TargetNotProvablyDistinct {
                    strides: dst_strides,
                    dims: out_dims,
                })
            }
        }

        // An empty tensor has an empty cover: `total` is 0 and no item is generated.
        let total = if dims.contains(&0) {
            0
        } else {
            out_dims
                .iter()
                .try_fold(1u64, |acc, &d| acc.checked_mul(d))
                .ok_or(hurray_core::Error::AddressOverflow)?
        };
        let item_elements = (TARGET_ITEM_BYTES / elem_bytes.max(1)).max(1);
        let items = total.div_ceil(item_elements);
        let item_count = usize::try_from(items).map_err(|_| Error::BeyondHostLimit {
            what: "the work-item count",
            value: i128::from(items),
            limit: usize::MAX as i128,
        })?;
        let contiguous_last = match (src_strides.last(), dst_strides.last()) {
            (Some(&s), Some(&d)) => s == 1 && d == 1,
            // A rank-0 tensor is a single element: there is no row to copy.
            _ => false,
        };
        Ok(Self {
            dims,
            out_dims,
            src_strides,
            dst_strides,
            src,
            dst,
            elem_bytes,
            total,
            item_elements,
            item_count,
            contiguous_last,
        })
    }

    pub(crate) fn item_count(&self) -> usize {
        self.item_count
    }

    /// The axes this remap collapses: a real source axis mapped onto a single output
    /// element. Empty unless the target broadcasts an axis the source does not.
    pub(crate) fn collapsed_axes(&self) -> Vec<u32> {
        // An empty tensor has no values, so there is no repetition to establish — and the
        // canonical strides of a shape with a zero extent carry a zero stride that is an
        // artefact of that extent rather than a broadcast.
        if self.total == 0 {
            return Vec::new();
        }
        self.dst_strides
            .iter()
            .zip(self.src_strides.iter())
            .enumerate()
            .filter(|(k, (&dst, &src))| dst == 0 && src != 0 && self.dims[*k] > 1)
            .map(|(k, _)| k as u32)
            .collect()
    }

    /// Runs work items `items`, copying elements from `src` into `dst`.
    pub(crate) fn run_items<D: DestinationBytes>(
        &self,
        src: &[u8],
        dst: &mut D,
        items: Range<usize>,
    ) -> RunReport {
        let start = (items.start as u64)
            .saturating_mul(self.item_elements)
            .min(self.total);
        let end = (items.end as u64)
            .saturating_mul(self.item_elements)
            .min(self.total);
        if start >= end {
            return RunReport::default();
        }
        let eb = self.elem_bytes as usize;
        let mut idx = vec![0u64; self.out_dims.len()];
        self.decode(start, &mut idx);
        let mut remaining = end - start;
        let mut elements = 0u64;
        while remaining > 0 {
            // A run of `n` elements that is contiguous in both layouts is one copy.
            let n = if self.contiguous_last {
                let last = idx.len() - 1;
                remaining.min(self.out_dims[last] - idx[last])
            } else {
                1
            };
            let src_at = self.byte_address(&self.src, &idx);
            let dst_at = self.byte_address(&self.dst, &idx);
            let len = n as usize * eb;
            debug_assert!(
                src_at + len <= self.src.extent as usize && src_at + len <= src.len(),
                "source address outside the geometry proved at plan time"
            );
            debug_assert!(
                dst_at + len <= self.dst.extent as usize,
                "destination address outside the geometry proved at plan time"
            );
            dst.write_at(dst_at, &src[src_at..src_at + len]);
            self.advance(&mut idx, n);
            remaining -= n;
            elements += n;
        }
        RunReport::new(elements, elements.saturating_mul(self.elem_bytes))
    }

    /// Establishes that the data repeats along every collapsed axis, which is what makes
    /// collapsing one value-preserving (ADR-044 § 14b).
    ///
    /// Reads the whole source: the claim is about every element, and a sampled check
    /// would make the conversion's exactness a probability.
    pub(crate) fn verify_repetition(&self, src: &[u8]) -> Result<()> {
        let collapsed = self.collapsed_axes();
        if collapsed.is_empty() {
            return Ok(());
        }
        let eb = self.elem_bytes as usize;
        let full_total = self
            .dims
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d))
            .ok_or(hurray_core::Error::AddressOverflow)?;
        let mut idx = vec![0u64; self.dims.len()];
        for flat in 0..full_total {
            decode_into(flat, &self.dims, &mut idx);
            // The element the collapsed layout keeps is the one at index 0 on every
            // collapsed axis; every other index along those axes must match it.
            let Some(&dim) = collapsed.iter().find(|&&d| idx[d as usize] != 0) else {
                continue;
            };
            let kept: Vec<u64> = idx
                .iter()
                .enumerate()
                .map(|(k, &i)| {
                    if collapsed.contains(&(k as u32)) {
                        0
                    } else {
                        i
                    }
                })
                .collect();
            let a = self.byte_address(&self.src, &idx);
            let b = self.byte_address(&self.src, &kept);
            if src[a..a + eb] != src[b..b + eb] {
                return Err(Error::RepetitionNotHeld {
                    dim,
                    index: idx.clone(),
                });
            }
        }
        Ok(())
    }

    /// The byte address of logical index `idx` in one of the two tensors.
    ///
    /// # Why plain `int64` arithmetic cannot overflow here
    ///
    /// Let `t_k = idx[k] × byte_strides[k]`. Since `0 ≤ idx[k] ≤ dims[k] - 1`, each `t_k`
    /// lies between `0` and the extreme term `byte_strides[k] × (dims[k] - 1)` that
    /// [`geometry`] already computed — on the same side of zero. [`geometry`] summed the
    /// negative extremes into `lowest` and the positive extremes into `highest_end`, and
    /// refused the conversion unless **both** fit in `[0, isize::MAX]`.
    ///
    /// Because every negative term is at most `0` and every positive term at least `0`,
    /// dropping any subset of terms keeps the sum inside `[lowest, highest_end]`. Every
    /// *partial* sum this function forms is such a subset, so the running total — and the
    /// result — lie in `[lowest, highest_end] ⊆ [0, isize::MAX]`. No intermediate value
    /// can leave `int64`, the result is non-negative, and the cast to `usize` is lossless
    /// on both 32- and 64-bit hosts.
    ///
    /// The `idx[k] as i64` cast is lossless for the same reason: [`geometry`] refused the
    /// conversion unless every `dims[k] - 1` fits in `int64`, and `idx[k]` never exceeds
    /// it.
    ///
    /// The `debug_assert` is a check of that reasoning, not the reason it holds.
    #[inline]
    fn byte_address(&self, geometry: &Geometry, idx: &[u64]) -> usize {
        let address = idx
            .iter()
            .zip(geometry.byte_strides.iter())
            .fold(geometry.base, |acc, (&i, &byte_stride)| {
                acc + i as i64 * byte_stride
            });
        debug_assert!(
            address >= geometry.lowest as i64 && (address as u64) < geometry.extent,
            "address {address} outside the proved range [{}, {})",
            geometry.lowest,
            geometry.extent
        );
        address as usize
    }

    #[inline]
    fn decode(&self, flat: u64, idx: &mut [u64]) {
        decode_into(flat, &self.out_dims, idx);
    }

    #[inline]
    fn advance(&self, idx: &mut [u64], by: u64) {
        if idx.is_empty() {
            return;
        }
        let mut k = idx.len() - 1;
        idx[k] += by;
        while idx[k] >= self.out_dims[k] {
            idx[k] = 0;
            if k == 0 {
                return;
            }
            k -= 1;
            idx[k] += 1;
        }
    }
}

/// Decodes a flat row-major position into a multi-dimensional index over `dims`.
#[inline]
fn decode_into(mut flat: u64, dims: &[u64], idx: &mut [u64]) {
    for k in (0..dims.len()).rev() {
        idx[k] = flat % dims[k];
        flat /= dims[k];
    }
}

/// Bytes per element, or an error for a type this crate cannot move yet.
///
/// Sub-byte types (`int4`, `bool`, the `float6` pair) are packed, so moving one element
/// is a bit-field edit rather than a copy, and a negative stride can land a tensor's base
/// part-way through a byte. Both are real conversions and both are their own pass; a
/// refusal here is better than a silently wrong byte.
///
/// Whole-byte elements are also what makes the disjointness argument of § 6 hold: two
/// distinct element offsets are `elem_bytes` apart or more, so their byte ranges cannot
/// overlap. A packed type would break that, which is a second reason the refusal is not
/// merely about convenience.
pub(crate) fn element_bytes(
    ty: ElementType,
    ext: Option<&hurray_core::ExtensionTypeDescriptor>,
) -> Result<u64> {
    let bits = match (ty.bit_width(), ext) {
        // An extension type declares `bit_width = 0` as a sentinel; its own descriptor
        // carries the real width, which is enough to *move* a value even though it is
        // not enough to interpret one (ADR-044 § 14g).
        (0, Some(e)) => {
            if e.packing_factor != 1 {
                return Err(Error::unsupported_element_type(
                    ty,
                    "a packed extension type needs the sub-byte packing pass",
                ));
            }
            e.bit_width
        }
        (0, None) => {
            return Err(Error::unsupported_element_type(
                ty,
                "the element width is unknown without an extension type descriptor",
            ))
        }
        (bits, _) => bits,
    };
    if bits == 0 || bits % 8 != 0 {
        return Err(Error::unsupported_element_type(
            ty,
            "sub-byte element types need the packing pass: moving one element is a \
             bit-field edit, not a copy",
        ));
    }
    Ok(u64::from(bits) / 8)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn remap(dims: Vec<u64>, src: Vec<i64>, dst: Vec<i64>, elem_bytes: u64) -> Remap {
        let shape = Shape::new(dims.clone()).unwrap();
        // A source with a negative stride needs its base placed above its own reach, which
        // is what a real descriptor's `byte_offset` does.
        let reach = negative_reach(&dims, &src, elem_bytes).unwrap();
        let byte_offset = u64::try_from(-reach).unwrap();
        let source = source_geometry(&dims, &src, byte_offset, elem_bytes).unwrap();
        let destination = destination_geometry(&dims, &dst, elem_bytes).unwrap();
        Remap::new(&shape, src, dst, elem_bytes, source, destination).unwrap()
    }

    /// The geometry contract, over the whole cover rather than over the addresses one run
    /// happens to touch: every address the engine can compute for either tensor lies inside
    /// the range that was proved at plan time, with room for a whole element.
    fn assert_addresses_stay_inside_their_geometry(remap: &Remap) {
        let mut idx = vec![0u64; remap.out_dims.len()];
        let elem = remap.elem_bytes as usize;
        for flat in 0..remap.total {
            decode_into(flat, &remap.out_dims, &mut idx);
            for (name, geometry) in [("source", &remap.src), ("destination", &remap.dst)] {
                let at = remap.byte_address(geometry, &idx);
                assert!(
                    at >= geometry.lowest as usize,
                    "{name} address {at} below the proved low bound {} at {idx:?}",
                    geometry.lowest
                );
                assert!(
                    at + elem <= geometry.extent as usize,
                    "{name} address {at} + {elem} past the proved extent {} at {idx:?}",
                    geometry.extent
                );
            }
        }
    }

    /// Which stride pairs take which execution path. A new path, or a path that stops being
    /// reachable, shows up here rather than as a quiet loss of coverage (rule 3 in this
    /// module's maintenance notes).
    #[test]
    fn path_selection_is_explicit() {
        // Bulk copy: both layouts advance the last axis by one element.
        for (dims, src, dst) in [
            (vec![3, 4], vec![4, 1], vec![8, 1]),  // padded rows
            (vec![3, 4], vec![4, 1], vec![-4, 1]), // reversed rows
            (vec![2, 3, 4], vec![12, 4, 1], vec![24, 4, 1]),
        ] {
            assert!(
                remap(dims.clone(), src.clone(), dst.clone(), 1).contiguous_last,
                "{dims:?}: {src:?} → {dst:?} should copy whole runs"
            );
        }
        // Per element: the last axis is strided on at least one side.
        for (dims, src, dst) in [
            (vec![3, 4], vec![4, 1], vec![1, 3]),  // → column-major
            (vec![3, 4], vec![1, 3], vec![4, 1]),  // column-major →
            (vec![3, 4], vec![4, 1], vec![16, 4]), // strided last axis
        ] {
            assert!(
                !remap(dims.clone(), src.clone(), dst.clone(), 1).contiguous_last,
                "{dims:?}: {src:?} → {dst:?} should move single elements"
            );
        }
        // A rank-0 tensor is one element and has no run to copy.
        assert!(!remap(vec![], vec![], vec![], 4).contiguous_last);
    }

    /// The geometry bound holds on both paths, at several element widths, with reversed and
    /// padded strides and a broadcast source.
    #[test]
    fn every_address_lies_inside_the_proved_geometry() {
        let cases: &[(Vec<u64>, Vec<i64>, Vec<i64>)] = &[
            (vec![3, 4], vec![4, 1], vec![1, 3]),
            (vec![3, 4], vec![4, 1], vec![8, 1]),
            (vec![3, 4], vec![-4, 1], vec![4, 1]),
            (vec![3, 4], vec![4, 1], vec![-4, 1]),
            (vec![3, 4], vec![0, 1], vec![4, 1]),
            (vec![3, 4], vec![4, 1], vec![0, 1]),
            (vec![2, 3, 4], vec![12, 4, 1], vec![1, 2, 6]),
            (vec![5], vec![1], vec![-1]),
        ];
        for (dims, src, dst) in cases {
            for elem_bytes in [1, 2, 4, 8, 16] {
                let remap = remap(dims.clone(), src.clone(), dst.clone(), elem_bytes);
                assert_addresses_stay_inside_their_geometry(&remap);
            }
        }
    }

    /// A destination's extent is exactly what its own addresses need: one element past the
    /// highest, and no slack. An optimization that over-allocates would hide a bound error.
    #[test]
    fn a_destination_extent_is_tight() {
        let remap = remap(vec![3, 4], vec![4, 1], vec![1, 3], 4);
        let mut idx = vec![0u64; 2];
        let mut highest = 0usize;
        for flat in 0..remap.total {
            decode_into(flat, &remap.out_dims, &mut idx);
            highest = highest.max(remap.byte_address(&remap.dst, &idx));
        }
        assert_eq!(
            highest + remap.elem_bytes as usize,
            remap.dst.extent as usize
        );
    }
}
