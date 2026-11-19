//! The dense remap engine: one element mover for every layout pair with an affine
//! stride form.
//!
//! Row-major, column-major and strided all address their buffer as
//! `offset = Σ index[k] × stride[k]` (`docs/spec/layouts/strided.md § Element Address`),
//! and `hurray_core::layout::LayoutDescriptor::dense_element_strides` is what reports the
//! stride vector for each. So a conversion between any two of them is one loop over the
//! output index space, reading at the source's strides and writing at the target's —
//! there is no per-layout code here, and none should be added.

use std::ops::Range;

use hurray_core::{ElementType, Shape};

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
    /// Bytes per element. Sub-byte types are refused at plan time, so this is exact.
    elem_bytes: u64,
    src_byte_offset: u64,
    dst_byte_offset: u64,
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
    /// Builds the recipe. Every size and offset is computed with checked arithmetic here
    /// so that the run loop can use plain arithmetic — the offsets it computes are
    /// bounded by the spans validated at plan time.
    pub(crate) fn new(
        shape: &Shape,
        src_strides: Vec<i64>,
        dst_strides: Vec<i64>,
        elem_bytes: u64,
        src_byte_offset: u64,
        dst_byte_offset: u64,
    ) -> Result<Self> {
        let dims = shape.dims().to_vec();
        let out_dims: Vec<u64> = dims
            .iter()
            .zip(dst_strides.iter())
            .map(|(&d, &s)| if s == 0 { 1 } else { d })
            .collect();
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
        let item_count = total.div_ceil(item_elements) as usize;
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
            elem_bytes,
            src_byte_offset,
            dst_byte_offset,
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
            let src_at = self.byte_address(self.src_byte_offset, &self.src_strides, &idx);
            let dst_at = self.byte_address(self.dst_byte_offset, &self.dst_strides, &idx);
            let len = n as usize * eb;
            debug_assert!(src_at + len <= src.len(), "source address out of bounds");
            dst.write_at(dst_at, &src[src_at..src_at + len]);
            self.advance(&mut idx, n);
            remaining -= n;
            elements += n;
        }
        RunReport::new(elements, elements * self.elem_bytes)
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
            let a = self.byte_address(self.src_byte_offset, &self.src_strides, &idx);
            let b = self.byte_address(self.src_byte_offset, &self.src_strides, &kept);
            if src[a..a + eb] != src[b..b + eb] {
                return Err(Error::RepetitionNotHeld {
                    dim,
                    index: idx.clone(),
                });
            }
        }
        Ok(())
    }

    #[inline]
    fn byte_address(&self, byte_offset: u64, strides: &[i64], idx: &[u64]) -> usize {
        let offset: i64 = idx
            .iter()
            .zip(strides.iter())
            .map(|(&i, &s)| i as i64 * s)
            .sum();
        (byte_offset as i64 + offset * self.elem_bytes as i64) as usize
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
    Ok(bits as u64 / 8)
}
