//! General subpaving layout element location.
//!
//! Spec: docs/spec/layouts/subpaving.md § Element Address

use crate::layout::{LayoutDescriptor, RegionDescriptor, SubpavingLayout};
use crate::{Error, Result, Shape};

use super::col_major::col_major_offset;
use super::row_major::row_major_offset;
use super::validate_index;

/// Location of an element within a subpaving layout.
///
/// Returned by [`SubpavingLayout::locate_element`]. The caller combines
/// `region_byte_offset + element_byte_delta(region_element_offset, element_type)`
/// to get the final byte address within `buffer_index`.
///
/// # Examples
///
/// ```
/// use hurray_core::layout::{RegionDescriptor, SubpavingLayout};
/// use hurray_core::Shape;
///
/// let regions = vec![
///     RegionDescriptor::new(vec![0, 0], vec![4, 4], 0x01, 0, 0).unwrap(),
///     RegionDescriptor::new(vec![0, 4], vec![4, 4], 0x01, 0, 64).unwrap(),
///     RegionDescriptor::new(vec![4, 0], vec![4, 4], 0x01, 0, 128).unwrap(),
///     RegionDescriptor::new(vec![4, 4], vec![4, 4], 0x01, 0, 192).unwrap(),
/// ];
/// let layout = SubpavingLayout::new(regions).unwrap();
/// let shape = Shape::new(vec![8, 8]).unwrap();
/// let loc = layout.locate_element(&[5, 6], &shape).unwrap();
/// assert_eq!(loc.region_index, 3);
/// assert_eq!(loc.region_element_offset, 6); // local [1,2] → 1*4+2=6
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubpavingLocation {
    /// Index of the matching region within [`SubpavingLayout::regions`].
    pub region_index: u32,
    /// Buffer table index for this region's data.
    pub buffer_index: u32,
    /// Byte offset into the region's buffer to the start of this region's data.
    pub region_byte_offset: u64,
    /// Linear element offset within the region's buffer (from `region_byte_offset`).
    pub region_element_offset: u64,
}

const MAX_SUBPAVING_DEPTH: usize = 8;

impl SubpavingLayout {
    /// Locates the element at `index` within a subpaving tensor of the given `shape`.
    ///
    /// Performs a linear scan over regions to find the one whose bounding box
    /// contains the index, then computes the element offset within that region.
    ///
    /// > **Note:** Region ordering is unspecified per the format spec; this
    /// > implementation uses a linear scan.
    /// > <!-- TODO(OQ-014.3): promote to sorted/indexed lookup once region ordering is resolved -->
    ///
    /// # Errors
    ///
    /// - [`Error::IndexRankMismatch`] — `index.len() != shape.rank()`.
    /// - [`Error::DynamicDimInIndexing`] — any shape dimension is `DYNAMIC`.
    /// - [`Error::IndexOutOfRange`] — any index component ≥ shape dimension.
    /// - [`Error::IndexNotInAnyRegion`] — no region contains the index.
    /// - [`Error::SubpavingNestingTooDeep`] — recursion depth exceeded 8 levels.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::layout::{RegionDescriptor, SubpavingLayout};
    /// use hurray_core::Shape;
    ///
    /// let regions = vec![
    ///     RegionDescriptor::new(vec![0, 0], vec![4, 4], 0x01, 0, 0).unwrap(),
    ///     RegionDescriptor::new(vec![4, 0], vec![4, 4], 0x01, 0, 64).unwrap(),
    /// ];
    /// let layout = SubpavingLayout::new(regions).unwrap();
    /// let shape = Shape::new(vec![8, 4]).unwrap();
    /// let loc = layout.locate_element(&[5, 2], &shape).unwrap();
    /// assert_eq!(loc.region_index, 1);
    /// assert_eq!(loc.region_element_offset, 6); // local [1,2] → 1*4+2=6
    /// ```
    pub fn locate_element(&self, index: &[u64], shape: &Shape) -> Result<SubpavingLocation> {
        self.locate_at_depth(index, shape, 0)
    }

    fn locate_at_depth(
        &self,
        index: &[u64],
        shape: &Shape,
        depth: usize,
    ) -> Result<SubpavingLocation> {
        if depth >= MAX_SUBPAVING_DEPTH {
            return Err(Error::SubpavingNestingTooDeep);
        }
        validate_index(index, shape)?;

        // Linear scan for the containing region.
        // TODO(OQ-014.3): Region ordering not specified; using linear scan.
        for (region_idx, region) in self.regions.iter().enumerate() {
            if !index_in_region(index, region) {
                continue;
            }

            let local_idx: Vec<u64> = index
                .iter()
                .zip(region.origin.iter())
                .map(|(&i, &o)| i - o)
                .collect();

            // Recursive subpaving: delegate to the inner SubpavingLayout.
            if region.region_layout_tag == 0x06 {
                let inner_shape = Shape::new(region.region_shape.clone())
                    .map_err(|e| Error::InvalidLayout(format!("invalid region shape: {e}")))?;
                let inner_sp = match region.inner_layout.as_deref() {
                    Some(LayoutDescriptor::Subpaving(sp)) => sp,
                    Some(_) => {
                        return Err(Error::InvalidLayout(
                            "inner_layout tag mismatch for recursive subpaving region".to_string(),
                        ))
                    }
                    None => {
                        return Err(Error::InvalidLayout(
                            "recursive subpaving region missing inner_layout".to_string(),
                        ))
                    }
                };
                return inner_sp.locate_at_depth(&local_idx, &inner_shape, depth + 1);
            }

            let region_element_offset = region_inner_offset(region, &local_idx)?;

            return Ok(SubpavingLocation {
                region_index: region_idx as u32,
                buffer_index: region.buffer_index,
                region_byte_offset: region.region_byte_offset,
                region_element_offset,
            });
        }

        Err(Error::IndexNotInAnyRegion {
            index: index.to_vec(),
        })
    }
}

fn index_in_region(index: &[u64], region: &RegionDescriptor) -> bool {
    index
        .iter()
        .zip(region.origin.iter())
        .zip(region.region_shape.iter())
        .all(|((&i, &o), &s)| i >= o && i < o + s)
}

fn region_inner_offset(region: &RegionDescriptor, local_idx: &[u64]) -> Result<u64> {
    match region.region_layout_tag {
        0x01 => row_major_offset(local_idx, &region.region_shape),
        0x02 => col_major_offset(local_idx, &region.region_shape),
        // Strided, tiled, Morton, Hilbert: dispatch through inner_layout.
        0x03 | 0x04 | 0x05 | 0x40 => {
            let inner = region.inner_layout.as_deref().ok_or_else(|| {
                Error::InvalidLayout(format!(
                    "region_layout_tag 0x{:02X} requires inner_layout",
                    region.region_layout_tag
                ))
            })?;
            let region_shape = Shape::new(region.region_shape.clone())
                .map_err(|e| Error::InvalidLayout(format!("invalid region shape: {e}")))?;
            inner.element_offset(local_idx, &region_shape)
        }
        tag => Err(Error::InvalidLayout(format!(
            "region_layout_tag 0x{tag:02X} is not supported in locate_element"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use crate::layout::{
        LayoutDescriptor, MortonLayout, RegionDescriptor, StridedLayout, SubpavingLayout,
    };
    use crate::{Error, Shape};

    // Spec docs/spec/layouts/subpaving.md § Element Address:
    // 8×8 tensor split into four 4×4 quadrants (all row-major, single buffer).
    //
    // Region layout:
    //   0: origin=[0,0], shape=[4,4], buffer_index=0, region_byte_offset=0
    //   1: origin=[0,4], shape=[4,4], buffer_index=0, region_byte_offset=64
    //   2: origin=[4,0], shape=[4,4], buffer_index=0, region_byte_offset=128
    //   3: origin=[4,4], shape=[4,4], buffer_index=0, region_byte_offset=192
    fn four_quadrant_layout() -> (SubpavingLayout, Shape) {
        let regions = vec![
            RegionDescriptor::new(vec![0, 0], vec![4, 4], 0x01, 0, 0).unwrap(),
            RegionDescriptor::new(vec![0, 4], vec![4, 4], 0x01, 0, 64).unwrap(),
            RegionDescriptor::new(vec![4, 0], vec![4, 4], 0x01, 0, 128).unwrap(),
            RegionDescriptor::new(vec![4, 4], vec![4, 4], 0x01, 0, 192).unwrap(),
        ];
        let layout = SubpavingLayout::new(regions).unwrap();
        let shape = Shape::new(vec![8, 8]).unwrap();
        (layout, shape)
    }

    // Spec example: element [5,6] → region 3 (origin [4,4]), local [1,2],
    // element_offset = row_major([1,2],[4,4]) = 1*4+2 = 6.
    #[test]
    fn subpaving_spec_example() {
        let (layout, shape) = four_quadrant_layout();
        let loc = layout.locate_element(&[5, 6], &shape).unwrap();
        assert_eq!(
            loc.region_index, 3,
            "expected region 3, got {}",
            loc.region_index
        );
        assert_eq!(
            loc.region_element_offset, 6,
            "expected element_offset 6, got {}",
            loc.region_element_offset
        );
        assert_eq!(loc.buffer_index, 0);
        assert_eq!(loc.region_byte_offset, 192);
    }

    // Element exactly at the inner corner of region 3: [4,4] → local [0,0],
    // element_offset = 0.
    #[test]
    fn subpaving_corner_element() {
        let (layout, shape) = four_quadrant_layout();
        let loc = layout.locate_element(&[4, 4], &shape).unwrap();
        assert_eq!(loc.region_index, 3);
        assert_eq!(loc.region_element_offset, 0);
    }

    // Element in region 0: [1,2] → local [1,2], offset = 1*4+2 = 6.
    #[test]
    fn subpaving_region_0() {
        let (layout, shape) = four_quadrant_layout();
        let loc = layout.locate_element(&[1, 2], &shape).unwrap();
        assert_eq!(loc.region_index, 0);
        assert_eq!(loc.region_element_offset, 6);
    }

    // Element in region 1: [2,5] → origin [0,4], local [2,1], offset = 2*4+1 = 9.
    #[test]
    fn subpaving_region_1() {
        let (layout, shape) = four_quadrant_layout();
        let loc = layout.locate_element(&[2, 5], &shape).unwrap();
        assert_eq!(loc.region_index, 1);
        assert_eq!(loc.region_element_offset, 9);
        assert_eq!(loc.region_byte_offset, 64);
    }

    // Element in region 2: [5,2] → origin [4,0], local [1,2], offset = 1*4+2 = 6.
    #[test]
    fn subpaving_region_2() {
        let (layout, shape) = four_quadrant_layout();
        let loc = layout.locate_element(&[5, 2], &shape).unwrap();
        assert_eq!(loc.region_index, 2);
        assert_eq!(loc.region_element_offset, 6);
        assert_eq!(loc.region_byte_offset, 128);
    }

    // Column-major region: element [5,6] in region 3 with col-major layout.
    // local [1,2], col_major_offset([1,2],[4,4]) = 1*1 + 2*4 = 9.
    #[test]
    fn subpaving_col_major_region() {
        let regions = vec![
            RegionDescriptor::new(vec![0, 0], vec![4, 4], 0x01, 0, 0).unwrap(),
            RegionDescriptor::new(vec![0, 4], vec![4, 4], 0x01, 0, 64).unwrap(),
            RegionDescriptor::new(vec![4, 0], vec![4, 4], 0x01, 0, 128).unwrap(),
            // Region 3 uses col-major (0x02).
            RegionDescriptor::new(vec![4, 4], vec![4, 4], 0x02, 0, 192).unwrap(),
        ];
        let layout = SubpavingLayout::new(regions).unwrap();
        let shape = Shape::new(vec![8, 8]).unwrap();
        let loc = layout.locate_element(&[5, 6], &shape).unwrap();
        assert_eq!(loc.region_index, 3);
        // col_major([1,2],[4,4]) = 1*1 + 2*4 = 9
        assert_eq!(loc.region_element_offset, 9);
    }

    // Error: index not in any region (gap in the layout).
    #[test]
    fn subpaving_index_not_in_region() {
        // Two non-covering regions: only [0,0..4) and [0,4..8).
        // Row 4+ is uncovered.
        let regions = vec![RegionDescriptor::new(vec![0, 0], vec![4, 8], 0x01, 0, 0).unwrap()];
        let layout = SubpavingLayout::new(regions).unwrap();
        let shape = Shape::new(vec![8, 8]).unwrap();
        let err = layout.locate_element(&[7, 0], &shape).unwrap_err();
        assert!(
            matches!(err, Error::IndexNotInAnyRegion { .. }),
            "expected IndexNotInAnyRegion, got {err:?}"
        );
    }

    // Error: index rank mismatch (1-component index into 2D shape).
    #[test]
    fn subpaving_rank_mismatch() {
        let (layout, shape) = four_quadrant_layout();
        let err = layout.locate_element(&[5], &shape).unwrap_err();
        assert!(
            matches!(err, Error::IndexRankMismatch { .. }),
            "expected IndexRankMismatch, got {err:?}"
        );
    }

    // Error: index out of bounds.
    #[test]
    fn subpaving_index_out_of_range() {
        let (layout, shape) = four_quadrant_layout();
        // index[0]=8 >= shape[0]=8.
        let err = layout.locate_element(&[8, 0], &shape).unwrap_err();
        assert!(
            matches!(err, Error::IndexOutOfRange { dim: 0, .. }),
            "expected IndexOutOfRange, got {err:?}"
        );
    }

    // Strided inner region: region 0 uses strided layout with strides [1, 4].
    // Element [2, 3] → local [2, 3], strided_offset = 2*1 + 3*4 = 14.
    #[test]
    fn subpaving_strided_inner_region() {
        let strided = LayoutDescriptor::Strided(StridedLayout::new(vec![1, 4]));
        let region = RegionDescriptor::new(vec![0, 0], vec![4, 4], 0x03, 0, 0)
            .unwrap()
            .with_inner_layout(strided)
            .unwrap();
        let layout = SubpavingLayout::new(vec![region]).unwrap();
        let shape = Shape::new(vec![4, 4]).unwrap();
        let loc = layout.locate_element(&[2, 3], &shape).unwrap();
        assert_eq!(loc.region_element_offset, 14); // 2*1 + 3*4
    }

    // Morton inner region: 4×4 region with Morton layout (2 bits per dim).
    // Element [3, 2] → Morton index interleaves bits of 3 (0b11) and 2 (0b10):
    // bit 0: x=1,y=0 → 0b01; bit 1: x=1,y=1 → 0b11 << 2 = 0b1100 → combined = 0b1101 = 13.
    // Actually let's compute: morton_bits=[2,2], dims 0 and 1, index [3, 2].
    // Bit 0 of dim0=3(0b11): bit0=1, bit1=1. Bit 0 of dim1=2(0b10): bit0=0, bit1=1.
    // h = bit0(d0)<<0 | bit0(d1)<<1 | bit1(d0)<<2 | bit1(d1)<<3
    //   = 1<<0 | 0<<1 | 1<<2 | 1<<3 = 1 + 0 + 4 + 8 = 13.
    #[test]
    fn subpaving_morton_inner_region() {
        let morton = LayoutDescriptor::Morton(MortonLayout::new(vec![2, 2]).unwrap());
        let region = RegionDescriptor::new(vec![0, 0], vec![4, 4], 0x05, 0, 0)
            .unwrap()
            .with_inner_layout(morton)
            .unwrap();
        let layout = SubpavingLayout::new(vec![region]).unwrap();
        let shape = Shape::new(vec![4, 4]).unwrap();
        let loc = layout.locate_element(&[3, 2], &shape).unwrap();
        assert_eq!(loc.region_element_offset, 13);
    }

    // Error: strided region without inner_layout → InvalidLayout.
    #[test]
    fn subpaving_strided_missing_inner_layout() {
        // region_layout_tag = 0x03 but inner_layout = None.
        let region = RegionDescriptor::new(vec![0, 0], vec![4, 4], 0x03, 0, 0).unwrap();
        let layout = SubpavingLayout::new(vec![region]).unwrap();
        let shape = Shape::new(vec![4, 4]).unwrap();
        let err = layout.locate_element(&[1, 1], &shape).unwrap_err();
        assert!(
            matches!(err, Error::InvalidLayout(_)),
            "expected InvalidLayout, got {err:?}"
        );
    }

    // Recursive subpaving: outer region covers [0,0..4×4), inner subpaving
    // splits it into two 2×4 halves (row-major).
    // Element [3, 2] → outer local [3, 2]; inner region 1 (origin [2,0]),
    // inner local [1, 2], row_major([1, 2], [2, 4]) = 1*4 + 2 = 6.
    #[test]
    fn subpaving_recursive() {
        let inner_regions = vec![
            RegionDescriptor::new(vec![0, 0], vec![2, 4], 0x01, 1, 0).unwrap(),
            RegionDescriptor::new(vec![2, 0], vec![2, 4], 0x01, 2, 0).unwrap(),
        ];
        let inner_sp = LayoutDescriptor::Subpaving(SubpavingLayout::new(inner_regions).unwrap());
        let outer_region = RegionDescriptor::new(vec![0, 0], vec![4, 4], 0x06, 0, 0)
            .unwrap()
            .with_inner_layout(inner_sp)
            .unwrap();
        let layout = SubpavingLayout::new(vec![outer_region]).unwrap();
        let shape = Shape::new(vec![4, 4]).unwrap();
        let loc = layout.locate_element(&[3, 2], &shape).unwrap();
        assert_eq!(loc.buffer_index, 2); // inner region 1's buffer
        assert_eq!(loc.region_byte_offset, 0);
        assert_eq!(loc.region_element_offset, 6); // row_major([1,2],[2,4]) = 6
    }

    // Recursive subpaving: nesting depth > 8 returns SubpavingNestingTooDeep.
    // Build 9 levels of nesting; the 9th call exceeds MAX_SUBPAVING_DEPTH.
    #[test]
    fn subpaving_recursive_too_deep() {
        // Build depth-9 nesting by wrapping a simple region in 8 subpaving layers.
        let leaf = SubpavingLayout::new(vec![
            RegionDescriptor::new(vec![0], vec![1], 0x01, 0, 0).unwrap()
        ])
        .unwrap();

        let mut current = leaf;
        for _ in 0..8 {
            let inner = LayoutDescriptor::Subpaving(current);
            let region = RegionDescriptor::new(vec![0], vec![1], 0x06, 0, 0)
                .unwrap()
                .with_inner_layout(inner)
                .unwrap();
            current = SubpavingLayout::new(vec![region]).unwrap();
        }

        let shape = Shape::new(vec![1]).unwrap();
        let err = current.locate_element(&[0], &shape).unwrap_err();
        assert!(
            matches!(err, Error::SubpavingNestingTooDeep),
            "expected SubpavingNestingTooDeep, got {err:?}"
        );
    }
}
