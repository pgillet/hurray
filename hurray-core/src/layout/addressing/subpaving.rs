//! General subpaving layout element location.
//!
//! Spec: docs/spec/layouts/subpaving.md § Element Address

use crate::layout::{RegionDescriptor, SubpavingLayout};
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
        // Layer 4 restriction: strided regions require stride fields not yet
        // carried by RegionDescriptor (see subpaving.md OQ-1).
        tag => Err(Error::InvalidLayout(format!(
            "region_layout_tag 0x{tag:02X} is not supported in locate_element at this layer"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{RegionDescriptor, SubpavingLayout};
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
}
