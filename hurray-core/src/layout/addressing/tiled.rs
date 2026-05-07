//! Tiled / blocked layout element offset computation.
//!
//! Spec: docs/spec/layouts/tiled.md § Element Address

use crate::layout::TiledLayout;
use crate::{Error, Result, Shape};

use super::col_major::col_major_offset;
use super::row_major::row_major_offset;
use super::{validate_index, ElementAddress};

impl ElementAddress for TiledLayout {
    fn element_offset(&self, index: &[u64], shape: &Shape) -> Result<u64> {
        validate_index(index, shape)?;
        if self.tile_shape.len() != shape.rank() {
            return Err(Error::IndexRankMismatch {
                index_rank: self.tile_shape.len(),
                shape_rank: shape.rank(),
            });
        }
        tiled_offset(self, index, shape.dims())
    }
}

/// Recursive tiled element offset computation.
///
/// `index` is the logical index into the tiled space; `dims` is the shape of
/// that space. Called recursively for `inner_layout == 0x04`.
fn tiled_offset(layout: &TiledLayout, index: &[u64], dims: &[u64]) -> Result<u64> {
    let rank = index.len();

    // 1. Tile index and intra-tile index per dimension.
    let mut tile_idx = vec![0u64; rank];
    let mut intra_idx = vec![0u64; rank];
    for k in 0..rank {
        tile_idx[k] = index[k] / layout.tile_shape[k];
        intra_idx[k] = index[k] % layout.tile_shape[k];
    }

    // 2. Tile-grid shape: ceil(dims[k] / tile_shape[k]).
    let tile_grid_dims: Vec<u64> = dims
        .iter()
        .zip(layout.tile_shape.iter())
        .map(|(&s, &t)| s.div_ceil(t))
        .collect();

    // 3. Linear tile number (in tile units).
    let tile_number: i64 = match layout.outer_layout {
        0x01 => row_major_offset(&tile_idx, &tile_grid_dims)? as i64,
        0x02 => col_major_offset(&tile_idx, &tile_grid_dims)? as i64,
        0x03 => strided_offset(
            &tile_idx,
            layout.outer_strides.as_ref().unwrap().strides.as_slice(),
        )?,
        _ => unreachable!("outer_layout validated in TiledLayout::new"),
    };

    // 4. Total elements per tile.
    let tile_size: i64 = layout.tile_shape.iter().try_fold(1i64, |acc, &d| {
        acc.checked_mul(d as i64).ok_or(Error::AddressOverflow)
    })?;

    // 5. Intra-tile element offset.
    let intra_offset: i64 = match layout.inner_layout {
        0x01 => row_major_offset(&intra_idx, &layout.tile_shape)? as i64,
        0x02 => col_major_offset(&intra_idx, &layout.tile_shape)? as i64,
        0x03 => strided_offset(
            &intra_idx,
            layout.inner_strides.as_ref().unwrap().strides.as_slice(),
        )?,
        0x04 => {
            let inner = layout
                .inner_tiled
                .as_ref()
                .expect("inner_tiled is Some when inner_layout == 0x04");
            // Recursive call: index into the outer tile's index space.
            tiled_offset(inner, &intra_idx, &layout.tile_shape)? as i64
        }
        _ => unreachable!("inner_layout validated in TiledLayout::new"),
    };

    // 6. Final offset = tile_number × tile_size + intra_offset.
    let total = tile_number
        .checked_mul(tile_size)
        .ok_or(Error::AddressOverflow)?
        .checked_add(intra_offset)
        .ok_or(Error::AddressOverflow)?;

    Ok(total as u64)
}

/// Computes a strided linear offset from an index and a stride slice (in `i64`).
///
/// Used for both outer (tile-grid) and inner (within-tile) strided layouts.
fn strided_offset(idx: &[u64], strides: &[i64]) -> Result<i64> {
    let mut sum: i64 = 0;
    for k in 0..idx.len() {
        let term = (idx[k] as i64)
            .checked_mul(strides[k])
            .ok_or(Error::AddressOverflow)?;
        sum = sum.checked_add(term).ok_or(Error::AddressOverflow)?;
    }
    Ok(sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{InnerStrides, OuterStrides, TiledLayout};
    use crate::Shape;

    // Spec docs/spec/layouts/tiled.md § Element Address:
    //
    // shape [6,8], tile_shape [2,4], outer=row-major (0x01), inner=row-major (0x01).
    //
    // Element [3,5]:
    //   tile_idx    = [3/2, 5/4] = [1, 1]
    //   intra_idx   = [3%2, 5%4] = [1, 1]
    //   tile_grid   = [6/2, 8/4] = [3, 2]
    //   tile_number = row_major([1,1], [3,2]) = 1*2+1 = 3
    //   tile_size   = 2*4 = 8
    //   intra_offset= row_major([1,1], [2,4]) = 1*4+1 = 5
    //   final       = 3*8 + 5 = 29
    #[test]
    fn tiled_spec_example() {
        let layout = TiledLayout::new(vec![2, 4], 0x01, 0x01, None, None, None).unwrap();
        let shape = Shape::new(vec![6, 8]).unwrap();
        assert_eq!(layout.element_offset(&[3, 5], &shape).unwrap(), 29);
    }

    // First element [0,0] is always at offset 0.
    #[test]
    fn tiled_first_element() {
        let layout = TiledLayout::new(vec![2, 4], 0x01, 0x01, None, None, None).unwrap();
        let shape = Shape::new(vec![6, 8]).unwrap();
        assert_eq!(layout.element_offset(&[0, 0], &shape).unwrap(), 0);
    }

    // Last element [5,7] in shape [6,8], tile [2,4], row-major / row-major:
    //   tile_idx    = [5/2, 7/4] = [2, 1]
    //   intra_idx   = [5%2, 7%4] = [1, 3]
    //   tile_grid   = [3, 2]
    //   tile_number = row_major([2,1],[3,2]) = 2*2+1 = 5
    //   tile_size   = 8
    //   intra_offset= row_major([1,3],[2,4]) = 1*4+3 = 7
    //   final       = 5*8 + 7 = 47
    #[test]
    fn tiled_last_element() {
        let layout = TiledLayout::new(vec![2, 4], 0x01, 0x01, None, None, None).unwrap();
        let shape = Shape::new(vec![6, 8]).unwrap();
        assert_eq!(layout.element_offset(&[5, 7], &shape).unwrap(), 47);
    }

    // Column-major inner layout:
    // shape [6,8], tile [2,4], outer=row-major (0x01), inner=col-major (0x02).
    //
    // Element [3,5]:
    //   tile_idx = [1,1], intra_idx = [1,1]
    //   tile_number = 3 (same as spec example, outer unchanged)
    //   tile_size   = 8
    //   intra_offset= col_major([1,1],[2,4]) = 1*1 + 1*2 = 3
    //   final       = 3*8 + 3 = 27
    #[test]
    fn tiled_col_major_inner() {
        let layout = TiledLayout::new(vec![2, 4], 0x01, 0x02, None, None, None).unwrap();
        let shape = Shape::new(vec![6, 8]).unwrap();
        assert_eq!(layout.element_offset(&[3, 5], &shape).unwrap(), 27);
    }

    // Recursive tiling:
    // shape [8,8], outer tile [4,4], outer=row-major, inner=tiled (0x04).
    // Inner tile [2,2], inner_outer=row-major, inner_inner=row-major.
    //
    // Element [3,5]:
    //   outer tile_idx  = [3/4, 5/4] = [0, 1]
    //   outer intra_idx = [3%4, 5%4] = [3, 1]
    //   outer tile_grid = [2, 2]
    //   outer tile_num  = row_major([0,1],[2,2]) = 0*2+1 = 1
    //   outer tile_size = 4*4 = 16
    //   inner (tile [2,2] in [4,4]):
    //     inner tile_idx  = [3/2, 1/2] = [1, 0]
    //     inner intra_idx = [3%2, 1%2] = [1, 1]
    //     inner tile_grid = [2, 2]
    //     inner tile_num  = row_major([1,0],[2,2]) = 1*2+0 = 2
    //     inner tile_size = 2*2 = 4
    //     inner intra     = row_major([1,1],[2,2]) = 1*2+1 = 3
    //     inner_offset    = 2*4 + 3 = 11
    //   final = 1*16 + 11 = 27
    #[test]
    fn tiled_recursive() {
        let inner = TiledLayout::new(vec![2, 2], 0x01, 0x01, None, None, None).unwrap();
        let outer =
            TiledLayout::new(vec![4, 4], 0x01, 0x04, None, None, Some(Box::new(inner))).unwrap();
        let shape = Shape::new(vec![8, 8]).unwrap();
        assert_eq!(outer.element_offset(&[3, 5], &shape).unwrap(), 27);
    }

    // Strided outer layout:
    // shape [4,4], tile [2,2], outer=strided (0x03) with strides [1,2],
    // inner=row-major.
    //
    // Element [2,3]:
    //   tile_idx  = [1, 1], intra_idx = [0, 1]
    //   tile_number = strided([1,1], strides=[1,2]) = 1*1 + 1*2 = 3
    //   tile_size   = 4
    //   intra_offset= row_major([0,1],[2,2]) = 0*2+1 = 1
    //   final       = 3*4 + 1 = 13
    #[test]
    fn tiled_strided_outer() {
        let outer_strides = OuterStrides::new(vec![1, 2]);
        let layout =
            TiledLayout::new(vec![2, 2], 0x03, 0x01, Some(outer_strides), None, None).unwrap();
        let shape = Shape::new(vec![4, 4]).unwrap();
        assert_eq!(layout.element_offset(&[2, 3], &shape).unwrap(), 13);
    }

    // Strided inner layout:
    // shape [4,4], tile [2,2], outer=row-major,
    // inner=strided (0x03) with strides [1,2] (col-major for 2×2 tile).
    //
    // Element [2,3]:
    //   tile_idx  = [1,1], intra_idx = [0,1]
    //   tile_number = row_major([1,1],[2,2]) = 1*2+1 = 3
    //   tile_size   = 4
    //   intra_offset= strided([0,1], strides=[1,2]) = 0*1 + 1*2 = 2
    //   final       = 3*4 + 2 = 14
    #[test]
    fn tiled_strided_inner() {
        let inner_strides = InnerStrides::new(vec![1, 2]);
        let layout =
            TiledLayout::new(vec![2, 2], 0x01, 0x03, None, Some(inner_strides), None).unwrap();
        let shape = Shape::new(vec![4, 4]).unwrap();
        assert_eq!(layout.element_offset(&[2, 3], &shape).unwrap(), 14);
    }

    // Error: tile_shape rank differs from shape rank.
    #[test]
    fn tiled_tile_shape_rank_mismatch() {
        let layout = TiledLayout::new(vec![2], 0x01, 0x01, None, None, None).unwrap();
        let shape = Shape::new(vec![6, 8]).unwrap();
        let err = layout.element_offset(&[3, 5], &shape).unwrap_err();
        assert!(
            matches!(err, Error::IndexRankMismatch { .. }),
            "expected IndexRankMismatch, got {err:?}"
        );
    }

    // Error: index out of bounds.
    #[test]
    fn tiled_index_out_of_range() {
        let layout = TiledLayout::new(vec![2, 4], 0x01, 0x01, None, None, None).unwrap();
        let shape = Shape::new(vec![6, 8]).unwrap();
        let err = layout.element_offset(&[6, 0], &shape).unwrap_err();
        assert!(
            matches!(err, Error::IndexOutOfRange { dim: 0, .. }),
            "expected IndexOutOfRange, got {err:?}"
        );
    }
}
