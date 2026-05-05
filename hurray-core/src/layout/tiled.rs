//! Tiled / blocked layout descriptor.
//!
//! Tag `0x04`. Carries tile shape, outer/inner layout tags, and optional
//! outer/inner explicit strides. The inner layout MAY itself be tiled,
//! making this structure recursive (max depth 8 per spec).
//! See `docs/spec/layouts/tiled.md`.

use crate::{Error, Result};

/// Maximum recursion depth for nested tiled layouts.
///
/// The spec RECOMMENDS a maximum of 8 levels; this implementation enforces it.
pub const MAX_TILED_DEPTH: usize = 8;

/// Outer strides for the tile grid (in units of **tiles**, not elements).
///
/// Present only when `outer_layout == 0x03` (strided).
///
/// # Examples
///
/// ```
/// use hurray_core::layout::OuterStrides;
///
/// let os = OuterStrides::new(vec![2, 1]);
/// assert_eq!(os.strides, [2, 1]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct OuterStrides {
    /// Outer tile-grid strides, one per dimension, in units of **tiles**.
    pub strides: Vec<i64>,
}

impl OuterStrides {
    /// Creates a new [`OuterStrides`] with the given per-dimension tile-grid strides.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::layout::OuterStrides;
    ///
    /// let os = OuterStrides::new(vec![3, 1]);
    /// assert_eq!(os.strides, [3, 1]);
    /// ```
    pub fn new(strides: Vec<i64>) -> Self {
        Self { strides }
    }
}

/// Inner strides within a single tile (in **logical elements**).
///
/// Present only when `inner_layout == 0x03` (strided).
///
/// # Examples
///
/// ```
/// use hurray_core::layout::InnerStrides;
///
/// let is = InnerStrides::new(vec![4, 1]);
/// assert_eq!(is.strides, [4, 1]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct InnerStrides {
    /// Per-dimension strides within a tile, in **logical elements**.
    pub strides: Vec<i64>,
}

impl InnerStrides {
    /// Creates a new [`InnerStrides`] with the given per-dimension element strides.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::layout::InnerStrides;
    ///
    /// let is = InnerStrides::new(vec![1, 4]);
    /// assert_eq!(is.strides, [1, 4]);
    /// ```
    pub fn new(strides: Vec<i64>) -> Self {
        Self { strides }
    }
}

/// Descriptor for the tiled / blocked layout.
///
/// A tiled layout partitions the tensor index space into uniform rectangular
/// tiles. Tile ordering is controlled by `outer_layout`; element ordering
/// within a tile is controlled by `inner_layout`. Both accept tags for
/// row-major (`0x01`), column-major (`0x02`), strided (`0x03`), or, for the
/// inner layout only, a recursively nested tiled layout (`0x04`).
///
/// Recursive tiling is useful for hierarchical GEMM blocking (e.g., 128×128
/// L2 tiles subdivided into 32×32 L1 tiles). Maximum nesting depth is
/// [`MAX_TILED_DEPTH`] (8 levels).
///
/// # Examples
///
/// ```
/// use hurray_core::layout::{LayoutDescriptor, TiledLayout};
///
/// // 2×4 tiles, row-major outer, row-major inner.
/// let layout = LayoutDescriptor::Tiled(Box::new(
///     TiledLayout::new(vec![2, 4], 0x01, 0x01, None, None, None).unwrap(),
/// ));
/// assert_eq!(layout.tag(), 0x04);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct TiledLayout {
    /// Tile dimensions. Every value MUST be > 0.
    /// `tile_shape.len()` MUST equal the tensor rank.
    pub tile_shape: Vec<u64>,

    /// Layout tag for tile-grid ordering.
    /// MUST be `0x01` (row-major), `0x02` (col-major), or `0x03` (strided).
    pub outer_layout: u8,

    /// Layout tag for element ordering within a tile.
    /// MUST be `0x01`, `0x02`, `0x03`, or `0x04` (recursive tiling).
    pub inner_layout: u8,

    /// Explicit outer (tile-grid) strides in units of **tiles**.
    /// Present iff `outer_layout == 0x03`.
    pub outer_strides: Option<OuterStrides>,

    /// Explicit inner strides in **logical elements** within a tile.
    /// Present iff `inner_layout == 0x03`.
    pub inner_strides: Option<InnerStrides>,

    /// Recursive inner tiling descriptor.
    /// Present iff `inner_layout == 0x04`.
    pub inner_tiled: Option<Box<TiledLayout>>,
}

impl TiledLayout {
    /// Creates a new [`TiledLayout`], validating that:
    /// - all `tile_shape` values are > 0,
    /// - `outer_layout` is `0x01`, `0x02`, or `0x03`,
    /// - `inner_layout` is `0x01`, `0x02`, `0x03`, or `0x04`,
    /// - `outer_strides` is `Some` iff `outer_layout == 0x03`,
    /// - `inner_strides` is `Some` iff `inner_layout == 0x03`,
    /// - `inner_tiled` is `Some` iff `inner_layout == 0x04`,
    /// - recursion depth does not exceed [`MAX_TILED_DEPTH`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidLayout`] on any constraint violation.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::layout::TiledLayout;
    ///
    /// // 4×4 tiles, row-major outer, column-major inner.
    /// let t = TiledLayout::new(vec![4, 4], 0x01, 0x02, None, None, None).unwrap();
    /// assert_eq!(t.tile_shape, [4, 4]);
    /// ```
    pub fn new(
        tile_shape: Vec<u64>,
        outer_layout: u8,
        inner_layout: u8,
        outer_strides: Option<OuterStrides>,
        inner_strides: Option<InnerStrides>,
        inner_tiled: Option<Box<TiledLayout>>,
    ) -> Result<Self> {
        Self::new_at_depth(
            tile_shape,
            outer_layout,
            inner_layout,
            outer_strides,
            inner_strides,
            inner_tiled,
            0,
        )
    }

    // Internal constructor that tracks recursion depth.
    fn new_at_depth(
        tile_shape: Vec<u64>,
        outer_layout: u8,
        inner_layout: u8,
        outer_strides: Option<OuterStrides>,
        inner_strides: Option<InnerStrides>,
        inner_tiled: Option<Box<TiledLayout>>,
        depth: usize,
    ) -> Result<Self> {
        if depth >= MAX_TILED_DEPTH {
            return Err(Error::InvalidLayout(format!(
                "tiled layout recursion depth {depth} exceeds maximum of {MAX_TILED_DEPTH}"
            )));
        }

        // All tile dimensions must be > 0.
        if tile_shape.is_empty() {
            return Err(Error::InvalidLayout(
                "tile_shape must not be empty".to_string(),
            ));
        }
        for (k, &dim) in tile_shape.iter().enumerate() {
            if dim == 0 {
                return Err(Error::InvalidLayout(format!(
                    "tile_shape[{k}] must be > 0, got 0"
                )));
            }
        }

        // outer_layout: must be 0x01, 0x02, or 0x03.
        if !matches!(outer_layout, 0x01..=0x03) {
            return Err(Error::InvalidLayout(format!(
                "outer_layout 0x{outer_layout:02X} is invalid: must be 0x01, 0x02, or 0x03"
            )));
        }

        // inner_layout: must be 0x01, 0x02, 0x03, or 0x04.
        if !matches!(inner_layout, 0x01..=0x04) {
            return Err(Error::InvalidLayout(format!(
                "inner_layout 0x{inner_layout:02X} is invalid: must be 0x01, 0x02, 0x03, or 0x04"
            )));
        }

        // outer_strides must be present iff outer_layout == 0x03.
        match (outer_layout, outer_strides.is_some()) {
            (0x03, false) => {
                return Err(Error::InvalidLayout(
                    "outer_strides must be present when outer_layout is 0x03 (strided)".to_string(),
                ));
            }
            (layout, true) if layout != 0x03 => {
                return Err(Error::InvalidLayout(format!(
                    "outer_strides present but outer_layout is 0x{layout:02X} (not strided)"
                )));
            }
            _ => {}
        }

        // inner_strides must be present iff inner_layout == 0x03.
        match (inner_layout, inner_strides.is_some()) {
            (0x03, false) => {
                return Err(Error::InvalidLayout(
                    "inner_strides must be present when inner_layout is 0x03 (strided)".to_string(),
                ));
            }
            (layout, true) if layout != 0x03 => {
                return Err(Error::InvalidLayout(format!(
                    "inner_strides present but inner_layout is 0x{layout:02X} (not strided)"
                )));
            }
            _ => {}
        }

        // inner_tiled must be present iff inner_layout == 0x04.
        match (inner_layout, inner_tiled.is_some()) {
            (0x04, false) => {
                return Err(Error::InvalidLayout(
                    "inner_tiled must be present when inner_layout is 0x04 (tiled)".to_string(),
                ));
            }
            (layout, true) if layout != 0x04 => {
                return Err(Error::InvalidLayout(format!(
                    "inner_tiled present but inner_layout is 0x{layout:02X} (not tiled)"
                )));
            }
            _ => {}
        }

        // Recursively validate the inner tiled descriptor at depth+1.
        // We rebuild it via new_at_depth to enforce depth tracking.
        let inner_tiled = if let Some(inner) = inner_tiled {
            // The inner TiledLayout was already validated by its own constructor;
            // we only need to check that adding it here doesn't exceed max depth.
            // Re-validate with depth+1 to enforce the global depth limit.
            let validated = Self::new_at_depth(
                inner.tile_shape.clone(),
                inner.outer_layout,
                inner.inner_layout,
                inner.outer_strides.clone(),
                inner.inner_strides.clone(),
                inner.inner_tiled.clone(),
                depth + 1,
            )?;
            Some(Box::new(validated))
        } else {
            None
        };

        Ok(Self {
            tile_shape,
            outer_layout,
            inner_layout,
            outer_strides,
            inner_strides,
            inner_tiled,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::LayoutDescriptor;

    #[test]
    fn tiled_tag_is_0x04() {
        let t = TiledLayout::new(vec![4, 4], 0x01, 0x01, None, None, None).unwrap();
        let layout = LayoutDescriptor::Tiled(Box::new(t));
        assert_eq!(layout.tag(), 0x04);
    }

    #[test]
    fn tiled_buffer_count_is_1() {
        use std::num::NonZeroU8;
        let t = TiledLayout::new(vec![4, 4], 0x01, 0x01, None, None, None).unwrap();
        let layout = LayoutDescriptor::Tiled(Box::new(t));
        assert_eq!(layout.buffer_count(), Some(NonZeroU8::new(1).unwrap()));
    }

    #[test]
    fn rejects_zero_tile_dim() {
        let err = TiledLayout::new(vec![4, 0], 0x01, 0x01, None, None, None).unwrap_err();
        assert!(matches!(err, Error::InvalidLayout(_)));
    }

    #[test]
    fn rejects_invalid_outer_layout() {
        let err = TiledLayout::new(vec![4, 4], 0x04, 0x01, None, None, None).unwrap_err();
        assert!(matches!(err, Error::InvalidLayout(_)));
    }

    #[test]
    fn rejects_invalid_inner_layout() {
        let err = TiledLayout::new(vec![4, 4], 0x01, 0x05, None, None, None).unwrap_err();
        assert!(matches!(err, Error::InvalidLayout(_)));
    }

    #[test]
    fn rejects_missing_outer_strides_when_outer_is_strided() {
        let err = TiledLayout::new(vec![4, 4], 0x03, 0x01, None, None, None).unwrap_err();
        assert!(matches!(err, Error::InvalidLayout(_)));
    }

    #[test]
    fn rejects_outer_strides_when_outer_not_strided() {
        let outer_strides = Some(OuterStrides {
            strides: vec![2, 1],
        });
        let err = TiledLayout::new(vec![4, 4], 0x01, 0x01, outer_strides, None, None).unwrap_err();
        assert!(matches!(err, Error::InvalidLayout(_)));
    }

    #[test]
    fn accepts_strided_outer_with_outer_strides() {
        let outer_strides = Some(OuterStrides {
            strides: vec![2, 1],
        });
        let t = TiledLayout::new(vec![4, 4], 0x03, 0x01, outer_strides, None, None);
        assert!(t.is_ok());
    }

    #[test]
    fn rejects_excessive_recursion_depth() {
        // Build a chain of TiledLayouts from the leaf up.
        // The leaf uses inner_layout=0x01 (row-major, no recursion).
        // Each wrapper uses inner_layout=0x04 (tiled, recurses into inner).
        fn make_tiled_at_depth(remaining: usize) -> Box<TiledLayout> {
            if remaining == 0 {
                // Leaf: no recursion.
                Box::new(TiledLayout::new(vec![2, 2], 0x01, 0x01, None, None, None).unwrap())
            } else {
                let inner = make_tiled_at_depth(remaining - 1);
                Box::new(TiledLayout::new(vec![2, 2], 0x01, 0x04, None, None, Some(inner)).unwrap())
            }
        }

        // MAX_TILED_DEPTH - 1 wrappers + 1 leaf = MAX_TILED_DEPTH total levels → valid.
        let _valid = make_tiled_at_depth(MAX_TILED_DEPTH - 1);

        // One more wrapper pushes total depth to MAX_TILED_DEPTH + 1 → rejected.
        let too_deep = TiledLayout::new(
            vec![2, 2],
            0x01,
            0x04,
            None,
            None,
            Some(make_tiled_at_depth(MAX_TILED_DEPTH - 1)),
        );
        assert!(
            too_deep.is_err(),
            "depth {} should be rejected (max is {})",
            MAX_TILED_DEPTH + 1,
            MAX_TILED_DEPTH
        );
    }
}
