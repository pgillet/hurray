//! Dense layout classes: row-major, column-major, strided, tiled, Morton, Hilbert.
//!
//! Every layout here stores elements in one buffer in a directly addressable order,
//! so `buffer_count` is 1 and `is_dense` is `True` for all of them.

use pyo3::prelude::*;
use pyo3::types::PyTuple;

use hurray_core::layout::{
    InnerStrides, OuterStrides, TiledLayout as CoreTiled, TAG_COL_MAJOR, TAG_ROW_MAJOR,
    TAG_STRIDED, TAG_TILED,
};
use hurray_core::LayoutDescriptor;

use super::{layout_err, tag_name, variant_mismatch, Layout};

// ── RowMajorLayout ────────────────────────────────────────────────────────────

/// Row-major (C-order) layout. Tag `0x01`.
///
/// The default: element `[i, j]` is at offset `i * ncols + j`.
///
/// ## Examples (Python)
///
/// ```python
/// import hurray
///
/// t = hurray.Tensor(bytes(16), hurray.float32, [4])
/// assert t.layout == hurray.RowMajorLayout()
/// ```
#[pyclass(name = "RowMajorLayout", extends = Layout, frozen)]
pub struct RowMajorLayout;

#[pymethods]
impl RowMajorLayout {
    /// Construct a row-major layout, which carries no parameters.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.RowMajorLayout().name == "row_major"
    /// ```
    #[new]
    pub fn new() -> PyClassInitializer<Self> {
        Layout::of(LayoutDescriptor::RowMajor).init(Self)
    }

    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert repr(hurray.RowMajorLayout()) == "RowMajorLayout()"
    /// ```
    pub fn __repr__(&self) -> String {
        "RowMajorLayout()".to_string()
    }
}

// ── ColMajorLayout ────────────────────────────────────────────────────────────

/// Column-major (Fortran-order) layout. Tag `0x02`.
///
/// ## Examples (Python)
///
/// ```python
/// import hurray
///
/// assert hurray.ColMajorLayout().name == "col_major"
/// ```
#[pyclass(name = "ColMajorLayout", extends = Layout, frozen)]
pub struct ColMajorLayout;

#[pymethods]
impl ColMajorLayout {
    /// Construct a column-major layout, which carries no parameters.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.ColMajorLayout().tag == 0x02
    /// ```
    #[new]
    pub fn new() -> PyClassInitializer<Self> {
        Layout::of(LayoutDescriptor::ColMajor).init(Self)
    }

    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert repr(hurray.ColMajorLayout()) == "ColMajorLayout()"
    /// ```
    pub fn __repr__(&self) -> String {
        "ColMajorLayout()".to_string()
    }
}

// ── StridedLayout ─────────────────────────────────────────────────────────────

/// Strided layout with explicit per-dimension strides. Tag `0x03`.
///
/// **Strides are in logical elements, not bytes**, and may be negative (a reversed
/// axis) or zero (a broadcast axis). A reader arriving from NumPy, whose strides are
/// in bytes, will otherwise read them wrongly.
///
/// ## Examples (Python)
///
/// ```python
/// import hurray
///
/// l = hurray.StridedLayout([4, 1])
/// assert l.strides == (4, 1)
/// assert l.name == "strided"
/// ```
#[pyclass(name = "StridedLayout", extends = Layout, frozen)]
pub struct StridedLayout;

#[pymethods]
impl StridedLayout {
    /// Construct a strided layout from strides **in logical elements**.
    ///
    /// `len(strides)` must equal the tensor's rank; that is checked when a tensor
    /// is built with this layout, since a layout on its own has no shape.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.StridedLayout([1, -4]).strides == (1, -4)
    /// ```
    #[new]
    pub fn new(strides: Vec<i64>) -> PyClassInitializer<Self> {
        Layout::of(LayoutDescriptor::Strided(
            hurray_core::layout::StridedLayout::new(strides),
        ))
        .init(Self)
    }

    /// The per-dimension strides, in logical elements.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.StridedLayout([4, 1]).strides == (4, 1)
    /// ```
    #[getter]
    pub fn strides(slf: PyRef<'_, Self>) -> PyResult<Py<PyTuple>> {
        match slf.as_super().descriptor() {
            LayoutDescriptor::Strided(l) => {
                Ok(PyTuple::new(slf.py(), l.strides.iter().copied())?.unbind())
            }
            other => Err(variant_mismatch("StridedLayout", other)),
        }
    }

    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert repr(hurray.StridedLayout([4, 1])) == "StridedLayout(strides=(4, 1))"
    /// ```
    pub fn __repr__(slf: PyRef<'_, Self>) -> PyResult<String> {
        let strides = Self::strides(slf)?;
        Python::attach(|py| {
            Ok(format!(
                "StridedLayout(strides={})",
                strides.bind(py).repr()?
            ))
        })
    }
}

// ── TiledLayout ───────────────────────────────────────────────────────────────

/// Tiled (blocked) layout. Tag `0x04`.
///
/// Elements are grouped into fixed-size tiles; `outer_layout` orders the tiles and
/// `inner_layout` orders the elements within a tile. A tiled layout may nest another
/// tiled layout as its inner layout, up to the depth bound core enforces.
///
/// ## Examples (Python)
///
/// ```python
/// import hurray
///
/// l = hurray.TiledLayout([4, 4], inner_layout="col_major")
/// assert l.tile_shape == (4, 4)
/// assert l.outer_layout == "row_major"
/// assert l.inner_layout == "col_major"
/// ```
#[pyclass(name = "TiledLayout", extends = Layout, frozen)]
pub struct TiledLayout;

#[pymethods]
impl TiledLayout {
    /// Construct a tiled layout.
    ///
    /// `outer_layout` is one of `"row_major"`, `"col_major"`, `"strided"`;
    /// `inner_layout` may additionally be `"tiled"`. Strides must be supplied
    /// exactly when the corresponding layout is `"strided"`, and `inner_tiled`
    /// exactly when `inner_layout` is `"tiled"` — core rejects any other
    /// combination.
    ///
    /// ## Errors
    ///
    /// - `ValueError` — a layout name that is not a legal tile ordering.
    /// - `hurray.InvalidDescriptorError` — an empty or zero tile shape, a
    ///   strides/layout mismatch, or nesting deeper than core allows.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// l = hurray.TiledLayout([32, 32])
    /// assert l.inner_tiled is None
    /// ```
    #[new]
    #[pyo3(signature = (
        tile_shape,
        outer_layout = "row_major",
        inner_layout = "row_major",
        outer_strides = None,
        inner_strides = None,
        inner_tiled = None,
    ))]
    pub fn new(
        tile_shape: Vec<u64>,
        outer_layout: &str,
        inner_layout: &str,
        outer_strides: Option<Vec<i64>>,
        inner_strides: Option<Vec<i64>>,
        inner_tiled: Option<PyRef<'_, TiledLayout>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let inner_tiled = match inner_tiled {
            Some(t) => match t.as_super().descriptor() {
                LayoutDescriptor::Tiled(l) => Some(l.clone()),
                other => return Err(variant_mismatch("TiledLayout", other)),
            },
            None => None,
        };
        let core = CoreTiled::new(
            tile_shape,
            tag_for_tile_name(outer_layout)?,
            tag_for_tile_name(inner_layout)?,
            outer_strides.map(OuterStrides::new),
            inner_strides.map(InnerStrides::new),
            inner_tiled,
        )
        .map_err(layout_err)?;
        Ok(Layout::of(LayoutDescriptor::Tiled(Box::new(core))).init(Self))
    }

    /// The tile extent along each dimension.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.TiledLayout([8, 8]).tile_shape == (8, 8)
    /// ```
    #[getter]
    pub fn tile_shape(slf: PyRef<'_, Self>) -> PyResult<Py<PyTuple>> {
        let core = tiled_of(&slf)?;
        Ok(PyTuple::new(slf.py(), core.tile_shape.iter().copied())?.unbind())
    }

    /// How tiles are ordered relative to each other, as a lowercase name.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.TiledLayout([8, 8]).outer_layout == "row_major"
    /// ```
    #[getter]
    pub fn outer_layout(slf: PyRef<'_, Self>) -> PyResult<&'static str> {
        Ok(tag_name(tiled_of(&slf)?.outer_layout))
    }

    /// How elements are ordered within a tile, as a lowercase name.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.TiledLayout([8, 8], inner_layout="col_major").inner_layout == "col_major"
    /// ```
    #[getter]
    pub fn inner_layout(slf: PyRef<'_, Self>) -> PyResult<&'static str> {
        Ok(tag_name(tiled_of(&slf)?.inner_layout))
    }

    /// Tile-level strides in logical elements, or `None` unless the outer layout is
    /// `"strided"`.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.TiledLayout([8, 8]).outer_strides is None
    /// ```
    #[getter]
    pub fn outer_strides(slf: PyRef<'_, Self>) -> PyResult<Option<Py<PyTuple>>> {
        let core = tiled_of(&slf)?;
        match &core.outer_strides {
            Some(s) => Ok(Some(
                PyTuple::new(slf.py(), s.strides.iter().copied())?.unbind(),
            )),
            None => Ok(None),
        }
    }

    /// Within-tile strides in logical elements, or `None` unless the inner layout is
    /// `"strided"`.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.TiledLayout([8, 8]).inner_strides is None
    /// ```
    #[getter]
    pub fn inner_strides(slf: PyRef<'_, Self>) -> PyResult<Option<Py<PyTuple>>> {
        let core = tiled_of(&slf)?;
        match &core.inner_strides {
            Some(s) => Ok(Some(
                PyTuple::new(slf.py(), s.strides.iter().copied())?.unbind(),
            )),
            None => Ok(None),
        }
    }

    /// The nested tiling, or `None` unless the inner layout is `"tiled"`.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// outer = hurray.TiledLayout(
    ///     [64, 64], inner_layout="tiled", inner_tiled=hurray.TiledLayout([8, 8])
    /// )
    /// assert outer.inner_tiled.tile_shape == (8, 8)
    /// ```
    #[getter]
    pub fn inner_tiled(slf: PyRef<'_, Self>) -> PyResult<Option<Py<PyAny>>> {
        let core = tiled_of(&slf)?;
        match &core.inner_tiled {
            Some(inner) => {
                let base = Layout::of(LayoutDescriptor::Tiled(inner.clone()));
                Ok(Some(Py::new(slf.py(), base.init(TiledLayout))?.into_any()))
            }
            None => Ok(None),
        }
    }

    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert repr(hurray.TiledLayout([4, 4])).startswith("TiledLayout(tile_shape=(4, 4)")
    /// ```
    pub fn __repr__(slf: PyRef<'_, Self>) -> PyResult<String> {
        Ok(tiled_repr(tiled_of(&slf)?, 0))
    }
}

/// The core tiled descriptor behind a `TiledLayout` façade.
fn tiled_of<'a>(slf: &'a PyRef<'_, TiledLayout>) -> PyResult<&'a CoreTiled> {
    match slf.as_super().descriptor() {
        LayoutDescriptor::Tiled(l) => Ok(l),
        other => Err(variant_mismatch("TiledLayout", other)),
    }
}

/// Depth-aware `repr` for a possibly-nested tiling.
///
/// Bounded independently of core's construction-time depth check: a descriptor
/// decoded by a future core with a larger bound must still print, not recurse away.
fn tiled_repr(core: &CoreTiled, depth: usize) -> String {
    const MAX_REPR_DEPTH: usize = 4;
    let tile_shape = core
        .tile_shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let tile_shape = if core.tile_shape.len() == 1 {
        format!("({tile_shape},)")
    } else {
        format!("({tile_shape})")
    };
    let inner = match &core.inner_tiled {
        None => String::new(),
        Some(_) if depth + 1 >= MAX_REPR_DEPTH => ", inner_tiled=...".to_string(),
        Some(i) => format!(", inner_tiled={}", tiled_repr(i, depth + 1)),
    };
    format!(
        "TiledLayout(tile_shape={tile_shape}, outer_layout='{}', inner_layout='{}'{inner})",
        tag_name(core.outer_layout),
        tag_name(core.inner_layout),
    )
}

/// The tag byte for a tile-ordering name.
///
/// Only the orderings a tile may use are accepted; core rejects the rest anyway,
/// but a name that is a real layout elsewhere deserves a message saying so.
fn tag_for_tile_name(name: &str) -> PyResult<u8> {
    match name {
        "row_major" => Ok(TAG_ROW_MAJOR),
        "col_major" => Ok(TAG_COL_MAJOR),
        "strided" => Ok(TAG_STRIDED),
        "tiled" => Ok(TAG_TILED),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "invalid tile ordering {other:?}: expected 'row_major', 'col_major', \
             'strided', or 'tiled'"
        ))),
    }
}

// ── MortonLayout ──────────────────────────────────────────────────────────────

/// Morton (Z-order curve) layout. Tag `0x05`.
///
/// `morton_bits[k]` is the number of index bits interleaved for dimension `k`, so
/// dimension `k` may be at most `2 ** morton_bits[k]` long.
///
/// ## Examples (Python)
///
/// ```python
/// import hurray
///
/// l = hurray.MortonLayout([3, 3])
/// assert l.morton_bits == (3, 3)
/// ```
#[pyclass(name = "MortonLayout", extends = Layout, frozen)]
pub struct MortonLayout;

#[pymethods]
impl MortonLayout {
    /// Construct a Morton layout from the per-dimension bit counts.
    ///
    /// ## Errors
    ///
    /// - `hurray.InvalidDescriptorError` — a bit count core rejects.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.MortonLayout([4, 4]).name == "morton"
    /// ```
    #[new]
    pub fn new(morton_bits: Vec<u32>) -> PyResult<PyClassInitializer<Self>> {
        let core = hurray_core::layout::MortonLayout::new(morton_bits).map_err(layout_err)?;
        Ok(Layout::of(LayoutDescriptor::Morton(core)).init(Self))
    }

    /// The number of interleaved index bits per dimension.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.MortonLayout([3, 3]).morton_bits == (3, 3)
    /// ```
    #[getter]
    pub fn morton_bits(slf: PyRef<'_, Self>) -> PyResult<Py<PyTuple>> {
        match slf.as_super().descriptor() {
            LayoutDescriptor::Morton(l) => {
                Ok(PyTuple::new(slf.py(), l.morton_bits.iter().copied())?.unbind())
            }
            other => Err(variant_mismatch("MortonLayout", other)),
        }
    }

    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert repr(hurray.MortonLayout([3, 3])) == "MortonLayout(morton_bits=(3, 3))"
    /// ```
    pub fn __repr__(slf: PyRef<'_, Self>) -> PyResult<String> {
        let bits = Self::morton_bits(slf)?;
        Python::attach(|py| {
            Ok(format!(
                "MortonLayout(morton_bits={})",
                bits.bind(py).repr()?
            ))
        })
    }
}

// ── HilbertLayout ─────────────────────────────────────────────────────────────

/// Hilbert curve layout. Tag `0x40`.
///
/// Every dimension must be exactly `2 ** hilbert_order` long, and `hilbert_rank`
/// must equal the tensor's rank.
///
/// ## Examples (Python)
///
/// ```python
/// import hurray
///
/// l = hurray.HilbertLayout(3, 2)
/// assert l.hilbert_order == 3
/// assert l.hilbert_rank == 2
/// ```
#[pyclass(name = "HilbertLayout", extends = Layout, frozen)]
pub struct HilbertLayout;

#[pymethods]
impl HilbertLayout {
    /// Construct a Hilbert layout from its order and rank.
    ///
    /// ## Errors
    ///
    /// - `hurray.InvalidDescriptorError` — an order or rank core rejects.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.HilbertLayout(2, 2).tag == 0x40
    /// ```
    #[new]
    pub fn new(hilbert_order: u32, hilbert_rank: u32) -> PyResult<PyClassInitializer<Self>> {
        let core = hurray_core::layout::HilbertLayout::new(hilbert_order, hilbert_rank)
            .map_err(layout_err)?;
        Ok(Layout::of(LayoutDescriptor::Hilbert(core)).init(Self))
    }

    /// The curve order: each dimension is `2 ** hilbert_order` long.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.HilbertLayout(3, 2).hilbert_order == 3
    /// ```
    #[getter]
    pub fn hilbert_order(slf: PyRef<'_, Self>) -> PyResult<u32> {
        match slf.as_super().descriptor() {
            LayoutDescriptor::Hilbert(l) => Ok(l.hilbert_order),
            other => Err(variant_mismatch("HilbertLayout", other)),
        }
    }

    /// The curve rank, which must equal the tensor's rank.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.HilbertLayout(3, 2).hilbert_rank == 2
    /// ```
    #[getter]
    pub fn hilbert_rank(slf: PyRef<'_, Self>) -> PyResult<u32> {
        match slf.as_super().descriptor() {
            LayoutDescriptor::Hilbert(l) => Ok(l.hilbert_rank),
            other => Err(variant_mismatch("HilbertLayout", other)),
        }
    }

    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert repr(hurray.HilbertLayout(3, 2)) == "HilbertLayout(hilbert_order=3, hilbert_rank=2)"
    /// ```
    pub fn __repr__(slf: PyRef<'_, Self>) -> PyResult<String> {
        match slf.as_super().descriptor() {
            LayoutDescriptor::Hilbert(l) => Ok(format!(
                "HilbertLayout(hilbert_order={}, hilbert_rank={})",
                l.hilbert_order, l.hilbert_rank
            )),
            other => Err(variant_mismatch("HilbertLayout", other)),
        }
    }
}
