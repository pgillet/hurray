//! Sparse layout classes: COO, CSR, CSC, CSF.
//!
//! ## `nnz` is required
//!
//! Every sparse layout takes `nnz` as a required argument. A layout object is a
//! *declaration* about buffers the caller supplies separately, and `nnz` is the one
//! parameter that says how much of them is meaningful — inferring it would mean
//! silently rewriting the caller's descriptor to match whatever bytes arrived.
//!
//! Inference belongs to the array-shaped constructors (`hurray.sparse_coo`,
//! `hurray.from_scipy`), which are handed the arrays and can derive it honestly.

use pyo3::prelude::*;
use pyo3::types::PyTuple;

use hurray_core::layout::{CooLayout as CoreCoo, CscLayout as CoreCsc, CsrLayout as CoreCsr};
use hurray_core::{CsfLayout as CoreCsf, LayoutDescriptor};

use super::{variant_mismatch, Layout};

// ── CooLayout ─────────────────────────────────────────────────────────────────

/// Coordinate-list sparse layout. Tag `0x06`. Two buffers: values, then indices.
///
/// ## Examples (Python)
///
/// ```python
/// import hurray
///
/// l = hurray.CooLayout(nnz=2, is_sorted=True)
/// assert l.nnz == 2
/// assert l.is_sorted
/// assert l.buffer_count == 2
/// ```
#[pyclass(name = "CooLayout", extends = Layout, frozen)]
pub struct CooLayout;

#[pymethods]
impl CooLayout {
    /// Construct a COO layout.
    ///
    /// `is_sorted` claims the coordinates are in lexicographic order; it defaults to
    /// `False`, which claims nothing.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.CooLayout(nnz=4).is_sorted is False
    /// ```
    #[new]
    #[pyo3(signature = (nnz, is_sorted = false))]
    pub fn new(nnz: u64, is_sorted: bool) -> PyClassInitializer<Self> {
        Layout::of(LayoutDescriptor::Coo(CoreCoo::new(nnz, is_sorted))).init(Self)
    }

    /// The number of stored non-zero elements.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.CooLayout(nnz=7).nnz == 7
    /// ```
    #[getter]
    pub fn nnz(slf: PyRef<'_, Self>) -> PyResult<u64> {
        match slf.as_super().descriptor() {
            LayoutDescriptor::Coo(l) => Ok(l.nnz),
            other => Err(variant_mismatch("CooLayout", other)),
        }
    }

    /// Whether the coordinates are stored in lexicographic order.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.CooLayout(nnz=7, is_sorted=True).is_sorted
    /// ```
    #[getter]
    pub fn is_sorted(slf: PyRef<'_, Self>) -> PyResult<bool> {
        match slf.as_super().descriptor() {
            LayoutDescriptor::Coo(l) => Ok(l.is_sorted),
            other => Err(variant_mismatch("CooLayout", other)),
        }
    }

    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert repr(hurray.CooLayout(nnz=2)) == "CooLayout(nnz=2, is_sorted=False)"
    /// ```
    pub fn __repr__(slf: PyRef<'_, Self>) -> PyResult<String> {
        match slf.as_super().descriptor() {
            LayoutDescriptor::Coo(l) => Ok(format!(
                "CooLayout(nnz={}, is_sorted={})",
                l.nnz,
                if l.is_sorted { "True" } else { "False" }
            )),
            other => Err(variant_mismatch("CooLayout", other)),
        }
    }
}

// ── CsrLayout ─────────────────────────────────────────────────────────────────

/// Compressed-sparse-row layout. Tag `0x07`. Rank 2 only.
///
/// Three buffers: values, column indices, row pointers.
///
/// ## Examples (Python)
///
/// ```python
/// import hurray
///
/// l = hurray.CsrLayout(nnz=4)
/// assert l.nnz == 4
/// assert l.buffer_count == 3
/// ```
#[pyclass(name = "CsrLayout", extends = Layout, frozen)]
pub struct CsrLayout;

#[pymethods]
impl CsrLayout {
    /// Construct a CSR layout.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.CsrLayout(nnz=4).name == "csr"
    /// ```
    #[new]
    pub fn new(nnz: u64) -> PyClassInitializer<Self> {
        Layout::of(LayoutDescriptor::Csr(CoreCsr::new(nnz))).init(Self)
    }

    /// The number of stored non-zero elements.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.CsrLayout(nnz=4).nnz == 4
    /// ```
    #[getter]
    pub fn nnz(slf: PyRef<'_, Self>) -> PyResult<u64> {
        match slf.as_super().descriptor() {
            LayoutDescriptor::Csr(l) => Ok(l.nnz),
            other => Err(variant_mismatch("CsrLayout", other)),
        }
    }

    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert repr(hurray.CsrLayout(nnz=4)) == "CsrLayout(nnz=4)"
    /// ```
    pub fn __repr__(slf: PyRef<'_, Self>) -> PyResult<String> {
        Ok(format!("CsrLayout(nnz={})", Self::nnz(slf)?))
    }
}

// ── CscLayout ─────────────────────────────────────────────────────────────────

/// Compressed-sparse-column layout. Tag `0x08`. Rank 2 only.
///
/// Three buffers: values, row indices, column pointers.
///
/// ## Examples (Python)
///
/// ```python
/// import hurray
///
/// l = hurray.CscLayout(nnz=4)
/// assert l.nnz == 4
/// assert l.name == "csc"
/// ```
#[pyclass(name = "CscLayout", extends = Layout, frozen)]
pub struct CscLayout;

#[pymethods]
impl CscLayout {
    /// Construct a CSC layout.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.CscLayout(nnz=4).buffer_count == 3
    /// ```
    #[new]
    pub fn new(nnz: u64) -> PyClassInitializer<Self> {
        Layout::of(LayoutDescriptor::Csc(CoreCsc::new(nnz))).init(Self)
    }

    /// The number of stored non-zero elements.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.CscLayout(nnz=4).nnz == 4
    /// ```
    #[getter]
    pub fn nnz(slf: PyRef<'_, Self>) -> PyResult<u64> {
        match slf.as_super().descriptor() {
            LayoutDescriptor::Csc(l) => Ok(l.nnz),
            other => Err(variant_mismatch("CscLayout", other)),
        }
    }

    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert repr(hurray.CscLayout(nnz=4)) == "CscLayout(nnz=4)"
    /// ```
    pub fn __repr__(slf: PyRef<'_, Self>) -> PyResult<String> {
        Ok(format!("CscLayout(nnz={})", Self::nnz(slf)?))
    }
}

// ── CsfLayout ─────────────────────────────────────────────────────────────────

/// Compressed-sparse-fiber layout. Tag `0x09`. Rank 3 and above.
///
/// The rank-N generalisation of CSR/CSC: a tree of `rank` levels, each contributing
/// a `pos` and a `crd` buffer, plus one values buffer — `2 * rank + 1` in total.
/// Those buffers have no named accessors on `hurray.Tensor`; reach them with
/// `t.buffer(index)`, in the order this layout describes.
///
/// ## Examples (Python)
///
/// ```python
/// import hurray
///
/// l = hurray.CsfLayout(nnz=5, mode_order=[0, 1, 2])
/// assert l.mode_order == (0, 1, 2)
/// assert l.buffer_count == 7
/// ```
#[pyclass(name = "CsfLayout", extends = Layout, frozen)]
pub struct CsfLayout;

#[pymethods]
impl CsfLayout {
    /// Construct a CSF layout.
    ///
    /// `mode_order` is the order in which dimensions are nested, and must be a
    /// permutation of `range(rank)` — checked against the tensor's shape when a
    /// tensor is built with this layout.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.CsfLayout(nnz=5, mode_order=[2, 0, 1]).nnz == 5
    /// ```
    #[new]
    pub fn new(nnz: u64, mode_order: Vec<u32>) -> PyClassInitializer<Self> {
        Layout::of(LayoutDescriptor::Csf(CoreCsf::new(nnz, mode_order))).init(Self)
    }

    /// The number of stored non-zero elements.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.CsfLayout(nnz=5, mode_order=[0, 1, 2]).nnz == 5
    /// ```
    #[getter]
    pub fn nnz(slf: PyRef<'_, Self>) -> PyResult<u64> {
        match slf.as_super().descriptor() {
            LayoutDescriptor::Csf(l) => Ok(l.nnz),
            other => Err(variant_mismatch("CsfLayout", other)),
        }
    }

    /// The dimension nesting order.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.CsfLayout(nnz=5, mode_order=[2, 0, 1]).mode_order == (2, 0, 1)
    /// ```
    #[getter]
    pub fn mode_order(slf: PyRef<'_, Self>) -> PyResult<Py<PyTuple>> {
        match slf.as_super().descriptor() {
            LayoutDescriptor::Csf(l) => {
                Ok(PyTuple::new(slf.py(), l.mode_order.iter().copied())?.unbind())
            }
            other => Err(variant_mismatch("CsfLayout", other)),
        }
    }

    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert repr(hurray.CsfLayout(nnz=5, mode_order=[0, 1, 2])) == \
    ///     "CsfLayout(nnz=5, mode_order=(0, 1, 2))"
    /// ```
    pub fn __repr__(slf: PyRef<'_, Self>) -> PyResult<String> {
        match slf.as_super().descriptor() {
            LayoutDescriptor::Csf(l) => {
                let nnz = l.nnz;
                let modes = PyTuple::new(slf.py(), l.mode_order.iter().copied())?;
                Ok(format!(
                    "CsfLayout(nnz={nnz}, mode_order={})",
                    modes.repr()?
                ))
            }
            other => Err(variant_mismatch("CsfLayout", other)),
        }
    }
}
