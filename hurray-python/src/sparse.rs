//! Python bindings for the Hurray sparse tensor type.
//!
//! Exposes [`SparseTensor`] as `hurray.SparseTensor`: a Python class that wraps
//! a sparse tensor descriptor and its component buffers (values + index arrays).
//!
//! ## Supported formats (D10)
//!
//! | Format | `SparseFormat` | Buffers |
//! |--------|---------------|---------|
//! | COO | `Coo` | values, indices `[nnz, rank]` uint64 |
//! | CSR | `Csr` | values, col_indices `[nnz]` uint64, row_ptr `[nrows+1]` uint64 |
//! | CSC | `Csc` | values, row_indices `[nnz]` uint64, col_ptr `[ncols+1]` uint64 |
//!
//! Attributes that don't apply to the active format raise `AttributeError` (D10).

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::IntoPyObjectExt;

use hurray_core::{
    layout::{CooLayout, CscLayout, CsrLayout},
    BufferHandle, ElementType, LayoutDescriptor, MemoryClass, Shape, SyncMode, TensorDescriptor,
    DESCRIPTOR_VERSION_MAJOR, DESCRIPTOR_VERSION_MINOR,
};

use crate::buffer::BufferStore;
use crate::device::Device;
use crate::dtype::Dtype;
use crate::errors::{BufferError, InvalidDescriptorError, UnsupportedError};
use crate::tensor::{is_tier1, Tensor};

// ── SparseFormat ──────────────────────────────────────────────────────────────

/// Which sparse storage format a [`SparseTensor`] uses.
///
/// D10: a single `SparseTensor` pyclass covers all three formats; format-specific
/// attributes raise `AttributeError` when accessed on a tensor of a different format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseFormat {
    Coo,
    Csr,
    Csc,
}

impl SparseFormat {
    fn as_str(self) -> &'static str {
        match self {
            SparseFormat::Coo => "coo",
            SparseFormat::Csr => "csr",
            SparseFormat::Csc => "csc",
        }
    }
}

// ── SparseTensor pyclass ──────────────────────────────────────────────────────

/// A Hurray sparse tensor: COO, CSR, or CSC format with zero-copy buffer access.
///
/// ## Construction
///
/// Use [`hurray.from_scipy`] to wrap a SciPy sparse matrix without copying, or
/// construct directly from component buffers (values + index arrays).
///
/// ## Format-specific attributes
///
/// | Attribute | COO | CSR | CSC |
/// |-----------|-----|-----|-----|
/// | `.values` | yes | yes | yes |
/// | `.indices` | yes | — | — |
/// | `.col_indices` | — | yes | — |
/// | `.row_ptr` | — | yes | — |
/// | `.row_indices` | — | — | yes |
/// | `.col_ptr` | — | — | yes |
///
/// Accessing an attribute that does not apply to the current format raises
/// `AttributeError`.
///
/// ## Examples (Python)
///
/// ```python
/// import numpy as np, hurray
///
/// # Construct a 3×3 CSR matrix with 4 non-zeros.
/// values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
/// col_idx = np.array([0, 2, 1, 2], dtype=np.uint64)
/// row_ptr = np.array([0, 1, 3, 4], dtype=np.uint64)
///
/// # (Use hurray.from_scipy for zero-copy from SciPy matrices.)
/// ```
#[pyclass(name = "SparseTensor")]
#[derive(Debug)]
pub struct SparseTensor {
    /// Tensor descriptor carrying element type, shape, layout, and buffer metadata.
    pub descriptor: TensorDescriptor,
    /// Which sparse storage format this tensor uses.
    pub format: SparseFormat,
    /// Buffer for the non-zero values array.
    pub values_buf: BufferStore,
    /// Buffer for the primary index array:
    /// - COO: `indices` `[nnz * rank]` uint64
    /// - CSR: `col_indices` `[nnz]` uint64
    /// - CSC: `row_indices` `[nnz]` uint64
    pub aux_buf_0: BufferStore,
    /// Buffer for the secondary index array (CSR/CSC only):
    /// - CSR: `row_ptr` `[nrows + 1]` uint64
    /// - CSC: `col_ptr` `[ncols + 1]` uint64
    /// - COO: `None`
    pub aux_buf_1: Option<BufferStore>,
    /// Python-side dtype handle for the values element type.
    pub dtype_py: Py<Dtype>,
    /// Python-side device handle.
    pub device_py: Py<Device>,
}

#[pymethods]
impl SparseTensor {
    // ── Properties ────────────────────────────────────────────────────────────

    /// The sparse storage format string: `"coo"`, `"csr"`, or `"csc"`.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert sparse.format == "csr"
    /// ```
    #[getter]
    pub fn format(&self) -> &str {
        self.format.as_str()
    }

    /// The logical shape of this sparse tensor as a tuple of `int`.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert sparse.shape == (1000, 1000)
    /// ```
    #[getter]
    pub fn shape(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        use hurray_core::DYNAMIC;
        let items: Vec<Py<PyAny>> = self
            .descriptor
            .shape
            .dims()
            .iter()
            .map(|&dim| -> PyResult<Py<PyAny>> {
                if dim == DYNAMIC {
                    Ok(py.None())
                } else {
                    dim.into_py_any(py)
                }
            })
            .collect::<PyResult<_>>()?;
        Ok(PyTuple::new(py, items)?.unbind())
    }

    /// Number of dimensions (rank) of this sparse tensor.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert sparse.ndim == 2
    /// ```
    #[getter]
    pub fn ndim(&self) -> usize {
        self.descriptor.shape.rank()
    }

    /// Number of explicitly stored (non-zero) elements.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert sparse.nnz == 42
    /// ```
    #[getter]
    pub fn nnz(&self) -> u64 {
        match &self.descriptor.layout {
            LayoutDescriptor::Coo(l) => l.nnz,
            LayoutDescriptor::Csr(l) => l.nnz,
            LayoutDescriptor::Csc(l) => l.nnz,
            // Unreachable: SparseTensor is only constructed with COO/CSR/CSC layouts.
            _ => 0,
        }
    }

    /// The element type of the non-zero values.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert sparse.dtype == hurray.float32
    /// ```
    #[getter]
    pub fn dtype(&self, py: Python<'_>) -> Py<Dtype> {
        self.dtype_py.clone_ref(py)
    }

    /// The device this tensor's buffers reside on.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert sparse.device.kind == "cpu"
    /// ```
    #[getter]
    pub fn device(&self, py: Python<'_>) -> Py<Device> {
        self.device_py.clone_ref(py)
    }

    // ── Component buffer accessors ────────────────────────────────────────────

    /// A `hurray.Tensor` view over the non-zero values buffer.
    ///
    /// Shape: `[nnz]`, dtype: same as `self.dtype`. Zero-copy — the view borrows
    /// this `SparseTensor`'s buffer.
    ///
    /// ## Examples
    ///
    /// ```python
    /// vals = sparse.values
    /// assert vals.shape == (sparse.nnz,)
    /// ```
    #[getter]
    pub fn values(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let s = slf.borrow();
        let nnz = s.nnz();
        let element_type = s.descriptor.element_type;
        let device_tag = s.device_py.borrow(py).tag;
        let memory_class = s.device_py.borrow(py).memory_class;
        let ptr = s.values_buf.as_ptr() as *mut u8;
        let len = s.values_buf.len();
        drop(s);

        let shape = Shape::new(vec![nnz])
            .map_err(|e| InvalidDescriptorError::new_err(format!("invalid shape: {e}")))?;
        // Keep this SparseTensor alive as the base for the borrowed Tensor view.
        let base: Py<PyAny> = slf.clone().into_any().unbind();
        let tensor = Tensor::new_borrowed_view(
            py,
            element_type,
            shape,
            device_tag,
            memory_class,
            base,
            ptr,
            len,
        )?;
        Ok(Py::new(py, tensor)?.into_any())
    }

    /// A `hurray.Tensor` view over the COO index buffer (shape `[nnz, rank]`, dtype `uint64`).
    ///
    /// Raises `AttributeError` for non-COO tensors (D10).
    ///
    /// ## Examples
    ///
    /// ```python
    /// idx = sparse.indices  # only valid for COO
    /// assert idx.shape == (sparse.nnz, sparse.ndim)
    /// ```
    #[getter]
    pub fn indices(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let s = slf.borrow();
        // D10: attributes that don't apply to the current format raise AttributeError.
        if s.format != SparseFormat::Coo {
            return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
                "'SparseTensor' object has no attribute 'indices'; this is a {} tensor",
                s.format.as_str()
            )));
        }
        let nnz = s.nnz();
        let rank = s.descriptor.shape.rank() as u64;
        let ptr = s.aux_buf_0.as_ptr() as *mut u8;
        let len = s.aux_buf_0.len();
        drop(s);

        let shape = Shape::new(vec![nnz, rank])
            .map_err(|e| InvalidDescriptorError::new_err(format!("invalid shape: {e}")))?;
        let base: Py<PyAny> = slf.clone().into_any().unbind();
        build_uint64_view(py, base, ptr, len, shape)
    }

    /// A `hurray.Tensor` view over the CSR column-index buffer (shape `[nnz]`, dtype `uint64`).
    ///
    /// Raises `AttributeError` for non-CSR tensors (D10).
    ///
    /// ## Examples
    ///
    /// ```python
    /// col_idx = sparse.col_indices  # only valid for CSR
    /// assert col_idx.shape == (sparse.nnz,)
    /// ```
    #[getter]
    pub fn col_indices(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let s = slf.borrow();
        // D10: attributes that don't apply to the current format raise AttributeError.
        if s.format != SparseFormat::Csr {
            return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
                "'SparseTensor' object has no attribute 'col_indices'; this is a {} tensor",
                s.format.as_str()
            )));
        }
        let nnz = s.nnz();
        let ptr = s.aux_buf_0.as_ptr() as *mut u8;
        let len = s.aux_buf_0.len();
        drop(s);

        let shape = Shape::new(vec![nnz])
            .map_err(|e| InvalidDescriptorError::new_err(format!("invalid shape: {e}")))?;
        let base: Py<PyAny> = slf.clone().into_any().unbind();
        build_uint64_view(py, base, ptr, len, shape)
    }

    /// A `hurray.Tensor` view over the CSR row-pointer buffer (shape `[nrows+1]`, dtype `uint64`).
    ///
    /// Raises `AttributeError` for non-CSR tensors (D10).
    ///
    /// ## Examples
    ///
    /// ```python
    /// rp = sparse.row_ptr  # only valid for CSR
    /// assert rp.shape == (sparse.shape[0] + 1,)
    /// ```
    #[getter]
    pub fn row_ptr(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let s = slf.borrow();
        // D10: attributes that don't apply to the current format raise AttributeError.
        if s.format != SparseFormat::Csr {
            return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
                "'SparseTensor' object has no attribute 'row_ptr'; this is a {} tensor",
                s.format.as_str()
            )));
        }
        let nrows = s.descriptor.shape.dims().first().copied().ok_or_else(|| {
            InvalidDescriptorError::new_err("CSR tensor shape must have at least 1 dimension")
        })?;
        let buf = s
            .aux_buf_1
            .as_ref()
            .ok_or_else(|| InvalidDescriptorError::new_err("CSR tensor missing row_ptr buffer"))?;
        let ptr = buf.as_ptr() as *mut u8;
        let len = buf.len();
        drop(s);

        let shape = Shape::new(vec![nrows + 1])
            .map_err(|e| InvalidDescriptorError::new_err(format!("invalid shape: {e}")))?;
        let base: Py<PyAny> = slf.clone().into_any().unbind();
        build_uint64_view(py, base, ptr, len, shape)
    }

    /// A `hurray.Tensor` view over the CSC row-index buffer (shape `[nnz]`, dtype `uint64`).
    ///
    /// Raises `AttributeError` for non-CSC tensors (D10).
    ///
    /// ## Examples
    ///
    /// ```python
    /// row_idx = sparse.row_indices  # only valid for CSC
    /// assert row_idx.shape == (sparse.nnz,)
    /// ```
    #[getter]
    pub fn row_indices(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let s = slf.borrow();
        // D10: attributes that don't apply to the current format raise AttributeError.
        if s.format != SparseFormat::Csc {
            return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
                "'SparseTensor' object has no attribute 'row_indices'; this is a {} tensor",
                s.format.as_str()
            )));
        }
        let nnz = s.nnz();
        let ptr = s.aux_buf_0.as_ptr() as *mut u8;
        let len = s.aux_buf_0.len();
        drop(s);

        let shape = Shape::new(vec![nnz])
            .map_err(|e| InvalidDescriptorError::new_err(format!("invalid shape: {e}")))?;
        let base: Py<PyAny> = slf.clone().into_any().unbind();
        build_uint64_view(py, base, ptr, len, shape)
    }

    /// A `hurray.Tensor` view over the CSC column-pointer buffer (shape `[ncols+1]`, dtype `uint64`).
    ///
    /// Raises `AttributeError` for non-CSC tensors (D10).
    ///
    /// ## Examples
    ///
    /// ```python
    /// cp = sparse.col_ptr  # only valid for CSC
    /// assert cp.shape == (sparse.shape[1] + 1,)
    /// ```
    #[getter]
    pub fn col_ptr(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let s = slf.borrow();
        // D10: attributes that don't apply to the current format raise AttributeError.
        if s.format != SparseFormat::Csc {
            return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
                "'SparseTensor' object has no attribute 'col_ptr'; this is a {} tensor",
                s.format.as_str()
            )));
        }
        let ncols = s.descriptor.shape.dims().get(1).copied().ok_or_else(|| {
            InvalidDescriptorError::new_err("CSC tensor shape must have at least 2 dimensions")
        })?;
        let buf = s
            .aux_buf_1
            .as_ref()
            .ok_or_else(|| InvalidDescriptorError::new_err("CSC tensor missing col_ptr buffer"))?;
        let ptr = buf.as_ptr() as *mut u8;
        let len = buf.len();
        drop(s);

        let shape = Shape::new(vec![ncols + 1])
            .map_err(|e| InvalidDescriptorError::new_err(format!("invalid shape: {e}")))?;
        let base: Py<PyAny> = slf.clone().into_any().unbind();
        build_uint64_view(py, base, ptr, len, shape)
    }

    // ── SciPy interop ─────────────────────────────────────────────────────────

    /// Convert this sparse tensor to a `scipy.sparse` matrix (zero-copy where possible).
    ///
    /// `copy=False` is passed to the SciPy constructor. SciPy may copy internally
    /// if it does not accept the `uint64` index dtype — this is outside Hurray's
    /// control and is documented here so callers can plan accordingly.
    ///
    /// ## Raises
    ///
    /// - `ImportError` — SciPy is not installed.
    /// - `hurray.UnsupportedError` — values dtype is Tier 2 / quantized (SciPy has
    ///   no equivalent) (D17).
    /// - `hurray.UnsupportedError` — COO tensors: use `.values` / `.indices` directly
    ///   and construct `scipy.sparse.coo_matrix` manually.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import scipy.sparse, hurray
    ///
    /// # (Assuming `sparse` is a hurray.SparseTensor with format="csr")
    /// m = sparse.to_scipy()
    /// assert isinstance(m, scipy.sparse.csr_matrix)
    /// ```
    pub fn to_scipy(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        {
            let s = slf.borrow();
            // D17: reject Tier 2 / quantized dtypes — SciPy has no equivalent.
            if !is_tier1(s.descriptor.element_type) {
                return Err(UnsupportedError::new_err(format!(
                    "to_scipy is not supported for '{}'; SciPy has no equivalent dtype",
                    crate::dtype::element_type_name(s.descriptor.element_type),
                )));
            }
            if s.format == SparseFormat::Coo {
                return Err(UnsupportedError::new_err(
                    "to_scipy is not supported for COO tensors; access .values and .indices \
                     directly and construct scipy.sparse.coo_matrix manually",
                ));
            }
        }

        // D7-pattern: import scipy lazily — hurray must not require scipy at module load.
        let sp = py.import("scipy.sparse").map_err(|_| {
            pyo3::exceptions::PyImportError::new_err(
                "scipy is not installed; install with: pip install scipy",
            )
        })?;

        // Build numpy view of values via __array__ (goes through DLPack; Tier 1 is safe).
        let values_py = SparseTensor::values(slf)?;
        let values_arr = values_py.bind(py).call_method0("__array__")?;

        let (nrows, ncols, nnz, fmt, aux1_len_elems) = {
            let s = slf.borrow();
            let dims = s.descriptor.shape.dims().to_vec();
            let nrows = dims[0];
            let ncols = dims[1];
            let nnz = s.nnz();
            let fmt = s.format;
            let aux1_len = match fmt {
                SparseFormat::Csr => nrows + 1,
                SparseFormat::Csc => ncols + 1,
                SparseFormat::Coo => unreachable!(),
            };
            (nrows, ncols, nnz, fmt, aux1_len)
        };

        // Build numpy views of index arrays directly from aux buffers.
        let aux0_shape = Shape::new(vec![nnz])
            .map_err(|e| InvalidDescriptorError::new_err(format!("invalid shape: {e}")))?;
        let aux0_py = {
            let s = slf.borrow();
            let ptr = s.aux_buf_0.as_ptr() as *mut u8;
            let len = s.aux_buf_0.len();
            drop(s);
            let base: Py<PyAny> = slf.clone().into_any().unbind();
            build_uint64_view(py, base, ptr, len, aux0_shape)?
        };
        let aux0_arr = aux0_py.bind(py).call_method0("__array__")?;

        let aux1_shape = Shape::new(vec![aux1_len_elems])
            .map_err(|e| InvalidDescriptorError::new_err(format!("invalid shape: {e}")))?;
        let aux1_py = {
            let s = slf.borrow();
            let buf = s.aux_buf_1.as_ref().ok_or_else(|| {
                InvalidDescriptorError::new_err("CSR/CSC tensor missing secondary index buffer")
            })?;
            let ptr = buf.as_ptr() as *mut u8;
            let len = buf.len();
            drop(s);
            let base: Py<PyAny> = slf.clone().into_any().unbind();
            build_uint64_view(py, base, ptr, len, aux1_shape)?
        };
        let aux1_arr = aux1_py.bind(py).call_method0("__array__")?;

        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("shape", PyTuple::new(py, [nrows, ncols])?)?;
        kwargs.set_item("copy", false)?;

        // csr_matrix((data, indices, indptr), shape=..., copy=False)
        let args = PyTuple::new(py, [&values_arr, &aux0_arr, &aux1_arr])?;
        let constructor_name = match fmt {
            SparseFormat::Csr => "csr_matrix",
            SparseFormat::Csc => "csc_matrix",
            SparseFormat::Coo => unreachable!(),
        };
        let matrix = sp.getattr(constructor_name)?.call((args,), Some(&kwargs))?;
        Ok(matrix.into())
    }

    // ── Dunders ───────────────────────────────────────────────────────────────

    /// Sparse tensors are unhashable — mutable objects must not be used as dict keys.
    pub fn __hash__(&self) -> PyResult<isize> {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "unhashable type: 'SparseTensor'",
        ))
    }

    /// Return a developer-friendly string representation.
    ///
    /// The output format is controlled by [`hurray.set_print_options`] /
    /// [`hurray.print_options`]:
    ///
    /// - `sparse_display="metadata"` *(default)*: compact SciPy-style summary:
    ///
    ///   ```text
    ///   hurray.SparseTensor(format='csr', shape=(3, 3), nnz=4, dtype=float32)
    ///   ```
    ///
    /// - `sparse_display="content"`: includes the per-format buffer arrays,
    ///   formatted via NumPy's own string representation:
    ///
    ///   ```text
    ///   hurray.SparseTensor(format='csr', shape=(3, 3), nnz=4, dtype=float32,
    ///       values=[1. 2. 3. 4.], col_indices=[0 2 1 2], row_ptr=[0 1 3 4])
    ///   ```
    ///
    ///   If NumPy is unavailable or array formatting fails, falls back silently
    ///   to the `"metadata"` string — `__repr__` never raises.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    /// import scipy.sparse as sp
    /// m = sp.csr_matrix(([1.0, 2.0], ([0, 1], [1, 0])), shape=(2, 2))
    /// t = hurray.from_scipy(m)
    /// print(repr(t))
    /// # hurray.SparseTensor(format='csr', shape=(2, 2), nnz=2, dtype=float64)
    ///
    /// with hurray.print_options(sparse_display="content"):
    ///     print(repr(t))
    /// # hurray.SparseTensor(format='csr', shape=(2, 2), nnz=2, dtype=float64,
    /// #     values=[1. 2.], col_indices=[1 0], row_ptr=[0 1 2])
    /// ```
    pub fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let s = slf.borrow();
        let shape_tuple = s.shape(py)?;
        let shape_str = shape_tuple.bind(py).repr()?.to_str()?.to_owned();
        let dtype_name = crate::dtype::element_type_name(s.descriptor.element_type);
        let fmt = s.format;
        let nnz = s.nnz();
        drop(s);

        // Build the invariant metadata prefix that both modes share.
        let metadata = format!(
            "hurray.SparseTensor(format='{}', shape={}, nnz={}, dtype={}",
            fmt.as_str(),
            shape_str,
            nnz,
            dtype_name,
        );

        // Check the ContextVar. Falls back to false (metadata) on any Python error.
        if !crate::print_options::sparse_display_is_content(py) {
            return Ok(format!("{metadata})"));
        }

        // Content mode: append per-format buffer arrays using NumPy formatting.
        // On any failure, fall back to the metadata string — __repr__ must not raise.
        match format_content_arrays(slf) {
            Ok(array_part) => Ok(format!("{metadata}, {array_part})")),
            Err(_) => Ok(format!("{metadata})")),
        }
    }

    /// Return the same string as `__repr__`.
    ///
    /// Respects the current `sparse_display` print option. See [`__repr__`] for
    /// full documentation.
    pub fn __str__(slf: &Bound<'_, Self>) -> PyResult<String> {
        SparseTensor::__repr__(slf)
    }
}

// ── Crate-internal helpers ────────────────────────────────────────────────────

/// Format the buffer arrays for `"content"` display mode.
///
/// Converts each format-appropriate index array and the values array to a NumPy
/// array (via the existing `__array__` accessor) and then calls NumPy's own
/// `array2string` for formatting, so the output respects the user's current
/// NumPy print options (precision, threshold, linewidth).
///
/// Returns a `key=[…], key=[…]` string (without the surrounding parens) ready
/// to be appended to the metadata prefix.
///
/// # Errors
///
/// Returns a `PyErr` if NumPy cannot be imported or any array conversion fails.
/// The caller (`__repr__`) silently falls back to `"metadata"` on error.
fn format_content_arrays(slf: &Bound<'_, SparseTensor>) -> PyResult<String> {
    let py = slf.py();
    // Content rendering needs numpy (the Tensor views format through it). If numpy is
    // absent, bail here so __repr__ falls back cleanly to the metadata string.
    py.import("numpy")?;

    let fmt = slf.borrow().format;

    /// Render a `hurray.Tensor` buffer view as its bare NumPy array string.
    ///
    /// Delegates to the Tensor's own `__str__` (the `numpy.frombuffer` path used by
    /// dense display) rather than `__array__`/DLPack, which does not work for the
    /// borrowed buffer views the sparse accessors return. `__str__` yields the compact
    /// inline `[1. 2. 3.]` form (NumPy's default separator), matching PyTorch's style.
    fn tensor_to_str(py: Python<'_>, tensor_obj: Py<PyAny>) -> PyResult<String> {
        tensor_obj.bind(py).str()?.extract::<String>()
    }

    let values_str = tensor_to_str(py, SparseTensor::values(slf)?)?;

    let array_part = match fmt {
        SparseFormat::Coo => {
            let indices_str = tensor_to_str(py, SparseTensor::indices(slf)?)?;
            format!("indices={indices_str}, values={values_str}")
        }
        SparseFormat::Csr => {
            let col_indices_str = tensor_to_str(py, SparseTensor::col_indices(slf)?)?;
            let row_ptr_str = tensor_to_str(py, SparseTensor::row_ptr(slf)?)?;
            format!("values={values_str}, col_indices={col_indices_str}, row_ptr={row_ptr_str}")
        }
        SparseFormat::Csc => {
            let row_indices_str = tensor_to_str(py, SparseTensor::row_indices(slf)?)?;
            let col_ptr_str = tensor_to_str(py, SparseTensor::col_ptr(slf)?)?;
            format!("values={values_str}, row_indices={row_indices_str}, col_ptr={col_ptr_str}")
        }
    };

    Ok(array_part)
}

/// Build a uint64 `hurray.Tensor` view over a raw buffer pointer.
///
/// `base` is kept alive by the resulting Tensor's `BufferStore`, ensuring `ptr`
/// remains valid for the view's lifetime.
///
/// # Safety
///
/// `ptr` must be valid for `len` bytes for as long as `base` is alive.
pub(crate) fn build_uint64_view(
    py: Python<'_>,
    base: Py<PyAny>,
    ptr: *mut u8,
    len: usize,
    shape: Shape,
) -> PyResult<Py<PyAny>> {
    let tensor = Tensor::new_borrowed_view(
        py,
        ElementType::Uint64,
        shape,
        hurray_core::DeviceTag::Cpu,
        MemoryClass::Standard,
        base,
        ptr,
        len,
    )?;
    Ok(Py::new(py, tensor)?.into_any())
}

/// Build a `SparseTensor` from three pre-validated component buffers and a descriptor.
///
/// Caller is responsible for ensuring `values_buf`, `aux_buf_0`, `aux_buf_1` are
/// consistent with `descriptor` and `format`.
pub(crate) fn make_sparse_tensor(
    py: Python<'_>,
    descriptor: TensorDescriptor,
    format: SparseFormat,
    values_buf: BufferStore,
    aux_buf_0: BufferStore,
    aux_buf_1: Option<BufferStore>,
) -> PyResult<SparseTensor> {
    let element_type = descriptor.element_type;
    let dtype_py = Py::new(
        py,
        Dtype {
            inner: element_type,
        },
    )?;
    let device_py = Py::new(
        py,
        Device {
            tag: hurray_core::DeviceTag::Cpu,
            memory_class: MemoryClass::Standard,
            device_id: 0,
        },
    )?;
    Ok(SparseTensor {
        descriptor,
        format,
        values_buf,
        aux_buf_0,
        aux_buf_1,
        dtype_py,
        device_py,
    })
}

// ── Alignment helper ──────────────────────────────────────────────────────────

/// Return the declared alignment for a buffer of `byte_size` bytes from Python.
///
/// Python/NumPy/SciPy allocators always produce at least 64-byte-aligned data,
/// so declaring `MIN_BUFFER_ALIGNMENT` for non-empty buffers is correct and
/// satisfies hurray-core's enforcement (`alignment ≥ 64` for non-empty buffers).
/// Empty buffers require only alignment=1 (the minimum valid power-of-two).
#[inline]
fn py_buf_alignment(byte_size: u64) -> u32 {
    if byte_size == 0 {
        1
    } else {
        hurray_core::MIN_BUFFER_ALIGNMENT
    }
}

/// Build the three `BufferHandle`s needed for a CSR/CSC descriptor.
pub(crate) fn build_three_buffer_handles(
    values_len: u64,
    aux0_len: u64,
    aux1_len: u64,
) -> PyResult<(BufferHandle, BufferHandle, BufferHandle)> {
    let bh_values = BufferHandle::new(
        values_len,
        py_buf_alignment(values_len),
        hurray_core::DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )
    .map_err(|e| BufferError::new_err(format!("values buffer: {e}")))?;
    let bh_aux0 = BufferHandle::new(
        aux0_len,
        py_buf_alignment(aux0_len),
        hurray_core::DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )
    .map_err(|e| BufferError::new_err(format!("index buffer 0: {e}")))?;
    let bh_aux1 = BufferHandle::new(
        aux1_len,
        py_buf_alignment(aux1_len),
        hurray_core::DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )
    .map_err(|e| BufferError::new_err(format!("index buffer 1: {e}")))?;
    Ok((bh_values, bh_aux0, bh_aux1))
}

/// Build a COO `TensorDescriptor`.
// TODO: activate in the COO Python construction pass (follow-on to 8a.4).
#[allow(dead_code)]
pub(crate) fn build_coo_descriptor(
    element_type: ElementType,
    shape: Shape,
    nnz: u64,
    values_len: u64,
    indices_len: u64,
) -> PyResult<TensorDescriptor> {
    let bh_values = BufferHandle::new(
        values_len,
        py_buf_alignment(values_len),
        hurray_core::DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )
    .map_err(|e| BufferError::new_err(format!("values buffer: {e}")))?;
    let bh_indices = BufferHandle::new(
        indices_len,
        py_buf_alignment(indices_len),
        hurray_core::DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )
    .map_err(|e| BufferError::new_err(format!("indices buffer: {e}")))?;

    TensorDescriptor::new(
        DESCRIPTOR_VERSION_MAJOR,
        DESCRIPTOR_VERSION_MINOR,
        element_type,
        shape,
        0,
        LayoutDescriptor::Coo(CooLayout::new(nnz, false)),
        vec![bh_values, bh_indices],
        None,
        None,
        None,
        None,
    )
    .map_err(|e| InvalidDescriptorError::new_err(format!("invalid descriptor: {e}")))
}

/// Build a CSR `TensorDescriptor`.
pub(crate) fn build_csr_descriptor(
    element_type: ElementType,
    shape: Shape,
    nnz: u64,
    values_len: u64,
    aux0_len: u64,
    aux1_len: u64,
) -> PyResult<TensorDescriptor> {
    let (bh_v, bh_0, bh_1) = build_three_buffer_handles(values_len, aux0_len, aux1_len)?;
    TensorDescriptor::new(
        DESCRIPTOR_VERSION_MAJOR,
        DESCRIPTOR_VERSION_MINOR,
        element_type,
        shape,
        0,
        LayoutDescriptor::Csr(CsrLayout::new(nnz)),
        vec![bh_v, bh_0, bh_1],
        None,
        None,
        None,
        None,
    )
    .map_err(|e| InvalidDescriptorError::new_err(format!("invalid descriptor: {e}")))
}

/// Build a CSC `TensorDescriptor`.
pub(crate) fn build_csc_descriptor(
    element_type: ElementType,
    shape: Shape,
    nnz: u64,
    values_len: u64,
    aux0_len: u64,
    aux1_len: u64,
) -> PyResult<TensorDescriptor> {
    let (bh_v, bh_0, bh_1) = build_three_buffer_handles(values_len, aux0_len, aux1_len)?;
    TensorDescriptor::new(
        DESCRIPTOR_VERSION_MAJOR,
        DESCRIPTOR_VERSION_MINOR,
        element_type,
        shape,
        0,
        LayoutDescriptor::Csc(CscLayout::new(nnz)),
        vec![bh_v, bh_0, bh_1],
        None,
        None,
        None,
        None,
    )
    .map_err(|e| InvalidDescriptorError::new_err(format!("invalid descriptor: {e}")))
}

// ── Registration ──────────────────────────────────────────────────────────────

/// Register `SparseTensor` on the `hurray` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SparseTensor>()?;
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use hurray_core::ElementType;
    use pyo3::Python;

    fn init() {
        pyo3::Python::initialize();
    }

    fn build_module(py: Python<'_>) -> pyo3::Bound<'_, pyo3::types::PyModule> {
        let m = pyo3::types::PyModule::new(py, "hurray").unwrap();
        crate::errors::register(&m).unwrap();
        crate::dtype::register(&m).unwrap();
        crate::device::register(&m).unwrap();
        crate::tensor::register(&m).unwrap();
        crate::print_options::register(&m).unwrap();
        register(&m).unwrap();
        let sys = py.import("sys").unwrap();
        let modules = sys.getattr("modules").unwrap();
        modules.set_item("hurray", &m).unwrap();
        m
    }

    /// Build a minimal COO SparseTensor for tests.
    fn make_coo(py: Python<'_>) -> SparseTensor {
        // 2×2 matrix, 2 non-zeros, float32 values
        let values: Vec<u8> = [5.0f32, 7.0].iter().flat_map(|f| f.to_le_bytes()).collect();
        // indices: [[0,0],[1,1]] stored row-major as [0,0,1,1]
        let indices: Vec<u8> = [0u64, 0, 1, 1]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let shape = Shape::new(vec![2, 2]).unwrap();
        let bh_values = BufferHandle::new(
            values.len() as u64,
            py_buf_alignment(values.len() as u64),
            hurray_core::DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )
        .unwrap();
        let bh_idx = BufferHandle::new(
            indices.len() as u64,
            py_buf_alignment(indices.len() as u64),
            hurray_core::DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )
        .unwrap();
        use hurray_core::layout::CooLayout;
        use hurray_core::{LayoutDescriptor, DESCRIPTOR_VERSION_MAJOR, DESCRIPTOR_VERSION_MINOR};
        let descriptor = TensorDescriptor::new(
            DESCRIPTOR_VERSION_MAJOR,
            DESCRIPTOR_VERSION_MINOR,
            ElementType::Float32,
            shape,
            0,
            LayoutDescriptor::Coo(CooLayout::new(2, false)),
            vec![bh_values, bh_idx],
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let dtype_py = Py::new(
            py,
            Dtype {
                inner: ElementType::Float32,
            },
        )
        .unwrap();
        let device_py = Py::new(
            py,
            Device {
                tag: hurray_core::DeviceTag::Cpu,
                memory_class: MemoryClass::Standard,
                device_id: 0,
            },
        )
        .unwrap();
        SparseTensor {
            descriptor,
            format: SparseFormat::Coo,
            values_buf: BufferStore::from_slice(&values),
            aux_buf_0: BufferStore::from_slice(&indices),
            aux_buf_1: None,
            dtype_py,
            device_py,
        }
    }

    /// Build a minimal CSC SparseTensor for tests.
    fn make_csc(py: Python<'_>) -> SparseTensor {
        // 2×2 matrix, 2 non-zeros, float32 values
        let values: Vec<u8> = [3.0f32, 6.0].iter().flat_map(|f| f.to_le_bytes()).collect();
        let row_indices: Vec<u8> = [0u64, 1].iter().flat_map(|v| v.to_le_bytes()).collect();
        let col_ptr: Vec<u8> = [0u64, 1, 2].iter().flat_map(|v| v.to_le_bytes()).collect();

        let shape = Shape::new(vec![2, 2]).unwrap();
        let descriptor = build_csc_descriptor(
            ElementType::Float32,
            shape,
            2,
            values.len() as u64,
            row_indices.len() as u64,
            col_ptr.len() as u64,
        )
        .unwrap();

        let dtype_py = Py::new(
            py,
            Dtype {
                inner: ElementType::Float32,
            },
        )
        .unwrap();
        let device_py = Py::new(
            py,
            Device {
                tag: hurray_core::DeviceTag::Cpu,
                memory_class: MemoryClass::Standard,
                device_id: 0,
            },
        )
        .unwrap();
        SparseTensor {
            descriptor,
            format: SparseFormat::Csc,
            values_buf: BufferStore::from_slice(&values),
            aux_buf_0: BufferStore::from_slice(&row_indices),
            aux_buf_1: Some(BufferStore::from_slice(&col_ptr)),
            dtype_py,
            device_py,
        }
    }

    /// Reset sparse_display to "metadata" — call at the end of any test that
    /// sets it to "content" so the ContextVar default is restored for subsequent
    /// tests sharing the same GIL acquisition.
    fn reset_sparse_display(py: Python<'_>) {
        crate::print_options::set_print_options(py, Some("metadata")).unwrap();
    }

    fn make_csr(py: Python<'_>) -> SparseTensor {
        // 3×3 matrix, 4 non-zeros, float32 values
        let values: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let col_idx: Vec<u8> = [0u64, 2, 1, 2]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let row_ptr: Vec<u8> = [0u64, 1, 3, 4]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let shape = Shape::new(vec![3, 3]).unwrap();
        let descriptor = build_csr_descriptor(
            ElementType::Float32,
            shape,
            4,
            values.len() as u64,
            col_idx.len() as u64,
            row_ptr.len() as u64,
        )
        .unwrap();

        let dtype_py = Py::new(
            py,
            Dtype {
                inner: ElementType::Float32,
            },
        )
        .unwrap();
        let device_py = Py::new(
            py,
            Device {
                tag: hurray_core::DeviceTag::Cpu,
                memory_class: MemoryClass::Standard,
                device_id: 0,
            },
        )
        .unwrap();

        SparseTensor {
            descriptor,
            format: SparseFormat::Csr,
            values_buf: BufferStore::from_slice(&values),
            aux_buf_0: BufferStore::from_slice(&col_idx),
            aux_buf_1: Some(BufferStore::from_slice(&row_ptr)),
            dtype_py,
            device_py,
        }
    }

    #[test]
    fn csr_properties() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let sparse = make_csr(py);
            assert_eq!(sparse.format(), "csr");
            assert_eq!(sparse.ndim(), 2);
            assert_eq!(sparse.nnz(), 4);
        });
    }

    #[test]
    fn csr_shape_tuple() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let sparse = make_csr(py);
            let shape = sparse.shape(py).unwrap();
            let repr = shape.bind(py).repr().unwrap().to_str().unwrap().to_owned();
            assert_eq!(repr, "(3, 3)");
        });
    }

    #[test]
    fn csr_repr_contains_fields() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            // __repr__ now takes &Bound<Self>; wrap the bare struct first.
            let sparse = Py::new(py, make_csr(py)).unwrap();
            let bound = sparse.bind(py);
            let r = SparseTensor::__repr__(bound).unwrap();
            assert!(r.contains("csr"), "repr should contain format");
            assert!(r.contains("float32"), "repr should contain dtype");
            assert!(r.contains("nnz=4"), "repr should contain nnz");
            // dtype must not be wrapped in hurray.Dtype('...')
            assert!(!r.contains("hurray.Dtype"), "repr dtype should be unquoted");
        });
    }

    #[test]
    fn sparse_str_equals_repr() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            // __repr__ and __str__ now take &Bound<Self>; wrap the bare struct first.
            let sparse = Py::new(py, make_csr(py)).unwrap();
            let bound = sparse.bind(py);
            assert_eq!(
                SparseTensor::__repr__(bound).unwrap(),
                SparseTensor::__str__(bound).unwrap(),
                "__str__ and __repr__ must be identical for SparseTensor"
            );
        });
    }

    #[test]
    fn csr_is_unhashable() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let sparse = make_csr(py);
            let err = sparse.__hash__().unwrap_err();
            assert!(err.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
        });
    }

    #[test]
    fn csr_indices_attribute_error() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let sparse = Py::new(py, make_csr(py)).unwrap();
            let bound = sparse.bind(py);
            let result = SparseTensor::indices(bound);
            assert!(result.is_err());
            assert!(result
                .unwrap_err()
                .is_instance_of::<pyo3::exceptions::PyAttributeError>(py));
        });
    }

    #[test]
    fn csc_col_indices_attribute_error() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            // Build a CSC variant.
            let values: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|f| f.to_le_bytes()).collect();
            let row_idx: Vec<u8> = [0u64, 1].iter().flat_map(|v| v.to_le_bytes()).collect();
            let col_ptr: Vec<u8> = [0u64, 1, 2].iter().flat_map(|v| v.to_le_bytes()).collect();

            let shape = Shape::new(vec![2, 2]).unwrap();
            let descriptor = build_csc_descriptor(
                ElementType::Float32,
                shape,
                2,
                values.len() as u64,
                row_idx.len() as u64,
                col_ptr.len() as u64,
            )
            .unwrap();

            let dtype_py = Py::new(
                py,
                Dtype {
                    inner: ElementType::Float32,
                },
            )
            .unwrap();
            let device_py = Py::new(
                py,
                Device {
                    tag: hurray_core::DeviceTag::Cpu,
                    memory_class: MemoryClass::Standard,
                    device_id: 0,
                },
            )
            .unwrap();

            let sparse = Py::new(
                py,
                SparseTensor {
                    descriptor,
                    format: SparseFormat::Csc,
                    values_buf: BufferStore::from_slice(&values),
                    aux_buf_0: BufferStore::from_slice(&row_idx),
                    aux_buf_1: Some(BufferStore::from_slice(&col_ptr)),
                    dtype_py,
                    device_py,
                },
            )
            .unwrap();
            let bound = sparse.bind(py);
            // col_indices is a CSR attribute — should raise AttributeError for CSC.
            let result = SparseTensor::col_indices(bound);
            assert!(result.is_err());
            assert!(result
                .unwrap_err()
                .is_instance_of::<pyo3::exceptions::PyAttributeError>(py));
        });
    }

    #[test]
    fn csr_values_view_shape_and_dtype() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let sparse = Py::new(py, make_csr(py)).unwrap();
            let bound = sparse.bind(py);
            let vals_obj = SparseTensor::values(bound).unwrap();
            let vals = vals_obj.bind(py);
            // shape should be (nnz,) = (4,)
            let shape = vals.getattr("shape").unwrap();
            let shape_repr = shape.repr().unwrap().to_str().unwrap().to_owned();
            assert_eq!(shape_repr, "(4,)");
            // dtype should be float32
            let dtype = vals.getattr("dtype").unwrap();
            let dtype_name: String = dtype.getattr("name").unwrap().extract().unwrap();
            assert_eq!(dtype_name, "float32");
        });
    }

    #[test]
    fn csr_row_ptr_view_shape() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let sparse = Py::new(py, make_csr(py)).unwrap();
            let bound = sparse.bind(py);
            // row_ptr for 3×3 matrix → shape (nrows+1,) = (4,)
            let rp_obj = SparseTensor::row_ptr(bound).unwrap();
            let rp = rp_obj.bind(py);
            let shape = rp.getattr("shape").unwrap();
            let shape_repr = shape.repr().unwrap().to_str().unwrap().to_owned();
            assert_eq!(shape_repr, "(4,)");
            let dtype = rp.getattr("dtype").unwrap();
            let dtype_name: String = dtype.getattr("name").unwrap().extract().unwrap();
            assert_eq!(dtype_name, "uint64");
        });
    }

    #[test]
    fn csr_col_indices_view_shape() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let sparse = Py::new(py, make_csr(py)).unwrap();
            let bound = sparse.bind(py);
            let ci_obj = SparseTensor::col_indices(bound).unwrap();
            let ci = ci_obj.bind(py);
            let shape = ci.getattr("shape").unwrap();
            let shape_repr = shape.repr().unwrap().to_str().unwrap().to_owned();
            assert_eq!(shape_repr, "(4,)");
        });
    }

    #[test]
    fn coo_to_scipy_raises_unsupported() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            // Build a COO SparseTensor
            let values: Vec<u8> = [1.0f32].iter().flat_map(|f| f.to_le_bytes()).collect();
            let indices: Vec<u8> = [0u64, 0u64].iter().flat_map(|v| v.to_le_bytes()).collect();
            let shape = Shape::new(vec![2, 2]).unwrap();
            let bh_values = BufferHandle::new(
                values.len() as u64,
                hurray_core::MIN_BUFFER_ALIGNMENT,
                hurray_core::DeviceTag::Cpu,
                SyncMode::ProducerSynced,
            )
            .unwrap();
            let bh_idx = BufferHandle::new(
                indices.len() as u64,
                hurray_core::MIN_BUFFER_ALIGNMENT,
                hurray_core::DeviceTag::Cpu,
                SyncMode::ProducerSynced,
            )
            .unwrap();
            let descriptor = TensorDescriptor::new(
                DESCRIPTOR_VERSION_MAJOR,
                DESCRIPTOR_VERSION_MINOR,
                ElementType::Float32,
                shape,
                0,
                LayoutDescriptor::Coo(CooLayout::new(1, false)),
                vec![bh_values, bh_idx],
                None,
                None,
                None,
                None,
            )
            .unwrap();
            let dtype_py = Py::new(
                py,
                Dtype {
                    inner: ElementType::Float32,
                },
            )
            .unwrap();
            let device_py = Py::new(
                py,
                Device {
                    tag: hurray_core::DeviceTag::Cpu,
                    memory_class: MemoryClass::Standard,
                    device_id: 0,
                },
            )
            .unwrap();
            let sparse = Py::new(
                py,
                SparseTensor {
                    descriptor,
                    format: SparseFormat::Coo,
                    values_buf: BufferStore::from_slice(&values),
                    aux_buf_0: BufferStore::from_slice(&indices),
                    aux_buf_1: None,
                    dtype_py,
                    device_py,
                },
            )
            .unwrap();
            let bound = sparse.bind(py);
            let result = SparseTensor::to_scipy(bound);
            assert!(result.is_err());
            assert!(result.unwrap_err().is_instance_of::<UnsupportedError>(py));
        });
    }

    // ── Print-options tests (Job 2) ───────────────────────────────────────────

    // Job 2.1 — Default mode is "metadata"; repr has no array keys.
    #[test]
    fn default_display_is_metadata_csr() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            // Ensure we start in metadata mode (in case another test leaked state).
            reset_sparse_display(py);
            let sparse = Py::new(py, make_csr(py)).unwrap();
            let bound = sparse.bind(py);
            let r = SparseTensor::__repr__(bound).unwrap();
            // Metadata mode: compact summary only.
            assert!(
                r.starts_with("hurray.SparseTensor("),
                "repr must start with type name"
            );
            assert!(r.contains("format='csr'"), "repr must contain format");
            assert!(r.contains("nnz=4"), "repr must contain nnz");
            assert!(r.contains("dtype=float32"), "repr must contain dtype");
            // Must NOT contain any array labels in default mode.
            assert!(
                !r.contains("values="),
                "default repr must not contain values= (got: {r})"
            );
            assert!(
                !r.contains("col_indices="),
                "default repr must not contain col_indices= (got: {r})"
            );
            assert!(
                !r.contains("row_ptr="),
                "default repr must not contain row_ptr= (got: {r})"
            );
        });
    }

    #[test]
    fn default_display_is_metadata_coo() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            reset_sparse_display(py);
            let sparse = Py::new(py, make_coo(py)).unwrap();
            let bound = sparse.bind(py);
            let r = SparseTensor::__repr__(bound).unwrap();
            assert!(r.contains("format='coo'"));
            assert!(
                !r.contains("indices="),
                "default COO repr must not contain indices="
            );
            assert!(
                !r.contains("values="),
                "default COO repr must not contain values="
            );
        });
    }

    #[test]
    fn default_display_is_metadata_csc() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            reset_sparse_display(py);
            let sparse = Py::new(py, make_csc(py)).unwrap();
            let bound = sparse.bind(py);
            let r = SparseTensor::__repr__(bound).unwrap();
            assert!(r.contains("format='csc'"));
            assert!(
                !r.contains("values="),
                "default CSC repr must not contain values="
            );
            assert!(
                !r.contains("row_indices="),
                "default CSC repr must not contain row_indices="
            );
            assert!(
                !r.contains("col_ptr="),
                "default CSC repr must not contain col_ptr="
            );
        });
    }

    // Job 2.2 — Content mode: ContextVar is correctly set; when NumPy is present
    // the repr includes array labels and values.  When NumPy is absent the spec
    // mandates a silent fallback to the metadata string — see Job 2.8 test below.
    //
    // These tests verify (a) set_print_options / sparse_display_is_content agree,
    // (b) __repr__ still produces a well-formed metadata string on fallback, and
    // (c) when numpy IS importable the array labels and values are present.
    #[test]
    fn content_mode_csr_contextvar_is_set_and_repr_does_not_raise() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            reset_sparse_display(py);
            crate::print_options::set_print_options(py, Some("content")).unwrap();

            // ContextVar must report content.
            assert!(
                crate::print_options::sparse_display_is_content(py),
                "sparse_display_is_content must be true after set_print_options(content)"
            );

            let sparse = Py::new(py, make_csr(py)).unwrap();
            let bound = sparse.bind(py);
            // __repr__ must never raise, regardless of whether NumPy is present.
            let r = SparseTensor::__repr__(bound).unwrap();
            // The repr must at minimum contain the invariant metadata fields.
            assert!(
                r.contains("format='csr'"),
                "repr must contain format (got: {r})"
            );
            assert!(r.contains("nnz=4"), "repr must contain nnz (got: {r})");
            assert!(
                r.contains("dtype=float32"),
                "repr must contain dtype (got: {r})"
            );

            // When NumPy is available: also assert the array labels.
            // When NumPy is absent: __repr__ falls back silently to metadata (Job 2.8).
            let numpy_available = py.import("numpy").is_ok();
            if numpy_available {
                assert!(
                    r.contains("values="),
                    "content CSR repr must contain values= when numpy present (got: {r})"
                );
                assert!(
                    r.contains("col_indices="),
                    "content CSR repr must contain col_indices= when numpy present (got: {r})"
                );
                assert!(
                    r.contains("row_ptr="),
                    "content CSR repr must contain row_ptr= when numpy present (got: {r})"
                );
                // Numeric values: [1,2,3,4], col_indices=[0,2,1,2], row_ptr=[0,1,3,4]
                assert!(
                    r.contains("1.") && r.contains("4."),
                    "values must appear (got: {r})"
                );
            }

            reset_sparse_display(py);
        });
    }

    #[test]
    fn content_mode_coo_contextvar_is_set_and_repr_does_not_raise() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            reset_sparse_display(py);
            crate::print_options::set_print_options(py, Some("content")).unwrap();

            assert!(crate::print_options::sparse_display_is_content(py));

            let sparse = Py::new(py, make_coo(py)).unwrap();
            let bound = sparse.bind(py);
            let r = SparseTensor::__repr__(bound).unwrap();
            assert!(
                r.contains("format='coo'"),
                "repr must contain format (got: {r})"
            );

            let numpy_available = py.import("numpy").is_ok();
            if numpy_available {
                assert!(
                    r.contains("indices="),
                    "content COO repr must contain indices= when numpy present (got: {r})"
                );
                assert!(
                    r.contains("values="),
                    "content COO repr must contain values= when numpy present (got: {r})"
                );
                // values=[5.0, 7.0]
                assert!(
                    r.contains("5.") && r.contains("7."),
                    "values must appear (got: {r})"
                );
            }

            reset_sparse_display(py);
        });
    }

    #[test]
    fn content_mode_csc_contextvar_is_set_and_repr_does_not_raise() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            reset_sparse_display(py);
            crate::print_options::set_print_options(py, Some("content")).unwrap();

            assert!(crate::print_options::sparse_display_is_content(py));

            let sparse = Py::new(py, make_csc(py)).unwrap();
            let bound = sparse.bind(py);
            let r = SparseTensor::__repr__(bound).unwrap();
            assert!(
                r.contains("format='csc'"),
                "repr must contain format (got: {r})"
            );

            let numpy_available = py.import("numpy").is_ok();
            if numpy_available {
                assert!(
                    r.contains("values="),
                    "content CSC repr must contain values= when numpy present (got: {r})"
                );
                assert!(
                    r.contains("row_indices="),
                    "content CSC repr must contain row_indices= when numpy present (got: {r})"
                );
                assert!(
                    r.contains("col_ptr="),
                    "content CSC repr must contain col_ptr= when numpy present (got: {r})"
                );
                // values=[3.0, 6.0]
                assert!(
                    r.contains("3.") && r.contains("6."),
                    "values must appear (got: {r})"
                );
            }

            reset_sparse_display(py);
        });
    }

    // Job 2.8 — NumPy-absent fallback: content mode silently returns metadata string.
    // This is always testable: __repr__ must not raise and must produce well-formed output.
    #[test]
    fn content_mode_fallback_to_metadata_when_numpy_absent() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            reset_sparse_display(py);

            // Only exercise the "fallback" branch when numpy really is absent.
            // When numpy is present this test still passes (repr succeeds), but we
            // cannot assert the metadata-only form because arrays will be present.
            let numpy_available = py.import("numpy").is_ok();
            if numpy_available {
                // Nothing to assert about the fallback path; skip.
                return;
            }

            crate::print_options::set_print_options(py, Some("content")).unwrap();
            assert!(crate::print_options::sparse_display_is_content(py));

            let sparse = Py::new(py, make_csr(py)).unwrap();
            let r = SparseTensor::__repr__(sparse.bind(py)).unwrap();

            // Spec: __repr__ never raises; falls back to the metadata string.
            assert!(
                r.starts_with("hurray.SparseTensor("),
                "fallback repr must be a valid metadata string (got: {r})"
            );
            assert!(
                r.contains("format='csr'"),
                "fallback repr must contain format (got: {r})"
            );
            assert!(
                r.contains("nnz=4"),
                "fallback repr must contain nnz (got: {r})"
            );
            // In the fallback the array labels must NOT appear (numpy couldn't render them).
            assert!(
                !r.contains("values="),
                "numpy-absent fallback must not contain values= (got: {r})"
            );

            reset_sparse_display(py);
        });
    }

    // Job 2.3 — __str__ still equals __repr__ in content mode.
    #[test]
    fn str_equals_repr_in_content_mode() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            reset_sparse_display(py);
            crate::print_options::set_print_options(py, Some("content")).unwrap();

            let sparse = Py::new(py, make_csr(py)).unwrap();
            let bound = sparse.bind(py);
            assert_eq!(
                SparseTensor::__repr__(bound).unwrap(),
                SparseTensor::__str__(bound).unwrap(),
                "__str__ must equal __repr__ in content mode"
            );

            reset_sparse_display(py);
        });
    }

    // Job 2.4 — get_print_options reflects the active value.
    #[test]
    fn get_print_options_returns_metadata_by_default() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            reset_sparse_display(py);
            let opts = crate::print_options::get_print_options(py).unwrap();
            // get_print_options returns a Py<PyAny> (PyDict); extract "sparse_display" via Python.
            let v: String = opts
                .bind(py)
                .get_item("sparse_display")
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(v, "metadata", "default sparse_display must be 'metadata'");
        });
    }

    #[test]
    fn get_print_options_reflects_content_after_set() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            reset_sparse_display(py);
            crate::print_options::set_print_options(py, Some("content")).unwrap();
            let opts = crate::print_options::get_print_options(py).unwrap();
            let v: String = opts
                .bind(py)
                .get_item("sparse_display")
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(v, "content", "sparse_display must be 'content' after set");
            reset_sparse_display(py);
        });
    }

    // Job 2.5 — Context manager scoping: content inside, metadata restored after.
    //
    // __enter__ and __exit__ are private #[pymethods]; drive them via Python's
    // call_method interface on the Bound object, exactly as Python does at runtime.
    #[test]
    fn print_options_ctx_scopes_and_restores() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            reset_sparse_display(py);

            // Before: metadata.
            assert!(!crate::print_options::sparse_display_is_content(py));

            // Build a PrintOptionsCtx, box it into a Python object, call protocol methods.
            let ctx =
                crate::print_options::PrintOptionsCtx::new(Some("content".to_owned())).unwrap();
            let bound_ctx = Py::new(py, ctx).unwrap();

            // __enter__: sets content.
            bound_ctx.bind(py).call_method0("__enter__").unwrap();
            assert!(
                crate::print_options::sparse_display_is_content(py),
                "inside context: must be content"
            );

            // Verify repr of CSR is in content mode inside the context.
            // When NumPy is present the array labels appear; when absent __repr__
            // falls back silently to the metadata string (spec: __repr__ never raises).
            let sparse = Py::new(py, make_csr(py)).unwrap();
            let r = SparseTensor::__repr__(sparse.bind(py)).unwrap();
            assert!(
                r.starts_with("hurray.SparseTensor("),
                "repr inside ctx must be well-formed (got: {r})"
            );
            if py.import("numpy").is_ok() {
                assert!(
                    r.contains("values="),
                    "repr inside ctx must be content when numpy present (got: {r})"
                );
            }

            // __exit__(None, None, None): restores metadata.
            let none = py.None();
            let nb = none.bind(py);
            bound_ctx
                .bind(py)
                .call_method1("__exit__", (nb, nb, nb))
                .unwrap();
            assert!(
                !crate::print_options::sparse_display_is_content(py),
                "after context: must be restored to metadata"
            );
        });
    }

    // Job 2.5 (nesting) — nested contexts restore correctly.
    #[test]
    fn print_options_ctx_nested_restores_correctly() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            reset_sparse_display(py);

            let none = py.None();
            let nb = none.bind(py);

            // Outer: enter content.
            let outer =
                crate::print_options::PrintOptionsCtx::new(Some("content".to_owned())).unwrap();
            let b_outer = Py::new(py, outer).unwrap();
            b_outer.bind(py).call_method0("__enter__").unwrap();
            assert!(crate::print_options::sparse_display_is_content(py));

            // Inner: enter metadata (overrides outer content).
            let inner =
                crate::print_options::PrintOptionsCtx::new(Some("metadata".to_owned())).unwrap();
            let b_inner = Py::new(py, inner).unwrap();
            b_inner.bind(py).call_method0("__enter__").unwrap();
            assert!(
                !crate::print_options::sparse_display_is_content(py),
                "inner ctx overrides outer: must be metadata"
            );

            // Exit inner: back to content.
            b_inner
                .bind(py)
                .call_method1("__exit__", (nb, nb, nb))
                .unwrap();
            assert!(
                crate::print_options::sparse_display_is_content(py),
                "after inner exit: must be back to content"
            );

            // Exit outer: back to original metadata.
            b_outer
                .bind(py)
                .call_method1("__exit__", (nb, nb, nb))
                .unwrap();
            assert!(
                !crate::print_options::sparse_display_is_content(py),
                "after outer exit: must be restored to metadata"
            );
        });
    }

    // Job 2.6 — Invalid values raise ValueError.
    #[test]
    fn set_print_options_invalid_value_raises_value_error() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let result = crate::print_options::set_print_options(py, Some("foo"));
            assert!(result.is_err(), "invalid value must return Err");
            assert!(
                result
                    .unwrap_err()
                    .is_instance_of::<pyo3::exceptions::PyValueError>(py),
                "invalid value must raise ValueError"
            );
        });
    }

    #[test]
    fn print_options_ctx_invalid_value_raises_value_error() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let result = crate::print_options::PrintOptionsCtx::new(Some("bad".to_owned()));
            assert!(
                result.is_err(),
                "invalid value in PrintOptionsCtx::new must return Err"
            );
            // PrintOptionsCtx does not derive Debug, so use .err().unwrap() instead of
            // .unwrap_err() (the latter requires T: Debug for its panic message).
            assert!(
                result
                    .err()
                    .unwrap()
                    .is_instance_of::<pyo3::exceptions::PyValueError>(py),
                "invalid value must raise ValueError"
            );
        });
    }

    // Job 2.7 — Dense Tensor repr is unaffected by sparse_display setting.
    #[test]
    fn content_mode_does_not_affect_dense_tensor_repr() {
        init();
        Python::attach(|py| {
            // Dense Tensor needs its own build_module; reuse the one from this module
            // which already has tensor registered.
            let _m = build_module(py);
            reset_sparse_display(py);

            // Build a simple float32 dense tensor.
            use pyo3::types::PyBytes;
            let floats: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
            let buf: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
            let py_buf = PyBytes::new(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Float32,
                },
            )
            .unwrap();
            let tensor = Py::new(
                py,
                crate::tensor::Tensor::new(py, py_buf.as_any(), dtype.bind(py), vec![4], None)
                    .unwrap(),
            )
            .unwrap();

            // Capture repr in metadata mode.
            let repr_metadata = tensor
                .bind(py)
                .call_method0("__repr__")
                .unwrap()
                .extract::<String>()
                .unwrap();

            // Switch to content mode.
            crate::print_options::set_print_options(py, Some("content")).unwrap();

            // Dense tensor repr must be identical — content mode is sparse-only.
            let repr_content = tensor
                .bind(py)
                .call_method0("__repr__")
                .unwrap()
                .extract::<String>()
                .unwrap();

            assert_eq!(
                repr_metadata, repr_content,
                "dense Tensor repr must be unaffected by sparse_display=content"
            );

            reset_sparse_display(py);
        });
    }
}
