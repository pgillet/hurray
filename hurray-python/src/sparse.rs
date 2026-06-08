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

// PyO3 0.22 macro expansion emits a redundant .into() on PyErr — suppress.
#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;
use pyo3::types::PyTuple;

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
        let items: Vec<PyObject> = self
            .descriptor
            .shape
            .dims()
            .iter()
            .map(|&dim| {
                if dim == DYNAMIC {
                    py.None()
                } else {
                    dim.to_object(py)
                }
            })
            .collect();
        Ok(PyTuple::new_bound(py, items).unbind())
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
    pub fn values(slf: &Bound<'_, Self>) -> PyResult<PyObject> {
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
        Ok(Py::new(py, tensor)?.into_py(py))
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
    pub fn indices(slf: &Bound<'_, Self>) -> PyResult<PyObject> {
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
    pub fn col_indices(slf: &Bound<'_, Self>) -> PyResult<PyObject> {
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
    pub fn row_ptr(slf: &Bound<'_, Self>) -> PyResult<PyObject> {
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
    pub fn row_indices(slf: &Bound<'_, Self>) -> PyResult<PyObject> {
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
    pub fn col_ptr(slf: &Bound<'_, Self>) -> PyResult<PyObject> {
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
    pub fn to_scipy(slf: &Bound<'_, Self>) -> PyResult<PyObject> {
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
        let sp = py.import_bound("scipy.sparse").map_err(|_| {
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

        let kwargs = pyo3::types::PyDict::new_bound(py);
        kwargs.set_item("shape", PyTuple::new_bound(py, [nrows, ncols]))?;
        kwargs.set_item("copy", false)?;

        // csr_matrix((data, indices, indptr), shape=..., copy=False)
        let args = PyTuple::new_bound(py, [&values_arr, &aux0_arr, &aux1_arr]);
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

    pub fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let shape_tuple = self.shape(py)?;
        let shape_str = shape_tuple.bind(py).repr()?.to_str()?.to_owned();
        let dtype_name = crate::dtype::element_type_name(self.descriptor.element_type);
        Ok(format!(
            "hurray.SparseTensor(format='{}', shape={}, nnz={}, dtype=hurray.Dtype('{}'))",
            self.format.as_str(),
            shape_str,
            self.nnz(),
            dtype_name,
        ))
    }
}

// ── Crate-internal helpers ────────────────────────────────────────────────────

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
) -> PyResult<PyObject> {
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
    Ok(Py::new(py, tensor)?.into_py(py))
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
        pyo3::prepare_freethreaded_python();
    }

    fn build_module(py: Python<'_>) -> pyo3::Bound<'_, pyo3::types::PyModule> {
        let m = pyo3::types::PyModule::new_bound(py, "hurray").unwrap();
        crate::errors::register(&m).unwrap();
        crate::dtype::register(&m).unwrap();
        crate::device::register(&m).unwrap();
        crate::tensor::register(&m).unwrap();
        register(&m).unwrap();
        let sys = py.import_bound("sys").unwrap();
        let modules = sys.getattr("modules").unwrap();
        modules.set_item("hurray", &m).unwrap();
        m
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
        Python::with_gil(|py| {
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
        Python::with_gil(|py| {
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
        Python::with_gil(|py| {
            let _m = build_module(py);
            let sparse = make_csr(py);
            let r = sparse.__repr__(py).unwrap();
            assert!(r.contains("csr"), "repr should contain format");
            assert!(r.contains("float32"), "repr should contain dtype");
            assert!(r.contains("nnz=4"), "repr should contain nnz");
        });
    }

    #[test]
    fn csr_is_unhashable() {
        init();
        Python::with_gil(|py| {
            let _m = build_module(py);
            let sparse = make_csr(py);
            let err = sparse.__hash__().unwrap_err();
            assert!(err.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
        });
    }

    #[test]
    fn csr_indices_attribute_error() {
        init();
        Python::with_gil(|py| {
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
        Python::with_gil(|py| {
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
        Python::with_gil(|py| {
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
        Python::with_gil(|py| {
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
        Python::with_gil(|py| {
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
        Python::with_gil(|py| {
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
}
