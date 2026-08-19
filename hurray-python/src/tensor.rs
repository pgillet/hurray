//! Python bindings for the Hurray tensor type.
//!
//! Exposes [`Tensor`] as `hurray.Tensor`: a Python class that holds a tensor
//! descriptor, a buffer (owned or borrowed), and Python-side dtype/device handles.
//!
//! ## Buffer ownership (D2)
//!
//! The buffer is stored in a [`BufferStore`] enum:
//! - `Owned` — copied from the caller-supplied bytes at construction time.
//! - `Borrowed` — zero-copy pointer into a NumPy array or similar source,
//!   with a strong Python reference keeping the source alive.
//!
//! See `buffer.rs` for the full safety contract.

use std::os::raw::c_void;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};
use pyo3::IntoPyObjectExt;

use hurray_core::{
    buffer_size_bytes, BufferHandle, LayoutDescriptor, Shape, SyncMode, TensorDescriptor,
    DESCRIPTOR_VERSION_MAJOR, DESCRIPTOR_VERSION_MINOR, DYNAMIC, MIN_BUFFER_ALIGNMENT,
};

use crate::buffer::BufferStore;
use crate::device::Device;
use crate::dlpack;
use crate::dtype::Dtype;
use crate::errors::{BufferError, CopyRequiredError, InvalidDescriptorError, UnsupportedError};
use crate::layout::{is_dense, layout_name, layout_to_py};

/// A Hurray tensor: element type, shape, device, and a data buffer.
///
/// ## Construction
///
/// ```python
/// hurray.Tensor(buffer, dtype, shape, device=None)
/// ```
///
/// | Parameter | Type | Default |
/// |-----------|------|---------|
/// | `buffer` | `bytes` or `bytearray` | — |
/// | `dtype` | `hurray.Dtype` | — |
/// | `shape` | `list[int]` | — |
/// | `device` | `hurray.Device` | `hurray.device.cpu` |
///
/// ## Zero-copy interop
///
/// Use [`hurray.from_numpy`] or [`hurray.from_torch`] to create tensors that share
/// the source buffer without copying. The `Tensor` holds a strong Python reference
/// to the source object so its buffer remains valid.
///
/// ## Examples (Python)
///
/// ```python
/// import struct, hurray
///
/// buf = struct.pack("6f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
/// t = hurray.Tensor(buf, hurray.float32, [2, 3])
///
/// assert t.shape == (2, 3)
/// assert t.ndim == 2
/// assert t.size == 6
/// assert t.dtype == hurray.float32
/// assert t.device.kind == "cpu"
/// ```
#[pyclass(name = "Tensor")]
#[derive(Debug)]
pub struct Tensor {
    /// Tensor descriptor carrying element type, shape, layout, and buffer metadata.
    pub descriptor: TensorDescriptor,
    /// Element data buffer — owned copy or zero-copy borrowed reference (D2).
    /// This is descriptor buffer index 0; every tensor has one.
    pub buffer: BufferStore,
    /// Buffers at descriptor indices `1..N`: quantization scales and zero points,
    /// sparse index arrays, block-paged page tables (ADR-030 § 3, buffer order is
    /// descriptor order).
    ///
    /// Split from `buffer` rather than folded into one `Vec` because an empty
    /// `Vec` does not allocate: a dense unquantized tensor — the overwhelmingly
    /// common case — pays nothing for multi-buffer support.
    pub aux_buffers: Vec<BufferStore>,
    /// Python-side dtype handle; holds the caller's object so
    /// `tensor.dtype is hurray.float32` holds when the user passes a singleton.
    pub dtype_py: Py<Dtype>,
    /// Python-side device handle (kept alive alongside the descriptor).
    pub device_py: Py<Device>,
}

#[pymethods]
impl Tensor {
    // ── Constructor ───────────────────────────────────────────────────────────

    /// Construct a `Tensor` from a bytes-like object, dtype, shape, and optional device.
    ///
    /// ## Layout
    ///
    /// `layout` takes a `hurray.Layout` object and defaults to row-major. It is a
    /// declaration about the buffers supplied alongside it, and the two must agree:
    /// the rank constraints, the buffer count, and each buffer's size are all checked
    /// here. A layout is never inferred from the buffers and the buffers are never
    /// reinterpreted to fit the layout — `hurray.CooLayout(nnz=4)` with a
    /// two-element values buffer raises rather than quietly becoming `nnz=2`.
    ///
    /// A layout **string** is not accepted: it could not carry `nnz` or `strides`,
    /// so it can only produce a descriptor that is wrong.
    ///
    /// ## Errors
    ///
    /// - `hurray.InvalidDescriptorError` — negative dimension in `shape`.
    /// - `hurray.BufferError` — buffer smaller than required by `dtype` + `shape`.
    /// - `hurray.BufferError` — `buffer` is not a `bytes` / `bytearray` object.
    /// - `TypeError` — `layout` is not a `hurray.Layout`.
    /// - `hurray.InvalidDescriptorError` — the layout contradicts the shape, or too
    ///   few buffers were supplied for it.
    /// - `hurray.BufferError` — a buffer is smaller than the layout implies.
    /// - `hurray.UnsupportedError` — a composite layout, which owns no buffers and
    ///   so cannot be expressed as a `hurray.Tensor`.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import struct, hurray
    ///
    /// buf = struct.pack("6f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    /// t = hurray.Tensor(buf, hurray.float32, [2, 3])
    /// assert t.size == 6
    ///
    /// # A CSR tensor: values, column indices, row pointers.
    /// csr = hurray.Tensor(
    ///     struct.pack("2f", 5.0, 7.0), hurray.float32, [2, 2],
    ///     aux_buffers=[struct.pack("2Q", 0, 1), struct.pack("3Q", 0, 1, 2)],
    ///     layout=hurray.CsrLayout(nnz=2),
    /// )
    /// assert csr.layout == hurray.CsrLayout(nnz=2)
    /// ```
    #[new]
    #[pyo3(signature = (
        buffer,
        dtype,
        shape,
        device = None,
        *,
        aux_buffers = None,
        layout = None,
        quantization = None,
        statistics = None,
        shard = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        py: Python<'_>,
        buffer: &Bound<'_, PyAny>,
        dtype: &Bound<'_, Dtype>,
        shape: Vec<i64>,
        device: Option<Py<Device>>,
        aux_buffers: Option<Vec<Py<PyAny>>>,
        layout: Option<&Bound<'_, PyAny>>,
        quantization: Option<&Bound<'_, PyAny>>,
        statistics: Option<&Bound<'_, crate::metadata::Statistics>>,
        shard: Option<&Bound<'_, crate::metadata::Shard>>,
    ) -> PyResult<Self> {
        // ── 1. Resolve device ────────────────────────────────────────────────
        let device_py: Py<Device> = match device {
            Some(d) => d,
            None => {
                // Default to hurray.device.cpu — retrieve the singleton via sys.modules
                // so `tensor.device is hurray.device.cpu` holds.
                let sys = py.import("sys")?;
                let modules = sys.getattr("modules")?;
                let device_mod = modules.get_item("hurray.device")?;
                let cpu_obj = device_mod.getattr("cpu")?;
                cpu_obj.extract::<Py<Device>>()?
            }
        };

        // Retain the caller's Dtype object so `tensor.dtype is hurray.float32`
        // holds when the user passes a registered singleton constant.
        let dtype_py: Py<Dtype> = dtype.clone().unbind();

        // ── 2. Parse and validate shape ──────────────────────────────────────
        let dims: Vec<u64> = shape
            .iter()
            .map(|&d| {
                if d < 0 {
                    Err(InvalidDescriptorError::new_err(format!(
                        "shape dimensions must be non-negative, got {d}"
                    )))
                } else {
                    Ok(d as u64)
                }
            })
            .collect::<PyResult<_>>()?;

        let hurray_shape = Shape::new(dims)
            .map_err(|e| InvalidDescriptorError::new_err(format!("invalid shape: {e}")))?;

        // ── 2b. Resolve the layout ───────────────────────────────────────────
        let layout_desc = match layout {
            Some(obj) => crate::layout::extract_layout(obj)?,
            None => LayoutDescriptor::RowMajor,
        };

        // A composite head owns no buffers at all, and this class is built around
        // holding one. Rejected here rather than allowed to fail deeper, where the
        // error would name a buffer table the caller never meant to supply.
        if layout_desc.is_virtual() {
            return Err(UnsupportedError::new_err(
                "a composite layout cannot be given to hurray.Tensor: a composite head \
                 owns no buffers, which this class cannot represent",
            ));
        }

        // ── 3. Extract buffer bytes ──────────────────────────────────────────
        let buf_bytes: &[u8] = if let Ok(b) = buffer.extract::<&[u8]>() {
            b
        } else if let Ok(b) = buffer.cast::<PyBytes>() {
            b.as_bytes()
        } else {
            return Err(BufferError::new_err(
                "expected bytes, bytearray, or buffer-compatible object",
            ));
        };

        // ── 4. Validate buffer size ──────────────────────────────────────────
        // Only a dense layout stores one element per logical index. A sparse or
        // indirect layout's buffer 0 holds nnz values or a page pool instead, and its
        // sizes come from the layout's own parameters in step 6b.
        if is_dense(&layout_desc) {
            let element_count = hurray_shape.element_count().unwrap_or(0);
            let expected = buffer_size_bytes(dtype.get().inner, element_count);
            if (buf_bytes.len() as u64) < expected {
                return Err(BufferError::new_err(format!(
                    "buffer too small: need at least {expected} bytes for {} elements of {}, \
                     got {}",
                    element_count,
                    dtype.get().name(),
                    buf_bytes.len(),
                )));
            }
        }

        // ── 5. Build the TensorDescriptor ────────────────────────────────────
        let (device_tag, memory_class) = {
            let dev = device_py.borrow(py);
            (dev.tag, dev.memory_class)
        };

        // SyncMode: CPU always uses ProducerSynced; for accelerators use
        // ProducerSynced as a safe default (the host has not yet issued a kernel).
        let sync_mode = SyncMode::ProducerSynced;

        let alignment = if buf_bytes.is_empty() {
            1
        } else {
            MIN_BUFFER_ALIGNMENT
        };

        let make_handle = |len: usize| {
            BufferHandle::with_memory_class(
                len as u64,
                if len == 0 { 1 } else { MIN_BUFFER_ALIGNMENT },
                device_tag,
                sync_mode,
                memory_class,
            )
            .map_err(|e| BufferError::new_err(format!("invalid buffer parameters: {e}")))
        };
        let _ = alignment;

        // ── 6. Auxiliary buffers (descriptor indices 1..N) ───────────────────
        let mut handles = vec![make_handle(buf_bytes.len())?];
        let mut aux_stores: Vec<BufferStore> = Vec::new();
        if let Some(aux) = aux_buffers {
            for (offset, obj) in aux.iter().enumerate() {
                let bound = obj.bind(py);
                let bytes: &[u8] = if let Ok(b) = bound.extract::<&[u8]>() {
                    b
                } else if let Ok(b) = bound.cast::<PyBytes>() {
                    b.as_bytes()
                } else {
                    return Err(BufferError::new_err(format!(
                        "aux_buffers[{offset}]: expected bytes, bytearray, or a                          buffer-compatible object"
                    )));
                };
                handles.push(make_handle(bytes.len())?);
                aux_stores.push(BufferStore::from_slice(bytes));
            }
        }

        // ── 6b. The layout must agree with the buffers it describes ──────────
        let buffer_lens: Vec<usize> = std::iter::once(buf_bytes.len())
            .chain(aux_stores.iter().map(|s| s.len()))
            .collect();
        crate::layout::validate_layout(
            &layout_desc,
            &hurray_shape,
            dtype.get().inner,
            &buffer_lens,
        )?;

        // ── 7. Optional descriptor sections ──────────────────────────────────
        let quant_desc = quantization
            .map(crate::quantization::extract_quantization)
            .transpose()?;

        // Reject an index that resolves to no buffer here, at authoring time. A
        // descriptor referencing a scale buffer that was never supplied encodes and
        // decodes cleanly, so the consumer would receive a dangling index instead of
        // an error — the exact failure the multi-buffer transport exists to prevent.
        if let Some(ref q) = quant_desc {
            hurray_core::validate_buffer_placement(q, &handles, 0).map_err(|e| {
                InvalidDescriptorError::new_err(format!("invalid quantization: {e}"))
            })?;
            crate::layout::validate_quantization_indices(&layout_desc, q)?;
        }

        let descriptor = TensorDescriptor::new(
            DESCRIPTOR_VERSION_MAJOR,
            DESCRIPTOR_VERSION_MINOR,
            dtype.get().inner,
            hurray_shape,
            0, // byte_offset: element [0,…,0] is at the start of the buffer
            layout_desc,
            handles,
            quant_desc.map(|q| q.encode_to_vec()),
            shard.map(|s| s.get().inner.clone()),
            statistics.map(|s| s.get().inner.clone()),
            None, // no extension type
        )
        .map_err(|e| InvalidDescriptorError::new_err(format!("invalid tensor descriptor: {e}")))?;

        Ok(Self {
            descriptor,
            buffer: BufferStore::from_slice(buf_bytes),
            aux_buffers: aux_stores,
            dtype_py,
            device_py,
        })
    }

    // ── Properties ────────────────────────────────────────────────────────────

    /// The element type of this tensor.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert t.dtype == hurray.float32
    /// ```
    #[getter]
    pub fn dtype(&self, py: Python<'_>) -> Py<Dtype> {
        self.dtype_py.clone_ref(py)
    }

    /// The shape of this tensor as a tuple of `int` (or `None` for dynamic dims).
    ///
    /// Dynamic dimensions (`DYNAMIC = u64::MAX` in the wire format) are mapped to
    /// `None`. All other dimensions are returned as Python `int`.
    ///
    /// ## Examples
    ///
    /// ```python
    /// t = hurray.Tensor(buf, hurray.float32, [2, 3])
    /// assert t.shape == (2, 3)
    /// ```
    #[getter]
    pub fn shape(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let items: Vec<Py<PyAny>> = self
            .descriptor
            .shape
            .dims()
            .iter()
            .map(|&dim| -> PyResult<Py<PyAny>> {
                if dim == DYNAMIC {
                    Ok(py.None())
                } else {
                    // dim is u64; Python int is arbitrary-precision so no truncation.
                    dim.into_py_any(py)
                }
            })
            .collect::<PyResult<_>>()?;
        Ok(PyTuple::new(py, items)?.unbind())
    }

    /// Number of dimensions (rank) of this tensor.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.Tensor(buf, hurray.float32, [2, 3]).ndim == 2
    /// ```
    #[getter]
    pub fn ndim(&self) -> usize {
        self.descriptor.shape.rank()
    }

    /// Total number of logical elements, or `None` if any dimension is dynamic.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.Tensor(buf, hurray.float32, [2, 3]).size == 6
    /// ```
    #[getter]
    pub fn size(&self) -> Option<u64> {
        self.descriptor.shape.element_count()
    }

    /// The device this tensor resides on.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert t.device.kind == "cpu"
    /// ```
    #[getter]
    pub fn device(&self, py: Python<'_>) -> Py<Device> {
        self.device_py.clone_ref(py)
    }

    /// Number of buffers this tensor holds, including the data buffer.
    ///
    /// `1` for an ordinary dense tensor; more when the descriptor references
    /// quantization scales, sparse index arrays, or a page table.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// t = hurray.Tensor(bytes(16), hurray.float32, [4])
    /// assert t.buffer_count == 1
    /// ```
    #[getter(buffer_count)]
    pub fn py_buffer_count(&self) -> usize {
        self.buffer_count()
    }

    /// The memory layout of this tensor, as a `hurray.Layout` object.
    ///
    /// The object carries that layout's parameters — `nnz`, `strides`, `page_size`,
    /// and so on — which a string could not. Its name is `layout.name`: one of
    /// `"row_major"`, `"col_major"`, `"strided"`, `"tiled"`, `"morton"`,
    /// `"hilbert"`, `"coo"`, `"csr"`, `"csc"`, `"csf"`, `"block_paged"`,
    /// `"composite"`, or `"extension"` for a private or unrecognised layout tag.
    ///
    /// Layout is a property of a tensor, not a different kind of object
    /// (ADR-031): a COO tensor and a row-major tensor are both `hurray.Tensor`.
    ///
    /// Read-only, and a fresh object each access: assigning a layout would silently
    /// reinterpret the existing buffers, and the object holds no reference back to
    /// this tensor, so `t.layout is t.layout` is `False` while `==` holds.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// t = hurray.Tensor(bytes(16), hurray.float32, [4])
    /// assert t.layout.name == "row_major"
    /// assert isinstance(t.layout, hurray.RowMajorLayout)
    /// assert t.layout == hurray.RowMajorLayout()
    /// ```
    #[getter]
    pub fn layout(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        layout_to_py(py, &self.descriptor.layout)
    }

    /// Number of stored non-zero elements, for layouts that track it.
    ///
    /// ## Errors
    ///
    /// - `AttributeError` — the layout has no notion of `nnz` (D10 discipline:
    ///   inapplicable attributes behave as absent, so `hasattr` reports the truth).
    ///
    /// ## Examples
    ///
    /// ```python
    /// import numpy as np, hurray
    ///
    /// t = hurray.sparse_coo(
    ///     np.array([5.0, 7.0], dtype=np.float32),
    ///     np.array([[0, 0], [1, 1]], dtype=np.uint64),
    ///     [2, 2],
    /// )
    /// assert t.nnz == 2
    /// assert not hasattr(hurray.Tensor(bytes(16), hurray.float32, [4]), "nnz")
    /// ```
    #[getter]
    pub fn nnz(&self) -> PyResult<u64> {
        match &self.descriptor.layout {
            LayoutDescriptor::Coo(l) => Ok(l.nnz),
            LayoutDescriptor::Csr(l) => Ok(l.nnz),
            LayoutDescriptor::Csc(l) => Ok(l.nnz),
            other => Err(no_such_attribute("nnz", layout_name(other))),
        }
    }

    /// A `hurray.Tensor` view over the stored values, for layouts that store them
    /// separately from their index structure.
    ///
    /// The view borrows this tensor's buffer — no copy — and keeps it alive.
    ///
    /// ## Errors
    ///
    /// - `AttributeError` — the layout has no separate values buffer.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import numpy as np, hurray
    ///
    /// t = hurray.sparse_coo(
    ///     np.array([5.0, 7.0], dtype=np.float32),
    ///     np.array([[0, 0], [1, 1]], dtype=np.uint64),
    ///     [2, 2],
    /// )
    /// assert t.values.shape == (2,)
    /// ```
    #[getter]
    pub fn values(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let (nnz, element_type) = {
            let t = slf.borrow();
            match &t.descriptor.layout {
                LayoutDescriptor::Coo(l) => (l.nnz, t.descriptor.element_type),
                LayoutDescriptor::Csr(l) => (l.nnz, t.descriptor.element_type),
                LayoutDescriptor::Csc(l) => (l.nnz, t.descriptor.element_type),
                other => return Err(no_such_attribute("values", layout_name(other))),
            }
        };
        let shape = Shape::new(vec![nnz])
            .map_err(|e| InvalidDescriptorError::new_err(format!("invalid shape: {e}")))?;
        Self::component_view(slf, 0, shape, element_type)
    }

    /// A `hurray.Tensor` view over the COO index buffer, shape `[nnz, rank]`, `uint64`.
    ///
    /// ## Errors
    ///
    /// - `AttributeError` — this is not a COO tensor.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert t.indices.shape == (t.nnz, t.ndim)   # COO only
    /// ```
    #[getter]
    pub fn indices(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let (nnz, rank) = {
            let t = slf.borrow();
            match &t.descriptor.layout {
                LayoutDescriptor::Coo(l) => (l.nnz, t.descriptor.shape.rank() as u64),
                other => return Err(no_such_attribute("indices", layout_name(other))),
            }
        };
        let shape = Shape::new(vec![nnz, rank])
            .map_err(|e| InvalidDescriptorError::new_err(format!("invalid shape: {e}")))?;
        Self::uint64_component_view(slf, 1, shape)
    }

    /// A `hurray.Tensor` view over the CSR column-index buffer, shape `[nnz]`, `uint64`.
    ///
    /// ## Errors
    ///
    /// - `AttributeError` — this is not a CSR tensor.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert t.col_indices.shape == (t.nnz,)      # CSR only
    /// ```
    #[getter]
    pub fn col_indices(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let nnz = {
            let t = slf.borrow();
            match &t.descriptor.layout {
                LayoutDescriptor::Csr(l) => l.nnz,
                other => return Err(no_such_attribute("col_indices", layout_name(other))),
            }
        };
        let shape = Shape::new(vec![nnz])
            .map_err(|e| InvalidDescriptorError::new_err(format!("invalid shape: {e}")))?;
        Self::uint64_component_view(slf, 1, shape)
    }

    /// A `hurray.Tensor` view over the CSR row-pointer buffer, shape `[nrows + 1]`, `uint64`.
    ///
    /// ## Errors
    ///
    /// - `AttributeError` — this is not a CSR tensor.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert t.row_ptr.shape == (t.shape[0] + 1,)  # CSR only
    /// ```
    #[getter]
    pub fn row_ptr(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let nrows = {
            let t = slf.borrow();
            match &t.descriptor.layout {
                LayoutDescriptor::Csr(_) => {
                    t.descriptor.shape.dims().first().copied().ok_or_else(|| {
                        InvalidDescriptorError::new_err(
                            "CSR tensor shape must have at least 1 dimension",
                        )
                    })?
                }
                other => return Err(no_such_attribute("row_ptr", layout_name(other))),
            }
        };
        let shape = Shape::new(vec![nrows + 1])
            .map_err(|e| InvalidDescriptorError::new_err(format!("invalid shape: {e}")))?;
        Self::uint64_component_view(slf, 2, shape)
    }

    /// A 1-D `uint8` view over the buffer at descriptor index `index`.
    ///
    /// The generic way to reach any buffer, including the ones with no named
    /// accessor: CSF has `2 * rank + 1` buffers and block-paged has three. Ask the
    /// tensor's `layout` what each index holds.
    ///
    /// `uint8` is the only honest element type here — the buffers of one tensor
    /// differ, with values taking the tensor's dtype, index buffers `uint64`, and
    /// MXFP scales `e8m0` — and it cannot misreport any of them. The view covers
    /// exactly the buffer's declared byte size.
    ///
    /// ## Errors
    ///
    /// - `IndexError` — no buffer at that index.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import struct, hurray
    ///
    /// csr = hurray.Tensor(
    ///     struct.pack("2f", 5.0, 7.0), hurray.float32, [2, 2],
    ///     aux_buffers=[struct.pack("2Q", 0, 1), struct.pack("3Q", 0, 1, 2)],
    ///     layout=hurray.CsrLayout(nnz=2),
    /// )
    /// assert csr.buffer(2).shape == (24,)      # row_ptr: 3 uint64 entries
    /// assert csr.buffer(2).dtype == hurray.uint8
    /// ```
    pub fn buffer(slf: &Bound<'_, Self>, index: usize) -> PyResult<Py<PyAny>> {
        let byte_size = {
            let t = slf.borrow();
            // The *declared* size, from the descriptor's buffer table — the same
            // number a consumer reading this descriptor off the wire would see.
            t.descriptor
                .buffers
                .get(index)
                .ok_or_else(|| {
                    pyo3::exceptions::PyIndexError::new_err(format!(
                        "no buffer at index {index}; this tensor has {}",
                        t.buffer_count()
                    ))
                })?
                .byte_size()
        };
        let shape = Shape::new(vec![byte_size])
            .map_err(|e| InvalidDescriptorError::new_err(format!("invalid shape: {e}")))?;
        Self::component_view(slf, index, shape, hurray_core::ElementType::Uint8)
    }

    /// A `hurray.Tensor` view over the CSC row-index buffer, shape `[nnz]`, `uint64`.
    ///
    /// ## Errors
    ///
    /// - `AttributeError` — this is not a CSC tensor.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert t.row_indices.shape == (t.nnz,)      # CSC only
    /// ```
    #[getter]
    pub fn row_indices(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let nnz = {
            let t = slf.borrow();
            match &t.descriptor.layout {
                LayoutDescriptor::Csc(l) => l.nnz,
                other => return Err(no_such_attribute("row_indices", layout_name(other))),
            }
        };
        let shape = Shape::new(vec![nnz])
            .map_err(|e| InvalidDescriptorError::new_err(format!("invalid shape: {e}")))?;
        Self::uint64_component_view(slf, 1, shape)
    }

    /// A `hurray.Tensor` view over the CSC column-pointer buffer, shape `[ncols + 1]`, `uint64`.
    ///
    /// ## Errors
    ///
    /// - `AttributeError` — this is not a CSC tensor.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert t.col_ptr.shape == (t.shape[1] + 1,)  # CSC only
    /// ```
    #[getter]
    pub fn col_ptr(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let ncols = {
            let t = slf.borrow();
            match &t.descriptor.layout {
                LayoutDescriptor::Csc(_) => {
                    t.descriptor.shape.dims().get(1).copied().ok_or_else(|| {
                        InvalidDescriptorError::new_err(
                            "CSC tensor shape must have at least 2 dimensions",
                        )
                    })?
                }
                other => return Err(no_such_attribute("col_ptr", layout_name(other))),
            }
        };
        let shape = Shape::new(vec![ncols + 1])
            .map_err(|e| InvalidDescriptorError::new_err(format!("invalid shape: {e}")))?;
        Self::uint64_component_view(slf, 2, shape)
    }

    /// The quantization scheme, or `None` if the tensor is not quantized.
    ///
    /// Returns the same class you would pass to the constructor —
    /// `PerTensorAffine`, `PerChannelAffine`, `PerBlockAffine`, `NF4`, or `MXFP` —
    /// so a descriptor read off disk or off the wire can be inspected, and passed
    /// straight back to build another tensor.
    ///
    /// ## Errors
    ///
    /// - `hurray.InvalidDescriptorError` — the stored quantization section could
    ///   not be decoded.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import struct, hurray
    ///
    /// t = hurray.Tensor(
    ///     bytes(8), hurray.int8, [2, 4],
    ///     aux_buffers=[struct.pack("2f", 0.02, 0.017)],
    ///     quantization=hurray.PerChannelAffine.symmetric(0, 1),
    /// )
    ///
    /// q = t.quantization
    /// assert q.axis == 0
    /// assert q.scale_buffer_index == 1
    ///
    /// assert hurray.Tensor(bytes(16), hurray.float32, [4]).quantization is None
    /// ```
    #[getter]
    pub fn quantization(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let Some(bytes) = self.descriptor.quantization.as_ref() else {
            return Ok(None);
        };
        // Decoding on read rather than caching a parsed copy: the descriptor stores
        // the encoded section, and re-parsing keeps one source of truth.
        let (desc, _read) = hurray_core::QuantizationDescriptor::decode(bytes).map_err(|e| {
            InvalidDescriptorError::new_err(format!("failed to decode quantization: {e}"))
        })?;
        crate::quantization::quantization_to_py(py, desc).map(Some)
    }

    /// The statistics section, or `None` if the tensor carries none.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// t = hurray.Tensor(
    ///     bytes(24), hurray.float32, [2, 3],
    ///     statistics=hurray.Statistics(nnz=6),
    /// )
    /// assert t.statistics.nnz == 6
    /// assert hurray.Tensor(bytes(16), hurray.float32, [4]).statistics is None
    /// ```
    #[getter]
    pub fn statistics(&self, py: Python<'_>) -> PyResult<Option<Py<crate::metadata::Statistics>>> {
        match self.descriptor.statistics.as_ref() {
            None => Ok(None),
            Some(s) => Py::new(py, crate::metadata::Statistics { inner: s.clone() }).map(Some),
        }
    }

    /// The shard descriptor, or `None` if this tensor is not a shard.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// t = hurray.Tensor(
    ///     bytes(24), hurray.float32, [2, 3],
    ///     shard=hurray.Shard([4, 3], [2, 0]),
    /// )
    /// assert t.shard.parent_shape == (4, 3)
    /// assert hurray.Tensor(bytes(16), hurray.float32, [4]).shard is None
    /// ```
    #[getter]
    pub fn shard(&self, py: Python<'_>) -> PyResult<Option<Py<crate::metadata::Shard>>> {
        match self.descriptor.shard.as_ref() {
            None => Ok(None),
            Some(s) => Py::new(py, crate::metadata::Shard { inner: s.clone() }).map(Some),
        }
    }

    /// Transpose view — **not yet implemented**.
    ///
    /// Raises `NotImplementedError`; lands in a future pass.
    #[getter(T)]
    pub fn t(&self) -> PyResult<()> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Tensor.T is not yet implemented",
        ))
    }

    // ── DLPack protocol ───────────────────────────────────────────────────────

    /// Return a DLPack v1.0 capsule (`"dltensor_versioned"`) for zero-copy buffer sharing.
    ///
    /// The capsule wraps a `DLManagedTensorVersioned` that holds a strong Python
    /// reference to this `Tensor`, keeping the buffer alive for the capsule's lifetime.
    ///
    /// ## Parameters
    ///
    /// - `stream` — accepted for API compatibility but ignored; all tensors are
    ///   `ProducerSynced` in this pass (D6). GPU stream synchronisation is deferred.
    /// - `max_version`, `dl_device`, `copy` — accepted for forward-compatibility (D8);
    ///   not yet honored.
    ///
    /// ## Errors
    ///
    /// - `builtins.BufferError` — element type not in DLPack (e.g. `bool`, `int4`) (D9).
    /// - `hurray.UnsupportedError` — layout cannot be expressed as DLPack strides.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import numpy as np, hurray
    ///
    /// buf = bytes(24)
    /// t = hurray.Tensor(buf, hurray.float32, [2, 3])
    /// capsule = t.__dlpack__()
    /// arr = np.from_dlpack(capsule)
    /// assert arr.shape == (2, 3)
    /// ```
    #[pyo3(signature = (*, stream=None, max_version=None, dl_device=None, copy=None))]
    pub fn __dlpack__(
        slf: &Bound<'_, Self>,
        // D6: stream accepted but ignored; all tensors are ProducerSynced in this pass.
        // D8: max_version, dl_device, copy accepted for forward-compat; not yet honored.
        stream: Option<Py<PyAny>>,
        max_version: Option<Py<PyAny>>,
        dl_device: Option<Py<PyAny>>,
        copy: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        // Suppress unused-variable warnings for intentionally-ignored parameters.
        let _ = (stream, max_version, dl_device, copy);
        let py = slf.py();
        let t = slf.borrow();
        // DLPack describes one densely-addressable buffer; a sparse or block-paged
        // tensor has no such single representation (ADR-031 § 3).
        if !is_dense(&t.descriptor.layout) {
            // builtins.BufferError, not hurray.BufferError: docs/impl/python-bindings.md
            // requires the built-in for __dlpack__, because that is what NumPy and
            // PyTorch catch when a producer cannot satisfy the protocol.
            return Err(pyo3::exceptions::PyBufferError::new_err(format!(
                "cannot export a {} tensor via DLPack; it has no single dense buffer — \
                 use __hurray_buffer__, or export the component views individually",
                layout_name(&t.descriptor.layout)
            )));
        }

        let (dev_tag, mem_class, dev_id) = {
            let dev = t.device_py.borrow(py);
            (dev.tag, dev.memory_class, dev.device_id)
        };

        let device_type = dlpack::device_to_dlpack(dev_tag, mem_class)?;
        let data_ptr = t.buffer.as_ptr() as *mut c_void;
        let shape = t.descriptor.shape.dims();

        // Keep the Tensor alive for the capsule's entire lifetime via a strong ref.
        let tensor_obj: Py<PyAny> = slf.clone().into_any().unbind();

        dlpack::build_capsule(
            py,
            tensor_obj,
            data_ptr,
            t.descriptor.element_type,
            shape,
            &t.descriptor.layout,
            device_type,
            dev_id,
        )
    }

    /// Return the `(DLDeviceType, device_id)` tuple for this tensor.
    ///
    /// ## Examples
    ///
    /// ```python
    /// device_type, device_id = t.__dlpack_device__()
    /// assert device_type == 1  # kDLCPU
    /// ```
    pub fn __dlpack_device__(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let dev = self.device_py.borrow(py);
        let device_type = dlpack::device_to_dlpack(dev.tag, dev.memory_class)?;
        let tup = PyTuple::new(py, [device_type, dev.device_id])?.unbind();
        Ok(tup)
    }

    // ── Native buffer protocol ────────────────────────────────────────────────

    /// Return a `"hurray_buffer"` PyCapsule for zero-copy buffer sharing between
    /// Hurray-aware Python extensions (ADR-023, Layer 8c).
    ///
    /// Unlike `__dlpack__`, this protocol preserves the full Hurray descriptor
    /// (device tag, memory class, sync mode, element type, shape, layout) without
    /// flattening to DLPack's `DLDeviceType` enum. Available on all dtypes.
    ///
    /// ## Parameters
    ///
    /// - `stream` — accepted for API parity with `__dlpack__`; all tensors are
    ///   `ProducerSynced` in this pass, so the value is noted but not acted on.
    ///
    /// ## Errors
    ///
    /// - `hurray.BufferError` — tensor has no buffer handles (should not occur in practice).
    /// - `hurray.InvalidDescriptorError` — descriptor could not be encoded.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// t = hurray.Tensor(bytes(16), hurray.float32, [4])
    /// cap = t.__hurray_buffer__()
    /// t2 = hurray.from_hurray_buffer(t)
    /// assert t2.shape == t.shape
    /// assert t2.dtype == t.dtype
    /// ```
    #[pyo3(signature = (stream = None))]
    pub fn __hurray_buffer__(
        slf: &Bound<'_, Self>,
        // ProducerSynced: stream hint accepted for API parity; not acted on in this pass.
        stream: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let _ = stream;
        let py = slf.py();
        let t = slf.borrow();

        if t.descriptor.buffers.is_empty() {
            return Err(BufferError::new_err(
                "tensor has no buffer handles; cannot produce a native buffer capsule",
            ));
        }

        // Pair each buffer store with its declared handle, in descriptor order
        // (ADR-030 § 3): the store supplies the address, the handle the metadata.
        let capsule_buffers: Vec<crate::native_buffer::CapsuleBuffer> = t
            .buffers()
            .zip(t.descriptor.buffers.iter())
            .map(|(store, handle)| {
                let byte_size = store.len() as u64;
                crate::native_buffer::CapsuleBuffer {
                    data_ptr: store.as_ptr() as *mut c_void,
                    byte_size,
                    // alignment=1 for empty buffers, to satisfy the handle constructor.
                    alignment: if byte_size == 0 {
                        1
                    } else {
                        handle.alignment()
                    },
                    device_tag: handle.device_tag(),
                    sync_mode: handle.sync_mode(),
                    memory_class: handle.memory_class(),
                }
            })
            .collect();

        // Encode descriptor before releasing the borrow.
        let descriptor = &t.descriptor;
        // Strong reference to the Tensor; kept alive via the capsule context.
        let tensor_obj: Py<PyAny> = slf.clone().into_any().unbind();

        crate::native_buffer::build_capsule(py, tensor_obj, descriptor, &capsule_buffers)
    }

    // ── SciPy interop ─────────────────────────────────────────────────────────

    /// Convert a CSR or CSC tensor to the matching `scipy.sparse` matrix
    /// (zero-copy where possible).
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
    /// - `hurray.UnsupportedError` — any layout other than CSR or CSC. For COO, use
    ///   `.values` / `.indices` directly and construct `scipy.sparse.coo_matrix`.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import scipy.sparse, hurray
    ///
    /// # (Assuming `sparse` is a hurray.Tensor with layout == "csr")
    /// m = sparse.to_scipy()
    /// assert isinstance(m, scipy.sparse.csr_matrix)
    /// ```
    pub fn to_scipy(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let is_csr =
            {
                let t = slf.borrow();
                // D17: reject Tier 2 / quantized dtypes — SciPy has no equivalent.
                if !is_tier1(t.descriptor.element_type) {
                    return Err(UnsupportedError::new_err(format!(
                        "to_scipy is not supported for '{}'; SciPy has no equivalent dtype",
                        crate::dtype::element_type_name(t.descriptor.element_type),
                    )));
                }
                match &t.descriptor.layout {
                    LayoutDescriptor::Csr(_) => true,
                    LayoutDescriptor::Csc(_) => false,
                    LayoutDescriptor::Coo(_) => return Err(UnsupportedError::new_err(
                        "to_scipy is not supported for COO tensors; access .values and .indices \
                         directly and construct scipy.sparse.coo_matrix manually",
                    )),
                    other => {
                        return Err(UnsupportedError::new_err(format!(
                            "to_scipy is not supported for a {} tensor; only csr and csc map to \
                         scipy.sparse matrix types",
                            crate::layout::layout_name(other)
                        )))
                    }
                }
            };

        // D7-pattern: import scipy lazily — hurray must not require scipy at module load.
        let sp = py.import("scipy.sparse").map_err(|_| {
            pyo3::exceptions::PyImportError::new_err(
                "scipy is not installed; install with: pip install scipy",
            )
        })?;

        // The component accessors already build borrowed views in descriptor order,
        // so to_scipy reuses them rather than reaching into the buffer table again.
        let values_arr = Tensor::values(slf)?.bind(py).call_method0("__array__")?;
        let aux0_arr = if is_csr {
            Tensor::col_indices(slf)?
        } else {
            Tensor::row_indices(slf)?
        };
        let aux0_arr = aux0_arr.bind(py).call_method0("__array__")?;
        let aux1_arr = if is_csr {
            Tensor::row_ptr(slf)?
        } else {
            Tensor::col_ptr(slf)?
        };
        let aux1_arr = aux1_arr.bind(py).call_method0("__array__")?;

        let (nrows, ncols) = {
            let t = slf.borrow();
            let dims = t.descriptor.shape.dims().to_vec();
            (dims[0], dims[1])
        };

        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("shape", PyTuple::new(py, [nrows, ncols])?)?;
        kwargs.set_item("copy", false)?;

        // csr_matrix((data, indices, indptr), shape=..., copy=False)
        let args = PyTuple::new(py, [&values_arr, &aux0_arr, &aux1_arr])?;
        let constructor_name = if is_csr { "csr_matrix" } else { "csc_matrix" };
        let matrix = sp.getattr(constructor_name)?.call((args,), Some(&kwargs))?;
        Ok(matrix.into())
    }

    // ── NumPy interop ─────────────────────────────────────────────────────────

    /// Return a NumPy `ndarray` backed by this tensor's buffer (zero-copy via DLPack).
    ///
    /// Only supported for CPU tensors with Tier 1 element types. The returned array
    /// shares the buffer — modifying it will modify this tensor's data.
    ///
    /// ## Parameters
    ///
    /// - `dtype` — if supplied, the returned array will be cast to this NumPy dtype.
    /// - `copy` — if `False`, raises `CopyRequiredError` when a cast is needed (NumPy
    ///   2.0 convention, NEP 47) (D4).
    ///
    /// ## Errors
    ///
    /// - `hurray.UnsupportedError` — non-CPU tensor or Tier 2 / quantized dtype.
    /// - `builtins.BufferError` — element type not representable in DLPack (e.g. bool).
    /// - `hurray.CopyRequiredError` — `copy=False` but a dtype cast is required (D4).
    ///
    /// ## Examples
    ///
    /// ```python
    /// import struct, numpy as np, hurray
    ///
    /// buf = struct.pack("6f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    /// t = hurray.Tensor(buf, hurray.float32, [2, 3])
    /// arr = t.__array__()
    /// assert arr.shape == (2, 3)
    /// assert arr.dtype == np.float32
    /// ```
    #[pyo3(signature = (dtype = None, copy = None))]
    pub fn __array__(
        slf: &Bound<'_, Self>,
        dtype: Option<Py<PyAny>>,
        // D4: copy=False + cast needed → CopyRequiredError (NumPy 2.0 NEP 47 convention).
        copy: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let t = slf.borrow();

        // NumPy's ndarray is dense; a sparse or block-paged buffer is not an array
        // of elements in index order (ADR-031 § 3).
        if !is_dense(&t.descriptor.layout) {
            return Err(UnsupportedError::new_err(format!(
                "cannot convert a {} tensor to a NumPy array; it has no dense element \
                 buffer — use .values / .indices, or .to_scipy() for a sparse matrix",
                layout_name(&t.descriptor.layout)
            )));
        }

        // Only CPU tensors — non-CPU would require device→host copy which NumPy can't do.
        {
            let dev = t.device_py.borrow(py);
            if dev.tag != hurray_core::DeviceTag::Cpu {
                return Err(UnsupportedError::new_err(
                    "__array__ requires a CPU tensor; non-CPU tensors cannot be \
                     converted to NumPy without a device→host copy",
                ));
            }
        }

        // Only Tier 1 types (NumPy has no dtype for int4, float8, quantized types).
        if !is_tier1(t.descriptor.element_type) {
            return Err(UnsupportedError::new_err(format!(
                "__array__ is not supported for '{}'; NumPy has no equivalent dtype",
                crate::dtype::element_type_name(t.descriptor.element_type),
            )));
        }

        // Reject a type DLPack cannot carry before handing the tensor over, so the
        // error names the element type instead of surfacing from inside NumPy.
        if crate::dlpack::element_type_to_dlpack(t.descriptor.element_type).is_none() {
            return Err(pyo3::exceptions::PyBufferError::new_err(format!(
                "element type '{}' cannot be represented in DLPack v1.0; \
                 use __hurray_buffer__ instead",
                crate::dtype::element_type_name(t.descriptor.element_type),
            )));
        }

        // Use DLPack as the bridge to NumPy — numpy.from_dlpack creates a zero-copy
        // view and registers its own finaliser that calls the DLPack deleter, which
        // holds a strong ref back to this Tensor. The memory chain is:
        //   ndarray → DLPack capsule → deleter → Tensor → BufferStore
        //
        // Hand NumPy the tensor, not a capsule: from_dlpack takes an object
        // implementing __dlpack__ — NumPy 2.1 dropped raw-capsule support — and this
        // way NumPy negotiates the stream and device arguments itself.
        drop(t); // release borrow before NumPy calls back into __dlpack__
        let np = py.import("numpy")?;
        let arr = np.call_method1("from_dlpack", (slf,))?;

        // Handle dtype cast request (D4).
        if let Some(target_dtype) = dtype {
            let target_dtype_bound = target_dtype.bind(py);
            // Check whether a cast would actually change the dtype.
            let arr_dtype = arr.getattr("dtype")?;
            let needs_cast = !arr_dtype.eq(target_dtype_bound)?;
            if needs_cast {
                // D4: copy=False and cast required → raise CopyRequiredError.
                let copy_false = copy
                    .as_ref()
                    .map(|c| c.bind(py).eq(false).unwrap_or(false))
                    .unwrap_or(false);
                if copy_false {
                    return Err(CopyRequiredError::new_err(
                        "copy=False was requested but a dtype cast is required; \
                         remove copy=False or pass dtype=None to allow the cast",
                    ));
                }
                let cast = arr.call_method1("astype", (target_dtype_bound,))?;
                return Ok(cast.into());
            }
        }

        Ok(arr.into())
    }

    // ── PyTorch interop ───────────────────────────────────────────────────────

    /// Return a `torch.Tensor` sharing this tensor's buffer (zero-copy via DLPack).
    ///
    /// `torch` is imported at call time — `import hurray` does not require PyTorch
    /// to be installed (D7).
    ///
    /// ## Errors
    ///
    /// - `ImportError` — PyTorch is not installed.
    /// - `builtins.BufferError` — element type not representable in DLPack.
    /// - `hurray.UnsupportedError` — device/layout not supported via DLPack.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    /// t = hurray.Tensor(bytes(16), hurray.float32, [4])
    /// torch_t = t.to_torch()
    /// ```
    pub fn to_torch(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        // D7: import torch at call time to avoid a hard dependency at module load.
        let torch = py.import("torch").map_err(|_| {
            pyo3::exceptions::PyImportError::new_err(
                "torch is not installed; install it with: pip install torch",
            )
        })?;
        let capsule = Tensor::__dlpack__(slf, None, None, None, None)?;
        let dlpack_mod = torch.getattr("utils")?.getattr("dlpack")?;
        let torch_tensor = dlpack_mod.call_method1("from_dlpack", (capsule,))?;
        Ok(torch_tensor.into())
    }

    // ── Dunders ───────────────────────────────────────────────────────────────

    /// Tensors are unhashable — mutable objects must not be used as dict keys.
    pub fn __hash__(&self) -> PyResult<isize> {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "unhashable type: 'Tensor'",
        ))
    }

    /// Return a developer-friendly string representation.
    ///
    /// For Tier 1 CPU tensors the data values are formatted using NumPy's
    /// `array2string`. Tier 2 types and non-CPU devices fall back to a
    /// metadata-only form: `hurray.Tensor(shape=…, dtype=…, device=…)`.
    ///
    /// ```python
    /// import hurray
    /// t = hurray.ones([2, 3], dtype=hurray.float32)
    /// print(repr(t))
    /// # hurray.Tensor([[1. 1. 1.]
    /// #  [1. 1. 1.]], dtype=float32)
    /// ```
    pub fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        // Sparse layouts have their own representation — component arrays rather
        // than a dense element grid — and their own print option.
        if matches!(
            slf.borrow().descriptor.layout,
            LayoutDescriptor::Coo(_) | LayoutDescriptor::Csr(_) | LayoutDescriptor::Csc(_)
        ) {
            return crate::sparse::sparse_repr(slf);
        }
        let this = slf.borrow();
        let et = this.descriptor.element_type;
        let dtype_name = crate::dtype::element_type_name(et);
        let dev = this.device_py.borrow(py);
        let is_cpu = dev.tag == hurray_core::DeviceTag::Cpu;

        // Show data values for Tier 1 CPU tensors; fall back on any error (e.g.
        // bfloat16 has no numpy dtype; non-CPU requires a device copy).
        if is_tier1(et) && is_cpu {
            if let Ok(data_str) = this.numpy_data_string(py) {
                return Ok(format!("hurray.Tensor({data_str}, dtype={dtype_name})"));
            }
        }

        // Fallback: metadata only.
        let shape_tuple = this.shape(py)?;
        let shape_str = shape_tuple.bind(py).repr()?.to_str()?.to_owned();
        let device_str = crate::device::device_repr(&dev);
        Ok(format!(
            "hurray.Tensor(shape={shape_str}, dtype={dtype_name}, device={device_str})"
        ))
    }

    /// Return a human-readable string of the tensor data.
    ///
    /// For Tier 1 CPU tensors this is the bare NumPy-style array string
    /// (no `hurray.Tensor(…)` wrapper). Falls back to `__repr__` for Tier 2
    /// types or non-CPU devices.
    ///
    /// ```python
    /// import hurray
    /// t = hurray.arange(4, dtype=hurray.float32)
    /// print(str(t))   # [0. 1. 2. 3.]
    /// ```
    pub fn __str__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        {
            let this = slf.borrow();
            let et = this.descriptor.element_type;
            let dev = this.device_py.borrow(py);
            if is_dense(&this.descriptor.layout)
                && is_tier1(et)
                && dev.tag == hurray_core::DeviceTag::Cpu
            {
                if let Ok(s) = this.numpy_data_string(py) {
                    return Ok(s);
                }
            }
        }
        Self::__repr__(slf)
    }
}

impl Tensor {
    /// Build a borrowed `Tensor` view over one of this tensor's buffers.
    ///
    /// `buffer_index` is the descriptor buffer-table index (ADR-030 § 3): 0 is the
    /// data buffer, 1..N the auxiliary ones. The view holds a strong reference to
    /// `slf`, so the borrowed pointer stays valid for the view's lifetime.
    fn component_view(
        slf: &Bound<'_, Self>,
        buffer_index: usize,
        shape: Shape,
        element_type: hurray_core::ElementType,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let (ptr, len, device_tag, memory_class) = {
            let t = slf.borrow();
            let store = match buffer_index {
                0 => &t.buffer,
                n => t.aux_buffers.get(n - 1).ok_or_else(|| {
                    InvalidDescriptorError::new_err(format!(
                        "tensor has no buffer at index {n}; it has {}",
                        t.buffer_count()
                    ))
                })?,
            };
            let dev = t.device_py.borrow(py);
            (
                store.as_ptr() as *mut u8,
                store.len(),
                dev.tag,
                dev.memory_class,
            )
        };

        // Keep this tensor alive as the base of the borrowed view.
        let base: Py<PyAny> = slf.clone().into_any().unbind();
        let view = Tensor::new_borrowed_view(
            py,
            element_type,
            shape,
            device_tag,
            memory_class,
            base,
            ptr,
            len,
        )?;
        Ok(Py::new(py, view)?.into_any())
    }

    /// [`Tensor::component_view`] for an index buffer, which is always `uint64`.
    fn uint64_component_view(
        slf: &Bound<'_, Self>,
        buffer_index: usize,
        shape: Shape,
    ) -> PyResult<Py<PyAny>> {
        Self::component_view(slf, buffer_index, shape, hurray_core::ElementType::Uint64)
    }
}

// ── Layout helpers ────────────────────────────────────────────────────────────

/// The error for an accessor that does not apply to this tensor's layout.
///
/// `AttributeError`, not `UnsupportedError`, so that `hasattr` tells the truth
/// (ADR-031 § 2 as amended; extends design decision D10 from the sparse formats to
/// every layout). With one class covering all layouts, feature detection is how
/// callers discover what a tensor supports.
pub(crate) fn no_such_attribute(attr: &str, layout: &str) -> PyErr {
    pyo3::exceptions::PyAttributeError::new_err(format!(
        "'Tensor' object has no attribute '{attr}'; this is a {layout} tensor"
    ))
}

// ── Buffer access ─────────────────────────────────────────────────────────────

impl Tensor {
    /// Iterates every buffer in **descriptor buffer-table order** (ADR-030 § 3):
    /// the data buffer first, then the auxiliary buffers.
    ///
    /// Every consumer that needs "all buffers of this tensor" MUST go through
    /// here, so the ordering invariant that buffer indices in quantization and
    /// layout descriptors rely on is stated exactly once.
    pub fn buffers(&self) -> impl Iterator<Item = &BufferStore> {
        std::iter::once(&self.buffer).chain(self.aux_buffers.iter())
    }

    /// The number of buffers this tensor holds, including the data buffer.
    pub fn buffer_count(&self) -> usize {
        1 + self.aux_buffers.len()
    }
}

// ── Display helpers ───────────────────────────────────────────────────────────

impl Tensor {
    /// Format tensor data as a NumPy-style string via `numpy.array2string`.
    ///
    /// Returns `Err` for types with no NumPy equivalent (e.g. bfloat16).
    fn numpy_data_string(&self, py: Python<'_>) -> PyResult<String> {
        let et = self.descriptor.element_type;
        let dtype_str = crate::creation::to_numpy_dtype_name(et).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("no numpy dtype for this element type")
        })?;

        let np = py.import("numpy")?;

        // SAFETY: GIL is held; Borrowed base is kept alive by self.buffer.
        let bytes = unsafe { self.buffer.as_slice() };
        let py_bytes = pyo3::types::PyBytes::new(py, bytes);

        let kw = pyo3::types::PyDict::new(py);
        kw.set_item("dtype", dtype_str)?;
        let arr_1d = np.call_method("frombuffer", (py_bytes,), Some(&kw))?;

        let shape: Vec<i64> = self
            .descriptor
            .shape
            .dims()
            .iter()
            .map(|&d| d as i64)
            .collect();
        let arr = arr_1d.call_method1("reshape", (shape,))?;

        np.call_method1("array2string", (&arr,))?
            .extract::<String>()
    }
}

// ── Crate-internal constructors ───────────────────────────────────────────────

impl Tensor {
    /// Construct a `Tensor` that borrows a slice of another Python object's buffer.
    ///
    /// `base` is a strong Python reference to the object that owns the allocation.
    /// The Tensor holds `base` alive for its own lifetime, ensuring the pointer
    /// remains valid (zero-copy borrow, same pattern as `from_numpy`).
    ///
    /// # Safety
    ///
    /// `ptr` must point to at least `len` bytes of valid memory for as long as
    /// `base` is alive.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use pyo3::prelude::*;
    /// # use hurray_core::{ElementType, Shape, DeviceTag, MemoryClass};
    /// # Python::attach(|py| -> PyResult<()> {
    /// // Internal use: called from sparse.rs to expose component buffers.
    /// # Ok(())
    /// # });
    /// ```
    // All eight arguments are structurally necessary for zero-copy construction;
    // grouping them into a struct would add overhead without clarity benefit.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_borrowed_view(
        py: Python<'_>,
        element_type: hurray_core::ElementType,
        shape: hurray_core::Shape,
        device_tag: hurray_core::DeviceTag,
        memory_class: hurray_core::MemoryClass,
        base: Py<PyAny>,
        ptr: *mut u8,
        len: usize,
    ) -> PyResult<Self> {
        use crate::errors::{BufferError, InvalidDescriptorError};
        use hurray_core::{
            LayoutDescriptor, SyncMode, DESCRIPTOR_VERSION_MAJOR, DESCRIPTOR_VERSION_MINOR,
        };

        // Use MIN_BUFFER_ALIGNMENT (64) for non-empty buffers: Python/NumPy/SciPy
        // allocators always produce at least 64-byte-aligned data, so declaring
        // this alignment is correct and satisfies hurray-core's enforcement.
        let alignment = if len == 0 { 1 } else { MIN_BUFFER_ALIGNMENT };
        let buffer_handle = BufferHandle::with_memory_class(
            len as u64,
            alignment,
            device_tag,
            SyncMode::ProducerSynced,
            memory_class,
        )
        .map_err(|e| BufferError::new_err(format!("invalid buffer parameters: {e}")))?;

        let descriptor = TensorDescriptor::new(
            DESCRIPTOR_VERSION_MAJOR,
            DESCRIPTOR_VERSION_MINOR,
            element_type,
            shape,
            0,
            LayoutDescriptor::RowMajor,
            vec![buffer_handle],
            None,
            None,
            None,
            None,
        )
        .map_err(|e| InvalidDescriptorError::new_err(format!("invalid tensor descriptor: {e}")))?;

        let dtype_py = Py::new(
            py,
            crate::dtype::Dtype {
                inner: element_type,
            },
        )?;
        let device_py = Py::new(
            py,
            crate::device::Device {
                tag: device_tag,
                memory_class,
                device_id: 0,
            },
        )?;

        // SAFETY: ptr points into base's allocation; base is kept alive by the
        // BufferStore, ensuring ptr remains valid for the Tensor's lifetime.
        let buffer = unsafe { BufferStore::borrowed(ptr, len, base) };

        Ok(Self {
            descriptor,
            buffer,
            aux_buffers: Vec::new(),
            dtype_py,
            device_py,
        })
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Return true for Tier 1 element types (the standard numeric types NumPy can represent).
pub(crate) fn is_tier1(ty: hurray_core::ElementType) -> bool {
    use hurray_core::ElementType::*;
    matches!(
        ty,
        Bool | Int8
            | Int16
            | Int32
            | Int64
            | Uint8
            | Uint16
            | Uint32
            | Uint64
            | Float16
            | BFloat16
            | Float32
            | Float64
            | Complex64
            | Complex128
    )
}

// ── Registration ──────────────────────────────────────────────────────────────

/// Register `Tensor` on the `hurray` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tensor>()?;
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use hurray_core::{DeviceTag, ElementType, MemoryClass};
    use pyo3::Python;
    use std::os::raw::c_int;

    fn init() {
        pyo3::Python::initialize();
    }

    pub(crate) fn build_module(py: Python<'_>) -> Bound<'_, pyo3::types::PyModule> {
        let m = pyo3::types::PyModule::new(py, "hurray").unwrap();
        crate::errors::register(&m).unwrap();
        crate::dtype::register(&m).unwrap();
        crate::device::register(&m).unwrap();
        register(&m).unwrap();
        let sys = py.import("sys").unwrap();
        let modules = sys.getattr("modules").unwrap();
        modules.set_item("hurray", &m).unwrap();
        m
    }

    fn float32_buf_2x3() -> Vec<u8> {
        let floats: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        floats.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    #[test]
    fn construction_float32_succeeds() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let buf = float32_buf_2x3();
            let py_buf = PyBytes::new(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Float32,
                },
            )
            .unwrap();
            let tensor = Tensor::new(
                py,
                py_buf.as_any(),
                dtype.bind(py),
                vec![2, 3],
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .expect("construction should succeed");

            assert_eq!(tensor.ndim(), 2);
            assert_eq!(tensor.size(), Some(6));
            let shape_repr = tensor
                .shape(py)
                .unwrap()
                .bind(py)
                .repr()
                .unwrap()
                .to_str()
                .unwrap()
                .to_owned();
            assert_eq!(shape_repr, "(2, 3)");
            assert_eq!(tensor.dtype(py).borrow(py).name(), "float32");
        });
    }

    #[test]
    fn construction_int4_succeeds() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let buf: Vec<u8> = vec![0xAB, 0xCD, 0xEF, 0x12];
            let py_buf = PyBytes::new(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Int4,
                },
            )
            .unwrap();
            let tensor = Tensor::new(
                py,
                py_buf.as_any(),
                dtype.bind(py),
                vec![8],
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .expect("int4 construction should succeed");
            assert_eq!(tensor.size(), Some(8));
            assert!(tensor.dtype(py).borrow(py).is_sub_byte());
        });
    }

    #[test]
    fn construction_buffer_too_small() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let buf: Vec<u8> = vec![0u8; 12];
            let py_buf = PyBytes::new(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Float32,
                },
            )
            .unwrap();
            let result = Tensor::new(
                py,
                py_buf.as_any(),
                dtype.bind(py),
                vec![2, 3],
                None,
                None,
                None,
                None,
                None,
                None,
            );
            assert!(result.is_err());
            assert!(result.unwrap_err().is_instance_of::<BufferError>(py));
        });
    }

    #[test]
    fn construction_negative_dim() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let buf = float32_buf_2x3();
            let py_buf = PyBytes::new(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Float32,
                },
            )
            .unwrap();
            let result = Tensor::new(
                py,
                py_buf.as_any(),
                dtype.bind(py),
                vec![-1, 3],
                None,
                None,
                None,
                None,
                None,
                None,
            );
            assert!(result.is_err());
            assert!(result
                .unwrap_err()
                .is_instance_of::<InvalidDescriptorError>(py));
        });
    }

    #[test]
    fn tensor_has_no_array_namespace() {
        // hurray.Tensor is not an Array API array (ADR-029): __array_namespace__ must
        // be absent so array-agnostic consumers cleanly reject it rather than break.
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let buf = float32_buf_2x3();
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
                Tensor::new(
                    py,
                    py_buf.as_any(),
                    dtype.bind(py),
                    vec![2, 3],
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap(),
            )
            .unwrap();
            assert!(
                !tensor.bind(py).hasattr("__array_namespace__").unwrap(),
                "__array_namespace__ must not exist on hurray.Tensor"
            );
        });
    }

    #[test]
    fn hurray_buffer_capsule_is_present() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let buf = float32_buf_2x3();
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
                Tensor::new(
                    py,
                    py_buf.as_any(),
                    dtype.bind(py),
                    vec![2, 3],
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap(),
            )
            .unwrap();
            assert!(
                tensor.bind(py).hasattr("__hurray_buffer__").unwrap(),
                "__hurray_buffer__ must be present in Layer 8c"
            );
            // Calling __hurray_buffer__() must return a PyCapsule without panicking.
            let capsule = tensor
                .bind(py)
                .call_method0("__hurray_buffer__")
                .expect("__hurray_buffer__ must not raise");
            // SAFETY: capsule is a borrowed Python object from the call above.
            let is_valid = unsafe {
                pyo3::ffi::PyCapsule_IsValid(capsule.as_ptr(), c"hurray_buffer".as_ptr())
            };
            assert_eq!(is_valid, 1, "capsule must be named 'hurray_buffer'");
        });
    }

    #[test]
    fn t_raises_not_implemented() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let buf = float32_buf_2x3();
            let py_buf = PyBytes::new(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Float32,
                },
            )
            .unwrap();
            let tensor = Tensor::new(
                py,
                py_buf.as_any(),
                dtype.bind(py),
                vec![2, 3],
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();
            assert!(tensor.t().is_err());
            assert!(tensor
                .t()
                .unwrap_err()
                .is_instance_of::<pyo3::exceptions::PyNotImplementedError>(py));
        });
    }

    #[test]
    fn repr_shows_tensor_prefix_and_dtype() {
        // Tests the guaranteed structure of repr.  Whether data values are shown
        // depends on numpy being importable at runtime; that path is covered by
        // examples/display.py which runs under maturin develop (with numpy present).
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let buf = float32_buf_2x3();
            let py_buf = PyBytes::new(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Float32,
                },
            )
            .unwrap();
            let tensor = Tensor::new(
                py,
                py_buf.as_any(),
                dtype.bind(py),
                vec![2, 3],
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();
            let bound = Py::new(py, tensor).unwrap().into_bound(py);
            let r = Tensor::__repr__(&bound).unwrap();
            assert!(
                r.starts_with("hurray.Tensor("),
                "repr should have hurray.Tensor prefix"
            );
            assert!(r.contains("float32"), "repr should contain dtype");
            // dtype must not use the old verbose hurray.Dtype('...') wrapper
            assert!(!r.contains("hurray.Dtype"), "repr dtype should be unquoted");
        });
    }

    #[test]
    fn repr_fallback_for_tier2_tensor() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            // Int4 has no numpy dtype — repr must fall back to metadata-only form.
            let buf = vec![0x21u8]; // two int4 nibbles
            let py_buf = PyBytes::new(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Int4,
                },
            )
            .unwrap();
            let tensor = Tensor::new(
                py,
                py_buf.as_any(),
                dtype.bind(py),
                vec![2],
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();
            let bound = Py::new(py, tensor).unwrap().into_bound(py);
            let r = Tensor::__repr__(&bound).unwrap();
            assert!(r.contains("shape="), "fallback repr should include shape=");
            assert!(r.contains("int4"), "fallback repr should include dtype");
            assert!(!r.contains("0x"), "fallback repr should not show raw bytes");
        });
    }

    #[test]
    fn str_is_non_empty_for_tier1() {
        // When numpy is present __str__ returns a bare array string (tested via
        // examples/display.py).  Without numpy it falls back to __repr__ — both
        // paths must return a non-empty, non-panicking string.
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let buf = float32_buf_2x3();
            let py_buf = PyBytes::new(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Float32,
                },
            )
            .unwrap();
            let tensor = Tensor::new(
                py,
                py_buf.as_any(),
                dtype.bind(py),
                vec![2, 3],
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();
            let bound = Py::new(py, tensor).unwrap().into_bound(py);
            let s = Tensor::__str__(&bound).unwrap();
            assert!(!s.is_empty(), "__str__ must not be empty");
            assert!(
                s.contains("float32") || s.contains("1."),
                "__str__ should contain dtype or data"
            );
        });
    }

    #[test]
    fn default_device_is_cpu() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let buf = float32_buf_2x3();
            let py_buf = PyBytes::new(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Float32,
                },
            )
            .unwrap();
            let tensor = Tensor::new(
                py,
                py_buf.as_any(),
                dtype.bind(py),
                vec![2, 3],
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();
            assert_eq!(tensor.device_py.borrow(py).kind(), "cpu");
        });
    }

    #[test]
    fn explicit_device_preserved() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let buf = float32_buf_2x3();
            let py_buf = PyBytes::new(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Float32,
                },
            )
            .unwrap();
            let cuda_device = Py::new(
                py,
                Device {
                    tag: DeviceTag::Cuda,
                    memory_class: MemoryClass::Standard,
                    device_id: 0,
                },
            )
            .unwrap();
            let tensor = Tensor::new(
                py,
                py_buf.as_any(),
                dtype.bind(py),
                vec![2, 3],
                Some(cuda_device),
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();
            assert_eq!(tensor.device_py.borrow(py).kind(), "cuda");
        });
    }

    #[test]
    fn shape_dynamic_dim() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let shape = Shape::new(vec![1, DYNAMIC, 768]).unwrap();
            let buffer = BufferHandle::new(0, 1, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap();
            let descriptor = TensorDescriptor::new(
                DESCRIPTOR_VERSION_MAJOR,
                DESCRIPTOR_VERSION_MINOR,
                ElementType::Float32,
                shape,
                0,
                LayoutDescriptor::RowMajor,
                vec![buffer],
                None,
                None,
                None,
                None,
            )
            .unwrap();
            let device_py = Py::new(
                py,
                Device {
                    tag: DeviceTag::Cpu,
                    memory_class: MemoryClass::Standard,
                    device_id: 0,
                },
            )
            .unwrap();
            let dtype_py = Py::new(
                py,
                Dtype {
                    inner: ElementType::Float32,
                },
            )
            .unwrap();
            let tensor = Tensor {
                descriptor,
                buffer: BufferStore::from_slice(&[]),
                aux_buffers: Vec::new(),
                dtype_py,
                device_py,
            };
            let shape_tuple = tensor.shape(py).unwrap();
            let dim1 = shape_tuple.bind(py).get_item(1).unwrap();
            assert!(dim1.is_none(), "DYNAMIC dimension should map to None");
            assert_eq!(tensor.size(), None);
        });
    }

    // ── Descriptor authoring (ADR-030 Pass 2) ────────────────────────────────

    /// Build the per-channel INT8 tensor from the #146 acceptance criteria:
    /// int8 data in buffer 0, float32 scales in buffer 1, referenced by index.
    fn per_channel_int8(py: Python<'_>) -> PyResult<Tensor> {
        let _m = build_module(py);
        let data = PyBytes::new(py, &[0u8; 8]);
        let scales = PyBytes::new(py, &[0u8; 8]);
        let dtype = Py::new(
            py,
            Dtype {
                inner: ElementType::Int8,
            },
        )?;
        let quant = Py::new(py, crate::quantization::PerChannelAffine::symmetric(0, 1)?)?;
        Tensor::new(
            py,
            data.as_any(),
            dtype.bind(py),
            vec![2, 4],
            None,
            Some(vec![scales.into_any().unbind()]),
            None,
            Some(quant.bind(py).as_any()),
            None,
            None,
        )
    }

    #[test]
    fn per_channel_quantized_tensor_carries_scheme_and_scale_buffer() {
        init();
        Python::attach(|py| {
            let tensor = per_channel_int8(py).unwrap();

            // Two buffers: data and scales, in descriptor order.
            assert_eq!(tensor.buffer_count(), 2);
            assert_eq!(tensor.descriptor.buffers.len(), 2);

            // The quantization section is present and decodes to per-channel affine.
            let encoded = tensor.descriptor.quantization.as_ref().unwrap();
            let (decoded, _read) = hurray_core::QuantizationDescriptor::decode(encoded).unwrap();
            match decoded {
                hurray_core::QuantizationDescriptor::PerChannelAffine(q) => {
                    assert_eq!(q.axis(), 0);
                    assert_eq!(q.scale_buffer_index(), 1);
                    assert_eq!(q.zero_point_buffer_index(), None);
                }
                other => panic!("expected per-channel affine, got {other:?}"),
            }
        });
    }

    #[test]
    fn quantized_descriptor_survives_encode_decode() {
        init();
        Python::attach(|py| {
            let tensor = per_channel_int8(py).unwrap();
            let bytes = tensor.descriptor.encode().unwrap();
            let back = hurray_core::TensorDescriptor::decode(&bytes).unwrap();
            assert_eq!(back, tensor.descriptor);
        });
    }

    #[test]
    fn scale_buffer_index_past_the_end_is_rejected_at_construction() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let data = PyBytes::new(py, &[0u8; 8]);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Int8,
                },
            )
            .unwrap();
            // Index 1 with no aux buffer: the descriptor would encode fine and hand a
            // consumer a dangling buffer index, so it must fail here instead.
            let quant = Py::new(
                py,
                crate::quantization::PerChannelAffine::symmetric(0, 1).unwrap(),
            )
            .unwrap();
            let err = Tensor::new(
                py,
                data.as_any(),
                dtype.bind(py),
                vec![2, 4],
                None,
                None,
                None,
                Some(quant.bind(py).as_any()),
                None,
                None,
            )
            .unwrap_err();
            assert!(
                err.to_string().contains("quantization"),
                "unexpected error: {err}"
            );
        });
    }

    #[test]
    fn statistics_and_shard_reach_the_descriptor() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let data = PyBytes::new(py, &[0u8; 24]);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Float32,
                },
            )
            .unwrap();
            let stats = Py::new(
                py,
                crate::metadata::Statistics::new(
                    Some(6),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap(),
            )
            .unwrap();
            let shard = Py::new(
                py,
                crate::metadata::Shard::new(vec![4, 3], vec![2, 0]).unwrap(),
            )
            .unwrap();

            let tensor = Tensor::new(
                py,
                data.as_any(),
                dtype.bind(py),
                vec![2, 3],
                None,
                None,
                None,
                None,
                Some(stats.bind(py)),
                Some(shard.bind(py)),
            )
            .unwrap();

            let stats_out = tensor.descriptor.statistics.as_ref().unwrap();
            assert_eq!(stats_out.nnz, 6);
            assert!(stats_out.computed_mask.nnz_valid());
            // Only nnz was supplied, so nothing else claims validity.
            assert!(!stats_out.computed_mask.value_range_valid());

            let shard_out = tensor.descriptor.shard.as_ref().unwrap();
            assert_eq!(shard_out.parent_shape, vec![4, 3]);
            assert_eq!(shard_out.shard_offset, vec![2, 0]);
        });
    }

    #[test]
    fn per_tensor_affine_needs_no_aux_buffer() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let data = PyBytes::new(py, &[0u8; 8]);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Int8,
                },
            )
            .unwrap();
            let quant = Py::new(
                py,
                crate::quantization::PerTensorAffine::new(0.02, 128).unwrap(),
            )
            .unwrap();
            let tensor = Tensor::new(
                py,
                data.as_any(),
                dtype.bind(py),
                vec![2, 4],
                None,
                None,
                None,
                Some(quant.bind(py).as_any()),
                None,
                None,
            )
            .unwrap();
            // Inline scale and zero point: still a single-buffer tensor.
            assert_eq!(tensor.buffer_count(), 1);
            assert!(tensor.descriptor.quantization.is_some());
        });
    }

    #[test]
    fn a_non_quantization_object_is_rejected() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let data = PyBytes::new(py, &[0u8; 8]);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Int8,
                },
            )
            .unwrap();
            let not_a_scheme = pyo3::types::PyString::new(py, "per-channel");
            let err = Tensor::new(
                py,
                data.as_any(),
                dtype.bind(py),
                vec![2, 4],
                None,
                None,
                None,
                Some(not_a_scheme.as_any()),
                None,
                None,
            )
            .unwrap_err();
            assert!(err
                .to_string()
                .contains("expected a quantization descriptor"));
        });
    }

    // ── Descriptor inspection (ADR-030 Pass 3) ───────────────────────────────

    #[test]
    fn quantization_getter_returns_the_scheme_that_was_authored() {
        init();
        Python::attach(|py| {
            let tensor = per_channel_int8(py).unwrap();
            let bound = Py::new(py, tensor).unwrap();
            let got = bound
                .borrow(py)
                .quantization(py)
                .unwrap()
                .expect("quantized tensor must report its scheme");

            // Comes back as the same class the constructor accepts.
            let q = got.bind(py);
            assert_eq!(q.getattr("axis").unwrap().extract::<u32>().unwrap(), 0);
            assert_eq!(
                q.getattr("scale_buffer_index")
                    .unwrap()
                    .extract::<u32>()
                    .unwrap(),
                1
            );
            assert!(q.getattr("zero_point_buffer_index").unwrap().is_none());
        });
    }

    #[test]
    fn an_unquantized_tensor_reports_none() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let buf = PyBytes::new(py, &[0u8; 16]);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Float32,
                },
            )
            .unwrap();
            let tensor = Tensor::new(
                py,
                buf.as_any(),
                dtype.bind(py),
                vec![4],
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();
            assert!(tensor.quantization(py).unwrap().is_none());
            assert!(tensor.statistics(py).unwrap().is_none());
            assert!(tensor.shard(py).unwrap().is_none());
            assert_eq!(tensor.py_buffer_count(), 1);
        });
    }

    #[test]
    fn statistics_and_shard_getters_report_what_was_set() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let buf = PyBytes::new(py, &[0u8; 24]);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Float32,
                },
            )
            .unwrap();
            let stats = Py::new(
                py,
                crate::metadata::Statistics::new(
                    Some(6),
                    None,
                    Some(-1.0),
                    Some(1.0),
                    Some(1.0),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap(),
            )
            .unwrap();
            let shard = Py::new(
                py,
                crate::metadata::Shard::new(vec![4, 3], vec![2, 0]).unwrap(),
            )
            .unwrap();
            let tensor = Tensor::new(
                py,
                buf.as_any(),
                dtype.bind(py),
                vec![2, 3],
                None,
                None,
                None,
                None,
                Some(stats.bind(py)),
                Some(shard.bind(py)),
            )
            .unwrap();

            let s = tensor.statistics(py).unwrap().unwrap();
            let s = s.borrow(py);
            assert_eq!(s.nnz(), Some(6));
            assert_eq!(s.value_abs_max(), Some(1.0));
            // Never supplied, so still not claimed after the round trip.
            assert_eq!(s.value_mean(), None);

            let sh = tensor.shard(py).unwrap().unwrap();
            let sh = sh.borrow(py);
            assert_eq!(
                sh.parent_shape(py).unwrap().extract::<Vec<u64>>().unwrap(),
                vec![4, 3]
            );
        });
    }

    #[test]
    fn an_inspected_scheme_can_rebuild_an_equivalent_tensor() {
        init();
        Python::attach(|py| {
            let original = per_channel_int8(py).unwrap();
            let descriptor = original.descriptor.clone();
            let bound = Py::new(py, original).unwrap();
            let scheme = bound.borrow(py).quantization(py).unwrap().unwrap();

            // The getter's return value is accepted by the constructor unchanged,
            // which is the point of returning the same classes.
            let data = PyBytes::new(py, &[0u8; 8]);
            let scales = PyBytes::new(py, &[0u8; 8]);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Int8,
                },
            )
            .unwrap();
            let rebuilt = Tensor::new(
                py,
                data.as_any(),
                dtype.bind(py),
                vec![2, 4],
                None,
                Some(vec![scales.into_any().unbind()]),
                None,
                Some(scheme.bind(py)),
                None,
                None,
            )
            .unwrap();

            assert_eq!(rebuilt.descriptor.quantization, descriptor.quantization);
        });
    }

    #[test]
    fn tensor_is_unhashable() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let buf = float32_buf_2x3();
            let py_buf = PyBytes::new(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Float32,
                },
            )
            .unwrap();
            let tensor = Tensor::new(
                py,
                py_buf.as_any(),
                dtype.bind(py),
                vec![2, 3],
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();
            let err = tensor.__hash__().unwrap_err();
            assert!(err.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
        });
    }

    #[test]
    fn dlpack_device_cpu() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let buf = float32_buf_2x3();
            let py_buf = PyBytes::new(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Float32,
                },
            )
            .unwrap();
            let tensor = Tensor::new(
                py,
                py_buf.as_any(),
                dtype.bind(py),
                vec![2, 3],
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();
            let tup = tensor.__dlpack_device__(py).unwrap();
            let tup_bound = tup.bind(py);
            let device_type: c_int = tup_bound.get_item(0).unwrap().extract().unwrap();
            let device_id: i32 = tup_bound.get_item(1).unwrap().extract().unwrap();
            assert_eq!(device_type, 1, "kDLCPU = 1");
            assert_eq!(device_id, 0);
        });
    }

    #[test]
    fn dlpack_capsule_created() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let buf = float32_buf_2x3();
            let py_buf = PyBytes::new(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Float32,
                },
            )
            .unwrap();
            let tensor_py = Py::new(
                py,
                Tensor::new(
                    py,
                    py_buf.as_any(),
                    dtype.bind(py),
                    vec![2, 3],
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap(),
            )
            .unwrap();
            let bound = tensor_py.bind(py);
            let capsule = Tensor::__dlpack__(bound, None, None, None, None).unwrap();
            // Verify it's a capsule by checking Python type name.
            let type_name = capsule.bind(py).get_type().name().unwrap().to_string();
            assert_eq!(type_name, "PyCapsule");
        });
    }

    #[test]
    fn dlpack_bool_raises_buffer_error() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            // 1 byte can hold 8 bools in Hurray's 1-bit packed format.
            let buf: Vec<u8> = vec![0u8; 1];
            let py_buf = PyBytes::new(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Bool,
                },
            )
            .unwrap();
            let tensor_py = Py::new(
                py,
                Tensor::new(
                    py,
                    py_buf.as_any(),
                    dtype.bind(py),
                    vec![8],
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap(),
            )
            .unwrap();
            let bound = tensor_py.bind(py);
            let result = Tensor::__dlpack__(bound, None, None, None, None);
            assert!(result.is_err());
            // D9: must be builtins.BufferError.
            assert!(result
                .unwrap_err()
                .is_instance_of::<pyo3::exceptions::PyBufferError>(py));
        });
    }

    #[test]
    fn construction_without_device_module_fails_gracefully() {
        init();
        Python::attach(|py| {
            let m = pyo3::types::PyModule::new(py, "hurray_bare").unwrap();
            crate::errors::register(&m).unwrap();
            crate::dtype::register(&m).unwrap();
            let buf = float32_buf_2x3();
            let py_buf = PyBytes::new(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Float32,
                },
            )
            .unwrap();
            let _ = Tensor::new(
                py,
                py_buf.as_any(),
                dtype.bind(py),
                vec![2, 3],
                None,
                None,
                None,
                None,
                None,
                None,
            );
        });
    }

    #[test]
    fn buffer_store_is_owned_after_construction() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let buf = float32_buf_2x3();
            let py_buf = PyBytes::new(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Float32,
                },
            )
            .unwrap();
            let tensor = Tensor::new(
                py,
                py_buf.as_any(),
                dtype.bind(py),
                vec![2, 3],
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();
            assert!(matches!(tensor.buffer, BufferStore::Owned(_)));
            assert_eq!(tensor.buffer.len(), 24); // 6 × f32 = 24 bytes
        });
    }
}
