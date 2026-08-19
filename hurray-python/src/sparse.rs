//! Sparse layout support for `hurray.Tensor`.
//!
//! Since ADR-031 there is no separate `SparseTensor` class: a sparse tensor is a
//! `hurray.Tensor` whose layout happens to be COO, CSR, or CSC. This module holds
//! the constructors, the component-buffer layout rules, and the sparse
//! representation used by `Tensor.__repr__`.
//!
//! ## Supported layouts
//!
//! | Layout | Buffers, in descriptor order |
//! |--------|------------------------------|
//! | COO | values, indices `[nnz, rank]` uint64 |
//! | CSR | values, col_indices `[nnz]` uint64, row_ptr `[nrows+1]` uint64 |
//! | CSC | values, row_indices `[nnz]` uint64, col_ptr `[ncols+1]` uint64 |
//!
//! Accessors that don't apply to a tensor's layout raise `AttributeError`, so
//! `hasattr` reports the truth — design decision D10, extended from the sparse
//! formats to every layout by ADR-031 § 2.

use pyo3::prelude::*;

use hurray_core::{
    layout::{CooLayout, CscLayout, CsrLayout},
    BufferHandle, ElementType, LayoutDescriptor, MemoryClass, Shape, SyncMode, TensorDescriptor,
    DESCRIPTOR_VERSION_MAJOR, DESCRIPTOR_VERSION_MINOR,
};

use crate::buffer::BufferStore;
use crate::device::Device;
use crate::dtype::Dtype;
use crate::errors::{BufferError, InvalidDescriptorError, UnsupportedError};
use crate::tensor::Tensor;

// ── SparseFormat ──────────────────────────────────────────────────────────────

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
/// Render a sparse `Tensor` as a string, honouring the `sparse_display` print option.
///
/// Lives here rather than in `tensor.rs` because it is entirely about the sparse
/// component arrays; `Tensor::__repr__` dispatches to it for COO/CSR/CSC layouts.
pub(crate) fn sparse_repr(slf: &Bound<'_, Tensor>) -> PyResult<String> {
    let py = slf.py();
    let t = slf.borrow();
    let shape_tuple = t.shape(py)?;
    let shape_str = shape_tuple.bind(py).repr()?.to_str()?.to_owned();
    let dtype_name = crate::dtype::element_type_name(t.descriptor.element_type);
    let layout = crate::layout::layout_name(&t.descriptor.layout);
    let nnz = t.nnz()?;
    drop(t);

    // Build the invariant metadata prefix that both modes share.
    let metadata = format!(
        "hurray.Tensor(layout='{layout}', shape={shape_str}, nnz={nnz}, dtype={dtype_name}"
    );

    // Check the ContextVar. Falls back to false (metadata) on any Python error.
    if !crate::print_options::sparse_display_is_content(py) {
        return Ok(format!("{metadata})"));
    }

    // Content mode: append per-layout buffer arrays using NumPy formatting.
    // On any failure, fall back to the metadata string — __repr__ must not raise.
    match format_content_arrays(slf) {
        Ok(array_part) => Ok(format!("{metadata}, {array_part})")),
        Err(_) => Ok(format!("{metadata})")),
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
fn format_content_arrays(slf: &Bound<'_, Tensor>) -> PyResult<String> {
    let py = slf.py();
    // Content rendering needs numpy (the Tensor views format through it). If numpy is
    // absent, bail here so __repr__ falls back cleanly to the metadata string.
    py.import("numpy")?;

    let layout = crate::layout::layout_name(&slf.borrow().descriptor.layout);

    /// Render a `hurray.Tensor` buffer view as its bare NumPy array string.
    ///
    /// Delegates to the Tensor's own `__str__` (the `numpy.frombuffer` path used by
    /// dense display) rather than `__array__`/DLPack, which does not work for the
    /// borrowed buffer views the sparse accessors return. `__str__` yields the compact
    /// inline `[1. 2. 3.]` form (NumPy's default separator), matching PyTorch's style.
    fn tensor_to_str(py: Python<'_>, tensor_obj: Py<PyAny>) -> PyResult<String> {
        tensor_obj.bind(py).str()?.extract::<String>()
    }

    let values_str = tensor_to_str(py, Tensor::values(slf)?)?;

    let array_part = match layout {
        "coo" => {
            let indices_str = tensor_to_str(py, Tensor::indices(slf)?)?;
            format!("indices={indices_str}, values={values_str}")
        }
        "csr" => {
            let col_indices_str = tensor_to_str(py, Tensor::col_indices(slf)?)?;
            let row_ptr_str = tensor_to_str(py, Tensor::row_ptr(slf)?)?;
            format!("values={values_str}, col_indices={col_indices_str}, row_ptr={row_ptr_str}")
        }
        "csc" => {
            let row_indices_str = tensor_to_str(py, Tensor::row_indices(slf)?)?;
            let col_ptr_str = tensor_to_str(py, Tensor::col_ptr(slf)?)?;
            format!("values={values_str}, row_indices={row_indices_str}, col_ptr={col_ptr_str}")
        }
        // sparse_repr only calls this for the three sparse layouts.
        other => {
            return Err(UnsupportedError::new_err(format!(
                "no component display for a {other} tensor"
            )))
        }
    };

    Ok(array_part)
}

/// Build a sparse `Tensor` from pre-validated component buffers and a descriptor.
///
/// Since ADR-031 there is one tensor class: the component buffers become the
/// tensor's buffer table in descriptor order — values at index 0, then the index
/// arrays — exactly as the layout's buffer-count rule requires.
///
/// Caller is responsible for ensuring the buffers are consistent with `descriptor`.
pub(crate) fn make_sparse_tensor(
    py: Python<'_>,
    descriptor: TensorDescriptor,
    values_buf: BufferStore,
    aux_buf_0: BufferStore,
    aux_buf_1: Option<BufferStore>,
) -> PyResult<Tensor> {
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
    let mut aux_buffers = vec![aux_buf_0];
    aux_buffers.extend(aux_buf_1);
    Ok(Tensor {
        descriptor,
        buffer: values_buf,
        aux_buffers,
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

/// Require a NumPy array to be C-contiguous (no silent copy on borrow).
fn require_c_contiguous(arr: &Bound<'_, PyAny>, name: &str) -> PyResult<()> {
    let c: bool = arr.getattr("flags")?.get_item("C_CONTIGUOUS")?.extract()?;
    if !c {
        return Err(UnsupportedError::new_err(format!(
            "{name} must be C-contiguous; call numpy.ascontiguousarray({name}) first"
        )));
    }
    Ok(())
}

/// Read a NumPy array's shape from `__array_interface__`.
fn array_shape(arr: &Bound<'_, PyAny>) -> PyResult<Vec<u64>> {
    let dims: Vec<i64> = arr
        .getattr("__array_interface__")?
        .get_item("shape")?
        .extract()?;
    Ok(dims.iter().map(|&d| d.max(0) as u64).collect())
}

/// Construct a COO [`Tensor`] from packed component arrays, zero-copy.
///
/// `values` is a 1-D array of `nnz` elements. `indices` is a 2-D `uint64` array of shape
/// `[nnz, rank]` giving each non-zero's coordinates in row-major (C-contiguous) order —
/// Hurray's packed COO layout. `shape` is the dense tensor shape; its length is the rank
/// and must equal `indices.shape[1]`.
///
/// Both arrays are shared without copying; the returned tensor holds strong references to
/// keep them alive. `scipy.sparse.coo_matrix` stores row/col as two arrays, so it is not
/// accepted directly — repack first:
/// `indices = numpy.stack([m.row, m.col], axis=1).astype(numpy.uint64)`.
///
/// ## Examples
///
/// ```python
/// import numpy as np, hurray
///
/// values = np.array([5.0, 7.0], dtype=np.float32)
/// indices = np.array([[0, 0], [1, 1]], dtype=np.uint64)  # [nnz, rank]
/// t = hurray.sparse_coo(values, indices, [2, 2])
/// assert t.layout == "coo"
/// assert t.nnz == 2
/// ```
#[pyfunction]
#[pyo3(signature = (values, indices, shape))]
pub fn sparse_coo(
    py: Python<'_>,
    values: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    shape: Vec<i64>,
) -> PyResult<Tensor> {
    if shape.iter().any(|&d| d < 0) {
        return Err(InvalidDescriptorError::new_err(
            "shape must have non-negative dimensions",
        ));
    }
    let rank = shape.len();
    let hurray_shape = Shape::new(shape.iter().map(|&d| d as u64).collect::<Vec<u64>>())
        .map_err(|e| InvalidDescriptorError::new_err(format!("invalid shape: {e}")))?;

    // indices: 2-D, C-contiguous, uint64, [nnz, rank].
    require_c_contiguous(indices, "indices")?;
    let idx_shape = array_shape(indices)?;
    if idx_shape.len() != 2 {
        return Err(InvalidDescriptorError::new_err(format!(
            "indices must be a 2-D array of shape [nnz, rank]; got {}-D",
            idx_shape.len()
        )));
    }
    let nnz = idx_shape[0];
    if idx_shape[1] != rank as u64 {
        return Err(InvalidDescriptorError::new_err(format!(
            "indices.shape[1] ({}) must equal the tensor rank ({rank})",
            idx_shape[1]
        )));
    }
    let (idx_ptr, idx_len, idx_et) = crate::scipy_interop::extract_numpy_buffer(py, indices, None)?;
    if idx_et != ElementType::Uint64 {
        return Err(UnsupportedError::new_err(format!(
            "indices must be uint64; got {idx_et:?}. Cast first: indices.astype(numpy.uint64)"
        )));
    }

    // values: 1-D, C-contiguous, length nnz.
    require_c_contiguous(values, "values")?;
    let val_shape = array_shape(values)?;
    if val_shape.len() != 1 || val_shape[0] != nnz {
        return Err(InvalidDescriptorError::new_err(format!(
            "values must be a 1-D array of length nnz ({nnz}); got shape {val_shape:?}"
        )));
    }
    let (val_ptr, val_len, element_type) =
        crate::scipy_interop::extract_numpy_buffer(py, values, None)?;

    let descriptor = build_coo_descriptor(
        element_type,
        hurray_shape,
        nnz,
        val_len as u64,
        idx_len as u64,
    )?;

    // Zero-copy: each source array is the base that keeps its borrowed buffer alive.
    let values_base: Py<PyAny> = values.clone().unbind();
    let indices_base: Py<PyAny> = indices.clone().unbind();
    // SAFETY: pointers come from the arrays' __array_interface__; the bases keep the
    // backing memory alive for the lifetime of the returned tensor's buffer views.
    let values_buf = unsafe { BufferStore::borrowed(val_ptr, val_len, values_base) };
    let indices_buf = unsafe { BufferStore::borrowed(idx_ptr, idx_len, indices_base) };

    make_sparse_tensor(py, descriptor, values_buf, indices_buf, None)
}

/// Build a COO `TensorDescriptor`. Used by [`sparse_coo`].
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

/// Register the sparse constructors on the `hurray` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sparse_coo, m)?)?;
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(crate) mod tests {
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

    /// Build a minimal COO tensor for tests.
    fn make_coo(py: Python<'_>) -> Tensor {
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
        Tensor {
            descriptor,
            buffer: BufferStore::from_slice(&values),
            aux_buffers: vec![BufferStore::from_slice(&indices)],
            dtype_py,
            device_py,
        }
    }

    /// Build a minimal CSC tensor for tests.
    fn make_csc(py: Python<'_>) -> Tensor {
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
        Tensor {
            descriptor,
            buffer: BufferStore::from_slice(&values),
            aux_buffers: vec![
                BufferStore::from_slice(&row_indices),
                BufferStore::from_slice(&col_ptr),
            ],
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

    pub(crate) fn make_csr(py: Python<'_>) -> Tensor {
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

        Tensor {
            descriptor,
            buffer: BufferStore::from_slice(&values),
            aux_buffers: vec![
                BufferStore::from_slice(&col_idx),
                BufferStore::from_slice(&row_ptr),
            ],
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
            assert_eq!(crate::layout::layout_name(&sparse.descriptor.layout), "csr");
            assert_eq!(sparse.ndim(), 2);
            assert_eq!(sparse.nnz().unwrap(), 4);
        });
    }

    #[test]
    fn sparse_coo_constructs_from_packed_arrays() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            if py.import("numpy").is_err() {
                return; // numpy not available in this environment
            }
            let np = py.import("numpy").unwrap();
            let vkw = pyo3::types::PyDict::new(py);
            vkw.set_item("dtype", "float32").unwrap();
            let values = np
                .call_method("array", (vec![5.0f64, 7.0],), Some(&vkw))
                .unwrap();
            let ikw = pyo3::types::PyDict::new(py);
            ikw.set_item("dtype", "uint64").unwrap();
            let indices = np
                .call_method("array", (vec![vec![0u64, 0], vec![1, 1]],), Some(&ikw))
                .unwrap();

            let t = sparse_coo(py, &values, &indices, vec![2, 2]).unwrap();
            assert_eq!(crate::layout::layout_name(&t.descriptor.layout), "coo");
            assert_eq!(t.ndim(), 2);
            assert_eq!(t.nnz().unwrap(), 2);
        });
    }

    #[test]
    fn sparse_coo_rejects_non_uint64_indices() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            if py.import("numpy").is_err() {
                return;
            }
            let np = py.import("numpy").unwrap();
            let values = np.call_method1("array", (vec![5.0f64],)).unwrap();
            // int32 indices — must be rejected (Hurray COO indices are uint64).
            let ikw = pyo3::types::PyDict::new(py);
            ikw.set_item("dtype", "int32").unwrap();
            let indices = np
                .call_method("array", (vec![vec![0i64, 0]],), Some(&ikw))
                .unwrap();
            let err = sparse_coo(py, &values, &indices, vec![2, 2]).unwrap_err();
            assert!(err.is_instance_of::<UnsupportedError>(py));
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
            let r = crate::sparse::sparse_repr(bound).unwrap();
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
                crate::sparse::sparse_repr(bound).unwrap(),
                Tensor::__str__(bound).unwrap(),
                "__str__ and __repr__ must be identical for a sparse tensor"
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
            let result = Tensor::indices(bound);
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
                Tensor {
                    descriptor,
                    buffer: BufferStore::from_slice(&values),
                    aux_buffers: vec![
                        BufferStore::from_slice(&row_idx),
                        BufferStore::from_slice(&col_ptr),
                    ],
                    dtype_py,
                    device_py,
                },
            )
            .unwrap();
            let bound = sparse.bind(py);
            // col_indices is a CSR attribute — should raise AttributeError for CSC.
            let result = Tensor::col_indices(bound);
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
            let vals_obj = Tensor::values(bound).unwrap();
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
            let rp_obj = Tensor::row_ptr(bound).unwrap();
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
            let ci_obj = Tensor::col_indices(bound).unwrap();
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
            // Build a COO tensor
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
                Tensor {
                    descriptor,
                    buffer: BufferStore::from_slice(&values),
                    aux_buffers: vec![BufferStore::from_slice(&indices)],
                    dtype_py,
                    device_py,
                },
            )
            .unwrap();
            let bound = sparse.bind(py);
            let result = Tensor::to_scipy(bound);
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
            let r = crate::sparse::sparse_repr(bound).unwrap();
            // Metadata mode: compact summary only.
            assert!(
                r.starts_with("hurray.Tensor("),
                "repr must start with type name"
            );
            assert!(r.contains("layout='csr'"), "repr must contain layout");
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
            let r = crate::sparse::sparse_repr(bound).unwrap();
            assert!(r.contains("layout='coo'"));
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
            let r = crate::sparse::sparse_repr(bound).unwrap();
            assert!(r.contains("layout='csc'"));
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
            let r = crate::sparse::sparse_repr(bound).unwrap();
            // The repr must at minimum contain the invariant metadata fields.
            assert!(
                r.contains("layout='csr'"),
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
            let r = crate::sparse::sparse_repr(bound).unwrap();
            assert!(
                r.contains("layout='coo'"),
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
            let r = crate::sparse::sparse_repr(bound).unwrap();
            assert!(
                r.contains("layout='csc'"),
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
            let r = crate::sparse::sparse_repr(sparse.bind(py)).unwrap();

            // Spec: __repr__ never raises; falls back to the metadata string.
            assert!(
                r.starts_with("hurray.Tensor("),
                "fallback repr must be a valid metadata string (got: {r})"
            );
            assert!(
                r.contains("layout='csr'"),
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
                crate::sparse::sparse_repr(bound).unwrap(),
                Tensor::__str__(bound).unwrap(),
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
            let r = crate::sparse::sparse_repr(sparse.bind(py)).unwrap();
            assert!(
                r.starts_with("hurray.Tensor("),
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
                crate::tensor::Tensor::new(
                    py,
                    py_buf.as_any(),
                    dtype.bind(py),
                    vec![4],
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
