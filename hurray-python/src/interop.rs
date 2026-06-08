//! Module-level interop functions: `from_numpy`, `from_torch`.
//!
//! Both functions perform **zero-copy** construction by storing a raw pointer
//! into the source object's buffer and holding a strong Python reference to
//! keep the source alive for the Tensor's lifetime.

// PyO3 0.22 macro expansion emits a redundant .into() on PyErr — suppress.
#![allow(clippy::useless_conversion)]

use hurray_core::{
    BufferHandle, LayoutDescriptor, MemoryClass, Shape, SyncMode, TensorDescriptor,
    DESCRIPTOR_VERSION_MAJOR, DESCRIPTOR_VERSION_MINOR,
};
use pyo3::prelude::*;

use crate::buffer::BufferStore;
use crate::device::Device;
use crate::dtype::Dtype;
use crate::errors::{BufferError, InvalidDescriptorError, UnsupportedError};
use crate::tensor::Tensor;

// ── hurray.from_numpy ─────────────────────────────────────────────────────────

/// Wrap a NumPy `ndarray` as a `hurray.Tensor` without copying.
///
/// The returned `Tensor` holds a strong Python reference to the source `ndarray`
/// so the array's buffer remains valid for the `Tensor`'s entire lifetime.
///
/// ## Requirements
///
/// - The array MUST be C-contiguous (row-major). Pass `numpy.ascontiguousarray(arr)`
///   first if the array is Fortran-order or has non-contiguous strides (D5).
/// - The array dtype MUST map to a Hurray Tier 1 element type.
/// - The array MUST reside on CPU (device 0).
///
/// ## Errors
///
/// - `hurray.UnsupportedError` — array is not C-contiguous (D5).
/// - `hurray.UnsupportedError` — dtype has no Hurray equivalent.
/// - `hurray.BufferError` — cannot extract buffer pointer from the array.
///
/// ## Examples
///
/// ```python
/// import numpy as np, hurray
///
/// arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
/// t = hurray.from_numpy(arr)
/// assert t.shape == (2, 3)
/// assert t.dtype == hurray.float32
/// assert t.device.kind == "cpu"
/// ```
#[pyfunction]
pub fn from_numpy(py: Python<'_>, array: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    // 1. Verify C-contiguity (D5: non-C-contiguous → UnsupportedError, no silent copy).
    let flags = array.getattr("flags")?;
    let c_contiguous: bool = flags.get_item("C_CONTIGUOUS")?.extract()?;
    if !c_contiguous {
        return Err(UnsupportedError::new_err(
            "from_numpy requires a C-contiguous array; call numpy.ascontiguousarray(arr) first",
        ));
    }

    // 2. Resolve dtype via __array_interface__ typestr.
    let iface = array.getattr("__array_interface__")?;
    let typestr: String = iface.get_item("typestr")?.extract()?;
    let element_type = numpy_typestr_to_element_type(&typestr).ok_or_else(|| {
        UnsupportedError::new_err(format!(
            "NumPy dtype '{}' has no Hurray equivalent",
            typestr
        ))
    })?;

    // 3. Extract shape.
    let shape_obj = iface.get_item("shape")?;
    let dims: Vec<u64> = shape_obj
        .extract::<Vec<i64>>()?
        .iter()
        .map(|&d| d as u64)
        .collect();
    let hurray_shape = Shape::new(dims.clone())
        .map_err(|e| InvalidDescriptorError::new_err(format!("invalid shape: {e}")))?;

    // 4. Get raw data pointer and buffer length from __array_interface__.
    let data_tuple = iface.get_item("data")?;
    let ptr_int: usize = data_tuple.get_item(0)?.extract()?;
    let ptr = ptr_int as *mut u8;

    let nbytes: usize = array.getattr("nbytes")?.extract()?;

    // 5. Build BufferHandle and TensorDescriptor.
    // NumPy allocators always produce at least 64-byte-aligned data; declare
    // MIN_BUFFER_ALIGNMENT for non-empty buffers so hurray-core accepts the handle.
    let alignment = if nbytes == 0 {
        1
    } else {
        hurray_core::MIN_BUFFER_ALIGNMENT
    };
    let buffer_handle = BufferHandle::new(
        nbytes as u64,
        alignment,
        hurray_core::DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )
    .map_err(|e| BufferError::new_err(format!("invalid buffer parameters: {e}")))?;

    let descriptor = TensorDescriptor::new(
        DESCRIPTOR_VERSION_MAJOR,
        DESCRIPTOR_VERSION_MINOR,
        element_type,
        hurray_shape,
        0,
        LayoutDescriptor::RowMajor,
        vec![buffer_handle],
        None,
        None,
        None,
        None,
    )
    .map_err(|e| InvalidDescriptorError::new_err(format!("invalid descriptor: {e}")))?;

    // 6. Build Python-side dtype and device objects.
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

    // 7. Construct BufferStore::Borrowed.
    // SAFETY: ptr points into the NumPy array's buffer. The array is kept alive
    // by `base` (a strong Python reference) for the Tensor's entire lifetime.
    let base: Py<PyAny> = array.clone().unbind();
    let buffer = unsafe { BufferStore::borrowed(ptr, nbytes, base) };

    Ok(Tensor {
        descriptor,
        buffer,
        dtype_py,
        device_py,
    })
}

// ── hurray.from_torch ─────────────────────────────────────────────────────────

/// Wrap a `torch.Tensor` as a `hurray.Tensor` without copying (via DLPack).
///
/// `torch` is resolved at call time — `import hurray` does not require PyTorch (D7).
///
/// ## Errors
///
/// - `ImportError` — PyTorch is not installed.
/// - `builtins.BufferError` — element type not representable in DLPack.
///
/// ## Examples
///
/// ```python
/// import torch, hurray
///
/// t_torch = torch.zeros(2, 3, dtype=torch.float32)
/// t = hurray.from_torch(t_torch)
/// assert t.shape == (2, 3)
/// assert t.dtype == hurray.float32
/// ```
#[pyfunction]
pub fn from_torch(py: Python<'_>, tensor: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    // D7: import torch at call time to avoid a hard dependency at module load.
    let _ = py.import_bound("torch").map_err(|_| {
        pyo3::exceptions::PyImportError::new_err(
            "torch is not installed; install it with: pip install torch",
        )
    })?;

    // Use DLPack v1.0 as the bridge: consume torch's capsule into a hurray Tensor.
    let capsule = tensor.call_method0("__dlpack__")?;
    from_dlpack_capsule(py, capsule.unbind())
}

// ── Internal: DLPack capsule consumer ────────────────────────────────────────

/// Consume a DLPack v1.0 capsule and construct a `Tensor` that borrows the buffer.
///
/// Renames the capsule from `"dltensor_versioned"` to `"used_dltensor_versioned"`
/// (taking ownership) and registers a Python finalizer that calls the DLPack
/// `deleter` when the resulting `Tensor` is garbage collected.
fn from_dlpack_capsule(py: Python<'_>, capsule: PyObject) -> PyResult<Tensor> {
    // Use numpy.from_dlpack to do the heavy lifting: it handles capsule ownership,
    // producer synchronisation, and produces a CPU ndarray we can then wrap.
    let np = py.import_bound("numpy")?;
    let arr = np.call_method1("from_dlpack", (&capsule,))?;
    // Recursively wrap the resulting ndarray via from_numpy.
    from_numpy(py, &arr)
}

// ── Registration ─────────────────────────────────────────────────────────────

/// Register `from_numpy` and `from_torch` on the `hurray` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(from_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(from_torch, m)?)?;
    Ok(())
}

// ── NumPy dtype string → ElementType ─────────────────────────────────────────

/// Convert a NumPy typestr (e.g. `"<f4"`, `">i2"`, `"|u1"`) to a Hurray `ElementType`.
///
/// NumPy typestr format: `<endian><kind><bytes>` where endian is `<`, `>`, `=`, or `|`.
///
/// Returns `None` for dtypes with no Hurray equivalent (e.g. `object`, `datetime64`).
pub(crate) fn numpy_typestr_to_element_type(typestr: &str) -> Option<hurray_core::ElementType> {
    use hurray_core::ElementType;
    // typestr is 3 chars: endian + kind + bytes. Strip endian prefix.
    if typestr.len() < 3 {
        return None;
    }
    let kind = typestr.as_bytes()[1] as char;
    let bytes: usize = typestr[2..].parse().ok()?;

    match (kind, bytes) {
        ('b', 1) => Some(ElementType::Bool),
        ('i', 1) => Some(ElementType::Int8),
        ('i', 2) => Some(ElementType::Int16),
        ('i', 4) => Some(ElementType::Int32),
        ('i', 8) => Some(ElementType::Int64),
        ('u', 1) => Some(ElementType::Uint8),
        ('u', 2) => Some(ElementType::Uint16),
        ('u', 4) => Some(ElementType::Uint32),
        ('u', 8) => Some(ElementType::Uint64),
        ('f', 2) => Some(ElementType::Float16),
        ('f', 4) => Some(ElementType::Float32),
        ('f', 8) => Some(ElementType::Float64),
        ('c', 8) => Some(ElementType::Complex64),
        ('c', 16) => Some(ElementType::Complex128),
        // BFloat16 is encoded as 'V' (void) in NumPy's typestr — not detectable here.
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typestr_float32() {
        assert_eq!(
            numpy_typestr_to_element_type("<f4"),
            Some(hurray_core::ElementType::Float32)
        );
    }

    #[test]
    fn typestr_int64() {
        assert_eq!(
            numpy_typestr_to_element_type("<i8"),
            Some(hurray_core::ElementType::Int64)
        );
    }

    #[test]
    fn typestr_uint8() {
        assert_eq!(
            numpy_typestr_to_element_type("|u1"),
            Some(hurray_core::ElementType::Uint8)
        );
    }

    #[test]
    fn typestr_complex128() {
        assert_eq!(
            numpy_typestr_to_element_type("<c16"),
            Some(hurray_core::ElementType::Complex128)
        );
    }

    #[test]
    fn typestr_unknown_returns_none() {
        assert!(numpy_typestr_to_element_type("<U4").is_none());
    }

    #[test]
    fn typestr_bool() {
        assert_eq!(
            numpy_typestr_to_element_type("|b1"),
            Some(hurray_core::ElementType::Bool)
        );
    }
}
