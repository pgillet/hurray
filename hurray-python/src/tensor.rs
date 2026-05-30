//! Python bindings for the Hurray tensor scaffold.
//!
//! Exposes [`Tensor`] as `hurray.Tensor` — a Python class that holds a tensor
//! descriptor and an owned copy of the element data buffer.
//!
//! ## Phase 8a.2 scope
//!
//! This phase implements the constructor, shape/dtype/device/ndim/size properties,
//! and the `__repr__` dunder. The following are intentionally absent and will land
//! in later phases:
//!
//! - `__array_namespace__` — 8a.3 (Array API compliance)
//! - `__dlpack__` / `__dlpack_device__` — 8a.3 (zero-copy DLPack)
//! - `__array__` — 8a.3 (NumPy interop)
//! - `__hurray_buffer__` — 8c (internal buffer protocol)
//! - `Tensor.T` — 8a.4 (transpose view)
//!
//! ## Buffer ownership (D1)
//!
//! The buffer is copied into a `Vec<u8>` at construction time. Zero-copy sharing
//! via `__dlpack__` lands in Phase 8a.3.

// PyO3 0.22 macro expansion emits a redundant .into() on PyErr for functions
// returning PyResult<()> — suppress the false positive across this module.
#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};

use hurray_core::{
    buffer_size_bytes, BufferHandle, LayoutDescriptor, Shape, SyncMode, TensorDescriptor,
    DESCRIPTOR_VERSION_MAJOR, DESCRIPTOR_VERSION_MINOR, DYNAMIC, MIN_BUFFER_ALIGNMENT,
};

use crate::device::Device;
use crate::dtype::Dtype;
use crate::errors::{BufferError, InvalidDescriptorError};

/// A Hurray tensor: element type, shape, device, and owned data buffer.
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
/// ## Buffer copy (Phase 8a.2)
///
/// The buffer is **copied** into an owned `Vec<u8>` at construction time.
/// Zero-copy sharing via `__dlpack__` and `__hurray_buffer__` will be added in
/// Phases 8a.3 and 8c respectively. Until then callers should not rely on the
/// buffer identity being preserved.
///
/// ## What is NOT present in Phase 8a.2
///
/// - `__array_namespace__` — `hasattr(tensor, '__array_namespace__')` returns `False` (D2)
/// - `__dlpack__` / `__dlpack_device__` — lands in Phase 8a.3
/// - `__array__` — lands in Phase 8a.3
/// - `__hurray_buffer__` — lands in Phase 8c
/// - `Tensor.T` — raises `NotImplementedError` (D6)
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
/// assert not hasattr(t, "__array_namespace__")
/// ```
#[pyclass(name = "Tensor")]
#[derive(Debug)]
pub struct Tensor {
    /// Tensor descriptor carrying element type, shape, layout, and buffer metadata.
    pub descriptor: TensorDescriptor,
    /// Owned copy of the element data buffer (D1: zero-copy deferred to Phase 8a.3).
    pub buffer: Vec<u8>,
    /// Python-side dtype handle; holds the same object the caller passed in, so
    /// `tensor.dtype is hurray.float32` holds when the user passes a singleton (D3).
    pub dtype_py: Py<Dtype>,
    /// Python-side device handle (kept alive alongside the descriptor).
    pub device_py: Py<Device>,
}

#[pymethods]
impl Tensor {
    // ── Constructor ───────────────────────────────────────────────────────────

    /// Construct a `Tensor` from a bytes-like object, dtype, shape, and optional device.
    ///
    /// ## Errors
    ///
    /// - `hurray.InvalidDescriptorError` — negative dimension in `shape`.
    /// - `hurray.BufferError` — buffer smaller than required by `dtype` + `shape`.
    /// - `hurray.BufferError` — `buffer` is not a `bytes` / `bytearray` object.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import struct, hurray
    ///
    /// buf = struct.pack("6f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    /// t = hurray.Tensor(buf, hurray.float32, [2, 3])
    /// assert t.size == 6
    /// ```
    #[new]
    #[pyo3(signature = (buffer, dtype, shape, device = None))]
    pub fn new(
        py: Python<'_>,
        buffer: &Bound<'_, PyAny>,
        dtype: &Bound<'_, Dtype>,
        shape: Vec<i64>,
        device: Option<Py<Device>>,
    ) -> PyResult<Self> {
        // ── 1. Resolve device ────────────────────────────────────────────────
        let device_py: Py<Device> = match device {
            Some(d) => d,
            None => {
                // Default to hurray.device.cpu — retrieve via sys.modules so
                // we get the singleton already constructed by device::register.
                let sys = py.import_bound("sys")?;
                let modules = sys.getattr("modules")?;
                let device_mod = modules.get_item("hurray.device")?;
                let cpu_obj = device_mod.getattr("cpu")?;
                cpu_obj.extract::<Py<Device>>()?
            }
        };

        // Retain the caller's Dtype object so `tensor.dtype is hurray.float32`
        // holds when the user passes a registered singleton constant (D3).
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

        // ── 3. Extract buffer bytes ──────────────────────────────────────────
        // Try &[u8] first (zero-copy for bytes/bytearray in CPython);
        // fall back to PyBytes extraction.
        let buf_bytes: &[u8] = if let Ok(b) = buffer.extract::<&[u8]>() {
            b
        } else if let Ok(b) = buffer.downcast::<PyBytes>() {
            b.as_bytes()
        } else {
            return Err(BufferError::new_err(
                "expected bytes, bytearray, or buffer-compatible object",
            ));
        };

        // ── 4. Validate buffer size ──────────────────────────────────────────
        // element_count is None for DYNAMIC shapes; buffer size cannot be validated
        // until all dims are resolved. Deferred to access time (Phase 8a.3+).
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

        let buffer_handle = BufferHandle::with_memory_class(
            buf_bytes.len() as u64,
            alignment,
            device_tag,
            sync_mode,
            memory_class,
        )
        .map_err(|e| BufferError::new_err(format!("invalid buffer parameters: {e}")))?;

        let descriptor = TensorDescriptor::new(
            DESCRIPTOR_VERSION_MAJOR,
            DESCRIPTOR_VERSION_MINOR,
            dtype.get().inner,
            hurray_shape,
            0, // byte_offset: element [0,…,0] is at the start of the buffer
            LayoutDescriptor::RowMajor,
            vec![buffer_handle],
            None, // no quantization
            None, // no shard
            None, // no statistics
            None, // no extension type
        )
        .map_err(|e| InvalidDescriptorError::new_err(format!("invalid tensor descriptor: {e}")))?;

        Ok(Self {
            descriptor,
            buffer: buf_bytes.to_vec(),
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
    /// Dynamic dimensions (represented as `u64::MAX` in the wire format) are
    /// mapped to `None`. All other dimensions are returned as Python `int`.
    ///
    /// ## Examples
    ///
    /// ```python
    /// t = hurray.Tensor(buf, hurray.float32, [2, 3])
    /// assert t.shape == (2, 3)
    /// ```
    #[getter]
    pub fn shape(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let items: Vec<PyObject> = self
            .descriptor
            .shape
            .dims()
            .iter()
            .map(|&dim| {
                if dim == DYNAMIC {
                    py.None()
                } else {
                    // dim is u64; use it directly so dims > i64::MAX are not truncated.
                    // Python int is arbitrary-precision and handles the full u64 range.
                    dim.to_object(py)
                }
            })
            .collect();
        Ok(PyTuple::new_bound(py, items).unbind())
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
    /// Returns `None` when the shape contains any `DYNAMIC` dimension (D7).
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

    /// Transpose view — **not yet implemented**.
    ///
    /// Raises `NotImplementedError` with a message pointing to Phase 8a.4 (D6).
    #[getter(T)]
    pub fn t(&self) -> PyResult<()> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Tensor.T is not yet implemented; lands in Phase 8a.4",
        ))
    }

    // ── Dunders ───────────────────────────────────────────────────────────────

    /// Tensors are unhashable — mutable objects must not be used as dict keys.
    ///
    /// Raises `TypeError: unhashable type: 'Tensor'`.
    fn __hash__(&self) -> PyResult<isize> {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "unhashable type: 'Tensor'",
        ))
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let shape_tuple = self.shape(py)?;
        let shape_str = shape_tuple.bind(py).repr()?.to_str()?.to_owned();
        let dtype_name = crate::dtype::element_type_name(self.descriptor.element_type);
        let dev = self.device_py.borrow(py);
        let device_repr = crate::device::device_repr(&dev);
        Ok(format!(
            "hurray.Tensor(shape={shape_str}, dtype=hurray.Dtype('{dtype_name}'), \
             device={device_repr})"
        ))
    }
}

// ── Registration ──────────────────────────────────────────────────────────────

/// Register `Tensor` on the `hurray` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tensor>()?;
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use hurray_core::{DeviceTag, ElementType, MemoryClass};
    use pyo3::Python;

    fn init() {
        pyo3::prepare_freethreaded_python();
    }

    /// Build a minimal `hurray` Python module, register dtype + device + tensor,
    /// and return it as a `Py<PyModule>`. Used to drive tests that need the full
    /// module graph (so `hurray.device.cpu` is resolvable).
    fn build_module(py: Python<'_>) -> Bound<'_, pyo3::types::PyModule> {
        let m = pyo3::types::PyModule::new_bound(py, "hurray").unwrap();
        crate::errors::register(&m).unwrap();
        crate::dtype::register(&m).unwrap();
        crate::device::register(&m).unwrap();
        register(&m).unwrap();
        // Register the module itself so sub-module lookups work.
        let sys = py.import_bound("sys").unwrap();
        let modules = sys.getattr("modules").unwrap();
        modules.set_item("hurray", &m).unwrap();
        m
    }

    fn float32_buf_2x3() -> Vec<u8> {
        // 6 × f32 = 24 bytes
        let floats: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        floats.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    #[test]
    fn construction_float32_succeeds() {
        init();
        Python::with_gil(|py| {
            let _m = build_module(py);
            let buf = float32_buf_2x3();
            let py_buf = PyBytes::new_bound(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Float32,
                },
            )
            .unwrap();
            let tensor = Tensor::new(py, py_buf.as_any(), &dtype.bind(py), vec![2, 3], None)
                .expect("construction should succeed");

            assert_eq!(tensor.ndim(), 2);
            assert_eq!(tensor.size(), Some(6));

            let shape_tuple = tensor.shape(py).unwrap();
            let shape_repr = shape_tuple
                .bind(py)
                .repr()
                .unwrap()
                .to_str()
                .unwrap()
                .to_owned();
            assert_eq!(shape_repr, "(2, 3)");

            let dtype_obj = tensor.dtype(py);
            let dtype_ref = dtype_obj.borrow(py);
            assert_eq!(dtype_ref.name(), "float32");
        });
    }

    #[test]
    fn construction_int4_succeeds() {
        init();
        Python::with_gil(|py| {
            let _m = build_module(py);
            // 8 elements × 4 bits = 32 bits = 4 bytes.
            let buf: Vec<u8> = vec![0xAB, 0xCD, 0xEF, 0x12];
            let py_buf = PyBytes::new_bound(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Int4,
                },
            )
            .unwrap();
            let tensor = Tensor::new(py, py_buf.as_any(), &dtype.bind(py), vec![8], None)
                .expect("int4 construction should succeed");

            assert_eq!(tensor.size(), Some(8));
            let dtype_obj = tensor.dtype(py);
            assert!(dtype_obj.borrow(py).is_sub_byte());
        });
    }

    #[test]
    fn construction_buffer_too_small() {
        init();
        Python::with_gil(|py| {
            let _m = build_module(py);
            let buf: Vec<u8> = vec![0u8; 12]; // too small for 6 × f32 = 24 bytes
            let py_buf = PyBytes::new_bound(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Float32,
                },
            )
            .unwrap();
            let result = Tensor::new(py, py_buf.as_any(), &dtype.bind(py), vec![2, 3], None);
            assert!(result.is_err(), "undersized buffer should return Err");
            let err = result.unwrap_err();
            assert!(
                err.is_instance_of::<BufferError>(py),
                "should be BufferError"
            );
        });
    }

    #[test]
    fn construction_negative_dim() {
        init();
        Python::with_gil(|py| {
            let _m = build_module(py);
            let buf = float32_buf_2x3();
            let py_buf = PyBytes::new_bound(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Float32,
                },
            )
            .unwrap();
            let result = Tensor::new(py, py_buf.as_any(), &dtype.bind(py), vec![-1, 3], None);
            assert!(result.is_err(), "negative dim should return Err");
            let err = result.unwrap_err();
            assert!(
                err.is_instance_of::<InvalidDescriptorError>(py),
                "should be InvalidDescriptorError"
            );
        });
    }

    #[test]
    fn no_array_namespace() {
        init();
        Python::with_gil(|py| {
            let _m = build_module(py);
            let buf = float32_buf_2x3();
            let py_buf = PyBytes::new_bound(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Float32,
                },
            )
            .unwrap();
            let tensor = Py::new(
                py,
                Tensor::new(py, py_buf.as_any(), &dtype.bind(py), vec![2, 3], None).unwrap(),
            )
            .unwrap();
            // `__array_namespace__` must NOT be present (D2).
            let has_attr = tensor.bind(py).hasattr("__array_namespace__").unwrap();
            assert!(
                !has_attr,
                "__array_namespace__ must not be on Tensor in 8a.2"
            );
        });
    }

    #[test]
    fn no_dlpack() {
        init();
        Python::with_gil(|py| {
            let _m = build_module(py);
            let buf = float32_buf_2x3();
            let py_buf = PyBytes::new_bound(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Float32,
                },
            )
            .unwrap();
            let tensor = Py::new(
                py,
                Tensor::new(py, py_buf.as_any(), &dtype.bind(py), vec![2, 3], None).unwrap(),
            )
            .unwrap();
            assert!(
                !tensor.bind(py).hasattr("__dlpack__").unwrap(),
                "__dlpack__ must not be on Tensor in 8a.2"
            );
        });
    }

    #[test]
    fn no_hurray_buffer() {
        init();
        Python::with_gil(|py| {
            let _m = build_module(py);
            let buf = float32_buf_2x3();
            let py_buf = PyBytes::new_bound(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Float32,
                },
            )
            .unwrap();
            let tensor = Py::new(
                py,
                Tensor::new(py, py_buf.as_any(), &dtype.bind(py), vec![2, 3], None).unwrap(),
            )
            .unwrap();
            assert!(
                !tensor.bind(py).hasattr("__hurray_buffer__").unwrap(),
                "__hurray_buffer__ must not be on Tensor in 8a.2"
            );
        });
    }

    #[test]
    fn t_raises_not_implemented() {
        init();
        Python::with_gil(|py| {
            let _m = build_module(py);
            let buf = float32_buf_2x3();
            let py_buf = PyBytes::new_bound(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Float32,
                },
            )
            .unwrap();
            let tensor =
                Tensor::new(py, py_buf.as_any(), &dtype.bind(py), vec![2, 3], None).unwrap();
            let result = tensor.t();
            assert!(result.is_err(), "Tensor.T should raise NotImplementedError");
            let err = result.unwrap_err();
            assert!(
                err.is_instance_of::<pyo3::exceptions::PyNotImplementedError>(py),
                "should be NotImplementedError"
            );
        });
    }

    #[test]
    fn repr_contains_shape_and_dtype() {
        init();
        Python::with_gil(|py| {
            let _m = build_module(py);
            let buf = float32_buf_2x3();
            let py_buf = PyBytes::new_bound(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Float32,
                },
            )
            .unwrap();
            let tensor =
                Tensor::new(py, py_buf.as_any(), &dtype.bind(py), vec![2, 3], None).unwrap();
            let r = tensor.__repr__(py).unwrap();
            assert!(r.contains("(2, 3)"), "repr should contain shape: got '{r}'");
            assert!(
                r.contains("float32"),
                "repr should contain dtype name: got '{r}'"
            );
        });
    }

    #[test]
    fn default_device_is_cpu() {
        init();
        Python::with_gil(|py| {
            let _m = build_module(py);
            let buf = float32_buf_2x3();
            let py_buf = PyBytes::new_bound(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Float32,
                },
            )
            .unwrap();
            let tensor =
                Tensor::new(py, py_buf.as_any(), &dtype.bind(py), vec![2, 3], None).unwrap();
            let dev = tensor.device_py.borrow(py);
            assert_eq!(dev.kind(), "cpu");
        });
    }

    #[test]
    fn explicit_device_preserved() {
        init();
        Python::with_gil(|py| {
            let _m = build_module(py);
            let buf = float32_buf_2x3();
            let py_buf = PyBytes::new_bound(py, &buf);
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
                &dtype.bind(py),
                vec![2, 3],
                Some(cuda_device),
            )
            .unwrap();
            let dev = tensor.device_py.borrow(py);
            assert_eq!(dev.kind(), "cuda");
        });
    }

    #[test]
    fn shape_dynamic_dim() {
        // TODO: expose DYNAMIC dim creation via TensorDescriptor in Phase 8a.3
        // when descriptor creation from wire format is added. For now verify that
        // the DYNAMIC constant maps to None in the shape getter by building a
        // descriptor directly.
        init();
        Python::with_gil(|py| {
            let _m = build_module(py);

            // Build a descriptor with a DYNAMIC dimension by-hand (bypassing
            // Tensor::new which requires non-negative dims from Python i64).
            let shape = Shape::new(vec![1, DYNAMIC, 768]).unwrap();
            let buffer = BufferHandle::new(
                0, // empty buffer — DYNAMIC size unknown
                1,
                DeviceTag::Cpu,
                SyncMode::ProducerSynced,
            )
            .unwrap();
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
                buffer: vec![],
                dtype_py,
                device_py,
            };

            // shape getter must map DYNAMIC to None.
            let shape_tuple = tensor.shape(py).unwrap();
            let shape_bound = shape_tuple.bind(py);
            let dim1 = shape_bound.get_item(1).unwrap();
            assert!(
                dim1.is_none(),
                "DYNAMIC dimension should map to None in Python shape"
            );

            // size() must be None when any dim is DYNAMIC.
            assert_eq!(tensor.size(), None);
        });
    }

    #[test]
    fn tensor_is_unhashable() {
        init();
        Python::with_gil(|py| {
            let _m = build_module(py);
            let buf = float32_buf_2x3();
            let py_buf = PyBytes::new_bound(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Float32,
                },
            )
            .unwrap();
            let tensor =
                Tensor::new(py, py_buf.as_any(), &dtype.bind(py), vec![2, 3], None).unwrap();
            let result = tensor.__hash__();
            assert!(result.is_err(), "Tensor.__hash__ must raise TypeError");
            let err = result.unwrap_err();
            assert!(
                err.is_instance_of::<pyo3::exceptions::PyTypeError>(py),
                "error should be TypeError"
            );
        });
    }

    /// Smoke-test: Tensor construction fails gracefully when `sys.modules`
    /// does not contain `hurray.device` (i.e. device::register was not called).
    #[test]
    fn construction_without_device_module_fails_gracefully() {
        init();
        Python::with_gil(|py| {
            // Minimal module without device::register.
            let m = pyo3::types::PyModule::new_bound(py, "hurray_bare").unwrap();
            crate::errors::register(&m).unwrap();
            crate::dtype::register(&m).unwrap();
            // Intentionally do NOT call device::register or tensor::register.

            let buf = float32_buf_2x3();
            let py_buf = PyBytes::new_bound(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Float32,
                },
            )
            .unwrap();
            // This call will fail because `hurray.device` is not in sys.modules.
            // We just verify it doesn't panic — it may return Ok or Err depending on
            // whether a stale `hurray.device` entry is still in sys.modules from a
            // previous test. Either outcome is acceptable.
            let _ = Tensor::new(py, py_buf.as_any(), &dtype.bind(py), vec![2, 3], None);
        });
    }
}
