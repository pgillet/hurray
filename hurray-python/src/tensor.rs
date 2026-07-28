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

// PyO3 0.22 macro expansion emits a redundant .into() on PyErr — suppress.
#![allow(clippy::useless_conversion)]

use std::os::raw::c_void;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};

use hurray_core::{
    buffer_size_bytes, BufferHandle, LayoutDescriptor, Shape, SyncMode, TensorDescriptor,
    DESCRIPTOR_VERSION_MAJOR, DESCRIPTOR_VERSION_MINOR, DYNAMIC, MIN_BUFFER_ALIGNMENT,
};

use crate::buffer::BufferStore;
use crate::device::Device;
use crate::dlpack;
use crate::dtype::Dtype;
use crate::errors::{BufferError, CopyRequiredError, InvalidDescriptorError, UnsupportedError};

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
    pub buffer: BufferStore,
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
                // Default to hurray.device.cpu — retrieve the singleton via sys.modules
                // so `tensor.device is hurray.device.cpu` holds.
                let sys = py.import_bound("sys")?;
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

        // ── 3. Extract buffer bytes ──────────────────────────────────────────
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
            buffer: BufferStore::from_slice(buf_bytes),
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
        let items: Vec<PyObject> = self
            .descriptor
            .shape
            .dims()
            .iter()
            .map(|&dim| {
                if dim == DYNAMIC {
                    py.None()
                } else {
                    // dim is u64; Python int is arbitrary-precision so no truncation.
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
        stream: Option<PyObject>,
        max_version: Option<PyObject>,
        dl_device: Option<PyObject>,
        copy: Option<PyObject>,
    ) -> PyResult<PyObject> {
        // Suppress unused-variable warnings for intentionally-ignored parameters.
        let _ = (stream, max_version, dl_device, copy);
        let py = slf.py();
        let t = slf.borrow();

        let (dev_tag, mem_class, dev_id) = {
            let dev = t.device_py.borrow(py);
            (dev.tag, dev.memory_class, dev.device_id)
        };

        let device_type = dlpack::device_to_dlpack(dev_tag, mem_class)?;
        let data_ptr = t.buffer.as_ptr() as *mut c_void;
        let shape = t.descriptor.shape.dims();

        // Keep the Tensor alive for the capsule's entire lifetime via a strong ref.
        let tensor_obj: PyObject = slf.clone().into_any().unbind();

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
        let tup = PyTuple::new_bound(py, [device_type, dev.device_id]).unbind();
        Ok(tup)
    }

    // ── Native buffer protocol ────────────────────────────────────────────────

    /// Return a `"hurray_buffer"` PyCapsule for zero-copy buffer sharing between
    /// Hurray-aware Python extensions (ADR-023, Layer 8c).
    ///
    /// Unlike `__dlpack__`, this protocol preserves the full Hurray descriptor
    /// (device tag, memory class, sync mode, element type, shape, layout) without
    /// flattening to DLPack's `DLDeviceType` enum. Available on all dtypes and
    /// in both strict and relaxed modes.
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
        stream: Option<PyObject>,
    ) -> PyResult<PyObject> {
        let _ = stream;
        let py = slf.py();
        let t = slf.borrow();

        let first_buf = t.descriptor.buffers.first().ok_or_else(|| {
            BufferError::new_err(
                "tensor has no buffer handles; cannot produce a native buffer capsule",
            )
        })?;

        let data_ptr = t.buffer.as_ptr() as *mut c_void;
        // Use alignment=1 for empty buffers to satisfy hurray_buffer_from_ptr's null-check.
        let byte_size = t.buffer.len() as u64;
        let alignment = if byte_size == 0 {
            1
        } else {
            first_buf.alignment()
        };
        let device_tag = first_buf.device_tag();
        let sync_mode = first_buf.sync_mode();
        let memory_class = first_buf.memory_class();

        // Encode descriptor before releasing the borrow.
        let descriptor = &t.descriptor;
        // Strong reference to the Tensor; kept alive via the capsule context.
        let tensor_obj: PyObject = slf.clone().into_any().unbind();

        crate::native_buffer::build_capsule(
            py,
            tensor_obj,
            descriptor,
            data_ptr,
            byte_size,
            alignment,
            device_tag,
            sync_mode,
            memory_class,
        )
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
        dtype: Option<PyObject>,
        // D4: copy=False + cast needed → CopyRequiredError (NumPy 2.0 NEP 47 convention).
        copy: Option<PyObject>,
    ) -> PyResult<PyObject> {
        let py = slf.py();
        let t = slf.borrow();

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

        // Use DLPack as the bridge to NumPy — numpy.from_dlpack creates a zero-copy
        // view and registers its own finaliser that calls the DLPack deleter, which
        // holds a strong ref back to this Tensor. The memory chain is:
        //   ndarray → DLPack capsule → deleter → Tensor → BufferStore
        drop(t); // release borrow before calling __dlpack__
        let capsule = Tensor::__dlpack__(slf, None, None, None, None)?;

        let np = py.import_bound("numpy")?;
        let arr = np.call_method1("from_dlpack", (capsule,))?;

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
    pub fn to_torch(slf: &Bound<'_, Self>) -> PyResult<PyObject> {
        let py = slf.py();
        // D7: import torch at call time to avoid a hard dependency at module load.
        let torch = py.import_bound("torch").map_err(|_| {
            pyo3::exceptions::PyImportError::new_err(
                "torch is not installed; install it with: pip install torch",
            )
        })?;
        let capsule = Tensor::__dlpack__(slf, None, None, None, None)?;
        let dlpack_mod = torch.getattr("utils")?.getattr("dlpack")?;
        let torch_tensor = dlpack_mod.call_method1("from_dlpack", (capsule,))?;
        Ok(torch_tensor.into())
    }

    // ── Array API namespace ───────────────────────────────────────────────────

    /// Return the Array API namespace (the `hurray` module) for this tensor.
    ///
    /// Raises `AttributeError` for Tier 2 dtypes in strict mode so that
    /// `hasattr(tensor, "__array_namespace__")` logic gated on calling this
    /// method behaves correctly for non-compliant tensors.
    ///
    /// Supported `api_version` values: `None`, `"2025.12"`. Other versions
    /// raise `ValueError`.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    /// t = hurray.zeros([3, 3])
    /// ns = t.__array_namespace__()
    /// assert ns is hurray
    /// ```
    #[pyo3(signature = (*, api_version = None))]
    pub fn __array_namespace__(
        &self,
        py: Python<'_>,
        api_version: Option<&str>,
    ) -> PyResult<PyObject> {
        // Raise AttributeError for Tier 2 types — these tensors do not comply
        // with the Array API, so consumers that call this method should fail.
        if !is_tier1(self.descriptor.element_type) {
            return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
                "__array_namespace__ is not available for Tier 2 dtype '{}' in strict mode",
                crate::dtype::element_type_name(self.descriptor.element_type)
            )));
        }
        match api_version {
            None | Some("2025.12") => {}
            Some(v) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unsupported api_version '{v}'; hurray implements '2025.12'"
                )));
            }
        }
        Ok(py.import_bound("hurray")?.into_any().unbind())
    }

    // ── Dunders ───────────────────────────────────────────────────────────────

    /// Tensors are unhashable — mutable objects must not be used as dict keys.
    fn __hash__(&self) -> PyResult<isize> {
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
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let et = self.descriptor.element_type;
        let dtype_name = crate::dtype::element_type_name(et);
        let dev = self.device_py.borrow(py);
        let is_cpu = dev.tag == hurray_core::DeviceTag::Cpu;

        // Show data values for Tier 1 CPU tensors; fall back on any error (e.g.
        // bfloat16 has no numpy dtype; non-CPU requires a device copy).
        if is_tier1(et) && is_cpu {
            if let Ok(data_str) = self.numpy_data_string(py) {
                return Ok(format!("hurray.Tensor({data_str}, dtype={dtype_name})"));
            }
        }

        // Fallback: metadata only.
        let shape_tuple = self.shape(py)?;
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
    fn __str__(&self, py: Python<'_>) -> PyResult<String> {
        let et = self.descriptor.element_type;
        let dev = self.device_py.borrow(py);
        if is_tier1(et) && dev.tag == hurray_core::DeviceTag::Cpu {
            if let Ok(s) = self.numpy_data_string(py) {
                return Ok(s);
            }
        }
        drop(dev);
        self.__repr__(py)
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

        let np = py.import_bound("numpy")?;

        // SAFETY: GIL is held; Borrowed base is kept alive by self.buffer.
        let bytes = unsafe { self.buffer.as_slice() };
        let py_bytes = pyo3::types::PyBytes::new_bound(py, bytes);

        let kw = pyo3::types::PyDict::new_bound(py);
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
    /// # Python::with_gil(|py| -> PyResult<()> {
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
            dtype_py,
            device_py,
        })
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Return true for Array API Tier 1 element types (those NumPy can represent).
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
mod tests {
    use super::*;
    use hurray_core::{DeviceTag, ElementType, MemoryClass};
    use pyo3::Python;
    use std::os::raw::c_int;

    fn init() {
        pyo3::prepare_freethreaded_python();
    }

    fn build_module(py: Python<'_>) -> Bound<'_, pyo3::types::PyModule> {
        let m = pyo3::types::PyModule::new_bound(py, "hurray").unwrap();
        crate::errors::register(&m).unwrap();
        crate::dtype::register(&m).unwrap();
        crate::device::register(&m).unwrap();
        register(&m).unwrap();
        let sys = py.import_bound("sys").unwrap();
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
            let tensor = Tensor::new(py, py_buf.as_any(), dtype.bind(py), vec![2, 3], None)
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
        Python::with_gil(|py| {
            let _m = build_module(py);
            let buf: Vec<u8> = vec![0xAB, 0xCD, 0xEF, 0x12];
            let py_buf = PyBytes::new_bound(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Int4,
                },
            )
            .unwrap();
            let tensor = Tensor::new(py, py_buf.as_any(), dtype.bind(py), vec![8], None)
                .expect("int4 construction should succeed");
            assert_eq!(tensor.size(), Some(8));
            assert!(tensor.dtype(py).borrow(py).is_sub_byte());
        });
    }

    #[test]
    fn construction_buffer_too_small() {
        init();
        Python::with_gil(|py| {
            let _m = build_module(py);
            let buf: Vec<u8> = vec![0u8; 12];
            let py_buf = PyBytes::new_bound(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Float32,
                },
            )
            .unwrap();
            let result = Tensor::new(py, py_buf.as_any(), dtype.bind(py), vec![2, 3], None);
            assert!(result.is_err());
            assert!(result.unwrap_err().is_instance_of::<BufferError>(py));
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
            let result = Tensor::new(py, py_buf.as_any(), dtype.bind(py), vec![-1, 3], None);
            assert!(result.is_err());
            assert!(result
                .unwrap_err()
                .is_instance_of::<InvalidDescriptorError>(py));
        });
    }

    #[test]
    fn array_namespace_tier1_present() {
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
                Tensor::new(py, py_buf.as_any(), dtype.bind(py), vec![2, 3], None).unwrap(),
            )
            .unwrap();
            // Tier 1 tensor must expose __array_namespace__ and return the hurray module.
            assert!(
                tensor.bind(py).hasattr("__array_namespace__").unwrap(),
                "__array_namespace__ must be present for Tier 1 tensors"
            );
            let ns = tensor
                .bind(py)
                .call_method0("__array_namespace__")
                .expect("__array_namespace__() should succeed for Tier 1");
            let hurray_mod = py.import_bound("hurray").unwrap();
            assert!(
                ns.is(&hurray_mod),
                "__array_namespace__() should return the hurray module"
            );
        });
    }

    #[test]
    fn array_namespace_tier2_raises_attribute_error() {
        init();
        Python::with_gil(|py| {
            let _m = build_module(py);
            let buf = vec![0xABu8, 0xCDu8, 0xEFu8, 0x12u8];
            let py_buf = PyBytes::new_bound(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Int4,
                },
            )
            .unwrap();
            let tensor = Py::new(
                py,
                Tensor::new(py, py_buf.as_any(), dtype.bind(py), vec![8], None).unwrap(),
            )
            .unwrap();
            // Tier 2 tensor: calling __array_namespace__() must raise AttributeError.
            let result = tensor.bind(py).call_method0("__array_namespace__");
            assert!(result.is_err());
            assert!(
                result
                    .unwrap_err()
                    .is_instance_of::<pyo3::exceptions::PyAttributeError>(py),
                "Tier 2 tensor must raise AttributeError from __array_namespace__()"
            );
        });
    }

    #[test]
    fn array_namespace_unsupported_version_raises_value_error() {
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
                Tensor::new(py, py_buf.as_any(), dtype.bind(py), vec![2, 3], None).unwrap(),
            )
            .unwrap();
            let kwargs = pyo3::types::PyDict::new_bound(py);
            kwargs.set_item("api_version", "2020.01").unwrap();
            let result = tensor
                .bind(py)
                .call_method("__array_namespace__", (), Some(&kwargs));
            assert!(result.is_err());
            assert!(
                result
                    .unwrap_err()
                    .is_instance_of::<pyo3::exceptions::PyValueError>(py),
                "Unknown api_version must raise ValueError"
            );
        });
    }

    #[test]
    fn hurray_buffer_capsule_is_present() {
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
                Tensor::new(py, py_buf.as_any(), dtype.bind(py), vec![2, 3], None).unwrap(),
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
                Tensor::new(py, py_buf.as_any(), dtype.bind(py), vec![2, 3], None).unwrap();
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
                Tensor::new(py, py_buf.as_any(), dtype.bind(py), vec![2, 3], None).unwrap();
            let r = tensor.__repr__(py).unwrap();
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
        Python::with_gil(|py| {
            let _m = build_module(py);
            // Int4 has no numpy dtype — repr must fall back to metadata-only form.
            let buf = vec![0x21u8]; // two int4 nibbles
            let py_buf = PyBytes::new_bound(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Int4,
                },
            )
            .unwrap();
            let tensor = Tensor::new(py, py_buf.as_any(), dtype.bind(py), vec![2], None).unwrap();
            let r = tensor.__repr__(py).unwrap();
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
                Tensor::new(py, py_buf.as_any(), dtype.bind(py), vec![2, 3], None).unwrap();
            let s = tensor.__str__(py).unwrap();
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
                Tensor::new(py, py_buf.as_any(), dtype.bind(py), vec![2, 3], None).unwrap();
            assert_eq!(tensor.device_py.borrow(py).kind(), "cpu");
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
                dtype.bind(py),
                vec![2, 3],
                Some(cuda_device),
            )
            .unwrap();
            assert_eq!(tensor.device_py.borrow(py).kind(), "cuda");
        });
    }

    #[test]
    fn shape_dynamic_dim() {
        init();
        Python::with_gil(|py| {
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
                dtype_py,
                device_py,
            };
            let shape_tuple = tensor.shape(py).unwrap();
            let dim1 = shape_tuple.bind(py).get_item(1).unwrap();
            assert!(dim1.is_none(), "DYNAMIC dimension should map to None");
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
                Tensor::new(py, py_buf.as_any(), dtype.bind(py), vec![2, 3], None).unwrap();
            let err = tensor.__hash__().unwrap_err();
            assert!(err.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
        });
    }

    #[test]
    fn dlpack_device_cpu() {
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
                Tensor::new(py, py_buf.as_any(), dtype.bind(py), vec![2, 3], None).unwrap();
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
            let tensor_py = Py::new(
                py,
                Tensor::new(py, py_buf.as_any(), dtype.bind(py), vec![2, 3], None).unwrap(),
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
        Python::with_gil(|py| {
            let _m = build_module(py);
            // 1 byte can hold 8 bools in Hurray's 1-bit packed format.
            let buf: Vec<u8> = vec![0u8; 1];
            let py_buf = PyBytes::new_bound(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Bool,
                },
            )
            .unwrap();
            let tensor_py = Py::new(
                py,
                Tensor::new(py, py_buf.as_any(), dtype.bind(py), vec![8], None).unwrap(),
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
        Python::with_gil(|py| {
            let m = pyo3::types::PyModule::new_bound(py, "hurray_bare").unwrap();
            crate::errors::register(&m).unwrap();
            crate::dtype::register(&m).unwrap();
            let buf = float32_buf_2x3();
            let py_buf = PyBytes::new_bound(py, &buf);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: ElementType::Float32,
                },
            )
            .unwrap();
            let _ = Tensor::new(py, py_buf.as_any(), dtype.bind(py), vec![2, 3], None);
        });
    }

    #[test]
    fn buffer_store_is_owned_after_construction() {
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
                Tensor::new(py, py_buf.as_any(), dtype.bind(py), vec![2, 3], None).unwrap();
            assert!(matches!(tensor.buffer, BufferStore::Owned(_)));
            assert_eq!(tensor.buffer.len(), 24); // 6 × f32 = 24 bytes
        });
    }
}
