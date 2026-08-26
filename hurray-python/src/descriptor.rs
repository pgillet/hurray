//! `hurray.Descriptor` — a tensor descriptor as a value, apart from its bytes.
//!
//! ```python
//! wire = tensor.descriptor.encode()
//! back = hurray.Descriptor.decode(wire)
//! assert back == tensor.descriptor
//! ```
//!
//! ## Why a class of its own
//!
//! The descriptor is the format's central artifact: self-delimiting, written before the
//! buffers it describes, and the only thing a consumer needs in order to know what the
//! following bytes are. Everything else `hurray-python` exposes has been a way of
//! *holding* one — a `Tensor` is a descriptor plus buffers — with no way to produce the
//! artifact itself, so a Python program could not put a Hurray tensor in a container of
//! its own, nor read one that arrived out of band.
//!
//! It is a separate class rather than a method on `Tensor` returning bytes because decode
//! has to return *something*, and that something has no buffers. Handing back a `Tensor`
//! with nothing in it would be the same class of lie ADR-037 removed from `alignment`.
//!
//! Not constructible from Python: a constructor would duplicate `hurray.Tensor`'s entire
//! parameter list to build the half of it that carries no data. Descriptors come from a
//! tensor, from a composite head, or from `decode`.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule, PyTuple};

use hurray_core::TensorDescriptor;

use crate::buffer_handle::BufferHandle;
use crate::device::Device;
use crate::dtype::Dtype;
use crate::errors::InvalidDescriptorError;

/// What a tensor declares about itself, without its data.
///
/// Obtained from `Tensor.descriptor`, `Composite.descriptor`, or
/// [`Descriptor::decode`] — never constructed directly.
///
/// ## Examples (Python)
///
/// ```python
/// import hurray
///
/// tensor = hurray.Tensor(bytes(48), hurray.float32, [3, 4])
/// descriptor = tensor.descriptor
///
/// assert descriptor.dtype is hurray.float32
/// assert descriptor.shape == (3, 4)
/// assert descriptor.layout == hurray.RowMajorLayout()
///
/// wire = descriptor.encode()
/// assert hurray.Descriptor.decode(wire) == descriptor
/// ```
#[pyclass(name = "Descriptor", frozen)]
pub struct Descriptor {
    pub(crate) inner: TensorDescriptor,
}

#[pymethods]
impl Descriptor {
    /// The element type of this tensor's data.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.Tensor(bytes(16), hurray.float32, [4]).descriptor.dtype is hurray.float32
    /// ```
    #[getter]
    pub fn dtype(&self, py: Python<'_>) -> PyResult<Py<Dtype>> {
        // The singleton, so `descriptor.dtype is hurray.float32` holds — the identity
        // the class documents, and the one `Tensor.dtype` already gives.
        crate::dtype::singleton(py, self.inner.element_type)
    }

    /// The shape, with `None` for a dynamic dimension.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.Tensor(bytes(16), hurray.float32, [4]).descriptor.shape == (4,)
    /// ```
    #[getter]
    pub fn shape(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        crate::tensor::shape_tuple(py, &self.inner.shape)
    }

    /// The number of dimensions.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.Tensor(bytes(16), hurray.float32, [2, 2]).descriptor.ndim == 2
    /// ```
    #[getter]
    pub fn ndim(&self) -> usize {
        self.inner.shape.rank()
    }

    /// The total element count, or `None` when a dimension is dynamic.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.Tensor(bytes(16), hurray.float32, [2, 2]).descriptor.size == 4
    /// assert hurray.Tensor(b"", hurray.float32, [None, 2]).descriptor.size is None
    /// ```
    #[getter]
    pub fn size(&self) -> Option<u64> {
        self.inner.shape.element_count()
    }

    /// The memory layout.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// descriptor = hurray.Tensor(bytes(16), hurray.float32, [4]).descriptor
    /// assert descriptor.layout == hurray.RowMajorLayout()
    /// ```
    #[getter]
    pub fn layout(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::layout::layout_to_py(py, &self.inner.layout)
    }

    /// One handle per buffer, in descriptor order.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// handles = hurray.Tensor(bytes(16), hurray.float32, [4]).descriptor.buffer_handles
    /// assert handles[0].byte_size == 16
    /// ```
    #[getter]
    pub fn buffer_handles(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let device_py = Py::new(py, self.device_object())?;
        let items: Vec<Py<BufferHandle>> = self
            .inner
            .buffers
            .iter()
            .map(|handle| {
                Py::new(
                    py,
                    BufferHandle {
                        inner: *handle,
                        device_py: device_py.clone_ref(py),
                    },
                )
            })
            .collect::<PyResult<_>>()?;
        Ok(PyTuple::new(py, items)?.unbind())
    }

    /// How many buffers this descriptor declares. `0` for a composite head.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.Tensor(bytes(16), hurray.float32, [4]).descriptor.buffer_count == 1
    /// ```
    #[getter]
    pub fn buffer_count(&self) -> usize {
        self.inner.buffers.len()
    }

    /// The device every buffer lives on (`buffer-protocol.md` § Device Colocation).
    ///
    /// A composite head declares no buffers and so no device; it reports CPU, which is
    /// what the wire carries for a descriptor with an empty buffer table.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.Tensor(bytes(16), hurray.float32, [4]).descriptor.device.kind == "cpu"
    /// ```
    #[getter]
    pub fn device(&self, py: Python<'_>) -> PyResult<Py<Device>> {
        Py::new(py, self.device_object())
    }

    /// The quantization scheme, or `None`.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.Tensor(bytes(16), hurray.float32, [4]).descriptor.quantization is None
    /// ```
    #[getter]
    pub fn quantization(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let Some(bytes) = self.inner.quantization.as_ref() else {
            return Ok(None);
        };
        let (scheme, _read) = hurray_core::QuantizationDescriptor::decode(bytes).map_err(|e| {
            InvalidDescriptorError::new_err(format!("failed to decode quantization: {e}"))
        })?;
        crate::quantization::quantization_to_py(py, scheme).map(Some)
    }

    /// The shard annotation, or `None`.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.Tensor(bytes(16), hurray.float32, [4]).descriptor.shard is None
    /// ```
    #[getter]
    pub fn shard(&self, py: Python<'_>) -> PyResult<Option<Py<crate::metadata::Shard>>> {
        match &self.inner.shard {
            Some(shard) => Py::new(
                py,
                crate::metadata::Shard {
                    inner: shard.clone(),
                },
            )
            .map(Some),
            None => Ok(None),
        }
    }

    /// The advisory statistics section, or `None`.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.Tensor(bytes(16), hurray.float32, [4]).descriptor.statistics is None
    /// ```
    #[getter]
    pub fn statistics(&self, py: Python<'_>) -> PyResult<Option<Py<crate::metadata::Statistics>>> {
        match &self.inner.statistics {
            Some(stats) => Py::new(
                py,
                crate::metadata::Statistics {
                    inner: stats.clone(),
                },
            )
            .map(Some),
            None => Ok(None),
        }
    }

    /// Byte offset from the start of buffer 0 to logical element `[0, …, 0]`.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.Tensor(bytes(16), hurray.float32, [4]).descriptor.byte_offset == 0
    /// ```
    #[getter]
    pub fn byte_offset(&self) -> u64 {
        self.inner.byte_offset
    }

    /// The format version this descriptor declares, as `(major, minor)`.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// major, minor = hurray.Tensor(bytes(16), hurray.float32, [4]).descriptor.version
    /// assert major == 1
    /// ```
    #[getter]
    pub fn version(&self) -> (u8, u8) {
        (self.inner.version_major, self.inner.version_minor)
    }

    /// The number of bytes [`Descriptor::encode`] will produce.
    ///
    /// The same number the wire's own `descriptor_length` field carries, which is what
    /// makes a descriptor self-delimiting.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// descriptor = hurray.Tensor(bytes(48), hurray.float32, [3, 4]).descriptor
    /// assert descriptor.encoded_len == len(descriptor.encode())
    /// ```
    #[getter]
    pub fn encoded_len(&self) -> PyResult<usize> {
        self.inner
            .encode()
            .map(|bytes| bytes.len())
            .map_err(|e| InvalidDescriptorError::new_err(format!("cannot encode: {e}")))
    }

    /// Encode to the binary descriptor, exactly as it appears on the wire.
    ///
    /// Self-delimiting: bytes 6–9 hold the total length, so a reader consumes it without
    /// any external framing. Put the result wherever a container of your own has room,
    /// and hand the buffers over beside it.
    ///
    /// ## Errors
    ///
    /// - `hurray.InvalidDescriptorError` — the descriptor cannot be encoded.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// wire = hurray.Tensor(bytes(48), hurray.float32, [3, 4]).descriptor.encode()
    /// assert isinstance(wire, bytes)
    /// ```
    pub fn encode(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let bytes = self
            .inner
            .encode()
            .map_err(|e| InvalidDescriptorError::new_err(format!("cannot encode: {e}")))?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }

    /// Decode a binary descriptor.
    ///
    /// Trailing bytes are permitted and ignored — a descriptor is self-delimiting, so
    /// what follows it in a stream is the buffers it describes, not part of it.
    ///
    /// ## Errors
    ///
    /// - `hurray.InvalidDescriptorError` — the bytes are not a valid descriptor.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// original = hurray.Tensor(bytes(48), hurray.float32, [3, 4]).descriptor
    /// assert hurray.Descriptor.decode(original.encode()) == original
    /// ```
    #[classmethod]
    pub fn decode(
        _cls: &Bound<'_, pyo3::types::PyType>,
        py: Python<'_>,
        data: &[u8],
    ) -> PyResult<Py<Descriptor>> {
        let inner = TensorDescriptor::decode(data)
            .map_err(|e| InvalidDescriptorError::new_err(format!("cannot decode: {e}")))?;
        Py::new(py, Descriptor { inner })
    }

    /// Value equality: two descriptors are equal when they say the same thing.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// a = hurray.Tensor(bytes(16), hurray.float32, [4]).descriptor
    /// b = hurray.Tensor(bytes(16), hurray.float32, [4]).descriptor
    /// assert a == b
    /// ```
    pub fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        match other.extract::<PyRef<'_, Descriptor>>() {
            Ok(other) => self.inner == other.inner,
            Err(_) => false,
        }
    }

    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// text = repr(hurray.Tensor(bytes(16), hurray.float32, [4]).descriptor)
    /// assert text.startswith("hurray.Descriptor(")
    /// ```
    pub fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let shape = self.shape(py)?.bind(py).repr()?.to_str()?.to_owned();
        Ok(format!(
            "hurray.Descriptor(dtype={}, shape={}, layout='{}', buffers={})",
            crate::dtype::element_type_name(self.inner.element_type),
            shape,
            crate::layout::layout_name(&self.inner.layout),
            self.inner.buffers.len(),
        ))
    }
}

impl Descriptor {
    /// The device its buffers agree on, or CPU when it declares none.
    ///
    /// Colocation makes the first buffer's device the descriptor's device; an empty
    /// buffer table belongs to a composite head, which owns no memory at all.
    fn device_object(&self) -> Device {
        match self.inner.buffers.first() {
            Some(handle) => Device {
                tag: handle.device_tag(),
                memory_class: handle.memory_class(),
                device_id: 0,
            },
            None => Device {
                tag: hurray_core::DeviceTag::Cpu,
                memory_class: hurray_core::MemoryClass::Standard,
                device_id: 0,
            },
        }
    }
}

// ── Registration ──────────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Descriptor>()?;
    Ok(())
}
