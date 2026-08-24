//! The buffer table's Python representation (ADR-037).
//!
//! ```python
//! for handle in tensor.buffer_handles:
//!     print(handle.byte_size, handle.alignment, handle.sync_mode)
//! ```
//!
//! ## A value object, not a view
//!
//! A `BufferHandle` is five scalars copied out of a buffer-table row. It holds **no
//! reference to its tensor and none to any buffer**, so collecting handles across a
//! stream pins nothing — the same discipline ADR-032 imposed on layout objects, and for
//! the same reason: in a zero-copy format, a metadata accessor that extends buffer
//! lifetime is a defect.
//!
//! That is why metadata and data have separate accessors. `t.buffer(i)` hands back a
//! view over bytes; `t.buffer_handles[i]` answers questions about those bytes without
//! touching them. On a CUDA tensor the second must work where the first cannot.
//!
//! ## Read-only, for two different reasons
//!
//! `alignment` is a fact the binding measures. `sync_mode` is a promise it cannot make:
//! `"event"` means a device event exists for a consumer to wait on, and no Python API
//! supplies one, so a settable field could only author a contract nothing could honour.

use pyo3::prelude::*;
use pyo3::types::PyModule;

use hurray_core::BufferHandle as CoreHandle;

use crate::device::Device;

/// One row of a tensor's buffer table: what a buffer declares about itself.
///
/// Not constructible from Python — there is no field a caller could supply that the
/// buffers do not already settle (ADR-037 § 4).
///
/// ## Examples (Python)
///
/// ```python
/// import hurray
///
/// t = hurray.Tensor(bytes(16), hurray.float32, [4])
/// handle = t.buffer_handles[0]
///
/// assert handle.byte_size == 16
/// assert handle.alignment >= hurray.MIN_BUFFER_ALIGNMENT
/// assert handle.sync_mode == "producer_synced"
/// assert handle.device is t.device        # colocation: one device per descriptor
/// ```
#[pyclass(name = "BufferHandle", frozen)]
pub struct BufferHandle {
    pub(crate) inner: CoreHandle,
    /// The tensor's own `Device` object, so `handle.device is t.device` holds.
    pub(crate) device_py: Py<Device>,
}

#[pymethods]
impl BufferHandle {
    /// The buffer's declared size in bytes.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// t = hurray.Tensor(bytes(16), hurray.float32, [4])
    /// assert t.buffer_handles[0].byte_size == 16
    /// ```
    #[getter]
    pub fn byte_size(&self) -> u64 {
        self.inner.byte_size()
    }

    /// The alignment of the buffer's base address, in bytes.
    ///
    /// Measured, never assumed: a buffer borrowed from NumPy declares what its address
    /// actually satisfies, and an owned buffer is allocated over-aligned so that the
    /// declaration is true. Always a power of two, and at least
    /// `hurray.MIN_BUFFER_ALIGNMENT` for a non-empty buffer.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// t = hurray.Tensor(bytes(4096), hurray.float32, [1024])
    /// assert t.buffer_handles[0].alignment >= 64
    /// ```
    #[getter]
    pub fn alignment(&self) -> u32 {
        self.inner.alignment()
    }

    /// When the buffer may be read: `"producer_synced"`, `"event"`, or
    /// `"consumer_stream"`.
    ///
    /// Anything this binding constructs is `"producer_synced"` — the interpreter cannot
    /// enqueue device work through this API, so it cannot promise anything else. A
    /// tensor decoded from a stream or a file reports what the producer declared, and a
    /// buffer that is not `"producer_synced"` will refuse the paths that hand out its
    /// bytes until the binding can honour the wait.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// t = hurray.Tensor(bytes(16), hurray.float32, [4])
    /// assert t.buffer_handles[0].sync_mode == "producer_synced"
    /// ```
    #[getter]
    pub fn sync_mode(&self) -> String {
        // core's Display is the single source of these strings, shared with
        // hurray-inspect, so the two cannot drift.
        self.inner.sync_mode().to_string()
    }

    /// The device this buffer lives on.
    ///
    /// Returns the tensor's own `Device` object: `buffer-protocol.md` § Device
    /// Colocation requires every buffer of one descriptor to share a device and memory
    /// class, so there is exactly one to report and no difference to branch on.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// t = hurray.Tensor(bytes(16), hurray.float32, [4])
    /// assert t.buffer_handles[0].device is t.device
    /// ```
    #[getter]
    pub fn device(&self, py: Python<'_>) -> Py<Device> {
        self.device_py.clone_ref(py)
    }

    /// Whether this buffer declares zero bytes.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert not hurray.Tensor(bytes(16), hurray.float32, [4]).buffer_handles[0].is_empty
    /// ```
    #[getter]
    pub fn is_empty(&self) -> bool {
        self.inner.byte_size() == 0
    }

    /// Value equality: two handles are equal when they declare the same thing.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// a = hurray.Tensor(bytes(16), hurray.float32, [4])
    /// b = hurray.Tensor(bytes(16), hurray.float32, [4])
    /// assert a.buffer_handles[0] == b.buffer_handles[0]
    /// ```
    pub fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        match other.extract::<PyRef<'_, BufferHandle>>() {
            Ok(o) => self.inner == o.inner,
            Err(_) => false,
        }
    }

    /// Hash of the declaration, consistent with `__eq__`.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// t = hurray.Tensor(bytes(16), hurray.float32, [4])
    /// assert len({t.buffer_handles[0], t.buffer_handles[0]}) == 1
    /// ```
    pub fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }

    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// t = hurray.Tensor(bytes(16), hurray.float32, [4])
    /// assert repr(t.buffer_handles[0]).startswith("BufferHandle(byte_size=16")
    /// ```
    pub fn __repr__(&self, py: Python<'_>) -> String {
        let device = crate::device::device_repr(&self.device_py.borrow(py));
        format!(
            "BufferHandle(byte_size={}, alignment={}, sync_mode='{}', device={})",
            self.inner.byte_size(),
            self.inner.alignment(),
            self.inner.sync_mode(),
            device,
        )
    }
}

// ── Registration ──────────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BufferHandle>()?;
    // The alignment floors the format defines, mirrored from hurray-core so a caller
    // can compare against them without hardcoding a literal.
    m.add("MIN_BUFFER_ALIGNMENT", hurray_core::MIN_BUFFER_ALIGNMENT)?;
    m.add("PAGE_ALIGNMENT", hurray_core::PAGE_ALIGNMENT)?;
    Ok(())
}
