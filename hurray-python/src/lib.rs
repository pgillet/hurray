//! # hurray-python
//!
//! Python bindings for the Hurray tensor interchange format.
//!
//! `hurray-python` is the Python codec and zero-copy bridge for the Hurray format —
//! it produces and consumes Hurray tensors and hands their buffers to the array
//! ecosystem without copying, via DLPack, the NumPy array protocols, and the native
//! Hurray buffer protocol. It is not an Array API implementation (see ADR-029). Built
//! with [PyO3](https://pyo3.rs).

use pyo3::prelude::*;

pub(crate) mod buffer;
mod buffer_handle;
mod composite;
mod creation;
mod descriptor;
mod device;
pub(crate) mod dlpack;
mod dtype;
pub mod errors;
mod file_io;
mod interop;
pub(crate) mod layout;
mod metadata;
mod native_protocol;
mod numpy_allocator;
mod print_options;
pub(crate) mod quantization;
mod scipy_interop;
mod sparse;
mod stream;
mod tensor;

/// Python bindings for the Hurray tensor interchange format.
///
/// `hurray` is a codec and a zero-copy bridge: it produces and consumes Hurray tensors and
/// hands their buffers to the array ecosystem without copying. It does no arithmetic — the
/// math belongs to whichever framework the buffer is handed to.
///
/// ```python
/// import hurray, numpy as np
///
/// # Wraps the array's buffer — no copy in, no copy out.
/// t = hurray.asarray(np.arange(12, dtype=np.float32).reshape(3, 4))
/// hurray.save("weights.hrry", {"w": t})
///
/// w = hurray.load("weights.hrry")["w"]
/// assert w.shape == (3, 4) and w.dtype == hurray.float32
/// np.asarray(w)[0]        # array([0., 1., 2., 3.], dtype=float32)
/// ```
///
/// ## The API, by what it is for
///
/// | Group | Names |
/// |-------|-------|
/// | The tensor | `Tensor`, `Composite`, `Descriptor` |
/// | Element types | `Dtype`, the type constants (`float32`, `int4`, `bfloat16`, …), the `dtype` submodule |
/// | Devices | `Device`, the device constants (`cpu`, `cuda`, …), the `device` submodule |
/// | Construction | `zeros`, `ones`, `full`, `empty` and their `_like` forms, `arange`, `linspace`, `eye` |
/// | Interop | `asarray`, `from_dlpack`, `from_numpy`, `from_torch`, `from_scipy`, `from_hurray`, `sparse_coo` |
/// | Layouts | `Layout` and one subclass per layout (`RowMajorLayout`, `CsrLayout`, `BlockPagedLayout`, …) |
/// | Quantization | `PerTensorAffine`, `PerChannelAffine`, `PerBlockAffine`, `NF4`, `MXFP`, `decode_quantization` |
/// | Buffers | `BufferHandle`, `aligned_allocator`, `MIN_BUFFER_ALIGNMENT`, `PAGE_ALIGNMENT` |
/// | Files | `save`, `load` |
/// | Streaming | `StreamWriter`, `StreamReader` |
/// | Display | `set_print_options`, `get_print_options`, `print_options` |
/// | Errors | `InvalidDescriptorError`, `BufferError`, `CopyRequiredError`, `UnsupportedError`, `FileError`, `StreamError`, `InternalError` |
///
/// ## Interchange protocols
///
/// A `Tensor` is a producer for `__dlpack__`, the NumPy array protocols, and Hurray's own
/// `__hurray__` protocol, and a consumer of all three. `__hurray__` is the only one that
/// carries the full descriptor — layout, quantization, multiple buffers — across a process
/// boundary; DLPack and NumPy carry a single strided buffer, which is all they model.
#[pymodule]
fn hurray(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    errors::register(m)?;
    dtype::register(m)?;
    device::register(m)?;
    tensor::register(m)?;
    interop::register(m)?;
    layout::register(m)?;
    sparse::register(m)?;
    scipy_interop::register(m)?;
    stream::register(m)?;
    native_protocol::register(m)?;
    numpy_allocator::register(m)?;
    buffer_handle::register(m)?;
    composite::register(m)?;
    creation::register(m)?;
    descriptor::register(m)?;
    file_io::register(m)?;
    print_options::register(m)?;
    quantization::register(m)?;
    metadata::register(m)?;
    Ok(())
}
