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
mod creation;
mod device;
pub(crate) mod dlpack;
mod dtype;
pub mod errors;
mod file_io;
mod interop;
mod metadata;
mod native_buffer;
mod print_options;
pub(crate) mod quantization;
mod scipy_interop;
mod sparse;
mod tensor;

/// Python module entry point.
///
/// Registers all public API items: version string, exception classes,
/// dtype/device submodules, and the `Tensor` class.
///
/// ## Module layout
///
/// | Name | Kind | Phase |
/// |------|------|-------|
/// | `hurray.__version__` | string | 8a.1 |
/// | `hurray.{Invalid,Buffer,Unsupported,Internal}Error` | exceptions | 8a.1 |
/// | `hurray.Dtype` | class | 8a.2 |
/// | `hurray.<tier1_type>` (e.g. `hurray.float32`) | `Dtype` constants | 8a.2 |
/// | `hurray.dtype` | submodule | 8a.2 |
/// | `hurray.Device` | class | 8a.2 |
/// | `hurray.device` | submodule | 8a.2 |
/// | `hurray.Tensor` | class | 8a.2 |
/// | `hurray.SparseTensor` | class | 8a.4 |
/// | `hurray.from_scipy` | function | 8a.4 |
/// | `hurray.sparse_coo` | function | 8a.4 |
/// | `hurray.from_hurray_buffer` | function | 8c |
/// | `hurray.Tensor.__hurray_buffer__` | method | 8c |
/// | `hurray.zeros` / `hurray.ones` / `hurray.full` / `hurray.empty` | functions | 8a.5 |
/// | `hurray.zeros_like` / `hurray.ones_like` / `hurray.full_like` / `hurray.empty_like` | functions | 8a.5 |
/// | `hurray.arange` / `hurray.linspace` / `hurray.eye` | functions | 8a.5 |
/// | `hurray.asarray` / `hurray.from_dlpack` | functions | 8a.5 |
/// | `hurray.load` / `hurray.save` | functions | 8b |
/// | `hurray.FileError` / `hurray.StreamError` | exceptions | 8b |
/// | `hurray.set_print_options` / `hurray.get_print_options` | functions | 8e |
/// | `hurray.print_options` | context-manager factory | 8e |
/// | `hurray.PrintOptionsCtx` | context manager | 8e |
#[pymodule]
fn hurray(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    errors::register(m)?;
    dtype::register(m)?;
    device::register(m)?;
    tensor::register(m)?;
    interop::register(m)?;
    sparse::register(m)?;
    scipy_interop::register(m)?;
    native_buffer::register(m)?;
    creation::register(m)?;
    file_io::register(m)?;
    print_options::register(m)?;
    quantization::register(m)?;
    metadata::register(m)?;
    Ok(())
}
