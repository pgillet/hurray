//! # hurray-python
//!
//! Python bindings for the Hurray tensor interchange format.
//!
//! Provides zero-copy interop with NumPy and PyTorch via the `__dlpack__` protocol,
//! and implements the [Python Array API Standard](https://data-apis.org/array-api/)
//! for Tier 1 element types. Built with [PyO3](https://pyo3.rs).
//!
//! ## Runtime modes
//!
//! This package defaults to **strict mode**, which enforces full Array API compliance.
//! Relaxed mode (allowing Tier 2 / quantized types through the Array API surface)
//! is reserved for a future release. See ADR-022.

use pyo3::prelude::*;

pub(crate) mod buffer;
mod device;
pub(crate) mod dlpack;
mod dtype;
pub mod errors;
mod interop;
mod modes;
mod scipy_interop;
mod sparse;
mod tensor;

/// Python module entry point.
///
/// Registers all public API items: version string, runtime mode functions,
/// context managers, exception classes, dtype/device submodules, and the
/// `Tensor` class.
///
/// ## Module layout
///
/// | Name | Kind | Phase |
/// |------|------|-------|
/// | `hurray.__version__` | string | 8a.1 |
/// | `hurray.set_strict` / `hurray.is_strict` | functions | 8a.1 |
/// | `hurray.StrictCtx` / `hurray.RelaxedCtx` | context managers | 8a.1 |
/// | `hurray.{Invalid,Buffer,Unsupported,Internal}Error` | exceptions | 8a.1 |
/// | `hurray.Dtype` | class | 8a.2 |
/// | `hurray.<tier1_type>` (e.g. `hurray.float32`) | `Dtype` constants | 8a.2 |
/// | `hurray.dtype` | submodule | 8a.2 |
/// | `hurray.Device` | class | 8a.2 |
/// | `hurray.device` | submodule | 8a.2 |
/// | `hurray.Tensor` | class | 8a.2 |
/// | `hurray.SparseTensor` | class | 8a.4 |
/// | `hurray.from_scipy` | function | 8a.4 |
#[pymodule]
fn hurray(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(modes::set_strict, m)?)?;
    m.add_function(wrap_pyfunction!(modes::is_strict, m)?)?;
    m.add_class::<modes::StrictCtx>()?;
    m.add_class::<modes::RelaxedCtx>()?;
    errors::register(m)?;
    dtype::register(m)?;
    device::register(m)?;
    tensor::register(m)?;
    interop::register(m)?;
    sparse::register(m)?;
    scipy_interop::register(m)?;
    Ok(())
}
