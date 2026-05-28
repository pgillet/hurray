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

pub mod errors;
mod modes;

/// Python module entry point.
///
/// Registers all public API items: version string, runtime mode functions,
/// context managers, and exception classes.
///
/// Registers the `hurray` Python module with all public API items.
#[pymodule]
fn hurray(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(modes::set_strict, m)?)?;
    m.add_function(wrap_pyfunction!(modes::is_strict, m)?)?;
    m.add_class::<modes::StrictCtx>()?;
    m.add_class::<modes::RelaxedCtx>()?;
    errors::register(m)?;
    Ok(())
}
