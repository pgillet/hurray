//! # hurray-python
//!
//! Python bindings for the hurray tensor interchange format.
//!
//! Provides zero-copy interop with NumPy and PyTorch via the `__dlpack__` protocol.
//! Built with [PyO3](https://pyo3.rs).

use pyo3::prelude::*;

/// Python module entry point.
#[pymodule]
fn hurray(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = m;
    Ok(())
}
