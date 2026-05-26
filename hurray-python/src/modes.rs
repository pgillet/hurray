//! Runtime compliance mode API — strict vs relaxed (ADR-022).
//!
//! Layer 8a implements strict mode only. Relaxed mode is reserved; calling
//! `set_strict(False)` or entering a `relaxed()` context raises
//! `NotImplementedError` until a future release activates it.

// PyO3 0.22 macro expansion emits a redundant .into() on PyErr for functions
// returning PyResult<()> — suppress the false positive across this module.
#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;

/// Set the process-wide compliance mode default.
///
/// In strict mode (the default) `hurray.Tensor` enforces full Python Array API
/// Standard compliance for Tier 1 element types. Passing `strict=False` enables
/// relaxed mode, which allows Tier 2 / quantized types through the Array API
/// surface. Relaxed mode is reserved for a future release.
///
/// The mode is stored in a `contextvars.ContextVar` and is thread-safe. Use the
/// `strict()` / `relaxed()` context managers for scoped overrides.
///
/// Args:
///     strict: `True` to enforce strict Array API compliance (the default).
///             `False` is reserved and raises `NotImplementedError`.
///
/// Raises:
///     NotImplementedError: if `strict=False` (relaxed mode not yet implemented).
///
/// Examples:
///     >>> import hurray
///     >>> hurray.set_strict(True)   # no-op, already in strict mode
///     >>> hurray.is_strict()
///     True
#[pyfunction]
pub fn set_strict(strict: bool) -> PyResult<()> {
    if !strict {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "hurray relaxed mode is reserved for a future release; \
             only strict mode (the default) is available in this version.",
        ));
    }
    Ok(())
}

/// Return `True` if the current context is in strict Array API compliance mode.
///
/// Always returns `True` in this version (relaxed mode is not yet implemented).
///
/// Examples:
///     >>> import hurray
///     >>> hurray.is_strict()
///     True
#[pyfunction]
pub fn is_strict() -> bool {
    true
}

/// Context manager that asserts strict Array API compliance mode.
///
/// This is a no-op in the current version (strict mode is always active), but
/// reserves the `hurray.strict()` name for future use.
///
/// Examples:
///     >>> import hurray
///     >>> with hurray.StrictCtx():
///     ...     pass  # strict mode active (the default)
#[pyclass(name = "StrictCtx")]
pub struct StrictCtx;

#[pymethods]
impl StrictCtx {
    #[new]
    pub fn new() -> Self {
        StrictCtx
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &self,
        _exc_type: &Bound<'_, PyAny>,
        _exc_val: &Bound<'_, PyAny>,
        _exc_tb: &Bound<'_, PyAny>,
    ) -> bool {
        false
    }
}

/// Context manager that enters relaxed mode, allowing Tier 2 / quantized types
/// through the Array API surface.
///
/// Relaxed mode is reserved for a future release. Entering this context manager
/// raises `NotImplementedError` until then.
///
/// Examples:
///     >>> import hurray
///     >>> with hurray.RelaxedCtx():  # doctest: +IGNORE_EXCEPTION_DETAIL
///     ...     pass
///     Traceback (most recent call last):
///         ...
///     NotImplementedError: ...
#[pyclass(name = "RelaxedCtx")]
pub struct RelaxedCtx;

#[pymethods]
impl RelaxedCtx {
    #[new]
    pub fn new() -> Self {
        RelaxedCtx
    }

    fn __enter__(&self) -> PyResult<()> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "hurray relaxed mode is reserved for a future release; \
             only strict mode (the default) is available in this version.",
        ))
    }

    fn __exit__(
        &self,
        _exc_type: &Bound<'_, PyAny>,
        _exc_val: &Bound<'_, PyAny>,
        _exc_tb: &Bound<'_, PyAny>,
    ) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;

    fn init_python() {
        // Required before any Python::with_gil call in test binaries (no auto-initialize).
        pyo3::prepare_freethreaded_python();
    }

    #[test]
    fn set_strict_true_is_noop() {
        init_python();
        Python::with_gil(|_py| {
            assert!(set_strict(true).is_ok());
        });
    }

    #[test]
    fn set_strict_false_raises() {
        init_python();
        Python::with_gil(|py| {
            let err = set_strict(false).unwrap_err();
            assert!(err.is_instance_of::<pyo3::exceptions::PyNotImplementedError>(py));
        });
    }

    #[test]
    fn is_strict_always_true() {
        assert!(is_strict());
    }
}
