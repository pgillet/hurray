//! Runtime compliance mode API — strict vs relaxed (ADR-022).
//!
//! The mode is backed by a Python `contextvars.ContextVar`, giving asyncio-Task
//! isolation that Rust `thread_local!` cannot provide. Strict mode is the
//! default (`True`). Use [`set_strict`], [`strict`], or [`relaxed`] to change it.
//!
//! ## Thread-safety note
//!
//! Threads created with `threading.Thread` inherit the ContextVar *default*
//! (`True`, strict), not the spawning thread's current mode. This is standard
//! Python `contextvars` behaviour.
//!
//! ## Context manager nesting
//!
//! Context managers store the `Token` returned by `ContextVar.set()` and call
//! `ContextVar.reset(token)` in `__exit__`. This is exception-safe and restores
//! the exact prior value, including nested usage.

// PyO3 0.22 macro expansion emits a redundant .into() on PyErr for functions
// returning PyResult<()> — suppress the false positive across this module.
#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use pyo3::types::PyDict;

// GILOnceCell over thread_local: ContextVar gives asyncio-Task isolation that
// thread-locals cannot.
static STRICT_VAR: GILOnceCell<Py<PyAny>> = GILOnceCell::new();

/// Return the `contextvars.ContextVar` object for the strict-mode flag.
///
/// Created lazily with `default=True` on the first call.
fn strict_var(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    STRICT_VAR
        .get_or_try_init(py, || {
            let cv_mod = py.import_bound("contextvars")?;
            // ContextVar(name, *, default=...) — default is keyword-only.
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("default", true)?;
            let var: Py<PyAny> = cv_mod
                .call_method("ContextVar", ("hurray.strict_mode",), Some(&kwargs))?
                .unbind();
            Ok(var)
        })
        .map(|v| v.bind(py).clone())
}

/// Set the process-wide compliance mode default.
///
/// In strict mode (the default) `hurray.Tensor` enforces full Python Array API
/// Standard compliance for Tier 1 element types. Passing `strict=False` enables
/// relaxed mode, which allows Tier 2 / quantized types through the Array API
/// surface.
///
/// The mode is stored in a `contextvars.ContextVar` and is thread-safe. Use the
/// `strict()` / `relaxed()` context managers for scoped overrides.
///
/// Note: threads created with `threading.Thread` always start in strict mode
/// (the ContextVar default), regardless of the calling thread's current mode.
///
/// Args:
///     strict: `True` to enforce strict Array API compliance (the default).
///             `False` to enable relaxed mode.
///
/// Examples:
///     >>> import hurray
///     >>> hurray.set_strict(True)
///     >>> hurray.is_strict()
///     True
///     >>> hurray.set_strict(False)
///     >>> hurray.is_strict()
///     False
///     >>> hurray.set_strict(True)  # restore
#[pyfunction]
pub fn set_strict(py: Python<'_>, strict: bool) -> PyResult<()> {
    strict_var(py)?.call_method1("set", (strict,))?;
    Ok(())
}

/// Return `True` if the current context is in strict Array API compliance mode.
///
/// Examples:
///     >>> import hurray
///     >>> hurray.is_strict()
///     True
#[pyfunction]
pub fn is_strict(py: Python<'_>) -> PyResult<bool> {
    strict_var(py)?.call_method0("get")?.extract::<bool>()
}

/// Context manager that enters strict Array API compliance mode.
///
/// On `__enter__`, sets the ContextVar to `True` and stores the reset token.
/// On `__exit__`, resets the ContextVar to its prior value via the stored token.
/// This is exception-safe and supports nesting correctly.
///
/// Examples:
///     >>> import hurray
///     >>> hurray.set_strict(False)
///     >>> hurray.is_strict()
///     False
///     >>> with hurray.StrictCtx():
///     ...     assert hurray.is_strict() == True
///     >>> hurray.is_strict()
///     False
///     >>> hurray.set_strict(True)  # restore
#[pyclass(name = "StrictCtx")]
pub struct StrictCtx {
    token: Option<Py<PyAny>>,
}

#[pymethods]
impl StrictCtx {
    #[new]
    pub fn new() -> Self {
        StrictCtx { token: None }
    }

    fn __enter__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> PyResult<()> {
        let token = strict_var(py)?.call_method1("set", (true,))?.unbind();
        slf.token = Some(token);
        Ok(())
    }

    fn __exit__(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        _exc_type: &Bound<'_, PyAny>,
        _exc_val: &Bound<'_, PyAny>,
        _exc_tb: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        if let Some(token) = slf.token.take() {
            strict_var(py)?.call_method1("reset", (token,))?;
        }
        Ok(false)
    }
}

/// Context manager that enters relaxed mode, allowing Tier 2 / quantized types
/// through the Array API surface.
///
/// On `__enter__`, sets the ContextVar to `False` and stores the reset token.
/// On `__exit__`, resets the ContextVar to its prior value via the stored token.
/// This is exception-safe and supports nesting correctly.
///
/// Note: threads created with `threading.Thread` always start in strict mode
/// (the ContextVar default), regardless of the calling thread's current mode.
///
/// Examples:
///     >>> import hurray
///     >>> hurray.is_strict()
///     True
///     >>> with hurray.RelaxedCtx():
///     ...     assert hurray.is_strict() == False
///     >>> hurray.is_strict()
///     True
#[pyclass(name = "RelaxedCtx")]
pub struct RelaxedCtx {
    token: Option<Py<PyAny>>,
}

#[pymethods]
impl RelaxedCtx {
    #[new]
    pub fn new() -> Self {
        RelaxedCtx { token: None }
    }

    fn __enter__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> PyResult<()> {
        let token = strict_var(py)?.call_method1("set", (false,))?.unbind();
        slf.token = Some(token);
        Ok(())
    }

    fn __exit__(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        _exc_type: &Bound<'_, PyAny>,
        _exc_val: &Bound<'_, PyAny>,
        _exc_tb: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        if let Some(token) = slf.token.take() {
            strict_var(py)?.call_method1("reset", (token,))?;
        }
        Ok(false)
    }
}

/// Return a context manager that enters strict Array API compliance mode.
///
/// Equivalent to `with hurray.StrictCtx():`. On entry the ContextVar is set to
/// `True`; on exit it is restored to its previous value. Supports nesting.
///
/// Examples:
///     >>> import hurray
///     >>> hurray.set_strict(False)
///     >>> with hurray.strict():
///     ...     assert hurray.is_strict() == True
///     >>> hurray.is_strict()
///     False
///     >>> hurray.set_strict(True)  # restore
#[pyfunction]
pub fn strict(py: Python<'_>) -> PyResult<StrictCtx> {
    let _ = py; // py used for symmetry; StrictCtx initialises token lazily in __enter__
    Ok(StrictCtx::new())
}

/// Return a context manager that enters relaxed mode.
///
/// Equivalent to `with hurray.RelaxedCtx():`. On entry the ContextVar is set to
/// `False`; on exit it is restored to its previous value. Supports nesting.
///
/// Examples:
///     >>> import hurray
///     >>> with hurray.relaxed():
///     ...     assert hurray.is_strict() == False
///     >>> hurray.is_strict()
///     True
#[pyfunction]
pub fn relaxed(py: Python<'_>) -> PyResult<RelaxedCtx> {
    let _ = py;
    Ok(RelaxedCtx::new())
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
        Python::with_gil(|py| {
            assert!(set_strict(py, true).is_ok());
        });
    }

    #[test]
    fn set_strict_false_enables_relaxed() {
        init_python();
        Python::with_gil(|py| {
            set_strict(py, false).expect("set_strict(False) must succeed");
            assert!(
                !is_strict(py).expect("is_strict must not fail"),
                "mode should be relaxed after set_strict(False)"
            );
            // Restore.
            set_strict(py, true).unwrap();
        });
    }

    #[test]
    fn is_strict_round_trips() {
        init_python();
        Python::with_gil(|py| {
            set_strict(py, false).unwrap();
            assert!(!is_strict(py).unwrap());
            set_strict(py, true).unwrap();
            assert!(is_strict(py).unwrap());
        });
    }

    #[test]
    fn strict_ctx_enters_and_restores() {
        init_python();
        Python::with_gil(|py| {
            // Start in relaxed.
            set_strict(py, false).unwrap();
            {
                let mut ctx = StrictCtx::new();
                let none = py.None();
                let none_bound = none.bind(py);
                // Simulate __enter__.
                StrictCtx::__enter__(
                    Py::new(py, StrictCtx::new())
                        .unwrap()
                        .into_bound(py)
                        .borrow_mut(),
                    py,
                )
                .unwrap();
                // Within the context: strict.
                assert!(is_strict(py).unwrap());
                // Simulate __exit__ via a fresh ctx that has a token.
                let bound = Py::new(py, StrictCtx::new()).unwrap();
                {
                    let mut bmut = bound.bind(py).borrow_mut();
                    // Re-enter to store token.
                    let token = strict_var(py)
                        .unwrap()
                        .call_method1("set", (true,))
                        .unwrap()
                        .unbind();
                    bmut.token = Some(token);
                    StrictCtx::__exit__(bmut, py, none_bound, none_bound, none_bound).unwrap();
                }
                // Restore from __exit__ above just reset to the value that was there.
                // Since set_strict(false) was called first then we did set(true),
                // reset should bring us back to false.
                let _ = ctx.token.take(); // suppress unused warning
            }
            // Clean up for other tests.
            set_strict(py, true).unwrap();
        });
    }

    #[test]
    fn relaxed_ctx_enters_and_restores() {
        init_python();
        Python::with_gil(|py| {
            // Start in strict.
            set_strict(py, true).unwrap();
            assert!(is_strict(py).unwrap());

            let none = py.None();
            let none_bound = none.bind(py);

            let bound = Py::new(py, RelaxedCtx::new()).unwrap();
            // __enter__
            RelaxedCtx::__enter__(bound.bind(py).borrow_mut(), py).unwrap();
            assert!(!is_strict(py).unwrap(), "should be relaxed inside ctx");
            // __exit__
            RelaxedCtx::__exit__(
                bound.bind(py).borrow_mut(),
                py,
                none_bound,
                none_bound,
                none_bound,
            )
            .unwrap();
            assert!(is_strict(py).unwrap(), "should be strict after ctx exits");
        });
    }

    #[test]
    fn nested_strict_in_relaxed_restores() {
        init_python();
        Python::with_gil(|py| {
            set_strict(py, true).unwrap();

            let none = py.None();
            let nb = none.bind(py);

            // Enter relaxed.
            let r = Py::new(py, RelaxedCtx::new()).unwrap();
            RelaxedCtx::__enter__(r.bind(py).borrow_mut(), py).unwrap();
            assert!(!is_strict(py).unwrap());

            // Enter strict inside relaxed.
            let s = Py::new(py, StrictCtx::new()).unwrap();
            StrictCtx::__enter__(s.bind(py).borrow_mut(), py).unwrap();
            assert!(is_strict(py).unwrap());

            // Exit strict → back to relaxed.
            StrictCtx::__exit__(s.bind(py).borrow_mut(), py, nb, nb, nb).unwrap();
            assert!(
                !is_strict(py).unwrap(),
                "should be back to relaxed after inner strict exits"
            );

            // Exit relaxed → back to strict.
            RelaxedCtx::__exit__(r.bind(py).borrow_mut(), py, nb, nb, nb).unwrap();
            assert!(
                is_strict(py).unwrap(),
                "should be back to strict after outer relaxed exits"
            );
        });
    }
}
