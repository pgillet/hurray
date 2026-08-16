//! Configurable display options for Hurray Python objects.
//!
//! The active options are backed by a Python `contextvars.ContextVar`, giving
//! asyncio-Task isolation that Rust `thread_local!` cannot provide. The default
//! for every option is the same as the pre-existing behaviour so that existing
//! code is never affected.
//!
//! ## Current options
//!
//! | Key | Values | Default | Effect |
//! |-----|--------|---------|--------|
//! | `sparse_display` | `"metadata"` \| `"content"` | `"metadata"` | Controls `Tensor.__repr__` for sparse layouts |
//!
//! ## Thread-safety note
//!
//! Threads created with `threading.Thread` inherit the ContextVar *default*
//! (`"metadata"`), not the spawning thread's current setting. This is standard
//! Python `contextvars` behaviour.
//!
//! ## Context manager nesting
//!
//! `PrintOptionsCtx` stores the `Token` returned by `ContextVar.set()` and calls
//! `ContextVar.reset(token)` in `__exit__`. This is exception-safe and restores
//! the exact prior value, including nested usage.

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyDict;

// PyOnceLock over thread_local: ContextVar gives asyncio-Task isolation that
// thread-locals cannot.
static SPARSE_DISPLAY_VAR: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// The two valid values for the `sparse_display` option.
const METADATA: &str = "metadata";
const CONTENT: &str = "content";

/// Return the `contextvars.ContextVar` object for the `sparse_display` option.
///
/// Created lazily with `default="metadata"` on the first call.
fn sparse_display_var(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    SPARSE_DISPLAY_VAR
        .get_or_try_init(py, || {
            let cv_mod = py.import("contextvars")?;
            // ContextVar(name, *, default=...) — default is keyword-only.
            let kwargs = PyDict::new(py);
            kwargs.set_item("default", METADATA)?;
            let var: Py<PyAny> = cv_mod
                .call_method(
                    "ContextVar",
                    ("hurray.print_options.sparse_display",),
                    Some(&kwargs),
                )?
                .unbind();
            Ok(var)
        })
        .map(|v| v.bind(py).clone())
}

/// Validate a `sparse_display` string and raise `ValueError` for unknown values.
// `py` is accepted for API symmetry with future option validators that may need it.
fn validate_sparse_display(value: &str) -> PyResult<()> {
    if value == METADATA || value == CONTENT {
        Ok(())
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "invalid sparse_display value {value:?}; expected 'metadata' or 'content'"
        )))
    }
}

/// Set one or more print options for the current asyncio task / thread context.
///
/// Options not provided (left as `None`) are unchanged; this keeps the signature
/// extensible as new options are added in the future.
///
/// ## Arguments
///
/// - `sparse_display` — controls how `Tensor.__repr__` renders a sparse-layout tensor:
///   - `"metadata"` *(default)* — SciPy-style compact summary:
///     `hurray.Tensor(layout='csr', shape=(3, 3), nnz=4, dtype=float32)`
///   - `"content"` — PyTorch-style, appending per-format buffer arrays after the
///     dtype field:
///     `hurray.Tensor(layout='csr', …, values=[…], col_indices=[…], row_ptr=[…])`
///
/// ## Raises
///
/// `ValueError` if `sparse_display` is not `"metadata"` or `"content"`.
///
/// ## Examples
///
/// ```python
/// import hurray
/// hurray.set_print_options(sparse_display="content")
/// hurray.set_print_options(sparse_display="metadata")  # restore default
/// ```
#[pyfunction]
#[pyo3(signature = (*, sparse_display = None))]
pub fn set_print_options(py: Python<'_>, sparse_display: Option<&str>) -> PyResult<()> {
    if let Some(v) = sparse_display {
        validate_sparse_display(v)?;
        sparse_display_var(py)?.call_method1("set", (v,))?;
    }
    Ok(())
}

/// Return the current print options as a `dict`.
///
/// Keys reflect the full set of configurable options; all values are strings.
///
/// ## Examples
///
/// ```python
/// import hurray
/// opts = hurray.get_print_options()
/// assert opts["sparse_display"] == "metadata"
/// ```
#[pyfunction]
pub fn get_print_options(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    let current: String = sparse_display_var(py)?
        .call_method0("get")?
        .extract::<String>()?;
    d.set_item("sparse_display", current)?;
    Ok(d.into())
}

/// Context manager that temporarily overrides print options for the current task.
///
/// On `__enter__`, sets the requested option(s) and stores the reset token.
/// On `__exit__`, resets each ContextVar to its prior value via the stored token.
/// This is exception-safe and supports nesting correctly.
///
/// ## Arguments
///
/// Same keyword arguments as [`set_print_options`].
///
/// ## Examples
///
/// ```python
/// import hurray
/// assert hurray.get_print_options()["sparse_display"] == "metadata"
/// with hurray.print_options(sparse_display="content"):
///     assert hurray.get_print_options()["sparse_display"] == "content"
/// assert hurray.get_print_options()["sparse_display"] == "metadata"
/// ```
#[pyclass(name = "PrintOptionsCtx")]
pub struct PrintOptionsCtx {
    /// The requested `sparse_display` override, or `None` for no change.
    sparse_display: Option<String>,
    /// Token from `ContextVar.set()`, stored so `__exit__` can call `reset()`.
    sparse_display_token: Option<Py<PyAny>>,
}

#[pymethods]
impl PrintOptionsCtx {
    #[new]
    #[pyo3(signature = (*, sparse_display = None))]
    pub fn new(sparse_display: Option<String>) -> PyResult<Self> {
        // Validate at construction time so errors surface at the `with …:` call site,
        // not inside __enter__. Reuse the shared validator so the accepted values and
        // error message live in one place.
        if let Some(ref v) = sparse_display {
            validate_sparse_display(v)?;
        }
        Ok(PrintOptionsCtx {
            sparse_display,
            sparse_display_token: None,
        })
    }

    fn __enter__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> PyResult<()> {
        // Compute the token while borrowing `sparse_display`, then assign the disjoint
        // `sparse_display_token` field afterwards — `PyRefMut` is a smart pointer, so
        // the two borrows cannot overlap and the value must outlive the immutable one.
        let token = match &slf.sparse_display {
            Some(v) => Some(
                sparse_display_var(py)?
                    .call_method1("set", (v.as_str(),))?
                    .unbind(),
            ),
            None => None,
        };
        slf.sparse_display_token = token;
        Ok(())
    }

    fn __exit__(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        _exc_type: &Bound<'_, PyAny>,
        _exc_val: &Bound<'_, PyAny>,
        _exc_tb: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        if let Some(token) = slf.sparse_display_token.take() {
            sparse_display_var(py)?.call_method1("reset", (token,))?;
        }
        Ok(false)
    }
}

/// Factory function returning a `PrintOptionsCtx` context manager.
///
/// Equivalent to `with hurray.PrintOptionsCtx(…):`. On entry the ContextVar(s)
/// are set; on exit they are restored to their previous values. Supports nesting.
///
/// ## Arguments
///
/// Same keyword arguments as [`set_print_options`].
///
/// ## Examples
///
/// ```python
/// import hurray
/// with hurray.print_options(sparse_display="content"):
///     assert hurray.get_print_options()["sparse_display"] == "content"
/// assert hurray.get_print_options()["sparse_display"] == "metadata"
/// ```
#[pyfunction]
#[pyo3(signature = (*, sparse_display = None))]
pub fn print_options(sparse_display: Option<String>) -> PyResult<PrintOptionsCtx> {
    PrintOptionsCtx::new(sparse_display)
}

// ── Crate-internal helpers ────────────────────────────────────────────────────

/// Return `true` if the current context has `sparse_display` set to `"content"`.
///
/// Falls back to `false` (i.e. `"metadata"` behaviour) if the ContextVar cannot
/// be read — this avoids raising from `__repr__`/`__str__`.
pub(crate) fn sparse_display_is_content(py: Python<'_>) -> bool {
    sparse_display_var(py)
        .and_then(|v| v.call_method0("get"))
        .and_then(|v| v.extract::<String>())
        .map(|s| s == CONTENT)
        // On any Python error (e.g. interpreter shutting down), fall back to metadata.
        .unwrap_or(false)
}

// ── Registration ──────────────────────────────────────────────────────────────

/// Register print-option API items on the `hurray` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_print_options, m)?)?;
    m.add_function(wrap_pyfunction!(get_print_options, m)?)?;
    m.add_function(wrap_pyfunction!(print_options, m)?)?;
    m.add_class::<PrintOptionsCtx>()?;
    Ok(())
}
