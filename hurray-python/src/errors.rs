//! Python exception hierarchy for hurray (ADR-022, python-bindings.md § Error Handling).
//!
//! All errors from the Rust core are surfaced as typed Python exceptions.
//! Callers that do not need to distinguish sub-types MAY catch the built-in
//! base class (`ValueError`, `NotImplementedError`, `RuntimeError`).
//!
//! # Exception class tree
//!
//! ```text
//! ValueError
//! ├── hurray.InvalidDescriptorError  — parse / validation errors
//! ├── hurray.BufferError             — buffer size / alignment errors
//! │                                    (distinct from builtins.BufferError)
//! └── hurray.CopyRequiredError       — copy=False requested but copy is needed
//! NotImplementedError
//! └── hurray.UnsupportedError        — unsupported element type or layout
//! RuntimeError
//! └── hurray.InternalError           — unexpected Rust panics
//! OSError
//! ├── hurray.FileError               — HRRYFILE I/O errors (Layer 8b)
//! └── hurray.StreamError             — streaming framing errors (Layer 8b)
//! ```

// PyO3 0.22 macro expansion emits a redundant .into() on PyErr — false positive.
#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;

pyo3::create_exception!(
    hurray,
    InvalidDescriptorError,
    pyo3::exceptions::PyValueError,
    "Raised when a Hurray tensor descriptor fails validation.\n\
     Subclass of :exc:`ValueError`."
);

pyo3::create_exception!(
    hurray,
    BufferError,
    pyo3::exceptions::PyValueError,
    "Raised for buffer size or alignment errors.\n\
     Subclass of :exc:`ValueError`.\n\n\
     .. note::\n\
        This is ``hurray.BufferError``, distinct from the built-in\n\
        ``builtins.BufferError`` used by the DLPack protocol for\n\
        non-representable element types."
);

pyo3::create_exception!(
    hurray,
    CopyRequiredError,
    pyo3::exceptions::PyValueError,
    "Raised when ``copy=False`` is passed to ``__array__`` but a copy is required\n\
     (e.g. a dtype cast is needed). Subclass of :exc:`ValueError`.\n\n\
     This follows the NumPy 2.0 ``__array__`` convention (NEP 47): the caller\n\
     requested zero-copy but the conversion cannot be done without copying."
);

pyo3::create_exception!(
    hurray,
    UnsupportedError,
    pyo3::exceptions::PyNotImplementedError,
    "Raised when an element type, memory layout, or device combination is\n\
     not supported by the current Hurray version.\n\
     Subclass of :exc:`NotImplementedError`."
);

pyo3::create_exception!(
    hurray,
    InternalError,
    pyo3::exceptions::PyRuntimeError,
    "Raised when an unexpected Rust panic is caught by the binding layer.\n\
     Subclass of :exc:`RuntimeError`.\n\n\
     If you see this exception, please file a bug report."
);

pyo3::create_exception!(
    hurray,
    FileError,
    pyo3::exceptions::PyOSError,
    "Raised by ``hurray.load()`` and ``hurray.save()`` for file-level errors:\n\
     file not found, permission denied, corrupt HRRYFILE container, unexpected EOF.\n\
     Subclass of :exc:`OSError`."
);

pyo3::create_exception!(
    hurray,
    StreamError,
    pyo3::exceptions::PyOSError,
    "Raised by the streaming reader/writer for framing errors:\n\
     frame corruption mid-stream, unexpected stream termination.\n\
     Subclass of :exc:`OSError`."
);

/// Register all exception classes on the `hurray` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "InvalidDescriptorError",
        m.py().get_type_bound::<InvalidDescriptorError>(),
    )?;
    m.add("BufferError", m.py().get_type_bound::<BufferError>())?;
    m.add(
        "CopyRequiredError",
        m.py().get_type_bound::<CopyRequiredError>(),
    )?;
    m.add(
        "UnsupportedError",
        m.py().get_type_bound::<UnsupportedError>(),
    )?;
    m.add("InternalError", m.py().get_type_bound::<InternalError>())?;
    m.add("FileError", m.py().get_type_bound::<FileError>())?;
    m.add("StreamError", m.py().get_type_bound::<StreamError>())?;
    Ok(())
}

/// Catch a Rust panic inside `f` and convert it to [`InternalError`].
///
/// Use this wrapper around any Rust call site that could panic due to an
/// internal invariant violation. PyO3's own wrapper already catches panics
/// (raising `RuntimeError`), but this ensures the more specific
/// `hurray.InternalError` subclass is raised with a clear message.
///
/// # Example
///
/// ```rust,no_run
/// # use pyo3::prelude::*;
/// # use hurray::errors::{catch_panic, InternalError};
/// # fn example(py: Python<'_>) -> PyResult<()> {
/// let result = catch_panic(|| {
///     // ... call into Rust core that might panic ...
///     Ok(42_i64)
/// })?;
/// # Ok(())
/// # }
/// ```
pub fn catch_panic<T, F>(f: F) -> PyResult<T>
where
    F: FnOnce() -> PyResult<T>,
{
    // SAFETY: AssertUnwindSafe is required because PyO3 types are not
    // UnwindSafe (they contain raw pointers). We only use catch_unwind for
    // error reporting — we do not resume execution after a panic.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or_else(|payload| {
        let msg = payload
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
            .unwrap_or("unexpected panic in Rust core");
        Err(InternalError::new_err(format!("internal error: {msg}")))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init_python() {
        pyo3::prepare_freethreaded_python();
    }

    #[test]
    fn invalid_descriptor_error_is_value_error() {
        init_python();
        Python::with_gil(|py| {
            let err = InvalidDescriptorError::new_err("bad descriptor");
            assert!(err.is_instance_of::<InvalidDescriptorError>(py));
            assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
        });
    }

    #[test]
    fn buffer_error_is_value_error() {
        init_python();
        Python::with_gil(|py| {
            let err = BufferError::new_err("misaligned");
            assert!(err.is_instance_of::<BufferError>(py));
            assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
        });
    }

    #[test]
    fn copy_required_error_is_value_error() {
        init_python();
        Python::with_gil(|py| {
            let err = CopyRequiredError::new_err("cast required");
            assert!(err.is_instance_of::<CopyRequiredError>(py));
            assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
        });
    }

    #[test]
    fn unsupported_error_is_not_implemented_error() {
        init_python();
        Python::with_gil(|py| {
            let err = UnsupportedError::new_err("int4 not supported here");
            assert!(err.is_instance_of::<UnsupportedError>(py));
            assert!(err.is_instance_of::<pyo3::exceptions::PyNotImplementedError>(py));
        });
    }

    #[test]
    fn internal_error_is_runtime_error() {
        init_python();
        Python::with_gil(|py| {
            let err = InternalError::new_err("unexpected state");
            assert!(err.is_instance_of::<InternalError>(py));
            assert!(err.is_instance_of::<pyo3::exceptions::PyRuntimeError>(py));
        });
    }

    #[test]
    fn catch_panic_converts_str_panic() {
        init_python();
        Python::with_gil(|_py| {
            let result: PyResult<i32> = catch_panic(|| panic!("something broke"));
            let err = result.unwrap_err();
            Python::with_gil(|py| {
                assert!(err.is_instance_of::<InternalError>(py));
                assert!(err.to_string().contains("something broke"));
            });
        });
    }

    #[test]
    fn catch_panic_passes_through_ok() {
        let result: PyResult<i32> = catch_panic(|| Ok(42));
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn file_error_is_os_error() {
        init_python();
        Python::with_gil(|py| {
            let err = FileError::new_err("file not found");
            assert!(err.is_instance_of::<FileError>(py));
            assert!(err.is_instance_of::<pyo3::exceptions::PyOSError>(py));
        });
    }

    #[test]
    fn stream_error_is_os_error() {
        init_python();
        Python::with_gil(|py| {
            let err = StreamError::new_err("framing error");
            assert!(err.is_instance_of::<StreamError>(py));
            assert!(err.is_instance_of::<pyo3::exceptions::PyOSError>(py));
        });
    }

    #[test]
    fn catch_panic_passes_through_err() {
        init_python();
        Python::with_gil(|py| {
            let result: PyResult<i32> = catch_panic(|| Err(UnsupportedError::new_err("nope")));
            let err = result.unwrap_err();
            assert!(err.is_instance_of::<UnsupportedError>(py));
        });
    }
}
