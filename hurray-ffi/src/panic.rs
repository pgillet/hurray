//! Panic barrier and null-pointer guard for the Hurray C ABI.
//!
//! Every `extern "C"` function body (except trivially infallible ones) MUST be
//! wrapped in [`catch`] so that Rust panics cannot unwind across the FFI
//! boundary, which is undefined behaviour in C.

use crate::status::{HurrayStatus, HURRAY_ERR_INTERNAL_PANIC};

// ── Panic barrier ─────────────────────────────────────────────────────────────

/// Calls `f` inside a `catch_unwind` fence and returns `HURRAY_ERR_INTERNAL_PANIC`
/// if `f` panics.
///
/// # Usage
///
/// Wrap the entire body of every `extern "C"` function (except infallible
/// trivial ones) in this call:
///
/// ```ignore
/// #[no_mangle]
/// pub unsafe extern "C" fn hurray_foo(handle: *mut HurrayBuffer) -> HurrayStatus {
///     catch(|| {
///         // ...body...
///         HURRAY_OK
///     })
/// }
/// ```
///
/// # Safety
///
/// `AssertUnwindSafe` is sound here: on panic the `INTERNAL_PANIC` status
/// signals to the caller that the handle is in an undefined state and MUST NOT
/// be reused; no invariant restoration is attempted inside the closure.
// SAFETY: AssertUnwindSafe is sound here — on panic, the INTERNAL_PANIC status
// signals the handle is in an undefined state and must not be reused; no
// invariant restoration is attempted.
pub(crate) fn catch<F: FnOnce() -> HurrayStatus>(f: F) -> HurrayStatus {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(status) => status,
        Err(_) => HURRAY_ERR_INTERNAL_PANIC,
    }
}

// ── Null-pointer guard ────────────────────────────────────────────────────────

/// Returns `HURRAY_ERR_NULL_POINTER` early if any of the listed raw pointers is null.
///
/// # Examples
///
/// ```ignore
/// null_check!(data, out_handle);
/// ```
macro_rules! null_check {
    ($($ptr:expr),+ $(,)?) => {
        $(
            if $ptr.is_null() {
                return $crate::status::HURRAY_ERR_NULL_POINTER;
            }
        )+
    };
}

pub(crate) use null_check;
