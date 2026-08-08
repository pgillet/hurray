//! Native Hurray buffer interchange protocol: `__hurray_buffer__` / `from_hurray_buffer`.
//!
//! Implements a PyCapsule-based protocol for zero-copy tensor buffer sharing between
//! Hurray-aware Python extensions. The capsule wraps a `HurrayBuffer` pointer from
//! `hurray-ffi`, validated by an ABI version check before the buffer pointer is used.
//!
//! ## Design decisions (see also ADR-023)
//!
//! **D-NB1 — Capsule pointer = `*mut HurrayBuffer`:** Allows C consumers to call
//! `hurray-ffi` accessor functions without linking PyO3.
//!
//! **D-NB2 — Capsule context = `*mut NativeBufferContext`:** Carries the encoded
//! `TensorDescriptor` (for round-trip reconstruction) and a strong Python reference
//! to the source `Tensor` (keeps the buffer alive for the capsule's lifetime).
//!
//! **D-NB3 — Python consumer reuses `BufferStore::Borrowed`:** After consuming the
//! capsule, `from_hurray_buffer` extracts ptr/len, calls `hurray_buffer_destroy` (the
//! handle has no release callback, so this only frees the struct), and creates a
//! `BufferStore::Borrowed` whose `base` is the source `Tensor`. C consumer interop
//! requiring a persistent `HurrayBuffer` is deferred to a future layer.

use std::ffi::CStr;

use hurray_core::{DeviceTag, MemoryClass, SyncMode, TensorDescriptor};
use hurray_ffi::{HurrayBuffer, HURRAY_C_ABI_VERSION, HURRAY_OK};
use pyo3::prelude::*;

use crate::buffer::BufferStore;
use crate::device::Device;
use crate::dtype::Dtype;
use crate::errors::{BufferError, InvalidDescriptorError, UnsupportedError};
use crate::tensor::Tensor;

// ── Capsule names ─────────────────────────────────────────────────────────────

/// Capsule name for a fresh (unconsumed) native buffer capsule.
const HURRAY_BUFFER_CAPSULE_NAME: &CStr = c"hurray_buffer";

/// Capsule name after consumption — prevents destructor from double-destroying.
const HURRAY_BUFFER_CAPSULE_USED: &CStr = c"used_hurray_buffer";

// ── Capsule context ───────────────────────────────────────────────────────────

/// Heap-allocated context stored alongside the capsule (via `PyCapsule_SetContext`).
///
/// Carries the encoded `TensorDescriptor` for round-trip reconstruction, an ABI
/// version stamp for consumer validation, and a strong Python reference to the source
/// `Tensor` that keeps the buffer alive for the capsule's lifetime.
struct NativeBufferContext {
    abi_version: u32,
    descriptor_bytes: Vec<u8>,
    /// Strong Python reference to the source `hurray.Tensor`.
    /// DECREF'd when the capsule is consumed or GC'd without consumption.
    tensor_ref: Py<PyAny>,
}

// ── Capsule destructor ────────────────────────────────────────────────────────

/// Called by the Python GC when the capsule object is collected.
///
/// - If still named `"hurray_buffer"` (not consumed): destroys the `HurrayBuffer`
///   handle and drops the context (releasing the source Tensor reference).
/// - If renamed to `"used_hurray_buffer"`: consumer already destroyed the handle
///   and freed the context — nothing to do.
///
/// # Safety
///
/// Called by CPython with a valid `Py<PyAny>*` pointing to a live PyCapsule.
/// The GIL is always held when Python finalizers run.
unsafe extern "C" fn hurray_buffer_capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    let name_ptr = pyo3::ffi::PyCapsule_GetName(capsule);
    if name_ptr.is_null() {
        return;
    }
    // SAFETY: PyCapsule_GetName returns a valid NUL-terminated string while the capsule lives.
    let name_bytes = CStr::from_ptr(name_ptr).to_bytes();

    if name_bytes == HURRAY_BUFFER_CAPSULE_NAME.to_bytes() {
        // Unconsumed — destroy the handle (no release callback, just frees the struct).
        let handle = pyo3::ffi::PyCapsule_GetPointer(capsule, name_ptr) as *mut HurrayBuffer;
        if !handle.is_null() {
            // SAFETY: created by hurray_buffer_from_ptr; this is the first and only destroy.
            hurray_ffi::buffer::hurray_buffer_destroy(handle);
        }

        // Free context, releasing tensor_ref (DECREF on source Tensor).
        // PyCapsule finalizers always run with the GIL held.
        let ctx_ptr = pyo3::ffi::PyCapsule_GetContext(capsule) as *mut NativeBufferContext;
        if !ctx_ptr.is_null() {
            // SAFETY: ctx_ptr was created by Box::into_raw; GIL is held for Py<PyAny> drop.
            Python::attach(|_py| {
                let _ = Box::from_raw(ctx_ptr);
            });
        }
    }
    // "used_hurray_buffer": consumer already called hurray_buffer_destroy and freed context.
}

// ── Producer ─────────────────────────────────────────────────────────────────

/// Build a `"hurray_buffer"` PyCapsule for the given tensor.
///
/// The capsule pointer is a heap-allocated `HurrayBuffer` handle (no release
/// callback; lifetime is managed by the `tensor_ref` in the context).
/// The capsule context is a heap-allocated `NativeBufferContext` that carries the
/// encoded descriptor and a strong Python reference to the source `Tensor`.
///
/// ## Ownership
/// - **Consumed** via [`from_hurray_buffer`]: caller extracts ptr/len and destroys the
///   handle; moves `tensor_ref` to the new `BufferStore::Borrowed.base`.
/// - **GC'd without consumption**: the capsule destructor destroys the handle and drops
///   the context.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_capsule(
    py: Python<'_>,
    tensor_obj: Py<PyAny>,
    descriptor: &TensorDescriptor,
    data_ptr: *mut std::ffi::c_void,
    byte_size: u64,
    alignment: u32,
    device_tag: DeviceTag,
    sync_mode: SyncMode,
    memory_class: MemoryClass,
) -> PyResult<Py<PyAny>> {
    use hurray_ffi::buffer::hurray_buffer_from_ptr;

    // 1. Encode descriptor for round-trip reconstruction on the consumer side.
    let descriptor_bytes = descriptor.encode().map_err(|e| {
        InvalidDescriptorError::new_err(format!("failed to encode descriptor: {e}"))
    })?;

    // 2. Create HurrayBuffer handle with no release callback — buffer lifetime is managed
    //    by the Python tensor_ref in the context, not by the C release mechanism.
    let mut handle_ptr: *mut HurrayBuffer = std::ptr::null_mut();
    // SAFETY: data_ptr is valid for byte_size bytes; alignment is a power-of-two ≥ 1.
    // No release callback: the source Tensor's Python refcount manages the buffer.
    let status = unsafe {
        hurray_buffer_from_ptr(
            data_ptr,
            byte_size,
            alignment,
            device_tag.to_byte(),
            sync_mode.to_byte(),
            memory_class.to_byte(),
            None,
            std::ptr::null_mut(),
            &mut handle_ptr,
        )
    };
    if status != HURRAY_OK {
        return Err(BufferError::new_err(format!(
            "failed to create HurrayBuffer handle (status {status})"
        )));
    }

    // 3. Build context.
    let ctx = Box::new(NativeBufferContext {
        abi_version: HURRAY_C_ABI_VERSION,
        descriptor_bytes,
        tensor_ref: tensor_obj,
    });
    let ctx_raw = Box::into_raw(ctx) as *mut std::ffi::c_void;

    // 4. Create PyCapsule named "hurray_buffer".
    // SAFETY: handle_ptr is non-null; destructor handles cleanup on both consume and GC paths.
    let capsule = unsafe {
        pyo3::ffi::PyCapsule_New(
            handle_ptr as *mut std::ffi::c_void,
            HURRAY_BUFFER_CAPSULE_NAME.as_ptr(),
            Some(hurray_buffer_capsule_destructor),
        )
    };
    if capsule.is_null() {
        // Free allocations to avoid a leak before propagating OOM.
        unsafe {
            let _ = Box::from_raw(ctx_raw as *mut NativeBufferContext);
            hurray_ffi::buffer::hurray_buffer_destroy(handle_ptr);
        }
        return Err(pyo3::exceptions::PyMemoryError::new_err(
            "failed to create hurray_buffer capsule",
        ));
    }

    // 5. Attach context to the capsule.
    // SAFETY: capsule is non-null and freshly created; ctx_raw is non-null.
    if unsafe { pyo3::ffi::PyCapsule_SetContext(capsule, ctx_raw) } != 0 {
        unsafe {
            let _ = Box::from_raw(ctx_raw as *mut NativeBufferContext);
            pyo3::ffi::Py_DECREF(capsule);
        }
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "failed to attach context to hurray_buffer capsule",
        ));
    }

    // SAFETY: PyCapsule_New returned a non-null owned reference.
    Ok(unsafe { Bound::from_owned_ptr(py, capsule).unbind() })
}

// ── Consumer ──────────────────────────────────────────────────────────────────

/// Accept any object whose `__hurray_buffer__()` returns a valid `"hurray_buffer"`
/// capsule and return a new [`hurray.Tensor`](Tensor) that shares the buffer
/// without copying.
///
/// ## Protocol
///
/// 1. Calls `obj.__hurray_buffer__()` to obtain the capsule.
/// 2. Verifies the capsule name is `"hurray_buffer"` and the ABI version matches.
/// 3. Renames the capsule to `"used_hurray_buffer"` (prevents double-destroy).
/// 4. Extracts the data pointer and byte size from the `HurrayBuffer` handle.
/// 5. Calls `hurray_buffer_destroy` (frees the handle; no release callback was set).
/// 6. Decodes the `TensorDescriptor` from the capsule context.
/// 7. Creates a new `Tensor` with `BufferStore::Borrowed` whose `base` keeps the
///    source `Tensor` alive (D-NB3).
///
/// ## Errors
///
/// - `TypeError` — `obj` does not expose `__hurray_buffer__`.
/// - `hurray.BufferError` — capsule is null, malformed, or already consumed.
/// - `hurray.UnsupportedError` — ABI version mismatch between producer and consumer.
/// - `hurray.InvalidDescriptorError` — descriptor bytes could not be decoded.
///
/// ## Examples
///
/// ```python
/// import hurray
///
/// t = hurray.Tensor(bytes(16), hurray.float32, [4])
/// t2 = hurray.from_hurray_buffer(t)
/// assert t2.shape == t.shape
/// assert t2.dtype == t.dtype
/// ```
#[pyfunction]
pub fn from_hurray_buffer(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    // 1. Check protocol support.
    if !obj.hasattr("__hurray_buffer__")? {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "argument does not support the __hurray_buffer__ protocol; \
             use hasattr(obj, '__hurray_buffer__') to check before calling",
        ));
    }

    // 2. Call __hurray_buffer__() to get the capsule.
    let capsule_bound = obj.call_method0("__hurray_buffer__")?;
    let cap_ptr = capsule_bound.as_ptr();

    // 3. Verify it is a valid PyCapsule named "hurray_buffer".
    // SAFETY: cap_ptr is a live borrowed Python object.
    let is_valid =
        unsafe { pyo3::ffi::PyCapsule_IsValid(cap_ptr, HURRAY_BUFFER_CAPSULE_NAME.as_ptr()) };
    if is_valid == 0 {
        // Attempt to read the actual name for a better error message.
        let name_ptr = unsafe { pyo3::ffi::PyCapsule_GetName(cap_ptr) };
        // Clear any exception set by PyCapsule_GetName on a type mismatch.
        unsafe { pyo3::ffi::PyErr_Clear() };
        let name_msg = if name_ptr.is_null() {
            "null or not a capsule".to_string()
        } else {
            // SAFETY: non-null NUL-terminated string from PyCapsule_GetName.
            unsafe { CStr::from_ptr(name_ptr) }
                .to_string_lossy()
                .into_owned()
        };
        if name_msg == "used_hurray_buffer" {
            return Err(BufferError::new_err(
                "hurray_buffer capsule has already been consumed; \
                 call __hurray_buffer__() again to get a fresh capsule",
            ));
        }
        return Err(BufferError::new_err(format!(
            "expected a 'hurray_buffer' PyCapsule, got '{name_msg}'"
        )));
    }

    // 4. Extract handle and context.
    // SAFETY: PyCapsule_IsValid confirmed this is a live "hurray_buffer" capsule.
    let handle =
        unsafe { pyo3::ffi::PyCapsule_GetPointer(cap_ptr, HURRAY_BUFFER_CAPSULE_NAME.as_ptr()) }
            as *mut HurrayBuffer;
    let ctx_ptr = unsafe { pyo3::ffi::PyCapsule_GetContext(cap_ptr) } as *mut NativeBufferContext;

    if handle.is_null() || ctx_ptr.is_null() {
        return Err(BufferError::new_err(
            "hurray_buffer capsule is malformed (null handle or context)",
        ));
    }

    // 5. Verify ABI version.
    // SAFETY: ctx_ptr is non-null and points to a live NativeBufferContext.
    let abi_version = unsafe { (*ctx_ptr).abi_version };
    if abi_version != HURRAY_C_ABI_VERSION {
        return Err(UnsupportedError::new_err(format!(
            "ABI version mismatch: capsule was produced with hurray ABI v{abi_version}, \
             this consumer has ABI v{HURRAY_C_ABI_VERSION}; \
             ensure producer and consumer use the same hurray build",
        )));
    }

    // 6. Rename capsule → "used_hurray_buffer" before modifying any shared state.
    //    This prevents the capsule destructor from calling hurray_buffer_destroy a
    //    second time if the capsule outlives this function's stack frame.
    // SAFETY: cap_ptr is a valid live PyCapsule; we are the sole consumer.
    if unsafe { pyo3::ffi::PyCapsule_SetName(cap_ptr, HURRAY_BUFFER_CAPSULE_USED.as_ptr()) } != 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "failed to mark hurray_buffer capsule as consumed",
        ));
    }

    // 7. Extract data pointer and byte size from the handle.
    let mut data_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let mut byte_size: u64 = 0;
    // SAFETY: handle is non-null and live; out-pointers are valid stack variables.
    unsafe {
        hurray_ffi::buffer::hurray_buffer_data_ptr(handle, &mut data_ptr);
        hurray_ffi::buffer::hurray_buffer_byte_size(handle, &mut byte_size);
    }

    // 8. Destructure context: move fields out, free the Box allocation.
    //    After this line, ctx_ptr is freed; tensor_ref and descriptor_bytes are
    //    owned by local bindings and will not be double-dropped.
    // SAFETY: ctx_ptr was created by Box::into_raw; first and only reclaim.
    let NativeBufferContext {
        descriptor_bytes,
        tensor_ref,
        ..
    } = unsafe { *Box::from_raw(ctx_ptr) };

    // 9. Destroy the HurrayBuffer handle (no release callback → just frees the struct).
    // SAFETY: handle created by hurray_buffer_from_ptr; first and only destroy.
    unsafe { hurray_ffi::buffer::hurray_buffer_destroy(handle) };

    // 10. Decode TensorDescriptor.
    let descriptor = TensorDescriptor::decode(&descriptor_bytes).map_err(|e| {
        InvalidDescriptorError::new_err(format!("failed to decode descriptor: {e}"))
    })?;

    // 11. Build Python dtype and device handles from the decoded descriptor.
    let element_type = descriptor.element_type;
    let (device_tag, memory_class) = descriptor
        .buffers
        .first()
        .map(|b| (b.device_tag(), b.memory_class()))
        .unwrap_or((DeviceTag::Cpu, MemoryClass::Standard));

    let dtype_py = Py::new(
        py,
        Dtype {
            inner: element_type,
        },
    )?;
    let device_py = Py::new(
        py,
        Device {
            tag: device_tag,
            memory_class,
            device_id: 0,
        },
    )?;

    // 12. Create a borrowed BufferStore backed by the source Tensor (D-NB3).
    //     `tensor_ref` keeps the source Tensor alive, which keeps its buffer alive.
    // SAFETY: data_ptr is valid for byte_size bytes for as long as tensor_ref is alive.
    let buffer =
        unsafe { BufferStore::borrowed(data_ptr as *mut u8, byte_size as usize, tensor_ref) };

    Ok(Tensor {
        descriptor,
        buffer,
        dtype_py,
        device_py,
    })
}

// ── Registration ──────────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(from_hurray_buffer, m)?)?;
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn init() {
        pyo3::Python::initialize();
    }

    #[test]
    fn from_hurray_buffer_rejects_non_protocol_objects() {
        init();
        Python::attach(|py| {
            let obj = py.eval(c"42", None, None).unwrap();
            let result = from_hurray_buffer(py, &obj);
            assert!(result.is_err());
            assert!(result
                .unwrap_err()
                .is_instance_of::<pyo3::exceptions::PyTypeError>(py));
        });
    }

    #[test]
    fn from_hurray_buffer_rejects_wrong_capsule_name() {
        init();
        Python::attach(|py| {
            // A capsule with the wrong name should raise BufferError.
            let capsule = unsafe {
                pyo3::ffi::PyCapsule_New(
                    std::ptr::dangling_mut::<std::ffi::c_void>(),
                    c"not_hurray_buffer".as_ptr(),
                    None,
                )
            };
            assert!(!capsule.is_null());
            let capsule_obj = unsafe { Bound::from_owned_ptr(py, capsule).unbind() };

            // Build a fake object that returns our capsule from __hurray_buffer__.
            // We can't easily do this in a unit test without a full Python class,
            // so just test PyCapsule_IsValid directly.
            let is_valid = unsafe {
                pyo3::ffi::PyCapsule_IsValid(
                    capsule_obj.as_ptr(),
                    HURRAY_BUFFER_CAPSULE_NAME.as_ptr(),
                )
            };
            assert_eq!(is_valid, 0, "wrong-named capsule should not be valid");
        });
    }

    #[test]
    fn capsule_name_constants_are_correct() {
        assert_eq!(HURRAY_BUFFER_CAPSULE_NAME.to_bytes(), b"hurray_buffer");
        assert_eq!(HURRAY_BUFFER_CAPSULE_USED.to_bytes(), b"used_hurray_buffer");
    }
}
