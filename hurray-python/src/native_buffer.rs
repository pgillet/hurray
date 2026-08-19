//! Native Hurray buffer interchange protocol: `__hurray_buffer__` / `from_hurray_buffer`.
//!
//! Implements a PyCapsule-based protocol for zero-copy tensor buffer sharing between
//! Hurray-aware Python extensions. The capsule wraps a `HurrayBufferList` pointer from
//! `hurray-ffi`, validated by an ABI version check before the pointer is used.
//!
//! ## Design decisions (see also ADR-023, ADR-030)
//!
//! **D-NB1 — Capsule pointer = `*mut HurrayBufferList`:** Allows C consumers to call
//! `hurray-ffi` accessor functions without linking PyO3. ADR-030 widened this from a
//! single `*mut HurrayBuffer` so that multi-buffer tensors — per-channel / NF4 / MXFP
//! quantization, sparse, block-paged — travel whole. Element `i` of the list is
//! descriptor buffer index `i`.
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
use hurray_ffi::{HurrayBuffer, HurrayBufferList, HURRAY_C_ABI_VERSION, HURRAY_OK};
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
/// - If still named `"hurray_buffer"` (not consumed): destroys the `HurrayBufferList`
///   — and with it every `HurrayBuffer` it owns — and drops the context (releasing
///   the source Tensor reference).
/// - If renamed to `"used_hurray_buffer"`: consumer already destroyed the list
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
        // Unconsumed — destroy the list, which destroys every handle it owns.
        let mut list = pyo3::ffi::PyCapsule_GetPointer(capsule, name_ptr) as *mut HurrayBufferList;
        if !list.is_null() {
            // SAFETY: created by hurray_buffer_list_new; this is the first and only destroy.
            hurray_ffi::buffer_list::hurray_buffer_list_destroy(&mut list);
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
    // "used_hurray_buffer": consumer already destroyed the list and freed the context.
}

// ── Producer ─────────────────────────────────────────────────────────────────

/// One buffer to place in a capsule, in descriptor buffer-table order.
pub(crate) struct CapsuleBuffer {
    pub data_ptr: *mut std::ffi::c_void,
    pub byte_size: u64,
    pub alignment: u32,
    pub device_tag: DeviceTag,
    pub sync_mode: SyncMode,
    pub memory_class: MemoryClass,
}

/// Build a `"hurray_buffer"` PyCapsule for the given tensor.
///
/// The capsule pointer is a heap-allocated `HurrayBufferList` owning one
/// `HurrayBuffer` per entry of `buffers`, in descriptor buffer-table order
/// (ADR-030 § 2–3). No release callbacks are registered; buffer lifetime is
/// managed by the `tensor_ref` in the context. The capsule context is a
/// heap-allocated `NativeBufferContext` carrying the encoded descriptor and a
/// strong Python reference to the source tensor.
///
/// ## Ownership
/// - **Consumed** via [`from_hurray_buffer`]: caller reads every buffer's ptr/len,
///   destroys the list, and moves `tensor_ref` into the new `BufferStore::Borrowed`
///   bases.
/// - **GC'd without consumption**: the capsule destructor destroys the list — and
///   with it every handle it owns — and drops the context.
pub(crate) fn build_capsule(
    py: Python<'_>,
    tensor_obj: Py<PyAny>,
    descriptor: &TensorDescriptor,
    buffers: &[CapsuleBuffer],
) -> PyResult<Py<PyAny>> {
    use hurray_ffi::buffer::hurray_buffer_from_ptr;
    use hurray_ffi::buffer_list::{
        hurray_buffer_list_destroy, hurray_buffer_list_new, hurray_buffer_list_push,
    };

    if buffers.is_empty() {
        return Err(BufferError::new_err(
            "tensor has no buffers; cannot produce a native buffer capsule",
        ));
    }

    // A capsule whose list length disagrees with the descriptor's buffer table would
    // hand the consumer a descriptor with unresolvable buffer indices (ADR-030 § 3).
    if buffers.len() != descriptor.buffers.len() {
        return Err(InvalidDescriptorError::new_err(format!(
            "descriptor declares {} buffers but {} were supplied",
            descriptor.buffers.len(),
            buffers.len()
        )));
    }

    // 1. Encode descriptor for round-trip reconstruction on the consumer side.
    let descriptor_bytes = descriptor.encode().map_err(|e| {
        InvalidDescriptorError::new_err(format!("failed to encode descriptor: {e}"))
    })?;

    // 2. Build the owning list, then push one handle per buffer in order.
    let mut list_ptr: *mut HurrayBufferList = std::ptr::null_mut();
    // SAFETY: out-pointer is a valid stack variable.
    let status = unsafe { hurray_buffer_list_new(buffers.len() as u64, &mut list_ptr) };
    if status != HURRAY_OK {
        return Err(BufferError::new_err(format!(
            "failed to create HurrayBufferList (status {status})"
        )));
    }

    for buf in buffers {
        // No release callback: the source tensor's Python refcount manages the memory.
        let mut handle_ptr: *mut HurrayBuffer = std::ptr::null_mut();
        // SAFETY: data_ptr is valid for byte_size bytes; alignment is a power-of-two ≥ 1.
        let status = unsafe {
            hurray_buffer_from_ptr(
                buf.data_ptr,
                buf.byte_size,
                buf.alignment,
                buf.device_tag.to_byte(),
                buf.sync_mode.to_byte(),
                buf.memory_class.to_byte(),
                None,
                std::ptr::null_mut(),
                &mut handle_ptr,
            )
        };
        if status != HURRAY_OK {
            // Drop the handles already pushed rather than leaking them.
            // SAFETY: list_ptr is live and owned by us at this point.
            unsafe { hurray_buffer_list_destroy(&mut list_ptr) };
            return Err(BufferError::new_err(format!(
                "failed to create HurrayBuffer handle (status {status})"
            )));
        }

        // SAFETY: both handles are live; push transfers ownership of handle_ptr.
        let status = unsafe { hurray_buffer_list_push(list_ptr, handle_ptr) };
        if status != HURRAY_OK {
            // handle_ptr is still ours on failure — destroy it, then the list.
            // SAFETY: push failed, so ownership did not transfer.
            unsafe { hurray_ffi::buffer::hurray_buffer_destroy(handle_ptr) };
            // SAFETY: list_ptr is live and owned by us.
            unsafe { hurray_buffer_list_destroy(&mut list_ptr) };
            return Err(BufferError::new_err(format!(
                "failed to append buffer to HurrayBufferList (status {status})"
            )));
        }
    }

    // 3. Build context.
    let ctx = Box::new(NativeBufferContext {
        abi_version: HURRAY_C_ABI_VERSION,
        descriptor_bytes,
        tensor_ref: tensor_obj,
    });
    let ctx_raw = Box::into_raw(ctx) as *mut std::ffi::c_void;

    // 4. Create PyCapsule named "hurray_buffer".
    // SAFETY: list_ptr is non-null; destructor handles cleanup on both consume and GC paths.
    let capsule = unsafe {
        pyo3::ffi::PyCapsule_New(
            list_ptr as *mut std::ffi::c_void,
            HURRAY_BUFFER_CAPSULE_NAME.as_ptr(),
            Some(hurray_buffer_capsule_destructor),
        )
    };
    if capsule.is_null() {
        // Free allocations to avoid a leak before propagating OOM.
        unsafe {
            let _ = Box::from_raw(ctx_raw as *mut NativeBufferContext);
            hurray_buffer_list_destroy(&mut list_ptr);
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

    // 4. Extract list and context.
    // SAFETY: PyCapsule_IsValid confirmed this is a live "hurray_buffer" capsule.
    let mut list =
        unsafe { pyo3::ffi::PyCapsule_GetPointer(cap_ptr, HURRAY_BUFFER_CAPSULE_NAME.as_ptr()) }
            as *mut HurrayBufferList;
    let ctx_ptr = unsafe { pyo3::ffi::PyCapsule_GetContext(cap_ptr) } as *mut NativeBufferContext;

    if list.is_null() || ctx_ptr.is_null() {
        return Err(BufferError::new_err(
            "hurray_buffer capsule is malformed (null list or context)",
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

    // 7. Read every buffer's pointer and size, in list order, before the list is
    //    destroyed. Handles from hurray_buffer_list_get are borrowed — the list owns
    //    them (ADR-030 § 2) — so they are read here and never destroyed individually.
    let mut buffer_count: u64 = 0;
    // SAFETY: list is non-null and live; out-pointer is a valid stack variable.
    unsafe { hurray_ffi::buffer_list::hurray_buffer_list_len(list, &mut buffer_count) };

    let mut raw_buffers: Vec<(*mut std::ffi::c_void, u64)> =
        Vec::with_capacity(buffer_count as usize);
    for index in 0..buffer_count {
        let mut borrowed: *mut HurrayBuffer = std::ptr::null_mut();
        // SAFETY: list is live; index < len; out-pointer is a valid stack variable.
        let status =
            unsafe { hurray_ffi::buffer_list::hurray_buffer_list_get(list, index, &mut borrowed) };
        if status != HURRAY_OK || borrowed.is_null() {
            // SAFETY: list is live and owned by us now that the capsule is consumed.
            unsafe { hurray_ffi::buffer_list::hurray_buffer_list_destroy(&mut list) };
            // SAFETY: ctx_ptr came from Box::into_raw and has not been reclaimed yet.
            unsafe { drop(Box::from_raw(ctx_ptr)) };
            return Err(BufferError::new_err(format!(
                "hurray_buffer capsule: buffer {index} could not be read (status {status})"
            )));
        }
        let mut data_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut byte_size: u64 = 0;
        // SAFETY: borrowed is a live handle owned by the list; out-pointers are valid.
        unsafe {
            hurray_ffi::buffer::hurray_buffer_data_ptr(borrowed, &mut data_ptr);
            hurray_ffi::buffer::hurray_buffer_byte_size(borrowed, &mut byte_size);
        }
        raw_buffers.push((data_ptr, byte_size));
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

    // 9. Destroy the list and every handle it owns (no release callbacks were
    //    registered, so this only frees the handle structs).
    // SAFETY: created by hurray_buffer_list_new; first and only destroy.
    unsafe { hurray_ffi::buffer_list::hurray_buffer_list_destroy(&mut list) };

    // 10. Decode TensorDescriptor.
    let descriptor = TensorDescriptor::decode(&descriptor_bytes).map_err(|e| {
        InvalidDescriptorError::new_err(format!("failed to decode descriptor: {e}"))
    })?;

    // A length mismatch means the descriptor's buffer indices cannot all resolve to a
    // transported buffer — reject rather than build a tensor with a dangling index
    // (ADR-030 § 3).
    if raw_buffers.len() != descriptor.buffers.len() {
        return Err(InvalidDescriptorError::new_err(format!(
            "capsule carries {} buffers but its descriptor declares {}",
            raw_buffers.len(),
            descriptor.buffers.len()
        )));
    }

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

    // 12. Create a borrowed BufferStore per transported buffer, each backed by the
    //     source tensor (D-NB3). Every store holds its own strong reference, so the
    //     source stays alive until the last of them is dropped.
    let mut stores = Vec::with_capacity(raw_buffers.len());
    for (data_ptr, byte_size) in raw_buffers {
        let base = tensor_ref.clone_ref(py);
        // SAFETY: data_ptr is valid for byte_size bytes for as long as base is alive.
        stores
            .push(unsafe { BufferStore::borrowed(data_ptr as *mut u8, byte_size as usize, base) });
    }
    let mut stores = stores.into_iter();
    let buffer = match stores.next() {
        Some(b) => b,
        None => {
            return Err(BufferError::new_err(
                "hurray_buffer capsule carries no buffers",
            ))
        }
    };
    let aux_buffers: Vec<BufferStore> = stores.collect();

    Ok(Tensor {
        descriptor,
        buffer,
        aux_buffers,
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

    // ── Multi-buffer capsule (ADR-030) ────────────────────────────────────────

    /// Build a two-buffer tensor: int8 data plus a float32 per-channel scale
    /// buffer, the shape a per-channel-quantized weight tensor has.
    fn two_buffer_tensor(py: Python<'_>) -> Tensor {
        use hurray_core::{
            BufferHandle, ElementType, LayoutDescriptor, Shape, SyncMode, TensorDescriptor,
            DESCRIPTOR_VERSION_MAJOR, DESCRIPTOR_VERSION_MINOR, MIN_BUFFER_ALIGNMENT,
        };

        let data = vec![7u8; 8];
        let scales = vec![1u8; 8];

        let bh = |len: u64| {
            BufferHandle::with_memory_class(
                len,
                MIN_BUFFER_ALIGNMENT,
                DeviceTag::Cpu,
                SyncMode::ProducerSynced,
                MemoryClass::Standard,
            )
            .unwrap()
        };

        let descriptor = TensorDescriptor::new(
            DESCRIPTOR_VERSION_MAJOR,
            DESCRIPTOR_VERSION_MINOR,
            ElementType::Int8,
            Shape::new(vec![2u64, 4]).unwrap(),
            0,
            LayoutDescriptor::RowMajor,
            vec![bh(8), bh(8)],
            None,
            None,
            None,
            None,
        )
        .unwrap();

        Tensor {
            descriptor,
            buffer: BufferStore::from_slice(&data),
            aux_buffers: vec![BufferStore::from_slice(&scales)],
            dtype_py: Py::new(
                py,
                Dtype {
                    inner: ElementType::Int8,
                },
            )
            .unwrap(),
            device_py: Py::new(
                py,
                Device {
                    tag: DeviceTag::Cpu,
                    memory_class: MemoryClass::Standard,
                    device_id: 0,
                },
            )
            .unwrap(),
        }
    }

    #[test]
    fn capsule_carries_every_buffer_and_round_trips() {
        init();
        Python::attach(|py| {
            let tensor = two_buffer_tensor(py);
            let bound = Py::new(py, tensor).unwrap().into_bound(py);

            let capsule = Tensor::__hurray_buffer__(&bound, None).unwrap();

            // The capsule pointer is a HurrayBufferList holding both buffers.
            let list = unsafe {
                pyo3::ffi::PyCapsule_GetPointer(
                    capsule.as_ptr(),
                    HURRAY_BUFFER_CAPSULE_NAME.as_ptr(),
                )
            } as *mut HurrayBufferList;
            assert!(!list.is_null());
            let mut len = 0u64;
            unsafe { hurray_ffi::buffer_list::hurray_buffer_list_len(list, &mut len) };
            assert_eq!(len, 2, "both buffers must travel in one capsule");

            // Consuming it reconstructs a two-buffer tensor with identical bytes.
            let rebuilt = from_hurray_buffer(py, &bound).unwrap();
            assert_eq!(rebuilt.buffer_count(), 2);
            let bytes: Vec<Vec<u8>> = rebuilt
                .buffers()
                .map(|b| unsafe { b.as_slice() }.to_vec())
                .collect();
            assert_eq!(bytes[0], vec![7u8; 8], "data buffer survives the hop");
            assert_eq!(bytes[1], vec![1u8; 8], "scale buffer survives the hop");
            assert_eq!(rebuilt.descriptor.buffers.len(), 2);
        });
    }

    #[test]
    fn single_buffer_tensor_is_just_the_n_equals_one_case() {
        init();
        Python::attach(|py| {
            let _m = crate::tensor::tests::build_module(py);
            let buf = pyo3::types::PyBytes::new(py, &[0u8; 16]);
            let dtype = Py::new(
                py,
                Dtype {
                    inner: hurray_core::ElementType::Float32,
                },
            )
            .unwrap();
            let tensor = Tensor::new(
                py,
                &buf,
                dtype.bind(py),
                vec![4],
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();
            let bound = Py::new(py, tensor).unwrap().into_bound(py);

            let rebuilt = from_hurray_buffer(py, &bound).unwrap();
            assert_eq!(rebuilt.buffer_count(), 1);
            assert!(rebuilt.aux_buffers.is_empty());
        });
    }

    #[test]
    fn build_capsule_rejects_a_buffer_count_that_disagrees_with_the_descriptor() {
        init();
        Python::attach(|py| {
            let tensor = two_buffer_tensor(py);
            let descriptor = tensor.descriptor.clone();
            let obj: Py<PyAny> = Py::new(py, tensor).unwrap().into_any();

            // One buffer supplied for a descriptor declaring two: the consumer would
            // receive a scale_buffer_index pointing at nothing.
            let only_one = [CapsuleBuffer {
                data_ptr: [0u8; 8].as_mut_ptr().cast(),
                byte_size: 8,
                alignment: 64,
                device_tag: DeviceTag::Cpu,
                sync_mode: SyncMode::ProducerSynced,
                memory_class: MemoryClass::Standard,
            }];
            let err = build_capsule(py, obj, &descriptor, &only_one).unwrap_err();
            assert!(
                err.to_string().contains("declares 2 buffers but 1"),
                "unexpected error: {err}"
            );
        });
    }
}
