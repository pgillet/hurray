//! Native Hurray buffer interchange protocol: `__hurray__` / `from_hurray`.
//!
//! Implements a PyCapsule-based protocol for zero-copy tensor buffer sharing between
//! Hurray-aware Python extensions. The capsule wraps a single `HurrayTensor` pointer from
//! `hurray-ffi`, validated by an ABI version check before anything else is read.
//!
//! ## Design decisions (see also ADR-023, ADR-030, ADR-043)
//!
//! **D-NB1 — Capsule pointer = `*mut HurrayTensor`, and nothing else:** a tensor is a
//! descriptor and its buffers, and the handle owns both, so a C consumer reaches either
//! half through `hurray-ffi` accessors without linking PyO3 — and cannot reach one half
//! without the other. ADR-030 widened the pointer from a single `*mut HurrayBuffer` to a
//! `HurrayBufferList` so multi-buffer tensors travel whole; ADR-043 narrowed it again to
//! the handle that owns that list. Element `i` of the list is descriptor buffer index `i`.
//!
//! **D-NB2 — No capsule context:** the encoded descriptor and the producing build's ABI
//! version used to ride in a separate `HurrayTensorContext` fetched with
//! `PyCapsule_GetContext`, paired with the pointer only by convention. Both now live in
//! the handle (ADR-043), so the producer sets no context and teardown is one call.
//!
//! **D-NB3 — The strong reference travels as the handle's opaque `owner`:** released
//! through `owner_release`, so no Python type reaches the C ABI.
//!
//! **D-NB4 — Python consumer reuses `BufferStore::Borrowed`:** after consuming the
//! capsule, `from_hurray` reads every buffer's ptr/len, clones the decoded descriptor,
//! calls `hurray_tensor_destroy` once, and creates one `BufferStore::Borrowed` per
//! buffer whose `base` is the source `Tensor`.

use std::ffi::CStr;

use hurray_core::{DeviceTag, MemoryClass, SyncMode, TensorDescriptor};
use hurray_ffi::{
    HurrayBuffer, HurrayBufferList, HurrayTensor, HURRAY_C_ABI_VERSION, HURRAY_OK,
};
use pyo3::prelude::*;

use crate::buffer::BufferStore;
use crate::device::Device;
use crate::dtype::Dtype;
use crate::errors::{BufferError, InvalidDescriptorError, UnsupportedError};
use crate::tensor::Tensor;

// ── Capsule names ─────────────────────────────────────────────────────────────

/// Capsule name for a fresh (unconsumed) native protocol capsule.
///
/// Renamed from `"hurray_tensor"` by ADR-043 § 5, once: the pointer's *type* changed, and
/// the version that would let a consumer detect that now lives inside the thing whose type
/// changed. A name mismatch is a hard `TypeError` in every consumer, in every language,
/// with no cooperation required; a stale consumer reading the pointer first would
/// otherwise get undefined behaviour.
const HURRAY_TENSOR_CAPSULE_NAME: &CStr = c"hurray_tensor_handle";

/// Capsule name after consumption — prevents destructor from double-destroying.
const HURRAY_TENSOR_CAPSULE_USED: &CStr = c"used_hurray_tensor_handle";

// ── The handle's owner ────────────────────────────────────────────────────────

/// What the tensor handle's opaque `owner` pointer actually is: a strong Python
/// reference to the source `Tensor`, keeping its buffers alive for the capsule's
/// lifetime. Released by [`release_owner`]; `hurray-ffi` never interprets it, which is
/// what keeps Python types out of the C ABI (ADR-043 § 1).
type CapsuleOwner = Py<PyAny>;

// ── Capsule destructor ────────────────────────────────────────────────────────

/// Called by the Python GC when the capsule object is collected.
///
/// - If still named `"hurray_tensor_handle"` (not consumed): destroys the
///   `HurrayTensor` — its descriptor, every `HurrayBuffer` its list owns, and the owner
///   reference to the source `Tensor`.
/// - If renamed to `"used_hurray_tensor_handle"`: the consumer already destroyed it —
///   nothing to do.
///
/// # Safety
///
/// Called by CPython with a valid `Py<PyAny>*` pointing to a live PyCapsule.
/// The GIL is always held when Python finalizers run.
///
/// A finalizer may also run *during interpreter finalization*, which is why this
/// function must never take a `Python` token — see [`release_context`].
unsafe extern "C" fn hurray_tensor_capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    let name_ptr = pyo3::ffi::PyCapsule_GetName(capsule);
    if name_ptr.is_null() {
        return;
    }
    // SAFETY: PyCapsule_GetName returns a valid NUL-terminated string while the capsule lives.
    let name_bytes = CStr::from_ptr(name_ptr).to_bytes();

    if name_bytes == HURRAY_TENSOR_CAPSULE_NAME.to_bytes() {
        // Unconsumed — one destroy releases the descriptor, every buffer handle, and the
        // owner reference, in an order the caller cannot get wrong (ADR-043 § 7).
        let mut tensor = pyo3::ffi::PyCapsule_GetPointer(capsule, name_ptr) as *mut HurrayTensor;
        if !tensor.is_null() {
            // SAFETY: created by HurrayTensor::from_core and destroyed exactly once —
            // either here or by the consumer, which renames the capsule first.
            hurray_ffi::tensor::hurray_tensor_destroy(&mut tensor);
        }
    }
    // "used_hurray_tensor_handle": the consumer already destroyed the handle.
}

/// Releases the source-tensor reference a capsule context owns.
///
/// Handed to the tensor handle as its `owner_release`, so this is the only
/// Python-touching code on the capsule path — which is why the finalization guard lives
/// here and nowhere else.
///
/// Deliberately does **not** go through `Python::attach`. A capsule finalizer can run
/// while the interpreter is being torn down, where `Python::attach` asserts that the
/// interpreter is initialized — and that panic cannot unwind across the `extern "C"`
/// boundary, so it aborts the process instead of exiting cleanly. The raw C API needs
/// no token, and once the interpreter is gone there is no refcount left to maintain:
/// the object's memory is released with the interpreter itself.
///
/// # Safety
///
/// `owner` must be the pointer produced by `Box::into_raw` in [`build_capsule`], and
/// must not have been reclaimed. The GIL must be held, which CPython guarantees for
/// capsule finalizers.
unsafe extern "C" fn release_owner(owner: *mut std::ffi::c_void) {
    if owner.is_null() {
        return;
    }
    let tensor_ref = *Box::from_raw(owner as *mut CapsuleOwner);
    // Consumes the strong reference without a token; we now owe exactly one DECREF.
    let obj = tensor_ref.into_ptr();
    if pyo3::ffi::Py_IsInitialized() != 0 {
        pyo3::ffi::Py_DECREF(obj);
    }
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

/// Build a `"hurray_tensor_handle"` PyCapsule for the given tensor.
///
/// The capsule pointer is a heap-allocated `HurrayTensor` owning the descriptor and a
/// `HurrayBufferList` with one `HurrayBuffer` per entry of `buffers`, in descriptor
/// buffer-table order (ADR-030 § 3, ADR-043 § 3). No buffer release callbacks are
/// registered; buffer lifetime is managed by the strong reference to the source tensor,
/// which travels as the handle's opaque `owner`. The capsule carries no context.
///
/// ## Ownership
/// - **Consumed** via [`from_hurray`]: caller reads every buffer's ptr/len, clones the
///   descriptor, destroys the handle, and holds its own reference to the source in the
///   new `BufferStore::Borrowed` bases.
/// - **GC'd without consumption**: the capsule destructor destroys the handle, which
///   releases both halves and the owner reference.
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
            "tensor has no buffers; cannot produce a native protocol capsule",
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

    // 1. Build the owning list, then push one handle per buffer in order.
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

    // 2. Fold the list into the tensor handle, which takes ownership of it. The
    //    descriptor goes in as the decoded type this binding already holds: the C
    //    constructor's encode-then-decode round trip buys validation a Python `Tensor`
    //    already has, since its descriptor was built type-checked rather than parsed
    //    (ADR-043 § 1). The source-tensor reference rides along as the opaque owner.
    let owner = Box::into_raw(Box::new(tensor_obj)) as *mut std::ffi::c_void;
    // SAFETY: list_ptr is a live list nothing else owns; owner comes from Box::into_raw
    // and stays valid until release_owner runs.
    let mut tensor_ptr = unsafe {
        HurrayTensor::from_core(
            HURRAY_C_ABI_VERSION,
            descriptor.clone(),
            list_ptr,
            owner,
            Some(release_owner),
        )
    };

    // 3. Create the PyCapsule, named for the handle it wraps (ADR-043 § 5). No context:
    //    a capsule carries one pointer, and both halves are reachable through it.
    // SAFETY: tensor_ptr is non-null; the destructor handles cleanup on both the consume
    // and the GC path.
    let capsule = unsafe {
        pyo3::ffi::PyCapsule_New(
            tensor_ptr as *mut std::ffi::c_void,
            HURRAY_TENSOR_CAPSULE_NAME.as_ptr(),
            Some(hurray_tensor_capsule_destructor),
        )
    };
    if capsule.is_null() {
        // One destroy frees the handle, its buffers and the owner reference, before the
        // OOM propagates.
        // SAFETY: created by HurrayTensor::from_core; first and only destroy.
        unsafe { hurray_ffi::tensor::hurray_tensor_destroy(&mut tensor_ptr) };
        return Err(pyo3::exceptions::PyMemoryError::new_err(
            "failed to create hurray_tensor_handle capsule",
        ));
    }

    // SAFETY: PyCapsule_New returned a non-null owned reference.
    Ok(unsafe { Bound::from_owned_ptr(py, capsule).unbind() })
}

// ── Consumer ──────────────────────────────────────────────────────────────────

/// Accept any object whose `__hurray__()` returns a valid `"hurray_tensor_handle"`
/// capsule and return a new [`hurray.Tensor`](Tensor) that shares the buffer
/// without copying.
///
/// ## Protocol
///
/// 1. Calls `obj.__hurray__()` to obtain the capsule.
/// 2. Verifies the capsule name is `"hurray_tensor_handle"` and the ABI version matches.
/// 3. Renames the capsule to `"used_hurray_tensor_handle"` (prevents double-destroy).
/// 4. Reads every buffer's data pointer and byte size through the handle's borrowed
///    buffer list.
/// 5. Clones the handle's decoded `TensorDescriptor` — no bytes, no decode step.
/// 6. Calls `hurray_tensor_destroy` once, which releases both halves and the producer's
///    owner reference.
/// 7. Creates a new `Tensor` with `BufferStore::Borrowed` whose `base` keeps the
///    source `Tensor` alive (D-NB4).
///
/// ## Errors
///
/// - `TypeError` — `obj` does not expose `__hurray__`.
/// - `hurray.BufferError` — capsule is null, malformed, or already consumed.
/// - `hurray.UnsupportedError` — ABI version mismatch between producer and consumer.
/// - `hurray.InvalidDescriptorError` — the capsule's buffer count disagrees with its
///   descriptor.
///
/// ## Examples
///
/// ```python
/// import hurray
///
/// t = hurray.Tensor(bytes(16), hurray.float32, [4])
/// t2 = hurray.from_hurray(t)
/// assert t2.shape == t.shape
/// assert t2.dtype == t.dtype
/// ```
#[pyfunction]
pub fn from_hurray(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    // 1. Check protocol support.
    if !obj.hasattr("__hurray__")? {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "argument does not support the __hurray__ protocol; \
             use hasattr(obj, '__hurray__') to check before calling",
        ));
    }

    // 2. Call __hurray__() to get the capsule.
    let capsule_bound = obj.call_method0("__hurray__")?;
    let cap_ptr = capsule_bound.as_ptr();

    // 3. Verify it is a valid PyCapsule named "hurray_tensor_handle".
    // SAFETY: cap_ptr is a live borrowed Python object.
    let is_valid =
        unsafe { pyo3::ffi::PyCapsule_IsValid(cap_ptr, HURRAY_TENSOR_CAPSULE_NAME.as_ptr()) };
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
        if name_msg == "used_hurray_tensor_handle" {
            return Err(BufferError::new_err(
                "hurray_tensor_handle capsule has already been consumed; \
                 call __hurray__() again to get a fresh capsule",
            ));
        }
        return Err(BufferError::new_err(format!(
            "expected a 'hurray_tensor_handle' PyCapsule, got '{name_msg}'"
        )));
    }

    // 4. Take the one pointer the capsule carries.
    // SAFETY: PyCapsule_IsValid confirmed this is a live "hurray_tensor_handle" capsule.
    let mut tensor =
        unsafe { pyo3::ffi::PyCapsule_GetPointer(cap_ptr, HURRAY_TENSOR_CAPSULE_NAME.as_ptr()) }
            as *mut HurrayTensor;

    if tensor.is_null() {
        return Err(BufferError::new_err(
            "hurray_tensor_handle capsule is malformed (null pointer)",
        ));
    }

    // 5. Verify the ABI version — the first thing read from any handle, and the only
    //    accessor guaranteed to work across versions (ADR-043 § 5).
    let mut abi_version: u32 = 0;
    // SAFETY: tensor is non-null and points to a live HurrayTensor.
    unsafe { hurray_ffi::tensor::hurray_tensor_abi_version(tensor, &mut abi_version) };
    if abi_version != HURRAY_C_ABI_VERSION {
        return Err(UnsupportedError::new_err(format!(
            "ABI version mismatch: capsule was produced with hurray ABI v{abi_version}, \
             this consumer has ABI v{HURRAY_C_ABI_VERSION}; \
             ensure producer and consumer use the same hurray build",
        )));
    }

    // 6. Rename capsule → "used_hurray_tensor_handle" before modifying any shared state.
    //    This prevents the capsule destructor from destroying the handle a second time if
    //    the capsule outlives this function's stack frame.
    // SAFETY: cap_ptr is a valid live PyCapsule; we are the sole consumer.
    if unsafe { pyo3::ffi::PyCapsule_SetName(cap_ptr, HURRAY_TENSOR_CAPSULE_USED.as_ptr()) } != 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "failed to mark hurray_tensor_handle capsule as consumed",
        ));
    }

    // 7. Read every buffer's pointer and size, in list order, before the handle is
    //    destroyed. Both the list and the handles in it are borrowed — the tensor owns
    //    them (ADR-043 § 1) — so they are read here and never destroyed individually.
    let mut list: *mut HurrayBufferList = std::ptr::null_mut();
    // SAFETY: tensor is live; out-pointer is a valid stack variable.
    unsafe { hurray_ffi::tensor::hurray_tensor_buffers(tensor, &mut list) };

    let mut buffer_count: u64 = 0;
    if !list.is_null() {
        // SAFETY: list is a live borrow owned by the tensor.
        unsafe { hurray_ffi::buffer_list::hurray_buffer_list_len(list, &mut buffer_count) };
    }

    let mut raw_buffers: Vec<(*mut std::ffi::c_void, u64)> =
        Vec::with_capacity(buffer_count as usize);
    for index in 0..buffer_count {
        let mut borrowed: *mut HurrayBuffer = std::ptr::null_mut();
        // SAFETY: list is live; index < len; out-pointer is a valid stack variable.
        let status =
            unsafe { hurray_ffi::buffer_list::hurray_buffer_list_get(list, index, &mut borrowed) };
        if status != HURRAY_OK || borrowed.is_null() {
            // SAFETY: the capsule is consumed, so the handle is ours to destroy; one call
            // releases the list and the owner reference too.
            unsafe { hurray_ffi::tensor::hurray_tensor_destroy(&mut tensor) };
            return Err(BufferError::new_err(format!(
                "hurray_tensor_handle capsule: buffer {index} could not be read \
                 (status {status})"
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

    // 8. Clone the handle's decoded descriptor. There are no encoded bytes to decode:
    //    the handle keeps only the decoded form (ADR-043 § 6), and cloning is what
    //    outlives the destroy below. A live handle always has one.
    // SAFETY: tensor is non-null and points to a live HurrayTensor.
    let descriptor = match unsafe { (*tensor).core_descriptor() } {
        Some(descriptor) => descriptor.clone(),
        None => {
            // SAFETY: the capsule is consumed, so the handle is ours to destroy.
            unsafe { hurray_ffi::tensor::hurray_tensor_destroy(&mut tensor) };
            return Err(BufferError::new_err(
                "hurray_tensor_handle capsule is malformed (no descriptor)",
            ));
        }
    };

    // The new tensor borrows from the source, so it holds its own reference to it.
    // `obj` is that source: from_hurray called `obj.__hurray__()` to get this capsule.
    let tensor_ref: Py<PyAny> = obj.clone().unbind();

    // 9. One destroy ends the handle: the descriptor, every buffer handle (no release
    //    callbacks were registered, so this only frees the structs), and the producer's
    //    reference to the source tensor. The payload stays alive because `tensor_ref`
    //    above is this consumer's own reference to it.
    // SAFETY: created by HurrayTensor::from_core; first and only destroy.
    unsafe { hurray_ffi::tensor::hurray_tensor_destroy(&mut tensor) };

    // 10. A length mismatch means the descriptor's buffer indices cannot all resolve to a
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
                "hurray_tensor capsule carries no buffers",
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
    m.add_function(wrap_pyfunction!(from_hurray, m)?)?;
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
    fn from_hurray_rejects_non_protocol_objects() {
        init();
        Python::attach(|py| {
            let obj = py.eval(c"42", None, None).unwrap();
            let result = from_hurray(py, &obj);
            assert!(result.is_err());
            assert!(result
                .unwrap_err()
                .is_instance_of::<pyo3::exceptions::PyTypeError>(py));
        });
    }

    #[test]
    fn from_hurray_rejects_wrong_capsule_name() {
        init();
        Python::attach(|py| {
            // A capsule with the wrong name should raise BufferError.
            let capsule = unsafe {
                pyo3::ffi::PyCapsule_New(
                    std::ptr::dangling_mut::<std::ffi::c_void>(),
                    c"not_hurray_tensor".as_ptr(),
                    None,
                )
            };
            assert!(!capsule.is_null());
            let capsule_obj = unsafe { Bound::from_owned_ptr(py, capsule).unbind() };

            // Build a fake object that returns our capsule from __hurray__.
            // We can't easily do this in a unit test without a full Python class,
            // so just test PyCapsule_IsValid directly.
            let is_valid = unsafe {
                pyo3::ffi::PyCapsule_IsValid(
                    capsule_obj.as_ptr(),
                    HURRAY_TENSOR_CAPSULE_NAME.as_ptr(),
                )
            };
            assert_eq!(is_valid, 0, "wrong-named capsule should not be valid");
        });
    }

    #[test]
    fn capsule_name_constants_are_correct() {
        assert_eq!(HURRAY_TENSOR_CAPSULE_NAME.to_bytes(), b"hurray_tensor_handle");
        assert_eq!(
            HURRAY_TENSOR_CAPSULE_USED.to_bytes(),
            b"used_hurray_tensor_handle"
        );
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

            let capsule = Tensor::__hurray__(&bound, None).unwrap();

            // The capsule pointer is a HurrayTensor owning both buffers and the
            // descriptor that indexes them.
            let handle = unsafe {
                pyo3::ffi::PyCapsule_GetPointer(
                    capsule.as_ptr(),
                    HURRAY_TENSOR_CAPSULE_NAME.as_ptr(),
                )
            } as *mut HurrayTensor;
            assert!(!handle.is_null());

            // And nothing rides in a context: there is no second fetch to forget
            // (ADR-043 § 3).
            let context = unsafe { pyo3::ffi::PyCapsule_GetContext(capsule.as_ptr()) };
            assert!(context.is_null(), "a capsule carries one pointer, no context");

            let mut list: *mut HurrayBufferList = std::ptr::null_mut();
            unsafe { hurray_ffi::tensor::hurray_tensor_buffers(handle, &mut list) };
            assert!(!list.is_null());
            let mut len = 0u64;
            unsafe { hurray_ffi::buffer_list::hurray_buffer_list_len(list, &mut len) };
            assert_eq!(len, 2, "both buffers must travel in one capsule");

            // The descriptor arrives through the same handle, already decoded.
            let declared = unsafe { (*handle).core_descriptor() }
                .expect("a live handle has a descriptor")
                .buffers
                .len();
            assert_eq!(declared, 2, "the pairing is structural, not conventional");

            // Consuming it reconstructs a two-buffer tensor with identical bytes.
            let rebuilt = from_hurray(py, &bound).unwrap();
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
                vec![Some(4)],
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();
            let bound = Py::new(py, tensor).unwrap().into_bound(py);

            let rebuilt = from_hurray(py, &bound).unwrap();
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
