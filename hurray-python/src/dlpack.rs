//! DLPack v1.0 FFI types and helper functions.
//!
//! Defines the C structs for `DLManagedTensorVersioned` (the v1.0 capsule format)
//! and provides helpers to translate between Hurray types and DLPack types.
//!
//! ## Design decisions
//!
//! **D1 — v1.0 capsule:** Hurray emits `"dltensor_versioned"` (DLPack v1.0) rather
//! than the legacy `"dltensor"` (v0.8). NumPy 2.0, PyTorch 2.1+, JAX, and CuPy all
//! produce and consume v1.0 capsules today.
//!
//! **D3 — NULL strides for row-major tensors:** When the tensor is C-contiguous
//! (row-major), the strides pointer is set to NULL per DLPack convention. Consumers
//! treat NULL strides as C-contiguous. Explicit strides are only emitted for layouts
//! that are genuinely non-contiguous.

use std::os::raw::{c_int, c_uint, c_void};

use hurray_core::{DeviceTag, ElementType, LayoutDescriptor, MemoryClass};
use pyo3::prelude::*;

use crate::errors::UnsupportedError;

// ── DLPack v1.0 C ABI ─────────────────────────────────────────────────────────

/// DLPack device type enum (subset used by Hurray).
///
/// Values are the DLPack integer constants; do NOT confuse with Hurray's
/// `DeviceTag` values — they are intentionally different (ADR-016).
#[allow(dead_code)]
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DLDeviceType {
    Cpu = 1,
    Cuda = 2,
    CudaHost = 3,
    OpenCl = 4,
    Vulkan = 7,
    Metal = 8,
    Rocm = 10,
    RocmHost = 11,
    OneApi = 14,
    WebGpu = 15,
    Hexagon = 16,
    CudaManaged = 13,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DLDevice {
    pub device_type: c_int,
    pub device_id: c_int,
}

/// DLPack data type code.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DLDataType {
    /// Type code (0=int, 1=uint, 2=float, 3=opaque, 4=bfloat, 5=complex).
    pub code: u8,
    /// Number of bits per element (e.g., 32 for float32).
    pub bits: u8,
    /// Number of elements in a vector (1 for scalar types).
    pub lanes: u16,
}

#[repr(C)]
pub struct DLTensor {
    /// Pointer to the data buffer.
    pub data: *mut c_void,
    pub device: DLDevice,
    /// Number of dimensions.
    pub ndim: c_int,
    pub dtype: DLDataType,
    /// Shape array (ndim elements). Owned by the DLManagedTensorVersioned.
    pub shape: *mut i64,
    /// Strides in elements (ndim elements), or NULL for C-contiguous (D3).
    pub strides: *mut i64,
    /// Byte offset from `data` to element [0, …, 0].
    pub byte_offset: u64,
}

/// DLPack v1.0 managed tensor. The capsule named `"dltensor_versioned"` wraps a
/// heap-allocated instance of this struct. Ownership is transferred to the capsule
/// destructor (or consumer on rename).
#[repr(C)]
pub struct DLManagedTensorVersioned {
    /// Major version — MUST be 1 for this struct layout.
    pub version: DLPackVersion,
    /// Opaque handle passed to `deleter`. Hurray uses this to hold a Py<PyAny>
    /// strong reference to the source Tensor, keeping its buffer alive.
    pub manager_ctx: *mut c_void,
    /// Called when the capsule is destroyed (not consumed) or at consumer finalisation.
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
    /// Bitmask of flags (none defined for v1.0 beyond the version field).
    pub flags: u64,
    /// The actual tensor — shape/strides pointers live in the same allocation.
    pub dl_tensor: DLTensor,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DLPackVersion {
    pub major: c_uint,
    pub minor: c_uint,
}

// DLManagedTensorVersioned contains raw pointers; we only pass it to Python
// capsules (single-threaded refcount) so asserting Send+Sync is safe here.
unsafe impl Send for DLManagedTensorVersioned {}
unsafe impl Sync for DLManagedTensorVersioned {}

// ── Capsule payload ────────────────────────────────────────────────────────────

/// Heap-allocated payload for a DLPack capsule.
///
/// Owns the shape/strides arrays pointed to by `DLTensor` and a strong Python
/// reference that keeps the source `Tensor` alive for the capsule's lifetime.
/// Allocated as `Box<DLPackPayload>` and stored via `manager_ctx`.
struct DLPackPayload {
    /// Strong reference to the source `hurray.Tensor` Python object.
    /// Kept for its drop side-effect: keeps the Tensor (and its buffer) alive for the
    /// capsule's lifetime. Never read — the value matters, not what's in it.
    #[allow(dead_code)]
    tensor_ref: Py<PyAny>,
    shape: Box<[i64]>,
    /// `None` for row-major (DLPack NULL strides convention, D3).
    strides: Option<Box<[i64]>>,
}

// ── Capsule lifecycle ──────────────────────────────────────────────────────────

/// DLPack deleter — called by the consumer when done, or by the capsule destructor
/// if the capsule was never consumed (still named `"dltensor_versioned"`).
///
/// Frees: the `DLPackPayload` (releases Python ref + shape/strides), then the
/// `DLManagedTensorVersioned` struct itself.
///
/// # Safety
/// `managed` must be a valid non-null pointer to a `DLManagedTensorVersioned`
/// previously allocated via `Box::into_raw`. Called at most once.
pub unsafe extern "C" fn dlpack_deleter(managed: *mut DLManagedTensorVersioned) {
    // Reconstruct and drop the payload: releases tensor Python ref + shape/strides.
    let _ = Box::from_raw((*managed).manager_ctx as *mut DLPackPayload);
    // Reconstruct and drop the managed struct itself.
    let _ = Box::from_raw(managed);
}

/// Capsule destructor registered with `PyCapsule_New`.
///
/// If the capsule is still named `"dltensor_versioned"` (not consumed), calls the
/// DLPack deleter to free everything. If renamed to `"used_dltensor_versioned"`,
/// the consumer is responsible — nothing to do.
///
/// # Safety
/// Called by the Python GC with a valid `Py<PyAny>*` pointing to a PyCapsule.
unsafe extern "C" fn capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    let name = pyo3::ffi::PyCapsule_GetName(capsule);
    if name.is_null() {
        return;
    }
    // SAFETY: name is a valid NUL-terminated C string from PyCapsule_GetName.
    let name_bytes = std::ffi::CStr::from_ptr(name).to_bytes();
    if name_bytes == b"dltensor_versioned" {
        // Not consumed — call deleter to free everything.
        let managed =
            pyo3::ffi::PyCapsule_GetPointer(capsule, name) as *mut DLManagedTensorVersioned;
        if !managed.is_null() {
            if let Some(del) = (*managed).deleter {
                del(managed);
            }
        }
    }
    // If "used_dltensor_versioned": consumer already called deleter — nothing to do.
}

// ── Capsule builder ────────────────────────────────────────────────────────────

/// Capsule name for a fresh (unconsumed) DLPack v1.0 tensor.
pub const DLPACK_CAPSULE_NAME: &std::ffi::CStr = c"dltensor_versioned";

/// Build a DLPack v1.0 PyCapsule for the given tensor.
///
/// ## Parameters
/// - `tensor_obj`: strong Python reference to the source `hurray.Tensor`; kept alive
///   for the capsule's lifetime via the `DLPackPayload.tensor_ref` field.
/// - `data_ptr`: pointer to the start of the element buffer.
/// - `element_type`: Hurray element type (translated to `DLDataType`).
/// - `shape`: tensor shape as `u64` dims; DYNAMIC (`u64::MAX`) maps to `-1`.
/// - `layout`: layout descriptor, used to compute DLPack strides (D3).
/// - `device_type`: pre-translated DLPack device type integer.
/// - `device_id`: device index (default `0`).
///
/// ## Errors
/// - `builtins.BufferError`: element type not representable in DLPack (e.g., bool, int4).
/// - `hurray.UnsupportedError`: layout cannot be expressed as DLPack strides.
/// - `PyMemoryError`: PyCapsule_New returned NULL.
#[allow(clippy::too_many_arguments)]
pub fn build_capsule(
    py: Python<'_>,
    tensor_obj: Py<PyAny>,
    data_ptr: *mut c_void,
    element_type: ElementType,
    shape: &[u64],
    layout: &LayoutDescriptor,
    device_type: c_int,
    device_id: i32,
) -> PyResult<Py<PyAny>> {
    // 1. Resolve DLDataType — fails for bool/int4/float8/quantized (D9).
    let dl_dtype = element_type_to_dlpack(element_type).ok_or_else(|| {
        pyo3::exceptions::PyBufferError::new_err(format!(
            "element type '{}' cannot be represented in DLPack v1.0; \
             bool is 1-bit packed in Hurray but 1-byte in DLPack — use __hurray__ instead",
            crate::dtype::element_type_name(element_type),
        ))
    })?;

    // 2. Build shape array (DYNAMIC = u64::MAX → -1 per DLPack convention).
    let dl_shape: Box<[i64]> = shape
        .iter()
        .map(|&d| {
            if d == hurray_core::DYNAMIC {
                -1
            } else {
                d as i64
            }
        })
        .collect();
    let ndim = dl_shape.len() as c_int;

    // 3. Compute strides (D3: None = NULL = C-contiguous).
    let strides_opt = strides_for_layout(layout, shape);
    let is_unsupported_layout = strides_opt
        .as_ref()
        .is_some_and(|v| v.is_empty() && ndim > 0);
    if is_unsupported_layout {
        return Err(UnsupportedError::new_err(
            "tensor layout cannot be expressed as DLPack strides (e.g. tiled, Morton); \
             use __hurray__ instead",
        ));
    }
    let dl_strides: Option<Box<[i64]>> = strides_opt.map(|v| v.into_boxed_slice());

    // 4. Build payload — owns shape/strides allocations and the tensor strong ref.
    let mut payload = Box::new(DLPackPayload {
        tensor_ref: tensor_obj,
        shape: dl_shape,
        strides: dl_strides,
    });

    let shape_ptr = payload.shape.as_mut_ptr();
    let strides_ptr = payload
        .strides
        .as_mut()
        .map_or(std::ptr::null_mut(), |s| s.as_mut_ptr());
    let manager_ctx = Box::into_raw(payload) as *mut c_void;

    // 5. Build DLManagedTensorVersioned on the heap.
    let managed = Box::new(DLManagedTensorVersioned {
        version: DLPackVersion { major: 1, minor: 0 },
        manager_ctx,
        deleter: Some(dlpack_deleter),
        flags: 0,
        dl_tensor: DLTensor {
            data: data_ptr,
            device: DLDevice {
                device_type,
                device_id,
            },
            ndim,
            dtype: dl_dtype,
            shape: shape_ptr,
            strides: strides_ptr,
            byte_offset: 0,
        },
    });
    let managed_ptr = Box::into_raw(managed);

    // 6. Create PyCapsule named "dltensor_versioned" (D1).
    // SAFETY: managed_ptr is a valid non-null heap allocation; capsule_destructor
    // will call dlpack_deleter if the capsule is GC'd without being consumed.
    let capsule = unsafe {
        pyo3::ffi::PyCapsule_New(
            managed_ptr as *mut c_void,
            DLPACK_CAPSULE_NAME.as_ptr(),
            Some(capsule_destructor),
        )
    };

    if capsule.is_null() {
        // PyCapsule_New failed — free allocations to avoid leak before propagating OOM.
        unsafe {
            let _ = Box::from_raw((*managed_ptr).manager_ctx as *mut DLPackPayload);
            let _ = Box::from_raw(managed_ptr);
        }
        return Err(pyo3::exceptions::PyMemoryError::new_err(
            "failed to create DLPack capsule",
        ));
    }

    // SAFETY: PyCapsule_New returned a non-null owned reference.
    Ok(unsafe { Bound::from_owned_ptr(py, capsule).unbind() })
}

// ── Type mapping helpers ───────────────────────────────────────────────────────

/// Translate a Hurray `ElementType` to a DLPack `DLDataType`.
///
/// Returns `None` for types that have no DLPack equivalent (sub-byte types,
/// float8 variants, quantized types). The caller MUST raise `builtins.BufferError`
/// in that case per the Array API spec.
///
/// **D9 — Bool dtype:** Hurray bool is 1-bit packed; DLPack bool is 1 byte.
/// There is no lossless zero-copy mapping, so we return `None` for `Bool`.
pub fn element_type_to_dlpack(ty: ElementType) -> Option<DLDataType> {
    // DLPack type codes: 0=int, 1=uint, 2=float, 4=bfloat, 5=complex
    let (code, bits): (u8, u8) = match ty {
        // D9: bool is 1-bit packed in Hurray; DLPack bool = 1 byte — not representable zero-copy.
        ElementType::Bool => return None,
        ElementType::Int8 => (0, 8),
        ElementType::Int16 => (0, 16),
        ElementType::Int32 => (0, 32),
        ElementType::Int64 => (0, 64),
        ElementType::Uint8 => (1, 8),
        ElementType::Uint16 => (1, 16),
        ElementType::Uint32 => (1, 32),
        ElementType::Uint64 => (1, 64),
        ElementType::Float16 => (2, 16),
        ElementType::Float32 => (2, 32),
        ElementType::Float64 => (2, 64),
        ElementType::BFloat16 => (4, 16),
        ElementType::Complex64 => (5, 64),
        ElementType::Complex128 => (5, 128),
        // Tier 2 / quantized types — not representable in DLPack.
        _ => return None,
    };
    Some(DLDataType {
        code,
        bits,
        lanes: 1,
    })
}

/// Translate `(DeviceTag, MemoryClass)` to a DLPack `DLDeviceType` integer.
///
/// Raises `hurray.UnsupportedError` for combinations that DLPack v1.0 cannot
/// represent (e.g., ROCm UNIFIED, PEER memory, private device tags).
///
/// See python-bindings.md §Device and Memory Class Mapping for the full table.
pub fn device_to_dlpack(tag: DeviceTag, memory_class: MemoryClass) -> PyResult<c_int> {
    let code = match (tag, memory_class) {
        (DeviceTag::Cpu, _) => DLDeviceType::Cpu as c_int,
        (DeviceTag::Cuda, MemoryClass::Standard) => DLDeviceType::Cuda as c_int,
        (DeviceTag::Cuda, MemoryClass::HostPinned) => DLDeviceType::CudaHost as c_int,
        (DeviceTag::Cuda, MemoryClass::Unified) => DLDeviceType::CudaManaged as c_int,
        (DeviceTag::Cuda, MemoryClass::Peer | MemoryClass::Private(_)) => {
            return Err(UnsupportedError::new_err(
                "CUDA PEER/Private memory has no DLPack equivalent; use __hurray__ instead",
            ))
        }
        (DeviceTag::Rocm, MemoryClass::Standard) => DLDeviceType::Rocm as c_int,
        (DeviceTag::Rocm, MemoryClass::HostPinned) => DLDeviceType::RocmHost as c_int,
        (DeviceTag::Rocm, MemoryClass::Unified | MemoryClass::Peer | MemoryClass::Private(_)) => {
            // ROCm UNIFIED has no DLPack equivalent (kDLROCMManaged does not exist in v1.0).
            return Err(UnsupportedError::new_err(
                "ROCm UNIFIED/PEER/Private memory has no DLPack equivalent; use __hurray__ instead",
            ));
        }
        (DeviceTag::Metal, _) => DLDeviceType::Metal as c_int,
        (DeviceTag::Vulkan, MemoryClass::Peer) => {
            return Err(UnsupportedError::new_err(
                "Vulkan PEER memory has no DLPack equivalent; use __hurray__ instead",
            ))
        }
        (DeviceTag::Vulkan, _) => DLDeviceType::Vulkan as c_int,
        (DeviceTag::WebGpu, _) => DLDeviceType::WebGpu as c_int,
        (DeviceTag::Hexagon, _) => DLDeviceType::Hexagon as c_int,
        (DeviceTag::LevelZero, MemoryClass::Peer) => {
            return Err(UnsupportedError::new_err(
                "Level Zero PEER memory has no DLPack equivalent; use __hurray__ instead",
            ))
        }
        (DeviceTag::LevelZero, _) => DLDeviceType::OneApi as c_int,
        (DeviceTag::OpenCl, _) => DLDeviceType::OpenCl as c_int,
        // Private device tags have no DLPack mapping; the binding must not fabricate one.
        (DeviceTag::Private(_), _) => {
            return Err(UnsupportedError::new_err(
                "private device tags have no DLPack mapping; use __hurray__ instead",
            ))
        }
    };
    Ok(code)
}

/// Compute explicit strides (in elements) for a C-contiguous tensor of the given shape.
///
/// Returns `None` for row-major tensors — the caller should pass NULL to DLPack
/// rather than allocating a strides array (D3: NULL strides = C-contiguous convention).
pub fn strides_for_layout(layout: &LayoutDescriptor, shape: &[u64]) -> Option<Vec<i64>> {
    match layout {
        // D3: row-major is C-contiguous — signal via NULL strides (None here).
        LayoutDescriptor::RowMajor => None,
        LayoutDescriptor::Strided(s) => {
            let strides: Vec<i64> = s.strides.to_vec();
            Some(strides)
        }
        // Column-major: strides go from innermost to outermost dimension.
        LayoutDescriptor::ColMajor => {
            let ndim = shape.len();
            if ndim == 0 {
                return None;
            }
            let mut strides = vec![1i64; ndim];
            for i in 1..ndim {
                strides[i] = strides[i - 1] * (shape[i - 1] as i64);
            }
            Some(strides)
        }
        // All other layouts (tiled, Morton, Hilbert, sparse) cannot be represented
        // as a simple strides array — signal as unsupported at the call site.
        _ => Some(vec![]), // sentinel: empty vec = unsupported layout
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float32_maps_to_dlpack() {
        let dt = element_type_to_dlpack(ElementType::Float32).unwrap();
        assert_eq!(dt.code, 2);
        assert_eq!(dt.bits, 32);
        assert_eq!(dt.lanes, 1);
    }

    #[test]
    fn bfloat16_maps_to_dlpack() {
        let dt = element_type_to_dlpack(ElementType::BFloat16).unwrap();
        assert_eq!(dt.code, 4);
        assert_eq!(dt.bits, 16);
    }

    #[test]
    fn complex128_maps_to_dlpack() {
        let dt = element_type_to_dlpack(ElementType::Complex128).unwrap();
        assert_eq!(dt.code, 5);
        assert_eq!(dt.bits, 128);
    }

    #[test]
    fn bool_has_no_dlpack_mapping() {
        // D9: Hurray bool is 1-bit packed; DLPack bool is 1 byte — no zero-copy mapping.
        assert!(element_type_to_dlpack(ElementType::Bool).is_none());
    }

    #[test]
    fn int4_has_no_dlpack_mapping() {
        assert!(element_type_to_dlpack(ElementType::Int4).is_none());
    }

    #[test]
    fn row_major_returns_null_strides() {
        // D3: row-major → None (caller sets strides pointer to NULL).
        let result = strides_for_layout(&LayoutDescriptor::RowMajor, &[2, 3]);
        assert!(result.is_none());
    }

    #[test]
    fn column_major_strides() {
        let result = strides_for_layout(&LayoutDescriptor::ColMajor, &[2, 3]).unwrap();
        assert_eq!(result, vec![1, 2]);
    }

    #[test]
    fn cpu_standard_dlpack_device() {
        pyo3::Python::initialize();
        Python::attach(|_py| {
            let code = device_to_dlpack(DeviceTag::Cpu, MemoryClass::Standard).unwrap();
            assert_eq!(code, DLDeviceType::Cpu as c_int);
        });
    }

    #[test]
    fn rocm_unified_raises_unsupported() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let result = device_to_dlpack(DeviceTag::Rocm, MemoryClass::Unified);
            assert!(result.is_err());
            assert!(result.unwrap_err().is_instance_of::<UnsupportedError>(py));
        });
    }
}
