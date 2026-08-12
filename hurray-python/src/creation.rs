//! Tensor construction functions: `zeros`, `ones`, `full`, `empty`, `*_like`,
//! `arange`, `linspace`, `eye`, `asarray`, `from_dlpack`.
//!
//! These constructors are Tier 1 only. Tier 2 dtypes (e.g. `int4`, `float8`) have no
//! meaningful fill/step semantics here and are rejected with `hurray.UnsupportedError`;
//! such tensors are produced by the interop and decode paths instead.

use hurray_core::{
    buffer_size_bytes, BufferHandle, DeviceTag, ElementType, LayoutDescriptor, MemoryClass, Shape,
    SyncMode, TensorDescriptor, DESCRIPTOR_VERSION_MAJOR, DESCRIPTOR_VERSION_MINOR,
    MIN_BUFFER_ALIGNMENT,
};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyComplex, PyFloat, PyInt};
#[cfg(test)]
use pyo3::IntoPyObjectExt;

use crate::buffer::BufferStore;
use crate::device::Device;
use crate::dtype::Dtype;
use crate::errors::{BufferError, InvalidDescriptorError, UnsupportedError};
use crate::tensor::{is_tier1, Tensor};

// ── Shared helpers ─────────────────────────────────────────────────────────────

/// Resolve `dtype=None` (→ Float64) or an explicit `hurray.Dtype` object.
fn resolve_dtype(dtype: Option<&Bound<'_, PyAny>>) -> PyResult<ElementType> {
    match dtype {
        None => Ok(ElementType::Float64),
        Some(obj) => {
            if let Ok(d) = obj.extract::<PyRef<Dtype>>() {
                Ok(d.inner)
            } else {
                Err(InvalidDescriptorError::new_err(format!(
                    "dtype must be a hurray.Dtype, got {}",
                    obj.get_type().name()?
                )))
            }
        }
    }
}

/// Resolve `device=None` (→ CPU host) or an explicit `hurray.Device` object.
fn resolve_device(device: Option<&Bound<'_, PyAny>>) -> PyResult<(DeviceTag, MemoryClass)> {
    match device {
        None => Ok((DeviceTag::Cpu, MemoryClass::Standard)),
        Some(obj) => {
            let dev = obj
                .extract::<PyRef<Device>>()
                .map_err(|_| InvalidDescriptorError::new_err("device must be a hurray.Device"))?;
            Ok((dev.tag, dev.memory_class))
        }
    }
}

/// Reject Tier 2 dtypes: these constructors only produce Tier 1 tensors.
fn check_tier1(ty: ElementType) -> PyResult<()> {
    if !is_tier1(ty) {
        return Err(UnsupportedError::new_err(format!(
            "dtype '{}' is a Tier 2 type and is not supported by tensor constructors",
            crate::dtype::element_type_name(ty)
        )));
    }
    Ok(())
}

/// Construct a `Tensor` from a raw owned buffer.
fn make_owned_tensor(
    py: Python<'_>,
    data: Vec<u8>,
    element_type: ElementType,
    shape: Shape,
    device_tag: DeviceTag,
    memory_class: MemoryClass,
) -> PyResult<Tensor> {
    let alignment = if data.is_empty() {
        1
    } else {
        MIN_BUFFER_ALIGNMENT
    };
    let buffer_handle = BufferHandle::with_memory_class(
        data.len() as u64,
        alignment,
        device_tag,
        SyncMode::ProducerSynced,
        memory_class,
    )
    .map_err(|e| BufferError::new_err(format!("invalid buffer parameters: {e}")))?;

    let descriptor = TensorDescriptor::new(
        DESCRIPTOR_VERSION_MAJOR,
        DESCRIPTOR_VERSION_MINOR,
        element_type,
        shape,
        0,
        LayoutDescriptor::RowMajor,
        vec![buffer_handle],
        None,
        None,
        None,
        None,
    )
    .map_err(|e| InvalidDescriptorError::new_err(format!("invalid tensor descriptor: {e}")))?;

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

    Ok(Tensor {
        descriptor,
        buffer: BufferStore::Owned(data.into_boxed_slice()),
        aux_buffers: Vec::new(),
        dtype_py,
        device_py,
    })
}

/// Parse a Python `list[int]` (or tuple) into a Hurray [`Shape`].
fn parse_shape(shape: Vec<i64>) -> PyResult<Shape> {
    let dims: Vec<u64> = shape
        .iter()
        .map(|&d| {
            if d < 0 {
                Err(InvalidDescriptorError::new_err(format!(
                    "shape dimensions must be non-negative, got {d}"
                )))
            } else {
                Ok(d as u64)
            }
        })
        .collect::<PyResult<_>>()?;
    Shape::new(dims).map_err(|e| InvalidDescriptorError::new_err(format!("invalid shape: {e}")))
}

// ── Buffer fill helpers ────────────────────────────────────────────────────────

/// Byte pattern for the value 1 for each non-Bool, non-packed Tier 1 type.
///
/// Bool is excluded — it uses bit packing and is handled separately.
fn one_bytes(element_type: ElementType) -> &'static [u8] {
    use ElementType::*;
    match element_type {
        Int8 | Uint8 => &[0x01],
        Int16 | Uint16 => &[0x01, 0x00],
        Int32 | Uint32 => &[0x01, 0x00, 0x00, 0x00],
        Int64 | Uint64 => &[0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        Float16 => &[0x00, 0x3c],             // f16(1.0) LE
        BFloat16 => &[0x80, 0x3f],            // bf16(1.0) LE
        Float32 => &[0x00, 0x00, 0x80, 0x3f], // f32(1.0) LE
        Float64 => &[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f], // f64(1.0) LE
        Complex64 => &[0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00], // (1+0i) f32 LE
        Complex128 => &[
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, // real 1.0 f64 LE
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // imag 0.0 f64 LE
        ],
        // Tier 2 types are rejected by check_tier1 before we get here.
        _ => unreachable!("one_bytes called with non-Tier-1 or Bool type"),
    }
}

/// Build an all-ones buffer for `n_elements` of `element_type`.
///
/// For Bool: sets all bits to 1 in full bytes, then masks the partial last byte
/// so padding bits are 0 (LSB-first per spec).
fn fill_ones(element_type: ElementType, n_elements: u64) -> Vec<u8> {
    let buf_size = buffer_size_bytes(element_type, n_elements) as usize;
    if element_type == ElementType::Bool {
        let mut buf = vec![0u8; buf_size];
        let full_bytes = (n_elements / 8) as usize;
        for b in &mut buf[..full_bytes] {
            *b = 0xFF;
        }
        let tail = (n_elements % 8) as u8;
        if tail > 0 {
            // Set the lowest `tail` bits to 1; upper padding bits remain 0.
            buf[full_bytes] = (1u8 << tail) - 1;
        }
        return buf;
    }
    let pattern = one_bytes(element_type);
    let mut buf = vec![0u8; buf_size];
    for chunk in buf.chunks_exact_mut(pattern.len()) {
        chunk.copy_from_slice(pattern);
    }
    buf
}

/// Convert a Python scalar to the little-endian bytes for `element_type`.
///
/// Called by `fill_scalar` for non-Bool types.
fn scalar_to_bytes(element_type: ElementType, value: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    use ElementType::*;
    match element_type {
        Int8 => Ok(value.extract::<i8>()?.to_le_bytes().to_vec()),
        Int16 => Ok(value.extract::<i16>()?.to_le_bytes().to_vec()),
        Int32 => Ok(value.extract::<i32>()?.to_le_bytes().to_vec()),
        Int64 => Ok(value.extract::<i64>()?.to_le_bytes().to_vec()),
        Uint8 => Ok(value.extract::<u8>()?.to_le_bytes().to_vec()),
        Uint16 => Ok(value.extract::<u16>()?.to_le_bytes().to_vec()),
        Uint32 => Ok(value.extract::<u32>()?.to_le_bytes().to_vec()),
        Uint64 => Ok(value.extract::<u64>()?.to_le_bytes().to_vec()),
        Float16 => {
            let v: f64 = value.extract()?;
            Ok(half::f16::from_f64(v).to_le_bytes().to_vec())
        }
        BFloat16 => {
            let v: f64 = value.extract()?;
            Ok(half::bf16::from_f64(v).to_le_bytes().to_vec())
        }
        Float32 => Ok(value.extract::<f32>()?.to_le_bytes().to_vec()),
        Float64 => Ok(value.extract::<f64>()?.to_le_bytes().to_vec()),
        Complex64 => {
            let real: f64 = value.getattr("real")?.extract()?;
            let imag: f64 = value.getattr("imag")?.extract()?;
            let mut bytes = Vec::with_capacity(8);
            bytes.extend_from_slice(&(real as f32).to_le_bytes());
            bytes.extend_from_slice(&(imag as f32).to_le_bytes());
            Ok(bytes)
        }
        Complex128 => {
            let real: f64 = value.getattr("real")?.extract()?;
            let imag: f64 = value.getattr("imag")?.extract()?;
            let mut bytes = Vec::with_capacity(16);
            bytes.extend_from_slice(&real.to_le_bytes());
            bytes.extend_from_slice(&imag.to_le_bytes());
            Ok(bytes)
        }
        // Bool is handled separately in fill_scalar; Tier 2 rejected earlier.
        _ => unreachable!("scalar_to_bytes called with Bool or Tier 2 type"),
    }
}

/// Fill a buffer with `value` repeated `n_elements` times.
///
/// Bool uses bit-level packing (LSB-first per spec, padding bits = 0).
/// Other types tile `scalar_to_bytes` across the buffer.
fn fill_scalar(
    element_type: ElementType,
    value: &Bound<'_, PyAny>,
    n_elements: u64,
) -> PyResult<Vec<u8>> {
    let buf_size = buffer_size_bytes(element_type, n_elements) as usize;

    if element_type == ElementType::Bool {
        let v: bool = value.extract()?;
        if v {
            // Reuse fill_ones which handles padding bits correctly.
            return Ok(fill_ones(ElementType::Bool, n_elements));
        }
        return Ok(vec![0u8; buf_size]);
    }

    let pattern = scalar_to_bytes(element_type, value)?;
    let mut buf = vec![0u8; buf_size];
    for chunk in buf.chunks_exact_mut(pattern.len()) {
        chunk.copy_from_slice(&pattern);
    }
    Ok(buf)
}

/// Infer the default dtype from a Python scalar (used by `full` when dtype=None).
///
/// Python type → Hurray type: bool→Bool, complex→Complex128, float→Float64,
/// int→Int64. (bool must be checked before int since bool is a subclass of int.)
fn infer_scalar_dtype(value: &Bound<'_, PyAny>) -> PyResult<ElementType> {
    if value.is_instance_of::<PyBool>() {
        Ok(ElementType::Bool)
    } else if value.is_instance_of::<PyComplex>() {
        Ok(ElementType::Complex128)
    } else if value.is_instance_of::<PyFloat>() {
        Ok(ElementType::Float64)
    } else if value.is_instance_of::<PyInt>() {
        Ok(ElementType::Int64)
    } else {
        Err(InvalidDescriptorError::new_err(format!(
            "cannot infer dtype from fill_value of type '{}'",
            value.get_type().name()?
        )))
    }
}

/// Convert an `f64` value to the little-endian bytes for `element_type`.
///
/// Used for sequence generation (`arange`, `linspace`, `eye`) where values are
/// computed as `f64` then cast to the target type.
fn f64_to_bytes(v: f64, et: ElementType) -> Vec<u8> {
    use ElementType::*;
    match et {
        Bool => unreachable!("Bool sequences use dedicated packing"),
        Int8 => (v as i8).to_le_bytes().to_vec(),
        Int16 => (v as i16).to_le_bytes().to_vec(),
        Int32 => (v as i32).to_le_bytes().to_vec(),
        Int64 => (v as i64).to_le_bytes().to_vec(),
        Uint8 => (v as u8).to_le_bytes().to_vec(),
        Uint16 => (v as u16).to_le_bytes().to_vec(),
        Uint32 => (v as u32).to_le_bytes().to_vec(),
        Uint64 => (v as u64).to_le_bytes().to_vec(),
        Float16 => half::f16::from_f64(v).to_le_bytes().to_vec(),
        BFloat16 => half::bf16::from_f64(v).to_le_bytes().to_vec(),
        Float32 => (v as f32).to_le_bytes().to_vec(),
        Float64 => v.to_le_bytes().to_vec(),
        Complex64 => {
            let mut b = Vec::with_capacity(8);
            b.extend_from_slice(&(v as f32).to_le_bytes());
            b.extend_from_slice(&0.0f32.to_le_bytes());
            b
        }
        Complex128 => {
            let mut b = Vec::with_capacity(16);
            b.extend_from_slice(&v.to_le_bytes());
            b.extend_from_slice(&0.0f64.to_le_bytes());
            b
        }
        _ => unreachable!("Tier 2 types rejected before sequence generation"),
    }
}

/// Convert a Hurray `ElementType` to the corresponding NumPy dtype name, or None
/// for types that NumPy does not natively support (e.g. `bfloat16`).
pub(crate) fn to_numpy_dtype_name(et: ElementType) -> Option<&'static str> {
    use ElementType::*;
    match et {
        Bool => Some("bool"),
        Int8 => Some("int8"),
        Int16 => Some("int16"),
        Int32 => Some("int32"),
        Int64 => Some("int64"),
        Uint8 => Some("uint8"),
        Uint16 => Some("uint16"),
        Uint32 => Some("uint32"),
        Uint64 => Some("uint64"),
        Float16 => Some("float16"),
        Float32 => Some("float32"),
        Float64 => Some("float64"),
        Complex64 => Some("complex64"),
        Complex128 => Some("complex128"),
        // NumPy has no native bfloat16 — callers must handle separately.
        _ => None,
    }
}

// ── Tensor construction functions ──────────────────────────────────────────────

/// Return a new tensor of the given shape, filled with zeros.
///
/// Default dtype is `float64`. Strict mode only.
///
/// ## Examples
///
/// ```python
/// import hurray
///
/// t = hurray.zeros([3, 4])
/// assert t.shape == (3, 4)
/// assert t.dtype == hurray.float64
/// ```
#[pyfunction]
#[pyo3(signature = (shape, *, dtype = None, device = None))]
pub fn zeros(
    py: Python<'_>,
    shape: Vec<i64>,
    dtype: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<Tensor> {
    let et = resolve_dtype(dtype)?;
    check_tier1(et)?;
    let (dtag, mc) = resolve_device(device)?;
    let s = parse_shape(shape)?;
    let n = s.element_count().unwrap_or(0);
    let buf = vec![0u8; buffer_size_bytes(et, n) as usize];
    make_owned_tensor(py, buf, et, s, dtag, mc)
}

/// Return a new tensor of the given shape, filled with ones.
///
/// Default dtype is `float64`. Strict mode only.
///
/// ## Examples
///
/// ```python
/// import hurray
///
/// t = hurray.ones([2, 3], dtype=hurray.float32)
/// assert t.shape == (2, 3)
/// assert t.dtype == hurray.float32
/// ```
#[pyfunction]
#[pyo3(signature = (shape, *, dtype = None, device = None))]
pub fn ones(
    py: Python<'_>,
    shape: Vec<i64>,
    dtype: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<Tensor> {
    let et = resolve_dtype(dtype)?;
    check_tier1(et)?;
    let (dtag, mc) = resolve_device(device)?;
    let s = parse_shape(shape)?;
    let n = s.element_count().unwrap_or(0);
    let buf = fill_ones(et, n);
    make_owned_tensor(py, buf, et, s, dtag, mc)
}

/// Return a new tensor of the given shape, filled with `fill_value`.
///
/// If `dtype` is `None`, it is inferred from `fill_value` (bool→Bool,
/// complex→Complex128, float→Float64, int→Int64). Strict mode only.
///
/// ## Examples
///
/// ```python
/// import hurray
///
/// t = hurray.full([2, 2], 3.14, dtype=hurray.float32)
/// assert t.shape == (2, 2)
/// assert t.dtype == hurray.float32
/// ```
#[pyfunction]
#[pyo3(signature = (shape, fill_value, *, dtype = None, device = None))]
pub fn full(
    py: Python<'_>,
    shape: Vec<i64>,
    fill_value: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<Tensor> {
    let et = if dtype.is_some() {
        resolve_dtype(dtype)?
    } else {
        infer_scalar_dtype(fill_value)?
    };
    check_tier1(et)?;
    let (dtag, mc) = resolve_device(device)?;
    let s = parse_shape(shape)?;
    let n = s.element_count().unwrap_or(0);
    let buf = fill_scalar(et, fill_value, n)?;
    make_owned_tensor(py, buf, et, s, dtag, mc)
}

/// Return a new uninitialized tensor of the given shape and type.
///
/// The buffer is zero-initialized (the contents of an "empty" tensor are unspecified).
/// Default dtype is `float64`. Tier 1 only.
///
/// ## Examples
///
/// ```python
/// import hurray
///
/// t = hurray.empty([4, 4], dtype=hurray.float64)
/// assert t.shape == (4, 4)
/// ```
#[pyfunction]
#[pyo3(signature = (shape, *, dtype = None, device = None))]
pub fn empty(
    py: Python<'_>,
    shape: Vec<i64>,
    dtype: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<Tensor> {
    let et = resolve_dtype(dtype)?;
    check_tier1(et)?;
    let (dtag, mc) = resolve_device(device)?;
    let s = parse_shape(shape)?;
    let n = s.element_count().unwrap_or(0);
    // Zero-fill is a safe default; the contents of an "empty" tensor are unspecified.
    let buf = vec![0u8; buffer_size_bytes(et, n) as usize];
    make_owned_tensor(py, buf, et, s, dtag, mc)
}

/// Return a zeros tensor with the same shape and dtype as `x`.
///
/// `dtype` and `device` override the source tensor's values when given.
///
/// ## Examples
///
/// ```python
/// import hurray
///
/// src = hurray.ones([3, 3], dtype=hurray.float32)
/// z = hurray.zeros_like(src)
/// assert z.shape == (3, 3)
/// assert z.dtype == hurray.float32
/// ```
#[pyfunction]
#[pyo3(signature = (x, *, dtype = None, device = None))]
pub fn zeros_like(
    py: Python<'_>,
    x: &Bound<'_, Tensor>,
    dtype: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<Tensor> {
    let xr = x.borrow();
    let et = if dtype.is_some() {
        resolve_dtype(dtype)?
    } else {
        xr.descriptor.element_type
    };
    check_tier1(et)?;
    let (dtag, mc) = if device.is_some() {
        resolve_device(device)?
    } else {
        let dev = xr.device_py.borrow(py);
        (dev.tag, dev.memory_class)
    };
    let s = xr.descriptor.shape.clone();
    let n = s.element_count().unwrap_or(0);
    let buf = vec![0u8; buffer_size_bytes(et, n) as usize];
    make_owned_tensor(py, buf, et, s, dtag, mc)
}

/// Return an all-ones tensor with the same shape and dtype as `x`.
///
/// `dtype` and `device` override the source tensor's values when given.
///
/// ## Examples
///
/// ```python
/// import hurray
///
/// src = hurray.zeros([2, 2], dtype=hurray.int32)
/// o = hurray.ones_like(src)
/// assert o.shape == (2, 2)
/// assert o.dtype == hurray.int32
/// ```
#[pyfunction]
#[pyo3(signature = (x, *, dtype = None, device = None))]
pub fn ones_like(
    py: Python<'_>,
    x: &Bound<'_, Tensor>,
    dtype: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<Tensor> {
    let xr = x.borrow();
    let et = if dtype.is_some() {
        resolve_dtype(dtype)?
    } else {
        xr.descriptor.element_type
    };
    check_tier1(et)?;
    let (dtag, mc) = if device.is_some() {
        resolve_device(device)?
    } else {
        let dev = xr.device_py.borrow(py);
        (dev.tag, dev.memory_class)
    };
    let s = xr.descriptor.shape.clone();
    let n = s.element_count().unwrap_or(0);
    let buf = fill_ones(et, n);
    make_owned_tensor(py, buf, et, s, dtag, mc)
}

/// Return a tensor filled with `fill_value` with the same shape and dtype as `x`.
///
/// `dtype` and `device` override the source tensor's values when given.
///
/// ## Examples
///
/// ```python
/// import hurray
///
/// src = hurray.zeros([4], dtype=hurray.float32)
/// f = hurray.full_like(src, 7.0)
/// assert f.shape == (4,)
/// assert f.dtype == hurray.float32
/// ```
#[pyfunction]
#[pyo3(signature = (x, fill_value, *, dtype = None, device = None))]
pub fn full_like(
    py: Python<'_>,
    x: &Bound<'_, Tensor>,
    fill_value: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<Tensor> {
    let xr = x.borrow();
    let et = if dtype.is_some() {
        resolve_dtype(dtype)?
    } else {
        xr.descriptor.element_type
    };
    check_tier1(et)?;
    let (dtag, mc) = if device.is_some() {
        resolve_device(device)?
    } else {
        let dev = xr.device_py.borrow(py);
        (dev.tag, dev.memory_class)
    };
    let s = xr.descriptor.shape.clone();
    let n = s.element_count().unwrap_or(0);
    let buf = fill_scalar(et, fill_value, n)?;
    make_owned_tensor(py, buf, et, s, dtag, mc)
}

/// Return an uninitialized tensor with the same shape and dtype as `x`.
///
/// `dtype` and `device` override the source tensor's values when given.
///
/// ## Examples
///
/// ```python
/// import hurray
///
/// src = hurray.ones([3], dtype=hurray.float64)
/// e = hurray.empty_like(src)
/// assert e.shape == (3,)
/// assert e.dtype == hurray.float64
/// ```
#[pyfunction]
#[pyo3(signature = (x, *, dtype = None, device = None))]
pub fn empty_like(
    py: Python<'_>,
    x: &Bound<'_, Tensor>,
    dtype: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<Tensor> {
    let xr = x.borrow();
    let et = if dtype.is_some() {
        resolve_dtype(dtype)?
    } else {
        xr.descriptor.element_type
    };
    check_tier1(et)?;
    let (dtag, mc) = if device.is_some() {
        resolve_device(device)?
    } else {
        let dev = xr.device_py.borrow(py);
        (dev.tag, dev.memory_class)
    };
    let s = xr.descriptor.shape.clone();
    let n = s.element_count().unwrap_or(0);
    let buf = vec![0u8; buffer_size_bytes(et, n) as usize];
    make_owned_tensor(py, buf, et, s, dtag, mc)
}

/// Return evenly spaced values over a given interval.
///
/// `arange(stop)` → values `[0, 1, …, stop)`.
/// `arange(start, stop, step)` → values `[start, start+step, …)`.
///
/// Dtype is inferred as `int64` if all of `start`, `stop`, `step` are Python
/// integers; `float64` otherwise. Pass an explicit `dtype` to override.
///
/// ## Examples
///
/// ```python
/// import hurray
///
/// t = hurray.arange(5)
/// assert t.shape == (5,)
/// assert t.dtype == hurray.int64
///
/// t2 = hurray.arange(0.0, 1.0, 0.25)
/// assert t2.shape == (4,)
/// assert t2.dtype == hurray.float64
/// ```
#[pyfunction]
#[pyo3(signature = (start, stop = None, step = None, *, dtype = None, device = None))]
pub fn arange(
    py: Python<'_>,
    start: &Bound<'_, PyAny>,
    stop: Option<&Bound<'_, PyAny>>,
    step: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<Tensor> {
    // Resolve effective start / stop / step.
    let (eff_start, eff_stop): (f64, f64) = match stop {
        None => (0.0, start.extract()?),
        Some(s) => (start.extract()?, s.extract()?),
    };
    let eff_step: f64 = step.map(|s| s.extract()).transpose()?.unwrap_or(1.0);

    if eff_step == 0.0 {
        return Err(InvalidDescriptorError::new_err(
            "arange: step must not be zero",
        ));
    }

    // Dtype inference: if all inputs are Python ints → int64; else → float64.
    let is_float_arg = |v: &Bound<'_, PyAny>| v.is_instance_of::<PyFloat>();
    let any_float = is_float_arg(start)
        || stop.map(is_float_arg).unwrap_or(false)
        || step.map(is_float_arg).unwrap_or(false);
    let et = if dtype.is_some() {
        resolve_dtype(dtype)?
    } else if any_float {
        ElementType::Float64
    } else {
        ElementType::Int64
    };
    check_tier1(et)?;

    // Count = max(0, ceil((stop - start) / step)).
    let raw_count = ((eff_stop - eff_start) / eff_step).ceil();
    let n = if raw_count > 0.0 { raw_count as u64 } else { 0 };

    let (dtag, mc) = resolve_device(device)?;

    // Bool sequences don't have sensible semantics — reject.
    if et == ElementType::Bool {
        return Err(UnsupportedError::new_err(
            "arange does not support dtype=bool; use zeros/ones instead",
        ));
    }

    let elem_bytes = buffer_size_bytes(et, 1) as usize;
    let mut buf = Vec::with_capacity(n as usize * elem_bytes);
    for i in 0..n {
        let v = eff_start + i as f64 * eff_step;
        buf.extend_from_slice(&f64_to_bytes(v, et));
    }

    let s = Shape::new(vec![n])
        .map_err(|e| InvalidDescriptorError::new_err(format!("arange shape error: {e}")))?;
    make_owned_tensor(py, buf, et, s, dtag, mc)
}

/// Return `num` evenly spaced values in the closed interval `[start, stop]`.
///
/// Default dtype is `float64`. Pass `endpoint=False` to exclude `stop`.
/// Strict mode only.
///
/// ## Examples
///
/// ```python
/// import hurray
///
/// t = hurray.linspace(0.0, 1.0, 5)
/// assert t.shape == (5,)
/// assert t.dtype == hurray.float64
/// ```
#[pyfunction]
#[pyo3(signature = (start, stop, num, *, dtype = None, device = None, endpoint = true))]
pub fn linspace(
    py: Python<'_>,
    start: f64,
    stop: f64,
    num: i64,
    dtype: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
    endpoint: bool,
) -> PyResult<Tensor> {
    if num < 0 {
        return Err(InvalidDescriptorError::new_err(format!(
            "linspace: num must be non-negative, got {num}"
        )));
    }
    let n = num as u64;
    let et = resolve_dtype(dtype)?;
    check_tier1(et)?;
    if et == ElementType::Bool {
        return Err(UnsupportedError::new_err(
            "linspace does not support dtype=bool",
        ));
    }
    let (dtag, mc) = resolve_device(device)?;

    let elem_bytes = buffer_size_bytes(et, 1) as usize;
    let mut buf = Vec::with_capacity(n as usize * elem_bytes);

    if n == 0 {
        // empty tensor
    } else if n == 1 {
        buf.extend_from_slice(&f64_to_bytes(start, et));
    } else {
        let divisor = if endpoint { (n - 1) as f64 } else { n as f64 };
        let step = (stop - start) / divisor;
        for i in 0..n {
            // Use start + i*step to minimise accumulated floating-point error.
            let v = start + i as f64 * step;
            buf.extend_from_slice(&f64_to_bytes(v, et));
        }
        // Clamp the last element to exactly `stop` when endpoint=true.
        if endpoint {
            let last_start = buf.len() - elem_bytes;
            buf.truncate(last_start);
            buf.extend_from_slice(&f64_to_bytes(stop, et));
        }
    }

    let s = Shape::new(vec![n])
        .map_err(|e| InvalidDescriptorError::new_err(format!("linspace shape error: {e}")))?;
    make_owned_tensor(py, buf, et, s, dtag, mc)
}

/// Return a 2-D tensor with ones on the k-th diagonal and zeros elsewhere.
///
/// Default dtype is `float64`. Strict mode only.
///
/// ## Examples
///
/// ```python
/// import hurray
///
/// t = hurray.eye(3)
/// assert t.shape == (3, 3)
/// assert t.dtype == hurray.float64
/// ```
#[pyfunction]
#[pyo3(signature = (n_rows, n_cols = None, *, k = 0, dtype = None, device = None))]
pub fn eye(
    py: Python<'_>,
    n_rows: i64,
    n_cols: Option<i64>,
    k: i64,
    dtype: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<Tensor> {
    if n_rows < 0 {
        return Err(InvalidDescriptorError::new_err(format!(
            "eye: n_rows must be non-negative, got {n_rows}"
        )));
    }
    let rows = n_rows as u64;
    let cols = match n_cols {
        None => rows,
        Some(c) if c >= 0 => c as u64,
        Some(c) => {
            return Err(InvalidDescriptorError::new_err(format!(
                "eye: n_cols must be non-negative, got {c}"
            )))
        }
    };

    let et = resolve_dtype(dtype)?;
    check_tier1(et)?;
    let (dtag, mc) = resolve_device(device)?;

    let s = Shape::new(vec![rows, cols])
        .map_err(|e| InvalidDescriptorError::new_err(format!("eye shape error: {e}")))?;

    let buf = if et == ElementType::Bool {
        fill_eye_bool(rows, cols, k)
    } else {
        fill_eye(rows, cols, k, et)
    };

    make_owned_tensor(py, buf, et, s, dtag, mc)
}

/// Fill an eye buffer for non-Bool types.
fn fill_eye(n_rows: u64, n_cols: u64, k: i64, et: ElementType) -> Vec<u8> {
    let n_elements = n_rows * n_cols;
    let elem_bytes = buffer_size_bytes(et, 1) as usize;
    let mut buf = vec![0u8; n_elements as usize * elem_bytes];
    let one = one_bytes(et);

    for i in 0..n_rows {
        let j_raw = i as i64 + k;
        if j_raw >= 0 && (j_raw as u64) < n_cols {
            let j = j_raw as u64;
            let byte_offset = (i * n_cols + j) as usize * elem_bytes;
            buf[byte_offset..byte_offset + elem_bytes].copy_from_slice(one);
        }
    }
    buf
}

/// Fill an eye buffer for Bool type (LSB-first bit packing, padding bits = 0).
fn fill_eye_bool(n_rows: u64, n_cols: u64, k: i64) -> Vec<u8> {
    let n_elements = n_rows * n_cols;
    let buf_size = buffer_size_bytes(ElementType::Bool, n_elements) as usize;
    let mut buf = vec![0u8; buf_size];

    for i in 0..n_rows {
        let j_raw = i as i64 + k;
        if j_raw >= 0 && (j_raw as u64) < n_cols {
            let j = j_raw as u64;
            let bit_idx = i * n_cols + j;
            let byte_idx = (bit_idx / 8) as usize;
            let bit_pos = (bit_idx % 8) as u8;
            buf[byte_idx] |= 1u8 << bit_pos; // LSB-first
        }
    }
    buf
}

/// Convert an object to a `hurray.Tensor`.
///
/// Input types handled:
/// - `hurray.Tensor` — returned as-is (or copied if `copy=True`).
/// - NumPy arrays — wrapped zero-copy via `from_numpy`.
/// - Objects with `__dlpack__` — wrapped zero-copy via DLPack.
/// - Python lists, tuples, scalars — converted via `numpy.asarray` first.
///
/// ## Errors
///
/// - `hurray.UnsupportedError` — `copy=False` and a copy would be required.
/// - `hurray.UnsupportedError` — `bfloat16` dtype requested (NumPy does not
///   support bfloat16 natively; use `from_numpy` on a pre-converted array).
///
/// ## Examples
///
/// ```python
/// import numpy as np, hurray
///
/// t = hurray.asarray([1.0, 2.0, 3.0])
/// assert t.shape == (3,)
/// assert t.dtype == hurray.float64
/// ```
#[pyfunction]
#[pyo3(signature = (obj, *, dtype = None, device = None, copy = None))]
pub fn asarray(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
    copy: Option<bool>,
) -> PyResult<Tensor> {
    let _ = device; // accepted for signature compatibility; only CPU is supported in this version
    let target_et = dtype.map(|d| resolve_dtype(Some(d))).transpose()?;
    if let Some(et) = target_et {
        check_tier1(et)?;
        if et == ElementType::BFloat16 {
            return Err(UnsupportedError::new_err(
                "asarray does not support bfloat16: NumPy has no native bfloat16 dtype. \
                 Use hurray.from_numpy on a pre-converted array, or hurray.from_torch.",
            ));
        }
    }

    // If obj is already a hurray.Tensor and no dtype conversion is needed, use
    // DLPack for zero-copy buffer sharing (creates a new Python wrapper but shares data).
    if let Ok(t) = obj.extract::<PyRef<Tensor>>() {
        let src_et = t.descriptor.element_type;
        let needs_dtype_conv = target_et.map(|et| et != src_et).unwrap_or(false);
        drop(t); // Release borrow before calling into obj.

        if !needs_dtype_conv && copy != Some(true) {
            let capsule = obj.call_method0("__dlpack__")?;
            return crate::interop::from_dlpack_capsule(py, capsule.unbind());
        }
        // Fall through to numpy for dtype conversion or explicit copy.
    }

    // Route through numpy: covers lists, numpy arrays, objects with __array__,
    // and objects with __dlpack__ (numpy ≥ 1.22 calls __dlpack__ transparently).
    let np = py.import("numpy")?;

    let arr = if let Some(et) = target_et {
        let np_dtype_name = to_numpy_dtype_name(et).ok_or_else(|| {
            UnsupportedError::new_err(format!(
                "dtype '{}' has no NumPy equivalent; cannot use asarray",
                crate::dtype::element_type_name(et)
            ))
        })?;
        np.call_method1("asarray", (obj, np_dtype_name))?
    } else {
        np.call_method1("asarray", (obj,))?
    };

    crate::interop::from_numpy(py, &arr)
}

/// Construct a tensor from a DLPack capsule or an object with `__dlpack__`.
///
/// The `from_dlpack` entry point. It calls `x.__dlpack__()` to obtain a DLPack v1.0
/// capsule and wraps the result as a `hurray.Tensor` (zero-copy on CPU). The `device`
/// and `copy` parameters are accepted for signature compatibility but only CPU is
/// supported in this version.
///
/// ## Errors
///
/// - `ImportError` — NumPy is not installed.
/// - `hurray.UnsupportedError` — source tensor is not on CPU.
///
/// ## Examples
///
/// ```python
/// import numpy as np, hurray
///
/// arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
/// t = hurray.from_dlpack(arr)
/// assert t.shape == (3,)
/// assert t.dtype == hurray.float64
/// ```
#[pyfunction]
#[pyo3(signature = (x, *, device = None, copy = None))]
#[allow(unused_variables)]
pub fn from_dlpack(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    // Accepted for signature compatibility; only CPU zero-copy is supported in this version.
    device: Option<&Bound<'_, PyAny>>,
    copy: Option<bool>,
) -> PyResult<Tensor> {
    let capsule = x.call_method0("__dlpack__")?;
    crate::interop::from_dlpack_capsule(py, capsule.unbind())
}

// ── Registration ──────────────────────────────────────────────────────────────

/// Register all Array API creation functions on the `hurray` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(full, m)?)?;
    m.add_function(wrap_pyfunction!(empty, m)?)?;
    m.add_function(wrap_pyfunction!(zeros_like, m)?)?;
    m.add_function(wrap_pyfunction!(ones_like, m)?)?;
    m.add_function(wrap_pyfunction!(full_like, m)?)?;
    m.add_function(wrap_pyfunction!(empty_like, m)?)?;
    m.add_function(wrap_pyfunction!(arange, m)?)?;
    m.add_function(wrap_pyfunction!(linspace, m)?)?;
    m.add_function(wrap_pyfunction!(eye, m)?)?;
    m.add_function(wrap_pyfunction!(asarray, m)?)?;
    m.add_function(wrap_pyfunction!(from_dlpack, m)?)?;
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;

    fn init() {
        pyo3::Python::initialize();
    }

    fn build_module(py: Python<'_>) -> pyo3::Bound<'_, pyo3::types::PyModule> {
        let m = pyo3::types::PyModule::new(py, "hurray").unwrap();
        crate::errors::register(&m).unwrap();
        crate::dtype::register(&m).unwrap();
        crate::device::register(&m).unwrap();
        crate::tensor::register(&m).unwrap();
        register(&m).unwrap();
        let sys = py.import("sys").unwrap();
        let modules = sys.getattr("modules").unwrap();
        modules.set_item("hurray", &m).unwrap();
        m
    }

    #[test]
    fn zeros_float64_default_dtype() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let t = zeros(py, vec![2, 3], None, None).unwrap();
            assert_eq!(t.descriptor.element_type, ElementType::Float64);
            assert_eq!(t.descriptor.shape.dims(), &[2, 3]);
            let buf_size = buffer_size_bytes(ElementType::Float64, 6) as usize;
            assert_eq!(t.buffer.len(), buf_size);
            // All bytes should be zero.
            let slice = unsafe { t.buffer.as_slice() };
            assert!(slice.iter().all(|&b| b == 0));
        });
    }

    #[test]
    fn ones_float32_pattern() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            // Build dtype object.
            let dtype_obj = Py::new(
                py,
                Dtype {
                    inner: ElementType::Float32,
                },
            )
            .unwrap();
            let t = ones(py, vec![2], Some(dtype_obj.bind(py).as_any()), None).unwrap();
            // Each 4-byte element should be [0x00, 0x00, 0x80, 0x3f] = 1.0f32 LE.
            let slice = unsafe { t.buffer.as_slice() };
            assert_eq!(&slice[0..4], &[0x00, 0x00, 0x80, 0x3f]);
            assert_eq!(&slice[4..8], &[0x00, 0x00, 0x80, 0x3f]);
        });
    }

    #[test]
    fn ones_bool_bit_packing() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let dtype_obj = Py::new(
                py,
                Dtype {
                    inner: ElementType::Bool,
                },
            )
            .unwrap();
            // 9 True elements → 2 bytes: first byte 0xFF, second byte 0x01.
            let t = ones(py, vec![9], Some(dtype_obj.bind(py).as_any()), None).unwrap();
            let slice = unsafe { t.buffer.as_slice() };
            assert_eq!(slice.len(), 2);
            assert_eq!(slice[0], 0xFF); // 8 True bits
            assert_eq!(slice[1], 0x01); // 1 True bit, 7 zero padding bits
        });
    }

    #[test]
    fn full_int32_scalar() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let fill_val_py = (42i32).into_py_any(py).unwrap();
            let fill_val = fill_val_py.bind(py);
            let dtype_obj = Py::new(
                py,
                Dtype {
                    inner: ElementType::Int32,
                },
            )
            .unwrap();
            let t = full(
                py,
                vec![3],
                fill_val,
                Some(dtype_obj.bind(py).as_any()),
                None,
            )
            .unwrap();
            let slice = unsafe { t.buffer.as_slice() };
            // 42i32 LE = [0x2a, 0x00, 0x00, 0x00]
            assert_eq!(&slice[0..4], &[0x2a, 0x00, 0x00, 0x00]);
        });
    }

    #[test]
    fn arange_int() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let start_py = (0i64).into_py_any(py).unwrap();
            let start = start_py.bind(py);
            let stop_py = (5i64).into_py_any(py).unwrap();
            let stop = stop_py.bind(py);
            let t = arange(py, start, Some(stop), None, None, None).unwrap();
            assert_eq!(t.descriptor.element_type, ElementType::Int64);
            assert_eq!(t.descriptor.shape.dims(), &[5]);
            let slice = unsafe { t.buffer.as_slice() };
            // Element 0: 0i64 LE
            assert_eq!(&slice[0..8], &[0, 0, 0, 0, 0, 0, 0, 0]);
            // Element 4: 4i64 LE
            assert_eq!(&slice[32..40], &[4, 0, 0, 0, 0, 0, 0, 0]);
        });
    }

    #[test]
    fn arange_single_arg() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let stop_py = (3i64).into_py_any(py).unwrap();
            let stop = stop_py.bind(py);
            let t = arange(py, stop, None, None, None, None).unwrap();
            assert_eq!(t.descriptor.shape.dims(), &[3]);
        });
    }

    #[test]
    fn linspace_five_points() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let t = linspace(py, 0.0, 1.0, 5, None, None, true).unwrap();
            assert_eq!(t.descriptor.element_type, ElementType::Float64);
            assert_eq!(t.descriptor.shape.dims(), &[5]);
            let slice = unsafe { t.buffer.as_slice() };
            // First element: 0.0 f64 LE.
            assert_eq!(&slice[0..8], &0.0f64.to_le_bytes());
            // Last element: 1.0 f64 LE.
            assert_eq!(&slice[32..40], &1.0f64.to_le_bytes());
        });
    }

    #[test]
    fn linspace_no_endpoint() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            // endpoint=False: 4 elements in [0, 1), step = 0.25.
            let t = linspace(py, 0.0, 1.0, 4, None, None, false).unwrap();
            assert_eq!(t.descriptor.shape.dims(), &[4]);
            let slice = unsafe { t.buffer.as_slice() };
            // Last element: 0.75 f64 LE.
            let last: f64 = f64::from_le_bytes(slice[24..32].try_into().unwrap());
            assert!((last - 0.75).abs() < 1e-10);
        });
    }

    #[test]
    fn eye_3x3_float64() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let t = eye(py, 3, None, 0, None, None).unwrap();
            assert_eq!(t.descriptor.shape.dims(), &[3, 3]);
            let slice = unsafe { t.buffer.as_slice() };
            // Element [0,0]: 1.0 f64; [0,1]: 0.0.
            assert_eq!(&slice[0..8], &1.0f64.to_le_bytes());
            assert_eq!(&slice[8..16], &0.0f64.to_le_bytes());
            // Element [1,1]: flat index 4 → byte offset 32.
            assert_eq!(&slice[32..40], &1.0f64.to_le_bytes());
        });
    }

    #[test]
    fn eye_bool_diagonal() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let dtype_obj = Py::new(
                py,
                Dtype {
                    inner: ElementType::Bool,
                },
            )
            .unwrap();
            // 3×3 Bool eye. 9 elements → 2 bytes.
            // Diagonal at positions 0, 4, 8 (bit indices).
            // byte 0: bits 0 and 4 set → 0b00010001 = 0x11.
            // byte 1: bit 0 set (element index 8 → bit 8 → byte 1, bit 0) → 0x01.
            let t = eye(py, 3, None, 0, Some(dtype_obj.bind(py).as_any()), None).unwrap();
            let slice = unsafe { t.buffer.as_slice() };
            assert_eq!(slice.len(), 2);
            assert_eq!(slice[0], 0x11);
            assert_eq!(slice[1], 0x01);
        });
    }

    #[test]
    fn tier2_dtype_rejected() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            let dtype_obj = Py::new(
                py,
                Dtype {
                    inner: ElementType::Int4,
                },
            )
            .unwrap();
            let result = zeros(py, vec![4], Some(dtype_obj.bind(py).as_any()), None);
            assert!(result.is_err());
            assert!(result.unwrap_err().is_instance_of::<UnsupportedError>(py));
        });
    }

    #[test]
    fn zeros_empty_shape() {
        init();
        Python::attach(|py| {
            let _m = build_module(py);
            // Shape [0] → 0 elements → empty buffer is valid.
            let t = zeros(py, vec![0], None, None).unwrap();
            assert_eq!(t.buffer.len(), 0);
        });
    }
}
