//! Python bindings for the five quantization schemes.
//!
//! These classes are thin wrappers over `hurray_core`'s quantization descriptors:
//! they carry the same fields and delegate every validation rule to core, so the
//! Python surface cannot drift from the wire format.
//!
//! ## Buffer indices, not buffers
//!
//! Every scheme except per-tensor affine references its scale (and optionally
//! zero-point) parameters by **buffer index** into the tensor's buffer table —
//! exactly as `docs/spec/quantization.md` defines them. The bytes themselves are
//! supplied separately, via `hurray.Tensor(..., aux_buffers=[...])`. That keeps the
//! Python classes faithful to what is on the wire, and is what lets a descriptor
//! read off disk be reconstructed exactly.
//!
//! An index that resolves to no buffer is caught when the tensor is built, not
//! later at the consumer — see `Tensor::new`.
//!
//! ## Symmetric vs asymmetric
//!
//! Schemes with an optional zero point expose two constructors rather than a flag.
//! Core encodes the symmetric case as the wire sentinel `0xFFFFFFFF` and models it
//! as `None`; separate constructors keep "asymmetric without a zero-point buffer"
//! unrepresentable instead of merely invalid.

use pyo3::prelude::*;
use pyo3::types::PyModule;

use hurray_core::{
    Mxfp as CoreMxfp, Nf4 as CoreNf4, PerBlockAffine as CorePerBlockAffine,
    PerChannelAffine as CorePerChannelAffine, PerTensorAffine as CorePerTensorAffine,
    QuantizationDescriptor,
};

use crate::dtype::Dtype;
use crate::errors::InvalidDescriptorError;

/// Map a core quantization error to the Python exception type.
fn quant_err(e: hurray_core::Error) -> PyErr {
    InvalidDescriptorError::new_err(e.to_string())
}

// ── PerTensorAffine ───────────────────────────────────────────────────────────

/// Per-tensor affine quantization: one scale and zero point for the whole tensor.
///
/// The only scheme whose parameters are **inline** in the descriptor — it needs no
/// scale buffer, which is why a per-tensor-affine tensor is still single-buffer.
///
/// `value = scale * (quantized - zero_point)`
///
/// ## Errors
///
/// - `hurray.InvalidDescriptorError` — `scale` is zero, NaN, or infinite.
///
/// ## Examples (Python)
///
/// ```python
/// import hurray
///
/// q = hurray.PerTensorAffine(0.02, 128)
/// assert q.scale == 0.02
/// assert q.zero_point == 128
/// ```
#[pyclass(name = "PerTensorAffine", frozen)]
#[derive(Debug)]
pub struct PerTensorAffine {
    pub(crate) inner: CorePerTensorAffine,
}

#[pymethods]
impl PerTensorAffine {
    /// Construct a per-tensor affine descriptor from an inline scale and zero point.
    ///
    /// ## Examples
    ///
    /// ```python
    /// q = hurray.PerTensorAffine(0.02, 0)   # symmetric: zero_point 0
    /// ```
    #[new]
    pub fn new(scale: f32, zero_point: i32) -> PyResult<Self> {
        Ok(Self {
            inner: CorePerTensorAffine::new(scale, zero_point).map_err(quant_err)?,
        })
    }

    /// The scale factor.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.PerTensorAffine(0.5, 0).scale == 0.5
    /// ```
    #[getter]
    pub fn scale(&self) -> f32 {
        self.inner.scale()
    }

    /// The zero point.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.PerTensorAffine(0.5, 7).zero_point == 7
    /// ```
    #[getter]
    pub fn zero_point(&self) -> i32 {
        self.inner.zero_point()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "PerTensorAffine(scale={}, zero_point={})",
            self.inner.scale(),
            self.inner.zero_point()
        )
    }
}

// ── PerChannelAffine ──────────────────────────────────────────────────────────

/// Per-channel affine quantization: one scale per slice along `axis`.
///
/// Scales live in their own buffer, so a per-channel-quantized tensor has at least
/// two buffers.
///
/// ## Examples (Python)
///
/// ```python
/// import hurray
///
/// q = hurray.PerChannelAffine.symmetric(axis=0, scale_buffer_index=1)
/// assert q.axis == 0
/// assert q.scale_buffer_index == 1
/// assert q.zero_point_buffer_index is None    # symmetric
/// ```
#[pyclass(name = "PerChannelAffine", frozen)]
#[derive(Debug)]
pub struct PerChannelAffine {
    pub(crate) inner: CorePerChannelAffine,
}

#[pymethods]
impl PerChannelAffine {
    /// Symmetric per-channel quantization: scales only, no zero points.
    ///
    /// ## Errors
    ///
    /// - `hurray.InvalidDescriptorError` — `axis` exceeds the rank cap.
    ///
    /// ## Examples
    ///
    /// ```python
    /// q = hurray.PerChannelAffine.symmetric(axis=0, scale_buffer_index=1)
    /// ```
    #[staticmethod]
    #[pyo3(signature = (axis, scale_buffer_index))]
    pub fn symmetric(axis: u32, scale_buffer_index: u32) -> PyResult<Self> {
        Ok(Self {
            inner: CorePerChannelAffine::new_symmetric(axis, scale_buffer_index)
                .map_err(quant_err)?,
        })
    }

    /// Asymmetric per-channel quantization: scales and per-channel zero points.
    ///
    /// ## Errors
    ///
    /// - `hurray.InvalidDescriptorError` — `axis` exceeds the rank cap, or the two
    ///   buffer indices are equal.
    ///
    /// ## Examples
    ///
    /// ```python
    /// q = hurray.PerChannelAffine.asymmetric(
    ///     axis=0, scale_buffer_index=1, zero_point_buffer_index=2
    /// )
    /// assert q.zero_point_buffer_index == 2
    /// ```
    #[staticmethod]
    #[pyo3(signature = (axis, scale_buffer_index, zero_point_buffer_index))]
    pub fn asymmetric(
        axis: u32,
        scale_buffer_index: u32,
        zero_point_buffer_index: u32,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CorePerChannelAffine::new_asymmetric(
                axis,
                scale_buffer_index,
                zero_point_buffer_index,
            )
            .map_err(quant_err)?,
        })
    }

    /// The quantized axis.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.PerChannelAffine.symmetric(1, 1).axis == 1
    /// ```
    #[getter]
    pub fn axis(&self) -> u32 {
        self.inner.axis()
    }

    /// Buffer index holding the per-channel scales.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.PerChannelAffine.symmetric(0, 1).scale_buffer_index == 1
    /// ```
    #[getter]
    pub fn scale_buffer_index(&self) -> u32 {
        self.inner.scale_buffer_index()
    }

    /// Buffer index holding the per-channel zero points, or `None` when symmetric.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.PerChannelAffine.symmetric(0, 1).zero_point_buffer_index is None
    /// ```
    #[getter]
    pub fn zero_point_buffer_index(&self) -> Option<u32> {
        self.inner.zero_point_buffer_index()
    }

    pub fn __repr__(&self) -> String {
        match self.inner.zero_point_buffer_index() {
            Some(zp) => format!(
                "PerChannelAffine.asymmetric(axis={}, scale_buffer_index={}, \
                 zero_point_buffer_index={zp})",
                self.inner.axis(),
                self.inner.scale_buffer_index()
            ),
            None => format!(
                "PerChannelAffine.symmetric(axis={}, scale_buffer_index={})",
                self.inner.axis(),
                self.inner.scale_buffer_index()
            ),
        }
    }
}

// ── PerBlockAffine ────────────────────────────────────────────────────────────

/// Per-block affine quantization: one scale per contiguous block of `block_size`
/// elements along `axis`.
///
/// ## Examples (Python)
///
/// ```python
/// import hurray
///
/// q = hurray.PerBlockAffine.symmetric(1, 64, 1, hurray.float32)
/// assert q.block_size == 64
/// ```
#[pyclass(name = "PerBlockAffine", frozen)]
#[derive(Debug)]
pub struct PerBlockAffine {
    pub(crate) inner: CorePerBlockAffine,
}

#[pymethods]
impl PerBlockAffine {
    /// Symmetric per-block quantization: scales only.
    ///
    /// ## Errors
    ///
    /// - `hurray.InvalidDescriptorError` — `axis` exceeds the rank cap, or
    ///   `block_size` is zero.
    ///
    /// ## Examples
    ///
    /// ```python
    /// q = hurray.PerBlockAffine.symmetric(1, 32, 1, hurray.float32)
    /// ```
    #[staticmethod]
    #[pyo3(signature = (axis, block_size, scale_buffer_index, scale_type))]
    pub fn symmetric(
        axis: u32,
        block_size: u32,
        scale_buffer_index: u32,
        scale_type: &Bound<'_, Dtype>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CorePerBlockAffine::new_symmetric(
                axis,
                block_size,
                scale_buffer_index,
                scale_type.get().inner,
            )
            .map_err(quant_err)?,
        })
    }

    /// Asymmetric per-block quantization: scales and per-block zero points.
    ///
    /// ## Errors
    ///
    /// - `hurray.InvalidDescriptorError` — `axis` exceeds the rank cap, `block_size`
    ///   is zero, or the two buffer indices are equal.
    ///
    /// ## Examples
    ///
    /// ```python
    /// q = hurray.PerBlockAffine.asymmetric(1, 32, 1, 2, hurray.float32)
    /// ```
    #[staticmethod]
    #[pyo3(signature = (axis, block_size, scale_buffer_index, zero_point_buffer_index, scale_type))]
    pub fn asymmetric(
        axis: u32,
        block_size: u32,
        scale_buffer_index: u32,
        zero_point_buffer_index: u32,
        scale_type: &Bound<'_, Dtype>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CorePerBlockAffine::new_asymmetric(
                axis,
                block_size,
                scale_buffer_index,
                zero_point_buffer_index,
                scale_type.get().inner,
            )
            .map_err(quant_err)?,
        })
    }

    /// The quantized axis.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.PerBlockAffine.symmetric(1, 32, 1, hurray.float32).axis == 1
    /// ```
    #[getter]
    pub fn axis(&self) -> u32 {
        self.inner.axis()
    }

    /// Elements per block along `axis`.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.PerBlockAffine.symmetric(1, 32, 1, hurray.float32).block_size == 32
    /// ```
    #[getter]
    pub fn block_size(&self) -> u32 {
        self.inner.block_size()
    }

    /// Buffer index holding the per-block scales.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.PerBlockAffine.symmetric(1, 32, 1, hurray.float32).scale_buffer_index == 1
    /// ```
    #[getter]
    pub fn scale_buffer_index(&self) -> u32 {
        self.inner.scale_buffer_index()
    }

    /// Buffer index holding the per-block zero points, or `None` when symmetric.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.PerBlockAffine.symmetric(1, 32, 1, hurray.float32).zero_point_buffer_index is None
    /// ```
    #[getter]
    pub fn zero_point_buffer_index(&self) -> Option<u32> {
        self.inner.zero_point_buffer_index()
    }

    /// Element type of the scale values: float16, bfloat16, or float32.
    ///
    /// ## Examples
    ///
    /// ```python
    /// q = hurray.PerBlockAffine.symmetric(1, 32, 1, hurray.float32)
    /// assert q.scale_type == hurray.float32
    /// ```
    #[getter]
    pub fn scale_type(&self, py: Python<'_>) -> PyResult<Py<Dtype>> {
        Py::new(
            py,
            Dtype {
                inner: self.inner.scale_type(),
            },
        )
    }

    pub fn __repr__(&self) -> String {
        format!(
            "PerBlockAffine(axis={}, block_size={}, scale_buffer_index={}, \
             zero_point_buffer_index={:?})",
            self.inner.axis(),
            self.inner.block_size(),
            self.inner.scale_buffer_index(),
            self.inner.zero_point_buffer_index()
        )
    }
}

// ── NF4 ───────────────────────────────────────────────────────────────────────

/// NF4 (NormalFloat4) block quantization: 4-bit codes indexing a fixed
/// information-theoretically optimal lookup table, with one scale per block.
///
/// ## Examples (Python)
///
/// ```python
/// import hurray
///
/// q = hurray.NF4(axis=1, block_size=64, scale_buffer_index=1)
/// assert q.block_size == 64
/// ```
#[pyclass(name = "NF4", frozen)]
#[derive(Debug)]
pub struct Nf4 {
    pub(crate) inner: CoreNf4,
}

#[pymethods]
impl Nf4 {
    /// Construct an NF4 descriptor.
    ///
    /// ## Errors
    ///
    /// - `hurray.InvalidDescriptorError` — `axis` exceeds the rank cap, or
    ///   `block_size` is invalid for the scheme.
    ///
    /// ## Examples
    ///
    /// ```python
    /// q = hurray.NF4(axis=1, block_size=64, scale_buffer_index=1)
    /// ```
    #[new]
    #[pyo3(signature = (axis, block_size, scale_buffer_index))]
    pub fn new(axis: u32, block_size: u32, scale_buffer_index: u32) -> PyResult<Self> {
        Ok(Self {
            inner: CoreNf4::new(axis, block_size, scale_buffer_index).map_err(quant_err)?,
        })
    }

    /// The quantized axis.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.NF4(1, 64, 1).axis == 1
    /// ```
    #[getter]
    pub fn axis(&self) -> u32 {
        self.inner.axis()
    }

    /// Elements per block along `axis`.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.NF4(1, 64, 1).block_size == 64
    /// ```
    #[getter]
    pub fn block_size(&self) -> u32 {
        self.inner.block_size()
    }

    /// Buffer index holding the per-block scales.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.NF4(1, 64, 1).scale_buffer_index == 1
    /// ```
    #[getter]
    pub fn scale_buffer_index(&self) -> u32 {
        self.inner.scale_buffer_index()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "NF4(axis={}, block_size={}, scale_buffer_index={})",
            self.inner.axis(),
            self.inner.block_size(),
            self.inner.scale_buffer_index()
        )
    }
}

// ── MXFP ──────────────────────────────────────────────────────────────────────

/// MXFP (OCP Microscaling) block quantization: an 8-bit shared exponent per block
/// alongside the element micro-floats.
///
/// ## Examples (Python)
///
/// ```python
/// import hurray
///
/// q = hurray.MXFP(axis=1, block_size=32, scale_buffer_index=1)
/// assert q.block_size == 32
/// ```
#[pyclass(name = "MXFP", frozen)]
#[derive(Debug)]
pub struct Mxfp {
    pub(crate) inner: CoreMxfp,
}

#[pymethods]
impl Mxfp {
    /// Construct an MXFP descriptor.
    ///
    /// ## Errors
    ///
    /// - `hurray.InvalidDescriptorError` — `axis` exceeds the rank cap, or
    ///   `block_size` is invalid for the scheme.
    ///
    /// ## Examples
    ///
    /// ```python
    /// q = hurray.MXFP(axis=1, block_size=32, scale_buffer_index=1)
    /// ```
    #[new]
    #[pyo3(signature = (axis, block_size, scale_buffer_index))]
    pub fn new(axis: u32, block_size: u32, scale_buffer_index: u32) -> PyResult<Self> {
        Ok(Self {
            inner: CoreMxfp::new(axis, block_size, scale_buffer_index).map_err(quant_err)?,
        })
    }

    /// The quantized axis.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.MXFP(1, 32, 1).axis == 1
    /// ```
    #[getter]
    pub fn axis(&self) -> u32 {
        self.inner.axis()
    }

    /// Elements per block along `axis`.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.MXFP(1, 32, 1).block_size == 32
    /// ```
    #[getter]
    pub fn block_size(&self) -> u32 {
        self.inner.block_size()
    }

    /// Buffer index holding the per-block shared exponents.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.MXFP(1, 32, 1).scale_buffer_index == 1
    /// ```
    #[getter]
    pub fn scale_buffer_index(&self) -> u32 {
        self.inner.scale_buffer_index()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "MXFP(axis={}, block_size={}, scale_buffer_index={})",
            self.inner.axis(),
            self.inner.block_size(),
            self.inner.scale_buffer_index()
        )
    }
}

// ── Conversion ────────────────────────────────────────────────────────────────

/// Convert any of the Python scheme objects into a core `QuantizationDescriptor`.
///
/// Accepting `PyAny` and dispatching on the concrete class keeps the five schemes
/// as independent types rather than one union class with unused fields.
pub(crate) fn extract_quantization(obj: &Bound<'_, PyAny>) -> PyResult<QuantizationDescriptor> {
    if let Ok(q) = obj.extract::<PyRef<PerTensorAffine>>() {
        return Ok(QuantizationDescriptor::PerTensorAffine(q.inner));
    }
    if let Ok(q) = obj.extract::<PyRef<PerChannelAffine>>() {
        return Ok(QuantizationDescriptor::PerChannelAffine(q.inner));
    }
    if let Ok(q) = obj.extract::<PyRef<PerBlockAffine>>() {
        return Ok(QuantizationDescriptor::PerBlockAffine(q.inner));
    }
    if let Ok(q) = obj.extract::<PyRef<Nf4>>() {
        return Ok(QuantizationDescriptor::Nf4(q.inner));
    }
    if let Ok(q) = obj.extract::<PyRef<Mxfp>>() {
        return Ok(QuantizationDescriptor::Mxfp(q.inner));
    }
    Err(InvalidDescriptorError::new_err(format!(
        "expected a quantization descriptor (PerTensorAffine, PerChannelAffine, \
         PerBlockAffine, NF4, or MXFP), got {}",
        obj.get_type().name()?
    )))
}

/// Wrap a decoded core `QuantizationDescriptor` in its Python class.
///
/// The inverse of [`extract_quantization`], used when reading a descriptor back
/// off the wire or off disk.
pub(crate) fn quantization_to_py(
    py: Python<'_>,
    desc: QuantizationDescriptor,
) -> PyResult<Py<PyAny>> {
    Ok(match desc {
        QuantizationDescriptor::PerTensorAffine(inner) => {
            Py::new(py, PerTensorAffine { inner })?.into_any()
        }
        QuantizationDescriptor::PerChannelAffine(inner) => {
            Py::new(py, PerChannelAffine { inner })?.into_any()
        }
        QuantizationDescriptor::PerBlockAffine(inner) => {
            Py::new(py, PerBlockAffine { inner })?.into_any()
        }
        QuantizationDescriptor::Nf4(inner) => Py::new(py, Nf4 { inner })?.into_any(),
        QuantizationDescriptor::Mxfp(inner) => Py::new(py, Mxfp { inner })?.into_any(),
    })
}

// ── Registration ──────────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PerTensorAffine>()?;
    m.add_class::<PerChannelAffine>()?;
    m.add_class::<PerBlockAffine>()?;
    m.add_class::<Nf4>()?;
    m.add_class::<Mxfp>()?;
    Ok(())
}
