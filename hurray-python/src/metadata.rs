//! Python bindings for the optional descriptor sections: statistics, shard, and
//! extension type.

use pyo3::prelude::*;
use pyo3::types::{PyModule, PyTuple};

use hurray_core::descriptor::ExtensionTypeDescriptor as CoreExtType;
use hurray_core::{ShardDescriptor as CoreShard, Statistics as CoreStats, StatisticsMask};

use crate::errors::InvalidDescriptorError;

// ── Statistics ────────────────────────────────────────────────────────────────

/// Precomputed statistics about a tensor's values.
///
/// Every field is optional. The descriptor's `computed_mask` — which records
/// *which* statistics are meaningful — is derived from the arguments you pass, so
/// a value can never be present with its validity bit unset. Omitted fields encode
/// as zero with their bit clear.
///
/// Statistics are grouped as the wire format groups them: `value_min`, `value_max`
/// and `value_abs_max` share one validity bit, as do `value_mean` and
/// `value_stddev`, and `nm_n`/`nm_m`. Supplying part of a group and not the rest is
/// an error rather than a silently half-filled section.
///
/// ## Errors
///
/// - `hurray.InvalidDescriptorError` — a grouped field was supplied without its
///   partners.
///
/// ## Examples (Python)
///
/// ```python
/// import hurray
///
/// s = hurray.Statistics(nnz=1024, value_min=-1.0, value_max=1.0, value_abs_max=1.0)
/// assert s.nnz == 1024
/// assert s.value_max == 1.0
/// assert s.value_mean is None      # not supplied, so not valid
/// ```
#[pyclass(name = "Statistics", frozen)]
#[derive(Debug)]
pub struct Statistics {
    pub(crate) inner: CoreStats,
}

#[pymethods]
impl Statistics {
    /// Build a statistics section from whichever values are known.
    ///
    /// ## Examples
    ///
    /// ```python
    /// # Sparsity only — nothing else is claimed.
    /// s = hurray.Statistics(sparsity_ratio=0.9)
    /// assert s.sparsity_ratio == 0.9
    /// assert s.nnz is None
    /// ```
    #[new]
    #[pyo3(signature = (
        *,
        nnz = None,
        sparsity_ratio = None,
        value_min = None,
        value_max = None,
        value_abs_max = None,
        value_mean = None,
        value_stddev = None,
        nm_n = None,
        nm_m = None,
        has_nan = None,
        has_inf = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        nnz: Option<u64>,
        sparsity_ratio: Option<f64>,
        value_min: Option<f64>,
        value_max: Option<f64>,
        value_abs_max: Option<f64>,
        value_mean: Option<f64>,
        value_stddev: Option<f64>,
        nm_n: Option<u8>,
        nm_m: Option<u8>,
        has_nan: Option<bool>,
        has_inf: Option<bool>,
    ) -> PyResult<Self> {
        let mut mask = 0u32;

        if nnz.is_some() {
            mask |= StatisticsMask::NNZ_VALID;
        }
        if sparsity_ratio.is_some() {
            mask |= StatisticsMask::SPARSITY_VALID;
        }

        // Grouped fields share one validity bit, so a partial group cannot be
        // encoded honestly — reject instead of silently zero-filling the rest.
        let range = [value_min, value_max, value_abs_max];
        match range.iter().filter(|v| v.is_some()).count() {
            0 => {}
            3 => mask |= StatisticsMask::VALUE_RANGE_VALID,
            _ => {
                return Err(InvalidDescriptorError::new_err(
                    "value_min, value_max and value_abs_max share one validity bit: \
                     supply all three or none",
                ))
            }
        }

        let stats = [value_mean, value_stddev];
        match stats.iter().filter(|v| v.is_some()).count() {
            0 => {}
            2 => mask |= StatisticsMask::VALUE_STATS_VALID,
            _ => {
                return Err(InvalidDescriptorError::new_err(
                    "value_mean and value_stddev share one validity bit: \
                     supply both or neither",
                ))
            }
        }

        match (nm_n, nm_m) {
            (None, None) => {}
            (Some(_), Some(_)) => mask |= StatisticsMask::NM_SPARSITY_VALID,
            _ => {
                return Err(InvalidDescriptorError::new_err(
                    "nm_n and nm_m share one validity bit: supply both or neither",
                ))
            }
        }

        match (has_nan, has_inf) {
            (None, None) => {}
            (Some(_), Some(_)) => mask |= StatisticsMask::NAN_INF_VALID,
            _ => {
                return Err(InvalidDescriptorError::new_err(
                    "has_nan and has_inf share one validity bit: supply both or neither",
                ))
            }
        }

        Ok(Self {
            inner: CoreStats {
                computed_mask: StatisticsMask(mask),
                nnz: nnz.unwrap_or(0),
                sparsity_ratio: sparsity_ratio.unwrap_or(0.0),
                value_min: value_min.unwrap_or(0.0),
                value_max: value_max.unwrap_or(0.0),
                value_abs_max: value_abs_max.unwrap_or(0.0),
                value_mean: value_mean.unwrap_or(0.0),
                value_stddev: value_stddev.unwrap_or(0.0),
                nm_n: nm_n.unwrap_or(0),
                nm_m: nm_m.unwrap_or(0),
                has_nan: has_nan.unwrap_or(false),
                has_inf: has_inf.unwrap_or(false),
            },
        })
    }

    /// Non-zero element count, or `None` if not computed.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.Statistics(nnz=10).nnz == 10
    /// ```
    #[getter]
    pub fn nnz(&self) -> Option<u64> {
        self.inner
            .computed_mask
            .nnz_valid()
            .then_some(self.inner.nnz)
    }

    /// Fraction of zero elements, or `None` if not computed.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.Statistics().sparsity_ratio is None
    /// ```
    #[getter]
    pub fn sparsity_ratio(&self) -> Option<f64> {
        self.inner
            .computed_mask
            .sparsity_valid()
            .then_some(self.inner.sparsity_ratio)
    }

    /// Minimum element value, or `None` if not computed.
    ///
    /// ## Examples
    ///
    /// ```python
    /// s = hurray.Statistics(value_min=-2.0, value_max=2.0, value_abs_max=2.0)
    /// assert s.value_min == -2.0
    /// ```
    #[getter]
    pub fn value_min(&self) -> Option<f64> {
        self.inner
            .computed_mask
            .value_range_valid()
            .then_some(self.inner.value_min)
    }

    /// Maximum element value, or `None` if not computed.
    ///
    /// ## Examples
    ///
    /// ```python
    /// s = hurray.Statistics(value_min=-2.0, value_max=2.0, value_abs_max=2.0)
    /// assert s.value_max == 2.0
    /// ```
    #[getter]
    pub fn value_max(&self) -> Option<f64> {
        self.inner
            .computed_mask
            .value_range_valid()
            .then_some(self.inner.value_max)
    }

    /// Maximum absolute element value, or `None` if not computed.
    ///
    /// ## Examples
    ///
    /// ```python
    /// s = hurray.Statistics(value_min=-2.0, value_max=1.0, value_abs_max=2.0)
    /// assert s.value_abs_max == 2.0
    /// ```
    #[getter]
    pub fn value_abs_max(&self) -> Option<f64> {
        self.inner
            .computed_mask
            .value_range_valid()
            .then_some(self.inner.value_abs_max)
    }

    /// Arithmetic mean, or `None` if not computed.
    ///
    /// ## Examples
    ///
    /// ```python
    /// s = hurray.Statistics(value_mean=0.5, value_stddev=0.1)
    /// assert s.value_mean == 0.5
    /// ```
    #[getter]
    pub fn value_mean(&self) -> Option<f64> {
        self.inner
            .computed_mask
            .value_stats_valid()
            .then_some(self.inner.value_mean)
    }

    /// Population standard deviation, or `None` if not computed.
    ///
    /// ## Examples
    ///
    /// ```python
    /// s = hurray.Statistics(value_mean=0.5, value_stddev=0.1)
    /// assert s.value_stddev == 0.1
    /// ```
    #[getter]
    pub fn value_stddev(&self) -> Option<f64> {
        self.inner
            .computed_mask
            .value_stats_valid()
            .then_some(self.inner.value_stddev)
    }

    /// N in N:M structured sparsity, or `None` if not computed.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.Statistics(nm_n=2, nm_m=4).nm_n == 2
    /// ```
    #[getter]
    pub fn nm_n(&self) -> Option<u8> {
        self.inner
            .computed_mask
            .nm_sparsity_valid()
            .then_some(self.inner.nm_n)
    }

    /// M in N:M structured sparsity, or `None` if not computed.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.Statistics(nm_n=2, nm_m=4).nm_m == 4
    /// ```
    #[getter]
    pub fn nm_m(&self) -> Option<u8> {
        self.inner
            .computed_mask
            .nm_sparsity_valid()
            .then_some(self.inner.nm_m)
    }

    /// Whether any NaN is present, or `None` if not computed.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.Statistics(has_nan=False, has_inf=False).has_nan is False
    /// ```
    #[getter]
    pub fn has_nan(&self) -> Option<bool> {
        self.inner
            .computed_mask
            .nan_inf_valid()
            .then_some(self.inner.has_nan)
    }

    /// Whether any infinity is present, or `None` if not computed.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.Statistics(has_nan=False, has_inf=True).has_inf is True
    /// ```
    #[getter]
    pub fn has_inf(&self) -> Option<bool> {
        self.inner
            .computed_mask
            .nan_inf_valid()
            .then_some(self.inner.has_inf)
    }

    /// The raw validity bitmask, as it appears on the wire.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.Statistics().computed_mask == 0
    /// assert hurray.Statistics(nnz=1).computed_mask == 1
    /// ```
    #[getter]
    pub fn computed_mask(&self) -> u32 {
        self.inner.computed_mask.0
    }

    pub fn __repr__(&self) -> String {
        // Names the statistics that are actually claimed, in constructor order, so the
        // repr rebuilds the object. `computed_mask=0x4` was the wire encoding of that
        // same information and said nothing about the values it gates.
        let mut fields: Vec<String> = Vec::new();
        let mut push = |name: &str, value: Option<String>| {
            if let Some(v) = value {
                fields.push(format!("{name}={v}"));
            }
        };

        push("nnz", self.nnz().map(|v| v.to_string()));
        push("sparsity_ratio", self.sparsity_ratio().map(float_repr));
        push("value_min", self.value_min().map(float_repr));
        push("value_max", self.value_max().map(float_repr));
        push("value_abs_max", self.value_abs_max().map(float_repr));
        push("value_mean", self.value_mean().map(float_repr));
        push("value_stddev", self.value_stddev().map(float_repr));
        push("nm_n", self.nm_n().map(|v| v.to_string()));
        push("nm_m", self.nm_m().map(|v| v.to_string()));
        push("has_nan", self.has_nan().map(bool_repr));
        push("has_inf", self.has_inf().map(bool_repr));

        format!("Statistics({})", fields.join(", "))
    }
}

/// Formats an `f64` the way Python writes one, so a repr stays a Python expression.
///
/// `{}` would print `1` for `1.0`, and non-finite values have no Python literal at all.
fn float_repr(v: f64) -> String {
    if v.is_finite() {
        format!("{v:?}")
    } else if v.is_nan() {
        "float('nan')".to_string()
    } else if v > 0.0 {
        "float('inf')".to_string()
    } else {
        "float('-inf')".to_string()
    }
}

/// Formats a `bool` the way Python writes one — Rust's `{}` gives `true`, not `True`.
fn bool_repr(v: bool) -> String {
    if v { "True" } else { "False" }.to_string()
}

// ── Shard ─────────────────────────────────────────────────────────────────────

/// Describes this tensor's position within a larger logical tensor.
///
/// `parent_shape` is the shape of the whole tensor; `shard_offset` is where this
/// piece starts along each dimension. Both must have the same length.
///
/// ## Errors
///
/// - `hurray.InvalidDescriptorError` — the two vectors differ in length.
///
/// ## Examples (Python)
///
/// ```python
/// import hurray
///
/// # The second half of a [1024, 512] tensor along dimension 0.
/// s = hurray.Shard(parent_shape=[1024, 512], shard_offset=[512, 0])
/// assert s.parent_shape == (1024, 512)
/// assert s.shard_offset == (512, 0)
/// ```
#[pyclass(name = "Shard", frozen)]
#[derive(Debug)]
pub struct Shard {
    pub(crate) inner: CoreShard,
}

#[pymethods]
impl Shard {
    /// Construct a shard descriptor.
    ///
    /// ## Examples
    ///
    /// ```python
    /// s = hurray.Shard([8, 8], [4, 0])
    /// assert s.shard_offset == (4, 0)
    /// ```
    #[new]
    #[pyo3(signature = (parent_shape, shard_offset))]
    pub fn new(parent_shape: Vec<u64>, shard_offset: Vec<u64>) -> PyResult<Self> {
        Ok(Self {
            inner: CoreShard::new(parent_shape, shard_offset)
                .map_err(|e| InvalidDescriptorError::new_err(e.to_string()))?,
        })
    }

    /// Shape of the logical parent tensor.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.Shard([8, 8], [4, 0]).parent_shape == (8, 8)
    /// ```
    #[getter]
    pub fn parent_shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.parent_shape)
    }

    /// Starting index of this shard within the parent, per dimension.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.Shard([8, 8], [4, 0]).shard_offset == (4, 0)
    /// ```
    #[getter]
    pub fn shard_offset<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.shard_offset)
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Shard(parent_shape={:?}, shard_offset={:?})",
            self.inner.parent_shape, self.inner.shard_offset
        )
    }
}

// ── ExtensionType ─────────────────────────────────────────────────────────────

/// Describes a private extension element type — one whose tag is in `0xF0`–`0xFE`.
///
/// The format reserves that tag range for types it does not standardize, and requires
/// every descriptor using one to carry this section. It is what lets a consumer that has
/// never heard of your type still size its buffers: `bit_width` and `packing_factor` are
/// enough to compute bytes without understanding a single value.
///
/// A tensor whose dtype is an extension tag MUST carry one, and a tensor whose dtype is
/// anything else MUST NOT — `hurray.Tensor` enforces both directions.
///
/// ## Sign fields
///
/// A float carries its sign in `sign_bits`, never in `is_signed`, which describes integer
/// types only. An unsigned float — the shape of the built-in exponent-only `float8_e8m0` —
/// is therefore expressible: `is_float=True, sign_bits=0`.
///
/// `packing_factor` is not an argument. The spec leaves exactly one legal value for a
/// given `bit_width`, so it is derived rather than restated.
///
/// ## Errors
///
/// - `hurray.InvalidDescriptorError` — `bit_width` is 0, or is sub-byte and not 1, 2 or 4;
///   `is_signed` is set on a float; `sign_bits` exceeds 1; or a float-only field is set on
///   an integer type.
///
/// ## Examples (Python)
///
/// ```python
/// import hurray
///
/// # A private 24-bit signed integer type.
/// ext = hurray.ExtensionType(bit_width=24, is_signed=True)
/// assert ext.packing_factor == 1
/// assert ext.buffer_size_bytes(10) == 30
///
/// # A private 4-bit type: two elements per byte, derived.
/// packed = hurray.ExtensionType(bit_width=4)
/// assert packed.packing_factor == 2
/// assert packed.buffer_size_bytes(7) == 4
/// ```
#[pyclass(name = "ExtensionType", frozen)]
#[derive(Debug)]
pub struct ExtensionType {
    pub(crate) inner: CoreExtType,
}

#[pymethods]
impl ExtensionType {
    /// Describe an extension element type.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// # A private 16-bit float: 1 sign bit, 5 exponent, 10 mantissa.
    /// half = hurray.ExtensionType(
    ///     bit_width=16, is_float=True,
    ///     sign_bits=1, exponent_bits=5, mantissa_bits=10, exponent_bias=15,
    ///     has_nan=True, has_inf=True,
    /// )
    /// assert half.is_float is True
    /// assert half.is_signed is False        # a float's sign is sign_bits
    /// ```
    #[new]
    #[pyo3(signature = (
        bit_width,
        *,
        is_float = false,
        is_signed = false,
        sign_bits = 0,
        exponent_bits = 0,
        mantissa_bits = 0,
        exponent_bias = 0,
        has_nan = false,
        has_inf = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        bit_width: u32,
        is_float: bool,
        is_signed: bool,
        sign_bits: u8,
        exponent_bits: u8,
        mantissa_bits: u8,
        exponent_bias: u32,
        has_nan: bool,
        has_inf: bool,
    ) -> PyResult<Self> {
        let packing_factor = derive_packing_factor(bit_width)?;
        let inner = CoreExtType::new(
            bit_width,
            packing_factor,
            is_float,
            is_signed,
            sign_bits,
            exponent_bits,
            mantissa_bits,
            exponent_bias,
            has_nan,
            has_inf,
        )
        .map_err(|e| InvalidDescriptorError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Bit width of one element.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.ExtensionType(bit_width=24).bit_width == 24
    /// ```
    #[getter]
    pub fn bit_width(&self) -> u32 {
        self.inner.bit_width
    }

    /// Elements packed per byte — `1` for whole-byte widths, `8 / bit_width` below that.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.ExtensionType(bit_width=8).packing_factor == 1
    /// assert hurray.ExtensionType(bit_width=1).packing_factor == 8
    /// ```
    #[getter]
    pub fn packing_factor(&self) -> u8 {
        self.inner.packing_factor
    }

    /// Whether the type is floating-point.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.ExtensionType(bit_width=8).is_float is False
    /// ```
    #[getter]
    pub fn is_float(&self) -> bool {
        self.inner.is_float
    }

    /// Whether the type is a signed *integer*. Always `False` for a float — see `sign_bits`.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.ExtensionType(bit_width=8, is_signed=True).is_signed is True
    /// ```
    #[getter]
    pub fn is_signed(&self) -> bool {
        self.inner.is_signed
    }

    /// Number of sign bits — `1` for a signed float, `0` for an unsigned one.
    ///
    /// ## Examples
    ///
    /// ```python
    /// # An exponent-only float, the float8_e8m0 shape: no sign, no mantissa.
    /// scale = hurray.ExtensionType(
    ///     bit_width=8, is_float=True, exponent_bits=8, exponent_bias=127,
    /// )
    /// assert scale.sign_bits == 0
    /// ```
    #[getter]
    pub fn sign_bits(&self) -> u8 {
        self.inner.sign_bits
    }

    /// Number of exponent bits, for float types.
    ///
    /// ## Examples
    ///
    /// ```python
    /// half = hurray.ExtensionType(
    ///     bit_width=16, is_float=True, sign_bits=1, exponent_bits=5, mantissa_bits=10,
    /// )
    /// assert half.exponent_bits == 5
    /// ```
    #[getter]
    pub fn exponent_bits(&self) -> u8 {
        self.inner.exponent_bits
    }

    /// Number of mantissa bits, for float types.
    ///
    /// ## Examples
    ///
    /// ```python
    /// half = hurray.ExtensionType(
    ///     bit_width=16, is_float=True, sign_bits=1, exponent_bits=5, mantissa_bits=10,
    /// )
    /// assert half.mantissa_bits == 10
    /// ```
    #[getter]
    pub fn mantissa_bits(&self) -> u8 {
        self.inner.mantissa_bits
    }

    /// Exponent bias, for float types.
    ///
    /// ## Examples
    ///
    /// ```python
    /// half = hurray.ExtensionType(
    ///     bit_width=16, is_float=True, sign_bits=1, exponent_bits=5, mantissa_bits=10,
    ///     exponent_bias=15,
    /// )
    /// assert half.exponent_bias == 15
    /// ```
    #[getter]
    pub fn exponent_bias(&self) -> u32 {
        self.inner.exponent_bias
    }

    /// Whether NaN is representable.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.ExtensionType(bit_width=8).has_nan is False
    /// ```
    #[getter]
    pub fn has_nan(&self) -> bool {
        self.inner.has_nan
    }

    /// Whether infinity is representable.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.ExtensionType(bit_width=8).has_inf is False
    /// ```
    #[getter]
    pub fn has_inf(&self) -> bool {
        self.inner.has_inf
    }

    /// Bytes needed to hold `element_count` elements of this type.
    ///
    /// `hurray.buffer_size_bytes` cannot answer this: an extension `Dtype` reports a
    /// `bit_width` of 0, because the real width lives here. Sub-byte widths round up to
    /// the next whole byte.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.ExtensionType(bit_width=24).buffer_size_bytes(10) == 30
    /// assert hurray.ExtensionType(bit_width=4).buffer_size_bytes(7) == 4   # ceil(7 / 2)
    /// assert hurray.ExtensionType(bit_width=4).buffer_size_bytes(0) == 0
    /// ```
    pub fn buffer_size_bytes(&self, element_count: u64) -> u64 {
        self.inner.buffer_size_bytes(element_count)
    }

    pub fn __repr__(&self) -> String {
        // Names only what was supplied: every field but bit_width defaults, and
        // packing_factor is not an argument at all, so printing it would give a repr
        // that cannot be evaluated back.
        let mut fields = vec![format!("bit_width={}", self.inner.bit_width)];
        let mut push = |name: &str, value: String| fields.push(format!("{name}={value}"));

        if self.inner.is_float {
            push("is_float", bool_repr(true));
        }
        if self.inner.is_signed {
            push("is_signed", bool_repr(true));
        }
        if self.inner.sign_bits != 0 {
            push("sign_bits", self.inner.sign_bits.to_string());
        }
        if self.inner.exponent_bits != 0 {
            push("exponent_bits", self.inner.exponent_bits.to_string());
        }
        if self.inner.mantissa_bits != 0 {
            push("mantissa_bits", self.inner.mantissa_bits.to_string());
        }
        if self.inner.exponent_bias != 0 {
            push("exponent_bias", self.inner.exponent_bias.to_string());
        }
        if self.inner.has_nan {
            push("has_nan", bool_repr(true));
        }
        if self.inner.has_inf {
            push("has_inf", bool_repr(true));
        }

        format!("ExtensionType({})", fields.join(", "))
    }
}

/// The one packing factor the spec permits for `bit_width`.
///
/// Whole-byte types pack one element per byte; sub-byte types pack `8 / bit_width`, and
/// only widths 1, 2 and 4 are legal. Since the value is fully determined, the Python
/// constructor derives it instead of asking the caller to restate it — the divergence
/// from `ExtensionTypeDescriptor::new` is deliberate and loses no expressiveness.
fn derive_packing_factor(bit_width: u32) -> PyResult<u8> {
    match bit_width {
        0 => Err(InvalidDescriptorError::new_err(
            "bit_width must be greater than 0",
        )),
        1 => Ok(8),
        2 => Ok(4),
        4 => Ok(2),
        w if w < 8 => Err(InvalidDescriptorError::new_err(format!(
            "sub-byte extension types must be 1, 2 or 4 bits wide, not {w}: other widths \
             are reserved to the built-in type space (metadata.md § Extension Type Section)"
        ))),
        _ => Ok(1),
    }
}

// ── Registration ──────────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Statistics>()?;
    m.add_class::<Shard>()?;
    m.add_class::<ExtensionType>()?;
    Ok(())
}
