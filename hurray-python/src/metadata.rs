//! Python bindings for the optional descriptor sections: statistics and shard.

use pyo3::prelude::*;
use pyo3::types::{PyModule, PyTuple};

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
        format!(
            "Statistics(computed_mask=0x{:X})",
            self.inner.computed_mask.0
        )
    }
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

// ── Registration ──────────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Statistics>()?;
    m.add_class::<Shard>()?;
    Ok(())
}
