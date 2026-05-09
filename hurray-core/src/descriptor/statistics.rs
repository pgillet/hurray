//! Statistics section — binary encode/decode.
//!
//! The statistics section is a fixed **72-byte** block present when the
//! `HAS_STATISTICS` flag is set. All fields are advisory (optimization hints)
//! and MUST NOT be relied upon for correctness.
//!
//! Wire layout (all little-endian):
//! ```text
//! offset  0: computed_mask   uint32
//! offset  4: _reserved       uint32   MUST be 0
//! offset  8: nnz             uint64
//! offset 16: sparsity_ratio  float64
//! offset 24: value_min       float64
//! offset 32: value_max       float64
//! offset 40: value_abs_max   float64
//! offset 48: value_mean      float64
//! offset 56: value_stddev    float64
//! offset 64: nm_n            uint8
//! offset 65: nm_m            uint8
//! offset 66: has_nan         uint8  (0x01 or 0x00)
//! offset 67: has_inf         uint8  (0x01 or 0x00)
//! offset 68: _reserved2      uint8[4]  MUST be 0
//! ```
//! Total = 72 bytes.

use crate::descriptor::cursor::{ByteCursor, ByteWriter};
use crate::{Error, Result};

/// Total byte length of the encoded statistics section.
const STATISTICS_BYTE_LEN: usize = 72;

/// Bitmask identifying which statistics fields contain valid data.
///
/// A reader MUST check the relevant bit before using any statistics field.
/// Fields whose bit is not set MUST be treated as unknown.
///
/// Bits 6–31 are reserved and MUST be `0`. A reader MUST reject a descriptor
/// whose `computed_mask` has any reserved bit set.
///
/// # Examples
///
/// ```
/// use hurray_core::descriptor::StatisticsMask;
///
/// let mask = StatisticsMask(StatisticsMask::NNZ_VALID | StatisticsMask::VALUE_RANGE_VALID);
/// assert!(mask.nnz_valid());
/// assert!(mask.value_range_valid());
/// assert!(!mask.sparsity_valid());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StatisticsMask(pub u32);

impl StatisticsMask {
    /// Bit 0: `nnz` field is valid.
    pub const NNZ_VALID: u32 = 1 << 0;
    /// Bit 1: `sparsity_ratio` field is valid.
    pub const SPARSITY_VALID: u32 = 1 << 1;
    /// Bit 2: `value_min`, `value_max`, `value_abs_max` are valid.
    pub const VALUE_RANGE_VALID: u32 = 1 << 2;
    /// Bit 3: `value_mean`, `value_stddev` are valid.
    pub const VALUE_STATS_VALID: u32 = 1 << 3;
    /// Bit 4: `nm_n`, `nm_m` are valid.
    pub const NM_SPARSITY_VALID: u32 = 1 << 4;
    /// Bit 5: `has_nan`, `has_inf` are valid.
    pub const NAN_INF_VALID: u32 = 1 << 5;

    /// Bitmask covering all reserved bits (bits 6 and above).
    const RESERVED_MASK: u32 = !0x3F;

    /// Returns `true` if the NNZ_VALID bit is set.
    #[inline]
    pub fn nnz_valid(self) -> bool {
        self.0 & Self::NNZ_VALID != 0
    }

    /// Returns `true` if the SPARSITY_VALID bit is set.
    #[inline]
    pub fn sparsity_valid(self) -> bool {
        self.0 & Self::SPARSITY_VALID != 0
    }

    /// Returns `true` if the VALUE_RANGE_VALID bit is set.
    #[inline]
    pub fn value_range_valid(self) -> bool {
        self.0 & Self::VALUE_RANGE_VALID != 0
    }

    /// Returns `true` if the VALUE_STATS_VALID bit is set.
    #[inline]
    pub fn value_stats_valid(self) -> bool {
        self.0 & Self::VALUE_STATS_VALID != 0
    }

    /// Returns `true` if the NM_SPARSITY_VALID bit is set.
    #[inline]
    pub fn nm_sparsity_valid(self) -> bool {
        self.0 & Self::NM_SPARSITY_VALID != 0
    }

    /// Returns `true` if the NAN_INF_VALID bit is set.
    #[inline]
    pub fn nan_inf_valid(self) -> bool {
        self.0 & Self::NAN_INF_VALID != 0
    }
}

/// Advisory statistics about a tensor's data buffer.
///
/// Statistics reflect the tensor data at write time. A reader MUST NOT rely on
/// any statistic for correctness — they are optimization hints only (algorithm
/// selection, memory pre-allocation, routing decisions).
///
/// Each group of fields is guarded by a bit in `computed_mask`. A reader MUST
/// check the relevant bit via [`StatisticsMask`] before using any field.
///
/// Note: derives `PartialEq` but NOT `Eq` — `f64` NaN semantics make `Eq` unsound.
///
/// # Examples
///
/// ```
/// use hurray_core::descriptor::{Statistics, StatisticsMask};
///
/// let stats = Statistics {
///     computed_mask: StatisticsMask(StatisticsMask::NNZ_VALID),
///     nnz: 42,
///     sparsity_ratio: 0.0,
///     value_min: 0.0,
///     value_max: 0.0,
///     value_abs_max: 0.0,
///     value_mean: 0.0,
///     value_stddev: 0.0,
///     nm_n: 0,
///     nm_m: 0,
///     has_nan: false,
///     has_inf: false,
/// };
/// assert!(stats.computed_mask.nnz_valid());
/// assert_eq!(stats.nnz, 42);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Statistics {
    /// Bitmask indicating which fields contain valid data.
    pub computed_mask: StatisticsMask,
    /// Number of non-zero elements. Valid when `NNZ_VALID` is set.
    pub nnz: u64,
    /// Fraction of zero elements `[0.0, 1.0]`. Valid when `SPARSITY_VALID` is set.
    pub sparsity_ratio: f64,
    /// Minimum element value (dequantized). Valid when `VALUE_RANGE_VALID` is set.
    pub value_min: f64,
    /// Maximum element value (dequantized). Valid when `VALUE_RANGE_VALID` is set.
    pub value_max: f64,
    /// Maximum absolute element value. Valid when `VALUE_RANGE_VALID` is set.
    pub value_abs_max: f64,
    /// Arithmetic mean of all elements. Valid when `VALUE_STATS_VALID` is set.
    pub value_mean: f64,
    /// Population standard deviation. Valid when `VALUE_STATS_VALID` is set.
    pub value_stddev: f64,
    /// N in N:M structured sparsity. Valid when `NM_SPARSITY_VALID` is set.
    pub nm_n: u8,
    /// M in N:M structured sparsity. Valid when `NM_SPARSITY_VALID` is set.
    pub nm_m: u8,
    /// `true` if at least one NaN element is present. Valid when `NAN_INF_VALID` is set.
    pub has_nan: bool,
    /// `true` if at least one infinity is present. Valid when `NAN_INF_VALID` is set.
    pub has_inf: bool,
}

impl Statistics {
    /// Encodes the statistics section into `w` as an exact 72-byte block.
    pub(crate) fn encode_into(&self, w: &mut ByteWriter) {
        w.write_u32_le(self.computed_mask.0); // offset  0
        w.write_u32_le(0u32); // offset  4 — _reserved
        w.write_u64_le(self.nnz); // offset  8
        w.write_f64_le(self.sparsity_ratio); // offset 16
        w.write_f64_le(self.value_min); // offset 24
        w.write_f64_le(self.value_max); // offset 32
        w.write_f64_le(self.value_abs_max); // offset 40
        w.write_f64_le(self.value_mean); // offset 48
        w.write_f64_le(self.value_stddev); // offset 56
        w.write_u8(self.nm_n); // offset 64
        w.write_u8(self.nm_m); // offset 65
        w.write_u8(u8::from(self.has_nan)); // offset 66
        w.write_u8(u8::from(self.has_inf)); // offset 67
        w.write_zeros(4); // offset 68 — _reserved2
    }

    /// Decodes a 72-byte statistics block from `cursor`.
    ///
    /// # Errors
    ///
    /// - [`Error::StatisticsReservedMaskBitsSet`] if `computed_mask` bits ≥ 6 are set.
    /// - [`Error::ReservedBytesNonZero`] if any `_reserved` field is non-zero.
    /// - [`Error::DescriptorTruncated`] if fewer than 72 bytes remain.
    pub(crate) fn decode_from(cursor: &mut ByteCursor<'_>) -> Result<Self> {
        let start_pos = cursor.pos();

        let computed_mask_raw = cursor.read_u32_le()?; // offset  0
        let reserved1 = cursor.read_u32_le()?; // offset  4
        let nnz = cursor.read_u64_le()?; // offset  8
        let sparsity_ratio = cursor.read_f64_le()?; // offset 16
        let value_min = cursor.read_f64_le()?; // offset 24
        let value_max = cursor.read_f64_le()?; // offset 32
        let value_abs_max = cursor.read_f64_le()?; // offset 40
        let value_mean = cursor.read_f64_le()?; // offset 48
        let value_stddev = cursor.read_f64_le()?; // offset 56
        let nm_n = cursor.read_u8()?; // offset 64
        let nm_m = cursor.read_u8()?; // offset 65
        let has_nan_raw = cursor.read_u8()?; // offset 66
        let has_inf_raw = cursor.read_u8()?; // offset 67
        let reserved2 = cursor.read_bytes(4)?; // offset 68

        // Validate that we read exactly STATISTICS_BYTE_LEN bytes.
        debug_assert_eq!(cursor.pos() - start_pos, STATISTICS_BYTE_LEN);

        // Reject reserved bits in computed_mask (bits 6–31).
        if computed_mask_raw & StatisticsMask::RESERVED_MASK != 0 {
            return Err(Error::StatisticsReservedMaskBitsSet {
                mask: computed_mask_raw,
            });
        }

        if reserved1 != 0 {
            return Err(Error::ReservedBytesNonZero {
                field: "statistics._reserved",
            });
        }

        if reserved2 != [0u8, 0, 0, 0] {
            return Err(Error::ReservedBytesNonZero {
                field: "statistics._reserved2",
            });
        }

        Ok(Self {
            computed_mask: StatisticsMask(computed_mask_raw),
            nnz,
            sparsity_ratio,
            value_min,
            value_max,
            value_abs_max,
            value_mean,
            value_stddev,
            nm_n,
            nm_m,
            has_nan: has_nan_raw != 0,
            has_inf: has_inf_raw != 0,
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::cursor::{ByteCursor, ByteWriter};

    fn sample_stats() -> Statistics {
        Statistics {
            computed_mask: StatisticsMask(
                StatisticsMask::NNZ_VALID
                    | StatisticsMask::SPARSITY_VALID
                    | StatisticsMask::VALUE_RANGE_VALID
                    | StatisticsMask::VALUE_STATS_VALID
                    | StatisticsMask::NM_SPARSITY_VALID
                    | StatisticsMask::NAN_INF_VALID,
            ),
            nnz: 1000,
            sparsity_ratio: 0.5,
            value_min: -1.0,
            value_max: 1.0,
            value_abs_max: 1.0,
            value_mean: 0.0,
            value_stddev: 0.5,
            nm_n: 2,
            nm_m: 4,
            has_nan: false,
            has_inf: true,
        }
    }

    #[test]
    fn statistics_round_trip() {
        let stats = sample_stats();
        let mut w = ByteWriter::new();
        stats.encode_into(&mut w);
        let bytes = w.into_vec();
        assert_eq!(bytes.len(), STATISTICS_BYTE_LEN);
        let mut c = ByteCursor::new(&bytes, bytes.len());
        let decoded = Statistics::decode_from(&mut c).unwrap();
        assert_eq!(decoded.computed_mask, stats.computed_mask);
        assert_eq!(decoded.nnz, stats.nnz);
        assert_eq!(decoded.sparsity_ratio, stats.sparsity_ratio);
        assert_eq!(decoded.value_min, stats.value_min);
        assert_eq!(decoded.value_max, stats.value_max);
        assert_eq!(decoded.value_abs_max, stats.value_abs_max);
        assert_eq!(decoded.value_mean, stats.value_mean);
        assert_eq!(decoded.value_stddev, stats.value_stddev);
        assert_eq!(decoded.nm_n, stats.nm_n);
        assert_eq!(decoded.nm_m, stats.nm_m);
        assert_eq!(decoded.has_nan, stats.has_nan);
        assert_eq!(decoded.has_inf, stats.has_inf);
    }

    #[test]
    fn statistics_reserved_mask_bits_rejected() {
        let mut w = ByteWriter::new();
        // Set bit 6 (first reserved bit).
        w.write_u32_le(0x0000_0040u32);
        w.write_zeros(68);
        let bytes = w.into_vec();
        let mut c = ByteCursor::new(&bytes, bytes.len());
        let err = Statistics::decode_from(&mut c).unwrap_err();
        assert!(matches!(err, Error::StatisticsReservedMaskBitsSet { .. }));
    }

    #[test]
    fn statistics_reserved_bytes_rejected() {
        let mut w = ByteWriter::new();
        w.write_u32_le(0u32); // computed_mask = 0 (valid)
        w.write_u32_le(1u32); // _reserved — non-zero, must be rejected
        w.write_zeros(64);
        let bytes = w.into_vec();
        let mut c = ByteCursor::new(&bytes, bytes.len());
        let err = Statistics::decode_from(&mut c).unwrap_err();
        assert!(matches!(err, Error::ReservedBytesNonZero { .. }));
    }

    #[test]
    fn statistics_reserved2_bytes_rejected() {
        let mut w = ByteWriter::new();
        w.write_u32_le(0u32); // computed_mask
        w.write_u32_le(0u32); // _reserved
        w.write_zeros(60); // nnz through has_inf
        w.write_u8(0x01); // _reserved2[0] non-zero
        w.write_zeros(3);
        let bytes = w.into_vec();
        let mut c = ByteCursor::new(&bytes, bytes.len());
        let err = Statistics::decode_from(&mut c).unwrap_err();
        assert!(matches!(err, Error::ReservedBytesNonZero { .. }));
    }

    #[test]
    fn encoded_length_is_72_bytes() {
        let mut w = ByteWriter::new();
        sample_stats().encode_into(&mut w);
        assert_eq!(w.len(), STATISTICS_BYTE_LEN);
    }
}
