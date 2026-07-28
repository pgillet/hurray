//! NF4 (NormalFloat4) quantization descriptor (scheme tag `0x04`).
//!
//! A non-linear 4-bit quantization scheme introduced by the QLoRA paper. Each
//! storage code in `[0, 15]` decodes to one of 16 fixed real-valued levels. Each
//! block along `axis` carries a single `absmax` scale stored in a separate buffer.
//!
//! See `docs/spec/quantization/nf4.md` for the normative definition.

use crate::{ElementType, Error, Result};

// ── Wire layout constants ─────────────────────────────────────────────────────

/// Scheme tag byte for NF4 quantization.
pub(crate) const SCHEME_TAG: u8 = 0x04;

/// Total descriptor length in bytes (header 4 + axis 4 + block_size 4 + scale_buf 4).
pub(crate) const ENCODED_LEN: usize = 16;

/// Version this implementation supports.
pub(crate) const SUPPORTED_VERSION: u8 = 0x01;

/// Minimum block size for NF4 (must be a power of two ≥ 8).
///
/// Below 8 elements per block the 16-point NF4 information content provides no
/// statistical benefit over a plain low-bit linear quantization.
pub const MIN_BLOCK_SIZE: u32 = 8;

/// No flags are defined for this scheme; all 16 bits must be zero.
const RESERVED_FLAGS_MASK: u16 = 0xFFFF;

/// Wire sentinel that must NOT appear in `scale_buffer_index` (symmetric sentinel).
const INVALID_BUF_SENTINEL: u32 = 0xFFFF_FFFF;

// Wire field offsets.
const OFFSET_AXIS: usize = 4;
const OFFSET_BLOCK_SIZE: usize = 8;
const OFFSET_SCALE_BUF: usize = 12;

// ── NF4 lookup table ──────────────────────────────────────────────────────────

/// The 16 NF4 quantization levels, indexed by the unsigned 4-bit storage code.
///
/// These values are fixed by the specification and MUST NOT be altered. They are
/// the `float32` decimal expansions of the exact levels from the QLoRA reference
/// implementation (`docs/spec/quantization/nf4.md § Lookup Table`).
///
/// The extra precision in the source literals is intentional: these are the
/// exact decimal representations of the IEEE 754 binary32 values; truncating
/// them would alter the bit pattern.
///
/// # Examples
///
/// ```
/// use hurray_core::NF4_LUT;
///
/// assert_eq!(NF4_LUT[0], -1.0f32);
/// assert_eq!(NF4_LUT[7], 0.0f32);
/// assert_eq!(NF4_LUT[15], 1.0f32);
/// assert_eq!(NF4_LUT.len(), 16);
/// ```
#[allow(clippy::excessive_precision)]
pub const NF4_LUT: [f32; 16] = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
];

// ── Nf4 ───────────────────────────────────────────────────────────────────────

/// Quantization parameters for NF4 (NormalFloat4) block quantization.
///
/// Each block along `axis` contains `block_size` elements, all sharing a single
/// `absmax` scale stored in `scale_buffer_index`. The dequantization formula for
/// element `q` (a 4-bit code in `[0, 15]`) with block index `b` is:
///
/// ```text
/// x_real = scale[b] * NF4_LUT[q]
/// ```
///
/// The block index computation follows the same rule as Per-Block Affine
/// (see `docs/spec/quantization/per-block-affine.md § Block Layout`).
///
/// # Wire format
///
/// Total descriptor length: **16 bytes** (including the 4-byte header).
///
/// | Offset | Field | Type |
/// |--------|-------|------|
/// | 4 | `axis` | `uint32` LE |
/// | 8 | `block_size` | `uint32` LE |
/// | 12 | `scale_buffer_index` | `uint32` LE |
///
/// No flags are defined; the `flags` field MUST be `0x0000`.
///
/// # Design notes
///
/// `PartialEq`, `Eq`, and `Hash` are all derived because this struct contains
/// no floating-point fields.
///
/// `Copy` because the struct is ≤ 24 bytes with no `Drop` glue.
///
/// # Examples
///
/// ```
/// use hurray_core::Nf4;
///
/// let q = Nf4::new(0, 64, 1).unwrap();
/// assert_eq!(q.axis(), 0);
/// assert_eq!(q.block_size(), 64);
/// assert_eq!(q.scale_buffer_index(), 1);
/// ```
// WHY Eq + Hash: no float fields.
// WHY Copy: ≤24 bytes, no Drop glue (design decision #7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Nf4 {
    axis: u32,
    block_size: u32,
    scale_buffer_index: u32,
}

impl Nf4 {
    /// Creates a new [`Nf4`] descriptor.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidBlockSize`] — `block_size` is not a power of two or is
    ///   less than [`MIN_BLOCK_SIZE`] (8).
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{Nf4, Error};
    ///
    /// assert!(Nf4::new(0, 64, 1).is_ok());
    /// assert!(Nf4::new(0, 128, 1).is_ok());
    ///
    /// // block_size < 8 is rejected.
    /// assert!(Nf4::new(0, 4, 1).is_err());
    /// // Non-power-of-two is rejected.
    /// assert!(Nf4::new(0, 48, 1).is_err());
    /// ```
    pub fn new(axis: u32, block_size: u32, scale_buffer_index: u32) -> Result<Self> {
        validate_block_size(block_size)?;
        Ok(Self {
            axis,
            block_size,
            scale_buffer_index,
        })
    }

    /// Returns the quantization axis index.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::Nf4;
    ///
    /// let q = Nf4::new(2, 64, 1).unwrap();
    /// assert_eq!(q.axis(), 2);
    /// ```
    #[inline]
    pub fn axis(&self) -> u32 {
        self.axis
    }

    /// Returns the number of logical elements per block along `axis`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::Nf4;
    ///
    /// let q = Nf4::new(0, 128, 1).unwrap();
    /// assert_eq!(q.block_size(), 128);
    /// ```
    #[inline]
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Returns the buffer table index of the per-block `absmax` scale array.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::Nf4;
    ///
    /// let q = Nf4::new(0, 64, 3).unwrap();
    /// assert_eq!(q.scale_buffer_index(), 3);
    /// ```
    #[inline]
    pub fn scale_buffer_index(&self) -> u32 {
        self.scale_buffer_index
    }

    /// Computes the number of blocks along `axis` for a given `shape_axis` size.
    ///
    /// Uses `ceil(shape_axis / block_size)`.
    ///
    /// Returns `0` when `shape_axis == 0` per the ADR-007 empty-axis carve-out.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::Nf4;
    ///
    /// let q = Nf4::new(0, 64, 1).unwrap();
    /// assert_eq!(q.num_blocks_per_axis(128), 2);
    /// assert_eq!(q.num_blocks_per_axis(65), 2); // partial trailing block
    /// assert_eq!(q.num_blocks_per_axis(0), 0);  // ADR-007 empty-axis carve-out
    /// ```
    pub fn num_blocks_per_axis(&self, shape_axis: u64) -> u64 {
        if shape_axis == 0 {
            // ADR-007 empty-axis carve-out: zero blocks, zero-byte scale buffer.
            return 0;
        }
        shape_axis.div_ceil(self.block_size as u64)
    }

    /// Validates this descriptor against the resolved `shape_axis` size.
    ///
    /// Rejects if `shape_axis` is the DYNAMIC sentinel (`u64::MAX`), or if
    /// `shape_axis > 0` and `block_size > shape_axis`.
    ///
    /// # Errors
    ///
    /// - [`Error::QuantizationShapeMismatch`] — `shape_axis` is the DYNAMIC sentinel,
    ///   or `shape_axis > 0` and `block_size > shape_axis`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{Nf4, DYNAMIC};
    ///
    /// let q = Nf4::new(0, 64, 1).unwrap();
    /// assert!(q.validate_against_shape_axis(128).is_ok());
    /// assert!(q.validate_against_shape_axis(64).is_ok());
    /// assert!(q.validate_against_shape_axis(0).is_ok()); // ADR-007: waived
    /// assert!(q.validate_against_shape_axis(32).is_err()); // block_size > shape_axis
    /// assert!(q.validate_against_shape_axis(DYNAMIC).is_err()); // dynamic dimension
    /// ```
    pub fn validate_against_shape_axis(&self, shape_axis: u64) -> Result<()> {
        // Reject the DYNAMIC sentinel: a dynamic dimension cannot be validated.
        if shape_axis == crate::shape::DYNAMIC {
            return Err(Error::QuantizationShapeMismatch {
                axis: self.axis,
                shape_axis,
                block_size: self.block_size,
                reason: "shape[axis] must not be the DYNAMIC sentinel (0xFFFFFFFFFFFFFFFF)",
            });
        }
        if shape_axis > 0 && self.block_size as u64 > shape_axis {
            return Err(Error::QuantizationShapeMismatch {
                axis: self.axis,
                shape_axis,
                block_size: self.block_size,
                reason: "NF4 block_size must not exceed shape[axis] when shape[axis] > 0",
            });
        }
        Ok(())
    }

    /// Returns the set of storage [`ElementType`]s that are valid for this scheme.
    ///
    /// Per `docs/spec/quantization/nf4.md § Valid Storage Types`: only `uint4`
    /// is valid for NF4.
    ///
    /// WHY `&'static [ElementType]`: no allocation per call (design decision #5).
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{ElementType, Nf4};
    ///
    /// assert_eq!(Nf4::valid_storage_types(), &[ElementType::Uint4]);
    /// assert!(!Nf4::valid_storage_types().contains(&ElementType::Int4));
    /// ```
    pub fn valid_storage_types() -> &'static [ElementType] {
        &[ElementType::Uint4]
    }

    // ── Crate-internal encode/decode ──────────────────────────────────────────

    /// Decodes the scheme-specific payload from `bytes`.
    ///
    /// `bytes` is the full descriptor slice (including the 4-byte header).
    /// `flags` are the header flags already read by the caller.
    pub(crate) fn decode_payload(flags: u16, bytes: &[u8]) -> Result<Self> {
        if bytes.len() < ENCODED_LEN {
            return Err(Error::QuantizationDescriptorTooShort {
                found: bytes.len(),
                needed: ENCODED_LEN,
            });
        }
        // No flags defined for NF4.
        if flags & RESERVED_FLAGS_MASK != 0 {
            return Err(Error::ReservedQuantizationFlagsBits {
                flags,
                mask: RESERVED_FLAGS_MASK,
            });
        }

        let axis = u32::from_le_bytes(
            bytes[OFFSET_AXIS..OFFSET_AXIS + 4]
                .try_into()
                .map_err(|_| Error::InvalidQuantization("axis slice error".into()))?,
        );
        let block_size = u32::from_le_bytes(
            bytes[OFFSET_BLOCK_SIZE..OFFSET_BLOCK_SIZE + 4]
                .try_into()
                .map_err(|_| Error::InvalidQuantization("block_size slice error".into()))?,
        );
        let scale_buf = u32::from_le_bytes(
            bytes[OFFSET_SCALE_BUF..OFFSET_SCALE_BUF + 4]
                .try_into()
                .map_err(|_| Error::InvalidQuantization("scale_buffer_index slice error".into()))?,
        );

        // block_size must be a power of two and >= 8.
        validate_block_size(block_size)?;

        // scale_buffer_index must not be the sentinel value 0xFFFFFFFF.
        if scale_buf == INVALID_BUF_SENTINEL {
            return Err(Error::InvalidQuantization(
                "NF4 scale_buffer_index must not be 0xFFFFFFFF".into(),
            ));
        }

        Ok(Self {
            axis,
            block_size,
            scale_buffer_index: scale_buf,
        })
    }

    /// Encodes the scheme-specific payload into `out`.
    ///
    /// `out` must be at least [`ENCODED_LEN`] bytes. The caller writes the 4-byte
    /// header; this method writes bytes 4–15.
    pub(crate) fn encode_payload(&self, out: &mut [u8]) {
        out[OFFSET_AXIS..OFFSET_AXIS + 4].copy_from_slice(&self.axis.to_le_bytes());
        out[OFFSET_BLOCK_SIZE..OFFSET_BLOCK_SIZE + 4]
            .copy_from_slice(&self.block_size.to_le_bytes());
        out[OFFSET_SCALE_BUF..OFFSET_SCALE_BUF + 4]
            .copy_from_slice(&self.scale_buffer_index.to_le_bytes());
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn validate_block_size(block_size: u32) -> Result<()> {
    if !block_size.is_power_of_two() || block_size < MIN_BLOCK_SIZE {
        return Err(Error::InvalidBlockSize {
            scheme_tag: SCHEME_TAG,
            block_size,
            min: MIN_BLOCK_SIZE,
            // NF4 has no upper bound in the spec; use u32::MAX as sentinel.
            max: u32::MAX,
        });
    }
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;

    // ── NF4_LUT ───────────────────────────────────────────────────────────────

    #[test]
    fn nf4_lut_has_16_entries() {
        assert_eq!(NF4_LUT.len(), 16);
    }

    #[test]
    fn nf4_lut_all_values_are_finite() {
        for (i, &v) in NF4_LUT.iter().enumerate() {
            assert!(v.is_finite(), "NF4_LUT[{i}] = {v} is not finite");
        }
    }

    #[test]
    fn nf4_lut_values_are_in_ascending_order() {
        // Spec requires monotonically non-decreasing values.
        for i in 1..NF4_LUT.len() {
            assert!(
                NF4_LUT[i] >= NF4_LUT[i - 1],
                "NF4_LUT is not ascending at index {i}: NF4_LUT[{}]={} > NF4_LUT[{}]={}",
                i - 1,
                NF4_LUT[i - 1],
                i,
                NF4_LUT[i]
            );
        }
    }

    #[test]
    // Deliberately asserting properties of the const NF4_LUT; the values are known at
    // compile time, which is exactly what this test pins down.
    #[allow(clippy::assertions_on_constants)]
    fn nf4_lut_straddles_zero_at_index_7_8() {
        // Per QLoRA paper: index 7 is 0.0 (the zero-crossing boundary),
        // index 8 is the first strictly positive value.
        assert!(NF4_LUT[7] <= 0.0, "NF4_LUT[7] should be <= 0.0");
        assert!(NF4_LUT[8] >= 0.0, "NF4_LUT[8] should be >= 0.0");
    }

    #[test]
    fn nf4_lut_first_value_approx_neg_one() {
        // Spec: first entry is approximately −1.0 (±0.01).
        let diff = (NF4_LUT[0] - (-1.0f32)).abs();
        assert!(
            diff < 0.01,
            "NF4_LUT[0] = {} is not within 0.01 of -1.0",
            NF4_LUT[0]
        );
    }

    #[test]
    fn nf4_lut_last_value_approx_pos_one() {
        // Spec: last entry is approximately +1.0 (±0.01).
        let diff = (NF4_LUT[15] - 1.0f32).abs();
        assert!(
            diff < 0.01,
            "NF4_LUT[15] = {} is not within 0.01 of 1.0",
            NF4_LUT[15]
        );
    }

    // ── Nf4::new ──────────────────────────────────────────────────────────────

    #[test]
    fn new_block_size_below_min_is_err() {
        // MIN_BLOCK_SIZE = 8; block_size = 4 is below minimum.
        assert!(matches!(
            Nf4::new(0, 4, 1),
            Err(Error::InvalidBlockSize { .. })
        ));
    }

    #[test]
    fn new_block_size_not_power_of_two_is_err() {
        assert!(matches!(
            Nf4::new(0, 12, 1),
            Err(Error::InvalidBlockSize { .. })
        ));
    }

    #[test]
    fn new_block_size_1_is_err() {
        assert!(matches!(
            Nf4::new(0, 1, 1),
            Err(Error::InvalidBlockSize { .. })
        ));
    }

    #[test]
    fn new_valid_min_block_size_is_ok() {
        assert!(Nf4::new(0, MIN_BLOCK_SIZE, 1).is_ok());
    }

    #[test]
    fn new_valid_block_size_16_is_ok() {
        assert!(Nf4::new(0, 16, 1).is_ok());
    }

    #[test]
    fn new_valid_block_size_64_is_ok() {
        assert!(Nf4::new(0, 64, 1).is_ok());
    }

    // ── num_blocks_per_axis ───────────────────────────────────────────────────

    #[test]
    fn num_blocks_per_axis_zero_shape_returns_zero() {
        // ADR-007 empty-axis carve-out.
        let q = Nf4::new(0, 64, 1).unwrap();
        assert_eq!(q.num_blocks_per_axis(0), 0);
    }

    #[test]
    fn num_blocks_per_axis_exact_divisibility() {
        let q = Nf4::new(0, 64, 1).unwrap();
        assert_eq!(q.num_blocks_per_axis(128), 2);
    }

    #[test]
    fn num_blocks_per_axis_partial_trailing_block_uses_div_ceil() {
        let q = Nf4::new(0, 64, 1).unwrap();
        // 65 / 64 = 1.015... → ceil = 2.
        assert_eq!(q.num_blocks_per_axis(65), 2);
    }

    // ── Round-trips ───────────────────────────────────────────────────────────

    fn encode_decode(q: &Nf4) -> Nf4 {
        let mut buf = vec![0u8; ENCODED_LEN];
        buf[0] = SCHEME_TAG;
        buf[1] = SUPPORTED_VERSION;
        buf[2] = 0;
        buf[3] = 0;
        q.encode_payload(&mut buf);
        Nf4::decode_payload(0, &buf).unwrap()
    }

    #[test]
    fn round_trip_preserves_axis_block_size_scale_buf() {
        let original = Nf4::new(2, 64, 5).unwrap();
        let decoded = encode_decode(&original);
        assert_eq!(decoded.axis(), original.axis());
        assert_eq!(decoded.block_size(), original.block_size());
        assert_eq!(decoded.scale_buffer_index(), original.scale_buffer_index());
    }

    #[test]
    fn round_trip_min_block_size() {
        let original = Nf4::new(0, MIN_BLOCK_SIZE, 1).unwrap();
        let decoded = encode_decode(&original);
        assert_eq!(decoded, original);
    }

    // ── validate_against_shape_axis ───────────────────────────────────────────

    #[test]
    fn validate_against_shape_axis_ok_when_block_size_le_shape() {
        let q = Nf4::new(0, 64, 1).unwrap();
        assert!(q.validate_against_shape_axis(64).is_ok());
        assert!(q.validate_against_shape_axis(128).is_ok());
    }

    #[test]
    fn validate_against_shape_axis_ok_when_shape_is_zero_adr007() {
        // ADR-007 carve-out: empty axis is accepted.
        let q = Nf4::new(0, 64, 1).unwrap();
        assert!(q.validate_against_shape_axis(0).is_ok());
    }

    #[test]
    fn validate_against_shape_axis_err_when_block_size_exceeds_shape() {
        let q = Nf4::new(0, 64, 1).unwrap();
        assert!(matches!(
            q.validate_against_shape_axis(32),
            Err(Error::QuantizationShapeMismatch { .. })
        ));
    }

    #[test]
    fn validate_against_shape_axis_err_for_dynamic_sentinel() {
        let q = Nf4::new(0, 64, 1).unwrap();
        assert!(matches!(
            q.validate_against_shape_axis(u64::MAX),
            Err(Error::QuantizationShapeMismatch { .. })
        ));
    }

    // ── decode_payload error paths ────────────────────────────────────────────

    #[test]
    fn decode_payload_too_short_is_err() {
        let short = vec![0u8; ENCODED_LEN - 1];
        assert!(matches!(
            Nf4::decode_payload(0, &short),
            Err(Error::QuantizationDescriptorTooShort { .. })
        ));
    }

    #[test]
    fn decode_payload_nonzero_flags_is_err() {
        let q = Nf4::new(0, 64, 1).unwrap();
        let mut buf = vec![0u8; ENCODED_LEN];
        buf[0] = SCHEME_TAG;
        buf[1] = SUPPORTED_VERSION;
        q.encode_payload(&mut buf);
        let nonzero_flags: u16 = 0x0002;
        assert!(matches!(
            Nf4::decode_payload(nonzero_flags, &buf),
            Err(Error::ReservedQuantizationFlagsBits { .. })
        ));
    }

    #[test]
    fn decode_payload_invalid_block_size_is_err() {
        let mut buf = vec![0u8; ENCODED_LEN];
        buf[0] = SCHEME_TAG;
        buf[1] = SUPPORTED_VERSION;
        buf[2] = 0;
        buf[3] = 0;
        buf[4..8].copy_from_slice(&0u32.to_le_bytes()); // axis
                                                        // block_size = 3 (not a power of two, and < MIN_BLOCK_SIZE=8)
        buf[8..12].copy_from_slice(&3u32.to_le_bytes());
        buf[12..16].copy_from_slice(&1u32.to_le_bytes()); // scale_buf
        assert!(matches!(
            Nf4::decode_payload(0, &buf),
            Err(Error::InvalidBlockSize { .. })
        ));
    }

    #[test]
    fn decode_payload_scale_buf_sentinel_is_err() {
        let mut buf = vec![0u8; ENCODED_LEN];
        buf[0] = SCHEME_TAG;
        buf[1] = SUPPORTED_VERSION;
        buf[2] = 0;
        buf[3] = 0;
        buf[4..8].copy_from_slice(&0u32.to_le_bytes()); // axis
        buf[8..12].copy_from_slice(&64u32.to_le_bytes()); // block_size
                                                          // scale_buffer_index = 0xFFFFFFFF (invalid sentinel)
        buf[12..16].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        assert!(matches!(
            Nf4::decode_payload(0, &buf),
            Err(Error::InvalidQuantization(_))
        ));
    }
}
