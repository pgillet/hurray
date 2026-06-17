//! MXFP (OCP Microscaling) quantization descriptor (scheme tag `0x05`).
//!
//! A block quantization format standardised by the Open Compute Project
//! Microscaling specification (OCP MX v1.0). Each contiguous block of elements
//! along a chosen axis shares a single `float8_e8m0` exponent-only scale stored
//! in a dedicated buffer. Unlike per-block affine, MXFP requires exact
//! divisibility — partial trailing blocks are **not** permitted.
//!
//! See `docs/spec/quantization/mxfp.md` for the normative definition.

use crate::{ElementType, Error, Result};

// ── Wire layout constants ─────────────────────────────────────────────────────

/// Scheme tag byte for MXFP quantization.
pub(crate) const SCHEME_TAG: u8 = 0x05;

/// Total descriptor length in bytes (header 4 + axis 4 + block_size 4 + scale_buf 4).
pub(crate) const ENCODED_LEN: usize = 16;

/// Version this implementation supports.
pub(crate) const SUPPORTED_VERSION: u8 = 0x01;

/// Minimum block size for MXFP (inclusive).
///
/// Values below 16 have no hardware Tensor Core support and are invalid under
/// any OCP MX revision.
pub const MIN_BLOCK_SIZE: u32 = 16;

/// Maximum block size for MXFP (inclusive).
pub const MAX_BLOCK_SIZE: u32 = 2048;

/// OCP MX v1.0 canonical block size.
pub const CANONICAL_BLOCK_SIZE: u32 = 32;

/// No flags are defined for this scheme; all 16 bits must be zero.
const RESERVED_FLAGS_MASK: u16 = 0xFFFF;

/// Wire sentinel that MUST NOT appear as `scale_buffer_index`.
const INVALID_BUF_SENTINEL: u32 = 0xFFFF_FFFF;

// Wire field offsets (relative to start of descriptor, including header).
const OFFSET_AXIS: usize = 4;
const OFFSET_BLOCK_SIZE: usize = 8;
const OFFSET_SCALE_BUF: usize = 12;

// ── Mxfp ──────────────────────────────────────────────────────────────────────

/// Quantization parameters for MXFP (OCP Microscaling) block quantization.
///
/// Each block of `block_size` consecutive elements along `axis` shares a single
/// `float8_e8m0` exponent-only scale. The scale values are stored as raw bytes
/// in the buffer identified by `scale_buffer_index`.
///
/// The dequantization formula for element `q` with block index `b` and scale
/// byte `e = scale[b]` is:
///
/// ```text
/// s      = 2^(e - 127)          (float8_e8m0 shared exponent)
/// x_real = s * value_of(q)      (float_value_of(q) or int_value_of(q))
/// ```
///
/// Unlike per-block affine, MXFP requires exact divisibility: `shape[axis]`
/// MUST be a positive multiple of `block_size`. No partial trailing blocks are
/// permitted.
///
/// # Wire format
///
/// Total descriptor length: **16 bytes** (including the 4-byte header).
///
/// | Offset | Field | Type |
/// |--------|-------|------|
/// | 0 | `scheme_tag` | `uint8` (must be `0x05`) |
/// | 1 | `scheme_version` | `uint8` (must be `0x01`) |
/// | 2 | `flags` | `uint16` LE (must be `0x0000`) |
/// | 4 | `axis` | `uint32` LE |
/// | 8 | `block_size` | `uint32` LE, power-of-two in `[16, 2048]` |
/// | 12 | `scale_buffer_index` | `uint32` LE |
///
/// # Design notes
///
/// `PartialEq`, `Eq`, and `Hash` are all derived because this struct contains
/// no floating-point fields — all fields are integers.
///
/// `Copy` because the struct is ≤ 24 bytes with no `Drop` glue.
///
/// # Examples
///
/// ```
/// use hurray_core::Mxfp;
///
/// let q = Mxfp::new(0, 32, 1).unwrap();
/// assert_eq!(q.axis(), 0);
/// assert_eq!(q.block_size(), 32);
/// assert_eq!(q.scale_buffer_index(), 1);
/// ```
// WHY Eq + Hash: no float fields.
// WHY Copy: ≤24 bytes, no Drop glue (design decision #7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Mxfp {
    axis: u32,
    block_size: u32,
    scale_buffer_index: u32,
}

impl Mxfp {
    /// Creates a new [`Mxfp`] descriptor.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidBlockSize`] — `block_size` is not a power of two, or
    ///   lies outside `[`[`MIN_BLOCK_SIZE`]`, `[`MAX_BLOCK_SIZE`]`]` (`[16, 2048]`).
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{Mxfp, Error};
    ///
    /// assert!(Mxfp::new(0, 32, 1).is_ok());
    /// assert!(Mxfp::new(0, 16, 1).is_ok());
    /// assert!(Mxfp::new(0, 2048, 1).is_ok());
    ///
    /// // block_size < 16 is rejected.
    /// assert!(Mxfp::new(0, 8, 1).is_err());
    /// // block_size > 2048 is rejected.
    /// assert!(Mxfp::new(0, 4096, 1).is_err());
    /// // Non-power-of-two is rejected.
    /// assert!(Mxfp::new(0, 48, 1).is_err());
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
    /// use hurray_core::Mxfp;
    ///
    /// let q = Mxfp::new(2, 32, 1).unwrap();
    /// assert_eq!(q.axis(), 2);
    /// ```
    #[inline]
    pub fn axis(&self) -> u32 {
        self.axis
    }

    /// Returns the number of logical elements per block along `axis`.
    ///
    /// Always a power of two in `[`[`MIN_BLOCK_SIZE`]`, `[`MAX_BLOCK_SIZE`]`]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::Mxfp;
    ///
    /// let q = Mxfp::new(0, 64, 1).unwrap();
    /// assert_eq!(q.block_size(), 64);
    /// ```
    #[inline]
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Returns the buffer table index of the per-block `float8_e8m0` scale array.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::Mxfp;
    ///
    /// let q = Mxfp::new(0, 32, 3).unwrap();
    /// assert_eq!(q.scale_buffer_index(), 3);
    /// ```
    #[inline]
    pub fn scale_buffer_index(&self) -> u32 {
        self.scale_buffer_index
    }

    /// Computes the number of blocks along `axis` for a given `shape_axis` size.
    ///
    /// MXFP requires exact divisibility: `shape_axis` MUST be a positive multiple
    /// of `block_size`. This is stricter than per-block affine, which uses
    /// `div_ceil` and allows partial trailing blocks.
    ///
    /// # Errors
    ///
    /// - [`Error::QuantizationShapeMismatch`] — `shape_axis` is not evenly
    ///   divisible by `block_size`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::Mxfp;
    ///
    /// let q = Mxfp::new(0, 32, 1).unwrap();
    /// assert_eq!(q.num_blocks_per_axis(64).unwrap(), 2);
    /// assert_eq!(q.num_blocks_per_axis(32).unwrap(), 1);
    ///
    /// // 65 is not a multiple of 32 — error.
    /// assert!(q.num_blocks_per_axis(65).is_err());
    /// // 0 is not a positive multiple — error.
    /// assert!(q.num_blocks_per_axis(0).is_err());
    /// ```
    pub fn num_blocks_per_axis(&self, shape_axis: u64) -> Result<u64> {
        // MXFP requires shape_axis > 0 AND exact divisibility by block_size.
        // Stricter than per-block-affine which uses div_ceil — exact divisibility
        // required by MXFP spec (mxfp.md § Validity Constraints).
        if shape_axis == 0 || shape_axis % self.block_size as u64 != 0 {
            return Err(Error::QuantizationShapeMismatch {
                axis: self.axis,
                shape_axis,
                block_size: self.block_size,
                reason: "MXFP requires shape[axis] to be a positive multiple of block_size; partial trailing blocks are not permitted",
            });
        }
        Ok(shape_axis / self.block_size as u64)
    }

    /// Validates that `shape_axis` satisfies the MXFP divisibility constraint.
    ///
    /// `shape_axis` MUST be greater than zero AND evenly divisible by `block_size`.
    ///
    /// # Errors
    ///
    /// - [`Error::QuantizationShapeMismatch`] — `shape_axis == 0` or
    ///   `shape_axis % block_size != 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::Mxfp;
    ///
    /// let q = Mxfp::new(0, 32, 1).unwrap();
    /// assert!(q.validate_against_shape_axis(64).is_ok());
    /// assert!(q.validate_against_shape_axis(32).is_ok());
    ///
    /// // Zero is rejected (not a positive multiple).
    /// assert!(q.validate_against_shape_axis(0).is_err());
    /// // 65 is not divisible by 32.
    /// assert!(q.validate_against_shape_axis(65).is_err());
    /// ```
    pub fn validate_against_shape_axis(&self, shape_axis: u64) -> Result<()> {
        self.num_blocks_per_axis(shape_axis).map(|_| ())
    }

    /// Validates the content of a scale buffer for MXFP conformance.
    ///
    /// Per `docs/spec/quantization/mxfp.md § Referenced Buffer`: the bit patterns
    /// `0x00` and `0xFF` are reserved (`NaN` per OCP MX v1.0 § 5.6) and MUST NOT
    /// appear in any scale byte.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidQuantization`] — any byte in `bytes` is `0x00` or `0xFF`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::Mxfp;
    ///
    /// let q = Mxfp::new(0, 32, 1).unwrap();
    ///
    /// // Valid scale bytes: non-zero, non-0xFF.
    /// assert!(q.validate_scale_bytes(&[0x01, 0x7F, 0x80, 0xFE]).is_ok());
    ///
    /// // 0x00 is forbidden.
    /// assert!(q.validate_scale_bytes(&[0x01, 0x00, 0x7F]).is_err());
    ///
    /// // 0xFF is forbidden.
    /// assert!(q.validate_scale_bytes(&[0x7F, 0xFF]).is_err());
    /// ```
    pub fn validate_scale_bytes(&self, bytes: &[u8]) -> Result<()> {
        for (i, &b) in bytes.iter().enumerate() {
            if b == 0x00 || b == 0xFF {
                return Err(Error::InvalidQuantization(format!(
                    "MXFP scale buffer byte at index {i} is 0x{b:02X}, which is a reserved \
                     float8_e8m0 NaN pattern (OCP MX v1.0 § 5.6); values 0x00 and 0xFF are \
                     forbidden in the scale buffer"
                )));
            }
        }
        Ok(())
    }

    /// Returns the set of storage [`ElementType`]s that are valid for this scheme.
    ///
    /// Per `docs/spec/quantization/mxfp.md § Valid Storage Types`.
    ///
    /// WHY `&'static [ElementType]`: no allocation per call; `slice::contains`
    /// over ≤10 items beats any hash structure (design decision #5).
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{ElementType, Mxfp};
    ///
    /// assert!(Mxfp::valid_storage_types().contains(&ElementType::Float8E4M3));
    /// assert!(Mxfp::valid_storage_types().contains(&ElementType::Int8));
    /// assert!(Mxfp::valid_storage_types().contains(&ElementType::Int4));
    /// assert!(!Mxfp::valid_storage_types().contains(&ElementType::Float32));
    /// ```
    pub fn valid_storage_types() -> &'static [ElementType] {
        &[
            ElementType::Float8E4M3, // MXFP8
            ElementType::Float8E5M2, // MXFP8
            ElementType::Float4E2M1, // MXFP4
            ElementType::Float6E2M3, // MXFP6
            ElementType::Float6E3M2, // MXFP6
            ElementType::Int8,       // MXINT8
            ElementType::Int4,       // MXINT4 / MXFP4 integer surrogate
        ]
    }

    // ── Crate-internal encode/decode ──────────────────────────────────────────

    /// Decodes the scheme-specific payload from `bytes`.
    ///
    /// `bytes` is the full descriptor slice (including the 4-byte header that the
    /// caller has already validated). `flags` are the header flags (must be zero
    /// for this scheme).
    pub(crate) fn decode_payload(flags: u16, bytes: &[u8]) -> Result<Self> {
        if bytes.len() < ENCODED_LEN {
            return Err(Error::QuantizationDescriptorTooShort {
                found: bytes.len(),
                needed: ENCODED_LEN,
            });
        }
        // No flags defined for MXFP.
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

        // block_size must be a power of two in [16, 2048].
        validate_block_size(block_size)?;

        // scale_buffer_index must not be the sentinel value 0xFFFFFFFF.
        if scale_buf == INVALID_BUF_SENTINEL {
            return Err(Error::InvalidQuantization(
                "MXFP scale_buffer_index must not be 0xFFFFFFFF".into(),
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
    /// header via [`super::QuantizationHeader::write`]; this method writes
    /// bytes 4–15.
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
    if !block_size.is_power_of_two() || !(MIN_BLOCK_SIZE..=MAX_BLOCK_SIZE).contains(&block_size) {
        return Err(Error::InvalidBlockSize {
            scheme_tag: SCHEME_TAG,
            block_size,
            min: MIN_BLOCK_SIZE,
            max: MAX_BLOCK_SIZE,
        });
    }
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ElementType, Error};

    // ── Mxfp::new ─────────────────────────────────────────────────────────────

    #[test]
    fn new_block_size_below_min_is_err() {
        // MIN_BLOCK_SIZE = 16; block_size = 8 is below.
        assert!(matches!(
            Mxfp::new(0, 8, 1),
            Err(Error::InvalidBlockSize { .. })
        ));
    }

    #[test]
    fn new_block_size_above_max_is_err() {
        // MAX_BLOCK_SIZE = 2048; block_size = 4096 is above.
        assert!(matches!(
            Mxfp::new(0, 4096, 1),
            Err(Error::InvalidBlockSize { .. })
        ));
    }

    #[test]
    fn new_block_size_not_power_of_two_is_err() {
        assert!(matches!(
            Mxfp::new(0, 48, 1),
            Err(Error::InvalidBlockSize { .. })
        ));
    }

    #[test]
    fn new_min_block_size_is_ok() {
        assert!(Mxfp::new(0, MIN_BLOCK_SIZE, 1).is_ok());
    }

    #[test]
    fn new_canonical_block_size_is_ok() {
        assert!(Mxfp::new(0, CANONICAL_BLOCK_SIZE, 1).is_ok());
    }

    #[test]
    fn new_max_block_size_is_ok() {
        assert!(Mxfp::new(0, MAX_BLOCK_SIZE, 1).is_ok());
    }

    // ── num_blocks_per_axis ───────────────────────────────────────────────────

    #[test]
    fn num_blocks_per_axis_zero_shape_is_err() {
        // MXFP requires a positive multiple — shape_axis=0 is always rejected.
        let q = Mxfp::new(0, 32, 1).unwrap();
        assert!(matches!(
            q.num_blocks_per_axis(0),
            Err(Error::QuantizationShapeMismatch { .. })
        ));
    }

    #[test]
    fn num_blocks_per_axis_exact_divisibility_is_ok() {
        let q = Mxfp::new(0, 32, 1).unwrap();
        assert_eq!(q.num_blocks_per_axis(64).unwrap(), 2);
    }

    #[test]
    fn num_blocks_per_axis_not_divisible_is_err() {
        // 65 is not a multiple of 32 — MXFP forbids partial trailing blocks.
        let q = Mxfp::new(0, 32, 1).unwrap();
        assert!(matches!(
            q.num_blocks_per_axis(65),
            Err(Error::QuantizationShapeMismatch { .. })
        ));
    }

    // ── validate_scale_bytes ──────────────────────────────────────────────────

    #[test]
    fn validate_scale_bytes_empty_slice_is_ok() {
        let q = Mxfp::new(0, 32, 1).unwrap();
        assert!(q.validate_scale_bytes(&[]).is_ok());
    }

    #[test]
    fn validate_scale_bytes_valid_bytes_is_ok() {
        let q = Mxfp::new(0, 32, 1).unwrap();
        assert!(q.validate_scale_bytes(&[0x01, 0x7F, 0x80, 0xFE]).is_ok());
    }

    #[test]
    fn validate_scale_bytes_zero_byte_is_err() {
        let q = Mxfp::new(0, 32, 1).unwrap();
        assert!(matches!(
            q.validate_scale_bytes(&[0x01, 0x00, 0x7F]),
            Err(Error::InvalidQuantization(_))
        ));
    }

    #[test]
    fn validate_scale_bytes_0xff_is_err() {
        let q = Mxfp::new(0, 32, 1).unwrap();
        assert!(matches!(
            q.validate_scale_bytes(&[0x7F, 0xFF]),
            Err(Error::InvalidQuantization(_))
        ));
    }

    #[test]
    fn validate_scale_bytes_mixed_valid_and_invalid_is_err() {
        let q = Mxfp::new(0, 32, 1).unwrap();
        // First byte is valid; second byte 0x00 is invalid.
        assert!(matches!(
            q.validate_scale_bytes(&[0x7F, 0x00]),
            Err(Error::InvalidQuantization(_))
        ));
    }

    // ── valid_storage_types ───────────────────────────────────────────────────

    #[test]
    fn valid_storage_types_contains_expected_float_and_int_types() {
        let types = Mxfp::valid_storage_types();
        assert!(types.contains(&ElementType::Float8E4M3));
        assert!(types.contains(&ElementType::Float8E5M2));
        assert!(types.contains(&ElementType::Float4E2M1));
        assert!(types.contains(&ElementType::Float6E2M3));
        assert!(types.contains(&ElementType::Float6E3M2));
        assert!(types.contains(&ElementType::Int8));
        assert!(types.contains(&ElementType::Int4));
    }

    #[test]
    fn valid_storage_types_does_not_contain_float32() {
        assert!(!Mxfp::valid_storage_types().contains(&ElementType::Float32));
    }

    // ── Round-trips ───────────────────────────────────────────────────────────

    fn encode_decode(q: &Mxfp) -> Mxfp {
        let mut buf = vec![0u8; ENCODED_LEN];
        buf[0] = SCHEME_TAG;
        buf[1] = SUPPORTED_VERSION;
        buf[2] = 0;
        buf[3] = 0;
        q.encode_payload(&mut buf);
        Mxfp::decode_payload(0, &buf).unwrap()
    }

    #[test]
    fn round_trip_canonical_block_size() {
        let original = Mxfp::new(0, CANONICAL_BLOCK_SIZE, 3).unwrap();
        let decoded = encode_decode(&original);
        assert_eq!(decoded, original);
    }

    #[test]
    fn round_trip_min_block_size() {
        let original = Mxfp::new(1, MIN_BLOCK_SIZE, 2).unwrap();
        let decoded = encode_decode(&original);
        assert_eq!(decoded, original);
    }

    #[test]
    fn round_trip_max_block_size() {
        let original = Mxfp::new(0, MAX_BLOCK_SIZE, 1).unwrap();
        let decoded = encode_decode(&original);
        assert_eq!(decoded, original);
    }
}
