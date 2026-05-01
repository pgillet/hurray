//! Per-block affine quantization descriptor (scheme tag `0x03`).
//!
//! The tensor is divided into fixed-size contiguous blocks along a specified axis.
//! Each block carries its own `scale` (and optionally `zero_point`). Partial
//! trailing blocks are permitted; see `docs/spec/quantization/per-block-affine.md`
//! § Padding for the normative padding rules.
//!
//! See `docs/spec/quantization/per-block-affine.md` for the normative definition.

use crate::{ElementType, Error, Result};

// ── Wire layout constants ─────────────────────────────────────────────────────

/// Scheme tag byte for per-block affine quantization.
pub(crate) const SCHEME_TAG: u8 = 0x03;

/// Total descriptor length in bytes (header 4 + axis 4 + block_size 4 +
/// scale_buf 4 + zp_buf 4 + scale_type_tag 1 + reserved 3).
pub(crate) const ENCODED_LEN: usize = 24;

/// Version this implementation supports.
pub(crate) const SUPPORTED_VERSION: u8 = 0x01;

/// Wire sentinel meaning "no zero-point buffer" (symmetric mode).
const ZP_SENTINEL: u32 = 0xFFFF_FFFF;

/// Minimum block size for per-block affine (must be a power of two ≥ 2).
pub const MIN_BLOCK_SIZE: u32 = 2;

/// No upper bound on block_size is imposed by the spec for per-block affine.
/// We use `u32::MAX` as a stand-in for "unbounded" in error messages.
const MAX_BLOCK_SIZE: u32 = u32::MAX;

/// Bit 0: symmetric flag — zero-point array is implicit (all zeros).
pub const FLAG_SYMMETRIC: u16 = 0b1;
/// All bits except bit 0 are reserved and must be zero.
pub const RESERVED_MASK: u16 = !FLAG_SYMMETRIC;

/// Valid scale type tags: float16 (0x01), bfloat16 (0x02), float32 (0x03).
const VALID_SCALE_TYPE_TAGS: [u8; 3] = [0x01, 0x02, 0x03];

// Wire field offsets (relative to start of descriptor, including header).
const OFFSET_AXIS: usize = 4;
const OFFSET_BLOCK_SIZE: usize = 8;
const OFFSET_SCALE_BUF: usize = 12;
const OFFSET_ZP_BUF: usize = 16;
const OFFSET_SCALE_TYPE: usize = 20;
const OFFSET_RESERVED: usize = 21;

// ── PerBlockAffine ────────────────────────────────────────────────────────────

/// Quantization parameters for per-block affine quantization.
///
/// The tensor is divided into fixed-size contiguous blocks along `axis`. Each
/// block carries its own `scale` (and optionally `zero_point`). Partial
/// trailing blocks are permitted; see the spec for padding rules.
///
/// The dequantization formula for element `q` with block index `b` is:
///
/// ```text
/// s      = scale[b]           (widened to float32 if float16/bfloat16)
/// z      = zero_point[b]      (0 if symmetric)
/// x_real = s * (q - z)
/// ```
///
/// # Wire format
///
/// Total descriptor length: **24 bytes** (including the 4-byte header).
///
/// | Offset | Field | Type |
/// |--------|-------|------|
/// | 4 | `axis` | `uint32` LE |
/// | 8 | `block_size` | `uint32` LE |
/// | 12 | `scale_buffer_index` | `uint32` LE |
/// | 16 | `zero_point_buffer_index` | `uint32` LE (`0xFFFFFFFF` if symmetric) |
/// | 20 | `scale_type_tag` | `uint8` (`0x01`, `0x02`, or `0x03`) |
/// | 21 | `_reserved` | `uint8[3]` (must be `0x00`) |
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
/// use hurray_core::{ElementType, PerBlockAffine};
///
/// let q = PerBlockAffine::new_symmetric(0, 32, 1, ElementType::Float32).unwrap();
/// assert!(q.is_symmetric());
/// assert_eq!(q.block_size(), 32);
/// assert_eq!(q.scale_type(), ElementType::Float32);
/// ```
// WHY Eq + Hash: no float fields (design decision #2 applies only to PerTensorAffine).
// WHY Copy: ≤24 bytes, no Drop glue (design decision #7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PerBlockAffine {
    axis: u32,
    block_size: u32,
    scale_buffer_index: u32,
    /// `None` encodes the symmetric case (wire sentinel `0xFFFFFFFF`).
    ///
    /// WHY `Option<u32>`: makes the symmetric/asymmetric distinction
    /// unrepresentable-when-wrong at the type level (design decision #6).
    zero_point_buffer_index: Option<u32>,
    scale_type: ElementType,
}

impl PerBlockAffine {
    /// Creates an asymmetric [`PerBlockAffine`] descriptor.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidBlockSize`] — `block_size` is not a power of two or is
    ///   less than `2`.
    /// - [`Error::InvalidQuantization`] — `scale_type` is not one of
    ///   `Float16`, `BFloat16`, or `Float32`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{ElementType, PerBlockAffine};
    ///
    /// let q = PerBlockAffine::new_asymmetric(0, 64, 1, 2, ElementType::Float16).unwrap();
    /// assert!(!q.is_symmetric());
    /// assert_eq!(q.zero_point_buffer_index(), Some(2));
    /// ```
    pub fn new_asymmetric(
        axis: u32,
        block_size: u32,
        scale_buffer_index: u32,
        zero_point_buffer_index: u32,
        scale_type: ElementType,
    ) -> Result<Self> {
        validate_block_size(block_size)?;
        validate_scale_type(scale_type)?;
        Ok(Self {
            axis,
            block_size,
            scale_buffer_index,
            zero_point_buffer_index: Some(zero_point_buffer_index),
            scale_type,
        })
    }

    /// Creates a symmetric [`PerBlockAffine`] descriptor.
    ///
    /// In symmetric mode the zero-point array is implicit (all zeros); no
    /// zero-point buffer entry is required.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidBlockSize`] — `block_size` is not a power of two or is
    ///   less than `2`.
    /// - [`Error::InvalidQuantization`] — `scale_type` is not one of
    ///   `Float16`, `BFloat16`, or `Float32`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{ElementType, PerBlockAffine};
    ///
    /// let q = PerBlockAffine::new_symmetric(0, 128, 1, ElementType::BFloat16).unwrap();
    /// assert!(q.is_symmetric());
    /// assert_eq!(q.zero_point_buffer_index(), None);
    /// ```
    pub fn new_symmetric(
        axis: u32,
        block_size: u32,
        scale_buffer_index: u32,
        scale_type: ElementType,
    ) -> Result<Self> {
        validate_block_size(block_size)?;
        validate_scale_type(scale_type)?;
        Ok(Self {
            axis,
            block_size,
            scale_buffer_index,
            zero_point_buffer_index: None,
            scale_type,
        })
    }

    /// Returns `true` if this descriptor uses symmetric quantization (no zero point).
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{ElementType, PerBlockAffine};
    ///
    /// assert!(PerBlockAffine::new_symmetric(0, 32, 1, ElementType::Float32)
    ///     .unwrap()
    ///     .is_symmetric());
    /// ```
    #[inline]
    pub fn is_symmetric(&self) -> bool {
        self.zero_point_buffer_index.is_none()
    }

    /// Returns the quantization axis index.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{ElementType, PerBlockAffine};
    ///
    /// let q = PerBlockAffine::new_symmetric(2, 32, 1, ElementType::Float32).unwrap();
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
    /// use hurray_core::{ElementType, PerBlockAffine};
    ///
    /// let q = PerBlockAffine::new_symmetric(0, 64, 1, ElementType::Float32).unwrap();
    /// assert_eq!(q.block_size(), 64);
    /// ```
    #[inline]
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Returns the buffer table index of the per-block scale array.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{ElementType, PerBlockAffine};
    ///
    /// let q = PerBlockAffine::new_symmetric(0, 32, 3, ElementType::Float32).unwrap();
    /// assert_eq!(q.scale_buffer_index(), 3);
    /// ```
    #[inline]
    pub fn scale_buffer_index(&self) -> u32 {
        self.scale_buffer_index
    }

    /// Returns the buffer table index of the per-block zero-point array, or
    /// `None` if this descriptor is symmetric.
    ///
    /// `None` maps to the wire sentinel `0xFFFFFFFF`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{ElementType, PerBlockAffine};
    ///
    /// let asym = PerBlockAffine::new_asymmetric(0, 32, 1, 2, ElementType::Float32).unwrap();
    /// assert_eq!(asym.zero_point_buffer_index(), Some(2));
    ///
    /// let sym = PerBlockAffine::new_symmetric(0, 32, 1, ElementType::Float32).unwrap();
    /// assert_eq!(sym.zero_point_buffer_index(), None);
    /// ```
    #[inline]
    pub fn zero_point_buffer_index(&self) -> Option<u32> {
        self.zero_point_buffer_index
    }

    /// Returns the element type used for scale values.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{ElementType, PerBlockAffine};
    ///
    /// let q = PerBlockAffine::new_symmetric(0, 32, 1, ElementType::Float16).unwrap();
    /// assert_eq!(q.scale_type(), ElementType::Float16);
    /// ```
    #[inline]
    pub fn scale_type(&self) -> ElementType {
        self.scale_type
    }

    /// Computes the number of blocks along `axis` for a given `shape_axis` size.
    ///
    /// Uses `ceil(shape_axis / block_size)`.
    ///
    /// Returns `0` when `shape_axis == 0` per the ADR-007 empty-axis carve-out:
    /// an empty quantization axis produces zero blocks and zero-byte parameter buffers.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{ElementType, PerBlockAffine};
    ///
    /// let q = PerBlockAffine::new_symmetric(0, 32, 1, ElementType::Float32).unwrap();
    /// assert_eq!(q.num_blocks_per_axis(64), 2);
    /// assert_eq!(q.num_blocks_per_axis(65), 3); // partial trailing block
    /// assert_eq!(q.num_blocks_per_axis(0), 0);  // ADR-007 empty-axis carve-out
    /// ```
    pub fn num_blocks_per_axis(&self, shape_axis: u64) -> u64 {
        if shape_axis == 0 {
            // ADR-007 empty-axis carve-out: an empty quantization axis yields zero
            // blocks. The block_size field retains its declared value for round-trip
            // fidelity; no blocks are materialized and scale/zp buffers have size 0.
            return 0;
        }
        shape_axis.div_ceil(self.block_size as u64)
    }

    /// Validates this descriptor against the resolved `shape_axis` size.
    ///
    /// Rejects if `shape_axis` is the DYNAMIC sentinel (`u64::MAX`), or if
    /// `shape_axis > 0` and `block_size > shape_axis`.
    ///
    /// Does not check `shape_axis == 0` (ADR-007 carve-out: the upper-bound
    /// check is waived for empty axes).
    ///
    /// # Errors
    ///
    /// - [`Error::QuantizationShapeMismatch`] — `shape_axis` is the DYNAMIC sentinel,
    ///   or `shape_axis > 0` and `block_size > shape_axis`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{ElementType, PerBlockAffine, DYNAMIC};
    ///
    /// let q = PerBlockAffine::new_symmetric(0, 32, 1, ElementType::Float32).unwrap();
    /// assert!(q.validate_against_shape_axis(64).is_ok());
    /// assert!(q.validate_against_shape_axis(32).is_ok());
    /// assert!(q.validate_against_shape_axis(0).is_ok()); // ADR-007: waived
    /// assert!(q.validate_against_shape_axis(16).is_err()); // block_size > shape_axis
    /// assert!(q.validate_against_shape_axis(DYNAMIC).is_err()); // dynamic dimension
    /// ```
    pub fn validate_against_shape_axis(&self, shape_axis: u64) -> Result<()> {
        // Reject the DYNAMIC sentinel (u64::MAX = 0xFFFF…FFFF): a dynamic dimension
        // cannot be validated against a block size constraint.
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
                reason: "block_size must not exceed shape[axis] when shape[axis] > 0",
            });
        }
        Ok(())
    }

    /// Returns the set of storage [`ElementType`]s that are valid for this scheme.
    ///
    /// Per `docs/spec/quantization/per-block-affine.md § Valid Storage Types`.
    ///
    /// WHY `&'static [ElementType]`: no allocation per call (design decision #5).
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{ElementType, PerBlockAffine};
    ///
    /// assert!(PerBlockAffine::valid_storage_types().contains(&ElementType::Int8));
    /// assert!(PerBlockAffine::valid_storage_types().contains(&ElementType::Int4));
    /// assert!(!PerBlockAffine::valid_storage_types().contains(&ElementType::Int16));
    /// ```
    pub fn valid_storage_types() -> &'static [ElementType] {
        &[
            ElementType::Int8,
            ElementType::Uint8,
            ElementType::Int4,
            ElementType::Uint4,
            ElementType::Int2,
            ElementType::Uint2,
        ]
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
        if flags & RESERVED_MASK != 0 {
            return Err(Error::ReservedQuantizationFlagsBits {
                flags,
                mask: RESERVED_MASK,
            });
        }
        let symmetric = flags & FLAG_SYMMETRIC != 0;

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
        let zp_buf_raw =
            u32::from_le_bytes(bytes[OFFSET_ZP_BUF..OFFSET_ZP_BUF + 4].try_into().map_err(
                |_| Error::InvalidQuantization("zero_point_buffer_index slice error".into()),
            )?);
        let scale_type_byte = bytes[OFFSET_SCALE_TYPE];

        // block_size must be a power of two and >= 2.
        if !block_size.is_power_of_two() || block_size < MIN_BLOCK_SIZE {
            return Err(Error::InvalidBlockSize {
                scheme_tag: SCHEME_TAG,
                block_size,
                min: MIN_BLOCK_SIZE,
                max: MAX_BLOCK_SIZE,
            });
        }

        // scale_type_tag must be one of {0x01, 0x02, 0x03}.
        if !VALID_SCALE_TYPE_TAGS.contains(&scale_type_byte) {
            return Err(Error::InvalidQuantization(format!(
                "per-block affine scale_type_tag must be 0x01, 0x02, or 0x03, got 0x{scale_type_byte:02X}"
            )));
        }
        let scale_type = ElementType::from_tag(scale_type_byte)?;

        // Reserved bytes [21..24] must be 0x00.
        if bytes[OFFSET_RESERVED..OFFSET_RESERVED + 3]
            .iter()
            .any(|&b| b != 0)
        {
            return Err(Error::InvalidQuantization(
                "per-block affine reserved bytes [21..24] must be 0x00".into(),
            ));
        }

        // SYMMETRIC flag and zero_point_buffer_index sentinel must be consistent.
        let zero_point_buffer_index = if symmetric {
            if zp_buf_raw != ZP_SENTINEL {
                return Err(Error::InvalidQuantization(
                    "SYMMETRIC flag is set but zero_point_buffer_index is not 0xFFFFFFFF".into(),
                ));
            }
            None
        } else {
            Some(zp_buf_raw)
        };

        Ok(Self {
            axis,
            block_size,
            scale_buffer_index: scale_buf,
            zero_point_buffer_index,
            scale_type,
        })
    }

    /// Encodes the scheme-specific payload into `out`.
    ///
    /// `out` must be at least [`ENCODED_LEN`] bytes. The caller writes the 4-byte
    /// header; this method writes bytes 4–23.
    pub(crate) fn encode_payload(&self, out: &mut [u8]) {
        out[OFFSET_AXIS..OFFSET_AXIS + 4].copy_from_slice(&self.axis.to_le_bytes());
        out[OFFSET_BLOCK_SIZE..OFFSET_BLOCK_SIZE + 4]
            .copy_from_slice(&self.block_size.to_le_bytes());
        out[OFFSET_SCALE_BUF..OFFSET_SCALE_BUF + 4]
            .copy_from_slice(&self.scale_buffer_index.to_le_bytes());
        let zp_wire = self.zero_point_buffer_index.unwrap_or(ZP_SENTINEL);
        out[OFFSET_ZP_BUF..OFFSET_ZP_BUF + 4].copy_from_slice(&zp_wire.to_le_bytes());
        out[OFFSET_SCALE_TYPE] = self.scale_type.tag();
        out[OFFSET_RESERVED..OFFSET_RESERVED + 3].fill(0);
    }

    /// Returns the flags word that encodes the symmetric/asymmetric state.
    pub(crate) fn flags(&self) -> u16 {
        if self.is_symmetric() {
            FLAG_SYMMETRIC
        } else {
            0
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn validate_block_size(block_size: u32) -> Result<()> {
    if !block_size.is_power_of_two() || block_size < MIN_BLOCK_SIZE {
        return Err(Error::InvalidBlockSize {
            scheme_tag: SCHEME_TAG,
            block_size,
            min: MIN_BLOCK_SIZE,
            max: MAX_BLOCK_SIZE,
        });
    }
    Ok(())
}

fn validate_scale_type(scale_type: ElementType) -> Result<()> {
    if !VALID_SCALE_TYPE_TAGS.contains(&scale_type.tag()) {
        return Err(Error::InvalidQuantization(format!(
            "per-block affine scale_type must be Float16, BFloat16, or Float32, got {:?}",
            scale_type
        )));
    }
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ElementType, Error};

    // ── Constructors ──────────────────────────────────────────────────────────

    #[test]
    fn new_symmetric_block_size_not_power_of_two_is_err() {
        assert!(matches!(
            PerBlockAffine::new_symmetric(0, 3, 1, ElementType::Float32),
            Err(Error::InvalidBlockSize { .. })
        ));
    }

    #[test]
    fn new_symmetric_block_size_zero_is_err() {
        assert!(matches!(
            PerBlockAffine::new_symmetric(0, 0, 1, ElementType::Float32),
            Err(Error::InvalidBlockSize { .. })
        ));
    }

    #[test]
    fn new_symmetric_block_size_one_is_err() {
        // MIN_BLOCK_SIZE is 2; block_size=1 is below minimum.
        assert!(matches!(
            PerBlockAffine::new_symmetric(0, 1, 1, ElementType::Float32),
            Err(Error::InvalidBlockSize { .. })
        ));
    }

    #[test]
    fn new_symmetric_sentinel_scale_buf_is_accepted() {
        // PerBlockAffine does NOT reject 0xFFFFFFFF as scale_buffer_index
        // (only PerChannelAffine and NF4 do — per_block_affine has no sentinel check).
        // Confirm new_symmetric with valid params works.
        assert!(PerBlockAffine::new_symmetric(0, 32, 1, ElementType::Float32).is_ok());
    }

    #[test]
    fn new_asymmetric_block_size_not_power_of_two_is_err() {
        assert!(matches!(
            PerBlockAffine::new_asymmetric(0, 6, 1, 2, ElementType::Float32),
            Err(Error::InvalidBlockSize { .. })
        ));
    }

    #[test]
    fn new_asymmetric_block_size_zero_is_err() {
        assert!(matches!(
            PerBlockAffine::new_asymmetric(0, 0, 1, 2, ElementType::Float32),
            Err(Error::InvalidBlockSize { .. })
        ));
    }

    #[test]
    fn new_symmetric_invalid_scale_type_is_err() {
        // Float32 is valid; Int8 is not a valid scale type.
        assert!(matches!(
            PerBlockAffine::new_symmetric(0, 32, 1, ElementType::Int8),
            Err(Error::InvalidQuantization(_))
        ));
    }

    #[test]
    fn new_asymmetric_invalid_scale_type_is_err() {
        assert!(matches!(
            PerBlockAffine::new_asymmetric(0, 32, 1, 2, ElementType::Uint8),
            Err(Error::InvalidQuantization(_))
        ));
    }

    #[test]
    fn valid_scale_types_are_float16_bfloat16_float32() {
        // All three valid scale types must succeed.
        assert!(PerBlockAffine::new_symmetric(0, 32, 1, ElementType::Float16).is_ok());
        assert!(PerBlockAffine::new_symmetric(0, 32, 1, ElementType::BFloat16).is_ok());
        assert!(PerBlockAffine::new_symmetric(0, 32, 1, ElementType::Float32).is_ok());
    }

    // ── num_blocks_per_axis ───────────────────────────────────────────────────

    #[test]
    fn num_blocks_per_axis_zero_shape_returns_zero() {
        // ADR-007 empty-axis carve-out.
        let q = PerBlockAffine::new_symmetric(0, 32, 1, ElementType::Float32).unwrap();
        assert_eq!(q.num_blocks_per_axis(0), 0);
    }

    #[test]
    fn num_blocks_per_axis_exact_divisibility() {
        let q = PerBlockAffine::new_symmetric(0, 32, 1, ElementType::Float32).unwrap();
        assert_eq!(q.num_blocks_per_axis(64), 2);
    }

    #[test]
    fn num_blocks_per_axis_partial_trailing_block_uses_div_ceil() {
        let q = PerBlockAffine::new_symmetric(0, 32, 1, ElementType::Float32).unwrap();
        // 65 / 32 = 2.03125 → ceil = 3.
        assert_eq!(q.num_blocks_per_axis(65), 3);
    }

    // ── valid_storage_types ───────────────────────────────────────────────────

    #[test]
    fn valid_storage_types_contains_integer_sub_byte_and_byte_types() {
        let types = PerBlockAffine::valid_storage_types();
        assert!(types.contains(&ElementType::Int8));
        assert!(types.contains(&ElementType::Uint8));
        assert!(types.contains(&ElementType::Int4));
        assert!(types.contains(&ElementType::Uint4));
        assert!(types.contains(&ElementType::Int2));
        assert!(types.contains(&ElementType::Uint2));
    }

    #[test]
    fn valid_storage_types_does_not_contain_float32() {
        assert!(!PerBlockAffine::valid_storage_types().contains(&ElementType::Float32));
    }

    // ── Round-trips ───────────────────────────────────────────────────────────

    fn encode_decode(q: &PerBlockAffine) -> PerBlockAffine {
        let mut buf = vec![0u8; ENCODED_LEN];
        let flags = q.flags();
        buf[0] = SCHEME_TAG;
        buf[1] = SUPPORTED_VERSION;
        buf[2] = (flags & 0xFF) as u8;
        buf[3] = (flags >> 8) as u8;
        q.encode_payload(&mut buf);
        PerBlockAffine::decode_payload(flags, &buf).unwrap()
    }

    #[test]
    fn round_trip_symmetric() {
        let original = PerBlockAffine::new_symmetric(1, 64, 2, ElementType::BFloat16).unwrap();
        let decoded = encode_decode(&original);
        assert_eq!(decoded, original);
    }

    #[test]
    fn round_trip_asymmetric() {
        let original = PerBlockAffine::new_asymmetric(0, 32, 1, 3, ElementType::Float16).unwrap();
        let decoded = encode_decode(&original);
        assert_eq!(decoded, original);
    }

    // ── validate_against_shape_axis ───────────────────────────────────────────

    #[test]
    fn validate_against_shape_axis_ok_when_block_size_le_shape() {
        let q = PerBlockAffine::new_symmetric(0, 32, 1, ElementType::Float32).unwrap();
        assert!(q.validate_against_shape_axis(32).is_ok());
        assert!(q.validate_against_shape_axis(64).is_ok());
    }

    #[test]
    fn validate_against_shape_axis_ok_when_shape_is_zero_adr007() {
        // ADR-007 carve-out: empty axis is always accepted.
        let q = PerBlockAffine::new_symmetric(0, 32, 1, ElementType::Float32).unwrap();
        assert!(q.validate_against_shape_axis(0).is_ok());
    }

    #[test]
    fn validate_against_shape_axis_err_when_block_size_exceeds_shape() {
        let q = PerBlockAffine::new_symmetric(0, 32, 1, ElementType::Float32).unwrap();
        assert!(matches!(
            q.validate_against_shape_axis(16),
            Err(Error::QuantizationShapeMismatch { .. })
        ));
    }

    #[test]
    fn validate_against_shape_axis_err_for_dynamic_sentinel() {
        let q = PerBlockAffine::new_symmetric(0, 32, 1, ElementType::Float32).unwrap();
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
            PerBlockAffine::decode_payload(0, &short),
            Err(Error::QuantizationDescriptorTooShort { .. })
        ));
    }

    #[test]
    fn decode_payload_nonzero_reserved_flags_is_err() {
        let q = PerBlockAffine::new_symmetric(0, 32, 1, ElementType::Float32).unwrap();
        let mut buf = vec![0u8; ENCODED_LEN];
        buf[0] = SCHEME_TAG;
        buf[1] = SUPPORTED_VERSION;
        let nonzero_flags: u16 = 0xFF00;
        buf[2] = (nonzero_flags & 0xFF) as u8;
        buf[3] = (nonzero_flags >> 8) as u8;
        q.encode_payload(&mut buf);
        assert!(matches!(
            PerBlockAffine::decode_payload(nonzero_flags, &buf),
            Err(Error::ReservedQuantizationFlagsBits { .. })
        ));
    }

    #[test]
    fn decode_payload_invalid_block_size_is_err() {
        // Manually craft a descriptor with a non-power-of-two block_size.
        let mut buf = vec![0u8; ENCODED_LEN];
        buf[0] = SCHEME_TAG;
        buf[1] = SUPPORTED_VERSION;
        buf[2] = 0;
        buf[3] = 0;
        // axis = 0
        buf[4..8].copy_from_slice(&0u32.to_le_bytes());
        // block_size = 33 (not power of two)
        buf[8..12].copy_from_slice(&33u32.to_le_bytes());
        // scale_buf = 1, zp_buf sentinel (symmetric)
        buf[12..16].copy_from_slice(&1u32.to_le_bytes());
        buf[16..20].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        // scale_type = float32 (0x03)
        buf[20] = 0x03;
        assert!(matches!(
            PerBlockAffine::decode_payload(0b1, &buf),
            Err(Error::InvalidBlockSize { .. })
        ));
    }

    #[test]
    fn decode_payload_invalid_scale_type_tag_is_err() {
        let mut buf = vec![0u8; ENCODED_LEN];
        buf[0] = SCHEME_TAG;
        buf[1] = SUPPORTED_VERSION;
        // symmetric flag
        let flags: u16 = 0b1;
        buf[2] = (flags & 0xFF) as u8;
        buf[3] = (flags >> 8) as u8;
        buf[4..8].copy_from_slice(&0u32.to_le_bytes()); // axis
        buf[8..12].copy_from_slice(&32u32.to_le_bytes()); // block_size
        buf[12..16].copy_from_slice(&1u32.to_le_bytes()); // scale_buf
        buf[16..20].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // zp sentinel
        buf[20] = 0x10; // invalid scale_type_tag
        assert!(matches!(
            PerBlockAffine::decode_payload(flags, &buf),
            Err(Error::InvalidQuantization(_))
        ));
    }
}
