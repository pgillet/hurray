//! Per-channel affine quantization descriptor (scheme tag `0x02`).
//!
//! One `scale` and `zero_point` pair per slice along a specified axis. The scale
//! and zero-point arrays are stored in separate buffers listed in the tensor
//! descriptor's buffer table.
//!
//! See `docs/spec/quantization/per-channel-affine.md` for the normative definition.

use crate::{ElementType, Error, Result};

// ── Wire layout constants ─────────────────────────────────────────────────────

/// Scheme tag byte for per-channel affine quantization.
#[allow(dead_code)]
pub(crate) const SCHEME_TAG: u8 = 0x02;

/// Total descriptor length in bytes (header 4 + axis 4 + scale_buf 4 + zp_buf 4
/// + scale_type_tag 1 + reserved 3).
pub(crate) const ENCODED_LEN: usize = 20;

/// Version this implementation supports.
pub(crate) const SUPPORTED_VERSION: u8 = 0x01;

/// Wire sentinel meaning "no zero-point buffer" (symmetric mode).
const ZP_SENTINEL: u32 = 0xFFFF_FFFF;

/// `scale_type_tag` is locked to `float32` (`0x03`) for `scheme_version = 0x01`.
const SCALE_TYPE_TAG_FLOAT32: u8 = 0x03;

/// Bit 0: symmetric flag — zero-point array is implicit (all zeros).
pub const FLAG_SYMMETRIC: u16 = 0b1;
/// All bits except bit 0 are reserved and must be zero.
pub const RESERVED_MASK: u16 = !FLAG_SYMMETRIC;

// Wire field offsets (relative to start of descriptor, including header).
const OFFSET_AXIS: usize = 4;
const OFFSET_SCALE_BUF: usize = 8;
const OFFSET_ZP_BUF: usize = 12;
const OFFSET_SCALE_TYPE: usize = 16;
const OFFSET_RESERVED: usize = 17;

// ── PerChannelAffine ──────────────────────────────────────────────────────────

/// Quantization parameters for per-channel affine quantization.
///
/// Each slice along `axis` has its own `scale` (and optionally `zero_point`).
/// The parameter arrays are stored in separate buffers referenced by index in
/// the tensor descriptor's buffer table.
///
/// The dequantization formula for element `q` at logical index
/// `[i_0, …, i_{rank-1}]` is:
///
/// ```text
/// c      = i_axis
/// x_real = scale[c] * (q - zero_point[c])
/// ```
///
/// When the `SYMMETRIC` flag is set, `zero_point[c]` is treated as `0`.
///
/// # Wire format
///
/// Total descriptor length: **20 bytes** (including the 4-byte header).
///
/// | Offset | Field | Type |
/// |--------|-------|------|
/// | 4 | `axis` | `uint32` LE |
/// | 8 | `scale_buffer_index` | `uint32` LE |
/// | 12 | `zero_point_buffer_index` | `uint32` LE (`0xFFFFFFFF` if symmetric) |
/// | 16 | `scale_type_tag` | `uint8` (must be `0x03` in v1) |
/// | 17 | `_reserved` | `uint8[3]` (must be `0x00`) |
///
/// # Design notes
///
/// `PartialEq`, `Eq`, and `Hash` are all derived because this struct contains
/// no floating-point fields — all fields are integers or `Option<u32>`.
///
/// `Copy` because the struct is ≤ 24 bytes with no `Drop` glue.
///
/// # Examples
///
/// ```
/// use hurray_core::PerChannelAffine;
///
/// // Asymmetric: separate scale (buffer 1) and zero-point (buffer 2) arrays.
/// let q = PerChannelAffine::new_asymmetric(0, 1, 2).unwrap();
/// assert!(!q.is_symmetric());
/// assert_eq!(q.zero_point_buffer_index(), Some(2));
///
/// // Symmetric: only a scale buffer, zero-point is implicitly zero.
/// let s = PerChannelAffine::new_symmetric(0, 1).unwrap();
/// assert!(s.is_symmetric());
/// assert_eq!(s.zero_point_buffer_index(), None);
/// ```
// WHY Eq + Hash: no float fields, so the IEEE 754 NaN problem does not apply.
// WHY Copy: ≤24 bytes, no Drop glue (design decision #7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PerChannelAffine {
    axis: u32,
    scale_buffer_index: u32,
    /// `None` encodes the symmetric case (wire sentinel `0xFFFFFFFF`).
    ///
    /// WHY `Option<u32>`: `None` maps to wire sentinel `0xFFFFFFFF`; makes the
    /// symmetric/asymmetric distinction unrepresentable-when-wrong at the type
    /// level (design decision #6).
    zero_point_buffer_index: Option<u32>,
    scale_type: ElementType,
}

impl PerChannelAffine {
    /// Creates an asymmetric [`PerChannelAffine`] descriptor.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidQuantization`] — `scale_buf` equals `0xFFFFFFFF`
    ///   (the zero-point sentinel value, which is not a valid scale buffer index).
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::PerChannelAffine;
    ///
    /// let q = PerChannelAffine::new_asymmetric(0, 1, 2).unwrap();
    /// assert!(!q.is_symmetric());
    /// assert_eq!(q.scale_buffer_index(), 1);
    /// assert_eq!(q.zero_point_buffer_index(), Some(2));
    /// ```
    pub fn new_asymmetric(
        axis: u32,
        scale_buffer_index: u32,
        zero_point_buffer_index: u32,
    ) -> Result<Self> {
        if scale_buffer_index == ZP_SENTINEL {
            return Err(Error::InvalidQuantization(
                "scale_buffer_index must not be 0xFFFFFFFF (reserved sentinel)".into(),
            ));
        }
        Ok(Self {
            axis,
            scale_buffer_index,
            zero_point_buffer_index: Some(zero_point_buffer_index),
            scale_type: ElementType::Float32,
        })
    }

    /// Creates a symmetric [`PerChannelAffine`] descriptor.
    ///
    /// In symmetric mode the zero-point array is implicit (all zeros); no
    /// zero-point buffer entry is required in the tensor descriptor's buffer table.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidQuantization`] — `scale_buf` equals `0xFFFFFFFF`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::PerChannelAffine;
    ///
    /// let q = PerChannelAffine::new_symmetric(1, 2).unwrap();
    /// assert!(q.is_symmetric());
    /// assert_eq!(q.zero_point_buffer_index(), None);
    /// ```
    pub fn new_symmetric(axis: u32, scale_buffer_index: u32) -> Result<Self> {
        if scale_buffer_index == ZP_SENTINEL {
            return Err(Error::InvalidQuantization(
                "scale_buffer_index must not be 0xFFFFFFFF (reserved sentinel)".into(),
            ));
        }
        Ok(Self {
            axis,
            scale_buffer_index,
            zero_point_buffer_index: None,
            scale_type: ElementType::Float32,
        })
    }

    /// Returns `true` if this descriptor uses symmetric quantization (no zero point).
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::PerChannelAffine;
    ///
    /// assert!(PerChannelAffine::new_symmetric(0, 1).unwrap().is_symmetric());
    /// assert!(!PerChannelAffine::new_asymmetric(0, 1, 2).unwrap().is_symmetric());
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
    /// use hurray_core::PerChannelAffine;
    ///
    /// let q = PerChannelAffine::new_symmetric(3, 1).unwrap();
    /// assert_eq!(q.axis(), 3);
    /// ```
    #[inline]
    pub fn axis(&self) -> u32 {
        self.axis
    }

    /// Returns the buffer table index of the per-channel scale array.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::PerChannelAffine;
    ///
    /// let q = PerChannelAffine::new_symmetric(0, 5).unwrap();
    /// assert_eq!(q.scale_buffer_index(), 5);
    /// ```
    #[inline]
    pub fn scale_buffer_index(&self) -> u32 {
        self.scale_buffer_index
    }

    /// Returns the buffer table index of the per-channel zero-point array, or
    /// `None` if this descriptor is symmetric.
    ///
    /// `None` maps to the wire sentinel `0xFFFFFFFF`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::PerChannelAffine;
    ///
    /// let asym = PerChannelAffine::new_asymmetric(0, 1, 2).unwrap();
    /// assert_eq!(asym.zero_point_buffer_index(), Some(2));
    ///
    /// let sym = PerChannelAffine::new_symmetric(0, 1).unwrap();
    /// assert_eq!(sym.zero_point_buffer_index(), None);
    /// ```
    #[inline]
    pub fn zero_point_buffer_index(&self) -> Option<u32> {
        self.zero_point_buffer_index
    }

    /// Returns the element type used for scale values (always `Float32` in v1).
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{ElementType, PerChannelAffine};
    ///
    /// let q = PerChannelAffine::new_symmetric(0, 1).unwrap();
    /// assert_eq!(q.scale_type(), ElementType::Float32);
    /// ```
    #[inline]
    pub fn scale_type(&self) -> ElementType {
        self.scale_type
    }

    /// Returns the set of storage [`ElementType`]s that are valid for this scheme.
    ///
    /// Per `docs/spec/quantization/per-channel-affine.md § Valid Storage Types`.
    ///
    /// WHY `&'static [ElementType]`: no allocation per call; `slice::contains`
    /// over ≤10 items beats any hash structure (design decision #5).
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{ElementType, PerChannelAffine};
    ///
    /// assert!(PerChannelAffine::valid_storage_types().contains(&ElementType::Int8));
    /// assert!(!PerChannelAffine::valid_storage_types().contains(&ElementType::Float32));
    /// ```
    pub fn valid_storage_types() -> &'static [ElementType] {
        &[
            ElementType::Int8,
            ElementType::Uint8,
            ElementType::Int16,
            ElementType::Uint16,
            ElementType::Int32,
            ElementType::Uint32,
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
        // Validate that no reserved flag bits are set.
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

        // scale_buffer_index must not be the sentinel.
        if scale_buf == ZP_SENTINEL {
            return Err(Error::InvalidQuantization(
                "scale_buffer_index must not be 0xFFFFFFFF".into(),
            ));
        }

        // scale_type_tag must be 0x03 (float32) for scheme_version 0x01.
        if scale_type_byte != SCALE_TYPE_TAG_FLOAT32 {
            return Err(Error::InvalidQuantization(format!(
                "per-channel affine scale_type_tag must be 0x03 (float32) in v1, got 0x{scale_type_byte:02X}"
            )));
        }

        // Reserved bytes [17..20] must be 0x00.
        if bytes[OFFSET_RESERVED..OFFSET_RESERVED + 3]
            .iter()
            .any(|&b| b != 0)
        {
            return Err(Error::InvalidQuantization(
                "per-channel affine reserved bytes [17..20] must be 0x00".into(),
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
            if zp_buf_raw == ZP_SENTINEL {
                return Err(Error::InvalidQuantization(
                    "SYMMETRIC flag is not set but zero_point_buffer_index is 0xFFFFFFFF".into(),
                ));
            }
            Some(zp_buf_raw)
        };

        Ok(Self {
            axis,
            scale_buffer_index: scale_buf,
            zero_point_buffer_index,
            scale_type: ElementType::Float32,
        })
    }

    /// Encodes the scheme-specific payload into `out`.
    ///
    /// `out` must be at least [`ENCODED_LEN`] bytes. The caller writes the 4-byte
    /// header; this method writes bytes 4–19.
    pub(crate) fn encode_payload(&self, out: &mut [u8]) {
        out[OFFSET_AXIS..OFFSET_AXIS + 4].copy_from_slice(&self.axis.to_le_bytes());
        out[OFFSET_SCALE_BUF..OFFSET_SCALE_BUF + 4]
            .copy_from_slice(&self.scale_buffer_index.to_le_bytes());
        let zp_wire = self.zero_point_buffer_index.unwrap_or(ZP_SENTINEL);
        out[OFFSET_ZP_BUF..OFFSET_ZP_BUF + 4].copy_from_slice(&zp_wire.to_le_bytes());
        out[OFFSET_SCALE_TYPE] = SCALE_TYPE_TAG_FLOAT32;
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ElementType, Error};

    // ── Constructors ──────────────────────────────────────────────────────────

    #[test]
    fn new_symmetric_sentinel_scale_buf_is_err() {
        // scale_buffer_index == 0xFFFFFFFF is the ZP sentinel — must be rejected.
        assert!(matches!(
            PerChannelAffine::new_symmetric(0, 0xFFFF_FFFF),
            Err(Error::InvalidQuantization(_))
        ));
    }

    #[test]
    fn new_asymmetric_sentinel_scale_buf_is_err() {
        assert!(matches!(
            PerChannelAffine::new_asymmetric(0, 0xFFFF_FFFF, 1),
            Err(Error::InvalidQuantization(_))
        ));
    }

    #[test]
    fn new_symmetric_is_symmetric() {
        let q = PerChannelAffine::new_symmetric(0, 1).unwrap();
        assert!(q.is_symmetric());
        assert_eq!(q.zero_point_buffer_index(), None);
    }

    #[test]
    fn new_asymmetric_is_not_symmetric() {
        let q = PerChannelAffine::new_asymmetric(0, 1, 2).unwrap();
        assert!(!q.is_symmetric());
        assert_eq!(q.zero_point_buffer_index(), Some(2));
    }

    #[test]
    fn scale_type_is_always_float32() {
        let q = PerChannelAffine::new_symmetric(0, 1).unwrap();
        assert_eq!(q.scale_type(), ElementType::Float32);
    }

    // ── Round-trips ───────────────────────────────────────────────────────────

    fn encode_decode(q: &PerChannelAffine) -> PerChannelAffine {
        let mut buf = vec![0u8; ENCODED_LEN];
        let flags = q.flags();
        // Write header: scheme_tag=0x02, version=0x01, flags LE.
        buf[0] = SCHEME_TAG;
        buf[1] = SUPPORTED_VERSION;
        buf[2] = (flags & 0xFF) as u8;
        buf[3] = (flags >> 8) as u8;
        q.encode_payload(&mut buf);
        PerChannelAffine::decode_payload(flags, &buf).unwrap()
    }

    #[test]
    fn round_trip_symmetric() {
        let original = PerChannelAffine::new_symmetric(3, 5).unwrap();
        let decoded = encode_decode(&original);
        assert_eq!(decoded, original);
    }

    #[test]
    fn round_trip_asymmetric() {
        let original = PerChannelAffine::new_asymmetric(1, 2, 4).unwrap();
        let decoded = encode_decode(&original);
        assert_eq!(decoded, original);
    }

    // ── decode_payload error cases ────────────────────────────────────────────

    #[test]
    fn decode_payload_too_short_is_err() {
        let short = vec![0u8; ENCODED_LEN - 1];
        assert!(matches!(
            PerChannelAffine::decode_payload(0, &short),
            Err(Error::QuantizationDescriptorTooShort { .. })
        ));
    }

    #[test]
    fn decode_payload_reserved_flag_bits_is_err() {
        let mut buf = vec![0u8; ENCODED_LEN];
        buf[0] = SCHEME_TAG;
        buf[1] = SUPPORTED_VERSION;
        // Write a valid scale_buffer_index.
        buf[8..12].copy_from_slice(&1u32.to_le_bytes());
        // Write ZP sentinel (symmetric).
        buf[12..16].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        buf[16] = 0x03; // valid scale_type_tag
                        // flags = 0x0002 — reserved bit.
        assert!(matches!(
            PerChannelAffine::decode_payload(0x0002, &buf),
            Err(Error::ReservedQuantizationFlagsBits { .. })
        ));
    }

    #[test]
    fn decode_payload_sentinel_scale_buf_is_err() {
        let mut buf = vec![0u8; ENCODED_LEN];
        buf[0] = SCHEME_TAG;
        buf[1] = SUPPORTED_VERSION;
        // scale_buffer_index = 0xFFFFFFFF.
        buf[8..12].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        buf[12..16].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        buf[16] = 0x03;
        assert!(matches!(
            PerChannelAffine::decode_payload(FLAG_SYMMETRIC, &buf),
            Err(Error::InvalidQuantization(_))
        ));
    }

    #[test]
    fn decode_payload_wrong_scale_type_tag_is_err() {
        let mut buf = vec![0u8; ENCODED_LEN];
        buf[0] = SCHEME_TAG;
        buf[1] = SUPPORTED_VERSION;
        buf[8..12].copy_from_slice(&1u32.to_le_bytes());
        buf[12..16].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        // scale_type_tag = 0x01 (float16) — must be 0x03 in v1.
        buf[16] = 0x01;
        assert!(matches!(
            PerChannelAffine::decode_payload(FLAG_SYMMETRIC, &buf),
            Err(Error::InvalidQuantization(_))
        ));
    }

    #[test]
    fn decode_payload_symmetric_flag_set_but_zp_not_sentinel_is_err() {
        let mut buf = vec![0u8; ENCODED_LEN];
        buf[0] = SCHEME_TAG;
        buf[1] = SUPPORTED_VERSION;
        buf[8..12].copy_from_slice(&1u32.to_le_bytes());
        // zp_buf = 2 (not sentinel), but SYMMETRIC flag is set — inconsistent.
        buf[12..16].copy_from_slice(&2u32.to_le_bytes());
        buf[16] = 0x03;
        assert!(matches!(
            PerChannelAffine::decode_payload(FLAG_SYMMETRIC, &buf),
            Err(Error::InvalidQuantization(_))
        ));
    }

    #[test]
    fn decode_payload_symmetric_flag_not_set_but_zp_sentinel_is_err() {
        let mut buf = vec![0u8; ENCODED_LEN];
        buf[0] = SCHEME_TAG;
        buf[1] = SUPPORTED_VERSION;
        buf[8..12].copy_from_slice(&1u32.to_le_bytes());
        // zp_buf = sentinel, but SYMMETRIC flag is NOT set — inconsistent.
        buf[12..16].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        buf[16] = 0x03;
        // flags = 0 (no SYMMETRIC bit).
        assert!(matches!(
            PerChannelAffine::decode_payload(0, &buf),
            Err(Error::InvalidQuantization(_))
        ));
    }

    #[test]
    fn decode_payload_nonzero_reserved_bytes_is_err() {
        let mut buf = vec![0u8; ENCODED_LEN];
        buf[0] = SCHEME_TAG;
        buf[1] = SUPPORTED_VERSION;
        buf[8..12].copy_from_slice(&1u32.to_le_bytes());
        buf[12..16].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        buf[16] = 0x03;
        // Pollute reserved byte [17].
        buf[17] = 0xAB;
        assert!(matches!(
            PerChannelAffine::decode_payload(FLAG_SYMMETRIC, &buf),
            Err(Error::InvalidQuantization(_))
        ));
    }
}
