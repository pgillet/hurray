//! # hurray-core
//!
//! Core types for the hurray tensor interchange format.
//!
//! This crate provides the format types, tensor descriptor, buffer handle, and
//! quantization descriptors. It has no I/O and no async dependencies — it is the
//! foundation for all other hurray crates.
//!
//! ## Feature flags
//!
//! | Feature | Effect |
//! |---------|--------|
//! | `serde` | Derives `serde::Serialize` / `serde::Deserialize` for all public types |

pub mod buffer;
pub mod descriptor;
pub mod element_type;
pub mod error;
pub mod layout;
pub mod quantization;
pub mod shape;

pub use buffer::{
    validate_colocation, BufferHandle, DeviceTag, PrivateTag, MIN_BUFFER_ALIGNMENT, PAGE_ALIGNMENT,
};
pub use descriptor::{
    DescriptorFlags, ExtensionTypeDescriptor, ShardDescriptor, Statistics, StatisticsMask,
    TensorDescriptor, DESCRIPTOR_VERSION_MAJOR, DESCRIPTOR_VERSION_MINOR, MAGIC,
};
pub use element_type::ElementType;
pub use error::{Error, Result};
pub use layout::{
    byte_address_from_element_offset, ElementAddress, LayoutDescriptor, SubpavingLocation,
};
pub use quantization::{
    validate_axis, validate_buffer_placement, Mxfp, Nf4, PerBlockAffine, PerChannelAffine,
    PerTensorAffine, QuantizationDescriptor, QuantizationSchemeTag, NF4_LUT,
};
pub use shape::{Shape, DYNAMIC, MAX_RANK};

/// Returns the minimum buffer size in bytes required to store `element_count`
/// contiguous elements of the given type.
///
/// This implements the buffer-size formulas defined in
/// `docs/spec/element-types.md § Buffer Size Calculation`:
///
/// - Whole-byte types (`bit_width ≥ 8`): `element_count × (bit_width / 8)`
/// - 6-bit types (`float6_e2m3`, `float6_e3m2`): `⌈N / 4⌉ × 3`
/// - Other sub-byte types (`bit_width ∈ {1, 2, 4}`): `⌈N × bit_width / 8⌉`
///
/// Uses 128-bit intermediate arithmetic to avoid overflow for large `element_count`.
///
/// # Examples
///
/// ```
/// use hurray_core::{ElementType, buffer_size_bytes};
///
/// // float32: 60 elements × 4 bytes = 240 bytes
/// assert_eq!(buffer_size_bytes(ElementType::Float32, 60), 240);
///
/// // int4: 7 elements → ⌈7/2⌉ = 4 bytes
/// assert_eq!(buffer_size_bytes(ElementType::Int4, 7), 4);
///
/// // float6_e2m3: 5 elements → ⌈5/4⌉ × 3 = 6 bytes
/// assert_eq!(buffer_size_bytes(ElementType::Float6E2M3, 5), 6);
///
/// // bool: 9 elements → ⌈9/8⌉ = 2 bytes
/// assert_eq!(buffer_size_bytes(ElementType::Bool, 9), 2);
///
/// // Zero elements always yields zero bytes.
/// assert_eq!(buffer_size_bytes(ElementType::Float32, 0), 0);
/// ```
pub fn buffer_size_bytes(ty: ElementType, element_count: u64) -> u64 {
    let bits = ty.bit_width() as u64;
    if bits >= 8 {
        // Whole-byte type: exact multiplication, no rounding needed.
        element_count * (bits / 8)
    } else if bits == 6 {
        // 6-bit packing: 4 elements per 3 bytes (ceil(N/4)*3).
        element_count.div_ceil(4) * 3
    } else {
        // Sub-byte power-of-two packing (bits ∈ {1, 2, 4}): ceil(N*bits/8).
        (element_count as u128 * bits as u128).div_ceil(8) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Whole-byte types ─────────────────────────────────────────────────────

    /// Spec § element-types Buffer Size Calculation:
    /// whole-byte types: element_count × (bit_width / 8).
    #[test]
    fn buffer_size_float32_60_elements() {
        // 60 × 4 bytes = 240
        assert_eq!(buffer_size_bytes(ElementType::Float32, 60), 240);
    }

    #[test]
    fn buffer_size_int64_1_element() {
        // 1 × 8 bytes = 8
        assert_eq!(buffer_size_bytes(ElementType::Int64, 1), 8);
    }

    #[test]
    fn buffer_size_float128_2_elements() {
        // 2 × 16 bytes = 32
        assert_eq!(buffer_size_bytes(ElementType::Float128, 2), 32);
    }

    #[test]
    fn buffer_size_uint8_5_elements() {
        // 5 × 1 byte = 5
        assert_eq!(buffer_size_bytes(ElementType::Uint8, 5), 5);
    }

    #[test]
    fn buffer_size_float16_4_elements() {
        // 4 × 2 bytes = 8
        assert_eq!(buffer_size_bytes(ElementType::Float16, 4), 8);
    }

    #[test]
    fn buffer_size_complex64_3_elements() {
        // Complex64 = 64 bits = 8 bytes; 3 × 8 = 24
        assert_eq!(buffer_size_bytes(ElementType::Complex64, 3), 24);
    }

    #[test]
    fn buffer_size_complex128_1_element() {
        // Complex128 = 128 bits = 16 bytes; 1 × 16 = 16
        assert_eq!(buffer_size_bytes(ElementType::Complex128, 1), 16);
    }

    // ── Sub-byte power-of-two types (bits ∈ {1, 2, 4}) ──────────────────────

    /// Spec § element-types Buffer Size Calculation:
    /// sub-byte power-of-two: ⌈N × bit_width / 8⌉.

    // Bool (1 bit)
    #[test]
    fn buffer_size_bool_9_elements() {
        // ⌈9 × 1 / 8⌉ = ⌈9/8⌉ = 2
        assert_eq!(buffer_size_bytes(ElementType::Bool, 9), 2);
    }

    #[test]
    fn buffer_size_bool_8_elements_exact_byte() {
        // ⌈8 × 1 / 8⌉ = 1
        assert_eq!(buffer_size_bytes(ElementType::Bool, 8), 1);
    }

    #[test]
    fn buffer_size_bool_1_element() {
        // ⌈1/8⌉ = 1
        assert_eq!(buffer_size_bytes(ElementType::Bool, 1), 1);
    }

    #[test]
    fn buffer_size_bool_16_elements() {
        // ⌈16/8⌉ = 2
        assert_eq!(buffer_size_bytes(ElementType::Bool, 16), 2);
    }

    // Int4 / Uint4 (4 bits)
    #[test]
    fn buffer_size_int4_7_elements() {
        // ⌈7 × 4 / 8⌉ = ⌈28/8⌉ = 4 (rounds up from 3.5)
        assert_eq!(buffer_size_bytes(ElementType::Int4, 7), 4);
    }

    #[test]
    fn buffer_size_int4_8_elements_exact_byte() {
        // ⌈8 × 4 / 8⌉ = 4 (exactly 4 bytes)
        assert_eq!(buffer_size_bytes(ElementType::Int4, 8), 4);
    }

    #[test]
    fn buffer_size_uint4_1_element() {
        // ⌈1 × 4 / 8⌉ = 1
        assert_eq!(buffer_size_bytes(ElementType::Uint4, 1), 1);
    }

    #[test]
    fn buffer_size_uint4_2_elements() {
        // ⌈2 × 4 / 8⌉ = 1 (exactly 1 byte)
        assert_eq!(buffer_size_bytes(ElementType::Uint4, 2), 1);
    }

    // Int2 / Uint2 (2 bits)
    #[test]
    fn buffer_size_int2_5_elements() {
        // ⌈5 × 2 / 8⌉ = ⌈10/8⌉ = 2
        assert_eq!(buffer_size_bytes(ElementType::Int2, 5), 2);
    }

    #[test]
    fn buffer_size_uint2_4_elements_exact_byte() {
        // ⌈4 × 2 / 8⌉ = 1 (exactly 1 byte)
        assert_eq!(buffer_size_bytes(ElementType::Uint2, 4), 1);
    }

    // Float4E2M1 (4 bits)
    #[test]
    fn buffer_size_float4e2m1_3_elements() {
        // ⌈3 × 4 / 8⌉ = ⌈12/8⌉ = 2
        assert_eq!(buffer_size_bytes(ElementType::Float4E2M1, 3), 2);
    }

    // ── 6-bit packing (Float6E2M3, Float6E3M2) ───────────────────────────────

    /// Spec § element-types Buffer Size Calculation:
    /// 6-bit types: ⌈N / 4⌉ × 3  (4 elements pack into 3 bytes).

    #[test]
    fn buffer_size_float6e2m3_4_elements_exact_group() {
        // ⌈4/4⌉ × 3 = 1 × 3 = 3
        assert_eq!(buffer_size_bytes(ElementType::Float6E2M3, 4), 3);
    }

    #[test]
    fn buffer_size_float6e2m3_5_elements_partial_group() {
        // ⌈5/4⌉ × 3 = 2 × 3 = 6
        assert_eq!(buffer_size_bytes(ElementType::Float6E2M3, 5), 6);
    }

    #[test]
    fn buffer_size_float6e2m3_8_elements_two_full_groups() {
        // ⌈8/4⌉ × 3 = 2 × 3 = 6
        assert_eq!(buffer_size_bytes(ElementType::Float6E2M3, 8), 6);
    }

    #[test]
    fn buffer_size_float6e2m3_9_elements_partial_third_group() {
        // ⌈9/4⌉ × 3 = 3 × 3 = 9
        assert_eq!(buffer_size_bytes(ElementType::Float6E2M3, 9), 9);
    }

    #[test]
    fn buffer_size_float6e3m2_4_elements() {
        // ⌈4/4⌉ × 3 = 3
        assert_eq!(buffer_size_bytes(ElementType::Float6E3M2, 4), 3);
    }

    #[test]
    fn buffer_size_float6e3m2_1_element() {
        // ⌈1/4⌉ × 3 = 1 × 3 = 3
        assert_eq!(buffer_size_bytes(ElementType::Float6E3M2, 1), 3);
    }

    // ── Zero elements ────────────────────────────────────────────────────────

    /// Zero elements must always produce 0 bytes, regardless of element type.
    #[test]
    fn buffer_size_zero_elements_whole_byte_type() {
        assert_eq!(buffer_size_bytes(ElementType::Float32, 0), 0);
    }

    #[test]
    fn buffer_size_zero_elements_bool() {
        assert_eq!(buffer_size_bytes(ElementType::Bool, 0), 0);
    }

    #[test]
    fn buffer_size_zero_elements_int4() {
        assert_eq!(buffer_size_bytes(ElementType::Int4, 0), 0);
    }

    #[test]
    fn buffer_size_zero_elements_int2() {
        assert_eq!(buffer_size_bytes(ElementType::Int2, 0), 0);
    }

    #[test]
    fn buffer_size_zero_elements_float6e2m3() {
        assert_eq!(buffer_size_bytes(ElementType::Float6E2M3, 0), 0);
    }

    // ── Large element counts (no overflow) ───────────────────────────────────

    /// A large element count for a byte-wide type must not overflow u64.
    #[test]
    fn buffer_size_large_count_uint8_no_overflow() {
        // u64::MAX elements of uint8 (1 byte each) = u64::MAX bytes
        assert_eq!(buffer_size_bytes(ElementType::Uint8, u64::MAX), u64::MAX);
    }

    /// A large element count for a 4-bit type uses 128-bit intermediate
    /// arithmetic so must not overflow.
    #[test]
    fn buffer_size_large_count_int4_no_overflow() {
        // 1 << 62 elements × 4 bits = 1 << 64 bits / 8 = 1 << 61 bytes — fits in u64
        let n = 1u64 << 62;
        let expected = n / 2; // ⌈n*4/8⌉ = n/2 (n is even)
        assert_eq!(buffer_size_bytes(ElementType::Int4, n), expected);
    }

    /// A large element count for bool uses 128-bit intermediates.
    #[test]
    fn buffer_size_large_count_bool_no_overflow() {
        // 1 << 63 bools = (1 << 63) / 8 bytes = 1 << 60 bytes — fits in u64
        let n = 1u64 << 63;
        let expected = n / 8;
        assert_eq!(buffer_size_bytes(ElementType::Bool, n), expected);
    }
}
