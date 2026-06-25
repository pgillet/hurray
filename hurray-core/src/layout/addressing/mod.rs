//! Layout address-computation traits and shared helpers.
//!
//! Implements the element-address formulas defined in
//! `docs/spec/memory-layout.md § Element Address Computation` and per-layout
//! spec files under `docs/spec/layouts/`.

pub mod block_paged;
pub mod col_major;
pub mod coo;
pub mod csc;
pub mod csr;
pub mod hilbert;
pub mod morton;
pub mod row_major;
pub mod strided;
pub mod subpaving;
pub mod tiled;

pub use subpaving::SubpavingLocation;

use crate::{ElementType, Error, Result, Shape, DYNAMIC};

/// Implemented by every dense layout descriptor.
///
/// Converts a multi-dimensional logical index into a linear element offset.
/// The offset is relative to `byte_offset` in the tensor descriptor, and may
/// be negative for strided layouts (encoded as a signed two's-complement `u64`).
///
/// # Contract
///
/// Implementations MUST:
/// - Reject `index.len() != shape.rank()` with [`Error::IndexRankMismatch`].
/// - Reject any `DYNAMIC` dimension with [`Error::DynamicDimInIndexing`].
/// - Reject any out-of-bounds index component with [`Error::IndexOutOfRange`].
///
/// # Examples
///
/// ```
/// use hurray_core::layout::{LayoutDescriptor, addressing::ElementAddress};
/// use hurray_core::Shape;
///
/// let shape = Shape::new(vec![3, 4]).unwrap();
/// let offset = LayoutDescriptor::RowMajor
///     .element_offset(&[1, 2], &shape)
///     .unwrap();
/// assert_eq!(offset, 6); // 1*4 + 2*1 = 6
/// ```
pub trait ElementAddress {
    /// Returns the linear element offset (in logical elements) for the given index.
    fn element_offset(&self, index: &[u64], shape: &Shape) -> Result<u64>;
}

/// Implemented by sparse layout descriptors; requires borrow of index buffers.
///
/// Kept `pub(crate)` until the first consumer (Layer 4) validates the API shape.
/// See OQ-014.1 for promotion criteria.
// Dead-code allowed: defined now for the stub impls; first used in Layer 4.
#[allow(dead_code)]
pub(crate) trait SparseElementAddress {
    /// Returns the storage offset of the given logical index,
    /// or `None` if the element is structurally absent (structural zero).
    fn sparse_element_offset(
        &self,
        index: &[u64],
        buffers: &SparseBuffers<'_>,
    ) -> Result<Option<u64>>;
}

/// Borrowed component buffers for sparse layout address lookup.
// Dead-code allowed: first constructed in Layer 4 when sparse addressing is wired up.
#[allow(dead_code)]
pub(crate) struct SparseBuffers<'a> {
    /// Raw bytes of each component buffer, in the order defined by the layout spec.
    pub buffers: &'a [&'a [u8]],
}

/// Converts a linear element offset to an absolute byte address within a buffer.
///
/// `element_offset` is treated as a **signed** two's-complement `u64`, so
/// negative offsets from strided layouts work correctly.
///
/// # Arguments
///
/// - `element_offset` — output of [`ElementAddress::element_offset`].
/// - `byte_offset` — the `byte_offset` field from the tensor descriptor.
/// - `element_type` — tensor's element type (determines byte-width arithmetic).
/// - `buffer_size` — total byte size of the buffer; used for bounds checking.
///
/// # Errors
///
/// - [`Error::AddressOverflow`] — intermediate arithmetic overflowed.
/// - [`Error::ByteAddressOverflow`] — address falls outside `[0, buffer_size)`.
///
/// # Spec
///
/// Implements `docs/spec/memory-layout.md § Element Address Computation`.
///
/// # Examples
///
/// ```
/// use hurray_core::ElementType;
/// use hurray_core::layout::addressing::byte_address_from_element_offset;
///
/// // float32 at element offset 6 with byte_offset 0 in a 100-byte buffer.
/// let addr = byte_address_from_element_offset(6, 0, ElementType::Float32, 100).unwrap();
/// assert_eq!(addr, 24); // 6 * 4 = 24
///
/// // bool at element offset 9 (fits in 2nd byte): byte = 0 + floor(9/8) = 1.
/// let addr = byte_address_from_element_offset(9, 0, ElementType::Bool, 100).unwrap();
/// assert_eq!(addr, 1);
/// ```
pub fn byte_address_from_element_offset(
    element_offset: u64,
    byte_offset: u64,
    element_type: ElementType,
    buffer_size: u64,
) -> Result<u64> {
    let signed = element_offset as i64;
    let bits = element_type.bit_width() as i64;

    let byte_delta: i64 = if bits >= 8 {
        // Whole-byte types: delta = signed_offset × (bits / 8).
        signed.checked_mul(bits / 8).ok_or(Error::AddressOverflow)?
    } else if bits == 6 {
        // 6-bit types pack 4 elements per 3 bytes; group-level addressing.
        // byte_delta = floor(signed / 4) × 3.
        signed
            .div_euclid(4)
            .checked_mul(3)
            .ok_or(Error::AddressOverflow)?
    } else {
        // Sub-byte power-of-two (bits ∈ {1, 2, 4}).
        // Packing factor P = 8 / bits; byte_delta = floor(signed / P).
        signed.div_euclid(8 / bits)
    };

    let byte_addr = (byte_offset as i64)
        .checked_add(byte_delta)
        .ok_or(Error::AddressOverflow)?;

    if byte_addr < 0 || byte_addr as u64 >= buffer_size {
        return Err(Error::ByteAddressOverflow { buffer_size });
    }
    Ok(byte_addr as u64)
}

/// Validates a multi-dimensional index against a shape.
///
/// Returns `Err` on rank mismatch, any DYNAMIC dimension, or any out-of-bounds
/// index component. Called at the start of every `ElementAddress` impl.
pub(crate) fn validate_index(index: &[u64], shape: &Shape) -> Result<()> {
    if index.len() != shape.rank() {
        return Err(Error::IndexRankMismatch {
            index_rank: index.len(),
            shape_rank: shape.rank(),
        });
    }
    for (k, (&idx, &dim)) in index.iter().zip(shape.dims().iter()).enumerate() {
        if dim == DYNAMIC {
            return Err(Error::DynamicDimInIndexing { dim: k as u32 });
        }
        if idx >= dim {
            return Err(Error::IndexOutOfRange {
                dim: k as u32,
                index: idx,
                size: dim,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::layout::{
        CooLayout, LayoutDescriptor, MortonLayout, RegionDescriptor, StridedLayout, SubpavingLayout,
    };
    use crate::{ElementType, Error, Shape};

    use super::byte_address_from_element_offset;

    // ── LayoutDescriptor::element_offset dispatch ─────────────────────────────

    // RowMajor dispatch: shape [3,4], element [1,2] → 1*4+2 = 6
    #[test]
    fn dispatch_row_major() {
        let shape = Shape::new(vec![3, 4]).unwrap();
        assert_eq!(
            LayoutDescriptor::RowMajor
                .element_offset(&[1, 2], &shape)
                .unwrap(),
            6
        );
    }

    // ColMajor dispatch: shape [3,4], element [1,2] → 1*1+2*3 = 7
    #[test]
    fn dispatch_col_major() {
        let shape = Shape::new(vec![3, 4]).unwrap();
        assert_eq!(
            LayoutDescriptor::ColMajor
                .element_offset(&[1, 2], &shape)
                .unwrap(),
            7
        );
    }

    // Strided dispatch: row-major strides [4,1], same result as RowMajor.
    #[test]
    fn dispatch_strided() {
        let shape = Shape::new(vec![3, 4]).unwrap();
        let layout = LayoutDescriptor::Strided(StridedLayout::new(vec![4, 1]));
        assert_eq!(layout.element_offset(&[1, 2], &shape).unwrap(), 6);
    }

    // Morton dispatch: shape [4,4], morton_bits [2,2], element [2,3] → 14
    #[test]
    fn dispatch_morton() {
        let shape = Shape::new(vec![4, 4]).unwrap();
        let layout = LayoutDescriptor::Morton(MortonLayout::new(vec![2, 2]).unwrap());
        assert_eq!(layout.element_offset(&[2, 3], &shape).unwrap(), 14);
    }

    // Subpaving returns LayoutRequiresMultiBuffer with the correct tag.
    #[test]
    fn dispatch_subpaving_returns_multi_buffer_error() {
        let region = RegionDescriptor::new(vec![0, 0], vec![4, 4], 0x01, 0, 0).unwrap();
        let layout = LayoutDescriptor::Subpaving(SubpavingLayout::new(vec![region]).unwrap());
        let shape = Shape::new(vec![4, 4]).unwrap();
        let err = layout.element_offset(&[0, 0], &shape).unwrap_err();
        assert!(
            matches!(err, Error::LayoutRequiresMultiBuffer { layout_tag: 0x06 }),
            "expected LayoutRequiresMultiBuffer{{0x06}}, got {err:?}"
        );
    }

    // COO returns LayoutRequiresMultiBuffer with tag 0x07.
    #[test]
    fn dispatch_coo_returns_multi_buffer_error() {
        let layout = LayoutDescriptor::Coo(CooLayout::new(0, false));
        let shape = Shape::new(vec![4, 4]).unwrap();
        let err = layout.element_offset(&[0, 0], &shape).unwrap_err();
        assert!(
            matches!(err, Error::LayoutRequiresMultiBuffer { layout_tag: 0x07 }),
            "expected LayoutRequiresMultiBuffer{{0x07}}, got {err:?}"
        );
    }

    // ── byte_address_from_element_offset ──────────────────────────────────────

    // Spec docs/spec/memory-layout.md § Element Address Computation:
    // Float32 at element offset 6, byte_offset 0, buffer 100 bytes → 6*4 = 24.
    #[test]
    fn byte_addr_float32_offset_6() {
        let addr = byte_address_from_element_offset(6, 0, ElementType::Float32, 100).unwrap();
        assert_eq!(addr, 24);
    }

    // Bool at element offset 9, byte_offset 0: floor(9/8) = 1.
    #[test]
    fn byte_addr_bool_offset_9() {
        let addr = byte_address_from_element_offset(9, 0, ElementType::Bool, 100).unwrap();
        assert_eq!(addr, 1);
    }

    // Int4 at element offset 7, byte_offset 0: floor(7/2) = 3.
    #[test]
    fn byte_addr_int4_offset_7() {
        let addr = byte_address_from_element_offset(7, 0, ElementType::Int4, 100).unwrap();
        assert_eq!(addr, 3);
    }

    // Uint4 at element offset 8, byte_offset 0: floor(8/2) = 4.
    #[test]
    fn byte_addr_uint4_offset_8() {
        let addr = byte_address_from_element_offset(8, 0, ElementType::Uint4, 100).unwrap();
        assert_eq!(addr, 4);
    }

    // Float64 at element offset 3, byte_offset 8: 8 + 3*8 = 32.
    #[test]
    fn byte_addr_float64_with_byte_offset() {
        let addr = byte_address_from_element_offset(3, 8, ElementType::Float64, 200).unwrap();
        assert_eq!(addr, 32);
    }

    // Negative strided offset:
    // Float32, element_offset = (-5_i64 as u64), byte_offset = 40, buffer = 100.
    // byte_delta = -5 * 4 = -20; byte_addr = 40 + (-20) = 20. Valid: 20 < 100.
    #[test]
    fn byte_addr_negative_strided_offset() {
        let element_offset = (-5i64) as u64;
        let addr = byte_address_from_element_offset(element_offset, 40, ElementType::Float32, 100)
            .unwrap();
        assert_eq!(addr, 20);
    }

    // ByteAddressOverflow when computed address >= buffer_size.
    // Float32, offset 25, byte_offset 0, buffer 100: 25*4 = 100 >= 100 → overflow.
    #[test]
    fn byte_addr_overflow_at_exact_buffer_size() {
        let err = byte_address_from_element_offset(25, 0, ElementType::Float32, 100).unwrap_err();
        assert!(
            matches!(err, Error::ByteAddressOverflow { buffer_size: 100 }),
            "expected ByteAddressOverflow{{100}}, got {err:?}"
        );
    }

    // ByteAddressOverflow when byte_offset insufficient to cover a negative delta.
    // Float32, element_offset = (-5_i64 as u64), byte_offset = 4, buffer = 100.
    // byte_delta = -20; byte_addr = 4 - 20 = -16 → negative → overflow.
    #[test]
    fn byte_addr_overflow_negative_offset_underflows_byte_offset() {
        let element_offset = (-5i64) as u64;
        let err = byte_address_from_element_offset(element_offset, 4, ElementType::Float32, 100)
            .unwrap_err();
        assert!(
            matches!(err, Error::ByteAddressOverflow { .. }),
            "expected ByteAddressOverflow, got {err:?}"
        );
    }

    // First valid element (offset 0) always succeeds even with buffer_size=1.
    #[test]
    fn byte_addr_offset_zero_is_always_valid() {
        let addr = byte_address_from_element_offset(0, 0, ElementType::Float32, 4).unwrap();
        assert_eq!(addr, 0);
    }

    // Last valid float32 element in a 40-byte buffer (10 elements, offset 9):
    // 9*4 = 36 < 40. ✓
    #[test]
    fn byte_addr_last_valid_float32_element() {
        let addr = byte_address_from_element_offset(9, 0, ElementType::Float32, 40).unwrap();
        assert_eq!(addr, 36);
    }
}
