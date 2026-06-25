//! Block-paged layout descriptor.
//!
//! Tag `0x0B`. Rank MUST be 3. Buffer count = 3 (page_pool + block_table + seq_ptr).
//! See `docs/spec/layouts/block-paged.md`.

/// KV-cache role for a block-paged tensor.
///
/// The wire value is a `uint8` in the additional descriptor fields of a
/// block-paged layout (see `docs/spec/layouts/block-paged.md § Additional Descriptor Fields`).
///
/// # Examples
///
/// ```
/// use hurray_core::layout::KvRole;
///
/// assert_eq!(KvRole::Key.wire_byte(), 0x00);
/// assert_eq!(KvRole::Value.wire_byte(), 0x01);
/// assert_eq!(KvRole::Fused.wire_byte(), 0x02);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum KvRole {
    /// Key tensor (`kv_role = 0x00`).
    Key,
    /// Value tensor (`kv_role = 0x01`).
    Value,
    /// Fused or non-KV generic tensor (`kv_role = 0x02`).
    Fused,
}

impl KvRole {
    /// Returns the wire byte for this role.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::layout::KvRole;
    ///
    /// assert_eq!(KvRole::Key.wire_byte(), 0x00);
    /// assert_eq!(KvRole::Value.wire_byte(), 0x01);
    /// assert_eq!(KvRole::Fused.wire_byte(), 0x02);
    /// ```
    #[inline]
    pub fn wire_byte(self) -> u8 {
        match self {
            Self::Key => 0x00,
            Self::Value => 0x01,
            Self::Fused => 0x02,
        }
    }

    /// Constructs a [`KvRole`] from its wire byte.
    ///
    /// Returns `None` for unrecognized values (future spec extensions).
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::layout::KvRole;
    ///
    /// assert_eq!(KvRole::from_wire(0x00), Some(KvRole::Key));
    /// assert_eq!(KvRole::from_wire(0x03), None);
    /// ```
    #[inline]
    pub fn from_wire(byte: u8) -> Option<Self> {
        match byte {
            0x00 => Some(Self::Key),
            0x01 => Some(Self::Value),
            0x02 => Some(Self::Fused),
            _ => None,
        }
    }
}

/// Element type for the `block_table` and `seq_ptr` index buffers.
///
/// Both buffers share the same index type, selected by `block_table_index_type`
/// in the block-paged descriptor fields. `U32` addresses up to ~4 billion pages;
/// `U64` is required for larger pools.
///
/// See `docs/spec/layouts/block-paged.md § Additional Descriptor Fields`.
///
/// # Examples
///
/// ```
/// use hurray_core::layout::BlockTableIndexType;
///
/// assert_eq!(BlockTableIndexType::U32.wire_byte(), 0x00);
/// assert_eq!(BlockTableIndexType::U64.wire_byte(), 0x01);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BlockTableIndexType {
    /// 32-bit unsigned index (`block_table_index_type = 0x00`). Default.
    U32,
    /// 64-bit unsigned index (`block_table_index_type = 0x01`).
    U64,
}

impl BlockTableIndexType {
    /// Returns the wire byte for this index type.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::layout::BlockTableIndexType;
    ///
    /// assert_eq!(BlockTableIndexType::U32.wire_byte(), 0x00);
    /// assert_eq!(BlockTableIndexType::U64.wire_byte(), 0x01);
    /// ```
    #[inline]
    pub fn wire_byte(self) -> u8 {
        match self {
            Self::U32 => 0x00,
            Self::U64 => 0x01,
        }
    }

    /// Constructs a [`BlockTableIndexType`] from its wire byte.
    ///
    /// Returns `None` for unrecognized values (future spec extensions).
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::layout::BlockTableIndexType;
    ///
    /// assert_eq!(BlockTableIndexType::from_wire(0x00), Some(BlockTableIndexType::U32));
    /// assert_eq!(BlockTableIndexType::from_wire(0x02), None);
    /// ```
    #[inline]
    pub fn from_wire(byte: u8) -> Option<Self> {
        match byte {
            0x00 => Some(Self::U32),
            0x01 => Some(Self::U64),
            _ => None,
        }
    }

    /// Returns the size in bytes of a single index element for this type.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::layout::BlockTableIndexType;
    ///
    /// assert_eq!(BlockTableIndexType::U32.element_bytes(), 4);
    /// assert_eq!(BlockTableIndexType::U64.element_bytes(), 8);
    /// ```
    #[inline]
    pub fn element_bytes(self) -> usize {
        match self {
            Self::U32 => 4,
            Self::U64 => 8,
        }
    }
}

/// The wire sentinel for `layer_index` meaning "not layer-scoped".
///
/// When `layer_index` is `0xFFFFFFFF` on the wire the field encodes
/// [`None`] in Rust; any other value encodes `Some(n)`.
pub(crate) const LAYER_INDEX_NONE: u32 = 0xFFFF_FFFF;

/// Descriptor for the block-paged indirect layout.
///
/// Block-paged stores a KV-cache tensor whose **paged axis** is divided into
/// fixed-size pages drawn from a shared **page pool**, with a **block table**
/// mapping each logical page position to a physical page ID. It is the
/// interchange form of a PagedAttention KV cache.
///
/// This layout is **defined only for rank-3 tensors** in this version of the
/// specification: `[total_tokens, num_heads, head_dim]`.
///
/// Three buffers are always required:
///
/// | Buffer | Name | Description |
/// |--------|------|-------------|
/// | 0 | `page_pool` | Flat pool of fixed-size pages. |
/// | 1 | `block_table` | Concatenated per-sequence page-ID lists. |
/// | 2 | `seq_ptr` | CSR-style offset array into `block_table`. |
///
/// When the tensor carries quantization parameters, additional buffers appear
/// at indices 3 and above per `docs/spec/quantization.md § Buffer Table Placement Rules`.
///
/// See `docs/spec/layouts/block-paged.md` for the full normative definition.
///
/// # Examples
///
/// ```
/// use hurray_core::layout::{BlockPagedLayout, BlockTableIndexType, KvRole, LayoutDescriptor};
///
/// let layout = BlockPagedLayout::new(16, 128, 0, 4, KvRole::Key, Some(3), BlockTableIndexType::U32);
/// assert_eq!(layout.page_size, 16);
/// assert_eq!(layout.num_pages, 128);
/// assert_eq!(layout.num_seqs, 4);
/// assert_eq!(layout.layer_index, Some(3));
///
/// let desc = LayoutDescriptor::BlockPaged(layout);
/// assert_eq!(desc.tag(), 0x0B);
/// assert_eq!(desc.buffer_count().map(|n| n.get()), Some(3));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct BlockPagedLayout {
    /// Tokens per page. MUST be >= 1.
    ///
    /// Common values are 16 and 32 (matching vLLM's default `block_size`).
    pub page_size: u32,

    /// Number of physical pages in `page_pool`.
    pub num_pages: u64,

    /// The axis subdivided into pages. MUST be 0 in this spec version.
    ///
    /// Kept as a field rather than a constant so that the wire format remains
    /// forward-compatible if a future spec version adds non-zero paged axes.
    pub paged_axis: u32,

    /// Number of sequences in the batch. MAY be 0 for an empty batch.
    pub num_seqs: u32,

    /// KV-cache role for this tensor (key, value, or fused/generic).
    pub kv_role: KvRole,

    /// Transformer layer index, or `None` if the tensor is not layer-scoped.
    ///
    /// Wire value `0xFFFFFFFF` maps to `None`; all other values map to `Some(n)`.
    pub layer_index: Option<u32>,

    /// Element type for both the `block_table` (buffer 1) and `seq_ptr` (buffer 2).
    ///
    /// `U32` is the default and addresses up to 4 billion pages.
    /// `U64` is required for larger pools.
    pub block_table_index_type: BlockTableIndexType,
}

impl BlockPagedLayout {
    /// Creates a new [`BlockPagedLayout`].
    ///
    /// # Arguments
    ///
    /// - `page_size` — tokens per page (MUST be validated >= 1 later via `validate_against_shape`).
    /// - `num_pages` — number of physical pages in the pool.
    /// - `paged_axis` — the axis subdivided into pages (MUST be 0 in this version).
    /// - `num_seqs` — number of sequences in the batch.
    /// - `kv_role` — KV-cache role for this tensor.
    /// - `layer_index` — transformer layer index (`None` = not layer-scoped).
    /// - `block_table_index_type` — element type for `block_table` and `seq_ptr` buffers.
    ///
    /// This constructor does not validate `page_size >= 1` or `paged_axis == 0` because
    /// the tensor shape is not available at construction time.  Call
    /// [`LayoutDescriptor::validate_against_shape`] to perform all invariant checks
    /// once the shape is known.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::layout::{BlockPagedLayout, BlockTableIndexType, KvRole};
    ///
    /// // Layer-3 key cache: page_size=16, 64 pages, 2 sequences, uint32 indices.
    /// let layout = BlockPagedLayout::new(
    ///     16, 64, 0, 2,
    ///     KvRole::Key,
    ///     Some(3),
    ///     BlockTableIndexType::U32,
    /// );
    /// assert_eq!(layout.page_size, 16);
    /// assert_eq!(layout.layer_index, Some(3));
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        page_size: u32,
        num_pages: u64,
        paged_axis: u32,
        num_seqs: u32,
        kv_role: KvRole,
        layer_index: Option<u32>,
        block_table_index_type: BlockTableIndexType,
    ) -> Self {
        Self {
            page_size,
            num_pages,
            paged_axis,
            num_seqs,
            kv_role,
            layer_index,
            block_table_index_type,
        }
    }

    /// Validates that a quantization scheme is compatible with this block-paged layout.
    ///
    /// Per `docs/spec/layouts/block-paged.md § Quantization Compatibility`:
    ///
    /// - **Per-tensor** (`scheme_tag = 0x01`): always compatible.
    /// - **Per-channel** (`scheme_tag = 0x02`): MUST NOT be applied on axis 0 (the
    ///   paged / token axis). `quant_axis` must be 1 (`num_heads`) or 2 (`head_dim`).
    /// - **Per-block-affine** (`scheme_tag = 0x03`): `quant_axis` MUST be 0 and
    ///   `quant_block_size` MUST equal `page_size`.
    /// - All other schemes: not validated here; returns `Ok(())`.
    ///
    /// Call this at the typed-quantization layer once you have a decoded
    /// [`crate::QuantizationDescriptor`]. The raw-bytes descriptor layer cannot
    /// perform this check.
    ///
    /// # Arguments
    ///
    /// - `scheme_tag` — the `scheme_tag` byte from the quantization descriptor header.
    /// - `quant_axis` — the `axis` field from the quantization descriptor (relevant for
    ///   per-channel and per-block-affine).
    /// - `quant_block_size` — the `block_size` field from the per-block-affine descriptor
    ///   (ignored for all other schemes).
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::InvalidQuantization`] if the scheme/layout combination
    /// violates the spec rules above.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::layout::{BlockPagedLayout, BlockTableIndexType, KvRole};
    ///
    /// let layout = BlockPagedLayout::new(16, 64, 0, 2, KvRole::Key, Some(0), BlockTableIndexType::U32);
    ///
    /// // Per-tensor (0x01) is always valid.
    /// assert!(layout.validate_quantization_compatibility(0x01, 0, 0).is_ok());
    ///
    /// // Per-channel on axis 1 (num_heads) is valid.
    /// assert!(layout.validate_quantization_compatibility(0x02, 1, 0).is_ok());
    ///
    /// // Per-channel on axis 0 (paged axis) is forbidden.
    /// assert!(layout.validate_quantization_compatibility(0x02, 0, 0).is_err());
    ///
    /// // Per-block-affine with axis=0 and block_size==page_size is valid.
    /// assert!(layout.validate_quantization_compatibility(0x03, 0, 16).is_ok());
    ///
    /// // Per-block-affine with block_size != page_size is forbidden.
    /// assert!(layout.validate_quantization_compatibility(0x03, 0, 32).is_err());
    /// ```
    pub fn validate_quantization_compatibility(
        &self,
        scheme_tag: u8,
        quant_axis: u32,
        quant_block_size: u32,
    ) -> crate::Result<()> {
        const SCHEME_PER_CHANNEL: u8 = 0x02;
        const SCHEME_PER_BLOCK_AFFINE: u8 = 0x03;

        match scheme_tag {
            SCHEME_PER_CHANNEL => {
                // Per-channel MUST NOT be applied to the paged axis (axis 0).
                // axis 1 (num_heads) and axis 2 (head_dim) are permitted.
                if quant_axis == 0 {
                    return Err(crate::Error::InvalidQuantization(
                        "block-paged: per-channel quantization (scheme 0x02) MUST NOT be applied \
                         to the paged axis (axis 0); use axis 1 (num_heads) or axis 2 (head_dim)"
                            .to_string(),
                    ));
                }
            }
            SCHEME_PER_BLOCK_AFFINE => {
                // axis MUST be 0 (the paged / token axis).
                if quant_axis != 0 {
                    return Err(crate::Error::InvalidQuantization(format!(
                        "block-paged: per-block-affine quantization (scheme 0x03) requires \
                         axis = 0, got axis = {quant_axis}"
                    )));
                }
                // block_size MUST equal page_size so scales align with page slots.
                if quant_block_size != self.page_size {
                    return Err(crate::Error::InvalidQuantization(format!(
                        "block-paged: per-block-affine quantization (scheme 0x03) requires \
                         block_size == page_size ({}), got block_size = {}",
                        self.page_size, quant_block_size
                    )));
                }
            }
            // Per-tensor (0x01) and all other schemes compose without paged-layout constraints.
            _ => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::LayoutDescriptor;
    use std::num::NonZeroU8;

    // ── KvRole wire mapping ───────────────────────────────────────────────────

    /// Spec §block-paged.md §Additional Descriptor Fields: kv_role wire bytes.
    #[test]
    fn kv_role_wire_byte_key_is_0x00() {
        assert_eq!(KvRole::Key.wire_byte(), 0x00);
    }

    #[test]
    fn kv_role_wire_byte_value_is_0x01() {
        assert_eq!(KvRole::Value.wire_byte(), 0x01);
    }

    #[test]
    fn kv_role_wire_byte_fused_is_0x02() {
        assert_eq!(KvRole::Fused.wire_byte(), 0x02);
    }

    #[test]
    fn kv_role_from_wire_round_trips_all_variants() {
        for variant in [KvRole::Key, KvRole::Value, KvRole::Fused] {
            let byte = variant.wire_byte();
            let decoded = KvRole::from_wire(byte);
            assert_eq!(
                decoded,
                Some(variant),
                "KvRole::from_wire(0x{byte:02X}) should return {variant:?}"
            );
        }
    }

    #[test]
    fn kv_role_from_wire_rejects_0x03() {
        assert_eq!(KvRole::from_wire(0x03), None);
    }

    #[test]
    fn kv_role_from_wire_rejects_0xff() {
        assert_eq!(KvRole::from_wire(0xFF), None);
    }

    // ── BlockTableIndexType wire mapping ──────────────────────────────────────

    /// Spec §block-paged.md §Additional Descriptor Fields: block_table_index_type wire bytes.
    #[test]
    fn block_table_index_type_wire_byte_u32_is_0x00() {
        assert_eq!(BlockTableIndexType::U32.wire_byte(), 0x00);
    }

    #[test]
    fn block_table_index_type_wire_byte_u64_is_0x01() {
        assert_eq!(BlockTableIndexType::U64.wire_byte(), 0x01);
    }

    #[test]
    fn block_table_index_type_from_wire_round_trips_all_variants() {
        for variant in [BlockTableIndexType::U32, BlockTableIndexType::U64] {
            let byte = variant.wire_byte();
            let decoded = BlockTableIndexType::from_wire(byte);
            assert_eq!(
                decoded,
                Some(variant),
                "BlockTableIndexType::from_wire(0x{byte:02X}) should return {variant:?}"
            );
        }
    }

    #[test]
    fn block_table_index_type_from_wire_rejects_0x02() {
        assert_eq!(BlockTableIndexType::from_wire(0x02), None);
    }

    #[test]
    fn block_table_index_type_from_wire_rejects_0xff() {
        assert_eq!(BlockTableIndexType::from_wire(0xFF), None);
    }

    #[test]
    fn block_table_index_type_element_bytes_u32_is_4() {
        assert_eq!(BlockTableIndexType::U32.element_bytes(), 4);
    }

    #[test]
    fn block_table_index_type_element_bytes_u64_is_8() {
        assert_eq!(BlockTableIndexType::U64.element_bytes(), 8);
    }

    // ── layer_index sentinel ─────────────────────────────────────────────────

    /// Spec §block-paged.md §Additional Descriptor Fields:
    /// wire value 0xFFFFFFFF encodes None; any other value encodes Some(n).
    #[test]
    fn layer_index_none_sentinel_value_is_0xffffffff() {
        assert_eq!(LAYER_INDEX_NONE, 0xFFFF_FFFF);
    }

    #[test]
    fn layer_index_some_zero_is_distinct_from_none() {
        // Some(0) must differ from None (0xFFFFFFFF sentinel).
        let layout_none =
            BlockPagedLayout::new(16, 64, 0, 2, KvRole::Key, None, BlockTableIndexType::U32);
        let layout_some_zero =
            BlockPagedLayout::new(16, 64, 0, 2, KvRole::Key, Some(0), BlockTableIndexType::U32);
        assert_ne!(layout_none, layout_some_zero);
        assert_eq!(layout_none.layer_index, None);
        assert_eq!(layout_some_zero.layer_index, Some(0));
    }

    #[test]
    fn layer_index_some_large_value_stored_correctly() {
        let n = 0xFFFF_FFFE_u32; // one below sentinel
        let layout =
            BlockPagedLayout::new(16, 64, 0, 2, KvRole::Key, Some(n), BlockTableIndexType::U32);
        assert_eq!(layout.layer_index, Some(n));
    }

    // ── BlockPagedLayout tag and buffer count ─────────────────────────────────

    /// Spec §block-paged.md: layout tag MUST be 0x0B.
    #[test]
    fn block_paged_layout_tag_is_0x0b() {
        let layout =
            BlockPagedLayout::new(16, 64, 0, 2, KvRole::Key, Some(0), BlockTableIndexType::U32);
        let desc = LayoutDescriptor::BlockPaged(layout);
        assert_eq!(desc.tag(), 0x0B);
    }

    /// Spec §block-paged.md §Buffer Table: buffer_count MUST be 3.
    #[test]
    fn block_paged_buffer_count_is_3() {
        let layout =
            BlockPagedLayout::new(16, 64, 0, 2, KvRole::Key, Some(0), BlockTableIndexType::U32);
        let desc = LayoutDescriptor::BlockPaged(layout);
        assert_eq!(desc.buffer_count(), NonZeroU8::new(3));
    }

    // ── Quantization compatibility ────────────────────────────────────────────

    /// Spec §block-paged.md §Quantization Compatibility: per-tensor (0x01) is always valid.
    #[test]
    fn quant_compat_per_tensor_always_accepted() {
        let layout =
            BlockPagedLayout::new(16, 64, 0, 2, KvRole::Key, Some(0), BlockTableIndexType::U32);
        // axis and block_size are irrelevant for per-tensor; any values pass.
        assert!(layout
            .validate_quantization_compatibility(0x01, 0, 0)
            .is_ok());
        assert!(layout
            .validate_quantization_compatibility(0x01, 99, 999)
            .is_ok());
    }

    /// Spec §block-paged.md §Quantization Compatibility: per-channel on axis 0 (paged axis) is forbidden.
    #[test]
    fn quant_compat_per_channel_on_axis_0_rejected() {
        let layout =
            BlockPagedLayout::new(16, 64, 0, 2, KvRole::Key, Some(0), BlockTableIndexType::U32);
        let result = layout.validate_quantization_compatibility(0x02, 0, 0);
        assert!(result.is_err(), "per-channel on axis 0 must be rejected");
        assert!(matches!(result, Err(crate::Error::InvalidQuantization(_))));
    }

    /// Spec §block-paged.md §Quantization Compatibility: per-channel on axis 1 (num_heads) is valid.
    #[test]
    fn quant_compat_per_channel_on_axis_1_accepted() {
        let layout =
            BlockPagedLayout::new(16, 64, 0, 2, KvRole::Key, Some(0), BlockTableIndexType::U32);
        assert!(layout
            .validate_quantization_compatibility(0x02, 1, 0)
            .is_ok());
    }

    /// Spec §block-paged.md §Quantization Compatibility: per-channel on axis 2 (head_dim) is valid.
    #[test]
    fn quant_compat_per_channel_on_axis_2_accepted() {
        let layout =
            BlockPagedLayout::new(16, 64, 0, 2, KvRole::Key, Some(0), BlockTableIndexType::U32);
        assert!(layout
            .validate_quantization_compatibility(0x02, 2, 0)
            .is_ok());
    }

    /// Spec §block-paged.md §Quantization Compatibility: per-block-affine with axis=0 and
    /// block_size == page_size is valid.
    #[test]
    fn quant_compat_per_block_affine_axis_0_block_size_eq_page_size_accepted() {
        let layout =
            BlockPagedLayout::new(16, 64, 0, 2, KvRole::Key, Some(0), BlockTableIndexType::U32);
        assert!(layout
            .validate_quantization_compatibility(0x03, 0, 16)
            .is_ok());
    }

    /// Spec §block-paged.md §Quantization Compatibility: per-block-affine with axis != 0 is rejected.
    #[test]
    fn quant_compat_per_block_affine_axis_nonzero_rejected() {
        let layout =
            BlockPagedLayout::new(16, 64, 0, 2, KvRole::Key, Some(0), BlockTableIndexType::U32);
        let result = layout.validate_quantization_compatibility(0x03, 1, 16);
        assert!(
            result.is_err(),
            "per-block-affine with axis=1 must be rejected"
        );
        assert!(matches!(result, Err(crate::Error::InvalidQuantization(_))));
    }

    /// Spec §block-paged.md §Quantization Compatibility: per-block-affine with
    /// block_size != page_size is rejected.
    #[test]
    fn quant_compat_per_block_affine_block_size_mismatch_rejected() {
        let layout =
            BlockPagedLayout::new(16, 64, 0, 2, KvRole::Key, Some(0), BlockTableIndexType::U32);
        let result = layout.validate_quantization_compatibility(0x03, 0, 32);
        assert!(
            result.is_err(),
            "per-block-affine with block_size=32 != page_size=16 must be rejected"
        );
        assert!(matches!(result, Err(crate::Error::InvalidQuantization(_))));
    }

    /// Unknown quantization scheme tag (>= 0x04) passes without error.
    #[test]
    fn quant_compat_unknown_scheme_tag_passes() {
        let layout =
            BlockPagedLayout::new(16, 64, 0, 2, KvRole::Key, Some(0), BlockTableIndexType::U32);
        assert!(layout
            .validate_quantization_compatibility(0x04, 0, 0)
            .is_ok());
        assert!(layout
            .validate_quantization_compatibility(0xFF, 0, 0)
            .is_ok());
    }
}
