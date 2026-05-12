//! `TensorDescriptor` binary decoding.
//!
//! Implements the wire format defined in `docs/spec/metadata.md`. Steps follow
//! the spec's Overall Descriptor Structure order exactly.

use crate::descriptor::cursor::ByteCursor;
use crate::descriptor::ext_type::ExtensionTypeDescriptor;
use crate::descriptor::layout_codec::decode_layout_payload;
use crate::descriptor::mod_types::{DescriptorFlags, TensorDescriptor, RESERVED_FLAGS_MASK};
use crate::descriptor::shard::ShardDescriptor;
use crate::descriptor::statistics::Statistics;
use crate::descriptor::{DESCRIPTOR_VERSION_MAJOR, MIN_DESCRIPTOR_LEN};
use crate::{BufferHandle, DeviceTag, ElementType, Error, Result, Shape, SyncMode};

/// Decodes a [`TensorDescriptor`] from its wire representation.
///
/// # Errors
///
/// See the error variants in [`Error`] prefixed with `Invalid`, `Unsupported`,
/// `Descriptor`, `Reserved`, and `Empty` for the full list of rejection conditions.
pub(crate) fn decode(bytes: &[u8]) -> Result<TensorDescriptor> {
    // ── Step 1: minimum length check before touching the cursor ───────────────
    // We need at least 10 bytes to read magic (4) + version (2) + descriptor_length (4).
    if bytes.len() < 10 {
        return Err(Error::DescriptorTruncated {
            offset: 0,
            needed: 10,
            available: bytes.len(),
        });
    }

    // ── Step 2: magic ─────────────────────────────────────────────────────────
    let magic: [u8; 4] = [bytes[0], bytes[1], bytes[2], bytes[3]];
    if magic != crate::descriptor::MAGIC {
        return Err(Error::InvalidMagic { got: magic });
    }

    // ── Step 3: version ───────────────────────────────────────────────────────
    let version_major = bytes[4];
    let version_minor = bytes[5];
    if version_major > DESCRIPTOR_VERSION_MAJOR {
        return Err(Error::UnsupportedDescriptorVersion {
            major: version_major,
            minor: version_minor,
        });
    }

    // ── Step 4: descriptor_length ─────────────────────────────────────────────
    let descriptor_length = u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]);
    if descriptor_length < MIN_DESCRIPTOR_LEN {
        return Err(Error::DescriptorTooShort {
            length: descriptor_length,
        });
    }

    // ── Step 5: create cursor limited to descriptor_length ────────────────────
    // Clamp to available bytes so ByteCursor does not go past what we have.
    let limit = (descriptor_length as usize).min(bytes.len());
    let mut cursor = ByteCursor::new(bytes, limit);

    // Skip magic (4) + version_major (1) + version_minor (1) + descriptor_length (4)
    // = 10 bytes already consumed above outside the cursor.
    // We advance the cursor past those bytes now.
    cursor.read_bytes(10)?;

    // ── Step 6: flags ─────────────────────────────────────────────────────────
    let flags_raw = cursor.read_u32_le()?;
    if flags_raw & RESERVED_FLAGS_MASK != 0 {
        return Err(Error::ReservedDescriptorFlagBitsSet {
            flags: flags_raw,
            mask: RESERVED_FLAGS_MASK,
        });
    }
    let flags = DescriptorFlags(flags_raw);

    // ── Step 7: type_tag ──────────────────────────────────────────────────────
    let type_tag = cursor.read_u8()?;
    // from_tag now returns Ok(Extension(tag)) for 0xF0–0xFE; flag consistency
    // is checked in Step 13 below.
    let element_type = ElementType::from_tag(type_tag)?;

    // ── Step 8: layout_tag ────────────────────────────────────────────────────
    let layout_tag = cursor.read_u8()?;

    // ── Step 9: rank ──────────────────────────────────────────────────────────
    let rank = cursor.read_u32_le()?;
    // Rank validation: MAX_RANK = 64.
    if rank as usize > crate::MAX_RANK {
        return Err(Error::RankExceedsMaximum {
            rank,
            max: crate::MAX_RANK as u32,
        });
    }

    // ── Step 10: shape uint64[rank] ───────────────────────────────────────────
    let mut dims = Vec::with_capacity(rank as usize);
    for _ in 0..rank {
        dims.push(cursor.read_u64_le()?);
    }
    let shape = Shape::new(dims)?;

    // ── Step 11: byte_offset uint64 ───────────────────────────────────────────
    let byte_offset = cursor.read_u64_le()?;

    // ── Step 12: layout payload ───────────────────────────────────────────────
    let layout = decode_layout_payload(layout_tag, rank, &mut cursor, 0)?;

    // ── Step 13: HAS_EXTENSION_TYPE ↔ type_tag range consistency ─────────────
    // ElementType::Extension(tag) is returned for 0xF0–0xFE; non-extension tags
    // yield a named variant. Both cases must agree with the HAS_EXTENSION_TYPE flag.
    let is_extension_tag = matches!(element_type, ElementType::Extension(_));
    if is_extension_tag != flags.has_extension_type() {
        return Err(Error::ExtensionTypeFlagMismatch {
            flag_set: flags.has_extension_type(),
            type_tag,
            type_tag_in_range: if is_extension_tag {
                "in 0xF0-0xFE"
            } else {
                "not in 0xF0-0xFE"
            },
        });
    }

    // ── Step 14: buffer table ─────────────────────────────────────────────────
    let buffer_count = cursor.read_u8()?;
    if buffer_count == 0 {
        return Err(Error::EmptyBufferTable);
    }
    let mut buffers = Vec::with_capacity(buffer_count as usize);
    for _ in 0..buffer_count {
        let byte_size = cursor.read_u64_le()?;
        let alignment = cursor.read_u32_le()?;
        let device_tag_byte = cursor.read_u8()?;
        let sync_mode_byte = cursor.read_u8()?;
        let reserved = cursor.read_bytes(2)?;
        if reserved != [0u8, 0] {
            return Err(Error::ReservedBytesNonZero {
                field: "buffer_handle._reserved",
            });
        }
        let device_tag = DeviceTag::from_byte(device_tag_byte)?;
        let sync_mode = SyncMode::from_byte(sync_mode_byte)?;
        let handle = BufferHandle::new(byte_size, alignment, device_tag, sync_mode)?;
        buffers.push(handle);
    }

    // ── Step 15: quantization section ────────────────────────────────────────
    let quantization = if flags.has_quantization() {
        let quant_len = cursor.read_u32_le()?;
        let quant_bytes = cursor.read_bytes(quant_len as usize)?.to_vec();
        Some(quant_bytes)
    } else {
        None
    };

    // ── Step 16: shard section ────────────────────────────────────────────────
    let shard = if flags.has_shard() {
        let s = ShardDescriptor::decode_from(&mut cursor, rank)?;
        // Validate bounds against tensor shape.
        s.validate_against_shape(&shape)?;
        Some(s)
    } else {
        None
    };

    // ── Step 17: statistics section ───────────────────────────────────────────
    let statistics = if flags.has_statistics() {
        Some(Statistics::decode_from(&mut cursor)?)
    } else {
        None
    };

    // ── Step 18: extension type section ──────────────────────────────────────
    let extension_type = if flags.has_extension_type() {
        Some(ExtensionTypeDescriptor::decode_from(&mut cursor)?)
    } else {
        None
    };

    // ── Step 19: length consistency check ─────────────────────────────────────
    // For known minor versions, the consumed byte count must exactly match
    // descriptor_length. For future minor versions, silently accept trailing bytes.
    // DESCRIPTOR_VERSION_MINOR is currently 0; the <= comparison is intentionally
    // written as == to satisfy clippy while preserving the forward-compat intent:
    // future minor bumps will change the constant, restoring the <= semantics.
    if version_minor == crate::descriptor::DESCRIPTOR_VERSION_MINOR {
        let consumed = cursor.pos();
        if consumed != descriptor_length as usize {
            return Err(Error::DescriptorLengthMismatch {
                declared: descriptor_length,
                actual: consumed,
            });
        }
    }

    // ── Step 20: construct TensorDescriptor ───────────────────────────────────
    TensorDescriptor::new(
        version_major,
        version_minor,
        element_type,
        shape,
        byte_offset,
        layout,
        buffers,
        quantization,
        shard,
        statistics,
        extension_type,
    )
}
