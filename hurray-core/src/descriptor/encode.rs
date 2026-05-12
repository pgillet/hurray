//! `TensorDescriptor` binary encoding.
//!
//! Implements the wire format defined in `docs/spec/metadata.md`. Section order:
//! ```text
//! [Fixed Header 20 bytes] [shape] [byte_offset] [Layout-specific]
//! [Buffer Table] [Quantization?] [Shard?] [Statistics?] [ExtensionType?]
//! ```
//! All multi-byte fields are little-endian.

use crate::descriptor::cursor::ByteWriter;
use crate::descriptor::layout_codec::encode_layout_payload;
use crate::descriptor::mod_types::TensorDescriptor;
use crate::{Error, Result};

/// Encodes a [`TensorDescriptor`] to its wire representation.
///
/// Returns `Err` only if the total encoded length would overflow `u32`.
pub(crate) fn encode(d: &TensorDescriptor) -> Result<Vec<u8>> {
    let mut w = ByteWriter::new();
    let rank = d.shape.rank() as u32;
    let flags = d.flags().0;

    // ── Fixed header (20 bytes) ───────────────────────────────────────────────
    // magic (4)
    w.write_bytes(&crate::descriptor::MAGIC);
    // version_major (1)
    w.write_u8(d.version_major);
    // version_minor (1)
    w.write_u8(d.version_minor);
    // descriptor_length placeholder — back-patched after we know total size (4)
    let length_offset = w.len(); // byte index 6
    w.write_u32_le(0u32);
    // flags (4)
    w.write_u32_le(flags);
    // type_tag (1)
    w.write_u8(d.element_type.tag());
    // layout_tag (1)
    w.write_u8(d.layout.tag());
    // rank (4)
    w.write_u32_le(rank);

    // ── shape: uint64[rank] ───────────────────────────────────────────────────
    for &dim in d.shape.dims() {
        w.write_u64_le(dim);
    }

    // ── byte_offset: uint64 ───────────────────────────────────────────────────
    w.write_u64_le(d.byte_offset);

    // ── Layout-specific fields ────────────────────────────────────────────────
    encode_layout_payload(&d.layout, rank, &mut w)?;

    // ── Buffer table ──────────────────────────────────────────────────────────
    // buffer_count: uint8 (max 255)
    if d.buffers.len() > 255 {
        return Err(Error::InvalidLayout(format!(
            "buffer_count {} exceeds the maximum of 255 (uint8 ceiling)",
            d.buffers.len()
        )));
    }
    w.write_u8(d.buffers.len() as u8);
    // Per-handle: byte_size u64 (8), alignment u32 (4), device_tag u8 (1),
    //             sync_mode u8 (1), _reserved u8[2] — total 16 bytes (ADR-018 § 3).
    for handle in &d.buffers {
        w.write_u64_le(handle.byte_size());
        w.write_u32_le(handle.alignment());
        w.write_u8(handle.device_tag().to_byte());
        w.write_u8(handle.sync_mode().to_byte());
        w.write_zeros(2); // _reserved
    }

    // ── Optional sections (in spec-mandated order) ────────────────────────────

    // Quantization (flag bit 0)
    if let Some(quant_bytes) = &d.quantization {
        let len = quant_bytes.len() as u32;
        w.write_u32_le(len);
        w.write_bytes(quant_bytes);
    }

    // Shard (flag bit 1)
    if let Some(shard) = &d.shard {
        shard.encode_into(&mut w);
    }

    // Statistics (flag bit 3)
    if let Some(stats) = &d.statistics {
        stats.encode_into(&mut w);
    }

    // Extension type (flag bit 2)
    if let Some(ext) = &d.extension_type {
        ext.encode_into(&mut w);
    }

    // ── Back-patch descriptor_length ──────────────────────────────────────────
    let total_len = w.len();
    if total_len > u32::MAX as usize {
        return Err(Error::DescriptorLengthMismatch {
            declared: u32::MAX,
            actual: total_len,
        });
    }
    let mut buf = w.into_vec();
    buf[length_offset..length_offset + 4].copy_from_slice(&(total_len as u32).to_le_bytes());

    Ok(buf)
}
