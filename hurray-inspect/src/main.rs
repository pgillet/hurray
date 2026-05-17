//! `hurray-inspect` — CLI tool for inspecting Hurray files.
//!
//! # Usage
//!
//! ```text
//! hurray-inspect <file>
//! hurray-inspect -          # read from stdin
//! ```
//!
//! Auto-detects the file type:
//!
//! - **HRRYFILE container** (magic `HRRYFILE`): displays the file header,
//!   trailer, index, KV metadata section, and each tensor's descriptor.
//! - **Raw tensor descriptor** (magic `HRRY`): displays the descriptor fields.

use std::{
    env, fs,
    io::{self, Read},
    process,
};

use hurray_core::{
    descriptor::TensorDescriptor,
    layout::{LayoutDescriptor, SubpavingLayout, TiledLayout},
    DYNAMIC,
};

// ── Error type ─────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("{0}")]
    Parse(#[from] hurray_core::Error),
    #[error("malformed HRRYFILE: {0}")]
    Hrryfile(String),
    #[error("usage: hurray-inspect <file>  (use '-' for stdin)")]
    Usage,
}

type Result<T> = std::result::Result<T, Error>;

// ── Table row ──────────────────────────────────────────────────────────────────

/// A single row in the output hex table.
struct Row {
    /// Byte offset of the first byte of this field within the file.
    offset: usize,
    /// Raw bytes of this field (empty for annotation-only rows).
    bytes: Vec<u8>,
    /// Human-readable field description.
    field: String,
}

// ── HRRYFILE format constants ──────────────────────────────────────────────────

const FILE_MAGIC: &[u8; 8] = b"HRRYFILE";
const FILE_HEADER_SIZE: usize = 64;
const TRAILER_SIZE: usize = 40;

const FLAG_HAS_KV_METADATA: u32 = 1 << 0;
const FLAG_SORTED_INDEX: u32 = 1 << 1;
const FLAG_HAS_INDEX_CRC32C: u32 = 1 << 2;

fn file_flags_display(flags: u32) -> String {
    let mut parts = Vec::new();
    if flags & FLAG_HAS_KV_METADATA != 0 {
        parts.push("HAS_KV_METADATA");
    }
    if flags & FLAG_SORTED_INDEX != 0 {
        parts.push("SORTED_INDEX");
    }
    if flags & FLAG_HAS_INDEX_CRC32C != 0 {
        parts.push("HAS_INDEX_CRC32C");
    }
    let reserved = flags & !0x07u32;
    let base = format!("0x{flags:08X}");
    let mut suffix = parts.join(" | ");
    if reserved != 0 {
        if !suffix.is_empty() {
            suffix.push_str(&format!(" | reserved=0x{reserved:08X}"));
        } else {
            suffix = format!("reserved=0x{reserved:08X}");
        }
    }
    if suffix.is_empty() {
        base
    } else {
        format!("{base} ({suffix})")
    }
}

fn kv_tag_name(tag: u8) -> &'static str {
    match tag {
        0x01 => "string",
        0x02 => "int64",
        0x03 => "uint64",
        0x04 => "float64",
        0x05 => "bool",
        0x06 => "bytes",
        0x07 => "array",
        _ => "unknown",
    }
}

// ── LE decode helpers ──────────────────────────────────────────────────────────

fn le_u8(b: &[u8]) -> u8 {
    b[0]
}
fn le_u16(b: &[u8]) -> u16 {
    // SAFETY: take() guarantees exactly 2 bytes
    u16::from_le_bytes(b[..2].try_into().unwrap())
}
fn le_u32(b: &[u8]) -> u32 {
    // SAFETY: take() guarantees exactly 4 bytes
    u32::from_le_bytes(b[..4].try_into().unwrap())
}
fn le_u64(b: &[u8]) -> u64 {
    // SAFETY: take() guarantees exactly 8 bytes
    u64::from_le_bytes(b[..8].try_into().unwrap())
}
fn le_i64(b: &[u8]) -> i64 {
    // SAFETY: take() guarantees exactly 8 bytes
    i64::from_le_bytes(b[..8].try_into().unwrap())
}
fn le_f64(b: &[u8]) -> f64 {
    // SAFETY: take() guarantees exactly 8 bytes
    f64::from_le_bytes(b[..8].try_into().unwrap())
}

// ── FReader — sequential cursor over a file byte slice ────────────────────────

/// Sequential byte reader that tracks absolute file offsets.
struct FReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> FReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn seek_to(&mut self, pos: usize) {
        self.pos = pos;
    }

    /// Consume `n` bytes and return `(absolute_offset, owned_bytes)`.
    fn take(&mut self, n: usize) -> Result<(usize, Vec<u8>)> {
        let offset = self.pos;
        let end = offset + n;
        if end > self.data.len() {
            return Err(Error::Hrryfile(format!(
                "unexpected end of file at offset {offset} \
                 (need {n} bytes, have {})",
                self.data.len().saturating_sub(offset)
            )));
        }
        let bytes = self.data[offset..end].to_vec();
        self.pos = end;
        Ok((offset, bytes))
    }

    fn row(&mut self, n: usize, field: String) -> Result<Row> {
        let (offset, bytes) = self.take(n)?;
        Ok(Row {
            offset,
            bytes,
            field,
        })
    }
}

// ── Reader — cursor for raw TensorDescriptor slices ───────────────────────────

/// Cursor over a byte slice bounded to `limit` bytes.
///
/// `base_offset` shifts all reported row offsets so they reflect the field's
/// absolute position within the containing file (0 for standalone descriptors).
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
    limit: usize,
    base_offset: usize,
}

impl<'a> Reader<'a> {
    fn with_base(data: &'a [u8], limit: usize, base_offset: usize) -> Self {
        Self {
            data,
            pos: 0,
            limit,
            base_offset,
        }
    }

    /// Consume `n` bytes and return a display [`Row`] labelled with `field`.
    fn take_row(&mut self, n: usize, field: String) -> Row {
        let offset = self.pos;
        let end = (self.pos + n).min(self.limit).min(self.data.len());
        let bytes = self.data[offset..end].to_vec();
        self.pos = end;
        Row {
            offset: self.base_offset + offset,
            bytes,
            field,
        }
    }

    /// Advance to `target`, emitting a row for any skipped bytes.
    fn skip_to(&mut self, target: usize) -> Option<Row> {
        if target <= self.pos || target > self.limit || target > self.data.len() {
            return None;
        }
        let offset = self.pos;
        let bytes = self.data[self.pos..target].to_vec();
        self.pos = target;
        Some(Row {
            offset: self.base_offset + offset,
            bytes,
            field: "(unknown / padding bytes)".to_string(),
        })
    }
}

// ── Layout tag name ─────────────────────────────────────────────────────────────

fn layout_tag_name(tag: u8) -> &'static str {
    match tag {
        0x01 => "row-major",
        0x02 => "column-major",
        0x03 => "strided",
        0x04 => "tiled",
        0x05 => "morton",
        0x06 => "subpaving",
        0x07 => "COO",
        0x08 => "CSR",
        0x09 => "CSC",
        0x40 => "hilbert",
        0xF0..=0xFE => "private-extension",
        _ => "unknown",
    }
}

// ── Layout-specific rows ───────────────────────────────────────────────────────

fn layout_rows(reader: &mut Reader<'_>, layout: &LayoutDescriptor) -> Vec<Row> {
    match layout {
        LayoutDescriptor::RowMajor | LayoutDescriptor::ColMajor => vec![],

        LayoutDescriptor::Strided(s) => s
            .strides
            .iter()
            .enumerate()
            .map(|(i, &v)| reader.take_row(8, format!("strides[{i}] = {v}")))
            .collect(),

        LayoutDescriptor::Tiled(t) => tiled_rows(reader, t),

        LayoutDescriptor::Morton(m) => m
            .morton_bits
            .iter()
            .enumerate()
            .map(|(i, &v)| reader.take_row(4, format!("morton_bits[{i}] = {v}")))
            .collect(),

        LayoutDescriptor::Subpaving(s) => subpaving_rows(reader, s),

        LayoutDescriptor::Coo(c) => vec![
            reader.take_row(8, format!("nnz = {}", c.nnz)),
            reader.take_row(1, format!("is_sorted = 0x{:02X}", u8::from(c.is_sorted))),
            reader.take_row(7, "COO._reserved".to_string()),
        ],

        LayoutDescriptor::Csr(c) => vec![
            reader.take_row(8, format!("nnz = {}", c.nnz)),
            reader.take_row(8, "CSR._reserved".to_string()),
        ],

        LayoutDescriptor::Csc(c) => vec![
            reader.take_row(8, format!("nnz = {}", c.nnz)),
            reader.take_row(8, "CSC._reserved".to_string()),
        ],

        LayoutDescriptor::Hilbert(h) => vec![
            reader.take_row(4, format!("hilbert_order = {}", h.hilbert_order)),
            reader.take_row(4, format!("hilbert_rank = {}", h.hilbert_rank)),
        ],

        LayoutDescriptor::PrivateExtension(p) => {
            let mut rows = vec![
                reader.take_row(
                    8,
                    format!("extension_layout_id = 0x{:016X}", p.extension_layout_id),
                ),
                reader.take_row(
                    4,
                    format!("extension_data_length = {}", p.extension_data.len()),
                ),
            ];
            if !p.extension_data.is_empty() {
                rows.push(reader.take_row(p.extension_data.len(), "extension_data".to_string()));
            }
            rows
        }

        LayoutDescriptor::Unknown(u) => {
            if u.raw_bytes.is_empty() {
                vec![]
            } else {
                vec![reader.take_row(
                    u.raw_bytes.len(),
                    format!("(unknown layout 0x{:02X} — raw bytes)", u.tag),
                )]
            }
        }

        // Required by #[non_exhaustive] — treat any future variant like Unknown.
        _ => vec![],
    }
}

fn tiled_rows(reader: &mut Reader<'_>, tiled: &TiledLayout) -> Vec<Row> {
    let mut rows = Vec::new();

    for (i, &v) in tiled.tile_shape.iter().enumerate() {
        rows.push(reader.take_row(8, format!("tile_shape[{i}] = {v}")));
    }
    rows.push(reader.take_row(
        1,
        format!(
            "outer_layout = 0x{:02X} ({})",
            tiled.outer_layout,
            layout_tag_name(tiled.outer_layout)
        ),
    ));
    rows.push(reader.take_row(
        1,
        format!(
            "inner_layout = 0x{:02X} ({})",
            tiled.inner_layout,
            layout_tag_name(tiled.inner_layout)
        ),
    ));
    rows.push(reader.take_row(2, "tiled._reserved".to_string()));

    if let Some(os) = &tiled.outer_strides {
        for (i, &v) in os.strides.iter().enumerate() {
            rows.push(reader.take_row(8, format!("outer_strides[{i}] = {v}")));
        }
    }
    if let Some(is) = &tiled.inner_strides {
        for (i, &v) in is.strides.iter().enumerate() {
            rows.push(reader.take_row(8, format!("inner_strides[{i}] = {v}")));
        }
    } else if let Some(inner) = &tiled.inner_tiled {
        rows.extend(tiled_rows(reader, inner));
    }

    rows
}

fn subpaving_rows(reader: &mut Reader<'_>, layout: &SubpavingLayout) -> Vec<Row> {
    let mut rows = Vec::new();

    rows.push(reader.take_row(4, format!("region_count = {}", layout.regions.len())));

    for (r, region) in layout.regions.iter().enumerate() {
        for (i, &v) in region.origin.iter().enumerate() {
            rows.push(reader.take_row(8, format!("region[{r}].origin[{i}] = {v}")));
        }
        for (i, &v) in region.region_shape.iter().enumerate() {
            rows.push(reader.take_row(8, format!("region[{r}].region_shape[{i}] = {v}")));
        }
        rows.push(reader.take_row(
            1,
            format!(
                "region[{r}].region_layout_tag = 0x{:02X} ({})",
                region.region_layout_tag,
                layout_tag_name(region.region_layout_tag)
            ),
        ));
        rows.push(reader.take_row(3, "region._reserved".to_string()));
        rows.push(reader.take_row(
            4,
            format!("region[{r}].buffer_index = {}", region.buffer_index),
        ));
        rows.push(reader.take_row(
            8,
            format!(
                "region[{r}].region_byte_offset = {}",
                region.region_byte_offset
            ),
        ));

        if let Some(inner) = &region.inner_layout {
            rows.extend(layout_rows(reader, inner));
        }
    }

    rows
}

// ── Descriptor display ─────────────────────────────────────────────────────────

/// Build rows for a tensor descriptor.
///
/// `base_offset` is added to all row offsets so they reflect the descriptor's
/// absolute position within the file (pass 0 for standalone descriptor files).
fn rows_from_descriptor(data: &[u8], desc: &TensorDescriptor, base_offset: usize) -> Vec<Row> {
    // descriptor_length is at bytes 6..10 (little-endian u32).
    let desc_len = u32::from_le_bytes(data[6..10].try_into().unwrap_or([0u8; 4])) as usize;
    let mut reader = Reader::with_base(data, desc_len, base_offset);
    let mut rows = Vec::new();

    // Fixed header (20 bytes).
    rows.push(reader.take_row(4, r#"magic = "HRRY""#.to_string()));
    rows.push(reader.take_row(1, format!("version_major = {}", desc.version_major)));
    rows.push(reader.take_row(1, format!("version_minor = {}", desc.version_minor)));
    rows.push(reader.take_row(4, format!("descriptor_length = {desc_len}")));
    rows.push(reader.take_row(4, format!("flags = 0x{:08X}", desc.flags().0)));
    rows.push(reader.take_row(
        1,
        format!(
            "type_tag = 0x{:02X} ({})",
            desc.element_type.tag(),
            desc.element_type
        ),
    ));
    let ltag = desc.layout.tag();
    rows.push(reader.take_row(
        1,
        format!("layout_tag = 0x{ltag:02X} ({})", layout_tag_name(ltag)),
    ));
    let rank = desc.shape.rank() as u32;
    rows.push(reader.take_row(4, format!("rank = {rank}")));

    // shape[0..rank] (rank × u64)
    for (i, &dim) in desc.shape.dims().iter().enumerate() {
        let display = if dim == DYNAMIC {
            "(dynamic)".to_string()
        } else {
            dim.to_string()
        };
        rows.push(reader.take_row(8, format!("shape[{i}] = {display}")));
    }

    // byte_offset (u64)
    rows.push(reader.take_row(8, format!("byte_offset = {}", desc.byte_offset)));

    // Layout-specific fields.
    rows.extend(layout_rows(&mut reader, &desc.layout));

    // Buffer table.
    rows.push(reader.take_row(1, format!("buffer_count = {}", desc.buffers.len())));

    // Per-handle: byte_size u64 (8), alignment u32 (4), device_tag u8 (1),
    //             sync_mode u8 (1), _reserved u8[2] — 16 bytes total (ADR-018).
    for (b, buf) in desc.buffers.iter().enumerate() {
        rows.push(reader.take_row(8, format!("buffer[{b}].byte_size = {}", buf.byte_size())));
        rows.push(reader.take_row(4, format!("buffer[{b}].alignment = {}", buf.alignment())));
        rows.push(reader.take_row(
            1,
            format!(
                "buffer[{b}].device_tag = 0x{:02X} ({})",
                buf.device_tag().to_byte(),
                buf.device_tag()
            ),
        ));
        rows.push(reader.take_row(
            1,
            format!(
                "buffer[{b}].sync_mode = 0x{:02X} ({})",
                buf.sync_mode().to_byte(),
                buf.sync_mode()
            ),
        ));
        rows.push(reader.take_row(2, format!("buffer[{b}]._reserved")));
    }

    // Optional sections in spec-mandated order: quantization, shard, statistics,
    // extension-type.  The encoder writes them in this order; the cursor must
    // advance in the same sequence.

    // Quantization (flag bit 0)
    if let Some(q) = &desc.quantization {
        rows.push(reader.take_row(4, format!("quantization_length = {}", q.len())));
        if !q.is_empty() {
            rows.push(reader.take_row(q.len(), "quantization_descriptor".to_string()));
        }
    }

    // Shard (flag bit 1)
    if let Some(s) = &desc.shard {
        for (i, &v) in s.parent_shape.iter().enumerate() {
            rows.push(reader.take_row(8, format!("parent_shape[{i}] = {v}")));
        }
        for (i, &v) in s.shard_offset.iter().enumerate() {
            rows.push(reader.take_row(8, format!("shard_offset[{i}] = {v}")));
        }
    }

    // Statistics (flag bit 3)
    if let Some(st) = &desc.statistics {
        rows.push(reader.take_row(
            4,
            format!("stats.computed_mask = 0x{:08X}", st.computed_mask.0),
        ));
        rows.push(reader.take_row(4, "stats._reserved".to_string()));
        rows.push(reader.take_row(8, format!("stats.nnz = {}", st.nnz)));
        rows.push(reader.take_row(8, format!("stats.sparsity_ratio = {}", st.sparsity_ratio)));
        rows.push(reader.take_row(8, format!("stats.value_min = {}", st.value_min)));
        rows.push(reader.take_row(8, format!("stats.value_max = {}", st.value_max)));
        rows.push(reader.take_row(8, format!("stats.value_abs_max = {}", st.value_abs_max)));
        rows.push(reader.take_row(8, format!("stats.value_mean = {}", st.value_mean)));
        rows.push(reader.take_row(8, format!("stats.value_stddev = {}", st.value_stddev)));
        rows.push(reader.take_row(1, format!("stats.nm_n = {}", st.nm_n)));
        rows.push(reader.take_row(1, format!("stats.nm_m = {}", st.nm_m)));
        rows.push(reader.take_row(1, format!("stats.has_nan = {}", u8::from(st.has_nan))));
        rows.push(reader.take_row(1, format!("stats.has_inf = {}", u8::from(st.has_inf))));
        rows.push(reader.take_row(4, "stats._reserved2".to_string()));
    }

    // Extension type (flag bit 2)
    if let Some(ext) = &desc.extension_type {
        rows.push(reader.take_row(4, format!("ext_type.bit_width = {}", ext.bit_width)));
        rows.push(reader.take_row(
            1,
            format!("ext_type.packing_factor = {}", ext.packing_factor),
        ));
        rows.push(reader.take_row(1, format!("ext_type.is_float = {}", u8::from(ext.is_float))));
        rows.push(reader.take_row(
            1,
            format!("ext_type.is_signed = {}", u8::from(ext.is_signed)),
        ));
        rows.push(reader.take_row(1, format!("ext_type.sign_bits = {}", ext.sign_bits)));
        rows.push(reader.take_row(1, format!("ext_type.exponent_bits = {}", ext.exponent_bits)));
        rows.push(reader.take_row(1, format!("ext_type.mantissa_bits = {}", ext.mantissa_bits)));
        rows.push(reader.take_row(2, "ext_type._reserved".to_string()));
        rows.push(reader.take_row(4, format!("ext_type.exponent_bias = {}", ext.exponent_bias)));
        rows.push(reader.take_row(1, format!("ext_type.has_nan = {}", u8::from(ext.has_nan))));
        rows.push(reader.take_row(1, format!("ext_type.has_inf = {}", u8::from(ext.has_inf))));
        rows.push(reader.take_row(2, "ext_type._reserved2".to_string()));
    }

    // Consume any trailing bytes within the descriptor window (future minor-version additions).
    if let Some(row) = reader.skip_to(desc_len) {
        rows.push(row);
    }

    rows
}

// ── HRRYFILE container inspection ─────────────────────────────────────────────

fn section_row(title: &str) -> Row {
    Row {
        offset: 0,
        bytes: vec![],
        field: format!("── {title} ──"),
    }
}

fn parse_kv_scalar(r: &mut FReader<'_>, rows: &mut Vec<Row>, label: &str, tag: u8) -> Result<()> {
    match tag {
        0x01 => {
            let (sl_off, sl) = r.take(4)?;
            let str_len = le_u32(&sl) as usize;
            rows.push(Row {
                offset: sl_off,
                bytes: sl,
                field: format!("{label}.str_len = {str_len}"),
            });
            let (sv_off, sv) = r.take(str_len)?;
            let s = String::from_utf8(sv.clone()).unwrap_or_else(|_| "(invalid UTF-8)".into());
            rows.push(Row {
                offset: sv_off,
                bytes: sv,
                field: format!("{label} = {s:?}"),
            });
        }
        0x02 => {
            let (vo, vb) = r.take(8)?;
            rows.push(Row {
                offset: vo,
                bytes: vb.clone(),
                field: format!("{label} = {}i64", le_i64(&vb)),
            });
        }
        0x03 => {
            let (vo, vb) = r.take(8)?;
            rows.push(Row {
                offset: vo,
                bytes: vb.clone(),
                field: format!("{label} = {}u64", le_u64(&vb)),
            });
        }
        0x04 => {
            let (vo, vb) = r.take(8)?;
            rows.push(Row {
                offset: vo,
                bytes: vb.clone(),
                field: format!("{label} = {}f64", le_f64(&vb)),
            });
        }
        0x05 => {
            let (vo, vb) = r.take(1)?;
            rows.push(Row {
                offset: vo,
                bytes: vb.clone(),
                field: format!("{label} = {}", vb[0] != 0),
            });
        }
        0x06 => {
            let (bl_off, bl) = r.take(4)?;
            let byte_len = le_u32(&bl) as usize;
            rows.push(Row {
                offset: bl_off,
                bytes: bl,
                field: format!("{label}.byte_len = {byte_len}"),
            });
            let (bv_off, bv) = r.take(byte_len)?;
            rows.push(Row {
                offset: bv_off,
                bytes: bv,
                field: format!("{label} = ({byte_len} bytes)"),
            });
        }
        _ => {
            rows.push(Row {
                offset: r.pos,
                bytes: vec![],
                field: format!("{label} = (unknown type 0x{tag:02X})"),
            });
        }
    }
    Ok(())
}

fn parse_kv_section(
    data: &[u8],
    r: &mut FReader<'_>,
    rows: &mut Vec<Row>,
    kv_offset: usize,
    kv_length: usize,
) -> Result<()> {
    r.seek_to(kv_offset);

    let (kc_off, kc) = r.take(4)?;
    let kv_count = le_u32(&kc) as usize;
    rows.push(section_row(&format!("KV METADATA ({kv_count} entries)")));
    rows.push(Row {
        offset: kc_off,
        bytes: kc,
        field: format!("kv_count = {kv_count}"),
    });

    for i in 0..kv_count {
        let (kl_off, kl) = r.take(2)?;
        let key_len = le_u16(&kl) as usize;
        rows.push(Row {
            offset: kl_off,
            bytes: kl,
            field: format!("kv[{i}].key_len = {key_len}"),
        });

        let (k_off, k) = r.take(key_len)?;
        let key = String::from_utf8(k.clone())
            .map_err(|_| Error::Hrryfile(format!("kv[{i}] key is not valid UTF-8")))?;
        rows.push(Row {
            offset: k_off,
            bytes: k,
            field: format!("kv[{i}].key = {key:?}"),
        });

        let (tag_off, tag_b) = r.take(1)?;
        let tag = le_u8(&tag_b);
        rows.push(Row {
            offset: tag_off,
            bytes: tag_b,
            field: format!("kv[{i}].type = 0x{tag:02X} ({})", kv_tag_name(tag)),
        });

        if tag == 0x07 {
            // Array: elem_type (1), count (4), then count * elem values
            let (et_off, et) = r.take(1)?;
            let elem_tag = le_u8(&et);
            rows.push(Row {
                offset: et_off,
                bytes: et,
                field: format!(
                    "kv[{i}].elem_type = 0x{elem_tag:02X} ({})",
                    kv_tag_name(elem_tag)
                ),
            });
            let (ac_off, ac) = r.take(4)?;
            let count = le_u32(&ac) as usize;
            rows.push(Row {
                offset: ac_off,
                bytes: ac,
                field: format!("kv[{i}].elem_count = {count}"),
            });
            for j in 0..count {
                parse_kv_scalar(r, rows, &format!("kv[{i}][{j}]"), elem_tag)?;
            }
        } else {
            parse_kv_scalar(r, rows, &format!("kv[{i}].value"), tag)?;
        }
    }

    let _ = (data, kv_length); // consumed for clarity; bounds already validated by open()
    Ok(())
}

fn inspect_hrryfile_inner(data: &[u8], rows: &mut Vec<Row>) -> Result<()> {
    let file_size = data.len();

    if file_size < FILE_HEADER_SIZE + TRAILER_SIZE {
        return Err(Error::Hrryfile(format!(
            "file too small: {file_size} bytes (minimum {})",
            FILE_HEADER_SIZE + TRAILER_SIZE
        )));
    }

    let mut r = FReader::new(data);

    // ── File header (64 bytes) ─────────────────────────────────────────────────
    rows.push(section_row("FILE HEADER (64 bytes)"));

    let (magic_off, magic) = r.take(8)?;
    rows.push(Row {
        offset: magic_off,
        bytes: magic,
        field: r#"file_magic = "HRRYFILE""#.to_string(),
    });

    let (vmaj_off, vmaj) = r.take(1)?;
    let version_major = le_u8(&vmaj);
    rows.push(Row {
        offset: vmaj_off,
        bytes: vmaj,
        field: format!("container_version_major = {version_major}"),
    });

    let (vmin_off, vmin) = r.take(1)?;
    let version_minor = le_u8(&vmin);
    rows.push(Row {
        offset: vmin_off,
        bytes: vmin,
        field: format!("container_version_minor = {version_minor}"),
    });

    rows.push(r.row(2, "_reserved".to_string())?);

    let (fl_off, fl) = r.take(4)?;
    let flags = le_u32(&fl);
    rows.push(Row {
        offset: fl_off,
        bytes: fl,
        field: format!("flags = {}", file_flags_display(flags)),
    });

    let (al_off, al) = r.take(4)?;
    let alignment = le_u32(&al);
    rows.push(Row {
        offset: al_off,
        bytes: al,
        field: format!("data_buffer_alignment = {alignment}"),
    });

    let (fdo_off, fdo) = r.take(8)?;
    let first_desc_offset = le_u64(&fdo);
    rows.push(Row {
        offset: fdo_off,
        bytes: fdo,
        field: format!("first_descriptor_offset = {first_desc_offset}"),
    });

    let (tch_off, tch) = r.take(8)?;
    let tensor_count_hint = le_u64(&tch);
    rows.push(Row {
        offset: tch_off,
        bytes: tch,
        field: if tensor_count_hint == u64::MAX {
            "tensor_count_hint = (unknown)".to_string()
        } else {
            format!("tensor_count_hint = {tensor_count_hint}")
        },
    });

    rows.push(r.row(28, "_reserved_header".to_string())?);

    // ── Trailer (last 40 bytes) ────────────────────────────────────────────────
    rows.push(section_row("TRAILER (40 bytes)"));

    let trailer_start = file_size - TRAILER_SIZE;
    r.seek_to(trailer_start);

    let (io_off, io) = r.take(8)?;
    let index_offset = le_u64(&io) as usize;
    rows.push(Row {
        offset: io_off,
        bytes: io,
        field: format!("index_offset = {index_offset}"),
    });

    let (il_off, il) = r.take(8)?;
    let index_length = le_u64(&il) as usize;
    rows.push(Row {
        offset: il_off,
        bytes: il,
        field: format!("index_length = {index_length}"),
    });

    let (ko_off, ko) = r.take(8)?;
    let kv_offset = le_u64(&ko) as usize;
    rows.push(Row {
        offset: ko_off,
        bytes: ko,
        field: format!("kv_offset = {kv_offset}"),
    });

    let (kl_off, kl_b) = r.take(4)?;
    let kv_length = le_u32(&kl_b) as usize;
    rows.push(Row {
        offset: kl_off,
        bytes: kl_b,
        field: format!("kv_length = {kv_length}"),
    });

    let (crc_off, crc_b) = r.take(4)?;
    let stored_crc = le_u32(&crc_b);
    rows.push(Row {
        offset: crc_off,
        bytes: crc_b,
        field: format!("index_crc32c = 0x{stored_crc:08X}"),
    });

    rows.push(r.row(4, "_reserved".to_string())?);

    let (tm_off, tm) = r.take(4)?;
    rows.push(Row {
        offset: tm_off,
        bytes: tm,
        field: r#"trailer_magic = "HRRY""#.to_string(),
    });

    // ── Index ──────────────────────────────────────────────────────────────────
    let sorted_label = if flags & FLAG_SORTED_INDEX != 0 {
        ", sorted"
    } else {
        ""
    };

    r.seek_to(index_offset);

    let (ec_off, ec) = r.take(8)?;
    let entry_count = le_u64(&ec) as usize;

    rows.push(section_row(&format!(
        "INDEX ({entry_count} entr{}{sorted_label})",
        if entry_count == 1 { "y" } else { "ies" }
    )));
    rows.push(Row {
        offset: ec_off,
        bytes: ec,
        field: format!("entry_count = {entry_count}"),
    });

    struct IndexEntry {
        name: String,
        descriptor_offset: usize,
        descriptor_length: usize,
        data_offset: u64,
        data_length: u64,
    }
    let mut entries: Vec<IndexEntry> = Vec::with_capacity(entry_count);

    for i in 0..entry_count {
        let (nl_off, nl) = r.take(2)?;
        let name_len = le_u16(&nl) as usize;
        rows.push(Row {
            offset: nl_off,
            bytes: nl,
            field: format!("entry[{i}].name_len = {name_len}"),
        });

        let (n_off, n_b) = r.take(name_len)?;
        let name = String::from_utf8(n_b.clone())
            .map_err(|_| Error::Hrryfile(format!("entry[{i}] name is not valid UTF-8")))?;
        rows.push(Row {
            offset: n_off,
            bytes: n_b,
            field: format!("entry[{i}].name = {name:?}"),
        });

        let (do_off, do_b) = r.take(8)?;
        let descriptor_offset = le_u64(&do_b) as usize;
        rows.push(Row {
            offset: do_off,
            bytes: do_b,
            field: format!("entry[{i}].descriptor_offset = {descriptor_offset}"),
        });

        let (dl_off, dl_b) = r.take(4)?;
        let descriptor_length = le_u32(&dl_b) as usize;
        rows.push(Row {
            offset: dl_off,
            bytes: dl_b,
            field: format!("entry[{i}].descriptor_length = {descriptor_length}"),
        });

        let (dto_off, dto) = r.take(8)?;
        let data_offset = le_u64(&dto);
        rows.push(Row {
            offset: dto_off,
            bytes: dto,
            field: format!("entry[{i}].data_offset = {data_offset}"),
        });

        let (dtl_off, dtl) = r.take(8)?;
        let data_length = le_u64(&dtl);
        rows.push(Row {
            offset: dtl_off,
            bytes: dtl,
            field: format!("entry[{i}].data_length = {data_length}"),
        });

        let (ef_off, ef) = r.take(4)?;
        let entry_flags = le_u32(&ef);
        rows.push(Row {
            offset: ef_off,
            bytes: ef,
            field: format!("entry[{i}].flags = 0x{entry_flags:08X}"),
        });

        entries.push(IndexEntry {
            name,
            descriptor_offset,
            descriptor_length,
            data_offset,
            data_length,
        });
    }

    // ── KV metadata (optional) ─────────────────────────────────────────────────
    let has_kv = (flags & FLAG_HAS_KV_METADATA != 0) || (kv_offset != 0 && kv_length != 0);
    if has_kv && kv_offset != 0 && kv_length != 0 {
        parse_kv_section(data, &mut r, rows, kv_offset, kv_length)?;
    }

    // ── Per-tensor descriptors ─────────────────────────────────────────────────
    for entry in &entries {
        rows.push(section_row(&format!(
            "TENSOR DESCRIPTOR: {:?}  (data at {}, {} bytes)",
            entry.name, entry.data_offset, entry.data_length
        )));

        let end = entry.descriptor_offset + entry.descriptor_length;
        let desc_data = data.get(entry.descriptor_offset..end).ok_or_else(|| {
            Error::Hrryfile(format!(
                "descriptor for {:?} is out of file bounds (offset={}, length={})",
                entry.name, entry.descriptor_offset, entry.descriptor_length
            ))
        })?;

        match TensorDescriptor::decode(desc_data) {
            Ok(desc) => rows.extend(rows_from_descriptor(
                desc_data,
                &desc,
                entry.descriptor_offset,
            )),
            Err(e) => rows.push(Row {
                offset: entry.descriptor_offset,
                bytes: vec![],
                field: format!("ERROR decoding descriptor: {e}"),
            }),
        }
    }

    Ok(())
}

fn inspect_hrryfile(data: &[u8]) -> (Vec<Row>, Option<Error>) {
    let mut rows = Vec::new();
    match inspect_hrryfile_inner(data, &mut rows) {
        Ok(()) => (rows, None),
        Err(e) => (rows, Some(e)),
    }
}

// ── Raw descriptor inspection ─────────────────────────────────────────────────

fn inspect(data: &[u8]) -> (Vec<Row>, Option<Error>) {
    // Dispatch to HRRYFILE container path when the file magic matches.
    if data.len() >= 8 && &data[..8] == FILE_MAGIC {
        return inspect_hrryfile(data);
    }

    match TensorDescriptor::decode(data) {
        Ok(desc) => (rows_from_descriptor(data, &desc, 0), None),
        Err(e) => {
            // Show the magic bytes (if present) so the caller can see what was read,
            // then let the error message explain why parsing failed.
            let partial = if data.len() >= 4 {
                vec![Row {
                    offset: 0,
                    bytes: data[..4].to_vec(),
                    field: format!(
                        "magic = {:?}",
                        std::str::from_utf8(&data[..4]).unwrap_or("(non-UTF-8)")
                    ),
                }]
            } else {
                vec![]
            };
            (partial, Some(Error::Parse(e)))
        }
    }
}

// ── Hex table rendering ────────────────────────────────────────────────────────

/// Format a byte slice as space-separated uppercase hex pairs (e.g. `"48 52 52 59"`).
fn hex_str(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|b| format!("{b:02X}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Split `bytes` into groups of at most `per_line` bytes and format each as a
/// space-separated hex string.
fn hex_chunks(bytes: &[u8], per_line: usize) -> Vec<String> {
    if bytes.is_empty() {
        return vec![String::new()];
    }
    bytes.chunks(per_line).map(hex_str).collect()
}

/// Print the 3-column hex table for `rows` to stdout.
///
/// Column widths:
/// - Offset: right-aligned in 6 chars
/// - Value (hex): left-aligned in 30 chars (10 bytes per line; wraps if longer)
/// - Field: remainder
///
/// Annotation rows (empty `bytes`) with a field starting with `"── "` are printed
/// as section separators with a blank line prefix.
fn print_table(rows: &[Row]) {
    println!("{:>6}  {:<30}  Field", "Offset", "Value (hex)");
    println!("{:->6}  {:-<30}  {:-<5}", "", "", "");

    for row in rows {
        if row.bytes.is_empty() {
            if row.field.starts_with("── ") {
                // Section separator — blank line + indented title for readability.
                println!();
                println!("       {}", row.field);
            } else {
                println!("{:>6}  {:<30}  {}", "", "", row.field);
            }
        } else {
            let mut first = true;
            for chunk in hex_chunks(&row.bytes, 10) {
                if first {
                    println!("{:>6}  {:<30}  {}", row.offset, chunk, row.field);
                    first = false;
                } else {
                    println!("{:>6}  {:<30}", "", chunk);
                }
            }
        }
    }
}

// ── Entry point ────────────────────────────────────────────────────────────────

fn run() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        return Err(Error::Usage);
    }

    let path = &args[1];
    let data: Vec<u8> = if path == "-" {
        let mut buf = Vec::new();
        io::stdin().read_to_end(&mut buf)?;
        buf
    } else {
        fs::read(path)?
    };

    let (rows, err) = inspect(&data);
    print_table(&rows);

    if let Some(e) = err {
        let error_row = Row {
            offset: 0,
            bytes: vec![],
            field: format!("ERROR: {e}"),
        };
        print_table(&[error_row]);
        eprintln!("error: {e}");
        process::exit(1);
    }

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        process::exit(1);
    }
}
