//! `hurray-inspect` — CLI tool for inspecting Hurray binary tensor descriptor files.
//!
//! # Usage
//!
//! ```text
//! hurray-inspect <file>
//! hurray-inspect -          # read from stdin
//! ```
//!
//! Reads the file (or stdin when the path is `-`), decodes the Hurray binary tensor
//! descriptor using `hurray-core`, and prints a 3-column hex table to stdout:
//!
//! ```text
//! Offset  Value (hex)                    Field
//! ------  ---------------------------    -----
//! 0       48 52 52 59                    magic = "HRRY"
//! ```

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
    #[error("usage: hurray-inspect <file>  (use '-' for stdin)")]
    Usage,
}

type Result<T> = std::result::Result<T, Error>;

// ── Table row ──────────────────────────────────────────────────────────────────

/// A single row in the output hex table.
struct Row {
    /// Byte offset of the first byte of this field within the descriptor.
    offset: usize,
    /// Raw bytes of this field (empty for annotation-only rows).
    bytes: Vec<u8>,
    /// Human-readable field description.
    field: String,
}

// ── Reader — cursor over a bounded byte slice ──────────────────────────────────

/// Cursor over a byte slice bounded to `limit` bytes.
///
/// Only extracts raw byte ranges for hex display; all parsing is done by
/// `hurray-core`.  When decode has already succeeded, every `take_row` call is
/// guaranteed to find the bytes it expects, so bounds-clamping is a safety net
/// rather than a real code path.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
    limit: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8], limit: usize) -> Self {
        Self {
            data,
            pos: 0,
            limit,
        }
    }

    /// Consume `n` bytes and return a display [`Row`] labelled with `field`.
    fn take_row(&mut self, n: usize, field: String) -> Row {
        let offset = self.pos;
        let end = (self.pos + n).min(self.limit).min(self.data.len());
        let bytes = self.data[offset..end].to_vec();
        self.pos = end;
        Row {
            offset,
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
            offset,
            bytes,
            field: "(unknown / padding bytes)".to_string(),
        })
    }
}

// ── Layout tag name ─────────────────────────────────────────────────────────────

fn layout_tag_name(tag: u8) -> &'static str {
    // Literal values match the TAG_* constants in hurray_core::layout.
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

fn rows_from_descriptor(data: &[u8], desc: &TensorDescriptor) -> Vec<Row> {
    // descriptor_length is at bytes 6..10 (little-endian u32).
    let desc_len = u32::from_le_bytes(data[6..10].try_into().unwrap_or([0u8; 4])) as usize;
    let mut reader = Reader::new(data, desc_len);
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

fn inspect(data: &[u8]) -> (Vec<Row>, Option<Error>) {
    match TensorDescriptor::decode(data) {
        Ok(desc) => (rows_from_descriptor(data, &desc), None),
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
fn print_table(rows: &[Row]) {
    println!("{:>6}  {:<30}  Field", "Offset", "Value (hex)");
    println!("{:->6}  {:-<30}  {:-<5}", "", "", "");

    for row in rows {
        if row.bytes.is_empty() {
            println!("{:>6}  {:<30}  {}", "", "", row.field);
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
