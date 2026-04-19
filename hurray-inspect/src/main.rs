//! `hurray-inspect` — CLI tool for inspecting Hurray binary tensor descriptor files.
//!
//! # Usage
//!
//! ```text
//! hurray-inspect <file>
//! hurray-inspect -          # read from stdin
//! ```
//!
//! Reads the file (or stdin when the path is `-`), parses the Hurray binary tensor
//! descriptor format as defined in `docs/spec/metadata.md`, and prints a 3-column
//! hex table to stdout:
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

// ──────────────────────────────────────────────────────────────────────────────
// Error type
// ──────────────────────────────────────────────────────────────────────────────

/// Errors produced by `hurray-inspect`.
#[derive(Debug, thiserror::Error)]
enum Error {
    /// Unexpected end of input while reading a field.
    #[error("unexpected end of input at offset {offset} while reading {field}")]
    UnexpectedEof { offset: usize, field: &'static str },

    /// The magic bytes are not `HRRY`.
    #[error("invalid magic bytes: expected 48 52 52 59, got {got}")]
    InvalidMagic { got: String },

    /// `descriptor_length` is smaller than the minimum valid size.
    #[error("descriptor_length {0} is smaller than the minimum valid size of 20 bytes")]
    DescriptorTooShort(u32),

    /// I/O error reading the input.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// Usage error (wrong number of arguments).
    #[error("usage: hurray-inspect <file>  (use '-' for stdin)")]
    Usage,
}

type Result<T> = std::result::Result<T, Error>;

// ──────────────────────────────────────────────────────────────────────────────
// Table row
// ──────────────────────────────────────────────────────────────────────────────

/// A single row in the output hex table.
struct Row {
    /// Byte offset of the first byte of this field within the descriptor.
    offset: usize,
    /// Raw bytes of this field.
    bytes: Vec<u8>,
    /// Human-readable field description.
    field: String,
}

// ──────────────────────────────────────────────────────────────────────────────
// Reader — cursor over a bounded byte slice
// ──────────────────────────────────────────────────────────────────────────────

/// A cursor-based reader over a byte slice, bounded to `limit` bytes.
///
/// Emits [`Row`] entries as it consumes fields.  The `limit` is the value of
/// `descriptor_length` and enforces the spec requirement that a reader MUST NOT
/// read beyond `descriptor_length` bytes.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
    limit: usize,
}

impl<'a> Reader<'a> {
    /// Create a new [`Reader`] over `data`, bounded to `limit` bytes.
    fn new(data: &'a [u8], limit: usize) -> Self {
        Self {
            data,
            pos: 0,
            limit,
        }
    }

    /// Current byte offset within the descriptor.
    fn offset(&self) -> usize {
        self.pos
    }

    /// Consume exactly `n` bytes and return them, or return an error.
    fn take(&mut self, n: usize, field: &'static str) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or(Error::UnexpectedEof {
            offset: self.pos,
            field,
        })?;
        if end > self.limit || end > self.data.len() {
            return Err(Error::UnexpectedEof {
                offset: self.pos,
                field,
            });
        }
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    /// Read a single `u8` field, emitting a [`Row`].
    fn read_u8(&mut self, field: impl Into<String>) -> Result<(u8, Row)> {
        let offset = self.pos;
        let bytes = self.take(1, "u8 field")?;
        let val = bytes[0];
        Ok((
            val,
            Row {
                offset,
                bytes: bytes.to_vec(),
                field: field.into(),
            },
        ))
    }

    /// Read a little-endian `u32` field, emitting a [`Row`].
    fn read_u32(&mut self, field: impl Into<String>) -> Result<(u32, Row)> {
        let offset = self.pos;
        let bytes = self.take(4, "u32 field")?;
        let val = u32::from_le_bytes(bytes.try_into().unwrap_or([0u8; 4]));
        Ok((
            val,
            Row {
                offset,
                bytes: bytes.to_vec(),
                field: field.into(),
            },
        ))
    }

    /// Read a little-endian `u64` field, emitting a [`Row`].
    fn read_u64(&mut self, field: impl Into<String>) -> Result<(u64, Row)> {
        let offset = self.pos;
        let bytes = self.take(8, "u64 field")?;
        let val = u64::from_le_bytes(bytes.try_into().unwrap_or([0u8; 8]));
        Ok((
            val,
            Row {
                offset,
                bytes: bytes.to_vec(),
                field: field.into(),
            },
        ))
    }

    /// Read a little-endian `i64` field, emitting a [`Row`].
    fn read_i64(&mut self, field: impl Into<String>) -> Result<(i64, Row)> {
        let offset = self.pos;
        let bytes = self.take(8, "i64 field")?;
        let val = i64::from_le_bytes(bytes.try_into().unwrap_or([0u8; 8]));
        Ok((
            val,
            Row {
                offset,
                bytes: bytes.to_vec(),
                field: field.into(),
            },
        ))
    }

    /// Read a little-endian `f64` field, emitting a [`Row`].
    fn read_f64(&mut self, field: impl Into<String>) -> Result<(f64, Row)> {
        let offset = self.pos;
        let bytes = self.take(8, "f64 field")?;
        let bits = u64::from_le_bytes(bytes.try_into().unwrap_or([0u8; 8]));
        let val = f64::from_bits(bits);
        Ok((
            val,
            Row {
                offset,
                bytes: bytes.to_vec(),
                field: field.into(),
            },
        ))
    }

    /// Read `n` raw bytes as a reserved/opaque block, emitting a [`Row`].
    fn read_raw(&mut self, n: usize, field: impl Into<String>) -> Result<Row> {
        let offset = self.pos;
        let bytes = self.take(n, "raw bytes")?;
        Ok(Row {
            offset,
            bytes: bytes.to_vec(),
            field: field.into(),
        })
    }

    /// Skip forward so that `self.pos` equals `target`.  Used to consume any
    /// bytes between the successfully parsed fields and the end of the
    /// descriptor (e.g., unknown minor-version additions).
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

// ──────────────────────────────────────────────────────────────────────────────
// Human-readable name helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Return the human-readable name for a `type_tag` byte.
fn type_tag_name(tag: u8) -> &'static str {
    match tag {
        0x01 => "float16",
        0x02 => "bfloat16",
        0x03 => "float32",
        0x04 => "float64",
        0x10 => "int8",
        0x11 => "uint8",
        0x12 => "int16",
        0x13 => "uint16",
        0x14 => "int32",
        0x15 => "uint32",
        0x16 => "int64",
        0x17 => "uint64",
        0x20 => "bool",
        0x40 => "float8_e4m3",
        0x41 => "float8_e5m2",
        0x42 => "float8_e8m0",
        0x48 => "int4",
        0x49 => "uint4",
        0x4A => "int2",
        0x4B => "uint2",
        0x50 => "complex64",
        0x51 => "complex128",
        _ => "unknown",
    }
}

/// Return the human-readable name for a `layout_tag` byte.
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
        0xF0..=0xFE => "extension",
        _ => "unknown",
    }
}

/// Return the human-readable name for a `device_tag` byte.
fn device_tag_name(tag: u8) -> &'static str {
    match tag {
        0x00 => "CPU",
        0x01 => "CUDA",
        0x02 => "ROCm",
        0x03 => "Metal",
        _ => "unknown",
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Hex formatting helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Format a byte slice as space-separated uppercase hex pairs (e.g. `"48 52 52 59"`).
fn hex_str(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|b| format!("{b:02X}"))
        .collect::<Vec<_>>()
        .join(" ")
}

// ──────────────────────────────────────────────────────────────────────────────
// Layout-specific field parser
// ──────────────────────────────────────────────────────────────────────────────

/// Parse layout-specific fields for the given `layout_tag` and `rank`.
///
/// Returns a list of rows.  If the layout has nested layout-specific fields
/// (tiled inner/outer strided, subpaving regions), those are parsed recursively
/// up to `depth_limit` levels deep.
fn parse_layout_fields(
    reader: &mut Reader<'_>,
    layout_tag: u8,
    rank: u32,
    depth_limit: u8,
) -> Result<Vec<Row>> {
    let mut rows = Vec::new();

    match layout_tag {
        // Row-major and column-major: no additional fields.
        0x01 | 0x02 => {}

        // Strided: strides i64[rank]
        0x03 => {
            for i in 0..rank {
                let (val, row) = reader.read_i64(format!("strides[{i}]"))?;
                rows.push(Row {
                    field: format!("strides[{i}] = {val}"),
                    ..row
                });
            }
        }

        // Tiled: tile_shape u64[rank], outer_layout u8, inner_layout u8, _reserved u8[2]
        // then conditional outer_strides / inner_strides / recursive tiling
        0x04 => {
            rows.extend(parse_tiled_fields(reader, rank, depth_limit)?);
        }

        // Morton: morton_bits u32[rank]
        0x05 => {
            for i in 0..rank {
                let (val, row) = reader.read_u32(format!("morton_bits[{i}]"))?;
                rows.push(Row {
                    field: format!("morton_bits[{i}] = {val}"),
                    ..row
                });
            }
        }

        // General Subpaving: region_count u32, then region_count region descriptors
        0x06 => {
            let (region_count, row) = reader.read_u32("region_count")?;
            rows.push(Row {
                field: format!("region_count = {region_count}"),
                ..row
            });
            rows.extend(parse_subpaving_regions(
                reader,
                rank,
                region_count,
                depth_limit,
            )?);
        }

        // COO: nnz u64, is_sorted u8, _reserved u8[7]
        0x07 => {
            let (val, row) = reader.read_u64("nnz")?;
            rows.push(Row {
                field: format!("nnz = {val}"),
                ..row
            });
            let (val, row) = reader.read_u8("is_sorted")?;
            rows.push(Row {
                field: format!("is_sorted = 0x{val:02X}"),
                ..row
            });
            rows.push(reader.read_raw(7, "COO._reserved")?);
        }

        // CSR: nnz u64, _reserved u8[8]
        0x08 => {
            let (val, row) = reader.read_u64("nnz")?;
            rows.push(Row {
                field: format!("nnz = {val}"),
                ..row
            });
            rows.push(reader.read_raw(8, "CSR._reserved")?);
        }

        // CSC: nnz u64, _reserved u8[8]
        0x09 => {
            let (val, row) = reader.read_u64("nnz")?;
            rows.push(Row {
                field: format!("nnz = {val}"),
                ..row
            });
            rows.push(reader.read_raw(8, "CSC._reserved")?);
        }

        // Hilbert: hilbert_order u32, hilbert_rank u32
        0x40 => {
            let (val, row) = reader.read_u32("hilbert_order")?;
            rows.push(Row {
                field: format!("hilbert_order = {val}"),
                ..row
            });
            let (val, row) = reader.read_u32("hilbert_rank")?;
            rows.push(Row {
                field: format!("hilbert_rank = {val}"),
                ..row
            });
        }

        // Extension layouts 0xF0–0xFE
        0xF0..=0xFE => {
            let (val, row) = reader.read_u64("extension_layout_id")?;
            rows.push(Row {
                field: format!("extension_layout_id = 0x{val:016X}"),
                ..row
            });
            let (ext_len, row) = reader.read_u32("extension_data_length")?;
            rows.push(Row {
                field: format!("extension_data_length = {ext_len}"),
                ..row
            });
            if ext_len > 0 {
                rows.push(reader.read_raw(ext_len as usize, "extension_data")?);
            }
        }

        other => {
            rows.push(Row {
                offset: reader.offset(),
                bytes: vec![],
                field: format!(
                    "(unknown layout tag 0x{other:02X} — cannot parse further layout fields)"
                ),
            });
        }
    }

    Ok(rows)
}

/// Parse the fields specific to the tiled layout (tag `0x04`), including
/// conditional `outer_strides`, `inner_strides`, and recursive inner tiling.
fn parse_tiled_fields(reader: &mut Reader<'_>, rank: u32, depth_limit: u8) -> Result<Vec<Row>> {
    let mut rows = Vec::new();

    for i in 0..rank {
        let (val, row) = reader.read_u64(format!("tile_shape[{i}]"))?;
        rows.push(Row {
            field: format!("tile_shape[{i}] = {val}"),
            ..row
        });
    }

    let (outer_layout, row) = reader.read_u8("outer_layout")?;
    rows.push(Row {
        field: format!(
            "outer_layout = 0x{outer_layout:02X} ({})",
            layout_tag_name(outer_layout)
        ),
        ..row
    });

    let (inner_layout, row) = reader.read_u8("inner_layout")?;
    rows.push(Row {
        field: format!(
            "inner_layout = 0x{inner_layout:02X} ({})",
            layout_tag_name(inner_layout)
        ),
        ..row
    });

    rows.push(reader.read_raw(2, "tiled._reserved")?);

    // Conditional outer strides
    if outer_layout == 0x03 {
        for i in 0..rank {
            let (val, row) = reader.read_i64(format!("outer_strides[{i}]"))?;
            rows.push(Row {
                field: format!("outer_strides[{i}] = {val}"),
                ..row
            });
        }
    }

    // Conditional inner strides or recursive tiling
    if inner_layout == 0x03 {
        for i in 0..rank {
            let (val, row) = reader.read_i64(format!("inner_strides[{i}]"))?;
            rows.push(Row {
                field: format!("inner_strides[{i}] = {val}"),
                ..row
            });
        }
    } else if inner_layout == 0x04 {
        if depth_limit == 0 {
            rows.push(Row {
                offset: reader.offset(),
                bytes: vec![],
                field: "(recursion limit reached — cannot parse nested tiled layout)".to_string(),
            });
        } else {
            rows.extend(parse_tiled_fields(reader, rank, depth_limit - 1)?);
        }
    }

    Ok(rows)
}

/// Parse `region_count` subpaving region descriptors.
fn parse_subpaving_regions(
    reader: &mut Reader<'_>,
    rank: u32,
    region_count: u32,
    depth_limit: u8,
) -> Result<Vec<Row>> {
    let mut rows = Vec::new();

    for r in 0..region_count {
        for i in 0..rank {
            let (val, row) = reader.read_u64(format!("region[{r}].origin[{i}]"))?;
            rows.push(Row {
                field: format!("region[{r}].origin[{i}] = {val}"),
                ..row
            });
        }
        for i in 0..rank {
            let (val, row) = reader.read_u64(format!("region[{r}].region_shape[{i}]"))?;
            rows.push(Row {
                field: format!("region[{r}].region_shape[{i}] = {val}"),
                ..row
            });
        }

        let (region_layout_tag, row) = reader.read_u8(format!("region[{r}].region_layout_tag"))?;
        rows.push(Row {
            field: format!(
                "region[{r}].region_layout_tag = 0x{region_layout_tag:02X} ({})",
                layout_tag_name(region_layout_tag)
            ),
            ..row
        });
        rows.push(reader.read_raw(3, "region._reserved")?);

        let (buf_idx, row) = reader.read_u32(format!("region[{r}].buffer_index"))?;
        rows.push(Row {
            field: format!("region[{r}].buffer_index = {buf_idx}"),
            ..row
        });

        let (byte_off, row) = reader.read_u64(format!("region[{r}].region_byte_offset"))?;
        rows.push(Row {
            field: format!("region[{r}].region_byte_offset = {byte_off}"),
            ..row
        });

        // Recursive layout-specific fields for this region
        if depth_limit == 0 {
            rows.push(Row {
                offset: reader.offset(),
                bytes: vec![],
                field: format!(
                    "region[{r}]: (recursion limit reached — cannot parse region layout fields)"
                ),
            });
        } else {
            rows.extend(parse_layout_fields(
                reader,
                region_layout_tag,
                rank,
                depth_limit - 1,
            )?);
        }
    }

    Ok(rows)
}

// ──────────────────────────────────────────────────────────────────────────────
// Top-level descriptor parser
// ──────────────────────────────────────────────────────────────────────────────

/// Parse the Hurray binary tensor descriptor from `data` and return all [`Row`]s.
///
/// On parse failure after the magic check, returns the rows collected so far
/// plus an error row, so the caller can still print a partial table.
fn parse_descriptor(data: &[u8]) -> (Vec<Row>, Option<Error>) {
    let mut rows = Vec::new();

    // We need at least 10 bytes (magic + version + descriptor_length) to
    // bootstrap the bounded reader.
    if data.len() < 10 {
        return (
            rows,
            Some(Error::UnexpectedEof {
                offset: 0,
                field: "fixed header",
            }),
        );
    }

    // --- Magic (4 bytes) — unbounded check before we know the limit ---
    let magic = &data[0..4];
    let magic_row = Row {
        offset: 0,
        bytes: magic.to_vec(),
        field: format!(
            "magic = {:?}",
            std::str::from_utf8(magic).unwrap_or("(non-UTF-8)")
        ),
    };
    rows.push(magic_row);

    if magic != b"HRRY" {
        return (
            rows,
            Some(Error::InvalidMagic {
                got: hex_str(magic),
            }),
        );
    }

    // Peek descriptor_length at offset 6 (4 bytes) before building the reader.
    let desc_len = u32::from_le_bytes(data[6..10].try_into().unwrap_or([0u8; 4]));
    if desc_len < 20 {
        // Still emit what we have, then error.
        return (rows, Some(Error::DescriptorTooShort(desc_len)));
    }

    let limit = (desc_len as usize).min(data.len());
    let mut reader = Reader::new(data, limit);
    // Advance past the 4 magic bytes already consumed above.
    reader.pos = 4;

    match parse_descriptor_body(&mut reader, &mut rows, desc_len) {
        Ok(()) => (rows, None),
        Err(e) => (rows, Some(e)),
    }
}

/// Parse everything after the magic bytes.  Mutates `rows` in place.
fn parse_descriptor_body(
    reader: &mut Reader<'_>,
    rows: &mut Vec<Row>,
    desc_len: u32,
) -> Result<()> {
    // version_major
    let (ver_major, row) = reader.read_u8("version_major")?;
    rows.push(Row {
        field: format!("version_major = {ver_major}"),
        ..row
    });

    // version_minor
    let (ver_minor, row) = reader.read_u8("version_minor")?;
    rows.push(Row {
        field: format!("version_minor = {ver_minor}"),
        ..row
    });

    // descriptor_length
    let (dl, row) = reader.read_u32("descriptor_length")?;
    rows.push(Row {
        field: format!("descriptor_length = {dl}"),
        ..row
    });

    // flags
    let (flags, row) = reader.read_u32("flags")?;
    rows.push(Row {
        field: format!("flags = 0x{flags:08X}"),
        ..row
    });

    // type_tag
    let (type_tag, row) = reader.read_u8("type_tag")?;
    rows.push(Row {
        field: format!("type_tag = 0x{type_tag:02X} ({})", type_tag_name(type_tag)),
        ..row
    });

    // layout_tag
    let (layout_tag, row) = reader.read_u8("layout_tag")?;
    rows.push(Row {
        field: format!(
            "layout_tag = 0x{layout_tag:02X} ({})",
            layout_tag_name(layout_tag)
        ),
        ..row
    });

    // rank
    let (rank, row) = reader.read_u32("rank")?;
    rows.push(Row {
        field: format!("rank = {rank}"),
        ..row
    });

    // shape[0..rank]
    for i in 0..rank {
        let (val, row) = reader.read_u64(format!("shape[{i}]"))?;
        let display = if val == u64::MAX {
            "(dynamic)".to_string()
        } else {
            val.to_string()
        };
        rows.push(Row {
            field: format!("shape[{i}] = {display}"),
            ..row
        });
    }

    // byte_offset
    let (byte_offset, row) = reader.read_u64("byte_offset")?;
    rows.push(Row {
        field: format!("byte_offset = {byte_offset}"),
        ..row
    });

    // Layout-specific fields (up to 8 recursion levels deep)
    rows.extend(parse_layout_fields(reader, layout_tag, rank, 8)?);

    // Buffer table: buffer_count u8
    let (buffer_count, row) = reader.read_u8("buffer_count")?;
    rows.push(Row {
        field: format!("buffer_count = {buffer_count}"),
        ..row
    });

    // buffer_count × 16-byte handles
    for b in 0..buffer_count {
        let (byte_size, row) = reader.read_u64(format!("buffer[{b}].byte_size"))?;
        rows.push(Row {
            field: format!("buffer[{b}].byte_size = {byte_size}"),
            ..row
        });

        let (alignment, row) = reader.read_u32(format!("buffer[{b}].alignment"))?;
        rows.push(Row {
            field: format!("buffer[{b}].alignment = {alignment}"),
            ..row
        });

        let (device_tag, row) = reader.read_u8(format!("buffer[{b}].device_tag"))?;
        rows.push(Row {
            field: format!(
                "buffer[{b}].device_tag = 0x{device_tag:02X} ({})",
                device_tag_name(device_tag)
            ),
            ..row
        });

        rows.push(reader.read_raw(3, "buffer._reserved")?);
    }

    // ── Optional sections (determined by flags) ──────────────────────────────

    // bit 0 — HAS_QUANTIZATION
    if flags & 0x01 != 0 {
        let (q_len, row) = reader.read_u32("quantization_length")?;
        rows.push(Row {
            field: format!("quantization_length = {q_len}"),
            ..row
        });
        if q_len > 0 {
            rows.push(reader.read_raw(q_len as usize, "quantization_descriptor")?);
        }
    }

    // bit 1 — HAS_SHARD
    if flags & 0x02 != 0 {
        for i in 0..rank {
            let (val, row) = reader.read_u64(format!("parent_shape[{i}]"))?;
            rows.push(Row {
                field: format!("parent_shape[{i}] = {val}"),
                ..row
            });
        }
        for i in 0..rank {
            let (val, row) = reader.read_u64(format!("shard_offset[{i}]"))?;
            rows.push(Row {
                field: format!("shard_offset[{i}] = {val}"),
                ..row
            });
        }
    }

    // bit 3 — HAS_STATISTICS (72-byte block)
    if flags & 0x08 != 0 {
        let (computed_mask, row) = reader.read_u32("stats.computed_mask")?;
        rows.push(Row {
            field: format!("stats.computed_mask = 0x{computed_mask:08X}"),
            ..row
        });
        rows.push(reader.read_raw(4, "stats._reserved")?);

        let (val, row) = reader.read_u64("stats.nnz")?;
        rows.push(Row {
            field: format!("stats.nnz = {val}"),
            ..row
        });
        let (val, row) = reader.read_f64("stats.sparsity_ratio")?;
        rows.push(Row {
            field: format!("stats.sparsity_ratio = {val}"),
            ..row
        });
        let (val, row) = reader.read_f64("stats.value_min")?;
        rows.push(Row {
            field: format!("stats.value_min = {val}"),
            ..row
        });
        let (val, row) = reader.read_f64("stats.value_max")?;
        rows.push(Row {
            field: format!("stats.value_max = {val}"),
            ..row
        });
        let (val, row) = reader.read_f64("stats.value_abs_max")?;
        rows.push(Row {
            field: format!("stats.value_abs_max = {val}"),
            ..row
        });
        let (val, row) = reader.read_f64("stats.value_mean")?;
        rows.push(Row {
            field: format!("stats.value_mean = {val}"),
            ..row
        });
        let (val, row) = reader.read_f64("stats.value_stddev")?;
        rows.push(Row {
            field: format!("stats.value_stddev = {val}"),
            ..row
        });
        let (val, row) = reader.read_u8("stats.nm_n")?;
        rows.push(Row {
            field: format!("stats.nm_n = {val}"),
            ..row
        });
        let (val, row) = reader.read_u8("stats.nm_m")?;
        rows.push(Row {
            field: format!("stats.nm_m = {val}"),
            ..row
        });
        let (val, row) = reader.read_u8("stats.has_nan")?;
        rows.push(Row {
            field: format!("stats.has_nan = {val}"),
            ..row
        });
        let (val, row) = reader.read_u8("stats.has_inf")?;
        rows.push(Row {
            field: format!("stats.has_inf = {val}"),
            ..row
        });
        rows.push(reader.read_raw(4, "stats._reserved2")?);
    }

    // bit 2 — HAS_EXTENSION_TYPE (20-byte block)
    if flags & 0x04 != 0 {
        let (val, row) = reader.read_u32("ext_type.bit_width")?;
        rows.push(Row {
            field: format!("ext_type.bit_width = {val}"),
            ..row
        });
        let (val, row) = reader.read_u8("ext_type.packing_factor")?;
        rows.push(Row {
            field: format!("ext_type.packing_factor = {val}"),
            ..row
        });
        let (val, row) = reader.read_u8("ext_type.is_float")?;
        rows.push(Row {
            field: format!("ext_type.is_float = {val}"),
            ..row
        });
        let (val, row) = reader.read_u8("ext_type.is_signed")?;
        rows.push(Row {
            field: format!("ext_type.is_signed = {val}"),
            ..row
        });
        let (val, row) = reader.read_u8("ext_type.sign_bits")?;
        rows.push(Row {
            field: format!("ext_type.sign_bits = {val}"),
            ..row
        });
        let (val, row) = reader.read_u8("ext_type.exponent_bits")?;
        rows.push(Row {
            field: format!("ext_type.exponent_bits = {val}"),
            ..row
        });
        let (val, row) = reader.read_u8("ext_type.mantissa_bits")?;
        rows.push(Row {
            field: format!("ext_type.mantissa_bits = {val}"),
            ..row
        });
        rows.push(reader.read_raw(2, "ext_type._reserved")?);
        let (val, row) = reader.read_u32("ext_type.exponent_bias")?;
        rows.push(Row {
            field: format!("ext_type.exponent_bias = {val}"),
            ..row
        });
        let (val, row) = reader.read_u8("ext_type.has_nan")?;
        rows.push(Row {
            field: format!("ext_type.has_nan = {val}"),
            ..row
        });
        let (val, row) = reader.read_u8("ext_type.has_inf")?;
        rows.push(Row {
            field: format!("ext_type.has_inf = {val}"),
            ..row
        });
        rows.push(reader.read_raw(2, "ext_type._reserved2")?);
    }

    // Consume any trailing bytes within the descriptor window (e.g., future
    // minor-version additions that this reader does not recognise).
    if let Some(row) = reader.skip_to(desc_len as usize) {
        rows.push(row);
    }

    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────────
// Table rendering
// ──────────────────────────────────────────────────────────────────────────────

/// Print the 3-column hex table for `rows` to stdout.
///
/// Column widths:
/// - Offset: right-aligned in 6 chars
/// - Value (hex): left-aligned in 30 chars
/// - Field: remainder
fn print_table(rows: &[Row]) {
    // Header
    println!("{:>6}  {:<30}  Field", "Offset", "Value (hex)");
    println!("{:->6}  {:-<30}  {:-<5}", "", "", "");

    for row in rows {
        if row.bytes.is_empty() {
            // Annotation-only row (e.g., recursion-limit notice)
            println!("{:>6}  {:<30}  {}", "", "", row.field);
        } else {
            // Format hex — may be long; we chunk at 30 chars (10 bytes) per line.
            // Each "XX " is 3 chars; 10 pairs occupy 29 chars, fitting the 30-char column.
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

/// Split `bytes` into groups of at most `per_line` bytes and format each as a
/// space-separated hex string.
fn hex_chunks(bytes: &[u8], per_line: usize) -> Vec<String> {
    if bytes.is_empty() {
        return vec![String::new()];
    }
    bytes.chunks(per_line).map(hex_str).collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// Entry point
// ──────────────────────────────────────────────────────────────────────────────

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

    let (rows, err) = parse_descriptor(&data);

    print_table(&rows);

    if let Some(e) = err {
        // Print the error as a final row so it appears in context.
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

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

// The `try_into().unwrap_or(...)` used in the read_* methods is only hit when
// the slice length does not match the array size, which can never occur because
// `take()` guarantees exactly the requested number of bytes.  We avoid
// `unwrap()` in public API code; the `unwrap_or` in private reader helpers is
// a safe fallback that satisfies the no-unwrap rule while keeping the code
// compact.  A dedicated `array_ref` macro would be cleaner but adds a
// dependency.
