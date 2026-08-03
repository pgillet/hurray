//! `hurray-inspect` — CLI tool for inspecting Hurray files.
//!
//! # Usage
//!
//! ```text
//! hurray-inspect [--data[=hex|numpy]] <file>
//! hurray-inspect -                           # read from stdin
//! ```
//!
//! Auto-detects the file type:
//!
//! - **HRRYFILE container** (magic `HRRYFILE`): displays the file header,
//!   trailer, index, KV metadata section, and each tensor's descriptor.
//! - **Raw tensor descriptor** (magic `HRRY`): displays the descriptor fields.
//!
//! With `--data` (or `--data=numpy`), tensor buffer values are printed after each
//! descriptor in NumPy-style nested bracket notation.  With `--data=hex`, the raw
//! buffer bytes are printed as a hex table.  For raw descriptors the buffer bytes
//! must immediately follow the descriptor in the same file.

use std::{
    env, fs,
    io::{self, Read},
    process,
};

use half::{bf16, f16};

use hurray_core::{
    descriptor::{MemberRole, TensorDescriptor},
    layout::{
        BlockTableIndexType, CombineOp, CompositionRule, KvRole, LayoutDescriptor, TiledLayout,
    },
    QuantizationDescriptor, DYNAMIC,
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
    #[error("usage: hurray-inspect [--data[=hex|numpy]] <file>  (use '-' for stdin)")]
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

// ── Data display mode ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
enum DataMode {
    None,
    Hex,
    Numpy,
}

// ── Value formatting ───────────────────────────────────────────────────────────

/// Format a single float as a string, adding a trailing `.` when the value
/// has no fractional part (matching NumPy's display convention).
fn format_float(v: f64) -> String {
    if v.is_nan() {
        return "nan".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let s = format!("{v}");
    if s.contains('.') || s.contains('e') || s.contains('E') {
        s
    } else {
        format!("{s}.")
    }
}

/// Decode a fixed-width little-endian integer at element index `idx`.
macro_rules! read_le {
    ($data:expr, $idx:expr, $n:expr, $ty:ty) => {{
        let i = ($idx * $n) as usize;
        if i + $n <= $data.len() {
            <$ty>::from_le_bytes($data[i..i + $n].try_into().unwrap()).to_string()
        } else {
            "?".into()
        }
    }};
}

/// Extracts a power-of-two sub-byte code (`bits` ∈ {1, 2, 4}) at element `idx`, LSB-first.
fn read_pow2_code(data: &[u8], idx: u64, bits: u8) -> Option<u32> {
    let per_byte = 8 / bits as u64;
    let byte_i = (idx / per_byte) as usize;
    let shift = ((idx % per_byte) * bits as u64) as u32;
    let mask = (1u32 << bits) - 1;
    data.get(byte_i).map(|&b| (b as u32 >> shift) & mask)
}

/// Extracts a 6-bit code at element `idx` from the 4-elements-per-3-bytes LSB-first packing
/// (`element-types.md` § 6-bit packing).
fn read_f6_code(data: &[u8], idx: u64) -> Option<u32> {
    let g = (idx / 4) as usize * 3;
    let (b0, b1, b2) = (
        *data.get(g)? as u32,
        *data.get(g + 1).unwrap_or(&0) as u32,
        *data.get(g + 2).unwrap_or(&0) as u32,
    );
    Some(match idx % 4 {
        0 => b0 & 0x3F,
        1 => (b0 >> 6) | ((b1 & 0x0F) << 2),
        2 => (b1 >> 4) | ((b2 & 0x03) << 4),
        _ => b2 >> 2,
    })
}

/// Sign-extends a `bits`-wide two's-complement `code` to `i64`.
fn sign_extend(code: u32, bits: u32) -> i64 {
    let shift = 64 - bits;
    ((code as i64) << shift) >> shift
}

/// Decodes an OCP micro-float `code` (sign|exponent|mantissa, LSB→MSB) to `f64`.
///
/// `has_inf`: all-ones exponent is infinity (mantissa 0) or NaN (e5m2). `nan_at_max`:
/// all-ones exponent AND mantissa is NaN but other max-exponent patterns are normal (e4m3).
/// When neither is set (e2m1, float6) the max exponent is an ordinary normal value.
fn micro_float(
    code: u32,
    exp_bits: u32,
    man_bits: u32,
    bias: i32,
    has_inf: bool,
    nan_at_max: bool,
) -> f64 {
    let sign = if (code >> (exp_bits + man_bits)) & 1 == 1 {
        -1.0
    } else {
        1.0
    };
    let exp = (code >> man_bits) & ((1 << exp_bits) - 1);
    let man_mask = (1u32 << man_bits) - 1;
    let man = code & man_mask;
    let max_exp = (1u32 << exp_bits) - 1;
    let man_div = (1u64 << man_bits) as f64;
    if exp == max_exp && has_inf {
        return if man == 0 {
            sign * f64::INFINITY
        } else {
            f64::NAN
        };
    }
    if exp == max_exp && nan_at_max && man == man_mask {
        return f64::NAN;
    }
    if exp == 0 {
        sign * (man as f64 / man_div) * 2f64.powi(1 - bias) // subnormal / zero
    } else {
        sign * (1.0 + man as f64 / man_div) * 2f64.powi(exp as i32 - bias) // normal
    }
}

/// Decodes a `float8_e8m0` byte (power-of-two scale; `0x00`/`0xFF` reserved → NaN).
fn e8m0(byte: u8) -> f64 {
    if byte == 0x00 || byte == 0xFF {
        f64::NAN
    } else {
        2f64.powi(byte as i32 - 127)
    }
}

/// Decode and format the element at logical index `idx` from `data`.
///
/// Complex types are formatted as `(real+imagj)`; `float128` (no stable Rust type) and any
/// unknown/private type are shown as raw hex.
fn format_scalar(et: hurray_core::ElementType, data: &[u8], idx: u64) -> String {
    use hurray_core::ElementType::*;

    match et {
        Bool => {
            let byte = (idx / 8) as usize;
            let bit = (idx % 8) as u8;
            if byte < data.len() {
                if (data[byte] >> bit) & 1 == 1 {
                    "True".into()
                } else {
                    "False".into()
                }
            } else {
                "?".into()
            }
        }
        Int8 => {
            let i = idx as usize;
            if i < data.len() {
                (data[i] as i8).to_string()
            } else {
                "?".into()
            }
        }
        Uint8 => {
            let i = idx as usize;
            if i < data.len() {
                data[i].to_string()
            } else {
                "?".into()
            }
        }
        Int16 => read_le!(data, idx, 2, i16),
        Uint16 => read_le!(data, idx, 2, u16),
        Int32 => read_le!(data, idx, 4, i32),
        Uint32 => read_le!(data, idx, 4, u32),
        Int64 => read_le!(data, idx, 8, i64),
        Uint64 => read_le!(data, idx, 8, u64),
        Float16 => {
            let i = (idx * 2) as usize;
            if i + 2 <= data.len() {
                let bits = u16::from_le_bytes(data[i..i + 2].try_into().unwrap());
                format_float(f16::from_bits(bits).to_f64())
            } else {
                "?".into()
            }
        }
        BFloat16 => {
            let i = (idx * 2) as usize;
            if i + 2 <= data.len() {
                let bits = u16::from_le_bytes(data[i..i + 2].try_into().unwrap());
                format_float(bf16::from_bits(bits).to_f64())
            } else {
                "?".into()
            }
        }
        Float32 => {
            let i = (idx * 4) as usize;
            if i + 4 <= data.len() {
                format_float(f32::from_le_bytes(data[i..i + 4].try_into().unwrap()) as f64)
            } else {
                "?".into()
            }
        }
        Float64 => {
            let i = (idx * 8) as usize;
            if i + 8 <= data.len() {
                format_float(f64::from_le_bytes(data[i..i + 8].try_into().unwrap()))
            } else {
                "?".into()
            }
        }
        Complex64 => {
            let i = (idx * 8) as usize;
            if i + 8 <= data.len() {
                let re = f32::from_le_bytes(data[i..i + 4].try_into().unwrap());
                let im = f32::from_le_bytes(data[i + 4..i + 8].try_into().unwrap());
                format!("({}+{}j)", format_float(re as f64), format_float(im as f64))
            } else {
                "?".into()
            }
        }
        Complex128 => {
            let i = (idx * 16) as usize;
            if i + 16 <= data.len() {
                let re = f64::from_le_bytes(data[i..i + 8].try_into().unwrap());
                let im = f64::from_le_bytes(data[i + 8..i + 16].try_into().unwrap());
                format!("({}+{}j)", format_float(re), format_float(im))
            } else {
                "?".into()
            }
        }
        // Sub-byte integers (LSB-first packing).
        Int4 => read_pow2_code(data, idx, 4).map_or("?".into(), |c| sign_extend(c, 4).to_string()),
        Uint4 => read_pow2_code(data, idx, 4).map_or("?".into(), |c| c.to_string()),
        Int2 => read_pow2_code(data, idx, 2).map_or("?".into(), |c| sign_extend(c, 2).to_string()),
        Uint2 => read_pow2_code(data, idx, 2).map_or("?".into(), |c| c.to_string()),

        // Tier 2 micro-floats (OCP OFP8 / MX).
        Float8E4M3 => data.get(idx as usize).map_or("?".into(), |&b| {
            format_float(micro_float(b as u32, 4, 3, 7, false, true))
        }),
        Float8E5M2 => data.get(idx as usize).map_or("?".into(), |&b| {
            format_float(micro_float(b as u32, 5, 2, 15, true, false))
        }),
        Float8E8M0 => data
            .get(idx as usize)
            .map_or("?".into(), |&b| format_float(e8m0(b))),
        Float4E2M1 => read_pow2_code(data, idx, 4).map_or("?".into(), |c| {
            format_float(micro_float(c, 2, 1, 1, false, false))
        }),
        Float6E2M3 => read_f6_code(data, idx).map_or("?".into(), |c| {
            format_float(micro_float(c, 2, 3, 1, false, false))
        }),
        Float6E3M2 => read_f6_code(data, idx).map_or("?".into(), |c| {
            format_float(micro_float(c, 3, 2, 3, false, false))
        }),

        // float128 (no stable Rust type) and any unknown/private type: raw hex bytes.
        _ => {
            let bw = et.bit_width();
            if bw == 0 || bw >= 8 {
                let n = bw.div_ceil(8).max(1) as usize;
                let i = (idx * n as u64) as usize;
                if i + n <= data.len() {
                    hex_str(&data[i..i + n])
                } else {
                    "?".into()
                }
            } else {
                // Sub-byte: decode the packed nibble/bits for this element.
                let elements_per_byte = 8 / bw as u64;
                let byte_i = (idx / elements_per_byte) as usize;
                let bit_off = ((idx % elements_per_byte) * bw as u64) as u8;
                let mask = (1u8 << bw) - 1;
                if byte_i < data.len() {
                    let raw = (data[byte_i] >> bit_off) & mask;
                    format!("0x{raw:01X}")
                } else {
                    "?".into()
                }
            }
        }
    }
}

/// Maximum number of elements shown before truncation (matches NumPy default).
const NUMPY_THRESHOLD: u64 = 1000;

/// Format tensor data as a NumPy-style nested-bracket string.
///
/// Elements beyond `NUMPY_THRESHOLD` are elided with `...`.
fn format_numpy(et: hurray_core::ElementType, shape: &[u64], data: &[u8]) -> String {
    let total: u64 = if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    };

    if total == 0 {
        return "[]".to_string();
    }

    if shape.is_empty() {
        return format_scalar(et, data, 0);
    }

    if total > NUMPY_THRESHOLD {
        let head = 3u64.min(total);
        let tail = 3u64.min(total - head);
        let mut parts: Vec<String> = (0..head).map(|i| format_scalar(et, data, i)).collect();
        parts.push("...".into());
        for i in (total - tail)..total {
            parts.push(format_scalar(et, data, i));
        }
        return format!("[{}]  # shape={shape:?}, {total} elements", parts.join(" "));
    }

    format_numpy_dim(et, shape, data, 0, 0)
}

/// Recursively build nested-bracket notation for dimension `depth`.
fn format_numpy_dim(
    et: hurray_core::ElementType,
    shape: &[u64],
    data: &[u8],
    depth: usize,
    offset: u64,
) -> String {
    let ndim = shape.len();

    if depth == ndim - 1 {
        // Innermost dimension: single row of space-separated values.
        let parts: Vec<String> = (0..shape[depth])
            .map(|i| format_scalar(et, data, offset + i))
            .collect();
        return format!("[{}]", parts.join(" "));
    }

    let stride: u64 = shape[depth + 1..].iter().product();
    let parts: Vec<String> = (0..shape[depth])
        .map(|i| format_numpy_dim(et, shape, data, depth + 1, offset + i * stride))
        .collect();

    // Between groups: blank line for all but the two innermost dims (numpy style).
    let indent = " ".repeat(depth + 1);
    let sep = if ndim - depth > 2 {
        format!("\n\n{indent}")
    } else {
        format!("\n{indent}")
    };
    format!("[{}]", parts.join(&sep))
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
        0x06 => "COO",
        0x07 => "CSR",
        0x08 => "CSC",
        0x09 => "CSF",
        0x0A => "block-paged",
        0x0B => "composite",
        0x40 => "hilbert",
        0xF0..=0xFE => "private-extension",
        _ => "unknown",
    }
}

// ── Quantization descriptor rows ─────────────────────────────────────────────────

fn quant_scheme_name(tag: u8) -> &'static str {
    match tag {
        0x01 => "per-tensor-affine",
        0x02 => "per-channel-affine",
        0x03 => "per-block-affine",
        0x04 => "NF4",
        0x05 => "MXFP",
        _ => "unknown",
    }
}

/// Renders the decoded quantization descriptor: the 4-byte header plus the per-scheme fields
/// (byte-accurate to `quantization/*.md`).
fn quant_rows(reader: &mut Reader<'_>, qd: &QuantizationDescriptor) -> Vec<Row> {
    let tag = qd.scheme_tag().tag();
    let mut rows = vec![
        reader.take_row(
            1,
            format!("scheme_tag = 0x{tag:02X} ({})", quant_scheme_name(tag)),
        ),
        reader.take_row(1, "scheme_version".to_string()),
        reader.take_row(2, "quant.flags".to_string()),
    ];
    let zp = |z: Option<u32>| match z {
        Some(x) => x.to_string(),
        None => "none (symmetric)".to_string(),
    };
    match qd {
        QuantizationDescriptor::PerTensorAffine(x) => {
            rows.push(reader.take_row(4, format!("scale = {}", x.scale())));
            rows.push(reader.take_row(4, format!("zero_point = {}", x.zero_point())));
            rows.push(reader.take_row(4, "quant._reserved".to_string()));
        }
        QuantizationDescriptor::PerChannelAffine(x) => {
            rows.push(reader.take_row(4, format!("axis = {}", x.axis())));
            rows.push(reader.take_row(
                4,
                format!("scale_buffer_index = {}", x.scale_buffer_index()),
            ));
            rows.push(reader.take_row(
                4,
                format!(
                    "zero_point_buffer_index = {}",
                    zp(x.zero_point_buffer_index())
                ),
            ));
            rows.push(reader.take_row(1, format!("scale_type = {}", x.scale_type())));
            rows.push(reader.take_row(3, "quant._reserved".to_string()));
        }
        QuantizationDescriptor::PerBlockAffine(x) => {
            rows.push(reader.take_row(4, format!("axis = {}", x.axis())));
            rows.push(reader.take_row(4, format!("block_size = {}", x.block_size())));
            rows.push(reader.take_row(
                4,
                format!("scale_buffer_index = {}", x.scale_buffer_index()),
            ));
            rows.push(reader.take_row(
                4,
                format!(
                    "zero_point_buffer_index = {}",
                    zp(x.zero_point_buffer_index())
                ),
            ));
            rows.push(reader.take_row(1, "scale_type_tag".to_string()));
            rows.push(reader.take_row(3, "quant._reserved".to_string()));
        }
        QuantizationDescriptor::Nf4(x) => {
            rows.push(reader.take_row(4, format!("axis = {}", x.axis())));
            rows.push(reader.take_row(4, format!("block_size = {}", x.block_size())));
            rows.push(reader.take_row(
                4,
                format!("scale_buffer_index = {}", x.scale_buffer_index()),
            ));
        }
        QuantizationDescriptor::Mxfp(x) => {
            rows.push(reader.take_row(4, format!("axis = {}", x.axis())));
            rows.push(reader.take_row(4, format!("block_size = {}", x.block_size())));
            rows.push(reader.take_row(
                4,
                format!("scale_buffer_index = {}", x.scale_buffer_index()),
            ));
        }
    }
    rows
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

        LayoutDescriptor::Csf(c) => {
            let mut rows = vec![reader.take_row(8, format!("nnz = {}", c.nnz))];
            for (i, &d) in c.mode_order.iter().enumerate() {
                rows.push(reader.take_row(4, format!("mode_order[{i}] = {d}")));
            }
            rows.push(reader.take_row(8, "CSF._reserved".to_string()));
            rows
        }

        LayoutDescriptor::BlockPaged(bp) => {
            let kv_n = match bp.kv_role {
                KvRole::Key => "key",
                KvRole::Value => "value",
                KvRole::Fused => "fused",
                _ => "unknown",
            };
            let bt_n = match bp.block_table_index_type {
                BlockTableIndexType::U32 => "uint32",
                BlockTableIndexType::U64 => "uint64",
                _ => "unknown",
            };
            let layer = match bp.layer_index {
                Some(x) => x.to_string(),
                None => "none".to_string(),
            };
            vec![
                reader.take_row(4, format!("page_size = {}", bp.page_size)),
                reader.take_row(8, format!("num_pages = {}", bp.num_pages)),
                reader.take_row(4, format!("paged_axis = {}", bp.paged_axis)),
                reader.take_row(4, format!("num_seqs = {}", bp.num_seqs)),
                reader.take_row(1, format!("kv_role = {kv_n}")),
                reader.take_row(4, format!("layer_index = {layer}")),
                reader.take_row(1, format!("block_table_index_type = {bt_n}")),
                reader.take_row(6, "block-paged._reserved".to_string()),
            ]
        }

        LayoutDescriptor::Composite(c) => {
            let rule_n = match &c.rule {
                CompositionRule::Partition => "partition",
                CompositionRule::Overlay(_) => "overlay",
                CompositionRule::Group => "group",
                _ => "unknown",
            };
            let op_n = match &c.rule {
                CompositionRule::Overlay(CombineOp::Replace) => "replace",
                CompositionRule::Overlay(CombineOp::Add) => "add",
                _ => "n/a",
            };
            vec![
                reader.take_row(1, format!("composition_rule = {rule_n}")),
                reader.take_row(1, format!("combine_op = {op_n}")),
                reader.take_row(2, "composite._reserved".to_string()),
                reader.take_row(4, format!("member_count = {}", c.member_count)),
            ]
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
        match QuantizationDescriptor::decode(q) {
            Ok((qd, _)) => rows.extend(quant_rows(&mut reader, &qd)),
            Err(_) if !q.is_empty() => {
                rows.push(
                    reader.take_row(q.len(), "quantization_descriptor (undecodable)".to_string()),
                );
            }
            Err(_) => {}
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

    // Composite member (flag bit 4) — appears after the extension-type section. Marks this
    // descriptor as a member of a composite (its head carries layout_tag 0x0B).
    if let Some(cm) = &desc.composite_member {
        let role_n = match cm.member_role {
            MemberRole::Base => "base",
            MemberRole::Correction => "correction",
            _ => "unknown",
        };
        rows.push(reader.take_row(1, format!("member_role = {role_n}")));
        rows.push(reader.take_row(15, "composite_member._reserved".to_string()));
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

// ── Data section rows ──────────────────────────────────────────────────────────

/// Append rows displaying the tensor buffer `buf` in the chosen `DataMode`.
///
/// `buf_file_offset` is the byte offset of `buf` within the file — used for
/// the offset column in hex-mode rows.
fn append_data_rows(
    rows: &mut Vec<Row>,
    desc: &TensorDescriptor,
    buf: &[u8],
    buf_file_offset: usize,
    mode: DataMode,
) {
    let et = desc.element_type;
    let shape = desc.shape.dims();
    rows.push(section_row(&format!(
        "TENSOR DATA ({}, shape={shape:?}, {} bytes)",
        et,
        buf.len()
    )));
    match mode {
        DataMode::Hex => {
            for (i, chunk) in buf.chunks(16).enumerate() {
                rows.push(Row {
                    offset: buf_file_offset + i * 16,
                    bytes: chunk.to_vec(),
                    field: String::new(),
                });
            }
        }
        DataMode::Numpy => {
            let s = format_numpy(et, shape, buf);
            for line in s.lines() {
                rows.push(Row {
                    offset: 0,
                    bytes: vec![],
                    field: line.to_string(),
                });
            }
        }
        DataMode::None => {}
    }
}

fn inspect_hrryfile_inner(data: &[u8], rows: &mut Vec<Row>, data_mode: DataMode) -> Result<()> {
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
            Ok(desc) => {
                rows.extend(rows_from_descriptor(
                    desc_data,
                    &desc,
                    entry.descriptor_offset,
                ));
                if data_mode != DataMode::None {
                    let data_start = entry.data_offset as usize;
                    let data_end = data_start.saturating_add(entry.data_length as usize);
                    if data_end <= data.len() && entry.data_length > 0 {
                        append_data_rows(
                            rows,
                            &desc,
                            &data[data_start..data_end],
                            data_start,
                            data_mode,
                        );
                    } else {
                        rows.push(section_row(&format!(
                            "TENSOR DATA {:?}: not available (data outside file bounds)",
                            entry.name
                        )));
                    }
                }
            }
            Err(e) => rows.push(Row {
                offset: entry.descriptor_offset,
                bytes: vec![],
                field: format!("ERROR decoding descriptor: {e}"),
            }),
        }
    }

    Ok(())
}

fn inspect_hrryfile(data: &[u8], data_mode: DataMode) -> (Vec<Row>, Option<Error>) {
    let mut rows = Vec::new();
    match inspect_hrryfile_inner(data, &mut rows, data_mode) {
        Ok(()) => (rows, None),
        Err(e) => (rows, Some(e)),
    }
}

// ── Raw descriptor inspection ─────────────────────────────────────────────────

fn inspect(data: &[u8], data_mode: DataMode) -> (Vec<Row>, Option<Error>) {
    // Dispatch to HRRYFILE container path when the file magic matches.
    if data.len() >= 8 && &data[..8] == FILE_MAGIC {
        return inspect_hrryfile(data, data_mode);
    }

    match TensorDescriptor::decode(data) {
        Ok(desc) => {
            let desc_len = u32::from_le_bytes(data[6..10].try_into().unwrap_or([0u8; 4])) as usize;
            let mut rows = rows_from_descriptor(data, &desc, 0);
            if data_mode != DataMode::None {
                let buf = data.get(desc_len..).unwrap_or(&[]);
                if buf.is_empty() {
                    rows.push(section_row(
                        "TENSOR DATA: not present in this file (descriptor only)",
                    ));
                } else {
                    append_data_rows(&mut rows, &desc, buf, desc_len, data_mode);
                }
            }
            (rows, None)
        }
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

fn parse_args(args: &[String]) -> Result<(String, DataMode)> {
    let mut path: Option<String> = None;
    let mut data_mode = DataMode::None;

    for arg in args.iter().skip(1) {
        if arg == "--data" || arg == "--data=numpy" {
            data_mode = DataMode::Numpy;
        } else if arg == "--data=hex" {
            data_mode = DataMode::Hex;
        } else if (arg.starts_with('-') && arg != "-") || path.is_some() {
            return Err(Error::Usage);
        } else {
            path = Some(arg.clone());
        }
    }

    path.map(|p| (p, data_mode)).ok_or(Error::Usage)
}

fn run() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let (path, data_mode) = parse_args(&args)?;

    let data: Vec<u8> = if path == "-" {
        let mut buf = Vec::new();
        io::stdin().read_to_end(&mut buf)?;
        buf
    } else {
        fs::read(&path)?
    };

    let (rows, err) = inspect(&data, data_mode);
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

#[cfg(test)]
mod tests {
    use super::*;
    use hurray_core::ElementType;

    // Build a minimal raw HRRY descriptor for testing.
    fn make_raw_descriptor(et: ElementType, dims: &[u64]) -> Vec<u8> {
        use hurray_core::{
            BufferHandle, LayoutDescriptor, MemoryClass, Shape, SyncMode, TensorDescriptor,
            DESCRIPTOR_VERSION_MAJOR, DESCRIPTOR_VERSION_MINOR,
        };
        let shape = Shape::new(dims.to_vec()).unwrap();
        let n: u64 = dims.iter().product();
        let byte_size = hurray_core::buffer_size_bytes(et, n);
        let bh = BufferHandle::with_memory_class(
            byte_size,
            64,
            hurray_core::DeviceTag::Cpu,
            SyncMode::ProducerSynced,
            MemoryClass::Standard,
        )
        .unwrap();
        let desc = TensorDescriptor::new(
            DESCRIPTOR_VERSION_MAJOR,
            DESCRIPTOR_VERSION_MINOR,
            et,
            shape,
            0,
            LayoutDescriptor::RowMajor,
            vec![bh],
            None,
            None,
            None,
            None,
        )
        .unwrap();
        desc.encode().unwrap()
    }

    #[test]
    fn format_float_adds_dot() {
        assert_eq!(format_float(1.0), "1.");
        assert_eq!(format_float(1.5), "1.5");
        assert_eq!(format_float(f64::NAN), "nan");
        assert_eq!(format_float(f64::INFINITY), "inf");
        assert_eq!(format_float(f64::NEG_INFINITY), "-inf");
    }

    #[test]
    fn format_scalar_float32() {
        let data = 1.0f32.to_le_bytes();
        assert_eq!(format_scalar(ElementType::Float32, &data, 0), "1.");
    }

    #[test]
    fn format_scalar_int32() {
        let data = 42i32.to_le_bytes();
        assert_eq!(format_scalar(ElementType::Int32, &data, 0), "42");
    }

    #[test]
    fn format_scalar_bool_true() {
        let data = [0b0000_0001u8];
        assert_eq!(format_scalar(ElementType::Bool, &data, 0), "True");
        assert_eq!(format_scalar(ElementType::Bool, &data, 1), "False");
    }

    #[test]
    fn format_scalar_uint4_int4_lsb_first() {
        // 0x83: low nibble = 0x3, high nibble = 0x8 (LSB-first packing).
        let data = [0x83u8];
        assert_eq!(format_scalar(ElementType::Uint4, &data, 0), "3");
        assert_eq!(format_scalar(ElementType::Uint4, &data, 1), "8");
        // int4: 0x8 is the two's-complement value -8.
        assert_eq!(format_scalar(ElementType::Int4, &data, 0), "3");
        assert_eq!(format_scalar(ElementType::Int4, &data, 1), "-8");
    }

    #[test]
    fn format_scalar_uint2_int2_lsb_first() {
        // 0b11_10_01_00: elements (LSB-first) 0,1,2,3.
        let data = [0b1110_0100u8];
        for (idx, want) in [(0, "0"), (1, "1"), (2, "2"), (3, "3")] {
            assert_eq!(format_scalar(ElementType::Uint2, &data, idx), want);
        }
        // int2: 0,1,-2,-1.
        for (idx, want) in [(0, "0"), (1, "1"), (2, "-2"), (3, "-1")] {
            assert_eq!(format_scalar(ElementType::Int2, &data, idx), want);
        }
    }

    #[test]
    fn format_scalar_float8_e4m3() {
        // exp=0b0111 (bias 7 → 2^0), man=0 → 1.0. byte = 0b0_0111_000 = 0x38.
        assert_eq!(format_scalar(ElementType::Float8E4M3, &[0x38], 0), "1.");
        // exp=0b1000 (2^1), man=0 → 2.0. byte = 0x40.
        assert_eq!(format_scalar(ElementType::Float8E4M3, &[0x40], 0), "2.");
        // sign=1, exp=0b0111, man=0 → -1.0. byte = 0xB8.
        assert_eq!(format_scalar(ElementType::Float8E4M3, &[0xB8], 0), "-1.");
        // 0x7F = S0 exp1111 man111 → NaN (E4M3 has no inf; max exp + max mantissa is NaN).
        assert_eq!(format_scalar(ElementType::Float8E4M3, &[0x7F], 0), "nan");
        // 0x00 → +0.
        assert_eq!(format_scalar(ElementType::Float8E4M3, &[0x00], 0), "0.");
    }

    #[test]
    fn format_scalar_float8_e5m2() {
        // exp=0b01111 (bias 15 → 2^0), man=0 → 1.0. byte = 0b0_01111_00 = 0x3C.
        assert_eq!(format_scalar(ElementType::Float8E5M2, &[0x3C], 0), "1.");
        // exp=0b11111, man=0 → inf (E5M2 has inf/nan like IEEE).
        assert_eq!(format_scalar(ElementType::Float8E5M2, &[0x7C], 0), "inf");
        // exp=0b11111, man!=0 → NaN.
        assert_eq!(format_scalar(ElementType::Float8E5M2, &[0x7D], 0), "nan");
    }

    #[test]
    fn format_scalar_float8_e8m0_scale() {
        // E8M0 is an unsigned power-of-two scale: byte 127 → 2^0 = 1.0.
        assert_eq!(format_scalar(ElementType::Float8E8M0, &[127], 0), "1.");
        assert_eq!(format_scalar(ElementType::Float8E8M0, &[128], 0), "2.");
        // 0x00 and 0xFF are the reserved NaN codes.
        assert_eq!(format_scalar(ElementType::Float8E8M0, &[0x00], 0), "nan");
        assert_eq!(format_scalar(ElementType::Float8E8M0, &[0xFF], 0), "nan");
    }

    #[test]
    fn format_scalar_float4_e2m1() {
        // e2m1, bias 1: exp=0b01, man=0 → 2^0 = 1.0. code = 0b010 = 2 (low nibble).
        assert_eq!(format_scalar(ElementType::Float4E2M1, &[0x02], 0), "1.");
        // exp=0b11, man=1 → (1.5)·2^(3-1) = 6.0. code = 0b111 = 7.
        assert_eq!(format_scalar(ElementType::Float4E2M1, &[0x07], 0), "6.");
        // high nibble is the element at idx 1.
        assert_eq!(format_scalar(ElementType::Float4E2M1, &[0x20], 1), "1.");
    }

    #[test]
    fn format_scalar_float6_packing() {
        // e2m3, bias 1: exp=0b01, man=0 → 1.0. 6-bit code = 0b001_000 = 0x08.
        assert_eq!(
            format_scalar(ElementType::Float6E2M3, &[0x08, 0, 0], 0),
            "1."
        );
        // e3m2, bias 3: exp=0b011, man=0 → 1.0. 6-bit code = 0b011_00 = 0x0C.
        assert_eq!(
            format_scalar(ElementType::Float6E3M2, &[0x0C, 0, 0], 0),
            "1."
        );
    }

    #[test]
    fn quant_rows_span_matches_encoded_len() {
        use hurray_core::quantization::PerTensorAffine;
        // The rendered rows must consume exactly the encoded descriptor — no more, no
        // less — or the cursor desyncs for the sections that follow quantization.
        let desc = QuantizationDescriptor::PerTensorAffine(PerTensorAffine::new(0.5, 128).unwrap());
        let bytes = desc.encode_to_vec();
        let mut reader = Reader::with_base(&bytes, bytes.len(), 0);
        let rows = quant_rows(&mut reader, &desc);
        let span: usize = rows.iter().map(|r| r.bytes.len()).sum();
        assert_eq!(span, bytes.len());
        // Header row decodes the scheme tag; a value row surfaces the scale.
        assert!(rows[0].field.contains("per-tensor-affine"));
        assert!(rows.iter().any(|r| r.field.contains("scale = 0.5")));
    }

    #[test]
    fn format_numpy_1d_int32() {
        let data: Vec<u8> = [1i32, 2, 3].iter().flat_map(|v| v.to_le_bytes()).collect();
        let s = format_numpy(ElementType::Int32, &[3], &data);
        assert_eq!(s, "[1 2 3]");
    }

    #[test]
    fn format_numpy_2d_float32() {
        let data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let s = format_numpy(ElementType::Float32, &[2, 2], &data);
        assert!(s.starts_with("[[1. 2.]"), "got: {s}");
        assert!(s.contains("[3. 4.]"), "got: {s}");
    }

    #[test]
    fn format_numpy_truncates_large() {
        let n = 2000u64;
        let data: Vec<u8> = (0..n).flat_map(|i| (i as i32).to_le_bytes()).collect();
        let s = format_numpy(ElementType::Int32, &[n], &data);
        assert!(s.contains("..."), "expected truncation: {s}");
    }

    #[test]
    fn parse_args_no_data() {
        let args = ["prog", "file.hrry"].map(String::from);
        let (path, mode) = parse_args(&args).unwrap();
        assert_eq!(path, "file.hrry");
        assert_eq!(mode, DataMode::None);
    }

    #[test]
    fn parse_args_data_numpy() {
        let args = ["prog", "--data", "file.hrry"].map(String::from);
        let (path, mode) = parse_args(&args).unwrap();
        assert_eq!(path, "file.hrry");
        assert_eq!(mode, DataMode::Numpy);
    }

    #[test]
    fn parse_args_data_hex() {
        let args = ["prog", "--data=hex", "file.hrry"].map(String::from);
        let (_, mode) = parse_args(&args).unwrap();
        assert_eq!(mode, DataMode::Hex);
    }

    #[test]
    fn parse_args_unknown_flag_is_error() {
        let args = ["prog", "--foo", "file.hrry"].map(String::from);
        assert!(parse_args(&args).is_err());
    }

    #[test]
    fn inspect_raw_descriptor_no_data() {
        let desc_bytes = make_raw_descriptor(ElementType::Float32, &[2, 3]);
        let (rows, err) = inspect(&desc_bytes, DataMode::None);
        assert!(err.is_none());
        assert!(rows.iter().any(|r| r.field.contains("float32")));
    }

    #[test]
    fn inspect_raw_descriptor_numpy_mode_no_buffer() {
        let desc_bytes = make_raw_descriptor(ElementType::Float32, &[2, 3]);
        let (rows, err) = inspect(&desc_bytes, DataMode::Numpy);
        assert!(err.is_none());
        // Buffer not present → section note, no panic
        assert!(rows
            .iter()
            .any(|r| r.field.contains("TENSOR DATA") || r.field.contains("not present")));
    }

    #[test]
    fn inspect_raw_descriptor_numpy_mode_with_buffer() {
        let mut file = make_raw_descriptor(ElementType::Float32, &[3]);
        // Append the 3 float32 values [1, 2, 3] after the descriptor.
        for v in [1.0f32, 2.0, 3.0] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        let (rows, err) = inspect(&file, DataMode::Numpy);
        assert!(err.is_none());
        let all = rows
            .iter()
            .map(|r| r.field.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(all.contains("[1. 2. 3.]"), "expected numpy row in: {all}");
    }
}
