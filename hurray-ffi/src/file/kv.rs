//! The file-level KV metadata section, in both directions.
//!
//! One tagged value type serves reads and writes. That is deliberate: a KV section read
//! from one file can be handed straight to another file's writer, which is the operation
//! a tool that rewrites a container actually performs. A read-only accessor set and a
//! separate write-only struct would have made that a transcription loop.
//!
//! On a read, the pointers inside a value borrow from the reader handle; on a write, they
//! borrow from the caller. Nothing is copied in either direction.

use hurray_io::file::KvValue;

use crate::status::{HurrayStatus, HURRAY_ERR_INVALID_FILE};

// ── Value kinds ───────────────────────────────────────────────────────────────
//
// The wire tags from `docs/spec/metadata.md` § KV Metadata Section, not a second
// numbering: one value, one tag, wherever it is observed.

/// UTF-8 string: `data` and `data_len`. Not NUL-terminated.
pub const HURRAY_KV_STRING: u8 = 0x01;
/// Signed 64-bit integer: `i64_value`.
pub const HURRAY_KV_INT64: u8 = 0x02;
/// Unsigned 64-bit integer: `u64_value`.
pub const HURRAY_KV_UINT64: u8 = 0x03;
/// IEEE 754 binary64: `f64_value`.
pub const HURRAY_KV_FLOAT64: u8 = 0x04;
/// Boolean: `bool_value`.
pub const HURRAY_KV_BOOL: u8 = 0x05;
/// Opaque bytes: `data` and `data_len`.
pub const HURRAY_KV_BYTES: u8 = 0x06;
/// Homogeneous array of scalars: `items` and `item_count`.
pub const HURRAY_KV_ARRAY: u8 = 0x07;

// ── Values ────────────────────────────────────────────────────────────────────

/// One typed KV value: a `kind` tag and the fields that tag selects.
///
/// A flat tagged struct rather than a union, because every binding generator downstream
/// of this header reads a struct correctly and few read a union well. Fields the tag does
/// not select are zero on a read and ignored on a write.
///
/// Arrays nest one level and no further — the format forbids an array of arrays — so an
/// `items` element is always a scalar.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HurrayKvValue {
    /// Which `HURRAY_KV_*` kind this value is.
    pub kind: u8,
    /// Set when `kind` is [`HURRAY_KV_BOOL`].
    pub bool_value: bool,
    /// Set when `kind` is [`HURRAY_KV_INT64`].
    pub i64_value: i64,
    /// Set when `kind` is [`HURRAY_KV_UINT64`].
    pub u64_value: u64,
    /// Set when `kind` is [`HURRAY_KV_FLOAT64`].
    pub f64_value: f64,
    /// First byte of a [`HURRAY_KV_STRING`] or [`HURRAY_KV_BYTES`] payload, else null.
    pub data: *const u8,
    /// Byte length of `data`. A string is UTF-8 and carries no NUL terminator.
    pub data_len: usize,
    /// First element of a [`HURRAY_KV_ARRAY`], else null.
    pub items: *const HurrayKvValue,
    /// Number of elements at `items`.
    pub item_count: usize,
}

impl Default for HurrayKvValue {
    /// A zeroed value with a null `kind`, which is what a C caller gets from `{0}`.
    fn default() -> Self {
        Self {
            kind: 0,
            bool_value: false,
            i64_value: 0,
            u64_value: 0,
            f64_value: 0.0,
            data: std::ptr::null(),
            data_len: 0,
            items: std::ptr::null(),
            item_count: 0,
        }
    }
}

/// One KV pair: a borrowed key and its value.
///
/// The key is UTF-8 and MUST NOT be empty; it carries no NUL terminator, so `key_len` is
/// not optional.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HurrayKvEntry {
    /// First byte of the key.
    pub key: *const u8,
    /// Key length in bytes.
    pub key_len: usize,
    /// The value this key maps to.
    pub value: HurrayKvValue,
}

impl Default for HurrayKvEntry {
    fn default() -> Self {
        Self {
            key: std::ptr::null(),
            key_len: 0,
            value: HurrayKvValue::default(),
        }
    }
}

// ── Rust → C: a reader handle's own snapshot ──────────────────────────────────

/// A reader handle's KV section, owned so that the pointers it hands out stay valid for
/// the life of the handle.
///
/// The core keeps its own copy, which it needs; this one exists because a C caller holds
/// pointers rather than values, and a pointer into the core would be invalidated the
/// moment the next read borrows it mutably.
pub(crate) struct KvTable {
    /// Owned keys and values. Every `data` pointer below points into these.
    _owned: Vec<(String, KvValue)>,
    /// Every array's elements, flattened. Fully populated before any pointer into it is
    /// taken, and never grown afterwards.
    _items: Vec<HurrayKvValue>,
    /// One entry per pair, in file order.
    entries: Vec<HurrayKvEntry>,
}

impl KvTable {
    /// Snapshots a decoded KV section into the C representation.
    pub(crate) fn new(kv: &[(String, KvValue)]) -> Self {
        let owned: Vec<(String, KvValue)> = kv.to_vec();

        // Exact capacity, because `items` must not reallocate once an entry points into
        // it. Arrays do not nest, so one pass over the top level counts every element.
        let total_items: usize = owned
            .iter()
            .map(|(_, v)| match v {
                KvValue::Array(elements) => elements.len(),
                _ => 0,
            })
            .sum();
        let mut items: Vec<HurrayKvValue> = Vec::with_capacity(total_items);

        // Two passes over the arrays: fill `items` first, then build the entries that
        // point into it, so no pointer is taken before the last push.
        let mut spans: Vec<(usize, usize)> = Vec::with_capacity(owned.len());
        for (_, value) in &owned {
            let start = items.len();
            if let KvValue::Array(elements) = value {
                for element in elements {
                    items.push(scalar_to_c(element));
                }
            }
            spans.push((start, items.len() - start));
        }

        let base = items.as_ptr();
        let entries = owned
            .iter()
            .zip(&spans)
            .map(|((key, value), &(start, len))| HurrayKvEntry {
                key: key.as_ptr(),
                key_len: key.len(),
                value: match value {
                    KvValue::Array(_) => HurrayKvValue {
                        kind: HURRAY_KV_ARRAY,
                        // SAFETY: `start + len` is within `items`, which has its final
                        // length here and never grows again.
                        items: unsafe { base.add(start) },
                        item_count: len,
                        ..HurrayKvValue::default()
                    },
                    scalar => scalar_to_c(scalar),
                },
            })
            .collect();

        Self {
            _owned: owned,
            _items: items,
            entries,
        }
    }

    /// The entries, in file order.
    pub(crate) fn entries(&self) -> &[HurrayKvEntry] {
        &self.entries
    }
}

/// Converts one non-array value. An `Array` reaches this only from a malformed section,
/// and is reported as the unknown kind `0` rather than silently flattened.
fn scalar_to_c(value: &KvValue) -> HurrayKvValue {
    match value {
        KvValue::String(s) => HurrayKvValue {
            kind: HURRAY_KV_STRING,
            data: s.as_ptr(),
            data_len: s.len(),
            ..HurrayKvValue::default()
        },
        KvValue::Int64(v) => HurrayKvValue {
            kind: HURRAY_KV_INT64,
            i64_value: *v,
            ..HurrayKvValue::default()
        },
        KvValue::Uint64(v) => HurrayKvValue {
            kind: HURRAY_KV_UINT64,
            u64_value: *v,
            ..HurrayKvValue::default()
        },
        KvValue::Float64(v) => HurrayKvValue {
            kind: HURRAY_KV_FLOAT64,
            f64_value: *v,
            ..HurrayKvValue::default()
        },
        KvValue::Bool(b) => HurrayKvValue {
            kind: HURRAY_KV_BOOL,
            bool_value: *b,
            ..HurrayKvValue::default()
        },
        KvValue::Bytes(b) => HurrayKvValue {
            kind: HURRAY_KV_BYTES,
            data: b.as_ptr(),
            data_len: b.len(),
            ..HurrayKvValue::default()
        },
        // `KvValue` is `#[non_exhaustive]`, and an array cannot be an array's element.
        _ => HurrayKvValue::default(),
    }
}

// ── C → Rust: what a writer is handed ─────────────────────────────────────────

/// Converts `count` caller-supplied entries into the pairs the writer core takes.
///
/// # Safety
///
/// `entries` MUST point to `count` readable [`HurrayKvEntry`] values, each with a key
/// readable for `key_len` bytes and a value whose own pointers are readable for their
/// declared lengths, all unmodified for the duration of the call.
pub(crate) unsafe fn kv_from_c(
    entries: *const HurrayKvEntry,
    count: usize,
) -> Result<Vec<(String, KvValue)>, HurrayStatus> {
    let mut out = Vec::with_capacity(count);
    // SAFETY: caller guarantees `count` readable entries at `entries`.
    for entry in std::slice::from_raw_parts(entries, count) {
        // SAFETY: caller guarantees `key_len` readable bytes at `key`.
        let key = str_from_c(entry.key, entry.key_len)?;
        out.push((key.to_owned(), value_from_c(&entry.value)?));
    }
    Ok(out)
}

/// Converts one caller-supplied value.
///
/// # Safety
///
/// `value`'s pointers MUST be readable for their declared lengths.
unsafe fn value_from_c(value: &HurrayKvValue) -> Result<KvValue, HurrayStatus> {
    match value.kind {
        HURRAY_KV_STRING => Ok(KvValue::String(
            str_from_c(value.data, value.data_len)?.to_owned(),
        )),
        HURRAY_KV_INT64 => Ok(KvValue::Int64(value.i64_value)),
        HURRAY_KV_UINT64 => Ok(KvValue::Uint64(value.u64_value)),
        HURRAY_KV_FLOAT64 => Ok(KvValue::Float64(value.f64_value)),
        HURRAY_KV_BOOL => Ok(KvValue::Bool(value.bool_value)),
        HURRAY_KV_BYTES => Ok(KvValue::Bytes(
            bytes_from_c(value.data, value.data_len)?.to_vec(),
        )),
        HURRAY_KV_ARRAY => {
            if value.items.is_null() && value.item_count > 0 {
                return Err(HURRAY_ERR_INVALID_FILE);
            }
            let mut elements = Vec::with_capacity(value.item_count);
            // SAFETY: caller guarantees `item_count` readable values at `items`.
            for element in std::slice::from_raw_parts(value.items, value.item_count) {
                if element.kind == HURRAY_KV_ARRAY {
                    // An array of arrays is not representable on the wire. Refused here
                    // rather than at encode time so the message names the caller's input.
                    return Err(HURRAY_ERR_INVALID_FILE);
                }
                elements.push(value_from_c(element)?);
            }
            Ok(KvValue::Array(elements))
        }
        _ => Err(HURRAY_ERR_INVALID_FILE),
    }
}

/// Borrows `len` bytes at `ptr`. A zero length needs no pointer.
///
/// # Safety
///
/// `ptr` MUST be readable for `len` bytes, unmodified for the duration of the call.
pub(crate) unsafe fn bytes_from_c(
    ptr: *const u8,
    len: usize,
) -> Result<&'static [u8], HurrayStatus> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(crate::status::HURRAY_ERR_NULL_POINTER);
    }
    // SAFETY: caller guarantees `len` readable bytes at `ptr`. The returned lifetime is
    // not really 'static; every caller drops the borrow before returning to C.
    Ok(std::slice::from_raw_parts(ptr, len))
}

/// Borrows `len` bytes at `ptr` as UTF-8.
///
/// # Safety
///
/// As [`bytes_from_c`].
pub(crate) unsafe fn str_from_c(ptr: *const u8, len: usize) -> Result<&'static str, HurrayStatus> {
    // SAFETY: delegated to the caller's guarantee on `ptr` and `len`.
    let bytes = bytes_from_c(ptr, len)?;
    // Names and keys are UTF-8 on the wire, so a non-UTF-8 one would make a file no
    // reader could parse: refused at the boundary, not written.
    std::str::from_utf8(bytes).map_err(|_| HURRAY_ERR_INVALID_FILE)
}
