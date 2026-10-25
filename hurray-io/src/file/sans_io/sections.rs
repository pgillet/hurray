//! The file container's fixed sections — the footer index and the KV block — encoded and
//! parsed as pure functions over bytes.
//!
//! Both directions live together because they are two readings of one wire layout: a tag
//! added on the write side without its parse arm is the bug this adjacency makes obvious.

use crate::file::types::{IndexEntry, KvValue};
use crate::{Error, Result};

// ── offsets ───────────────────────────────────────────────────────────────────

/// The next multiple of `alignment` at or after `offset`.
///
/// `alignment` is validated when the header is parsed (power of two, non-zero), so the
/// remainder cannot divide by zero here.
pub(crate) fn align_up(offset: u64, alignment: u64) -> u64 {
    let rem = offset % alignment;
    if rem == 0 {
        offset
    } else {
        offset + (alignment - rem)
    }
}

/// How many padding bytes take `offset` to the next `alignment` boundary.
pub(crate) fn pad_to(offset: u64, alignment: u64) -> u64 {
    align_up(offset, alignment) - offset
}

// ── parsing ───────────────────────────────────────────────────────────────────

fn read_u16(bytes: &[u8], pos: &mut usize) -> Result<u16> {
    if *pos + 2 > bytes.len() {
        return Err(Error::InvalidHeader("truncated field (u16)".into()));
    }
    let v = u16::from_le_bytes(bytes[*pos..*pos + 2].try_into().unwrap_or([0, 0]));
    *pos += 2;
    Ok(v)
}

fn read_u32(bytes: &[u8], pos: &mut usize) -> Result<u32> {
    if *pos + 4 > bytes.len() {
        return Err(Error::InvalidHeader("truncated field (u32)".into()));
    }
    let v = u32::from_le_bytes(bytes[*pos..*pos + 4].try_into().unwrap_or([0; 4]));
    *pos += 4;
    Ok(v)
}

fn read_u64(bytes: &[u8], pos: &mut usize) -> Result<u64> {
    if *pos + 8 > bytes.len() {
        return Err(Error::InvalidHeader("truncated field (u64)".into()));
    }
    let v = u64::from_le_bytes(bytes[*pos..*pos + 8].try_into().unwrap_or([0; 8]));
    *pos += 8;
    Ok(v)
}

fn read_bytes_slice<'a>(bytes: &'a [u8], pos: &mut usize, len: usize) -> Result<&'a [u8]> {
    if *pos + len > bytes.len() {
        return Err(Error::InvalidHeader("truncated byte slice".into()));
    }
    let s = &bytes[*pos..*pos + len];
    *pos += len;
    Ok(s)
}

/// Parses the footer index section: one entry per tensor, names unique.
pub(crate) fn parse_index(bytes: &[u8]) -> Result<Vec<IndexEntry>> {
    let mut pos = 0usize;
    let count = read_u64(bytes, &mut pos)? as usize;
    // Entry size is at least 32 bytes plus a 1-byte name, so a `count` larger than the
    // section can hold is a lie: reserving for it would be an attacker-chosen allocation.
    let mut entries = Vec::with_capacity(count.min(bytes.len() / 32));
    let mut seen = std::collections::HashSet::new();

    for _ in 0..count {
        let name_len = read_u16(bytes, &mut pos)? as usize;
        if name_len == 0 {
            return Err(Error::TensorNameEmpty);
        }
        let name_bytes = read_bytes_slice(bytes, &mut pos, name_len)?;
        let name = std::str::from_utf8(name_bytes)
            .map_err(|_| Error::InvalidHeader("tensor name is not valid UTF-8".into()))?
            .to_string();
        if !seen.insert(name.clone()) {
            return Err(Error::DuplicateTensorName(name));
        }

        let descriptor_offset = read_u64(bytes, &mut pos)?;
        let descriptor_length = read_u32(bytes, &mut pos)?;
        let data_offset = read_u64(bytes, &mut pos)?;
        let data_length = read_u64(bytes, &mut pos)?;
        let _flags = read_u32(bytes, &mut pos)?; // reserved

        entries.push(IndexEntry {
            name,
            descriptor_offset,
            descriptor_length,
            data_offset,
            data_length,
        });
    }

    Ok(entries)
}

/// Parses the file-level KV metadata section.
pub(crate) fn parse_kv(bytes: &[u8]) -> Result<Vec<(String, KvValue)>> {
    let mut pos = 0usize;
    let count = read_u32(bytes, &mut pos)? as usize;
    let mut entries = Vec::with_capacity(count.min(bytes.len() / 4));
    let mut seen = std::collections::HashSet::new();

    for _ in 0..count {
        let key_len = read_u16(bytes, &mut pos)? as usize;
        if key_len == 0 {
            return Err(Error::KvKeyEmpty);
        }
        let key_bytes = read_bytes_slice(bytes, &mut pos, key_len)?;
        let key = std::str::from_utf8(key_bytes)
            .map_err(|_| Error::InvalidHeader("KV key is not valid UTF-8".into()))?
            .to_string();
        if !seen.insert(key.clone()) {
            return Err(Error::DuplicateKvKey(key));
        }

        let value_tag = read_bytes_slice(bytes, &mut pos, 1)?[0];
        let value = parse_kv_value(value_tag, bytes, &mut pos)?;
        entries.push((key, value));
    }

    Ok(entries)
}

fn parse_kv_value(tag: u8, bytes: &[u8], pos: &mut usize) -> Result<KvValue> {
    match tag {
        0x01 => {
            let len = read_u32(bytes, pos)? as usize;
            let s = read_bytes_slice(bytes, pos, len)?;
            let s = std::str::from_utf8(s)
                .map_err(|_| Error::InvalidHeader("KV string is not valid UTF-8".into()))?
                .to_string();
            Ok(KvValue::String(s))
        }
        0x02 => Ok(KvValue::Int64(read_u64(bytes, pos)? as i64)),
        0x03 => Ok(KvValue::Uint64(read_u64(bytes, pos)?)),
        0x04 => {
            let raw = read_u64(bytes, pos)?;
            Ok(KvValue::Float64(f64::from_le_bytes(raw.to_le_bytes())))
        }
        0x05 => {
            let b = read_bytes_slice(bytes, pos, 1)?[0];
            match b {
                0x00 => Ok(KvValue::Bool(false)),
                0x01 => Ok(KvValue::Bool(true)),
                _ => Err(Error::InvalidHeader(format!(
                    "invalid KV bool byte 0x{b:02X}"
                ))),
            }
        }
        0x06 => {
            let len = read_u32(bytes, pos)? as usize;
            let v = read_bytes_slice(bytes, pos, len)?.to_vec();
            Ok(KvValue::Bytes(v))
        }
        0x07 => {
            let elem_tag = read_bytes_slice(bytes, pos, 1)?[0];
            if elem_tag == 0x07 || elem_tag > 0x06 {
                return Err(Error::InvalidHeader(format!(
                    "invalid KV array element tag 0x{elem_tag:02X}"
                )));
            }
            let count = read_u32(bytes, pos)? as usize;
            let mut elements = Vec::with_capacity(count.min(bytes.len()));
            for _ in 0..count {
                elements.push(parse_kv_value(elem_tag, bytes, pos)?);
            }
            Ok(KvValue::Array(elements))
        }
        _ => Err(Error::InvalidHeader(format!(
            "unknown KV value tag 0x{tag:02X}"
        ))),
    }
}

// ── encoding ──────────────────────────────────────────────────────────────────

pub(crate) fn validate_kv_keys(kv: &[(String, KvValue)]) -> Result<()> {
    let mut seen = std::collections::HashSet::new();
    for (key, _) in kv {
        if key.is_empty() {
            return Err(Error::KvKeyEmpty);
        }
        if key.len() > 65535 {
            return Err(Error::KvKeyTooLong { len: key.len() });
        }
        if !seen.insert(key.as_str()) {
            return Err(Error::DuplicateKvKey(key.clone()));
        }
    }
    Ok(())
}

pub(crate) fn encode_kv_section(kv: &[(String, KvValue)]) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&(kv.len() as u32).to_le_bytes());
    for (key, value) in kv {
        buf.extend_from_slice(&(key.len() as u16).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        encode_kv_value(&mut buf, value)?;
    }
    Ok(buf)
}

fn kv_wire_tag(value: &KvValue) -> u8 {
    match value {
        KvValue::String(_) => 0x01,
        KvValue::Int64(_) => 0x02,
        KvValue::Uint64(_) => 0x03,
        KvValue::Float64(_) => 0x04,
        KvValue::Bool(_) => 0x05,
        KvValue::Bytes(_) => 0x06,
        KvValue::Array(_) => 0x07,
    }
}

fn encode_kv_value(buf: &mut Vec<u8>, value: &KvValue) -> Result<()> {
    buf.push(kv_wire_tag(value));
    encode_kv_payload(buf, value)
}

fn encode_kv_payload(buf: &mut Vec<u8>, value: &KvValue) -> Result<()> {
    match value {
        KvValue::String(s) => {
            buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        }
        KvValue::Int64(v) => buf.extend_from_slice(&v.to_le_bytes()),
        KvValue::Uint64(v) => buf.extend_from_slice(&v.to_le_bytes()),
        KvValue::Float64(v) => buf.extend_from_slice(&v.to_le_bytes()),
        KvValue::Bool(v) => buf.push(u8::from(*v)),
        KvValue::Bytes(v) => {
            buf.extend_from_slice(&(v.len() as u32).to_le_bytes());
            buf.extend_from_slice(v);
        }
        KvValue::Array(elements) => {
            if elements.is_empty() {
                return Err(Error::InvalidHeader("KV array must not be empty".into()));
            }
            let elem_tag = kv_wire_tag(&elements[0]);
            if elem_tag == 0x07 {
                return Err(Error::InvalidHeader(
                    "KV array elements must not be arrays".into(),
                ));
            }
            for elem in elements {
                if kv_wire_tag(elem) != elem_tag {
                    return Err(Error::InvalidHeader(
                        "KV array elements must all have the same type".into(),
                    ));
                }
            }
            buf.push(elem_tag);
            buf.extend_from_slice(&(elements.len() as u32).to_le_bytes());
            for elem in elements {
                encode_kv_payload(buf, elem)?;
            }
        }
    }
    Ok(())
}

pub(crate) fn encode_index_section(entries: &[IndexEntry]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    for e in entries {
        buf.extend_from_slice(&(e.name.len() as u16).to_le_bytes());
        buf.extend_from_slice(e.name.as_bytes());
        buf.extend_from_slice(&e.descriptor_offset.to_le_bytes());
        buf.extend_from_slice(&e.descriptor_length.to_le_bytes());
        buf.extend_from_slice(&e.data_offset.to_le_bytes());
        buf.extend_from_slice(&e.data_length.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // flags, reserved
    }
    buf
}
