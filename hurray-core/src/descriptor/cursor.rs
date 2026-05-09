//! Internal byte-level read/write primitives for descriptor encoding and decoding.
//!
//! [`ByteCursor`] provides bounds-checked sequential reads from a byte slice.
//! [`ByteWriter`] provides infallible sequential writes to a growing `Vec<u8>`.
//!
//! Both types are `pub(crate)` — callers outside this crate interact only with
//! the public `TensorDescriptor::encode` / `TensorDescriptor::decode` API.

use crate::{Error, Result};

// ── ByteCursor ────────────────────────────────────────────────────────────────

/// A bounds-checked sequential reader over a byte slice.
///
/// The cursor has a `limit` that is typically set to `descriptor_length`; reads
/// are rejected once `pos >= limit`, even if bytes remain in the underlying
/// slice beyond that point.  This enforces the spec rule that a reader MUST NOT
/// read beyond `descriptor_length` bytes.
pub(crate) struct ByteCursor<'a> {
    data: &'a [u8],
    pos: usize,
    limit: usize,
}

impl<'a> ByteCursor<'a> {
    /// Creates a new cursor over `data` with the given read limit.
    ///
    /// `limit` is clamped to `data.len()` so callers need not guard against
    /// overflow when setting `limit = descriptor_length as usize`.
    pub(crate) fn new(data: &'a [u8], limit: usize) -> Self {
        let limit = limit.min(data.len());
        Self {
            data,
            pos: 0,
            limit,
        }
    }

    /// Returns the current read position in bytes from the start of the slice.
    #[inline]
    pub(crate) fn pos(&self) -> usize {
        self.pos
    }

    /// Returns the number of bytes still available for reading before the limit.
    #[inline]
    pub(crate) fn remaining(&self) -> usize {
        self.limit.saturating_sub(self.pos)
    }

    /// Reads exactly `n` bytes and advances the position.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DescriptorTruncated`] if fewer than `n` bytes remain.
    pub(crate) fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.limit {
            return Err(Error::DescriptorTruncated {
                offset: self.pos,
                needed: n,
                available: self.remaining(),
            });
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    /// Reads a single `u8`.
    pub(crate) fn read_u8(&mut self) -> Result<u8> {
        let b = self.read_bytes(1)?;
        Ok(b[0])
    }

    /// Reads a little-endian `u32`.
    pub(crate) fn read_u32_le(&mut self) -> Result<u32> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Reads a little-endian `u64`.
    pub(crate) fn read_u64_le(&mut self) -> Result<u64> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// Reads a little-endian `i64`.
    pub(crate) fn read_i64_le(&mut self) -> Result<i64> {
        let b = self.read_bytes(8)?;
        Ok(i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// Reads a little-endian IEEE 754 `f64`.
    pub(crate) fn read_f64_le(&mut self) -> Result<f64> {
        let b = self.read_bytes(8)?;
        Ok(f64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
}

// ── ByteWriter ────────────────────────────────────────────────────────────────

/// An infallible sequential writer that grows an internal `Vec<u8>`.
///
/// All methods are infallible; memory allocation failure will panic (consistent
/// with Rust's standard library allocation model for Vec).
pub(crate) struct ByteWriter {
    buf: Vec<u8>,
}

impl ByteWriter {
    /// Creates a new writer pre-allocated for 128 bytes.
    ///
    /// 128 bytes covers all minimal descriptors (fixed header + small shape +
    /// one buffer handle) without excessive waste on the heap.
    pub(crate) fn new() -> Self {
        Self {
            buf: Vec::with_capacity(128),
        }
    }

    /// Consumes the writer and returns the accumulated bytes.
    pub(crate) fn into_vec(self) -> Vec<u8> {
        self.buf
    }

    /// Returns the number of bytes written so far.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.buf.len()
    }

    /// Appends a raw byte slice.
    #[inline]
    pub(crate) fn write_bytes(&mut self, b: &[u8]) {
        self.buf.extend_from_slice(b);
    }

    /// Appends a single `u8`.
    #[inline]
    pub(crate) fn write_u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    /// Appends a little-endian `u32`.
    #[inline]
    pub(crate) fn write_u32_le(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Appends a little-endian `u64`.
    #[inline]
    pub(crate) fn write_u64_le(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Appends a little-endian `i64`.
    #[inline]
    pub(crate) fn write_i64_le(&mut self, v: i64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Appends a little-endian IEEE 754 `f64`.
    #[inline]
    pub(crate) fn write_f64_le(&mut self, v: f64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Appends `n` zero bytes.
    #[inline]
    pub(crate) fn write_zeros(&mut self, n: usize) {
        self.buf.resize(self.buf.len() + n, 0u8);
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_read_u8_ok() {
        let data = [0x42u8];
        let mut c = ByteCursor::new(&data, data.len());
        assert_eq!(c.read_u8().unwrap(), 0x42);
        assert_eq!(c.remaining(), 0);
    }

    #[test]
    fn cursor_read_beyond_limit_returns_truncated() {
        let data = [0x01u8, 0x02, 0x03, 0x04, 0x05];
        // limit = 2, so only 2 bytes are accessible
        let mut c = ByteCursor::new(&data, 2);
        c.read_u8().unwrap();
        c.read_u8().unwrap();
        let err = c.read_u8().unwrap_err();
        assert!(matches!(err, Error::DescriptorTruncated { .. }));
    }

    #[test]
    fn cursor_read_u64_ok() {
        let v = 0x0102030405060708u64;
        let data = v.to_le_bytes();
        let mut c = ByteCursor::new(&data, data.len());
        assert_eq!(c.read_u64_le().unwrap(), v);
    }

    #[test]
    fn cursor_limit_clamps_to_slice_len() {
        let data = [1u8, 2, 3];
        let c = ByteCursor::new(&data, 100);
        assert_eq!(c.remaining(), 3);
    }

    #[test]
    fn writer_round_trip_u32() {
        let mut w = ByteWriter::new();
        w.write_u32_le(0xDEAD_BEEF);
        let v = w.into_vec();
        assert_eq!(v, [0xEF, 0xBE, 0xAD, 0xDE]);
    }

    #[test]
    fn writer_write_zeros() {
        let mut w = ByteWriter::new();
        w.write_zeros(4);
        assert_eq!(w.into_vec(), [0, 0, 0, 0]);
    }
}
