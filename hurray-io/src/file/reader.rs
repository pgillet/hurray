use bytes::{Bytes, BytesMut};
use hurray_core::{CompositeValidator, LayoutDescriptor, TensorDescriptor};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncSeek, AsyncSeekExt, SeekFrom};

use crate::file::types::{IndexEntry, KvValue};
use crate::file::{
    file_flags, FILE_HEADER_SIZE, FILE_MAGIC, SUPPORTED_CONTAINER_VERSION_MAJOR, TRAILER_MAGIC,
    TRAILER_SIZE,
};
use crate::{Error, Result};

/// Default maximum composite nesting depth for [`FileReader::read_composite`].
pub const DEFAULT_MAX_COMPOSITE_DEPTH: usize = 64;

/// A tensor read from a Hurray file: descriptor plus zero-copy buffer views.
#[derive(Debug)]
pub struct FileTensor {
    /// The tensor's name as recorded in the file index.
    pub name: String,
    /// The decoded tensor descriptor.
    pub descriptor: TensorDescriptor,
    /// Raw buffer bytes, one [`Bytes`] per buffer handle.
    pub buffers: Vec<Bytes>,
}

/// One member of a composite read from a file: a plain tensor or a nested composite.
#[derive(Debug)]
pub enum FileItem {
    /// A single (non-composite) tensor.
    Tensor(FileTensor),
    /// A nested composite.
    Composite(FileComposite),
}

impl FileItem {
    /// The item's governing descriptor: the tensor's descriptor, or the composite head.
    pub fn descriptor(&self) -> &TensorDescriptor {
        match self {
            FileItem::Tensor(t) => &t.descriptor,
            FileItem::Composite(c) => &c.head,
        }
    }
}

/// A composite tensor read from a file: its head plus its ordered members.
///
/// Membership was recovered from the head's `member_count` and file-offset adjacency, then
/// validated with [`CompositeValidator`] (member count, partition coverage, overlay
/// ordering). Members are ordered as written; each may itself be a composite (nesting).
#[derive(Debug)]
pub struct FileComposite {
    /// The head's name, as recorded in the file index.
    pub name: String,
    /// The composite head descriptor (owns no data buffers).
    pub head: TensorDescriptor,
    /// The members, in write order. Each may itself be a composite.
    pub members: Vec<FileItem>,
}

/// Reads tensors from a seekable Hurray file.
///
/// Open with [`FileReader::open`], then look up tensors by name with
/// [`read_tensor`][FileReader::read_tensor] or inspect metadata with
/// [`tensor_names`][FileReader::tensor_names] and [`kv`][FileReader::kv].
///
/// # Examples
///
/// ```no_run
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use hurray_io::file::FileReader;
///
/// let file = tokio::fs::File::open("model.hrry").await?;
/// let mut reader = FileReader::open(file).await?;
/// println!("tensors: {:?}", reader.tensor_names().collect::<Vec<_>>());
/// let tensor = reader.read_tensor("embeddings").await?;
/// println!("buffer[0]: {} bytes", tensor.buffers[0].len());
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct FileReader<R> {
    inner: R,
    data_buffer_alignment: u64,
    max_composite_depth: usize,
    /// Footer index: one entry per tensor, in file-write order (or sorted if
    /// `SORTED_INDEX` was set by the writer).
    pub index: Vec<IndexEntry>,
    /// File-level KV metadata (empty if the file has no KV section).
    pub kv: Vec<(String, KvValue)>,
}

impl<R: AsyncRead + AsyncSeek + Unpin> FileReader<R> {
    /// Opens a Hurray file: reads the header, trailer, index, and KV section.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidFileMagic`] — first 8 bytes are not `HRRYFILE`
    /// - [`Error::UnsupportedContainerVersion`] — major version > 1
    /// - [`Error::ReservedFileFlagBits`] — header `file_flags` has unknown bits set
    /// - [`Error::InvalidTrailerMagic`] — trailer `trailer_magic` is not `HRRY`
    /// - [`Error::IndexOverrunsTrailer`] — index section overlaps the trailer
    /// - [`Error::IndexCrc32cMismatch`] — CRC-32C verification failed
    /// - [`Error::DuplicateTensorName`] — index contains duplicate names
    /// - [`Error::Io`] / [`Error::UnexpectedEof`] — I/O failures
    pub async fn open(mut inner: R) -> Result<Self> {
        // ── File header ───────────────────────────────────────────────────────
        inner.seek(SeekFrom::Start(0)).await?;
        let mut header = [0u8; 64];
        inner
            .read_exact(&mut header)
            .await
            .map_err(eof_to_unexpected)?;

        if &header[0..8] != FILE_MAGIC {
            return Err(Error::InvalidFileMagic);
        }

        let container_version_major = header[8];
        if container_version_major > SUPPORTED_CONTAINER_VERSION_MAJOR {
            return Err(Error::UnsupportedContainerVersion {
                major: container_version_major,
            });
        }

        let file_flags_val = u32::from_le_bytes(header[12..16].try_into().unwrap());
        if file_flags_val & file_flags::RESERVED_MASK != 0 {
            return Err(Error::ReservedFileFlagBits {
                flags: file_flags_val,
            });
        }

        let data_buffer_alignment = u32::from_le_bytes(header[16..20].try_into().unwrap()) as u64;

        // ── Trailer ───────────────────────────────────────────────────────────
        let file_size = inner.seek(SeekFrom::End(0)).await?;
        if file_size < FILE_HEADER_SIZE + TRAILER_SIZE {
            return Err(Error::InvalidHeader(
                "file too small to contain a valid trailer".into(),
            ));
        }

        inner
            .seek(SeekFrom::Start(file_size - TRAILER_SIZE))
            .await?;
        let mut trailer = [0u8; 40];
        inner
            .read_exact(&mut trailer)
            .await
            .map_err(eof_to_unexpected)?;

        if &trailer[36..40] != TRAILER_MAGIC {
            return Err(Error::InvalidTrailerMagic);
        }

        let index_offset = u64::from_le_bytes(trailer[0..8].try_into().unwrap());
        let index_length = u64::from_le_bytes(trailer[8..16].try_into().unwrap());
        let kv_offset = u64::from_le_bytes(trailer[16..24].try_into().unwrap());
        let kv_length = u32::from_le_bytes(trailer[24..28].try_into().unwrap());
        let stored_crc = u32::from_le_bytes(trailer[28..32].try_into().unwrap());

        if index_offset + index_length > file_size - TRAILER_SIZE {
            return Err(Error::IndexOverrunsTrailer);
        }

        // ── Index section ────────────────────────────────────────────────────
        inner.seek(SeekFrom::Start(index_offset)).await?;
        let mut index_bytes = vec![0u8; index_length as usize];
        inner
            .read_exact(&mut index_bytes)
            .await
            .map_err(eof_to_unexpected)?;

        if file_flags_val & file_flags::HAS_INDEX_CRC32C != 0 {
            let computed = crc32c::crc32c(&index_bytes);
            if computed != stored_crc {
                return Err(Error::IndexCrc32cMismatch {
                    stored: stored_crc,
                    computed,
                });
            }
        }

        let index = parse_index(&index_bytes)?;

        // ── KV section ───────────────────────────────────────────────────────
        // Accept KV if either the flag or the trailer fields indicate its presence.
        // A streaming writer may not set HAS_KV_METADATA in the header (no seek-back),
        // so kv_offset != 0 is treated as authoritative.
        let has_kv = (file_flags_val & file_flags::HAS_KV_METADATA != 0)
            || (kv_offset != 0 && kv_length != 0);
        let kv = if has_kv && kv_offset != 0 && kv_length != 0 {
            inner.seek(SeekFrom::Start(kv_offset)).await?;
            let mut kv_bytes = vec![0u8; kv_length as usize];
            inner
                .read_exact(&mut kv_bytes)
                .await
                .map_err(eof_to_unexpected)?;
            parse_kv(&kv_bytes)?
        } else {
            Vec::new()
        };

        Ok(Self {
            inner,
            data_buffer_alignment,
            max_composite_depth: DEFAULT_MAX_COMPOSITE_DEPTH,
            index,
            kv,
        })
    }

    /// Sets the maximum composite nesting depth accepted by
    /// [`read_composite`][FileReader::read_composite]. Default:
    /// [`DEFAULT_MAX_COMPOSITE_DEPTH`].
    pub fn with_max_composite_depth(mut self, max_composite_depth: usize) -> Self {
        self.max_composite_depth = max_composite_depth;
        self
    }

    /// Returns the names of all tensors in the file, in index order.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.index.iter().map(|e| e.name.as_str())
    }

    /// Returns the file-level KV metadata.
    pub fn kv(&self) -> &[(String, KvValue)] {
        &self.kv
    }

    /// Decodes and returns the descriptor for `name` without reading buffer data.
    ///
    /// Useful for inspecting dtype, shape, or layout before deciding whether to
    /// load the full tensor.
    pub async fn read_descriptor(&mut self, name: &str) -> Result<TensorDescriptor> {
        let entry = self.find_entry(name)?;
        let offset = entry.descriptor_offset;
        let len = entry.descriptor_length as usize;

        self.inner.seek(SeekFrom::Start(offset)).await?;
        let mut buf = vec![0u8; len];
        self.inner
            .read_exact(&mut buf)
            .await
            .map_err(eof_to_unexpected)?;
        Ok(TensorDescriptor::decode(&buf)?)
    }

    /// Reads and returns a complete tensor (descriptor + all buffers).
    pub async fn read_tensor(&mut self, name: &str) -> Result<FileTensor> {
        let entry = self.find_entry(name)?;
        let desc_offset = entry.descriptor_offset;
        let desc_len = entry.descriptor_length as usize;
        let data_offset = entry.data_offset;
        let tensor_name = entry.name.clone();

        // Decode descriptor
        self.inner.seek(SeekFrom::Start(desc_offset)).await?;
        let mut desc_buf = vec![0u8; desc_len];
        self.inner
            .read_exact(&mut desc_buf)
            .await
            .map_err(eof_to_unexpected)?;
        let desc = TensorDescriptor::decode(&desc_buf)?;

        // Read buffers from data region
        self.inner.seek(SeekFrom::Start(data_offset)).await?;
        let mut current = data_offset;
        let n = desc.buffers.len();
        let mut buffers = Vec::with_capacity(n);

        for (i, handle) in desc.buffers.iter().enumerate() {
            let byte_size = handle.byte_size() as usize;
            let mut buf = BytesMut::with_capacity(byte_size);
            buf.resize(byte_size, 0);
            self.inner
                .read_exact(&mut buf)
                .await
                .map_err(eof_to_unexpected)?;
            current += byte_size as u64;
            buffers.push(buf.freeze());

            // Between consecutive buffers, seek past the alignment padding.
            if i < n - 1 {
                let aligned = align_up(current, self.data_buffer_alignment);
                if aligned > current {
                    self.inner.seek(SeekFrom::Start(aligned)).await?;
                    current = aligned;
                }
            }
        }

        Ok(FileTensor {
            name: tensor_name,
            descriptor: desc,
            buffers,
        })
    }

    /// Reads a composite tensor by its head name, reassembling the head with its members.
    ///
    /// Members are recovered by **file-offset adjacency**: the head's declared
    /// `member_count` tensors written immediately after it, recursively for nested
    /// composites. Recovery keys on the descriptor offset, not index position, so it is
    /// correct even when the file used the `SORTED_INDEX` option (which reorders the index
    /// array but not the file layout). The reassembled group is validated with
    /// [`CompositeValidator`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use hurray_io::file::FileReader;
    ///
    /// let file = tokio::fs::File::open("model.hrry").await?;
    /// let mut reader = FileReader::open(file).await?;
    /// let composite = reader.read_composite("weight").await?;
    /// println!("{} member(s)", composite.members.len());
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// - [`Error::TensorNotFound`] — no tensor named `head_name`
    /// - [`Error::NotAComposite`] — `head_name` exists but is not a composite head
    /// - [`Error::TornComposite`] — fewer members follow the head than it declares
    /// - [`Error::CompositeNestingTooDeep`] — nesting exceeded `max_composite_depth`
    /// - [`Error::Core`] — composite validation failed
    pub async fn read_composite(&mut self, head_name: &str) -> Result<FileComposite> {
        // Offset-ordered snapshot of names: membership is recovered by write order (file
        // offset), independent of the index array's order (which SORTED_INDEX permutes).
        let mut ordered: Vec<(u64, String)> = self
            .index
            .iter()
            .map(|e| (e.descriptor_offset, e.name.clone()))
            .collect();
        ordered.sort_unstable_by_key(|(offset, _)| *offset);

        let start = ordered
            .iter()
            .position(|(_, name)| name == head_name)
            .ok_or_else(|| Error::TensorNotFound(head_name.to_string()))?;

        let names: Vec<String> = ordered.into_iter().map(|(_, name)| name).collect();
        let mut cursor = start;
        match self.consume_item(&names, &mut cursor, 0).await? {
            FileItem::Composite(c) => Ok(c),
            FileItem::Tensor(t) => Err(Error::NotAComposite(t.name)),
        }
    }

    /// Consumes one item at `names[*cursor]` (advancing the cursor), recursing into members
    /// when it is a composite head.
    async fn consume_item(
        &mut self,
        names: &[String],
        cursor: &mut usize,
        depth: usize,
    ) -> Result<FileItem> {
        let name = names[*cursor].clone();
        *cursor += 1;

        let desc = self.read_descriptor(&name).await?;
        let member_count = match &desc.layout {
            LayoutDescriptor::Composite(c) => c.member_count,
            _ => {
                let tensor = self.read_tensor(&name).await?;
                return Ok(FileItem::Tensor(tensor));
            }
        };

        if depth >= self.max_composite_depth {
            return Err(Error::CompositeNestingTooDeep {
                limit: self.max_composite_depth,
            });
        }

        let mut validator = CompositeValidator::new(&desc)?;
        let mut members = Vec::with_capacity(member_count as usize);
        for i in 0..member_count {
            if *cursor >= names.len() {
                return Err(Error::TornComposite {
                    declared: member_count,
                    actual: i,
                });
            }
            // Box the recursive call: an async fn cannot name its own future.
            let item = Box::pin(self.consume_item(names, cursor, depth + 1)).await?;
            validator.push_member(item.descriptor())?;
            members.push(item);
        }
        validator.finish()?;

        Ok(FileItem::Composite(FileComposite {
            name,
            head: desc,
            members,
        }))
    }

    /// Returns the underlying reader.
    pub fn into_inner(self) -> R {
        self.inner
    }

    fn find_entry(&self, name: &str) -> Result<&IndexEntry> {
        self.index
            .iter()
            .find(|e| e.name == name)
            .ok_or_else(|| Error::TensorNotFound(name.to_string()))
    }
}

// ── Parsing helpers ───────────────────────────────────────────────────────────

fn align_up(offset: u64, alignment: u64) -> u64 {
    let rem = offset % alignment;
    if rem == 0 {
        offset
    } else {
        offset + (alignment - rem)
    }
}

fn eof_to_unexpected(e: std::io::Error) -> Error {
    if e.kind() == std::io::ErrorKind::UnexpectedEof {
        Error::UnexpectedEof
    } else {
        Error::Io(e)
    }
}

fn read_u16(bytes: &[u8], pos: &mut usize) -> Result<u16> {
    if *pos + 2 > bytes.len() {
        return Err(Error::InvalidHeader("truncated field (u16)".into()));
    }
    let v = u16::from_le_bytes(bytes[*pos..*pos + 2].try_into().unwrap());
    *pos += 2;
    Ok(v)
}

fn read_u32(bytes: &[u8], pos: &mut usize) -> Result<u32> {
    if *pos + 4 > bytes.len() {
        return Err(Error::InvalidHeader("truncated field (u32)".into()));
    }
    let v = u32::from_le_bytes(bytes[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    Ok(v)
}

fn read_u64(bytes: &[u8], pos: &mut usize) -> Result<u64> {
    if *pos + 8 > bytes.len() {
        return Err(Error::InvalidHeader("truncated field (u64)".into()));
    }
    let v = u64::from_le_bytes(bytes[*pos..*pos + 8].try_into().unwrap());
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

fn parse_index(bytes: &[u8]) -> Result<Vec<IndexEntry>> {
    let mut pos = 0usize;
    let count = read_u64(bytes, &mut pos)? as usize;
    let mut entries = Vec::with_capacity(count);
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

fn parse_kv(bytes: &[u8]) -> Result<Vec<(String, KvValue)>> {
    let mut pos = 0usize;
    let count = read_u32(bytes, &mut pos)? as usize;
    let mut entries = Vec::with_capacity(count);
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

        let value_tag = if pos < bytes.len() {
            let t = bytes[pos];
            pos += 1;
            t
        } else {
            return Err(Error::InvalidHeader("KV value tag missing".into()));
        };

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
            let mut elements = Vec::with_capacity(count);
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
