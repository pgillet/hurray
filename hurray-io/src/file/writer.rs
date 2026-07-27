use std::collections::HashSet;

use hurray_core::{CompositeValidator, TensorDescriptor};
use tokio::io::{AsyncWrite, AsyncWriteExt};

use crate::file::types::{FileWriterOptions, KvValue};
use crate::file::{file_flags, FILE_HEADER_SIZE, FILE_MAGIC, TRAILER_MAGIC};
use crate::{Error, Result};

/// A named node to write as part of a composite via [`FileWriter::write_composite`]:
/// either a plain tensor or a nested composite. Every node carries a name because every
/// tensor — head and members alike — gets its own footer-index entry (ADR-027 § Binding).
pub enum FileCompositeNode<'a> {
    /// A single (non-composite) tensor member.
    Tensor {
        /// Unique tensor name for this member's index entry.
        name: &'a str,
        /// The member's tensor descriptor.
        descriptor: &'a TensorDescriptor,
        /// One byte-slice per buffer handle in `descriptor.buffers`.
        buffers: &'a [&'a [u8]],
    },
    /// A nested composite member: its head plus its own members.
    Composite {
        /// Unique tensor name for the nested head's index entry.
        name: &'a str,
        /// The nested composite's head descriptor.
        head: &'a TensorDescriptor,
        /// The nested composite's members, in order.
        members: &'a [FileCompositeNode<'a>],
    },
}

impl FileCompositeNode<'_> {
    /// The node's governing descriptor: the tensor's descriptor, or the nested head.
    fn descriptor(&self) -> &TensorDescriptor {
        match self {
            FileCompositeNode::Tensor { descriptor, .. } => descriptor,
            FileCompositeNode::Composite { head, .. } => head,
        }
    }
}

/// Recursively validates a composite node tree, writing nothing. Reuses
/// [`CompositeValidator`] at every level so [`FileWriter::write_composite`] can reject a
/// torn/invalid composite before emitting any tensor.
fn validate_composite_node(
    head: &TensorDescriptor,
    members: &[FileCompositeNode<'_>],
) -> Result<()> {
    let mut validator = CompositeValidator::new(head)?;
    for member in members {
        validator.push_member(member.descriptor())?;
    }
    validator.finish()?;

    for member in members {
        if let FileCompositeNode::Composite {
            head: nested_head,
            members: nested_members,
            ..
        } = member
        {
            validate_composite_node(nested_head, nested_members)?;
        }
    }
    Ok(())
}

struct InternalEntry {
    name: String,
    descriptor_offset: u64,
    descriptor_length: u32,
    data_offset: u64,
    data_length: u64,
}

/// Writes a Hurray file in a single forward pass without seeks.
///
/// The file format is:
/// ```text
/// [ File header      ]  64 bytes
/// [ Tensor region    ]  descriptor → pad → buffers → pad  (repeated)
/// [ KV section       ]  optional, written by finish()
/// [ Index section    ]  written by finish()
/// [ Trailer          ]  40 bytes
/// ```
///
/// # Note on `HAS_KV_METADATA`
///
/// This writer sets `HAS_KV_METADATA = 0` in the file header because KV
/// content is not known until [`finish`][FileWriter::finish] is called and a
/// streaming writer cannot seek back to patch the header. The trailer's
/// `kv_offset` field is the canonical indicator of KV presence; [`FileReader`]
/// uses that field rather than the header flag.
///
/// [`FileReader`]: crate::file::FileReader
///
/// # Examples
///
/// ```no_run
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use hurray_core::{
///     BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape,
///     SyncMode, TensorDescriptor, MIN_BUFFER_ALIGNMENT,
/// };
/// use hurray_io::file::{FileWriter, KvValue};
///
/// let handle = BufferHandle::new(64, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced)?;
/// let shape = Shape::new(vec![4u64, 4]).unwrap();
/// let desc = TensorDescriptor::new(
///     1, 0, ElementType::Float32, shape, 0,
///     LayoutDescriptor::RowMajor, vec![handle], None, None, None, None,
/// )?;
/// let data = vec![0u8; 64];
///
/// let file = tokio::fs::File::create("model.hrry").await?;
/// let mut writer = FileWriter::new(file).await?;
/// writer.write_tensor("embeddings", &desc, &[&data]).await?;
/// writer.finish(vec![
///     ("model".to_string(), KvValue::String("llama-3".to_string())),
/// ]).await?;
/// # Ok(())
/// # }
/// ```
pub struct FileWriter<W> {
    inner: W,
    alignment: u64,
    sorted_index: bool,
    current_offset: u64,
    entries: Vec<InternalEntry>,
    seen_names: HashSet<String>,
}

static ZEROS: [u8; 4096] = [0u8; 4096];

async fn write_zeroes<W: AsyncWrite + Unpin>(w: &mut W, mut count: u64) -> Result<()> {
    while count > 0 {
        let n = count.min(ZEROS.len() as u64) as usize;
        w.write_all(&ZEROS[..n]).await?;
        count -= n as u64;
    }
    Ok(())
}

fn pad_to(offset: u64, alignment: u64) -> u64 {
    let rem = offset % alignment;
    if rem == 0 {
        0
    } else {
        alignment - rem
    }
}

impl<W: AsyncWrite + Unpin> FileWriter<W> {
    /// Creates a writer with default options (4096-byte buffer alignment, unsorted index).
    pub async fn new(inner: W) -> Result<Self> {
        Self::with_options(inner, FileWriterOptions::default()).await
    }

    /// Creates a writer with custom options.
    pub async fn with_options(mut inner: W, options: FileWriterOptions) -> Result<Self> {
        options.validate().map_err(Error::InvalidHeader)?;

        let align = options.data_buffer_alignment as u64;
        let sorted_index = options.sorted_index;

        // File flags known at write time. HAS_KV_METADATA is not set here
        // because KV content is supplied to finish(); see type-level note.
        let mut flags: u32 = file_flags::HAS_INDEX_CRC32C;
        if sorted_index {
            flags |= file_flags::SORTED_INDEX;
        }

        // 64-byte file header
        let mut header = [0u8; 64];
        header[0..8].copy_from_slice(FILE_MAGIC);
        header[8] = 1; // container_version_major
        header[9] = 0; // container_version_minor
                       // [10..12] _reserved = 0
        header[12..16].copy_from_slice(&flags.to_le_bytes());
        header[16..20].copy_from_slice(&(align as u32).to_le_bytes());
        // first_descriptor_offset = 64 (immediately after the header)
        header[20..28].copy_from_slice(&FILE_HEADER_SIZE.to_le_bytes());
        // tensor_count_hint = UINT64_MAX (unknown at write time)
        header[28..36].copy_from_slice(&u64::MAX.to_le_bytes());
        // [36..64] _reserved_header = 0

        inner.write_all(&header).await?;

        Ok(Self {
            inner,
            alignment: align,
            sorted_index,
            current_offset: FILE_HEADER_SIZE,
            entries: Vec::new(),
            seen_names: HashSet::new(),
        })
    }

    /// Encodes and writes one tensor.
    ///
    /// # Errors
    ///
    /// - [`Error::TensorNameEmpty`] / [`Error::TensorNameTooLong`] — invalid name
    /// - [`Error::DuplicateTensorName`] — name already written to this file
    /// - [`Error::MultiBufferLengthMismatch`] — wrong buffer count
    /// - [`Error::BufferSizeMismatch`] — buffer length ≠ handle `byte_size`
    /// - [`Error::Core`] — descriptor encoding failed
    /// - [`Error::Io`] — underlying write error
    pub async fn write_tensor(
        &mut self,
        name: &str,
        desc: &TensorDescriptor,
        buffers: &[&[u8]],
    ) -> Result<()> {
        if name.is_empty() {
            return Err(Error::TensorNameEmpty);
        }
        if name.len() > 65535 {
            return Err(Error::TensorNameTooLong { len: name.len() });
        }
        if !self.seen_names.insert(name.to_string()) {
            return Err(Error::DuplicateTensorName(name.to_string()));
        }

        if buffers.len() != desc.buffers.len() {
            return Err(Error::MultiBufferLengthMismatch {
                declared: desc.buffers.len(),
                actual: buffers.len(),
            });
        }
        for (i, (handle, buf)) in desc.buffers.iter().zip(buffers).enumerate() {
            let declared = handle.byte_size();
            let actual = buf.len() as u64;
            if declared != actual {
                return Err(Error::BufferSizeMismatch {
                    index: i,
                    declared,
                    actual,
                });
            }
        }

        let descriptor_offset = self.current_offset;
        let encoded = desc.encode()?;
        self.inner.write_all(&encoded).await?;
        self.current_offset += encoded.len() as u64;
        let descriptor_length = encoded.len() as u32;

        // Pad descriptor end → first data-buffer alignment boundary
        let pad = pad_to(self.current_offset, self.alignment);
        write_zeroes(&mut self.inner, pad).await?;
        self.current_offset += pad;

        let data_offset = self.current_offset;
        let n = buffers.len();

        for (i, buf) in buffers.iter().enumerate() {
            self.inner.write_all(buf).await?;
            self.current_offset += buf.len() as u64;

            // Pad between consecutive buffers (not after the last one)
            if i < n - 1 {
                let pad = pad_to(self.current_offset, self.alignment);
                write_zeroes(&mut self.inner, pad).await?;
                self.current_offset += pad;
            }
        }

        // data_length covers bytes from data_offset through the last byte of the
        // last buffer, excluding any trailing alignment padding for the next descriptor.
        let data_length = self.current_offset - data_offset;

        // Pad to 8-byte boundary so the next descriptor starts correctly.
        let pad = pad_to(self.current_offset, 8);
        write_zeroes(&mut self.inner, pad).await?;
        self.current_offset += pad;

        self.entries.push(InternalEntry {
            name: name.to_string(),
            descriptor_offset,
            descriptor_length,
            data_offset,
            data_length,
        });

        Ok(())
    }

    /// Writes a composite tensor: its head, then every member's descriptor and data,
    /// contiguously and in order (ADR-027 § Binding).
    ///
    /// Every tensor — the head and each member — gets its own footer-index entry, so all
    /// are individually addressable by name via [`read_tensor`][crate::file::FileReader::read_tensor].
    /// Membership is recoverable by [`read_composite`][crate::file::FileReader::read_composite]
    /// from the head's `member_count` plus file-offset adjacency (the members are the tensors
    /// written immediately after the head). Nested composites are written recursively.
    ///
    /// The whole group is validated up front — reusing [`CompositeValidator`] for member
    /// count and per-rule constraints (partition exact-cover, overlay ordering) — before any
    /// tensor is written.
    ///
    /// # Errors
    ///
    /// - [`Error::Core`] — the head is not a valid composite head, or validation failed
    /// - the name/buffer errors of [`write_tensor`][FileWriter::write_tensor] for the head
    ///   or any member
    /// - [`Error::Io`] — underlying write error
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use hurray_core::{
    ///     layout::{CompositeLayout, CompositionRule, LayoutDescriptor},
    ///     ElementType, Shape, TensorDescriptor,
    /// };
    /// use hurray_io::file::{FileCompositeNode, FileWriter};
    ///
    /// let head = TensorDescriptor::new(
    ///     1, 0, ElementType::Float32, Shape::new(vec![8u64, 8]).unwrap(), 0,
    ///     LayoutDescriptor::Composite(CompositeLayout::new(CompositionRule::Partition, 2).unwrap()),
    ///     vec![], None, None, None, None,
    /// )?;
    /// # let members: Vec<FileCompositeNode> = vec![];
    /// let file = tokio::fs::File::create("model.hrry").await?;
    /// let mut writer = FileWriter::new(file).await?;
    /// writer.write_composite("weight", &head, &members).await?;
    /// writer.finish(vec![]).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn write_composite(
        &mut self,
        head_name: &str,
        head: &TensorDescriptor,
        members: &[FileCompositeNode<'_>],
    ) -> Result<()> {
        // Validate the whole tree before writing any tensor.
        validate_composite_node(head, members)?;

        // Head first (no data buffers), then each member in order.
        self.write_tensor(head_name, head, &[]).await?;
        self.write_composite_members(members).await
    }

    /// Writes each member (recursively for nested composites) after the head.
    async fn write_composite_members(&mut self, members: &[FileCompositeNode<'_>]) -> Result<()> {
        for member in members {
            match member {
                FileCompositeNode::Tensor {
                    name,
                    descriptor,
                    buffers,
                } => {
                    self.write_tensor(name, descriptor, buffers).await?;
                }
                FileCompositeNode::Composite {
                    name,
                    head,
                    members,
                } => {
                    self.write_tensor(name, head, &[]).await?;
                    // Box the recursive call: an async fn cannot name its own future.
                    Box::pin(self.write_composite_members(members)).await?;
                }
            }
        }
        Ok(())
    }

    /// Writes the KV section, footer index, and trailer, then flushes.
    ///
    /// `kv` is a list of `(key, value)` pairs. Keys must be non-empty, at most
    /// 65 535 bytes, and unique within the list.
    pub async fn finish(mut self, kv: Vec<(String, KvValue)>) -> Result<W> {
        validate_kv_keys(&kv)?;

        // The tensor region is already 8-aligned after the last write_tensor().
        // No additional padding needed before the KV section.

        let (kv_offset, kv_length) = if kv.is_empty() {
            (0u64, 0u32)
        } else {
            let kv_bytes = encode_kv_section(&kv)?;
            let kv_offset = self.current_offset;
            self.inner.write_all(&kv_bytes).await?;
            self.current_offset += kv_bytes.len() as u64;

            // Pad to 8-byte boundary for the index section
            let pad = pad_to(self.current_offset, 8);
            write_zeroes(&mut self.inner, pad).await?;
            self.current_offset += pad;

            (kv_offset, kv_bytes.len() as u32)
        };

        if self.sorted_index {
            self.entries
                .sort_unstable_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));
        }

        let index_bytes = encode_index_section(&self.entries);
        let index_offset = self.current_offset;
        let index_length = index_bytes.len() as u64;
        let index_crc32c = crc32c::crc32c(&index_bytes);

        self.inner.write_all(&index_bytes).await?;
        self.current_offset += index_length;

        // 40-byte trailer
        let mut trailer = [0u8; 40];
        trailer[0..8].copy_from_slice(&index_offset.to_le_bytes());
        trailer[8..16].copy_from_slice(&index_length.to_le_bytes());
        trailer[16..24].copy_from_slice(&kv_offset.to_le_bytes());
        trailer[24..28].copy_from_slice(&kv_length.to_le_bytes());
        trailer[28..32].copy_from_slice(&index_crc32c.to_le_bytes());
        // [32..36] _reserved = 0
        trailer[36..40].copy_from_slice(TRAILER_MAGIC);

        self.inner.write_all(&trailer).await?;
        self.inner.flush().await?;

        Ok(self.inner)
    }
}

// ── Encoding helpers ──────────────────────────────────────────────────────────

fn validate_kv_keys(kv: &[(String, KvValue)]) -> Result<()> {
    let mut seen = HashSet::new();
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

fn encode_kv_section(kv: &[(String, KvValue)]) -> Result<Vec<u8>> {
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

fn encode_index_section(entries: &[InternalEntry]) -> Vec<u8> {
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
