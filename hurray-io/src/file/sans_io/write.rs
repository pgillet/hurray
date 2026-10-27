//! The sequential write half of the sans-io file core.
//!
//! The file container is written in one forward pass and never seeks — the footer index is
//! built as tensors go out and emitted last — so this side needs no range vocabulary. What
//! it does need is ADR-041 § 3's rule: payloads stay borrowed and are emitted one slice at
//! a time, never assembled, never copied.

use std::collections::HashSet;

use hurray_core::{CompositeValidator, TensorDescriptor};

use super::sections::{
    align_up, encode_index_section, encode_kv_section, pad_to, validate_kv_keys,
};
use crate::file::types::{FileCompositeNode, FileWriterOptions, IndexEntry, KvValue};
use crate::file::{file_flags, FILE_HEADER_SIZE, FILE_MAGIC, TRAILER_MAGIC};
use crate::stream::{Unusable, WriteStep};
use crate::{Error, Result};

/// Alignment padding is emitted from here rather than materialised: a `data_buffer_alignment`
/// of 2 MiB would otherwise mean a 2 MiB buffer of zeroes per tensor.
static ZEROS: [u8; 4096] = [0u8; 4096];

/// One stretch of bytes to emit, in file order.
#[derive(Debug, Clone, Copy)]
enum Seg {
    /// Bytes the *operation* owns: the file header, a descriptor, or the footer. Held by
    /// the op rather than the core so that abandoning it commits nothing (§ 6b).
    Owned,
    /// `len` padding bytes.
    Zeros { len: u64 },
    /// Caller payload `index`, borrowed and never copied (§ 3).
    Payload { index: usize },
}

/// Writes the Hurray file container into bytes a driver sends onward.
///
/// Each tensor, and the footer, is submitted as one operation; the core emits the
/// descriptor, the alignment padding and each payload in turn, and says when the item is
/// written. The footer index is accumulated as it goes, so nothing has to be re-read and
/// no offset is ever patched.
///
/// # Examples
///
/// ```
/// use std::io::Write;
///
/// use hurray_core::{
///     BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode,
///     TensorDescriptor, MIN_BUFFER_ALIGNMENT,
/// };
/// use hurray_io::file::FileWriterCore;
/// use hurray_io::stream::WriteStep;
///
/// let handle =
///     BufferHandle::new(64, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced)?;
/// let descriptor = TensorDescriptor::new(
///     1, 0, ElementType::Float32, Shape::new(vec![4u64, 4])?, 0,
///     LayoutDescriptor::RowMajor, vec![handle], None, None, None, None,
/// )?;
/// let payload = [7u8; 64];
/// // Bound, not a temporary: the operation borrows the payloads until the item is
/// // written, which is ADR-041 § 3's lifetime rule showing up in the type system.
/// let buffers: [&[u8]; 1] = [&payload];
///
/// /// Drives one operation to completion over any `io::Write`.
/// fn drive(mut op: hurray_io::file::FileWriteOp<'_>, sink: &mut Vec<u8>) -> hurray_io::Result<()> {
///     while let WriteStep::Output { .. } = op.step()? {
///         let src = op.output_src();
///         sink.write_all(src).map_err(hurray_io::Error::Io)?;
///         let n = src.len();
///         op.report_written(n)?;
///     }
///     Ok(())
/// }
///
/// let mut file = Vec::new();
/// let mut core = FileWriterCore::new();
/// drive(core.header()?, &mut file)?;
/// drive(core.tensor("embeddings", &descriptor, &buffers)?, &mut file)?;
/// drive(core.finish(vec![])?, &mut file)?;
///
/// assert_eq!(&file[0..8], b"HRRYFILE");
/// assert_eq!(&file[file.len() - 4..], b"HRRY");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct FileWriterCore {
    alignment: u64,
    sorted_index: bool,
    /// Absolute offset of the first byte not yet reported written.
    offset: u64,
    entries: Vec<IndexEntry>,
    seen_names: HashSet<String>,
    unusable: Option<Unusable>,
    /// Set when the item in flight has had a byte accepted, cleared when it completes
    /// (§ 6b). On the core, not the op, because a driver that errors and drops the op
    /// reports nothing — the poison cannot depend on being told.
    torn: bool,
    header_written: bool,
    finished: bool,
}

impl Default for FileWriterCore {
    fn default() -> Self {
        Self::new()
    }
}

impl FileWriterCore {
    /// Creates a core with default options: 4096-byte data alignment, unsorted index.
    ///
    /// # Examples
    ///
    /// See the type-level example.
    pub fn new() -> Self {
        // Default options are valid by construction, so this cannot fail — which is why
        // `new` returns `Self` and only `with_options` returns a `Result`.
        Self::build(FileWriterOptions::default())
    }

    /// Creates a core with custom options.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidHeader`] — `data_buffer_alignment` is not a power of two in
    /// `[4096, 2097152]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_io::file::{FileWriterCore, FileWriterOptions};
    ///
    /// let core = FileWriterCore::with_options(FileWriterOptions {
    ///     data_buffer_alignment: 8192,
    ///     sorted_index: true,
    /// })?;
    /// assert!(!core.is_unusable());
    /// # Ok::<(), hurray_io::Error>(())
    /// ```
    pub fn with_options(options: FileWriterOptions) -> Result<Self> {
        options.validate().map_err(Error::InvalidHeader)?;
        Ok(Self::build(options))
    }

    fn build(options: FileWriterOptions) -> Self {
        Self {
            alignment: u64::from(options.data_buffer_alignment),
            sorted_index: options.sorted_index,
            offset: 0,
            entries: Vec::new(),
            seen_names: HashSet::new(),
            unusable: None,
            torn: false,
            header_written: false,
            finished: false,
        }
    }

    /// Builds the 64-byte header on demand.
    ///
    /// Derived rather than stored so that a header operation owns its own bytes and an
    /// abandoned one leaves the core untouched (§ 6b). Called once per file.
    fn encode_header(&self) -> Vec<u8> {
        // File flags known at write time. HAS_KV_METADATA is not set here because the KV
        // content is supplied to `finish`, and this writer never seeks back to amend the
        // header; readers treat a non-zero trailer `kv_offset` as authoritative.
        let mut flags: u32 = file_flags::HAS_INDEX_CRC32C;
        if self.sorted_index {
            flags |= file_flags::SORTED_INDEX;
        }

        // `FILE_HEADER_SIZE`, not a literal 64: the header writes that same constant into
        // its own `first_descriptor_offset` field below.
        let mut header = vec![0u8; FILE_HEADER_SIZE as usize];
        header[0..8].copy_from_slice(FILE_MAGIC);
        header[8] = 1; // container_version_major
        header[9] = 0; // container_version_minor
                       // [10..12] _reserved = 0
        header[12..16].copy_from_slice(&flags.to_le_bytes());
        header[16..20].copy_from_slice(&(self.alignment as u32).to_le_bytes());
        // first_descriptor_offset = 64 (immediately after the header)
        header[20..28].copy_from_slice(&FILE_HEADER_SIZE.to_le_bytes());
        // tensor_count_hint = UINT64_MAX (unknown at write time)
        header[28..36].copy_from_slice(&u64::MAX.to_le_bytes());
        // [36..64] _reserved_header = 0
        header
    }

    /// Emits the 64-byte file header.
    ///
    /// Drive this before the first tensor: a core with no header written refuses
    /// [`tensor`][Self::tensor] and [`finish`][Self::finish].
    ///
    /// # Errors
    ///
    /// - [`Error::CoreUnusable`] — the core is poisoned
    /// - [`Error::CoreFailed`] — the header is already written, or the file is finished
    ///
    /// # Examples
    ///
    /// See the type-level example.
    pub fn header(&mut self) -> Result<FileWriteOp<'_>> {
        if let Some(reason) = self.retired() {
            return Err(Error::CoreUnusable(reason));
        }
        // Checked in the same order as a second `finish`, so a header after the trailer
        // reports `CoreFailed` for the same reason.
        if self.finished || self.header_written {
            return Err(Error::CoreFailed);
        }
        let bytes = self.encode_header();
        Ok(FileWriteOp {
            plan: vec![Seg::Owned],
            core: self,
            bytes,
            payloads: &[],
            cursor: 0,
            written: 0,
            emitted_bytes: 0,
            commit: Some(Commit::Header),
            failed: false,
        })
    }

    /// Submits one tensor: its descriptor, its alignment padding, and its payloads.
    ///
    /// The returned operation borrows `buffers` until it reports
    /// [`WriteStep::ItemWritten`]; the core holds no pointer into them afterwards (§ 3).
    /// The tensor's index entry is recorded at that same point.
    ///
    /// # Errors
    ///
    /// - [`Error::TensorNameEmpty`] / [`Error::TensorNameTooLong`] — invalid name
    /// - [`Error::DuplicateTensorName`] — the name is already in this file
    /// - [`Error::MultiBufferLengthMismatch`] — `buffers.len()` is not the descriptor's
    ///   buffer count
    /// - [`Error::BufferSizeMismatch`] — a slice length differs from its handle's
    ///   `byte_size`
    /// - [`Error::Core`] — the descriptor does not encode
    /// - [`Error::CoreFailed`] — the footer has already been written
    /// - [`Error::CoreUnusable`] — the sink was poisoned mid-item
    ///
    /// # Examples
    ///
    /// See the type-level example.
    pub fn tensor<'a>(
        &'a mut self,
        name: &str,
        descriptor: &TensorDescriptor,
        buffers: &'a [&'a [u8]],
    ) -> Result<FileWriteOp<'a>> {
        if let Some(reason) = self.retired() {
            return Err(Error::CoreUnusable(reason));
        }
        // The trailer is the last thing in the file, so there is no legal tensor after it.
        if self.finished {
            return Err(Error::CoreFailed);
        }
        if !self.header_written {
            return Err(Error::HeaderNotWritten);
        }
        if name.is_empty() {
            return Err(Error::TensorNameEmpty);
        }
        if name.len() > 65535 {
            return Err(Error::TensorNameTooLong { len: name.len() });
        }
        // `contains`, not `insert`: the name is consumed when the item completes, so an
        // operation that never emits a byte leaves it free (§ 6b).
        if self.seen_names.contains(name) {
            return Err(Error::DuplicateTensorName(name.to_string()));
        }
        if buffers.len() != descriptor.buffers.len() {
            return Err(Error::MultiBufferLengthMismatch {
                declared: descriptor.buffers.len(),
                actual: buffers.len(),
            });
        }
        for (i, (handle, payload)) in descriptor.buffers.iter().zip(buffers).enumerate() {
            let declared = handle.byte_size();
            let actual = payload.len() as u64;
            if declared != actual {
                return Err(Error::BufferSizeMismatch {
                    index: i,
                    declared,
                    actual,
                });
            }
        }
        let encoded = descriptor.encode()?;
        // Exact, with nothing staged on the core: every byte an earlier item planned has
        // either been reported written or belongs to an op that poisoned the core (§ 6b).
        let descriptor_offset = self.offset;
        let descriptor_length = encoded.len() as u32;

        let mut plan = vec![Seg::Owned];

        // Padding from the descriptor's end to the first data buffer. Keyed on there
        // *being* a first data buffer: `buffer_count = 0` is reachable only for a
        // composite head (ADR-027 D-1), which owns no buffer to align, and
        // `file-format.md § Composite Tensors` forbids padding after one. An empty tensor
        // still declares one buffer of `byte_size = 0`, so it keeps its padding — which
        // is why the condition is the buffer table, not the payload size.
        let descriptor_end = descriptor_offset + u64::from(descriptor_length);
        let data_offset = if buffers.is_empty() {
            descriptor_end
        } else {
            align_up(descriptor_end, self.alignment)
        };
        if data_offset > descriptor_end {
            plan.push(Seg::Zeros {
                len: data_offset - descriptor_end,
            });
        }

        let mut cursor = data_offset;
        let last = buffers.len().saturating_sub(1);
        for (index, payload) in buffers.iter().enumerate() {
            plan.push(Seg::Payload { index });
            cursor += payload.len() as u64;
            if index < last {
                let padding = pad_to(cursor, self.alignment);
                if padding > 0 {
                    plan.push(Seg::Zeros { len: padding });
                    cursor += padding;
                }
            }
        }
        // `data_length` covers data_offset through the last byte of the last buffer,
        // excluding the trailing padding that aligns the next descriptor.
        let data_length = cursor - data_offset;

        // Pad to 8 bytes so the next descriptor starts on its own boundary.
        let padding = pad_to(cursor, 8);
        if padding > 0 {
            plan.push(Seg::Zeros { len: padding });
        }

        Ok(FileWriteOp {
            plan,
            core: self,
            bytes: encoded,
            payloads: buffers,
            cursor: 0,
            written: 0,
            emitted_bytes: 0,
            // The name travels inside the entry, which already carries it.
            commit: Some(Commit::Tensor(IndexEntry {
                name: name.to_string(),
                descriptor_offset,
                descriptor_length,
                data_offset,
                data_length,
            })),
            failed: false,
        })
    }

    /// Submits the footer: the KV section, the index, and the 40-byte trailer.
    ///
    /// # Errors
    ///
    /// - [`Error::KvKeyEmpty`] / [`Error::KvKeyTooLong`] / [`Error::DuplicateKvKey`] —
    ///   invalid KV keys
    /// - [`Error::InvalidHeader`] — a KV array is empty, nested, or not homogeneous
    /// - [`Error::CoreFailed`] — the footer has already been written
    /// - [`Error::CoreUnusable`] — the sink was poisoned mid-item
    ///
    /// # Examples
    ///
    /// See the type-level example.
    pub fn finish(&mut self, kv: Vec<(String, KvValue)>) -> Result<FileWriteOp<'_>> {
        if let Some(reason) = self.retired() {
            return Err(Error::CoreUnusable(reason));
        }
        if self.finished {
            return Err(Error::CoreFailed);
        }
        if !self.header_written {
            return Err(Error::HeaderNotWritten);
        }
        validate_kv_keys(&kv)?;

        // The tensor region is already 8-aligned, so the KV section starts where the last
        // tensor left off. Footer padding is at most 7 bytes, so it is staged inline
        // rather than streamed from `ZEROS`.
        let footer_offset = self.offset;
        let mut footer = Vec::new();

        let (kv_offset, kv_length) = if kv.is_empty() {
            (0u64, 0u32)
        } else {
            let kv_bytes = encode_kv_section(&kv)?;
            let kv_length = kv_bytes.len() as u32;
            footer.extend_from_slice(&kv_bytes);
            // Absolute, not relative: the 8-byte boundary the index needs is a boundary in
            // the file, and `footer_offset` is only incidentally aligned.
            let padding = pad_to(footer_offset + footer.len() as u64, 8) as usize;
            footer.resize(footer.len() + padding, 0);
            (footer_offset, kv_length)
        };

        // Sorted into a local copy, not in place: `finish` may be retried after a footer
        // that emitted nothing, and § 6b forbids an operation mutating state the core
        // carries past it. (The sort is idempotent, so this is about the rule, not a bug.)
        let index_bytes = if self.sorted_index {
            let mut sorted = self.entries.clone();
            sorted.sort_unstable_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));
            encode_index_section(&sorted)
        } else {
            encode_index_section(&self.entries)
        };
        let index_offset = footer_offset + footer.len() as u64;
        let index_length = index_bytes.len() as u64;
        let index_crc32c = crc32c::crc32c(&index_bytes);
        footer.extend_from_slice(&index_bytes);

        let mut trailer = [0u8; 40];
        trailer[0..8].copy_from_slice(&index_offset.to_le_bytes());
        trailer[8..16].copy_from_slice(&index_length.to_le_bytes());
        trailer[16..24].copy_from_slice(&kv_offset.to_le_bytes());
        trailer[24..28].copy_from_slice(&kv_length.to_le_bytes());
        trailer[28..32].copy_from_slice(&index_crc32c.to_le_bytes());
        // [32..36] _reserved = 0
        trailer[36..40].copy_from_slice(TRAILER_MAGIC);
        footer.extend_from_slice(&trailer);

        Ok(FileWriteOp {
            plan: vec![Seg::Owned],
            core: self,
            bytes: footer,
            payloads: &[],
            cursor: 0,
            written: 0,
            emitted_bytes: 0,
            commit: Some(Commit::Footer),
            failed: false,
        })
    }

    /// Whether a write put an unknown number of bytes on the sink, leaving the file torn.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_io::file::FileWriterCore;
    ///
    /// assert!(!FileWriterCore::new().is_unusable());
    /// ```
    pub fn is_unusable(&self) -> bool {
        self.retired().is_some()
    }

    /// The reason this core is retired, if it is.
    ///
    /// `unusable` wins over `torn`: a lost count is strictly worse than a known-incomplete
    /// item. Read through here at every site so the precedence cannot drift.
    fn retired(&self) -> Option<Unusable> {
        self.unusable
            .or(self.torn.then_some(Unusable::IncompleteOutput))
    }

    /// Retires the core because the output can no longer be completed.
    ///
    /// For a unit larger than one item — a composite group, whose head is already on the
    /// sink when a member fails. A `FileWriteOp` cannot express that: the failure may come
    /// from *submission*, with no operation in hand.
    pub(crate) fn poison(&mut self, reason: Unusable) {
        self.unusable = Some(reason);
    }
}

/// One item being written: a tensor, the header, or the footer.
///
/// Borrowing both the core and the caller's payloads for its lifetime is what enforces
/// ADR-041 § 3's ownership rule in the type system — the core cannot hold a payload
/// pointer after the operation ends, because the operation is what holds them.
///
/// # Examples
///
/// See [`FileWriterCore`].
pub struct FileWriteOp<'a> {
    core: &'a mut FileWriterCore,
    /// The stretches to emit, in file order.
    plan: Vec<Seg>,
    /// This operation's own bytes: the header, a descriptor, or the footer.
    bytes: Vec<u8>,
    /// Payload slices of the tensor being emitted, in buffer order.
    payloads: &'a [&'a [u8]],
    /// Which entry of `plan` is being emitted.
    cursor: usize,
    /// Bytes of the current stretch already written.
    written: usize,
    /// Bytes of *this item* already accepted by the sink — per item, not per stretch. The
    /// § 6b predicate.
    emitted_bytes: u64,
    /// What this operation changes on the core once every byte is out. `take`n at
    /// completion so a second `step` after `ItemWritten` stays idempotent.
    commit: Option<Commit>,
    failed: bool,
}

/// What an operation owes the core once its last byte is out (§ 6b).
#[derive(Debug)]
enum Commit {
    Header,
    Tensor(IndexEntry),
    Footer,
}

impl FileWriteOp<'_> {
    /// Advances the operation and returns what it needs next. Never blocks.
    ///
    /// # Errors
    ///
    /// - [`Error::CoreFailed`] — a previous call ended this operation
    /// - [`Error::CoreUnusable`] — the core was poisoned mid-item
    ///
    /// # Examples
    ///
    /// See [`FileWriterCore`].
    pub fn step(&mut self) -> Result<WriteStep> {
        if let Some(reason) = self.core.unusable {
            return Err(Error::CoreUnusable(reason));
        }
        if self.failed {
            return Err(Error::CoreFailed);
        }
        loop {
            let Some(seg) = self.plan.get(self.cursor).copied() else {
                // Every byte is out: the borrowed payloads are released and the operation
                // pays what it owes. This is the single commit point of § 6b — the name and
                // the index entry move together, which is what #248 was about.
                self.core.torn = false;
                match self.commit.take() {
                    Some(Commit::Header) => self.core.header_written = true,
                    Some(Commit::Tensor(entry)) => {
                        self.core.seen_names.insert(entry.name.clone());
                        self.core.entries.push(entry);
                    }
                    Some(Commit::Footer) => self.core.finished = true,
                    None => {}
                }
                return Ok(WriteStep::ItemWritten);
            };

            let remaining = self.remaining_in(seg);
            if remaining == 0 {
                self.cursor += 1;
                self.written = 0;
                continue;
            }
            return Ok(WriteStep::Output { len: remaining });
        }
    }

    /// The bytes to write now: one stretch of one segment, never a concatenation (§ 3).
    ///
    /// For a payload this is a slice of the caller's own memory; for padding it is a
    /// window onto a shared page of zeroes, so the same bytes may be offered repeatedly
    /// until the padding is out.
    ///
    /// # Examples
    ///
    /// See [`FileWriterCore`].
    pub fn output_src(&self) -> &[u8] {
        match self.plan.get(self.cursor).copied() {
            Some(Seg::Owned) => &self.bytes[self.written.min(self.bytes.len())..],
            Some(Seg::Zeros { len }) => {
                let left = len - self.written as u64;
                let take = left.min(ZEROS.len() as u64) as usize;
                &ZEROS[..take]
            }
            Some(Seg::Payload { index }) => match self.payloads.get(index) {
                Some(payload) => &payload[self.written.min(payload.len())..],
                None => &[],
            },
            None => &[],
        }
    }

    /// Reports that the sink accepted `n` bytes of [`output_src`][Self::output_src].
    ///
    /// A short write re-offers the remainder.
    ///
    /// # Errors
    ///
    /// - [`Error::ZeroLengthWrite`] — `n == 0`, which § 4 forbids treating as progress. A
    ///   sink that cannot accept bytes is reported with
    ///   [`report_sink_closed`][Self::report_sink_closed] instead.
    /// - [`Error::InvalidProgressReport`] — more bytes reported than were offered. An
    ///   unknown number of bytes is then on the sink and cannot be un-written, so this
    ///   poisons the core (§ 6).
    ///
    /// # Examples
    ///
    /// See [`FileWriterCore`].
    pub fn report_written(&mut self, n: usize) -> Result<()> {
        if let Some(reason) = self.core.unusable {
            return Err(Error::CoreUnusable(reason));
        }
        if self.failed {
            return Err(Error::CoreFailed);
        }
        if n == 0 {
            // § 4: a zero-length write is not success. The position is still known, so the
            // operation stays usable — the driver may retry or declare the sink closed.
            return Err(Error::ZeroLengthWrite);
        }
        let Some(seg) = self.plan.get(self.cursor).copied() else {
            return Err(Error::InvalidProgressReport {
                reported: n,
                offered: 0,
            });
        };
        let offered = self.remaining_in(seg);
        if n > offered {
            self.core.unusable = Some(Unusable::LostWriteCount);
            self.failed = true;
            return Err(Error::InvalidProgressReport {
                reported: n,
                offered,
            });
        }

        if self.emitted_bytes == 0 {
            // Poison at the first accepted byte, not in `Drop`: `Drop` can be skipped by
            // `forget`, delayed by a live op, or run while unwinding. The transient state
            // is unobservable anyway — the op holds `&mut core` for its whole life.
            //
            // Strictly this is the first *reported* byte. ADR-041 § 4 is what closes the
            // gap: the transfer and its count report form one uninterruptible step, so a
            // byte the sink took is always a byte reported.
            self.core.torn = true;
        }
        self.emitted_bytes += n as u64;
        self.core.offset += n as u64;
        // One buffer per segment now, so `written` tracks every kind of stretch.
        self.written += n;
        Ok(())
    }

    /// Bytes of this item already accepted by the sink, padding included.
    ///
    /// Zero means nothing of the item has left the driver, which is exactly when the item
    /// may be resubmitted after a failure (§ 6b).
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_io::file::FileWriterCore;
    ///
    /// let mut core = FileWriterCore::new();
    /// let mut op = core.header()?;
    /// // Stepping offers bytes; only a report counts them.
    /// op.step()?;
    /// assert_eq!(op.emitted(), 0);
    ///
    /// op.report_written(16)?;
    /// assert_eq!(op.emitted(), 16);
    /// # Ok::<(), hurray_io::Error>(())
    /// ```
    pub fn emitted(&self) -> u64 {
        self.emitted_bytes
    }

    /// Declares the sink closed while this item is in flight.
    ///
    /// Returns the error to propagate. A partially written file has a torn item in it and
    /// no footer, so this poisons the core: a fresh operation on the same sink is not a
    /// recovery (§ 6).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn main() -> hurray_io::Result<()> {
    /// # let mut core = hurray_io::file::FileWriterCore::new();
    /// let mut op = core.header()?;
    /// let error = op.report_sink_closed();
    /// assert!(matches!(error, hurray_io::Error::SinkClosed));
    /// # Ok(())
    /// # }
    /// ```
    pub fn report_sink_closed(&mut self) -> Error {
        // Not `LostWriteCount`: the count is known exactly. What is lost is the ability to
        // finish the item, which is a different claim (§ 6b).
        self.core.unusable = Some(Unusable::IncompleteOutput);
        self.failed = true;
        Error::SinkClosed
    }

    fn remaining_in(&self, seg: Seg) -> usize {
        match seg {
            Seg::Owned => self.bytes.len().saturating_sub(self.written),
            Seg::Zeros { len } => {
                let left = len - self.written as u64;
                left.min(ZEROS.len() as u64) as usize
            }
            Seg::Payload { index } => match self.payloads.get(index) {
                Some(payload) => payload.len().saturating_sub(self.written),
                None => 0,
            },
        }
    }
}

/// One tensor of a flattened composite, ready to hand to
/// [`FileWriterCore::tensor`][FileWriterCore::tensor].
///
/// # Examples
///
/// See [`flatten_composite`].
pub struct CompositeWrite<'a> {
    /// The tensor's name, which becomes its footer-index entry.
    pub name: &'a str,
    /// Its descriptor. A composite head's declares no buffers.
    pub descriptor: &'a TensorDescriptor,
    /// Its payloads, one slice per buffer handle. Empty for a head.
    pub buffers: &'a [&'a [u8]],
}

/// Validates a composite tree and flattens it into the order its tensors go on disk.
///
/// Every tensor — the head and each member, nested or not — gets its own footer-index
/// entry (ADR-027 § Binding), and membership is recovered on read from the head's
/// `member_count` plus file-offset adjacency. So the order here *is* the membership: head
/// first, then each member in turn, depth first.
///
/// The whole tree is validated before anything is returned — member counts, partition
/// exact-cover, overlay ordering, at every level — so a torn or invalid composite never
/// reaches a sink. The traversal is an explicit stack rather than recursion (§ 10).
///
/// # Errors
///
/// [`Error::Core`] — a head is not a valid composite head, or a level failed validation.
///
/// # Examples
///
/// ```
/// use hurray_core::{
///     layout::{CompositeLayout, CompositionRule, LayoutDescriptor},
///     BufferHandle, DeviceTag, ElementType, Shape, SyncMode, TensorDescriptor,
///     MIN_BUFFER_ALIGNMENT,
/// };
/// use hurray_io::file::{flatten_composite, FileCompositeNode};
///
/// let head = TensorDescriptor::new(
///     1, 0, ElementType::Float32, Shape::new(vec![2u64, 2])?, 0,
///     LayoutDescriptor::Composite(CompositeLayout::new(CompositionRule::Group, 1)?),
///     vec![], None, None, None, None,
/// )?;
/// let handle =
///     BufferHandle::new(16, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced)?;
/// let member = TensorDescriptor::new(
///     1, 0, ElementType::Float32, Shape::new(vec![2u64, 2])?, 0,
///     LayoutDescriptor::RowMajor, vec![handle], None, None, None, None,
/// )?;
/// let data = [0u8; 16];
/// let buffers: [&[u8]; 1] = [&data];
/// let members = [FileCompositeNode::Tensor {
///     name: "w.0",
///     descriptor: &member,
///     buffers: &buffers,
/// }];
///
/// let order = flatten_composite("w", &head, &members)?;
/// assert_eq!(order.len(), 2);
/// assert_eq!(order[0].name, "w");
/// assert!(order[0].buffers.is_empty());
/// assert_eq!(order[1].name, "w.0");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn flatten_composite<'a>(
    head_name: &'a str,
    head: &'a TensorDescriptor,
    members: &'a [FileCompositeNode<'a>],
) -> Result<Vec<CompositeWrite<'a>>> {
    const NO_BUFFERS: &[&[u8]] = &[];

    let mut order = vec![CompositeWrite {
        name: head_name,
        descriptor: head,
        buffers: NO_BUFFERS,
    }];

    // Each frame is one open head's remaining members, so the deepest frame is the one
    // being consumed — the call stack `write_composite_members` used, made visible.
    let mut stack = vec![(head, members, 0usize)];

    while let Some((open_head, nodes, next)) = stack.pop() {
        if next == 0 {
            // Validate a level once, on the way in.
            let mut validator = CompositeValidator::new(open_head)?;
            for node in nodes {
                validator.push_member(node.descriptor())?;
            }
            validator.finish()?;
        }
        let Some(node) = nodes.get(next) else {
            continue;
        };
        stack.push((open_head, nodes, next + 1));

        match node {
            FileCompositeNode::Tensor {
                name,
                descriptor,
                buffers,
            } => order.push(CompositeWrite {
                name,
                descriptor,
                buffers,
            }),
            FileCompositeNode::Composite {
                name,
                head: nested_head,
                members: nested_members,
            } => {
                order.push(CompositeWrite {
                    name,
                    descriptor: nested_head,
                    buffers: NO_BUFFERS,
                });
                stack.push((nested_head, nested_members, 0));
            }
        }
    }

    Ok(order)
}
