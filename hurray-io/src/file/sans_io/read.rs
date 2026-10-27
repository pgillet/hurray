//! The random-access read half of the sans-io file core.
//!
//! Where the stream core asks for *the next* bytes, this one asks for bytes
//! `[offset, offset + len)` (ADR-041 § 7). The file is addressed, not consumed: the
//! trailer sits at `size - 40`, the index is located from it, and tensors are found by
//! name. That difference in request vocabulary is the whole reason this is a second
//! machine rather than a mode of the first.

use hurray_core::{CompositeValidator, LayoutDescriptor, TensorDescriptor};

use super::sections::{align_up, parse_index, parse_kv};
use crate::file::types::{
    FileComposite, FileItem, FileTensor, IndexEntry, KvValue, DEFAULT_MAX_COMPOSITE_DEPTH,
};
use crate::file::{
    file_flags, FILE_HEADER_SIZE, FILE_MAGIC, MAX_DATA_BUFFER_ALIGNMENT,
    SUPPORTED_CONTAINER_VERSION_MAJOR, TRAILER_MAGIC, TRAILER_SIZE,
};
use crate::stream::{Destination, DestinationAlloc, DestinationRequest, SystemDestinations};
use crate::{Error, Result};

/// What the core needs next from its driver.
///
/// Exactly one request is outstanding at a time, and the driver answers it — with
/// [`report_filled`][FileReaderCore::report_filled] or one of the `take_*` methods —
/// before the core produces another (ADR-041 § 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileStep {
    /// Read the bytes at `[offset, offset + len)` into
    /// [`input_dst`][FileReaderCore::input_dst], then report how many arrived.
    ///
    /// Filling fewer than `len` is normal — the remainder is requested again — but `0`
    /// means the file is shorter than its own index claims, which is a truncated file.
    Range {
        /// Absolute offset from the start of the file.
        offset: u64,
        /// Bytes still wanted at `offset`. Never `0`.
        len: usize,
    },
    /// The file is open and no job is in flight: call `begin_descriptor`, `begin_tensor`
    /// or `begin_composite`.
    Ready,
    /// A descriptor is ready. Collect it with
    /// [`take_descriptor`][FileReaderCore::take_descriptor].
    Descriptor,
    /// A tensor is ready. Collect it with [`take_tensor`][FileReaderCore::take_tensor].
    Tensor,
    /// A composite is ready. Collect it with
    /// [`take_composite`][FileReaderCore::take_composite].
    Composite,
}

/// What the core is working towards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Job {
    /// A descriptor on its own, no buffers read.
    Descriptor,
    /// One complete tensor.
    Tensor,
    /// A composite: a walk over the tensors that follow its head.
    Walk,
}

/// Where the machine is.
#[derive(Debug, Clone, Copy)]
enum State {
    /// Filling the 64-byte file header.
    Header,
    /// Filling the 40-byte trailer.
    Trailer,
    /// Filling the footer index section.
    Index,
    /// Filling the KV metadata section.
    Kv,
    /// Open, nothing in flight.
    Ready,
    /// Filling a tensor descriptor.
    Descriptor,
    /// Filling buffer `index` of the tensor under construction.
    Buffer { index: usize },
    /// A result is waiting to be taken.
    Done(Done),
    /// The open sequence failed: the file is not a Hurray file, or its footer is
    /// malformed. Unlike a failed job, this is terminal — there is nothing to retry.
    Failed,
}

#[derive(Debug, Clone, Copy)]
enum Done {
    Descriptor,
    Tensor,
    Composite,
}

/// One composite head whose members are still arriving (ADR-041 § 10's explicit stack).
struct Open {
    name: String,
    head: TensorDescriptor,
    member_count: u32,
    members: Vec<FileItem>,
    validator: CompositeValidator,
}

/// A composite traversal in progress.
///
/// Membership is recovered by **file-offset adjacency**: the head's declared
/// `member_count` tensors written immediately after it, recursively. The cursor walks
/// names ordered by descriptor offset, which is write order — not index order, which
/// `SORTED_INDEX` permutes.
struct Walk {
    names: Vec<String>,
    cursor: usize,
    stack: Vec<Open>,
}

/// The tensor currently being assembled.
struct Pending<D> {
    name: String,
    descriptor: TensorDescriptor,
    /// Buffers already filled and frozen, in declaration order.
    buffers: Vec<bytes::Bytes>,
    /// The destination for the buffer being filled.
    dest: Option<D>,
    /// Absolute offset the next buffer starts at.
    next_offset: u64,
}

/// Reads a Hurray file from bytes a driver fetches by range.
///
/// The core owns the memory every payload byte lands in, so the transport's write into it
/// is the only copy (ADR-041 § 2), and that memory carries the alignment the buffer handle
/// *declares* (§ 5).
///
/// **Every request is absolute, which is what makes this core unloseable.** The read
/// position advances only when a count is reported, so a driver that abandons a transfer —
/// because its future was cancelled, or because a count came back it could not trust —
/// simply calls [`step`][Self::step] again and is handed the same range. Re-filling a
/// partly filled destination from the start of the range is idempotent. The stream core
/// has an [`Unusable`][crate::stream::Unusable] state for a lost byte count; this one needs
/// none, and for the same reason a job that fails leaves the core usable: the next
/// `begin_*` starts from the index, not from wherever the file pointer happened to stop.
///
/// # Examples
///
/// Reading a file with no runtime anywhere — the driver is three `std::io` calls:
///
/// ```
/// use std::io::{Read, Seek, SeekFrom};
///
/// use hurray_io::file::{FileReaderCore, FileStep};
/// # use hurray_core::{
/// #     BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode,
/// #     TensorDescriptor, MIN_BUFFER_ALIGNMENT,
/// # };
/// # use hurray_io::file::{FileWriteOp, FileWriterCore};
/// # use hurray_io::stream::WriteStep;
/// #
/// # // A one-tensor file, built with the write core.
/// # fn drive(mut op: FileWriteOp<'_>, sink: &mut Vec<u8>) -> hurray_io::Result<()> {
/// #     while let WriteStep::Output { .. } = op.step()? {
/// #         let src = op.output_src();
/// #         sink.extend_from_slice(src);
/// #         let n = src.len();
/// #         op.report_written(n)?;
/// #     }
/// #     Ok(())
/// # }
/// # let handle =
/// #     BufferHandle::new(256, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced)?;
/// # let descriptor = TensorDescriptor::new(
/// #     1, 0, ElementType::Float32, Shape::new(vec![8u64, 8])?, 0,
/// #     LayoutDescriptor::RowMajor, vec![handle], None, None, None, None,
/// # )?;
/// # let payload: Vec<u8> = (0..=255u8).collect();
/// # let buffers: [&[u8]; 1] = [&payload];
/// # let mut bytes = Vec::new();
/// # let mut writer = FileWriterCore::new();
/// # drive(writer.header()?, &mut bytes)?;
/// # drive(writer.tensor("embeddings", &descriptor, &buffers)?, &mut bytes)?;
/// # drive(writer.finish(vec![])?, &mut bytes)?;
/// let mut file = std::io::Cursor::new(bytes);
/// let size = file.seek(SeekFrom::End(0))?;
///
/// let mut core = FileReaderCore::new(size);
/// // Open: header, trailer, index, KV.
/// loop {
///     match core.step()? {
///         FileStep::Range { offset, len } => {
///             file.seek(SeekFrom::Start(offset))?;
///             let n = file.read(&mut core.input_dst()[..len])?;
///             core.report_filled(n)?;
///         }
///         FileStep::Ready => break,
///         other => panic!("unexpected {other:?} while opening"),
///     }
/// }
/// assert_eq!(core.index().len(), 1);
///
/// // Then one tensor, by name.
/// core.begin_tensor("embeddings")?;
/// let tensor = loop {
///     match core.step()? {
///         FileStep::Range { offset, len } => {
///             file.seek(SeekFrom::Start(offset))?;
///             let n = file.read(&mut core.input_dst()[..len])?;
///             core.report_filled(n)?;
///         }
///         FileStep::Tensor => break core.take_tensor().expect("a tensor is ready"),
///         other => panic!("unexpected {other:?} while reading"),
///     }
/// };
/// assert_eq!(tensor.buffers[0].as_ref(), &payload[..]);
/// // The destination carried the declared alignment, so a SIMD load is safe on it.
/// assert_eq!(tensor.buffers[0].as_ptr() as usize % 64, 0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct FileReaderCore<A: DestinationAlloc = SystemDestinations> {
    size: u64,
    data_buffer_alignment: u64,
    max_composite_depth: usize,
    destinations: A,
    state: State,
    job: Option<Job>,
    /// Header flags, kept until the index CRC is checked against the trailer's.
    flags: u32,
    index: Vec<IndexEntry>,
    kv: Vec<(String, KvValue)>,
    /// `(offset, length)` of the KV section, read from the trailer.
    kv_range: (u64, u32),
    /// The index CRC-32C the trailer records.
    stored_crc: u32,
    /// Absolute offset of the next byte wanted.
    cursor: u64,
    /// Bytes the current target wants in total.
    target: usize,
    /// Bytes of that target already in place.
    filled: usize,
    /// Sections and descriptors land here; payloads never do (§ 2).
    staging: Vec<u8>,
    /// The name and data offset of the entry being read.
    entry_name: String,
    entry_data_offset: u64,
    pending: Option<Pending<A::Dest>>,
    walk: Option<Walk>,
    descriptor_out: Option<TensorDescriptor>,
    tensor_out: Option<FileTensor>,
    composite_out: Option<FileComposite>,
}

// Manual, because the allocator and its destinations carry no `Debug` bound — and
// `FileReader` has derived `Debug` since 0.1.0, so the core it now holds needs one.
impl<A: DestinationAlloc> std::fmt::Debug for FileReaderCore<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileReaderCore")
            .field("size", &self.size)
            .field("state", &self.state)
            .field("tensors", &self.index.len())
            .field("kv_entries", &self.kv.len())
            .finish_non_exhaustive()
    }
}

// The default allocator's constructor: `FileReaderCore` with no type argument.
impl FileReaderCore<SystemDestinations> {
    /// Creates a core for a file of `size` bytes, reading into host allocations.
    ///
    /// The size is a constructor argument rather than a request the core makes: every
    /// driver already knows it — a seek to the end, a `stat`, or the transport vtable's
    /// `size` callback — and a size the core cannot act on buys nothing.
    ///
    /// # Examples
    ///
    /// See the type-level example.
    pub fn new(size: u64) -> Self {
        Self::with_destinations(size, SystemDestinations)
    }
}

impl<A: DestinationAlloc> FileReaderCore<A> {
    /// Creates a core whose read destinations come from `destinations`.
    ///
    /// The seam ADR-041 left open: a binding may supply the allocation so a read lands
    /// directly in memory its runtime owns, as long as the destination satisfies the
    /// alignment the handle declares.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_io::file::FileReaderCore;
    /// use hurray_io::stream::SystemDestinations;
    ///
    /// let core = FileReaderCore::with_destinations(4096, SystemDestinations);
    /// assert!(!core.is_open());
    /// ```
    pub fn with_destinations(size: u64, destinations: A) -> Self {
        Self {
            size,
            // Replaced by the header's value; never used before then.
            data_buffer_alignment: 1,
            max_composite_depth: DEFAULT_MAX_COMPOSITE_DEPTH,
            destinations,
            state: State::Header,
            job: None,
            flags: 0,
            index: Vec::new(),
            kv: Vec::new(),
            kv_range: (0, 0),
            stored_crc: 0,
            cursor: 0,
            target: FILE_HEADER_SIZE as usize,
            filled: 0,
            staging: vec![0u8; FILE_HEADER_SIZE as usize],
            entry_name: String::new(),
            entry_data_offset: 0,
            pending: None,
            walk: None,
            descriptor_out: None,
            tensor_out: None,
            composite_out: None,
        }
    }

    /// Sets the maximum composite nesting depth accepted by
    /// [`begin_composite`][Self::begin_composite]. Default:
    /// [`DEFAULT_MAX_COMPOSITE_DEPTH`].
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_io::file::FileReaderCore;
    ///
    /// let core = FileReaderCore::new(4096).with_max_composite_depth(8);
    /// assert!(!core.is_open());
    /// ```
    pub fn with_max_composite_depth(mut self, max_composite_depth: usize) -> Self {
        self.max_composite_depth = max_composite_depth;
        self
    }

    /// Advances the machine and returns what it needs next.
    ///
    /// Never blocks and never waits: every transition here is a parse, an allocation or
    /// bookkeeping, which is what lets one core serve a blocking driver and an event loop
    /// alike (ADR-041 § 4).
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidFileMagic`], [`Error::UnsupportedContainerVersion`],
    ///   [`Error::ReservedFileFlagBits`], [`Error::InvalidTrailerMagic`],
    ///   [`Error::IndexOverrunsTrailer`], [`Error::IndexCrc32cMismatch`],
    ///   [`Error::DuplicateTensorName`] — the file's own structure is wrong
    /// - [`Error::Core`] — a descriptor failed to decode
    /// - [`Error::UnallocatableBuffer`] — a destination the allocator cannot provide
    /// - [`Error::TornComposite`], [`Error::CompositeNestingTooDeep`],
    ///   [`Error::NotAComposite`] — a composite walk could not be completed
    /// - [`Error::CoreFailed`] — the open sequence failed earlier
    ///
    /// # Examples
    ///
    /// See the type-level example.
    pub fn step(&mut self) -> Result<FileStep> {
        loop {
            match self.state {
                State::Failed => return Err(Error::CoreFailed),
                State::Ready => return Ok(FileStep::Ready),
                State::Done(Done::Descriptor) => return Ok(FileStep::Descriptor),
                State::Done(Done::Tensor) => return Ok(FileStep::Tensor),
                State::Done(Done::Composite) => return Ok(FileStep::Composite),
                State::Header
                | State::Trailer
                | State::Index
                | State::Kv
                | State::Descriptor
                | State::Buffer { .. } => {
                    if self.filled < self.target {
                        return Ok(FileStep::Range {
                            offset: self.cursor,
                            len: self.target - self.filled,
                        });
                    }
                    self.advance()?;
                }
            }
        }
    }

    /// The memory the current [`FileStep::Range`] should be read into.
    ///
    /// Exactly as long as the outstanding [`FileStep::Range`]'s `len`, and its address
    /// does not move while that request stands, so a driver may hold a pointer into it
    /// across a suspension (§ 6a).
    ///
    /// # Examples
    ///
    /// See the type-level example.
    pub fn input_dst(&mut self) -> &mut [u8] {
        let filled = self.filled;
        match self.state {
            State::Buffer { .. } => match self.pending.as_mut().and_then(|p| p.dest.as_mut()) {
                Some(dest) => &mut dest.as_mut_slice()[filled..],
                // Unreachable while a Buffer request is outstanding; an empty slice makes
                // a driver that ignores `step` report zero rather than corrupt memory.
                None => &mut [],
            },
            State::Header | State::Trailer | State::Index | State::Kv | State::Descriptor => {
                &mut self.staging[filled..]
            }
            _ => &mut [],
        }
    }

    /// Reports that `n` bytes of the outstanding range were read.
    ///
    /// Records a count and nothing else: every parse, allocation and state change happens
    /// in [`step`][Self::step], which keeps the window a cancellation can land in as
    /// narrow as possible (§ 6).
    ///
    /// # Errors
    ///
    /// - [`Error::UnexpectedEof`] — `n` was `0`: the file is shorter than its index claims
    /// - [`Error::InvalidProgressReport`] — `n` exceeded the bytes offered, or no range
    ///   was outstanding
    ///
    /// # Examples
    ///
    /// See the type-level example.
    pub fn report_filled(&mut self, n: usize) -> Result<()> {
        let offered = match self.state {
            State::Header
            | State::Trailer
            | State::Index
            | State::Kv
            | State::Descriptor
            | State::Buffer { .. } => self.target - self.filled,
            State::Failed => return Err(Error::CoreFailed),
            _ => 0,
        };
        if offered == 0 || n > offered {
            return Err(Error::InvalidProgressReport {
                reported: n,
                offered,
            });
        }
        if n == 0 {
            return Err(self.fail(Error::UnexpectedEof));
        }
        self.filled += n;
        self.cursor += n as u64;
        Ok(())
    }

    /// Starts reading the descriptor for `name`, without reading any buffer data.
    ///
    /// Supersedes any job already in flight: the index makes every offset absolute, so
    /// there is nothing to unwind.
    ///
    /// # Errors
    ///
    /// - [`Error::TensorNotFound`] — no tensor of that name is in the index
    /// - [`Error::CoreFailed`] — the open sequence failed, or has not finished
    ///
    /// # Examples
    ///
    /// See the type-level example.
    pub fn begin_descriptor(&mut self, name: &str) -> Result<()> {
        self.begin_job(Job::Descriptor)?;
        self.start_descriptor(name)
    }

    /// Starts reading the complete tensor named `name`: descriptor, then every buffer.
    ///
    /// Supersedes any job already in flight.
    ///
    /// # Errors
    ///
    /// As [`begin_descriptor`][Self::begin_descriptor].
    ///
    /// # Examples
    ///
    /// See the type-level example.
    pub fn begin_tensor(&mut self, name: &str) -> Result<()> {
        self.begin_job(Job::Tensor)?;
        self.start_descriptor(name)
    }

    /// Starts reading the composite whose head is named `head_name`, reassembling the head
    /// with its members.
    ///
    /// # Errors
    ///
    /// - [`Error::TensorNotFound`] — no tensor of that name is in the index
    /// - [`Error::CoreFailed`] — the open sequence failed, or has not finished
    ///
    /// Walking it may further fail with [`Error::NotAComposite`],
    /// [`Error::TornComposite`] or [`Error::CompositeNestingTooDeep`], reported from
    /// [`step`][Self::step].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn main() -> hurray_io::Result<()> {
    /// # let mut core = hurray_io::file::FileReaderCore::new(4096);
    /// core.begin_composite("weight")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn begin_composite(&mut self, head_name: &str) -> Result<()> {
        self.begin_job(Job::Walk)?;

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

        self.walk = Some(Walk {
            names: ordered.into_iter().map(|(_, name)| name).collect(),
            cursor: start,
            stack: Vec::new(),
        });
        self.walk_next()
    }

    /// Takes the descriptor [`FileStep::Descriptor`] announced, leaving the core ready.
    ///
    /// # Examples
    ///
    /// See the type-level example.
    pub fn take_descriptor(&mut self) -> Option<TensorDescriptor> {
        let descriptor = self.descriptor_out.take()?;
        self.finish_job();
        Some(descriptor)
    }

    /// Takes the tensor [`FileStep::Tensor`] announced, leaving the core ready.
    ///
    /// # Examples
    ///
    /// See the type-level example.
    pub fn take_tensor(&mut self) -> Option<FileTensor> {
        let tensor = self.tensor_out.take()?;
        self.finish_job();
        Some(tensor)
    }

    /// Takes the composite [`FileStep::Composite`] announced, leaving the core ready.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn main() -> hurray_io::Result<()> {
    /// # let mut core = hurray_io::file::FileReaderCore::new(4096);
    /// if let Some(composite) = core.take_composite() {
    ///     println!("{} member(s)", composite.members.len());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn take_composite(&mut self) -> Option<FileComposite> {
        let composite = self.composite_out.take()?;
        self.finish_job();
        Some(composite)
    }

    /// The footer index: one entry per tensor, in the order the file records it.
    ///
    /// Empty until [`FileStep::Ready`].
    ///
    /// # Examples
    ///
    /// See the type-level example.
    pub fn index(&self) -> &[IndexEntry] {
        &self.index
    }

    /// The file-level KV metadata, empty if the file has none.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_io::file::FileReaderCore;
    ///
    /// let core = FileReaderCore::new(4096);
    /// assert!(core.kv().is_empty());
    /// ```
    pub fn kv(&self) -> &[(String, KvValue)] {
        &self.kv
    }

    /// Whether the open sequence has finished, so the index and KV are available.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_io::file::FileReaderCore;
    ///
    /// assert!(!FileReaderCore::new(4096).is_open());
    /// ```
    pub fn is_open(&self) -> bool {
        !matches!(
            self.state,
            State::Header | State::Trailer | State::Index | State::Kv | State::Failed
        )
    }

    // ── transitions ───────────────────────────────────────────────────────────

    /// A failed *job* leaves the core usable: every offset comes from the index, so the
    /// next `begin_*` is unaffected by where this one stopped. A failure during the open
    /// sequence is terminal, because there is no index to start again from.
    fn fail(&mut self, error: Error) -> Error {
        match self.state {
            State::Header | State::Trailer | State::Index | State::Kv => self.fail_open(error),
            _ => {
                self.state = State::Ready;
                self.job = None;
                self.pending = None;
                self.walk = None;
                error
            }
        }
    }

    fn fail_open(&mut self, error: Error) -> Error {
        self.state = State::Failed;
        error
    }

    fn begin_job(&mut self, job: Job) -> Result<()> {
        if !self.is_open() {
            return Err(Error::CoreFailed);
        }
        self.job = Some(job);
        self.pending = None;
        self.walk = None;
        self.descriptor_out = None;
        self.tensor_out = None;
        self.composite_out = None;
        Ok(())
    }

    fn finish_job(&mut self) {
        self.job = None;
        self.state = State::Ready;
    }

    /// Stages a range of `len` bytes at `offset` for the given state.
    fn request(&mut self, offset: u64, len: usize, state: State) {
        self.cursor = offset;
        self.target = len;
        self.filled = 0;
        self.staging.clear();
        self.staging.resize(len, 0);
        self.state = state;
    }

    fn advance(&mut self) -> Result<()> {
        match self.state {
            State::Header => self.finish_header(),
            State::Trailer => self.finish_trailer(),
            State::Index => self.finish_index(),
            State::Kv => self.finish_kv(),
            State::Descriptor => self.finish_descriptor(),
            State::Buffer { index } => self.finish_buffer(index),
            // `advance` is only reached from the filling states.
            _ => Err(self.fail(Error::CoreFailed)),
        }
    }

    fn finish_header(&mut self) -> Result<()> {
        // Checked here rather than before the header request, so a file too short to even
        // hold a header still fails as the truncated file it is: the header range comes
        // back short, and a zero-byte read is `UnexpectedEof`.
        if self.size < FILE_HEADER_SIZE + TRAILER_SIZE {
            return Err(self.fail_open(Error::InvalidHeader(
                "file too small to contain a valid trailer".into(),
            )));
        }

        let header = &self.staging;
        if header[0..8] != FILE_MAGIC[..] {
            return Err(self.fail_open(Error::InvalidFileMagic));
        }

        let container_version_major = header[8];
        if container_version_major > SUPPORTED_CONTAINER_VERSION_MAJOR {
            return Err(self.fail_open(Error::UnsupportedContainerVersion {
                major: container_version_major,
            }));
        }

        let flags = u32::from_le_bytes(header[12..16].try_into().unwrap_or([0; 4]));
        if flags & file_flags::RESERVED_MASK != 0 {
            return Err(self.fail_open(Error::ReservedFileFlagBits { flags }));
        }
        self.flags = flags;

        let alignment = u32::from_le_bytes(header[16..20].try_into().unwrap_or([0; 4]));
        // `file-format.md § File Header` makes this a power of two in [4096, 2 MiB]. The
        // check is not pedantry: `0` reaches the padding arithmetic as a divisor.
        if !alignment.is_power_of_two() || alignment > MAX_DATA_BUFFER_ALIGNMENT {
            return Err(self.fail_open(Error::InvalidHeader(format!(
                "data_buffer_alignment {alignment} is not a power of two in \
                 [1, {MAX_DATA_BUFFER_ALIGNMENT}]"
            ))));
        }
        self.data_buffer_alignment = u64::from(alignment);

        self.request(
            self.size - TRAILER_SIZE,
            TRAILER_SIZE as usize,
            State::Trailer,
        );
        Ok(())
    }

    fn finish_trailer(&mut self) -> Result<()> {
        let trailer = &self.staging;
        if trailer[36..40] != TRAILER_MAGIC[..] {
            return Err(self.fail_open(Error::InvalidTrailerMagic));
        }

        let index_offset = u64::from_le_bytes(trailer[0..8].try_into().unwrap_or([0; 8]));
        let index_length = u64::from_le_bytes(trailer[8..16].try_into().unwrap_or([0; 8]));
        let kv_offset = u64::from_le_bytes(trailer[16..24].try_into().unwrap_or([0; 8]));
        let kv_length = u32::from_le_bytes(trailer[24..28].try_into().unwrap_or([0; 4]));
        self.stored_crc = u32::from_le_bytes(trailer[28..32].try_into().unwrap_or([0; 4]));
        self.kv_range = (kv_offset, kv_length);

        // Checked: a crafted trailer can make these sum past u64, and an overflow panic in
        // a parser fed untrusted bytes is a worse outcome than the error it would raise.
        match index_offset.checked_add(index_length) {
            Some(end) if end <= self.size - TRAILER_SIZE => {}
            _ => return Err(self.fail_open(Error::IndexOverrunsTrailer)),
        }

        let length = match usize::try_from(index_length) {
            Ok(length) => length,
            Err(_) => return Err(self.fail_open(Error::IndexOverrunsTrailer)),
        };
        self.request(index_offset, length, State::Index);
        Ok(())
    }

    fn finish_index(&mut self) -> Result<()> {
        if self.flags & file_flags::HAS_INDEX_CRC32C != 0 {
            let computed = crc32c::crc32c(&self.staging);
            if computed != self.stored_crc {
                return Err(self.fail_open(Error::IndexCrc32cMismatch {
                    stored: self.stored_crc,
                    computed,
                }));
            }
        }

        self.index = match parse_index(&self.staging) {
            Ok(index) => index,
            Err(e) => return Err(self.fail_open(e)),
        };

        // Accept KV if either the flag or the trailer fields indicate its presence. A
        // streaming writer may not set HAS_KV_METADATA in the header (no seek-back), so
        // kv_offset != 0 is treated as authoritative.
        let (kv_offset, kv_length) = self.kv_range;
        let has_kv = (self.flags & file_flags::HAS_KV_METADATA != 0) || kv_offset != 0;
        if has_kv && kv_offset != 0 && kv_length != 0 {
            self.request(kv_offset, kv_length as usize, State::Kv);
        } else {
            self.state = State::Ready;
        }
        Ok(())
    }

    fn finish_kv(&mut self) -> Result<()> {
        self.kv = match parse_kv(&self.staging) {
            Ok(kv) => kv,
            Err(e) => return Err(self.fail_open(e)),
        };
        self.state = State::Ready;
        Ok(())
    }

    /// Looks `name` up in the index and asks for its descriptor bytes.
    fn start_descriptor(&mut self, name: &str) -> Result<()> {
        let entry = self
            .index
            .iter()
            .find(|e| e.name == name)
            .ok_or_else(|| Error::TensorNotFound(name.to_string()))?;

        let offset = entry.descriptor_offset;
        let length = entry.descriptor_length as usize;
        self.entry_name = entry.name.clone();
        self.entry_data_offset = entry.data_offset;
        self.request(offset, length, State::Descriptor);
        Ok(())
    }

    fn finish_descriptor(&mut self) -> Result<()> {
        let descriptor = match TensorDescriptor::decode(&self.staging) {
            Ok(descriptor) => descriptor,
            Err(e) => return Err(self.fail(e.into())),
        };

        match self.job {
            Some(Job::Descriptor) => {
                self.descriptor_out = Some(descriptor);
                self.state = State::Done(Done::Descriptor);
                Ok(())
            }
            Some(Job::Tensor) => self.begin_buffers(descriptor),
            Some(Job::Walk) => self.walk_descriptor(descriptor),
            None => Err(self.fail(Error::CoreFailed)),
        }
    }

    /// Enters the buffer-filling states for `descriptor`, or completes it at once when it
    /// declares no buffers — the composite head case.
    fn begin_buffers(&mut self, descriptor: TensorDescriptor) -> Result<()> {
        let count = descriptor.buffers.len();
        self.pending = Some(Pending {
            name: std::mem::take(&mut self.entry_name),
            descriptor,
            buffers: Vec::with_capacity(count),
            dest: None,
            next_offset: self.entry_data_offset,
        });
        if count == 0 {
            return self.complete_tensor();
        }
        self.begin_buffer(0)
    }

    /// Allocates the destination for buffer `index` and asks for its bytes.
    fn begin_buffer(&mut self, index: usize) -> Result<()> {
        let (handle, offset) = match self.pending.as_ref() {
            Some(pending) => match pending.descriptor.buffers.get(index) {
                Some(handle) => (*handle, pending.next_offset),
                None => return Err(self.fail(Error::CoreFailed)),
            },
            None => return Err(self.fail(Error::CoreFailed)),
        };

        let request = DestinationRequest {
            index,
            byte_size: handle.byte_size(),
            alignment: handle.alignment(),
        };
        let len = match usize::try_from(request.byte_size) {
            Ok(len) => len,
            Err(_) => {
                return Err(self.fail(Error::UnallocatableBuffer {
                    index,
                    byte_size: request.byte_size,
                    alignment: request.alignment,
                }))
            }
        };
        let dest = match self.destinations.allocate(request) {
            Ok(dest) => dest,
            Err(e) => return Err(self.fail(e)),
        };
        match self.pending.as_mut() {
            Some(pending) => pending.dest = Some(dest),
            None => return Err(self.fail(Error::CoreFailed)),
        }

        // `request` would resize the staging buffer; a payload never lands there (§ 2).
        self.cursor = offset;
        self.target = len;
        self.filled = 0;
        self.state = State::Buffer { index };
        Ok(())
    }

    /// Buffer `index` is full: freeze it, then move to the next one or finish the tensor.
    fn finish_buffer(&mut self, index: usize) -> Result<()> {
        let count = match self.pending.as_mut() {
            Some(pending) => match pending.dest.take() {
                Some(dest) => {
                    pending.buffers.push(dest.freeze());
                    pending.descriptor.buffers.len()
                }
                None => return Err(self.fail(Error::CoreFailed)),
            },
            None => return Err(self.fail(Error::CoreFailed)),
        };

        let next = index + 1;
        if next < count {
            // Each buffer begins on a `data_buffer_alignment` boundary, so the padding
            // after this one is skipped rather than read.
            let offset = align_up(self.cursor, self.data_buffer_alignment);
            match self.pending.as_mut() {
                Some(pending) => pending.next_offset = offset,
                None => return Err(self.fail(Error::CoreFailed)),
            }
            self.begin_buffer(next)
        } else {
            self.complete_tensor()
        }
    }

    fn complete_tensor(&mut self) -> Result<()> {
        let pending = match self.pending.take() {
            Some(pending) => pending,
            None => return Err(self.fail(Error::CoreFailed)),
        };
        let tensor = FileTensor {
            name: pending.name,
            descriptor: pending.descriptor,
            buffers: pending.buffers,
        };

        match self.job {
            Some(Job::Tensor) => {
                self.tensor_out = Some(tensor);
                self.state = State::Done(Done::Tensor);
                Ok(())
            }
            Some(Job::Walk) => self.deliver(FileItem::Tensor(tensor)),
            _ => Err(self.fail(Error::CoreFailed)),
        }
    }

    // ── the composite walk (§ 10) ─────────────────────────────────────────────
    //
    // The same shape as `CompositeAssembler`, not the same code: that one assembles the
    // stream's nameless items as they are pushed at it, while this one pulls named items
    // out of the index in file order. Sharing them would mean a name-shaped hole in the
    // stream's types, which have no names on the wire.

    /// Asks for the next item in the walk.
    fn walk_next(&mut self) -> Result<()> {
        let (name, exhausted) = match self.walk.as_mut() {
            Some(walk) => match walk.names.get(walk.cursor) {
                Some(name) => {
                    let name = name.clone();
                    walk.cursor += 1;
                    (name, false)
                }
                None => (String::new(), true),
            },
            None => return Err(self.fail(Error::CoreFailed)),
        };

        if exhausted {
            // The file ran out with a composite still open. The innermost one is reported:
            // that is the level whose members actually went missing.
            let torn = self
                .walk
                .as_ref()
                .and_then(|walk| walk.stack.last())
                .map(|open| Error::TornComposite {
                    declared: open.member_count,
                    actual: open.members.len() as u32,
                })
                .unwrap_or(Error::CoreFailed);
            return Err(self.fail(torn));
        }

        self.start_descriptor(&name)
    }

    /// A descriptor arrived during a walk: open a composite, or read the tensor's buffers.
    fn walk_descriptor(&mut self, descriptor: TensorDescriptor) -> Result<()> {
        let member_count = match &descriptor.layout {
            LayoutDescriptor::Composite(layout) => layout.member_count,
            _ => {
                // The name the walk started from must be a head; anything else is the
                // caller's mistake, caught before its buffers are read.
                if self.walk.as_ref().is_some_and(|walk| walk.stack.is_empty()) {
                    let name = std::mem::take(&mut self.entry_name);
                    return Err(self.fail(Error::NotAComposite(name)));
                }
                return self.begin_buffers(descriptor);
            }
        };

        let too_deep = match self.walk.as_ref() {
            Some(walk) => walk.stack.len() >= self.max_composite_depth,
            None => return Err(self.fail(Error::CoreFailed)),
        };
        if too_deep {
            return Err(self.fail(Error::CompositeNestingTooDeep {
                limit: self.max_composite_depth,
            }));
        }

        let validator = match CompositeValidator::new(&descriptor) {
            Ok(validator) => validator,
            Err(e) => return Err(self.fail(e.into())),
        };
        let open = Open {
            name: std::mem::take(&mut self.entry_name),
            head: descriptor,
            member_count,
            members: Vec::new(),
            validator,
        };
        match self.walk.as_mut() {
            Some(walk) => walk.stack.push(open),
            None => return Err(self.fail(Error::CoreFailed)),
        }
        self.walk_next()
    }

    /// Hands `item` to the innermost open head, closing heads — and their parents — as
    /// each one fills. Loops rather than recurses (§ 10).
    fn deliver(&mut self, item: FileItem) -> Result<()> {
        let mut item = item;
        loop {
            // Pop and push back rather than peek: the closing branch needs to own the
            // head, and `CompositeValidator::finish` consumes the validator.
            let top = match self.walk.as_mut() {
                Some(walk) => walk.stack.pop(),
                None => return Err(self.fail(Error::CoreFailed)),
            };
            let Some(mut top) = top else {
                // Nothing left open: the walk produced its composite.
                let FileItem::Composite(composite) = item else {
                    return Err(self.fail(Error::CoreFailed));
                };
                self.composite_out = Some(composite);
                self.walk = None;
                self.state = State::Done(Done::Composite);
                return Ok(());
            };

            if let Err(e) = top.validator.push_member(item.descriptor()) {
                return Err(self.fail(e.into()));
            }
            top.members.push(item);

            if (top.members.len() as u32) < top.member_count {
                match self.walk.as_mut() {
                    Some(walk) => walk.stack.push(top),
                    None => return Err(self.fail(Error::CoreFailed)),
                }
                return self.walk_next();
            }

            let Open {
                name,
                head,
                members,
                validator,
                ..
            } = top;
            if let Err(e) = validator.finish() {
                return Err(self.fail(e.into()));
            }
            // A closed composite is itself an item: a member of its parent, or the one the
            // caller gets.
            item = FileItem::Composite(FileComposite {
                name,
                head,
                members,
            });
        }
    }
}
