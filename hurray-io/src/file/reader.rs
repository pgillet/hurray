use hurray_core::TensorDescriptor;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncSeek, AsyncSeekExt, SeekFrom};

use crate::file::sans_io::{FileReaderCore, FileStep};
// `pub use`, not `use`: 0.1.0 published these at `file::reader::*` as well as at
// `file::*`, because this module is public and defined them. They belong to the core now,
// and the published paths keep resolving. `hurray_io::file::*` is canonical.
pub use crate::file::types::{FileComposite, FileItem, FileTensor, DEFAULT_MAX_COMPOSITE_DEPTH};
use crate::file::types::{IndexEntry, KvValue};
use crate::{Error, Result};

/// Reads tensors from a seekable Hurray file.
///
/// Open with [`FileReader::open`], then look up tensors by name with
/// [`read_tensor`][FileReader::read_tensor] or inspect metadata with
/// [`tensor_names`][FileReader::tensor_names] and [`kv`][FileReader::kv].
///
/// This is a driver, not a parser: [`FileReaderCore`] holds the container format and
/// decides which bytes it needs, and this type only seeks to the range the core asks for
/// and reports how many bytes arrived (ADR-041 § 7).
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
    core: FileReaderCore,
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
        let size = inner.seek(SeekFrom::End(0)).await?;
        let mut core = FileReaderCore::new(size);

        loop {
            match core.step()? {
                FileStep::Range { offset, len } => fill(&mut inner, &mut core, offset, len).await?,
                FileStep::Ready => break,
                // Until the file is open the core answers nothing else.
                _ => return Err(Error::CoreFailed),
            }
        }

        // Copies, because 0.1.0 published `index` and `kv` as fields. The core keeps its
        // own: it needs them for every name lookup and for the composite walk.
        Ok(Self {
            index: core.index().to_vec(),
            kv: core.kv().to_vec(),
            inner,
            core,
        })
    }

    /// Sets the maximum composite nesting depth accepted by
    /// [`read_composite`][FileReader::read_composite]. Default:
    /// [`DEFAULT_MAX_COMPOSITE_DEPTH`].
    pub fn with_max_composite_depth(mut self, max_composite_depth: usize) -> Self {
        self.core = self.core.with_max_composite_depth(max_composite_depth);
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
        self.core.begin_descriptor(name)?;
        match self.drive().await? {
            FileStep::Descriptor => self.core.take_descriptor().ok_or(Error::CoreFailed),
            _ => Err(Error::CoreFailed),
        }
    }

    /// Reads and returns a complete tensor (descriptor + all buffers).
    pub async fn read_tensor(&mut self, name: &str) -> Result<FileTensor> {
        self.core.begin_tensor(name)?;
        match self.drive().await? {
            FileStep::Tensor => self.core.take_tensor().ok_or(Error::CoreFailed),
            _ => Err(Error::CoreFailed),
        }
    }

    /// Reads a composite tensor by its head name, reassembling the head with its members.
    ///
    /// Members are recovered by **file-offset adjacency**: the head's declared
    /// `member_count` tensors written immediately after it, recursively for nested
    /// composites. Recovery keys on the descriptor offset, not index position, so it is
    /// correct even when the file used the `SORTED_INDEX` option (which reorders the index
    /// array but not the file layout). The reassembled group is validated with
    /// [`CompositeValidator`][hurray_core::CompositeValidator].
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
        self.core.begin_composite(head_name)?;
        match self.drive().await? {
            FileStep::Composite => self.core.take_composite().ok_or(Error::CoreFailed),
            _ => Err(Error::CoreFailed),
        }
    }

    /// Returns the underlying reader.
    pub fn into_inner(self) -> R {
        self.inner
    }

    /// Answers every range the core asks for, and returns the step that ends the job.
    async fn drive(&mut self) -> Result<FileStep> {
        loop {
            match self.core.step()? {
                FileStep::Range { offset, len } => {
                    fill(&mut self.inner, &mut self.core, offset, len).await?
                }
                other => return Ok(other),
            }
        }
    }
}

/// Answers one [`FileStep::Range`]: seek, read into the core's destination, report the
/// count.
///
/// A free function rather than a method so the seek and the destination can come from
/// disjoint field borrows, which is also what `open` needs before the reader exists.
///
/// The transfer and the report are one step with nothing suspended between them, the shape
/// ADR-041 § 6a requires of every driver.
async fn fill<R: AsyncRead + AsyncSeek + Unpin>(
    inner: &mut R,
    core: &mut FileReaderCore,
    offset: u64,
    len: usize,
) -> Result<()> {
    inner.seek(SeekFrom::Start(offset)).await?;
    let n = loop {
        // `read`, not `read_exact`: a bulk helper cannot report how many bytes it moved
        // when it fails, and that count is the read position (§ 6). Here the position is
        // absolute anyway, so even a lost count would be recoverable — but a short read
        // must still be reported as what it is.
        match inner.read(&mut core.input_dst()[..len]).await {
            Ok(n) => break n,
            // `read_exact`, which this replaces, absorbed this internally.
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            // A read that errors moved no bytes, so the count is known to be zero and the
            // core's position survives.
            Err(e) => return Err(Error::Io(e)),
        }
    };
    // `n == 0` means the file is shorter than its own index claims; the core calls it.
    core.report_filled(n)
}
