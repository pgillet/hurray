use hurray_core::TensorDescriptor;
use tokio::io::{AsyncWrite, AsyncWriteExt};

use crate::file::sans_io::{flatten_composite, FileWriteOp, FileWriterCore};
// `pub use`, not `use`: 0.1.0 published this at `file::writer::FileCompositeNode` too (see
// reader.rs). `hurray_io::file::FileCompositeNode` is canonical.
pub use crate::file::types::FileCompositeNode;
use crate::file::types::{FileWriterOptions, KvValue};
use crate::stream::WriteStep;
use crate::{Error, Result};

/// Writes a Hurray file in a single forward pass without seeks.
///
/// The file format is:
/// ```text
/// [ File header      ]  64 bytes
/// [ Tensor 0         ]  descriptor + padding + data buffers
/// [ Tensor 1         ]
///   ...
/// [ KV section       ]  optional, written by finish()
/// [ Footer index     ]  written by finish()
/// [ Trailer          ]  40 bytes, written by finish()
/// ```
///
/// This is a driver, not an encoder: [`FileWriterCore`] holds the container format, the
/// offsets and the growing index, and this type only writes the bytes the core offers and
/// reports how many the sink took (ADR-041 § 7).
///
/// `HAS_KV_METADATA` is not set in the header because KV content is supplied to
/// [`finish`][FileWriter::finish], after the header has already gone out, and this writer
/// never seeks back. Readers treat a non-zero trailer `kv_offset` as authoritative.
///
/// # Examples
///
/// ```no_run
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use hurray_core::{
///     BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode,
///     TensorDescriptor, MIN_BUFFER_ALIGNMENT,
/// };
/// use hurray_io::file::{FileWriter, KvValue};
///
/// let handle = BufferHandle::new(
///     16, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced,
/// )?;
/// let descriptor = TensorDescriptor::new(
///     1, 0, ElementType::Float32, Shape::new(vec![2u64, 2])?, 0,
///     LayoutDescriptor::RowMajor, vec![handle], None, None, None, None,
/// )?;
/// let data = [0u8; 16];
///
/// let file = tokio::fs::File::create("model.hrry").await?;
/// let mut writer = FileWriter::new(file).await?;
/// writer.write_tensor("weight", &descriptor, &[&data]).await?;
/// writer
///     .finish(vec![("producer".into(), KvValue::String("hurray".into()))])
///     .await?;
/// # Ok(())
/// # }
/// ```
pub struct FileWriter<W> {
    inner: W,
    core: FileWriterCore,
}

impl<W: AsyncWrite + Unpin> FileWriter<W> {
    /// Creates a writer with default options (4096-byte buffer alignment, unsorted index).
    pub async fn new(inner: W) -> Result<Self> {
        Self::with_options(inner, FileWriterOptions::default()).await
    }

    /// Creates a writer with custom options.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidHeader`] — `data_buffer_alignment` is not a power of two in
    ///   `[4096, 2097152]`
    /// - [`Error::Io`] — the header could not be written
    pub async fn with_options(mut inner: W, options: FileWriterOptions) -> Result<Self> {
        let mut core = FileWriterCore::with_options(options)?;
        drain(&mut core.header(), &mut inner).await?;
        Ok(Self { inner, core })
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
    /// - [`Error::SinkClosed`] — the sink stopped accepting bytes mid-tensor
    /// - [`Error::Io`] — underlying write error
    pub async fn write_tensor(
        &mut self,
        name: &str,
        desc: &TensorDescriptor,
        buffers: &[&[u8]],
    ) -> Result<()> {
        let mut op = self.core.tensor(name, desc, buffers)?;
        drain(&mut op, &mut self.inner).await
    }

    /// Writes a composite tensor: its head, then every member's descriptor and data,
    /// contiguously and in order (ADR-027 § Binding).
    ///
    /// Every tensor — the head and each member — gets its own footer-index entry, so all
    /// are individually addressable by name via
    /// [`read_tensor`][crate::file::FileReader::read_tensor]. Membership is recoverable by
    /// [`read_composite`][crate::file::FileReader::read_composite] from the head's
    /// `member_count` plus file-offset adjacency (the members are the tensors written
    /// immediately after the head). Nested composites are written in the same pass.
    ///
    /// The whole group is validated up front — reusing
    /// [`CompositeValidator`][hurray_core::CompositeValidator] for member count and
    /// per-rule constraints (partition exact-cover, overlay ordering) — before any tensor
    /// is written.
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
        // The tree is validated and flattened into write order first, so a torn or invalid
        // composite never reaches the sink — and the nesting is walked with an explicit
        // stack in the core rather than by recursing here (§ 10).
        for node in flatten_composite(head_name, head, members)? {
            let mut op = self.core.tensor(node.name, node.descriptor, node.buffers)?;
            drain(&mut op, &mut self.inner).await?;
        }
        Ok(())
    }

    /// Writes the KV section, footer index, and trailer, then flushes.
    ///
    /// `kv` is a list of `(key, value)` pairs. Keys must be non-empty, at most
    /// 65 535 bytes, and unique within the list.
    ///
    /// # Errors
    ///
    /// - [`Error::KvKeyEmpty`] / [`Error::KvKeyTooLong`] / [`Error::DuplicateKvKey`] —
    ///   invalid KV keys
    /// - [`Error::InvalidHeader`] — a KV array is empty, nested, or not homogeneous
    /// - [`Error::SinkClosed`] — the sink stopped accepting bytes mid-footer
    /// - [`Error::Io`] — underlying write error
    pub async fn finish(mut self, kv: Vec<(String, KvValue)>) -> Result<W> {
        let mut op = self.core.finish(kv)?;
        drain(&mut op, &mut self.inner).await?;
        self.inner.flush().await?;
        Ok(self.inner)
    }
}

/// Writes every byte of one submitted item to `sink`.
///
/// A free function rather than a method: the operation holds a mutable borrow of the
/// writer's core for as long as it lives, so only disjoint *field* borrows let the sink be
/// used at the same time.
async fn drain<W: AsyncWrite + Unpin>(op: &mut FileWriteOp<'_>, sink: &mut W) -> Result<()> {
    while let WriteStep::Output { .. } = op.step()? {
        let n = {
            let src = op.output_src();
            loop {
                // `write`, not `write_all`: a failed bulk write cannot report how many
                // bytes it moved, and that count is the file position (§ 6).
                match sink.write(src).await {
                    Ok(n) => break n,
                    // `write_all`, which this replaces, absorbed this internally.
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    // A write that errors put no bytes on the sink, so the count is known
                    // to be zero and the file is still intact up to here (§ 6).
                    Err(e) => return Err(Error::Io(e)),
                }
            }
        };

        if n == 0 {
            // § 4: a zero-length write is never progress. A closed sink is its own signal,
            // not an I/O error to retry.
            return Err(op.report_sink_closed());
        }
        op.report_written(n)?;
    }
    Ok(())
}
