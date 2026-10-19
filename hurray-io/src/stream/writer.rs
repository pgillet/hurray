use hurray_core::TensorDescriptor;
use tokio::io::{AsyncWrite, AsyncWriteExt};

use crate::stream::sans_io::{StreamWriterCore, WriteOp, WriteStep};
// `pub use`: 0.1.0 published this at `stream::writer::CompositeNode` too (see reader.rs).
pub use crate::stream::types::CompositeNode;
use crate::{Error, Result};

/// Writes tensors to an async sink in the Hurray streaming wire format.
///
/// The wire format is bare concatenation: each tensor is represented as its
/// encoded [`TensorDescriptor`] immediately followed by each buffer's raw bytes.
/// No outer framing, no alignment padding between tensors.
///
/// This is a driver, not an encoder: [`StreamWriterCore`] holds the wire format and emits
/// the bytes, and this type only hands them to the sink (ADR-041 § 7). Payloads are never
/// assembled or copied on the way — each slice goes to the sink as it is (§ 3).
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
/// use hurray_io::stream::StreamWriter;
///
/// let handle = BufferHandle::new(64, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced)?;
/// let shape = Shape::new(vec![4u64, 4]).unwrap();
/// let desc = TensorDescriptor::new(
///     1, 0,
///     ElementType::Float32,
///     shape,
///     0,
///     LayoutDescriptor::RowMajor,
///     vec![handle],
///     None, None, None, None,
/// )?;
/// let data = vec![0u8; 64];
///
/// let mut wire = Vec::<u8>::new();
/// let mut writer = StreamWriter::new(&mut wire);
/// writer.write_tensor(&desc, &[&data]).await?;
/// writer.finish().await?;
/// # Ok(())
/// # }
/// ```
pub struct StreamWriter<W> {
    inner: W,
    core: StreamWriterCore,
}

impl<W: AsyncWrite + Unpin> StreamWriter<W> {
    /// Creates a writer that accepts any valid [`SyncMode`][hurray_core::SyncMode].
    pub fn new(inner: W) -> Self {
        Self {
            inner,
            core: StreamWriterCore::new(),
        }
    }

    /// Creates a writer that rejects buffers whose `sync_mode` is not
    /// [`ProducerSynced`][hurray_core::SyncMode::ProducerSynced].
    ///
    /// Use this when the stream crosses machine boundaries where GPU and
    /// semaphore-based sync primitives are not transferable.
    pub fn cross_machine(inner: W) -> Self {
        Self {
            inner,
            core: StreamWriterCore::cross_machine(),
        }
    }

    /// Encodes and writes one tensor.
    ///
    /// # Errors
    ///
    /// - [`Error::MultiBufferLengthMismatch`] — `buffers.len()` ≠ `desc.buffers.len()`
    /// - [`Error::BufferSizeMismatch`] — a buffer's byte length ≠ its handle's `byte_size`
    /// - [`Error::InvalidCrossMachineSyncMode`] — cross-machine mode and a buffer has a
    ///   non-`ProducerSynced` sync mode
    /// - [`Error::SinkClosed`] — the sink accepted zero bytes mid-item
    /// - [`Error::CoreUnusable`] — an earlier write left this writer's sink poisoned
    /// - [`Error::Core`] — descriptor encoding failed
    /// - [`Error::Io`] — underlying write error
    pub async fn write_tensor(&mut self, desc: &TensorDescriptor, buffers: &[&[u8]]) -> Result<()> {
        let mut op = self.core.tensor(desc, buffers)?;
        drain(&mut op, &mut self.inner).await
    }

    /// Encodes and writes a composite tensor: its head followed by every member's
    /// descriptor and data, in order (ADR-027 § Binding).
    ///
    /// The group is validated *before* any byte is written — reusing
    /// [`CompositeValidator`][hurray_core::CompositeValidator] to check `member_count` and
    /// the per-rule constraints (partition exact-cover / non-overlap, overlay base-first
    /// ordering) — so a torn or invalid composite never reaches the wire. Members that are
    /// themselves composites are written as part of the same operation (head precedes its
    /// members at every level), preserving the forward, self-delimiting,
    /// back-reference-free wire contract.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use hurray_core::{
    ///     layout::{CompositeLayout, CompositionRule, LayoutDescriptor},
    ///     ElementType, Shape, ShardDescriptor, TensorDescriptor,
    /// };
    /// use hurray_io::stream::{CompositeNode, StreamWriter};
    ///
    /// // A partition head [8, 8] split into two [8, 4] members (buffers elided here).
    /// let head = TensorDescriptor::new(
    ///     1, 0, ElementType::Float32, Shape::new(vec![8u64, 8]).unwrap(), 0,
    ///     LayoutDescriptor::Composite(CompositeLayout::new(CompositionRule::Partition, 2).unwrap()),
    ///     vec![], None, None, None, None,
    /// )?;
    /// # let member_descs: Vec<TensorDescriptor> = vec![];
    /// # let members: Vec<CompositeNode> = vec![];
    /// let mut wire = Vec::<u8>::new();
    /// let mut writer = StreamWriter::new(&mut wire);
    /// writer.write_composite(&head, &members).await?;
    /// writer.finish().await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// - [`Error::Core`] — the head is not a valid composite head, or validation failed
    ///   (member-count mismatch, partition coverage, overlay ordering)
    /// - the same buffer/sync/sink errors as [`write_tensor`][StreamWriter::write_tensor]
    ///   for each member
    /// - [`Error::Io`] — underlying write error
    pub async fn write_composite(
        &mut self,
        head: &TensorDescriptor,
        members: &[CompositeNode<'_>],
    ) -> Result<()> {
        let mut op = self.core.composite(head, members)?;
        drain(&mut op, &mut self.inner).await
    }

    /// Flushes the underlying sink and returns it.
    pub async fn finish(mut self) -> Result<W> {
        self.inner.flush().await?;
        Ok(self.inner)
    }
}

/// Writes every byte of one submitted item to `sink`.
///
/// A free function rather than a method: the operation holds a mutable borrow of the
/// writer's core for as long as it lives, so only disjoint *field* borrows let the sink be
/// used at the same time.
async fn drain<W: AsyncWrite + Unpin>(op: &mut WriteOp<'_>, sink: &mut W) -> Result<()> {
    while let WriteStep::Output { .. } = op.step()? {
        let n = {
            let src = op.output_src();
            loop {
                match sink.write(src).await {
                    Ok(n) => break n,
                    // `write_all`, which this replaces, absorbed this internally.
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    // A write that errors put no bytes on the wire, so the count is known
                    // to be zero and neither core nor sink is poisoned (§ 6).
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
