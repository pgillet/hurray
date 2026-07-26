use hurray_core::{CompositeValidator, SyncMode, TensorDescriptor};
use tokio::io::{AsyncWrite, AsyncWriteExt};

use crate::{Error, Result};

/// A node to emit as part of a composite via [`StreamWriter::write_composite`]: either a
/// plain tensor (its descriptor plus one byte-slice per buffer) or a nested composite (its
/// head plus its own members). The recursive [`Composite`][CompositeNode::Composite] arm
/// lets a member be a composite in its own right (ADR-027 § Binding).
pub enum CompositeNode<'a> {
    /// A single (non-composite) tensor: descriptor + one byte-slice per declared buffer.
    Tensor {
        /// The member's tensor descriptor.
        descriptor: &'a TensorDescriptor,
        /// One byte-slice per buffer handle in `descriptor.buffers`.
        buffers: &'a [&'a [u8]],
    },
    /// A nested composite: a data-less head plus its ordered members.
    Composite {
        /// The nested composite's head descriptor.
        head: &'a TensorDescriptor,
        /// The nested composite's members, in order.
        members: &'a [CompositeNode<'a>],
    },
}

impl CompositeNode<'_> {
    /// The node's governing descriptor: the tensor's descriptor, or the nested head.
    fn descriptor(&self) -> &TensorDescriptor {
        match self {
            CompositeNode::Tensor { descriptor, .. } => descriptor,
            CompositeNode::Composite { head, .. } => head,
        }
    }
}

/// Writes tensors to an async sink in the Hurray streaming wire format.
///
/// The wire format is bare concatenation: each tensor is represented as its
/// encoded [`TensorDescriptor`] immediately followed by each buffer's raw bytes.
/// No outer framing, no alignment padding between tensors.
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
    enforce_cross_machine_sync: bool,
}

impl<W: AsyncWrite + Unpin> StreamWriter<W> {
    /// Creates a writer that accepts any valid [`SyncMode`].
    pub fn new(inner: W) -> Self {
        Self {
            inner,
            enforce_cross_machine_sync: false,
        }
    }

    /// Creates a writer that rejects buffers whose `sync_mode` is not
    /// [`SyncMode::ProducerSynced`].
    ///
    /// Use this when the stream crosses machine boundaries where GPU and
    /// semaphore-based sync primitives are not transferable.
    pub fn cross_machine(inner: W) -> Self {
        Self {
            inner,
            enforce_cross_machine_sync: true,
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
    /// - [`Error::Core`] — descriptor encoding failed
    /// - [`Error::Io`] — underlying write error
    pub async fn write_tensor(&mut self, desc: &TensorDescriptor, buffers: &[&[u8]]) -> Result<()> {
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
            if self.enforce_cross_machine_sync && handle.sync_mode() != SyncMode::ProducerSynced {
                return Err(Error::InvalidCrossMachineSyncMode {
                    index: i,
                    actual: handle.sync_mode().to_byte(),
                });
            }
        }

        let encoded = desc.encode()?;
        self.inner.write_all(&encoded).await?;

        for buf in buffers {
            self.inner.write_all(buf).await?;
        }

        Ok(())
    }

    /// Encodes and writes a composite tensor: its head followed by every member's
    /// descriptor and data, in order (ADR-027 § Binding).
    ///
    /// The group is validated *before* any byte is written — reusing
    /// [`CompositeValidator`] to check `member_count` and the per-rule constraints
    /// (partition exact-cover / non-overlap, overlay base-first ordering) — so a torn or
    /// invalid composite never reaches the wire. Members that are themselves composites are
    /// written recursively (head precedes its members at every level), preserving the
    /// forward, self-delimiting, back-reference-free wire contract.
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
    /// - the same buffer/sync errors as [`write_tensor`][StreamWriter::write_tensor] for
    ///   each member
    /// - [`Error::Io`] — underlying write error
    pub async fn write_composite(
        &mut self,
        head: &TensorDescriptor,
        members: &[CompositeNode<'_>],
    ) -> Result<()> {
        // Validate every level up front, so a torn/invalid composite — at any nesting
        // depth — never reaches the wire.
        validate_composite_node(head, members)?;
        self.write_composite_unchecked(head, members).await
    }

    /// Writes an already-validated composite (head, then each member recursively).
    async fn write_composite_unchecked(
        &mut self,
        head: &TensorDescriptor,
        members: &[CompositeNode<'_>],
    ) -> Result<()> {
        // The head owns no data buffers (composite-head invariant).
        self.write_tensor(head, &[]).await?;

        for member in members {
            match member {
                CompositeNode::Tensor {
                    descriptor,
                    buffers,
                } => {
                    self.write_tensor(descriptor, buffers).await?;
                }
                CompositeNode::Composite {
                    head: nested_head,
                    members: nested_members,
                } => {
                    // Already validated by write_composite's up-front pass; box the
                    // recursive call since an async fn cannot name its own future.
                    Box::pin(self.write_composite_unchecked(nested_head, nested_members)).await?;
                }
            }
        }

        Ok(())
    }

    /// Flushes the underlying sink and returns it.
    pub async fn finish(mut self) -> Result<W> {
        self.inner.flush().await?;
        Ok(self.inner)
    }
}

/// Recursively validates a composite node and all nested composites, writing nothing.
///
/// Reuses [`CompositeValidator`] at each level (member count + partition coverage / overlay
/// ordering). Pure and synchronous, so [`StreamWriter::write_composite`] can validate the
/// entire tree before emitting a single byte.
fn validate_composite_node(head: &TensorDescriptor, members: &[CompositeNode<'_>]) -> Result<()> {
    let mut validator = CompositeValidator::new(head)?;
    for member in members {
        validator.push_member(member.descriptor())?;
    }
    validator.finish()?;

    for member in members {
        if let CompositeNode::Composite {
            head: nested_head,
            members: nested_members,
        } = member
        {
            validate_composite_node(nested_head, nested_members)?;
        }
    }
    Ok(())
}
