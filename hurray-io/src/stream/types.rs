//! Types shared by the sans-io stream core and the Tokio drivers over it.
//!
//! These describe what the stream wire format carries — a tensor, a composite, the limits
//! a reader enforces — so they belong to the core rather than to any one driver, and are
//! available without the `tokio` feature (ADR-041 § 1).

use bytes::Bytes;
use hurray_core::TensorDescriptor;

/// Default maximum descriptor size: 16 MiB.
///
/// Limits memory allocation when reading untrusted streams.
pub const DEFAULT_MAX_DESCRIPTOR_BYTES: u64 = 16 * 1024 * 1024;

/// Default maximum composite nesting depth.
///
/// A composite member may itself be a composite (ADR-027 § Binding). This bounds the
/// recursion so a maliciously deep composite on an untrusted stream cannot exhaust the
/// stack. Matches the format's rank cap of 64 in spirit.
pub const DEFAULT_MAX_COMPOSITE_DEPTH: usize = 64;

/// The limits a reader enforces on an untrusted stream.
///
/// Held by [`StreamReaderCore`][super::StreamReaderCore] and by every driver over it, so
/// a C consumer and an event loop enforce the same ceilings.
///
/// # Examples
///
/// ```
/// use hurray_io::stream::{StreamReaderOptions, DEFAULT_MAX_COMPOSITE_DEPTH};
///
/// let options = StreamReaderOptions {
///     max_buffer_bytes: 1 << 30,
///     ..Default::default()
/// };
/// assert_eq!(options.max_composite_depth, DEFAULT_MAX_COMPOSITE_DEPTH);
/// ```
pub struct StreamReaderOptions {
    /// Maximum byte length of a single descriptor. Default: 16 MiB.
    pub max_descriptor_bytes: u64,
    /// Maximum byte length of a single buffer. Default: [`u64::MAX`] (unbounded).
    pub max_buffer_bytes: u64,
    /// Reject buffers whose `sync_mode` is not
    /// [`ProducerSynced`][hurray_core::SyncMode::ProducerSynced].
    ///
    /// Enable when the stream crosses machine boundaries.
    pub enforce_cross_machine_sync: bool,
    /// Maximum composite nesting depth. Default: [`DEFAULT_MAX_COMPOSITE_DEPTH`].
    pub max_composite_depth: usize,
}

impl Default for StreamReaderOptions {
    fn default() -> Self {
        Self {
            max_descriptor_bytes: DEFAULT_MAX_DESCRIPTOR_BYTES,
            max_buffer_bytes: u64::MAX,
            enforce_cross_machine_sync: false,
            max_composite_depth: DEFAULT_MAX_COMPOSITE_DEPTH,
        }
    }
}

/// A decoded tensor: its descriptor plus its buffers.
///
/// Each element in `buffers` corresponds to the [`hurray_core::BufferHandle`]
/// at the same index in `descriptor.buffers`, and carries the base address alignment that
/// handle declares (ADR-041 § 5).
///
/// # Examples
///
/// See [`StreamReaderCore`][super::StreamReaderCore] for a tensor read end to end.
#[derive(Debug)]
pub struct StreamTensor {
    /// The decoded tensor descriptor.
    pub descriptor: TensorDescriptor,
    /// Raw buffer bytes, one [`Bytes`] per buffer handle.
    pub buffers: Vec<Bytes>,
}

/// One decoded item from the stream: either a plain tensor or a composite.
///
/// A composite's members are themselves [`StreamItem`]s, so nested composites (ADR-027
/// § Binding) are represented as a tree.
///
/// # Examples
///
/// See [`StreamReaderCore`][super::StreamReaderCore].
#[derive(Debug)]
pub enum StreamItem {
    /// A single (non-composite) tensor.
    Tensor(StreamTensor),
    /// A composite: a data-less head plus its ordered members.
    Composite(StreamComposite),
}

impl StreamItem {
    /// The item's governing descriptor: the tensor's descriptor, or the composite head.
    ///
    /// # Examples
    ///
    /// See [`StreamReaderCore`][super::StreamReaderCore].
    pub fn descriptor(&self) -> &TensorDescriptor {
        match self {
            StreamItem::Tensor(t) => &t.descriptor,
            StreamItem::Composite(c) => &c.head,
        }
    }
}

/// A decoded composite tensor: its data-less head plus its ordered members.
///
/// Membership and ordering are exactly as they appeared on the wire (head precedes its
/// members; ADR-027 § Binding). The head, `member_count`, and per-rule constraints
/// (partition coverage, overlay ordering) have already been validated via
/// [`hurray_core::CompositeValidator`].
///
/// # Examples
///
/// See [`StreamReaderCore`][super::StreamReaderCore].
#[derive(Debug)]
pub struct StreamComposite {
    /// The composite head descriptor (owns no data buffers).
    pub head: TensorDescriptor,
    /// The members, in wire order. Each may itself be a composite (nesting).
    pub members: Vec<StreamItem>,
}

/// A node to emit as part of a composite: either a plain tensor (its descriptor plus one
/// byte-slice per buffer) or a nested composite (its head plus its own members).
///
/// The recursive [`Composite`][CompositeNode::Composite] arm lets a member be a composite
/// in its own right (ADR-027 § Binding). Payload slices stay borrowed — the writer never
/// copies or concatenates them (ADR-041 § 3).
///
/// # Examples
///
/// See [`StreamWriterCore::composite`][super::StreamWriterCore::composite].
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
    ///
    /// # Examples
    ///
    /// ```
    /// # use hurray_core::{
    /// #     BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode,
    /// #     TensorDescriptor, MIN_BUFFER_ALIGNMENT,
    /// # };
    /// use hurray_io::stream::CompositeNode;
    ///
    /// # let handle =
    /// #     BufferHandle::new(4, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced)?;
    /// # let descriptor = TensorDescriptor::new(
    /// #     1, 0, ElementType::Float32, Shape::new(vec![1u64])?, 0,
    /// #     LayoutDescriptor::RowMajor, vec![handle], None, None, None, None,
    /// # )?;
    /// let data = [0u8; 4];
    /// let buffers: [&[u8]; 1] = [&data];
    /// let node = CompositeNode::Tensor { descriptor: &descriptor, buffers: &buffers };
    /// assert_eq!(node.descriptor().shape.dims(), &[1]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn descriptor(&self) -> &TensorDescriptor {
        match self {
            CompositeNode::Tensor { descriptor, .. } => descriptor,
            CompositeNode::Composite { head, .. } => head,
        }
    }
}
