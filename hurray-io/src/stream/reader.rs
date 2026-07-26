use bytes::{Bytes, BytesMut};
use hurray_core::{CompositeValidator, LayoutDescriptor, SyncMode, TensorDescriptor};
use tokio::io::{AsyncRead, AsyncReadExt};

use crate::stream::frame;
use crate::{Error, Result};

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

/// Options for [`StreamReader`].
pub struct StreamReaderOptions {
    /// Maximum byte length of a single descriptor. Default: 16 MiB.
    pub max_descriptor_bytes: u64,
    /// Maximum byte length of a single buffer. Default: [`u64::MAX`] (unbounded).
    pub max_buffer_bytes: u64,
    /// Reject buffers whose `sync_mode` is not [`SyncMode::ProducerSynced`].
    ///
    /// Enable when the stream crosses machine boundaries.
    pub enforce_cross_machine_sync: bool,
    /// Maximum composite nesting depth for [`next_item`][StreamReader::next_item].
    /// Default: [`DEFAULT_MAX_COMPOSITE_DEPTH`].
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

/// A decoded tensor: its descriptor plus zero-copy buffer views.
///
/// Each element in `buffers` corresponds to the [`hurray_core::BufferHandle`]
/// at the same index in `descriptor.buffers`.
#[derive(Debug)]
pub struct StreamTensor {
    /// The decoded tensor descriptor.
    pub descriptor: TensorDescriptor,
    /// Raw buffer bytes, one [`Bytes`] per buffer handle.
    pub buffers: Vec<Bytes>,
}

/// One decoded item from the stream: either a plain tensor or a composite.
///
/// Returned by [`next_item`][StreamReader::next_item]. A composite's members are
/// themselves [`StreamItem`]s, so nested composites (ADR-027 § Binding) are represented
/// as a tree.
#[derive(Debug)]
pub enum StreamItem {
    /// A single (non-composite) tensor.
    Tensor(StreamTensor),
    /// A composite: a data-less head plus its ordered members.
    Composite(StreamComposite),
}

impl StreamItem {
    /// The item's governing descriptor: the tensor's descriptor, or the composite head.
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
/// [`CompositeValidator`].
#[derive(Debug)]
pub struct StreamComposite {
    /// The composite head descriptor (owns no data buffers).
    pub head: TensorDescriptor,
    /// The members, in wire order. Each may itself be a composite (nesting).
    pub members: Vec<StreamItem>,
}

/// Reads tensors from an async source in the Hurray streaming wire format.
///
/// Call [`next_tensor`][StreamReader::next_tensor] in a loop until it returns
/// `Ok(None)` (clean EOF).
///
/// # Examples
///
/// ```no_run
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use hurray_io::stream::StreamReader;
///
/// let wire: &[u8] = &[]; // replace with actual data
/// let mut reader = StreamReader::new(wire);
/// while let Some(tensor) = reader.next_tensor().await? {
///     println!(
///         "tensor with {} buffer(s)",
///         tensor.buffers.len()
///     );
/// }
/// # Ok(())
/// # }
/// ```
pub struct StreamReader<R> {
    inner: R,
    options: StreamReaderOptions,
}

impl<R: AsyncRead + Unpin> StreamReader<R> {
    /// Creates a reader with default options.
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            options: StreamReaderOptions::default(),
        }
    }

    /// Creates a reader that enforces `ProducerSynced` on every buffer.
    ///
    /// Use when the stream was produced by a remote machine and GPU/semaphore
    /// sync primitives are not meaningful locally.
    pub fn cross_machine(inner: R) -> Self {
        Self {
            inner,
            options: StreamReaderOptions {
                enforce_cross_machine_sync: true,
                ..Default::default()
            },
        }
    }

    /// Creates a reader with custom options.
    pub fn with_options(inner: R, options: StreamReaderOptions) -> Self {
        Self { inner, options }
    }

    /// Reads the next tensor from the stream.
    ///
    /// Returns `Ok(None)` on a clean EOF (no bytes remaining before a descriptor starts).
    ///
    /// # Errors
    ///
    /// - [`Error::UnexpectedEof`] — stream ended mid-descriptor or mid-buffer
    /// - [`Error::InvalidHeader`] — malformed descriptor prefix
    /// - [`Error::FrameTooLarge`] — descriptor or buffer exceeds configured limit
    /// - [`Error::InvalidCrossMachineSyncMode`] — cross-machine mode and a non-`ProducerSynced` buffer
    /// - [`Error::Core`] — descriptor decode failed
    /// - [`Error::Io`] — underlying read error
    pub async fn next_tensor(&mut self) -> Result<Option<StreamTensor>> {
        let desc = match frame::read_descriptor(&mut self.inner, self.options.max_descriptor_bytes)
            .await?
        {
            Some(d) => d,
            None => return Ok(None),
        };

        let buffers = self.read_buffers(&desc).await?;

        Ok(Some(StreamTensor {
            descriptor: desc,
            buffers,
        }))
    }

    /// Reads the next item, assembling composites (head + members) into a
    /// [`StreamItem::Composite`] and validating them.
    ///
    /// When the next descriptor on the wire is a composite head (`layout_tag = 0x0B`), this
    /// reads its declared `member_count` members — each of which may itself be a composite
    /// (recursion) — validates the group with [`CompositeValidator`] (member count, plus
    /// partition coverage / overlay ordering), and returns them together. Otherwise it
    /// behaves like [`next_tensor`][StreamReader::next_tensor], returning a
    /// [`StreamItem::Tensor`]. Returns `Ok(None)` on a clean EOF.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use hurray_io::stream::{StreamItem, StreamReader};
    ///
    /// let wire: &[u8] = &[]; // replace with actual data
    /// let mut reader = StreamReader::new(wire);
    /// while let Some(item) = reader.next_item().await? {
    ///     match item {
    ///         StreamItem::Tensor(t) => println!("tensor, {} buffer(s)", t.buffers.len()),
    ///         StreamItem::Composite(c) => println!("composite, {} member(s)", c.members.len()),
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// In addition to the errors of [`next_tensor`][StreamReader::next_tensor]:
    ///
    /// - [`Error::TornComposite`] — the stream ended before the head's `member_count`
    ///   members were read
    /// - [`Error::CompositeNestingTooDeep`] — nesting exceeded `max_composite_depth`
    /// - [`Error::Core`] — composite validation failed (e.g. partition does not cover the
    ///   index space, overlay ordering, member-count mismatch)
    pub async fn next_item(&mut self) -> Result<Option<StreamItem>> {
        self.read_item(0).await
    }

    /// Reads and freezes every buffer declared by `desc`, enforcing the buffer-size limit
    /// and (in cross-machine mode) the sync-mode requirement.
    async fn read_buffers(&mut self, desc: &TensorDescriptor) -> Result<Vec<Bytes>> {
        let mut buffers = Vec::with_capacity(desc.buffers.len());

        for (i, handle) in desc.buffers.iter().enumerate() {
            if self.options.enforce_cross_machine_sync
                && handle.sync_mode() != SyncMode::ProducerSynced
            {
                return Err(Error::InvalidCrossMachineSyncMode {
                    index: i,
                    actual: handle.sync_mode().to_byte(),
                });
            }

            let byte_size = handle.byte_size();

            if byte_size > self.options.max_buffer_bytes {
                return Err(Error::FrameTooLarge {
                    kind: "buffer",
                    value: byte_size,
                    limit: self.options.max_buffer_bytes,
                });
            }

            // One allocation per buffer; resize then freeze gives a Bytes with no extra copy.
            let mut buf = BytesMut::with_capacity(byte_size as usize);
            buf.resize(byte_size as usize, 0);
            self.inner
                .read_exact(&mut buf)
                .await
                .map_err(frame::map_unexpected_eof)?;
            buffers.push(buf.freeze());
        }

        Ok(buffers)
    }

    /// Reads one item at composite-nesting `depth`, recursing into members.
    async fn read_item(&mut self, depth: usize) -> Result<Option<StreamItem>> {
        let desc = match frame::read_descriptor(&mut self.inner, self.options.max_descriptor_bytes)
            .await?
        {
            Some(d) => d,
            None => return Ok(None),
        };

        let member_count = match &desc.layout {
            LayoutDescriptor::Composite(c) => c.member_count,
            // Not a composite head: read its buffers and return a plain tensor.
            _ => {
                let buffers = self.read_buffers(&desc).await?;
                return Ok(Some(StreamItem::Tensor(StreamTensor {
                    descriptor: desc,
                    buffers,
                })));
            }
        };

        if depth >= self.options.max_composite_depth {
            return Err(Error::CompositeNestingTooDeep {
                limit: self.options.max_composite_depth,
            });
        }

        // Validate the group as its members arrive, reusing the core validator: member
        // count, and per-rule constraints (partition coverage, overlay base-first ordering).
        let mut validator = CompositeValidator::new(&desc)?;
        let mut members = Vec::with_capacity(member_count as usize);
        for i in 0..member_count {
            // Box the recursive call: an async fn cannot name its own future inline.
            let item = match Box::pin(self.read_item(depth + 1)).await? {
                Some(item) => item,
                None => {
                    return Err(Error::TornComposite {
                        declared: member_count,
                        actual: i,
                    })
                }
            };
            validator.push_member(item.descriptor())?;
            members.push(item);
        }
        validator.finish()?;

        Ok(Some(StreamItem::Composite(StreamComposite {
            head: desc,
            members,
        })))
    }

    /// Returns the underlying reader.
    pub fn into_inner(self) -> R {
        self.inner
    }
}
