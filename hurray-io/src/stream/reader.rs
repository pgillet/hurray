use bytes::Bytes;
use hurray_core::{CompositeValidator, LayoutDescriptor, SyncMode, TensorDescriptor};
use tokio::io::{AsyncRead, AsyncReadExt};

use crate::aligned::AlignedBuf;
use crate::stream::frame;
use crate::stream::types::{StreamComposite, StreamItem, StreamReaderOptions, StreamTensor};
use crate::{Error, Result};

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

            // Allocate at the *declared* alignment, not the allocator's default: the
            // consumer is entitled to an aligned SIMD load against it (#235).
            let mut dst = AlignedBuf::zeroed(byte_size, handle.alignment()).ok_or(
                Error::UnallocatableBuffer {
                    index: i,
                    byte_size,
                    alignment: handle.alignment(),
                },
            )?;
            self.inner
                .read_exact(dst.as_mut_slice())
                .await
                .map_err(frame::map_unexpected_eof)?;
            buffers.push(dst.freeze());
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
