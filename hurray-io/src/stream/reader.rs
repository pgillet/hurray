use bytes::{Bytes, BytesMut};
use hurray_core::{SyncMode, TensorDescriptor};
use tokio::io::{AsyncRead, AsyncReadExt};

use crate::stream::frame;
use crate::{Error, Result};

/// Default maximum descriptor size: 16 MiB.
///
/// Limits memory allocation when reading untrusted streams.
pub const DEFAULT_MAX_DESCRIPTOR_BYTES: u64 = 16 * 1024 * 1024;

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
}

impl Default for StreamReaderOptions {
    fn default() -> Self {
        Self {
            max_descriptor_bytes: DEFAULT_MAX_DESCRIPTOR_BYTES,
            max_buffer_bytes: u64::MAX,
            enforce_cross_machine_sync: false,
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

        Ok(Some(StreamTensor {
            descriptor: desc,
            buffers,
        }))
    }

    /// Returns the underlying reader.
    pub fn into_inner(self) -> R {
        self.inner
    }
}
