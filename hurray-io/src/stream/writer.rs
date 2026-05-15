use hurray_core::{SyncMode, TensorDescriptor};
use tokio::io::{AsyncWrite, AsyncWriteExt};

use crate::{Error, Result};

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

    /// Flushes the underlying sink and returns it.
    pub async fn finish(mut self) -> Result<W> {
        self.inner.flush().await?;
        Ok(self.inner)
    }
}
