use tokio::io::{AsyncRead, AsyncReadExt};

use crate::stream::sans_io::{CompositeAssembler, ReadStep, StreamReaderCore};
// `pub use`, not `use`: 0.1.0 published these at `stream::reader::*` as well as at
// `stream::*`, because this module is public and defined them. They belong to the core
// now, and the published paths keep resolving. `hurray_io::stream::*` is canonical.
pub use crate::stream::types::{
    StreamComposite, StreamItem, StreamReaderOptions, StreamTensor, DEFAULT_MAX_COMPOSITE_DEPTH,
    DEFAULT_MAX_DESCRIPTOR_BYTES,
};
use crate::{Error, Result};

/// Reads tensors from an async source in the Hurray streaming wire format.
///
/// Call [`next_tensor`][StreamReader::next_tensor] in a loop until it returns
/// `Ok(None)` (clean EOF).
///
/// This is a driver, not a decoder: [`StreamReaderCore`] holds the wire format and the
/// read position, and this type only moves bytes into the destination the core asks for
/// (ADR-041 § 7).
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
    core: StreamReaderCore,
    // Held across calls, not created per call: between successful `next_item` calls it is
    // provably empty (a call returns only once a *top-level* item closes), so this changes
    // nothing observable today — but a half-assembled tree is exactly what has to survive
    // a suspension when the readiness-driven driver arrives (ADR-041 § 6a).
    assembler: CompositeAssembler,
}

impl<R: AsyncRead + Unpin> StreamReader<R> {
    /// Creates a reader with default options.
    pub fn new(inner: R) -> Self {
        Self::with_options(inner, StreamReaderOptions::default())
    }

    /// Creates a reader that enforces `ProducerSynced` on every buffer.
    ///
    /// Use when the stream was produced by a remote machine and GPU/semaphore
    /// sync primitives are not meaningful locally.
    pub fn cross_machine(inner: R) -> Self {
        Self::with_options(
            inner,
            StreamReaderOptions {
                enforce_cross_machine_sync: true,
                ..Default::default()
            },
        )
    }

    /// Creates a reader with custom options.
    pub fn with_options(inner: R, options: StreamReaderOptions) -> Self {
        Self {
            inner,
            assembler: CompositeAssembler::new(options.max_composite_depth),
            core: StreamReaderCore::with_options(options),
        }
    }

    /// Reads the next tensor from the stream.
    ///
    /// Returns `Ok(None)` on a clean EOF (no bytes remaining before a descriptor starts).
    ///
    /// Oblivious to composition: a composite head comes back like any other tensor, with
    /// no buffers, followed by its members in wire order. Use
    /// [`next_item`][StreamReader::next_item] to have them assembled.
    ///
    /// # Errors
    ///
    /// - [`Error::UnexpectedEof`] — stream ended mid-descriptor or mid-buffer
    /// - [`Error::InvalidHeader`] — malformed descriptor prefix
    /// - [`Error::FrameTooLarge`] — descriptor or buffer exceeds configured limit
    /// - [`Error::InvalidCrossMachineSyncMode`] — cross-machine mode and a non-`ProducerSynced` buffer
    /// - [`Error::UnallocatableBuffer`] — a buffer's declared size and alignment cannot be allocated
    /// - [`Error::Core`] — descriptor decode failed
    /// - [`Error::Io`] — underlying read error
    pub async fn next_tensor(&mut self) -> Result<Option<StreamTensor>> {
        loop {
            match self.core.step()? {
                ReadStep::Input { .. } => self.fill().await?,
                // `take_tensor` yields `Some` exactly when `step` reported `Tensor`.
                ReadStep::Tensor => return Ok(Some(self.take_tensor()?)),
                ReadStep::Finished => return Ok(None),
            }
        }
    }

    /// Reads the next item, assembling composites (head + members) into a
    /// [`StreamItem::Composite`] and validating them.
    ///
    /// When the next descriptor on the wire is a composite head (`layout_tag = 0x0B`), this
    /// reads its declared `member_count` members — each of which may itself be a composite
    /// (nesting) — validates the group with
    /// [`CompositeValidator`][hurray_core::CompositeValidator] (member count, plus
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
        loop {
            match self.core.step()? {
                ReadStep::Input { .. } => self.fill().await?,
                ReadStep::Tensor => {
                    let tensor = self.take_tensor()?;
                    if let Some(item) = self.assembler.push(tensor)? {
                        return Ok(Some(item));
                    }
                }
                ReadStep::Finished => {
                    self.assembler.finish()?;
                    return Ok(None);
                }
            }
        }
    }

    /// Answers one [`ReadStep::Input`]: read into the core's destination, report the count.
    ///
    /// The transfer and the report are one step with nothing suspended between them, which
    /// is the shape ADR-041 § 6a requires of every driver.
    async fn fill(&mut self) -> Result<()> {
        let n = loop {
            // Disjoint field borrows: the destination comes from `core`, the bytes from
            // `inner`.
            match self.inner.read(self.core.input_dst()).await {
                Ok(n) => break n,
                // `read_exact`, which this replaces, absorbed this internally.
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                // A read that errors moved no bytes, so the count is known to be zero and
                // the core's position survives — no need to declare it lost (§ 6).
                Err(e) => return Err(Error::Io(e)),
            }
        };
        // `n == 0` is end of input: the core decides whether that is clean or torn.
        self.core.report_filled(n)
    }

    /// Collects the tensor the core just announced.
    fn take_tensor(&mut self) -> Result<StreamTensor> {
        self.core.take_tensor().ok_or(Error::CoreFailed)
    }

    /// Returns the underlying reader.
    pub fn into_inner(self) -> R {
        self.inner
    }
}
