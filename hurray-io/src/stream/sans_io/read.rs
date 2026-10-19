//! The sequential read half of the sans-io stream core.
//!
//! One state machine, driven one step at a time: it asks for bytes, is told how many
//! arrived, and produces tensors. It never reads, never waits, and holds no transport.
//!
//! Tensors, not items: a composite head leaves here as the buffer-less tensor it is on the
//! wire, and [`CompositeAssembler`][super::CompositeAssembler] puts the tree back together.

use hurray_core::{BufferHandle, SyncMode, TensorDescriptor};

use super::dest::{Destination, DestinationAlloc, DestinationRequest, SystemDestinations};
use super::Unusable;
use crate::stream::types::{StreamReaderOptions, StreamTensor};
use crate::{Error, Result};

/// magic(4) + version(1) + flags(1) + descriptor_length(4)
const PREFIX_LEN: usize = 10;

/// What the core needs next from its driver.
///
/// Exactly one request is outstanding at a time, and the driver answers it — with
/// [`report_filled`][StreamReaderCore::report_filled] or
/// [`take_tensor`][StreamReaderCore::take_tensor] — before the core produces another
/// (ADR-041 § 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadStep {
    /// Fill up to `len` bytes of [`input_dst`][StreamReaderCore::input_dst], then report
    /// how many arrived. Filling fewer is normal; `0` means end of input.
    Input {
        /// Bytes remaining for the current target. Never `0`.
        len: usize,
    },
    /// A tensor is complete. Collect it with
    /// [`take_tensor`][StreamReaderCore::take_tensor].
    Tensor,
    /// End of input, reached cleanly at a tensor boundary.
    Finished,
}

/// The tensor currently being assembled.
struct Pending<D> {
    descriptor: TensorDescriptor,
    /// Buffers already filled and frozen, in declaration order.
    buffers: Vec<bytes::Bytes>,
    /// The destination for the buffer being filled.
    dest: Option<D>,
}

/// Where the machine is. `Failed` and `Unusable` are distinct by design: a format error
/// ends the stream with the position still known, while `Unusable` means a byte count was
/// lost and the position is unrecoverable (ADR-041 § 6).
#[derive(Debug, Clone, Copy)]
enum State {
    /// Filling the 10-byte frame prefix. End of input *here* is the clean one.
    Prefix,
    /// Filling the rest of the descriptor.
    Descriptor,
    /// Filling buffer `index` of the pending tensor.
    Buffer { index: usize },
    /// A tensor is waiting to be taken.
    TensorReady,
    /// End of input at a tensor boundary.
    Finished,
    /// A previous call returned an error; the stream cannot continue.
    Failed,
    /// Bytes moved and could not be accounted for.
    Unusable(Unusable),
}

/// Decodes the Hurray stream wire format from bytes a driver supplies.
///
/// The core owns the destination every payload byte lands in, so the transport's write into
/// it is the only copy (ADR-041 § 2), and that destination carries the alignment the buffer
/// handle *declares* (§ 5). Position is kept across every short read, so a driver that
/// stops — because its transport would block, or because its future was cancelled — can
/// resume on the next call (§ 6).
///
/// # Examples
///
/// Driving it over a slice, with no runtime anywhere:
///
/// ```
/// use hurray_core::{
///     BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode,
///     TensorDescriptor, MIN_BUFFER_ALIGNMENT,
/// };
/// use hurray_io::stream::{ReadStep, StreamReaderCore, StreamWriterCore, WriteStep};
///
/// // A 64-byte float32 tensor, encoded with the write core.
/// let handle =
///     BufferHandle::new(64, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced)?;
/// let descriptor = TensorDescriptor::new(
///     1, 0, ElementType::Float32, Shape::new(vec![4u64, 4])?, 0,
///     LayoutDescriptor::RowMajor, vec![handle], None, None, None, None,
/// )?;
/// let payload: Vec<u8> = (0u8..64).collect();
/// // Bound, not a temporary: the operation borrows the payloads until the item is
/// // written, which is ADR-041 § 3's lifetime rule showing up in the type system.
/// let buffers: [&[u8]; 1] = [&payload];
///
/// let mut wire = Vec::new();
/// let mut writer = StreamWriterCore::new();
/// let mut op = writer.tensor(&descriptor, &buffers)?;
/// while let WriteStep::Output { .. } = op.step()? {
///     let src = op.output_src();
///     wire.extend_from_slice(src);
///     let n = src.len();
///     op.report_written(n)?;
/// }
///
/// // Read it back, handing the core 7 bytes at a time to exercise short reads.
/// let mut core = StreamReaderCore::new();
/// let mut rest = &wire[..];
/// let mut tensors = Vec::new();
/// loop {
///     match core.step()? {
///         ReadStep::Input { len } => {
///             let n = len.min(7).min(rest.len());
///             core.input_dst()[..n].copy_from_slice(&rest[..n]);
///             rest = &rest[n..];
///             core.report_filled(n)?;
///         }
///         ReadStep::Tensor => tensors.extend(core.take_tensor()),
///         ReadStep::Finished => break,
///     }
/// }
///
/// assert_eq!(tensors.len(), 1);
/// let tensor = &tensors[0];
/// assert_eq!(tensor.buffers[0].as_ref(), &payload[..]);
/// // The destination carried the declared alignment, so a SIMD load is safe on it.
/// assert_eq!(tensor.buffers[0].as_ptr() as usize % 64, 0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct StreamReaderCore<A: DestinationAlloc = SystemDestinations> {
    options: StreamReaderOptions,
    destinations: A,
    state: State,
    /// Descriptor bytes, prefix included, for the frame being read.
    staging: Vec<u8>,
    /// Length of the target currently being filled: the prefix, the rest of the
    /// descriptor, or one buffer.
    target: usize,
    /// Bytes of that target already filled.
    filled: usize,
    pending: Option<Pending<A::Dest>>,
    tensor: Option<StreamTensor>,
}

impl Default for StreamReaderCore {
    fn default() -> Self {
        Self::new()
    }
}

// The default allocator's constructors: `StreamReaderCore` with no type argument.
impl StreamReaderCore<SystemDestinations> {
    /// Creates a core with default options and the system allocator.
    ///
    /// # Examples
    ///
    /// See the type-level example.
    pub fn new() -> Self {
        Self::with_options(StreamReaderOptions::default())
    }

    /// Creates a core with custom options.
    ///
    /// `max_descriptor_bytes`, `max_buffer_bytes` and `enforce_cross_machine_sync` apply
    /// here; `max_composite_depth` does not, because this core does not assemble
    /// composites — it bounds
    /// [`CompositeAssembler`][super::CompositeAssembler] instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_io::stream::{StreamReaderCore, StreamReaderOptions};
    ///
    /// let core = StreamReaderCore::with_options(StreamReaderOptions {
    ///     max_buffer_bytes: 1 << 20,
    ///     ..Default::default()
    /// });
    /// assert!(!core.is_unusable());
    /// ```
    pub fn with_options(options: StreamReaderOptions) -> Self {
        Self::with_destinations(options, SystemDestinations)
    }
}

impl<A: DestinationAlloc> StreamReaderCore<A> {
    /// Creates a core whose read destinations come from `destinations`.
    ///
    /// This is the seam ADR-041 left open: a binding may supply the allocation, so a read
    /// can land directly in memory its runtime owns, as long as the destination satisfies
    /// the alignment the handle declares.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_io::stream::{
    ///     AlignedDestination, DestinationAlloc, DestinationRequest, StreamReaderCore,
    ///     StreamReaderOptions, SystemDestinations,
    /// };
    ///
    /// /// Counts what the core asks for, then defers to the default allocator.
    /// struct Counting {
    ///     seen: Vec<(u64, u32)>,
    /// }
    ///
    /// impl DestinationAlloc for Counting {
    ///     type Dest = AlignedDestination;
    ///
    ///     fn allocate(
    ///         &mut self,
    ///         request: DestinationRequest,
    ///     ) -> hurray_io::Result<Self::Dest> {
    ///         self.seen.push((request.byte_size, request.alignment));
    ///         SystemDestinations.allocate(request)
    ///     }
    /// }
    ///
    /// let core = StreamReaderCore::with_destinations(
    ///     StreamReaderOptions::default(),
    ///     Counting { seen: Vec::new() },
    /// );
    /// assert!(!core.is_unusable());
    /// ```
    pub fn with_destinations(options: StreamReaderOptions, destinations: A) -> Self {
        Self {
            options,
            destinations,
            state: State::Prefix,
            staging: vec![0u8; PREFIX_LEN],
            target: PREFIX_LEN,
            filled: 0,
            pending: None,
            tensor: None,
        }
    }

    /// Advances the machine and returns what it needs next.
    ///
    /// Never blocks and never waits: every transition here is decode, allocate or
    /// bookkeeping, which is what lets one core serve a blocking driver and an event loop
    /// alike (ADR-041 § 4).
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidHeader`], [`Error::Core`] — the frame or descriptor is malformed
    /// - [`Error::FrameTooLarge`] — a descriptor or buffer exceeds the configured limit
    /// - [`Error::InvalidCrossMachineSyncMode`] — cross-machine mode and a buffer that is
    ///   not `ProducerSynced`
    /// - [`Error::UnallocatableBuffer`] — a destination the allocator cannot provide
    /// - [`Error::CoreFailed`], [`Error::CoreUnusable`] — a previous call ended this core
    ///
    /// # Examples
    ///
    /// See the type-level example.
    pub fn step(&mut self) -> Result<ReadStep> {
        loop {
            match self.state {
                State::Unusable(reason) => return Err(Error::CoreUnusable(reason)),
                State::Failed => return Err(Error::CoreFailed),
                State::Finished => return Ok(ReadStep::Finished),
                State::TensorReady => return Ok(ReadStep::Tensor),
                State::Prefix => match self.remaining() {
                    0 => self.begin_descriptor()?,
                    len => return Ok(ReadStep::Input { len }),
                },
                State::Descriptor => match self.remaining() {
                    0 => self.decode_descriptor()?,
                    len => return Ok(ReadStep::Input { len }),
                },
                State::Buffer { index } => match self.remaining() {
                    0 => self.finish_buffer(index)?,
                    len => return Ok(ReadStep::Input { len }),
                },
            }
        }
    }

    /// The destination for the outstanding [`ReadStep::Input`] request.
    ///
    /// Its address does not move while the request stands, so a driver may hold a pointer
    /// into it across a suspension (ADR-041 § 6a). Empty when no request is outstanding.
    ///
    /// # Examples
    ///
    /// See the type-level example.
    pub fn input_dst(&mut self) -> &mut [u8] {
        match self.state {
            State::Prefix | State::Descriptor => &mut self.staging[self.filled..self.target],
            State::Buffer { .. } => {
                let filled = self.filled;
                match self.pending.as_mut().and_then(|p| p.dest.as_mut()) {
                    Some(dest) => &mut dest.as_mut_slice()[filled..],
                    None => &mut [],
                }
            }
            _ => &mut [],
        }
    }

    /// Reports that `n` bytes were filled into [`input_dst`][Self::input_dst].
    ///
    /// Deliberately the cheapest call in the API: it records a count and nothing else, so
    /// a driver can pair the transfer and this report inside one synchronous callback,
    /// where no cancellation can interleave (ADR-041 § 6a).
    ///
    /// A short fill is normal — the core keeps its position and re-offers the remainder.
    /// `n == 0` means the transport reached end of input: clean at an item boundary, an
    /// error anywhere else.
    ///
    /// # Errors
    ///
    /// - [`Error::UnexpectedEof`] — end of input mid-item
    /// - [`Error::InvalidProgressReport`] — more bytes reported than were offered, which
    ///   also makes the core unusable: the position is lost
    ///
    /// # Examples
    ///
    /// See the type-level example.
    pub fn report_filled(&mut self, n: usize) -> Result<()> {
        match self.state {
            State::Unusable(reason) => return Err(Error::CoreUnusable(reason)),
            State::Failed => return Err(Error::CoreFailed),
            State::Finished | State::TensorReady => {
                return Err(Error::InvalidProgressReport {
                    reported: n,
                    offered: 0,
                })
            }
            _ => {}
        }

        if n == 0 {
            // § 4: only a zero-length read is end of input, and only at a tensor
            // boundary is it the clean one.
            if matches!(self.state, State::Prefix) && self.filled == 0 {
                self.state = State::Finished;
                return Ok(());
            }
            return Err(self.fail(Error::UnexpectedEof));
        }

        let offered = self.remaining();
        if n > offered {
            // Bytes moved that the core cannot place: its position no longer matches the
            // transport's, and no inspection recovers it (§ 6 — 0x00 is valid data).
            self.state = State::Unusable(Unusable::LostReadCount);
            return Err(Error::InvalidProgressReport {
                reported: n,
                offered,
            });
        }
        self.filled += n;
        Ok(())
    }

    /// Takes the completed tensor announced by [`ReadStep::Tensor`].
    ///
    /// Returns `None` when no tensor is ready. The tensor is handed over only once every
    /// one of its buffers is full, so a partially filled destination is never observable
    /// (ADR-041 § 6).
    ///
    /// A composite head arrives here like any other tensor, carrying no buffers; putting
    /// the tree back together is [`CompositeAssembler`][crate::stream::CompositeAssembler]'s
    /// job, not the core's.
    ///
    /// # Examples
    ///
    /// See the type-level example.
    pub fn take_tensor(&mut self) -> Option<StreamTensor> {
        if !matches!(self.state, State::TensorReady) {
            return None;
        }
        let tensor = self.tensor.take();
        self.reset_to_prefix();
        tensor
    }

    /// Marks the core unusable: a transport moved bytes whose count was lost.
    ///
    /// The position is unrecoverable, so every later call reports it by name rather than
    /// letting the core diverge silently from the transport (ADR-041 § 6). A readiness-
    /// driven driver never needs this; one built on a transfer-and-resolve primitive such
    /// as `sock_recv_into` does, for the window between the transfer and the report.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_io::stream::StreamReaderCore;
    ///
    /// let mut core = StreamReaderCore::new();
    /// core.mark_unusable();
    /// assert!(core.is_unusable());
    /// assert!(core.step().is_err());
    /// ```
    pub fn mark_unusable(&mut self) {
        self.state = State::Unusable(Unusable::LostReadCount);
    }

    /// Whether the core has been marked unusable.
    ///
    /// # Examples
    ///
    /// See [`mark_unusable`][Self::mark_unusable].
    pub fn is_unusable(&self) -> bool {
        matches!(self.state, State::Unusable(_))
    }

    // ── transitions ───────────────────────────────────────────────────────────

    /// Bytes still wanted for the current target.
    fn remaining(&self) -> usize {
        match self.state {
            State::Prefix | State::Descriptor | State::Buffer { .. } => {
                self.target.saturating_sub(self.filled)
            }
            _ => 0,
        }
    }

    /// Records a terminal failure and hands the error back for propagation.
    fn fail(&mut self, error: Error) -> Error {
        self.state = State::Failed;
        error
    }

    /// Prefix complete: size the staging buffer for the rest of the descriptor.
    fn begin_descriptor(&mut self) -> Result<()> {
        let descriptor_length = u32::from_le_bytes(
            self.staging[6..10]
                .try_into()
                .unwrap_or([0xFF, 0xFF, 0xFF, 0xFF]),
        ) as u64;

        if descriptor_length < PREFIX_LEN as u64 {
            let error = Error::InvalidHeader(format!(
                "descriptor_length {descriptor_length} is below minimum {PREFIX_LEN}"
            ));
            return Err(self.fail(error));
        }
        let limit = self.options.max_descriptor_bytes;
        let total = match usize::try_from(descriptor_length) {
            Ok(total) if descriptor_length <= limit => total,
            _ => {
                let error = Error::FrameTooLarge {
                    kind: "descriptor",
                    value: descriptor_length,
                    limit,
                };
                return Err(self.fail(error));
            }
        };

        self.staging.resize(total, 0);
        self.target = total;
        self.state = State::Descriptor;
        Ok(())
    }

    /// Descriptor complete: decode it and start filling its buffers.
    fn decode_descriptor(&mut self) -> Result<()> {
        let descriptor = match TensorDescriptor::decode(&self.staging) {
            Ok(descriptor) => descriptor,
            Err(e) => return Err(self.fail(e.into())),
        };

        // A composite head needs no special case here: it is the one descriptor that
        // declares no buffers, so it completes immediately and is handed over like any
        // other tensor.
        let buffer_count = descriptor.buffers.len();
        self.pending = Some(Pending {
            descriptor,
            buffers: Vec::with_capacity(buffer_count),
            dest: None,
        });
        if buffer_count == 0 {
            return self.complete_tensor();
        }
        self.begin_buffer(0)
    }

    /// Allocates the destination for buffer `index` and enters `Buffer { index }`.
    fn begin_buffer(&mut self, index: usize) -> Result<()> {
        let handle = match self.buffer_handle(index) {
            Some(handle) => handle,
            None => return Err(self.fail(Error::CoreFailed)),
        };

        if self.options.enforce_cross_machine_sync && handle.sync_mode() != SyncMode::ProducerSynced
        {
            let error = Error::InvalidCrossMachineSyncMode {
                index,
                actual: handle.sync_mode().to_byte(),
            };
            return Err(self.fail(error));
        }
        if handle.byte_size() > self.options.max_buffer_bytes {
            let error = Error::FrameTooLarge {
                kind: "buffer",
                value: handle.byte_size(),
                limit: self.options.max_buffer_bytes,
            };
            return Err(self.fail(error));
        }

        let request = DestinationRequest {
            index,
            byte_size: handle.byte_size(),
            alignment: handle.alignment(),
        };
        let len = match usize::try_from(request.byte_size) {
            Ok(len) => len,
            Err(_) => {
                let error = Error::UnallocatableBuffer {
                    index,
                    byte_size: request.byte_size,
                    alignment: request.alignment,
                };
                return Err(self.fail(error));
            }
        };
        let dest = match self.destinations.allocate(request) {
            Ok(dest) => dest,
            Err(e) => return Err(self.fail(e)),
        };

        match self.pending.as_mut() {
            Some(pending) => pending.dest = Some(dest),
            None => return Err(self.fail(Error::CoreFailed)),
        }
        self.target = len;
        self.filled = 0;
        self.state = State::Buffer { index };
        Ok(())
    }

    /// Buffer `index` is full: freeze it, then move to the next one or finish the tensor.
    fn finish_buffer(&mut self, index: usize) -> Result<()> {
        let (frozen, buffer_count) = match self.pending.as_mut() {
            Some(pending) => match pending.dest.take() {
                Some(dest) => (dest.freeze(), pending.descriptor.buffers.len()),
                None => return Err(self.fail(Error::CoreFailed)),
            },
            None => return Err(self.fail(Error::CoreFailed)),
        };
        match self.pending.as_mut() {
            Some(pending) => pending.buffers.push(frozen),
            None => return Err(self.fail(Error::CoreFailed)),
        }

        let next = index + 1;
        if next < buffer_count {
            self.begin_buffer(next)
        } else {
            self.complete_tensor()
        }
    }

    /// Hands the pending tensor to the caller.
    fn complete_tensor(&mut self) -> Result<()> {
        let pending = match self.pending.take() {
            Some(pending) => pending,
            None => return Err(self.fail(Error::CoreFailed)),
        };
        self.tensor = Some(StreamTensor {
            descriptor: pending.descriptor,
            buffers: pending.buffers,
        });
        self.state = State::TensorReady;
        Ok(())
    }

    /// Rewinds to the start of the next frame.
    fn reset_to_prefix(&mut self) {
        self.filled = 0;
        self.staging.clear();
        self.staging.resize(PREFIX_LEN, 0);
        self.target = PREFIX_LEN;
        self.state = State::Prefix;
    }

    fn buffer_handle(&self, index: usize) -> Option<BufferHandle> {
        self.pending
            .as_ref()?
            .descriptor
            .buffers
            .get(index)
            .copied()
    }
}
