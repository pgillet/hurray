//! The sequential write half of the sans-io stream core.
//!
//! Where the read core *owns* its destination, this one owns nothing: it hands the driver
//! the caller's own payload slices, one at a time, and never assembles them (ADR-041 § 3).

use hurray_core::{CompositeValidator, SyncMode, TensorDescriptor};

use super::Unusable;
use crate::stream::types::CompositeNode;
use crate::{Error, Result};

/// What the core has for its driver.
///
/// One request outstanding at a time, answered with
/// [`report_written`][WriteOp::report_written] (ADR-041 § 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteStep {
    /// Write [`output_src`][WriteOp::output_src] — `len` bytes — then report how many the
    /// sink accepted. Fewer is normal; `0` is never progress.
    Output {
        /// Bytes remaining for the current slice. Never `0`.
        len: usize,
    },
    /// Every byte of the submitted item is written. The borrowed payloads are released:
    /// the core holds no pointer into them from here on (ADR-041 § 3).
    ItemWritten,
}

/// Encodes the Hurray stream wire format into bytes a driver writes.
///
/// A plain tensor or a whole composite is submitted as one operation; the core emits the
/// descriptor, then each payload in turn, and says when the item is written. Validation
/// happens on submission, so a torn or invalid item never reaches a sink.
///
/// # Examples
///
/// ```
/// use hurray_core::{
///     BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode,
///     TensorDescriptor, MIN_BUFFER_ALIGNMENT,
/// };
/// use hurray_io::stream::{StreamWriterCore, WriteStep};
///
/// let handle =
///     BufferHandle::new(8, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced)?;
/// let descriptor = TensorDescriptor::new(
///     1, 0, ElementType::Float64, Shape::new(vec![1u64])?, 0,
///     LayoutDescriptor::RowMajor, vec![handle], None, None, None, None,
/// )?;
/// let payload = [1u8; 8];
/// // Bound, not a temporary: the operation borrows the payloads until the item is
/// // written, which is ADR-041 § 3's lifetime rule showing up in the type system.
/// let buffers: [&[u8]; 1] = [&payload];
///
/// let mut sink = Vec::new();
/// let mut core = StreamWriterCore::new();
/// let mut op = core.tensor(&descriptor, &buffers)?;
/// loop {
///     match op.step()? {
///         WriteStep::Output { .. } => {
///             // A sink that accepts 3 bytes at a time: a short write is normal.
///             let src = op.output_src();
///             let n = src.len().min(3);
///             sink.extend_from_slice(&src[..n]);
///             op.report_written(n)?;
///         }
///         WriteStep::ItemWritten => break,
///     }
/// }
/// assert_eq!(sink.len(), descriptor.encode()?.len() + payload.len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct StreamWriterCore {
    enforce_cross_machine_sync: bool,
    /// Set once the core may no longer be used; survives the [`WriteOp`] that set it.
    unusable: Option<Unusable>,
}

impl Default for StreamWriterCore {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamWriterCore {
    /// Creates a core that accepts any valid [`SyncMode`].
    ///
    /// # Examples
    ///
    /// See the type-level example.
    pub fn new() -> Self {
        Self {
            enforce_cross_machine_sync: false,
            unusable: None,
        }
    }

    /// Creates a core that rejects buffers whose `sync_mode` is not
    /// [`SyncMode::ProducerSynced`], for a stream that crosses machine boundaries.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_io::stream::StreamWriterCore;
    ///
    /// let core = StreamWriterCore::cross_machine();
    /// assert!(!core.is_unusable());
    /// ```
    pub fn cross_machine() -> Self {
        Self {
            enforce_cross_machine_sync: true,
            unusable: None,
        }
    }

    /// Submits one tensor: its descriptor followed by one payload slice per buffer.
    ///
    /// The payloads stay borrowed for the operation's lifetime and are released when it
    /// reports [`WriteStep::ItemWritten`] — which is also the point a binding may drop the
    /// reference pinning them (ADR-041 § 3).
    ///
    /// # Errors
    ///
    /// - [`Error::MultiBufferLengthMismatch`] — `buffers.len()` ≠ `descriptor.buffers.len()`
    /// - [`Error::BufferSizeMismatch`] — a slice's length ≠ its handle's `byte_size`
    /// - [`Error::InvalidCrossMachineSyncMode`] — cross-machine mode and a buffer that is
    ///   not `ProducerSynced`
    /// - [`Error::Core`] — the descriptor does not encode
    /// - [`Error::CoreUnusable`] — a previous write lost its byte count, so this core and
    ///   its sink are poisoned
    ///
    /// # Examples
    ///
    /// See the type-level example.
    pub fn tensor<'a>(
        &'a mut self,
        descriptor: &'a TensorDescriptor,
        buffers: &'a [&'a [u8]],
    ) -> Result<WriteOp<'a>> {
        if let Some(reason) = self.unusable {
            return Err(Error::CoreUnusable(reason));
        }
        self.validate_tensor(descriptor, buffers)?;
        let staging = descriptor.encode()?;
        Ok(WriteOp {
            core: self,
            staging,
            payloads: buffers,
            state: OpState::Descriptor,
            written: 0,
            stack: Vec::new(),
        })
    }

    /// Submits a composite: its head, then every member's descriptor and payloads, in
    /// order (ADR-027 § Binding).
    ///
    /// The whole tree is validated before a byte is emitted, at every nesting level.
    ///
    /// # Errors
    ///
    /// The errors of [`tensor`][Self::tensor] for each member, plus [`Error::Core`] when
    /// the head is not a valid composite head or the group fails validation (member count,
    /// partition coverage, overlay ordering).
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{
    ///     layout::{CompositeLayout, CompositionRule},
    ///     BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode,
    ///     TensorDescriptor, MIN_BUFFER_ALIGNMENT,
    /// };
    /// use hurray_io::stream::{CompositeNode, StreamWriterCore, WriteStep};
    ///
    /// let head = TensorDescriptor::new(
    ///     1, 0, ElementType::Float32, Shape::new(vec![1u64])?, 0,
    ///     LayoutDescriptor::Composite(CompositeLayout::new(CompositionRule::Group, 1)?),
    ///     vec![], None, None, None, None,
    /// )?;
    /// let handle =
    ///     BufferHandle::new(4, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced)?;
    /// let member = TensorDescriptor::new(
    ///     1, 0, ElementType::Float32, Shape::new(vec![1u64])?, 0,
    ///     LayoutDescriptor::RowMajor, vec![handle], None, None, None, None,
    /// )?;
    /// let data = [7u8; 4];
    /// let buffers: [&[u8]; 1] = [&data];
    /// let members = [CompositeNode::Tensor { descriptor: &member, buffers: &buffers }];
    ///
    /// let mut core = StreamWriterCore::new();
    /// let mut op = core.composite(&head, &members)?;
    /// let mut emitted = 0usize;
    /// while let WriteStep::Output { len } = op.step()? {
    ///     emitted += len;
    ///     op.report_written(len)?;
    /// }
    /// assert_eq!(emitted, head.encode()?.len() + member.encode()?.len() + data.len());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn composite<'a>(
        &'a mut self,
        head: &'a TensorDescriptor,
        members: &'a [CompositeNode<'a>],
    ) -> Result<WriteOp<'a>> {
        if let Some(reason) = self.unusable {
            return Err(Error::CoreUnusable(reason));
        }
        self.validate_composite(head, members)?;
        // The head owns no data buffers (composite-head invariant).
        let staging = head.encode()?;
        Ok(WriteOp {
            core: self,
            staging,
            payloads: &[],
            state: OpState::Descriptor,
            written: 0,
            stack: vec![Frame { members, next: 0 }],
        })
    }

    /// Whether this core — and the sink behind it — has been poisoned.
    ///
    /// # Examples
    ///
    /// See [`WriteOp::mark_unusable`].
    pub fn is_unusable(&self) -> bool {
        self.unusable.is_some()
    }

    /// Checks one tensor's payload slices against its buffer table, writing nothing.
    fn validate_tensor(&self, descriptor: &TensorDescriptor, buffers: &[&[u8]]) -> Result<()> {
        if buffers.len() != descriptor.buffers.len() {
            return Err(Error::MultiBufferLengthMismatch {
                declared: descriptor.buffers.len(),
                actual: buffers.len(),
            });
        }
        for (i, (handle, buf)) in descriptor.buffers.iter().zip(buffers).enumerate() {
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
        Ok(())
    }

    /// Validates a composite and every composite nested inside it, writing nothing.
    fn validate_composite(
        &self,
        head: &TensorDescriptor,
        members: &[CompositeNode<'_>],
    ) -> Result<()> {
        let mut validator = CompositeValidator::new(head)?;
        for member in members {
            validator.push_member(member.descriptor())?;
        }
        validator.finish()?;

        for member in members {
            match member {
                CompositeNode::Tensor {
                    descriptor,
                    buffers,
                } => self.validate_tensor(descriptor, buffers)?,
                CompositeNode::Composite { head, members } => {
                    self.validate_composite(head, members)?
                }
            }
        }
        Ok(())
    }
}

/// Composite members still to emit at one nesting level.
struct Frame<'a> {
    members: &'a [CompositeNode<'a>],
    next: usize,
}

/// What part of the current node is being emitted.
#[derive(Debug, Clone, Copy)]
enum OpState {
    /// Emitting the staged descriptor bytes.
    Descriptor,
    /// Emitting payload `index` of the current tensor.
    Payload { index: usize },
    /// Every byte of the item is out.
    Done,
    /// A previous call ended this operation.
    Failed,
}

/// One item being written: a tensor or a whole composite.
///
/// Borrowing both the core and the caller's payloads for its lifetime is what enforces
/// ADR-041 § 3's ownership rule in the type system — the core cannot hold a payload
/// pointer after the operation ends, because the operation is what holds them.
///
/// # Examples
///
/// See [`StreamWriterCore`].
pub struct WriteOp<'a> {
    core: &'a mut StreamWriterCore,
    /// Encoded descriptor bytes of the node being emitted.
    staging: Vec<u8>,
    /// Payload slices of the node being emitted, in buffer order.
    payloads: &'a [&'a [u8]],
    state: OpState,
    /// Bytes of the current slice already written.
    written: usize,
    /// Members awaiting emission, outermost first: ADR-041 § 10's explicit stack, in place
    /// of the recursion `write_composite_unchecked` used.
    stack: Vec<Frame<'a>>,
}

impl WriteOp<'_> {
    /// Advances the operation and returns what it needs next. Never blocks.
    ///
    /// # Errors
    ///
    /// - [`Error::Core`] — a member descriptor does not encode
    /// - [`Error::CoreFailed`] — a previous call ended this operation
    /// - [`Error::CoreUnusable`] — the core was poisoned mid-item
    ///
    /// # Examples
    ///
    /// See [`StreamWriterCore`].
    pub fn step(&mut self) -> Result<WriteStep> {
        if let Some(reason) = self.core.unusable {
            return Err(Error::CoreUnusable(reason));
        }
        loop {
            match self.state {
                OpState::Failed => return Err(Error::CoreFailed),
                OpState::Done => return Ok(WriteStep::ItemWritten),
                OpState::Descriptor => {
                    if self.written < self.staging.len() {
                        return Ok(WriteStep::Output {
                            len: self.staging.len() - self.written,
                        });
                    }
                    self.begin_payload(0)?;
                }
                OpState::Payload { index } => {
                    let remaining = self.remaining();
                    if remaining > 0 {
                        return Ok(WriteStep::Output { len: remaining });
                    }
                    self.begin_payload(index + 1)?;
                }
            }
        }
    }

    /// The bytes to write for the outstanding [`WriteStep::Output`] request.
    ///
    /// Either the staged descriptor or one of the caller's payload slices — never a copy
    /// of them, and never a concatenation (ADR-041 § 3).
    ///
    /// # Examples
    ///
    /// See [`StreamWriterCore`].
    pub fn output_src(&self) -> &[u8] {
        match self.state {
            OpState::Descriptor => &self.staging[self.written.min(self.staging.len())..],
            OpState::Payload { index } => match self.payloads.get(index) {
                Some(payload) => &payload[self.written.min(payload.len())..],
                None => &[],
            },
            _ => &[],
        }
    }

    /// Reports that the sink accepted `n` bytes of [`output_src`][Self::output_src].
    ///
    /// A short write re-offers the remainder.
    ///
    /// # Errors
    ///
    /// - [`Error::ZeroLengthWrite`] — `n == 0`, which § 4 forbids treating as progress. A
    ///   sink that cannot accept bytes is reported with
    ///   [`report_sink_closed`][Self::report_sink_closed] instead.
    /// - [`Error::InvalidProgressReport`] — more bytes reported than were offered. An
    ///   unknown number of payload bytes is then on the wire and cannot be un-sent, so
    ///   this poisons the core and its sink (ADR-041 § 6a).
    ///
    /// # Examples
    ///
    /// See [`StreamWriterCore`].
    pub fn report_written(&mut self, n: usize) -> Result<()> {
        if let Some(reason) = self.core.unusable {
            return Err(Error::CoreUnusable(reason));
        }
        if matches!(self.state, OpState::Failed) {
            return Err(Error::CoreFailed);
        }
        if n == 0 {
            // § 4: a zero-length write is not success. Position is still known, so the
            // operation stays usable — the driver may retry or declare the sink closed.
            return Err(Error::ZeroLengthWrite);
        }
        let offered = self.remaining();
        if n > offered {
            self.core.unusable = Some(Unusable::LostWriteCount);
            self.state = OpState::Failed;
            return Err(Error::InvalidProgressReport {
                reported: n,
                offered,
            });
        }
        self.written += n;
        Ok(())
    }

    /// Declares the sink closed while this item is in flight.
    ///
    /// Returns the error to propagate. A partially written item is already on the wire and
    /// a stream cannot be un-sent, so this poisons the core as well: a fresh writer over
    /// the same connection is not a recovery (ADR-041 § 6a).
    ///
    /// # Examples
    ///
    /// See [`mark_unusable`][Self::mark_unusable].
    pub fn report_sink_closed(&mut self) -> Error {
        self.core.unusable = Some(Unusable::LostWriteCount);
        self.state = OpState::Failed;
        Error::SinkClosed
    }

    /// Marks the core unusable: a write moved bytes whose count was lost.
    ///
    /// The read side's equivalent damages only the core's own position; here the bytes are
    /// already on the wire, so the sink is poisoned too (ADR-041 § 6a).
    ///
    /// # Examples
    ///
    /// ```
    /// # use hurray_core::{
    /// #     BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode,
    /// #     TensorDescriptor, MIN_BUFFER_ALIGNMENT,
    /// # };
    /// use hurray_io::stream::StreamWriterCore;
    ///
    /// # let handle =
    /// #     BufferHandle::new(4, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced)?;
    /// # let descriptor = TensorDescriptor::new(
    /// #     1, 0, ElementType::Float32, Shape::new(vec![1u64])?, 0,
    /// #     LayoutDescriptor::RowMajor, vec![handle], None, None, None, None,
    /// # )?;
    /// # let payload = [0u8; 4];
    /// # let buffers: [&[u8]; 1] = [&payload];
    /// let mut core = StreamWriterCore::new();
    /// let mut op = core.tensor(&descriptor, &buffers)?;
    /// op.mark_unusable();
    /// drop(op);
    ///
    /// // The poison outlives the operation: this sink is not reusable.
    /// assert!(core.is_unusable());
    /// assert!(core.tensor(&descriptor, &buffers).is_err());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn mark_unusable(&mut self) {
        self.core.unusable = Some(Unusable::LostWriteCount);
        self.state = OpState::Failed;
    }

    /// Bytes still to write for the current slice.
    fn remaining(&self) -> usize {
        match self.state {
            OpState::Descriptor => self.staging.len().saturating_sub(self.written),
            OpState::Payload { index } => match self.payloads.get(index) {
                Some(payload) => payload.len().saturating_sub(self.written),
                None => 0,
            },
            _ => 0,
        }
    }

    /// Moves to payload `index` of the current node, or past it to the next node.
    fn begin_payload(&mut self, index: usize) -> Result<()> {
        self.written = 0;
        if index < self.payloads.len() {
            self.state = OpState::Payload { index };
            return Ok(());
        }
        self.next_node()
    }

    /// Takes the next composite member off the stack, or finishes the item.
    fn next_node(&mut self) -> Result<()> {
        loop {
            let Some(frame) = self.stack.last_mut() else {
                self.state = OpState::Done;
                self.payloads = &[];
                return Ok(());
            };
            let Some(node) = frame.members.get(frame.next) else {
                self.stack.pop();
                continue;
            };
            frame.next += 1;

            match node {
                CompositeNode::Tensor {
                    descriptor,
                    buffers,
                } => {
                    self.staging = match descriptor.encode() {
                        Ok(staging) => staging,
                        Err(e) => {
                            self.state = OpState::Failed;
                            return Err(e.into());
                        }
                    };
                    self.payloads = buffers;
                }
                CompositeNode::Composite { head, members } => {
                    self.staging = match head.encode() {
                        Ok(staging) => staging,
                        Err(e) => {
                            self.state = OpState::Failed;
                            return Err(e.into());
                        }
                    };
                    self.payloads = &[];
                    self.stack.push(Frame { members, next: 0 });
                }
            }
            self.written = 0;
            self.state = OpState::Descriptor;
            return Ok(());
        }
    }
}
