//! Reassembly of composite tensors from the flat tensors a stream carries.
//!
//! This is bookkeeping, not I/O. The wire carries a composite as its head followed by its
//! members, each of which may be a composite in turn (ADR-027 § Binding), and nothing about
//! putting that tree back together needs a byte to arrive: by the time a tensor reaches this
//! module it is already decoded. So it lives beside the byte-level state machine rather than
//! inside it — [`StreamReaderCore`][super::StreamReaderCore] stays a machine over bytes, and
//! every driver that wants structure layers this on top.
//!
//! That split is also what lets one reader offer both views of the same stream: flat tensors
//! in wire order, or assembled composites, from the same core.

use hurray_core::{CompositeValidator, LayoutDescriptor, TensorDescriptor};

use crate::stream::types::{StreamComposite, StreamItem, StreamTensor};
use crate::{Error, Result};

/// A composite head whose members have not all arrived.
struct Open {
    head: TensorDescriptor,
    member_count: u32,
    members: Vec<StreamItem>,
    validator: CompositeValidator,
}

/// Rebuilds composite tensors from a stream of flat tensors.
///
/// Feed every tensor the stream yields, in wire order, to [`push`][Self::push]. Plain
/// tensors come straight back out; a composite head opens a group that stays open until its
/// declared `member_count` members have been fed, at which point the whole composite is
/// returned as one item. Nesting is tracked with an explicit stack rather than recursion
/// (ADR-041 § 10), so arbitrarily deep composites cost stack space in the heap, not on the
/// call stack — and a half-assembled tree survives between calls.
///
/// Members are validated as they arrive, with the same [`CompositeValidator`] the writer
/// uses: member count, partition exact-cover, and overlay base-first ordering.
///
/// # Examples
///
/// ```
/// use hurray_core::{
///     layout::{CompositeLayout, CompositionRule},
///     BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode,
///     TensorDescriptor, MIN_BUFFER_ALIGNMENT,
/// };
/// use hurray_io::stream::{
///     CompositeAssembler, StreamItem, StreamTensor, DEFAULT_MAX_COMPOSITE_DEPTH,
/// };
///
/// // A group of one, and the tensor that fills it.
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
///
/// let mut assembler = CompositeAssembler::new(DEFAULT_MAX_COMPOSITE_DEPTH);
///
/// // The head opens the group: nothing comes out yet.
/// let head_tensor = StreamTensor { descriptor: head, buffers: vec![] };
/// assert!(assembler.push(head_tensor)?.is_none());
/// assert_eq!(assembler.depth(), 1);
///
/// // The member closes it, and the whole composite comes back as one item.
/// let member_tensor = StreamTensor {
///     descriptor: member,
///     buffers: vec![bytes::Bytes::from_static(&[0u8; 4])],
/// };
/// let item = assembler.push(member_tensor)?.expect("the group is complete");
/// let StreamItem::Composite(composite) = item else { panic!("expected a composite") };
/// assert_eq!(composite.members.len(), 1);
/// assert_eq!(assembler.depth(), 0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct CompositeAssembler {
    max_depth: usize,
    stack: Vec<Open>,
}

impl CompositeAssembler {
    /// Creates an assembler that refuses nesting deeper than `max_depth`.
    ///
    /// [`DEFAULT_MAX_COMPOSITE_DEPTH`][crate::stream::DEFAULT_MAX_COMPOSITE_DEPTH] is the
    /// limit the readers apply.
    ///
    /// # Examples
    ///
    /// See the type-level example.
    pub fn new(max_depth: usize) -> Self {
        Self {
            max_depth,
            stack: Vec::new(),
        }
    }

    /// Feeds the next tensor from the stream.
    ///
    /// Returns `Some` when a top-level item is complete — a plain tensor outside any
    /// composite, or the outermost composite once its last member arrives — and `None`
    /// while a composite is still being filled.
    ///
    /// # Errors
    ///
    /// - [`Error::CompositeNestingTooDeep`] — a head arrived at `max_depth`
    /// - [`Error::Core`] — the head is not a valid composite head, or the group failed
    ///   validation (member count, partition coverage, overlay ordering)
    ///
    /// # Examples
    ///
    /// See the type-level example.
    pub fn push(&mut self, tensor: StreamTensor) -> Result<Option<StreamItem>> {
        if let LayoutDescriptor::Composite(layout) = &tensor.descriptor.layout {
            let member_count = layout.member_count;
            // Depth is the number of heads already open, which is what the recursive
            // reader this replaces counted as its call depth.
            if self.stack.len() >= self.max_depth {
                return Err(Error::CompositeNestingTooDeep {
                    limit: self.max_depth,
                });
            }
            let validator = CompositeValidator::new(&tensor.descriptor)?;
            self.stack.push(Open {
                head: tensor.descriptor,
                member_count,
                members: Vec::new(),
                validator,
            });
            return Ok(None);
        }

        self.deliver(StreamItem::Tensor(tensor))
    }

    /// Reports end of input.
    ///
    /// # Errors
    ///
    /// [`Error::TornComposite`] — the stream ended with a composite still open. The
    /// innermost one is reported: that is the level whose members actually went missing.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_io::stream::{CompositeAssembler, DEFAULT_MAX_COMPOSITE_DEPTH};
    ///
    /// // Nothing open, so end of input is clean.
    /// let assembler = CompositeAssembler::new(DEFAULT_MAX_COMPOSITE_DEPTH);
    /// assert!(assembler.finish().is_ok());
    /// ```
    pub fn finish(&self) -> Result<()> {
        match self.stack.last() {
            None => Ok(()),
            Some(open) => Err(Error::TornComposite {
                declared: open.member_count,
                actual: open.members.len() as u32,
            }),
        }
    }

    /// How many composite heads are currently open.
    ///
    /// `0` between top-level items.
    ///
    /// # Examples
    ///
    /// See the type-level example.
    pub fn depth(&self) -> usize {
        self.stack.len()
    }

    /// Hands `item` to the innermost open head, closing heads — and their parents — as each
    /// one fills. Returns the item once no head is left to take it.
    fn deliver(&mut self, item: StreamItem) -> Result<Option<StreamItem>> {
        let mut item = item;
        loop {
            // Pop and push back rather than peek: the closing branch needs to own the
            // head, and `CompositeValidator::finish` consumes the validator.
            let Some(mut top) = self.stack.pop() else {
                return Ok(Some(item));
            };

            top.validator.push_member(item.descriptor())?;
            top.members.push(item);

            if (top.members.len() as u32) < top.member_count {
                self.stack.push(top);
                return Ok(None);
            }

            let Open {
                head,
                members,
                validator,
                ..
            } = top;
            validator.finish()?;
            // A closed composite is itself an item: a member of its parent, or the one the
            // caller gets. Loop rather than recurse (§ 10).
            item = StreamItem::Composite(StreamComposite { head, members });
        }
    }
}
