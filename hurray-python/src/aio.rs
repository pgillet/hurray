//! The private seam `hurray/aio.py` drives (ADR-041 § 9, ADR-042).
//!
//! Python's event loop is the driver. It awaits *readiness only* — `loop.add_reader` /
//! `add_writer` — and every transfer happens in one synchronous call here that moves the
//! bytes **and** reports the count to the core before returning. That is ADR-041 § 6a's
//! requirement, and it makes the § 6a case it names unreachable by construction: there is
//! no suspension point between a `read(2)` landing bytes in the core's destination and
//! `report_filled` telling the core they arrived.
//!
//! ## Nothing here is public API
//!
//! Every class is underscore-prefixed and excluded from `hurray.aio.__all__`. The
//! supported surface is `hurray.aio.StreamReader` / `StreamWriter`, written in Python
//! because the readiness logic belongs where the loop is.
//!
//! ## Why the GIL is not released
//!
//! The rest of this binding wraps every transfer in `py.detach`, because a blocking
//! `read` or `write` can hold the thread for an unbounded time. Here it cannot: the
//! descriptor is non-blocking and the loop has already said it is ready, so the syscall
//! returns immediately or returns `EAGAIN`. Releasing the GIL per syscall would cost more
//! than it saves — and `WriteOp` is not `Send`, so a detached closure could not hold it.
//!
//! ## Why a descriptor and not a transport object
//!
//! A `memoryview` into the core's destination must never reach Python: `step()` replaces
//! that destination when the buffer completes, so a view that outlived one step would be
//! a use-after-free. Keeping the transfer on this side of the boundary is what prevents
//! one from existing (ADR-042 § 2). Unix-only, the same boundary as issue #215.

use std::os::fd::{AsRawFd, OwnedFd, RawFd};

use pyo3::prelude::*;
use pyo3::types::PyModule;

use hurray_core::TensorDescriptor;
use hurray_io::stream::{
    CompositeAssembler, CompositeNode, ReadStep, StreamItem, StreamReaderCore, StreamReaderOptions,
    StreamWriterCore, WriteOp, WriteStep,
};
use hurray_io::Error;

use crate::composite::{owned_tree, OwnedNode, OwnedTree, SendSlice};
use crate::errors::StreamError;
use crate::stream::{dup_fd, stream_err, stream_item_to_py};
use crate::tensor::Tensor;

// ── The status codes the Python driver switches on ────────────────────────────
//
// Three outcomes, so three small integers, mirrored by named constants in `aio.py`.
// A `#[pyclass]` enum would be the typed answer and costs a class in the module's
// namespace — including in the repr-contract guard — for a value that never leaves the
// two files that define it.

/// The core wants bytes: await readiness, then call `fill` / `pump`.
const BLOCKED: u8 = 0;
/// An item is ready (`take_item`), or the submitted item is fully written.
const READY: u8 = 1;
/// Clean end of stream.
const FINISHED: u8 = 2;

/// Whether an I/O error means "not ready yet" rather than failure.
///
/// `Interrupted` counts: a syscall cut short by a signal moved no byte and reported no
/// count, so waiting for readiness again is the same answer as for `EAGAIN`. The blocking
/// drivers retry it in place instead, because they have a thread to retry on.
fn would_block(e: &std::io::Error) -> bool {
    matches!(
        e.kind(),
        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
    )
}

// ── Reader ────────────────────────────────────────────────────────────────────

/// The reader half of the seam: a `StreamReaderCore`, its assembler, and a descriptor.
#[pyclass(name = "_AsyncStreamReader", unsendable)]
pub struct AsyncStreamReader {
    core: StreamReaderCore,
    /// Held across calls like the blocking driver's: between completed items it is
    /// provably empty, since `step` reports `READY` only when a top-level item closes.
    assembler: CompositeAssembler,
    /// The item `step` completed, waiting for `take_item`.
    pending: Option<StreamItem>,
    fd: OwnedFd,
}

#[pymethods]
impl AsyncStreamReader {
    #[new]
    #[pyo3(signature = (
        fd,
        max_descriptor_bytes = None,
        max_buffer_bytes = None,
        max_composite_depth = None,
        cross_machine = false,
    ))]
    fn new(
        fd: RawFd,
        max_descriptor_bytes: Option<u64>,
        max_buffer_bytes: Option<u64>,
        max_composite_depth: Option<usize>,
        cross_machine: bool,
    ) -> PyResult<Self> {
        // Non-blocking mode is set by the Python driver through `os.set_blocking`, on
        // the descriptor `fileno` hands back — stdlib, and the loop's business.
        let owned = dup_fd(fd)?;
        // Unset means the core's own default, not a number copied into this binding —
        // the same resolution `hurray.StreamReader` does.
        let defaults = StreamReaderOptions::default();
        let options = StreamReaderOptions {
            max_descriptor_bytes: max_descriptor_bytes.unwrap_or(defaults.max_descriptor_bytes),
            max_buffer_bytes: max_buffer_bytes.unwrap_or(defaults.max_buffer_bytes),
            max_composite_depth: max_composite_depth.unwrap_or(defaults.max_composite_depth),
            enforce_cross_machine_sync: cross_machine,
        };
        let depth = options.max_composite_depth;
        Ok(Self {
            core: StreamReaderCore::with_options(options),
            assembler: CompositeAssembler::new(depth),
            pending: None,
            fd: owned,
        })
    }

    /// The descriptor to register with `loop.add_reader`.
    ///
    /// This object's own `dup`, not the caller's: readiness is awaited on exactly the
    /// descriptor [`fill`][Self::fill] reads from, so the two cannot drift apart.
    fn fileno(&self) -> RawFd {
        self.fd.as_raw_fd()
    }

    /// Drives the core as far as it goes without more bytes.
    ///
    /// Returns [`READY`] with an item waiting in [`take_item`][Self::take_item],
    /// [`BLOCKED`] when the core wants input, or [`FINISHED`] at a clean end of stream.
    fn step(&mut self) -> PyResult<u8> {
        loop {
            match self.core.step().map_err(stream_err)? {
                ReadStep::Input { .. } => return Ok(BLOCKED),
                ReadStep::Tensor => {
                    // `take_tensor` yields `Some` exactly when `step` reported `Tensor`.
                    let tensor = self.core.take_tensor().ok_or(Error::CoreFailed);
                    let tensor = tensor.map_err(stream_err)?;
                    if let Some(item) = self.assembler.push(tensor).map_err(stream_err)? {
                        self.pending = Some(item);
                        return Ok(READY);
                    }
                }
                ReadStep::Finished => {
                    self.assembler.finish().map_err(stream_err)?;
                    return Ok(FINISHED);
                }
            }
        }
    }

    /// Reads into the core's destination and reports the count, in one call.
    ///
    /// Returns `True` when bytes moved, `False` when the descriptor was not actually
    /// ready. `EAGAIN` MUST NOT be reported to the core as a count of zero: zero means
    /// end of input, so reporting it would turn a spurious wakeup into a truncated
    /// stream.
    fn fill(&mut self) -> PyResult<bool> {
        let n = match read_fd(self.fd.as_raw_fd(), self.core.input_dst()) {
            Ok(n) => n,
            Err(e) if would_block(&e) => return Ok(false),
            Err(e) => return Err(stream_err(Error::Io(e))),
        };
        // `n == 0` is end of input; whether that is clean or torn is the core's call.
        self.core.report_filled(n).map_err(stream_err)?;
        Ok(true)
    }

    /// The item the last [`READY`] step completed, as a `hurray.Tensor` or
    /// `hurray.Composite`.
    fn take_item(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self.pending.take() {
            Some(item) => stream_item_to_py(py, item),
            None => Err(StreamError::new_err("no item is ready")),
        }
    }
}

/// Borrows `fd` as a `File` for exactly one syscall.
///
/// `ManuallyDrop` because `File` closes its descriptor on drop and this one is owned by
/// the reader or writer object. The alternative is a `libc` dependency for `read`/`write`,
/// which buys nothing: `std`'s own `io::Error` already maps `EAGAIN` to
/// `ErrorKind::WouldBlock`, which is the distinction this module turns on.
fn borrow_fd(fd: RawFd) -> std::mem::ManuallyDrop<std::fs::File> {
    use std::os::fd::FromRawFd;
    // SAFETY: `fd` is valid for the duration of the call — it is owned by the object that
    // passed it in — and the `File` is never dropped, so it never closes it.
    std::mem::ManuallyDrop::new(unsafe { std::fs::File::from_raw_fd(fd) })
}

/// One `read(2)` into `dst`.
fn read_fd(fd: RawFd, dst: &mut [u8]) -> std::io::Result<usize> {
    if dst.is_empty() {
        return Ok(0);
    }
    std::io::Read::read(&mut *borrow_fd(fd), dst)
}

/// One `write(2)` from `src`.
fn write_fd(fd: RawFd, src: &[u8]) -> std::io::Result<usize> {
    if src.is_empty() {
        return Ok(0);
    }
    std::io::Write::write(&mut *borrow_fd(fd), src)
}

// ── The pinned payload: why this module has `unsafe` at all ───────────────────

/// Everything a submitted item's [`WriteOp`] borrows, at addresses that do not move.
///
/// A `WriteOp<'a>` borrows the core, the descriptors and the payload slices for as long as
/// it lives, and an async writer MUST hold it across a suspension: dropping it mid-item
/// sets the core's `torn` flag and poisons the writer (ADR-041 § 6b), and buffering the
/// item's wire bytes instead is the second payload copy § 2 exists to forbid. So one
/// object owns both the data and a borrow of it, which is a relationship no lifetime can
/// express — hence the erasure in [`Pinned::erase`], and hence this type.
///
/// Every field is boxed. Pushing to one of these `Vec`s may reallocate the `Vec`, but it
/// never moves a `Box`'s pointee, so a reference taken into a boxed value stays valid for
/// the life of this `Pinned` — including across the move that returns it from
/// [`Pinned::tensor`] or [`Pinned::composite`].
#[derive(Default)]
struct Pinned {
    /// Strong Python references to the objects that own the payload bytes. The bytes are
    /// not ours: they belong to a `hurray.Tensor`'s `BufferStore`, and these keep the
    /// tensors — and so the stores — alive while the op reads them (ADR-041 § 6).
    owners: Vec<Py<PyAny>>,
    // `Vec<Box<T>>`, not `Vec<T>`: a `Vec` moves its elements when it reallocates, which
    // would invalidate every erased reference already handed out. The box is what makes
    // the address stable, so clippy's advice is wrong here specifically.
    #[allow(clippy::vec_box)]
    descriptors: Vec<Box<TensorDescriptor>>,
    /// One per node: that node's `&[&[u8]]`, which `CompositeNode::Tensor` borrows.
    slices: Vec<Box<[&'static [u8]]>>,
    /// One per composite level: the members `CompositeNode::Composite` borrows.
    nodes: Vec<Box<[CompositeNode<'static>]>>,
}

impl Pinned {
    /// Erases a reference's lifetime to `'static`.
    ///
    /// # Safety
    ///
    /// The referent MUST be a boxed value this `Pinned` owns, and the caller MUST keep the
    /// `Pinned` alive — and never remove anything from it — for as long as the erased
    /// reference is reachable. [`AsyncStreamWriter`] is the only caller, and it satisfies
    /// both: nothing is ever removed, and its field order drops the op first.
    unsafe fn erase<T: ?Sized>(value: &T) -> &'static T {
        // SAFETY: forwarded from this function's own contract.
        unsafe { &*(value as *const T) }
    }

    /// Pins one descriptor and hands back an erased reference to it.
    fn descriptor(&mut self, descriptor: TensorDescriptor) -> &'static TensorDescriptor {
        let boxed = Box::new(descriptor);
        // SAFETY: the referent lives inside the box, so moving the box into the vector
        // below does not move it, and nothing is ever removed from the vector. See
        // `erase` for the rest of the contract.
        let erased = unsafe { Self::erase(&*boxed) };
        self.descriptors.push(boxed);
        erased
    }

    /// Pins one node's payload slices and hands back an erased reference to them.
    ///
    /// # Safety
    ///
    /// Each [`SendSlice`] MUST point at `len` live bytes owned by an object in
    /// [`owners`][Self::owners].
    unsafe fn payloads(&mut self, buffers: &[SendSlice]) -> &'static [&'static [u8]] {
        let boxed: Box<[&'static [u8]]> = buffers
            .iter()
            // SAFETY: forwarded from this function's own contract.
            .map(|s| unsafe { std::slice::from_raw_parts(s.0, s.1) })
            .collect();
        // SAFETY: as in `descriptor` — a boxed slice's contents do not move with the box.
        let erased = unsafe { Self::erase(&*boxed) };
        self.slices.push(boxed);
        erased
    }

    /// Pins one composite level and hands back an erased reference to its members.
    fn level(&mut self, nodes: Vec<CompositeNode<'static>>) -> &'static [CompositeNode<'static>] {
        let boxed = nodes.into_boxed_slice();
        // SAFETY: as in `descriptor`.
        let erased = unsafe { Self::erase(&*boxed) };
        self.nodes.push(boxed);
        erased
    }

    /// Flattens an owned composite tree into pinned nodes, children before parents.
    ///
    /// This is what `composite.rs`'s `with_stream_nodes` cannot be: that builder is
    /// continuation-passing because each node's slice vector lives in the stack frame that
    /// built it, which is safe and is the right shape for a *synchronous* write — and
    /// unusable here, where the tree has to outlive the frame that built it. The two are
    /// deliberately not merged: pointing the sync writer at this one would trade its safe
    /// builder for an erased-lifetime one to save duplication, which is the wrong way
    /// round.
    ///
    /// # Safety
    ///
    /// As [`payloads`][Self::payloads], for every tensor node in the tree.
    unsafe fn tree(&mut self, owned: &[OwnedNode]) -> &'static [CompositeNode<'static>] {
        let mut level = Vec::with_capacity(owned.len());
        for node in owned {
            level.push(match node {
                OwnedNode::Tensor {
                    descriptor,
                    buffers,
                } => CompositeNode::Tensor {
                    descriptor: self.descriptor(descriptor.clone()),
                    // SAFETY: forwarded from this function's own contract.
                    buffers: unsafe { self.payloads(buffers) },
                },
                OwnedNode::Composite { head, members } => CompositeNode::Composite {
                    head: self.descriptor(head.clone()),
                    // SAFETY: as above. Children are pinned before their parent, so the
                    // parent's reference is to boxes that already exist.
                    members: unsafe { self.tree(members) },
                },
            });
        }
        self.level(level)
    }
}

// ── Writer ────────────────────────────────────────────────────────────────────

/// The writer half of the seam: a core, one submitted item's pinned payload, and the
/// operation that borrows both.
///
/// **Field order is load-bearing.** Rust drops fields in declaration order, and `op`
/// borrows every field below it, so `op` must come first. Reordering this type is a
/// use-after-free, not a style change.
#[pyclass(name = "_AsyncStreamWriter", unsendable)]
pub struct AsyncStreamWriter {
    /// The item in flight. `None` between items.
    op: Option<WriteOp<'static>>,
    /// What the op borrows: descriptors, payload slices, and the Python objects that own
    /// the bytes. Cleared when an item completes or is abandoned.
    pinned: Pinned,
    /// Boxed so its address is stable while `op` holds a reference to it.
    core: Box<StreamWriterCore>,
    fd: OwnedFd,
}

#[pymethods]
impl AsyncStreamWriter {
    #[new]
    #[pyo3(signature = (fd, cross_machine))]
    fn new(fd: RawFd, cross_machine: bool) -> PyResult<Self> {
        let owned = dup_fd(fd)?;
        Ok(Self {
            op: None,
            pinned: Pinned::default(),
            core: Box::new(if cross_machine {
                StreamWriterCore::cross_machine()
            } else {
                StreamWriterCore::new()
            }),
            fd: owned,
        })
    }

    /// The descriptor to register with `loop.add_writer`. See [`AsyncStreamReader::fileno`].
    fn fileno(&self) -> RawFd {
        self.fd.as_raw_fd()
    }

    /// Submits one tensor. Nothing is written yet: [`pump`][Self::pump] does that.
    fn begin_tensor(&mut self, tensor: &Bound<'_, Tensor>) -> PyResult<()> {
        self.guard_idle()?;
        let (descriptor, buffers) = {
            let borrowed = tensor.borrow();
            let buffers: Vec<SendSlice> = borrowed
                .buffers()
                .map(|store| {
                    // SAFETY: the GIL is held, so the store and its base are alive.
                    let slice = unsafe { store.as_slice() };
                    SendSlice(slice.as_ptr(), slice.len())
                })
                .collect();
            (borrowed.descriptor.clone(), buffers)
        };
        // Pinned before the op exists, and kept until the item completes: the op reads
        // these bytes long after this call returns (ADR-041 § 3).
        self.pinned.owners.push(tensor.clone().into_any().unbind());
        let descriptor = self.pinned.descriptor(descriptor);
        // SAFETY: every slice points into a `BufferStore` of the tensor just pushed to
        // `owners`, which holds a strong reference for as long as the payload is pinned.
        let payloads = unsafe { self.pinned.payloads(&buffers) };
        self.submit(|core| core.tensor(descriptor, payloads))
    }

    /// Submits one composite — head, then members, as one item.
    fn begin_composite(
        &mut self,
        py: Python<'_>,
        composite: &Bound<'_, crate::composite::Composite>,
    ) -> PyResult<()> {
        self.guard_idle()?;
        let tree: OwnedTree = owned_tree(py, &composite.borrow());
        // One strong reference to the composite covers the whole tree: it owns its
        // members, which own their tensors, which own the stores the slices point into.
        self.pinned
            .owners
            .push(composite.clone().into_any().unbind());
        let head = self.pinned.descriptor(tree.head.clone());
        // SAFETY: every slice in the tree points into a store owned by the composite just
        // pushed to `owners`.
        let members = unsafe { self.pinned.tree(&tree.members) };
        self.submit(|core| core.composite(head, members))
    }

    /// Writes what it can of the submitted item and reports the count, in one call.
    ///
    /// Returns [`READY`] when the item is fully written, or [`BLOCKED`] when the sink is
    /// not ready — in which case the operation is still in flight and the caller must
    /// await writability and call again.
    ///
    /// The byte count is reported before this returns, which is what makes a cancelled
    /// `await` land only where no count can be lost (ADR-041 § 6a).
    fn pump(&mut self) -> PyResult<u8> {
        match self.pump_inner() {
            // Any error ends the item: the core keeps whatever state it earned — a lost
            // count, or `torn` from a byte already accepted — so the next submission
            // reports it. What must not survive is the operation itself, or a second
            // write would be refused as "already in flight" rather than for the real
            // reason.
            Err(e) => {
                self.release();
                Err(e)
            }
            ok => ok,
        }
    }

    /// Drops the item in flight, as a cancelled `await` must.
    ///
    /// Nothing else is needed, and nothing is reported: the core sets its own `torn` flag
    /// at an item's first accepted byte and clears it when the item completes, so after
    /// this call [`poisoned`][Self::poisoned] already answers ADR-041 § 6b's question.
    /// An item that emitted nothing leaves the core exactly as it was and the same tensor
    /// may be submitted again; one that emitted anything has put bytes on a wire that
    /// cannot be un-sent, and every later submission is refused.
    ///
    /// `WriteOp::emitted()` is public and would make the same decision here explicitly.
    /// Consulting it would be a second copy of a predicate the core already evaluates —
    /// two places to keep in step, for no new answer.
    fn abandon(&mut self) {
        self.release();
    }

    /// Whether the core has retired itself: a lost count, or a half-emitted item.
    ///
    /// The `op.is_none()` guard is load-bearing twice over. It keeps this method from
    /// touching `self.core` while an operation holds the only live reference to it — the
    /// invariant [`submit`][Self::submit] relies on, and `&&` short-circuits so the field
    /// is not merely unread but unevaluated. And it is the right answer anyway: while an
    /// item is in flight the core's `torn` flag means "this item has started", not "this
    /// writer is dead", and the operation itself clears it on completion. A core that was
    /// already unusable could not have accepted the submission in the first place.
    fn poisoned(&self) -> bool {
        self.op.is_none() && self.core.is_unusable()
    }
}

impl AsyncStreamWriter {
    /// [`pump`][Self::pump]'s body, so one place can clean up after any failure.
    fn pump_inner(&mut self) -> PyResult<u8> {
        let fd = self.fd.as_raw_fd();
        let Some(op) = self.op.as_mut() else {
            return Err(StreamError::new_err("no item has been submitted"));
        };
        loop {
            match op.step().map_err(stream_err)? {
                WriteStep::ItemWritten => {
                    self.release();
                    return Ok(READY);
                }
                WriteStep::Output { .. } => {
                    let n = match write_fd(fd, op.output_src()) {
                        Ok(n) => n,
                        Err(e) if would_block(&e) => return Ok(BLOCKED),
                        Err(e) => return Err(stream_err(Error::Io(e))),
                    };
                    if n == 0 {
                        // § 4: a zero-length write is never progress, and a closed sink is
                        // its own signal rather than an error to retry.
                        return Err(stream_err(op.report_sink_closed()));
                    }
                    op.report_written(n).map_err(stream_err)?;
                }
            }
        }
    }

    /// Rejects a second submission while one is in flight.
    fn guard_idle(&self) -> PyResult<()> {
        if self.op.is_some() {
            return Err(StreamError::new_err(
                "an item is already being written; await the current write before the next",
            ));
        }
        Ok(())
    }

    /// Runs one `StreamWriterCore` submission and stores the operation it returns.
    ///
    /// The closure takes the core as an erased `&'static mut` so the `WriteOp` it builds
    /// can be stored beside it. On failure the pins are released, so a rejected
    /// submission leaves the writer idle rather than holding a payload nobody will read.
    fn submit(
        &mut self,
        f: impl FnOnce(&'static mut StreamWriterCore) -> hurray_io::Result<WriteOp<'static>>,
    ) -> PyResult<()> {
        let ptr: *mut StreamWriterCore = &raw mut *self.core;
        // SAFETY: the core lives in a box this struct owns, so its address is stable for
        // the struct's lifetime and the reference is unaliased: while `op` is `Some`,
        // every method on this type reaches the core through the op and never through
        // `self.core`. `op` is declared first, so it is dropped before the box.
        let core: &'static mut StreamWriterCore = unsafe { &mut *ptr };
        match f(core) {
            Ok(op) => {
                self.op = Some(op);
                Ok(())
            }
            Err(e) => {
                self.release();
                Err(stream_err(e))
            }
        }
    }

    /// Ends the item in flight: the op goes first, then what it borrowed.
    fn release(&mut self) {
        self.op = None;
        self.pinned = Pinned::default();
    }
}

// ── Registration ──────────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AsyncStreamReader>()?;
    m.add_class::<AsyncStreamWriter>()?;
    m.add("_AIO_BLOCKED", BLOCKED)?;
    m.add("_AIO_READY", READY)?;
    m.add("_AIO_FINISHED", FINISHED)?;
    Ok(())
}
