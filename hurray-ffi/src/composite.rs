//! Writing a composite from C: the node array, and the borrowed tree it becomes.
//!
//! The mirror of [`crate::item`]. A C caller describes a composite the way the format
//! does — a head, then its members, each node declaring how many of the nodes after it
//! are its own ([ADR-027](../../../docs/adr/ADR-027-composite-tensors.md) § Binding) — and
//! hands the whole thing over as one preorder array:
//!
//! ```text
//!   index  kind       name    member_count  payloads
//!   0      composite  "w"     2             -             <- the head, nodes[0]
//!   1      tensor     "w.0"   0             1 slice
//!   2      composite  "w.1"   2             -
//!   3      tensor     "w.1.0" 0             1 slice
//!   4      tensor     "w.1.1" 0             1 slice
//! ```
//!
//! One array rather than a tree of pointers, for the reason [`crate::item`] gives: a
//! pointer tree crossing an ABI can be cyclic, and preorder with member counts cannot
//! describe a cycle. It is also one allocation for the caller instead of one per level.
//!
//! `hurray-io` wants that as a tree of borrowed nodes — `CompositeNode<'a>` for a stream,
//! `FileCompositeNode<'a>` for a file — so this module turns one into the other, with an
//! explicit stack and no recursion (ADR-041 § 10), and with the payload slices left
//! borrowed from the caller the whole way (§ 3).

use hurray_core::TensorDescriptor;
use hurray_io::file::FileCompositeNode;
use hurray_io::stream::CompositeNode;

use crate::{
    descriptor::HurrayDescriptor,
    file::kv::{bytes_from_c, str_from_c},
    item::{HURRAY_ITEM_COMPOSITE, HURRAY_ITEM_TENSOR},
    status::{
        HurrayStatus, HURRAY_ERR_INDEX_OUT_OF_BOUNDS, HURRAY_ERR_INTERNAL,
        HURRAY_ERR_NOT_COMPOSITE, HURRAY_ERR_NULL_POINTER,
    },
    stream::HurrayIoSlice,
};

// ── Nodes ─────────────────────────────────────────────────────────────────────

/// One node of a composite to write: a tensor member, or a head whose members follow it.
///
/// A flat tagged struct rather than a union, for the reason
/// [`HurrayKvValue`][crate::HurrayKvValue] is one: every binding generator downstream of
/// this header reads a struct correctly and few read a union well. Fields the `kind` does
/// not select are ignored.
///
/// Every pointer borrows from the caller and MUST stay valid and unmodified until the
/// write call returns. Nothing here is copied — not the descriptors, and above all not
/// the payloads (ADR-041 § 3).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HurrayCompositeNode {
    /// [`HURRAY_ITEM_TENSOR`] or [`HURRAY_ITEM_COMPOSITE`].
    pub kind: u8,
    /// First byte of this node's tensor name, UTF-8 and not NUL-terminated.
    ///
    /// Required on the file surface, where every tensor — head and member alike — gets its
    /// own footer-index entry. Ignored on the stream surface, which carries no names.
    pub name: *const u8,
    /// Byte length of `name`.
    pub name_len: usize,
    /// The node's descriptor: a member's tensor descriptor, or a head's composite one.
    pub descriptor: *const HurrayDescriptor,
    /// One slice per buffer handle in `descriptor`'s buffer table, in that order.
    ///
    /// Ignored for a composite node: a head owns no data buffers, which its descriptor
    /// already states — and a head whose descriptor says otherwise is refused by the
    /// validator, not here.
    pub payloads: *const HurrayIoSlice,
    /// Number of slices at `payloads`.
    pub payload_count: usize,
    /// How many of the nodes following this one are its direct members. `0` for a tensor.
    pub member_count: u32,
}

impl Default for HurrayCompositeNode {
    /// A zeroed node, which is what a C caller gets from `{0}`.
    fn default() -> Self {
        Self {
            kind: HURRAY_ITEM_TENSOR,
            name: std::ptr::null(),
            name_len: 0,
            descriptor: std::ptr::null(),
            payloads: std::ptr::null(),
            payload_count: 0,
            member_count: 0,
        }
    }
}

// ── The structure of the array ────────────────────────────────────────────────

/// Resolves the preorder array into each node's list of direct members.
///
/// This is the whole structural check: the array MUST be exactly one tree. A head that
/// declares more members than the array carries, and nodes trailing after the root's
/// subtree is complete, are both [`HURRAY_ERR_INDEX_OUT_OF_BOUNDS`] — which is what each
/// of them is, a member count that does not fit the array it indexes.
///
/// A single forward pass with a stack of open heads, so a malformed array is rejected
/// before a descriptor is touched.
fn members_of(nodes: &[HurrayCompositeNode]) -> Result<Vec<Vec<usize>>, HurrayStatus> {
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    // Each entry is an open head and the number of members it is still owed. A head owed
    // zero is never on the stack, which is what makes the `-= 1` below safe without a
    // check: it is popped the instant it fills.
    let mut open: Vec<(usize, u32)> = Vec::new();

    for (index, node) in nodes.iter().enumerate() {
        if index > 0 {
            let Some(parent) = open.last_mut() else {
                // The root's subtree is complete and the array continues: whatever
                // follows belongs to no head.
                return Err(HURRAY_ERR_INDEX_OUT_OF_BOUNDS);
            };
            parent.1 -= 1;
            members[parent.0].push(index);
            while matches!(open.last(), Some((_, 0))) {
                open.pop();
            }
        }
        if node.kind == HURRAY_ITEM_COMPOSITE && node.member_count > 0 {
            open.push((index, node.member_count));
        }
    }

    if !open.is_empty() {
        // A head declared members the array does not carry.
        return Err(HURRAY_ERR_INDEX_OUT_OF_BOUNDS);
    }
    Ok(members)
}

// ── Stable storage for the tree ───────────────────────────────────────────────

/// Owns the member arrays and payload-slice arrays a borrowed tree points into.
///
/// The tree `hurray-io` wants is a graph of slices, and every slice needs an owner that
/// outlives the write call. This is that owner: one `Box` per array, so an array's address
/// survives later pushes into the `Vec` — the property ADR-042 § 4 wrote down for the
/// async writer's node arena, and the reason `Vec<Box<[T]>>` is not `Vec<Vec<T>>` here.
///
/// Nothing is ever removed or mutated after it is pushed.
struct Arena<T> {
    #[allow(clippy::vec_box)] // see above: the pointee's address must not move
    arrays: Vec<Box<[T]>>,
}

// Written out rather than derived: `#[derive(Default)]` would demand `T: Default`, and a
// borrowed composite node has no default.
impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self { arrays: Vec::new() }
    }
}

impl<T: 'static> Arena<T> {
    /// Stores one array and borrows it back for the rest of the call.
    ///
    /// The `'static` is the same documented lie [`bytes_from_c`] tells: the slice is valid
    /// for as long as this arena, and every caller drops the borrow before returning to C.
    fn store(&mut self, array: Vec<T>) -> &'static [T] {
        let boxed = array.into_boxed_slice();
        let ptr = boxed.as_ptr();
        let len = boxed.len();
        self.arrays.push(boxed);
        // SAFETY: `ptr` points into a heap allocation owned by a `Box` this arena now
        // holds. A `Box`'s pointee does not move when the `Vec` holding the box
        // reallocates, and nothing in this type ever mutates or removes an array, so the
        // slice stays valid until the arena is dropped — which the caller does only after
        // the write it built the tree for has returned.
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

/// The owners of one converted tree, kept alive for the duration of a write call.
pub(crate) struct TreeArena<T: 'static> {
    nodes: Arena<T>,
    payloads: Arena<&'static [u8]>,
}

impl<T: 'static> Default for TreeArena<T> {
    fn default() -> Self {
        Self {
            nodes: Arena::default(),
            payloads: Arena::default(),
        }
    }
}

// ── C → Rust: the borrowed tree ───────────────────────────────────────────────

/// Borrows a descriptor handle for the rest of the call.
///
/// # Safety
///
/// `ptr` MUST be null or point to a live [`HurrayDescriptor`] that outlives the call.
unsafe fn descriptor_from_c(
    ptr: *const HurrayDescriptor,
) -> Result<&'static TensorDescriptor, HurrayStatus> {
    if ptr.is_null() {
        return Err(HURRAY_ERR_NULL_POINTER);
    }
    // SAFETY: non-null and caller-guaranteed live. The `'static` is [`bytes_from_c`]'s
    // documented lie: the borrow ends before this call returns to C.
    Ok((*ptr).inner())
}

/// Borrows one node's payload slices, leaving the bytes where the caller put them.
///
/// # Safety
///
/// The node's `payloads` MUST describe `payload_count` readable [`HurrayIoSlice`] values,
/// each readable for its own `len`, unmodified for the duration of the call.
unsafe fn payloads_from_c(node: &HurrayCompositeNode) -> Result<Vec<&'static [u8]>, HurrayStatus> {
    if node.payload_count == 0 {
        return Ok(Vec::new());
    }
    if node.payloads.is_null() {
        return Err(HURRAY_ERR_NULL_POINTER);
    }
    // SAFETY: non-null and caller-guaranteed to describe `payload_count` readable slices.
    let slices = std::slice::from_raw_parts(node.payloads, node.payload_count);
    let mut borrowed = Vec::with_capacity(slices.len());
    for slice in slices {
        // SAFETY: caller guarantees `len` readable bytes at `ptr`; a zero length needs no
        // pointer.
        borrowed.push(bytes_from_c(slice.ptr, slice.len)?);
    }
    Ok(borrowed)
}

/// Converts the preorder array into the nameless tree a stream writer takes.
///
/// Returns the head's descriptor and its direct members, which is exactly what
/// `StreamWriterCore::composite` asks for. Nodes are built in reverse preorder — a node's
/// members always sit after it, so by the time a head is built its members already are —
/// which is how the nesting is rebuilt without recursion.
///
/// # Safety
///
/// Every pointer in `nodes` MUST satisfy the contract documented on
/// [`HurrayCompositeNode`] for the duration of the call.
pub(crate) unsafe fn stream_tree(
    arena: &mut TreeArena<CompositeNode<'static>>,
    nodes: &[HurrayCompositeNode],
) -> Result<(&'static TensorDescriptor, &'static [CompositeNode<'static>]), HurrayStatus> {
    let members_of_node = root_check(nodes)?;
    let mut built: Vec<Option<CompositeNode<'static>>> = (0..nodes.len()).map(|_| None).collect();

    for index in (0..nodes.len()).rev() {
        let node = &nodes[index];
        // SAFETY: delegated to this function's own contract.
        let descriptor = descriptor_from_c(node.descriptor)?;
        built[index] = Some(if node.kind == HURRAY_ITEM_COMPOSITE {
            CompositeNode::Composite {
                head: descriptor,
                members: arena
                    .nodes
                    .store(take_members(&mut built, &members_of_node[index])?),
            }
        } else {
            // SAFETY: delegated to this function's own contract.
            let payloads = payloads_from_c(node)?;
            CompositeNode::Tensor {
                descriptor,
                buffers: arena.payloads.store(payloads),
            }
        });
    }

    match built[0].take() {
        Some(CompositeNode::Composite { head, members }) => Ok((head, members)),
        // `root_check` already refused a non-composite root.
        _ => Err(HURRAY_ERR_INTERNAL),
    }
}

/// Converts the preorder array into the named tree a file writer takes.
///
/// Returns the head's name and descriptor plus its direct members, which is what
/// `flatten_composite` asks for.
///
/// Deliberately not shared with [`stream_tree`]: `CompositeNode` and `FileCompositeNode`
/// are distinct types with no common trait upstream, and only one of them has names. A
/// local trait to join them would be more surface than the thirty lines it saves — step
/// 4b's conclusion for the same pair of write ops.
///
/// # Safety
///
/// Every pointer in `nodes` MUST satisfy the contract documented on
/// [`HurrayCompositeNode`] for the duration of the call.
pub(crate) unsafe fn file_tree(
    arena: &mut TreeArena<FileCompositeNode<'static>>,
    nodes: &[HurrayCompositeNode],
) -> Result<
    (
        &'static str,
        &'static TensorDescriptor,
        &'static [FileCompositeNode<'static>],
    ),
    HurrayStatus,
> {
    let members_of_node = root_check(nodes)?;
    let mut built: Vec<Option<FileCompositeNode<'static>>> =
        (0..nodes.len()).map(|_| None).collect();

    for index in (0..nodes.len()).rev() {
        let node = &nodes[index];
        // SAFETY: delegated to this function's own contract.
        let descriptor = descriptor_from_c(node.descriptor)?;
        let name = str_from_c(node.name, node.name_len)?;
        built[index] = Some(if node.kind == HURRAY_ITEM_COMPOSITE {
            FileCompositeNode::Composite {
                name,
                head: descriptor,
                members: arena
                    .nodes
                    .store(take_members(&mut built, &members_of_node[index])?),
            }
        } else {
            // SAFETY: delegated to this function's own contract.
            let payloads = payloads_from_c(node)?;
            FileCompositeNode::Tensor {
                name,
                descriptor,
                buffers: arena.payloads.store(payloads),
            }
        });
    }

    match built[0].take() {
        Some(FileCompositeNode::Composite {
            name,
            head,
            members,
        }) => Ok((name, head, members)),
        // `root_check` already refused a non-composite root.
        _ => Err(HURRAY_ERR_INTERNAL),
    }
}

/// Validates the array's shape and that its root is a composite head.
fn root_check(nodes: &[HurrayCompositeNode]) -> Result<Vec<Vec<usize>>, HurrayStatus> {
    // A composite write with no head, or headed by a plain tensor, is the caller asking
    // for a composite where there is none — `hurray_writer_write_tensor` writes a tensor.
    match nodes.first() {
        Some(root) if root.kind == HURRAY_ITEM_COMPOSITE => members_of(nodes),
        _ => Err(HURRAY_ERR_NOT_COMPOSITE),
    }
}

/// Takes the already-built members at `indices` out of `built`, in order.
fn take_members<T>(built: &mut [Option<T>], indices: &[usize]) -> Result<Vec<T>, HurrayStatus> {
    let mut members = Vec::with_capacity(indices.len());
    for &index in indices {
        // Reverse preorder guarantees every member index is greater than its head's, so
        // each one is built and untaken: a `None` here would be this module's own bug.
        let Some(member) = built.get_mut(index).and_then(Option::take) else {
            return Err(HURRAY_ERR_INTERNAL);
        };
        members.push(member);
    }
    Ok(members)
}
