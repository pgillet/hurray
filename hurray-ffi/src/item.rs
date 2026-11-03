//! A read item: one tensor, or a composite and everything under it.
//!
//! `hurray-io` hands a driver a sum type — `StreamItem` and `FileItem` are each a tensor
//! or a composite whose members are items in turn — and C has no sum type to receive it
//! in. This module is the answer ADR-041 § Open Questions asked for: an item crosses as
//! the sequence the **format itself** uses, a head followed by its members, with each
//! node declaring how many of the nodes after it are its own
//! ([ADR-027](../../../docs/adr/ADR-027-composite-tensors.md) § Binding).
//!
//! So walking a composite from C is a loop over an array, and the nesting is in the
//! `member_count` column:
//!
//! ```text
//!   index  kind       name    member_count
//!   0      composite  "w"     2              <- the head, nodes[0]
//!   1      tensor     "w.0"   0
//!   2      composite  "w.1"   2              <- a nested head: its members follow it
//!   3      tensor     "w.1.0" 0
//!   4      tensor     "w.1.1" 0
//! ```
//!
//! Preorder rather than a tree of pointers, for a reason that outlives taste: a pointer
//! tree crossing an ABI can be cyclic, and a walker on either side loops forever on one.
//! An array with member counts cannot describe a cycle at all — every child sits after
//! its parent.
//!
//! Both surfaces produce this: [`hurray_reader_next_item`][crate::hurray_reader_next_item]
//! from the stream core's assembler, and
//! [`hurray_file_reader_read_composite`][crate::hurray_file_reader_read_composite] from
//! the file core's walk. A composite is the same shape whichever one carried it.

use hurray_core::TensorDescriptor;
use hurray_io::file::FileItem;
use hurray_io::stream::StreamItem;

use crate::{
    panic::{catch, null_check},
    status::{HurrayStatus, HURRAY_ERR_INDEX_OUT_OF_BOUNDS, HURRAY_ERR_INTERNAL, HURRAY_OK},
    stream::tensor::{hurray_tensor_destroy, tensor_handle, HurrayTensor},
};

// ── Node kinds ────────────────────────────────────────────────────────────────

/// A plain tensor: it owns buffers and no node after it belongs to it.
///
/// Unlike the `HURRAY_KV_*` tags, these two are **not** wire values: the wire has no
/// tensor-or-composite tag, it has a composite *layout* (tag `0x0B`). The kind is here so
/// that walking the array needs no descriptor decode, and so that a head with
/// `member_count == 0` — which the layout permits — is still distinguishable from a
/// tensor.
pub const HURRAY_ITEM_TENSOR: u8 = 0;

/// A composite head: it owns no buffers, and the next `member_count` sibling-level nodes
/// are its members.
pub const HURRAY_ITEM_COMPOSITE: u8 = 1;

// ── Nodes ─────────────────────────────────────────────────────────────────────

/// One node of an item, as a C caller sees it.
///
/// A flat tagged struct rather than a union, for the reason
/// [`HurrayKvValue`][crate::HurrayKvValue] is one: every binding generator downstream of
/// this header reads a struct correctly and few read a union well.
///
/// Every pointer in it **borrows** from the [`HurrayItem`] it came from and is valid
/// until [`hurray_item_destroy`]. In particular `tensor` MUST NOT be passed to
/// [`hurray_tensor_destroy`] — the item owns it.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HurrayItemNode {
    /// [`HURRAY_ITEM_TENSOR`] or [`HURRAY_ITEM_COMPOSITE`].
    pub kind: u8,
    /// First byte of the node's name, or null when it has none.
    ///
    /// A file names every tensor, heads and members alike, because each gets its own
    /// footer-index entry. A stream names none: the names are not on the wire, so there
    /// is nothing to borrow and this is null.
    pub name: *const u8,
    /// Byte length of `name`; `0` when `name` is null. UTF-8, not NUL-terminated.
    pub name_len: usize,
    /// The node's tensor handle, borrowed: its descriptor and its buffers.
    ///
    /// A composite head is the buffer-less tensor it is on the wire, so this is non-null
    /// for every node and its buffer list is empty for a head.
    pub tensor: *mut HurrayTensor,
    /// How many of the nodes following this one are its direct members. `0` for a tensor.
    pub member_count: u32,
}

impl Default for HurrayItemNode {
    /// A zeroed node, which is what a C caller gets from `{0}`.
    fn default() -> Self {
        Self {
            kind: HURRAY_ITEM_TENSOR,
            name: std::ptr::null(),
            name_len: 0,
            tensor: std::ptr::null_mut(),
            member_count: 0,
        }
    }
}

/// One node, with the parts the item owns.
struct OwnedNode {
    kind: u8,
    /// Owned, so the pointer handed to C stays valid for the item's life.
    name: Option<String>,
    /// Owned; destroyed by [`hurray_item_destroy`].
    tensor: *mut HurrayTensor,
    member_count: u32,
}

// ── HurrayItem ────────────────────────────────────────────────────────────────

/// Opaque handle to one read item: a tensor, or a composite and its whole subtree.
///
/// Not `#[repr(C)]`; the layout is an implementation detail. The handle **owns** every
/// node's tensor and name, and one [`hurray_item_destroy`] releases all of them.
///
/// A plain tensor is an item of exactly one node, so a caller that does not care about
/// composites can read `hurray_item_node(item, 0, &node)` and ignore the rest.
pub struct HurrayItem {
    /// Double-destroy sentinel (debug builds only).
    #[cfg(debug_assertions)]
    sentinel: u64,
    /// Preorder: a parent precedes its members, members are in wire order.
    nodes: Vec<OwnedNode>,
}

/// Sentinel placed in `HurrayItem.sentinel` on allocation (debug builds only).
#[cfg(debug_assertions)]
const SENTINEL_LIVE: u64 = 0x17E4_0000_17E4_0000;

/// Sentinel written before the allocation is freed, to detect a double destroy in debug
/// builds.
#[cfg(debug_assertions)]
const SENTINEL_DEAD: u64 = 0x17E4_DEAD_17E4_DEAD;

// SAFETY: mirrors the HurrayTensor contract — the C ABI is single-thread-at-a-time per
// handle, so Send only permits moving the box across threads during construction and
// destruction, never concurrent access.
unsafe impl Send for HurrayItem {}

// ── Accessors ─────────────────────────────────────────────────────────────────

/// The item's own kind: [`HURRAY_ITEM_TENSOR`] or [`HURRAY_ITEM_COMPOSITE`].
///
/// The kind of node `0`, which a caller that only wants to know whether a composite
/// arrived can ask for without walking.
///
/// # Safety
///
/// `handle` and `out_kind` MUST be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn hurray_item_kind(
    handle: *const HurrayItem,
    out_kind: *mut u8,
) -> HurrayStatus {
    catch(|| {
        null_check!(handle, out_kind);
        // SAFETY: handle is non-null and points to a live HurrayItem, which the C ABI's
        // one-handle-one-thread rule says nothing else is touching.
        let item = &*handle;
        // Every constructor pushes the root first, so an item always has a node.
        let Some(root) = item.nodes.first() else {
            return HURRAY_ERR_INTERNAL;
        };
        *out_kind = root.kind;
        HURRAY_OK
    })
}

/// How many nodes the item has in total, counting the root and every nested member.
///
/// `1` for a plain tensor.
///
/// # Safety
///
/// `handle` and `out_count` MUST be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn hurray_item_node_count(
    handle: *const HurrayItem,
    out_count: *mut u64,
) -> HurrayStatus {
    catch(|| {
        null_check!(handle, out_count);
        // SAFETY: handle is non-null and points to a live HurrayItem.
        let item = &*handle;
        *out_count = item.nodes.len() as u64;
        HURRAY_OK
    })
}

/// Copies node `index` into `out_node`.
///
/// Nodes are in preorder: node `0` is the root, and a node's `member_count` direct members
/// begin at the next index, each followed immediately by its own subtree. The pointers
/// inside the node borrow from the item and are valid until [`hurray_item_destroy`].
///
/// Returns [`HURRAY_ERR_INDEX_OUT_OF_BOUNDS`] if `index` is past the end.
///
/// # Safety
///
/// `handle` MUST point to a live [`HurrayItem`], and `out_node` MUST be a valid, non-null
/// pointer writable for one [`HurrayItemNode`].
#[no_mangle]
pub unsafe extern "C" fn hurray_item_node(
    handle: *const HurrayItem,
    index: u64,
    out_node: *mut HurrayItemNode,
) -> HurrayStatus {
    catch(|| {
        null_check!(handle, out_node);
        // SAFETY: handle is non-null and points to a live HurrayItem.
        let item = &*handle;
        let Some(node) = item.nodes.get(index as usize) else {
            return HURRAY_ERR_INDEX_OUT_OF_BOUNDS;
        };
        *out_node = HurrayItemNode {
            kind: node.kind,
            name: node.name.as_ref().map_or(std::ptr::null(), |n| n.as_ptr()),
            name_len: node.name.as_ref().map_or(0, |n| n.len()),
            tensor: node.tensor,
            member_count: node.member_count,
        };
        HURRAY_OK
    })
}

// ── Destructor ────────────────────────────────────────────────────────────────

/// Destroys an item, every tensor and name it owns, then writes null through `handle`.
///
/// One call for a whole composite subtree: the nodes' `tensor` pointers were borrowed and
/// are released here, so a caller that destroyed them individually would double-free.
///
/// # Safety
///
/// - `handle` MUST be a valid, non-null pointer to a `*mut HurrayItem` produced by
///   [`hurray_reader_next_item`][crate::hurray_reader_next_item] or
///   [`hurray_file_reader_read_composite`][crate::hurray_file_reader_read_composite].
/// - That item MUST NOT have been destroyed already.
#[no_mangle]
pub unsafe extern "C" fn hurray_item_destroy(handle: *mut *mut HurrayItem) -> HurrayStatus {
    catch(|| {
        null_check!(handle);
        let item = *handle;
        if item.is_null() {
            return HURRAY_ERR_INTERNAL;
        }

        #[cfg(debug_assertions)]
        {
            // SAFETY: item is non-null and points to a live HurrayItem. Read and mark
            // through the raw pointer before Box::from_raw, so the dead marker lands in
            // the heap slot a second destroy would inspect.
            if (*item).sentinel == SENTINEL_DEAD {
                return HURRAY_ERR_INTERNAL;
            }
            (*item).sentinel = SENTINEL_DEAD;
        }

        // SAFETY: produced by `item_handle` via Box::into_raw; the caller guarantees this
        // is the first and only destroy call.
        let mut boxed = Box::from_raw(item);
        for node in boxed.nodes.iter_mut() {
            let tensor = std::mem::replace(&mut node.tensor, std::ptr::null_mut());
            if !tensor.is_null() {
                // SAFETY: owned by this item and destroyed exactly once; nulled first so
                // a re-entrant destroy cannot free it twice.
                let mut tensor = tensor;
                hurray_tensor_destroy(&mut tensor);
            }
        }

        drop(boxed);
        *handle = std::ptr::null_mut();
        HURRAY_OK
    })
}

// ── Handing a decoded item to C ───────────────────────────────────────────────

/// Builds one node, taking ownership of a decoded tensor.
fn owned_node(
    kind: u8,
    name: Option<String>,
    descriptor: TensorDescriptor,
    buffers: Vec<bytes::Bytes>,
    member_count: u32,
) -> OwnedNode {
    OwnedNode {
        kind,
        name,
        tensor: Box::into_raw(tensor_handle(descriptor, buffers)),
        member_count,
    }
}

/// Flattens a decoded stream item into the preorder array a C caller walks.
///
/// Nameless: a stream carries no tensor names.
pub(crate) fn item_handle(item: StreamItem) -> Box<HurrayItem> {
    let mut nodes = Vec::new();
    // An explicit stack, not recursion (ADR-041 § 10) — a composite's depth is bounded by
    // the reader's own limit, but this traversal needs no stack frame per level to say so.
    // Members are pushed in reverse so the LIFO yields them in wire order.
    let mut stack = vec![item];
    while let Some(next) = stack.pop() {
        match next {
            StreamItem::Tensor(tensor) => nodes.push(owned_node(
                HURRAY_ITEM_TENSOR,
                None,
                tensor.descriptor,
                tensor.buffers,
                0,
            )),
            StreamItem::Composite(composite) => {
                // From the members present, not from the head's declared `member_count`:
                // this number describes the array that follows, and the assembler has
                // already refused any group where the two disagree.
                nodes.push(owned_node(
                    HURRAY_ITEM_COMPOSITE,
                    None,
                    composite.head,
                    Vec::new(),
                    composite.members.len() as u32,
                ));
                stack.extend(composite.members.into_iter().rev());
            }
        }
    }
    Box::new(HurrayItem {
        #[cfg(debug_assertions)]
        sentinel: SENTINEL_LIVE,
        nodes,
    })
}

/// Flattens a decoded file item into the preorder array a C caller walks.
///
/// Named: every tensor in a file has a footer-index entry, heads and members alike.
///
/// Deliberately not shared with [`item_handle`]: `StreamItem` and `FileItem` are distinct
/// types with no common trait upstream, and the names are not an optional field on one of
/// them but absent from the wire entirely. A local trait to join them would be more
/// surface than the fifteen lines it saves — step 4b's conclusion for the same pair of
/// write ops.
pub(crate) fn file_item_handle(item: FileItem) -> Box<HurrayItem> {
    let mut nodes = Vec::new();
    let mut stack = vec![item];
    while let Some(next) = stack.pop() {
        match next {
            FileItem::Tensor(tensor) => nodes.push(owned_node(
                HURRAY_ITEM_TENSOR,
                Some(tensor.name),
                tensor.descriptor,
                tensor.buffers,
                0,
            )),
            FileItem::Composite(composite) => {
                nodes.push(owned_node(
                    HURRAY_ITEM_COMPOSITE,
                    Some(composite.name),
                    composite.head,
                    Vec::new(),
                    composite.members.len() as u32,
                ));
                stack.extend(composite.members.into_iter().rev());
            }
        }
    }
    Box::new(HurrayItem {
        #[cfg(debug_assertions)]
        sentinel: SENTINEL_LIVE,
        nodes,
    })
}
