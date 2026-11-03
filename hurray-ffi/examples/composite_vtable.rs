//! Demonstrates composites across the C surface: one node array written, one item read
//! back, on both the stream and the file (ADR-041 § 10).
//!
//! C has no sum type, so a composite crosses the boundary the way the format itself
//! carries it: a head followed by its members, in preorder, with every node declaring how
//! many of the nodes after it are its own. Writing one is an array; walking one is a loop
//! with a stack of counts — `walk` below is the whole algorithm, and it is the same shape
//! a C consumer writes.
//!
//! The one difference between the surfaces is names. A file gives every tensor its own
//! footer-index entry, head and member alike, so each node carries a name; a stream
//! carries no names at all, so each node's is null.
//!
//! Run with:
//!
//! ```text
//! cargo run --example composite_vtable -p hurray-ffi
//! ```

use std::ffi::c_void;

use hurray_core::{
    layout::{CompositeLayout, CompositionRule},
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode, TensorDescriptor,
    MIN_BUFFER_ALIGNMENT,
};
use hurray_ffi::buffer::{hurray_buffer_alignment, hurray_buffer_data_ptr};
use hurray_ffi::buffer_list::{hurray_buffer_list_get, hurray_buffer_list_len};
use hurray_ffi::descriptor::hurray_descriptor_destroy;
use hurray_ffi::{
    hurray_descriptor_decode, hurray_file_reader_destroy, hurray_file_reader_new,
    hurray_file_reader_read_composite, hurray_file_writer_destroy, hurray_file_writer_finish,
    hurray_file_writer_new, hurray_file_writer_write_composite, hurray_item_destroy,
    hurray_item_node, hurray_item_node_count, hurray_reader_destroy, hurray_reader_new,
    hurray_reader_next_item, hurray_tensor_buffers, hurray_writer_destroy, hurray_writer_new,
    hurray_writer_write_composite, HurrayBuffer, HurrayBufferList, HurrayCompositeNode,
    HurrayDescriptor, HurrayFileReader, HurrayFileWriter, HurrayIoSlice, HurrayItem,
    HurrayItemNode, HurrayReader, HurrayStatus, HurrayTransport, HurrayWriter,
    HURRAY_C_ABI_VERSION, HURRAY_ITEM_COMPOSITE, HURRAY_ITEM_TENSOR, HURRAY_OK,
};

// ── The caller's transport ────────────────────────────────────────────────────

/// Stands in for a socket or a file: a byte buffer, a sequential read cursor, and a
/// deliberately small transfer size so the short-transfer paths are the ones exercised.
struct Store {
    bytes: Vec<u8>,
    read_pos: usize,
    chunk: usize,
}

/// # Safety
///
/// `ctx` must point to a live `Store`, `dst` must be writable for `len` bytes, and
/// `out_filled` must be writable.
unsafe extern "C" fn store_read(
    ctx: *mut c_void,
    dst: *mut u8,
    len: usize,
    out_filled: *mut usize,
) -> HurrayStatus {
    let store = &mut *(ctx as *mut Store);
    let n = len.min(store.chunk).min(store.bytes.len() - store.read_pos);
    std::ptr::copy_nonoverlapping(store.bytes.as_ptr().add(store.read_pos), dst, n);
    store.read_pos += n;
    // Zero means end of input; fewer bytes than asked for does not.
    *out_filled = n;
    HURRAY_OK
}

/// # Safety
///
/// `ctx` must point to a live `Store`, `dst` must be writable for `len` bytes, and
/// `out_filled` must be writable.
unsafe extern "C" fn store_read_at(
    ctx: *mut c_void,
    offset: u64,
    dst: *mut u8,
    len: usize,
    out_filled: *mut usize,
) -> HurrayStatus {
    let store = &mut *(ctx as *mut Store);
    let start = (offset as usize).min(store.bytes.len());
    let n = len.min(store.chunk).min(store.bytes.len() - start);
    std::ptr::copy_nonoverlapping(store.bytes.as_ptr().add(start), dst, n);
    // The range is absolute, so a failed or short call costs the reader nothing: it simply
    // asks again.
    *out_filled = n;
    HURRAY_OK
}

/// # Safety
///
/// `ctx` must point to a live `Store` and `out_size` must be writable.
unsafe extern "C" fn store_size(ctx: *mut c_void, out_size: *mut u64) -> HurrayStatus {
    *out_size = (*(ctx as *mut Store)).bytes.len() as u64;
    HURRAY_OK
}

/// # Safety
///
/// `ctx` must point to a live `Store`, `src` must be readable for `len` bytes, and
/// `out_written` must be writable.
unsafe extern "C" fn store_write(
    ctx: *mut c_void,
    src: *const u8,
    len: usize,
    out_written: *mut usize,
) -> HurrayStatus {
    let store = &mut *(ctx as *mut Store);
    let n = len.min(store.chunk);
    store
        .bytes
        .extend_from_slice(std::slice::from_raw_parts(src, n));
    // Never report 0: a sink accepting nothing is a closed sink, not progress.
    *out_written = n;
    HURRAY_OK
}

fn vtable(store: &mut Store) -> HurrayTransport {
    HurrayTransport {
        ctx: store as *mut Store as *mut c_void,
        read: Some(store_read),
        read_at: Some(store_read_at),
        size: Some(store_size),
        write: Some(store_write),
        // Hurray never frees the context; `store` is owned by main, so there is nothing to
        // release. A caller whose transport is heap-allocated passes its free function here.
        release: None,
    }
}

// ── The composite ─────────────────────────────────────────────────────────────

/// A group head of `member_count` members. Group is the permissive composition rule: its
/// members may differ arbitrarily in shape and type.
fn group_head(member_count: u32) -> TensorDescriptor {
    TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(vec![1u64]).expect("a rank-1 shape"),
        0,
        LayoutDescriptor::Composite(
            CompositeLayout::new(CompositionRule::Group, member_count).expect("a group head"),
        ),
        // A composite head owns no data buffers: it describes a structure, not bytes.
        vec![],
        None,
        None,
        None,
        None,
    )
    .expect("a composite descriptor")
}

fn member(byte_size: u64) -> TensorDescriptor {
    let handle = BufferHandle::new(
        byte_size,
        MIN_BUFFER_ALIGNMENT,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )
    .expect("a 64-byte-aligned CPU buffer handle");
    TensorDescriptor::new(
        1,
        0,
        ElementType::Uint8,
        Shape::new(vec![byte_size]).expect("a rank-1 shape"),
        0,
        LayoutDescriptor::RowMajor,
        vec![handle],
        None,
        None,
        None,
        None,
    )
    .expect("a row-major descriptor")
}

/// A C caller names a descriptor by decoding its bytes.
fn descriptor_handle(descriptor: &TensorDescriptor) -> *mut HurrayDescriptor {
    let bytes = descriptor.encode().expect("a descriptor encodes");
    let mut handle: *mut HurrayDescriptor = std::ptr::null_mut();
    // SAFETY: `bytes` is readable for its own length and `handle` is a live local.
    let status = unsafe { hurray_descriptor_decode(bytes.as_ptr(), bytes.len(), &mut handle) };
    assert_eq!(status, HURRAY_OK, "decode failed with status {status}");
    handle
}

// ── Walking an item ───────────────────────────────────────────────────────────

/// Prints an item's tree, indented — the whole preorder walk, and the only algorithm a
/// consumer needs.
///
/// `remaining` is a stack of "members still owed" for each open head: a node is pushed
/// when it declares members, and heads are popped as they fill. The depth of that stack is
/// the indentation.
///
/// # Safety
///
/// `item` must point to a live `HurrayItem`.
unsafe fn walk(item: *mut HurrayItem) {
    let mut count: u64 = 0;
    assert_eq!(hurray_item_node_count(item, &mut count), HURRAY_OK);

    let mut remaining: Vec<u32> = Vec::new();
    for index in 0..count {
        let mut node = HurrayItemNode::default();
        assert_eq!(hurray_item_node(item, index, &mut node), HURRAY_OK);

        if let Some(owed) = remaining.last_mut() {
            *owed -= 1;
        }
        let depth = remaining.len();
        while matches!(remaining.last(), Some(0)) {
            remaining.pop();
        }
        if node.member_count > 0 {
            remaining.push(node.member_count);
        }

        let name = if node.name.is_null() {
            // A stream carries no names, so there is nothing to borrow.
            "(unnamed)".to_string()
        } else {
            String::from_utf8_lossy(std::slice::from_raw_parts(node.name, node.name_len))
                .into_owned()
        };

        let mut list: *mut HurrayBufferList = std::ptr::null_mut();
        assert_eq!(hurray_tensor_buffers(node.tensor, &mut list), HURRAY_OK);
        let mut buffers: u64 = 0;
        assert_eq!(hurray_buffer_list_len(list, &mut buffers), HURRAY_OK);

        let indent = "  ".repeat(depth);
        if node.kind == HURRAY_ITEM_COMPOSITE {
            println!(
                "{indent}- {name}: composite head, {} member(s)",
                node.member_count
            );
        } else {
            let mut buffer: *mut HurrayBuffer = std::ptr::null_mut();
            assert_eq!(hurray_buffer_list_get(list, 0, &mut buffer), HURRAY_OK);
            let mut data: *mut c_void = std::ptr::null_mut();
            let mut alignment: u32 = 0;
            assert_eq!(hurray_buffer_data_ptr(buffer, &mut data), HURRAY_OK);
            assert_eq!(hurray_buffer_alignment(buffer, &mut alignment), HURRAY_OK);
            // The transport wrote straight into this allocation, at the alignment the
            // buffer handle declares — so a SIMD load on it is sound.
            assert_eq!(data as usize % alignment as usize, 0);
            println!(
                "{indent}- {name}: tensor, {buffers} buffer(s), {alignment}-byte aligned at {data:p}"
            );
        }
    }
}

fn main() {
    println!("Hurray C ABI version {HURRAY_C_ABI_VERSION}\n");

    // group { inner { leaf }, tail } — one array, in preorder, head first.
    let head = group_head(2);
    let inner = group_head(1);
    let leaf = member(64);
    let tail = member(32);
    let leaf_data: Vec<u8> = (0u8..64).collect();
    let tail_data: Vec<u8> = vec![0xA5; 32];

    let handles = [
        descriptor_handle(&head),
        descriptor_handle(&inner),
        descriptor_handle(&leaf),
        descriptor_handle(&tail),
    ];
    let leaf_slice = [HurrayIoSlice {
        ptr: leaf_data.as_ptr(),
        len: leaf_data.len(),
    }];
    let tail_slice = [HurrayIoSlice {
        ptr: tail_data.as_ptr(),
        len: tail_data.len(),
    }];

    // Names are what the file surface needs and the stream surface ignores, so one array
    // serves both: `hurray_writer_write_composite` never looks at them.
    let names = ["group", "group.0", "group.0.0", "group.1"];
    let nodes = [
        HurrayCompositeNode {
            kind: HURRAY_ITEM_COMPOSITE,
            name: names[0].as_ptr(),
            name_len: names[0].len(),
            descriptor: handles[0],
            member_count: 2,
            ..HurrayCompositeNode::default()
        },
        HurrayCompositeNode {
            kind: HURRAY_ITEM_COMPOSITE,
            name: names[1].as_ptr(),
            name_len: names[1].len(),
            descriptor: handles[1],
            member_count: 1,
            ..HurrayCompositeNode::default()
        },
        HurrayCompositeNode {
            kind: HURRAY_ITEM_TENSOR,
            name: names[2].as_ptr(),
            name_len: names[2].len(),
            descriptor: handles[2],
            payloads: leaf_slice.as_ptr(),
            payload_count: leaf_slice.len(),
            member_count: 0,
        },
        HurrayCompositeNode {
            kind: HURRAY_ITEM_TENSOR,
            name: names[3].as_ptr(),
            name_len: names[3].len(),
            descriptor: handles[3],
            payloads: tail_slice.as_ptr(),
            payload_count: tail_slice.len(),
            member_count: 0,
        },
    ];

    // ── The stream: one item in, one item out ────────────────────────────────
    let mut wire = Store {
        bytes: Vec::new(),
        read_pos: 0,
        chunk: 24,
    };

    let transport = vtable(&mut wire);
    let mut writer: *mut HurrayWriter = std::ptr::null_mut();
    // SAFETY: the vtable and out-pointer are live locals; the write callback is set.
    let status = unsafe { hurray_writer_new(&transport, std::ptr::null(), &mut writer) };
    assert_eq!(status, HURRAY_OK, "hurray_writer_new: {status}");

    // SAFETY: every handle in `nodes` is live, and the payloads outlive this call — all
    // § 3's borrowed-payload rule asks of a blocking surface. The whole tree is validated
    // before a byte goes out.
    let status = unsafe { hurray_writer_write_composite(writer, nodes.as_ptr(), nodes.len()) };
    assert_eq!(status, HURRAY_OK, "write_composite: {status}");
    // SAFETY: the writer is live and destroyed exactly once.
    unsafe { hurray_writer_destroy(&mut writer) };
    println!("{} bytes on the wire\n", wire.bytes.len());

    let transport = vtable(&mut wire);
    let mut reader: *mut HurrayReader = std::ptr::null_mut();
    // SAFETY: the vtable and out-pointer are live locals; null options asks for the
    // documented defaults, including a composite nesting limit of 64.
    let status = unsafe { hurray_reader_new(&transport, std::ptr::null(), &mut reader) };
    assert_eq!(status, HURRAY_OK, "hurray_reader_new: {status}");

    println!("read back from the stream, as one item:");
    // SAFETY: the reader is live and `item` is a live local.
    unsafe {
        let mut item: *mut HurrayItem = std::ptr::null_mut();
        let status = hurray_reader_next_item(reader, &mut item);
        assert_eq!(status, HURRAY_OK, "next_item: {status}");
        walk(item);
        // One destroy for the whole subtree: the nodes' tensors were borrowed from it.
        hurray_item_destroy(&mut item);
        hurray_reader_destroy(&mut reader);
    }

    // ── The file: addressed by the head's name ───────────────────────────────
    let mut file = Store {
        bytes: Vec::new(),
        read_pos: 0,
        chunk: 4096,
    };

    let transport = vtable(&mut file);
    let mut file_writer: *mut HurrayFileWriter = std::ptr::null_mut();
    // SAFETY: the vtable and out-pointer are live locals; null options asks for 4096-byte
    // data alignment and an index in write order.
    let status = unsafe { hurray_file_writer_new(&transport, std::ptr::null(), &mut file_writer) };
    assert_eq!(status, HURRAY_OK, "hurray_file_writer_new: {status}");

    // SAFETY: as above — and here the names matter: every tensor of the group, head
    // included, gets its own footer-index entry.
    unsafe {
        let status = hurray_file_writer_write_composite(file_writer, nodes.as_ptr(), nodes.len());
        assert_eq!(status, HURRAY_OK, "file write_composite: {status}");
        let status = hurray_file_writer_finish(file_writer, std::ptr::null(), 0);
        assert_eq!(status, HURRAY_OK, "finish: {status}");
        hurray_file_writer_destroy(&mut file_writer);
    }
    println!("\n{} bytes in the file\n", file.bytes.len());

    let transport = vtable(&mut file);
    let mut file_reader: *mut HurrayFileReader = std::ptr::null_mut();
    // SAFETY: the vtable and out-pointer are live locals; `read_at` and `size` are set,
    // which opening a container requires — it is located from its own end.
    let status = unsafe { hurray_file_reader_new(&transport, &mut file_reader) };
    assert_eq!(status, HURRAY_OK, "hurray_file_reader_new: {status}");

    println!("read back from the file, by the head's name:");
    // SAFETY: the reader is live, the name is readable for its length, and `item` is a
    // live local.
    unsafe {
        let name = "group";
        let mut item: *mut HurrayItem = std::ptr::null_mut();
        let status =
            hurray_file_reader_read_composite(file_reader, name.as_ptr(), name.len(), &mut item);
        assert_eq!(status, HURRAY_OK, "read_composite: {status}");
        walk(item);
        hurray_item_destroy(&mut item);
        hurray_file_reader_destroy(&mut file_reader);
    }

    // SAFETY: each handle is live and destroyed exactly once.
    for handle in handles {
        unsafe { hurray_descriptor_destroy(handle) };
    }

    println!(
        "\nmembership was recovered on the stream from wire order, and in the file from the \
         index — the same tree either way, and no payload copied after the transport wrote it"
    );
}
