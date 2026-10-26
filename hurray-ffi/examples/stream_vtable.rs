//! Demonstrates the C stream surface: a reader and a writer over a transport vtable
//! (ADR-041 § 8).
//!
//! A C consumer never hands Hurray a path or a file descriptor. It hands over a context
//! pointer and two callbacks, and Hurray asks them for bytes. Nothing async is linked:
//! the decoding is a state machine, and all the blocking happens inside the caller's own
//! callbacks — here, a `Vec<u8>` standing in for a socket.
//!
//! Run with:
//!
//! ```text
//! cargo run --example stream_vtable -p hurray-ffi
//! ```

use std::ffi::c_void;

use hurray_core::{
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode, TensorDescriptor,
    MIN_BUFFER_ALIGNMENT,
};
use hurray_ffi::buffer::{
    hurray_buffer_alignment, hurray_buffer_byte_size, hurray_buffer_data_ptr,
};
use hurray_ffi::buffer_list::{hurray_buffer_list_get, hurray_buffer_list_len};
use hurray_ffi::descriptor::{hurray_descriptor_destroy, hurray_descriptor_rank};
use hurray_ffi::{
    hurray_descriptor_decode, hurray_reader_destroy, hurray_reader_new, hurray_reader_next_tensor,
    hurray_tensor_buffers, hurray_tensor_descriptor, hurray_tensor_destroy, hurray_writer_destroy,
    hurray_writer_new, hurray_writer_write_tensor, HurrayBuffer, HurrayBufferList,
    HurrayDescriptor, HurrayIoSlice, HurrayReader, HurrayStatus, HurrayTensor, HurrayTransport,
    HurrayWriter, HURRAY_C_ABI_VERSION, HURRAY_END_OF_STREAM, HURRAY_OK,
};

// ── The caller's transport ────────────────────────────────────────────────────

/// Stands in for a socket: a byte buffer, a read cursor, and a deliberately small
/// transfer size so the short-read and short-write paths are the ones exercised.
struct Pipe {
    bytes: Vec<u8>,
    read_pos: usize,
    /// Bytes moved per callback. A real socket's limit is the kernel's; a short transfer
    /// is normal and is not an error.
    chunk: usize,
}

/// # Safety
///
/// `ctx` must point to a live `Pipe`, `dst` must be writable for `len` bytes, and
/// `out_filled` must be writable.
unsafe extern "C" fn pipe_read(
    ctx: *mut c_void,
    dst: *mut u8,
    len: usize,
    out_filled: *mut usize,
) -> HurrayStatus {
    let pipe = &mut *(ctx as *mut Pipe);
    let n = len.min(pipe.chunk).min(pipe.bytes.len() - pipe.read_pos);
    std::ptr::copy_nonoverlapping(pipe.bytes.as_ptr().add(pipe.read_pos), dst, n);
    pipe.read_pos += n;
    // Writing 0 means end of input. Writing fewer bytes than asked for does not.
    *out_filled = n;
    HURRAY_OK
}

/// # Safety
///
/// `ctx` must point to a live `Pipe`, `src` must be readable for `len` bytes, and
/// `out_written` must be writable.
unsafe extern "C" fn pipe_write(
    ctx: *mut c_void,
    src: *const u8,
    len: usize,
    out_written: *mut usize,
) -> HurrayStatus {
    let pipe = &mut *(ctx as *mut Pipe);
    let n = len.min(pipe.chunk);
    pipe.bytes
        .extend_from_slice(std::slice::from_raw_parts(src, n));
    // Never report 0 here: a sink accepting nothing is a closed sink, not progress.
    *out_written = n;
    HURRAY_OK
}

fn vtable(pipe: &mut Pipe) -> HurrayTransport {
    HurrayTransport {
        ctx: pipe as *mut Pipe as *mut c_void,
        read: Some(pipe_read),
        write: Some(pipe_write),
        // Hurray never frees the context; `pipe` is owned by main, so there is nothing to
        // release. A caller whose transport is heap-allocated passes its free function here.
        release: None,
        ..HurrayTransport::default()
    }
}

// ── The tensors ───────────────────────────────────────────────────────────────

fn descriptor_for(byte_size: u64) -> TensorDescriptor {
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

/// A C caller names a descriptor by decoding its bytes. (Building one from scratch through
/// the C ABI is a separate surface, not yet implemented.)
fn descriptor_handle(descriptor: &TensorDescriptor) -> *mut HurrayDescriptor {
    let bytes = descriptor.encode().expect("a descriptor encodes");
    let mut handle: *mut HurrayDescriptor = std::ptr::null_mut();
    // SAFETY: `bytes` is readable for its own length and `handle` is a live local.
    let status = unsafe { hurray_descriptor_decode(bytes.as_ptr(), bytes.len(), &mut handle) };
    assert_eq!(status, HURRAY_OK, "decode failed with status {status}");
    handle
}

fn main() {
    println!("Hurray C ABI version {HURRAY_C_ABI_VERSION}\n");

    let mut pipe = Pipe {
        bytes: Vec::new(),
        read_pos: 0,
        chunk: 24,
    };

    // ── Write two tensors through the vtable ──────────────────────────────────
    let descriptor = descriptor_for(64);
    let payloads: [Vec<u8>; 2] = [vec![0xA5; 64], (0u8..64).collect()];

    let transport = vtable(&mut pipe);
    let mut writer: *mut HurrayWriter = std::ptr::null_mut();
    // SAFETY: the vtable and the out-pointer are live locals; the write callback is set.
    let status = unsafe { hurray_writer_new(&transport, std::ptr::null(), &mut writer) };
    assert_eq!(status, HURRAY_OK, "hurray_writer_new: {status}");

    let handle = descriptor_handle(&descriptor);
    for payload in &payloads {
        let slice = HurrayIoSlice {
            ptr: payload.as_ptr(),
            len: payload.len(),
        };
        // SAFETY: the writer and descriptor handles are live, and `payload` outlives this
        // call — which is all § 3's borrowed-payload rule asks of a blocking surface.
        let status = unsafe { hurray_writer_write_tensor(writer, handle, &slice, 1) };
        assert_eq!(status, HURRAY_OK, "write_tensor: {status}");
        println!("wrote a {}-byte tensor", payload.len());
    }
    // SAFETY: both handles are live and destroyed exactly once.
    unsafe {
        hurray_descriptor_destroy(handle);
        hurray_writer_destroy(&mut writer);
    }
    assert!(writer.is_null(), "destroy nulls the handle");
    println!("\n{} bytes on the wire\n", pipe.bytes.len());

    // ── Read them back ───────────────────────────────────────────────────────
    let transport = vtable(&mut pipe);
    let mut reader: *mut HurrayReader = std::ptr::null_mut();
    // SAFETY: the vtable and out-pointer are live locals; the read callback is set. Null
    // options asks for the documented defaults.
    let status = unsafe { hurray_reader_new(&transport, std::ptr::null(), &mut reader) };
    assert_eq!(status, HURRAY_OK, "hurray_reader_new: {status}");

    let mut index = 0usize;
    loop {
        let mut tensor: *mut HurrayTensor = std::ptr::null_mut();
        // SAFETY: the reader is live and `tensor` is a live local.
        let status = unsafe { hurray_reader_next_tensor(reader, &mut tensor) };
        if status == HURRAY_END_OF_STREAM {
            println!("end of stream");
            break;
        }
        assert_eq!(status, HURRAY_OK, "next_tensor: {status}");

        // SAFETY: `tensor` is live; both accessors hand back borrowed handles that stay
        // valid until the tensor is destroyed.
        unsafe {
            let mut desc: *mut HurrayDescriptor = std::ptr::null_mut();
            hurray_tensor_descriptor(tensor, &mut desc);
            let mut rank: u32 = 0;
            hurray_descriptor_rank(desc, &mut rank);

            let mut list: *mut HurrayBufferList = std::ptr::null_mut();
            hurray_tensor_buffers(tensor, &mut list);
            let mut count: u64 = 0;
            hurray_buffer_list_len(list, &mut count);

            let mut buffer: *mut HurrayBuffer = std::ptr::null_mut();
            hurray_buffer_list_get(list, 0, &mut buffer);
            let mut data: *mut c_void = std::ptr::null_mut();
            let mut size: u64 = 0;
            let mut alignment: u32 = 0;
            hurray_buffer_data_ptr(buffer, &mut data);
            hurray_buffer_byte_size(buffer, &mut size);
            hurray_buffer_alignment(buffer, &mut alignment);

            let bytes = std::slice::from_raw_parts(data as *const u8, size as usize);
            assert_eq!(bytes, &payloads[index][..], "payload round-tripped");
            // The transport wrote straight into this allocation, at the alignment the
            // buffer handle declares — so a SIMD load on it is sound.
            assert_eq!(data as usize % alignment as usize, 0);

            println!(
                "tensor {index}: rank {rank}, {count} buffer(s), {size} bytes, \
                 {alignment}-byte aligned at {data:p}"
            );

            // One destroy for the descriptor and every buffer: the order cannot be got
            // wrong because the caller does not choose it.
            hurray_tensor_destroy(&mut tensor);
        }
        assert!(tensor.is_null());
        index += 1;
    }

    // SAFETY: the reader is live and destroyed exactly once.
    unsafe { hurray_reader_destroy(&mut reader) };
    assert!(reader.is_null());
    println!("\nread {index} tensor(s); no payload was copied after the transport wrote it");
}
