//! Demonstrates the C file surface: writing and then randomly reading a real `HRRYFILE`
//! through the transport vtable (ADR-041 §§ 7, 8).
//!
//! The file here is an actual file on disk, and it is reached the way a C consumer
//! reaches anything: through callbacks. The writer needs only `write`, because a file is
//! written in one forward pass. The reader needs `read_at` and `size`, because a
//! container is *addressed* — its trailer sits at `size - 40`, its index is located from
//! there, and a tensor is found by name without touching the rest of the file.
//!
//! No runtime is linked into any of this. The decoding is a state machine; every blocking
//! call is the caller's own `std::fs` call inside a callback.
//!
//! Run with:
//!
//! ```text
//! cargo run --example file_vtable -p hurray-ffi
//! ```

use std::ffi::c_void;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};

use hurray_core::{
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode, TensorDescriptor,
    MIN_BUFFER_ALIGNMENT,
};
use hurray_ffi::buffer::{
    hurray_buffer_alignment, hurray_buffer_byte_size, hurray_buffer_data_ptr,
};
use hurray_ffi::buffer_list::hurray_buffer_list_get;
use hurray_ffi::descriptor::{hurray_descriptor_destroy, hurray_descriptor_rank};
use hurray_ffi::{
    hurray_descriptor_decode, hurray_file_reader_destroy, hurray_file_reader_kv_count,
    hurray_file_reader_kv_entry, hurray_file_reader_new, hurray_file_reader_read_descriptor,
    hurray_file_reader_read_tensor, hurray_file_reader_tensor_count,
    hurray_file_reader_tensor_name, hurray_file_writer_destroy, hurray_file_writer_finish,
    hurray_file_writer_new, hurray_file_writer_write_tensor, hurray_tensor_buffers,
    hurray_tensor_descriptor, hurray_tensor_destroy, HurrayBuffer, HurrayBufferList,
    HurrayDescriptor, HurrayFileReader, HurrayFileWriter, HurrayIoSlice, HurrayKvEntry,
    HurrayKvValue, HurrayStatus, HurrayTensor, HurrayTransport, HURRAY_C_ABI_VERSION,
    HURRAY_ERR_NOT_FOUND, HURRAY_ERR_TRANSPORT, HURRAY_KV_STRING, HURRAY_KV_UINT64, HURRAY_OK,
};

// ── The caller's transport ────────────────────────────────────────────────────

/// A file on disk, reached through callbacks. A C caller would hand over a `FILE*`, a
/// descriptor, or a memory-mapped span just as easily — Hurray never learns which.
struct DiskFile {
    inner: File,
}

/// Answers one absolute range request.
///
/// Seek-then-read inside the callback, which is what makes a positional read portable:
/// the cursor is this callback's business and Hurray never relies on it.
///
/// # Safety
///
/// `ctx` must point to a live `DiskFile`, `dst` must be writable for `len` bytes, and
/// `out_filled` must be writable.
unsafe extern "C" fn disk_read_at(
    ctx: *mut c_void,
    offset: u64,
    dst: *mut u8,
    len: usize,
    out_filled: *mut usize,
) -> HurrayStatus {
    let file = &mut *(ctx as *mut DiskFile);
    *out_filled = 0;

    if file.inner.seek(SeekFrom::Start(offset)).is_err() {
        // No bytes moved, which is what makes the request retryable.
        return HURRAY_ERR_TRANSPORT;
    }
    let slice = std::slice::from_raw_parts_mut(dst, len);
    match file.inner.read(slice) {
        // Fewer bytes than asked for is normal: the rest of the range is requested again.
        Ok(n) => {
            *out_filled = n;
            HURRAY_OK
        }
        Err(_) => HURRAY_ERR_TRANSPORT,
    }
}

/// Reports the file's total size, which is where the container is located from.
///
/// # Safety
///
/// `ctx` must point to a live `DiskFile` and `out_size` must be writable.
unsafe extern "C" fn disk_size(ctx: *mut c_void, out_size: *mut u64) -> HurrayStatus {
    let file = &*(ctx as *mut DiskFile);
    match file.inner.metadata() {
        Ok(meta) => {
            *out_size = meta.len();
            HURRAY_OK
        }
        Err(_) => HURRAY_ERR_TRANSPORT,
    }
}

/// # Safety
///
/// `ctx` must point to a live `DiskFile`, `src` must be readable for `len` bytes, and
/// `out_written` must be writable.
unsafe extern "C" fn disk_write(
    ctx: *mut c_void,
    src: *const u8,
    len: usize,
    out_written: *mut usize,
) -> HurrayStatus {
    let file = &mut *(ctx as *mut DiskFile);
    *out_written = 0;
    match file.inner.write(std::slice::from_raw_parts(src, len)) {
        // Never 0 on success: zero accepted bytes means the sink is closed, which Hurray
        // reports as its own status rather than as progress.
        Ok(n) => {
            *out_written = n;
            HURRAY_OK
        }
        Err(_) => HURRAY_ERR_TRANSPORT,
    }
}

fn vtable(file: &mut DiskFile) -> HurrayTransport {
    HurrayTransport {
        ctx: file as *mut DiskFile as *mut c_void,
        write: Some(disk_write),
        read_at: Some(disk_read_at),
        size: Some(disk_size),
        // The context is ours, so Hurray must not free it; `read` is for streams.
        ..HurrayTransport::default()
    }
}

// ── The tensors ───────────────────────────────────────────────────────────────

fn descriptor(elements: u64) -> TensorDescriptor {
    let byte_size = elements * 4;
    let handle = BufferHandle::new(
        byte_size,
        MIN_BUFFER_ALIGNMENT,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )
    .expect("buffer handle");
    TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(vec![elements]).expect("shape"),
        0,
        LayoutDescriptor::RowMajor,
        vec![handle],
        None,
        None,
        None,
        None,
    )
    .expect("descriptor")
}

/// Wraps a descriptor in the C handle. A C caller gets one by decoding descriptor bytes;
/// originating one from C needs an encode surface that does not exist yet.
unsafe fn descriptor_handle(descriptor: &TensorDescriptor) -> *mut HurrayDescriptor {
    let bytes = descriptor.encode().expect("encode");
    let mut handle: *mut HurrayDescriptor = std::ptr::null_mut();
    assert_eq!(
        hurray_descriptor_decode(bytes.as_ptr(), bytes.len(), &mut handle),
        HURRAY_OK
    );
    handle
}

fn main() {
    println!("C ABI version: {HURRAY_C_ABI_VERSION}");

    let path = std::env::temp_dir().join("hurray-file-vtable-example.hrry");
    let embeddings: Vec<u8> = (0..1024u32)
        .flat_map(|i| (i as f32).to_le_bytes())
        .collect();
    let bias: Vec<u8> = (0..16u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let embeddings_descriptor = descriptor(1024);
    let bias_descriptor = descriptor(16);

    // ── Write ─────────────────────────────────────────────────────────────────

    let mut sink = DiskFile {
        inner: File::create(&path).expect("create"),
    };

    unsafe {
        let transport = vtable(&mut sink);
        let mut writer: *mut HurrayFileWriter = std::ptr::null_mut();
        // Null options: 4096-byte data alignment, index in write order. The 64-byte file
        // header goes out during this call.
        assert_eq!(
            hurray_file_writer_new(&transport, std::ptr::null(), &mut writer),
            HURRAY_OK
        );

        for (name, descriptor, payload) in [
            ("embeddings", &embeddings_descriptor, &embeddings),
            ("bias", &bias_descriptor, &bias),
        ] {
            let handle = descriptor_handle(descriptor);
            let slice = HurrayIoSlice {
                ptr: payload.as_ptr(),
                len: payload.len(),
            };
            assert_eq!(
                hurray_file_writer_write_tensor(
                    writer,
                    name.as_ptr(),
                    name.len(),
                    handle,
                    &slice,
                    1
                ),
                HURRAY_OK
            );
            assert_eq!(hurray_descriptor_destroy(handle), HURRAY_OK);
            println!("wrote {name} ({} bytes of payload)", payload.len());
        }

        // The KV section: borrowed keys and values, copied as the footer goes out.
        let producer = "hurray-ffi example";
        let kv = [
            HurrayKvEntry {
                key: "producer".as_ptr(),
                key_len: "producer".len(),
                value: HurrayKvValue {
                    kind: HURRAY_KV_STRING,
                    data: producer.as_ptr(),
                    data_len: producer.len(),
                    ..HurrayKvValue::default()
                },
            },
            HurrayKvEntry {
                key: "parameter_count".as_ptr(),
                key_len: "parameter_count".len(),
                value: HurrayKvValue {
                    kind: HURRAY_KV_UINT64,
                    u64_value: 1040,
                    ..HurrayKvValue::default()
                },
            },
        ];

        // Until this returns, the bytes on disk are not a container: the index a reader
        // opens from is written here.
        assert_eq!(
            hurray_file_writer_finish(writer, kv.as_ptr(), kv.len()),
            HURRAY_OK
        );
        assert_eq!(hurray_file_writer_destroy(&mut writer), HURRAY_OK);
        assert!(writer.is_null());
    }
    sink.inner.flush().expect("flush");
    drop(sink);
    println!("finished {}", path.display());

    // ── Read, by name and without reading the rest ────────────────────────────

    let mut source = DiskFile {
        inner: File::open(&path).expect("open"),
    };

    unsafe {
        let transport = vtable(&mut source);
        let mut reader: *mut HurrayFileReader = std::ptr::null_mut();
        // Opening reads the header, the trailer, the index and the KV section — and
        // nothing else.
        assert_eq!(hurray_file_reader_new(&transport, &mut reader), HURRAY_OK);

        let mut count: u64 = 0;
        assert_eq!(
            hurray_file_reader_tensor_count(reader, &mut count),
            HURRAY_OK
        );
        print!("{count} tensors:");
        for index in 0..count {
            let mut name: *const u8 = std::ptr::null();
            let mut len: usize = 0;
            assert_eq!(
                hurray_file_reader_tensor_name(reader, index, &mut name, &mut len),
                HURRAY_OK
            );
            // Borrowed from the reader, and not NUL-terminated.
            let name = std::str::from_utf8(std::slice::from_raw_parts(name, len)).expect("utf8");
            print!(" {name}");
        }
        println!();

        // A descriptor on its own: what a loader inspects before deciding whether it
        // wants the tensor at all. No data buffer is read.
        let mut descriptor: *mut HurrayDescriptor = std::ptr::null_mut();
        assert_eq!(
            hurray_file_reader_read_descriptor(
                reader,
                "bias".as_ptr(),
                "bias".len(),
                &mut descriptor
            ),
            HURRAY_OK
        );
        let mut rank: u32 = 0;
        assert_eq!(hurray_descriptor_rank(descriptor, &mut rank), HURRAY_OK);
        println!("bias: rank {rank}, no data read");
        assert_eq!(hurray_descriptor_destroy(descriptor), HURRAY_OK);

        // The second tensor, read without touching the first.
        let mut tensor: *mut HurrayTensor = std::ptr::null_mut();
        assert_eq!(
            hurray_file_reader_read_tensor(
                reader,
                "embeddings".as_ptr(),
                "embeddings".len(),
                &mut tensor
            ),
            HURRAY_OK
        );

        let mut borrowed: *mut HurrayDescriptor = std::ptr::null_mut();
        assert_eq!(hurray_tensor_descriptor(tensor, &mut borrowed), HURRAY_OK);
        let mut list: *mut HurrayBufferList = std::ptr::null_mut();
        assert_eq!(hurray_tensor_buffers(tensor, &mut list), HURRAY_OK);
        let mut buffer: *mut HurrayBuffer = std::ptr::null_mut();
        assert_eq!(hurray_buffer_list_get(list, 0, &mut buffer), HURRAY_OK);

        let mut data: *mut c_void = std::ptr::null_mut();
        let mut byte_size: u64 = 0;
        let mut alignment: u32 = 0;
        assert_eq!(hurray_buffer_data_ptr(buffer, &mut data), HURRAY_OK);
        assert_eq!(hurray_buffer_byte_size(buffer, &mut byte_size), HURRAY_OK);
        assert_eq!(hurray_buffer_alignment(buffer, &mut alignment), HURRAY_OK);

        let bytes = std::slice::from_raw_parts(data as *const u8, byte_size as usize);
        assert_eq!(bytes, &embeddings[..]);
        // The transport wrote straight into memory the core allocated at the alignment
        // the buffer handle declares, so a SIMD load on it is sound.
        assert_eq!(data as usize % alignment as usize, 0);
        println!("embeddings: {byte_size} bytes at {alignment}-byte alignment, verified");

        // One destroy for both halves of the tensor.
        assert_eq!(hurray_tensor_destroy(&mut tensor), HURRAY_OK);

        // A name that is not in the index is a question, not a defect: the reader stays
        // usable afterwards.
        assert_eq!(
            hurray_file_reader_read_tensor(reader, "absent".as_ptr(), 6, &mut tensor),
            HURRAY_ERR_NOT_FOUND
        );

        // The KV section, in file order.
        let mut kv_count: u64 = 0;
        assert_eq!(
            hurray_file_reader_kv_count(reader, &mut kv_count),
            HURRAY_OK
        );
        for index in 0..kv_count {
            let mut entry = HurrayKvEntry::default();
            assert_eq!(
                hurray_file_reader_kv_entry(reader, index, &mut entry),
                HURRAY_OK
            );
            let key = std::str::from_utf8(std::slice::from_raw_parts(entry.key, entry.key_len))
                .expect("utf8");
            match entry.value.kind {
                HURRAY_KV_STRING => {
                    let value = std::str::from_utf8(std::slice::from_raw_parts(
                        entry.value.data,
                        entry.value.data_len,
                    ))
                    .expect("utf8");
                    println!("kv {key} = {value:?}");
                }
                HURRAY_KV_UINT64 => println!("kv {key} = {}", entry.value.u64_value),
                other => println!("kv {key} = <kind {other}>"),
            }
        }

        assert_eq!(hurray_file_reader_destroy(&mut reader), HURRAY_OK);
    }

    std::fs::remove_file(&path).expect("remove");
}
