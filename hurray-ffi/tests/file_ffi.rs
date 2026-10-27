//! Integration tests for the C file surface: the range-request half of the transport
//! vtable, the file handles, and the KV section in both directions.
//!
//! Everything is driven through the C entry points over an in-memory transport, because
//! the properties under test are the ABI's, not the core's — the core has its own tests.
//!
//! Two of the tests write bytes and compare them with the frozen corpus in
//! `hurray-io/tests/fixtures/file/`, which was emitted by the pre-sans-io `FileWriter`.
//! That is the real byte-compatibility check: a file written through C must be the file
//! Hurray has always written. Never regenerate those fixtures — see their README.

use std::ffi::c_void;

use hurray_core::{
    buffer_size_bytes,
    layout::{CooLayout, LayoutDescriptor},
    BufferHandle, DeviceTag, ElementType, Shape, SyncMode, TensorDescriptor, MIN_BUFFER_ALIGNMENT,
};
use hurray_ffi::{
    buffer::{hurray_buffer_alignment, hurray_buffer_byte_size, hurray_buffer_data_ptr},
    buffer_list::{hurray_buffer_list_get, hurray_buffer_list_len},
    descriptor::{hurray_descriptor_destroy, hurray_descriptor_element_type_tag},
    hurray_descriptor_decode, hurray_file_reader_destroy, hurray_file_reader_kv_count,
    hurray_file_reader_kv_entry, hurray_file_reader_new, hurray_file_reader_read_descriptor,
    hurray_file_reader_read_tensor, hurray_file_reader_tensor_count,
    hurray_file_reader_tensor_name, hurray_file_writer_destroy, hurray_file_writer_finish,
    hurray_file_writer_new, hurray_file_writer_write_tensor, hurray_tensor_buffers,
    hurray_tensor_destroy, HurrayBuffer, HurrayBufferList, HurrayDescriptor, HurrayFileReader,
    HurrayFileWriter, HurrayFileWriterOptions, HurrayIoSlice, HurrayKvEntry, HurrayKvValue,
    HurrayStatus, HurrayTensor, HurrayTransport, HURRAY_ERR_INDEX_OUT_OF_BOUNDS,
    HURRAY_ERR_INVALID_FILE, HURRAY_ERR_INVALID_MAGIC, HURRAY_ERR_NOT_FOUND,
    HURRAY_ERR_NULL_POINTER, HURRAY_ERR_SINK_CLOSED, HURRAY_ERR_TRANSPORT,
    HURRAY_ERR_UNEXPECTED_EOF, HURRAY_KV_ARRAY, HURRAY_KV_BOOL, HURRAY_KV_BYTES, HURRAY_KV_FLOAT64,
    HURRAY_KV_INT64, HURRAY_KV_STRING, HURRAY_KV_UINT64, HURRAY_OK,
};

// ── An in-memory file transport ───────────────────────────────────────────────

/// A scriptable file transport: answers range requests out of `bytes`, collects a
/// writer's output in `output`, and can be told to move bytes in small chunks or to fail
/// at a chosen point.
#[derive(Default)]
struct MemFile {
    /// Bytes a reader addresses.
    bytes: Vec<u8>,
    /// Bytes a writer produced.
    output: Vec<u8>,
    /// Maximum bytes moved per callback; `0` means "as many as offered".
    chunk: usize,
    /// Total bytes handed to a reader, over every call.
    served: usize,
    /// Fail the next `read_at` that asks for this offset, once.
    fail_read_at: Option<u64>,
    /// Report a size this many bytes short of the truth, simulating a truncated file.
    truncate_to: Option<u64>,
    /// Fail the write callback once, when the sink holds this many bytes.
    fail_write_at: Option<usize>,
    /// Accept zero bytes from this sink size onwards.
    close_write_at: Option<usize>,
    /// How many times the release callback ran.
    released: u32,
}

impl MemFile {
    fn reading(bytes: Vec<u8>) -> Box<Self> {
        Box::new(Self {
            bytes,
            ..Self::default()
        })
    }

    fn writing() -> Box<Self> {
        Box::new(Self::default())
    }
}

/// # Safety
///
/// `ctx` must be a live `MemFile`; `dst` writable for `len`; `out_filled` writable.
unsafe extern "C" fn mem_read_at(
    ctx: *mut c_void,
    offset: u64,
    dst: *mut u8,
    len: usize,
    out_filled: *mut usize,
) -> HurrayStatus {
    let f = &mut *(ctx as *mut MemFile);
    *out_filled = 0;

    if f.fail_read_at == Some(offset) {
        // One-shot, and no bytes moved: the contract that makes a retry sound.
        f.fail_read_at = None;
        return HURRAY_ERR_TRANSPORT;
    }

    let start = offset as usize;
    if start > f.bytes.len() {
        return HURRAY_ERR_TRANSPORT;
    }
    let available = f.bytes.len() - start;
    let mut n = len.min(available);
    if f.chunk > 0 {
        n = n.min(f.chunk);
    }
    std::ptr::copy_nonoverlapping(f.bytes.as_ptr().add(start), dst, n);
    f.served += n;
    *out_filled = n;
    HURRAY_OK
}

/// # Safety
///
/// `ctx` must be a live `MemFile`; `out_size` writable.
unsafe extern "C" fn mem_size(ctx: *mut c_void, out_size: *mut u64) -> HurrayStatus {
    let f = &*(ctx as *mut MemFile);
    *out_size = f.truncate_to.unwrap_or(f.bytes.len() as u64);
    HURRAY_OK
}

/// # Safety
///
/// `ctx` must be a live `MemFile`; `src` readable for `len`; `out_written` writable.
unsafe extern "C" fn mem_write(
    ctx: *mut c_void,
    src: *const u8,
    len: usize,
    out_written: *mut usize,
) -> HurrayStatus {
    let f = &mut *(ctx as *mut MemFile);
    *out_written = 0;

    if let Some(at) = f.fail_write_at {
        if f.output.len() >= at {
            f.fail_write_at = None;
            return HURRAY_ERR_TRANSPORT;
        }
    }
    if let Some(at) = f.close_write_at {
        if f.output.len() >= at {
            // Zero accepted bytes with OK: a closed sink, not a failure.
            return HURRAY_OK;
        }
    }

    let mut n = len;
    if f.chunk > 0 {
        n = n.min(f.chunk);
    }
    f.output
        .extend_from_slice(std::slice::from_raw_parts(src, n));
    *out_written = n;
    HURRAY_OK
}

/// # Safety
///
/// `ctx` must be a live `MemFile`.
unsafe extern "C" fn mem_release(ctx: *mut c_void) {
    (*(ctx as *mut MemFile)).released += 1;
}

fn vtable(ctx: &mut MemFile) -> HurrayTransport {
    HurrayTransport {
        ctx: ctx as *mut MemFile as *mut c_void,
        write: Some(mem_write),
        release: Some(mem_release),
        read_at: Some(mem_read_at),
        size: Some(mem_size),
        // A file reader never consumes sequentially: `read` stays null.
        ..HurrayTransport::default()
    }
}

// ── Handles ───────────────────────────────────────────────────────────────────

unsafe fn reader_over(file: &mut MemFile) -> *mut HurrayFileReader {
    let vt = vtable(file);
    let mut out: *mut HurrayFileReader = std::ptr::null_mut();
    assert_eq!(hurray_file_reader_new(&vt, &mut out), HURRAY_OK);
    out
}

unsafe fn writer_over(
    file: &mut MemFile,
    options: Option<HurrayFileWriterOptions>,
) -> *mut HurrayFileWriter {
    let vt = vtable(file);
    let mut out: *mut HurrayFileWriter = std::ptr::null_mut();
    let options_ptr = options
        .as_ref()
        .map_or(std::ptr::null(), |o| o as *const HurrayFileWriterOptions);
    assert_eq!(
        hurray_file_writer_new(&vt, options_ptr, &mut out),
        HURRAY_OK
    );
    out
}

/// Wraps a descriptor in the C handle, the only way a C caller can name one today.
unsafe fn descriptor_handle(descriptor: &TensorDescriptor) -> *mut HurrayDescriptor {
    let bytes = descriptor.encode().unwrap();
    let mut out: *mut HurrayDescriptor = std::ptr::null_mut();
    assert_eq!(
        hurray_descriptor_decode(bytes.as_ptr(), bytes.len(), &mut out),
        HURRAY_OK
    );
    out
}

unsafe fn write_one(
    writer: *mut HurrayFileWriter,
    name: &str,
    descriptor: &TensorDescriptor,
    payloads: &[&[u8]],
) -> HurrayStatus {
    let handle = descriptor_handle(descriptor);
    let slices: Vec<HurrayIoSlice> = payloads
        .iter()
        .map(|p| HurrayIoSlice {
            ptr: p.as_ptr(),
            len: p.len(),
        })
        .collect();
    let status = hurray_file_writer_write_tensor(
        writer,
        name.as_ptr(),
        name.len(),
        handle,
        slices.as_ptr(),
        slices.len(),
    );
    hurray_descriptor_destroy(handle);
    status
}

unsafe fn read_tensor(reader: *mut HurrayFileReader, name: &str) -> *mut HurrayTensor {
    let mut tensor: *mut HurrayTensor = std::ptr::null_mut();
    let status = hurray_file_reader_read_tensor(reader, name.as_ptr(), name.len(), &mut tensor);
    assert_eq!(status, HURRAY_OK, "reading {name}");
    tensor
}

/// Borrows buffer `index` of a tensor handle as a byte slice, with its declared alignment.
unsafe fn buffer_of(tensor: *mut HurrayTensor, index: u64) -> (&'static [u8], u32) {
    let mut list: *mut HurrayBufferList = std::ptr::null_mut();
    assert_eq!(hurray_tensor_buffers(tensor, &mut list), HURRAY_OK);

    let mut buffer: *mut HurrayBuffer = std::ptr::null_mut();
    assert_eq!(hurray_buffer_list_get(list, index, &mut buffer), HURRAY_OK);

    let mut data: *mut c_void = std::ptr::null_mut();
    let mut size: u64 = 0;
    let mut alignment: u32 = 0;
    assert_eq!(hurray_buffer_data_ptr(buffer, &mut data), HURRAY_OK);
    assert_eq!(hurray_buffer_byte_size(buffer, &mut size), HURRAY_OK);
    assert_eq!(hurray_buffer_alignment(buffer, &mut alignment), HURRAY_OK);

    (
        std::slice::from_raw_parts(data as *const u8, size as usize),
        alignment,
    )
}

unsafe fn buffer_count(tensor: *mut HurrayTensor) -> u64 {
    let mut list: *mut HurrayBufferList = std::ptr::null_mut();
    assert_eq!(hurray_tensor_buffers(tensor, &mut list), HURRAY_OK);
    let mut len: u64 = 0;
    assert_eq!(hurray_buffer_list_len(list, &mut len), HURRAY_OK);
    len
}

unsafe fn tensor_names(reader: *mut HurrayFileReader) -> Vec<String> {
    let mut count: u64 = 0;
    assert_eq!(
        hurray_file_reader_tensor_count(reader, &mut count),
        HURRAY_OK
    );
    (0..count)
        .map(|i| {
            let mut ptr: *const u8 = std::ptr::null();
            let mut len: usize = 0;
            assert_eq!(
                hurray_file_reader_tensor_name(reader, i, &mut ptr, &mut len),
                HURRAY_OK
            );
            String::from_utf8(std::slice::from_raw_parts(ptr, len).to_vec()).unwrap()
        })
        .collect()
}

// ── The descriptors the frozen corpus is made of ──────────────────────────────
//
// Deliberately re-stated here rather than imported: these bytes are what the C writer has
// to reproduce, and a fixture whose meaning can be changed from another crate's test file
// is not frozen.

fn buf(byte_size: u64, alignment: u32) -> BufferHandle {
    BufferHandle::new(
        byte_size,
        alignment,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )
    .unwrap()
}

fn payload(len: usize, salt: u8) -> Vec<u8> {
    (0..len).map(|i| (i % 251) as u8 ^ salt).collect()
}

fn row_major() -> TensorDescriptor {
    TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(vec![8u64, 8]).unwrap(),
        0,
        LayoutDescriptor::RowMajor,
        vec![buf(256, MIN_BUFFER_ALIGNMENT)],
        None,
        None,
        None,
        None,
    )
    .unwrap()
}

fn plain_2x2() -> TensorDescriptor {
    TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(vec![2u64, 2]).unwrap(),
        0,
        LayoutDescriptor::RowMajor,
        vec![buf(
            buffer_size_bytes(ElementType::Float32, 4),
            MIN_BUFFER_ALIGNMENT,
        )],
        None,
        None,
        None,
        None,
    )
    .unwrap()
}

fn coo() -> TensorDescriptor {
    let nnz = 8u64;
    TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(vec![8u64, 8]).unwrap(),
        0,
        LayoutDescriptor::Coo(CooLayout::new(nnz, false)),
        vec![buf(32, 128), buf(nnz * 2 * 8, 4096)],
        None,
        None,
        None,
        None,
    )
    .unwrap()
}

fn fixture(name: &str) -> Vec<u8> {
    let path = format!(
        "{}/../hurray-io/tests/fixtures/file/{}",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
    std::fs::read(&path).unwrap_or_else(|e| panic!("reading fixture {path}: {e}"))
}

// ── Round trip ────────────────────────────────────────────────────────────────

#[test]
fn a_file_written_through_c_reads_back_through_c() {
    unsafe {
        let descriptor = row_major();
        let data = payload(256, 0x00);

        let mut sink = MemFile::writing();
        let mut writer = writer_over(&mut sink, None);
        assert_eq!(
            write_one(writer, "embeddings", &descriptor, &[&data]),
            HURRAY_OK
        );
        assert_eq!(
            hurray_file_writer_finish(writer, std::ptr::null(), 0),
            HURRAY_OK
        );
        assert_eq!(hurray_file_writer_destroy(&mut writer), HURRAY_OK);
        assert!(writer.is_null());
        assert_eq!(sink.released, 1);

        let mut source = MemFile::reading(std::mem::take(&mut sink.output));
        let mut reader = reader_over(&mut source);

        assert_eq!(tensor_names(reader), ["embeddings"]);

        let mut tensor = read_tensor(reader, "embeddings");
        let (bytes, declared) = buffer_of(tensor, 0);
        assert_eq!(bytes, &data[..]);
        // § 5: the alignment the handle declares, not the allocator's default.
        assert_eq!(bytes.as_ptr() as usize % declared as usize, 0);

        assert_eq!(hurray_tensor_destroy(&mut tensor), HURRAY_OK);
        assert_eq!(hurray_file_reader_destroy(&mut reader), HURRAY_OK);
        assert_eq!(source.released, 1);
    }
}

#[test]
fn a_tensor_survives_the_reader_that_produced_it() {
    unsafe {
        let mut source = MemFile::reading(fixture("row_major.hrry"));
        let mut reader = reader_over(&mut source);
        let mut tensor = read_tensor(reader, "embeddings");

        // The tensor owns its buffers, so destroying the reader cannot invalidate them.
        assert_eq!(hurray_file_reader_destroy(&mut reader), HURRAY_OK);
        let (bytes, _) = buffer_of(tensor, 0);
        assert_eq!(bytes, &payload(256, 0x00)[..]);
        assert_eq!(hurray_tensor_destroy(&mut tensor), HURRAY_OK);
    }
}

// ── Byte compatibility with the frozen corpus ─────────────────────────────────

#[test]
fn the_c_writer_emits_the_frozen_row_major_bytes() {
    unsafe {
        let descriptor = row_major();
        let data = payload(256, 0x00);

        let mut sink = MemFile::writing();
        // Seven bytes a call: a short write must not change what lands on the sink.
        sink.chunk = 7;
        let mut writer = writer_over(&mut sink, None);
        assert_eq!(
            write_one(writer, "embeddings", &descriptor, &[&data]),
            HURRAY_OK
        );
        assert_eq!(
            hurray_file_writer_finish(writer, std::ptr::null(), 0),
            HURRAY_OK
        );
        assert_eq!(hurray_file_writer_destroy(&mut writer), HURRAY_OK);

        assert_eq!(
            sink.output,
            fixture("row_major.hrry"),
            "the C writer no longer emits the bytes 0.1.0 emitted"
        );
    }
}

#[test]
fn the_c_writer_emits_the_frozen_kv_sorted_bytes() {
    unsafe {
        let zeta = row_major();
        let alpha = plain_2x2();
        let zeta_data = payload(256, 0x55);
        let alpha_data = payload(16, 0x66);

        let mut sink = MemFile::writing();
        let options = HurrayFileWriterOptions {
            data_buffer_alignment: 8192,
            sorted_index: true,
        };
        let mut writer = writer_over(&mut sink, Some(options));
        // Written out of byte order: the sorted index is the writer's job, not the
        // caller's.
        assert_eq!(write_one(writer, "zeta", &zeta, &[&zeta_data]), HURRAY_OK);
        assert_eq!(
            write_one(writer, "alpha", &alpha, &[&alpha_data]),
            HURRAY_OK
        );

        let kv = kv_section();
        assert_eq!(
            hurray_file_writer_finish(writer, kv.as_ptr(), kv.len()),
            HURRAY_OK
        );
        assert_eq!(hurray_file_writer_destroy(&mut writer), HURRAY_OK);

        assert_eq!(
            sink.output,
            fixture("kv_sorted.hrry"),
            "the C writer's KV section, index sort or alignment has drifted"
        );
    }
}

#[test]
fn the_c_reader_reads_the_frozen_file_corpus() {
    unsafe {
        // row_major.hrry — one tensor, default options, no KV
        let mut source = MemFile::reading(fixture("row_major.hrry"));
        // Seven bytes a call: every range must survive being answered piecemeal.
        source.chunk = 7;
        let mut reader = reader_over(&mut source);
        assert_eq!(tensor_names(reader), ["embeddings"]);
        let mut tensor = read_tensor(reader, "embeddings");
        assert_eq!(buffer_of(tensor, 0).0, &payload(256, 0x00)[..]);
        assert_eq!(hurray_tensor_destroy(&mut tensor), HURRAY_OK);
        assert_eq!(hurray_file_reader_destroy(&mut reader), HURRAY_OK);

        // multi_buffer.hrry — two buffers declaring 128 and 4096, so the data region
        // carries inter-buffer padding the reader must skip.
        let mut source = MemFile::reading(fixture("multi_buffer.hrry"));
        let mut reader = reader_over(&mut source);
        let mut tensor = read_tensor(reader, "sparse");
        assert_eq!(buffer_count(tensor), 2);
        let (values, values_alignment) = buffer_of(tensor, 0);
        let (indices, indices_alignment) = buffer_of(tensor, 1);
        assert_eq!(values, &payload(32, 0x11)[..]);
        assert_eq!(indices, &payload(128, 0x22)[..]);
        assert_eq!(values_alignment, 128);
        assert_eq!(indices_alignment, 4096);
        assert_eq!(values.as_ptr() as usize % 128, 0);
        assert_eq!(indices.as_ptr() as usize % 4096, 0);
        assert_eq!(hurray_tensor_destroy(&mut tensor), HURRAY_OK);
        assert_eq!(hurray_file_reader_destroy(&mut reader), HURRAY_OK);

        // kv_sorted.hrry — sorted index, every KV tag
        let mut source = MemFile::reading(fixture("kv_sorted.hrry"));
        let mut reader = reader_over(&mut source);
        assert_eq!(tensor_names(reader), ["alpha", "zeta"]);
        let mut tensor = read_tensor(reader, "zeta");
        assert_eq!(buffer_of(tensor, 0).0, &payload(256, 0x55)[..]);
        assert_eq!(hurray_tensor_destroy(&mut tensor), HURRAY_OK);
        let mut tensor = read_tensor(reader, "alpha");
        assert_eq!(buffer_of(tensor, 0).0, &payload(16, 0x66)[..]);
        assert_eq!(hurray_tensor_destroy(&mut tensor), HURRAY_OK);
        assert_eq!(hurray_file_reader_destroy(&mut reader), HURRAY_OK);

        // nested_composite.hrry — five entries; a composite head is a buffer-less tensor
        for name in ["nested_composite.hrry", "nested_composite_padded_head.hrry"] {
            let mut source = MemFile::reading(fixture(name));
            let mut reader = reader_over(&mut source);
            assert_eq!(
                tensor_names(reader),
                ["group", "part", "part.0", "part.1", "group.1"],
                "{name}"
            );
            let mut head = read_tensor(reader, "group");
            assert_eq!(buffer_count(head), 0, "{name}: a head carries no buffers");
            assert_eq!(hurray_tensor_destroy(&mut head), HURRAY_OK);
            let mut tile = read_tensor(reader, "part.0");
            assert_eq!(buffer_of(tile, 0).0, &payload(128, 0x77)[..], "{name}");
            assert_eq!(hurray_tensor_destroy(&mut tile), HURRAY_OK);
            let mut tail = read_tensor(reader, "group.1");
            assert_eq!(buffer_of(tail, 0).0, &payload(16, 0x99)[..], "{name}");
            assert_eq!(hurray_tensor_destroy(&mut tail), HURRAY_OK);
            assert_eq!(hurray_file_reader_destroy(&mut reader), HURRAY_OK);
        }
    }
}

// ── The KV section ────────────────────────────────────────────────────────────

/// The seven KV wire tags, as a C caller spells them.
fn kv_section() -> Vec<HurrayKvEntry> {
    const MODEL: &str = "hurray-test";
    const DIGEST: [u8; 4] = [0xDE, 0xAD, 0xBE, 0xEF];
    // Leaked deliberately: the array's elements must outlive the entry that points at
    // them, and a test that frees them is testing its own lifetime bookkeeping.
    let layer_sizes: &'static [HurrayKvValue] = Box::leak(Box::new([
        HurrayKvValue {
            kind: HURRAY_KV_INT64,
            i64_value: 4096,
            ..HurrayKvValue::default()
        },
        HurrayKvValue {
            kind: HURRAY_KV_INT64,
            i64_value: 11008,
            ..HurrayKvValue::default()
        },
        HurrayKvValue {
            kind: HURRAY_KV_INT64,
            i64_value: 4096,
            ..HurrayKvValue::default()
        },
    ]));

    fn entry(key: &'static str, value: HurrayKvValue) -> HurrayKvEntry {
        HurrayKvEntry {
            key: key.as_ptr(),
            key_len: key.len(),
            value,
        }
    }

    vec![
        entry(
            "model",
            HurrayKvValue {
                kind: HURRAY_KV_STRING,
                data: MODEL.as_ptr(),
                data_len: MODEL.len(),
                ..HurrayKvValue::default()
            },
        ),
        entry(
            "epoch",
            HurrayKvValue {
                kind: HURRAY_KV_INT64,
                i64_value: -7,
                ..HurrayKvValue::default()
            },
        ),
        entry(
            "tokens",
            HurrayKvValue {
                kind: HURRAY_KV_UINT64,
                u64_value: 1 << 40,
                ..HurrayKvValue::default()
            },
        ),
        entry(
            "temperature",
            HurrayKvValue {
                kind: HURRAY_KV_FLOAT64,
                f64_value: 0.8125,
                ..HurrayKvValue::default()
            },
        ),
        entry(
            "quantized",
            HurrayKvValue {
                kind: HURRAY_KV_BOOL,
                bool_value: true,
                ..HurrayKvValue::default()
            },
        ),
        entry(
            "digest",
            HurrayKvValue {
                kind: HURRAY_KV_BYTES,
                data: DIGEST.as_ptr(),
                data_len: DIGEST.len(),
                ..HurrayKvValue::default()
            },
        ),
        entry(
            "layer_sizes",
            HurrayKvValue {
                kind: HURRAY_KV_ARRAY,
                items: layer_sizes.as_ptr(),
                item_count: layer_sizes.len(),
                ..HurrayKvValue::default()
            },
        ),
    ]
}

/// Reads every KV entry out of a reader, in file order.
unsafe fn kv_entries(reader: *mut HurrayFileReader) -> Vec<HurrayKvEntry> {
    let mut count: u64 = 0;
    assert_eq!(hurray_file_reader_kv_count(reader, &mut count), HURRAY_OK);
    (0..count)
        .map(|i| {
            let mut entry = HurrayKvEntry::default();
            assert_eq!(
                hurray_file_reader_kv_entry(reader, i, &mut entry),
                HURRAY_OK
            );
            entry
        })
        .collect()
}

unsafe fn key_of(entry: &HurrayKvEntry) -> &str {
    std::str::from_utf8(std::slice::from_raw_parts(entry.key, entry.key_len)).unwrap()
}

#[test]
fn every_kv_tag_reads_back_through_c() {
    unsafe {
        let mut source = MemFile::reading(fixture("kv_sorted.hrry"));
        let mut reader = reader_over(&mut source);
        let entries = kv_entries(reader);

        let keys: Vec<&str> = entries.iter().map(|e| key_of(e)).collect();
        assert_eq!(
            keys,
            [
                "model",
                "epoch",
                "tokens",
                "temperature",
                "quantized",
                "digest",
                "layer_sizes"
            ]
        );

        assert_eq!(entries[0].value.kind, HURRAY_KV_STRING);
        assert_eq!(
            std::slice::from_raw_parts(entries[0].value.data, entries[0].value.data_len),
            b"hurray-test"
        );
        assert_eq!(entries[1].value.kind, HURRAY_KV_INT64);
        assert_eq!(entries[1].value.i64_value, -7);
        assert_eq!(entries[2].value.kind, HURRAY_KV_UINT64);
        assert_eq!(entries[2].value.u64_value, 1 << 40);
        assert_eq!(entries[3].value.kind, HURRAY_KV_FLOAT64);
        assert_eq!(entries[3].value.f64_value, 0.8125);
        assert_eq!(entries[4].value.kind, HURRAY_KV_BOOL);
        assert!(entries[4].value.bool_value);
        assert_eq!(entries[5].value.kind, HURRAY_KV_BYTES);
        assert_eq!(
            std::slice::from_raw_parts(entries[5].value.data, entries[5].value.data_len),
            &[0xDE, 0xAD, 0xBE, 0xEF]
        );

        assert_eq!(entries[6].value.kind, HURRAY_KV_ARRAY);
        assert_eq!(entries[6].value.item_count, 3);
        let items = std::slice::from_raw_parts(entries[6].value.items, 3);
        let values: Vec<i64> = items.iter().map(|v| v.i64_value).collect();
        assert!(items.iter().all(|v| v.kind == HURRAY_KV_INT64));
        assert_eq!(values, [4096, 11008, 4096]);

        let mut out_of_range = HurrayKvEntry::default();
        assert_eq!(
            hurray_file_reader_kv_entry(reader, 7, &mut out_of_range),
            HURRAY_ERR_INDEX_OUT_OF_BOUNDS
        );
        assert_eq!(hurray_file_reader_destroy(&mut reader), HURRAY_OK);
    }
}

#[test]
fn a_kv_section_read_through_c_writes_back_through_c() {
    unsafe {
        // The operation this surface exists for: copy one file's metadata into another
        // without transcribing it. The entries go straight from the reader to the writer.
        let mut source = MemFile::reading(fixture("kv_sorted.hrry"));
        let mut reader = reader_over(&mut source);
        let entries = kv_entries(reader);

        let descriptor = plain_2x2();
        let data = payload(16, 0x66);
        let mut sink = MemFile::writing();
        let mut writer = writer_over(&mut sink, None);
        assert_eq!(write_one(writer, "alpha", &descriptor, &[&data]), HURRAY_OK);
        assert_eq!(
            hurray_file_writer_finish(writer, entries.as_ptr(), entries.len()),
            HURRAY_OK
        );
        assert_eq!(hurray_file_writer_destroy(&mut writer), HURRAY_OK);
        // The borrowed pointers were the reader's, and are not needed past the write.
        assert_eq!(hurray_file_reader_destroy(&mut reader), HURRAY_OK);

        let mut copy = MemFile::reading(std::mem::take(&mut sink.output));
        let mut reader = reader_over(&mut copy);
        let copied = kv_entries(reader);
        let keys: Vec<&str> = copied.iter().map(|e| key_of(e)).collect();
        assert_eq!(
            keys,
            [
                "model",
                "epoch",
                "tokens",
                "temperature",
                "quantized",
                "digest",
                "layer_sizes"
            ]
        );
        assert_eq!(copied[2].value.u64_value, 1 << 40);
        assert_eq!(copied[6].value.item_count, 3);
        assert_eq!(hurray_file_reader_destroy(&mut reader), HURRAY_OK);
    }
}

#[test]
fn an_array_of_arrays_is_refused() {
    unsafe {
        let inner = [HurrayKvValue {
            kind: HURRAY_KV_ARRAY,
            ..HurrayKvValue::default()
        }];
        let key = "nested";
        let entry = HurrayKvEntry {
            key: key.as_ptr(),
            key_len: key.len(),
            value: HurrayKvValue {
                kind: HURRAY_KV_ARRAY,
                items: inner.as_ptr(),
                item_count: 1,
                ..HurrayKvValue::default()
            },
        };

        let mut sink = MemFile::writing();
        let mut writer = writer_over(&mut sink, None);
        assert_eq!(
            hurray_file_writer_finish(writer, &entry, 1),
            HURRAY_ERR_INVALID_FILE
        );
        assert_eq!(hurray_file_writer_destroy(&mut writer), HURRAY_OK);
    }
}

// ── Descriptor-only reads ─────────────────────────────────────────────────────

#[test]
fn reading_a_descriptor_does_not_read_the_data() {
    unsafe {
        let mut source = MemFile::reading(fixture("row_major.hrry"));
        let mut reader = reader_over(&mut source);
        let after_open = source.served;

        let name = "embeddings";
        let mut descriptor: *mut HurrayDescriptor = std::ptr::null_mut();
        assert_eq!(
            hurray_file_reader_read_descriptor(reader, name.as_ptr(), name.len(), &mut descriptor),
            HURRAY_OK
        );

        let mut tag: u8 = 0;
        assert_eq!(
            hurray_descriptor_element_type_tag(descriptor, &mut tag),
            HURRAY_OK
        );
        assert_eq!(tag, ElementType::Float32.tag());
        assert_eq!(hurray_descriptor_destroy(descriptor), HURRAY_OK);

        // The descriptor is well under 256 bytes, so reading it cannot have touched the
        // 256-byte data region: that is the point of asking for one.
        let descriptor_bytes = source.served - after_open;
        assert!(
            descriptor_bytes < 256,
            "a descriptor-only read served {descriptor_bytes} bytes"
        );

        assert_eq!(hurray_file_reader_destroy(&mut reader), HURRAY_OK);
    }
}

// ── Errors, and what stays usable after one ───────────────────────────────────

#[test]
fn a_name_that_is_not_in_the_index_is_not_found() {
    unsafe {
        let mut source = MemFile::reading(fixture("row_major.hrry"));
        let mut reader = reader_over(&mut source);

        let name = "absent";
        let mut tensor: *mut HurrayTensor = std::ptr::null_mut();
        assert_eq!(
            hurray_file_reader_read_tensor(reader, name.as_ptr(), name.len(), &mut tensor),
            HURRAY_ERR_NOT_FOUND
        );
        assert!(tensor.is_null());

        // A failed job leaves the reader usable: the request named nothing, so nothing
        // about the reader's position was lost (ADR-041 § 6, as pass 4a refined it).
        let mut tensor = read_tensor(reader, "embeddings");
        assert_eq!(buffer_of(tensor, 0).0, &payload(256, 0x00)[..]);
        assert_eq!(hurray_tensor_destroy(&mut tensor), HURRAY_OK);
        assert_eq!(hurray_file_reader_destroy(&mut reader), HURRAY_OK);
    }
}

#[test]
fn a_failed_range_read_is_retryable() {
    unsafe {
        let bytes = fixture("row_major.hrry");
        let mut source = MemFile::reading(bytes);
        let mut reader = reader_over(&mut source);

        // Fail the read of the data region once. The range is absolute, so re-issuing the
        // same call must read the same bytes rather than resume somewhere else.
        let data_offset = 4096u64;
        source.fail_read_at = Some(data_offset);

        let name = "embeddings";
        let mut tensor: *mut HurrayTensor = std::ptr::null_mut();
        assert_eq!(
            hurray_file_reader_read_tensor(reader, name.as_ptr(), name.len(), &mut tensor),
            HURRAY_ERR_TRANSPORT
        );
        assert!(tensor.is_null());

        let mut tensor = read_tensor(reader, "embeddings");
        assert_eq!(buffer_of(tensor, 0).0, &payload(256, 0x00)[..]);
        assert_eq!(hurray_tensor_destroy(&mut tensor), HURRAY_OK);
        assert_eq!(hurray_file_reader_destroy(&mut reader), HURRAY_OK);
    }
}

#[test]
fn bytes_that_are_not_a_container_fail_to_open() {
    unsafe {
        let mut source = MemFile::reading(vec![0x7Fu8; 4096]);
        let vt = vtable(&mut source);
        let mut reader: *mut HurrayFileReader = std::ptr::null_mut();
        assert_eq!(
            hurray_file_reader_new(&vt, &mut reader),
            HURRAY_ERR_INVALID_MAGIC
        );
        assert!(reader.is_null());
        // The vtable's ctx is released even though no handle was produced.
        assert_eq!(source.released, 1);
    }
}

#[test]
fn a_file_shorter_than_its_own_structure_fails_to_open() {
    unsafe {
        let mut source = MemFile::reading(fixture("row_major.hrry"));
        // The size the transport reports is the truth the core works from; claiming more
        // bytes than exist is how a truncated file presents.
        let real = source.bytes.len() as u64;
        source.truncate_to = Some(real + 8192);
        let vt = vtable(&mut source);
        let mut reader: *mut HurrayFileReader = std::ptr::null_mut();
        let status = hurray_file_reader_new(&vt, &mut reader);
        assert!(
            status == HURRAY_ERR_UNEXPECTED_EOF || status == HURRAY_ERR_TRANSPORT,
            "unexpected status {status}"
        );
        assert!(reader.is_null());
    }
}

#[test]
fn a_reader_needs_both_range_callbacks() {
    unsafe {
        let mut source = MemFile::reading(fixture("row_major.hrry"));
        let mut reader: *mut HurrayFileReader = std::ptr::null_mut();

        // `read` alone is a stream transport, and a file is not a stream.
        let sequential = HurrayTransport {
            ctx: &mut *source as *mut MemFile as *mut c_void,
            read_at: Some(mem_read_at),
            ..HurrayTransport::default()
        };
        assert_eq!(
            hurray_file_reader_new(&sequential, &mut reader),
            HURRAY_ERR_NULL_POINTER
        );
        assert!(reader.is_null());

        assert_eq!(
            hurray_file_reader_new(std::ptr::null(), &mut reader),
            HURRAY_ERR_NULL_POINTER
        );
    }
}

#[test]
fn a_writer_needs_a_write_callback() {
    unsafe {
        let mut sink = MemFile::writing();
        let read_only = HurrayTransport {
            ctx: &mut *sink as *mut MemFile as *mut c_void,
            read_at: Some(mem_read_at),
            size: Some(mem_size),
            ..HurrayTransport::default()
        };
        let mut writer: *mut HurrayFileWriter = std::ptr::null_mut();
        assert_eq!(
            hurray_file_writer_new(&read_only, std::ptr::null(), &mut writer),
            HURRAY_ERR_NULL_POINTER
        );
        assert!(writer.is_null());
    }
}

#[test]
fn an_out_of_range_alignment_is_refused() {
    unsafe {
        let mut sink = MemFile::writing();
        let vt = vtable(&mut sink);
        let options = HurrayFileWriterOptions {
            // Not a power of two, and below the 4096 floor.
            data_buffer_alignment: 100,
            sorted_index: false,
        };
        let mut writer: *mut HurrayFileWriter = std::ptr::null_mut();
        assert_eq!(
            hurray_file_writer_new(&vt, &options, &mut writer),
            HURRAY_ERR_INVALID_FILE
        );
        assert!(writer.is_null());
        // Nothing was written, because the header never went out.
        assert!(sink.output.is_empty());
    }
}

#[test]
fn a_duplicate_tensor_name_is_refused() {
    unsafe {
        let descriptor = plain_2x2();
        let data = payload(16, 0x01);
        let mut sink = MemFile::writing();
        let mut writer = writer_over(&mut sink, None);

        assert_eq!(
            write_one(writer, "weight", &descriptor, &[&data]),
            HURRAY_OK
        );
        assert_eq!(
            write_one(writer, "weight", &descriptor, &[&data]),
            HURRAY_ERR_INVALID_FILE
        );
        // An empty name is the same class of refusal.
        assert_eq!(
            write_one(writer, "", &descriptor, &[&data]),
            HURRAY_ERR_INVALID_FILE
        );

        // Refused before any byte went out, so the file is still completable.
        assert_eq!(
            hurray_file_writer_finish(writer, std::ptr::null(), 0),
            HURRAY_OK
        );
        assert_eq!(hurray_file_writer_destroy(&mut writer), HURRAY_OK);

        let mut source = MemFile::reading(std::mem::take(&mut sink.output));
        let mut reader = reader_over(&mut source);
        assert_eq!(tensor_names(reader), ["weight"]);
        assert_eq!(hurray_file_reader_destroy(&mut reader), HURRAY_OK);
    }
}

#[test]
fn finishing_twice_fails() {
    unsafe {
        let mut sink = MemFile::writing();
        let mut writer = writer_over(&mut sink, None);
        assert_eq!(
            hurray_file_writer_finish(writer, std::ptr::null(), 0),
            HURRAY_OK
        );
        let complete = sink.output.len();
        assert!(hurray_file_writer_finish(writer, std::ptr::null(), 0) < 0);
        // A refused second footer puts no bytes on the sink.
        assert_eq!(sink.output.len(), complete);
        assert_eq!(hurray_file_writer_destroy(&mut writer), HURRAY_OK);
    }
}

#[test]
fn a_sink_that_closes_mid_tensor_reports_sink_closed() {
    unsafe {
        let descriptor = row_major();
        let data = payload(256, 0x00);
        let mut sink = MemFile::writing();
        let mut writer = writer_over(&mut sink, None);

        // Accept the first 128 bytes of the tensor, then nothing.
        sink.close_write_at = Some(sink.output.len() + 128);
        assert_eq!(
            write_one(writer, "embeddings", &descriptor, &[&data]),
            HURRAY_ERR_SINK_CLOSED
        );
        // Poisoned: the footer cannot rescue a torn tensor.
        assert!(hurray_file_writer_finish(writer, std::ptr::null(), 0) < 0);
        assert_eq!(hurray_file_writer_destroy(&mut writer), HURRAY_OK);
    }
}

#[test]
fn a_file_write_that_fails_before_its_first_byte_is_retryable() {
    unsafe {
        let descriptor = plain_2x2();
        let data = payload(16, 0x42);
        let mut sink = MemFile::writing();
        let mut writer = writer_over(&mut sink, None);

        // The header is already out, so failing at exactly its length fails this tensor
        // before any of its bytes reach the sink. The callback's own status is reported,
        // because it says why the file stopped.
        sink.fail_write_at = Some(sink.output.len());
        assert_eq!(
            write_one(writer, "weight", &descriptor, &[&data]),
            HURRAY_ERR_TRANSPORT
        );

        // Retryable, under the same name: the failed attempt emitted nothing, so it
        // consumed neither the name nor a byte of the file (§ 6b).
        sink.fail_write_at = None;
        assert_eq!(
            write_one(writer, "weight", &descriptor, &[&data]),
            HURRAY_OK
        );
        assert_eq!(
            hurray_file_writer_finish(writer, std::ptr::null(), 0),
            HURRAY_OK
        );
        assert_eq!(hurray_file_writer_destroy(&mut writer), HURRAY_OK);

        // A complete one-tensor container, which the retracted contract said was
        // impossible.
        let mut source = MemFile::reading(std::mem::take(&mut sink.output));
        let vt = vtable(&mut source);
        let mut reader: *mut HurrayFileReader = std::ptr::null_mut();
        assert_eq!(hurray_file_reader_new(&vt, &mut reader), HURRAY_OK);
        assert!(!reader.is_null());
        let mut count: u64 = 0;
        assert_eq!(
            hurray_file_reader_tensor_count(reader, &mut count),
            HURRAY_OK
        );
        assert_eq!(count, 1);
        let mut r = reader;
        assert_eq!(hurray_file_reader_destroy(&mut r), HURRAY_OK);
    }
}

#[test]
fn a_payload_whose_length_disagrees_with_the_descriptor_is_refused() {
    unsafe {
        let descriptor = coo();
        let values = payload(32, 0x11);
        let indices = payload(64, 0x22); // declared 128

        let mut sink = MemFile::writing();
        let mut writer = writer_over(&mut sink, None);
        assert!(write_one(writer, "sparse", &descriptor, &[&values, &indices]) < 0);
        // One payload for a two-buffer tensor is the same class of refusal.
        assert!(write_one(writer, "sparse", &descriptor, &[&values]) < 0);
        assert_eq!(hurray_file_writer_destroy(&mut writer), HURRAY_OK);
    }
}

// ── Null and index guards ─────────────────────────────────────────────────────

#[test]
fn null_arguments_are_refused_rather_than_dereferenced() {
    unsafe {
        let mut source = MemFile::reading(fixture("row_major.hrry"));
        let mut reader = reader_over(&mut source);

        let mut count: u64 = 0;
        assert_eq!(
            hurray_file_reader_tensor_count(std::ptr::null(), &mut count),
            HURRAY_ERR_NULL_POINTER
        );
        assert_eq!(
            hurray_file_reader_tensor_count(reader, std::ptr::null_mut()),
            HURRAY_ERR_NULL_POINTER
        );
        assert_eq!(
            hurray_file_reader_kv_count(reader, std::ptr::null_mut()),
            HURRAY_ERR_NULL_POINTER
        );

        let mut ptr: *const u8 = std::ptr::null();
        let mut len: usize = 0;
        assert_eq!(
            hurray_file_reader_tensor_name(reader, 99, &mut ptr, &mut len),
            HURRAY_ERR_INDEX_OUT_OF_BOUNDS
        );

        let mut tensor: *mut HurrayTensor = std::ptr::null_mut();
        assert_eq!(
            hurray_file_reader_read_tensor(reader, std::ptr::null(), 4, &mut tensor),
            HURRAY_ERR_NULL_POINTER
        );

        assert_eq!(hurray_file_reader_destroy(&mut reader), HURRAY_OK);
        // A destroyed handle leaves the caller's pointer null, and destroying null again
        // is reported rather than dereferenced.
        assert!(reader.is_null());
        assert!(hurray_file_reader_destroy(&mut reader) < 0);
    }
}

/// The terminal half of § 6b through C. `chunk` forces several writes per item, so the
/// failure lands *after* bytes of the tensor are already on the sink — a torn item, which
/// the transport's own status would understate.
#[test]
fn a_file_write_that_fails_after_its_first_byte_poisons_the_writer() {
    unsafe {
        let descriptor = plain_2x2();
        let data = payload(16, 0x42);
        let mut sink = MemFile::writing();
        sink.chunk = 8;
        let mut writer = writer_over(&mut sink, None);

        // Past the header and into the tensor.
        sink.fail_write_at = Some(sink.output.len() + 16);
        assert_eq!(
            write_one(writer, "weight", &descriptor, &[&data]),
            HURRAY_ERR_SINK_CLOSED
        );

        // Terminal: the file cannot be made conforming by writing forward.
        assert!(write_one(writer, "weight", &descriptor, &[&data]) < 0);
        assert!(write_one(writer, "other", &descriptor, &[&data]) < 0);
        assert!(hurray_file_writer_finish(writer, std::ptr::null(), 0) < 0);
        assert_eq!(hurray_file_writer_destroy(&mut writer), HURRAY_OK);

        // Not a container: no trailer was ever written.
        let mut source = MemFile::reading(std::mem::take(&mut sink.output));
        let vt = vtable(&mut source);
        let mut reader: *mut HurrayFileReader = std::ptr::null_mut();
        assert!(hurray_file_reader_new(&vt, &mut reader) < 0);
        assert!(reader.is_null());
    }
}

/// ADR-041's struck open question, and the one § 6b retry reachable *only* from C — the
/// tokio `finish(self)` consumes its sink. The data is already on the sink and only the
/// index is missing, so a footer that emitted nothing is worth retrying.
#[test]
fn a_file_footer_that_fails_before_its_first_byte_may_be_retried() {
    unsafe {
        let descriptor = plain_2x2();
        let data = payload(16, 0x42);
        let mut sink = MemFile::writing();
        let mut writer = writer_over(&mut sink, None);

        assert_eq!(
            write_one(writer, "weight", &descriptor, &[&data]),
            HURRAY_OK
        );

        // The footer's very first write fails, so none of it reached the sink.
        sink.fail_write_at = Some(sink.output.len());
        assert_eq!(
            hurray_file_writer_finish(writer, std::ptr::null(), 0),
            HURRAY_ERR_TRANSPORT
        );

        // Retried, and the file is complete.
        assert_eq!(
            hurray_file_writer_finish(writer, std::ptr::null(), 0),
            HURRAY_OK
        );
        assert_eq!(hurray_file_writer_destroy(&mut writer), HURRAY_OK);

        let mut source = MemFile::reading(std::mem::take(&mut sink.output));
        let vt = vtable(&mut source);
        let mut reader: *mut HurrayFileReader = std::ptr::null_mut();
        assert_eq!(hurray_file_reader_new(&vt, &mut reader), HURRAY_OK);
        let mut count: u64 = 0;
        assert_eq!(
            hurray_file_reader_tensor_count(reader, &mut count),
            HURRAY_OK
        );
        assert_eq!(count, 1);
        let mut r = reader;
        assert_eq!(hurray_file_reader_destroy(&mut r), HURRAY_OK);
    }
}
