//! Integration tests for the C stream surface: handles, the transport vtable, and the
//! ADR-041 §§ 4–6 rules it has to keep.
//!
//! Everything is driven through the C entry points over an in-memory transport, because
//! the properties under test are the ABI's, not the core's — the core has its own tests.

use std::ffi::c_void;

use hurray_core::{
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode, TensorDescriptor,
    MIN_BUFFER_ALIGNMENT,
};
use hurray_ffi::{
    buffer::{hurray_buffer_alignment, hurray_buffer_byte_size, hurray_buffer_data_ptr},
    buffer_list::{hurray_buffer_list_get, hurray_buffer_list_len},
    descriptor::{hurray_descriptor_buffer_count, hurray_descriptor_shape},
    hurray_descriptor_decode, hurray_reader_destroy, hurray_reader_new, hurray_reader_next_tensor,
    hurray_tensor_buffers, hurray_tensor_descriptor, hurray_tensor_destroy, hurray_writer_destroy,
    hurray_writer_new, hurray_writer_write_tensor, HurrayBuffer, HurrayBufferList,
    HurrayDescriptor, HurrayIoSlice, HurrayReader, HurrayReaderOptions, HurrayStatus, HurrayTensor,
    HurrayTransport, HurrayWriter, HurrayWriterOptions, HURRAY_END_OF_STREAM, HURRAY_ERR_INTERNAL,
    HURRAY_ERR_NULL_POINTER, HURRAY_ERR_SINK_CLOSED, HURRAY_ERR_TRANSPORT,
    HURRAY_ERR_UNEXPECTED_EOF, HURRAY_OK,
};

// ── An in-memory transport ────────────────────────────────────────────────────

/// A scriptable byte transport: feeds `input` to a reader, collects a writer's bytes in
/// `output`, and can be told to move bytes in small chunks or to fail at a chosen point.
#[derive(Default)]
struct MemTransport {
    /// Bytes a reader consumes.
    input: Vec<u8>,
    /// Read cursor into `input`.
    pos: usize,
    /// Bytes a writer produced.
    output: Vec<u8>,
    /// Maximum bytes moved per callback; `0` means "as many as offered".
    chunk: usize,
    /// Fail the read callback once, when the cursor has reached this offset.
    fail_read_at: Option<usize>,
    /// Fail the write callback once, when the sink holds this many bytes.
    fail_write_at: Option<usize>,
    /// Accept zero bytes from this sink size onwards.
    close_write_at: Option<usize>,
    /// How many times the release callback ran.
    released: u32,
}

impl MemTransport {
    fn reading(input: Vec<u8>) -> Box<Self> {
        Box::new(Self {
            input,
            ..Self::default()
        })
    }

    fn writing() -> Box<Self> {
        Box::new(Self::default())
    }
}

/// # Safety
///
/// `ctx` must be a live `MemTransport`; `dst` writable for `len`; `out_filled` writable.
unsafe extern "C" fn mem_read(
    ctx: *mut c_void,
    dst: *mut u8,
    len: usize,
    out_filled: *mut usize,
) -> HurrayStatus {
    let t = &mut *(ctx as *mut MemTransport);
    *out_filled = 0;

    if t.fail_read_at == Some(t.pos) {
        // One-shot, and no bytes moved: the contract that makes a retry sound.
        t.fail_read_at = None;
        return HURRAY_ERR_TRANSPORT;
    }

    let available = t.input.len() - t.pos;
    let mut n = len.min(available);
    if t.chunk > 0 {
        n = n.min(t.chunk);
    }
    std::ptr::copy_nonoverlapping(t.input.as_ptr().add(t.pos), dst, n);
    t.pos += n;
    *out_filled = n;
    HURRAY_OK
}

/// # Safety
///
/// `ctx` must be a live `MemTransport`; `src` readable for `len`; `out_written` writable.
unsafe extern "C" fn mem_write(
    ctx: *mut c_void,
    src: *const u8,
    len: usize,
    out_written: *mut usize,
) -> HurrayStatus {
    let t = &mut *(ctx as *mut MemTransport);
    *out_written = 0;

    if let Some(at) = t.fail_write_at {
        if t.output.len() >= at {
            t.fail_write_at = None;
            return HURRAY_ERR_TRANSPORT;
        }
    }
    if let Some(at) = t.close_write_at {
        if t.output.len() >= at {
            // Zero accepted bytes with OK: a closed sink, not a failure.
            return HURRAY_OK;
        }
    }

    let mut n = len;
    if t.chunk > 0 {
        n = n.min(t.chunk);
    }
    t.output
        .extend_from_slice(std::slice::from_raw_parts(src, n));
    *out_written = n;
    HURRAY_OK
}

/// # Safety
///
/// `ctx` must be a live `MemTransport`.
unsafe extern "C" fn mem_release(ctx: *mut c_void) {
    (*(ctx as *mut MemTransport)).released += 1;
}

fn vtable(ctx: &mut MemTransport) -> HurrayTransport {
    HurrayTransport {
        ctx: ctx as *mut MemTransport as *mut c_void,
        read: Some(mem_read),
        write: Some(mem_write),
        release: Some(mem_release),
        ..HurrayTransport::default()
    }
}

// ── Descriptors and handles ───────────────────────────────────────────────────

fn row_major(byte_size: u64, alignment: u32, sync: SyncMode) -> TensorDescriptor {
    let handle = BufferHandle::new(byte_size, alignment, DeviceTag::Cpu, sync).unwrap();
    TensorDescriptor::new(
        1,
        0,
        ElementType::Uint8,
        Shape::new(vec![byte_size]).unwrap(),
        0,
        LayoutDescriptor::RowMajor,
        vec![handle],
        None,
        None,
        None,
        None,
    )
    .unwrap()
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

unsafe fn reader_over(transport: &mut MemTransport) -> *mut HurrayReader {
    let vt = vtable(transport);
    let mut out: *mut HurrayReader = std::ptr::null_mut();
    assert_eq!(
        hurray_reader_new(&vt, std::ptr::null(), &mut out),
        HURRAY_OK
    );
    out
}

unsafe fn writer_over(transport: &mut MemTransport) -> *mut HurrayWriter {
    let vt = vtable(transport);
    let mut out: *mut HurrayWriter = std::ptr::null_mut();
    assert_eq!(
        hurray_writer_new(&vt, std::ptr::null(), &mut out),
        HURRAY_OK
    );
    out
}

unsafe fn write_one(
    writer: *mut HurrayWriter,
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
    let status = hurray_writer_write_tensor(writer, handle, slices.as_ptr(), slices.len());
    hurray_ffi::descriptor::hurray_descriptor_destroy(handle);
    status
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

/// Encodes one tensor to bytes through the C writer.
unsafe fn encode_through_c(descriptor: &TensorDescriptor, payloads: &[&[u8]]) -> Vec<u8> {
    let mut sink = MemTransport::writing();
    let mut writer = writer_over(&mut sink);
    assert_eq!(write_one(writer, descriptor, payloads), HURRAY_OK);
    assert_eq!(hurray_writer_destroy(&mut writer), HURRAY_OK);
    std::mem::take(&mut sink.output)
}

// ── Round trip ────────────────────────────────────────────────────────────────

#[test]
fn a_tensor_written_through_c_reads_back_through_c() {
    unsafe {
        let payload: Vec<u8> = (0u8..64).collect();
        let descriptor = row_major(64, MIN_BUFFER_ALIGNMENT, SyncMode::ProducerSynced);
        let wire = encode_through_c(&descriptor, &[&payload]);

        let mut source = MemTransport::reading(wire);
        let mut reader = reader_over(&mut source);

        let mut tensor: *mut HurrayTensor = std::ptr::null_mut();
        assert_eq!(hurray_reader_next_tensor(reader, &mut tensor), HURRAY_OK);

        let (bytes, _) = buffer_of(tensor, 0);
        assert_eq!(bytes, &payload[..]);

        let mut desc: *mut HurrayDescriptor = std::ptr::null_mut();
        assert_eq!(hurray_tensor_descriptor(tensor, &mut desc), HURRAY_OK);
        let mut count: u32 = 0;
        assert_eq!(hurray_descriptor_buffer_count(desc, &mut count), HURRAY_OK);
        assert_eq!(count, 1);
        let mut rank: usize = 1;
        let mut dims = [0u64; 1];
        assert_eq!(
            hurray_descriptor_shape(desc, dims.as_mut_ptr(), &mut rank),
            HURRAY_OK
        );
        assert_eq!(dims, [64]);

        assert_eq!(hurray_tensor_destroy(&mut tensor), HURRAY_OK);
        assert!(tensor.is_null());
        assert_eq!(
            hurray_reader_next_tensor(reader, &mut tensor),
            HURRAY_END_OF_STREAM
        );
        assert_eq!(hurray_reader_destroy(&mut reader), HURRAY_OK);
    }
}

#[test]
fn several_tensors_arrive_in_order() {
    unsafe {
        let descriptor = row_major(64, MIN_BUFFER_ALIGNMENT, SyncMode::ProducerSynced);
        let mut sink = MemTransport::writing();
        let mut writer = writer_over(&mut sink);
        for fill in [1u8, 2, 3] {
            let payload = vec![fill; 64];
            assert_eq!(write_one(writer, &descriptor, &[&payload]), HURRAY_OK);
        }
        assert_eq!(hurray_writer_destroy(&mut writer), HURRAY_OK);

        let mut source = MemTransport::reading(std::mem::take(&mut sink.output));
        let mut reader = reader_over(&mut source);
        for expected in [1u8, 2, 3] {
            let mut tensor: *mut HurrayTensor = std::ptr::null_mut();
            assert_eq!(hurray_reader_next_tensor(reader, &mut tensor), HURRAY_OK);
            let (bytes, _) = buffer_of(tensor, 0);
            assert!(bytes.iter().all(|b| *b == expected));
            assert_eq!(hurray_tensor_destroy(&mut tensor), HURRAY_OK);
        }
        let mut tensor: *mut HurrayTensor = std::ptr::null_mut();
        assert_eq!(
            hurray_reader_next_tensor(reader, &mut tensor),
            HURRAY_END_OF_STREAM
        );
        assert_eq!(hurray_reader_destroy(&mut reader), HURRAY_OK);
    }
}

// ── § 4: partial transfers ────────────────────────────────────────────────────

#[test]
fn a_transport_that_moves_one_byte_at_a_time_still_round_trips() {
    unsafe {
        let payload: Vec<u8> = (0u8..128).collect();
        let descriptor = row_major(128, MIN_BUFFER_ALIGNMENT, SyncMode::ProducerSynced);

        // Short writes.
        let mut sink = MemTransport::writing();
        sink.chunk = 1;
        let mut writer = writer_over(&mut sink);
        assert_eq!(write_one(writer, &descriptor, &[&payload]), HURRAY_OK);
        assert_eq!(hurray_writer_destroy(&mut writer), HURRAY_OK);

        // Short reads.
        let mut source = MemTransport::reading(std::mem::take(&mut sink.output));
        source.chunk = 1;
        let mut reader = reader_over(&mut source);
        let mut tensor: *mut HurrayTensor = std::ptr::null_mut();
        assert_eq!(hurray_reader_next_tensor(reader, &mut tensor), HURRAY_OK);
        let (bytes, _) = buffer_of(tensor, 0);
        assert_eq!(bytes, &payload[..]);
        assert_eq!(hurray_tensor_destroy(&mut tensor), HURRAY_OK);
        assert_eq!(hurray_reader_destroy(&mut reader), HURRAY_OK);
    }
}

#[test]
fn zero_bytes_read_mid_tensor_is_unexpected_eof_not_end_of_stream() {
    unsafe {
        let payload: Vec<u8> = (0u8..64).collect();
        let descriptor = row_major(64, MIN_BUFFER_ALIGNMENT, SyncMode::ProducerSynced);
        let mut wire = encode_through_c(&descriptor, &[&payload]);
        wire.truncate(wire.len() - 8);

        let mut source = MemTransport::reading(wire);
        let mut reader = reader_over(&mut source);
        let mut tensor: *mut HurrayTensor = std::ptr::null_mut();
        assert_eq!(
            hurray_reader_next_tensor(reader, &mut tensor),
            HURRAY_ERR_UNEXPECTED_EOF
        );
        assert!(tensor.is_null());
        assert_eq!(hurray_reader_destroy(&mut reader), HURRAY_OK);
    }
}

#[test]
fn an_empty_stream_is_end_of_stream_immediately() {
    unsafe {
        let mut source = MemTransport::reading(Vec::new());
        let mut reader = reader_over(&mut source);
        let mut tensor: *mut HurrayTensor = std::ptr::null_mut();
        assert_eq!(
            hurray_reader_next_tensor(reader, &mut tensor),
            HURRAY_END_OF_STREAM
        );
        assert_eq!(hurray_reader_destroy(&mut reader), HURRAY_OK);
    }
}

#[test]
fn a_sink_that_accepts_zero_bytes_mid_tensor_closes_and_poisons_the_writer() {
    unsafe {
        let payload: Vec<u8> = (0u8..64).collect();
        let descriptor = row_major(64, MIN_BUFFER_ALIGNMENT, SyncMode::ProducerSynced);

        let mut sink = MemTransport::writing();
        sink.chunk = 16;
        sink.close_write_at = Some(32);
        let mut writer = writer_over(&mut sink);
        assert_eq!(
            write_one(writer, &descriptor, &[&payload]),
            HURRAY_ERR_SINK_CLOSED
        );

        // Poisoned: a fresh item on the same writer is refused, because a truncated one is
        // already on the wire.
        assert_ne!(write_one(writer, &descriptor, &[&payload]), HURRAY_OK);
        assert_eq!(hurray_writer_destroy(&mut writer), HURRAY_OK);
    }
}

// ── § 6: a failing callback leaves the position intact ────────────────────────

#[test]
fn a_failing_read_is_retryable_and_resumes_mid_tensor() {
    unsafe {
        let payload: Vec<u8> = (0u8..64).collect();
        let descriptor = row_major(64, MIN_BUFFER_ALIGNMENT, SyncMode::ProducerSynced);
        let wire = encode_through_c(&descriptor, &[&payload]);
        let descriptor_len = wire.len() - 64;

        let mut source = MemTransport::reading(wire);
        source.chunk = 8;
        // Fail once, well inside the payload.
        source.fail_read_at = Some(descriptor_len + 16);

        let mut reader = reader_over(&mut source);
        let mut tensor: *mut HurrayTensor = std::ptr::null_mut();
        assert_eq!(
            hurray_reader_next_tensor(reader, &mut tensor),
            HURRAY_ERR_TRANSPORT
        );
        assert!(tensor.is_null());

        // The same call again: the reader resumed rather than restarted, which is only
        // observable because the bytes already consumed are not re-fed.
        assert_eq!(hurray_reader_next_tensor(reader, &mut tensor), HURRAY_OK);
        let (bytes, _) = buffer_of(tensor, 0);
        assert_eq!(bytes, &payload[..]);

        assert_eq!(hurray_tensor_destroy(&mut tensor), HURRAY_OK);
        assert_eq!(hurray_reader_destroy(&mut reader), HURRAY_OK);
    }
}

#[test]
fn a_write_that_fails_before_its_first_byte_is_retryable() {
    unsafe {
        let payload: Vec<u8> = (0u8..64).collect();
        let descriptor = row_major(64, MIN_BUFFER_ALIGNMENT, SyncMode::ProducerSynced);

        let mut sink = MemTransport::writing();
        sink.fail_write_at = Some(0);
        let mut writer = writer_over(&mut sink);
        assert_eq!(
            write_one(writer, &descriptor, &[&payload]),
            HURRAY_ERR_TRANSPORT
        );
        assert!(sink.output.is_empty());

        // Nothing was emitted, so nothing needs un-sending: the writer is still good.
        assert_eq!(write_one(writer, &descriptor, &[&payload]), HURRAY_OK);
        assert_eq!(hurray_writer_destroy(&mut writer), HURRAY_OK);

        // And what it wrote is a readable stream.
        let mut source = MemTransport::reading(std::mem::take(&mut sink.output));
        let mut reader = reader_over(&mut source);
        let mut tensor: *mut HurrayTensor = std::ptr::null_mut();
        assert_eq!(hurray_reader_next_tensor(reader, &mut tensor), HURRAY_OK);
        assert_eq!(buffer_of(tensor, 0).0, &payload[..]);
        assert_eq!(hurray_tensor_destroy(&mut tensor), HURRAY_OK);
        assert_eq!(hurray_reader_destroy(&mut reader), HURRAY_OK);
    }
}

#[test]
fn a_write_that_fails_after_its_first_byte_poisons_the_writer() {
    unsafe {
        let payload: Vec<u8> = (0u8..64).collect();
        let descriptor = row_major(64, MIN_BUFFER_ALIGNMENT, SyncMode::ProducerSynced);

        let mut sink = MemTransport::writing();
        sink.chunk = 8;
        sink.fail_write_at = Some(8);
        let mut writer = writer_over(&mut sink);
        // A torn item is on the wire, so the recoverable status is not available.
        assert_eq!(
            write_one(writer, &descriptor, &[&payload]),
            HURRAY_ERR_SINK_CLOSED
        );
        assert_ne!(write_one(writer, &descriptor, &[&payload]), HURRAY_OK);
        assert_eq!(hurray_writer_destroy(&mut writer), HURRAY_OK);
    }
}

// ── § 5: the declared alignment reaches C ─────────────────────────────────────

#[test]
fn buffers_carry_the_alignment_the_handle_declares() {
    for alignment in [64u32, 128, 4096] {
        unsafe {
            let payload: Vec<u8> = (0u8..=255).collect();
            let descriptor = row_major(256, alignment, SyncMode::ProducerSynced);
            let wire = encode_through_c(&descriptor, &[&payload]);

            let mut source = MemTransport::reading(wire);
            let mut reader = reader_over(&mut source);
            let mut tensor: *mut HurrayTensor = std::ptr::null_mut();
            assert_eq!(hurray_reader_next_tensor(reader, &mut tensor), HURRAY_OK);

            let (bytes, declared) = buffer_of(tensor, 0);
            assert_eq!(declared, alignment);
            assert_eq!(
                bytes.as_ptr() as usize % alignment as usize,
                0,
                "a buffer declared {alignment}-byte aligned must be {alignment}-byte aligned"
            );
            assert_eq!(bytes, &payload[..]);

            assert_eq!(hurray_tensor_destroy(&mut tensor), HURRAY_OK);
            assert_eq!(hurray_reader_destroy(&mut reader), HURRAY_OK);
        }
    }
}

// ── Frozen wire corpus ────────────────────────────────────────────────────────

#[test]
fn the_c_reader_decodes_the_frozen_0_1_0_fixtures() {
    // These bytes were emitted by 0.1.0's StreamWriter and must never be regenerated; see
    // hurray-io/tests/fixtures/README.md. A failure here is a wire regression.
    let fixtures = [
        ("row_major.bin", 1usize),
        ("coo.bin", 1),
        ("quantized.bin", 1),
        // Two tensors back to back.
        ("cross_machine.bin", 2),
        // A group of two whose first member is a partition of two. The C reader is flat,
        // so each composite head arrives as the buffer-less tensor it is on the wire:
        // group head, partition head, its two members, then the group's second member.
        ("nested_composite.bin", 5),
    ];
    for (name, expected_tensors) in fixtures {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../hurray-io/tests/fixtures/");
        let bytes = std::fs::read(format!("{path}{name}")).unwrap();
        unsafe {
            let mut source = MemTransport::reading(bytes);
            source.chunk = 7;
            let mut reader = reader_over(&mut source);

            let mut seen = 0usize;
            loop {
                let mut tensor: *mut HurrayTensor = std::ptr::null_mut();
                let status = hurray_reader_next_tensor(reader, &mut tensor);
                if status == HURRAY_END_OF_STREAM {
                    break;
                }
                assert_eq!(status, HURRAY_OK, "{name} failed to decode");
                let mut list: *mut HurrayBufferList = std::ptr::null_mut();
                assert_eq!(hurray_tensor_buffers(tensor, &mut list), HURRAY_OK);
                let mut len: u64 = 0;
                assert_eq!(hurray_buffer_list_len(list, &mut len), HURRAY_OK);
                for index in 0..len {
                    let (bytes, declared) = buffer_of(tensor, index);
                    assert_eq!(
                        bytes.as_ptr() as usize % declared as usize,
                        0,
                        "{name} buffer {index} is under-aligned"
                    );
                }
                assert_eq!(hurray_tensor_destroy(&mut tensor), HURRAY_OK);
                seen += 1;
            }
            assert_eq!(seen, expected_tensors, "{name} tensor count");
            assert_eq!(hurray_reader_destroy(&mut reader), HURRAY_OK);
        }
    }
}

// ── Multi-buffer ──────────────────────────────────────────────────────────────

#[test]
fn a_multi_buffer_tensor_round_trips_with_every_payload_in_order() {
    unsafe {
        let scales = BufferHandle::new(
            64,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )
        .unwrap();
        let data = BufferHandle::new(
            128,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )
        .unwrap();
        let descriptor = TensorDescriptor::new(
            1,
            0,
            ElementType::Uint8,
            Shape::new(vec![128u64]).unwrap(),
            0,
            LayoutDescriptor::RowMajor,
            vec![data, scales],
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let first = vec![0xAAu8; 128];
        let second = vec![0x55u8; 64];
        let wire = encode_through_c(&descriptor, &[&first, &second]);

        let mut source = MemTransport::reading(wire);
        let mut reader = reader_over(&mut source);
        let mut tensor: *mut HurrayTensor = std::ptr::null_mut();
        assert_eq!(hurray_reader_next_tensor(reader, &mut tensor), HURRAY_OK);

        assert_eq!(buffer_of(tensor, 0).0, &first[..]);
        assert_eq!(buffer_of(tensor, 1).0, &second[..]);

        assert_eq!(hurray_tensor_destroy(&mut tensor), HURRAY_OK);
        assert_eq!(hurray_reader_destroy(&mut reader), HURRAY_OK);
    }
}

#[test]
fn a_payload_count_that_disagrees_with_the_descriptor_is_refused_before_any_byte() {
    unsafe {
        let descriptor = row_major(64, MIN_BUFFER_ALIGNMENT, SyncMode::ProducerSynced);
        let payload = vec![0u8; 64];

        let mut sink = MemTransport::writing();
        let mut writer = writer_over(&mut sink);
        assert_ne!(
            write_one(writer, &descriptor, &[&payload, &payload]),
            HURRAY_OK
        );
        assert!(
            sink.output.is_empty(),
            "validation must happen before a byte reaches the sink"
        );
        assert_eq!(hurray_writer_destroy(&mut writer), HURRAY_OK);
    }
}

// ── Options ───────────────────────────────────────────────────────────────────

#[test]
fn a_cross_machine_writer_refuses_a_device_local_sync_mode() {
    unsafe {
        // A device-local synchronization handle means nothing to another machine, and only
        // a device buffer can carry one: a CPU buffer is ProducerSynced by construction.
        let handle =
            BufferHandle::new(64, 4096, DeviceTag::Cuda, SyncMode::ConsumerStream).unwrap();
        let descriptor = TensorDescriptor::new(
            1,
            0,
            ElementType::Uint8,
            Shape::new(vec![64u64]).unwrap(),
            0,
            LayoutDescriptor::RowMajor,
            vec![handle],
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let payload = vec![0u8; 64];

        let mut sink = MemTransport::writing();
        let vt = vtable(&mut sink);
        let options = HurrayWriterOptions {
            enforce_cross_machine_sync: true,
        };
        let mut writer: *mut HurrayWriter = std::ptr::null_mut();
        assert_eq!(hurray_writer_new(&vt, &options, &mut writer), HURRAY_OK);

        assert_ne!(write_one(writer, &descriptor, &[&payload]), HURRAY_OK);
        assert!(sink.output.is_empty());
        assert_eq!(hurray_writer_destroy(&mut writer), HURRAY_OK);
    }
}

#[test]
fn a_descriptor_limit_is_enforced_before_the_descriptor_is_decoded() {
    unsafe {
        let payload = vec![0u8; 64];
        let descriptor = row_major(64, MIN_BUFFER_ALIGNMENT, SyncMode::ProducerSynced);
        let wire = encode_through_c(&descriptor, &[&payload]);

        let mut source = MemTransport::reading(wire);
        let vt = vtable(&mut source);
        let options = HurrayReaderOptions {
            max_descriptor_bytes: 4,
            max_buffer_bytes: u64::MAX,
            enforce_cross_machine_sync: false,
        };
        let mut reader: *mut HurrayReader = std::ptr::null_mut();
        assert_eq!(hurray_reader_new(&vt, &options, &mut reader), HURRAY_OK);

        let mut tensor: *mut HurrayTensor = std::ptr::null_mut();
        assert_eq!(
            hurray_reader_next_tensor(reader, &mut tensor),
            hurray_ffi::HURRAY_ERR_LIMIT_EXCEEDED
        );
        assert_eq!(hurray_reader_destroy(&mut reader), HURRAY_OK);
    }
}

// ── Handle hygiene ────────────────────────────────────────────────────────────

#[test]
fn a_transport_without_the_callback_its_handle_needs_is_refused() {
    unsafe {
        let mut t = MemTransport::writing();
        let write_only = HurrayTransport {
            ctx: &mut *t as *mut MemTransport as *mut c_void,
            read: None,
            write: Some(mem_write),
            release: None,
            ..HurrayTransport::default()
        };
        let mut reader: *mut HurrayReader = std::ptr::null_mut();
        assert_eq!(
            hurray_reader_new(&write_only, std::ptr::null(), &mut reader),
            HURRAY_ERR_NULL_POINTER
        );

        let read_only = HurrayTransport {
            ctx: &mut *t as *mut MemTransport as *mut c_void,
            read: Some(mem_read),
            write: None,
            release: None,
            ..HurrayTransport::default()
        };
        let mut writer: *mut HurrayWriter = std::ptr::null_mut();
        assert_eq!(
            hurray_writer_new(&read_only, std::ptr::null(), &mut writer),
            HURRAY_ERR_NULL_POINTER
        );
    }
}

#[test]
fn destroying_a_handle_releases_its_transport_context_exactly_once() {
    unsafe {
        let mut source = MemTransport::reading(Vec::new());
        let mut reader = reader_over(&mut source);
        assert_eq!(source.released, 0);
        assert_eq!(hurray_reader_destroy(&mut reader), HURRAY_OK);
        assert!(reader.is_null());
        assert_eq!(source.released, 1);

        let mut sink = MemTransport::writing();
        let mut writer = writer_over(&mut sink);
        assert_eq!(hurray_writer_destroy(&mut writer), HURRAY_OK);
        assert_eq!(sink.released, 1);
    }
}

#[test]
fn a_tensor_outlives_the_reader_that_produced_it() {
    unsafe {
        let payload: Vec<u8> = (0u8..64).collect();
        let descriptor = row_major(64, MIN_BUFFER_ALIGNMENT, SyncMode::ProducerSynced);
        let wire = encode_through_c(&descriptor, &[&payload]);

        let mut source = MemTransport::reading(wire);
        let mut reader = reader_over(&mut source);
        let mut tensor: *mut HurrayTensor = std::ptr::null_mut();
        assert_eq!(hurray_reader_next_tensor(reader, &mut tensor), HURRAY_OK);
        assert_eq!(hurray_reader_destroy(&mut reader), HURRAY_OK);

        // The buffers belong to the tensor, not to the reader.
        assert_eq!(buffer_of(tensor, 0).0, &payload[..]);
        assert_eq!(hurray_tensor_destroy(&mut tensor), HURRAY_OK);
    }
}

#[test]
fn null_arguments_and_double_destroys_are_reported_not_crashes() {
    unsafe {
        let mut null_reader: *mut HurrayReader = std::ptr::null_mut();
        assert_eq!(
            hurray_reader_new(std::ptr::null(), std::ptr::null(), &mut null_reader),
            HURRAY_ERR_NULL_POINTER
        );
        assert_eq!(
            hurray_reader_destroy(std::ptr::null_mut()),
            HURRAY_ERR_NULL_POINTER
        );
        assert_eq!(hurray_reader_destroy(&mut null_reader), HURRAY_ERR_INTERNAL);

        let mut tensor: *mut HurrayTensor = std::ptr::null_mut();
        assert_eq!(
            hurray_reader_next_tensor(std::ptr::null_mut(), &mut tensor),
            HURRAY_ERR_NULL_POINTER
        );

        let payload: Vec<u8> = (0u8..64).collect();
        let descriptor = row_major(64, MIN_BUFFER_ALIGNMENT, SyncMode::ProducerSynced);
        let wire = encode_through_c(&descriptor, &[&payload]);
        let mut source = MemTransport::reading(wire);
        let mut reader = reader_over(&mut source);
        assert_eq!(hurray_reader_next_tensor(reader, &mut tensor), HURRAY_OK);
        assert_eq!(hurray_tensor_destroy(&mut tensor), HURRAY_OK);
        // Nulled by the first destroy, so a second one has nothing to free.
        assert_eq!(hurray_tensor_destroy(&mut tensor), HURRAY_ERR_INTERNAL);
        assert_eq!(hurray_reader_destroy(&mut reader), HURRAY_OK);
    }
}
