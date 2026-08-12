//! Integration tests for the Hurray C ABI layer.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};

use hurray_core::{
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, MemoryClass, Shape, SyncMode,
    TensorDescriptor, MIN_BUFFER_ALIGNMENT,
};
use hurray_ffi::{
    buffer::{
        hurray_buffer_alignment, hurray_buffer_byte_size, hurray_buffer_data_ptr,
        hurray_buffer_destroy, hurray_buffer_device_tag, hurray_buffer_memory_class,
        hurray_buffer_sync_mode,
    },
    descriptor::{
        hurray_descriptor_buffer_count, hurray_descriptor_byte_offset, hurray_descriptor_destroy,
        hurray_descriptor_element_type_tag, hurray_descriptor_layout_tag, hurray_descriptor_rank,
        hurray_descriptor_shape,
    },
    hurray_buffer_from_ptr, hurray_c_abi_version, hurray_descriptor_decode,
    sync::{
        hurray_buffer_handoff_consumer_stream, hurray_buffer_handoff_event,
        hurray_buffer_handoff_producer_synced, HurraySyncConsumerStreamPayload,
        HurraySyncEventPayload,
    },
    HurrayBuffer, HURRAY_ERR_BUFFER_TOO_SMALL, HURRAY_ERR_INTERNAL, HURRAY_ERR_INTERNAL_PANIC,
    HURRAY_ERR_INVALID_LAYOUT, HURRAY_ERR_INVALID_MAGIC, HURRAY_ERR_INVALID_SYNC_MODE,
    HURRAY_ERR_INVALID_TYPE, HURRAY_ERR_NULL_POINTER, HURRAY_ERR_SYNC_MODE_MISMATCH,
    HURRAY_ERR_VERSION_MISMATCH, HURRAY_OK,
};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Creates a CPU buffer handle via the C ABI from a Vec<u8>.
///
/// The vec must outlive the handle. Returns the raw `*mut HurrayBuffer`.
unsafe fn make_cpu_buffer(v: &mut Vec<u8>) -> *mut HurrayBuffer {
    let mut out: *mut HurrayBuffer = std::ptr::null_mut();
    let status = hurray_buffer_from_ptr(
        v.as_mut_ptr() as *mut c_void,
        v.len() as u64,
        MIN_BUFFER_ALIGNMENT,
        DeviceTag::Cpu.to_byte(),
        SyncMode::ProducerSynced.to_byte(),
        MemoryClass::Standard.to_byte(),
        None,
        std::ptr::null_mut(),
        &mut out,
    );
    assert_eq!(
        status, HURRAY_OK,
        "make_cpu_buffer failed with status {status}"
    );
    out
}

/// Encodes a minimal float32 [2,3] row-major descriptor.
fn encode_simple_descriptor() -> Vec<u8> {
    let shape = Shape::new(vec![2u64, 3]).unwrap();
    let buf = BufferHandle::new(
        64,
        MIN_BUFFER_ALIGNMENT,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )
    .unwrap();
    let desc = TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        shape,
        0,
        LayoutDescriptor::RowMajor,
        vec![buf],
        None,
        None,
        None,
        None,
    )
    .unwrap();
    desc.encode().unwrap()
}

/// Encodes a rank-3 float32 descriptor for shape-roundtrip tests.
fn encode_rank3_descriptor() -> Vec<u8> {
    let shape = Shape::new(vec![4u64, 5, 6]).unwrap();
    let buf = BufferHandle::new(
        64,
        MIN_BUFFER_ALIGNMENT,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )
    .unwrap();
    let desc = TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        shape,
        0,
        LayoutDescriptor::RowMajor,
        vec![buf],
        None,
        None,
        None,
        None,
    )
    .unwrap();
    desc.encode().unwrap()
}

// ── Phase 1 — ABI version and status codes ───────────────────────────────────

#[test]
fn abi_version_is_3() {
    // Raised 2 -> 3 by ADR-030: the native buffer capsule now wraps a
    // HurrayBufferList, so a v2 consumer must be told rather than dereference it.
    assert_eq!(hurray_c_abi_version(), 3);
}

#[test]
fn status_codes_have_expected_values() {
    assert_eq!(HURRAY_OK, 0);
    assert_eq!(HURRAY_ERR_INVALID_MAGIC, -1);
    assert_eq!(HURRAY_ERR_VERSION_MISMATCH, -2);
    assert_eq!(HURRAY_ERR_INVALID_LAYOUT, -3);
    assert_eq!(HURRAY_ERR_INVALID_TYPE, -4);
    assert_eq!(HURRAY_ERR_BUFFER_TOO_SMALL, -5);
    assert_eq!(HURRAY_ERR_NULL_POINTER, -6);
    assert_eq!(HURRAY_ERR_INTERNAL_PANIC, -7);
    assert_eq!(HURRAY_ERR_INVALID_SYNC_MODE, -8);
    assert_eq!(HURRAY_ERR_SYNC_MODE_MISMATCH, -9);
    assert_eq!(HURRAY_ERR_INTERNAL, -10);
}

// ── Phase 2 — Buffer handle ───────────────────────────────────────────────────

#[test]
fn buffer_from_ptr_null_data_returns_null_pointer() {
    let mut out: *mut HurrayBuffer = std::ptr::null_mut();
    let status = unsafe {
        hurray_buffer_from_ptr(
            std::ptr::null_mut(),
            1024,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu.to_byte(),
            SyncMode::ProducerSynced.to_byte(),
            MemoryClass::Standard.to_byte(),
            None,
            std::ptr::null_mut(),
            &mut out,
        )
    };
    assert_eq!(status, HURRAY_ERR_NULL_POINTER);
}

#[test]
fn buffer_from_ptr_null_out_handle_returns_null_pointer() {
    let mut v = vec![0u8; 1024];
    let status = unsafe {
        hurray_buffer_from_ptr(
            v.as_mut_ptr() as *mut c_void,
            1024,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu.to_byte(),
            SyncMode::ProducerSynced.to_byte(),
            MemoryClass::Standard.to_byte(),
            None,
            std::ptr::null_mut(),
            std::ptr::null_mut(), // null out_handle
        )
    };
    assert_eq!(status, HURRAY_ERR_NULL_POINTER);
}

#[test]
fn buffer_from_ptr_invalid_device_tag_rejected() {
    let mut v = vec![0u8; 1024];
    let mut out: *mut HurrayBuffer = std::ptr::null_mut();
    let status = unsafe {
        hurray_buffer_from_ptr(
            v.as_mut_ptr() as *mut c_void,
            1024,
            MIN_BUFFER_ALIGNMENT,
            0x04, // Vulkan — was reserved in spec, but 0x09 is the first reserved range
            SyncMode::ProducerSynced.to_byte(),
            MemoryClass::Standard.to_byte(),
            None,
            std::ptr::null_mut(),
            &mut out,
        )
    };
    // 0x04 is Vulkan (valid), so use a byte in the reserved range 0x09..=0xEF
    let _ = status; // discard

    let mut out2: *mut HurrayBuffer = std::ptr::null_mut();
    let status2 = unsafe {
        hurray_buffer_from_ptr(
            v.as_mut_ptr() as *mut c_void,
            1024,
            MIN_BUFFER_ALIGNMENT,
            0x09, // reserved device tag → INVALID_TYPE
            SyncMode::ProducerSynced.to_byte(),
            MemoryClass::Standard.to_byte(),
            None,
            std::ptr::null_mut(),
            &mut out2,
        )
    };
    assert_eq!(status2, HURRAY_ERR_INVALID_TYPE);
}

#[test]
fn buffer_from_ptr_invalid_sync_mode_rejected() {
    let mut v = vec![0u8; 1024];
    let mut out: *mut HurrayBuffer = std::ptr::null_mut();
    let status = unsafe {
        hurray_buffer_from_ptr(
            v.as_mut_ptr() as *mut c_void,
            1024,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu.to_byte(),
            0x03, // reserved sync mode → INVALID_SYNC_MODE
            MemoryClass::Standard.to_byte(),
            None,
            std::ptr::null_mut(),
            &mut out,
        )
    };
    assert_eq!(status, HURRAY_ERR_INVALID_SYNC_MODE);
}

// Release-callback and double-free test (debug builds only).
static RELEASE_COUNT: AtomicU32 = AtomicU32::new(0);

unsafe extern "C" fn count_release(_buf: *mut c_void, _ctx: *mut c_void) {
    RELEASE_COUNT.fetch_add(1, Ordering::SeqCst);
}

#[test]
fn buffer_lifecycle_and_release_callback() {
    RELEASE_COUNT.store(0, Ordering::SeqCst);

    let mut v = vec![0u8; 4096];
    let mut out: *mut HurrayBuffer = std::ptr::null_mut();
    let status = unsafe {
        hurray_buffer_from_ptr(
            v.as_mut_ptr() as *mut c_void,
            v.len() as u64,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu.to_byte(),
            SyncMode::ProducerSynced.to_byte(),
            MemoryClass::Standard.to_byte(),
            Some(count_release),
            std::ptr::null_mut(),
            &mut out,
        )
    };
    assert_eq!(status, HURRAY_OK);
    assert!(!out.is_null());

    // Destroy once — callback should fire.
    let destroy_status = unsafe { hurray_buffer_destroy(out) };
    assert_eq!(destroy_status, HURRAY_OK);
    assert_eq!(RELEASE_COUNT.load(Ordering::SeqCst), 1);
    // Note: a second destroy is not tested here because the raw pointer is
    // dangling after the first destroy — reading from it to check the sentinel
    // is UB once the allocator has reclaimed the slot.  The sentinel write in
    // hurray_buffer_destroy is best-effort for callers who race before reclaim.
}

#[test]
fn buffer_accessors_roundtrip() {
    // 4096-byte CUDA buffer, Event sync, Standard memory class, 4096-byte alignment.
    let mut v = vec![0u8; 4096];
    let mut out: *mut HurrayBuffer = std::ptr::null_mut();
    let status = unsafe {
        hurray_buffer_from_ptr(
            v.as_mut_ptr() as *mut c_void,
            4096,
            4096,                            // PAGE_ALIGNMENT
            DeviceTag::Cuda.to_byte(),       // 0x01
            SyncMode::Event.to_byte(),       // 0x01
            MemoryClass::Standard.to_byte(), // 0x00
            None,
            std::ptr::null_mut(),
            &mut out,
        )
    };
    assert_eq!(status, HURRAY_OK);

    let mut byte_size: u64 = 0;
    let mut alignment: u32 = 0;
    let mut device_tag: u8 = 0xFF;
    let mut sync_mode: u8 = 0xFF;
    let mut memory_class: u8 = 0xFF;
    let mut data_ptr: *mut c_void = std::ptr::null_mut();

    unsafe {
        assert_eq!(hurray_buffer_byte_size(out, &mut byte_size), HURRAY_OK);
        assert_eq!(hurray_buffer_alignment(out, &mut alignment), HURRAY_OK);
        assert_eq!(hurray_buffer_device_tag(out, &mut device_tag), HURRAY_OK);
        assert_eq!(hurray_buffer_sync_mode(out, &mut sync_mode), HURRAY_OK);
        assert_eq!(
            hurray_buffer_memory_class(out, &mut memory_class),
            HURRAY_OK
        );
        assert_eq!(hurray_buffer_data_ptr(out, &mut data_ptr), HURRAY_OK);
    }

    assert_eq!(byte_size, 4096);
    assert_eq!(alignment, 4096);
    assert_eq!(device_tag, DeviceTag::Cuda.to_byte());
    assert_eq!(sync_mode, SyncMode::Event.to_byte());
    assert_eq!(memory_class, MemoryClass::Standard.to_byte());
    assert_eq!(data_ptr, v.as_mut_ptr() as *mut c_void);

    unsafe { hurray_buffer_destroy(out) };
}

// ── Phase 3 — Descriptor handle ───────────────────────────────────────────────

#[test]
fn descriptor_decode_null_bytes_rejected() {
    let mut out: *mut hurray_ffi::HurrayDescriptor = std::ptr::null_mut();
    let status = unsafe { hurray_descriptor_decode(std::ptr::null(), 0, &mut out) };
    assert_eq!(status, HURRAY_ERR_NULL_POINTER);
}

#[test]
fn descriptor_decode_malformed_bytes_rejected() {
    let garbage = [0x00u8, 0xDE, 0xAD, 0xBE, 0xEF];
    let mut out: *mut hurray_ffi::HurrayDescriptor = std::ptr::null_mut();
    let status = unsafe { hurray_descriptor_decode(garbage.as_ptr(), garbage.len(), &mut out) };
    // Either bad magic or layout/truncation error depending on content.
    assert!(
        status == HURRAY_ERR_INVALID_MAGIC || status == HURRAY_ERR_INVALID_LAYOUT,
        "unexpected status {status}"
    );
}

#[test]
fn descriptor_decode_and_accessors() {
    let encoded = encode_simple_descriptor();
    let mut out: *mut hurray_ffi::HurrayDescriptor = std::ptr::null_mut();
    let status = unsafe { hurray_descriptor_decode(encoded.as_ptr(), encoded.len(), &mut out) };
    assert_eq!(status, HURRAY_OK);
    assert!(!out.is_null());

    let mut rank: u32 = 0xFF;
    let mut type_tag: u8 = 0xFF;
    let mut layout_tag: u8 = 0xFF;
    let mut byte_offset: u64 = 0xFFFF;
    let mut buf_count: u32 = 0xFF;

    unsafe {
        assert_eq!(hurray_descriptor_rank(out, &mut rank), HURRAY_OK);
        assert_eq!(
            hurray_descriptor_element_type_tag(out, &mut type_tag),
            HURRAY_OK
        );
        assert_eq!(
            hurray_descriptor_layout_tag(out, &mut layout_tag),
            HURRAY_OK
        );
        assert_eq!(
            hurray_descriptor_byte_offset(out, &mut byte_offset),
            HURRAY_OK
        );
        assert_eq!(
            hurray_descriptor_buffer_count(out, &mut buf_count),
            HURRAY_OK
        );
    }

    assert_eq!(rank, 2); // shape [2, 3]
    assert_eq!(type_tag, ElementType::Float32.tag());
    assert_eq!(layout_tag, LayoutDescriptor::RowMajor.tag());
    assert_eq!(byte_offset, 0);
    assert_eq!(buf_count, 1);

    unsafe { hurray_descriptor_destroy(out) };
}

#[test]
fn descriptor_shape_roundtrip() {
    let encoded = encode_rank3_descriptor();
    let mut out: *mut hurray_ffi::HurrayDescriptor = std::ptr::null_mut();
    let status = unsafe { hurray_descriptor_decode(encoded.as_ptr(), encoded.len(), &mut out) };
    assert_eq!(status, HURRAY_OK);

    let mut dims = [0u64; 3];
    let mut rank: usize = 3; // capacity = 3
    let status = unsafe { hurray_descriptor_shape(out, dims.as_mut_ptr(), &mut rank) };
    assert_eq!(status, HURRAY_OK);
    assert_eq!(rank, 3);
    assert_eq!(dims, [4, 5, 6]);

    unsafe { hurray_descriptor_destroy(out) };
}

#[test]
fn descriptor_shape_buffer_too_small() {
    let encoded = encode_rank3_descriptor();
    let mut out: *mut hurray_ffi::HurrayDescriptor = std::ptr::null_mut();
    unsafe { hurray_descriptor_decode(encoded.as_ptr(), encoded.len(), &mut out) };

    let mut rank: usize = 0; // capacity = 0 → too small
    let status = unsafe { hurray_descriptor_shape(out, std::ptr::null_mut(), &mut rank) };
    assert_eq!(status, HURRAY_ERR_BUFFER_TOO_SMALL);
    assert_eq!(
        rank, 3,
        "true rank must be written even on BUFFER_TOO_SMALL"
    );

    unsafe { hurray_descriptor_destroy(out) };
}

// ── Phase 4 — Sync-mode handoffs ─────────────────────────────────────────────

/// Creates a CUDA buffer with the given sync mode via the C ABI.
unsafe fn make_cuda_buffer(v: &mut Vec<u8>, sync_mode_byte: u8) -> *mut HurrayBuffer {
    let mut out: *mut HurrayBuffer = std::ptr::null_mut();
    let status = hurray_buffer_from_ptr(
        v.as_mut_ptr() as *mut c_void,
        v.len() as u64,
        MIN_BUFFER_ALIGNMENT,
        DeviceTag::Cuda.to_byte(), // 0x01
        sync_mode_byte,
        MemoryClass::Standard.to_byte(),
        None,
        std::ptr::null_mut(),
        &mut out,
    );
    assert_eq!(status, HURRAY_OK);
    out
}

// Dummy event-release function for tests.
unsafe extern "C" fn noop_event_release(_handle: *mut c_void, _ctx: *mut c_void) {}

#[test]
fn sync_handoff_producer_synced_pass() {
    let mut v = vec![0u8; 4096];
    let buf = unsafe { make_cpu_buffer(&mut v) };
    let status = unsafe { hurray_buffer_handoff_producer_synced(buf) };
    assert_eq!(status, HURRAY_OK);
    unsafe { hurray_buffer_destroy(buf) };
}

#[test]
fn sync_handoff_producer_synced_wrong_mode() {
    let mut v = vec![0u8; 4096];
    let buf = unsafe { make_cuda_buffer(&mut v, SyncMode::Event.to_byte()) };
    let status = unsafe { hurray_buffer_handoff_producer_synced(buf) };
    assert_eq!(status, HURRAY_ERR_SYNC_MODE_MISMATCH);
    unsafe { hurray_buffer_destroy(buf) };
}

#[test]
fn sync_handoff_event_pass() {
    let mut v = vec![0u8; 4096];
    let buf = unsafe { make_cuda_buffer(&mut v, SyncMode::Event.to_byte()) };

    // Use a non-null dummy value as the event handle.
    let fake_event: *mut c_void = &mut 42u64 as *mut u64 as *mut c_void;
    let payload = HurraySyncEventPayload {
        sync_handle: fake_event,
        sync_handle_device_tag: DeviceTag::Cuda.to_byte(),
        event_release_fn: Some(noop_event_release),
        event_release_context: std::ptr::null_mut(),
    };
    let status = unsafe { hurray_buffer_handoff_event(buf, &payload) };
    assert_eq!(status, HURRAY_OK);
    unsafe { hurray_buffer_destroy(buf) };
}

#[test]
fn sync_handoff_event_null_sync_handle() {
    let mut v = vec![0u8; 4096];
    let buf = unsafe { make_cuda_buffer(&mut v, SyncMode::Event.to_byte()) };

    let payload = HurraySyncEventPayload {
        sync_handle: std::ptr::null_mut(), // null → mismatch
        sync_handle_device_tag: DeviceTag::Cuda.to_byte(),
        event_release_fn: Some(noop_event_release),
        event_release_context: std::ptr::null_mut(),
    };
    let status = unsafe { hurray_buffer_handoff_event(buf, &payload) };
    assert_eq!(status, HURRAY_ERR_SYNC_MODE_MISMATCH);
    unsafe { hurray_buffer_destroy(buf) };
}

#[test]
fn sync_handoff_event_device_tag_mismatch() {
    let mut v = vec![0u8; 4096];
    let buf = unsafe { make_cuda_buffer(&mut v, SyncMode::Event.to_byte()) };

    let fake_event: *mut c_void = &mut 42u64 as *mut u64 as *mut c_void;
    let payload = HurraySyncEventPayload {
        sync_handle: fake_event,
        sync_handle_device_tag: DeviceTag::Rocm.to_byte(), // wrong device
        event_release_fn: Some(noop_event_release),
        event_release_context: std::ptr::null_mut(),
    };
    let status = unsafe { hurray_buffer_handoff_event(buf, &payload) };
    assert_eq!(status, HURRAY_ERR_SYNC_MODE_MISMATCH);
    unsafe { hurray_buffer_destroy(buf) };
}

#[test]
fn sync_handoff_consumer_stream_pass() {
    let mut v = vec![0u8; 4096];
    let buf = unsafe { make_cuda_buffer(&mut v, SyncMode::ConsumerStream.to_byte()) };

    let fake_stream: *mut c_void = &mut 99u64 as *mut u64 as *mut c_void;
    let payload = HurraySyncConsumerStreamPayload {
        consumer_stream: fake_stream,
        consumer_stream_device_tag: DeviceTag::Cuda.to_byte(),
    };
    let status = unsafe { hurray_buffer_handoff_consumer_stream(buf, &payload) };
    assert_eq!(status, HURRAY_OK);
    unsafe { hurray_buffer_destroy(buf) };
}

#[test]
fn sync_handoff_consumer_stream_null_stream() {
    let mut v = vec![0u8; 4096];
    let buf = unsafe { make_cuda_buffer(&mut v, SyncMode::ConsumerStream.to_byte()) };

    let payload = HurraySyncConsumerStreamPayload {
        consumer_stream: std::ptr::null_mut(), // null → mismatch
        consumer_stream_device_tag: DeviceTag::Cuda.to_byte(),
    };
    let status = unsafe { hurray_buffer_handoff_consumer_stream(buf, &payload) };
    assert_eq!(status, HURRAY_ERR_SYNC_MODE_MISMATCH);
    unsafe { hurray_buffer_destroy(buf) };
}
