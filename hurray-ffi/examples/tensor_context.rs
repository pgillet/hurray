//! Reading a native-protocol capsule from outside Python (ADR-034).
//!
//! A capsule carries two things: a `HurrayBufferList` as its pointer, and a
//! `HurrayTensorContext` as its context. The list holds the bytes; the context
//! holds the descriptor that says what those bytes are.
//!
//! Before ADR-034 the context was a Rust struct private to `hurray-python`, so a
//! consumer in any other language got buffers with no element type, no shape, and
//! no layout. This example walks the path such a consumer now takes. Nothing here
//! touches Python — it is the same sequence a Go or Julia binding would follow,
//! written in Rust only because that is what this repository builds.
//!
//! Run with:
//!
//!     cargo run -p hurray-ffi --example tensor_context

use std::ffi::c_void;

use hurray_core::{
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, MemoryClass, Shape, SyncMode,
    TensorDescriptor, MIN_BUFFER_ALIGNMENT,
};
use hurray_ffi::{
    buffer::{hurray_buffer_byte_size, hurray_buffer_data_ptr},
    buffer_list::{
        hurray_buffer_list_destroy, hurray_buffer_list_get, hurray_buffer_list_len,
        hurray_buffer_list_new, hurray_buffer_list_push,
    },
    descriptor::{
        hurray_descriptor_destroy, hurray_descriptor_element_type_tag, hurray_descriptor_rank,
        hurray_descriptor_shape,
    },
    hurray_buffer_from_ptr, hurray_c_abi_version, hurray_descriptor_decode,
    tensor_context::{
        hurray_tensor_context_abi_version, hurray_tensor_context_descriptor,
        hurray_tensor_context_destroy, hurray_tensor_context_new,
    },
    HurrayBuffer, HurrayBufferList, HurrayDescriptor, HurrayTensorContext, HURRAY_OK,
};

/// Release callback for the example's heap buffer.
///
/// # Safety
///
/// Called once by `hurray_buffer_destroy` with the pointer and context given to
/// `hurray_buffer_from_ptr`.
unsafe extern "C" fn release_buffer(data: *mut c_void, _context: *mut c_void) {
    drop(Vec::from_raw_parts(data as *mut u8, 24, 24));
}

/// Release callback for whatever owns the tensor.
///
/// A real producer parks its own reference here — `hurray-python` puts a Python
/// object in it. The C ABI never looks inside.
///
/// # Safety
///
/// Called once by `hurray_tensor_context_destroy` with the `owner` pointer.
unsafe extern "C" fn release_owner(owner: *mut c_void) {
    println!("  owner released (opaque pointer {owner:p})");
}

fn main() {
    // ── Producer: a 2×3 float32 tensor ───────────────────────────────────────

    let shape = Shape::new(vec![2u64, 3]).expect("valid shape");
    let handle = BufferHandle::new(
        24, // 6 float32 elements
        MIN_BUFFER_ALIGNMENT,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )
    .expect("valid buffer handle");
    let descriptor = TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        shape,
        0,
        LayoutDescriptor::RowMajor,
        vec![handle],
        None,
        None,
        None,
        None,
    )
    .expect("valid descriptor");
    let encoded = descriptor.encode().expect("descriptor encodes");

    println!("=== Producer ===");
    println!("  descriptor: {} bytes", encoded.len());

    // The element data, leaked to the release callback.
    let mut data: Vec<u8> = vec![0u8; 24];
    let data_ptr = data.as_mut_ptr() as *mut c_void;
    std::mem::forget(data);

    let mut buffer: *mut HurrayBuffer = std::ptr::null_mut();
    let status = unsafe {
        hurray_buffer_from_ptr(
            data_ptr,
            24,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu.to_byte(),
            SyncMode::ProducerSynced.to_byte(),
            MemoryClass::Standard.to_byte(),
            Some(release_buffer),
            std::ptr::null_mut(),
            &mut buffer,
        )
    };
    assert_eq!(status, HURRAY_OK);

    let mut list: *mut HurrayBufferList = std::ptr::null_mut();
    assert_eq!(unsafe { hurray_buffer_list_new(1, &mut list) }, HURRAY_OK);
    assert_eq!(
        unsafe { hurray_buffer_list_push(list, buffer) },
        HURRAY_OK,
        "the list takes ownership of the handle"
    );

    let mut ctx: *mut HurrayTensorContext = std::ptr::null_mut();
    let status = unsafe {
        hurray_tensor_context_new(
            hurray_c_abi_version(),
            encoded.as_ptr(),
            encoded.len() as u64,
            0xBEEF as *mut c_void, // stands in for the producer's own reference
            Some(release_owner),
            &mut ctx,
        )
    };
    assert_eq!(status, HURRAY_OK);
    println!("  context built, ABI v{}", hurray_c_abi_version());

    // ── Consumer: everything below could be another language ─────────────────

    println!("\n=== Consumer ===");

    // 1. The version, before anything else is trusted. An accessor added in a
    //    later ABI version would not exist in an older consumer's headers, so
    //    this check is what makes the rest safe to call.
    let mut abi_version: u32 = 0;
    assert_eq!(
        unsafe { hurray_tensor_context_abi_version(ctx, &mut abi_version) },
        HURRAY_OK
    );
    if abi_version != hurray_c_abi_version() {
        println!("  ABI mismatch (v{abi_version}) — refusing to dereference");
        return;
    }
    println!("  ABI v{abi_version} matches; safe to read on");

    // 2. The descriptor: a borrow owned by the context, valid until it is destroyed.
    let mut bytes: *const u8 = std::ptr::null();
    let mut len: u64 = 0;
    assert_eq!(
        unsafe { hurray_tensor_context_descriptor(ctx, &mut bytes, &mut len) },
        HURRAY_OK
    );
    println!("  descriptor: {len} bytes borrowed from the context");

    // 3. Decode it — this is what was impossible before ADR-034.
    let mut decoded: *mut HurrayDescriptor = std::ptr::null_mut();
    assert_eq!(
        unsafe { hurray_descriptor_decode(bytes, len as usize, &mut decoded) },
        HURRAY_OK
    );

    let mut type_tag: u8 = 0;
    let mut rank: u32 = 0;
    unsafe {
        hurray_descriptor_element_type_tag(decoded, &mut type_tag);
        hurray_descriptor_rank(decoded, &mut rank);
    }
    let mut dims = vec![0u64; rank as usize];
    let mut capacity = dims.len();
    unsafe { hurray_descriptor_shape(decoded, dims.as_mut_ptr(), &mut capacity) };
    println!("  element type tag: 0x{type_tag:02X}  rank: {rank}  shape: {dims:?}");

    // 4. Now the buffers mean something: the descriptor said what they hold.
    let mut count: u64 = 0;
    unsafe { hurray_buffer_list_len(list, &mut count) };
    println!("  buffers: {count}");
    for index in 0..count {
        let mut borrowed: *mut HurrayBuffer = std::ptr::null_mut();
        unsafe { hurray_buffer_list_get(list, index, &mut borrowed) };
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let mut size: u64 = 0;
        unsafe {
            hurray_buffer_data_ptr(borrowed, &mut ptr);
            hurray_buffer_byte_size(borrowed, &mut size);
        }
        println!("    buffer[{index}]: {size} bytes at {ptr:p}");
    }

    // ── Teardown ─────────────────────────────────────────────────────────────

    println!("\n=== Teardown ===");
    unsafe { hurray_descriptor_destroy(decoded) };
    // Destroying the context runs release_owner; destroying the list runs
    // release_buffer for every handle it owns.
    assert_eq!(
        unsafe { hurray_tensor_context_destroy(&mut ctx) },
        HURRAY_OK
    );
    assert_eq!(unsafe { hurray_buffer_list_destroy(&mut list) }, HURRAY_OK);
    assert!(ctx.is_null() && list.is_null(), "both handles are nulled");
    println!("  context and list destroyed");
}
