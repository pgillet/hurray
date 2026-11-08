//! Reading a native-protocol capsule from outside Python (ADR-043).
//!
//! A capsule carries **one** thing: a `HurrayTensor *`. The handle owns the descriptor
//! that says what the bytes are — element type, shape, layout, quantization — and the
//! buffers that hold them, so a consumer holding the pointer cannot have one without the
//! other. Before ADR-043 it was two independent fetches, a pointer and a context, paired
//! only by convention, and torn down in a required order.
//!
//! Nothing here touches Python — it is the same sequence a Go or Julia binding would
//! follow, written in Rust only because that is what this repository builds.
//!
//! Run with:
//!
//!     cargo run -p hurray-ffi --example tensor

use std::ffi::c_void;

use hurray_core::{
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, MemoryClass, Shape, SyncMode,
    TensorDescriptor, MIN_BUFFER_ALIGNMENT,
};
use hurray_ffi::{
    buffer::{hurray_buffer_byte_size, hurray_buffer_data_ptr},
    buffer_list::{hurray_buffer_list_get, hurray_buffer_list_len, hurray_buffer_list_push},
    descriptor::{
        hurray_descriptor_element_type_tag, hurray_descriptor_rank, hurray_descriptor_shape,
    },
    hurray_buffer_from_ptr, hurray_buffer_list_new, hurray_c_abi_version,
    tensor::{
        hurray_tensor_abi_version, hurray_tensor_buffers, hurray_tensor_descriptor,
        hurray_tensor_destroy, hurray_tensor_new,
    },
    HurrayBuffer, HurrayBufferList, HurrayDescriptor, HurrayTensor, HURRAY_OK,
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
/// A real producer parks its own reference here — `hurray-python` puts a Python object in
/// it. The C ABI never looks inside.
///
/// # Safety
///
/// Called once by `hurray_tensor_destroy` with the `owner` pointer.
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

    // The list is the builder: one push per descriptor buffer-table entry, then
    // hurray_tensor_new takes ownership of the finished list.
    let mut list: *mut HurrayBufferList = std::ptr::null_mut();
    assert_eq!(unsafe { hurray_buffer_list_new(1, &mut list) }, HURRAY_OK);
    assert_eq!(
        unsafe { hurray_buffer_list_push(list, buffer) },
        HURRAY_OK,
        "the list takes ownership of the handle"
    );

    let mut tensor: *mut HurrayTensor = std::ptr::null_mut();
    let status = unsafe {
        hurray_tensor_new(
            hurray_c_abi_version(),
            encoded.as_ptr(),
            encoded.len() as u64,
            list,                  // ownership transfers to the tensor
            0xBEEF as *mut c_void, // stands in for the producer's own reference
            Some(release_owner),
            &mut tensor,
        )
    };
    assert_eq!(status, HURRAY_OK);
    println!("  tensor built, ABI v{}", hurray_c_abi_version());

    // Construction decoded the descriptor, so bytes that do not parse are refused here
    // rather than one process later.
    let mut rejected: *mut HurrayTensor = std::ptr::null_mut();
    let mut spare: *mut HurrayBufferList = std::ptr::null_mut();
    assert_eq!(unsafe { hurray_buffer_list_new(0, &mut spare) }, HURRAY_OK);
    let status = unsafe {
        hurray_tensor_new(
            hurray_c_abi_version(),
            b"not a descriptor".as_ptr(),
            16,
            spare,
            std::ptr::null_mut(),
            None,
            &mut rejected,
        )
    };
    assert_ne!(status, HURRAY_OK);
    assert!(rejected.is_null());
    println!("  an unparseable descriptor is refused at construction (status {status})");
    // The list is still the caller's on failure, so it is the caller who destroys it.
    assert_eq!(
        unsafe { hurray_ffi::buffer_list::hurray_buffer_list_destroy(&mut spare) },
        HURRAY_OK
    );

    // ── Consumer: everything below could be another language ─────────────────

    println!("\n=== Consumer ===");

    // 1. The version, before anything else is trusted. An accessor added in a later ABI
    //    version would not exist in an older consumer's headers, so this check is what
    //    makes the rest safe to call.
    let mut abi_version: u32 = 0;
    assert_eq!(
        unsafe { hurray_tensor_abi_version(tensor, &mut abi_version) },
        HURRAY_OK
    );
    if abi_version != hurray_c_abi_version() {
        println!("  ABI mismatch (v{abi_version}) — refusing to dereference");
        return;
    }
    println!("  ABI v{abi_version} matches; safe to read on");

    // 2. The descriptor: a borrowed handle, owned by the tensor and already decoded.
    //    There are no descriptor bytes to hand out and nothing to decode — the fields
    //    are read straight off the handle.
    let mut decoded: *mut HurrayDescriptor = std::ptr::null_mut();
    assert_eq!(
        unsafe { hurray_tensor_descriptor(tensor, &mut decoded) },
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

    // 3. The buffers mean something, because the same handle said what they hold.
    let mut buffers: *mut HurrayBufferList = std::ptr::null_mut();
    assert_eq!(
        unsafe { hurray_tensor_buffers(tensor, &mut buffers) },
        HURRAY_OK
    );
    let mut count: u64 = 0;
    unsafe { hurray_buffer_list_len(buffers, &mut count) };
    println!("  buffers: {count}");
    for index in 0..count {
        let mut borrowed: *mut HurrayBuffer = std::ptr::null_mut();
        unsafe { hurray_buffer_list_get(buffers, index, &mut borrowed) };
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
    // One call: the descriptor, every buffer handle (running release_buffer for each),
    // and the owner reference (running release_owner). Both halves were borrows, so
    // neither is destroyed separately — and there is no order to get wrong.
    assert_eq!(unsafe { hurray_tensor_destroy(&mut tensor) }, HURRAY_OK);
    assert!(tensor.is_null(), "destroy nulls the caller's pointer");
    println!("  tensor destroyed");
}
