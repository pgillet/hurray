//! Example: create a CPU buffer via the Hurray C ABI, inspect its metadata,
//! and destroy it.
//!
//! Run with:
//! ```text
//! cargo run --example buffer_from_ptr -p hurray-ffi
//! ```

use std::ffi::c_void;

use hurray_core::{DeviceTag, MemoryClass, SyncMode, MIN_BUFFER_ALIGNMENT};
use hurray_ffi::{
    buffer::{
        hurray_buffer_alignment, hurray_buffer_byte_size, hurray_buffer_data_ptr,
        hurray_buffer_destroy, hurray_buffer_device_tag, hurray_buffer_memory_class,
        hurray_buffer_sync_mode,
    },
    hurray_buffer_from_ptr, hurray_c_abi_version, HurrayBuffer, HURRAY_C_ABI_VERSION, HURRAY_OK,
};

/// Release callback: prints a message and frees the heap allocation.
///
/// # Safety
///
/// `buf` must be the pointer originally passed to `hurray_buffer_from_ptr`.
/// `_ctx` is unused in this example.
unsafe extern "C" fn on_release(buf: *mut c_void, _ctx: *mut c_void) {
    println!("buffer released");
    // Reconstruct the Vec so Rust drops and frees it.
    // SAFETY: buf was created via Vec::into_raw_parts (see below), and this
    // callback is called exactly once from hurray_buffer_destroy.
    let _ = Vec::from_raw_parts(buf as *mut u8, 1024, 1024);
}

fn main() {
    // ── ABI version check ─────────────────────────────────────────────────────
    let abi_ver = hurray_c_abi_version();
    println!("Hurray C ABI version: {abi_ver}");
    // Against the constant, not a literal: a literal goes stale the next time the
    // ABI is bumped, and says nothing useful when it does.
    assert_eq!(
        abi_ver, HURRAY_C_ABI_VERSION,
        "hurray_c_abi_version() must report the compiled-in ABI version"
    );

    // ── Create a 1 KB CPU buffer on the heap ──────────────────────────────────
    // Leak the Vec so the raw pointer remains valid until the release callback.
    let mut storage: Vec<u8> = vec![0u8; 1024];
    let data_ptr: *mut c_void = storage.as_mut_ptr() as *mut c_void;
    // Forget the Vec — ownership is transferred to the release callback.
    std::mem::forget(storage);

    let mut handle: *mut HurrayBuffer = std::ptr::null_mut();
    let status = unsafe {
        hurray_buffer_from_ptr(
            data_ptr,
            1024,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu.to_byte(),
            SyncMode::ProducerSynced.to_byte(),
            MemoryClass::Standard.to_byte(),
            Some(on_release),
            std::ptr::null_mut(), // no release context needed
            &mut handle,
        )
    };
    assert_eq!(status, HURRAY_OK, "hurray_buffer_from_ptr failed: {status}");
    println!("buffer created: handle={handle:?}");

    // ── Read accessors ────────────────────────────────────────────────────────
    let mut byte_size: u64 = 0;
    let mut alignment: u32 = 0;
    let mut device_tag: u8 = 0;
    let mut sync_mode: u8 = 0;
    let mut memory_class: u8 = 0;
    let mut ptr_out: *mut c_void = std::ptr::null_mut();

    unsafe {
        hurray_buffer_byte_size(handle, &mut byte_size);
        hurray_buffer_alignment(handle, &mut alignment);
        hurray_buffer_device_tag(handle, &mut device_tag);
        hurray_buffer_sync_mode(handle, &mut sync_mode);
        hurray_buffer_memory_class(handle, &mut memory_class);
        hurray_buffer_data_ptr(handle, &mut ptr_out);
    }

    println!("  byte_size    = {byte_size}");
    println!("  alignment    = {alignment}");
    println!("  device_tag   = 0x{device_tag:02X} (cpu=0x00)");
    println!("  sync_mode    = 0x{sync_mode:02X} (producer_synced=0x00)");
    println!("  memory_class = 0x{memory_class:02X} (standard=0x00)");
    println!("  data_ptr     = {ptr_out:?}");

    // ── Destroy the handle (triggers release callback) ────────────────────────
    let destroy_status = unsafe { hurray_buffer_destroy(handle) };
    assert_eq!(
        destroy_status, HURRAY_OK,
        "hurray_buffer_destroy failed: {destroy_status}"
    );

    println!("done");
}
