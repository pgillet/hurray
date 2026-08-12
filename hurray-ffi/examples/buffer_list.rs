//! Demonstrates the multi-buffer C ABI carrier: `HurrayBufferList` (ADR-030).
//!
//! A tensor whose descriptor references more than one buffer — per-channel /
//! NF4 / MXFP quantization, sparse layouts, block-paged — needs all of its
//! buffers to travel together. The list is that carrier: one owner, one destroy,
//! borrowed access to each element.
//!
//! Run with:
//!
//! ```text
//! cargo run --example buffer_list -p hurray-ffi
//! ```

use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

use hurray_ffi::buffer::{hurray_buffer_byte_size, hurray_buffer_from_ptr};
use hurray_ffi::buffer_list::{
    hurray_buffer_list_destroy, hurray_buffer_list_get, hurray_buffer_list_len,
    hurray_buffer_list_new, hurray_buffer_list_push,
};
use hurray_ffi::status::HURRAY_ERR_INDEX_OUT_OF_BOUNDS;
use hurray_ffi::{HurrayBuffer, HurrayBufferList, HURRAY_C_ABI_VERSION, HURRAY_OK};

/// Buffers must be 64-byte aligned (SIMD minimum), so allocate them that way.
#[repr(align(64))]
struct Aligned([u8; 64]);

/// Counts release-callback invocations so the example can show that destroying
/// the list releases every buffer exactly once.
static RELEASES: AtomicUsize = AtomicUsize::new(0);

unsafe extern "C" fn on_release(_data: *mut c_void, _ctx: *mut c_void) {
    RELEASES.fetch_add(1, Ordering::SeqCst);
}

fn make_buffer(data: &mut Aligned, byte_size: u64) -> *mut HurrayBuffer {
    let mut handle: *mut HurrayBuffer = std::ptr::null_mut();
    // SAFETY: data is 64-byte aligned and at least byte_size long; the out-pointer
    // is a valid stack variable.
    let status = unsafe {
        hurray_buffer_from_ptr(
            data.0.as_mut_ptr().cast(),
            byte_size,
            64,
            0x00, // device_tag: CPU
            0x00, // sync_mode: ProducerSynced
            0x00, // memory_class: Standard
            Some(on_release),
            std::ptr::null_mut(),
            &mut handle,
        )
    };
    assert_eq!(status, HURRAY_OK);
    handle
}

fn main() {
    println!("=== C ABI version ===");
    println!("  HURRAY_C_ABI_VERSION = {HURRAY_C_ABI_VERSION}");
    println!("  (ADR-030 raised this 2 -> 3: the capsule now wraps a list, not a buffer)");

    // ── Build a two-buffer tensor's worth of handles ──────────────────────────
    //
    // Buffer 0 is the quantized weight data, buffer 1 the per-channel scales —
    // the arrangement a PerChannelAffine descriptor's scale_buffer_index refers to.

    let mut weights = Aligned([0xAB; 64]);
    let mut scales = Aligned([0x01; 64]);

    let mut list: *mut HurrayBufferList = std::ptr::null_mut();
    // SAFETY: out-pointer is a valid stack variable.
    assert_eq!(unsafe { hurray_buffer_list_new(2, &mut list) }, HURRAY_OK);

    // Push order IS descriptor buffer-table order (ADR-030 § 3).
    for (label, handle) in [
        ("weights", make_buffer(&mut weights, 64)),
        ("scales", make_buffer(&mut scales, 32)),
    ] {
        // SAFETY: both handles are live; push transfers ownership to the list.
        assert_eq!(unsafe { hurray_buffer_list_push(list, handle) }, HURRAY_OK);
        println!("  pushed {label}");
    }

    // ── Read it back ──────────────────────────────────────────────────────────

    println!("\n=== Reading the list ===");
    let mut len: u64 = 0;
    // SAFETY: list is live; out-pointer is a valid stack variable.
    unsafe { hurray_buffer_list_len(list, &mut len) };
    println!("  length = {len}");

    for index in 0..len {
        let mut borrowed: *mut HurrayBuffer = std::ptr::null_mut();
        // SAFETY: list is live and index < len.
        let status = unsafe { hurray_buffer_list_get(list, index, &mut borrowed) };
        assert_eq!(status, HURRAY_OK);

        let mut byte_size: u64 = 0;
        // SAFETY: borrowed is a live handle owned by the list.
        unsafe { hurray_buffer_byte_size(borrowed, &mut byte_size) };
        println!("  buffer[{index}]: {byte_size} bytes");

        // NOTE: `borrowed` must NOT be destroyed here. The list owns it; destroying
        // it individually would double-free when the list is destroyed.
    }

    // ── Bounds checking ───────────────────────────────────────────────────────

    println!("\n=== Out-of-range access ===");
    let mut missing: *mut HurrayBuffer = std::ptr::null_mut();
    // SAFETY: list is live; the out-pointer is valid. Index 2 is past the end.
    let status = unsafe { hurray_buffer_list_get(list, 2, &mut missing) };
    assert_eq!(status, HURRAY_ERR_INDEX_OUT_OF_BOUNDS);
    println!("  index 2 of a 2-element list -> HURRAY_ERR_INDEX_OUT_OF_BOUNDS");

    // ── Destroy ───────────────────────────────────────────────────────────────

    println!("\n=== Destroying the list ===");
    // SAFETY: list was created by hurray_buffer_list_new; first and only destroy.
    assert_eq!(unsafe { hurray_buffer_list_destroy(&mut list) }, HURRAY_OK);
    println!(
        "  release callbacks fired: {}",
        RELEASES.load(Ordering::SeqCst)
    );
    println!("  caller's pointer is now null: {}", list.is_null());

    // Destroy nulls the caller's pointer, so cleanup paths are idempotent.
    // SAFETY: *list is null, which the destructor treats as a no-op.
    assert_eq!(unsafe { hurray_buffer_list_destroy(&mut list) }, HURRAY_OK);
    println!("  second destroy is a safe no-op");
}
