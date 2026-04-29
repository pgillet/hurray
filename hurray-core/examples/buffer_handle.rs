//! Demonstrates the buffer protocol: device tags, buffer handles, and colocation.
//!
//! Run with:
//!
//! ```text
//! cargo run --example buffer_handle -p hurray-core
//! ```

use hurray_core::{
    validate_colocation, BufferHandle, DeviceTag, Error, MIN_BUFFER_ALIGNMENT, PAGE_ALIGNMENT,
};

fn main() {
    // ── Device tag round-trip ─────────────────────────────────────────────────

    println!("=== Device tag round-trip ===");
    for byte in [0x00u8, 0x01, 0x02, 0x03, 0xF0, 0xF2, 0xFE] {
        let tag = DeviceTag::from_byte(byte).unwrap();
        println!("  byte 0x{byte:02X} → {tag} → 0x{:02X}", tag.to_byte());
    }

    // ── Invalid device tags ───────────────────────────────────────────────────

    println!("\n=== Invalid device tags ===");
    for bad_byte in [0xFF, 0x04, 0x80] {
        match DeviceTag::from_byte(bad_byte) {
            Err(Error::InvalidDeviceTag(b)) => {
                println!("  0x{b:02X}: permanently invalid (sentinel)");
            }
            Err(Error::ReservedDeviceTag(b)) => {
                println!("  0x{b:02X}: reserved for future spec versions");
            }
            _ => {}
        }
    }

    // ── CPU buffer with SIMD alignment ────────────────────────────────────────

    println!("\n=== CPU buffer (SIMD alignment) ===");
    let cpu_buffer =
        BufferHandle::new(1024, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu).expect("valid CPU buffer");
    println!("  Size: {} bytes", cpu_buffer.byte_size());
    println!(
        "  Alignment: {} bytes (SIMD minimum)",
        cpu_buffer.alignment()
    );
    println!("  Device: {}", cpu_buffer.device_tag());
    println!("  Empty: {}", cpu_buffer.is_empty());

    // ── GPU buffer with page alignment ────────────────────────────────────────

    println!("\n=== CUDA buffer (page alignment) ===");
    let cuda_buffer =
        BufferHandle::new(8192, PAGE_ALIGNMENT, DeviceTag::Cuda).expect("valid CUDA buffer");
    println!("  Size: {} bytes", cuda_buffer.byte_size());
    println!(
        "  Alignment: {} bytes (page-aligned for GPU)",
        cuda_buffer.alignment()
    );
    println!("  Device: {}", cuda_buffer.device_tag());
    println!("  Empty: {}", cuda_buffer.is_empty());

    // ── Empty buffer ──────────────────────────────────────────────────────────

    println!("\n=== Empty buffer ===");
    let empty = BufferHandle::empty(DeviceTag::Rocm);
    println!("  Size: {} bytes", empty.byte_size());
    println!(
        "  Alignment: {} (allows minimal alignment for empty buffers)",
        empty.alignment()
    );
    println!("  Device: {}", empty.device_tag());
    println!("  Empty: {}", empty.is_empty());
    println!("  → Readers MUST NOT dereference the pointer of an empty buffer");

    // ── Private device tag ────────────────────────────────────────────────────

    println!("\n=== Private device tag (custom hardware) ===");
    let private_tag = DeviceTag::from_byte(0xF2).unwrap();
    let custom_buffer = BufferHandle::new(4096, MIN_BUFFER_ALIGNMENT, private_tag)
        .expect("valid custom-device buffer");
    println!("  Device tag: {}", custom_buffer.device_tag());
    println!("  Is private: {}", custom_buffer.device_tag().is_private());
    println!(
        "  Wire byte: 0x{:02X}",
        custom_buffer.device_tag().to_byte()
    );
    println!("  → Private tags must not be exchanged between independent implementations");

    // ── Invalid alignment: non-power-of-two ────────────────────────────────────

    println!("\n=== Alignment validation: non-power-of-two ===");
    if let Err(Error::AlignmentNotPowerOfTwo { alignment }) =
        BufferHandle::new(512, 63, DeviceTag::Cpu)
    {
        println!("  alignment={alignment}: rejected (not a power of two)");
    }

    // ── Invalid alignment: below SIMD minimum ─────────────────────────────────

    println!("\n=== Alignment validation: below SIMD minimum ===");
    if let Err(Error::AlignmentBelowMinimum { alignment, minimum }) =
        BufferHandle::new(512, 32, DeviceTag::Cpu)
    {
        println!(
            "  alignment={alignment}, minimum={minimum}: rejected \
             (non-empty buffers must be SIMD-aligned)"
        );
    }

    // ── Valid empty buffer with low alignment ─────────────────────────────────

    println!("\n=== Empty buffer allows any power-of-two alignment ===");
    let empty_aligned = BufferHandle::new(0, 1, DeviceTag::Metal).expect("valid empty buffer");
    println!("  Empty buffer with alignment=1: accepted");
    println!("  Byte size: {}", empty_aligned.byte_size());

    // ── Colocation validation: all same device (pass) ────────────────────────

    println!("\n=== Colocation validation: all on CPU (pass) ===");
    let handles = [
        BufferHandle::new(1024, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu).unwrap(),
        BufferHandle::new(512, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu).unwrap(),
        BufferHandle::new(256, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu).unwrap(),
    ];
    if let Ok(tag) = validate_colocation(&handles) {
        println!("  All {} buffers on: {}", handles.len(), tag);
    }

    // ── Colocation validation: mixed devices (fail) ─────────────────────────

    println!("\n=== Colocation validation: mixed devices (fail) ===");
    let mixed = [
        BufferHandle::new(1024, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu).unwrap(),
        BufferHandle::new(512, PAGE_ALIGNMENT, DeviceTag::Cuda).unwrap(),
    ];
    if let Err(Error::DeviceTagMismatch { expected, found }) = validate_colocation(&mixed) {
        println!("  Buffer 0: device 0x{expected:02X}; Buffer 1: device 0x{found:02X}",);
        println!("  → Rejected: all buffers in a tensor must reside on the same device");
    }

    // ── Colocation validation: empty list (fail) ──────────────────────────────

    println!("\n=== Colocation validation: empty list (fail) ===");
    if let Err(Error::EmptyBufferList) = validate_colocation(&[]) {
        println!("  Rejected: at least one handle required to determine colocation");
    }

    // ── Metal device (Apple Silicon) ──────────────────────────────────────────

    println!("\n=== Metal device (Apple Silicon unified memory) ===");
    let metal = BufferHandle::new(2048, PAGE_ALIGNMENT, DeviceTag::Metal).unwrap();
    println!("  Device: {}", metal.device_tag());
    println!("  Size: {}", metal.byte_size());
    println!("  Alignment: {}", metal.alignment());

    // ── ROCm device ───────────────────────────────────────────────────────────

    println!("\n=== ROCm device ===");
    let rocm = BufferHandle::new(4096, PAGE_ALIGNMENT, DeviceTag::Rocm).unwrap();
    println!("  Device: {}", rocm.device_tag());
    println!("  Size: {}", rocm.byte_size());

    println!("\n=== Summary ===");
    println!("  • Device tags identify where a buffer resides (CPU, GPU, or private)");
    println!("  • Non-empty buffers must use SIMD alignment (≥64 bytes)");
    println!("  • Page-aligned buffers (≥4096 bytes) are recommended for GPU/IPC");
    println!("  • All buffers in a tensor must reside on the same device");
    println!("  • Empty buffers allow any power-of-two alignment and are never dereferenced");
}
