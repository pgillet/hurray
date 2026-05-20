//! Demonstrates the buffer protocol: device tags, memory classes, buffer handles, and colocation.
//!
//! Run with:
//!
//! ```text
//! cargo run --example buffer_handle -p hurray-core
//! ```

use hurray_core::{
    validate_colocation, BufferHandle, DeviceTag, Error, MemoryClass, SyncMode,
    MIN_BUFFER_ALIGNMENT, PAGE_ALIGNMENT,
};

fn main() {
    // ── Device tag round-trip ─────────────────────────────────────────────────

    println!("=== Device tag round-trip ===");
    for byte in [
        0x00u8, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0xF0, 0xF2, 0xFE,
    ] {
        let tag = DeviceTag::from_byte(byte).unwrap();
        println!("  byte 0x{byte:02X} → {tag} → 0x{:02X}", tag.to_byte());
    }

    // ── Invalid device tags ───────────────────────────────────────────────────

    println!("\n=== Invalid device tags ===");
    for bad_byte in [0xFF, 0x09, 0x80] {
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
    let cpu_buffer = BufferHandle::new(
        1024,
        MIN_BUFFER_ALIGNMENT,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )
    .expect("valid CPU buffer");
    println!("  Size: {} bytes", cpu_buffer.byte_size());
    println!(
        "  Alignment: {} bytes (SIMD minimum)",
        cpu_buffer.alignment()
    );
    println!("  Device: {}", cpu_buffer.device_tag());
    println!("  Sync mode: {}", cpu_buffer.sync_mode());
    println!("  Memory class: {}", cpu_buffer.memory_class());
    println!("  Empty: {}", cpu_buffer.is_empty());

    // ── GPU buffer with page alignment ────────────────────────────────────────

    println!("\n=== CUDA buffer (page alignment, Event sync) ===");
    let cuda_buffer = BufferHandle::new(8192, PAGE_ALIGNMENT, DeviceTag::Cuda, SyncMode::Event)
        .expect("valid CUDA buffer");
    println!("  Size: {} bytes", cuda_buffer.byte_size());
    println!(
        "  Alignment: {} bytes (page-aligned for GPU)",
        cuda_buffer.alignment()
    );
    println!("  Device: {}", cuda_buffer.device_tag());
    println!("  Sync mode: {}", cuda_buffer.sync_mode());
    println!("  Memory class: {}", cuda_buffer.memory_class());

    // ── Memory class round-trip ───────────────────────────────────────────────

    println!("\n=== Memory class round-trip ===");
    for (name, class) in [
        ("Standard", MemoryClass::Standard),
        ("HostPinned", MemoryClass::HostPinned),
        ("Unified", MemoryClass::Unified),
        ("Peer", MemoryClass::Peer),
        ("Private(0xF1)", MemoryClass::from_byte(0xF1).unwrap()),
    ] {
        let byte = class.to_byte();
        let recovered = MemoryClass::from_byte(byte).unwrap();
        println!("  {name}: 0x{byte:02X} → {recovered}");
    }

    // ── CUDA unified (managed) memory ─────────────────────────────────────────

    println!("\n=== CUDA unified (managed) memory ===");
    let unified = BufferHandle::with_memory_class(
        8192,
        PAGE_ALIGNMENT,
        DeviceTag::Cuda,
        SyncMode::ProducerSynced,
        MemoryClass::Unified,
    )
    .expect("valid unified buffer");
    println!("  Device: {}", unified.device_tag());
    println!("  Memory class: {}", unified.memory_class());
    println!("  → CPU and GPU can both access this buffer without explicit transfers");

    // ── Host-pinned memory ────────────────────────────────────────────────────

    println!("\n=== Host-pinned memory ===");
    let pinned = BufferHandle::with_memory_class(
        4096,
        PAGE_ALIGNMENT,
        DeviceTag::Cuda,
        SyncMode::Event,
        MemoryClass::HostPinned,
    )
    .expect("valid host-pinned buffer");
    println!("  Device: {}", pinned.device_tag());
    println!("  Memory class: {}", pinned.memory_class());
    println!("  → GPU DMA can read this buffer directly without staging");

    // ── Private memory class ──────────────────────────────────────────────────

    println!("\n=== Private memory class (vendor extension) ===");
    let vendor_class = MemoryClass::from_byte(0xF1).unwrap();
    let vendor_buf = BufferHandle::with_memory_class(
        2048,
        MIN_BUFFER_ALIGNMENT,
        DeviceTag::from_byte(0xF0).unwrap(),
        SyncMode::ProducerSynced,
        vendor_class,
    )
    .expect("valid vendor buffer");
    println!(
        "  Memory class is private: {}",
        vendor_buf.memory_class().is_private()
    );
    println!("  Wire byte: 0x{:02X}", vendor_buf.memory_class().to_byte());

    // ── Empty buffer ──────────────────────────────────────────────────────────

    println!("\n=== Empty buffer ===");
    let empty = BufferHandle::empty(DeviceTag::Rocm);
    println!("  Size: {} bytes", empty.byte_size());
    println!(
        "  Alignment: {} (allows minimal alignment for empty buffers)",
        empty.alignment()
    );
    println!("  Device: {}", empty.device_tag());
    println!("  Sync mode: {}", empty.sync_mode());
    println!("  Memory class: {}", empty.memory_class());
    println!("  Empty: {}", empty.is_empty());
    println!("  → Readers MUST NOT dereference the pointer of an empty buffer");

    // ── Private device tag ────────────────────────────────────────────────────

    println!("\n=== Private device tag (custom hardware) ===");
    let private_tag = DeviceTag::from_byte(0xF2).unwrap();
    let custom_buffer = BufferHandle::new(
        4096,
        MIN_BUFFER_ALIGNMENT,
        private_tag,
        SyncMode::ProducerSynced,
    )
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
        BufferHandle::new(512, 63, DeviceTag::Cpu, SyncMode::ProducerSynced)
    {
        println!("  alignment={alignment}: rejected (not a power of two)");
    }

    // ── Invalid alignment: below SIMD minimum ─────────────────────────────────

    println!("\n=== Alignment validation: below SIMD minimum ===");
    if let Err(Error::AlignmentBelowMinimum { alignment, minimum }) =
        BufferHandle::new(512, 32, DeviceTag::Cpu, SyncMode::ProducerSynced)
    {
        println!(
            "  alignment={alignment}, minimum={minimum}: rejected \
             (non-empty buffers must be SIMD-aligned)"
        );
    }

    // ── CPU + non-ProducerSynced rejected ─────────────────────────────────────

    println!("\n=== Sync mode validation: CPU must use ProducerSynced ===");
    if let Err(Error::InvalidSyncMode(b)) =
        BufferHandle::new(512, 64, DeviceTag::Cpu, SyncMode::Event)
    {
        println!(
            "  sync_mode=0x{b:02X} (Event) on CPU: rejected (CPU buffers must use ProducerSynced)"
        );
    }

    // ── Valid empty buffer with low alignment ─────────────────────────────────

    println!("\n=== Empty buffer allows any power-of-two alignment ===");
    let empty_aligned = BufferHandle::new(0, 1, DeviceTag::Metal, SyncMode::ProducerSynced)
        .expect("valid empty buffer");
    println!("  Empty buffer with alignment=1: accepted");
    println!("  Byte size: {}", empty_aligned.byte_size());

    // ── Colocation validation: all same device + class (pass) ────────────────

    println!("\n=== Colocation validation: all on CPU, Standard class (pass) ===");
    let handles = [
        BufferHandle::new(
            1024,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )
        .unwrap(),
        BufferHandle::new(
            512,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )
        .unwrap(),
        BufferHandle::new(
            256,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )
        .unwrap(),
    ];
    if let Ok(tag) = validate_colocation(&handles) {
        println!("  All {} buffers on: {}", handles.len(), tag);
    }

    // ── Colocation validation: mixed devices (fail) ─────────────────────────

    println!("\n=== Colocation validation: mixed devices (fail) ===");
    let mixed_devices = [
        BufferHandle::new(
            1024,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )
        .unwrap(),
        BufferHandle::new(
            512,
            PAGE_ALIGNMENT,
            DeviceTag::Cuda,
            SyncMode::ProducerSynced,
        )
        .unwrap(),
    ];
    if let Err(Error::DeviceTagMismatch { expected, found }) = validate_colocation(&mixed_devices) {
        println!("  Buffer 0: device 0x{expected:02X}; Buffer 1: device 0x{found:02X}");
        println!("  → Rejected: all buffers in a tensor must reside on the same device");
    }

    // ── Colocation validation: mixed memory classes (fail) ───────────────────

    println!("\n=== Colocation validation: mixed memory classes (fail) ===");
    let standard_buf =
        BufferHandle::new(4096, PAGE_ALIGNMENT, DeviceTag::Cuda, SyncMode::Event).unwrap();
    let unified_buf = BufferHandle::with_memory_class(
        4096,
        PAGE_ALIGNMENT,
        DeviceTag::Cuda,
        SyncMode::Event,
        MemoryClass::Unified,
    )
    .unwrap();
    if let Err(Error::MemoryClassMismatch { expected, found }) =
        validate_colocation(&[standard_buf, unified_buf])
    {
        println!("  Buffer 0: class 0x{expected:02X}; Buffer 1: class 0x{found:02X}");
        println!("  → Rejected: all buffers in a tensor must use the same memory class");
    }

    // ── Colocation validation: empty list (fail) ──────────────────────────────

    println!("\n=== Colocation validation: empty list (fail) ===");
    if let Err(Error::EmptyBufferList) = validate_colocation(&[]) {
        println!("  Rejected: at least one handle required to determine colocation");
    }

    // ── Metal device (Apple Silicon) ──────────────────────────────────────────

    println!("\n=== Metal device (Apple Silicon unified memory) ===");
    let metal = BufferHandle::new(2048, PAGE_ALIGNMENT, DeviceTag::Metal, SyncMode::Event).unwrap();
    println!("  Device: {}", metal.device_tag());
    println!(
        "  Memory class: {} (Standard = Metal device memory)",
        metal.memory_class()
    );
    println!("  Size: {}", metal.byte_size());
    println!("  Sync mode: {}", metal.sync_mode());

    // ── ROCm device ───────────────────────────────────────────────────────────

    println!("\n=== ROCm device ===");
    let rocm = BufferHandle::new(
        4096,
        PAGE_ALIGNMENT,
        DeviceTag::Rocm,
        SyncMode::ConsumerStream,
    )
    .unwrap();
    println!("  Device: {}", rocm.device_tag());
    println!("  Size: {}", rocm.byte_size());
    println!("  Sync mode: {}", rocm.sync_mode());
    println!("  Memory class: {}", rocm.memory_class());

    println!("\n=== Summary ===");
    println!("  • Device tags identify where a buffer resides (CPU, GPU, or private hardware)");
    println!("  • Memory class describes how the buffer is accessible (Standard, HostPinned, Unified, Peer)");
    println!("  • SyncMode describes producer–consumer ordering (ADR-018)");
    println!(
        "  • CPU buffers must use ProducerSynced; GPU buffers may use Event or ConsumerStream"
    );
    println!("  • Non-empty buffers must use SIMD alignment (≥64 bytes)");
    println!("  • Page-aligned buffers (≥4096 bytes) are recommended for GPU/IPC");
    println!("  • All buffers in a tensor must share both the same device tag and memory class");
    println!("  • Empty buffers allow any power-of-two alignment and are never dereferenced");
}
