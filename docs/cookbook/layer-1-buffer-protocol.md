# Layer 1: Buffer Protocol

## Purpose

A **buffer handle** declares a tensor's data buffer: its size in bytes, alignment guarantee, and which device (CPU, GPU, or custom) it resides in. A **device tag** identifies the memory space. A **memory class** describes *how* the buffer is accessible — standard device-private, host-pinned, unified, or peer-accessible. Together they form the bridge between the descriptor's binary metadata and the actual memory location — the handle does not hold a pointer (that comes out-of-band) but carries the rules readers must follow to safely dereference the data.

## Creating Buffer Handles

The most common case: a CPU buffer with SIMD alignment (64 bytes minimum):

```rust
use hurray_core::{BufferHandle, DeviceTag, SyncMode, MIN_BUFFER_ALIGNMENT};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a 1 KB CPU buffer with SIMD alignment.
    let handle = BufferHandle::new(1024, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced)?;
    
    assert_eq!(handle.byte_size(), 1024);
    assert_eq!(handle.alignment(), 64);
    assert_eq!(handle.device_tag(), DeviceTag::Cpu);
    assert_eq!(handle.sync_mode(), SyncMode::ProducerSynced);
    
    Ok(())
}
```

For GPU or IPC buffers, use page alignment (4096 bytes):

```rust
use hurray_core::{BufferHandle, DeviceTag, SyncMode, PAGE_ALIGNMENT};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // CUDA buffer aligned to one page — safe for GPU + IPC transport.
    let gpu_buffer = BufferHandle::new(8192, PAGE_ALIGNMENT, DeviceTag::Cuda, SyncMode::Event)?;
    
    assert_eq!(gpu_buffer.alignment(), 4096);
    assert_eq!(gpu_buffer.device_tag(), DeviceTag::Cuda);
    
    Ok(())
}
```

## Choosing Alignment

| Device | Alignment | Why |
|--------|-----------|-----|
| **CPU (SIMD)** | 64 bytes | Minimum for AVX-512, NEON, SVE without per-op negotiation |
| **GPU, IPC, RDMA** | 4096 bytes | Host page size; avoids cross-page pinning and TLB fragmentation |
| **Custom** (private tag) | ≥64 bytes | Implementation-defined; typically matches SIMD or page boundary |

Always use the strongest alignment you can guarantee — readers may rely on it for performance.

## Empty Buffers

A tensor with zero elements (e.g., shape `[5, 0, 10]`) has zero-byte buffers. Use `BufferHandle::empty()`:

```rust
use hurray_core::{BufferHandle, DeviceTag};

fn main() {
    // Empty buffer — no data, alignment is waived.
    let empty = BufferHandle::empty(DeviceTag::Cpu);
    
    assert!(empty.is_empty());
    assert_eq!(empty.byte_size(), 0);
    assert_eq!(empty.alignment(), 1); // Any power-of-two is valid
}
```

Readers MUST NOT dereference the pointer of an empty buffer. In C ABI contexts, it may be a null pointer; in others, it may be non-null but uninitialized. Do not read or write.

## Memory Class

A buffer's **memory class** describes how it is accessible, orthogonally to which device it resides on. The default is `Standard` (device-private memory), but other classes enable zero-copy sharing patterns:

| Class | Wire byte | Meaning |
|-------|-----------|---------|
| `Standard` | `0x00` | Device-private memory; default for all devices |
| `HostPinned` | `0x01` | CPU-accessible pinned memory (e.g., CUDA `cudaMallocHost`) |
| `Unified` | `0x02` | Unified/managed memory accessible from both CPU and GPU |
| `Peer` | `0x03` | Peer-to-peer memory accessible from a second GPU |

Use `BufferHandle::new()` for the common case (Standard); use `BufferHandle::with_memory_class()` when the class is known:

```rust
use hurray_core::{BufferHandle, DeviceTag, MemoryClass, SyncMode, PAGE_ALIGNMENT};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // CUDA unified (managed) memory — CPU and GPU can both access it directly.
    let unified = BufferHandle::with_memory_class(
        8192,
        PAGE_ALIGNMENT,
        DeviceTag::Cuda,
        SyncMode::ProducerSynced,
        MemoryClass::Unified,
    )?;
    assert_eq!(unified.memory_class(), MemoryClass::Unified);

    // Host-pinned memory — GPU DMA can read it without staging.
    let pinned = BufferHandle::with_memory_class(
        4096,
        PAGE_ALIGNMENT,
        DeviceTag::Cuda,
        SyncMode::Event,
        MemoryClass::HostPinned,
    )?;
    assert_eq!(pinned.memory_class(), MemoryClass::HostPinned);
    
    Ok(())
}
```

The memory class round-trips through the wire format:

```rust
use hurray_core::MemoryClass;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Serialize: type → byte
    let original = MemoryClass::Unified;
    let byte = original.to_byte();
    assert_eq!(byte, 0x02);
    
    // Deserialize: byte → type
    let recovered = MemoryClass::from_byte(byte)?;
    assert_eq!(original, recovered);
    
    Ok(())
}
```

Private memory classes (`0xF0`–`0xFE`) are available for vendor-specific extensions, following the same pattern as private device tags:

```rust
use hurray_core::{BufferHandle, DeviceTag, MemoryClass, SyncMode, MIN_BUFFER_ALIGNMENT};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vendor_class = MemoryClass::from_byte(0xF1)?;
    let handle = BufferHandle::with_memory_class(
        2048,
        MIN_BUFFER_ALIGNMENT,
        DeviceTag::from_byte(0xF0)?,
        SyncMode::ProducerSynced,
        vendor_class,
    )?;
    assert!(handle.memory_class().is_private());
    
    Ok(())
}
```

## Private Device Tags

For experimental or vendor-specific hardware, use the private range (`0xF0`–`0xFE`):

```rust
use hurray_core::{BufferHandle, DeviceTag, SyncMode, MIN_BUFFER_ALIGNMENT};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a private device tag for a custom accelerator (e.g., TPU, custom FPGA).
    let custom_device = DeviceTag::from_byte(0xF2)?;
    let handle = BufferHandle::new(4096, MIN_BUFFER_ALIGNMENT, custom_device, SyncMode::ProducerSynced)?;
    
    assert!(custom_device.is_private());
    assert_eq!(custom_device.to_byte(), 0xF2);
    
    Ok(())
}
```

**Important:** Private tags must not be exchanged between independent implementations without an out-of-band agreement on semantics. Use only when both producer and consumer control the device tag value.

## Alignment Validation

Alignment must be a power of two:

```rust
use hurray_core::{BufferHandle, DeviceTag, Error, SyncMode};

fn main() {
    // Alignment is not a power of two — rejected.
    let result = BufferHandle::new(512, 63, DeviceTag::Cpu, SyncMode::ProducerSynced);
    assert!(matches!(result, Err(Error::AlignmentNotPowerOfTwo { alignment: 63 })));
}
```

For non-empty buffers, alignment must be at least 64 bytes:

```rust
use hurray_core::{BufferHandle, DeviceTag, Error, SyncMode};

fn main() {
    // Non-empty buffer with alignment below SIMD minimum — rejected.
    let result = BufferHandle::new(512, 32, DeviceTag::Cpu, SyncMode::ProducerSynced);
    assert!(matches!(
        result,
        Err(Error::AlignmentBelowMinimum { alignment: 32, minimum: 64 })
    ));
    
    // Empty buffers allow any power-of-two alignment, including 1.
    let empty = BufferHandle::new(0, 1, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap();
    assert!(empty.is_empty());
}
```

## Device Colocation

All buffers in a single tensor (data + quantization parameters) must reside on the same device **and** in the same memory class. Validate this before processing:

```rust
use hurray_core::{BufferHandle, DeviceTag, SyncMode, validate_colocation};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_buffer = BufferHandle::new(1024, 64, DeviceTag::Cpu, SyncMode::ProducerSynced)?;
    let scale_buffer = BufferHandle::new(16, 64, DeviceTag::Cpu, SyncMode::ProducerSynced)?;
    
    // All on CPU with Standard memory class — passes.
    let device = validate_colocation(&[data_buffer, scale_buffer])?;
    assert_eq!(device, DeviceTag::Cpu);
    
    Ok(())
}
```

Mixed devices are rejected:

```rust
use hurray_core::{BufferHandle, DeviceTag, Error, SyncMode, validate_colocation};

fn main() {
    let cpu_buf = BufferHandle::new(1024, 64, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap();
    let gpu_buf = BufferHandle::new(256, 4096, DeviceTag::Cuda, SyncMode::ProducerSynced).unwrap();
    
    // Different devices — fails.
    let result = validate_colocation(&[cpu_buf, gpu_buf]);
    assert!(matches!(
        result,
        Err(Error::DeviceTagMismatch { expected: 0x00, found: 0x01 })
    ));
}
```

Mixed memory classes are also rejected — even when all buffers share the same device:

```rust
use hurray_core::{BufferHandle, DeviceTag, Error, MemoryClass, SyncMode, PAGE_ALIGNMENT, validate_colocation};

fn main() {
    let standard = BufferHandle::new(4096, PAGE_ALIGNMENT, DeviceTag::Cuda, SyncMode::Event).unwrap();
    let unified = BufferHandle::with_memory_class(
        4096, PAGE_ALIGNMENT, DeviceTag::Cuda, SyncMode::Event, MemoryClass::Unified,
    ).unwrap();
    
    // Same device, different memory class — fails.
    let result = validate_colocation(&[standard, unified]);
    assert!(matches!(
        result,
        Err(Error::MemoryClassMismatch { expected: 0x00, found: 0x02 })
    ));
}
```

Why? Quantized tensor kernels dereference both data and quantization parameters. Cross-device and cross-class transfers are expensive; colocation ensures efficient access. If buffers must use different memory classes, emit a separate tensor descriptor.

## Device Tag Round-Trip

Serialize a device to its wire byte and back:

```rust
use hurray_core::DeviceTag;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Serialize: type → byte
    let original = DeviceTag::Cuda;
    let byte = original.to_byte();
    assert_eq!(byte, 0x01);
    
    // Deserialize: byte → type
    let recovered = DeviceTag::from_byte(byte)?;
    assert_eq!(original, recovered);
    
    println!("Round-trip: {} → 0x{:02X} → {}", original, byte, recovered);
    
    Ok(())
}
```

Bytes in the range `0x09`–`0xEF` (reserved for future spec versions) and `0xFF` (permanently invalid) are rejected:

```rust
use hurray_core::{DeviceTag, Error};

fn main() {
    assert!(matches!(DeviceTag::from_byte(0x09), Err(Error::ReservedDeviceTag(_))));
    assert!(matches!(DeviceTag::from_byte(0xFF), Err(Error::InvalidDeviceTag(_))));
}
```

## Named Device Tags

The spec defines nine named device types:

| Tag | Variant | Use |
|-----|---------|-----|
| `0x00` | `DeviceTag::Cpu` | Host memory |
| `0x01` | `DeviceTag::Cuda` | NVIDIA CUDA GPU |
| `0x02` | `DeviceTag::Rocm` | AMD ROCm GPU |
| `0x03` | `DeviceTag::Metal` | Apple Silicon (Metal/MPS) |
| `0x04` | `DeviceTag::Vulkan` | Vulkan cross-platform GPU compute |
| `0x05` | `DeviceTag::WebGpu` | WebGPU (browser inference) |
| `0x06` | `DeviceTag::Hexagon` | Qualcomm HVX/HMX DSP |
| `0x07` | `DeviceTag::LevelZero` | Intel Level Zero / oneAPI |
| `0x08` | `DeviceTag::OpenCl` | OpenCL (embedded/legacy GPU) |

Tags `0x09`–`0xEF` are reserved; `0xF0`–`0xFE` are private; `0xFF` is permanently invalid.

## Key Takeaways

- **DeviceTag** identifies where a buffer resides (CPU, GPU, or custom hardware)
- **MemoryClass** describes how it is accessible: Standard (device-private), HostPinned, Unified, or Peer
- **Alignment** must be a power of two; at least 64 bytes for non-empty, any power-of-two for empty
- **Page alignment** (4096 bytes) recommended for GPU and IPC buffers
- **Colocation validation** requires all buffers to share both the same device tag and the same memory class
- **Private tags** (`0xF0`–`0xFE`) allow vendor-specific devices or memory classes but require out-of-band agreement
- **Empty buffers** are never dereferenced; alignment rules are waived

See `docs/spec/buffer-protocol.md` for the normative specification.
