# Layer 2: Quantization Descriptors

## Purpose

Quantization descriptors specify how tensor elements are dequantized when retrieved from storage. Hurray supports five schemes covering per-tensor, per-channel, and per-block quantization strategies—essential for inference on quantized LLMs and diffusion models. Each scheme is encoded into a binary descriptor that precedes the tensor data.

## Per-Tensor Affine Quantization (INT8)

When a single scale and zero point apply uniformly to all elements (the simplest case):

<div class="lang-tabs">

```rust
use hurray_core::{PerTensorAffine, QuantizationDescriptor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a per-tensor affine descriptor: scale=0.015625, zero_point=128
    let q = PerTensorAffine::new(0.015625, 128)?;
    let desc = QuantizationDescriptor::PerTensorAffine(q);

    // Encode to bytes (16 bytes total: 4-byte header + 8 bytes scale/zp + 4 reserved)
    let mut buf = vec![0u8; desc.encoded_len()];
    let written = desc.encode_into(&mut buf)?;
    println!("Encoded: {} bytes", written); // 16

    // Decode back
    let (decoded, consumed) = QuantizationDescriptor::decode(&buf)?;
    assert_eq!(consumed, 16);
    assert_eq!(decoded, desc);

    // Dequantization formula: x_real = scale * (q - zero_point)
    // where q is a raw int8 value from storage
    Ok(())
}
```

```python
import hurray

# scale=0.015625, zero_point=128
scheme = hurray.PerTensorAffine(0.015625, 128)

wire = scheme.encode()
assert len(wire) == 16          # 4-byte header + 8 bytes scale/zp + 4 reserved

assert hurray.decode_quantization(wire) == scheme

# Dequantization formula: x_real = scale * (q - zero_point), where q is a raw
# int8 from storage. Applying it is the consuming framework's job, not the codec's.
```

</div>

**Use case:** Uniform quantization across an entire weight matrix or activation tensor.

## Per-Channel Affine Quantization (Output Channel Scaling)

When each output channel has its own scale (typical in INT8 quantized LLM weights):

<div class="lang-tabs">

```rust
use hurray_core::{PerChannelAffine, QuantizationDescriptor, BufferHandle, DeviceTag, MIN_BUFFER_ALIGNMENT};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Symmetric mode: scale array only, zero point implicit (all zeros)
    let q_sym = PerChannelAffine::new_symmetric(
        0,  // quantization axis (e.g., output channels)
        1,  // scale buffer index in tensor's buffer table
    )?;
    let desc_sym = QuantizationDescriptor::PerChannelAffine(q_sym);

    let encoded_sym = desc_sym.encode_to_vec();
    println!("Symmetric per-channel: {} bytes", encoded_sym.len()); // 20

    // Asymmetric mode: both scale and zero_point arrays
    let q_asym = PerChannelAffine::new_asymmetric(
        1,  // quantization axis
        2,  // scale buffer index
        3,  // zero_point buffer index
    )?;
    let desc_asym = QuantizationDescriptor::PerChannelAffine(q_asym);

    let encoded_asym = desc_asym.encode_to_vec();
    println!("Asymmetric per-channel: {} bytes", encoded_asym.len()); // 20

    // Dequantization formula: x_real = scale[c] * (q - zero_point[c])
    // where c = logical_index[axis]
    Ok(())
}
```

```python
import hurray

scheme = hurray.PerChannelAffine.symmetric(axis=0, scale_buffer_index=1)

assert scheme.axis == 0
assert hurray.decode_quantization(scheme.encode()) == scheme
```

</div>

**Use case:** Per-output-channel quantization in transformer weight matrices; achieves better accuracy than per-tensor.

## Per-Block Affine Quantization (QLoRA-Style)

Divide a tensor into fixed-size blocks along one axis; each block carries its own scale and (optionally) zero point:

<div class="lang-tabs">

```rust
use hurray_core::{PerBlockAffine, QuantizationDescriptor, ElementType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Symmetric: scale only (zero_point implicit)
    let q_sym = PerBlockAffine::new_symmetric(
        0,                      // quantization axis
        64,                     // block size (must be power of two ≥ 2)
        1,                      // scale buffer index
        ElementType::Float32,   // scale element type
    )?;
    let desc_sym = QuantizationDescriptor::PerBlockAffine(q_sym);

    let encoded_sym = desc_sym.encode_to_vec();
    println!("Per-block symmetric: {} bytes", encoded_sym.len()); // 24

    // Asymmetric: scale and zero_point arrays
    let q_asym = PerBlockAffine::new_asymmetric(
        1,                      // quantization axis
        32,                     // block size
        2,                      // scale buffer index
        3,                      // zero_point buffer index
        ElementType::Float16,   // scale in float16 (more compact)
    )?;
    let desc_asym = QuantizationDescriptor::PerBlockAffine(q_asym);

    let encoded_asym = desc_asym.encode_to_vec();
    println!("Per-block asymmetric: {} bytes", encoded_asym.len()); // 24

    // Dequantization formula: x_real = scale[b] * (q - zero_point[b])
    // where b = block_index = logical_index[axis] / block_size
    Ok(())
}
```

```python
import hurray

scheme = hurray.PerBlockAffine.symmetric(
    axis=0, block_size=64, scale_buffer_index=1, scale_type=hurray.float32
)

assert scheme.symmetric
assert hurray.decode_quantization(scheme.encode()) == scheme
```

</div>

**Scale types:** `Float16`, `BFloat16`, or `Float32` (controlled per descriptor).

**Use case:** QLoRA-style quantization; balances compression and accuracy in large language models.

## NF4 Block Quantization

A non-linear 4-bit scheme with 16 fixed quantization levels (from QLoRA). Each block has a single absolute-maximum scale:

<div class="lang-tabs">

```rust
use hurray_core::{Nf4, QuantizationDescriptor, NF4_LUT};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // NF4: 4-bit storage, 16 fixed levels
    let q = Nf4::new(
        0,      // quantization axis
        64,     // block size (must be power of two ≥ 8)
        1,      // scale buffer index (contains absmax values)
    )?;
    let desc = QuantizationDescriptor::Nf4(q);

    let encoded = desc.encode_to_vec();
    println!("NF4: {} bytes", encoded.len()); // 16

    // Inspect the fixed NF4 lookup table
    println!("NF4 levels (indexed by 4-bit code):");
    for (i, &level) in NF4_LUT.iter().enumerate() {
        println!("  [{}] = {:.4}", i, level);
    }

    // Dequantization formula: x_real = scale[b] * NF4_LUT[q]
    // where q ∈ [0, 15] (the 4-bit storage code)
    // and b = block_index = logical_index[axis] / block_size
    Ok(())
}
```

```python
import hurray

scheme = hurray.NF4(axis=0, block_size=64, scale_buffer_index=1)

assert scheme.block_size == 64
assert hurray.decode_quantization(scheme.encode()) == scheme
```

</div>

**Block size constraint:** Must be a power of two ≥ 8.

**Use case:** Quantization of weight matrices in LLMs using the QLoRA approach; achieves ≤4-bit effective precision with minimal accuracy loss.

## MXFP Block Quantization (OCP Microscaling)

Open Compute Project Microscaling (OCP MX) format: blocks share a single exponent-only scale in float8_e8m0 format. Requires exact divisibility (no partial trailing blocks):

<div class="lang-tabs">

```rust
use hurray_core::{Mxfp, QuantizationDescriptor, CANONICAL_BLOCK_SIZE};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // MXFP: standard block size from OCP MX spec
    let q = Mxfp::new(
        0,                      // quantization axis
        CANONICAL_BLOCK_SIZE,   // 32 (canonical OCP MX v1.0 block size)
        1,                      // scale buffer index
    )?;
    let desc = QuantizationDescriptor::Mxfp(q);

    let encoded = desc.encode_to_vec();
    println!("MXFP: {} bytes", encoded.len()); // 16

    // Alternative: custom block size (still must be power-of-two in [16, 2048])
    let q_custom = Mxfp::new(0, 64, 1)?;
    println!("MXFP with block_size=64: valid");

    // IMPORTANT: Unlike per-block affine, MXFP requires exact divisibility.
    // If shape[axis] is 1000 and block_size is 32, this descriptor is INVALID
    // because 1000 is not divisible by 32.

    // Dequantization formula: x_real = 2^(e - 127) * q
    // where e is the float8_e8m0 scale byte from scale buffer
    Ok(())
}
```

```python
import hurray

scheme = hurray.MXFP(axis=0, block_size=32, scale_buffer_index=1)

assert scheme.block_size == 32
assert hurray.decode_quantization(scheme.encode()) == scheme
```

</div>

**Block size constraint:** Power of two in `[16, 2048]` (inclusive).

**Divisibility requirement:** `shape[axis]` must be a positive multiple of `block_size`. No partial blocks.

**Use case:** Hardware-friendly quantization for inference accelerators supporting OCP Microscaling (NVIDIA H100+, etc.).

## Choosing a Quantization Scheme

| Scheme | Scope | Grain | Storage | Best For |
|--------|-------|-------|---------|----------|
| Per-Tensor Affine | Entire tensor | Single scale/ZP | 8 bytes payload | Uniform quantization, simplicity |
| Per-Channel Affine | One axis | One scale/ZP per slice | Separate scale/ZP buffers | LLM weight quantization (INT8) |
| Per-Block Affine | Blocks along axis | One scale/ZP per block | Separate scale/ZP buffers | Moderate compression (QLoRA) |
| NF4 | Blocks along axis | One absmax per block | Fixed 16-level LUT | Aggressive 4-bit quantization |
| MXFP | Blocks along axis | Exponent-only scale | Separate scale buffer (float8_e8m0) | Hardware-accelerated OCP MX targets |

## Validating Buffer Placement

Quantization descriptors reference external buffers (for scales and zero points). Validate placement before building a tensor descriptor:

<div class="lang-tabs">

```rust
use hurray_core::{
    BufferHandle, DeviceTag, Nf4, QuantizationDescriptor, SyncMode,
    validate_buffer_placement, MIN_BUFFER_ALIGNMENT,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create buffers on the same device
    let data_buf = BufferHandle::new(4096, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced)?;
    let scale_buf = BufferHandle::new(256, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced)?;
    let buffers = [data_buf, scale_buf];

    // Create an NF4 descriptor with scale at buffer index 1
    let q = Nf4::new(0, 64, 1)?;
    let desc = QuantizationDescriptor::Nf4(q);

    // Validate: scale_buffer_index=1 must be in range, not alias data (index 0),
    // and be on the same device.
    validate_buffer_placement(&desc, &buffers, 0)?;

    println!("Buffer placement valid!");
    Ok(())
}
```

```python
import hurray

# Python checks placement when the tensor is built, against the buffers it was
# actually given — there is no separate descriptor to validate in isolation.
tensor = hurray.Tensor(
    bytes(4096),
    hurray.dtype.int4,
    [64, 64],
    aux_buffers=[bytes(256)],
    quantization=hurray.NF4(axis=0, block_size=64, scale_buffer_index=1),
)
assert tensor.buffer_count == 2

# An index that names no buffer is refused.
try:
    hurray.Tensor(
        bytes(4096),
        hurray.dtype.int4,
        [64, 64],
        aux_buffers=[bytes(256)],
        quantization=hurray.NF4(axis=0, block_size=64, scale_buffer_index=7),
    )
    raise AssertionError("buffer 7 does not exist")
except hurray.InvalidDescriptorError as exc:
    print(exc)
```

</div>

**Constraints checked:**
1. All quantization parameter buffer indices are within `buffers.len()`.
2. No quantization buffer index equals the data buffer index (no aliasing).
3. All quantization buffers are on the same device as the data buffer.

## Encoding and Decoding

Use `encode_into` for streaming writers (zero-alloc) or `encode_to_vec` for convenience:

<div class="lang-tabs">

```rust
use hurray_core::{PerTensorAffine, QuantizationDescriptor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let desc = QuantizationDescriptor::PerTensorAffine(
        PerTensorAffine::new(0.5, 0)?
    );

    // Zero-alloc path (required for streaming)
    let mut buf = vec![0u8; desc.encoded_len()];
    let written = desc.encode_into(&mut buf)?;
    assert_eq!(written, desc.encoded_len());

    // Decode and verify round-trip
    let (decoded, consumed) = QuantizationDescriptor::decode(&buf)?;
    assert_eq!(consumed, written);
    assert_eq!(decoded, desc);

    println!("Roundtrip successful!");
    Ok(())
}
```

```python
import hurray

scheme = hurray.PerTensorAffine(0.5, 0)

# 16 to 24 bytes, so there is no zero-alloc variant to reach for: the section
# is smaller than the cost of arranging to avoid the allocation.
wire = scheme.encode()
assert len(wire) == 16

assert hurray.decode_quantization(wire) == scheme
```

</div>

`decode_quantization` returns whichever of the five classes the section's scheme tag
names, so a caller reads the bytes without knowing in advance which scheme wrote them —
which is what a tagged section is for. Trailing bytes are ignored: the section carries its
own length, which is what lets it sit inside a descriptor with other sections after it.

**Note:** `encode_into` is preferred in hot paths (streaming readers/writers) because it avoids allocation. The descriptor itself is small (16–24 bytes), so `encode_to_vec` is fine for initialization paths.
