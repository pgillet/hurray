# Quantized Inference

Hurray treats quantization as **first-class metadata**, orthogonal to the storage type:
a tensor is quantized if and only if it carries a quantization descriptor. Five schemes
are normative — per-tensor affine, per-channel affine, per-block affine, NF4 (QLoRA), and
MXFP (OCP Microscaling) — and the quantization parameters (scales, zero-points) live in
**separate buffer-table entries**, never interleaved with the data, so both stay
zero-copy.

This recipe builds a per-block-affine `int4` weight tensor in Rust, round-trips it, and
reads the scheme back.

```rust
use hurray_core::{
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, PerBlockAffine,
    QuantizationDescriptor, Shape, SyncMode, TensorDescriptor, MIN_BUFFER_ALIGNMENT,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A [256, 256] int4 weight matrix, per-block affine along axis 1, block_size 64.
    let shape = Shape::new(vec![256u64, 256])?;

    // Buffer 0 — packed int4 data: 256 × 256 values, 2 per byte = 32,768 bytes.
    let data = BufferHandle::new(
        32_768, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced,
    )?;
    // Buffer 1 — float32 scales, one per block: 256 rows × (256 / 64) = 1024 scales.
    let scales = BufferHandle::new(
        1024 * 4, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced,
    )?;

    // Per-block affine (symmetric): axis 1, block_size 64, scales in buffer index 1,
    // scale element type float32.
    let quant = QuantizationDescriptor::PerBlockAffine(
        PerBlockAffine::new_symmetric(1, 64, 1, ElementType::Float32)?,
    )
    .encode_to_vec();

    let desc = TensorDescriptor::new(
        1, 0,
        ElementType::Int4,          // storage type is orthogonal to the scheme
        shape,
        0,
        LayoutDescriptor::RowMajor,
        vec![data, scales],         // data + scale buffers
        Some(quant),                // HAS_QUANTIZATION
        None, None, None,           // no shard / statistics / extension-type
    )?;

    // Round-trip the descriptor.
    let bytes = desc.encode()?;
    let decoded = TensorDescriptor::decode(&bytes)?;
    assert_eq!(decoded, desc);

    // Read the scheme back from the (raw) quantization section.
    let (q, _) = QuantizationDescriptor::decode(decoded.quantization.as_ref().unwrap())?;
    println!("scheme tag = 0x{:02X}", q.scheme_tag().tag()); // 0x03 = per-block-affine
    if let QuantizationDescriptor::PerBlockAffine(pb) = q {
        println!("axis = {}, block_size = {}", pb.axis(), pb.block_size());
    }
    Ok(())
}
```

## Dequantization

The scheme's dequantization formula is normative (see
[Layer 2: Quantization Descriptors](layer-2-quantization-descriptors.md) and
`docs/spec/quantization.md`). For per-block affine, each element uses the scale (and, for
asymmetric, the zero-point) of the block it belongs to:

```text
value = (q_code - zero_point) * scale[block_index]
```

For a **symmetric** descriptor the zero-point is implicitly `0`, so `value = q_code *
scale[block]`. The block index is derived from the element's coordinate on the quantized
axis and `block_size`.

## Other schemes

- **NF4** (QLoRA) uses a fixed 16-level lookup table — `hurray_core::NF4_LUT` — plus a
  per-block scale: `QuantizationDescriptor::Nf4(Nf4::new(axis, block_size, scale_buffer))`.
- **MXFP** (OCP Microscaling) pairs an 8-bit shared exponent per block with the element
  micro-floats: `QuantizationDescriptor::Mxfp(Mxfp::new(axis, block_size, scale_buffer))`.
- **Per-tensor / per-channel affine** cover the classic INT8 cases
  (`PerTensorAffine::new`, `PerChannelAffine::new_symmetric` / `new_asymmetric`).

`hurray_core::validate_buffer_placement` checks that the scale/zero-point buffer indices a
descriptor references actually exist in the buffer table.

## Reading quantized tensors elsewhere

The Python bindings cannot yet construct quantized tensors. This is a current limitation,
not a design boundary: `hurray-python` is meant to expose everything `hurray-core` and
`hurray-io` can express. The blocker is that the bindings are single-buffer end to end,
while per-channel, NF4, and MXFP descriptors reference a separate scale buffer — tracked
in [#146](https://github.com/pgillet/hurray/issues/146). Per-tensor affine is the one
scheme that survives the trip today, because its scale and zero point are inline.

To inspect a quantized descriptor byte by byte (scheme, axis, block size, buffer indices),
use the CLI:

```bash
hurray-inspect weights.hrry
```

See [hurray-inspect CLI](hurray-inspect-cli.md); it decodes every quantization scheme and
formats Tier 2 / sub-byte element values.
