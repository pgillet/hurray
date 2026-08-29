# Converting an Externally Quantized Tensor

An external toolchain hands you quantized weights, per-block scales, and a
zero-point plane. Two of those three transfer as-is.

Hurray dequantizes as `scale * (q - zero_point)` — the stored zero point is the
value subtracted, with no implicit bias. Toolchains in the GPTQ family have
conventionally stored `zero_point - 1` and removed the bias in the loader, so the
number in the file and the number Hurray subtracts are not the same number.
Nothing in either file records which convention produced it.

That makes this the rare conversion error with no symptom. A descriptor built
from un-normalized values satisfies every validity constraint of its scheme, and
a reader has no way to detect the discrepancy — it simply decodes every element
off by one scale step.

## What transfers

| From the toolchain | Into Hurray | Changes? |
|---|---|---|
| Quantized weights | buffer 0 | no |
| Per-block scales | scale buffer, `float32` | no |
| Zero-point plane | zero-point buffer, one `int32` per block | **yes** — rebuilt |

The last row is the useful one. Hurray's per-block affine scheme takes one
`int32` per block, while the foreign plane packs two 4-bit values per byte. You
are rebuilding that buffer either way; the normalization is one line inside a
transform you are already writing, not a step to remember afterwards.

## The conversion

<div class="lang-tabs">

```rust
use hurray_core::{
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, PerBlockAffine,
    QuantizationDescriptor, Shape, SyncMode, TensorDescriptor,
    DESCRIPTOR_VERSION_MAJOR, DESCRIPTOR_VERSION_MINOR, MIN_BUFFER_ALIGNMENT,
};

/// 4-bit zero points packed two per byte, low nibble first, each holding
/// `zero_point - 1`.
fn normalize_zero_points(packed: &[u8], count: usize) -> Vec<i32> {
    (0..count)
        .map(|i| {
            let byte = packed[i / 2];
            let nibble = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            i32::from(nibble) + 1 // remove the toolchain's bias, here or never
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A [2, 16] int8 weight, quantized in blocks of 8 along axis 1: 4 blocks.
    let zero_points = normalize_zero_points(&[0x67, 0x78], 4);
    assert_eq!(zero_points, vec![8, 7, 9, 8]);

    let handle = |len: u64| {
        BufferHandle::new(len, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced)
    };

    // Buffer 0: weights. Buffer 1: float32 scales. Buffer 2: int32 zero points.
    let buffers = vec![handle(32)?, handle(16)?, handle(16)?];

    let quant = QuantizationDescriptor::PerBlockAffine(
        PerBlockAffine::new_asymmetric(1, 8, 1, 2, ElementType::Float32)?,
    );

    let desc = TensorDescriptor::new(
        DESCRIPTOR_VERSION_MAJOR,
        DESCRIPTOR_VERSION_MINOR,
        ElementType::Int8,
        Shape::new(vec![2u64, 16])?,
        0,
        LayoutDescriptor::RowMajor,
        buffers,
        Some(quant.encode_to_vec()),
        None, // no shard
        None, // no statistics
        None, // no extension type
    )?;

    assert_eq!(desc.buffers.len(), 3);
    Ok(())
}
```

```python
import struct
import hurray

def normalize_zero_points(packed, count):
    """4-bit zero points packed two per byte, each holding zero_point - 1."""
    out = []
    for i in range(count):
        byte = packed[i // 2]
        nibble = byte & 0x0F if i % 2 == 0 else byte >> 4
        out.append(nibble + 1)   # remove the toolchain's bias, here or never
    return out

# A [2, 16] int8 weight, quantized in blocks of 8 along axis 1: 4 blocks.
zero_points = normalize_zero_points(bytes([0x67, 0x78]), 4)
assert zero_points == [8, 7, 9, 8]

quant = hurray.PerBlockAffine.asymmetric(1, 8, 1, 2, hurray.float32)

weights = hurray.Tensor(
    bytes(32),                                  # buffer 0: the weights
    hurray.int8,
    [2, 16],
    aux_buffers=[
        struct.pack("4f", 0.02, 0.015, 0.025, 0.01),   # buffer 1: scales
        struct.pack("4i", *zero_points),               # buffer 2: zero points
    ],
    quantization=quant,
)

assert weights.buffer_count == 3
```

</div>

The scales went in untouched. The zero points did not.

## What skipping it costs

Hurray describes tensors; it does not compute on them, so there is no
`dequantize` in the library and there should not be one. The formula below is
read out of the spec, not called from an API:

```rust
fn main() {
    // One block: scale 0.02, true zero point 8, one quantized value.
    let (s, q) = (0.02f32, 9i32);

    let correct = s * (q - 8) as f32; // normalized
    let wrong = s * (q - 7) as f32;   // foreign value copied straight through

    assert!((wrong - correct - s).abs() < 1e-6); // off by exactly one scale step
}
```

One scale step, on every element, in both descriptors that a reader accepts
without complaint. Which is the whole reason the normalization is normative:
`quantization.md` § Zero-Point Convention places the obligation on the writer,
because after the buffer is written the information needed to detect the mistake
is gone.

## Scope

This recipe converts buffers you already hold in memory. Hurray is an
interchange format, not a converter library — there is no GPTQ, safetensors, or
GGUF reader here, and parsing those files is the caller's job.

Two limits worth knowing before you plan a conversion:

- **Activation-order grouping is out of scope.** GPTQ's act-order variant
  reorders the quantized axis, which per-block affine cannot express. Only the
  permutation-free subset — contiguous groups along one axis — maps onto scheme
  `0x03`. See `quantization.md` § Extension Schemes.
- **GGUF K-quants are out of scope.** `Q2_K`–`Q6_K` scale their per-block scales
  by a second super-block factor; per-block affine carries one scale array. See
  [Authoring Quantized Tensors](authoring-quantized-tensors.md) for what each
  scheme does cover.

## See also

- [Authoring Quantized Tensors](authoring-quantized-tensors.md) — building a
  descriptor from parameters you own
- [Quantized Inference](quantized-inference.md) — choosing a scheme
- [Multi-Buffer Tensors](multi-buffer-tensors.md) — how the parameter buffers travel
- `cargo run --example convert_external_quantized -p hurray-core`
- `python hurray-python/examples/convert_external_quantized.py`
