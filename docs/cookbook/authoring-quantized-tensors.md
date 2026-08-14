# Authoring Quantized Tensors

Hurray *describes* quantization; it does not compute it. These types package scales
and zero points you already have — from your own quantizer, or from a model you are
converting — into a descriptor any Hurray reader understands.

## Where the parameters live

| Scheme | Parameters | Buffers |
|---|---|---|
| Per-tensor affine | one scale + zero point, **inline** in the descriptor | 1 |
| Per-channel affine | one scale per slice along an axis | 2 (+1 if asymmetric) |
| Per-block affine | one scale per block of `block_size` | 2 (+1 if asymmetric) |
| NF4 | one scale per block | 2 |
| MXFP | one shared exponent per block | 2 |

Only per-tensor affine keeps its parameters in the descriptor. Every other scheme
puts them in a separate buffer and refers to it **by index** — so building one means
supplying the parameter bytes *and* the index that points at them.

## Building one

<div class="lang-tabs">

```rust
use hurray_core::{
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, PerChannelAffine,
    QuantizationDescriptor, Shape, SyncMode, TensorDescriptor,
    DESCRIPTOR_VERSION_MAJOR, DESCRIPTOR_VERSION_MINOR, MIN_BUFFER_ALIGNMENT,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let handle = |len: u64| {
        BufferHandle::new(len, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced)
    };

    // Buffer 0 is the int8 data; buffer 1 holds one float32 scale per row.
    let buffers = vec![handle(8)?, handle(8)?];

    // The scheme names buffer 1 by index — symmetric, so no zero points.
    let quant = QuantizationDescriptor::PerChannelAffine(
        PerChannelAffine::new_symmetric(0, 1)?,
    );

    let desc = TensorDescriptor::new(
        DESCRIPTOR_VERSION_MAJOR,
        DESCRIPTOR_VERSION_MINOR,
        ElementType::Int8,
        Shape::new(vec![2u64, 4])?,
        0,
        LayoutDescriptor::RowMajor,
        buffers,
        Some(quant.encode_to_vec()),
        None, // no shard
        None, // no statistics
        None, // no extension type
    )?;

    println!("descriptor: {} bytes", desc.encode()?.len());
    Ok(())
}
```

```python
import struct
import hurray

# One float32 scale per row of a [2, 4] tensor.
scales = struct.pack("2f", 0.02, 0.017)

# The scheme names buffer 1 by index — symmetric, so no zero points.
quant = hurray.PerChannelAffine.symmetric(axis=0, scale_buffer_index=1)

weights = hurray.Tensor(
    bytes(8),              # buffer 0: the int8 weights
    hurray.int8,
    [2, 4],
    aux_buffers=[scales],  # buffer 1: what scale_buffer_index points at
    quantization=quant,
)
```

</div>

Per-tensor affine needs no companion buffer, because its parameters are inline:

```python
q = hurray.PerTensorAffine(0.02, 128)
t = hurray.Tensor(bytes(8), hurray.int8, [2, 4], quantization=q)   # still 1 buffer
```

## An index that points at nothing is refused

A descriptor claiming a scale buffer that was never supplied encodes and decodes
perfectly well — the consumer simply finds a dangling index. So it is rejected where
the mistake was made:

```python
# No aux_buffers, but the scheme references buffer 1.
hurray.Tensor(bytes(8), hurray.int8, [2, 4],
              quantization=hurray.PerChannelAffine.symmetric(0, 1))
# hurray.InvalidDescriptorError: invalid quantization: ...
```

## Symmetric and asymmetric

Schemes with an optional zero point expose two constructors rather than a flag, so
"asymmetric but no zero-point buffer" cannot be expressed at all:

```python
sym  = hurray.PerChannelAffine.symmetric(axis=0, scale_buffer_index=1)
asym = hurray.PerChannelAffine.asymmetric(
    axis=0, scale_buffer_index=1, zero_point_buffer_index=2
)

assert sym.zero_point_buffer_index is None      # wire sentinel 0xFFFFFFFF
assert asym.zero_point_buffer_index == 2
```

Per-block affine additionally declares the element type of its scales, which must be
float16, bfloat16, or float32:

```python
q = hurray.PerBlockAffine.symmetric(1, 32, 1, hurray.float32)
assert q.scale_type == hurray.float32
```

## Statistics

Each statistic carries a validity bit saying whether it means anything. You pass
values and the mask is derived, so a number can never be present with its bit unset:

```python
s = hurray.Statistics(nnz=1024, value_min=-1.0, value_max=1.0, value_abs_max=1.0)

assert s.nnz == 1024
assert s.value_mean is None        # not supplied, so not claimed
```

Fields that share one bit on the wire must be supplied together — `value_min` /
`value_max` / `value_abs_max`, `value_mean` / `value_stddev`, `nm_n` / `nm_m`, and
`has_nan` / `has_inf`. A partial group raises rather than silently zero-filling the
rest:

```python
hurray.Statistics(value_min=-1.0)   # InvalidDescriptorError
```

## Shard

Records this tensor's position inside a larger logical one:

```python
shard = hurray.Shard(parent_shape=[1024, 512], shard_offset=[512, 0])
piece = hurray.Tensor(bytes(8), hurray.int8, [2, 4], shard=shard)
```

## Reading it back

A consumer that receives a tensor — off disk, off the wire, or over the native
protocol — can ask what it is holding. The getters return the same classes the
constructor accepts, so an inspected scheme can be passed straight back to build
another tensor.

<div class="lang-tabs">

```rust
use hurray_core::{QuantizationDescriptor, TensorDescriptor};

fn describe(desc: &TensorDescriptor) -> Result<(), Box<dyn std::error::Error>> {
    match desc.quantization.as_ref() {
        None => println!("not quantized"),
        Some(bytes) => {
            let (scheme, _read) = QuantizationDescriptor::decode(bytes)?;
            match scheme {
                QuantizationDescriptor::PerChannelAffine(q) => println!(
                    "per-channel: axis {}, scales in buffer {}",
                    q.axis(),
                    q.scale_buffer_index()
                ),
                other => println!("scheme: {other:?}"),
            }
        }
    }
    Ok(())
}
```

```python
import hurray

loaded = hurray.load("weights.hrry")["w"]

q = loaded.quantization
if q is None:
    print("not quantized")
else:
    print(f"per-channel: axis {q.axis}, scales in buffer {q.scale_buffer_index}")

print(f"buffers: {loaded.buffer_count}")
```

</div>

`statistics` and `shard` read back the same way, and every section is `None` when
absent:

```python
t = hurray.Tensor(bytes(16), hurray.float32, [4])

assert t.quantization is None
assert t.statistics is None
assert t.shard is None
```

A statistic that was never supplied stays unclaimed after the round trip — the
validity mask travels with the values:

```python
s = hurray.Statistics(nnz=6)
t = hurray.Tensor(bytes(24), hurray.float32, [2, 3], statistics=s)

assert t.statistics.nnz == 6
assert t.statistics.value_mean is None
```

## Checking what you built

`save()` writes every buffer, so the scales travel with the weights:

```python
hurray.save("weights.hrry", {"w": weights})
```

`hurray-inspect` then shows the scheme byte by byte:

```text
   141  14 00 00 00                     quantization_length = 20
   145  02                              scheme_tag = 0x02 (per-channel-affine)
   149  00 00 00 00                     axis = 0
   153  01 00 00 00                     scale_buffer_index = 1
   157  FF FF FF FF                     zero_point_buffer_index = none (symmetric)
   161  03                              scale_type = float32
```

## See also

- [Quantized Inference](quantized-inference.md) — choosing a scheme
- [Multi-Buffer Tensors](multi-buffer-tensors.md) — how the scale buffer travels
- [hurray-inspect CLI](hurray-inspect-cli.md) — reading a descriptor byte by byte
- `cargo run --example quantization_roundtrip -p hurray-core`
- `python hurray-python/examples/quantized_authoring.py`
