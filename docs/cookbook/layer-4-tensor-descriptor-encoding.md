# Layer 4: Tensor Descriptor Encoding

## Purpose

A `TensorDescriptor` is the top-level carrier for all metadata required to interpret a tensor's
data buffer: element type, rank, shape, memory layout, buffer handles, and optional quantization,
shard, statistics, and extension-type annotations.

The binary format is defined in `docs/spec/metadata.md`. A 20-byte fixed header is followed by
variable-length core fields, layout-specific payload, a buffer table, and up to four optional
sections selected by a flags bitmask.

Runnable example: `cargo run --example encode_decode_descriptor`

## Quick reference: wire format sections

| Section | Always present? | Controlled by |
|---------|----------------|---------------|
| Fixed header (20 bytes) | Yes | Always |
| Shape `uint64[rank]` | Yes | `shape.rank()` |
| `byte_offset uint64` | Yes | Always |
| Layout payload | Yes | `layout.tag()` |
| Buffer table | Yes | `buffers.len()` |
| Quantization | `HAS_QUANTIZATION` flag | `quantization.is_some()` |
| Shard | `HAS_SHARD` flag | `shard.is_some()` |
| Extension type | `HAS_EXTENSION_TYPE` flag | `extension_type.is_some()` |
| Statistics | `HAS_STATISTICS` flag | `statistics.is_some()` |

## Encoding and decoding

The spec's worked example — `float32 [3, 4]` row-major, one CPU buffer — encodes to exactly
61 bytes:

<div class="lang-tabs">

```rust
use hurray_core::{
    BufferHandle, DeviceTag, ElementType, Shape, SyncMode, MIN_BUFFER_ALIGNMENT,
    descriptor::TensorDescriptor,
    layout::LayoutDescriptor,
};

let shape  = Shape::new(vec![3u64, 4]).unwrap();
// 192 = 3 × 64 bytes (one cache line per row) — the spec's worked example allocation.
// The tensor data itself needs only 3×4×4 = 48 bytes; byte_size records the physical
// allocation which may exceed the data footprint. In real code use buffer_size_bytes().
let buffer = BufferHandle::new(192, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap();

let desc = TensorDescriptor::new(
    1, 0,                      // version_major, version_minor
    ElementType::Float32,
    shape,
    0,                         // byte_offset
    LayoutDescriptor::RowMajor,
    vec![buffer],
    None,                      // quantization
    None,                      // shard
    None,                      // statistics
    None,                      // extension_type
).unwrap();

let bytes = desc.encode().unwrap();
assert_eq!(bytes.len(), 61);   // spec worked example

// Decode back — descriptor is byte-exact round-trip.
let decoded = TensorDescriptor::decode(&bytes).unwrap();
assert_eq!(decoded, desc);
```

```python
import hurray

# A descriptor comes from a tensor rather than being built on its own: it is the
# half of a tensor that travels first, not a separate thing to assemble.
tensor = hurray.Tensor(bytes(192), hurray.float32, [3, 4])
descriptor = tensor.descriptor

wire = descriptor.encode()
assert len(wire) == 61          # spec worked example

# Decode back — byte-exact round trip.
assert hurray.Descriptor.decode(wire) == descriptor
```

</div>

`Descriptor` is not constructible from Python: a constructor would duplicate
`hurray.Tensor`'s whole parameter list to build the half of it that carries no data.
Descriptors come from a tensor, from a composite head (`Composite.descriptor`), or from
`decode`.

## Advisory statistics

Attach pre-computed statistics (value range, NaN/Inf presence, etc.) using
`Statistics` and `StatisticsMask`. Only the bits set in `computed_mask` carry
valid values; all other fields are zero and MUST be ignored by readers:

<div class="lang-tabs">

```rust
use hurray_core::{
    BufferHandle, DeviceTag, ElementType, Shape, SyncMode, MIN_BUFFER_ALIGNMENT,
    descriptor::{Statistics, StatisticsMask, TensorDescriptor},
    layout::LayoutDescriptor,
};

// Construct Statistics with all fields explicitly — only VALUE_RANGE_VALID and
// NAN_INF_VALID bits are set; unset-mask fields are zero (undefined by spec).
let stats = Statistics {
    computed_mask: StatisticsMask(
        StatisticsMask::VALUE_RANGE_VALID | StatisticsMask::NAN_INF_VALID,
    ),
    nnz: 0,
    sparsity_ratio: 0.0,
    value_min: -1.0,
    value_max:  1.0,
    value_abs_max: 1.0,
    value_mean: 0.0,
    value_stddev: 0.0,
    nm_n: 0,
    nm_m: 0,
    has_nan: false,
    has_inf: false,
};

let shape  = Shape::new(vec![8u64, 8]).unwrap();
let buffer = BufferHandle::new(128, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap();

let desc = TensorDescriptor::new(
    1, 0, ElementType::Float16, shape, 0,
    LayoutDescriptor::ColMajor, vec![buffer],
    None, None, Some(stats), None,
).unwrap();

// Statistics section appends 72 bytes; encode then decode.
let bytes   = desc.encode().unwrap();
let decoded = TensorDescriptor::decode(&bytes).unwrap();
let s = decoded.statistics.as_ref().unwrap();
assert!(s.computed_mask.value_range_valid());
assert!(!s.has_nan);
```

```python
import hurray

# Python derives computed_mask from which arguments you pass, so a value can
# never be present with its validity bit unset.
# value_min, value_max and value_abs_max share one validity bit, so they are
# supplied together — Python enforces that rather than letting you set a bit
# for a value you did not compute.
stats = hurray.Statistics(
    value_min=-1.0, value_max=1.0, value_abs_max=1.0, has_nan=False, has_inf=False
)

tensor = hurray.Tensor(bytes(128), hurray.float16, [8, 8], statistics=stats)

decoded = hurray.Descriptor.decode(tensor.descriptor.encode())

assert decoded.statistics.value_min == -1.0
assert decoded.statistics.has_nan is False
```

</div>

## Shard annotations

When a tensor is a rectangular sub-region of a larger logical tensor (e.g., a row shard of a
matrix), attach a `ShardDescriptor`. The `parent_shape` rank must match the tensor's rank and
`shard_offset[k] + shape[k] <= parent_shape[k]` must hold for every dimension `k`:

<div class="lang-tabs">

```rust
use hurray_core::{
    BufferHandle, DeviceTag, ElementType, Shape, SyncMode, MIN_BUFFER_ALIGNMENT,
    descriptor::{ShardDescriptor, TensorDescriptor},
    layout::LayoutDescriptor,
};

// Shard: rows 2048..3071 of a 4096×1024 parent matrix.
let shape  = Shape::new(vec![1024u64, 1024]).unwrap();
let buffer = BufferHandle::new(524_288, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap();
let shard  = ShardDescriptor::new(
    vec![4096u64, 1024], // parent_shape
    vec![2048u64, 0],    // shard_offset (origin in parent)
).unwrap();

let desc = TensorDescriptor::new(
    1, 0, ElementType::Int4, shape, 0,
    LayoutDescriptor::RowMajor, vec![buffer],
    None, Some(shard), None, None,
).unwrap();

let bytes   = desc.encode().unwrap();
let decoded = TensorDescriptor::decode(&bytes).unwrap();

let s = decoded.shard.as_ref().unwrap();
assert_eq!(s.parent_shape, [4096, 1024]);
assert_eq!(s.shard_offset, [2048, 0]);
```

```python
import hurray

# Shard: rows 2048..3071 of a 4096x1024 parent matrix.
tensor = hurray.Tensor(
    bytes(524_288),
    hurray.dtype.int4,
    [1024, 1024],
    shard=hurray.Shard([4096, 1024], [2048, 0]),
)

decoded = hurray.Descriptor.decode(tensor.descriptor.encode())

assert decoded.shard.parent_shape == (4096, 1024)
assert decoded.shard.shard_offset == (2048, 0)
```

</div>

## Validation errors

`TensorDescriptor::new` rejects invalid combinations:

```rust
use hurray_core::{
    BufferHandle, DeviceTag, ElementType, Shape, SyncMode, MIN_BUFFER_ALIGNMENT, Error,
    descriptor::TensorDescriptor,
    layout::LayoutDescriptor,
};

let shape  = Shape::new(vec![4u64, 4]).unwrap();
let buffer = BufferHandle::new(64, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap();

// Empty buffer table is rejected.
let result = TensorDescriptor::new(
    1, 0, ElementType::Float32, shape.clone(), 0,
    LayoutDescriptor::RowMajor, vec![], // no buffers
    None, None, None, None,
);
assert!(matches!(result, Err(Error::EmptyBufferTable)));

// Extension type flag must be consistent with the element type tag.
// Float32 (tag 0x03) is not an extension type — providing ExtensionTypeDescriptor is an error.
use hurray_core::descriptor::ExtensionTypeDescriptor;
// An 8-bit float, 1-4-3. A float's sign is sign_bits; is_signed stays false.
let ext = ExtensionTypeDescriptor::new(8, 1, true, false, 1, 4, 3, 7, true, false).unwrap();
let result = TensorDescriptor::new(
    1, 0, ElementType::Float32, shape.clone(), 0,
    LayoutDescriptor::RowMajor, vec![buffer.clone()],
    None, None, None,
    Some(ext), // mismatch: Float32 is not an extension type
);
assert!(matches!(result, Err(Error::ExtensionTypeFlagMismatch { .. })));
```

## Decode errors

`TensorDescriptor::decode` rejects malformed inputs:

```rust
use hurray_core::{descriptor::TensorDescriptor, Error};

// Truncated input.
let result = TensorDescriptor::decode(&[0x48, 0x52, 0x52, 0x59]);
assert!(matches!(
    result,
    Err(Error::DescriptorTooShort { .. } | Error::DescriptorTruncated { .. })
));

// Wrong magic bytes.
let mut bad = vec![0u8; 61];
bad[0..4].copy_from_slice(b"BAAD");
let result = TensorDescriptor::decode(&bad);
assert!(matches!(result, Err(Error::InvalidMagic { .. })));
```

## Wire format anatomy (61-byte example)

```text
Offset  Size  Field
──────  ────  ─────────────────────────────────────────────────────────────
0x00    4     magic "HRRY" (0x48 0x52 0x52 0x59)
0x04    1     version_major = 0x01
0x05    1     version_minor = 0x00
0x06    4     descriptor_length = 61 (0x3D 0x00 0x00 0x00, little-endian)
0x0A    4     flags = 0x00000000 (no optional sections)
0x0E    1     type_tag = 0x03 (float32)
0x0F    1     layout_tag = 0x01 (row-major)
0x10    4     rank = 2 (0x02 0x00 0x00 0x00)
0x14    8     shape[0] = 3 (0x03 0x00 0x00 0x00 0x00 0x00 0x00 0x00)
0x1C    8     shape[1] = 4 (0x04 0x00 0x00 0x00 0x00 0x00 0x00 0x00)
0x24    8     byte_offset = 0
              ── layout payload: RowMajor has no additional bytes ──
0x2C    1     buffer_count = 1
0x2D    8     buffer[0].size_bytes = 192
0x35    4     buffer[0].alignment = 64 (0x40 0x00 0x00 0x00, little-endian uint32)
0x39    1     buffer[0].device_tag = 0x00 (CPU)
0x3A    3     buffer[0]._reserved = 0 0 0
              ── no optional sections (flags == 0) ──
```
