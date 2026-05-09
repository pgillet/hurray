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

```rust
use hurray_core::{
    BufferHandle, DeviceTag, ElementType, Shape, MIN_BUFFER_ALIGNMENT,
    descriptor::TensorDescriptor,
    layout::LayoutDescriptor,
};

let shape  = Shape::new(vec![3u64, 4]).unwrap();
let buffer = BufferHandle::new(192, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu).unwrap();

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

## Advisory statistics

Attach pre-computed statistics (value range, NaN/Inf presence, etc.) using
`Statistics` and `StatisticsMask`. Only the bits set in `computed_mask` carry
valid values; all other fields are zero and MUST be ignored by readers:

```rust
use hurray_core::{
    BufferHandle, DeviceTag, ElementType, Shape, MIN_BUFFER_ALIGNMENT,
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
let buffer = BufferHandle::new(128, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu).unwrap();

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

## Shard annotations

When a tensor is a rectangular sub-region of a larger logical tensor (e.g., a row shard of a
matrix), attach a `ShardDescriptor`. The `parent_shape` rank must match the tensor's rank and
`shard_offset[k] + shape[k] <= parent_shape[k]` must hold for every dimension `k`:

```rust
use hurray_core::{
    BufferHandle, DeviceTag, ElementType, Shape, MIN_BUFFER_ALIGNMENT,
    descriptor::{ShardDescriptor, TensorDescriptor},
    layout::LayoutDescriptor,
};

// Shard: rows 2048..3071 of a 4096×1024 parent matrix.
let shape  = Shape::new(vec![1024u64, 1024]).unwrap();
let buffer = BufferHandle::new(524_288, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu).unwrap();
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

## Validation errors

`TensorDescriptor::new` rejects invalid combinations:

```rust
use hurray_core::{
    BufferHandle, DeviceTag, ElementType, Shape, MIN_BUFFER_ALIGNMENT, Error,
    descriptor::TensorDescriptor,
    layout::LayoutDescriptor,
};

let shape  = Shape::new(vec![4u64, 4]).unwrap();
let buffer = BufferHandle::new(64, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu).unwrap();

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
let ext = ExtensionTypeDescriptor::new(8, 1, false, false, 1, 0, 7, 0, false, false).unwrap();
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
assert!(matches!(result, Err(Error::DescriptorTooShort | Error::DescriptorTruncated)));

// Wrong magic bytes.
let mut bad = vec![0u8; 61];
bad[0..4].copy_from_slice(b"BAAD");
let result = TensorDescriptor::decode(&bad);
assert!(matches!(result, Err(Error::InvalidMagic(_))));
```

## Wire format anatomy (61-byte example)

```
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
0x35    1     buffer[0].log2_alignment = 6  (2^6 = 64 bytes)
0x36    2     buffer[0].device_tag = 0x0000 (CPU)
0x38    8     buffer[0]._reserved = 0
              ── no optional sections (flags == 0) ──
```
