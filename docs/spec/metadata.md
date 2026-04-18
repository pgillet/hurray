# Metadata — Hurray Format Specification

> **Status:** Draft

## Scope

This section defines the binary encoding of the **tensor descriptor**: the
self-describing header that precedes every tensor data buffer in the Hurray format.
The tensor descriptor encodes all information required to interpret a tensor: its
element type, rank, shape, memory layout, buffer table, and optional quantization and
shard annotations.

> **Note (non-normative):** The tensor descriptor is designed to be self-delimiting:
> a receiver can determine its total byte length from the first 10 bytes, without
> reading the entire descriptor. This property is essential for streaming readers.

## Normative Requirements

> This section uses RFC 2119 key words: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT,
> SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

---

## Overall Descriptor Structure

A tensor descriptor consists of a **fixed header**, followed by **variable-length
core fields**, followed by **layout-specific fields**, followed by a **buffer table**,
followed by zero or more **optional sections** selected by the flags field.

```
[Fixed Header]         20 bytes
[shape]                8 × rank bytes
[byte_offset]          8 bytes
[Layout-specific]      variable (layout-dependent)
[Buffer Table]         variable
[Quantization]         variable, present if HAS_QUANTIZATION flag is set
[Shard]                variable, present if HAS_SHARD flag is set
[Extension Type]       variable, present if HAS_EXTENSION_TYPE flag is set
```

All multi-byte fields MUST be encoded in little-endian byte order (least significant
byte at the lowest address).

---

## Fixed Header

The fixed header occupies the first 20 bytes of every tensor descriptor.

| Offset | Field | Type | Description |
|--------|-------|------|-------------|
| 0 | `magic` | `bytes[4]` | Magic bytes: `0x48 0x52 0x52 0x59` (ASCII "HRRY"). |
| 4 | `version_major` | `uint8` | Major format version. Current value: `0x01`. |
| 5 | `version_minor` | `uint8` | Minor format version. Current value: `0x00`. |
| 6 | `descriptor_length` | `uint32` | Total length of the tensor descriptor in bytes, including the fixed header. A reader MUST use this field to advance past the descriptor without parsing all fields. |
| 10 | `flags` | `uint32` | Descriptor flags bitmask (see [Flags](#flags)). |
| 14 | `type_tag` | `uint8` | Element type tag (see `element-types.md`). |
| 15 | `layout_tag` | `uint8` | Memory layout tag (see `memory-layout.md`). |
| 16 | `rank` | `uint32` | Number of dimensions. A rank of `0` denotes a scalar tensor. |

A reader MUST reject a descriptor whose `magic` field is not `0x48 0x52 0x52 0x59`.

A reader MUST reject a descriptor whose `version_major` exceeds the reader's
supported major version.

A reader MUST reject a descriptor whose `descriptor_length` is less than 20 (the
minimum valid descriptor size).

A reader MUST NOT read beyond `descriptor_length` bytes when parsing a descriptor.

### Flags

| Bit | Name | Meaning |
|-----|------|---------|
| 0 | `HAS_QUANTIZATION` | A quantization descriptor section is present (see [Quantization Section](#quantization-section)). |
| 1 | `HAS_SHARD` | A shard descriptor section is present (see [Shard Section](#shard-section)). |
| 2 | `HAS_EXTENSION_TYPE` | An extension type descriptor section is present. MUST be set if and only if `type_tag` is in the range `0xF0`–`0xFE` (see [Extension Type Section](#extension-type-section)). |
| 3–31 | (reserved) | MUST be `0`. A reader MUST reject a descriptor with any reserved flag bit set. |

---

## Core Variable Fields

Immediately following the fixed header:

### shape

`uint64[rank]` — the size of each dimension, in ascending dimension order (dimension
0 first). Each value MUST be greater than or equal to 0. A dimension size of 0
indicates an empty tensor.

The value `0xFFFFFFFFFFFFFFFF` (`UINT64_MAX`) is the **dynamic dimension sentinel**:
it indicates that the dimension's size is not statically known. A reader MUST NOT
compute buffer sizes, strides, or element counts for a dimension carrying this
sentinel without first resolving it to a concrete value.

For a scalar tensor (`rank = 0`), this field is absent (zero bytes).

### byte_offset

`uint64` — the byte offset from the start of buffer 0 in the buffer table to the
element at logical index `[0, 0, ..., 0]`. MUST be less than or equal to the byte
size of buffer 0.

For sub-byte types (`bool`, `int4`, `uint4`, `int2`, `uint2`), `byte_offset` MUST
point to a byte boundary. The first element begins at bit 0 of that byte.

---

## Layout-Specific Fields

Immediately following `byte_offset`, the layout-specific fields for the layout
identified by `layout_tag` are encoded. If the layout has no additional fields, this
section is absent.

### Row-Major (`0x01`) and Column-Major (`0x02`)

No additional fields. Strides are implicit and computed as defined in
`memory-layout.md`.

### Strided (`0x03`)

| Field | Type | Description |
|-------|------|-------------|
| `strides` | `int64[rank]` | Stride of each dimension in logical elements. |

### Tiled / Blocked (`0x04`)

| Field | Type | Description |
|-------|------|-------------|
| `tile_shape` | `uint64[rank]` | Tile size along each dimension. Every value MUST be greater than 0. |
| `outer_layout` | `uint8` | Layout tag for tile-grid ordering. MUST be `0x01`, `0x02`, or `0x03`. |
| `inner_layout` | `uint8` | Layout tag for element ordering within each tile. MUST be `0x01`, `0x02`, `0x03`, or `0x04` (recursive tiling). |
| `_reserved` | `uint8[2]` | MUST be `0x00`. |

If `outer_layout` is `0x03` (strided):

| Field | Type | Description |
|-------|------|-------------|
| `outer_strides` | `int64[rank]` | Outer strides in units of tiles (not elements). |

If `inner_layout` is `0x03` (strided):

| Field | Type | Description |
|-------|------|-------------|
| `inner_strides` | `int64[rank]` | Inner strides in logical elements within a tile. |

If `inner_layout` is `0x04` (recursive tiling), the tiled layout-specific fields are
encoded recursively at this point, beginning with `tile_shape`.

A reader MUST enforce a maximum recursion depth for nested tiled descriptors. The
RECOMMENDED limit is 8 levels. A reader MUST reject a descriptor that exceeds its
configured recursion limit.

### Morton (Z-Order Curve) (`0x05`)

| Field | Type | Description |
|-------|------|-------------|
| `morton_bits` | `uint32[rank]` | Number of bits used per dimension in the Morton encoding. Each value MUST be greater than 0. |

### General Subpaving (`0x06`)

| Field | Type | Description |
|-------|------|-------------|
| `region_count` | `uint32` | Number of regions. MUST be greater than 0. |

Followed by `region_count` **region descriptors**, each encoded as:

| Field | Type | Description |
|-------|------|-------------|
| `origin` | `uint64[rank]` | Starting index of the region along each dimension (inclusive). |
| `region_shape` | `uint64[rank]` | Size of the region along each dimension. Every value MUST be greater than 0. |
| `region_layout_tag` | `uint8` | Layout of elements within this region. MUST NOT be `0x00` or `0xFF`. |
| `_reserved` | `uint8[3]` | MUST be `0x00`. |
| `buffer_index` | `uint32` | Index of the data buffer in the buffer table that holds this region. |
| `region_byte_offset` | `uint64` | Byte offset within the referenced buffer to the start of this region's data. |

After `region_byte_offset`, the layout-specific fields for `region_layout_tag` are
encoded inline (recursively if `region_layout_tag` is `0x04` or `0x06`).

### Hilbert Curve (`0x40`)

| Field | Type | Description |
|-------|------|-------------|
| `hilbert_order` | `uint32` | Order of the Hilbert curve. MUST be greater than 0. |
| `hilbert_rank` | `uint32` | Number of curve dimensions. MUST equal `rank`. MUST be greater than or equal to 2. |

### Extension Layouts (`0xF0`–`0xFE`)

| Field | Type | Description |
|-------|------|-------------|
| `extension_layout_id` | `uint64` | Implementation-defined layout identifier. |
| `extension_data_length` | `uint32` | Byte length of the opaque metadata that follows. |
| `extension_data` | `bytes[extension_data_length]` | Opaque layout-specific metadata. |

A reader that does not recognise `extension_layout_id` MUST reject the descriptor,
unless operating in permissive mode.

---

## Buffer Table

Immediately following the layout-specific fields, the buffer table is encoded.

| Field | Type | Description |
|-------|------|-------------|
| `buffer_count` | `uint8` | Number of buffer handles. MUST be at least 1. For all dense layout tags (`0x01`–`0x06`, `0x40`), MUST be exactly `0x01`. |

Followed by `buffer_count` **buffer handles**, each encoded as 16 bytes:

| Offset | Field | Type | Description |
|--------|-------|------|-------------|
| 0 | `byte_size` | `uint64` | Size of the buffer in bytes. |
| 8 | `alignment` | `uint32` | Minimum buffer alignment in bytes. MUST be a power of two and MUST be at least 64. |
| 12 | `device_tag` | `uint8` | Device where this buffer resides (see `buffer-protocol.md` and Device Tags in `interchange.md`). |
| 13 | `_reserved` | `uint8[3]` | MUST be `0x00`. |

> **Note (non-normative):** For sparse layout tags (COO, CSR, etc., to be assigned in
> a future revision), `buffer_count` will exceed 1. Each entry holds a distinct
> component array (values, indices, pointers). The layout-specific fields use
> `buffer_index` to address individual entries. Requiring `buffer_count = 0x01` for
> all current dense layouts costs one byte but gives every descriptor a uniform
> structure: decoders always read the count first and allocate the right number of
> handles without special-casing.

---

## Quantization Section

Present if and only if the `HAS_QUANTIZATION` flag (bit 0) is set.

| Field | Type | Description |
|-------|------|-------------|
| `quantization_length` | `uint32` | Byte length of the quantization descriptor that follows. |
| `quantization_descriptor` | `bytes[quantization_length]` | Binary encoding of the quantization descriptor, as defined in `quantization.md`. |

A reader that encounters `HAS_QUANTIZATION` but does not support quantized types MUST
reject the descriptor unless operating in permissive mode.

---

## Shard Section

Present if and only if the `HAS_SHARD` flag (bit 1) is set.

| Field | Type | Description |
|-------|------|-------------|
| `parent_shape` | `uint64[rank]` | Shape of the logical parent tensor. MUST have the same rank as the tensor. |
| `shard_offset` | `uint64[rank]` | Starting index of this shard within the parent tensor along each dimension. |

The constraint `shard_offset[k] + shape[k] <= parent_shape[k]` MUST hold for every
dimension `k`. A reader MUST reject a shard descriptor that violates this constraint.

---

## Extension Type Section

Present if and only if the `HAS_EXTENSION_TYPE` flag (bit 2) is set. This flag MUST
be set whenever `type_tag` is in the range `0xF0`–`0xFE`, and MUST NOT be set
otherwise.

Per ADR-001, extension type tags MUST carry an inline descriptor providing at minimum
the bit width and packing parameters required to compute buffer sizes.

The extension type descriptor is 20 bytes:

| Offset | Field | Type | Description |
|--------|-------|------|-------------|
| 0 | `bit_width` | `uint32` | Bit width of one element. MUST be greater than 0. |
| 4 | `packing_factor` | `uint8` | Elements packed per byte. MUST be 1 for whole-byte types (`bit_width >= 8`). For sub-byte types, MUST equal `8 / bit_width`; `bit_width` MUST be a power of two less than 8. |
| 5 | `is_float` | `uint8` | `0x01` if floating-point, `0x00` if integer. |
| 6 | `is_signed` | `uint8` | `0x01` if signed integer. MUST be `0x00` for float types. |
| 7 | `sign_bits` | `uint8` | Number of sign bits (for float types). MUST be 0 or 1. MUST be `0x00` for integer types. |
| 8 | `exponent_bits` | `uint8` | Number of exponent bits (for float types). MUST be `0x00` for integer types. |
| 9 | `mantissa_bits` | `uint8` | Number of mantissa bits (for float types). MUST be `0x00` for integer types. |
| 10 | `_reserved` | `uint8[2]` | MUST be `0x00`. |
| 12 | `exponent_bias` | `uint32` | Exponent bias (for float types). MUST be `0x00000000` for integer types. |
| 16 | `has_nan` | `uint8` | `0x01` if NaN is representable (float types only). |
| 17 | `has_inf` | `uint8` | `0x01` if infinity is representable (float types only). |
| 18 | `_reserved2` | `uint8[2]` | MUST be `0x00`. |

A reader MUST use `bit_width` and `packing_factor` to compute buffer sizes for
tensors with extension type tags, even if it does not interpret the numeric semantics
of the type.

---

## Version Compatibility

A reader MUST reject a descriptor whose `version_major` exceeds the reader's
supported major version.

A reader encountering a `version_minor` greater than its supported minor version
SHOULD accept the descriptor but MUST NOT interpret fields beyond what its supported
minor version defines. The `descriptor_length` field allows the reader to skip the
descriptor entirely if desired.

> **Note (non-normative):** Minor version increments add optional fields or new flag
> bits. A reader built against version 1.0 will correctly skip a 1.1 descriptor by
> consuming exactly `descriptor_length` bytes, because all reserved flag bits were
> required to be 0 in 1.0. Major version increments signal backward-incompatible
> changes; a reader MUST NOT attempt to parse a descriptor with an unsupported major
> version.

---

## Worked Example

A rank-2 `float32` tensor with shape `[3, 4]` in row-major layout, one buffer of
192 bytes aligned to 64 bytes on CPU, no optional sections:

```
Offset  Value (hex)                   Field
------  ----------------------------  -----
0       48 52 52 59                   magic = "HRRY"
4       01                            version_major = 1
5       00                            version_minor = 0
6       3D 00 00 00                   descriptor_length = 61
10      00 00 00 00                   flags = 0x00000000 (no optional sections)
14      03                            type_tag = 0x03 (float32)
15      01                            layout_tag = 0x01 (row-major)
16      02 00 00 00                   rank = 2
20      03 00 00 00 00 00 00 00       shape[0] = 3
28      04 00 00 00 00 00 00 00       shape[1] = 4
36      00 00 00 00 00 00 00 00       byte_offset = 0
                                      (no layout-specific fields for row-major)
44      01                            buffer_count = 1
45      C0 00 00 00 00 00 00 00       buffer[0].byte_size = 192
53      40 00 00 00                   buffer[0].alignment = 64
57      00                            buffer[0].device_tag = 0x00 (CPU)
58      00 00 00                      buffer[0]._reserved
```

Total: 61 bytes. Fixed header (20) + shape (16) + byte_offset (8) + buffer table
(1 + 16) = 61.

---

## Interaction with Other Sections

- **Element Types (`element-types.md`)**: defines `type_tag` values and the bit-width,
  packing, and alignment properties used during buffer size computation.
- **Memory Layout (`memory-layout.md`)**: defines `layout_tag` values and the
  layout-specific fields encoded in this descriptor.
- **Quantization (`quantization.md`)**: defines the binary format of the
  `quantization_descriptor` payload in the quantization section.
- **Buffer Protocol (`buffer-protocol.md`)**: defines buffer ownership, device memory
  semantics, and release callback conventions referenced by the buffer table entries.
- **Interchange (`interchange.md`)**: the tensor descriptor defined here is transmitted
  verbatim in `TENSOR_DESCRIPTOR` and `TENSOR_PUT` message payloads, followed by
  transport-specific fields (`total_data_bytes`, `shard_index`, `total_shards`).

---

## Open Questions

> **[OQ-1]:** Should the descriptor include a checksum field (e.g., CRC-32 of the
> descriptor body) for corruption detection? This would add 4 bytes to every descriptor
> and require a full-pass computation on write. Alternatively, corruption detection
> could be delegated to the transport layer (TCP checksum, TLS). Resolution pending.

> **[OQ-2]:** The binary encoding of the quantization descriptor is deferred to
> `quantization.md`. The `quantization_length` prefix ensures readers can skip it
> safely in the interim.
