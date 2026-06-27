# Metadata — Hurray Format Specification

> **Status:** Draft

## Scope

This section defines the binary encoding of the **tensor descriptor**: the
self-describing header that precedes every tensor data buffer in the Hurray format.
The tensor descriptor encodes all information required to interpret a tensor: its
element type, rank, shape, memory layout, buffer table, and optional quantization and
shard annotations.

> **Note (non-normative):** The tensor descriptor is designed to be self-delimiting:
> a receiver can determine its total byte length from the first 10 bytes (the
> `descriptor_length` field occupies bytes 6–9) and MAY skip the descriptor entirely
> without parsing any layout-specific fields. Skipping past the *whole* descriptor is
> what only requires the first 10 bytes; locating or skipping a particular section
> *within* the descriptor (e.g. the buffer table or the quantization section) requires
> reading through the preceding fields. This property is essential for streaming readers.

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
[Statistics]           72 bytes, present if HAS_STATISTICS flag is set
[Extension Type]       20 bytes, present if HAS_EXTENSION_TYPE flag is set
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
| 15 | `layout_tag` | `uint8` | Memory layout tag (see `memory-layout.md`). See `data-model.md` § Scalar Tensors for layout restrictions that apply when `rank = 0`. |
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
| 3 | `HAS_STATISTICS` | A statistics section is present (see [Statistics Section](#statistics-section)). |
| 4–31 | (reserved) | MUST be `0`. A reader MUST reject a descriptor with any reserved flag bit set. |

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
| `region_layout_length` | `uint32` | Byte count of the inner layout payload that follows. MUST be `0` for `region_layout_tag` values `0x01` and `0x02`. |
| `region_layout_payload` | `bytes[region_layout_length]` | Layout-specific fields for `region_layout_tag`, encoded identically to the Layout-Specific Fields section above for that tag, with the tag byte omitted. |

Recursive subpaving (`region_layout_tag = 0x06`) is permitted. A reader MUST
reject any descriptor where the subpaving nesting depth exceeds 8 levels.

### COO (`0x07`)

| Field | Type | Description |
|-------|------|-------------|
| `nnz` | `uint64` | Number of stored (non-zero) elements. MAY be 0 for an empty sparse tensor. |
| `is_sorted` | `uint8` | `0x01` if the non-zeros are sorted in lexicographic index order (dimension 0 major); `0x00` otherwise. |
| `_reserved` | `uint8[7]` | MUST be `0x00`. |

See `layouts/coo.md` for buffer table composition, storage order, and validity
constraints.

### CSR (`0x08`)

| Field | Type | Description |
|-------|------|-------------|
| `nnz` | `uint64` | Number of stored (non-zero) elements. MAY be 0 for an empty sparse matrix. |
| `_reserved` | `uint8[8]` | MUST be `0x00`. |

See `layouts/csr.md` for buffer table composition, storage invariants, and
rank-2 restriction.

### CSC (`0x09`)

| Field | Type | Description |
|-------|------|-------------|
| `nnz` | `uint64` | Number of stored (non-zero) elements. MAY be 0 for an empty sparse matrix. |
| `_reserved` | `uint8[8]` | MUST be `0x00`. |

See `layouts/csc.md` for buffer table composition, storage invariants, and
rank-2 restriction.

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
| `buffer_count` | `uint8` | Number of buffer handles. MUST be at least 1. For dense layout tags (`0x01`–`0x06`, `0x40`) without quantization, MUST be exactly `0x01`. For quantized dense tensors, MUST equal `0x01` plus the number of quantization-parameter buffers required by the active scheme (see `quantization.md` § Buffer Table Placement Rules). |

The maximum value is `255`, imposed by the `uint8` wire type. This limit applies
to the sum of data and quantization-parameter buffers. Implementations that
require more than 255 buffers MUST use multiple tensor descriptors.

Followed by `buffer_count` **buffer handles**, each encoded as 16 bytes:

| Offset | Field | Type | Description |
|--------|-------|------|-------------|
| 0 | `byte_size` | `uint64` | Size of the buffer in bytes. |
| 8 | `alignment` | `uint32` | Minimum buffer alignment in bytes. MUST be a power of two and MUST be at least 64. |
| 12 | `device_tag` | `uint8` | Device where this buffer resides (see `buffer-protocol.md` and Device Tags in `interchange.md`). |
| 13 | `_reserved` | `uint8[3]` | MUST be `0x00`. |

The `_reserved` bytes MUST be `0x00`. A conforming reader in strict mode MUST
reject a descriptor containing any buffer handle whose `_reserved` bytes are not
all `0x00`.

> **Note (non-normative):** For sparse layout tags (COO `0x07`, CSR `0x08`, CSC `0x09`, CSF `0x0A`), `buffer_count` exceeds 1 — each entry holds a distinct component array (values, indices, pointers). For CSF the count is rank-dependent, `2·rank + 1` (one `values` buffer plus a `pos`/`crd` pair per level). For quantized dense tensors, quantization-parameter buffers (scales, zero-points) extend the buffer table beyond the layout baseline. The layout-defined minimum is always `0x01` for dense layouts; quantization schemes append their parameter buffers on top.

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

## Statistics Section

Present if and only if the `HAS_STATISTICS` flag (bit 3) is set.

The statistics section is a fixed-size **72-byte** block. All statistics are
**advisory**: they reflect the tensor data at write time. A reader MUST NOT rely on
any statistic for correctness; statistics MAY be used only as optimization hints
(algorithm selection, memory pre-allocation, routing decisions).

> **Note (non-normative):** A streaming writer that has not processed the entire
> tensor buffer before emitting the descriptor (e.g., a pipeline stage forwarding data
> on the fly) MUST omit the statistics section (`HAS_STATISTICS` not set) rather than
> emitting invalid statistics. A writer that knows only a subset of statistics (e.g.,
> `nnz` is known from a sparse format but `value_mean` was not computed) MUST mark
> the unknown fields as not valid in `computed_mask`.

### computed_mask

The `computed_mask` field (first 4 bytes of the section) is a bitmask indicating
which statistics fields contain valid data. A reader MUST check the relevant bit
before using any field. Fields whose bit is not set MUST be treated as unknown,
regardless of their encoded value.

| Bit | Name | Covers |
|-----|------|--------|
| 0 | `NNZ_VALID` | `nnz` |
| 1 | `SPARSITY_VALID` | `sparsity_ratio` |
| 2 | `VALUE_RANGE_VALID` | `value_min`, `value_max`, `value_abs_max` |
| 3 | `VALUE_STATS_VALID` | `value_mean`, `value_stddev` |
| 4 | `NM_SPARSITY_VALID` | `nm_n`, `nm_m` |
| 5 | `NAN_INF_VALID` | `has_nan`, `has_inf` |
| 6–31 | (reserved) | MUST be `0`. |

A conforming reader MUST reject a descriptor whose `computed_mask` has any
reserved bit set (any bit greater than or equal to 6, given the six defined
statistics fields above).

### Field Encoding

The 72-byte statistics block is encoded as follows. All multi-byte fields are
little-endian.

| Offset | Field | Type | Description |
|--------|-------|------|-------------|
| 0 | `computed_mask` | `uint32` | Validity bitmask (see above). |
| 4 | `_reserved` | `uint32` | MUST be `0x00000000`. |
| 8 | `nnz` | `uint64` | Number of non-zero elements. Valid when `NNZ_VALID` is set. |
| 16 | `sparsity_ratio` | `float64` | Fraction of zero elements: `(total_elements - nnz) / total_elements`. Range `[0.0, 1.0]`. Valid when `SPARSITY_VALID` is set. |
| 24 | `value_min` | `float64` | Minimum element value, dequantized to `float64`. Valid when `VALUE_RANGE_VALID` is set. |
| 32 | `value_max` | `float64` | Maximum element value, dequantized to `float64`. Valid when `VALUE_RANGE_VALID` is set. |
| 40 | `value_abs_max` | `float64` | Maximum absolute element value (`max(abs(value_min), abs(value_max))`). Key input for symmetric quantization range calibration. Valid when `VALUE_RANGE_VALID` is set. |
| 48 | `value_mean` | `float64` | Arithmetic mean of all elements, dequantized to `float64`. Valid when `VALUE_STATS_VALID` is set. |
| 56 | `value_stddev` | `float64` | Population standard deviation of all elements, dequantized to `float64`. MUST be greater than or equal to `0.0`. Valid when `VALUE_STATS_VALID` is set. |
| 64 | `nm_n` | `uint8` | N in the N:M structured sparsity pattern (e.g., `2` for 2:4 sparsity). `0x00` if not applicable. Valid when `NM_SPARSITY_VALID` is set. |
| 65 | `nm_m` | `uint8` | M in the N:M structured sparsity pattern (e.g., `4` for 2:4 sparsity). MUST satisfy `nm_n <= nm_m`. `0x00` if not applicable. Valid when `NM_SPARSITY_VALID` is set. |
| 66 | `has_nan` | `uint8` | `0x01` if at least one NaN element is present; `0x00` if no NaN was found. Valid when `NAN_INF_VALID` is set. |
| 67 | `has_inf` | `uint8` | `0x01` if at least one positive or negative infinity is present; `0x00` otherwise. Valid when `NAN_INF_VALID` is set. |
| 68 | `_reserved2` | `uint8[4]` | MUST be `0x00`. |

> **Note (non-normative):** `value_min`, `value_max`, `value_abs_max`, `value_mean`,
> and `value_stddev` are always expressed in `float64` regardless of the tensor's
> element type. For quantized tensors, these values reflect dequantized (real-valued)
> statistics, not the raw quantized storage values. For `bool` types, the statistics
> are defined over the integer domain `{0, 1}`.

> **Note (non-normative):** N:M structured sparsity is particularly relevant for
> NVIDIA Ampere/Ada/Hopper Tensor Cores, which provide hardware-accelerated 2:4
> sparsity (2 non-zeros in every group of 4 consecutive elements). Declaring the
> N:M pattern in the descriptor lets a receiver select the sparse kernel path without
> scanning the buffer.

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
| 4 | `packing_factor` | `uint8` | Number of elements packed per byte. MUST be exactly `1` when `bit_width` is greater than or equal to `8`. When `bit_width` is less than `8`, `bit_width` MUST be one of `1`, `2`, or `4`, and `packing_factor` MUST equal `8 / bit_width` (that is, `8`, `4`, or `2` respectively). All other sub-byte widths — including but not limited to `3`, `5`, `6`, and `7` bits — MUST NOT be encoded as extension types. A reader MUST reject an extension type descriptor that violates these constraints. |
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

Sub-byte element widths that are not a power of two (notably `6`-bit) are reserved to the built-in type tag space. Implementors requiring an interchange-portable non-power-of-two sub-byte type MUST request a built-in tag assignment through the specification governance process rather than encoding the type in the private extension range. The extension descriptor's whole-byte and power-of-two sub-byte width restriction ensures that buffer-size computation remains a single integer formula (`ceil(N / packing_factor)` for sub-byte, `N * (bit_width / 8)` for whole-byte) without rational arithmetic.

> **Note (non-normative):** The 6-bit `float6_e2m3` (`0x44`) and `float6_e3m2` (`0x45`) types are built-in (Tier 2) and use a dedicated 4-elements-per-3-bytes packing defined in `element-types.md`. Their packing rule is not expressible as `8 / bit_width` and is therefore not delegable to the generic extension descriptor. The extension descriptor is designed for private, implementation-defined types whose layout fits the simple "elements per byte" model; richer packings remain the prerogative of the standardized type system.

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

> **[OQ-1]:** ~~Should the descriptor include a CRC-32 checksum field?~~ **Resolved:** No checksum in the descriptor. Integrity is delegated to the transport/storage layer (TCP, TLS, ECC, ZFS). Adding 4 bytes and a full-pass CRC on every descriptor would penalise in-process and IPC interchange where corruption is not a realistic threat. If file-level integrity is needed, it belongs in the file format footer (see `file-format.md` OQ-3).

> **[OQ-2]:** ~~The binary encoding of the quantization descriptor is deferred to `quantization.md`.~~ **Resolved:** The encoding is fully defined in `quantization.md`: a fixed 4-byte header (`scheme_tag`, `scheme_version`, `flags`) followed by a per-scheme payload. Complete byte-level layouts are specified in `quantization/per-tensor-affine.md`, `quantization/per-channel-affine.md`, `quantization/per-block-affine.md`, `quantization/nf4.md`, and `quantization/mxfp.md`. The `quantization_length` prefix in `metadata.md` allows readers to skip unrecognised schemes safely.
