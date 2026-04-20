# Quantization — Hurray Format Specification

> **Status:** Draft

## Scope

This section defines the binary encoding of the **quantization descriptor**: the
opaque payload that appears inside the Quantization Section of a tensor descriptor
(see `metadata.md` § Quantization Section). It specifies the set of supported
quantization schemes, the wire format for each scheme's parameters, and the
normative dequantization formula that a reader MUST apply to recover real-valued
elements from the quantized storage buffer.

A Hurray **quantized tensor** has two parts:

1. A **storage type** — an integer or float8 type tag defined in `element-types.md`
   that describes how each element is encoded in the tensor's data buffer.
2. A **quantization descriptor** — the subject of this file, which describes how
   to map those storage values to real-valued elements.

> **Note (non-normative):** The tensor descriptor's `type_tag` field always refers
> to the storage type. Hurray does not allocate separate type tags for quantized
> formats; a tensor is quantized if and only if the `HAS_QUANTIZATION` flag
> (bit 0 of `flags`) is set in the tensor descriptor. This separation keeps the
> type system orthogonal to the quantization scheme space.

## Normative Requirements

> This section uses RFC 2119 key words: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT,
> SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

---

## Relationship to Other Sections

- **`metadata.md`** defines the Quantization Section framing. The section is
  present if and only if the `HAS_QUANTIZATION` flag (bit 0) is set in the tensor
  descriptor. It consists of a `uint32` `quantization_length` prefix followed by
  `quantization_length` bytes of payload. This file defines the contents of those
  payload bytes.
- **`element-types.md`** defines the storage type tags (`type_tag` values) that
  may appear in a quantized tensor descriptor. Each scheme defined below lists
  the set of storage types it accepts.
- **`memory-layout.md`** defines the buffer table: an ordered list of buffer
  handles indexed from `0`. Several schemes below store their per-block scale
  and zero-point arrays in buffers other than buffer 0; those schemes reference
  those buffers by index.

---

## Descriptor Header

Every quantization descriptor begins with a **fixed 4-byte header**.

| Offset | Field | Type | Description |
|--------|-------|------|-------------|
| 0 | `scheme_tag` | `uint8` | Identifies the quantization scheme. Values are assigned below. |
| 1 | `scheme_version` | `uint8` | Version of the scheme-specific encoding. Current value for all schemes defined here: `0x01`. |
| 2 | `flags` | `uint16` | Scheme-specific flags bitmask. Reserved bits MUST be `0`. |

Immediately following the 4-byte header, **scheme-specific fields** are encoded.
The total length of the descriptor (header plus scheme-specific fields, plus any
trailing padding chosen by the writer) MUST equal the `quantization_length`
prefix defined in `metadata.md`.

A reader MUST dispatch on `scheme_tag` after reading the first byte. A reader
that does not recognise `scheme_tag` MUST reject the tensor descriptor, unless
operating in permissive mode. In permissive mode, the reader MAY skip past the
descriptor using the `quantization_length` prefix but MUST NOT dereference the
tensor data buffer.

A reader MUST reject a descriptor whose `scheme_version` exceeds the highest
version defined in this specification for the given `scheme_tag`.

A reader MUST reject a descriptor with any reserved `flags` bit set.

A reader MUST NOT read beyond `quantization_length` bytes when parsing the
descriptor.

All multi-byte fields in the quantization descriptor MUST be encoded in
little-endian byte order.

### Scheme Tag Space

| Range | Allocation |
|-------|------------|
| `0x00` | Reserved (invalid) |
| `0x01` – `0x3F` | Tier 1 schemes (MUST be supported by conforming implementations that advertise quantization support) |
| `0x40` – `0x7F` | Tier 2 schemes (OPTIONAL) |
| `0x80` – `0xEF` | Reserved for future specification versions |
| `0xF0` – `0xFE` | Implementation-private extension schemes |
| `0xFF` | Reserved (invalid) |

A reader MUST reject a descriptor whose `scheme_tag` is `0x00` or `0xFF`.

Tags in the range `0x80` – `0xEF` MUST NOT be used by any implementation; they
are reserved for future specification versions.

Tags in the range `0xF0` – `0xFE` MAY be used by implementations for private
schemes. Tensors using private scheme tags MUST NOT be exchanged between
independent implementations unless both parties have agreed on the semantics
out of band.

### Assigned Scheme Tags

| Scheme | Tag | Tier | Section |
|--------|-----|------|---------|
| Per-tensor affine | `0x01` | 1 | [§ Per-Tensor Affine](#per-tensor-affine-0x01) |
| Per-channel affine | `0x02` | 1 | [§ Per-Channel Affine](#per-channel-affine-0x02) |
| Per-block affine | `0x03` | 1 | [§ Per-Block Affine](#per-block-affine-0x03) |
| NF4 (NormalFloat4) | `0x04` | 2 | [§ NF4](#nf4-normalfloat4-0x04) |
| MXFP (OCP Microscaling) | `0x05` | 2 | [§ MXFP](#mxfp-ocp-microscaling-0x05) |

---

## Per-Tensor Affine (`0x01`)

A single `scale` and `zero_point` pair applies to every element of the tensor.
This scheme covers both asymmetric quantization (arbitrary `zero_point`) and
symmetric quantization (`zero_point = 0`).

### Binary Encoding

Total descriptor length: **16 bytes**.

| Offset | Field | Type | Description |
|--------|-------|------|-------------|
| 0 | `scheme_tag` | `uint8` | MUST be `0x01`. |
| 1 | `scheme_version` | `uint8` | MUST be `0x01`. |
| 2 | `flags` | `uint16` | MUST be `0x0000`. No flags are defined for this scheme. |
| 4 | `scale` | `float32` | Dequantization scale. MUST be a finite, non-zero value. |
| 8 | `zero_point` | `int32` | Quantization zero point. For symmetric quantization, MUST be `0x00000000`. |
| 12 | `_reserved` | `uint8[4]` | MUST be `0x00`. |

### Dequantization Formula

For each storage element `q`:

```
x_real = scale * (q - zero_point)
```

The subtraction is performed in signed 32-bit integer arithmetic. The result of
the subtraction MUST then be converted to `float32` and multiplied by `scale`.
The real-valued element type produced by dequantization is `float32`. A consumer
MAY further convert to `float64` or to a lower-precision float type; such
conversion is out of scope for this specification.

### Validity Constraints

- `scale` MUST NOT be zero, NaN, or infinity. A reader MUST reject a descriptor
  that violates this constraint.
- `zero_point` MUST lie within the representable range of the storage type. For
  example, for a `uint8` storage type, `zero_point` MUST be in `[0, 255]`. A
  reader MUST reject a descriptor that violates this constraint.

### Valid Storage Types

The storage type (`type_tag` in the tensor descriptor) MUST be one of:

- `int8` (`0x10`), `uint8` (`0x11`)
- `int16` (`0x12`), `uint16` (`0x13`)
- `int32` (`0x14`), `uint32` (`0x15`)
- `int4` (`0x48`), `uint4` (`0x49`)
- `int2` (`0x4A`), `uint2` (`0x4B`)

A reader MUST reject a descriptor whose storage type is not in this list.

---

## Per-Channel Affine (`0x02`)

One `scale` and `zero_point` pair per slice along a specified axis. The scale
and zero-point arrays are stored in **separate buffers** listed in the tensor
descriptor's buffer table.

### Binary Encoding

Total descriptor length: **16 bytes**.

| Offset | Field | Type | Description |
|--------|-------|------|-------------|
| 0 | `scheme_tag` | `uint8` | MUST be `0x02`. |
| 1 | `scheme_version` | `uint8` | MUST be `0x01`. |
| 2 | `flags` | `uint16` | Scheme-specific flags (see below). Reserved bits MUST be `0`. |
| 4 | `axis` | `uint32` | Index of the quantized axis. MUST be strictly less than `rank`. |
| 8 | `scale_buffer_index` | `uint32` | Index in the buffer table of the buffer holding the `scale` array. |
| 12 | `zero_point_buffer_index` | `uint32` | Index in the buffer table of the buffer holding the `zero_point` array. |

**Flags bits:**

| Bit | Name | Meaning |
|-----|------|---------|
| 0 | `SYMMETRIC` | If set, the `zero_point` array is implicitly all zeros; `zero_point_buffer_index` MUST be `0xFFFFFFFF`. |
| 1–15 | (reserved) | MUST be `0`. |

### Referenced Buffers

The `scale` buffer MUST contain exactly `shape[axis]` consecutive `float32`
values in little-endian byte order, starting at byte offset `0` within the
referenced buffer. Its byte size MUST be exactly `shape[axis] * 4`.

The `zero_point` buffer — present only if the `SYMMETRIC` flag is not set —
MUST contain exactly `shape[axis]` consecutive `int32` values in little-endian
byte order, starting at byte offset `0` within the referenced buffer. Its byte
size MUST be exactly `shape[axis] * 4`.

A reader MUST reject a descriptor whose `scale_buffer_index` or (when the
`SYMMETRIC` flag is not set) `zero_point_buffer_index` is greater than or equal
to `buffer_count` in the buffer table.

A reader MUST reject a descriptor whose `scale_buffer_index` or
`zero_point_buffer_index` equals the buffer index used by the layout for tensor
data (typically `0` for dense layouts).

> **Note (non-normative):** Because dense-layout descriptors require
> `buffer_count = 0x01`, a per-channel-affine quantized dense tensor requires
> `buffer_count` to be at least 2 (symmetric case) or 3 (asymmetric case). This
> is the mechanism by which quantization schemes extend the buffer table beyond
> the dense-layout minimum.

### Dequantization Formula

For a storage element `q` at logical index `[i_0, i_1, ..., i_{rank-1}]`:

```
c = i_axis
x_real = scale[c] * (q - zero_point[c])
```

If the `SYMMETRIC` flag is set, `zero_point[c]` is treated as `0` for all `c`.

### Validity Constraints

- `axis` MUST satisfy `axis < rank`.
- `shape[axis]` MUST NOT equal `0xFFFFFFFFFFFFFFFF` (the dynamic dimension
  sentinel): per-channel quantization requires a statically known channel count.
- Every element of the `scale` array MUST be a finite, non-zero `float32` value.
- Every element of the `zero_point` array (when present) MUST lie within the
  representable range of the storage type.

A reader MAY defer the per-element validity check on the scale and zero-point
arrays to the first dequantization attempt, but MUST perform the axis and shape
checks before accepting the descriptor.

### Valid Storage Types

Same set as Per-Tensor Affine:

- `int8`, `uint8`, `int16`, `uint16`, `int32`, `uint32`
- `int4`, `uint4`, `int2`, `uint2`

---

## Per-Block Affine (`0x03`)

The tensor is divided into fixed-size, contiguous blocks along a specified axis.
Each block carries its own `scale` (and optionally `zero_point`). This scheme
covers the GGUF family of linear block-quantized formats (e.g., `Q8_0`, `Q4_0`,
`Q4_1`) at the descriptor level; specific GGUF-style layouts are representable
by choosing the appropriate storage type, block size, and flags.

### Binary Encoding

Total descriptor length: **24 bytes**.

| Offset | Field | Type | Description |
|--------|-------|------|-------------|
| 0 | `scheme_tag` | `uint8` | MUST be `0x03`. |
| 1 | `scheme_version` | `uint8` | MUST be `0x01`. |
| 2 | `flags` | `uint16` | Scheme-specific flags (see below). Reserved bits MUST be `0`. |
| 4 | `axis` | `uint32` | Index of the axis along which the tensor is divided into blocks. MUST be strictly less than `rank`. |
| 8 | `block_size` | `uint32` | Number of **logical elements** per block along `axis`. MUST be a power of two and MUST be greater than or equal to `2`. |
| 12 | `scale_buffer_index` | `uint32` | Index in the buffer table of the buffer holding the `scale` array. |
| 16 | `zero_point_buffer_index` | `uint32` | Index in the buffer table of the buffer holding the `zero_point` array. Ignored when the `SYMMETRIC` flag is set; writers SHOULD set it to `0xFFFFFFFF` in that case. |
| 20 | `scale_type_tag` | `uint8` | Storage type of the scale values. MUST be `0x01` (`float16`), `0x02` (`bfloat16`), or `0x03` (`float32`). |
| 21 | `_reserved` | `uint8[3]` | MUST be `0x00`. |

**Flags bits:**

| Bit | Name | Meaning |
|-----|------|---------|
| 0 | `SYMMETRIC` | If set, the `zero_point` array is implicitly all zeros; `zero_point_buffer_index` MUST be `0xFFFFFFFF`. |
| 1–15 | (reserved) | MUST be `0`. |

### Block Layout

Let `S = shape[axis]` (resolved; MUST NOT be the dynamic dimension sentinel) and
`K = block_size`. The number of blocks along `axis` is:

```
num_blocks_per_axis = ceil(S / K)
```

The total number of blocks across the whole tensor is:

```
num_blocks = num_blocks_per_axis * product(shape[j] for j != axis)
```

Block index `b` at a tensor position `[i_0, ..., i_{rank-1}]` is computed as
follows. Let `outer` be the linear index formed from all dimensions except
`axis` using row-major order over those dimensions. Then:

```
b = outer * num_blocks_per_axis + floor(i_axis / K)
```

> **Note (non-normative):** This mapping preserves the tensor's row-major
> traversal order along non-quantized dimensions, which matches GGUF's linear
> layout for 2-D weight matrices. Writers targeting column-major traversal
> SHOULD use a column-major tensor layout rather than altering the block
> mapping.

### Padding

If `S` is not a multiple of `K`, the final block along `axis` for each outer
position contains only `S mod K` valid elements. The storage buffer MUST still
allocate space for a full block of `K` elements; the unused trailing elements
within the final block MUST be set to `0x00` bytes by the writer. A reader MUST
ignore these padding elements when dequantizing.

The scale (and zero-point) arrays MUST contain one entry per block, including
the partial final block.

### Referenced Buffers

The `scale` buffer MUST contain exactly `num_blocks` consecutive values of
`scale_type_tag` in little-endian byte order, starting at byte offset `0`. Its
byte size MUST be exactly `num_blocks * sizeof(scale_type_tag)`.

The `zero_point` buffer — present only if the `SYMMETRIC` flag is not set —
MUST contain exactly `num_blocks` consecutive `int32` values in little-endian
byte order, starting at byte offset `0`. Its byte size MUST be exactly
`num_blocks * 4`.

A reader MUST reject a descriptor whose `scale_buffer_index` or (when
applicable) `zero_point_buffer_index` is greater than or equal to
`buffer_count`.

### Dequantization Formula

Let `b` be the block index for a storage element `q` at logical position
`[i_0, ..., i_{rank-1}]`, computed as above. Let `s = scale[b]` and, if the
`SYMMETRIC` flag is set, `z = 0`, else `z = zero_point[b]`.

```
x_real = s * (q - z)
```

The multiplication is performed in `float32` arithmetic. If `scale_type_tag` is
`float16` or `bfloat16`, `s` MUST be widened to `float32` losslessly before the
multiplication.

### Validity Constraints

- `axis` MUST satisfy `axis < rank`.
- `block_size` MUST be a power of two in the range `[2, shape[axis]]` (inclusive).
  A reader MUST reject a descriptor whose `block_size` exceeds `shape[axis]`.
- `shape[axis]` MUST NOT equal `0xFFFFFFFFFFFFFFFF`.
- Every `scale` value MUST be a finite, non-zero value.
- `scale_type_tag` MUST be one of `0x01`, `0x02`, `0x03`.

### Valid Storage Types

- `int8`, `uint8`, `int4`, `uint4`, `int2`, `uint2`

> **Note (non-normative):** Wider integer storage types (`int16` and above) are
> not permitted for per-block affine because block quantization is only
> beneficial at low bit widths. A writer wishing to apply per-block scaling to
> wider storage types should use per-channel affine (`0x02`) instead.

---

## NF4 (NormalFloat4) (`0x04`)

A non-linear 4-bit quantization scheme introduced by the QLoRA paper. Each
storage code in `[0, 15]` decodes to one of 16 fixed real-valued levels chosen
to be information-theoretically optimal for weights drawn from a standard
normal distribution. Each block carries a single `absmax` scale; there is no
per-element zero point.

### Binary Encoding

Total descriptor length: **16 bytes**.

| Offset | Field | Type | Description |
|--------|-------|------|-------------|
| 0 | `scheme_tag` | `uint8` | MUST be `0x04`. |
| 1 | `scheme_version` | `uint8` | MUST be `0x01`. |
| 2 | `flags` | `uint16` | MUST be `0x0000`. No flags are defined for this scheme. |
| 4 | `axis` | `uint32` | Index of the axis along which the tensor is divided into blocks. MUST be strictly less than `rank`. |
| 8 | `block_size` | `uint32` | Number of logical elements per block along `axis`. MUST be a power of two; RECOMMENDED values are `64` (bitsandbytes default) or `128`. |
| 12 | `scale_buffer_index` | `uint32` | Index in the buffer table of the buffer holding the per-block `absmax` scales. |

### Lookup Table

The 16 NF4 levels are fixed by this specification and MUST NOT be altered by
readers or writers. Indexed by the unsigned 4-bit storage code `q`:

| `q` | `nf4[q]` |
|-----|----------|
| 0 | `-1.0` |
| 1 | `-0.6961928009986877` |
| 2 | `-0.5250730514526367` |
| 3 | `-0.39491748809814453` |
| 4 | `-0.28444138169288635` |
| 5 | `-0.18477343022823334` |
| 6 | `-0.09105003625154495` |
| 7 | `0.0` |
| 8 | `0.07958029955625534` |
| 9 | `0.16093020141124725` |
| 10 | `0.24611230194568634` |
| 11 | `0.33791524171829224` |
| 12 | `0.44070982933044434` |
| 13 | `0.5626170039176941` |
| 14 | `0.7229568362236023` |
| 15 | `1.0` |

The table values are given here as `float32` decimal expansions of the exact
levels from the QLoRA reference implementation. Implementations MUST use these
`float32` values verbatim; deriving the table from first principles at runtime
is NOT RECOMMENDED and MUST match these values bit-for-bit if attempted.

### Referenced Buffer

The `scale` buffer MUST contain exactly `num_blocks` consecutive `float32`
`absmax` values in little-endian byte order, starting at byte offset `0` within
the referenced buffer. `num_blocks` is computed identically to the Per-Block
Affine scheme. Its byte size MUST be exactly `num_blocks * 4`.

### Dequantization Formula

Let `b` be the block index for a storage element `q` at logical position
`[i_0, ..., i_{rank-1}]` (computed as in Per-Block Affine). Let `s = scale[b]`.

```
x_real = s * nf4[q]
```

The multiplication is performed in `float32` arithmetic.

### Validity Constraints

- `axis` MUST satisfy `axis < rank`.
- `block_size` MUST be a power of two in the range `[8, shape[axis]]`.
- `shape[axis]` MUST NOT equal `0xFFFFFFFFFFFFFFFF`.
- Every `scale` value MUST be a finite, non-negative `float32`.
- `scale_buffer_index` MUST be a valid index into the buffer table and MUST NOT
  refer to the tensor data buffer.

### Valid Storage Types

- `uint4` (`0x49`) — REQUIRED.

A reader MUST reject an NF4 descriptor whose storage type is not `uint4`.

> **Note (non-normative):** NF4 storage codes are conceptually unsigned (they
> index into a signed-valued lookup table); `uint4` is the correct storage type.
> The packing order follows the standard `uint4` rule from `element-types.md`:
> element `2k` in the low nibble of byte `k`, element `2k+1` in the high nibble.

---

## MXFP (OCP Microscaling) (`0x05`)

A block quantization format standardized by the Open Compute Project
Microscaling specification (OCP MX v1.0). A block of `32` contiguous elements
shares a single `float8_e8m0` exponent-only scale. Each element within the
block is stored as one of several supported narrow numeric types. This is the
format used by NVIDIA Blackwell Tensor Cores for MXFP8/MXFP6/MXFP4 compute.

### Binary Encoding

Total descriptor length: **16 bytes**.

| Offset | Field | Type | Description |
|--------|-------|------|-------------|
| 0 | `scheme_tag` | `uint8` | MUST be `0x05`. |
| 1 | `scheme_version` | `uint8` | MUST be `0x01`. |
| 2 | `flags` | `uint16` | MUST be `0x0000`. No flags are defined for this scheme. |
| 4 | `axis` | `uint32` | Index of the axis along which the tensor is divided into microscaling blocks. MUST be strictly less than `rank`. |
| 8 | `block_size` | `uint32` | Number of logical elements per microscaling block along `axis`. MUST be exactly `32` in this version of the specification. |
| 12 | `scale_buffer_index` | `uint32` | Index in the buffer table of the buffer holding the per-block `float8_e8m0` scales. |

### Referenced Buffer

The `scale` buffer MUST contain exactly `num_blocks` consecutive
`float8_e8m0` values, one byte each, starting at byte offset `0` within the
referenced buffer. Its byte size MUST be exactly `num_blocks` bytes.
`num_blocks` is computed identically to the Per-Block Affine scheme, with
`block_size = 32`.

The bit pattern `0xFF` in any scale byte is reserved (see `element-types.md`)
and MUST NOT appear in the scale buffer. A reader encountering `0xFF` in the
scale buffer MUST treat the descriptor as invalid.

### Dequantization Formula

Let `b` be the block index for a storage element at logical position
`[i_0, ..., i_{rank-1}]`. Let `e = scale[b]` be the `float8_e8m0` byte. The
shared exponent scale is:

```
s = 2^(e - 127)
```

If the storage type is a float8 type (`float8_e4m3`, `float8_e5m2`):

```
x_real = s * float_value_of(q)
```

where `float_value_of(q)` is the real number represented by the storage element
`q` interpreted according to `element-types.md`.

If the storage type is an integer type (`int8`, `int4`):

```
x_real = s * int_value_of(q)
```

where `int_value_of(q)` is the signed integer value of `q`. This MXFP integer
form is equivalent to per-block symmetric affine quantization with a
power-of-two scale and no zero point.

All arithmetic is performed in `float32` (or higher) precision; the shared
exponent `s` is itself exactly representable in `float32` for every valid
`float8_e8m0` bit pattern except `0xFF` (which is prohibited above).

### Validity Constraints

- `axis` MUST satisfy `axis < rank`.
- `block_size` MUST be exactly `32`.
- `shape[axis]` MUST NOT equal `0xFFFFFFFFFFFFFFFF`.
- `shape[axis]` MUST be a positive multiple of `32`. Unlike Per-Block Affine and
  NF4, MXFP does NOT permit partial trailing blocks; the OCP MX specification
  requires exact block alignment. A reader MUST reject a descriptor whose
  `shape[axis]` is not a positive multiple of `32`.
- `scale_buffer_index` MUST be a valid index into the buffer table and MUST NOT
  refer to the tensor data buffer.

### Valid Storage Types

The storage type MUST be one of:

- `float8_e4m3` (`0x40`) — MXFP8
- `float8_e5m2` (`0x41`) — MXFP8
- `int8` (`0x10`) — MXINT8
- `int4` (`0x48`) — MXFP4 surrogate (integer-valued variant)

> **Note (non-normative):** The OCP MX specification also defines MXFP4
> (`float4_e2m1`) and MXFP6 (`float6_e2m3`, `float6_e3m2`). These element types
> are not yet in the Tier 2 type list (see `element-types.md` open questions
> OQ-4 and OQ-5). When those types are assigned tags, they will be permitted
> here via a minor version increment. Until then, MXFP4 is representable only
> via the `int4` storage surrogate.

---

## Buffer Table Placement Rules

Every scheme that references a buffer (per-channel, per-block, NF4, MXFP) adds
entries to the tensor descriptor's buffer table beyond the buffers used by the
layout itself. The following rules apply across all quantization schemes:

1. The tensor data buffer always occupies a buffer table index determined by
   the layout (typically `0` for dense layouts). Quantization-parameter buffers
   MUST occupy distinct indices.
2. A quantization-parameter buffer MUST NOT be shared with the tensor data
   buffer. A reader MUST reject a descriptor that violates this rule.
3. Two quantization-parameter buffers (e.g., scale and zero-point) MAY reside
   in the same buffer table entry if and only if the entry's `byte_size`
   accommodates both arrays laid out end-to-end, and both arrays start at
   byte offset `0`. Since each parameter descriptor field specifies its own
   `*_buffer_index`, this case is expressed by writing the same index into
   both fields; readers MUST then interpret the buffer as the concatenation
   `[scales | zero_points]`. Writers SHOULD prefer distinct buffers for
   clarity; sharing is permitted only as an optimization.
4. A quantization-parameter buffer's `device_tag` (see `metadata.md` buffer
   handle format) MUST match the tensor data buffer's `device_tag`. A reader
   MUST reject a descriptor that violates this rule.

> **Note (non-normative):** The device-colocation rule ensures that quantized
> tensor kernels can dereference both the data and the quantization parameters
> without triggering cross-device transfers. A writer that needs to materialize
> quantization parameters on a different device must emit a separate tensor.

---

## Extension Schemes (`0xF0` – `0xFE`)

Implementations MAY define private quantization schemes using scheme tags in
`0xF0` – `0xFE`. The binary encoding of an extension scheme descriptor is
unconstrained beyond the fixed 4-byte header; the writer and reader MUST agree
on the payload format out of band.

> **Note (non-normative):** Extension schemes are the mechanism for
> implementation-specific or experimental quantization formats (e.g., GPTQ,
> AWQ group quantization with per-group permutations, or hardware-vendor-
> specific packings). Schemes that prove broadly useful SHOULD be proposed for
> assignment in the Tier 1 or Tier 2 range through a specification revision.

---

## Version Compatibility

Adding a new scheme tag MUST be accompanied by a minor version increment of the
overall format (`version_minor` in the fixed header; see `metadata.md`). A
reader at an earlier minor version will encounter an unrecognised `scheme_tag`
and reject the descriptor per the rules above, which is the intended behaviour.

Adding a new field to an existing scheme (or repurposing a reserved byte) MUST
be accompanied by a `scheme_version` increment for that scheme. A reader MUST
compare `scheme_version` against the highest version it supports for the given
`scheme_tag` and reject the descriptor if the version is newer than supported.

Removing a scheme or changing the dequantization formula for an existing
`(scheme_tag, scheme_version)` pair is a backward-incompatible change and MUST
be accompanied by a major version increment.

---

## Worked Example

A rank-2 `int8`-stored weight tensor with shape `[768, 1024]`, per-channel
affine quantization along axis 0 (asymmetric), no statistics or shard sections.
The tensor descriptor's buffer table carries three buffers:

- Buffer 0 — tensor data, `768 * 1024 = 786432` bytes, `int8` storage.
- Buffer 1 — scale array, `768 * 4 = 3072` bytes, `float32`.
- Buffer 2 — zero-point array, `768 * 4 = 3072` bytes, `int32`.

Quantization descriptor bytes (16 total):

```
Offset  Value (hex)                   Field
------  ----------------------------  -----
0       02                            scheme_tag = 0x02 (per-channel affine)
1       01                            scheme_version = 1
2       00 00                         flags = 0x0000 (asymmetric)
4       00 00 00 00                   axis = 0
8       01 00 00 00                   scale_buffer_index = 1
12      02 00 00 00                   zero_point_buffer_index = 2
```

The `quantization_length` prefix in the tensor descriptor's Quantization
Section would be `0x00000010` (16).

Dequantization of element `q` at logical position `[c, k]`:

```
x_real = scale[c] * (q - zero_point[c])
```

---

## Interaction with Other Sections

- **`metadata.md`**: The quantization descriptor defined here is the payload of
  the Quantization Section, prefixed by a `uint32` `quantization_length`. The
  `HAS_QUANTIZATION` flag (bit 0) gates the section's presence.
- **`element-types.md`**: Defines the storage type tags permitted in quantized
  tensor descriptors. Each scheme above enumerates its permitted set.
- **`memory-layout.md`**: Defines the buffer table structure that quantization
  parameters reference by index. Note that dense-layout descriptors impose
  `buffer_count = 0x01`; quantization schemes that reference additional buffers
  extend the buffer table beyond that minimum.
- **`buffer-protocol.md`**: Defines buffer alignment, device-tag semantics, and
  ownership. Quantization parameter buffers follow the same rules as data
  buffers.

---

## Open Questions

> **[OQ-1]:** Should this file define a normative encoding for double
> quantization (quantizing the scale buffer itself, as in bitsandbytes QLoRA
> "nested" quantization)? The current design makes the scale buffer a flat
> `float32` array. Double quantization would require either a recursive
> descriptor or a separate scheme tag. Deferred until a concrete interop need
> arises.

> **[OQ-2]:** The MXFP scheme currently fixes `block_size = 32` per the OCP MX
> v1.0 specification. Future OCP revisions (or NVIDIA-specific variants) may
> define alternative block sizes. Should `block_size` be left parameterised at
> the scheme level now to avoid a new scheme tag later, or should the fixed
> constraint be enforced to ensure strict OCP conformance? Current text enforces
> `block_size = 32`.

> **[OQ-3]:** Per-channel and per-block schemes currently limit the scale type
> to `float32` (per-channel) or `float16/bfloat16/float32` (per-block). Should
> per-channel support lower-precision scale types as well? The storage saving
> is modest (a few KB per layer) and the precision cost is non-trivial for
> per-channel weight quantization. Deferred.

> **[OQ-4]:** Should the descriptor include an explicit `num_blocks` field for
> the block-based schemes, or is deriving it from `shape[axis]` and
> `block_size` sufficient? Current text derives it. Deriving is simpler but
> requires a shape resolution step on the reader side; an explicit field would
> be redundant but self-contained. Deferred.

> **[OQ-5]:** NF4 decoding requires the fixed 16-entry lookup table. Should
> this table be duplicated into a normative reference appendix (to insulate
> against accidental mutation of the main table in future edits), or is the
> inline specification sufficient? Current text keeps it inline.
