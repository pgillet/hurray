# MXFP (OCP Microscaling) Quantization — Hurray Format Specification

**Scheme tag:** `0x05` | **Tier:** 2

> This section uses RFC 2119 key words: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT,
> SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

## Description

A block quantization format standardized by the Open Compute Project
Microscaling specification (OCP MX v1.0). A block of `32` contiguous elements
shares a single `float8_e8m0` exponent-only scale. Each element within the
block is stored as one of several supported narrow numeric types. This is the
format used by NVIDIA Blackwell Tensor Cores for MXFP8/MXFP6/MXFP4 compute.

The block index computation follows the same rule as Per-Block Affine
(see `quantization/per-block-affine.md` § Block Layout), with `block_size`
fixed at `32`.

## Binary Encoding

Total descriptor length: **16 bytes**.

| Offset | Field | Type | Description |
|--------|-------|------|-------------|
| 0 | `scheme_tag` | `uint8` | MUST be `0x05`. |
| 1 | `scheme_version` | `uint8` | MUST be `0x01`. |
| 2 | `flags` | `uint16` | MUST be `0x0000`. No flags are defined for this scheme. |
| 4 | `axis` | `uint32` | Index of the axis along which the tensor is divided into microscaling blocks. MUST be strictly less than `rank`. |
| 8 | `block_size` | `uint32` | Number of logical elements per microscaling block along `axis`. MUST be exactly `32` in this version of the specification. |
| 12 | `scale_buffer_index` | `uint32` | Index in the buffer table of the buffer holding the per-block `float8_e8m0` scales. |

All multi-byte fields MUST be encoded in little-endian byte order.

## Referenced Buffer

The `scale` buffer MUST contain exactly `num_blocks` consecutive
`float8_e8m0` values, one byte each, starting at byte offset `0` within the
referenced buffer. Its byte size MUST be exactly `num_blocks` bytes.
`num_blocks` is computed identically to the Per-Block Affine scheme, with
`block_size = 32`.

The bit pattern `0xFF` in any scale byte is reserved (see `element-types.md`)
and MUST NOT appear in the scale buffer. A reader encountering `0xFF` in the
scale buffer MUST treat the descriptor as invalid.

## Dequantization Formula

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

## Validity Constraints

- `axis` MUST satisfy `axis < rank`.
- `block_size` MUST be exactly `32`.
- `shape[axis]` MUST NOT equal `0xFFFFFFFFFFFFFFFF`.
- `shape[axis]` MUST be a positive multiple of `32`. Unlike Per-Block Affine and
  NF4, MXFP does NOT permit partial trailing blocks; the OCP MX specification
  requires exact block alignment. A reader MUST reject a descriptor whose
  `shape[axis]` is not a positive multiple of `32`.
- `scale_buffer_index` MUST be a valid index into the buffer table and MUST NOT
  refer to the tensor data buffer.

## Valid Storage Types

The storage type (`type_tag` in the tensor descriptor) MUST be one of:

- `float8_e4m3` (`0x40`) — MXFP8
- `float8_e5m2` (`0x41`) — MXFP8
- `int8` (`0x10`) — MXINT8
- `int4` (`0x48`) — MXFP4 surrogate (integer-valued variant)

A reader MUST reject an MXFP descriptor whose storage type is not in this list.

> **Note (non-normative):** The OCP MX specification also defines MXFP4
> (`float4_e2m1`) and MXFP6 (`float6_e2m3`, `float6_e3m2`). These element types
> are not yet in the Tier 2 type list (see `element-types.md` open questions
> OQ-4 and OQ-5). When those types are assigned tags, they will be permitted
> here via a minor version increment. Until then, MXFP4 is representable only
> via the `int4` storage surrogate.
