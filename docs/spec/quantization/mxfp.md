# MXFP (OCP Microscaling) Quantization — Hurray Format Specification

**Scheme tag:** `0x05` | **Tier:** 2

> This section uses RFC 2119 key words: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT,
> SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

## Description

A block quantization format standardized by the Open Compute Project
Microscaling specification (OCP MX v1.0). A contiguous block of elements along
a chosen axis shares a single `float8_e8m0` exponent-only scale. Each element
within the block is stored as one of several supported narrow numeric types.
This is the format used by NVIDIA Blackwell Tensor Cores for MXFP8/MXFP6/MXFP4
compute.

The block size is a descriptor field (`block_size`). The OCP MX v1.0 canonical
value is `32`; other power-of-two values are permitted by this specification to
accommodate future OCP revisions and hardware variants.

The block index computation follows the same rule as Per-Block Affine
(see `quantization/per-block-affine.md` § Block Layout), with the
descriptor-specified `block_size`.

## Binary Encoding

Total descriptor length: **16 bytes**.

| Offset | Field | Type | Description |
|--------|-------|------|-------------|
| 0 | `scheme_tag` | `uint8` | MUST be `0x05`. |
| 1 | `scheme_version` | `uint8` | MUST be `0x01`. For the version compatibility policy, see `quantization.md` § Version Compatibility. |
| 2 | `flags` | `uint16` | MUST be `0x0000`. No flags are defined for this scheme. |
| 4 | `axis` | `uint32` | Index of the axis along which the tensor is divided into microscaling blocks. MUST be strictly less than `rank`. |
| 8 | `block_size` | `uint32` | Number of logical elements per microscaling block along `axis`. MUST be a power of two in the range `[16, 2048]`. The OCP MX v1.0 canonical value is `32`. |
| 12 | `scale_buffer_index` | `uint32` | Index in the buffer table of the buffer holding the per-block `float8_e8m0` scales. |

All multi-byte fields MUST be encoded in little-endian byte order.

## Referenced Buffer

The `scale` buffer MUST contain exactly `num_blocks` consecutive
`float8_e8m0` values, one byte each, starting at byte offset `0` within the
referenced buffer. Its byte size MUST be exactly `num_blocks` bytes.
`num_blocks` is computed identically to the Per-Block Affine scheme using the
descriptor-specified `block_size`, with the additional MXFP constraint that
`shape[axis]` is a positive multiple of `block_size` (no partial trailing
block — see Validity Constraints below). The full count across the whole
tensor is:

```
num_blocks = (shape[axis] / block_size) * product(shape[j] for j ≠ axis)
```

(exact division — `shape[axis]` MUST be a positive multiple of `block_size`;
see Validity Constraints). See `quantization/per-block-affine.md` § Block
Layout for the derivation; MXFP differs only in disallowing partial trailing
blocks.

The bit patterns `0x00` and `0xFF` in any scale byte are reserved (NaN per OCP MX v1.0 § 5.6; see `element-types.md` § float8_e8m0) and MUST NOT appear in the scale buffer. A reader encountering `0x00` or `0xFF` in the scale buffer MUST treat the descriptor as invalid.

## Dequantization Formula

Let `b` be the block index for a storage element at logical position
`[i_0, ..., i_{rank-1}]`. Let `e = scale[b]` be the `float8_e8m0` byte. The
shared exponent scale is:

```
s = 2^(e - 127)
```

If the storage type is a float type (`float8_e4m3`, `float8_e5m2`, `float4_e2m1`, `float6_e2m3`, `float6_e3m2`):

```
x_real = s * float_value_of(q)
```

where `float_value_of(q)` is the real number represented by the storage element
`q` interpreted according to `element-types.md` and the OCP MX v1.0 value maps
for the respective type.

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
- `block_size` MUST be a power of two in the range `[16, 2048]`. A reader MUST reject a descriptor whose `block_size` is not a power of two, is less than `16`, or exceeds `2048`. Values below 16 have no hardware Tensor Core support and are not valid under any OCP MX revision.
- `shape[axis]` MUST NOT equal `0xFFFFFFFFFFFFFFFF`.
- `shape[axis]` MUST be a positive multiple of `block_size`. Unlike Per-Block Affine and NF4, MXFP does NOT permit partial trailing blocks; all blocks MUST be full. A reader MUST reject a descriptor whose `shape[axis]` is not a positive multiple of `block_size`.
- `scale_buffer_index` MUST be a valid index into the buffer table and MUST NOT
  refer to the tensor data buffer.

## Valid Storage Types

The storage type (`type_tag` in the tensor descriptor) MUST be one of:

- `float8_e4m3` (`0x40`) — MXFP8
- `float8_e5m2` (`0x41`) — MXFP8
- `float4_e2m1` (`0x43`) — MXFP4
- `float6_e2m3` (`0x44`) — MXFP6
- `float6_e3m2` (`0x45`) — MXFP6
- `int8` (`0x10`) — MXINT8
- `int4` (`0x48`) — MXINT4 / MXFP4 integer-valued surrogate

A reader MUST reject an MXFP descriptor whose storage type is not in this list.
