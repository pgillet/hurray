# Per-Channel Affine Quantization — Hurray Format Specification

**Scheme tag:** `0x02` | **Tier:** 1

> This section uses RFC 2119 key words: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT,
> SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

## Description

One `scale` and `zero_point` pair per slice along a specified axis. The scale
and zero-point arrays are stored in **separate buffers** listed in the tensor
descriptor's buffer table.

> **Note (non-normative):** Because dense-layout descriptors require
> `buffer_count = 0x01`, a per-channel-affine quantized dense tensor requires
> `buffer_count` to be at least 2 (symmetric case) or 3 (asymmetric case). This
> is the mechanism by which quantization schemes extend the buffer table beyond
> the dense-layout minimum.

## Binary Encoding

Total descriptor length: **16 bytes**.

| Offset | Field | Type | Description |
|--------|-------|------|-------------|
| 0 | `scheme_tag` | `uint8` | MUST be `0x02`. |
| 1 | `scheme_version` | `uint8` | MUST be `0x01`. |
| 2 | `flags` | `uint16` | Scheme-specific flags (see below). Reserved bits MUST be `0`. |
| 4 | `axis` | `uint32` | Index of the quantized axis. MUST be strictly less than `rank`. |
| 8 | `scale_buffer_index` | `uint32` | Index in the buffer table of the buffer holding the `scale` array. |
| 12 | `zero_point_buffer_index` | `uint32` | Index in the buffer table of the buffer holding the `zero_point` array. |

All multi-byte fields MUST be encoded in little-endian byte order.

**Flags bits:**

| Bit | Name | Meaning |
|-----|------|---------|
| 0 | `SYMMETRIC` | If set, the `zero_point` array is implicitly all zeros; `zero_point_buffer_index` MUST be `0xFFFFFFFF`. |
| 1–15 | (reserved) | MUST be `0`. |

## Referenced Buffers

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

## Dequantization Formula

For a storage element `q` at logical index `[i_0, i_1, ..., i_{rank-1}]`:

```
c = i_axis
x_real = scale[c] * (q - zero_point[c])
```

If the `SYMMETRIC` flag is set, `zero_point[c]` is treated as `0` for all `c`.

## Validity Constraints

- `axis` MUST satisfy `axis < rank`.
- `shape[axis]` MUST NOT equal `0xFFFFFFFFFFFFFFFF` (the dynamic dimension
  sentinel): per-channel quantization requires a statically known channel count.
- Every element of the `scale` array MUST be a finite, non-zero `float32` value.
- Every element of the `zero_point` array (when present) MUST lie within the
  representable range of the storage type.

A reader MAY defer the per-element validity check on the scale and zero-point
arrays to the first dequantization attempt, but MUST perform the axis and shape
checks before accepting the descriptor.

## Valid Storage Types

The storage type (`type_tag` in the tensor descriptor) MUST be one of:

- `int8` (`0x10`), `uint8` (`0x11`)
- `int16` (`0x12`), `uint16` (`0x13`)
- `int32` (`0x14`), `uint32` (`0x15`)
- `int4` (`0x48`), `uint4` (`0x49`)
- `int2` (`0x4A`), `uint2` (`0x4B`)

A reader MUST reject a descriptor whose storage type is not in this list.

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
