# NF4 (NormalFloat4) Quantization — Hurray Format Specification

**Scheme tag:** `0x04` | **Tier:** 2

> This section uses RFC 2119 key words: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT,
> SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

## Description

A non-linear 4-bit quantization scheme introduced by the QLoRA paper. Each
storage code in `[0, 15]` decodes to one of 16 fixed real-valued levels chosen
to be information-theoretically optimal for weights drawn from a standard
normal distribution. Each block carries a single `absmax` scale; there is no
per-element zero point.

The block index computation follows the same rule as Per-Block Affine
(see `quantization/per-block-affine.md` § Block Layout).

## Binary Encoding

Total descriptor length: **16 bytes**.

| Offset | Field | Type | Description |
|--------|-------|------|-------------|
| 0 | `scheme_tag` | `uint8` | MUST be `0x04`. |
| 1 | `scheme_version` | `uint8` | MUST be `0x01`. |
| 2 | `flags` | `uint16` | MUST be `0x0000`. No flags are defined for this scheme. |
| 4 | `axis` | `uint32` | Index of the axis along which the tensor is divided into blocks. MUST be strictly less than `rank`. |
| 8 | `block_size` | `uint32` | Number of logical elements per block along `axis`. MUST be a power of two; RECOMMENDED values are `64` (bitsandbytes default) or `128`. |
| 12 | `scale_buffer_index` | `uint32` | Index in the buffer table of the buffer holding the per-block `absmax` scales. |

All multi-byte fields MUST be encoded in little-endian byte order.

## Lookup Table

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

## Referenced Buffer

The `scale` buffer MUST contain exactly `num_blocks` consecutive `float32`
`absmax` values in little-endian byte order, starting at byte offset `0` within
the referenced buffer. `num_blocks` is computed identically to the Per-Block
Affine scheme. Its byte size MUST be exactly `num_blocks * 4`.

## Dequantization Formula

Let `b` be the block index for a storage element `q` at logical position
`[i_0, ..., i_{rank-1}]` (computed as in Per-Block Affine). Let `s = scale[b]`.

```
x_real = s * nf4[q]
```

The multiplication is performed in `float32` arithmetic.

## Validity Constraints

- `axis` MUST satisfy `axis < rank`.
- `block_size` MUST be a power of two in the range `[8, shape[axis]]`.
- `shape[axis]` MUST NOT equal `0xFFFFFFFFFFFFFFFF`.
- Every `scale` value MUST be a finite, non-negative `float32`.
- `scale_buffer_index` MUST be a valid index into the buffer table and MUST NOT
  refer to the tensor data buffer.

## Valid Storage Types

The storage type (`type_tag` in the tensor descriptor) MUST be `uint4` (`0x49`).

A reader MUST reject an NF4 descriptor whose storage type is not `uint4`.

> **Note (non-normative):** NF4 storage codes are conceptually unsigned (they
> index into a signed-valued lookup table); `uint4` is the correct storage type.
> The packing order follows the standard `uint4` rule from `element-types.md`:
> element `2k` in the low nibble of byte `k`, element `2k+1` in the high nibble.
