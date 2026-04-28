# Per-Block Affine Quantization — Hurray Format Specification

**Scheme tag:** `0x03` | **Tier:** 1

> This section uses RFC 2119 key words: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT,
> SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

## Description

The tensor is divided into fixed-size, contiguous blocks along a specified axis.
Each block carries its own `scale` (and optionally `zero_point`). This scheme
covers the GGUF family of linear block-quantized formats (e.g., `Q8_0`, `Q4_0`,
`Q4_1`) at the descriptor level; specific GGUF-style layouts are representable
by choosing the appropriate storage type, block size, and flags.

## Binary Encoding

Total descriptor length: **24 bytes**.

| Offset | Field | Type | Description |
|--------|-------|------|-------------|
| 0 | `scheme_tag` | `uint8` | MUST be `0x03`. |
| 1 | `scheme_version` | `uint8` | MUST be `0x01`. For the version compatibility policy, see `quantization.md` § Version Compatibility. |
| 2 | `flags` | `uint16` | Scheme-specific flags (see below). Reserved bits MUST be `0`. |
| 4 | `axis` | `uint32` | Index of the axis along which the tensor is divided into blocks. MUST be strictly less than `rank`. |
| 8 | `block_size` | `uint32` | Number of **logical elements** per block along `axis`. MUST be a power of two and MUST be greater than or equal to `2`. |
| 12 | `scale_buffer_index` | `uint32` | Index in the buffer table of the buffer holding the `scale` array. |
| 16 | `zero_point_buffer_index` | `uint32` | Index in the buffer table of the buffer holding the `zero_point` array. Ignored when the `SYMMETRIC` flag is set; writers SHOULD set it to `0xFFFFFFFF` in that case. |
| 20 | `scale_type_tag` | `uint8` | Storage type of the scale values. MUST be `0x01` (`float16`), `0x02` (`bfloat16`), or `0x03` (`float32`). |
| 21 | `_reserved` | `uint8[3]` | MUST be `0x00`. |

All multi-byte fields MUST be encoded in little-endian byte order.

**Flags bits:**

| Bit | Name | Meaning |
|-----|------|---------|
| 0 | `SYMMETRIC` | If set, the `zero_point` array is implicitly all zeros; `zero_point_buffer_index` MUST be `0xFFFFFFFF`. |
| 1–15 | (reserved) | MUST be `0`. |

## Block Layout

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

## Padding

If `S` is not a multiple of `K`, the final block along `axis` for each outer
position contains only `S mod K` valid elements. The storage buffer MUST still
allocate space for a full block of `K` elements; the unused trailing elements
within the final block MUST be set to `0x00` bytes by the writer. A reader MUST
ignore these padding elements when dequantizing.

The scale (and zero-point) arrays MUST contain one entry per block, including
the partial final block.

## Referenced Buffers

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

## Dequantization Formula

Let `b` be the block index for a storage element `q` at logical position
`[i_0, ..., i_{rank-1}]`, computed as above. Let `s = scale[b]` and, if the
`SYMMETRIC` flag is set, `z = 0`, else `z = zero_point[b]`.

```
x_real = s * (q - z)
```

The multiplication is performed in `float32` arithmetic. If `scale_type_tag` is
`float16` or `bfloat16`, `s` MUST be widened to `float32` losslessly before the
multiplication.

## Validity Constraints

- `axis` MUST satisfy `axis < rank`.
- `block_size` MUST be a power of two and MUST be greater than or equal to `2`.
- When `shape[axis]` is greater than `0`, `block_size` MUST be less than or equal to `shape[axis]`. A reader MUST reject a descriptor whose `block_size` exceeds a non-zero `shape[axis]`.
- When `shape[axis]` equals `0` (an empty quantization axis, per ADR-007), the upper-bound check is waived. The `block_size` field MUST still be a power of two greater than or equal to `2`, but its value has no effect on buffer sizing: `num_blocks` evaluates to `0`, and the scale and zero-point buffers MUST have byte size `0` (their pointers MAY be null per `buffer-protocol.md`).
- `shape[axis]` MUST NOT equal `0xFFFFFFFFFFFFFFFF`.
- Every `scale` value MUST be a finite, non-zero value.
- `scale_type_tag` MUST be one of `0x01`, `0x02`, `0x03`.

> **Note (non-normative):** Permitting `block_size` to exceed `shape[axis]` when the axis is empty preserves the producer's declared quantization granularity across shape changes (for example, a filter that selects zero rows from an otherwise per-block-quantized weight tensor). The descriptor remains structurally valid and round-trippable; no blocks are materialized.

> **Note (non-normative):** The case `block_size = shape[axis]` — a single
> block covering the entire quantized axis — is intentionally permitted. It
> is a degenerate but valid configuration that produces semantics equivalent
> to per-tensor affine quantization along that axis (one shared
> `scale`/`zero_point` pair per outer position). Readers MUST handle it
> identically to any other `block_size` value; there is no separate code
> path.

## Valid Storage Types

The storage type (`type_tag` in the tensor descriptor) MUST be one of:

- `int8` (`0x10`), `uint8` (`0x11`)
- `int4` (`0x48`), `uint4` (`0x49`)
- `int2` (`0x4A`), `uint2` (`0x4B`)

A reader MUST reject a descriptor whose storage type is not in this list.

> **Note (non-normative):** Wider integer storage types (`int16` and above) are
> not permitted for per-block affine because block quantization is only
> beneficial at low bit widths. A writer wishing to apply per-block scaling to
> wider storage types should use per-channel affine (`0x02`) instead.
