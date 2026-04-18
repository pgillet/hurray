# Morton (Z-Order Curve) Layout — Hurray Format Specification

**Layout tag:** `0x05` | **Tier:** 1

> This section uses RFC 2119 key words: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT,
> SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

## Description

The Morton layout stores elements by interleaving the bits of their dimension indices,
producing a linear order with good spatial locality for multi-dimensional access
patterns.

## Additional Descriptor Fields

| Field | Type | Description |
|-------|------|-------------|
| `morton_bits` | `uint32[rank]` | Number of bits used per dimension in the Morton encoding. Each value MUST be greater than 0. |

## Dimension Size Constraints

For each dimension `k`, `shape[k]` MUST satisfy `shape[k] <= 2^morton_bits[k]`.

## Morton Index Computation

For element `[i_0, i_1, ..., i_{r-1}]`, the Morton code is computed by interleaving
bits in round-robin order, starting from the least significant bit of dimension 0:

```
morton_code = 0
for bit_position b = 0, 1, 2, ...:
    for dimension d = 0, 1, ..., rank - 1:
        if b < morton_bits[d]:
            morton_code |= ((i_d >> b) & 1) << (b * rank + d)
```

The element at Morton code `m` is stored at linear offset `m` in the buffer.

## Buffer Size

The buffer MUST hold `2^(sum(morton_bits[k] for all k))` elements.

> **Note (non-normative):** For non-power-of-two dimension sizes, `morton_bits[k]`
> SHOULD be set to `ceil(log2(shape[k]))` to minimise padding. The worst-case padding
> factor per dimension is strictly less than 2×. Writers for whom padding waste is
> unacceptable SHOULD prefer a tiled or row-major layout.

## Example

Rank-2 tensor with shape `[4, 4]`, `morton_bits = [2, 2]`.

Element `[2, 3]`: `i_0 = 2` = `0b10`, `i_1 = 3` = `0b11`.
Interleaved (LSB first, dim 0 then dim 1): bit 0 of `i_0`=0, bit 0 of `i_1`=1,
bit 1 of `i_0`=1, bit 1 of `i_1`=1. Morton code = `0b1110 = 14`.
Element `[2, 3]` is stored at linear offset 14.
