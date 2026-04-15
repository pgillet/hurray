# ADR-005: Morton Layout Uses Zero-Padding for Non-Power-of-Two Dimensions

## Status

Accepted

## Context

The Morton (Z-order curve) layout uses `morton_bits[k]` to define the number of bits
allocated to each dimension in the interleaved Morton code. The buffer must hold
`2^(sum(morton_bits))` elements. When a dimension size is not a power of two,
`morton_bits[k]` must be set to `ceil(log2(shape[k]))`, padding the dimension to the
next power of two. Elements with Morton codes corresponding to indices outside the
tensor's shape are padding with undefined values.

OQ-4 asked whether this zero-padding approach should be mandated, or whether a
"compact Morton" scheme should be defined to eliminate the padding waste.

## Decision

The zero-padding approach is mandated. Writers SHOULD set `morton_bits[k]` to the
minimum value satisfying `shape[k] <= 2^morton_bits[k]`. The buffer holds
`2^(sum(morton_bits))` elements; padding elements are undefined and readers MUST NOT
access them as tensor data.

## Alternatives Considered

- **Compact Morton addressing**: a bijective mapping from Morton codes to valid
  elements, skipping codes that fall outside the tensor's shape. Rejected because:
  1. It requires non-trivial per-access index computation (lookup tables or specialised
     bit manipulation), breaking the branchless bit-interleaving that makes Morton fast.
  2. It significantly complicates the implementation and the spec (the mapping is not
     self-evident and requires a normative algorithm).
  3. Morton layouts are inherently power-of-two structures; non-power-of-two use is
     already an unusual choice. Writers for whom padding waste is unacceptable should
     use a tiled or row-major layout instead.

## Consequences

- The padding factor per dimension is strictly less than 2× in the worst case
  (when `shape[k] = 2^(b-1) + 1`). For typical ML dimensions (224, 256, 512, etc.)
  the waste is small or zero.
- Morton index computation remains a simple, branchless bit-interleaving operation
  with no special cases.
- The `morton_bits[k]` field gives writers explicit control over the trade-off between
  waste and address space: a writer MAY choose a larger `morton_bits[k]` than the
  minimum (e.g. for alignment reasons), at the cost of more padding.
