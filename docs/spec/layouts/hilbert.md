# Hilbert Curve Layout — Hurray Format Specification

**Layout tag:** `0x40` | **Tier:** 2

> This section uses RFC 2119 key words: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT,
> SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

## Description

The Hilbert curve layout stores elements according to a Hilbert space-filling curve,
which provides better locality than the Morton curve: consecutive Hilbert indices
always correspond to physically adjacent elements (L∞ distance = 1 in index space),
with no large jumps. This benefits access patterns that traverse multi-dimensional
regions with spatial coherence (e.g., image convolution, volumetric sampling,
point-cloud processing), at the cost of a more expensive index computation than Morton.

> **Note (non-normative):** The Hilbert curve is particularly effective for 2D and 3D
> spatial tensors where Morton's large jumps at quadrant boundaries cause cache misses.
> For 1D access patterns or cases where index-computation cost dominates, row-major or
> Morton layouts are preferable.

## Additional Descriptor Fields

| Field | Type | Description |
|-------|------|-------------|
| `hilbert_order` | `uint32` | Order of the Hilbert curve. MUST be greater than 0. Each tensor dimension MUST equal `2^hilbert_order`. |
| `hilbert_rank` | `uint32` | Number of curve dimensions. MUST equal the tensor's `rank`. MUST be greater than or equal to 2. |

## Validity Constraints

A conforming reader MUST reject a Hilbert-curve descriptor that violates any
of the following constraints:

1. `shape[k] = 2^hilbert_order` for every `k` in `[0, hilbert_rank)`. All tensor
   dimensions MUST be a power of two equal to `2^hilbert_order`.
2. `hilbert_rank` MUST equal the tensor's `rank`.
3. `hilbert_rank` MUST be greater than or equal to `2`.
4. `hilbert_order` MUST be greater than `0`.

This layout MUST NOT be used for rank-0 (scalar) tensors (constraints 2 and 3
together exclude rank 0). See `data-model.md` § Scalar Tensors.

## Buffer Size

The buffer MUST hold exactly `2^(hilbert_rank * hilbert_order)` elements.

## Normative Index Mapping

The normative index mapping is the algorithm defined by Skilling (2004) (see
`references.md`). Conforming implementations MUST use this algorithm. All arithmetic
is integer arithmetic; `^` denotes bitwise XOR; array indexing is zero-based.

Let `r = hilbert_rank`, `p = hilbert_order`. Coordinates `X[0..r-1]` are each in
`[0, 2^p)`. Hilbert index `h` is in `[0, 2^(r*p))`.

**Bit packing:** bit `(b * r + (r - 1 - d))` of `h` holds bit `b` of `X[d]`,
for `b = 0, ..., p-1` and `d = 0, ..., r-1`.

**`CoordsToHilbert(X[0..r-1], r, p)` → `h`:**

```
M = 1 << (p - 1)
Q = M
while Q > 1:
    P = Q - 1
    for i = 0 to r-1:
        if X[i] & Q:
            X[0] ^= P
        else:
            t = (X[0] ^ X[i]) & P
            X[0] ^= t; X[i] ^= t
    Q >>= 1
for i = 1 to r-1:
    X[i] ^= X[i-1]
t = 0; Q = M
while Q > 1:
    if X[r-1] & Q: t ^= Q - 1
    Q >>= 1
for i = 0 to r-1:
    X[i] ^= t
h = 0
for b = 0 to p-1:
    for d = 0 to r-1:
        h |= ((X[d] >> b) & 1) << (b * r + (r - 1 - d))
return h
```

**`HilbertToCoords(h, r, p)` → `X[0..r-1]`:**

```
X = [0] * r
for b = 0 to p-1:
    for d = 0 to r-1:
        X[d] |= ((h >> (b * r + (r - 1 - d))) & 1) << b
t = X[r-1] >> 1
for i = r-1 downto 1:
    X[i] ^= X[i-1]
X[0] ^= t
Q = 2
while Q != (1 << p):
    P = Q - 1
    for i = r-1 downto 0:
        if X[i] & Q:
            X[0] ^= P
        else:
            t = (X[0] ^ X[i]) & P
            X[0] ^= t; X[i] ^= t
    Q <<= 1
return X
```

## Conformance Check

Selected index mappings for `r = 2`, `p = 2` (shape `[4, 4]`):

| `h` | `X[0]` | `X[1]` | | `h` | `X[0]` | `X[1]` |
|-----|--------|--------|-|-----|--------|--------|
| 0 | 0 | 0 | | 8 | 2 | 2 |
| 1 | 1 | 0 | | 9 | 2 | 3 |
| 2 | 1 | 1 | | 10 | 3 | 3 |
| 3 | 0 | 1 | | 11 | 3 | 2 |
| 4 | 0 | 2 | | 12 | 3 | 1 |
| 5 | 0 | 3 | | 13 | 2 | 1 |
| 6 | 1 | 3 | | 14 | 2 | 0 |
| 7 | 1 | 2 | | 15 | 3 | 0 |

Consecutive entries differ by exactly 1 in exactly one coordinate. Implementations
SHOULD validate against this table as a conformance check.

> **Note (non-normative):** The MUST-level normative reference for the index mapping
> is the `CoordsToHilbert` / `HilbertToCoords` algorithm defined above. This table is
> provided as a SHOULD-level conformance aid; if a discrepancy is ever observed
> between table and algorithm, the algorithm output prevails.
