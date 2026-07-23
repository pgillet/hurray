# COO (Coordinate) Sparse Layout — Hurray Format Specification

**Layout tag:** `0x06` | **Tier:** 1

> This section uses RFC 2119 key words: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT,
> SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

## Description

The COO (Coordinate) format stores a sparse tensor as a list of explicitly enumerated
non-zero elements. Each non-zero is described by its logical index tuple and its value.
COO is the most general sparse format: it imposes no ordering requirement on the
non-zeros and supports arbitrary rank.

> **Note (non-normative):** COO is simple to construct (append-only) and easy to
> convert to other formats. It is less efficient for row-wise random access than CSR,
> but is well-suited for assembly (e.g., accumulating contributions from finite-element
> computations) and as an interchange format between sparse kernels.

## Buffer Table

A COO tensor descriptor MUST have `buffer_count = 2` in the buffer table.

| Buffer index | Name | Element type | Length | Description |
|---|---|---|---|---|
| 0 | `values` | tensor element type | `nnz` elements | Non-zero element values, in storage order. |
| 1 | `indices` | `uint64` | `nnz × rank` elements | Logical index tuples of the non-zeros, stored in row-major order: `indices[i * rank + d]` is the coordinate of the `i`-th non-zero along dimension `d`. |

## Additional Descriptor Fields

| Field | Type | Description |
|-------|------|-------------|
| `nnz` | `uint64` | Number of stored (non-zero) elements. MAY be 0 for an empty sparse tensor. |
| `is_sorted` | `uint8` | `0x01` if the non-zeros are sorted in lexicographic index order (dimension 0 major); `0x00` otherwise. |
| `_reserved` | `uint8[7]` | MUST be `0x00`. |

## Storage Order

Non-zeros are stored at positions `0` through `nnz - 1` in `values` and `indices`.
The `i`-th non-zero has:
- Value: `values[i]`
- Logical index: `[indices[i * rank + 0], indices[i * rank + 1], ..., indices[i * rank + (rank-1)]]`

If `is_sorted = 0x01`, the non-zeros MUST appear in **lexicographic index order**
(i.e., sorted by dimension 0 first, then dimension 1, etc.). A conforming reader MAY
use binary search for element lookup when `is_sorted = 0x01`.

If `is_sorted = 0x00`, no ordering guarantee is made. Readers MUST perform a linear
scan to locate a specific element.

## Validity Constraints

This layout MUST NOT be used for rank-0 (scalar) tensors. See
`data-model.md` § Scalar Tensors.

A conforming writer MUST ensure:

1. Every stored index tuple is within bounds: `0 <= indices[i * rank + d] < shape[d]`
   for all `i` in `[0, nnz)` and all `d` in `[0, rank)`.
2. No two stored entries share the same index tuple (no duplicate coordinates).
3. If `is_sorted = 0x01`, entries are in strictly increasing lexicographic order
   (no ties, since duplicate coordinates are forbidden).

A conforming reader SHOULD validate constraints (1) and (2) and MUST reject
descriptors that violate them, unless operating in permissive mode.

## Buffer Size

- `values` buffer: `nnz * element_byte_width` bytes (or `ceil(nnz / packing_factor)`
  for sub-byte types).
- `indices` buffer: `nnz * rank * 8` bytes (`uint64` elements).

Both buffers MUST satisfy the alignment requirements in `buffer-protocol.md`.

> **Note (non-normative):** The `byte_offset` field in the common descriptor header
> is not meaningful for sparse layouts — there is no single "first element" at a
> fixed offset. For COO tensors, `byte_offset` MUST be set to `0x0000000000000000`.

## Element Lookup

To retrieve the value at logical index `idx[0..rank-1]`:

1. If `is_sorted = 0x01`, perform a binary search on the `indices` buffer (comparing
   full `rank`-tuples in lexicographic order) to find a matching entry.
2. If `is_sorted = 0x00`, perform a linear scan over all `nnz` entries.
3. If a match is found at position `i`, return `values[i]`.
4. If no match is found, the element is implicitly **zero** (or the zero value of the
   element type).

## Interaction with Statistics Section

When the `HAS_STATISTICS` flag is set, the `nnz` field in the statistics section
MUST match the `nnz` field in the COO descriptor, which is authoritative. The `sparsity_ratio` SHOULD be
computed as `1.0 - (nnz / total_elements)` where `total_elements` is the product of
all `shape` values.

## Example

Rank-2 sparse tensor with shape `[4, 4]`, element type `float32`, 3 non-zeros:

```
Non-zeros: (0, 1) = 1.5,  (2, 0) = -0.5,  (3, 3) = 2.0
nnz = 3, is_sorted = 0x01

values  (buffer 0, 12 bytes):  [1.5, -0.5, 2.0]  as float32 LE

indices (buffer 1, 3×2×8 = 48 bytes, uint64 LE):
  entry 0: [0, 1]
  entry 1: [2, 0]
  entry 2: [3, 3]
```
