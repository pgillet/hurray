# General Subpaving Layout — Hurray Format Specification

**Layout tag:** `0x06` | **Tier:** 1

> This section uses RFC 2119 key words: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT,
> SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

## Description

The general subpaving layout describes an **irregular subpaving**: the tensor's index
space is partitioned into a set of non-overlapping rectangular regions (boxes), each
with its own layout descriptor. It is the most general layout in Hurray, intended for
tensors with heterogeneous structure — mixed dense/sparse regions, or regions with
different tiling strategies.

> **Note (non-normative):** The general subpaving layout carries more descriptor
> overhead than simpler layouts and requires a region lookup step per element access.
> It is not intended for performance-critical inner loops over dense data. Its primary
> use case is describing tensors with structurally heterogeneous regions (e.g., a mix
> of dense tiles and sparse blocks, or a tensor assembled from independently-produced
> shards with different inner layouts).

## Additional Descriptor Fields

| Field | Type | Description |
|-------|------|-------------|
| `region_count` | `uint32` | Number of regions. MUST be greater than 0. |

Followed by `region_count` **region descriptors**, each encoded as:

| Field | Type | Description |
|-------|------|-------------|
| `origin` | `uint64[rank]` | Starting index of the region along each dimension (inclusive). |
| `region_shape` | `uint64[rank]` | Size of the region along each dimension. Every value MUST be greater than 0. |
| `region_layout_tag` | `uint8` | Layout of elements within this region. MUST NOT be `0x00` or `0xFF`. |
| `_reserved` | `uint8[3]` | MUST be `0x00`. |
| `buffer_index` | `uint32` | Index into the buffer table for this region's data. |
| `region_byte_offset` | `uint64` | Byte offset within the referenced buffer to the start of this region's data. |

After `region_byte_offset`, the layout-specific fields for the region's
`region_layout_tag` are encoded inline (recursively if `region_layout_tag` is
`0x04` or `0x06`). The field names `region_layout_tag` and `region_byte_offset`
match the binary encoding given in `metadata.md` § General Subpaving (`0x06`).

## Coverage Constraint

The union of all regions MUST exactly cover every element in the tensor's index
space: for every valid index `[i_0, i_1, ..., i_{r-1}]` (where `0 <= i_k < shape[k]`
for all `k`), there MUST be exactly one region whose bounding box contains that index.

## Non-Overlap Constraint

Regions MUST NOT overlap. Two regions `A` and `B` overlap if, for every dimension `k`:

```
A.origin[k] < B.origin[k] + B.region_shape[k]
AND
B.origin[k] < A.origin[k] + A.region_shape[k]
```

A conforming writer MUST produce a valid subpaving. A conforming reader SHOULD
validate coverage and non-overlap and MUST reject violating descriptors unless in
permissive mode.

## Element Address

To locate element `[i_0, i_1, ..., i_{r-1}]`:

1. Find the region whose bounding box contains the index: `origin[k] <= i[k] < origin[k] + region_shape[k]` for all `k`.
2. Compute the local index: `local[k] = i[k] - origin[k]`.
3. Apply the region's layout addressing (per `region_layout_tag`) to `local` to compute the offset within the region's buffer at its `region_byte_offset`.

## Example

Rank-2 tensor with shape `[8, 8]` split into four 4×4 quadrants in row-major order:

- Region 0: `origin = [0, 0]`, `region_shape = [4, 4]`, `region_layout_tag = 0x01`
- Region 1: `origin = [0, 4]`, `region_shape = [4, 4]`, `region_layout_tag = 0x01`
- Region 2: `origin = [4, 0]`, `region_shape = [4, 4]`, `region_layout_tag = 0x01`
- Region 3: `origin = [4, 4]`, `region_shape = [4, 4]`, `region_layout_tag = 0x01`

Element `[5, 6]` falls in region 3 (origin `[4, 4]`). Local index: `[1, 2]`.
Offset within region: `1 * 4 + 2 = 6`.
