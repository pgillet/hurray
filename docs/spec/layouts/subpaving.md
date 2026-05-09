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
| `region_layout_length` | `uint32` | Byte count of the inner layout payload that follows. MUST be `0` for `region_layout_tag` values `0x01` and `0x02`. |
| `region_layout_payload` | `bytes[region_layout_length]` | Layout-specific fields for `region_layout_tag`, encoded identically to `metadata.md` § Layout-Specific Fields for that tag, with the tag byte omitted. |

The `region_layout_length` field enables forward-skipping: a reader that does not
recognise `region_layout_tag` MAY skip `region_layout_length` bytes and continue
parsing subsequent RegionDescriptors. A strict-mode reader MUST reject unrecognised
inner layout tags.

Recursive subpaving (`region_layout_tag = 0x06`) is permitted. A reader MUST
reject any descriptor where the subpaving nesting depth exceeds 8 levels.

## Coverage Constraint

The union of all regions MUST exactly cover every element in the tensor's index
space: for every valid index `[i_0, i_1, ..., i_{r-1}]` (where `0 <= i_k < shape[k]`
for all `k`), there MUST be exactly one region whose bounding box contains that index.

## Region Order

Writers SHOULD emit regions in ascending lexicographic order of `origin`,
comparing dimensions from index `0` to `rank-1`. Readers MUST NOT rely on
this order for correctness; a conforming reader MUST accept regions in any order.

> **Note (non-normative):** Writers that already produce regions in scan order
> (row-major, column-major, or tile-traversal order) satisfy this SHOULD without
> extra work. Readers MAY check whether the region array is already sorted on
> input; if so, lookup MAY use binary search on `origin`. Otherwise the reader
> MAY sort once at parse time or fall back to a linear scan. None of these
> acceleration strategies are part of the format.

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
