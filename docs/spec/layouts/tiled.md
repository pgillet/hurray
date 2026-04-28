# Tiled / Blocked Layout — Hurray Format Specification

**Layout tag:** `0x04` | **Tier:** 1

> This section uses RFC 2119 key words: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT,
> SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

## Description

A tiled layout partitions the tensor's index space into uniform rectangular tiles
(blocks). This is a **regular subpaving**: all tiles have the same shape and tile the
index space without overlap or gap (with possible padding at the boundaries).

## Additional Descriptor Fields

| Field | Type | Description |
|-------|------|-------------|
| `tile_shape` | `uint64[rank]` | Tile size along each dimension. Every value MUST be greater than 0. |
| `outer_layout` | `uint8` | Layout tag for tile-grid ordering. MUST be `0x01`, `0x02`, or `0x03`. |
| `inner_layout` | `uint8` | Layout tag for element ordering within each tile. MUST be `0x01`, `0x02`, `0x03`, or `0x04` (recursive tiling). |
| `_reserved` | `uint8[2]` | MUST be `0x00`. |

If `outer_layout` is `0x03` (strided):

| Field | Type | Description |
|-------|------|-------------|
| `outer_strides` | `int64[rank]` | Outer strides in units of **tiles** (not elements). |

If `inner_layout` is `0x03` (strided):

| Field | Type | Description |
|-------|------|-------------|
| `inner_strides` | `int64[rank]` | Inner strides in **logical elements** within a tile. |

If `inner_layout` is `0x04` (recursive tiling), the tiled layout-specific fields are
encoded recursively beginning with `tile_shape`. A reader MUST enforce a maximum
recursion depth (RECOMMENDED: 8 levels) and MUST reject descriptors that exceed it.

## Element Address

To locate element `[i_0, i_1, ..., i_{r-1}]`:

1. Compute the tile index: `t_k = floor(i_k / tile_shape[k])` for each dimension.
2. Compute the intra-tile offset: `e_k = i_k mod tile_shape[k]` for each dimension.
3. Compute the linear tile number using the outer layout applied to `[t_0, ..., t_{r-1}]`. Multiply by `total_tile_elements` to get the byte offset to the tile start.
4. Compute the intra-tile linear offset using the inner layout applied to `[e_0, ..., e_{r-1}]`.
5. Final linear element offset = (3) + (4).

## Boundary Padding

When a dimension size is not evenly divisible by the tile size, partial tiles exist at
the boundary. The buffer MUST contain storage for full tiles, including padding.
Padding element values are undefined; readers MUST NOT access elements whose logical
index exceeds `shape`.

Number of tiles per dimension:

```
num_tiles[k] = ceil(shape[k] / tile_shape[k])
```

Total buffer elements:

```
total_tile_elements = product(tile_shape[k] for all k)
total_tiles        = product(num_tiles[k] for all k)
buffer_elements    = total_tiles * total_tile_elements
```

## Validity Constraints

This layout MUST NOT be used for rank-0 (scalar) tensors. See
`data-model.md` § Scalar Tensors.

## Recursive Tiling

> **Note (non-normative):** Recursive tiling is useful for hierarchical blocking in
> GEMM kernels (e.g., 128×128 L2 tiles subdivided into 32×32 L1 tiles). It is also
> the natural expression of cache-oblivious recursive blocking.

## Example

Rank-2 tensor with shape `[6, 8]`, `tile_shape = [2, 4]`, `outer_layout = 0x01`
(row-major), `inner_layout = 0x01` (row-major).

- Tile grid shape: `[3, 2]`. Outer strides (implicit): `[2, 1]` in tiles.
- Each tile: 8 elements. Inner strides (implicit): `[4, 1]` in elements.
- Element `[3, 5]`: tile `[1, 1]`, intra-tile `[1, 1]`.
  Tile linear index = `1 * 2 + 1 = 3`. Tile start = `3 * 8 = 24`.
  Intra-tile offset = `1 * 4 + 1 = 5`. Final offset = `29`.
