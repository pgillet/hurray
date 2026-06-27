# CSF (Compressed Sparse Fiber) Layout — Hurray Format Specification

**Layout tag:** `0x0A` | **Tier:** 1

> This section uses RFC 2119 key words: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT,
> SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

## Description

The CSF (Compressed Sparse Fiber) format is the rank-N generalisation of CSR/CSC. It
stores a sparse tensor as a tree of `rank` levels, where each level compresses one
mode (logical dimension) with a `(pos, crd)` pair: a `pos` pointer array delimiting
each parent's children, and a `crd` coordinate array naming those children. A single
`values` array holds the non-zero values at the leaves. CSF is the natural higher-rank
complement to COO for structured sparsity — sparse attention masks, sparse
activations, and higher-order tensor factorisations.

Every level of a CSF tree is **compressed**: there is no per-mode dense/compressed
distinction in this version of the specification. Dense-outer rank-2 cases are covered
by CSR/CSC.

> **Note (non-normative):** CSF is defined only for **rank ≥ 3 tensors** in this
> version. Rank-2 sparse matrices are served by CSR ([csr.md](csr.md)) and CSC
> ([csc.md](csc.md)), whose dense outer pointer array is more compact and more
> interop-canonical. COO ([coo.md](coo.md)) (which supports any rank) and CSR/CSC are
> preferable for rank ≤ 2. A future revision could admit dense levels via a
> `mode_format` field; a reader of this version rejecting such a descriptor is intended
> versioning behaviour.

A conforming implementation MUST reject a CSF descriptor whose `rank` is less than 3.

## Mode Ordering

A CSF descriptor carries a `mode_order: uint32[rank]` field, a permutation of the
integers `0..rank-1`. `mode_order[L]` is the logical dimension stored at tree level
`L`, where level `0` is the outermost level and level `rank-1` is the leaf level
directly above `values`. The bounding size for level `L` — the upper bound on every
coordinate stored at that level — is `shape[mode_order[L]]`.

`mode_order` affects storage traversal only; it does not change the tensor's logical
shape.

A conforming reader MUST honour any valid `mode_order` permutation for lookup and
iteration. A conforming reader MUST NOT reject a CSF descriptor merely because
`mode_order` is non-identity. A conforming reader MUST reject a descriptor whose
`mode_order` is not a valid permutation of `0..rank-1` (e.g. contains a duplicate or an
out-of-range value), unless operating in permissive mode.

When a writer has no access-pattern preference, the identity ordering
`[0, 1, ..., rank-1]` (row-major, outer-to-inner) SHOULD be used as the default.

> **Note (non-normative):** `mode_order` is a performance knob: it lets a writer match
> the tree's nesting to its access pattern. The identity ordering is preferred as the
> default because it is reproducible across writers and keeps the outermost (slowest-
> varying) logical dimension at the top of the tree, which is cache-friendly for
> row-major access.

## Buffer Table

A CSF tensor descriptor MUST have `buffer_count = 2 * rank + 1` in the buffer table
(plus any quantization-parameter buffers, which follow at indices `2 * rank + 1` and
up).

| Buffer index | Name | Element type | Length | Description |
|---|---|---|---|---|
| 0 | `values` | tensor element type | `nnz` elements | Non-zero values, ordered by tree traversal (leaf order). `values[p]` is the value reached by descending to leaf position `p`. |
| `2L + 1` | `pos_L` | `uint64` | `n_{L-1} + 1` elements (`2` for level 0) | Level-`L` pointer array. `pos_L[k]` and `pos_L[k+1]` delimit the children of parent `k` in `crd_L`. |
| `2L + 2` | `crd_L` | `uint64` | `n_L` elements (`nnz` for the leaf level) | Level-`L` coordinate array. `crd_L[i]` is a coordinate along logical dimension `mode_order[L]`. |

where `n_L` is the number of nodes stored at level `L`, `n_{-1} = 1` (a single virtual
root), and `n_{rank-1} = nnz`. The top-level pointer array `pos_0` always has length 2
and equals `[0, n_0]`.

## Additional Descriptor Fields

| Field | Type | Description |
|-------|------|-------------|
| `nnz` | `uint64` | Number of stored (non-zero) elements. MAY be 0 for an empty sparse tensor. |
| `mode_order` | `uint32[rank]` | Permutation of `0..rank-1`; `mode_order[L]` is the logical dimension stored at level `L`. See § Mode Ordering. |
| `_reserved` | `uint8[8]` | MUST be `0x00`. |

## Validity Constraints

This layout MUST NOT be used for rank-0 (scalar), rank-1, or rank-2 tensors. CSF
requires `rank >= 3` (see § Description); rank-1 and rank-2 cases are served by COO,
CSR, and CSC. Rank remains capped at 64 by `data-model.md`.

## Storage Invariants

A conforming writer MUST ensure, for every level `L` in `0..rank-1`:

1. `pos_L[0] = 0`.
2. `pos_L` is non-decreasing: `pos_L[k] <= pos_L[k+1]` for all `k`.
3. The terminal `pos` entry equals the level's stored count: `pos_0[1] = n_0`,
   `pos_L[n_{L-1}] = n_L`, and `n_{rank-1} = nnz`.
4. Within each parent slice `crd_L[pos_L[k]..pos_L[k+1])`, the coordinates MUST be
   sorted in strictly increasing order (no duplicate siblings).
5. All coordinates are within bounds: `0 <= crd_L[i] < shape[mode_order[L]]` for all
   `i`.

In addition, `mode_order` MUST be a valid permutation of `0..rank-1`.

For an empty tensor (`nnz = 0`), `pos_0` MUST be `[0, 0]`, `values` and every `crd_L`
MUST have length 0, and for each level `L >= 1` `pos_L` MUST be `[0]` (length 1, since
`n_{L-1} = 0`).

A conforming reader SHOULD validate these invariants and MUST reject descriptors that
violate them, unless operating in permissive mode.

## Buffer Size

- `values` buffer: `nnz * element_byte_width` bytes (or `ceil(nnz / packing_factor)`
  for sub-byte types).
- `pos_L` buffer: `(n_{L-1} + 1) * 8` bytes (`uint64` elements); `pos_0` is always
  `2 * 8 = 16` bytes.
- `crd_L` buffer: `n_L * 8` bytes (`uint64` elements); the leaf `crd_{rank-1}` is
  `nnz * 8` bytes.

All buffers MUST satisfy the alignment requirements in `buffer-protocol.md`.

For CSF tensors, `byte_offset` MUST be set to `0x0000000000000000`.

> **Note (non-normative):** As with COO, CSR, and CSC, the `byte_offset` field in the
> common descriptor header is not meaningful for CSF — there is no single "first
> element" at a fixed offset; the first stored element is located by descending the
> tree.

## Element Lookup

To retrieve the value at logical index `idx = [idx[0], ..., idx[rank-1]]`:

1. Permute the query into storage order: the coordinate sought at level `L` is
   `q_L = idx[mode_order[L]]`.
2. Initialise the parent position `p = 0` (the virtual root).
3. For each level `L` from `0` to `rank-1`:
   - The children of the current parent occupy the slice
     `crd_L[pos_L[p] .. pos_L[p+1])`.
   - Perform a binary search on that slice for `q_L`.
   - If the binary search finds `q_L` at relative offset `k` within the slice (so
     that `crd_L[pos_L[p] + k] == q_L`), set `p = pos_L[p] + k` — the **absolute**
     index into `crd_L` — and continue to the next level.
   - If not found, the element is implicitly **zero**; stop.
4. After the leaf level (`L = rank-1`), `p` is the leaf position; return `values[p]`.

Binary search is valid at each level because sibling coordinates within a parent slice
are sorted (invariant 4).

## Iteration

To iterate over all non-zeros, descend the tree in depth-first, leaf order. The
following sketch accumulates the storage-order coordinate tuple `c[0..rank-1]`; the
logical index is recovered by placing `c[L]` at dimension `mode_order[L]`:

```
def visit(L, p):
    for i = pos_L[p] to pos_L[p+1] - 1:
        c[L] = crd_L[i]
        if L == rank - 1:
            emit(c, values[i])      # i is the leaf position
        else:
            visit(L + 1, i)

visit(0, 0)
```

## Relationship to CSR / CSC / COO

CSF generalises CSR/CSC to arbitrary rank but does **not** replace them. For rank-2
tensors, writers SHOULD prefer CSR ([csr.md](csr.md)) or CSC ([csc.md](csc.md)), whose
dense outer pointer array is more compact and more interop-canonical; for rank-1 and
rank-2 coordinate lists, writers SHOULD prefer COO ([coo.md](coo.md)). CSF MUST NOT be
used below rank 3.

A rank-2 CSF tree with `mode_order = [0, 1]` is structurally analogous to CSR (`pos_0`
the trivial root pointer, `pos_1` the row pointer, `crd_1` the column indices), and
with `mode_order = [1, 0]` to CSC — but CSF stores an explicit top-level `pos_0`/`crd_0`
pair that CSR/CSC fold into their dense outer array, so the layouts are not
byte-identical.

## Quantization Compatibility

The `values` buffer (buffer index 0) holds the quantized leaf values, exactly as for
COO and CSR. Quantization schemes decorate the leaves: per-tensor, per-channel,
per-block, NF4, and MXFP compose as for COO/CSR, following the placement rules in
`quantization.md`. Quantization-parameter buffers occupy buffer indices `2 * rank + 1`
and up. All quantization-parameter buffers MUST carry the same `device_tag` as the
`values` buffer.

## Sharding

A CSF tensor MUST NOT carry a shard descriptor in this version of the specification.
Sharding a CSF tree (the interaction between a shard's `shard_offset` and the
per-level `pos`/`crd` structure) is deferred to a future revision.

## Interaction with Statistics Section

When the `HAS_STATISTICS` flag is set, the `nnz` field in the statistics section
SHOULD match the `nnz` field in the CSF descriptor. The `sparsity_ratio` SHOULD be
`1.0 - (nnz / total_elements)`, where `total_elements` is the product of all `shape`
entries.

## Example

Rank-3 sparse tensor with shape `[2, 3, 4]`, element type `float32`,
`mode_order = [0, 1, 2]` (identity), 4 non-zeros:

```
Non-zeros (logical index -> value):
  (0, 0, 1) -> 1.0
  (0, 2, 3) -> 2.0
  (1, 1, 0) -> 3.0
  (1, 1, 2) -> 4.0

nnz = 4
buffer_count = 2 * 3 + 1 = 7

Level 0 (dimension 0, bound 2):
  pos_0   (buffer 1): [0, 2]
  crd_0   (buffer 2): [0, 1]              # two non-empty slices: i=0, i=1

Level 1 (dimension 1, bound 3):
  pos_1   (buffer 3): [0, 2, 3]           # parent 0 has 2 children, parent 1 has 1
  crd_1   (buffer 4): [0, 2, 1]           # i=0: j=0,2 ; i=1: j=1

Level 2 (dimension 2, bound 4):
  pos_2   (buffer 5): [0, 1, 2, 4]        # leaf delimiters
  crd_2   (buffer 6): [1, 3, 0, 2]        # the k coordinates

values    (buffer 0): [1.0, 2.0, 3.0, 4.0]
```

Lookup of `(1, 1, 2)`:

```
q = [1, 1, 2]                 # mode_order is identity

Level 0: parent p = 0, slice crd_0[pos_0[0]..pos_0[1]) = crd_0[0..2) = [0, 1]
         search 1 -> relative offset k = 1; p = pos_0[0] + 1 = 0 + 1 = 1
Level 1: slice crd_1[pos_1[1]..pos_1[2]) = crd_1[2..3) = [1]
         search 1 -> relative offset k = 0; p = pos_1[1] + 0 = 2 + 0 = 2
Level 2: slice crd_2[pos_2[2]..pos_2[3]) = crd_2[2..4) = [0, 2]
         search 2 -> relative offset k = 1; p = pos_2[2] + 1 = 2 + 1 = 3

leaf position p = 3 -> values[3] = 4.0
```
