# CSR (Compressed Sparse Row) Layout — Hurray Format Specification

**Layout tag:** `0x07` | **Tier:** 1

> This section uses RFC 2119 key words: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT,
> SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

## Description

The CSR (Compressed Sparse Row) format stores a sparse matrix (rank-2 tensor) by
compressing the row structure: each row is represented by a contiguous slice of a
flat non-zero array, with a separate pointer array indicating where each row begins.
CSR is the most widely-used sparse matrix format in scientific computing and machine
learning (e.g., sparse attention, graph neural networks, sparse weight matrices).

> **Note (non-normative):** CSR is defined only for **rank-2 tensors** in this
> version of the specification. Generalisation to arbitrary rank (Compressed Sparse
> Fiber, CSF) is specified in [CSF (Compressed Sparse Fiber)](csf.md).

A conforming implementation MUST reject a CSR descriptor whose `rank` is not 2.

## Buffer Table

A CSR tensor descriptor MUST have `buffer_count = 3` in the buffer table.

| Buffer index | Name | Element type | Length | Description |
|---|---|---|---|---|
| 0 | `values` | tensor element type | `nnz` elements | Non-zero values in row-major order (all non-zeros of row 0 first, then row 1, etc.). |
| 1 | `col_indices` | `uint64` | `nnz` elements | Column index of each non-zero. `col_indices[i]` is the column of `values[i]`. |
| 2 | `row_ptr` | `uint64` | `nrows + 1` elements | Row pointer array. `row_ptr[i]` is the index into `values` / `col_indices` of the first non-zero in row `i`. `row_ptr[nrows] = nnz`. |

where `nrows = shape[0]`.

## Additional Descriptor Fields

| Field | Type | Description |
|-------|------|-------------|
| `nnz` | `uint64` | Number of stored (non-zero) elements. MAY be 0 for an empty sparse matrix. |
| `_reserved` | `uint8[8]` | MUST be `0x00`. |

## Validity Constraints

This layout MUST NOT be used for rank-0 (scalar) tensors. See
`data-model.md` § Scalar Tensors. CSR is further restricted to rank 2 (see
§ Description).

## Storage Invariants

A conforming writer MUST ensure:

1. `row_ptr[0] = 0` and `row_ptr[nrows] = nnz`.
2. `row_ptr` is non-decreasing: `row_ptr[i] <= row_ptr[i+1]` for all `i`.
3. Within each row `i`, the non-zeros in `col_indices[row_ptr[i]..row_ptr[i+1])` MUST
   be sorted in strictly increasing order (no duplicate column indices per row).
4. All column indices are within bounds: `0 <= col_indices[j] < shape[1]` for all `j`.

A conforming reader SHOULD validate these invariants and MUST reject descriptors that
violate them, unless operating in permissive mode.

## Buffer Size

- `values` buffer: `nnz * element_byte_width` bytes (or `ceil(nnz / packing_factor)`
  for sub-byte types).
- `col_indices` buffer: `nnz * 8` bytes (`uint64` elements).
- `row_ptr` buffer: `(nrows + 1) * 8` bytes (`uint64` elements), where
  `nrows = shape[0]`.

All buffers MUST satisfy the alignment requirements in `buffer-protocol.md`.

For CSR tensors, `byte_offset` MUST be set to `0x0000000000000000`.

> **Note (non-normative):** The `byte_offset` field in the common descriptor header
> is not meaningful for CSR — there is no single "first element" at a fixed offset.

## Element Lookup

To retrieve the value at row `r`, column `c`:

1. The non-zeros of row `r` occupy positions `row_ptr[r]` through `row_ptr[r+1] - 1`
   in `values` and `col_indices`.
2. Perform a binary search on `col_indices[row_ptr[r]..row_ptr[r+1])` for value `c`.
3. If found at position `j`, return `values[j]`.
4. If not found, the element is implicitly **zero**.

Binary search is valid because column indices within each row are sorted (invariant 3).

## Row Iteration

To iterate over all non-zeros in row `r`:

```
for j = row_ptr[r] to row_ptr[r+1] - 1:
    col = col_indices[j]
    val = values[j]
```

## Interaction with Statistics Section

When the `HAS_STATISTICS` flag is set, the `nnz` field in the statistics section
MUST match the `nnz` field in the CSR descriptor, which is authoritative. The `sparsity_ratio` SHOULD be
`1.0 - (nnz / (shape[0] * shape[1]))`.

## Example

Rank-2 sparse matrix with shape `[4, 5]`, element type `float32`:

```
Dense representation:
  row 0: [1.0,  0,   0,  2.0,  0 ]
  row 1: [ 0,   0,  3.0,  0,   0 ]
  row 2: [ 0,  4.0,  0,   0,  5.0]
  row 3: [ 0,   0,   0,   0,   0 ]

nnz = 5

values      (buffer 0): [1.0, 2.0, 3.0, 4.0, 5.0]
col_indices (buffer 1): [0,   3,   2,   1,   4  ]
row_ptr     (buffer 2): [0,   2,   3,   5,   5  ]
```

`row_ptr[3] = row_ptr[4] = 5` because row 3 has no non-zeros.
