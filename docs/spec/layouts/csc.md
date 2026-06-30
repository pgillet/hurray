# CSC (Compressed Sparse Column) Layout — Hurray Format Specification

**Layout tag:** `0x09` | **Tier:** 1

> Also known as: **CCS (Compressed Column Storage)**

> This section uses RFC 2119 key words: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT,
> SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

## Description

The CSC (Compressed Sparse Column) format is the column analog of CSR. It stores a
sparse matrix (rank-2 tensor) by compressing the column structure: each column is
represented by a contiguous slice of a flat non-zero array, with a separate pointer
array indicating where each column begins.

CSC is preferred when the workload requires efficient column access or column
iteration — e.g., sparse matrix-vector products with column-major dense vectors,
column pivoting in sparse direct solvers, and graph algorithms that traverse
out-edges (adjacency stored column-wise).

> **Note (non-normative):** CSC is defined only for **rank-2 tensors** in this
> version of the specification, consistent with CSR; generalisation to arbitrary rank
> is provided by the Compressed Sparse Fiber (CSF) layout (see [csf.md](csf.md)). The format is also known as
> Compressed Column Storage (CCS) in some communities (e.g., MATLAB, some LAPACK
> interfaces). The two names refer to the same format.

A conforming implementation MUST reject a CSC descriptor whose `rank` is not 2.

## Buffer Table

A CSC tensor descriptor MUST have `buffer_count = 3` in the buffer table.

| Buffer index | Name | Element type | Length | Description |
|---|---|---|---|---|
| 0 | `values` | tensor element type | `nnz` elements | Non-zero values in column-major order (all non-zeros of column 0 first, then column 1, etc.). |
| 1 | `row_indices` | `uint64` | `nnz` elements | Row index of each non-zero. `row_indices[i]` is the row of `values[i]`. |
| 2 | `col_ptr` | `uint64` | `ncols + 1` elements | Column pointer array. `col_ptr[j]` is the index into `values` / `row_indices` of the first non-zero in column `j`. `col_ptr[ncols] = nnz`. |

where `ncols = shape[1]`.

## Additional Descriptor Fields

| Field | Type | Description |
|-------|------|-------------|
| `nnz` | `uint64` | Number of stored (non-zero) elements. MAY be 0 for an empty sparse matrix. |
| `_reserved` | `uint8[8]` | MUST be `0x00`. |

## Validity Constraints

This layout MUST NOT be used for rank-0 (scalar) tensors. See
`data-model.md` § Scalar Tensors. CSC is further restricted to rank 2 (see
§ Description).

## Storage Invariants

A conforming writer MUST ensure:

1. `col_ptr[0] = 0` and `col_ptr[ncols] = nnz`.
2. `col_ptr` is non-decreasing: `col_ptr[j] <= col_ptr[j+1]` for all `j`.
3. Within each column `j`, the non-zeros in `row_indices[col_ptr[j]..col_ptr[j+1])`
   MUST be sorted in strictly increasing order (no duplicate row indices per column).
4. All row indices are within bounds: `0 <= row_indices[i] < shape[0]` for all `i`.

A conforming reader SHOULD validate these invariants and MUST reject descriptors that
violate them, unless operating in permissive mode.

## Buffer Size

- `values` buffer: `nnz * element_byte_width` bytes (or `ceil(nnz / packing_factor)`
  for sub-byte types).
- `row_indices` buffer: `nnz * 8` bytes (`uint64` elements).
- `col_ptr` buffer: `(ncols + 1) * 8` bytes (`uint64` elements), where
  `ncols = shape[1]`.

All buffers MUST satisfy the alignment requirements in `buffer-protocol.md`.

> **Note (non-normative):** As with CSR and COO, the `byte_offset` field in the
> common descriptor header is not meaningful for CSC. For CSC tensors, `byte_offset`
> MUST be set to `0x0000000000000000`.

## Element Lookup

To retrieve the value at row `r`, column `c`:

1. The non-zeros of column `c` occupy positions `col_ptr[c]` through
   `col_ptr[c+1] - 1` in `values` and `row_indices`.
2. Perform a binary search on `row_indices[col_ptr[c]..col_ptr[c+1])` for value `r`.
3. If found at position `i`, return `values[i]`.
4. If not found, the element is implicitly **zero**.

Binary search is valid because row indices within each column are sorted (invariant 3).

## Column Iteration

To iterate over all non-zeros in column `c`:

```
for i = col_ptr[c] to col_ptr[c+1] - 1:
    row = row_indices[i]
    val = values[i]
```

## Relationship to CSR

CSC is the transpose of CSR: the CSC representation of matrix `A` is equivalent to
the CSR representation of `A^T` with `shape` swapped. Implementations that support
both formats MAY convert between them by transposing the pointer and index arrays.

## Interaction with Statistics Section

When the `HAS_STATISTICS` flag is set, the `nnz` field in the statistics section
MUST match the `nnz` field in the CSC descriptor, which is authoritative. The `sparsity_ratio` SHOULD be
`1.0 - (nnz / (shape[0] * shape[1]))`.

## Example

The same matrix as in the CSR example (shape `[4, 5]`, element type `float32`):

```
Dense representation:
  row 0: [1.0,  0,   0,  2.0,  0 ]
  row 1: [ 0,   0,  3.0,  0,   0 ]
  row 2: [ 0,  4.0,  0,   0,  5.0]
  row 3: [ 0,   0,   0,   0,   0 ]

nnz = 5

values      (buffer 0): [1.0, 4.0, 3.0, 2.0, 5.0]
row_indices (buffer 1): [0,   2,   1,   0,   2  ]
col_ptr     (buffer 2): [0,   1,   2,   3,   4,   5]
```

`col_ptr[j+1] - col_ptr[j]` = number of non-zeros in column `j`:
col 0: 1 (value 1.0 at row 0), col 1: 1 (value 4.0 at row 2),
col 2: 1 (value 3.0 at row 1), col 3: 1 (value 2.0 at row 0),
col 4: 1 (value 5.0 at row 2).
