# Layer 3: Layout Descriptors

## Purpose

A layout descriptor tells a reader how the elements of a tensor are arranged in memory. Every tensor descriptor includes exactly one layout tag byte followed by layout-specific fields. The `hurray-core` `LayoutDescriptor` enum models all layouts defined in the spec, from the zero-overhead unit variants (`RowMajor`, `ColMajor`) to sparse multi-buffer formats and permissive-mode passthrough.

## Quick reference: layout tags and buffer counts

| Variant | Tag | Buffer count | Notes |
|---------|-----|-------------|-------|
| `RowMajor` | `0x01` | 1 | No fields; strides are implicit |
| `ColMajor` | `0x02` | 1 | No fields; strides are implicit |
| `Strided` | `0x03` | 1 | Explicit `strides: Vec<i64>`; negative/zero valid |
| `Tiled` | `0x04` | 1 | Tile shape, outer/inner layout tags, optional strides; recursive |
| `Morton` | `0x05` | 1 | Per-dimension bit counts |
| `Subpaving` | `0x06` | 1 | List of rectangular regions, each with its own inner layout |
| `Coo` | `0x07` | 2 | `nnz`, `is_sorted`; values + index buffers |
| `Csr` | `0x08` | 3 | `nnz`; values + col_indices + row_ptr; rank-2 only |
| `Csc` | `0x09` | 3 | `nnz`; values + row_indices + col_ptr; rank-2 only |
| `Csf` | `0x0A` | `2·rank+1` | `nnz`, `mode_order` permutation; values + per-level pos/crd; rank-3+ generalization of CSR/CSC. See [csf.md](../spec/layouts/csf.md) |
| `BlockPaged` | `0x0B` | 3 | PagedAttention KV cache; page_pool + block_table + seq_ptr; rank-3 only. See [block-paged-kv-cache.md](block-paged-kv-cache.md) |
| `Hilbert` | `0x40` | 1 | `hilbert_order`, `hilbert_rank`; dims must be `2^order` |
| `PrivateExtension` | `0xF0`–`0xFE` | `None` | Opaque; requires out-of-band agreement |
| `Unknown` | any unrecognised | `None` | Permissive mode only; never dereference data |

## Constructing dense layouts

Unit variants need no constructor:

```rust
use hurray_core::layout::LayoutDescriptor;

let rm = LayoutDescriptor::RowMajor;
let cm = LayoutDescriptor::ColMajor;
assert_eq!(rm.tag(), 0x01);
assert_eq!(cm.tag(), 0x02);
```

Strided layout — explicit per-dimension strides in logical elements. Negative strides reverse a dimension; zero strides broadcast (virtual dimension, no physical replication):

```rust
use hurray_core::layout::{LayoutDescriptor, StridedLayout};

// Row-major strides for a 3×4 tensor: last dim varies fastest.
let rm_strides = LayoutDescriptor::Strided(StridedLayout::new(vec![4, 1]));

// Same tensor with first dimension reversed.
let reversed = LayoutDescriptor::Strided(StridedLayout::new(vec![-4, 1]));

// Broadcast along dimension 0: all rows map to row 0.
let broadcast = LayoutDescriptor::Strided(StridedLayout::new(vec![0, 1]));
```

## Tiled / blocked layout

2×4 tiles with row-major outer ordering and column-major inner ordering:

```rust
use hurray_core::layout::{LayoutDescriptor, TiledLayout};

let tiled = LayoutDescriptor::Tiled(Box::new(
    TiledLayout::new(
        vec![2, 4], // tile_shape
        0x01,       // outer_layout: row-major
        0x02,       // inner_layout: column-major
        None,       // outer_strides: None (implicit for row-major outer)
        None,       // inner_strides: None
        None,       // inner_tiled: None (not recursive)
    ).unwrap(),
));
```

Strided tile grid — outer_strides must be provided when `outer_layout == 0x03`:

```rust
use hurray_core::layout::{LayoutDescriptor, OuterStrides, TiledLayout};

let tiled_strided = LayoutDescriptor::Tiled(Box::new(
    TiledLayout::new(
        vec![2, 2],
        0x03, // strided outer
        0x01, // row-major inner
        Some(OuterStrides::new(vec![2, 1])), // tile-grid strides in units of tiles
        None,
        None,
    ).unwrap(),
));
```

Recursive tiling (two levels of blocking, useful for hierarchical GEMM caches):

```rust
use hurray_core::layout::TiledLayout;

let inner = TiledLayout::new(vec![4, 4], 0x01, 0x01, None, None, None).unwrap();
let outer = TiledLayout::new(
    vec![32, 32],
    0x01,
    0x04, // inner_layout is itself tiled
    None,
    None,
    Some(Box::new(inner)),
).unwrap();
```

Maximum recursion depth is 8 levels; deeper nesting returns `Error::InvalidLayout`.

## Sparse layouts

COO — two buffers (values + flat index array):

```rust
use hurray_core::layout::{CooLayout, LayoutDescriptor};

let coo = LayoutDescriptor::Coo(CooLayout::new(
    42,   // nnz
    true, // is_sorted: non-zeros in lexicographic order
));
assert_eq!(coo.buffer_count().map(|n| n.get()), Some(2));
```

CSR — three buffers (values + col_indices + row_ptr), rank-2 only:

```rust
use hurray_core::layout::{CsrLayout, LayoutDescriptor};

let csr = LayoutDescriptor::Csr(CsrLayout::new(100)); // nnz = 100
assert_eq!(csr.buffer_count().map(|n| n.get()), Some(3));
```

CSC — three buffers (values + row_indices + col_ptr), rank-2 only:

```rust
use hurray_core::layout::{CscLayout, LayoutDescriptor};

let csc = LayoutDescriptor::Csc(CscLayout::new(100));
assert_eq!(csc.buffer_count().map(|n| n.get()), Some(3));
```

CSF (Compressed Sparse Fiber) — the rank-N (rank ≥ 3) generalization of CSR/CSC, with
`2·rank + 1` buffers (`values` plus a `pos`/`crd` pair per level). The buffer count is
derived from the rank, which `CsfLayout` carries via its `mode_order` permutation
(`mode_order[L]` is the logical dimension stored at level `L`). Writers SHOULD prefer
CSR/CSC for rank-2 sparse matrices and reserve CSF for rank ≥ 3:

```rust
use hurray_core::layout::{CsfLayout, LayoutDescriptor};
use hurray_core::Shape;

// Rank-3 sparse tensor, identity mode order, 4 non-zeros.
let csf = LayoutDescriptor::Csf(CsfLayout::new(4, vec![0, 1, 2]));
assert_eq!(csf.tag(), 0x0A);
assert_eq!(csf.buffer_count().map(|n| n.get()), Some(7)); // 2*3 + 1

// rank ≥ 3 only; CSR/CSC own rank-2.
let shape = Shape::new(vec![2, 3, 4]).unwrap();
assert!(csf.validate_against_shape(&shape).is_ok());
assert!(csf
    .validate_against_shape(&Shape::new(vec![3, 4]).unwrap())
    .is_err());
```

See [csf.md](../spec/layouts/csf.md) for the full per-level buffer layout and lookup.

## Space-filling curve layouts

Morton (Z-order) — per-dimension bit counts control how many index bits are
interleaved per dimension. Each `shape[k]` must satisfy `shape[k] <= 2^morton_bits[k]`:

```rust
use hurray_core::layout::{LayoutDescriptor, MortonLayout};
use hurray_core::Shape;

// 4×4 tensor: each dim needs 2 bits (4 <= 2^2).
let morton = LayoutDescriptor::Morton(MortonLayout::new(vec![2, 2]).unwrap());
let shape = Shape::new(vec![4, 4]).unwrap();
morton.validate_against_shape(&shape).unwrap();
```

Hilbert curve — all dims must equal `2^hilbert_order`; rank must be >= 2:

```rust
use hurray_core::layout::{HilbertLayout, LayoutDescriptor};
use hurray_core::Shape;

// 8×8×8 tensor: order=3 (8 = 2^3), rank=3.
let hilbert = LayoutDescriptor::Hilbert(HilbertLayout::new(3, 3).unwrap());
let shape = Shape::new(vec![8, 8, 8]).unwrap();
hilbert.validate_against_shape(&shape).unwrap();
```

## General subpaving layout

Irregular partitioning: split an 8×8 tensor into four 4×4 row-major quadrants,
each pointing into a different byte offset of the same buffer:

```rust
use hurray_core::layout::{LayoutDescriptor, RegionDescriptor, SubpavingLayout};
use hurray_core::Shape;

let regions = vec![
    RegionDescriptor::new(vec![0, 0], vec![4, 4], 0x01, 0, 0).unwrap(),
    RegionDescriptor::new(vec![0, 4], vec![4, 4], 0x01, 0, 64).unwrap(),
    RegionDescriptor::new(vec![4, 0], vec![4, 4], 0x01, 0, 128).unwrap(),
    RegionDescriptor::new(vec![4, 4], vec![4, 4], 0x01, 0, 192).unwrap(),
];
let layout = LayoutDescriptor::Subpaving(SubpavingLayout::new(regions).unwrap());

let shape = Shape::new(vec![8, 8]).unwrap();
layout.validate_against_shape(&shape).unwrap();
```

`validate_against_shape` checks that regions don't overlap and don't exceed the
tensor's bounds. Overlapping regions return `Error::InvalidLayout`.

## Tag introspection and validation

```rust
use hurray_core::layout::{
    validate_layout_tag_strict, is_invalid_tag, is_reserved_tag, is_private_tag,
    LayoutDescriptor, UnknownLayout,
};
use hurray_core::Error;

// Check individual tag categories without constructing a descriptor.
// 0x10 is a genuinely unassigned tag in the Tier-1 reserved range.
assert!(is_invalid_tag(0x00));
assert!(is_reserved_tag(0x10));
assert!(is_private_tag(0xF3));

// Strict-mode validation: rejects invalid, reserved, and private tags.
assert!(validate_layout_tag_strict(0x01).is_ok());
assert!(matches!(validate_layout_tag_strict(0x00), Err(Error::InvalidLayoutTag(0x00))));
assert!(matches!(validate_layout_tag_strict(0x10), Err(Error::ReservedLayoutTag(0x10))));
assert!(matches!(validate_layout_tag_strict(0xF0), Err(Error::PrivateLayoutTag(0xF0))));

// Permissive mode: wrap unrecognised tags in Unknown for passthrough.
// The reader must NOT dereference the tensor data buffer for Unknown layouts.
let unknown = LayoutDescriptor::Unknown(UnknownLayout::new(0x10, vec![]).unwrap());
assert_eq!(unknown.tag(), 0x10);
assert!(unknown.buffer_count().is_none());
```

## Validating a descriptor against a tensor shape

`validate_against_shape` is called by Layer 4 (tensor descriptor) to enforce
layout-specific rank and dimension constraints. Call it explicitly when building
descriptors to catch mismatches early:

```rust
use hurray_core::layout::{CsrLayout, LayoutDescriptor};
use hurray_core::Shape;

let csr = LayoutDescriptor::Csr(CsrLayout::new(5));

// Rank-2: valid.
assert!(csr.validate_against_shape(&Shape::new(vec![4, 5]).unwrap()).is_ok());

// Rank-3: rejected — CSR is only defined for rank-2 tensors.
assert!(csr.validate_against_shape(&Shape::new(vec![2, 3, 4]).unwrap()).is_err());
```

## Subpaving validation

`validate_against_shape` enforces the full subpaving contract from
`subpaving.md`: each region lies within the tensor bounds, regions do not overlap,
and — per the spec's **Coverage Constraint** — the regions **exactly cover** the
index space. Coverage is checked via `∑ region volumes == total elements`: because
the regions are already verified to be in-bounds and non-overlapping, that equality
is sufficient to guarantee no gaps (an `O(n·rank)` check, not a cell-by-cell scan).
Coverage validation is skipped only when a dimension is dynamic or a volume product
overflows `uint64`, since the total then cannot be computed reliably.

## Private extension layouts

For hardware-specific panel/pack formats agreed out of band:

```rust
use hurray_core::layout::{LayoutDescriptor, PrivateExtensionLayout};

let private = LayoutDescriptor::PrivateExtension(
    PrivateExtensionLayout::new(
        0xF0,                    // tag: must be 0xF0–0xFE
        0xDEAD_BEEF_0000_0001,   // implementation-defined layout ID
        vec![0x01, 0x00, 0x04],  // opaque metadata
    ).unwrap(),
);
// buffer_count is None: the format doesn't know how many buffers this needs.
assert!(private.buffer_count().is_none());
```
