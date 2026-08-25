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
| `Coo` | `0x06` | 2 | `nnz`, `is_sorted`; values + index buffers |
| `Csr` | `0x07` | 3 | `nnz`; values + col_indices + row_ptr; rank-2 only |
| `Csc` | `0x08` | 3 | `nnz`; values + row_indices + col_ptr; rank-2 only |
| `Csf` | `0x09` | `2·rank+1` | `nnz`, `mode_order` permutation; values + per-level pos/crd; rank-3+ generalization of CSR/CSC. See [CSF (Compressed Sparse Fiber)](../spec/layouts/csf.md) |
| `BlockPaged` | `0x0A` | 3 | PagedAttention KV cache; page_pool + block_table + seq_ptr; rank-3 only. See [Block-Paged KV Cache](block-paged-kv-cache.md) |
| `Hilbert` | `0x40` | 1 | `hilbert_order`, `hilbert_rank`; dims must be `2^order` |
| `PrivateExtension` | `0xF0`–`0xFE` | `None` | Opaque; requires out-of-band agreement |
| `Unknown` | any unrecognised | `None` | Permissive mode only; never dereference data |

## Constructing dense layouts

Unit variants need no constructor:

<div class="lang-tabs">

```rust
use hurray_core::layout::LayoutDescriptor;

let rm = LayoutDescriptor::RowMajor;
let cm = LayoutDescriptor::ColMajor;
assert_eq!(rm.tag(), 0x01);
assert_eq!(cm.tag(), 0x02);
```

```python
import hurray

rm = hurray.RowMajorLayout()
cm = hurray.ColMajorLayout()
assert rm.tag == 0x01
assert cm.tag == 0x02
```

</div>

Python spells each layout as its own class rather than a tag byte (ADR-032); the
tag is still there on every one of them.

Strided layout — explicit per-dimension strides in logical elements. Negative strides reverse a dimension; zero strides broadcast (virtual dimension, no physical replication):

<div class="lang-tabs">

```rust
use hurray_core::layout::{LayoutDescriptor, StridedLayout};

// Row-major strides for a 3×4 tensor: last dim varies fastest.
let rm_strides = LayoutDescriptor::Strided(StridedLayout::new(vec![4, 1]));

// Same tensor with first dimension reversed.
let reversed = LayoutDescriptor::Strided(StridedLayout::new(vec![-4, 1]));

// Broadcast along dimension 0: all rows map to row 0.
let broadcast = LayoutDescriptor::Strided(StridedLayout::new(vec![0, 1]));
```

```python
import hurray

# Row-major strides for a 3x4 tensor: last dim varies fastest.
rm_strides = hurray.StridedLayout([4, 1])

# Same tensor with first dimension reversed.
reversed_ = hurray.StridedLayout([-4, 1])

# Broadcast along dimension 0: all rows map to row 0.
broadcast = hurray.StridedLayout([0, 1])

assert rm_strides.strides == (4, 1)
```

</div>

## Tiled / blocked layout

2×4 tiles with row-major outer ordering and column-major inner ordering:

<div class="lang-tabs">

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

```python
import hurray

tiled = hurray.TiledLayout(
    [2, 4],                    # tile_shape
    outer_layout="row_major",
    inner_layout="col_major",
)
```

</div>

The nested layouts are named, not tagged: `"row_major"` rather than `0x01`. Python
reads the tag back off `layout.tag` when it needs the wire value.

Strided tile grid — outer_strides must be provided when `outer_layout == 0x03`:

<div class="lang-tabs">

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

```python
import hurray

tiled_strided = hurray.TiledLayout(
    [2, 2],
    outer_layout="strided",            # tile-grid strides required
    inner_layout="row_major",
    outer_strides=[2, 1],              # in units of tiles
)
```

</div>

Recursive tiling (two levels of blocking, useful for hierarchical GEMM caches):

<div class="lang-tabs">

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

```python
import hurray

inner = hurray.TiledLayout([4, 4], "row_major", "row_major")
outer = hurray.TiledLayout(
    [32, 32],
    outer_layout="row_major",
    inner_layout="tiled",              # the inner layout is itself tiled
    inner_tiled=inner,
)
```

</div>

Maximum recursion depth is 8 levels; deeper nesting returns `Error::InvalidLayout`.

## Sparse layouts

COO — two buffers (values + flat index array):

<div class="lang-tabs">

```rust
use hurray_core::layout::{CooLayout, LayoutDescriptor};

let coo = LayoutDescriptor::Coo(CooLayout::new(
    42,   // nnz
    true, // is_sorted: non-zeros in lexicographic order
));
assert_eq!(coo.buffer_count().map(|n| n.get()), Some(2));
```

```python
import hurray

coo = hurray.CooLayout(
    nnz=42,
    is_sorted=True,     # non-zeros in lexicographic order
)
assert coo.buffer_count == 2
```

</div>

CSR — three buffers (values + col_indices + row_ptr), rank-2 only:

<div class="lang-tabs">

```rust
use hurray_core::layout::{CsrLayout, LayoutDescriptor};

let csr = LayoutDescriptor::Csr(CsrLayout::new(100)); // nnz = 100
assert_eq!(csr.buffer_count().map(|n| n.get()), Some(3));
```

```python
import hurray

csr = hurray.CsrLayout(nnz=100)
assert csr.buffer_count == 3
```

</div>

CSC — three buffers (values + row_indices + col_ptr), rank-2 only:

<div class="lang-tabs">

```rust
use hurray_core::layout::{CscLayout, LayoutDescriptor};

let csc = LayoutDescriptor::Csc(CscLayout::new(100));
assert_eq!(csc.buffer_count().map(|n| n.get()), Some(3));
```

```python
import hurray

csc = hurray.CscLayout(nnz=100)
assert csc.buffer_count == 3
```

</div>

CSF (Compressed Sparse Fiber) — the rank-N (rank ≥ 3) generalization of CSR/CSC, with
`2·rank + 1` buffers (`values` plus a `pos`/`crd` pair per level). The buffer count is
derived from the rank, which `CsfLayout` carries via its `mode_order` permutation
(`mode_order[L]` is the logical dimension stored at level `L`). Writers SHOULD prefer
CSR/CSC for rank-2 sparse matrices and reserve CSF for rank ≥ 3:

<div class="lang-tabs">

```rust
use hurray_core::layout::{CsfLayout, LayoutDescriptor};
use hurray_core::Shape;

// Rank-3 sparse tensor, identity mode order, 4 non-zeros.
let csf = LayoutDescriptor::Csf(CsfLayout::new(4, vec![0, 1, 2]));
assert_eq!(csf.tag(), 0x09);
assert_eq!(csf.buffer_count().map(|n| n.get()), Some(7)); // 2*3 + 1

// rank ≥ 3 only; CSR/CSC own rank-2.
let shape = Shape::new(vec![2, 3, 4]).unwrap();
assert!(csf.validate_against_shape(&shape).is_ok());
assert!(csf
    .validate_against_shape(&Shape::new(vec![3, 4]).unwrap())
    .is_err());
```

```python
import hurray

# Rank-3 sparse tensor, identity mode order, 4 non-zeros.
csf = hurray.CsfLayout(nnz=4, mode_order=[0, 1, 2])
assert csf.tag == 0x09
assert csf.buffer_count == 7        # 2*3 + 1

# rank >= 3 only; CSR/CSC own rank-2.
csf.validate_against_shape([2, 3, 4])
try:
    csf.validate_against_shape([3, 4])
    raise AssertionError("rank-2 should be refused")
except hurray.InvalidDescriptorError:
    pass
```

</div>

See [CSF (Compressed Sparse Fiber)](../spec/layouts/csf.md) for the full per-level buffer layout and lookup.

## Space-filling curve layouts

Morton (Z-order) — per-dimension bit counts control how many index bits are
interleaved per dimension. Each `shape[k]` must satisfy `shape[k] <= 2^morton_bits[k]`:

<div class="lang-tabs">

```rust
use hurray_core::layout::{LayoutDescriptor, MortonLayout};
use hurray_core::Shape;

// 4×4 tensor: each dim needs 2 bits (4 <= 2^2).
let morton = LayoutDescriptor::Morton(MortonLayout::new(vec![2, 2]).unwrap());
let shape = Shape::new(vec![4, 4]).unwrap();
morton.validate_against_shape(&shape).unwrap();
```

```python
import hurray

# 4x4 tensor: each dim needs 2 bits (4 <= 2^2).
morton = hurray.MortonLayout([2, 2])
morton.validate_against_shape([4, 4])
```

</div>

Hilbert curve — all dims must equal `2^hilbert_order`; rank must be >= 2:

<div class="lang-tabs">

```rust
use hurray_core::layout::{HilbertLayout, LayoutDescriptor};
use hurray_core::Shape;

// 8×8×8 tensor: order=3 (8 = 2^3), rank=3.
let hilbert = LayoutDescriptor::Hilbert(HilbertLayout::new(3, 3).unwrap());
let shape = Shape::new(vec![8, 8, 8]).unwrap();
hilbert.validate_against_shape(&shape).unwrap();
```

```python
import hurray

# 8x8x8 tensor: order=3 (8 = 2^3), rank=3.
hilbert = hurray.HilbertLayout(hilbert_order=3, hilbert_rank=3)
hilbert.validate_against_shape([8, 8, 8])
```

</div>

## Tag introspection and validation

<div class="lang-tabs">

```rust
use hurray_core::layout::{
    validate_layout_tag_strict, is_invalid_tag, is_named_tag, is_reserved_tag,
    is_private_tag, LayoutDescriptor, UnknownLayout,
};
use hurray_core::Error;

// Check individual tag categories without constructing a descriptor.
// 0x10 is a genuinely unassigned tag in the Tier-1 reserved range.
assert!(is_invalid_tag(0x00));
assert!(is_named_tag(0x07));      // CSR — this crate knows how to check it
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

// Only genuinely unrecognised tags: "unknown" is a claim, and it has to be true.
// A named tag wrapped this way would skip every check its own variant applies
// while still encoding to that tag on the wire.
assert!(matches!(UnknownLayout::new(0x07, vec![]), Err(Error::NamedLayoutTag(0x07))));
assert!(matches!(UnknownLayout::new(0xF0, vec![]), Err(Error::PrivateLayoutTag(0xF0))));
```

```python
import hurray

# Classify a tag without constructing a descriptor. The four categories partition
# the byte space, so one call answers the question four predicates would.
assert hurray.layout_tag_kind(0x07) == "named"      # CSR
assert hurray.layout_tag_kind(0x10) == "reserved"
assert hurray.layout_tag_kind(0xF3) == "private"
assert hurray.layout_tag_kind(0x00) == "invalid"

# Permissive mode: wrap an unrecognised tag for passthrough.
# The reader must NOT dereference the tensor data buffer for an Unknown layout.
unknown = hurray.UnknownLayout(0x10, b"")
assert unknown.tag == 0x10
assert unknown.buffer_count is None

# Only genuinely unrecognised tags: "unknown" is a claim, and it has to be true.
# A named tag wrapped this way would skip every check its own class applies while
# still encoding to that tag on the wire.
for taken in (0x07, 0xF0):
    try:
        hurray.UnknownLayout(taken, b"")
        raise AssertionError(f"0x{taken:02X} is not unknown")
    except ValueError as exc:
        print(exc)
```

</div>

The four kinds call for different reactions, which is why the classification is worth
having: **reserved** most likely means the producer is newer than this reader, so relaying
the tensor on is reasonable while interpreting its bytes is not; **private** belongs to an
out-of-band agreement; **invalid** means corruption or a framing error.

## Validating a descriptor against a tensor shape

`validate_against_shape` is called by Layer 4 (tensor descriptor) to enforce
layout-specific rank and dimension constraints. Call it explicitly when building
descriptors to catch mismatches early:

<div class="lang-tabs">

```rust
use hurray_core::layout::{CsrLayout, LayoutDescriptor};
use hurray_core::Shape;

let csr = LayoutDescriptor::Csr(CsrLayout::new(5));

// Rank-2: valid.
assert!(csr.validate_against_shape(&Shape::new(vec![4, 5]).unwrap()).is_ok());

// Rank-3: rejected — CSR is only defined for rank-2 tensors.
assert!(csr.validate_against_shape(&Shape::new(vec![2, 3, 4]).unwrap()).is_err());
```

```python
import hurray

csr = hurray.CsrLayout(nnz=5)

# Rank-2: valid.
csr.validate_against_shape([4, 5])

# Rank-3: rejected - CSR is only defined for rank-2 tensors.
try:
    csr.validate_against_shape([2, 3, 4])
    raise AssertionError("rank-3 should be refused")
except hurray.InvalidDescriptorError:
    pass
```

</div>

`hurray.Tensor` runs this for you at construction. Call it directly when you are
choosing a layout for a shape you have not built a tensor for yet.

## Private extension layouts

For hardware-specific panel/pack formats agreed out of band:

<div class="lang-tabs">

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

```python
import hurray

private = hurray.PrivateExtensionLayout(
    0xF0,                       # tag: must be 0xF0-0xFE
    0xDEAD_BEEF_0000_0001,      # implementation-defined layout ID
    b"\x01\x00\x04",            # opaque metadata
)
# buffer_count is None: the format doesn't know how many buffers this needs.
assert private.buffer_count is None
```

</div>
