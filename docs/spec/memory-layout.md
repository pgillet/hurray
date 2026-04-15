# Memory Layout -- Hurray Format Specification

> **Status:** Draft

## Scope

This section defines how tensor elements are arranged in memory. It specifies the
addressing model that maps a tensor's logical index space to byte positions within a
data buffer. Hurray supports a range of memory layouts -- from simple contiguous
arrangements to tiled, space-filling-curve, and general subpaving layouts -- to
accommodate the diverse access patterns required by modern AI/ML inference pipelines.

> **Note (non-normative):** The unifying mathematical concept behind all Hurray layouts
> is the **subpaving**: a finite collection of non-overlapping boxes (rectangular
> regions) that together tile a tensor's index space. A contiguous row-major tensor
> is a trivial subpaving (one box covering the entire space). A tiled tensor is a
> regular subpaving. A tensor with mixed dense/sparse regions is an irregular subpaving.
> See [Subpaving (Wikipedia)](https://en.wikipedia.org/wiki/Subpaving) for the
> mathematical background. This concept motivates the layout taxonomy but does not
> introduce additional normative requirements beyond those stated for each named layout.

## Normative Requirements

> This section uses RFC 2119 key words: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT,
> SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

## Layout Taxonomy

Every tensor descriptor MUST include a **layout tag** that identifies the memory
layout of the tensor's data. The layout tag is encoded as a `uint8` value. The tag
space is partitioned as follows:

| Range | Allocation |
|-------|------------|
| `0x00` | Reserved (invalid) |
| `0x01` -- `0x3F` | Core named layouts (Tier 1) |
| `0x40` -- `0x7F` | Extended named layouts (Tier 2) |
| `0x80` -- `0xEF` | Reserved for future specification versions |
| `0xF0` -- `0xFE` | Implementation-private extension layouts |
| `0xFF` | Reserved (invalid) |

A conforming reader MUST reject a tensor descriptor containing a layout tag of `0x00`
or `0xFF`.

A conforming reader MUST reject a tensor descriptor containing a layout tag it does
not recognize, unless operating in permissive mode. In permissive mode, the reader
MAY accept the descriptor but MUST NOT attempt to dereference or interpret the tensor
data buffer.

### Layout Tags

| Layout | Tag | Tier | Section |
|--------|-----|------|---------|
| Row-major (C order) | `0x01` | 1 | [Row-Major](#row-major-c-order) |
| Column-major (Fortran order) | `0x02` | 1 | [Column-Major](#column-major-fortran-order) |
| Strided | `0x03` | 1 | [Strided](#strided) |
| Tiled / Blocked | `0x04` | 1 | [Tiled / Blocked](#tiled--blocked) |
| Morton (Z-order) | `0x05` | 1 | [Morton (Z-Order Curve)](#morton-z-order-curve) |
| Subpaving (general) | `0x06` | 1 | [General Subpaving](#general-subpaving) |
| Hilbert curve | `0x40` | 2 | [Hilbert Curve](#hilbert-curve) |

Writers choose the layout. Hurray imposes no requirement on which layout a writer
selects; any layout from the table above (or from the extension range, by prior
agreement) is valid.

---

## Common Fields

All layouts share the following fields in the tensor descriptor. These fields are
defined here once; individual layout sections specify additional layout-specific fields.

### Rank and Shape

- **`rank`** (`uint32`): the number of dimensions. A rank of 0 denotes a scalar tensor
  (a single element).
- **`shape`** (`uint64[rank]`): the size of each dimension. Each value MUST be greater
  than or equal to 0. A dimension size of 0 indicates an empty tensor (zero elements).

> **Note (non-normative):** Zero-size dimensions are valid and useful for representing
> placeholder tensors or empty batches. A tensor with shape `[3, 0, 5]` has zero total
> elements.

A dimension size equal to `0xFFFFFFFFFFFFFFFF` (`UINT64_MAX`) is the **dynamic
dimension sentinel**. It indicates that the dimension's size is not statically known
and will be resolved at runtime. A reader MUST NOT compute buffer sizes, strides, or
element counts using a dynamic dimension sentinel without first resolving it.

### Byte Offset

- **`byte_offset`** (`uint64`): the offset in bytes from the start of the data buffer
  to the element at logical index `[0, 0, ..., 0]`. This value MUST be less than or
  equal to the buffer's total byte size. A `byte_offset` of 0 means the first element
  begins at the start of the buffer.

For sub-byte types (`bool`, `int4`, `uint4`, `int2`, `uint2`), `byte_offset` MUST
point to a byte boundary. The first element of the tensor starts at bit 0 of the byte
at `byte_offset`. See [Sub-Byte Types and Strides](#sub-byte-types-and-strides) for
further details.

---

## Tier 1 -- Core Named Layouts

All conforming implementations MUST support reading tensor descriptors for every Tier 1
layout. An implementation MUST correctly interpret the descriptor metadata (shape,
strides, buffer bounds) for any Tier 1 layout. Whether the implementation can perform
computation on data in every layout is outside the scope of this specification.

### Row-Major (C Order)

**Layout tag:** `0x01`

In row-major layout, elements are stored with the **last dimension varying fastest**.
The strides are implicit and MUST NOT be present in the descriptor for this layout tag.

The implicit strides are computed as:

```
strides[rank - 1] = 1
strides[i] = shape[i + 1] * strides[i + 1]    for i = rank - 2, ..., 0
```

All strides are in **logical elements**.

**Element address:** the linear element offset of element `[i_0, i_1, ..., i_{r-1}]`
in a row-major tensor of rank `r` is:

```
offset = sum(i_k * strides[k] for k = 0, ..., r - 1)
```

The byte address is computed from the element offset using the rules in
[Element Address Computation](#element-address-computation).

**Buffer size:** for a contiguous row-major tensor, the minimum buffer size is
`num_elements * element_byte_width` for whole-byte types, or
`ceil(num_elements / packing_factor)` for sub-byte types, where `num_elements` is
the product of all dimension sizes and `packing_factor` is defined in
`element-types.md`.

**Example:** A rank-2 tensor with shape `[3, 4]` in row-major layout has implicit
strides `[4, 1]`. Element `[1, 2]` is at linear offset `1 * 4 + 2 * 1 = 6`.

### Column-Major (Fortran Order)

**Layout tag:** `0x02`

In column-major layout, elements are stored with the **first dimension varying
fastest**. The strides are implicit and MUST NOT be present in the descriptor for this
layout tag.

The implicit strides are computed as:

```
strides[0] = 1
strides[i] = shape[i - 1] * strides[i - 1]    for i = 1, ..., rank - 1
```

All strides are in **logical elements**.

**Element address:** computed identically to row-major using the column-major strides.

**Example:** A rank-2 tensor with shape `[3, 4]` in column-major layout has implicit
strides `[1, 3]`. Element `[1, 2]` is at linear offset `1 * 1 + 2 * 3 = 7`.

### Strided

**Layout tag:** `0x03`

The strided layout is the most general of the simple (non-tiled, non-curve) layouts.
It generalizes both row-major and column-major by allowing an arbitrary stride value
per dimension.

**Additional descriptor fields:**

- **`strides`** (`int64[rank]`): the stride of each dimension, in **logical elements**.
  This field MUST be present when the layout tag is `0x03`.

**Stride semantics:**

- A **positive** stride advances forward through the buffer.
- A **negative** stride advances backward through the buffer. A negative stride on
  dimension `k` reverses that dimension: logical index 0 maps to the highest physical
  offset along that axis.
- A stride of **zero** on dimension `k` means that all indices along dimension `k` map
  to the same physical data. This is a **broadcast** (virtual) dimension. The data is
  not physically replicated.

Negative and zero strides are valid. A conforming implementation MUST support
negative strides and zero strides in descriptors with the strided layout tag.

**Element address:** the linear element offset of element `[i_0, i_1, ..., i_{r-1}]`
is:

```
offset = sum(i_k * strides[k] for k = 0, ..., r - 1)
```

When negative strides are present, the offset may be negative relative to the base
address at `byte_offset`. The `byte_offset` field MUST be set such that the element
at `[0, 0, ..., 0]` is addressable within the buffer. Specifically, the physical
address of any valid element MUST lie within the buffer's bounds.

**Contiguity:** a strided tensor is **dense** (contiguous with no gaps) if and only if
the absolute values of its strides form a permutation of the row-major strides for the
same shape (i.e., the strides describe some permutation of dimensions, possibly with
reversals). Implementations SHOULD NOT assume that a strided tensor is dense without
verifying this condition.

**Buffer size for strided layouts:** the minimum buffer size in bytes for a strided
tensor MUST be large enough to contain every addressable element. Formally, the
required range in elements is:

```
max_offset = sum(max(0, strides[k] * (shape[k] - 1)) for all k)
min_offset = sum(min(0, strides[k] * (shape[k] - 1)) for all k)
range_elements = max_offset - min_offset + 1
```

The minimum buffer size in bytes depends on the element type; see
[Element Address Computation](#element-address-computation).

**Example:** A rank-2 tensor with shape `[3, 4]` and strides `[-4, 1]` represents a
row-major matrix with the row order reversed. `byte_offset` would point to what would
be element `[2, 0]` in a non-reversed matrix. Element `[0, 0]` is at offset
`0 * (-4) + 0 * 1 = 0` relative to `byte_offset`, and element `[2, 3]` is at offset
`2 * (-4) + 3 * 1 = -5`. The buffer MUST be large enough to cover the full range from
offset -8 to offset 0 (inclusive), i.e., 9 elements.

### Tiled / Blocked

**Layout tag:** `0x04`

A tiled layout partitions the tensor's index space into uniform rectangular tiles
(blocks). This is a **regular subpaving**: all tiles have the same shape, and they tile
the index space without overlap or gap (with possible padding at the boundaries).

**Additional descriptor fields:**

- **`tile_shape`** (`uint64[rank]`): the size of each tile along each dimension.
  Every value MUST be greater than 0. This field MUST have the same length as `rank`.
- **`outer_layout`** (`uint8`): the layout tag describing how tiles are arranged in
  memory. MUST be one of `0x01` (row-major), `0x02` (column-major), or `0x03`
  (strided). The outer layout operates over a **tile grid** whose shape is
  `ceil(shape[k] / tile_shape[k])` for each dimension `k`.
- **`inner_layout`** (`uint8`): the layout tag describing how elements are arranged
  within each tile. MUST be one of `0x01` (row-major), `0x02` (column-major), or
  `0x03` (strided).
- **`outer_strides`** (`int64[rank]`): present if and only if `outer_layout` is
  `0x03` (strided). Strides for tile grid addressing, in units of **tiles** (not
  elements).
- **`inner_strides`** (`int64[rank]`): present if and only if `inner_layout` is
  `0x03` (strided). Strides for element addressing within a tile, in **logical
  elements**.

**Recursive tiling:** the tiled layout MAY be nested: the `inner_layout` or
`outer_layout` MAY itself be `0x04` (tiled), enabling multi-level blocking (e.g.,
L1-tile within L2-tile). When nested, the inner tiled layout carries its own
`tile_shape`, `outer_layout`, `inner_layout`, and associated strides. The nesting
depth is not limited by this specification, but implementations MAY impose a
maximum recursion depth and MUST reject descriptors that exceed it.

> **Note (non-normative):** Recursive tiling is useful for hierarchical blocking in
> GEMM kernels (e.g., 128x128 L2 tiles subdivided into 32x32 L1 tiles, each stored
> in row-major order). It is also the natural expression of cache-oblivious recursive
> blocking.

**Boundary padding:** when a dimension size is not evenly divisible by the
corresponding tile size, partial tiles exist at the boundary. The data buffer MUST
contain storage for full tiles, including the padding region of partial tiles. The
values of padding elements are undefined; readers MUST NOT access elements whose
logical index exceeds the tensor's shape.

Formally, the number of tiles along dimension `k` is:

```
num_tiles[k] = ceil(shape[k] / tile_shape[k])
```

The total buffer size is:

```
total_tile_elements = product(tile_shape[k] for all k)
total_tiles = product(num_tiles[k] for all k)
buffer_elements = total_tiles * total_tile_elements
```

Buffer size in bytes is computed from `buffer_elements` using the element type's
byte width (or packing factor for sub-byte types).

**Element address computation:** to locate element `[i_0, i_1, ..., i_{r-1}]`:

1. Compute the tile index for each dimension: `t_k = floor(i_k / tile_shape[k])`.
2. Compute the intra-tile offset for each dimension: `e_k = i_k mod tile_shape[k]`.
3. Compute the linear tile number using the outer layout strides applied to
   `[t_0, t_1, ..., t_{r-1}]`. Multiply by `total_tile_elements` to get the
   element offset to the start of the tile.
4. Compute the intra-tile linear offset using the inner layout strides applied to
   `[e_0, e_1, ..., e_{r-1}]`.
5. The final linear element offset is the sum of (3) and (4).

**Example:** A rank-2 tensor with shape `[6, 8]`, `tile_shape = [2, 4]`,
`outer_layout = 0x01` (row-major), `inner_layout = 0x01` (row-major).

- Tile grid shape: `[3, 2]`. Outer strides (implicit row-major): `[2, 1]` in tiles.
- Each tile has 8 elements. Inner strides (implicit row-major): `[4, 1]` in elements.
- Element `[3, 5]`: tile `[1, 1]`, intra-tile `[1, 1]`.
  Tile linear index = `1 * 2 + 1 = 3`. Tile start = `3 * 8 = 24`.
  Intra-tile offset = `1 * 4 + 1 = 5`. Final offset = `24 + 5 = 29`.

### Morton (Z-Order Curve)

**Layout tag:** `0x05`

The Morton layout (also known as Z-order curve) stores elements by interleaving the
bits of their dimension indices, producing a linear order with good spatial locality
for multi-dimensional access patterns.

**Additional descriptor fields:**

- **`morton_bits`** (`uint32[rank]`): the number of bits used for each dimension in the
  Morton encoding. This field MUST be present when the layout tag is `0x05`.

**Morton index computation:** for element `[i_0, i_1, ..., i_{r-1}]`, the Morton
code is computed by interleaving the bits of the dimension indices in round-robin
order, starting from the least significant bit of dimension 0:

```
morton_code = 0
for bit_position b = 0, 1, 2, ...:
    for dimension d = 0, 1, ..., rank - 1:
        if b < morton_bits[d]:
            morton_code |= ((i_d >> b) & 1) << (b * rank + d)
```

The element at Morton code `m` is stored at linear offset `m` in the buffer.

**Dimension size constraints:** for each dimension `k`, the dimension size `shape[k]`
MUST satisfy `shape[k] <= 2^morton_bits[k]`. The value `morton_bits[k]` MUST be
greater than 0 for each dimension.

> **Note (non-normative):** For non-power-of-two dimension sizes, `morton_bits[k]`
> SHOULD be set to `ceil(log2(shape[k]))` — the smallest value satisfying the
> constraint — to minimise padding waste. The worst-case padding factor per dimension
> is strictly less than 2×. A "compact Morton" scheme that eliminates padding entirely
> was considered and rejected: it requires non-trivial per-access index computation
> (lookup tables or specialised bit manipulation), breaks branchless bit-interleaving,
> and defeats the spatial-locality advantage of Morton layouts. Writers for whom
> padding waste is unacceptable SHOULD prefer a tiled or row-major layout instead.
> See ADR-005.

**Buffer size:** the buffer MUST hold `2^(sum(morton_bits[k] for all k))` elements.

**Example:** A rank-2 tensor with shape `[4, 4]` and `morton_bits = [2, 2]`. The
Morton code for element `[2, 3]` is computed by interleaving bits:

- `i_0 = 2` = binary `10`, `i_1 = 3` = binary `11`.
- Interleaved (LSB first, dim 0 then dim 1): bit 0 of `i_0` = 0, bit 0 of `i_1` = 1,
  bit 1 of `i_0` = 1, bit 1 of `i_1` = 1. Morton code = `0b1110` = 14.
- Element `[2, 3]` is stored at linear offset 14 in the buffer.

### General Subpaving

**Layout tag:** `0x06`

The general subpaving layout describes an **irregular subpaving**: the tensor's index
space is partitioned into a set of non-overlapping rectangular regions (boxes), each
with its own layout descriptor. This is the most general layout in Hurray. It is
intended for tensors with heterogeneous structure -- for example, mixed dense/sparse
regions, or regions with different tiling strategies.

**Additional descriptor fields:**

- **`region_count`** (`uint32`): the number of regions in the subpaving. MUST be
  greater than 0.
- **`regions`**: an array of `region_count` region descriptors. Each region descriptor
  contains:
  - **`origin`** (`uint64[rank]`): the starting index of the region along each
    dimension (inclusive).
  - **`region_shape`** (`uint64[rank]`): the size of the region along each dimension.
    Every value MUST be greater than 0.
  - **`layout_tag`** (`uint8`): the layout of elements within this region. MUST be
    one of the core layout tags (`0x01` through `0x05`), or another `0x06` for
    recursive subpaving. MUST NOT be `0x00` or `0xFF`.
  - **Layout-specific fields**: as required by the region's `layout_tag` (e.g.,
    `strides` if `layout_tag` is `0x03`, `tile_shape` / `outer_layout` /
    `inner_layout` if `0x04`, etc.).
  - **`buffer_index`** (`uint32`): the index of the data buffer that holds this
    region's data. All regions MAY reference the same buffer or different buffers.
  - **`byte_offset`** (`uint64`): the byte offset within the referenced buffer where
    this region's data begins.

**Coverage constraint:** the union of all regions MUST exactly cover every element in
the tensor's index space. That is, for every valid index `[i_0, i_1, ..., i_{r-1}]`
(where `0 <= i_k < shape[k]` for all `k`), there MUST be exactly one region whose
bounding box contains that index.

**Non-overlap constraint:** regions MUST NOT overlap. Two regions overlap if their
bounding boxes share any index point. Formally, regions `A` and `B` overlap if, for
every dimension `k`:

```
A.origin[k] < B.origin[k] + B.region_shape[k]
AND
B.origin[k] < A.origin[k] + A.region_shape[k]
```

A conforming writer MUST produce a valid subpaving (full coverage, no overlap). A
conforming reader SHOULD validate the coverage and non-overlap constraints and MUST
reject descriptors that violate them, unless operating in permissive mode.

**Element address computation:** to locate element `[i_0, i_1, ..., i_{r-1}]`:

1. Find the region whose bounding box contains the index. A region with origin `o` and
   shape `s` contains index `i` if `o[k] <= i[k] < o[k] + s[k]` for all `k`.
2. Compute the local index within the region: `local[k] = i[k] - origin[k]`.
3. Apply the region's layout addressing to `local` to compute the element offset
   within the region's buffer at the region's `byte_offset`.

**Example:** A rank-2 tensor with shape `[8, 8]` split into four 4x4 quadrants, each
stored in row-major order:

- Region 0: `origin = [0, 0]`, `region_shape = [4, 4]`, `layout_tag = 0x01`.
- Region 1: `origin = [0, 4]`, `region_shape = [4, 4]`, `layout_tag = 0x01`.
- Region 2: `origin = [4, 0]`, `region_shape = [4, 4]`, `layout_tag = 0x01`.
- Region 3: `origin = [4, 4]`, `region_shape = [4, 4]`, `layout_tag = 0x01`.

Each region's data occupies 16 elements in its buffer. Element `[5, 6]` falls in
region 3 (origin `[4, 4]`). Local index: `[1, 2]`. Offset within region: `1 * 4 + 2 = 6`.

> **Note (non-normative):** The general subpaving layout carries more descriptor
> overhead than simpler layouts and requires a region lookup step per element access.
> It is not intended for performance-critical inner loops over dense data. Its
> primary use case is describing tensors with structurally heterogeneous regions
> (e.g., a mixture of dense tiles and sparse blocks, or a tensor assembled from
> independently-produced shards with different inner layouts).

---

## Tier 2 -- Extended Named Layouts

Tier 2 layouts are OPTIONAL. Conforming implementations MAY support any subset of
Tier 2 layouts, including none. Implementations that do not support a given Tier 2
layout MUST reject (or, in permissive mode, skip) descriptors using that layout tag.

### Hilbert Curve

**Layout tag:** `0x40`

The Hilbert curve layout stores elements according to a Hilbert space-filling curve,
which provides better locality than the Morton curve: consecutive Hilbert indices
always correspond to physically adjacent elements (L∞ distance = 1 in index space),
with no large jumps. This property benefits access patterns that traverse
multi-dimensional regions with spatial coherence (e.g. image convolution, volumetric
sampling, point-cloud processing), at the cost of a more expensive index computation
than Morton.

> **Note (non-normative):** The Hilbert curve is particularly effective for 2D and 3D
> spatial tensors (image processing, volumetric inference) where Morton's large jumps
> at quadrant boundaries cause cache misses. For 1D access patterns or cases where
> index-computation cost dominates, row-major or Morton layouts are preferable.

**Additional descriptor fields:**

- **`hilbert_order`** (`uint32`): the order of the Hilbert curve. The tensor's
  dimensions MUST each be equal to `2^hilbert_order`. MUST be greater than 0. This
  field MUST be present when the layout tag is `0x40`.
- **`hilbert_rank`** (`uint32`): the number of dimensions of the Hilbert curve.
  MUST equal the tensor's `rank`. MUST be greater than or equal to 2.

**Buffer size:** the buffer MUST hold exactly `2^(hilbert_rank * hilbert_order)`
elements.

#### Normative Index Mapping

The normative index mapping is the algorithm defined by Skilling (2004) (see
`references.md`). Conforming implementations MUST use this algorithm to map between
Hilbert indices and element coordinates. The algorithm is reproduced below as
normative pseudocode.

Let `r = hilbert_rank` and `p = hilbert_order`. Coordinates are `X[0..r-1]`, each in
`[0, 2^p)`. The Hilbert index `h` is a non-negative integer in `[0, 2^(r*p))`.

**Bit packing convention:** the Hilbert index `h` is packed from the coordinates as
follows. Bit `(b * r + (r - 1 - d))` of `h` (counting from bit 0 at the LSB) holds
bit `b` of coordinate `X[d]`, for `b = 0, ..., p-1` and `d = 0, ..., r-1`.

**Algorithm `CoordsToHilbert(X[0..r-1], r, p)` → `h`:**

```
// Step 1: excess-work transform (modifies X in-place)
M = 1 << (p - 1)
Q = M
while Q > 1:
    P = Q - 1
    for i = 0 to r-1:
        if X[i] & Q:
            X[0] ^= P
        else:
            t = (X[0] ^ X[i]) & P
            X[0] ^= t
            X[i] ^= t
    Q >>= 1

// Step 2: Gray encode with rotation
for i = 1 to r-1:
    X[i] ^= X[i-1]
t = 0
Q = M
while Q > 1:
    if X[r-1] & Q:
        t ^= Q - 1
    Q >>= 1
for i = 0 to r-1:
    X[i] ^= t

// Step 3: pack bits into Hilbert index
h = 0
for b = 0 to p-1:
    for d = 0 to r-1:
        h |= ((X[d] >> b) & 1) << (b * r + (r - 1 - d))
return h
```

**Algorithm `HilbertToCoords(h, r, p)` → `X[0..r-1]`:**

```
// Step 1: unpack Hilbert index into coordinate array
X = [0] * r
for b = 0 to p-1:
    for d = 0 to r-1:
        X[d] |= ((h >> (b * r + (r - 1 - d))) & 1) << b

// Step 2: inverse Gray decode with rotation
t = X[r-1] >> 1
for i = r-1 downto 1:
    X[i] ^= X[i-1]
X[0] ^= t

// Step 3: inverse excess-work transform
Q = 2
while Q != (1 << p):
    P = Q - 1
    for i = r-1 downto 0:
        if X[i] & Q:
            X[0] ^= P
        else:
            t = (X[0] ^ X[i]) & P
            X[0] ^= t
            X[i] ^= t
    Q <<= 1
return X
```

All arithmetic in the algorithms above is integer arithmetic. The `^` operator denotes
bitwise XOR. Array indexing is zero-based. The algorithms MUST produce identical
results on all conforming implementations for identical inputs.

**Example:** rank-2 tensor with shape `[4, 4]` (`r = 2`, `p = 2`,
`hilbert_order = 2`, `hilbert_rank = 2`). Selected index mappings:

| `h` | `X[0]` | `X[1]` | | `h` | `X[0]` | `X[1]` |
|-----|--------|--------|-|-----|--------|--------|
| 0 | 0 | 0 | | 8 | 2 | 2 |
| 1 | 1 | 0 | | 9 | 2 | 3 |
| 2 | 1 | 1 | | 10 | 3 | 3 |
| 3 | 0 | 1 | | 11 | 3 | 2 |
| 4 | 0 | 2 | | 12 | 3 | 1 |
| 5 | 0 | 3 | | 13 | 3 | 0 |
| 6 | 1 | 3 | | 14 | 2 | 0 |
| 7 | 1 | 2 | | 15 | 2 | 1 |

Consecutive entries differ by exactly 1 in exactly one coordinate (the Hamiltonian
path property of Hilbert curves). Implementations SHOULD validate this property
against the table above as a conformance check.

---

## Extension Layouts

Layout tags in the range `0xF0` -- `0xFE` are reserved for implementation-private
extension layouts. Tensors using extension layout tags MUST NOT be exchanged between
independent implementations unless both parties have agreed on the layout semantics
out of band.

An extension layout descriptor MUST include at minimum:

- **`extension_layout_id`** (`uint64`): a unique identifier for the extension layout,
  chosen by the implementation. No central registry is defined.
- **`extension_data`** (`byte sequence`): opaque layout-specific metadata. The length
  MUST be encoded as a `uint32` preceding the byte sequence.

> **Note (non-normative):** Candidate extension layouts include:
> - **Panel/Pack format** -- hardware-specific BLAS/BLIS computational formats used
>   between pipeline stages. These are not named layouts because their parameters
>   (panel width, register block dimensions, cache line size, SIMD width) are
>   implementation- and hardware-specific, and no portable normative definition is
>   possible. They are explicitly supported via the extension layout mechanism combined
>   with content negotiation in the network transport protocol (see `interchange.md`):
>   a client advertises an extension layout tag with opaque hardware metadata, and
>   the server transcodes and packs accordingly. The packed buffer is consumed directly
>   by the BLAS kernel on the client side and is never forwarded or stored.
> - **NVIDIA Tensor Core fragment layouts** -- hardware-internal layouts that are
>   out of scope for a portable interchange format.

---

## Element Address Computation

This section defines the general procedure for converting a logical element offset
(as computed by the layout-specific addressing rules above) into a byte address within
the data buffer.

### Whole-Byte Types

For element types with bit width >= 8, the byte address of element at linear offset
`offset` is:

```
byte_address = byte_offset + offset * (bit_width / 8)
```

where `bit_width` is the element type's bit width as defined in `element-types.md`,
and `byte_offset` is the tensor descriptor's byte offset field.

### Sub-Byte Types and Strides

For sub-byte types (`bool`, `int4`, `uint4`, `int2`, `uint2`), strides are expressed
in **logical elements**, consistent with all other layouts. The mapping from a logical
element offset to a bit position within the buffer follows the packing rules defined
in `element-types.md`.

Given a linear element offset `offset` (computed from the layout's addressing formula):

- **Packing factor** `P`: the number of elements packed per byte (8 for `bool`,
  2 for 4-bit types, 4 for 2-bit types).
- **Bit width** `B`: the number of bits per element (1, 4, or 2 respectively).
- **Byte index**: `byte_offset + floor(offset / P)`.
- **Bit position within byte**: `(offset mod P) * B`, counting from the least
  significant bit.

For strided layouts with sub-byte types, the stride values are in logical elements.
The implementation MUST first compute the linear element offset using the strides, then
apply the packing formula above. This means that non-contiguous strides on sub-byte
types are valid but the resulting physical access pattern may not pack elements
efficiently.

> **Note (non-normative):** In practice, sub-byte types are almost always used with
> contiguous (row-major or column-major) layouts. Strided sub-byte tensors are
> supported for completeness -- for example, to represent a slice of a packed int4
> tensor -- but writers SHOULD prefer contiguous layouts for sub-byte data when
> possible, as strided sub-byte access requires bit-level manipulation per element.

---

## Alignment

Buffer alignment requirements are defined in `buffer-protocol.md`. This section states
the layout-level constraints that interact with alignment.

- The data buffer referenced by a tensor descriptor MUST be aligned to at least **64
  bytes**. This requirement applies regardless of layout or element type.
- For tiled layouts, each tile's data SHOULD start at a naturally aligned boundary
  (i.e., the byte offset of the first element of each tile SHOULD be a multiple of the
  element type's natural alignment as defined in `element-types.md`). Writers SHOULD
  insert padding between tiles if necessary to achieve this. When such padding is
  present, the tile stride values MUST account for it.
- For sub-byte types, the byte offset of the tensor's first element MUST be
  byte-aligned (the `byte_offset` field is in bytes and is always integral).
- Page-aligned buffers (typically 4096 bytes) SHOULD be used when the tensor is shared
  across processes or with GPU devices. See `buffer-protocol.md` for details.

---

## Splittability and Sharding

A tensor MAY be described as a **shard**: a rectangular sub-region of a larger logical
tensor. A shard is a regular Hurray tensor (with its own shape, layout, and buffer)
that additionally carries a **shard descriptor** indicating its position within the
parent tensor's index space.

**Shard descriptor fields:**

- **`parent_shape`** (`uint64[rank]`): the shape of the logical parent tensor. MUST
  have the same rank as the shard.
- **`shard_offset`** (`uint64[rank]`): the starting index of this shard within the
  parent tensor, along each dimension.

The shard's `shape` (from the tensor descriptor) defines the extent of the shard. The
following constraint MUST hold for every dimension `k`:

```
shard_offset[k] + shape[k] <= parent_shape[k]
```

A shard descriptor is OPTIONAL. Tensors without a shard descriptor are standalone.

> **Note (non-normative):** The shard descriptor intentionally uses the simplest
> possible representation: an axis-aligned box (hyperrectangle) defined by
> `shard_offset` and `shape`. This covers all practical sharding patterns in ML
> inference (batch splitting, tensor parallelism, pipeline stages), which always
> produce rectangular sub-regions. A more general subpaving region descriptor was
> considered but rejected: it conflates memory layout (how elements are arranged)
> with logical partitioning (which sub-region of the parent this shard represents),
> and non-rectangular shards have no identified use case in the target domain.
> See ADR-004.

> **Note (non-normative):** Sharding describes how a logical tensor is divided into
> independently-stored pieces. Protocol-level splitting (splitting a tensor's data
> across multiple stream frames for transmission) is a separate concern and is covered
> in `interchange.md`. A shard may itself be split across multiple stream frames.

---

## Sparse Layouts

### Buffer Table

Every tensor descriptor contains a **buffer table**: an ordered list of buffer handles,
each with its own byte size and alignment. The buffer table is encoded as a `uint8`
count field followed by that many buffer handle entries, as defined in
`buffer-protocol.md`.

For dense tensors (all layout tags `0x01`–`0x06` and `0x40`), the buffer table MUST
contain exactly one entry (count = `0x01`). All layout-specific `buffer_index` fields
(e.g., in the general subpaving layout) index into this table.

For sparse layout tags (defined below), the buffer table MUST contain the number of
entries specified by the layout. Each buffer in the table holds a distinct component
array (values, indices, pointers) whose element type and length are defined by the
sparse layout tag. Buffer entries within the table MUST NOT overlap in memory.

> **Note (non-normative):** Requiring a `uint8` count of `0x01` for dense tensors
> costs one byte on the wire but gives every descriptor a uniform structure. Decoders
> can always read the count first and allocate the right number of buffer handles
> without special-casing dense vs. sparse. The general subpaving layout already uses
> `buffer_index` per region; the buffer table formalises what those indices refer to.

### Sparse Format Candidates

The following sparse storage formats are candidates for future specification as named
layout tags. Each requires the buffer count and component semantics listed:

| Format | Layout tag | Buffers | Buffer 0 | Buffer 1 | Buffer 2 |
|--------|-----------|---------|----------|----------|----------|
| COO (Coordinate) | TBD | 2 | `values` (tensor element type, `nnz` elements) | `indices` (`uint64`, `nnz × rank` elements) | — |
| CSR (Compressed Sparse Row) | TBD | 3 | `values` (tensor element type, `nnz` elements) | `col_indices` (`uint64`, `nnz` elements) | `row_ptr` (`uint64`, `nrows + 1` elements) |
| CSC (Compressed Sparse Column) | TBD | 3 | `values` (tensor element type, `nnz` elements) | `row_indices` (`uint64`, `nnz` elements) | `col_ptr` (`uint64`, `ncols + 1` elements) |
| BSR (Block Sparse Row) | TBD | 3 | `values` (tensor element type, `nnz_blocks × block_rows × block_cols` elements) | `col_indices` (`uint64`, `nnz_blocks` elements) | `row_ptr` (`uint64`, `nblock_rows + 1` elements) |
| ELLPACK | TBD | 2 | `values` (tensor element type, `nrows × max_nnz_per_row` elements) | `col_indices` (`uint64`, `nrows × max_nnz_per_row` elements) | — |

Layout tags and full normative definitions for these formats will be assigned in a
future revision of this specification.

> **Note (non-normative):** The general subpaving layout (`0x06`) can describe certain
> sparse-like structures (e.g., a tensor where only some rectangular regions contain
> data) without requiring a separate sparse format. However, true sparse formats like
> CSR provide significantly more compact storage for tensors with many scattered
> zero elements.

---

## Custom Layouts

Any layout expressible as a composition of the named primitives (strides, tiling, curve
type) using the general subpaving layout or recursive tiling is representable within
Hurray without the extension mechanism. For example, a tensor with row-major tiles
arranged in Morton order could be expressed as a general subpaving where the region
ordering follows a Z-curve, or as a tiled layout with a Morton outer layout (if
supported by a future extension of the tiling descriptor).

Truly opaque custom layouts -- those not expressible through any combination of named
layouts -- MUST use the extension mechanism (layout tags `0xF0` -- `0xFE`) with an
out-of-band agreement between producer and consumer on the layout semantics.

---

## Interaction with Other Sections

- **Element Types (`element-types.md`)**: defines bit widths, packing rules, and
  natural alignment for each element type. Memory layout addressing uses these
  properties to convert logical element offsets to byte addresses.
- **Quantization (`quantization.md`)**: quantized tensors use a storage element type
  and may use block-quantized layouts where the block structure interacts with the
  memory layout's tiling.
- **Buffer Protocol (`buffer-protocol.md`)**: defines buffer ownership, alignment
  requirements, device memory semantics, and the release callback mechanism. Memory
  layout descriptors reference buffers described by the buffer protocol.
- **Metadata (`metadata.md`)**: defines the binary encoding of the tensor descriptor,
  including all layout-specific fields defined in this section.
- **Interchange (`interchange.md`)**: defines how tensor descriptors and data buffers
  are transmitted between processes. Protocol-level splitting of tensor data across
  stream frames is defined there, distinct from logical sharding defined here.
