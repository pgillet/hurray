# Memory Layout -- Hurray Format Specification

> **Status:** Draft

## Scope

This section defines how tensor elements are arranged in memory. It specifies the
addressing model that maps a tensor's logical index space to byte positions within a
data buffer. Hurray supports a range of memory layouts — from simple contiguous
arrangements to tiled, space-filling-curve, sparse, and composite (virtual) layouts —
to accommodate the diverse access patterns required by modern AI/ML inference pipelines.

> **Note (non-normative):** The unifying mathematical concept behind all Hurray dense
> layouts is the **subpaving**: a finite collection of non-overlapping boxes
> (rectangular regions) that together tile a tensor's index space. A contiguous
> row-major tensor is a trivial subpaving (one box). A tiled tensor is a regular
> subpaving. See [Subpaving (Wikipedia)](https://en.wikipedia.org/wiki/Subpaving).

## Normative Requirements

> This section uses RFC 2119 key words: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT,
> SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

---

## Layout Taxonomy

Every tensor descriptor MUST include a **layout tag** (`uint8`) that identifies the
memory layout of the tensor's data. The tag space is partitioned as follows:

| Range | Allocation |
|-------|------------|
| `0x00` | Reserved (invalid) |
| `0x01` – `0x3F` | Core named layouts (Tier 1) |
| `0x40` – `0x7F` | Extended named layouts (Tier 2) |
| `0x80` – `0xEF` | Reserved for future specification versions |
| `0xF0` – `0xFE` | Implementation-private extension layouts |
| `0xFF` | Reserved (invalid) |

A conforming reader MUST reject a tensor descriptor containing a layout tag of `0x00`
or `0xFF`.

A conforming reader MUST reject a tensor descriptor containing a layout tag it does
not recognise, unless operating in permissive mode. In permissive mode, the reader
MAY accept the descriptor but MUST NOT dereference or interpret the tensor data buffer.

### Named Layout Tags

| Layout | Tag | Tier | Type | Spec |
|--------|-----|------|------|------|
| Row-major (C order) | `0x01` | 1 | Dense | [layouts/row-major.md](layouts/row-major.md) |
| Column-major (Fortran order) | `0x02` | 1 | Dense | [layouts/column-major.md](layouts/column-major.md) |
| Strided | `0x03` | 1 | Dense | [layouts/strided.md](layouts/strided.md) |
| Tiled / Blocked | `0x04` | 1 | Dense | [layouts/tiled.md](layouts/tiled.md) |
| Morton (Z-order) | `0x05` | 1 | Dense | [layouts/morton.md](layouts/morton.md) |
| COO (Coordinate) | `0x06` | 1 | Sparse | [layouts/coo.md](layouts/coo.md) |
| CSR (Compressed Sparse Row) | `0x07` | 1 | Sparse | [layouts/csr.md](layouts/csr.md) |
| CSC (Compressed Sparse Column) | `0x08` | 1 | Sparse | [layouts/csc.md](layouts/csc.md) |
| CSF (Compressed Sparse Fiber) | `0x09` | 1 | Sparse | [layouts/csf.md](layouts/csf.md) |
| Block-paged | `0x0A` | 1 | Indirect | [layouts/block-paged.md](layouts/block-paged.md) |
| Composite / Virtual | `0x0B` | 1 | Virtual | [layouts/composite.md](layouts/composite.md) |
| Hilbert curve | `0x40` | 2 | Dense | [layouts/hilbert.md](layouts/hilbert.md) |

The **Type** column classifies each layout's addressing model:

- **Dense** — every logical element exists and maps to a physical position by an affine
  stride formula.
- **Sparse** — only stored (non-zero) elements are materialised; unstored coordinates
  are implicitly zero.
- **Indirect** — every logical element exists (no implicit zeros), but the mapping from
  a logical index to a physical position is non-affine and resolved through an index
  structure (e.g. a block table) rather than an affine stride formula.
- **Virtual** — the descriptor owns no data buffers; it presents a logical view assembled
  from a set of member tensors (the head of a composite; see `layouts/composite.md`).

> **Note (non-normative):** Rows marked "(reserved — planned)" name a tag that is
> earmarked for a layout whose spec section does not yet exist. A reader treats such a tag
> exactly as it treats any unrecognised tag: it MUST reject a descriptor bearing that tag
> unless operating in permissive mode, in which case it MUST NOT dereference the tensor
> data buffer (see the unrecognised-tag rule above). The reservation only records intent
> so the tag is not reassigned before the layout is specified.

Writers choose the layout. Hurray imposes no requirement on which layout a writer
selects; any layout from the table above (or from the extension range, by prior
agreement) is valid.

---

## Common Fields

All layouts share the following fields in the tensor descriptor. These fields are
defined once here; individual layout files specify additional layout-specific fields.

### Rank and Shape

- **`rank`** (`uint32`): the number of dimensions. `0` denotes a scalar tensor.
- **`shape`** (`uint64[rank]`): the size of each dimension. Each value MUST be
  greater than or equal to 0. A size of 0 indicates an empty tensor.

> **Note (non-normative):** Zero-size dimensions are valid for placeholder tensors
> or empty batches. A tensor with shape `[3, 0, 5]` has zero total elements.

The value `0xFFFFFFFFFFFFFFFF` (`UINT64_MAX`) is the **dynamic dimension sentinel**:
the dimension's size is not statically known and MUST be resolved before use.

### byte_offset

- **`byte_offset`** (`uint64`): offset in bytes from the start of buffer 0 to the
  element at logical index `[0, 0, ..., 0]`. MUST be ≤ the buffer's byte size.

For sub-byte types (`bool`, `int4`, `uint4`, `int2`, `uint2`), `byte_offset` MUST
point to a byte boundary.

For **sparse layouts** (COO, CSR, CSC, CSF, and future sparse tags) and **indirect
layouts** (block-paged, and future indirect tags), the concept of a "first element at
a fixed offset" does not apply: the first logical element is located through an index
structure, not at a fixed offset. For these tensors, `byte_offset` MUST be set to
`0x0000000000000000`.

For the **virtual layout** (composite head, tag `0x0B`), there is no data buffer at all;
`byte_offset` MUST be set to `0x0000000000000000`. See `layouts/composite.md`.

---

## Element Address Computation

### Whole-Byte Types

For element types with bit width ≥ 8, the byte address of the element at linear
offset `offset` (as computed by the layout-specific addressing formula) is:

```
byte_address = byte_offset + offset * (bit_width / 8)
```

### Sub-Byte Types

For sub-byte types (`bool`, `int4`, `uint4`, `int2`, `uint2`), strides are expressed
in **logical elements**. Given a linear element offset `offset`:

- **Packing factor** `P`: elements per byte (8 for `bool`, 2 for 4-bit, 4 for 2-bit).
- **Bit width** `B`: bits per element (1, 4, or 2).
- **Byte index**: `byte_offset + floor(offset / P)`.
- **Bit position within byte**: `(offset mod P) * B` (counting from LSB).

For strided layouts with sub-byte types, strides are in logical elements; the
implementation MUST compute the linear element offset using the strides, then apply
the packing formula above.

> **Note (non-normative):** Sub-byte types are almost always used with contiguous
> layouts. Strided sub-byte tensors are supported for completeness but require
> bit-level manipulation per element. Writers SHOULD prefer contiguous layouts for
> sub-byte data.

> **Note (non-normative):** The 6-bit types (`float6_e2m3`, `float6_e3m2`) pack 4
> elements per 3 bytes (see `element-types.md` § Buffer Size Calculation). However,
> because their bit width (6) does not divide evenly into 8 bits, the standard
> sub-byte bit-addressing model (where bit position within a byte is well-defined)
> does not apply cleanly. Elements are always addressed at the group level (3 bytes
> per group of 4 elements); individual element extraction within a group is defined
> by the bit layout in `element-types.md` § Encoding but is not generalised here.

> **[OQ-1]:** Should a normative group-based addressing formula for 6-bit types be
> defined in this section, or is the `element-types.md` encoding sufficient for
> implementors? Until resolved, conforming implementations MAY treat 6-bit types
> as requiring group-level buffer access rather than single-element byte addressing.

---

## Alignment

- The data buffer MUST be aligned to at least **64 bytes** regardless of layout or
  element type.
- For tiled layouts, each tile's data SHOULD start at a naturally aligned boundary.
  Writers SHOULD insert inter-tile padding when needed; tile stride values MUST
  account for any padding.
- For sub-byte types, `byte_offset` MUST be byte-aligned.
- Page-aligned buffers (typically 4096 bytes) SHOULD be used when the tensor is shared
  across processes or with GPU devices. See `buffer-protocol.md`.

---

## Buffer Table

Every tensor descriptor contains a **buffer table**: an ordered list of buffer handles.
The buffer table is encoded as a `uint8` count followed by that many buffer handle
entries, as defined in `metadata.md`.

For **dense layouts** (tags `0x01`–`0x05`, `0x40`), the buffer table MUST contain at least **one** entry. Non-quantized dense tensors MUST have exactly `buffer_count = 0x01`. Quantized dense tensors MUST have `buffer_count = 0x01` plus the number of quantization-parameter buffers required by the active scheme (see `quantization.md` § Buffer Table Placement Rules).

For **sparse layouts** (tags `0x06`, `0x07`, `0x08`, `0x09`, and future sparse tags),
the buffer table MUST contain the number of entries specified by that layout's
individual spec file. Each buffer holds a distinct component array (values, indices,
pointers). Most sparse layouts have a fixed buffer count; CSF (`0x09`) is the
exception, with a rank-dependent count of `2 × rank + 1` (see `layouts/csf.md`).

For **indirect layouts** (tag `0x0A`, and future indirect tags), the buffer table
MUST contain at least **three** entries (`buffer_count >= 3`): a values buffer plus the
index/pointer buffers that resolve the logical-to-physical mapping. For block-paged
(`0x0A`) these are buffer 0 = `page_pool`, buffer 1 = `block_table`, and buffer 2 =
`seq_ptr`. When the tensor is quantized, the quantization-parameter buffers follow at
indices 3 and up, per `quantization.md` § Buffer Table Placement Rules. See that
layout's individual spec file for the exact buffer table.

For the **virtual layout** (composite head, tag `0x0B`), the buffer table MUST be empty
(`buffer_count = 0x00`): the head owns no data and supplies its logical view through its
member tensors. See `layouts/composite.md`.

---

## Splittability and Sharding

A tensor MAY be described as a **shard**: a rectangular sub-region of a larger logical
tensor. A shard carries a **shard descriptor** in the tensor descriptor (see
`metadata.md` § Shard Section) indicating its position within the parent index space.

**Shard descriptor fields:**

- **`parent_shape`** (`uint64[rank]`): shape of the logical parent tensor.
- **`shard_offset`** (`uint64[rank]`): starting index of this shard within the parent
  along each dimension.

The constraint `shard_offset[k] + shape[k] <= parent_shape[k]` MUST hold for every
dimension `k`.

> **Note (non-normative):** Sharding always produces rectangular sub-regions
> (hyperrectangles), which covers all practical partitioning patterns in ML inference
> (batch splitting, tensor parallelism, pipeline stages). Protocol-level splitting of
> tensor data across stream frames is a separate concern covered in `interchange.md`.

---

## Extension Layouts

Layout tags in the range `0xF0`–`0xFE` are reserved for implementation-private
extension layouts. Tensors using these tags MUST NOT be exchanged between independent
implementations unless both parties have agreed on the layout semantics out of band.

An extension layout descriptor MUST include:

- **`extension_layout_id`** (`uint64`): implementation-defined unique identifier.
- **`extension_data`** (`byte sequence`): opaque layout-specific metadata, preceded
  by a `uint32` byte-length field.

> **Note (non-normative):** Extension layouts are the mechanism for hardware-specific
> panel/pack formats used between BLAS pipeline stages. A client advertises an
> extension layout tag with opaque hardware metadata during capability negotiation
> (see `interchange.md`); the server transcodes and packs accordingly.

---

## Custom Layouts

Any layout expressible as a composition of the named primitives — strides, tiling,
space-filling curves — using a composite partition (`layouts/composite.md`) or recursive
tiling is representable without the extension mechanism. Truly opaque custom layouts MUST
use extension tags (`0xF0`–`0xFE`) with an out-of-band semantic agreement.

---

## Interaction with Other Sections

- **Element Types (`element-types.md`)**: defines bit widths, packing rules, and
  natural alignment for each element type.
- **Quantization (`quantization.md`)**: block-quantized layouts interact with the
  memory layout tiling structure.
- **Buffer Protocol (`buffer-protocol.md`)**: defines buffer ownership, alignment
  requirements, and device memory semantics.
- **Metadata (`metadata.md`)**: defines the binary encoding of all layout-specific
  fields in the tensor descriptor.
- **Interchange (`interchange.md`)**: defines how tensor descriptors and data buffers
  are transmitted. Protocol-level data splitting is distinct from logical sharding.
