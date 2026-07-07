# ADR-026: General Subpaving as Nested Region Descriptors with Per-Region Buffer Sub-Tables

## Status

Superseded by ADR-027 (Composite Tensors — Head + Members + Composition Rule)

Superseded while still in Draft; the "Supersedes: ADR-015" intent below therefore never took
effect. ADR-015 remains the record of the currently implemented inline region encoding until
ADR-027's implementation lands.

Supersedes: ADR-015 (Subpaving Region Inline Layout Encoding)

## Context

A spec-checker audit of the General Subpaving layout (`docs/spec/layouts/subpaving.md`,
tag `0x06`) surfaced five design-level findings (F-1, F-2, F-4, F-5, F-10). Their common
root cause: subpaving is classified Dense and inherits the dense-layout rule
`buffer_count == 1`, yet each `RegionDescriptor` (per ADR-015) carries a single
`buffer_index` + `region_byte_offset` and its `region_layout_payload` carries only the
inner layout's *scalar* descriptor fields — never buffer references. Sparse
(`0x07`–`0x0A`) and indirect (`0x0B`) inner layouts bind their component arrays (values,
indices, pointers, page pool, block table) **positionally** to a buffer table with a
mandated exact count, so a sparse or indirect inner region is syntactically encodable but
semantically uninterpretable (F-1). The per-region `buffer_index` also contradicts the
`buffer_count == 1` rule (F-2); private-extension inner tags have undefined buffer needs
(F-4); per-region quantization is undefined (F-5); and the tensor-level `byte_offset` is
unused by subpaving addressing (F-10).

The project's array-database vision treats **heterogeneous chunked tensors as
first-class**: one very large logical tensor whose regions differ in structure —
dense tiles beside sparse blocks beside paged blocks — is a target use case, not an edge
case. A prior draft of this ADR proposed a dense-only inner-layout whitelist (deferring
sparse-in-subpaving); that direction is **rejected**. This ADR adopts full support for
sparse and indirect inner regions in v1.0 via a **nested descriptor per region, each
carrying its own buffer sub-table**.

Constraints in force: streamability (descriptors precede data; self-delimiting; no
back-references; no end-of-file index — `README.md`, `interchange.md`); the flat
buffer-table model in which handle *properties* live in the descriptor and buffer
*locations* are supplied positionally by the transport (file-format data region walked in
table order at `data_buffer_alignment`; streaming `TENSOR_DATA` frames; in-process C ABI);
the `uint8` (max 255) buffer-count wire ceiling; the 64-byte minimum per-buffer alignment
and the tensor-wide device-colocation rule (`buffer-protocol.md`); the ADR-017/019
extensibility and evolvability contracts.

## Decision

The general subpaving layout is redefined as a **container of nested region descriptors**.
Each region carries a self-delimiting descriptor body that includes the region's own
layout-specific fields, its own **buffer sub-table**, and an optional per-region
quantization section. Any layout tag — dense, sparse, indirect, recursive subpaving, or
private extension — MAY be a region layout, because each region declares and owns the
buffers its layout requires.

### D1 — Region descriptor is a trimmed "descriptor-tail" profile, not a byte-for-byte full TensorDescriptor

A region does **not** repeat the full tensor-descriptor frame. Fields that a full
`TensorDescriptor` carries but that would be pure redundancy or a new mismatch failure
mode per region are **omitted and inherited from the outer descriptor**:

- `magic`, `version_major`, `version_minor` — inherited (a nested magic per region would
  waste 4 bytes/region and add a consistency check).
- `rank` — inherited; `origin` and `region_shape` are already `uint64[rank]`.
- `shape` — a region's shape **is** its `region_shape`; there is no separate nested shape
  field, so the "nested shape MUST equal region_shape" hazard cannot arise by
  construction.
- `element_type` — inherited. A subpaving tensor has exactly one element type in v1.0;
  per-region element types are out of scope (a region's sparse `values` buffer uses the
  outer element type, index/pointer buffers use their layout-defined `uint64`, exactly as
  a top-level sparse tensor).
- `shard`, `statistics`, `extension_type` — forbidden per region (see D5).

Each region is encoded as a fixed prefix followed by a length-delimited body:

```
Region prefix:
| origin              | uint64[rank] | region start index, inclusive            |
| region_shape        | uint64[rank] | region extent; every value > 0           |
| region_layout_tag   | uint8        | any valid layout tag (0x01–0x0B, 0x40,   |
|                     |              | 0xF0–0xFE); MUST NOT be 0x00/0xFF        |
| region_flags        | uint8        | bit 0 = HAS_REGION_QUANTIZATION;         |
|                     |              | bits 1–7 reserved, MUST be 0             |
| _reserved           | uint8[2]     | MUST be 0x00                             |
| region_body_length  | uint32       | byte count of the body that follows      |

Region body (region_body_length bytes):
| byte_offset         | uint64       | per the region layout's own byte_offset  |
|                     |              | rule (see D4)                            |
| layout-specific     | variable     | fields for region_layout_tag, encoded as |
|   fields            |              | in metadata.md § Layout-Specific Fields, |
|                     |              | tag byte omitted (recursive for 0x06)    |
| buffer_count        | uint8        | region sub-table size (see D2)           |
| buffer_handles      | 16 × count   | the region's own buffer handles          |
| quantization        | present iff  | uint32 length + quantization_descriptor  |
|   section           | flags bit 0  | bytes (see D5)                           |
```

The outer subpaving layout-specific field remains `region_count: uint32` (> 0), followed
by `region_count` region descriptors.

`region_body_length` generalises ADR-015's `region_layout_length`: it enables a reader
that does not recognise a region's inner layout to skip the whole region body and continue
parsing subsequent regions (permissive mode); a strict-mode reader MUST reject an
unrecognised region layout tag.

### D2 — Buffer binding: outer table empty, effective table is the flattened region sub-tables

The buffer *properties* of every region live in that region's sub-table inside its body;
buffer *locations* continue to be supplied positionally by the transport. The binding
rule:

- **The outer subpaving descriptor's top-level buffer table is empty: `buffer_count = 0`.**
  This is a deliberate carve-out from the current `metadata.md` rule "`buffer_count` MUST
  be at least 1", which is amended to admit `0` for tag `0x06`.
- **The tensor's effective buffer list is the depth-first, region-order concatenation of
  every region's sub-table**, recursing into nested subpavings. For each region in region
  order: if the region layout is a leaf, emit its sub-table buffers in sub-table order
  (layout data buffers first, then quantization-parameter buffers per D5); if the region
  is itself a subpaving, recurse. This ordering is fully determined by the descriptor,
  contains no back-references, and is the order in which the file-format data region and
  streaming `TENSOR_DATA` frames lay the buffers down. Streamability is preserved.
- **Per-region sub-table size is exact.** A region's `buffer_count` MUST equal the number
  of buffers its layout requires (dense = 1, COO = 2, CSR/CSC = 3, CSF = 2·rank+1,
  block-paged ≥ 3, private extension = whatever the region declares) **plus** the number
  of quantization-parameter buffers required by the region's active scheme when
  `HAS_REGION_QUANTIZATION` is set. A reader MUST reject a region whose sub-table is
  over- or under-supplied (this is F-2's bounds and no-dangling safety rules, reborn
  per-region).

**The 255-buffer ceiling is fully dissolved.** The `uint8` cap now applies per region
(≤ 255 buffers per region), while `region_count` is `uint32`. A subpaving of thousands of
rank-3 CSF regions (7 buffers each) is representable; the effective buffer count is bounded
only by `region_count × 255`, i.e. effectively unbounded. This resolves the CSF-exhaustion
concern that made the per-region-buffer-list alternative (old Option B) unattractive.

### D3 — Any layout tag may be a region layout (resolves F-1, F-4)

Because each region owns its buffers, the ADR-015-era restriction is removed:
`region_layout_tag` MAY be any valid layout tag — dense (`0x01`–`0x06`, `0x40`), sparse
(`0x07`–`0x0A`), indirect (`0x0B`), or private extension (`0xF0`–`0xFE`) — subject to that
layout's own rank and shape constraints validated against `region_shape` (e.g. a
block-paged or CSF region forces the whole tensor to the rank that layout requires). A
private-extension region declares its own `buffer_count` in its sub-table; the sub-table
count is authoritative for extension layouts whose needs are otherwise out-of-band. The
standard private-tag interoperability caveat (no cross-implementation exchange without
out-of-band agreement) applies unchanged (F-4 resolved: permitted, with that caveat).

### D4 — byte_offset (resolves F-10)

The **outer** subpaving descriptor's tensor-level `byte_offset` MUST be
`0x0000000000000000` (there is no single first element at a fixed offset; element `[0,…,0]`
is located through region lookup). Each **region body** carries its own `byte_offset`
governed by that region layout's own rule: for dense region layouts it MAY be non-zero and
MUST be ≤ the region's buffer-0 size; for sparse and indirect region layouts, and for a
nested subpaving region, it MUST be `0`, exactly as those layouts require at top level.

### D5 — Per-region quantization falls out for free (resolves F-5)

A region MAY carry a quantization section in its body, gated by `region_flags` bit 0. The
`quantization.md` § Buffer Table Placement Rules apply **within the region's sub-table
unchanged**: the region's quantization-parameter buffers occupy the sub-table indices after
its layout data buffers, MUST NOT alias the data buffer, and MUST share the region's
`device_tag` and `memory_class`. Heterogeneous per-region quantization (different schemes in
different regions) is therefore expressible in v1.0 at no extra machinery cost, because the
quantization descriptor is carried as opaque length-prefixed bytes (as it already is at the
top level).

`HAS_SHARD`, `HAS_STATISTICS`, and `HAS_EXTENSION_TYPE` are **forbidden** per region in
v1.0 (`region_flags` bits 1–7 MUST be 0): a region is not independently a shard of a parent
(the whole subpaving MAY be a shard), per-region statistics are deferred, and element type
is inherited so per-region extension-type descriptors are meaningless.

### D6 — Validation set

Coverage and non-overlap are unchanged (they are properties of `origin`/`region_shape`
only, independent of region contents; the volume-sum coverage check remains valid). New
normative rules:

1. Each region's layout MUST validate against its own `region_shape`
   (`validate_against_shape` applied recursively).
2. Each region's sub-table `buffer_count` MUST exactly equal its layout requirement plus
   its quantization-parameter requirement (D2). Unconditional MUST-reject on mismatch
   (memory safety).
3. **Device colocation is tensor-wide over the flattened buffer set.** All buffers of all
   regions MUST share the same `device_tag` and `memory_class`
   (`buffer-protocol.md` § Device Colocation, applied to the effective buffer list). A
   heterogeneous-device subpaving is out of scope in v1.0 and noted as a future open
   question.
4. Recursion depth: a subpaving region nested inside a subpaving increments depth; a reader
   MUST reject nesting deeper than 8 levels (the existing `MAX_SUBPAVING_DEPTH` /
   `MAX_TILED_DEPTH` guard, on both encode and decode).
5. `region_flags` reserved bits and region `_reserved` bytes MUST be 0.

### D7 — Addressing API is redefined to region-resolution + per-region delegation

A subpaving element address can no longer be a single pure-arithmetic offset, because a
sparse or indirect region's value lookup is **data-dependent**: locating element `[i,j]` in
a COO region requires searching that region's `indices` buffer *contents*, yielding either
`values[p]` or an implicit zero — this cannot be expressed as a byte offset without reading
buffer bytes, which the descriptor-only addressing layer does not possess. Therefore:

- `SubpavingLayout::locate_element` is redefined to resolve the containing region and the
  local index within it (pure arithmetic, recursing through nested subpavings), returning a
  handle: `{ region_index, local_index, &RegionDescriptor }`.
- For a **dense** region, addressing then returns the element's byte offset within the
  region's sub-table buffer 0 (via the region layout's existing `element_offset`).
- For a **sparse or indirect** region, addressing returns a "requires buffer lookup" result
  carrying the region index, local index, and the region's buffer sub-table; the caller (a
  higher layer that holds actual buffer memory) performs the value lookup using that
  layout's Element Lookup algorithm (`coo.md`/`csr.md`/`csc.md`/`csf.md`/`block-paged.md`).

This is consistent with the existing model: top-level sparse tensors already return
`Error::LayoutRequiresMultiBuffer` from pure-offset addressing. Subpaving simply delegates
per region. It is nonetheless a genuine public-API change to `SubpavingLocation` and is
called out as the single largest code cost below.

## Alternatives Considered

**Dense-only inner-layout whitelist (prior draft's Option A).** Restrict `region_layout_tag`
to dense tags and defer sparse-in-subpaving. Rejected by decision: it forecloses the
heterogeneous-sparsity array-database use case the project explicitly wants in v1.0.

**Per-region buffer list (Option B): replace `buffer_index` with `buffer_index_count` +
`buffer_index[]` indexing a single flat outer buffer table.** Rejected: the flat outer
table is `uint8`-counted, so the 255-buffer ceiling caps the whole tensor — a handful of
CSF regions exhaust it. Nesting per-region sub-tables (this ADR) moves the cap per-region
and dissolves it. Option B also still could not carry per-region quantization without
further extension.

**Packed single-buffer sub-format (Option C): concatenate a sparse region's component
arrays into one buffer slice at computed offsets.** Rejected: it re-creates the "single
buffer with offsets" approach ADR-002 rejected for sparse — heterogeneous element types in
one slice, computed sub-offsets that break the 64-byte per-component alignment guarantee,
and loss of independent zero-copy component sharing.

**Byte-for-byte full `TensorDescriptor` per region.** Rejected in favour of the trimmed
descriptor-tail profile (D1): repeating magic/version/rank/element_type/shape per region
wastes bytes across potentially millions of regions and manufactures a "MUST equal the
outer value" consistency check for each repeated field. The trimmed profile inherits those
fields and keeps only what genuinely varies per region (layout, byte_offset, buffers,
quantization).

**Keep the outer buffer table non-empty as the flattened list.** Rejected: it would place
the flattened list back under the `uint8` outer count, reinstating the 255 ceiling. The
outer `buffer_count = 0` carve-out (D2) is the price of an unbounded effective count.

## Consequences

### Positive

- Heterogeneous-sparsity tensors — dense tiles, sparse blocks, and paged blocks in one
  logical tensor — are first-class in v1.0, serving the array-database vision.
- The 255-buffer ceiling is dissolved for subpaving (per-region cap, `uint32` region
  count).
- Per-region quantization and per-region layout diversity fall out of one mechanism; F-1,
  F-2, F-4, F-5, F-10 are all resolved coherently.
- Streamability, self-delimitation, and no-back-reference properties are preserved: the
  effective buffer order is a pure function of the descriptor, laid down in region order.

### Negative / obligations created

- **New wire format for regions.** ADR-015's `RegionDescriptor` encoding
  (`buffer_index` + `region_byte_offset` + `region_layout_length` + layout-only payload) is
  replaced by the descriptor-tail profile. ADR-015 is superseded. Every existing
  subpaving descriptor byte layout, doc-comment, and round-trip test is invalidated.
- **`metadata.md` invariant relaxed.** "`buffer_count` MUST be at least 1" gains a
  subpaving exception (`0`). Every reader that assumes ≥ 1 for all layouts must special-case
  `0x06`. `TensorDescriptor::new`/`decode`'s `EmptyBufferTable` check must exempt subpaving.
- **Transport must flatten.** The file-format reader/writer, the index `data_length`
  computation, and the streaming `TENSOR_DATA` walk must iterate the *effective* (flattened)
  buffer list for subpaving rather than the top-level table, and must handle > 255 effective
  buffers.
- **Addressing API redesign** (D7): `SubpavingLocation` changes from a pure offset to a
  region-resolution enum; sparse/indirect regions return a "requires buffer lookup" result.
  Downstream callers of `locate_element` must adapt.
- **Codec layering change.** `decode_region`/`encode_region` must now encode a buffer
  sub-table and an optional quantization section inside the layout payload. Buffer-handle
  codec and quantization-section codec currently live in the descriptor-level
  `encode.rs`/`decode.rs`; they must be factored into shared helpers that
  `layout_codec.rs` can call. `layout_codec` gains a dependency on the buffer-table codec.
- **Per-region wire overhead.** Each region costs `16·rank` bytes (origin + region_shape)
  plus the body; for tensors with millions of regions this is significant. This design
  trades wire compactness for generality and streamability.

### Risks

- **Effective-buffer flattening bugs.** The flattening order is load-bearing (it defines
  the on-disk / on-wire data order). Mitigation: a single normative flattening algorithm in
  `memory-layout.md`, one shared implementation, and round-trip tests through the file
  format with mixed dense/sparse regions.
- **Naive reader mis-reads `buffer_count = 0`.** A reader that does not understand tag
  `0x06` sees an empty top-level table; it MUST already reject unknown layout tags in strict
  mode, and MUST NOT dereference data in permissive mode, so `0` is safe.
- **Device-colocation over-restriction.** Tensor-wide colocation forbids per-region devices;
  the array-DB use case may eventually want per-region device placement. Deferred as an OQ,
  not closed off (an additive relaxation under the evolvability contract).

## Compatibility Impact

During the pre-1.0 draft period this redefines the region wire format and relaxes the
`buffer_count >= 1` invariant for tag `0x06`. No previously *interpretable* descriptor is
silently changed (sparse/private inner regions were never interpretable). Under ADR-019,
the additive features left open here (per-region statistics, per-region element type,
heterogeneous-device subpaving) arrive later as gated additive changes without rebinding any
`1.x` value. Supersedes ADR-015.

## Date

2026-07-05
