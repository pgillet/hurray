# ADR-015: Subpaving Region Inline Layout Encoding

## Status
Accepted

## Context

The general subpaving layout (`0x06`) partitions a tensor's index space into
non-overlapping rectangular regions, each with its own inner layout. Each
RegionDescriptor carries a `region_layout_tag` byte that identifies the inner
layout, but the format left the encoding of layout-specific fields for that tag
(e.g., strides for `0x03`, tile dimensions for `0x04`, Morton bits for `0x05`,
Hilbert order/rank for `0x40`, or a nested region list for recursive `0x06`)
undefined.

This was filed as `subpaving.md OQ-1`. Without a normative encoding, the reference
implementation could only handle row-major (`0x01`) and column-major (`0x02`)
inner regions — both of which have no additional descriptor fields. The Layer 4
restriction note was added to `subpaving.md` and the addressing code pending
resolution of this OQ.

Three options were considered:

**(a) Reuse the top-level layout encoding inline, with a length prefix**
Add `region_layout_length: uint32` immediately after `region_byte_offset`. The
next `region_layout_length` bytes carry the layout-specific fields for
`region_layout_tag`, using the same field encoding defined in `metadata.md`
§ Layout-Specific Fields for that tag — with the tag byte itself omitted (it is
already present in `region_layout_tag`). Set `region_layout_length = 0` for
layouts that have no additional fields (row-major `0x01`, column-major `0x02`).

**(b) Tag-specific inline structs without a length field**
Encode the layout-specific fields inline in a fixed, tag-defined order with no
framing length. Readers must know the exact byte size for every tag they may
encounter; unknown tags make the descriptor un-parsable without forward-skipping.

**(c) A separate "inner layout table" referenced by index**
Add a parallel table of inner layout descriptors to the subpaving header, and
have each RegionDescriptor carry an index into that table instead of inline
fields. More compact when many regions share the same inner layout, but
introduces a non-sequential encoding dependency and is harder to stream.

## Decision

Use **option (a)**: a `region_layout_length: uint32` (little-endian) prefix
immediately following `region_byte_offset` in every RegionDescriptor, followed
by `region_layout_length` bytes of layout-specific fields using the same
per-tag encoding defined in `metadata.md` § Layout-Specific Fields (tag byte
omitted).

The RegionDescriptor binary layout becomes:

| Field | Type | Description |
|-------|------|-------------|
| `origin` | `uint64[rank]` | Starting index along each dimension (inclusive). |
| `region_shape` | `uint64[rank]` | Size along each dimension. Every value MUST be > 0. |
| `region_layout_tag` | `uint8` | Inner layout tag. MUST NOT be `0x00` or `0xFF`. |
| `_reserved` | `uint8[3]` | MUST be `0x00`. |
| `buffer_index` | `uint32` | Index into the tensor's buffer table for this region. |
| `region_byte_offset` | `uint64` | Byte offset to the start of this region's data in the buffer. |
| `region_layout_length` | `uint32` | Byte count of the inner layout payload that follows. MUST be `0` for `region_layout_tag` values `0x01` and `0x02`. |
| `region_layout_payload` | `bytes[region_layout_length]` | Layout-specific fields for `region_layout_tag`, encoded identically to `metadata.md` § Layout-Specific Fields for that tag, with the tag byte omitted. |

An 8-level nesting depth limit applies: a reader MUST reject any descriptor
where the total subpaving recursion depth exceeds 8. This limit matches the
existing implementation constant `MAX_SUBPAVING_DEPTH`.

In the Rust reference implementation, `RegionDescriptor` gains a new field:
```
inner_layout: Option<Box<LayoutDescriptor>>
```
`None` for tags that carry no additional fields (`0x01`, `0x02`); `Some` for all
other tags, where the `LayoutDescriptor` variant must have `tag() == region_layout_tag`.

## Alternatives Considered

**Option (b) — tag-specific inline structs, no length field** was rejected because
it requires every reader to know the exact byte width of every possible inner
layout tag. An unknown future tag would make the remaining RegionDescriptors
un-parsable. The length prefix enables safe forward-skipping and is consistent
with the length-prefixed approach used throughout the Hurray format.

**Option (c) — inner layout table** was rejected because it introduces a
backward reference (each RegionDescriptor points into a table that precedes it
in the byte stream only if the table is in the header), which conflicts with the
format's streamability requirement. It also adds per-table parsing complexity
for what is often a small number of distinct inner layouts.

## Consequences

- Every RegionDescriptor is 4 bytes larger due to the mandatory
  `region_layout_length` field. For the common case of row-major or col-major
  inner regions the payload length is 0, so the only overhead is the 4-byte
  length field itself.
- True recursive subpaving (inner `region_layout_tag = 0x06`) is now encodable;
  the 8-level depth cap prevents unbounded recursion at parse time.
- The `subpaving.md` spec must be updated: remove OQ-1, add `region_layout_length`
  and `region_layout_payload` to the RegionDescriptor table.
- `metadata.md` § General Subpaving (`0x06`) must be updated with the same
  RegionDescriptor table change.
- The `RegionDescriptor::new()` constructor signature is unchanged; a
  `with_inner_layout()` builder is added for the non-trivial case.
- Layer 4 restriction in `addressing/subpaving.rs` is lifted; `region_inner_offset`
  now dispatches through `inner_layout` for all supported tags.
- `OQ-014.3` (region ordering/pre-sorting) remains open and is tracked separately.
