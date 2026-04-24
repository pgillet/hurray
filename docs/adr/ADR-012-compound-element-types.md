# ADR-012: Compound Element Types as Logical Metadata Over a Trailing Dimension

## Status

Accepted

## Context

The spec currently defines a strictly numeric scalar element-type system
(`element-types.md`). A recurring inference/graphics use case is to group a
small fixed number of scalars into a single logical element: an RGB pixel
tensor, a vector-field tensor, a complex tuple, a quaternion. The canonical
worked example is a tensor of shape `[H, W, 3]` with element type `uint8`
that a consumer wishes to view as shape `[H, W]` with a compound element
representing an RGB pixel — without copying the underlying bytes.

Two design poles were surveyed:

- **Compound-as-scalar-dtype** (NumPy / HDF5 / NetCDF-4 / Zarr v2 / Vulkan
  texel formats): the compound is a first-class dtype. Shape `[H, W]` with
  dtype `rgb24` is a distinct descriptor from `[H, W, 3]` `uint8`. The two
  share bytes only when the trailing `uint8` axis is contiguous.
- **Compound-as-logical-metadata** (Apache Arrow `FixedShapeTensor`
  canonical extension): the compound is extension metadata over a primitive
  trailing dimension. Shape `[H, W, 3]` `uint8` carries metadata saying
  "the innermost dim is a tuple of 3 named components".

The zero-copy-first invariant is the hardest constraint. A compound
reinterpretation is safe only when the axis being collapsed is
**contiguous (unit stride), packed (no inter-field padding), and
minor-most**. Arbitrarily-aligned compounds cannot zero-copy with a
primitive tensor of the same shape.

The Hurray target ecosystem — DLPack, SafeTensors, GGUF — has no compound
type concept; any compound dtype Hurray defines must have a deterministic
lowering to a primitive-plus-trailing-dim representation for interop.
Hardware vector loads (`uchar4`, `half2`, `float4`) are power-of-two sized
and power-of-two aligned; the image-natural `rgb24` (3 × uint8) is
hardware-hostile.

This ADR resolves six open questions:

1. Compound-as-dtype vs compound-as-metadata: which design pole?
2. Alignment policy: packed, max-member, or next-power-of-two?
3. Recursion: allow nested compounds?
4. Non-power-of-two compounds (e.g., `rgb24`): permit?
5. Field names: required, optional, or absent?
6. Where does the extension land in the spec?

## Decision

Hurray v1 adopts **compound-as-logical-metadata over a trailing dimension**,
modelled on the Apache Arrow `FixedShapeTensor` canonical extension, with a
binary descriptor encoding defined from first principles (no JSON, no
NumPy dtype string dependency). The resolution of each open question follows.

### D1. Design pole: compound-as-metadata

A compound element type is **not** a new value in the `type_tag` space. It
is an optional **Compound Annotation Section** attached to a tensor
descriptor whose `type_tag` is an existing primitive Tier 1 or Tier 2 type.
The annotation declares that the innermost `N` logical elements of the
tensor — along the final dimension — form a fixed-size tuple of `N`
components, each of the tensor's primitive element type.

A tensor of shape `[H, W, 3]` `uint8` MAY carry a Compound Annotation
Section declaring a 3-tuple with component names `("r", "g", "b")`. A
reader that understands the annotation MAY present the tensor to consumers as
shape `[H, W]` with compound element type; a reader that does not MUST treat
the tensor exactly as `[H, W, 3]` `uint8`. The underlying descriptor, buffer,
strides, and byte layout are identical in both views.

The annotation is gated by a new flag bit `HAS_COMPOUND` (bit 4 of `flags`
in `metadata.md`).

### D2. Binary encoding of the Compound Annotation Section

The section is present if and only if `HAS_COMPOUND` (bit 4) is set.

| Offset | Field              | Type       | Description |
|--------|--------------------|------------|-------------|
| 0      | `section_length`   | `uint32`   | Total byte length of this section, including this field. |
| 4      | `component_count`  | `uint8`    | Number of components. MUST be `>= 2` and `<= 64`. |
| 5      | `flags`            | `uint8`    | Compound flags (see below). |
| 6      | `_reserved`        | `uint8[2]` | MUST be `0x00`. |
| 8      | `name_table_length`| `uint32`   | Byte length of the name table. `0x00000000` iff `HAS_COMPONENT_NAMES` is clear. |
| 12     | `name_table`       | `bytes[…]` | Present iff `HAS_COMPONENT_NAMES` is set. See encoding below. |

**Compound flags** (one byte):

| Bit | Name                  | Meaning |
|-----|-----------------------|---------|
| 0   | `HAS_COMPONENT_NAMES` | A component name table follows. |
| 1–7 | (reserved)            | MUST be `0x00`. A reader MUST reject a section with any reserved bit set. |

**Component name table** (present iff `HAS_COMPONENT_NAMES` is set):

The name table consists of exactly `component_count` entries encoded
back-to-back. Each entry is:

| Field         | Type              | Description |
|---------------|-------------------|-------------|
| `name_length` | `uint16`          | Byte length of the UTF-8 name. MUST be `>= 1` and `<= 64`. |
| `name`        | `bytes[name_length]` | UTF-8 encoded component name. MUST NOT contain a `0x00` byte. |

Names MUST be pairwise distinct within a single compound annotation. A
reader MUST reject an annotation with duplicate component names.

The section is self-delimiting via `section_length`, allowing readers that
do not wish to interpret the compound to skip it without parsing the name
table.

### D3. Contiguity preconditions (the zero-copy contract)

The Compound Annotation MAY be attached to a tensor descriptor if and only
if all of the following conditions hold. A writer MUST NOT emit
`HAS_COMPOUND` unless every condition is satisfied; a reader MUST reject a
descriptor with `HAS_COMPOUND` set that violates any condition.

1. **Rank.** `rank >= 1`. Scalar tensors (`rank = 0`) MUST NOT carry a
   compound annotation.
2. **Trailing-dim size.** `shape[rank - 1] == component_count`.
3. **No dynamic trailing dim.** `shape[rank - 1]` MUST NOT be the dynamic
   dimension sentinel (`0xFFFFFFFFFFFFFFFF`).
4. **Unit stride on the trailing dim.** The effective logical stride along
   dimension `rank - 1` MUST be `1`:
   - For `layout_tag` `0x01` (row-major), this is implicit and satisfied.
   - For `layout_tag` `0x03` (strided), `strides[rank - 1]` MUST equal `1`.
   - For all other layout tags (column-major, tiled, Morton, Hilbert,
     subpaving, and all sparse layouts), `HAS_COMPOUND` MUST NOT be set.
5. **Packed storage.** The storage type MUST be a whole-byte primitive
   (`bit_width >= 8`), or a sub-byte type whose packing factor exactly
   divides `component_count` (so a compound element spans an integer number
   of bytes with no intra-element fragmentation):
   - `int4`/`uint4`: permitted iff `component_count` is even.
   - `int2`/`uint2`: permitted iff `component_count` is a multiple of 4.
   - `bool`: permitted iff `component_count` is a multiple of 8.
6. **No quantization.** `HAS_QUANTIZATION` and `HAS_COMPOUND` MUST NOT both
   be set on the same descriptor.
7. **Extension type compatibility.** If `type_tag` is an extension tag
   (`0xF0`–`0xFE`), the extension type descriptor's `bit_width` and
   `packing_factor` are used to evaluate condition 5.

> **Note (non-normative):** Conditions 4 and 5 together guarantee that the
> byte-for-byte content of the tensor is identical between the compound view
> (shape `shape[0..rank-1]`) and the primitive view (shape `shape[0..rank]`).
> No data motion is ever required for this reinterpretation.

### D4. Alignment policy: packed only

Compound elements are **packed**. There is no inter-field padding and no
trailing padding. The byte size of one compound element is exactly
`component_count * sizeof(primitive)` for whole-byte primitives, and exactly
`component_count / packing_factor` bytes for sub-byte primitives (which
condition 5 forces to be an integer).

Max-member alignment, next-power-of-two padding, and explicit padding fields
are all **prohibited**. A tensor that requires interior padding to match a
hardware vector type (e.g., RGBA32 from three RGB channels plus one alpha)
MUST be represented by allocating the padding as a real trailing-dim element,
not as invisible padding.

### D5. Recursion: not permitted

A Compound Annotation Section describes a **flat tuple of primitive
components**. Nested compounds are not permitted in v1. Components are always
of the tensor's primitive `type_tag`; there is no encoding path for a
component that is itself compound.

### D6. Non-power-of-two compounds: permitted

`component_count` MAY be any integer in `[2, 64]`. Image-natural widths
(`3` for RGB, `5` for RGBA+depth) are not hardware-primitive on any GPU, but
they are legitimate logical groupings broadly used in capture and compute
pipelines. The upper bound of 64 matches the normative rank cap (ADR-008).

> **Note (non-normative):** Consumers that care about hardware vectorization
> will naturally use `component_count` values of 2 or 4. Consumers that care
> about image semantics will use 3 or 4. The spec does not prefer one.

### D7. Field names: optional

Component names are **optional** (`HAS_COMPONENT_NAMES` flag). When absent,
components are addressed by zero-based index. When present, the name table
carries exactly `component_count` UTF-8 names, each `1`–`64` bytes long and
pairwise distinct within the annotation. Names are **advisory**: a reader
MUST NOT use a component name to perform any correctness-affecting decoding.

### D8. DLPack lowering

DLPack has no compound type. A Hurray producer exporting a compound tensor to
a DLPack consumer MUST lower it as follows:

1. The exported DLPack tensor has **rank `rank`** (the full primitive rank,
   including the trailing dim of size `component_count`).
2. Shape, strides, byte offset, device, and data pointer are carried over
   unchanged from the primitive view.
3. The compound annotation is **dropped** on export. It is not
   round-trippable through DLPack. A subsequent DLPack-to-Hurray import
   produces a descriptor with the primitive shape and no `HAS_COMPOUND` flag.

### D9. Spec placement

The compound annotation is specified in a new section
`docs/spec/compound-types.md`, cross-referenced from `element-types.md`,
`data-model.md`, and `metadata.md`. It is not folded into `element-types.md`
— the type-tag space remains strictly primitive.

## Alternatives Considered

**Compound-as-scalar-dtype (NumPy / HDF5 / Zarr v2).** Assign each compound
a distinct `type_tag`, with an inline descriptor enumerating component types,
names, and offsets. Rejected: forces dtype proliferation or a
parameterized-dtype mechanism that undermines the closed `type_tag` enumeration
(see ADR-001). HDF5's experience shows that flexible compounds require a
conversion engine that contradicts zero-copy-first.

**Compound with configurable field alignment (C-struct parity).** Permit
packed, max-member, and next-power-of-two alignment modes. Rejected: breaks
byte-identity between primitive and compound views, eliminating zero-copy.
Hardware-aligned compounds that matter (`uchar4`, `float4`) are already
naturally packed by being power-of-two in component count.

**Recursive / nested compounds.** Rejected: every nested compound can be
flattened into a higher-rank tensor with a single-level annotation. No
expressive loss. No hardware concept for nested compounds in ML inference.

**Required field names.** Rejected: forces every producer to invent names
for unnamed tuples, inflating descriptors with no information content.

**Fold annotation into `element-types.md`.** Rejected: `element-types.md`
specifies storage types (what bytes mean). The compound annotation is a
logical grouping over existing storage. Co-locating them would obscure the
invariant that `type_tag` always decodes bytes unaided.

## Compatibility Impact

`HAS_COMPOUND` is bit 4 of the flags field, currently in the reserved range.
Introducing it is a **format minor version increment** (from `1.0` to `1.1`).
A v1.0 reader encountering a v1.1 descriptor with `HAS_COMPOUND` set MUST
reject it — this is already implied by the reserved-flag rejection rule in
`metadata.md`. A v1.1 reader MUST correctly handle a v1.0 descriptor as a
primitive tensor.

## Consequences

- New spec section `docs/spec/compound-types.md` MUST be written (see spec
  mandate below).
- `docs/spec/metadata.md` MUST document `HAS_COMPOUND` as flag bit 4, shrink
  the reserved range to bits 5–31, add a Compound Annotation Section row in
  the descriptor structure table, and record the v1.1 version bump.
- `docs/spec/element-types.md` MUST add a non-normative cross-reference to
  `compound-types.md` clarifying that `type_tag` remains strictly primitive.
- `docs/spec/data-model.md` MUST add a non-normative cross-reference noting
  that a tensor MAY carry a compound annotation without changing its bytes.
- `docs/spec/README.md` MUST add `compound-types.md` to the table of
  contents.
- `docs/impl/compliance.md` MUST include test vectors for:
  (a) Valid compound annotation, 3-component uint8 row-major, named `("r", "g", "b")`.
  (b) `HAS_COMPOUND` with `shape[rank-1] != component_count` — MUST be rejected.
  (c) `HAS_COMPOUND` on strided layout with `strides[rank-1] != 1` — MUST be rejected.
  (d) `HAS_COMPOUND` + `HAS_QUANTIZATION` both set — MUST be rejected.
  (e) v1.0 reader against v1.1 descriptor with `HAS_COMPOUND` set — MUST reject.
- Future ADRs MAY relax the "no quantization + compound" exclusion and MAY
  introduce nested or scalar-dtype compounds without invalidating v1 consumers.
