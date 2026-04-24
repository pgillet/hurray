# Compound Annotation — Hurray Format Specification

> **Status:** Draft

## Scope

This section defines the **Compound Annotation**: an optional, purely logical
metadata section attached to a tensor descriptor that declares its innermost
dimension as a fixed-size tuple of named or unnamed components. A compound
annotation never changes the tensor's bytes, shape, strides, storage type, or
buffer layout — it only changes the consumer-facing view. A reader that does
not understand or chooses to ignore a compound annotation MUST process the
tensor as a primitive tensor of rank `rank` with storage type `type_tag`. The
bytes, strides, element addresses, and zero-copy semantics are identical in
both views.

## Normative Requirements

> This section uses RFC 2119 key words: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT,
> SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

All multi-byte fields defined in this section are encoded in little-endian
byte order (least significant byte at the lowest address).

---

## Annotation Model

A **compound annotation** declares that the innermost dimension of a tensor
(dimension `rank - 1`) is a **fixed-size tuple** of `component_count`
components. All components share the tensor's primitive element type
(`type_tag`); there is no mechanism for mixed-type components and no
recursion into further compound types.

The annotation is present on a tensor descriptor if and only if the
`HAS_COMPOUND` flag (bit 4 of the descriptor `flags` field; see
`metadata.md`) is set. When the annotation is present:

- The tensor's underlying bytes, shape, strides, `byte_offset`, buffer table,
  and `type_tag` are unaffected.
- A consumer that understands the annotation MAY present the tensor at
  logical rank `rank - 1`, with element type "tuple of `component_count`
  primitives".
- A consumer that does not understand the annotation MUST present the tensor
  at its primitive rank `rank` with its primitive `type_tag`.

No compound type is ever represented in the `type_tag` field. The `type_tag`
space defined in `element-types.md` remains strictly primitive.

---

## Binary Encoding

The Compound Annotation Section is present if and only if `HAS_COMPOUND`
(bit 4 of the descriptor `flags`) is set. Its byte-level encoding is:

| Offset | Field              | Type       | Description |
|--------|--------------------|------------|-------------|
| 0      | `section_length`   | `uint32`   | Total byte length of this section, including this field. |
| 4      | `component_count`  | `uint8`    | Number of components. MUST be `>= 2` and `<= 64`. |
| 5      | `flags`            | `uint8`    | Compound flags (see [Compound Flags](#compound-flags)). |
| 6      | `_reserved`        | `uint8[2]` | MUST be `0x00`. A reader MUST reject the section if any of these bytes is non-zero. |
| 8      | `name_table_length`| `uint32`   | Byte length of the name table. MUST be `0x00000000` if and only if `HAS_COMPONENT_NAMES` is clear. |
| 12     | `name_table`       | `bytes[name_table_length]` | Present if and only if `HAS_COMPONENT_NAMES` is set. See [Component Name Table](#component-name-table). |

All multi-byte fields are little-endian.

The section is self-delimiting via `section_length`, allowing a reader that
does not wish to interpret the compound annotation to skip it without
parsing the name table.

### Compound Flags

The `flags` field is a single byte:

| Bit | Name                  | Meaning |
|-----|-----------------------|---------|
| 0   | `HAS_COMPONENT_NAMES` | A component name table follows after `name_table_length`. |
| 1–7 | (reserved)            | MUST be `0`. A reader MUST reject a section with any reserved bit set. |

### Component Name Table

When `HAS_COMPONENT_NAMES` is set, the `name_table` field contains exactly
`component_count` name entries encoded back-to-back with no alignment
padding. Each entry is:

| Field         | Type                 | Description |
|---------------|----------------------|-------------|
| `name_length` | `uint16`             | Byte length of the UTF-8 name. MUST be `>= 1` and `<= 64`. |
| `name`        | `bytes[name_length]` | UTF-8 encoded component name. MUST NOT contain a `0x00` byte. |

A reader MUST reject a Compound Annotation Section that violates any of the
following:

- `component_count < 2` or `component_count > 64`.
- Any reserved flag bit is set.
- Either byte of `_reserved` at offset 6 is non-zero.
- `name_table_length` is non-zero and `HAS_COMPONENT_NAMES` is clear.
- `name_table_length` is zero and `HAS_COMPONENT_NAMES` is set.
- The number of name entries in the name table does not equal
  `component_count`.
- Any `name_length` is `0` or greater than `64`.
- Any `name` contains a `0x00` byte.
- Any `name` is not valid UTF-8.
- Two or more names in the same annotation are byte-identical (duplicate
  component names are prohibited).

---

## Zero-Copy Preconditions

The Compound Annotation MAY be attached to a tensor descriptor only when
every one of the following conditions is satisfied. A writer MUST NOT emit
`HAS_COMPOUND` unless every condition holds. A reader MUST reject a
descriptor in which `HAS_COMPOUND` is set and any condition is violated.

1. **Rank.** `rank >= 1`. A scalar tensor (`rank = 0`) MUST NOT carry a
   Compound Annotation.
2. **Trailing-dim size.** `shape[rank - 1]` MUST equal `component_count`.
3. **No dynamic trailing dim.** `shape[rank - 1]` MUST NOT be the dynamic
   dimension sentinel `0xFFFFFFFFFFFFFFFF`.
4. **Unit stride on the trailing dim.** The logical stride along dimension
   `rank - 1` MUST be `1`:
   - For `layout_tag` `0x01` (row-major), this is implicit and satisfied
     automatically.
   - For `layout_tag` `0x03` (strided), `strides[rank - 1]` MUST equal `1`.
   - For all other layout tags (column-major `0x02`, tiled `0x04`, Morton
     `0x05`, general subpaving `0x06`, COO `0x07`, CSR `0x08`, CSC `0x09`,
     Hilbert `0x40`, and all extension layouts), `HAS_COMPOUND` MUST NOT be
     set.
5. **Packed storage.** The storage type MUST be a whole-byte primitive
   (`bit_width >= 8`), or a sub-byte type whose packing factor exactly
   divides `component_count` so that one compound element spans an integer
   number of bytes with no intra-element bit fragmentation:
   - `int4` / `uint4`: permitted if and only if `component_count` is even.
   - `int2` / `uint2`: permitted if and only if `component_count` is a
     multiple of `4`.
   - `bool`: permitted if and only if `component_count` is a multiple of
     `8`.
6. **No quantization.** `HAS_QUANTIZATION` (bit 0) and `HAS_COMPOUND`
   (bit 4) MUST NOT both be set on the same descriptor.
7. **Extension type compatibility.** If `type_tag` is in the extension
   range (`0xF0`–`0xFE`), condition 5 MUST be evaluated using the
   `bit_width` and `packing_factor` carried in the Extension Type
   descriptor (see `metadata.md`). Whole-byte extension types
   (`bit_width >= 8`, `packing_factor = 1`) are always permitted; sub-byte
   extension types are permitted if and only if the extension's
   `packing_factor` exactly divides `component_count`.

> **Note (non-normative):** Conditions 4 and 5 together guarantee that the
> byte-for-byte content of the tensor is identical between the compound
> view (shape `shape[0..rank-1]`) and the primitive view (shape
> `shape[0..rank]`). No data motion is ever required for the
> reinterpretation. This is the zero-copy contract: a consumer that
> prefers the primitive view can read the same buffer, at the same byte
> offset, with the same strides, and obtain the same bytes.

---

## Packed-Only Alignment

Compound elements are **packed**. There is no inter-field padding between
adjacent components of a single compound element, and there is no trailing
padding after the last component. The byte size of one compound element is
exactly:

- For whole-byte primitives: `component_count * (bit_width / 8)` bytes.
- For permitted sub-byte primitives: `component_count / packing_factor`
  bytes, which is always an integer by condition 5 of
  [Zero-Copy Preconditions](#zero-copy-preconditions).

Max-member alignment, next-power-of-two padding, and explicit padding
fields are **prohibited**. A producer that needs interior padding to match
a hardware vector type (for example, mapping three RGB channels to a
four-lane `uchar4` SIMD register) MUST allocate the padding as a real
trailing-dimension element, not as invisible padding within a compound
element. In such cases the tensor MUST declare the larger `component_count`
(for example, `4` for RGBA) and the producer MUST write an explicit value
for every component.

---

## No Recursion

A Compound Annotation describes a **flat tuple of primitive components**.
Nested compound types MUST NOT occur in v1. The annotation encoding
provides no mechanism for a component that is itself compound: every
component has the tensor's primitive `type_tag`, and the tensor's
`type_tag` is always a primitive tag as defined in `element-types.md`.

A producer that conceptually requires nesting MUST flatten its structure
into a higher-rank tensor with a single-level annotation.

---

## Component Count

`component_count` MUST be an integer in the inclusive range `[2, 64]`. A
writer MUST NOT emit a Compound Annotation with `component_count = 0` or
`component_count = 1`; a reader MUST reject such an annotation.

The upper bound of `64` matches the normative rank cap of Hurray tensors
(see `data-model.md` and ADR-008).

> **Note (non-normative):** Typical values are `2` (for packed 2-vectors
> and complex-like pairs), `3` (RGB, XYZ), and `4` (RGBA, quaternions,
> hardware-aligned 32-bit or 128-bit vectors). Larger values occur in
> spectral imaging and embedding-projection tensors but are uncommon.

---

## Component Names

Component names are OPTIONAL, gated by the `HAS_COMPONENT_NAMES` flag bit.
When absent, components are addressed by zero-based index (`0`,
`1`, ..., `component_count - 1`). When present, the name table carries
exactly `component_count` UTF-8 names, each between `1` and `64` bytes long
and pairwise distinct within the annotation.

Component names are **advisory**. A reader MUST NOT use a component name
to perform any correctness-affecting decoding decision — specifically,
nothing in the buffer layout, byte offsets, strides, or element addresses
depends on whether names are present or what they are. A reader MAY
ignore the name table entirely while still correctly consuming the
tensor.

---

## DLPack Lowering

DLPack has no compound type concept. A Hurray producer exporting a
tensor with a Compound Annotation to a DLPack consumer MUST lower it as
follows:

1. The exported DLPack tensor MUST have rank equal to the Hurray tensor's
   primitive rank `rank` (including the trailing dimension of size
   `component_count`).
2. The exported DLPack tensor's `shape`, `strides`, `byte_offset`,
   `device_type`, `device_id`, and `data` pointer MUST be carried over
   unchanged from the primitive view of the Hurray tensor.
3. The Compound Annotation MUST be dropped on export. It is not
   round-trippable through DLPack: a subsequent DLPack-to-Hurray import
   produces a descriptor with the primitive shape and `HAS_COMPOUND` not
   set.

> **Note (non-normative):** Producers that wish to preserve compound
> semantics across a DLPack boundary MUST carry the annotation in an
> out-of-band channel (for example, a side metadata structure managed by
> the producer and consumer). The zero-copy data path is unaffected; only
> the component-naming metadata is lost.

---

## Layout Compatibility

> **Note (non-normative):** The following table summarises which layout
> tags are compatible with `HAS_COMPOUND`, derived from the normative
> preconditions in [Zero-Copy Preconditions](#zero-copy-preconditions).

| `layout_tag` | Layout name            | Compatible? | Notes |
|--------------|------------------------|-------------|-------|
| `0x01`       | Row-major              | Yes         | Unit trailing stride is implicit. |
| `0x02`       | Column-major           | No          | Trailing stride is always the outer product, never `1`. |
| `0x03`       | Strided                | Conditional | Permitted if and only if `strides[rank - 1] == 1`. |
| `0x04`       | Tiled / blocked        | No          | Trailing-element contiguity depends on tile geometry; not guaranteed. |
| `0x05`       | Morton (Z-order)       | No          | Adjacent logical elements are not byte-adjacent. |
| `0x06`       | General subpaving      | No          | Per-region layouts cannot uniformly guarantee trailing contiguity. |
| `0x07`       | COO (sparse)           | No          | No meaningful contiguous trailing dimension. |
| `0x08`       | CSR (sparse)           | No          | No meaningful contiguous trailing dimension. |
| `0x09`       | CSC (sparse)           | No          | No meaningful contiguous trailing dimension. |
| `0x40`       | Hilbert curve          | No          | Adjacent logical elements are not byte-adjacent. |
| `0xF0`–`0xFE`| Extension layouts      | No          | Extension layouts MUST NOT carry `HAS_COMPOUND`. |

---

## Worked Examples

> **Note (non-normative):** The following examples illustrate typical
> Compound Annotation usage. They are non-normative; the normative
> requirements are stated in the preceding sections.

### Example 1 — RGB pixel tensor

A tensor of shape `[H, W, 3]` with `type_tag = 0x11` (`uint8`),
`layout_tag = 0x01` (row-major), and a Compound Annotation with:

- `component_count = 3`
- `HAS_COMPONENT_NAMES` set
- Names: `"r"`, `"g"`, `"b"`

A consumer that understands the annotation MAY present the tensor as a
rank-2 tensor of shape `[H, W]` whose element type is an RGB pixel with
three `uint8` components. A consumer that ignores the annotation sees the
same tensor as rank-3, shape `[H, W, 3]`, `uint8`. The bytes on the wire
and in memory are identical in both views.

### Example 2 — Quaternion tensor

A tensor of shape `[N, 4]` with `type_tag = 0x03` (`float32`),
`layout_tag = 0x01` (row-major), and a Compound Annotation with:

- `component_count = 4`
- `HAS_COMPONENT_NAMES` set
- Names: `"w"`, `"x"`, `"y"`, `"z"`

Compound size: `4 * 4 = 16` bytes per quaternion.

### Example 3 — Unnamed 2-tuple

A tensor of shape `[M, N, 2]` with `type_tag = 0x01` (`float16`),
`layout_tag = 0x01` (row-major), and a Compound Annotation with:

- `component_count = 2`
- `HAS_COMPONENT_NAMES` clear
- `name_table_length = 0`
- No name table bytes

Components are addressed as index `0` and index `1`. Compound size:
`2 * 2 = 4` bytes per tuple.

### Example 4 — CUDA `uchar4` equivalent

A tensor of shape `[H, W, 4]` with `type_tag = 0x11` (`uint8`),
`layout_tag = 0x01` (row-major), and a Compound Annotation with:

- `component_count = 4`
- `HAS_COMPONENT_NAMES` set
- Names: `"r"`, `"g"`, `"b"`, `"a"`

Compound size: `4 * 1 = 4` bytes per pixel, exactly matching the layout
of a hardware 32-bit aligned load such as CUDA's `uchar4`. Because
`component_count` is a power of two, the minor dimension is naturally
aligned to the compound size, and consumers that support vector loads
can issue a single 32-bit access per compound element without any
reinterpretation of the underlying bytes.

---

## Interaction with Other Sections

- **`element-types.md`**: defines the primitive `type_tag` values that a
  compound-annotated tensor may carry. No compound tag is defined there.
- **`data-model.md`**: describes the logical tensor model; a compound
  annotation never changes the model's shape, rank, or storage type.
- **`metadata.md`**: defines the `HAS_COMPOUND` flag bit, the ordering of
  the Compound Annotation Section within the descriptor byte stream, and
  a brief byte-level summary cross-referenced to this section.
- **`memory-layout.md`** and `layouts/`: define the layout tags referenced
  by the [Layout Compatibility](#layout-compatibility) table.
- **`quantization.md`**: `HAS_QUANTIZATION` and `HAS_COMPOUND` are
  mutually exclusive on a single descriptor (condition 6 of
  [Zero-Copy Preconditions](#zero-copy-preconditions)).

---

## Version

`HAS_COMPOUND` is introduced in Hurray format version `1.1`. A v1.0
reader encountering a v1.1 descriptor with `HAS_COMPOUND` set MUST reject
it; this is already implied by the reserved-flag rejection rule in v1.0
(see `metadata.md` and `versioning.md`).
