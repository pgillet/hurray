# Versioning — Hurray Format Specification

> **Status:** Draft

> This section uses RFC 2119 key words: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT,
> SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

## Scope

This section defines the **versioning model** for the Hurray format and the
compatibility policy that conforming readers and writers MUST follow.

The Hurray format carries three independent version axes:

1. **Descriptor version** — the format version of the tensor descriptor itself.
2. **Container version** — the format version of the file-level container.
3. **Quantization scheme version** — the version of an individual quantization
   scheme's wire encoding.

The three axes evolve independently: a single major version of the descriptor
may coexist with multiple major versions of the container, and quantization
schemes may version their parameter encodings without affecting either of the
other two axes.

> **Note (non-normative):** Decoupling these axes avoids forcing a global
> version bump every time a single component evolves. A new scheme tag, for
> example, only requires a descriptor minor version increment — it does not
> require the container or any other scheme to change versions.

---

## Version Axes

### Descriptor Version (`version_major` / `version_minor`)

The descriptor version describes the wire encoding of the tensor descriptor as
defined in `metadata.md`.

- **Location**: tensor descriptor fixed header, byte `4` (`version_major`,
  `uint8`) and byte `5` (`version_minor`, `uint8`). See `metadata.md` § Fixed
  Header.
- **Current values**: `version_major = 0x01`, `version_minor = 0x00` (i.e.,
  descriptor format version `1.0`).

A change to the descriptor encoding falls into one of three classes (see
[§ Change Classification](#change-classification)).

A `version_minor` increment is REQUIRED when any of the following changes are
made to the descriptor encoding:

- A new optional flag bit is defined in the descriptor `flags` field.
- A new optional descriptor section is defined (gated by a new flag bit).
- A new optional trailing field is appended to an existing section without
  changing the offset of any field defined in the previous minor version.
- A new `type_tag`, `layout_tag`, or `device_tag` value is allocated within
  the existing tag space.
- A new quantization scheme tag is allocated within the existing scheme tag
  space.

A `version_major` increment is REQUIRED when any of the following changes are
made:

- An existing field's offset, type, or semantics changes.
- An existing flag bit's meaning changes.
- An existing tag value's meaning changes.
- A previously OPTIONAL field becomes REQUIRED, or vice versa.
- A previously valid encoding is forbidden.
- The minimum descriptor length grows.

Reader behaviour for the descriptor version is defined in
[§ Compatibility Matrix](#compatibility-matrix). The normative reader rules are
established in `metadata.md` § Fixed Header and § Version Compatibility; this
section does not duplicate them, it only classifies the changes that justify
each rule.

### Container Version (`container_version_major` / `container_version_minor`)

The container version describes the wire encoding of the Hurray file format as
defined in `file-format.md`.

- **Location**: file header, byte `8` (`container_version_major`, `uint8`) and
  byte `9` (`container_version_minor`, `uint8`). See `file-format.md` § File
  Header.
- **Current values**: `container_version_major = 0x01`,
  `container_version_minor = 0x00` (i.e., container format version `1.0`).

The container version is **independent** of the descriptor version. A file at
container version `1.0` MAY contain tensor descriptors at descriptor version
`1.0`, `1.1`, or any later compatible descriptor version. A reader MUST NOT
infer the descriptor version from the container version, or vice versa.

A `container_version_minor` increment is REQUIRED when any of the following
changes are made to the container encoding:

- A new optional file-level flag bit is defined in `file_flags`.
- A new optional KV value tag is allocated within the existing KV value tag
  space.
- A new optional trailing field is appended to the file header, the trailer,
  or an index entry without changing the offset of any field defined in the
  previous minor version.

A `container_version_major` increment is REQUIRED when any of the following
changes are made:

- An existing field's offset, type, or semantics changes in the file header,
  index, or trailer.
- The trailer length changes.
- An existing flag bit's meaning changes.
- The data buffer alignment minimum or maximum bounds change.
- A previously OPTIONAL section becomes REQUIRED, or vice versa.

The normative reader rules for the container version are established in
`file-format.md` § File Header.

### Quantization Scheme Version (`scheme_version`)

The quantization scheme version describes the wire encoding of an individual
quantization scheme's parameter payload as defined in `quantization.md`.

- **Location**: quantization descriptor header, byte `1` (`scheme_version`,
  `uint8`). See `quantization.md` § Descriptor Header.
- **Current values**: `scheme_version = 0x01` for every scheme defined in
  `quantization.md` and its sub-files.

Each `(scheme_tag, scheme_version)` pair is **independently versioned**.
Incrementing the version of one scheme does not affect any other scheme.

A `scheme_version` increment is REQUIRED when any of the following changes are
made to a scheme's parameter encoding:

- A new field is added to the scheme's payload.
- A reserved byte or flag bit is repurposed.
- The interpretation of an existing field changes.
- The dequantization formula changes (this is also a backward-incompatible
  change at the descriptor level — see below).

Adding a brand-new scheme does not increment any existing scheme's
`scheme_version`; it allocates a new `scheme_tag` and starts that tag at
`scheme_version = 0x01`. Allocating the new `scheme_tag` requires a descriptor
minor version increment, as established in `quantization.md` § Version
Compatibility.

The normative reader rule for the scheme version is established in
`quantization.md` § Descriptor Header: a reader MUST reject a descriptor whose
`scheme_version` exceeds the highest version defined in this specification for
the given `scheme_tag`.

> **Note (non-normative):** Removing a scheme or changing the dequantization
> formula for an existing `(scheme_tag, scheme_version)` pair is a
> backward-incompatible change to the descriptor encoding and therefore
> requires a descriptor `version_major` increment, not merely a
> `scheme_version` bump. The `scheme_version` field exists to support
> backward-compatible additions to a scheme's parameter encoding within a
> single descriptor major version.

---

## Change Classification

Changes to any of the three version axes fall into one of three classes.

| Class | Wire-format effect | Required version bump |
|-------|-------------------|-----------------------|
| **MAJOR** | Backward-incompatible. A reader built for the previous major version cannot correctly parse data written at the new major version. | Major version increment on the affected axis. |
| **MINOR** | Backward-compatible addition. A reader built for the previous minor version can still parse data written at the new minor version (ignoring new optional content). | Minor version increment on the affected axis. |
| **PATCH** | No wire-format change. Documentation clarifications, editorial fixes, examples, and test vector additions. | No version increment. |

### Examples of MAJOR changes

- Removing a field, flag bit, or tag value.
- Changing the offset, type, or semantics of an existing field.
- Adding a new mandatory field that all writers MUST emit and all readers MUST
  parse.
- Changing the dequantization formula for an existing
  `(scheme_tag, scheme_version)` pair.
- Changing the meaning of an existing flag bit.
- Changing the encoding of magic bytes, the descriptor length field, or any
  field a reader is required to consult before parsing the rest of a structure.

### Examples of MINOR changes

- Adding a new optional flag bit to a flags field.
- Adding a new optional section gated by a new flag bit.
- Allocating a new `type_tag`, `layout_tag`, `device_tag`, KV value tag, or
  quantization `scheme_tag` within the existing tag space.
- Adding a new optional trailing field to an existing section without
  disturbing the offsets of fields defined in the previous minor version.

### Examples of PATCH changes

- Clarifying ambiguous wording.
- Adding non-normative notes or worked examples.
- Adding test vectors.
- Fixing typos that do not affect interpretation.

---

## Compatibility Matrix

Let `R` denote the highest version a reader supports on a given axis, and `W`
denote the version recorded in the data being read on the same axis. The
compatibility rules apply identically to all three axes (descriptor, container,
and quantization scheme).

| Relationship | Reader behaviour |
|--------------|------------------|
| `R.major < W.major` | MUST reject the data. |
| `R.major > W.major` | MUST reject the data, unless this specification defines an explicit migration path for the older major version. No such migration path is defined for any axis at version `1.x`. |
| `R.major == W.major`, `R.minor < W.minor` | MUST parse all fields defined at `R.minor`. MUST ignore any optional fields, flag bits, sections, or tag values that are not defined at `R.minor`. MUST use the relevant length prefix (`descriptor_length`, `quantization_length`, or section length in the file index) to skip unknown trailing content. |
| `R.major == W.major`, `R.minor >= W.minor` | Normal read. All fields written at `W.minor` are defined at `R.minor` or earlier. |

A reader that encounters a flag bit, tag value, or scheme not defined at its
supported minor version MUST treat the affected feature as unknown:

- For an unknown **descriptor flag bit**: a reader MUST reject the descriptor,
  because flag bits in descriptor minor version `1.0` are required to be `0`
  for all reserved positions, so an unknown flag bit at `1.x > 0` indicates the
  writer used a feature the reader does not understand.
- For an unknown **type tag** (outside the extension range `0xF0`–`0xFE`): a
  reader MUST reject the descriptor.
- For an unknown **layout tag** (outside the extension range): a reader MUST
  reject the descriptor.
- For an unknown **quantization scheme tag**: a reader MUST reject the
  descriptor unless operating in permissive mode, as defined in
  `quantization.md` § Descriptor Header.
- For an unknown **KV value tag**: a reader MUST reject the file, as defined
  in `file-format.md` § KV Value Types.
- For an unknown **file flag bit**: a reader MUST reject the file, as defined
  in `file-format.md` § File Flags.

> **Note (non-normative):** The asymmetry between "unknown trailing content
> MUST be ignored" and "unknown flag bits / tags MUST be rejected" is
> intentional. Trailing content is opt-in by definition: it can only be
> reached via a length prefix the reader already trusts. Unknown flag bits or
> tags signal that the writer relied on a feature whose semantics are
> unknowable to the reader, which would lead to silent misinterpretation of
> the data buffer.

---

## Writer Requirements

A conforming writer:

- MUST set `version_major` and `version_minor` to the highest descriptor
  version whose features it actually uses. A writer that only uses descriptor
  `1.0` features MUST emit `version_major = 0x01`, `version_minor = 0x00`,
  even if the writer's implementation is aware of higher minor versions.
- MUST set `container_version_major` and `container_version_minor` to the
  highest container version whose features it actually uses, under the same
  rule.
- MUST set `scheme_version` to the lowest scheme version whose features it
  actually uses for each emitted quantization descriptor.
- MUST NOT set any reserved flag bit, reserved tag value, or reserved byte
  unless this specification has defined the corresponding feature.

> **Note (non-normative):** Writers SHOULD emit the lowest version number
> compatible with the features they use, to maximise the population of readers
> that can consume the output. A writer that knows it only emits descriptor
> `1.0` features SHOULD emit `1.0`, not `1.5`, even if the writer's library
> supports both.

---

## Version Registry

Current values for every version field defined in this specification:

| Axis | Field | Location | Current value |
|------|-------|----------|---------------|
| Descriptor | `version_major` | tensor descriptor, byte `4` | `0x01` |
| Descriptor | `version_minor` | tensor descriptor, byte `5` | `0x00` |
| Container | `container_version_major` | file header, byte `8` | `0x01` |
| Container | `container_version_minor` | file header, byte `9` | `0x00` |
| Quantization scheme | `scheme_version` (per-tensor affine, `scheme_tag = 0x01`) | quantization descriptor, byte `1` | `0x01` |
| Quantization scheme | `scheme_version` (per-channel affine, `scheme_tag = 0x02`) | quantization descriptor, byte `1` | `0x01` |
| Quantization scheme | `scheme_version` (per-block affine, `scheme_tag = 0x03`) | quantization descriptor, byte `1` | `0x01` |
| Quantization scheme | `scheme_version` (NF4, `scheme_tag = 0x04`) | quantization descriptor, byte `1` | `0x01` |
| Quantization scheme | `scheme_version` (MXFP, `scheme_tag = 0x05`) | quantization descriptor, byte `1` | `0x01` |

> **Note (non-normative):** The full set of allocated `scheme_tag` values is
> maintained in `quantization.md` § Scheme Tag Space and its sub-files. This
> registry tracks only the version of each scheme's wire encoding, not the
> tag allocation itself. When a new scheme is added to `quantization.md`, a
> corresponding row MUST be added to this table.

---

## Relationship to Other Sections

- **`metadata.md`** establishes the normative reader rules for the descriptor
  version and the structure of the descriptor that the version describes.
- **`file-format.md`** establishes the normative reader rules for the
  container version and the structure of the file that the version describes.
- **`quantization.md`** establishes the normative reader rules for the
  quantization scheme version and defines which scheme tags exist.
- **`element-types.md`**, **`memory-layout.md`**, and the per-layout files
  under `layouts/` define the tag spaces (type tags, layout tags) whose
  allocation is gated by the descriptor version policy stated in this section.
