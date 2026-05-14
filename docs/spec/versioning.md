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

Minor amendments MUST comply with the [Evolvability Contract § Spec
Amendment Rules](#spec-amendment-rules) (in particular S6 — no fixed-offset
additions) and the defaults table in [§ Defaults for Appended Trailing
Fields](#defaults-for-appended-trailing-fields) (S4).

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

## Extensibility Contract

This section states the **stability guarantees** the Hurray format makes to
implementors and downstream tools across the lifetime of major version `1.x`.
It is the normative counterpart to the "Extensible" property listed in
[`README.md`](README.md) § Scope and Goals. The guarantee period begins at
descriptor and container version `1.0`; pre-`1.0` drafts are explicitly
excluded (see [§ Out of Scope](#out-of-scope) below).

The per-tag-space mechanics — reserved-range layouts, extension-range
boundaries, and tag allocation tables — are normatively defined in the
per-section files. This section does not restate those rules; it cross-
references them:

- Element type tag space and extension range: see
  [`element-types.md`](element-types.md) § Type Tag Space.
- Layout tag space, extension range, and per-layout reserved bytes: see
  [`memory-layout.md`](memory-layout.md) § Layout Tag Space and the per-layout
  files under [`layouts/`](layouts/).
- Device tag space and extension range: see
  [`buffer-protocol.md`](buffer-protocol.md) § Device Tag Space.
- Quantization scheme tag space, scheme reserved ranges, and permissive-mode
  parsing: see [`quantization.md`](quantization.md) § Scheme Tag Space and
  § Descriptor Header.
- KV value tag space and file flag bits: see
  [`file-format.md`](file-format.md) § KV Value Types and § File Flags.

### Commitments

For the lifetime of major version `1.x`, this specification commits to the
following invariants. Conforming readers, writers, and downstream tools MAY
rely on every one of them.

1. **Reserved tag ranges are stable.** Reserved tag ranges defined by this
   specification — across every public tag space (element type, layout,
   device, quantization scheme, KV value, and flag bits) — MUST NOT be
   repurposed, narrowed, or removed within major version `1.x`. A reserved
   range allocated at `1.0` MUST remain reserved with the same boundaries
   and the same intended use class throughout `1.x`.
2. **Implementation-private ranges remain implementation-private.** The
   implementation-private ranges `0xF0`–`0xFE` for element type tags, layout
   tags, and device tags MUST remain implementation-private for the lifetime
   of major version `1.x`. This specification MUST NOT allocate any named
   public value into a private range, and a future minor revision MUST NOT
   reclaim a private range for public allocation. Equivalent
   implementation-private ranges defined for other tag spaces in their per-
   section files (e.g., the quantization scheme private range in
   `quantization.md`) are subject to the same guarantee.
3. **Reserved flag bits remain available for feature gating.** Reserved flag
   bits in the tensor descriptor header `flags` field, the file header
   `file_flags` field, and every per-section flag field defined in this
   specification MUST remain available for backward-compatible feature gating
   throughout major version `1.x`. A reserved flag bit MUST NOT be removed
   or have its reserved status withdrawn within `1.x`; when allocated, it
   MUST be allocated as an optional feature gated by a minor version
   increment, per [§ Change Classification](#change-classification).
4. **Every variable-length section is length-prefixed.** Every variable-length
   section in the tensor descriptor and the file format MUST carry a length
   prefix that allows an older reader to skip unknown trailing content
   without rejecting the structure. A future minor revision of the descriptor
   or container MUST NOT introduce a variable-length section that lacks such
   a length prefix. The applicable prefixes (`descriptor_length`,
   `quantization_length`, file index entry section lengths, and any
   equivalent fields defined in future minor revisions) MUST continue to
   bound exactly the bytes whose interpretation may change.
5. **Permissive-mode parsing is preserved.** A reader MUST be able to parse
   the tensor descriptor's shape and buffer table even when it cannot
   interpret an unknown layout tag or an unknown quantization scheme tag.
   This specification MUST NOT, within major version `1.x`, introduce a
   change that requires interpreting a layout tag or a quantization scheme
   tag in order to recover the shape, rank, element type, or buffer table.
   The exact behaviour of permissive mode for each tag space is defined in
   its per-section file (see [`quantization.md`](quantization.md) §
   Descriptor Header and [`memory-layout.md`](memory-layout.md) § Layout
   Tag Space).
6. **The three version axes evolve independently.** The descriptor,
   container, and per-quantization-scheme versions MUST evolve independently.
   Adding a new feature on one axis MUST NOT force a version bump on the
   others. A descriptor minor increment MUST NOT require a container minor
   increment; a scheme-version bump for a single `(scheme_tag,
   scheme_version)` pair MUST NOT require any change to the descriptor or
   container version; a container minor increment MUST NOT require any
   change to the descriptor or to any scheme version. This commitment
   complements [§ Version Axes](#version-axes), which establishes the
   axes themselves.
7. **Public tag allocation goes through the spec amendment process.** Any
   new public tag value — including a new element type tag, layout tag,
   device tag, quantization scheme tag, KV value tag, or named flag bit —
   MUST be added by a spec amendment that increments the appropriate minor
   version (per [§ Change Classification](#change-classification)). New
   public named values MUST NOT be added by implementations independently of
   the specification. Implementations that need a private value MUST use the
   appropriate implementation-private range and remain subject to commitment
   (2) above.

> **Note (non-normative):** Together, commitments (1)–(7) form the stable
> "extension surface" that downstream array databases, runtime registries,
> language bindings, and compatibility-testing harnesses can build against.
> A new tag allocated at descriptor `1.5` is guaranteed to retain its
> meaning, encoding, and tag-space neighbourhood through descriptor `1.99`.

### Out of Scope

The Extensibility Contract is a finite guarantee. The following properties
are deliberately **not** guaranteed by this specification, and conforming
readers, writers, and tools MUST NOT rely on them:

1. **Forward compatibility across major versions is out of scope.** This
   contract applies within major version `1.x` only. A reader MUST reject
   data whose major version on any axis exceeds the reader's supported
   major version, per [§ Compatibility Matrix](#compatibility-matrix). The
   guarantees in [§ Commitments](#commitments) do not transfer to major
   version `2.x` or beyond; a future major version MAY revise tag spaces,
   reserved ranges, flag bits, and length-prefix conventions without
   preserving `1.x` semantics.
2. **Interpretation of unknown content is out of scope.** Permissive-mode
   parsing allows a reader to extract shape, rank, element type, and buffer
   table when a layout tag or scheme tag is unknown. It does not authorise
   the reader to interpret the associated data buffer. A reader that does
   not understand the layout tag or quantization scheme tag of a tensor
   MUST NOT attempt to dereference, dequantize, or otherwise interpret the
   data buffer; doing so would constitute silent misinterpretation, which
   this contract explicitly forbids.
3. **Interoperability of implementation-private tags is out of scope.** Tag
   values in any implementation-private range (e.g., `0xF0`–`0xFE` for
   element type, layout, and device tags, and equivalent ranges in other tag
   spaces) MUST NOT be exchanged between independent implementations without
   an out-of-band agreement on their meaning. Encountering a private-range
   value from an unknown source MUST be treated as an unknown tag under the
   rules in [§ Compatibility Matrix](#compatibility-matrix); permissive-mode
   parsing remains available where defined, but no semantic interoperability
   is implied.
4. **A runtime plugin or codec mechanism is out of scope.** This
   specification provides no registered plugin interface, no dynamic codec
   loader, and no implementation-supplied extension descriptor. New element
   types, layouts, devices, quantization schemes, and KV value tags MUST be
   added by a spec amendment under commitment (7); they MUST NOT be added by
   a runtime registration call, a sidecar manifest, or any implementation-
   private mechanism.
5. **User-defined non-numeric element types are out of scope.** The element
   type extension range defined in [`element-types.md`](element-types.md)
   exists only for new numeric encodings (integer, floating-point, and
   numerically-equivalent storage types). It MUST NOT be used to encode
   strings, structured records, opaque blobs, references to other tensors,
   or any other non-numeric content. Carrying non-numeric data over Hurray
   is the responsibility of the KV metadata section in
   [`file-format.md`](file-format.md), not of the element type system.
6. **Back-compatibility of pre-`1.0` drafts is out of scope.** The
   Extensibility Contract begins at descriptor and container version `1.0`.
   Pre-`1.0` draft versions of this specification MAY have used different
   tag allocations, reserved ranges, or wire encodings, and conforming
   `1.x` readers and writers MUST NOT assume any compatibility with them. A
   reader encountering data that claims a pre-`1.0` version SHOULD reject
   it; it MAY accept it only if the reader was explicitly configured to
   consume draft data and has applied an implementation-defined migration.

> **Note (non-normative):** The boundary between "what we promise" and
> "what we deliberately do not promise" is what makes the Extensibility
> Contract usable. Without the out-of-scope list, downstream tooling could
> infer guarantees that the format cannot actually defend — for example,
> assuming that a private-range tag from one runtime is meaningful in
> another, or that permissive-mode parsing implies safe buffer access.
> Calling out non-commitments is part of the contract.

---

## Evolvability Contract

Evolvability and extensibility are complementary but distinct. The
Extensibility Contract names *where* the format can grow — the extension
surface of reserved tag ranges, implementation-private ranges, reserved flag
bits, and length-prefixed sections. The Evolvability Contract names *how*
that growth is staged across versions: which compatibility direction a
reader can rely on when it encounters data from a different minor version on
the same major axis, and which spec-amendment moves are admissible at each
step. Together, the two contracts form the stability surface that downstream
implementations build against.

### Compatibility Direction

- **BACKWARD (CD1):** A reader at minor `M` MUST correctly parse data
  written at any minor `N ∈ {0, …, M}` on the same major version on the
  relevant axis.
- **FORWARD_ADDITIVE (CD2):** A reader at minor `M` reading data written at
  minor `N > M` on the same major version MUST correctly parse every field
  defined at minor `M` — the fixed header, the buffer table, and every
  length-prefixed section whose gating flag bit is defined at minor `M`,
  including trailing bytes of those sections up to the prefix length. The
  reader MUST reject the data if it encounters any flag bit or public tag
  value not defined at minor `M`, except within the permissive-mode
  exceptions for layout tags and quantization scheme tags defined in
  [`memory-layout.md`](memory-layout.md) § Layout Tag Space and
  [`quantization.md`](quantization.md) § Descriptor Header.

  > **Note (non-normative):** The asymmetry between rejecting new
  > flag-gated sections and skipping additive trailing bytes is
  > intentional. An unknown flag bit may gate a new section whose
  > presence changes the semantics of the data buffer; the reader has no
  > safe way to ignore such a gate. Additive trailing bytes inside an
  > already-known length-prefixed section, by contrast, extend a
  > structure whose framing the reader already understands and can step
  > past using the existing length prefix.

- **CD3:** A reader supporting major `K` on a given axis MUST reject data
  whose major version on that axis is `K + 1` or higher.
- **CD4:** Cross-major reading is not automatic. A `K+1`-major reader is
  not required to read `K`-major data; it MAY do so only via the migration
  specification required by S5 below.

### Writer Evolution Rules

- **W3:** A writer MUST NOT emit a deprecated public tag value or flag
  bit when a non-deprecated equivalent exists.
- **W4:** When a writer appends an optional trailing field to an
  existing length-prefixed section under a `version_minor` increment,
  the writer MUST emit that field at the documented offset and MUST
  update the enclosing length prefix accordingly. A writer MUST NOT
  emit a partial trailing field.

> **Note (non-normative):** Rules W1 and W2 are the writer requirements
> stated in [§ Writer Requirements](#writer-requirements) below. W3 and
> W4 extend them with the evolution-specific constraints needed by the
> Evolvability Contract.

### Reader Evolution Rules

- **R3:** A deprecated public tag value MUST be treated as semantically
  equivalent to its non-deprecated definition. Deprecation MUST NOT
  change a value's wire semantics.
- **R4:** When a reader at minor `M` encounters a length-prefixed
  section shorter than the section length defined for minor `M` (i.e.,
  data written at some minor `N < M`), the reader MUST treat every
  field beyond the data's section length as carrying its documented
  default from the defaults table in
  [§ Defaults for Appended Trailing Fields](#defaults-for-appended-trailing-fields)
  below.

> **Note (non-normative):** Rules R1 and R2 are the reader behaviours
> stated in [§ Compatibility Matrix](#compatibility-matrix) above. R3
> and R4 extend them with the evolution-specific constraints needed by
> the Evolvability Contract.

### Spec Amendment Rules

- **S1:** New public tag values MUST be allocated from the documented
  public reserved range of their tag space (see [§ Extensibility
  Contract](#extensibility-contract), commitments 1 and 7).
- **S2 (Anti-rebind):** An allocated public tag value MUST NOT be
  rebound to a different meaning within the same major version, even
  after deprecation.
- **S3 (Deprecation convention):** A deprecated tag table entry MUST be
  marked "deprecated since 1.N" and SHOULD point to a replacement.
  Deprecation is a writer-facing signal only — see R3 for reader
  obligations.
- **S4 (Defaults for trailing fields):** Any new optional trailing
  field appended to an existing section MUST have a normatively
  documented default in the same minor revision. The default MUST be
  recorded in
  [§ Defaults for Appended Trailing Fields](#defaults-for-appended-trailing-fields)
  below.
- **S5 (Migration commitment):** A future major version MUST be
  accompanied by a normative migration specification mapping the prior
  major version's encoding onto the new major version's encoding for
  every field, flag bit, and tag value that survives the transition.
- **S6 (No fixed-offset additions):** A new field in a minor revision
  MUST be gated by a flag bit, a tag value, or a length-prefixed
  trailing extension. A minor amendment MUST NOT allocate a new field
  at a fixed offset that an older reader would parse as part of an
  existing structure.

### Defaults for Appended Trailing Fields

The following table records the default value that a reader at the prior
minor version conceptually sees for each trailing field appended under a
minor bump. This table is normative for R4. It MUST be updated as part
of any spec amendment that adds a trailing field.

| Section | Field | Introduced in | Default for prior-minor readers |
|---------|-------|---------------|----------------------------------|
| *(empty — no trailing fields have been appended at `1.0`)* | | | |

> **Note (non-normative):** The table is currently empty because no
> optional trailing fields have been appended to any section at
> descriptor version `1.0`. The first such amendment MUST add its
> entry here.

### Anti-Patterns

> **Note (non-normative):** This sub-section is non-normative
> commentary; the operative prohibition lives in the bullet items
> themselves, which use MUST NOT to bind future spec amendments to
> the choice made here.

- **Per-field numeric tagging (Protobuf, Thrift):** imposing a tag and
  a length-or-type word on every field defeats fixed-offset zero-copy
  reads and forces a per-field decode loop even for readers that need
  only a small subset of fields. This approach MUST NOT be adopted as
  the descriptor's encoding strategy within major version `1.x`.
- **vtables (FlatBuffers):** adding a per-object vtable indirection
  breaks single-pass streamability (the vtable is referenced by an
  offset that may point backward relative to the object) and adds
  extra cache-line traffic per field access. This approach MUST NOT
  be adopted as the descriptor's encoding strategy within major
  version `1.x`.
- **Hurray's evolvability mechanism, by contrast, is the flag-bit +
  length-prefix model:** new sections are gated by flag bits and
  framed by length prefixes; additive trailing fields live behind
  the enclosing length prefix; tag spaces grow only within their
  documented reserved ranges. This combination preserves fixed-offset
  zero-copy reads for every field defined at the reader's minor
  version while still admitting backward-compatible extension.

### Worked Example

> **Note (non-normative):** This sub-section is illustrative. The
> hypothetical `bias_correction` field described below is not part of
> the format at descriptor version `1.0`; no MXFP trailing field has
> been appended at `1.0`. The example shows how W4, R4, and S4 work
> together when a future minor revision appends a trailing field.

**Worked Example: MXFP Scheme Evolution Across Minor Versions**

Suppose a future descriptor version `1.1` appends an optional
`bias_correction` field (`float32`, 4 bytes) to the MXFP quantization
scheme payload (see [`quantization/mxfp.md`](quantization/mxfp.md) §
Binary Encoding) immediately after the existing `scale_buffer_index`
field, extending the MXFP descriptor from 16 bytes to 20 bytes. The
following steps illustrate the contract:

1. **Spec amendment (S4).** The `1.1` amendment records
   `bias_correction` in the defaults table in
   [§ Defaults for Appended Trailing Fields](#defaults-for-appended-trailing-fields)
   with default value `0.0` (IEEE 754 `float32` zero).
2. **`1.1` writer (W4).** The writer emits the full 20-byte MXFP
   payload including `bias_correction`, and sets the enclosing
   `quantization_length` prefix to `20` (plus any further trailing
   bytes added by an even later minor revision the writer
   participates in).
3. **`1.0` reader against `1.1` data (CD2, R4).** The reader sees
   `quantization_length ≥ 20` but parses only the first 16 bytes
   defined at `1.0`. The trailing 4 bytes are skipped via the
   length prefix; the reader synthesises `bias_correction = 0.0`
   per the defaults table, even though it never actually reads the
   bytes (a `1.0` reader does not know `bias_correction` exists, so
   the synthesised default is only observable to a downstream
   `1.1` consumer that subsequently reparses the data).
4. **`1.1` reader against `1.1` data.** The reader parses all
   20 bytes, including `bias_correction`, and interprets the field
   directly.
5. **`1.1` reader against `1.0` data (CD1, R4).** The reader sees
   `quantization_length = 16` (the `1.0` MXFP length). Every field
   beyond the data's section length is treated as carrying its
   documented default; the reader synthesises
   `bias_correction = 0.0` per the defaults table and proceeds with
   dequantization as if the writer had emitted the default
   explicitly. No descriptor rejection occurs.

This example does not introduce any normative requirement that is not
already stated by W4, R4, and S4 — it only illustrates how those rules
compose.

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
