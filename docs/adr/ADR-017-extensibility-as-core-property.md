# ADR-017: Extensibility as a Core Property

## Status
Accepted

## Context

`docs/spec/README.md` § Scope and Goals already includes the line "Be extensible
without breaking existing readers", but the term is undefined. The mechanisms that
implement extensibility are scattered across multiple sections:

- **Tag-space partitioning** with a private extension range (`0xF0`–`0xFE`) for
  element types (`element-types.md`), layout tags (`memory-layout.md`), and
  device tags (`buffer-protocol.md`).
- **Reserved-for-future-spec ranges** (e.g. `0x80`–`0xEF`) that no implementation
  may consume.
- **Reserved flag bits** in the descriptor header for future optional sections.
- **Length-prefixed sections** that allow unknown trailing bytes to be skipped
  by older readers.
- **Independent version axes** (descriptor, container, quantization scheme) per
  `versioning.md`, with a `MAJOR` / `MINOR` / `PATCH` change classification.
- **Permissive-mode parsing** for unknown layout and quantization scheme tags.
- **`ExtensionTypeDescriptor`** for custom numeric types within the extension tag range.

The risk of leaving extensibility implicit:

- Future spec authors and implementors discover the extensibility contract by
  reverse-engineering the tag tables and version policy, and may inadvertently break it.
- External implementors evaluating Hurray cannot quickly determine the extensibility posture.
- The boundary between "private extensions are forever" and "private extensions are
  unsupported" is ambiguous.

The risk of making it explicit:

- A named guarantee invites maximalist interpretations ("a v1 reader will accept
  anything I put in the descriptor").
- It may be read as a commitment that every reserved range will eventually be filled,
  or that the spec will accommodate any feature a downstream user wants.
- It locks the existing extension-point inventory into a contractual surface that
  cannot be reduced without a major version bump.

## Decision

Hurray adopts **extensibility** as a named core property of the format and protocol.
The property is defined precisely and narrowly to avoid over-promising.

### What Hurray commits to

The format MUST provide the following extension points, and these extension points
MUST remain stable across all minor versions of a given major version:

1. **Reserved tag ranges** in the element-type, layout, device-tag, and
   quantization-scheme tag spaces. The spec MUST NOT repurpose, narrow, or remove
   any reserved range within a major version.
2. **Implementation-private tag ranges** (`0xF0`–`0xFE` for element types, layouts,
   and device tags). These ranges MUST remain implementation-private for the lifetime
   of major version `1.x`. The spec MUST NOT allocate any named value into a private range.
3. **Reserved flag bits** in the descriptor header, file header, and per-section flag
   fields. Bits reserved at version `1.0` MUST remain available for backward-compatible
   feature gating throughout `1.x`.
4. **Length-prefixed sections**. Every variable-length section in the descriptor and
   file format MUST carry a length prefix that enables an older reader to skip unknown
   trailing content without rejecting the whole structure.
5. **Permissive-mode parsing** for unknown layout tags and unknown quantization scheme
   tags. A reader MUST be able to parse the descriptor's shape and buffer table even
   when it cannot interpret the layout or scheme.
6. **Independent version axes**. The descriptor, container, and per-quantization-scheme
   versions MUST evolve independently. Adding a new feature on one axis MUST NOT force
   a version bump on the others.
7. **A spec amendment process for named values**. Any new public tag value (element
   type, layout, device, quantization scheme, KV value tag, or flag bit) MUST go
   through a spec amendment that increments the appropriate minor version. New named
   values are added by the spec, not by implementations.

### What Hurray explicitly does NOT commit to

1. **No forward compatibility across major versions.** A reader MUST reject data at
   a higher major version on the relevant axis.
2. **No commitment to interpret unknown content.** Permissive mode allows the descriptor
   to be parsed; it does not allow the data buffer to be interpreted. A reader that does
   not understand a quantization scheme MUST NOT attempt to dequantize.
3. **No interoperability of private-range values.** Tags in the private range MUST NOT
   be exchanged between independent implementations without out-of-band agreement.
4. **No runtime plugin or codec mechanism.** The extension surface is a curated tag
   space, not a runtime-loadable codec pipeline. Adding a layout or scheme requires
   a spec amendment, not a registered plugin.
5. **No user-defined non-numeric element types.** The extension range exists for new
   numeric encodings, not for arbitrary user types.
6. **No back-compatibility guarantee for pre-`1.0` drafts.** The extensibility
   guarantee begins at `1.0`.

### Where this is documented

1. **`docs/spec/README.md` § Scope and Goals** — extensibility is added as a named
   bullet replacing the current "Be extensible without breaking existing readers" line,
   with a pointer to the "Extensibility Contract" section in `versioning.md`.
2. **`docs/spec/versioning.md`** — a new "Extensibility Contract" section is added that
   enumerates the seven commitments and six non-commitments above in RFC 2119 normative
   language. It cross-references the per-tag-space rules already in `element-types.md`,
   `memory-layout.md`, `buffer-protocol.md`, and `quantization.md` rather than
   restating them.
3. **No new normative text** is needed in the individual per-section spec files.

## Alternatives Considered

**Leave extensibility implicit.** Continue relying on the one-line goal in README.md and
per-section tag tables. Rejected because implementors discover the contract piecemeal, and
the boundary between "private ranges are forever" and "private ranges are unsupported"
remains ambiguous. The cost of naming it is one section in `versioning.md`.

**Name it as a goal but add no normative text.** Add the bullet to the README without a
normative contract section. Rejected — a goal without normative backing is not a contract;
spec-checker cannot verify compliance against an unstated rule.

**Define extensibility as a runtime plugin contract (Zarr v3-style).** Treat extensions as
runtime-loadable codecs. Rejected — conflicts with zero-copy, language-agnostic, and
streaming goals. Changes Hurray's category from "stable interchange format" to "extensible
compute substrate."

**Narrow the contract to reserved bits and tag ranges only, excluding permissive mode and
independent version axes.** Rejected — permissive mode and independent version axes are
already normative; excluding them from the named property splits the extensibility surface
across "named guarantees" and "unnamed guarantees" for no benefit.

## Consequences

- External evaluators find the extensibility posture in one place (`versioning.md`
  § Extensibility Contract).
- Spec-checker audits gain a concrete normative anchor: every amendment that touches a
  reserved range, private range, length-prefixed section, or version axis is testable
  against the contract.
- The non-commitments are explicit, providing documented responses to requests for
  runtime-loadable codecs, cross-vendor private tags, or user-defined element types.
- The contract creates a documented expectation that reserved ranges remain reserved across
  `1.x`. The spec MUST NOT recover bytes from a reserved range mid-major-version.
- This decision is **non-breaking**: it documents existing behaviour and adds no new
  wire-format constraints.

### Follow-up work

1. **`format-spec-writer`**: edit `docs/spec/README.md` § Scope and Goals to replace
   "Be extensible without breaking existing readers" with the named extensibility bullet.
2. **`format-spec-writer`**: add "Extensibility Contract" section to
   `docs/spec/versioning.md` (after § Compatibility Matrix, before § Writer Requirements)
   with the seven commitments and six non-commitments in RFC 2119 language, cross-referencing
   the per-section files.
3. **`spec-checker`**: add a checklist item "Does this amendment preserve the
   Extensibility Contract?" to `docs/SPEC_CHECKLIST.md` once the section is in place.
4. No code changes required in any `hurray-*` crate.
