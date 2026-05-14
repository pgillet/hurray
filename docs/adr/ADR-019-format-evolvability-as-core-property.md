# ADR-019: Format Evolvability as a Core Property

## Status

Proposed

## Context

ADR-017 named **extensibility** as a core property of Hurray and codified the stable extension surface (reserved tag ranges, private ranges, reserved flag bits, length-prefixed sections, permissive-mode parsing, independent version axes, spec-amendment process). Extensibility — also called **evolvability**, modifiability, or plasticity depending on the literature — is the single property of a format that makes it easy for engineers to change it in the future, adapting it for unanticipated use cases as requirements change. ADR-017 defined the extension *surface*; it did not formally specify the operational *rules* for using that surface: how backward and forward compatibility are achieved, what a reader is obliged to do when it meets data written by a newer minor version, how tags are added, and how an enum variant is safely deprecated and removed.

The mechanics required for safe evolution are largely already in place:

- Magic bytes (`HRRY`, `HRRYFILE`).
- Three independent version axes (descriptor `(major, minor)`, container `(major, minor)`, per-scheme `scheme_version`).
- `descriptor_length` in bytes 6–9 (self-delimiting from the first 10 bytes).
- Length prefixes on every optional section, gated by a flag bit.
- Reserved tag ranges and reserved flag bits (MUST be zero until allocated).
- Permissive-mode parsing for unknown layout and quantization scheme tags.
- A change-classification table (MAJOR / MINOR / PATCH) and compatibility matrix in `versioning.md`.

A focused review against the evolution rules of Protobuf, Avro, Thrift, FlatBuffers, and the DDIA Chapter 4 framework surfaced seven concrete gaps:

1. **Compatibility direction is never named with industry-standard vocabulary.** The mechanics implement backward compatibility within `1.x` and a particular shape of forward compatibility, but the spec never uses those words.
2. **Forward compatibility is asymmetric and undocumented.** Inside an existing length-prefixed section, an old reader silently skips trailing bytes added in a later minor version (forward-compatible). For a new optional section gated by a new flag bit, an old reader MUST reject (not forward-compatible). The asymmetry is correct, but it is not explained.
3. **No defaults table for newly appended trailing fields.** If `1.1` appends a field to an existing section, the spec does not say what value a `1.0` reader conceptually sees for that field.
4. **No anti-rebind rule.** Nothing prevents a future minor revision from reusing the same tag-byte value for a different meaning after the original is deprecated. Protobuf's `reserved` keyword closes this gap explicitly.
5. **No migration-spec commitment.** When `2.0` ships, the spec does not promise a normative migration path from `1.x`.
6. **No deprecation convention.** There is no wire signal or spec convention for "this tag value still works but writers SHOULD NOT emit it."
7. **No spec-writer guard rail** preventing a future minor amendment from allocating a new field at a fixed offset that an older reader would mis-parse (instead of properly gating it behind a flag bit or length-prefixed section).

The research also confirmed two design choices that are NOT appropriate for Hurray:

- **Protobuf per-field tagging** imposes per-field tag/length-type overhead on every read, defeating fixed-offset zero-copy.
- **FlatBuffers vtables** add an indirection table per object that breaks the "read the next field at a known offset" property required by streaming readers.

Hurray's existing **flag-bit + length-prefix** model is the correct analogue for a zero-copy, fixed-offset format and must remain the sole evolution mechanism for the optional surface of the descriptor and container.

This ADR closes the seven gaps above by naming the compatibility direction, adopting normative writer/reader/spec rules, and elevating evolvability to a named core property alongside extensibility. It introduces no new wire-format constraints; it documents and disciplines the existing one.

## Decision

Hurray formally specifies the operational rules for its extensibility property. **Format evolvability** — the same concept as extensibility, modifiability, or plasticity — is codified as Core Property #12: the normative contract that defines how the extension surface (Core Property #11) is used safely over time. The contract is defined by four compatibility direction declarations, two normative writer rules, two normative reader rules, six normative spec-amendment rules, and an explicit rejection of per-field tagging and vtables.

### CD — Compatibility direction (named and bounded)

Within major version `1.x`, on each of the three axes (descriptor, container, per-scheme):

- **CD1 — BACKWARD compatible within a major.** A reader at minor `M` MUST correctly parse data written at any minor `N ∈ {0, …, M}`. This is the existing behaviour; CD1 names it.
- **CD2 — FORWARD_ADDITIVE within a major.** A reader at minor `M` reading data written at minor `N > M` MUST correctly parse every field defined at minor `M` (fixed header up to `M`, buffer table, sections gated by flag bits defined at `M`, and trailing bytes of those sections up to the prefix length). The reader MUST reject the data when it encounters any flag bit or public tag value not defined at minor `M`, subject only to the permissive-mode exceptions defined in `quantization.md` § Descriptor Header and `memory-layout.md` § Layout Tag Space.
- **CD3 — No forward compatibility across major versions.** A reader supporting major `K` MUST reject data whose declared major version on the relevant axis is `K + 1` or higher.
- **CD4 — No automatic backward-transitive compatibility across major versions.** A reader supporting major `K + 1` is not required to read major-`K` data directly. Cross-major reading is supported only via the migration specification described in S5.

The CD2 name **FORWARD_ADDITIVE** is normative. It is defined as: *a reader correctly parses every part of newer-minor data that its own minor version defines, ignores trailing additive content inside known length-prefixed sections, and rejects newer-minor data that uses any feature gate (flag bit or public tag value) not defined at the reader's minor*. It is deliberately a stricter property than Protobuf's "ignore unknown fields" forward compatibility, because Hurray's zero-copy fixed-offset model cannot safely skip unknown gated sections whose semantics may affect the data buffer.

### W — Writer rules (new, normative)

- **W3.** A writer MUST NOT emit a deprecated public tag value or a deprecated flag bit when a non-deprecated equivalent exists.
- **W4.** When a writer appends an optional trailing field to an existing length-prefixed section under a `version_minor` increment, the writer MUST emit that field at the documented offset for that section and MUST update the enclosing length prefix accordingly. A writer MUST NOT emit a partial trailing field.

### R — Reader rules (new, normative)

- **R3.** A deprecated public tag value MUST be treated as semantically equivalent to its non-deprecated definition. Deprecation MUST NOT change a value's wire semantics; deprecation only signals "writers SHOULD prefer the replacement."
- **R4.** When a reader at minor `M` encounters a length-prefixed section whose encoded length is shorter than the length defined for minor `M` (because the data was written at minor `N < M`), the reader MUST treat every field beyond the data's section length as carrying its **documented default** for minor `M`. The defaults table is normative and MUST be maintained per S4.

### S — Spec amendment rules (new, normative)

- **S1.** A new public tag value MUST be allocated from the documented public reserved range of its tag space (element type, layout, device, quantization scheme, KV value, or named flag bit). It MUST NOT be allocated from an implementation-private range.
- **S2 — Anti-rebind.** An allocated public tag value MUST NOT be rebound to a different meaning within the same major version, even after it has been marked deprecated. Once allocated, a tag-byte value's meaning is fixed for the lifetime of `1.x`.
- **S3 — Deprecation convention.** When a public tag value or named flag bit is deprecated, the relevant tag table in the spec MUST mark it `deprecated since 1.N` and SHOULD include a pointer to its replacement. Deprecation is a writer-facing signal only; deprecation MUST NOT change reader behaviour (per R3).
- **S4 — Defaults for appended trailing fields.** Any new optional trailing field appended to an existing section under a minor bump MUST be accompanied by a normatively documented default in the same minor revision. The default is what a reader at the prior minor conceptually sees for that field, per R4. Defaults MUST be expressible without reference to other fields in the same descriptor unless that dependency is documented.
- **S5 — Major-version migration commitment.** A future major version (e.g., descriptor `2.0`, container `2.0`) MUST be accompanied by a normative migration specification mapping the prior major version's encoding to the new one for every tensor it can represent. The migration spec is normative for tools but does not impose a runtime obligation on a `2.x` reader to consume `1.x` data.
- **S6 — No fixed-offset additions in minor revisions.** A new field added under a minor bump MUST be gated by a flag bit (for a new section), by a tag value (for a new variant), or by a length-prefixed trailing extension of an existing section. A minor amendment MUST NOT allocate a new field at a fixed offset that an older reader would parse as part of an existing structure.

### Anti-patterns explicitly rejected

The following evolution mechanisms are incompatible with Hurray's zero-copy fixed-offset access model and MUST NOT be adopted within major version `1.x`:

- **Per-field tagging (Protobuf-style).** Every read would require scanning a tag/length/type sequence to locate the next field, defeating the property that a reader can compute the offset of any field from a small set of inputs (rank, layout tag, flag bits).
- **vtables (FlatBuffers-style).** Each tensor descriptor would carry a per-object virtual table indirecting every field access, losing single-pass streamability and the self-delimiting property.

Hurray's evolvability surface is the **flag-bit + length-prefix** model:

- New optional content is gated by a flag bit and length-prefixed. Old readers that don't know the bit reject; old readers that don't enter the section can skip it via the length prefix.
- Additive growth of an existing section happens by appending trailing bytes under a minor bump (W4 + R4 + S4 + S6). Old readers see defaults for the appended fields; new readers parse them.
- Tag spaces grow only within reserved ranges, never by rebinding (S1 + S2).

### Resolutions of the open questions

- **OQ-A (spec fingerprint KV entry).** **Deferred, non-blocking.** A future `hurray.spec_fingerprint` KV entry MAY be defined in `file-format.md` § KV Value Types for archival forensics. Not part of this ADR's normative scope.
- **OQ-B (descriptor header reserved bytes for `compat_flags`).** **Rejected on premise.** Bytes 6–9 of the descriptor header are `descriptor_length` (`uint32`), not reserved. Future compatibility-flag bits MUST be allocated from the reserved bits of the descriptor `flags` field (currently bits 4–31), which is the correct architectural channel. A minor amendment MUST NOT allocate a new byte at a fixed offset per S6.
- **OQ-C (name for selective forward compatibility).** **Resolved as FORWARD_ADDITIVE.** See CD2.
- **OQ-D (worked example placement).** **Resolved in two parts.** One worked example is added inline in `versioning.md` (the MXFP `scheme_version` scenario). A full evolution playbook with multiple scenarios is deferred to `docs/cookbook/evolution-playbook.md` as a Layer 5+ `doc-updater` deliverable.

### New Core Property #12

Core Property #11 (Extensibility, ADR-017) names the *surface* — the tag ranges, flag bits, and length-prefixed sections that allow the format to grow. Core Property #12 names the *rules* — the normative contract that governs how that surface is used safely over time. Together they form one continuous property; the split is editorial, not conceptual.

The following paragraph is added to `README.md` after Core Property 11 (Array Database Foundation):

> #### 12. Format Evolvability (Operational Rules for Extensibility)
>
> Hurray defines normative rules for how the format changes over time, formally specifying how the extension surface from Core Property #11 is used in practice. Within major version `1.x`, the format is BACKWARD-compatible (a reader at minor `M` parses any minor `N ≤ M` correctly) and FORWARD_ADDITIVE (a reader at minor `M` correctly parses every part of newer-minor data that its own minor defines, ignores additive trailing bytes inside known length-prefixed sections, and rejects newer-minor data that uses an unknown flag bit or unknown public tag value). Public tag values are never rebound once allocated; deprecated tags retain their original semantics. A future major version is accompanied by a normative migration specification. Per-field tagging (Protobuf) and vtables (FlatBuffers) are explicitly rejected: they are incompatible with Hurray's zero-copy fixed-offset access model. See [`versioning.md`](versioning.md) § Evolvability Contract for the full normative definition.

## Alternatives Considered

**Leave the compatibility direction unnamed.** Rejected: implementors comparing Hurray to Protobuf/Avro/FlatBuffers cannot match the property to a familiar label, and `spec-checker` has no anchor for auditing compatibility direction. Naming costs almost nothing.

**Adopt full forward compatibility (Protobuf-style) by silently ignoring unknown gated sections.** Replace CD2 with "a reader MUST skip any unknown flag-gated section using its length prefix." Rejected: an unknown flag bit signals that the writer relied on a feature whose semantics are unknowable to the reader. Silently dropping the feature could allow a reader to parse a tensor's shape while mis-interpreting its data buffer if the unknown section changes how `byte_offset` or `sync_mode` is interpreted. FORWARD_ADDITIVE captures exactly the cases where skipping is genuinely safe (trailing bytes inside an already-understood section).

**Adopt per-field tags or vtables to gain Protobuf/FlatBuffers-style evolvability.** Rejected on architectural grounds (see § Anti-patterns rejected).

**Defer S5 (major-version migration commitment) to the actual `2.0` planning cycle.** Rejected: the value of S5 is the *promise* made before any user bets a workflow on `1.x` archives. Deferring it weakens the evolvability property at the exact point users need it most.

**Encode deprecation as a wire-level flag bit.** Rejected: deprecation is a writer-facing recommendation, not a reader-facing state. R3 mandates identical wire semantics for deprecated and non-deprecated tags, so a wire flag would carry no information for the reader. A spec-table annotation (S3) is the right channel.

**Allow tag rebind after a grace period.** Rejected: a tag value's meaning being stable for the lifetime of `1.x` is more valuable than recovering byte space. Tag spaces have large reserved ranges; exhaustion within `1.x` is not a realistic concern.

## Consequences

### Positive

- The compatibility posture of Hurray is expressible in a single sentence per axis ("BACKWARD within `1.x`, FORWARD_ADDITIVE within `1.x`, no cross-major automatic compat, migration via S5").
- Spec amendments have a normative checklist (S1–S6) that `spec-checker` can apply mechanically.
- Long-lived archives are protected by the anti-rebind rule (S2) and the migration commitment (S5).
- The deprecation convention (S3 + R3 + W3) lets the spec retire wire forms cleanly without breaking readers.

### Negative / obligations created

- Every future minor amendment MUST document a default for any appended trailing field (S4). New editorial obligation on `format-spec-writer`.
- S6 constrains spec authors: a future minor revision MUST gate new content behind flag bits, tag values, or length-prefixed trailing extensions.
- S5 binds the project to writing a migration specification before any future major bump can be Accepted.
- A `docs/cookbook/evolution-playbook.md` deliverable is created for Layer 5+.

### Risks

- **FORWARD_ADDITIVE misread as "Hurray is forward-compatible."** Mitigation: the name explicitly contains `ADDITIVE` and `versioning.md` § Evolvability Contract explains the asymmetry with a worked example.
- **Defaults table becoming inconsistent.** Mitigation: S4 mandates the default appears in the same minor revision that introduces the trailing field; `spec-checker` gains a corresponding checklist item.
- **S2 misread as forbidding tag reuse across majors.** It is not. S2 binds `1.x` only; a `2.0` migration spec MAY remap tag values entirely.

### Compatibility impact

This ADR introduces **no new wire-format constraints** and **no new fields**. It documents and disciplines existing behaviour and adds editorial rules for future spec amendments. W3 and W4 constrain future writers; they do not invalidate any existing writer's output, because no deprecated tags exist yet and no trailing fields have been appended.

## Handoff

- `format-spec-writer`: add **§ Evolvability Contract** to `versioning.md` (sub-sections: Compatibility Direction, Writer Evolution Rules, Reader Evolution Rules, Spec Amendment Rules, Defaults for Appended Trailing Fields, Anti-Patterns, worked MXFP example). Add cross-reference in § Change Classification MINOR row to S6. Add § Core Property 12 to `README.md`. Update `docs/SPEC_CHECKLIST.md` preamble (13→14 categories, 11→12 Core Properties) and insert new § 12 Format Evolvability category with the 6 checklist items listed below.
- `spec-checker`: the new § 12 checklist items are: (1) appended trailing fields have documented defaults (S4); (2) new public tags allocated from reserved ranges (S1); (3) no allocated tag rebound (S2); (4) deprecated tags marked in their table (S3); (5) new fields gated, not at fixed offsets (S6); (6) FORWARD_ADDITIVE preserved for prior-minor readers (CD2).
- `doc-updater` (Layer 5+): create `docs/cookbook/evolution-playbook.md` with four worked scenarios (new flag-gated section, appended trailing field, new KV value tag, cross-major migration outline).

## Date

2026-05-14
