# ADR-040: One field identifier space covers the layout payload and all five optional sections

## Status

Accepted (2026-10-12)

Settles the identifier-space question **ADR-039** deferred, and renames the three
accessors it introduced. Supersedes ADR-039 §§ 1–3 on naming only; its mechanics —
the `uint32_t` tag space, the widening rule, the capacity idiom, the status pair, the
`level` parameter, the placement on `HurrayDescriptor` — are unchanged and still
govern. Closes [issue #231](https://github.com/pgillet/hurray/issues/231).

## Context

ADR-039 settled how a structured descriptor sub-object crosses the C ABI: a field is
named by a `uint32` identifier whose high half-word says which sub-object it addresses,
and read through a typed accessor. It shipped that for the layout payload (PR #228) and
left one question open, to be answered by whichever section came first:

> **Whether the five optional sections share one identifier enum** with reserved
> ranges, as § 3 assumes, **or take one type each.** Settled when the first of them is
> implemented; the range discipline in § 3 leaves both open.

All five are now being implemented at once, so the question is live. They are the last
sub-objects `hurray-core` can describe and C cannot read: every one is ✅ for
`hurray-core` and `hurray-python` on the generated status page and ❌ for `hurray-ffi`,
and `website/coverage-matrix.toml` names five `hurray_descriptor_*` symbols for them
that do not exist in `hurray.h`.

Two further facts bear on the answer. First, the sections introduce field types the
layout payload does not have: per-tensor affine `zero_point` is an `int32` and its
`scale` a `float32`, and six of the statistics are `float64`. ADR-039 § 4 anticipated
exactly this — *"should a signed scalar field ever be specified, it gets its own
accessor then — additively, without touching these three."* Second, `Statistics`
guards each field group with a bit in `computed_mask`, and the spec requires a reader
to check that bit before using a field.

## Decision

### 1. One identifier space, not five

`HurrayLayoutField` becomes **`HurrayDescriptorField`**, a single `uint32_t` tag space
addressing the layout payload and every optional section:

```text
field_id = (range << 16) | (discriminant << 8) | field_index
```

| Range | Sub-object | Discriminant byte |
|---|---|---|
| `0x0001` | layout payload | layout tag |
| `0x0002` | quantization | scheme tag; `0x00` addresses the section itself |
| `0x0003` | statistics | `0x00` |
| `0x0004` | shard | `0x00` |
| `0x0005` | extension type | `0x00` |
| `0x0006` | composite member | `0x00` |

This is what ADR-039 § 3 already assumed, and it keeps the applicability check a
comparison. Five separate types would have multiplied the accessor family by five for
no gain: a caller must know which sub-object it is addressing in either design, and in
this one the identifier says so.

Quantization is the only section with an inner sum type, so it is the only one that
uses the discriminant byte — for the scheme tag, exactly as the layout range uses the
layout tag. `0x00` is not a valid scheme tag, which frees that slot for
`HURRAY_QUANT_FIELD_SCHEME_TAG`: the field a caller must read first, since it decides
which scheme-specific identifiers apply. The other four sections are flat and carry
`0x00` there, leaving 256 field indices each and the discriminant byte available should
one of them ever grow a variant.

Allocation follows ADR-019's anti-rebind rule: a retired identifier is never reused for
a different field. 43 identifiers are allocated here — 18 quantization, 12 statistics,
2 shard, 10 extension type, 1 composite member.

### 2. The accessors are renamed, not duplicated

One space addressing six sub-objects cannot be read through functions named for one of
them. The three shipped symbols are renamed and two are added:

| Before (ABI 5) | After (ABI 6) |
|---|---|
| `hurray_descriptor_layout_field_u64` | `hurray_descriptor_field_u64` |
| `hurray_descriptor_layout_field_u64_array` | `hurray_descriptor_field_u64_array` |
| `hurray_descriptor_layout_field_i64_array` | `hurray_descriptor_field_i64_array` |
| — | `hurray_descriptor_field_i64` |
| — | `hurray_descriptor_field_f64` |

The `HURRAY_LAYOUT_FIELD_*` constants keep their names; they do name layout fields. The
new constants are prefixed by section: `HURRAY_QUANT_FIELD_*`, `HURRAY_STATS_FIELD_*`,
`HURRAY_SHARD_FIELD_*`, `HURRAY_EXT_TYPE_FIELD_*`, `HURRAY_COMPOSITE_MEMBER_FIELD_*`.

`..._i64` serves per-tensor affine `zero_point`; `..._f64` serves `scale` and the six
`float64` statistics. Both widen losslessly from their wire widths. A signed read is
not a convenience here: an unsigned accessor would report a `zero_point` of `-128` as
`2^64 - 128`, and a quantized dequantization cannot absorb that.

No deprecated aliases. The ABI window is open until 1.0 by the policy on
[issue #225](https://github.com/pgillet/hurray/issues/225), and carrying two names for
one function past this pass would cost more than the relink does.

### 3. A statistic reads as absent unless its guard bit is set

`HURRAY_ERR_FIELD_NOT_PRESENT` covers an absent section, an unset optional field —
and, for the statistics, a field whose `computed_mask` bit is clear. The output is left
untouched, as it is for any other absence.

This enforces the spec rule rather than restating it. The statistics fields are
physically present whatever the mask says, each holding whatever the writer left there,
so the only failure mode of this section is a reader that forgets to test the mask —
and a C caller is exactly the reader with no type system to remind it.
`HURRAY_STATS_FIELD_COMPUTED_MASK` reads the raw mask for a caller that would rather
branch on it directly.

### 4. The quantization section is decoded per call

`TensorDescriptor` stores the quantization payload verbatim, because typed decode
belongs to a layer with schema context. A field read therefore decodes it on each call
rather than caching a typed form in `HurrayDescriptor`.

Caching would move a possible failure into `hurray_descriptor_decode`, which today
accepts a descriptor whose quantization bytes it never typed — a decode that currently
succeeds would start failing. The payload is at most 24 bytes of fixed-width fields, so
the work is arithmetic, not allocation.

### 5. C ABI version 5 → 6

The rename removes three exported symbols, so a v5 consumer must relink. Two new
functions and 43 new identifiers are additive.

## Alternatives Considered

**A parallel `hurray_descriptor_section_field_*` family, keeping the layout names.**
Rejected: the identifier's range already tells a caller which sub-object it is
addressing, so a second family would duplicate every signature and every dispatch to
re-encode information the argument carries. It would also leave the question of which
family a future sub-object joins, permanently.

**One identifier type per section (the deferred alternative).** Rejected: five types
force either five accessor families or one family with an untyped parameter, which is
the shared space with extra ceremony. The embedded range already partitions the space,
and C has no type-level benefit to collect — these are `uint32_t` typedefs precisely
because a C caller may pass any integer.

**Named accessors for the sections** (`hurray_descriptor_statistics_nnz`, …). Rejected
for the reason ADR-039 rejected it for layouts: 43 symbols now, and the count grows
with every section the format gains. ADR-019's thesis is that this format evolves by
allocating tag values.

**A `repr(C)` struct per section.** Rejected for ADR-039 § Alternatives' reasons,
which apply unchanged: it publishes Rust layout across the boundary, and it cannot
express `Statistics`'s mask-guarded optionality without the caller reimplementing the
guard.

**Keeping `zero_point` in the `u64` accessor, documented as a bit pattern.** Rejected:
it makes every caller cast, and a caller that forgets gets a plausible-looking huge
number rather than an error.

## Consequences

- All five section rows flip to ✅ for `hurray-ffi` on the generated status page. What
  remains ❌ there is Level 2 (no encode entry point in C at all, deferred pending the
  async bridge design) and Level 3 (unscheduled by decision, stated on the page).
- A v5 consumer must relink. There is no released ABI 5 consumer outside this
  repository; 0.1.0 shipped ABI 4.
- The accessor family is now five functions and will not grow again for a new
  sub-object — only for a new field *type*, and the format has no scalar type left that
  `u64`/`i64`/`f64` cannot carry.
- Reading a quantization field costs a decode. Measured in fixed-width arithmetic over
  ≤ 24 bytes; if a caller ever reads enough fields per descriptor for it to matter, the
  fix is a typed-section handle, not a cache inside this one.
- The guard-bit enforcement means a C caller cannot reproduce a Rust caller's ability to
  read a stale statistic deliberately. That is the intended asymmetry; `hurray-core`
  keeps the fields public for a writer that is computing them.

## Required Documentation Amendments

- `docs/impl/c-ffi.md` — the renamed family, the two new accessors, the range table.
- `docs/cookbook/layer-7-c-ffi.md` — read a quantization scheme and its parameters, a
  shard's parent shape, and a mask-guarded statistic from C.
- `website/coverage-matrix.toml` — the five section rows' `c_symbols` lists, and the
  `level-1` row's accessor names.
- `docs/SUMMARY.md` — add this ADR.
- ADR-039 — a Status note pointing here for the naming.

## Open Questions Deferred

- **The writer side.** Unchanged from ADR-039: `HurrayWriter` is a typedef with nothing
  behind it and there is no encode entry point, deferred pending the async bridge
  design. Not an accessor question.
- **Whether a composite head's member descriptors are reachable from C.**
  `HURRAY_LAYOUT_FIELD_COMPOSITE_MEMBER_COUNT` says how many follow and
  `HURRAY_COMPOSITE_MEMBER_FIELD_ROLE` reads one member's role once that member's own
  descriptor has been decoded, but C has no way to walk from a head to its members —
  that is a streaming/container concern, not a field read, and it waits on the Layer 5
  C surface.
