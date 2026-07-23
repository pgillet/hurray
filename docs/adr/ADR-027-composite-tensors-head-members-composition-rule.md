# ADR-027: Composite Tensors — Head + Members + Composition Rule

## Status

Accepted — scope limited to **partition**, **group**, and **sealed overlay**.

**Versioned overlay is descoped from v1.0**, deferred to a future ADR: `member_version`,
the `0xFFFFFFFF` open-composite sentinel, file append + footer regeneration for overlays,
and time-travel reads. Rationale: that machinery's value is driven almost entirely by the
array-database vision, which is explicitly long-term/not-current-sprint; it introduces
unbounded, indefinitely-open cross-descriptor state (no closing verdict, ever) versus the
bounded, ADR-026-precedented state partition/sealed-overlay already require; and it adds
file-mutation (append + footer regen) and version-monotonicity machinery to Layers 5–8
for a use case not yet on the roadmap. Sealed overlay (definite count, stream-order
precedence, no `member_version`) ships now — it is cheap and delivers SpQR/KVQuant-style
outlier quantization, an immediate, published use case. See the sections below, each
annotated where content was trimmed accordingly.

Supersedes: ADR-026 (Subpaving Nested Region Descriptors) — see § Consequences
Amends the deferral scope of: ADR-010 (Multi-Tensor Collections Deferred)

> **Amended 2026-07-23 (layout-tag renumber):** The composite head's layout tag is
> reassigned `0x0C` → `0x0B`. The General Subpaving layout (former tag `0x06`) was retired
> from v1.0 entirely, and the tags that followed it — COO through Composite — were shifted
> down by one to keep the Tier-1 named layout range contiguous (`0x01`–`0x0B`, no hole),
> since Hurray is pre-release with no compatibility obligation. The head is therefore now a
> **named Tier-1 tag (`0x0B`)**, not a tag borrowed from the reserved range `0x0C`–`0x3F`;
> and tag `0x06` is now **permanently COO's**. D1 and § Disposition of ADR-026 below are
> updated to match. Other in-text references to `0x0C` (as the head tag) and to "inline
> `0x06`" compaction elsewhere in this ADR predate this amendment and are retained as the
> historical record, superseded by this note.

## Context

Three capabilities that Hurray has treated as distinct are, on inspection, one idea seen
from three angles:

- **Subpaving (layout 0x06, ADR-026):** one logical tensor whose index space is a
  partition of heterogeneous regions, each with its own layout, buffers, and (per ADR-026
  D5) quantization.
- **Sharding (shard section, ADR-004):** a tensor that declares itself a rectangular
  sub-region (`shard_offset` + `shape`) of a larger logical `parent_shape`.
- **Tensor grouping (ADR-010, deferred):** several tensors delivered together under one
  logical identity (multi-output inference; weight collections).

ADR-026's design work drove the recognition. To make sparse/paged regions work, ADR-026 had
to (a) let regions carry their own buffer sub-tables, (b) invent a `buffer_count = 0` head
carve-out, and (c) forbid or defer per-region statistics, per-region element type, and
per-region device placement (D5). But a *region that carries its own layout, buffers, and
quantization is very nearly a full tensor descriptor*, and *a region positioned in a parent
index space is exactly a shard*. If a region simply **were** a full tensor descriptor with a
shard section, every capability ADR-026 had to hand-build — and several it had to forbid —
would fall out of machinery that already exists (ADR-004 shard section; ordinary member
descriptors carry dtype, layout, buffers, quantization, statistics, and device tags for
free).

Prior art (`docs/prior-art.md` § 8) confirms two composition models a single-partition
layout cannot span:

- **Partition** (exact-cover, non-overlap): AMReX/Chombo `DisjointBoxLayout`, OpenVDB
  tiles, HDF5 Virtual Datasets. This is what subpaving implements.
- **Overlay** (overlapping composition: a base spanning the space plus scattered corrections
  at shared indices): SpQR / KVQuant outlier quantization, and — the case that makes overlay
  strategically important — **TileDB timestamped fragments**, i.e. tensor data *versioning*:
  region/partial updates and time-travel reads over a fixed index space. This directly
  serves the array-database vision.

This ADR unifies subpaving, sharding, and grouping under one primitive — the **composite
tensor** — and adds the overlay model. **Originally proposed** (2026-07-07 Draft) to also
specify **versioned overlay** (partial updates + time travel) as a first-class v1.0 feature;
on review (see § Status) that half is descoped to a future ADR, since the array-database use
case it serves is explicitly long-term rather than current, and it is the one part of this
design that introduces unbounded, file-mutating state. What ships in v1.0 is partition,
group, and **sealed** (non-versioned) overlay.

Constraints preserved: streamability (descriptors precede data; self-delimiting; no
back-references; no end-of-file index); zero-copy; 64-byte alignment; language-agnostic
naming; RFC 2119; the ADR-017/019 extensibility and evolvability contracts (for *future*,
post-1.0 additions — everything in this ADR's v1.0 scope ships in the initial v1.0 format, as
the format
is pre-release).

## Decision

A **composite tensor** is a **head** descriptor plus an ordered set of **member** tensors,
combined by a declared **composition rule**.

### D1 — The head is a virtual (data-less) tensor descriptor under layout tag 0x0B

> Amended 2026-07-23: head tag `0x0C` → `0x0B` (see § Status).

The head is an ordinary tensor descriptor with `layout_tag = 0x0B` ("Composite / Virtual",
Tier 1, a new addressing category *Virtual* alongside Dense / Sparse / Indirect). It:

- carries the composite's **logical shape** (`shape`) and **logical element type**
  (`type_tag`) — the view the composite presents to a consumer;
- owns **no data**: `buffer_count` MUST be `0` and `byte_offset` MUST be `0`.

Tag `0x0B` is a **named Tier-1 layout tag**, not one borrowed from the reserved range: it
was assigned when the General Subpaving layout was retired from v1.0 and COO through Composite
shifted down by one to close the gap (see § Status). Because Hurray is pre-release, it is
allocated as part of the initial v1.0 format (no version-increment ceremony). A strict reader
rejects an unrecognised `0x0B`; a permissive reader may read the head's shape and dtype but
MUST NOT dereference data (there is none).

The head's layout-specific fields encode the composition rule:

| Field | Type | Description |
|-------|------|-------------|
| `composition_rule` | `uint8` | `0x01` partition, `0x02` overlay, `0x03` group. `0x00` and `0x04`–`0xEF` reserved; `0xF0`–`0xFE` private; `0xFF` invalid. |
| `combine_op` | `uint8` | Overlay only: `0x01` replace (last-wins), `0x02` add. MUST be `0x00` for partition and group. |
| `_reserved` | `uint8[2]` | MUST be `0x00`. |
| `member_count` | `uint32` | Number of member tensors that immediately follow. **v1.0: MUST be a definite count for all composition rules, including overlay.** The sentinel `0xFFFFFFFF` (an **open composite**, see D3) is RESERVED — a strict v1.0 reader MUST reject it; open composites are deferred to a future ADR. |

### D2 — Members are ordinary tensor descriptors positioned by the shard section

A member is a complete, ordinary `TensorDescriptor` (its own layout — dense, sparse, paged,
or a nested composite — its own buffers, quantization, statistics, device tags). For
**partition** and **overlay** composites, each member MUST carry a **shard section**
(ADR-004; `metadata.md` § Shard Section) whose `parent_shape` equals the head's logical
`shape`; the member's `shard_offset` and its own `shape` define its box in the head's index
space. For **group** composites, members MAY omit the shard section.

**Overlay members additionally carry a Composite Member section** (a new optional descriptor
section gated by descriptor flag bit 4, `HAS_COMPOSITE_MEMBER`, appended after the Extension
Type section):

| Field | Type | Description |
|-------|------|-------------|
| `member_role` | `uint8` | `0x00` correction, `0x01` base. `0x02`–`0xFF` reserved (see § Deferred: tombstones). |
| `_reserved` | `uint8[15]` | MUST be `0x00`. |

**v1.0 carries `member_role` only.** `member_version` is deferred: a sealed overlay's
precedence is plain stream/emission order (last-wins under `combine_op = replace`), so no
explicit version field is needed until versioned overlay (time-travel) is taken up. The
reserved padding leaves room for a future ADR to add `member_version` as an additive field
under the ADR-017/019 evolvability contract, without reallocating the section.

Partition and group members do not carry this section. This is the crux of the unification:
**a region ≡ a shard ≡ a member.** Per-member layout, buffers, quantization, statistics, and
device placement all come from the ordinary descriptor machinery — including the three things
ADR-026 D5 had to forbid or defer.

> Plain sharding (ADR-004 / interchange parallel transfer) is the status quo: members
> without a head. The head *upgrades* an ephemeral shard set into a persistent,
> composition-typed collection.

### D3 — Binding: forward stream adjacency, no new namespace, all v1.0 composites definite-count

A head with a **definite** `member_count = N` binds the **next N self-delimiting tensors** in
stream / file write order as its members. This is a **forward** promise (head precedes
members precede their data), not a back-reference, so it is streamable for readers and
writers and introduces no name namespace. It works uniformly across transports:

- **In-process:** the head handle plus an array of member handles (in-memory; no wire
  concern).
- **IPC / network streaming:** the head's `TENSOR_DESCRIPTOR`, then each member's
  `TENSOR_DESCRIPTOR` → `TENSOR_DATA` → `TENSOR_DATA_END`.
- **File:** the head, then the members' descriptors+data, written contiguously in the tensor
  region; every tensor gets a footer-index entry; membership is recovered from the head's
  `member_count` plus descriptor-offset order (preserved regardless of `SORTED_INDEX`).

**Open composites — deferred.** The `0xFFFFFFFF` open-composite sentinel and its
append-oriented membership-delimitation rules (maximal run of Composite-Member-tagged
tensors; file tensor-region delimitation between heads) are reserved for a future ADR
(versioned overlay). **All v1.0 composites — partition, group, and overlay — use a
definite `member_count`,** bound uniformly by the forward-adjacency rule above.

Nested composites are permitted for definite-count composites (pre-order parse, depth cap
8). Explicit member identifiers for out-of-order random access are not defined in this
version (see Deferred).

### D4 — Composition semantics

**Partition (`0x01`).** Members' shard boxes MUST exactly cover the head's index space with no
overlap (ADR-026 D6 validation, evaluated across members). Each logical index belongs to
exactly one member, so the composite view is **zero-copy**: value lookup = box selection +
that member's own addressing. This is subpaving semantics as a collection.

**Overlay (`0x02`, v1.0: sealed only).** One member is the **base** (`member_role = 0x01`,
the first member, box spanning the whole space); the rest are **corrections** whose boxes
MAY overlap. See D6 for precedence and reads. Per-member storage is zero-copy; the *merged
logical view* is computed by the consumer (exactly as SpQR/KVQuant-aware and array-DB
kernels operate).

**Group (`0x03`).** Unordered, no spatial semantics; members are independent tensors under one
head identity (multi-output inference). The head's `shape`/`type_tag` are advisory; members
MAY differ arbitrarily. This occupies ADR-010's grouping gap using adjacency, not naming.

**Element type across members.** For partition and overlay, each member MAY declare a
different **stored** element type and its own quantization, but each member's **decoded**
value type MUST equal the head's `type_tag`. Example (SpQR): head `type_tag = float16`; base
= `int4` + per-block quantization → decodes to `float16`; outlier correction = `float16`
sparse (COO). Overlay combine happens in `float16`. Dequantization already yields a canonical
real-valued view, so this needs no new machinery. Composite versioning changes **values over
a fixed index space**; shape evolution is out of scope.

### D6 — Sealed overlay: precedence, combine, and reads (v1.0)

**Precedence (single-pass friendly, no explicit version).** Precedence is **emission/stream
order — later wins**: writers emit the base first, then corrections in the order they take
effect. A single-pass reader applies members as they arrive; no version field or reordering
is needed. (A future ADR may reintroduce an explicit `member_version` for time-travel — see
Status.)

**Reads.** Apply the base, then all corrections in emission order, under `combine_op`
(replace: the topmost member covering an index wins within its box; add: base plus the sum
of covering corrections), evaluated in the head's element type. A correction's box
replaces/adds within its box only; outside it, lower-precedence members (down to the base)
show through.

**`combine_op` for v1.0.** Both `0x01` replace and `0x02` add are defined. Replace is the
default (a region overwrite); add serves residual/outlier overlays (e.g. SpQR/KVQuant).

**Sealed only.** A v1.0 overlay is a complete snapshot: definite `member_count`, not
appendable without rewriting the head (D3). Versioned/appendable overlay is deferred.

**Deletes.** Logical delete (reverting a region to base, or masking it) is **out of scope for
v1.0**, stated explicitly. A region is reverted by writing a new correction carrying the
desired values. A tombstone member kind (`member_role = 0x02`, data-less, revert-to-base-
within-box) is reserved for a future addition (the role byte is already present).

### D5 — Validation: cross-member, stateful, bounded

Per-member checks are immediate: shard `parent_shape` == head `shape`; box in bounds; decoded
dtype == head `type_tag`; `combine_op` legal for the rule; for overlay, the Composite Member
section is present with a valid `member_role`; the first overlay member is the base
(`role = 0x01`) and spans the index space.

Close-time checks depend on the rule:

- **Partition (definite count):** on receiving the Nth member, run exact-cover + non-overlap
  over the N boxes (ADR-026 D6: volume-sum + sweep/pairwise). Overlap or a gap → reject.
- **Sealed overlay (definite count):** close at the Nth member; base-spans is checked at the
  first member; overlap is legal; no exact-cover.
- **Group (definite count):** close at the Nth member; no coverage or overlap check (members
  MAY differ arbitrarily, D4).

All v1.0 composition rules use a definite count, so all validation is **bounded**: a reader
accumulates state only up to the known N, reaches one verdict at the Nth member, and is done.
(Open, unbounded, every-prefix-valid overlay validation is deferred — see Status.)

Failure semantics: a per-member violation MUST cause rejection of the composite (network
stream: `ERROR` + close). A **torn definite-count composite** (stream ends before N members,
of any composition rule) is incomplete: a strict reader MUST reject it; a permissive reader
MAY expose the arrived members as independent shard tensors but MUST NOT present the
composite as complete.

## Alternatives Considered

**Head via a `HAS_COMPOSITION` flag on an ordinary descriptor (no new tag).** Rejected: the
head has no data, so any real `layout_tag` would misdescribe it; a Virtual tag `0x0C` with
`buffer_count = 0` is honest and reuses the descriptor frame.

**A new message/container kind outside the tensor descriptor.** Rejected: breaks the
"everything is a self-delimiting tensor descriptor" uniformity and forces new framing in
`interchange.md` and `file-format.md`.

**Explicit group IDs (a `group_id` namespace) as the binding.** Rejected for v1:
reintroduces the namespace ADR-010 warned against; forward adjacency binds streamably without
it. Left as a Deferred item for out-of-order random access.

**Wall-clock timestamps as the version axis (TileDB-style).** Considered for a future
versioned-overlay ADR; not part of v1.0 (no version axis ships at all — see Status). A
writer-controlled logical sequence number would be simpler, reproducible, and skew-free than
wall-clock time if/when versioning is taken up; wall-clock time, if needed, would go in KV
metadata.

**Explicit `member_version` for v1.0 (implicit version = position, rejected in the other
direction).** Considered and **descoped**: an explicit version field would give a stable,
addressable time-travel key, but time-travel is not a v1.0 feature, so the field would be
dead weight. v1.0 uses plain emission order (D6). Left as future work alongside the open
sentinel.

**Sealed fixed-count composites only, with versioning via external file conventions.**
This is effectively what v1.0 ships (sealed-only overlay, no version axis). A true append-only
version log (open sentinel, immutable head, footer-regeneration append) remains the better
design *if and when* versioning is taken up — descoped here, not rejected outright.

**Keep ADR-026's trimmed "descriptor-tail" region profile for inline 0x06.** Rejected in favour
of a region being a *full* nested descriptor with a shard section, so inline-region ≡ member
exactly.

**Leave subpaving (ADR-026) and composites as two mechanisms (composites in 1.1).** Rejected:
two overlapping mechanisms would diverge; the value here is unification.

## Consequences

### Disposition of ADR-026

**ADR-026 is marked Superseded by ADR-027.** Its durable insights survive and are
generalised: nested descriptors, the data-less head, and full per-region capability. What
changes:

- The bespoke ADR-026 D1 "trimmed region tail" wire is **dropped**. A partition composite's
  regions are **members = full tensor descriptors with shard sections** (the "uniform
  full-descriptor variant"). This removes the trimmed-profile consistency rules, the bespoke
  addressing-API redesign, and the `layout_codec`→buffer/quant relayering ADR-026 required.
- ADR-026 D5's forbidden/deferred items (per-region statistics, per-region element type,
  per-region device) become **supported for free** (a member is an ordinary descriptor).
- **Inline subpaving compaction is dropped, and tag `0x06` is permanently reassigned to
  COO** (amended 2026-07-23; see § Status). ADR-026's inline single-frame compaction of a
  *partition* composite is no longer associated with tag `0x06` and is not available for any
  future subpaving-compaction use; reviving that idea would require a **fresh** layout tag
  allocated from the reserved range `0x0C`–`0x3F`. The region ↔ member equivalence it relied
  on (inline region origin ↔ member `shard_offset`; region within head `shape` ↔ member
  `parent_shape`) survives conceptually, but the collection form (members) is the v1.0
  deliverable for partition.

Net: ADR-027's core is smaller than ADR-026's would have been (reuses the shard section and
ordinary members), at the cost of a persistent grouping concept threaded through the transport
layers (below).

### Reopening ADR-010 (addressed head-on)

ADR-010 deferred a *named/indexed archive container*; ADR-027 is a streamable composition
primitive, not that. Against ADR-010's four deferral reasons: (1) no string names/uniqueness
policy — binding is `member_count` + adjacency; (2) no general namespace — a composite is a
flat, bounded, single-level (optionally nested, depth-capped) grouping tied to one head; (3)
the header/footer-index-vs-streamability tension — resolved by forward adjacency (no header
index, no footer index, no back-reference); (4) no KV-metadata pressure — composites carry
composition structure, not arbitrary metadata. ADR-010's core decision (no named
`hurray-archive` in v1) **stands**; ADR-027 fills only the grouping gap, by composition.

### Scope, layers touched, and schedule cost (honest)

**Lands in v1.0 (spec + hurray-core):** the `0x0C` head descriptor; the composition-rule
payload codec; the `HAS_COMPOSITE_MEMBER` section and its codec (`member_role` only);
shard-based member positioning (shard section already exists); the cross-member stateful
validator (a `CompositeValidator` accumulating member boxes, bounded to a definite count);
and the sealed-overlay read model (current view only) as a *specified* semantic (the merge
itself is a consumer concern). This is a bounded addition to the Layer-4 core; it does
**not** materially delay Layers 0–4.

**Staged with their layers (not pulled forward):**

- **Layer 5 (streaming):** head→member adjacency; composite "close" (definite counts) on the
  Nth `TENSOR_DATA_END`; reuse of shard-consistency validation.
- **Layer 6 (file):** head + members as consecutive index entries.
- **Layer 7 (FFI):** a composite handle kind (head handle + member iterator).
- **Layer 8 (Python):** a `CompositeTensor` view; `__dlpack__` per member; overlay current
  view materialised on demand.

**Deferred (Open Questions):** versioned/open overlay in full — `member_version`, the
`0xFFFFFFFF` sentinel, append + footer-regeneration, time-travel reads (see Status; the
primary deferral); explicit member IDs / out-of-order random access; wall-clock timestamp
versioning; tombstone / logical-delete (`member_role = 0x02`); inline 0x06 compaction;
nested composites inside open overlays; heterogeneous per-member device placement.

**Schedule statement:** ADR-027 replaces (does not add to) the ADR-026 partition
implementation budget and is smaller there. With versioned overlay descoped, it introduces a
persistent grouping concept (bounded, definite-count binding + close-time validation) across
Layers 5–8, but not the open-ended file append/footer-regeneration or version-cutoff read
paths — those are deferred with the feature that needed them. The v1.0 transport/binding
cost is therefore comparable in shape to partition's, not materially larger.

### Positive

- One primitive unifies subpaving, sharding, and grouping; overlay (SpQR/KVQuant) is
  first-class as a sealed snapshot.
- Per-member dtype/layout/quantization/statistics/device come from existing machinery.
- Streamable (forward adjacency), zero-copy per member, no new namespace.
- Bounded, definite-count validation for every v1.0 composition rule (D5) — no open-ended
  state, no file-mutation story, kept out of v1.0 until actually needed.

### Negative / obligations

- A persistent, cross-descriptor, stateful validation + framing concept (head→member
  binding; partition's coverage check; sealed-overlay's base-span check) is new to
  Layers 5–8, though bounded to a definite count in every case.
- Overlay's merged view is consumer-computed, not zero-copy at the composite level.

### Risks

- **Overlay misread as zero-copy** — mitigated by the explicit "structure described, merge
  computed" statement (D4/D6).
- **Scope creep into a general versioned DB** — mitigated directly by descoping the version
  axis, the open sentinel, and file append/footer-regeneration from v1.0 entirely (see
  Status), not merely by scoping out deletes and wall-clock timestamps as before.
- **Deferred work resurfaces as a rushed addition later** — versioned overlay is real,
  array-DB-vision-serving work, not abandoned; when it's picked up it should get its own ADR,
  research pass, and spec-checker audit rather than being reconstituted ad hoc.

## Compatibility Impact

Hurray is **pre-release**; the scope of this ADR that ships within the initial **v1.0**
format is: tag `0x0C`, descriptor flag bit 4 (`HAS_COMPOSITE_MEMBER`) and its section
(`member_role` only), the `buffer_count = 0` head rule, and `combine_op` — for partition,
group, and **sealed** overlay only. No minor-version increment is involved. **The
`0xFFFFFFFF` open-composite sentinel and `member_version` are RESERVED, not usable in
v1.0**, and are deferred to a future ADR alongside the rest of the Deferred list (member
IDs, tombstones, inline 0x06, wall-clock timestamps) — each an additive minor under the
ADR-017/019 evolvability contract that rebinds no v1.0 value. Supersedes ADR-026; leaves
ADR-004 and ADR-010's core decisions intact.

## Date

2026-07-07 (Draft); scope narrowed and Accepted 2026-07-23 (versioned overlay descoped —
see § Status)
