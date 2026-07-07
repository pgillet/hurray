# ADR-027: Composite Tensors — Head + Members + Composition Rule

## Status

Draft

Supersedes: ADR-026 (Subpaving Nested Region Descriptors) — see § Consequences
Amends the deferral scope of: ADR-010 (Multi-Tensor Collections Deferred)

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
tensor** — adds the overlay model, and specifies **versioned overlay** (partial updates +
time travel) as a first-class v1.0 feature.

Constraints preserved: streamability (descriptors precede data; self-delimiting; no
back-references; no end-of-file index); zero-copy; 64-byte alignment; language-agnostic
naming; RFC 2119; the ADR-017/019 extensibility and evolvability contracts (for *future*,
post-1.0 additions — everything in this ADR ships in the initial v1.0 format, as the format
is pre-release).

## Decision

A **composite tensor** is a **head** descriptor plus an ordered set of **member** tensors,
combined by a declared **composition rule**.

### D1 — The head is a virtual (data-less) tensor descriptor under layout tag 0x0C

The head is an ordinary tensor descriptor with `layout_tag = 0x0C` ("Composite / Virtual",
Tier 1, a new addressing category *Virtual* alongside Dense / Sparse / Indirect). It:

- carries the composite's **logical shape** (`shape`) and **logical element type**
  (`type_tag`) — the view the composite presents to a consumer;
- owns **no data**: `buffer_count` MUST be `0` and `byte_offset` MUST be `0`.

Tag `0x0C` is drawn from the reserved layout range (`0x0C`–`0x3F`). Because Hurray is
pre-release, it is allocated as part of the initial v1.0 format (no version-increment
ceremony). A strict reader rejects an unrecognised `0x0C`; a permissive reader may read the
head's shape and dtype but MUST NOT dereference data (there is none).

The head's layout-specific fields encode the composition rule:

| Field | Type | Description |
|-------|------|-------------|
| `composition_rule` | `uint8` | `0x01` partition, `0x02` overlay, `0x03` group. `0x00` and `0x04`–`0xEF` reserved; `0xF0`–`0xFE` private; `0xFF` invalid. |
| `combine_op` | `uint8` | Overlay only: `0x01` replace (last-wins), `0x02` add. MUST be `0x00` for partition and group. |
| `_reserved` | `uint8[2]` | MUST be `0x00`. |
| `member_count` | `uint32` | Number of member tensors that immediately follow. The sentinel `0xFFFFFFFF` denotes an **open composite** (see D3); it is permitted only for overlay. |

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
| `member_version` | `uint64` | Logical version (writer-controlled). MUST be non-decreasing in emission/stream order (D6). |
| `member_role` | `uint8` | `0x00` correction, `0x01` base. `0x02`–`0xFF` reserved (see § Deferred: tombstones). |
| `_reserved` | `uint8[7]` | MUST be `0x00`. |

Partition and group members do not carry this section. This is the crux of the unification:
**a region ≡ a shard ≡ a member.** Per-member layout, buffers, quantization, statistics, and
device placement all come from the ordinary descriptor machinery — including the three things
ADR-026 D5 had to forbid or defer.

> Plain sharding (ADR-004 / interchange parallel transfer) is the status quo: members
> without a head. The head *upgrades* an ephemeral shard set into a persistent,
> composition-typed collection.

### D3 — Binding: forward stream adjacency, no new namespace; open vs sealed composites

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

**Open composites (versioning).** An overlay head MAY set `member_count = 0xFFFFFFFF`,
declaring an **open, appendable** composite. Membership is delimited by:

- **Stream:** the maximal run of consecutive following tensors, each carrying a Composite
  Member section, immediately after the open head. The first following tensor lacking a
  Composite Member section (including any head) ends the run; stream/session close also ends
  it. Single-pass, self-delimiting, no back-reference.
- **File:** the tensor-region entries between this head and the next head or the end of the
  tensor region.

Partition and group composites MUST use a definite count. Nested composites are permitted for
definite-count composites (pre-order parse, depth cap 8); an **open** overlay's members MUST
be leaf tensors (no nested composite members). Explicit member identifiers for out-of-order
random access are not defined in this version (see Deferred).

### D4 — Composition semantics

**Partition (`0x01`).** Members' shard boxes MUST exactly cover the head's index space with no
overlap (ADR-026 D6 validation, evaluated across members). Each logical index belongs to
exactly one member, so the composite view is **zero-copy**: value lookup = box selection +
that member's own addressing. This is subpaving semantics as a collection.

**Overlay (`0x02`).** One member is the **base** (`member_role = 0x01`, the first member, box
spanning the whole space); the rest are **corrections** whose boxes MAY overlap. See D6 for
versioning, precedence, and reads. Per-member storage is zero-copy; the *merged logical view*
is computed by the consumer (exactly as SpQR/KVQuant-aware and array-DB kernels operate).

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

### D6 — Versioned overlay: partial updates and time travel (v1.0)

Overlay composites provide tensor data versioning.

**Version carrier.** Each overlay member carries a `member_version: uint64` in its Composite
Member section (D2). It is a **logical sequence number**, writer-controlled — not a wall-clock
timestamp (simpler, reproducible, no clock skew; applications needing wall-clock time MAY
record it in file-format KV metadata). The base has the lowest version; corrections carry
increasing versions. Multiple members MAY share a version (one logical update spanning several
boxes).

**Precedence (single-pass friendly).** Writers MUST emit members in **non-decreasing
`member_version` order** (base first). Precedence is therefore **emission/stream order —
later wins** — and a reader applying members in stream order automatically applies them in
version order. Explicit versions are used for the time-travel cutoff, not for reordering.

**Reads.**

- **Current view:** apply the base, then all corrections in emission order, under `combine_op`
  (replace: the topmost member covering an index wins within its box; add: base plus the sum
  of covering corrections), evaluated in the head's element type.
- **Time-travel view as of version `V`:** apply the base plus only corrections with
  `member_version <= V`, in emission order. Because emission order respects version order, a
  single-pass reader simply stops at the first member with `member_version > V`.
- **Partial update:** a correction's box replaces/adds within its box only; outside it,
  lower-precedence members (down to the base) show through.

**`combine_op` firmed for v1.0.** Both `0x01` replace and `0x02` add are defined. Replace is
the default and the dominant versioning semantic (a partial update overwrites a region); add
serves residual/outlier overlays.

**Open (appendable) vs sealed (snapshot).** An open overlay (`member_count = 0xFFFFFFFF`,
D3) is an append-only version log: the head, once written, is **immutable**, and history grows
by appending members after it. A sealed overlay (definite count) is a complete history
snapshot and is not appendable without rewriting the head. Appendable/versioned overlays MUST
use the open sentinel.

**File append story.** Appending a new version to an open overlay in an `HRRYFILE` =
write the new member's descriptor+data into the tensor region (after all existing tensors,
before the footer), then **regenerate the footer** (KV + index + trailer, relocated to the new
EOF). Existing tensor bytes and their absolute offsets, the head, and all prior members are
**unchanged** — immutability is preserved; only the footer is rewritten (an additive index
entry plus a moved trailer). This is a single fresh footer pass over the extended file,
consistent with `file-format.md`'s no-backward-seek writer.

**Deletes.** Logical delete (reverting a region to base, or masking it) is **out of scope for
v1.0**, stated explicitly. A region is reverted by writing a new correction (at a higher
version) carrying the desired values. A tombstone member kind (`member_role = 0x02`,
data-less, revert-to-base-within-box) is reserved for a future addition (the role byte is
already present).

### D5 — Validation: cross-member, stateful; every prefix of an open overlay is valid

Per-member checks are immediate: shard `parent_shape` == head `shape`; box in bounds; decoded
dtype == head `type_tag`; `combine_op` legal for the rule; for overlay, the Composite Member
section is present and `member_version` is non-decreasing; the first overlay member is the base
(`role = 0x01`) and spans the index space.

Close-time checks depend on the rule:

- **Partition (definite count):** on receiving the Nth member, run exact-cover + non-overlap
  over the N boxes (ADR-026 D6: volume-sum + sweep/pairwise). Overlap or a gap → reject.
- **Sealed overlay (definite count):** close at the Nth member; base-spans is checked at the
  first member; overlap is legal; no exact-cover.
- **Open overlay (`0xFFFFFFFF`):** there is no close and no completeness verdict. Per-member
  checks and the first-member base-spans check are the entire validation surface. **Every
  prefix of the history is itself a valid composite** (as of its last version) — a log-
  structured resilience property: a truncated open-overlay stream is not an error, it is a
  valid earlier snapshot.

Failure semantics: a per-member violation MUST cause rejection of the composite (network
stream: `ERROR` + close). A **torn definite-count group** (stream ends before N members) is
incomplete: a strict reader MUST reject it; a permissive reader MAY expose the arrived members
as independent shard tensors but MUST NOT present the composite as complete. A torn **open**
overlay is valid up to its last complete member (per the prefix property above).

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

**Wall-clock timestamps as the version axis (TileDB-style).** Rejected: a writer-controlled
logical sequence number is simpler, reproducible, and skew-free; wall-clock time, if needed,
goes in KV metadata.

**Version via member ordinal only (implicit version = position).** Rejected: an explicit
`member_version` gives a stable, addressable time-travel key, allows several members to share
one logical version, and survives re-serialization; the non-decreasing-emission-order rule
keeps single-pass reading as simple as the ordinal scheme.

**Sealed fixed-count composites only, with versioning via external file conventions.**
Rejected: appending a correction would require rewriting the head (violating immutability and
requiring a backward seek). The open sentinel makes the head immutable and the history
append-only.

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
- **Layout tag 0x06 (inline subpaving) is retained only as an OPTIONAL single-frame
  compaction** of a *partition* composite, via a normative region ↔ member equivalence
  (inline region origin ↔ member `shard_offset`; region within head `shape` ↔ member
  `parent_shape`). **Inline 0x06 is deferred**; the collection form is the v1.0 deliverable.

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
payload codec; the open-composite sentinel; the `HAS_COMPOSITE_MEMBER` section and its codec;
shard-based member positioning (shard section already exists); the cross-member stateful
validator (a `CompositeValidator` accumulating member boxes + a version/precedence check); and
the versioned-overlay read model as a *specified* semantic (the merge itself is a consumer
concern). This is a bounded addition to the Layer-4 core; it does **not** materially delay
Layers 0–4.

**Staged with their layers (not pulled forward):**

- **Layer 5 (streaming):** head→member adjacency; the open-overlay member run and its
  termination; composite "close" (definite counts) on the Nth `TENSOR_DATA_END`; reuse of
  shard-consistency validation.
- **Layer 6 (file):** head + members as consecutive index entries; open-overlay membership by
  tensor-region delimitation; the **append + footer-regeneration** flow for versioned
  overlays; time-travel reads by scanning member versions in the index.
- **Layer 7 (FFI):** a composite handle kind (head handle + member iterator; a version-cutoff
  read parameter for overlays).
- **Layer 8 (Python):** a `CompositeTensor` view; `__dlpack__` per member; overlay current /
  as-of-`V` views materialised on demand.

**Deferred (Open Questions):** explicit member IDs / out-of-order random access; wall-clock
timestamp versioning; tombstone / logical-delete (`member_role = 0x02`); inline 0x06
compaction; nested composites inside open overlays; heterogeneous per-member device placement.

**Schedule statement:** ADR-027 replaces (does not add to) the ADR-026 partition
implementation budget and is likely smaller there; but it introduces a persistent grouping +
versioning concept threading through Layers 5–8 (notably the file append/footer-regeneration
and version-cutoff read paths). The v1.0 core work is bounded and non-blocking; the
transport/binding cost is real and is paid incrementally as those layers are built.

### Positive

- One primitive unifies subpaving, sharding, and grouping; overlay (SpQR/KVQuant) is
  first-class; **versioned overlay delivers array-database region/partial updates and
  time-travel reads** with a log-structured, every-prefix-valid history.
- Per-member dtype/layout/quantization/statistics/device come from existing machinery.
- Streamable (forward adjacency; open-overlay append preserves immutability), zero-copy per
  member, no new namespace.

### Negative / obligations

- A persistent, cross-descriptor, stateful validation + framing concept (now including a
  version axis and open-ended membership) is new to Layers 5–8.
- Versioned-overlay file append requires footer regeneration (append + rewrite trailer/index),
  not a pure in-place append.
- Overlay's merged/time-travel view is consumer-computed, not zero-copy at the composite level.

### Risks

- **Open-overlay membership ambiguity on a stream** — mitigated by the "maximal run of
  Composite-Member-tagged tensors, terminated by any head/plain tensor/close" rule (D3).
- **Torn history** — mitigated by the every-prefix-valid property (D5); a truncated open
  overlay is a valid earlier snapshot.
- **Overlay misread as zero-copy** — mitigated by the explicit "structure described, merge
  computed" statement (D4/D6).
- **Scope creep into a general versioned DB** — mitigated by scoping deletes and wall-clock
  timestamps out, keeping ADR-010's named-archive deferral intact, and bounding composites to
  composition + a logical version axis.

## Compatibility Impact

Hurray is **pre-release**; this ADR ships entirely within the initial **v1.0** format. Tag
`0x0C`, descriptor flag bit 4 (`HAS_COMPOSITE_MEMBER`) and its section, the `buffer_count = 0`
head rule, `combine_op`, the open-composite sentinel, and versioned overlay are all part of
v1.0 — no minor-version increment is involved. The ADR-017/019 evolvability contract governs
only **future, post-1.0** additions listed under Deferred (member IDs, tombstones, inline
0x06, wall-clock timestamps), each of which would be an additive minor that rebinds no v1.0
value. Supersedes ADR-026; leaves ADR-004 and ADR-010's core decisions intact.

## Date

2026-07-07
