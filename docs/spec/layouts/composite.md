# Composite / Virtual Tensor — Hurray Format Specification

**Layout tag:** `0x0B` | **Tier:** 1 | **Type:** Virtual

> This section uses RFC 2119 key words: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT,
> SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

## Description

A **composite tensor** is a **head** descriptor plus an ordered set of **member**
tensors, combined under a declared **composition rule**. The head presents a single
logical view — one `shape` and one `type_tag` — while the members supply the actual
data, each as a complete, ordinary tensor descriptor with its own layout, buffers,
quantization, statistics, and device placement.

The head is a **virtual** (data-less) descriptor: it owns no buffers. This is a new
addressing category, **Virtual**, alongside Dense, Sparse, and Indirect (see
`memory-layout.md` § Layout Taxonomy). A composite unifies three previously distinct
capabilities under one primitive:

- **Partition** — a logical tensor whose index space is an exact, non-overlapping tiling
  of heterogeneous regions, each a full member tensor (see § Composition Semantics ›
  Partition for the coverage and non-overlap constraints).
- **Overlay** — a base tensor spanning the whole index space plus scattered corrections
  at shared indices (SpQR / KVQuant outlier quantization).
- **Group** — several independent tensors delivered together under one logical identity
  (multi-output inference; weight collections).

> **Note (non-normative):** A member is an ordinary `TensorDescriptor`. Everything a
> region needs — per-member layout, buffers, quantization, statistics, device tags —
> comes from the existing descriptor machinery, so the composite adds no per-member wire
> format of its own beyond the small Composite Member section (§ Composite Member
> Section). The crux of the unification is that **a region is a shard is a member.**

> **Note (non-normative):** This version specifies only the **v1.0-accepted** scope:
> partition, group, and **sealed** overlay. Versioned / open overlay — an appendable,
> time-travel-capable overlay identified by an open-composite `member_count` sentinel and
> a per-member version field — is **deferred to a future ADR** (see ADR-027 § Status).
> Wire space is reserved for it (the `member_count` sentinel `0xFFFFFFFF` in § Head
> Layout-Specific Fields, and the reserved padding in § Composite Member Section) so the
> future feature is an additive change, not a re-layout. Those reserved encodings are
> **not usable in v1.0**; the normative rejection rules for them appear in
> § Head Layout-Specific Fields and § Composite Member Section below.

## Head Descriptor

The head is an ordinary tensor descriptor (see `metadata.md`) with `layout_tag = 0x0B`.
It carries the composite's **logical shape** (`shape`) and **logical element type**
(`type_tag`) — the view the composite presents to a consumer.

The head owns no data:

- `buffer_count` MUST be `0x00`.
- `byte_offset` MUST be `0x0000000000000000`.

A reader MUST reject a head descriptor whose `buffer_count` is not `0x00`, or whose
`byte_offset` is not `0x0000000000000000`.

The head MUST NOT set the `HAS_COMPOSITE_MEMBER` flag (bit 4); that flag applies to
members only (§ Composite Member Section). Because the head owns no data buffer, it MUST
NOT set the `HAS_QUANTIZATION` flag (there is no stored data to dequantize). When a head is
itself a member of an enclosing partition or overlay composite (a nested composite; see
§ Binding), it MUST carry a shard section exactly as any other member does.

A strict reader MUST reject an unrecognised `layout_tag = 0x0B`. A permissive reader MAY
read the head's `shape` and `type_tag` but MUST NOT dereference tensor data through the
head (there is none).

## Head Layout-Specific Fields

Immediately following the head's `byte_offset` (per `metadata.md` § Layout-Specific
Fields), the composite head encodes the following fields. All multi-byte fields are
little-endian.

| Field | Type | Description |
|-------|------|-------------|
| `composition_rule` | `uint8` | `0x01` partition, `0x02` overlay, `0x03` group. `0x00` and `0x04`–`0xEF` are reserved; `0xF0`–`0xFE` are implementation-private; `0xFF` is invalid. |
| `combine_op` | `uint8` | Overlay only: `0x01` replace (last-wins), `0x02` add. MUST be `0x00` for partition (`0x01`) and group (`0x03`). |
| `_reserved` | `uint8[2]` | MUST be `0x00`. |
| `member_count` | `uint32` | Number of member tensors that immediately follow the head. MUST be a definite count for every composition rule. |

A reader MUST reject a head whose `composition_rule` is `0x00`, `0xFF`, or in the reserved
range `0x04`–`0xEF`. A reader that does not recognise an implementation-private
`composition_rule` in `0xF0`–`0xFE` MUST reject the head unless operating in permissive
mode with an out-of-band agreement.

A reader MUST reject a head whose `combine_op` is not `0x00` when `composition_rule` is
`0x01` (partition) or `0x03` (group). When `composition_rule` is `0x02` (overlay), a
reader MUST reject a head whose `combine_op` is not `0x01` or `0x02`.

A reader MUST reject a head whose `_reserved` bytes are not all `0x00`.

**`member_count` sentinel — RESERVED.** The value `0xFFFFFFFF` denotes an **open
composite** and is RESERVED for a future ADR (versioned / open overlay). A strict v1.0
reader MUST reject a head whose `member_count` is `0xFFFFFFFF`. All v1.0 composites —
partition, group, and overlay — use a definite `member_count`.

> **Note (non-normative):** `member_count = 0x00000000` (a head with no members) is a
> definite count of zero. It is syntactically valid but carries no data view; writers are
> not expected to emit it. Validation for each rule at "the Nth member" is trivially
> satisfied at N = 0 except where a rule requires a base member (overlay), which such a
> head cannot supply — see § Validation.

## Members

A member is a complete, ordinary `TensorDescriptor` with its own `layout_tag` (dense,
sparse, indirect, or a nested composite), its own buffer table, and its own optional
quantization, statistics, and device tags.

### Shard section

For **partition** (`0x01`) and **overlay** (`0x02`) composites, every member MUST carry a
shard section (`HAS_SHARD` flag set; see `metadata.md` § Shard Section, and ADR-004). Its
`parent_shape` MUST equal the head's logical `shape`, and its `shard_offset` together with
the member's own `shape` define the member's box in the head's index space:

```
member covers, along dimension k, the half-open range
    [ shard_offset[k], shard_offset[k] + shape[k] )
```

For **group** (`0x03`) composites, members MAY omit the shard section (group members have
no spatial relationship to the head; see § Composition Semantics).

### Element type across members

For partition and overlay, each member MAY declare its own **stored** element type and its
own quantization scheme, but each member's **decoded** value type MUST equal the head's
`type_tag`. Dequantization already yields a canonical real-valued view, so no new
machinery is required.

> **Note (non-normative):** SpQR example — head `type_tag = float16`; the base member is
> `int4` with per-block-affine quantization decoding to `float16`; an outlier correction
> member is a `float16` COO sparse tensor. The overlay combine is evaluated in `float16`
> (the head's type). Composite versioning would change **values over a fixed index
> space**; shape evolution is out of scope.

For group composites, the head's `shape` and `type_tag` are advisory and members MAY
differ arbitrarily (see § Composition Semantics).

## Composite Member Section

**Overlay members** carry a **Composite Member section**, a new optional descriptor
section gated by descriptor flag bit 4, `HAS_COMPOSITE_MEMBER`, defined in
`metadata.md` § Composite Member Section and appended after the Extension Type section.
Its single v1.0 field is `member_role`:

| `member_role` | Meaning |
|---------------|---------|
| `0x00` | correction |
| `0x01` | base |
| `0x02`–`0xFF` | RESERVED (see § Deferred below) |

Partition and group members MUST NOT carry a Composite Member section (they MUST NOT set
`HAS_COMPOSITE_MEMBER`). An overlay member MUST carry one.

> **Note (non-normative):** v1.0 carries `member_role` only. A sealed overlay's precedence
> is plain stream / emission order (§ Composition Semantics), so no explicit version field
> is needed. The section's reserved padding leaves room for a future ADR to add a
> `member_version` field additively without reallocating the section.

## Binding

A head with a definite `member_count = N` binds the **next N self-delimiting tensors** in
stream / file write order as its members. This is a **forward** promise — the head
precedes its members, which precede their data — not a back-reference. It introduces no
name namespace and is streamable for both readers and writers.

- **In-process:** the head handle plus an array of N member handles (no wire concern).
- **IPC / network streaming:** the framing and "close" rules are defined in
  `interchange.md` § Composite Tensor Streaming.
- **File:** the head + members occupy consecutive index entries in the tensor region; the
  recovery rule is defined in `file-format.md` § Composite Tensors.

Nested composites are permitted: a member MAY itself be a head with `layout_tag = 0x0B`,
parsed pre-order. A reader MUST enforce a maximum composite nesting depth of 8 levels and
MUST reject a descriptor that exceeds it (the same recursion-depth discipline used for
nested Tiled layouts, `metadata.md` § Layout-Specific Fields › Tiled / Blocked).

> **Note (non-normative):** Plain sharding (members without a head; see `interchange.md`
> § Parallel Transfers) is the status quo. The head *upgrades* an ephemeral shard set into
> a persistent, composition-typed collection. Explicit member identifiers for out-of-order
> random access are not defined in this version (see § Deferred).

## Composition Semantics

### Partition (`0x01`)

Members' shard boxes MUST exactly cover the head's index space with no overlap. Each
logical index belongs to exactly one member, so the composite view is **zero-copy**: value
lookup at logical index `idx` is (1) select the member whose box contains `idx`,
(2) compute the local index `idx - shard_offset`, (3) apply that member's own addressing.

**Coverage constraint.** The union of all member boxes MUST exactly cover every element in
the head's index space: for every valid index `[i_0, i_1, ..., i_{r-1}]` (where
`0 <= i_k < shape[k]` for all `k`, `shape` being the head's `shape`), there MUST be
exactly one member whose box — `[shard_offset[k], shard_offset[k] + shape[k])` along each
dimension `k` (§ Members › Shard section) — contains that index.

**Non-overlap constraint.** Two members' boxes `A` and `B` overlap if, for every
dimension `k`:

```
A.shard_offset[k] < B.shard_offset[k] + B.shape[k]
AND
B.shard_offset[k] < A.shard_offset[k] + A.shape[k]
```

Member boxes MUST NOT overlap. A conforming writer MUST produce a partition composite
whose members satisfy both constraints. A conforming reader SHOULD validate them and MUST
reject a violating composite unless in permissive mode (§ Validation).

### Overlay (`0x02`, v1.0: sealed only)

One member is the **base** (`member_role = 0x01`): it MUST be the **first** member and its
box MUST span the whole index space (its shard `shard_offset` is all-zero and its `shape`
equals the head's `shape`). The remaining members are **corrections** (`member_role =
0x00`) whose boxes MAY overlap one another and the base.

**Precedence is emission / stream order — later wins.** The writer emits the base first,
then corrections in the order they take effect. A single-pass reader applies members as
they arrive; no version field or reordering is required.

**Reads** apply the base, then all corrections in emission order, under `combine_op`,
evaluated in the head's `type_tag`:

- `combine_op = 0x01` (replace): within a correction's box, the topmost (latest-emitted)
  member covering an index wins; outside every correction's box, the base shows through.
- `combine_op = 0x02` (add): the value at an index is the base value plus the sum of all
  covering corrections' values at that index.

A correction's box replaces or adds **within its box only**; outside it, lower-precedence
members (down to the base) show through. Per-member storage is zero-copy, but the **merged
logical view is computed by the consumer** — a sealed overlay is not zero-copy at the
composite level.

> **Note (non-normative):** Replace serves region overwrites; add serves residual / outlier
> overlays (SpQR / KVQuant). A v1.0 overlay is a complete snapshot: definite
> `member_count`, not appendable without rewriting the head. Logical delete (reverting a
> region to base, or masking it) is out of scope for v1.0; a region is reverted by writing
> a new correction that carries the desired values. A data-less tombstone member kind is
> reserved (see § Deferred).

### Group (`0x03`)

Members are independent tensors under one head identity, with **no spatial semantics** and
**no ordering semantics**. The head's `shape` and `type_tag` are advisory; members MAY
differ arbitrarily in rank, shape, element type, layout, and device. This occupies the
grouping gap (see ADR-010) using forward adjacency, not naming.

## Validation

Validation is **cross-member and stateful but bounded**: because every v1.0 composition
rule uses a definite `member_count = N`, a reader accumulates state only up to N, reaches
one verdict at the Nth member, and is done.

### Per-member checks (immediate)

On each member, a reader MUST verify:

1. For partition and overlay: the member carries a shard section whose `parent_shape`
   equals the head's `shape`, and whose box is in bounds
   (`shard_offset[k] + shape[k] <=` the head's `shape[k]` for every dimension `k`).
2. The member's **decoded** value type equals the head's `type_tag` (§ Members).
3. `combine_op` is legal for the composition rule (§ Head Layout-Specific Fields).
4. For overlay: the member carries a Composite Member section with a valid `member_role`
   (`0x00` or `0x01`). The **first** overlay member MUST be the base (`member_role =
   0x01`) and MUST span the index space (all-zero `shard_offset`, `shape` equal to the
   head's `shape`). Every subsequent overlay member MUST have `member_role = 0x00`.
5. For partition and group: the member MUST NOT set `HAS_COMPOSITE_MEMBER`.

A per-member violation MUST cause rejection of the whole composite.

### Close-time checks (at the Nth member)

- **Partition:** on receiving the Nth member, run the exact-cover and non-overlap checks
  (§ Composition Semantics › Partition) over the N boxes. A gap or an overlap MUST cause
  rejection.
- **Sealed overlay:** close at the Nth member. The base-span is already checked at the
  first member (per-member check 4); overlap between corrections is legal; there is no
  exact-cover requirement. A reader MUST reject an overlay head (`composition_rule =
  0x02`) with `member_count = 0x00000000`: overlay requires a base member (§ Composition
  Semantics), which a zero-member composite cannot supply.
- **Group:** close at the Nth member. There is no coverage or overlap check (members MAY
  differ arbitrarily).

### Torn composite

A **torn** composite — the stream or file ends before all N members have arrived, for any
composition rule — is incomplete. A strict reader MUST reject it. A permissive reader MAY
expose the arrived members as independent shard tensors but MUST NOT present the composite
as complete.

## Deferred

> **Note (non-normative):** The following are reserved for a future ADR (versioned / open
> overlay and related work) and are **not part of v1.0**: the `0xFFFFFFFF` open-composite
> `member_count` sentinel and its append-oriented membership rules; the `member_version`
> field (the Composite Member section's reserved padding holds room for it); time-travel
> reads; file append with footer regeneration; a tombstone member kind (`member_role =
> 0x02`, data-less, revert-to-base-within-box); explicit member identifiers for
> out-of-order random access; wall-clock timestamp versioning; an optional inline
> single-frame compaction of a partition composite (a future revival of this idea would
> need a fresh layout tag allocated from the reserved range `0x0C`–`0x3F`; the old
> subpaving tag `0x06` is permanently reassigned to COO and is not available for it); and
> heterogeneous per-member device placement. See ADR-027 § Status and § Consequences.

## Example

A sealed overlay (SpQR-style) with a `float16` logical view of shape `[4096, 4096]`,
`combine_op = 0x01` (replace), one base and one correction:

```
Head (layout_tag = 0x0B):
  shape            = [4096, 4096]
  type_tag         = float16
  buffer_count     = 0x00
  byte_offset      = 0x0000000000000000
  composition_rule = 0x02   (overlay)
  combine_op       = 0x01   (replace)
  member_count     = 2

Member 0 (base):
  shape            = [4096, 4096]
  type_tag         = int4    (stored), decodes to float16
  layout_tag       = 0x01    (row-major), per-block-affine quantization
  HAS_SHARD:  parent_shape = [4096, 4096], shard_offset = [0, 0]
  HAS_COMPOSITE_MEMBER:  member_role = 0x01  (base)

Member 1 (correction):
  shape            = [4096, 4096]
  type_tag         = float16
  layout_tag       = 0x06    (COO sparse: scattered outliers)
  HAS_SHARD:  parent_shape = [4096, 4096], shard_offset = [0, 0]
  HAS_COMPOSITE_MEMBER:  member_role = 0x00  (correction)
```

The merged logical view is: for each index, the correction's outlier value if present,
otherwise the dequantized base value. The reader computes this merge; it is not zero-copy
at the composite level.

A partition composite with a `float32` logical view of shape `[8, 8]`, split into two
`[8, 4]` members:

```
Head: shape = [8, 8], type_tag = float32, buffer_count = 0,
      composition_rule = 0x01 (partition), combine_op = 0x00, member_count = 2

Member 0: shape = [8, 4], HAS_SHARD parent_shape = [8, 8], shard_offset = [0, 0]
Member 1: shape = [8, 4], HAS_SHARD parent_shape = [8, 8], shard_offset = [0, 4]
```

The two boxes exactly cover `[8, 8]` with no overlap; element `[3, 6]` resolves to member 1
at local index `[3, 2]`. The view is zero-copy.
