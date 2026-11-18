# Transcode Conversion Matrix

Which representation conversions `hurray-transcode` performs, which it performs only after a
survey, which it refuses permanently, and which are not implemented yet. One row per
conversion, with the crate version it became available in.

This page is the versioned counterpart to
[ADR-044](../adr/ADR-044-hurray-transcode-crate.md) § 14. That section audits the
*architectural* surface — for every field of the tensor descriptor, whether it can differ
between two descriptors of the same logical value, and therefore whether a conversion exists
at all. It is deliberately larger than what the crate does, and a reader who needs to know
what works today should not have to infer it from an ADR's audit. Hence this table.

> **Note (non-normative):** No part of this page is normative. The format specification
> requires no implementation to transcode anything — `interchange.md`'s `TRANSCODING`
> capability is a negotiated Level 3 protocol feature, not a conversion requirement. What a
> conforming implementation must provide is in [Compliance](compliance.md); what the
> implementations actually provide is in [Implementation Status](implementation-status.md).
> This page covers one crate's conversion surface, which is neither.

**Status as of this writing: `hurray-transcode` does not exist.** ADR-044 is the design pass
that precedes its first line of code, so every implementable row below reads `deferred` and
every version column reads `—`. The rows that read `refused` are the useful content today:
those are decided, permanent, and the reason is recorded.

Unlike [Implementation Status](implementation-status.md), this page is **hand-maintained**.
It is updated in the same pass that adds, gates or refuses a conversion — a matrix maintained
after the fact is a matrix nobody trusts. Whether it can instead be generated from the
crate's own dispatch table, and drift-checked in CI, is an open question for the pass that
writes that table.

**Status vocabulary**

| Status | Meaning |
|---|---|
| `supported` | Planned from descriptors, run from buffers. Available since the version named. |
| `survey-gated` | Supported, but a survey over the data must first establish that the conversion is admissible or size its output. A survey can refuse. |
| `deferred` | In scope, not implemented yet. |
| `refused` | Permanently out of scope. The reason is part of the published surface. |

A `refused` row is a decision, not a gap: re-proposing one means arguing with the reason in
its last column, which is the point of recording it.

## Element types

| Conversion | Status | Since | Governed by |
|---|---|---|---|
| float → wider float (`float16`/`bfloat16` → `float32` → `float64`) | `deferred` | — | § 15 |
| float → narrower float, target with NaN and infinity (`float16`, `bfloat16`, `float8_e5m2`) | `deferred` | — | § 15 |
| float → narrower float, target with neither (`float8_e4m3`, `float4_e2m1`, `float6_e2m3`, `float6_e3m2`) | `deferred`, survey-gated | — | § 15 |
| integer widening (`int8` → `int32`, `uint8` → `uint32`, …) | `deferred` | — | § 14a |
| integer narrowing (`int32` → `int8`, …) | `deferred`, survey-gated | — | § 14a |
| `bool` → `int8`/`uint8` over the `{0, 1}` domain | `deferred` | — | § 14a |
| `int32` → `float32` and other integer → float casts | *undecided* | — | § 14a, open question |
| signed ↔ unsigned at the same width | `refused` | — | A bit-identical reinterpretation with a different logical value: `-1` and `255` are not the same number (§ 14a). |
| integer → `bool` | `refused` | — | A predicate, not a representation (§ 14a). |
| any cast of an extension-typed (private) tensor | `refused` | — | The extension descriptor carries enough to size a buffer, not enough to interpret a value (§ 14g). |

The arithmetic of every cast above — rounding mode, overflow, subnormals, signed zero, NaN
payloads — is pinned by ADR-044 § 15, which also explains why the specification does not pin
it yet.

## Layouts

| Conversion | Status | Since | Governed by |
|---|---|---|---|
| any dense layout → any dense layout (row-major, column-major, strided, tiled, Morton, Hilbert) | `deferred` | — | § 14, `memory-layout.md` |
| negative stride → positive stride (axis reversal normalised) | `deferred` | — | § 14b |
| zero stride materialised (broadcast expanded) | `deferred` | — | § 14b |
| repeated axis collapsed back to a zero stride | `deferred`, survey-gated | — | § 14b |
| dense → sparse (COO, CSR, CSC, CSF) | `deferred`, survey-gated | — | § 4, § 7(c) |
| sparse → dense | `deferred` | — | § 14 |
| sparse → sparse | `deferred` | — | § 14 |
| block-paged → dense (pages gathered) | `deferred` | — | § 14, § 6 |
| sub-byte repacking (`int4`, `bool` packing order) | `deferred` | — | § 14, `memory-layout.md` |
| rank change (reshape) | `refused` | — | Rank is logical: a reshape is a descriptor change, and materialising nothing is not this crate's job ([#113](https://github.com/pgillet/hurray/issues/113), corollary 1). |

Dense → sparse is survey-gated for sizing rather than admissibility: the number of stored
entries is in no descriptor. The same survey is what makes the run partitionable, since each
chunk's output offset is the exclusive prefix sum of the per-chunk counts (§ 6).

## Buffer placement

| Conversion | Status | Since | Governed by |
|---|---|---|---|
| alignment re-placement | `deferred` | — | § 14 |
| `byte_offset` rebasing (to offset 0, or packing tensors into one buffer) | `deferred` | — | § 14e |
| host → device, device → host, pinned-host staging | `deferred` (feature-gated, CI compile-tested only) | — | § 11 |
| on-device repacking (a device kernel that changes layout in place) | `refused` | — | A per-vendor repacking kernel is a kernel; the crate's name is the first argument against it (§ 11). Device → device is explicit composition: device → host, convert, host → device. |

## Quantization

| Conversion | Status | Since | Governed by |
|---|---|---|---|
| quantize: float → per-tensor / per-channel / per-block affine, NF4, MXFP | `deferred` | — | § 8, § 9, § 15 |
| dequantize: any scheme → `float32` or `float64` | `deferred` | — | § 7, § 15 |
| dequantize → narrower float | `deferred` | — | § 7 (dequantize, then a narrowing cast — lossy by composition) |
| per-tensor affine → per-channel affine (scale and zero point replicated) | `deferred` | — | § 14c |
| per-channel affine → per-block affine (each channel's scale replicated over its blocks) | `deferred` | — | § 14c |
| symmetric → asymmetric (all-zero zero-point buffer) | `deferred` | — | § 14c |
| per-channel → per-tensor, per-block → per-channel, asymmetric → symmetric | `deferred`, survey-gated | — | § 14c — admissible only if every replicated scale is in fact equal and every zero point in fact zero. |
| requantization (any lossy scheme → scheme) | `refused` | — | The loss would be a property of the path, not of the target, and it compounds. A promotion leaves the storage buffer byte-identical; a requantization rewrites it — that is the test (§ 1, § 14c). |
| NF4 or MXFP ↔ affine, in either direction | `refused` | — | A non-uniform grid and a power-of-two shared exponent have no exact affine equivalent (§ 14c). |
| `scale_type_tag` conversion | `refused` | — | The specification requires `float32` for every affine scheme in this version (§ 14c, `quantization.md`). |
| quantization of an extension-typed tensor | `refused` | — | § 14g. |

Quantization parameters are always derived from the data through a survey, never from the
descriptor's advisory statistics (§ 8). A caller that supplies its own range gets saturation
rather than an error, and the run reports how many elements saturated (§ 9).

## Sharding and composition

| Conversion | Status | Since | Governed by |
|---|---|---|---|
| shard split (one tensor → a shard grid, per ADR-004's shard section) | `deferred` | — | § 14d |
| shard merge (a complete shard set → its parent) | `deferred` | — | § 14d — an incomplete or overlapping set is refused at plan time, from the shard sections alone. |
| composite flatten (head + members → one dense tensor) | `deferred` | — | § 14, ADR-027 |
| composite split (one tensor → a partition composite) | `deferred` | — | § 14 |
| composite split into an **overlay**, `combine_op = replace` | `deferred` | — | § 14h — the target boxes determine every member's values. |
| composite split into an **overlay**, `combine_op = add` | `refused` | — | Infinitely many base-and-correction decompositions sum to the same tensor, so the plan is not a function of the descriptor sets (§ 1, § 14h). A caller that wants a residual overlay knows which residual it wants. |
| overlay materialisation applying `CombineOp` | `deferred` | — | § 14h, § 15 — the composite's logical value *is* the combination, so materialising it preserves that value; layer order follows ADR-027 and the additions happen in the head's element type with no wider accumulator. |

## Surveys and non-plan operations

| Operation | Status | Since | Governed by |
|---|---|---|---|
| statistics survey (the descriptor's statistics section, recomputed on request) | `deferred` | — | § 7 |
| data-level validation (COO coordinates, `indptr` monotonicity, block tables, composite partitions, actual alignment) | `deferred` | — | § 2, § 13 |
| canonical digest over the logical value | `deferred` | — | § 10 — not published until the algorithm is pinned, probably normatively: a canonical digest two implementations compute differently is worth nothing. |
| byte swap (`to_native` / `from_native`, in place, per buffer by that buffer's own element type) | `deferred` | — | § 12 — not a plan: the distinguishing input is which side of the format boundary the bytes sit on, which no descriptor records. |
| survey of an extension-typed tensor | `refused` | — | § 14g. |

## Not conversions at all

Three descriptor differences look like conversions and are not, because no byte moves. They
are listed so the audit's coverage is visible here too, and they belong to
[#113](https://github.com/pgillet/hurray/issues/113) rather than to this crate:

| Descriptor difference | Why it is out |
|---|---|
| `version_major` / `version_minor` | A descriptor rewrite; corollary 1. |
| `rank` and `shape` reinterpretation (reshape) | Rank and shape are logical; corollary 1. |
| A dynamic-dimension sentinel resolved to a concrete extent | A descriptor-only change, and a plan requires a fully resolved shape — an unresolved one is refused at plan time (§ 14f). |

A request whose plan turns out to be *all aliases* — a target equal to the source, an
alignment already satisfied, a `byte_offset` already zero — is **not** in this table and is
not an error: it is a supported conversion with an empty body, and it succeeds, writing
nothing (§ 5).
