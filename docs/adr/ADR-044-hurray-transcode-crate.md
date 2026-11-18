# ADR-044: `hurray-transcode` — one membership rule, and a plan that is a function of two descriptors

## Status

Proposed (2026-11-17)

The architect pass for [issue #272](https://github.com/pgillet/hurray/issues/272): the
crate boundary against `hurray-core`, the error surface, and the sync-core/parallel-edge
shape, decided before the first line is written. It also resolves that issue's one open
question — **device support is movement-only** (§ 11).

> **The pass was run twice.** § 14's audit found eight conversions the issue's list did not
> name, so the decisions were re-derived against the larger scope rather than left standing
> on the smaller one. Four moved: a plan now publishes each output buffer as **aliased or
> produced** (§ 5), which makes corollary 1 computable rather than merely stated; work-item
> disjointness is an **output** property, not an input one (§ 6); statistics propagation
> needs an **arity** qualifier, because the first pass's rule was written for one-to-one and
> is wrong for a split (§ 7); and a survey can now **refuse** a conversion, which is a sixth
> error family (§ 13). The audit paid for itself before any code was written, which is the
> argument for running it on every pass that adds scope.

> **Then reviewed.** A review of the two passes found one internal contradiction and one
> false claim, and both are corrected here rather than argued with: a plan whose every
> output buffer is an alias was returned as a no-op by § 5 and refused at plan time by
> § 13 — it is now **`Ready` with zero work items** (§ 5), and corollary 1's refusals are
> stated separately from it; and § 7's claim that pairwise merging makes a survey
> independent of the chunking is **true only in exact arithmetic**, so the guarantees are
> now split three ways, with the digest pinned to index order (§ 7). It also found three
> holes where the design was silent rather than wrong, each of which becomes a contract
> here: the **numerical arithmetic** of a narrowing cast, which the spec declares out of its
> own scope and which is now pinned in full (§ 15, finding 5); what the **destinations hold
> after a failed run** (§ 13); and the fact that a `ReadyPlan` reached through a survey is
> **only as true as the bytes the survey read** (§ 4). The first was the expensive one to
> have found after publication, which is the argument for a review pass between the
> architect and the planner. Two smaller things came with it: § 1 now says that "logical
> value" means *the value the format defines*, which is what makes § 14h's `CombineOp::Add`
> an instance of the rule rather than an exception to it, and the versioned conversion
> matrix the audit is deliberately not becomes a documentation deliverable (§ 16).

**ADR-014** (layout address computation lives in `hurray-core`), **ADR-041 §§ 1–2**
(sans-io core; who owns payload memory) and **ADR-027** (composite composition rules) are
load-bearing here. None of them is amended.

## Context

No crate in this workspace reads a payload byte as a *value*. That is not an accident of
scheduling — it is the shape of the thing. `hurray_core::BufferHandle` carries a
`byte_size`, an `alignment`, a `DeviceTag`, a `SyncMode` and a `MemoryClass`, and **no
pointer**; `hurray-io` moves buffers it never interprets; `hurray-inspect` decodes
descriptors and prints payload bytes as hex. Dequantization appears nowhere in the
workspace. The cookbook page on converting an externally quantized tensor rebuilds a
zero-point plane by hand, in the user's own loop, because there is nothing to call.

Meanwhile the format specifies conversions that no implementation performs.
`interchange.md` § On-the-Fly Transcoding, § Layout Negotiation and § Server Layout
Selection are specified in full and implemented nowhere, and the Implementation Status
page says so: the whole Level 3 section reads `spec-only`. Four consumers are queued
behind the same missing mechanism —
[#110](https://github.com/pgillet/hurray/issues/110) (layout optimization policy),
[#226](https://github.com/pgillet/hurray/issues/226) (Compliance Level 3),
[#111](https://github.com/pgillet/hurray/issues/111) (Redis module) and
[#207](https://github.com/pgillet/hurray/issues/207) (foreign-file scanning) — and each
of them would otherwise grow its own conversion loops.

The risk is not that the crate is hard. It is that "convert a tensor" has no natural
stopping point, and the first thing `CLAUDE.md` says about this project is that it is not
a compute library. So the boundary has to be a rule that can refuse things, not a list
that can be appended to.

## Decision

### 1. The membership rule, and its executable form

The rule, as issue #272 states it:

> An operation belongs in `hurray-transcode` **iff** it changes only the tensor's
> representation, never its logical value — the mapping from index to value is preserved
> exactly, or with a precision loss the **target representation itself declares** — and
> the conversion is fully determined by the source and target descriptors.

With its two corollaries: **anything expressible as a descriptor change alone stays out**
(that is [#113](https://github.com/pgillet/hurray/issues/113) — materialising is in,
annotating is out), and **loss must be a property of the target, not of the path** (which
is what rules requantization out).

**"Logical value" means the value the format defines**, and the rule is relative to that
definition rather than to an intuition about arithmetic. For a dense tensor it is the
element stored at an index; for a quantized tensor it is the value the scheme's
dequantization arithmetic yields; for an overlay composite it is the `CombineOp`
combination ADR-027 defines over the layers. So § 14h's `CombineOp::Add` — which performs
additions — is admitted *by* the rule rather than excused from it, and an operation that
preserves a value Hurray never defined is not admitted at all. The boundary moves when the
format moves, and only then: the clause to argue with, when a future pass wants to widen
this crate, is the format's definition of the value, not this crate's taste.

The rule is not new to this project. `interchange.md` § On-the-Fly Transcoding already
says it normatively for the protocol: "the logical tensor (the values at every index) MUST
be identical before and after transcoding. Only the memory layout changes." This ADR
generalises that sentence to element types and quantization, where the loss is declared by
the target, and gives it a mechanical form:

> **The selection of a conversion and its rules are a pure function of the source
> descriptor and the target descriptor. Its admissibility and its output sizes may require
> the bytes.**

Both halves are load-bearing, and the first pass stated only the first — "a plan is a pure
function of two descriptors" is the sentence § 4's survey family contradicts. The size of a
COO `values` buffer is not in any descriptor, and neither is the fact that every `int32`
value in a buffer fits in `int8`. What *is* a function of the descriptors alone is **which**
transformation is performed and **by what rule**, and that is the half the membership test
needs: an operation whose very choice of transformation depends on the data is reading
values to decide what to compute, which is a compute library's job. A survey never changes
the rule; it establishes whether the rule applies and how much room its output needs.

The first half is also the rule as a type, not as prose. § 4 makes the plan a real object
built from two descriptors and nothing else, so an operation that cannot be planned from
descriptors alone cannot be expressed — a matmul has no plan, `arange` has no source
descriptor, a reshape annotation produces a plan with no work in it, and a requantization
needs to know which path the data took. The clause earns its keep twice: once as the
paragraph a future pass is measured against, and once as the signature that makes the
measurement automatic.

One generalisation, found by the § 14 audit: three operations are not one-to-one — shard
split and composite split are one-to-many, shard merge and composite flatten are
many-to-one. So the rule is over the source and target descriptor **sets**, and its first
half is a pure function of those two sets. Nothing else about it changes; the destinations
of § 5 are already a slice.

### 2. The crate boundary: the first crate that reads payload bytes as values

`hurray-core` **describes**; `hurray-transcode` is the first crate in the workspace that
**reads payload bytes as values**. That sentence is the whole boundary, and it is drawn
where the code already draws it: `BufferHandle` has no pointer, so everything `hurray-core`
can check is descriptor-internal, and everything this crate exists to do needs the bytes.

Three consequences, in the direction each must go:

- **No format semantics move into `hurray-transcode`.** Index-to-address computation stays
  in `hurray-core` (ADR-014; `byte_address_from_element_offset`, `ElementAddress`), and so
  do `NF4_LUT`, the MXFP scale encoding, `buffer_size_bytes`, packing factors,
  `valid_storage_types` and every validity constraint already implemented there. When a
  conversion needs a format fact `hurray-core` does not expose, the fix is to expose it
  from `hurray-core` — never to re-derive it here. A second implementation of the format's
  arithmetic, living next to the first, is the specific failure this rule exists to
  prevent.
- **No descriptor construction rules move in either.** An output descriptor is built
  through `hurray-core`'s own constructors and validators, so it is valid by construction
  and no output can be more permissive than a decoded one.
- **Nothing moves back out.** Data-dependent validation — COO coordinates sorted, in
  bounds and non-duplicated, CSR `indptr` monotonic, a block table within its pool, a
  composite partition covering its index space in the data, a base address actually
  meeting its declared alignment — stays *here*, even though it feels like validation,
  because it needs bytes. `CompositeValidator` stays in `hurray-core` and gains a
  data-level counterpart here, not a replacement.

### 3. Dependency direction, and why `hurray-io` is not a dependency

```
hurray-core  ←  hurray-transcode  ←  (later, optionally) hurray-io, hurray-ffi, hurray-python
```

`hurray-transcode` depends on `hurray-core` and nothing else in the workspace. It does not
depend on `hurray-io`, and the reason is the first consumer: #226's Level 3 server needs
`hurray-io` to *call* transcoding to honour `ALLOW_TRANSCODE`, which is only acyclic in
this direction. The crate also takes no `bytes` dependency and names no `Bytes` in its API
(§ 5), so it stays usable from a caller that has never heard of our I/O layer.

**The crate's scope is deliberately wider than the protocol's.** `TRANSCODING` and
`ALLOW_TRANSCODE` are layout-scoped and lossless: `interchange.md` permits a server to
change the memory layout, and nothing more. A server MUST NOT read this crate's element
casting or quantization as something negotiation authorises it to do on a client's behalf.
Policy — which conversion to perform, and whether a conversion is cheap enough to perform
at all (`TRANSCODE_LIMIT_EXCEEDED`) — belongs to the caller, as #272 requires: this crate
performs a conversion it is told to perform and holds no heuristic.

### 4. Three operation shapes: plan, survey, run

Every data-touching operation in the crate is one of three shapes, and the shapes are what
keeps § 1's rule enforceable.

| Shape | Reads | Produces | Example |
|---|---|---|---|
| **plan** | descriptors only | an output descriptor, output buffer handles, and a partition of the work | tiled → row-major |
| **survey** | payload bytes | facts; never a tensor | statistics, a digest, the `nnz` of a dense tensor |
| **run** | payload bytes | payload bytes | the conversion itself |

A plan is built from two descriptors, exposes the output descriptor and the output buffer
handles *before any byte moves*, and is the only way to reach a run. Its output descriptor
is final: the statistics propagation of § 7 is decided during planning, from descriptors,
so nothing about the output's description depends on the data — except in the one family
where it genuinely does.

**That family is dense → sparse, and a survey is how it is admitted rather than hidden.**
The size of a COO `values` buffer is the number of stored entries, which no descriptor
knows. So planning returns a sum type, not a struct:

```text
Planned::Ready(ReadyPlan)              // sizes known from descriptors alone
Planned::NeedsSurvey(SurveyPlan)       // sizes depend on the data
    SurveyPlan::survey(&inputs) -> ReadyPlan
```

Both arms converge on one `ReadyPlan` with one `run`, so "ran a plan whose sizes were
never determined" is unrepresentable rather than an error variant — the same preference for
a sum type over a flag plus a runtime check that ADR-041 § 6a settled for the C stream
surface. The survey that sizes a sparse output yields the exact `nnz` as a byproduct, which
is also what makes the subsequent run partitionable (§ 6).

**A survey serves two purposes, and the second one can say no.** Sizing is the first
(dense → sparse). *Admissibility* is the second, and § 14 is what put it there: integer
narrowing must establish that the data fits, collapsing a broadcast must establish that the
repetition holds, and demoting per-block to per-channel or asymmetric to symmetric must
establish that the scales are uniform or the zero points zero. So `survey` returns a
`ReadyPlan` **or refuses** — on conforming data, for a target it does not fit (§ 13). It
also means a survey reads *buffer* bytes rather than payload bytes specifically: the scale
and zero-point buffers are what the demotions interrogate.

**A `ReadyPlan` reached through a survey is only as true as the bytes it read.** A survey
establishes a fact about data — `nnz` is 412, every value fits in `int8`, every replicated
scale is equal — and the run that follows acts on it. The inputs handed to `run` MUST
therefore be the same bytes the survey read; **mutating an input between survey and run
voids the plan, and the crate cannot detect that it happened.** A `BufferHandle` carries no
pointer (§ 2) and a plan keeps no digest of what it surveyed, so there is nothing to
compare against, and adding a checksum would mean a second pass over the data to defend
against a caller mutating its own buffers mid-conversion. This is a documented caller
obligation, which is the only honest place for it.

What the obligation does *not* extend to is soundness. A plan's sizes are checked against
the destinations the caller actually passes (§ 13, unusable destination), and every
data-level validation a run performs is performed on the bytes in front of it, not on the
survey's recollection. So a caller that breaks the contract gets a wrong value, a
saturated element, or an error — never a write outside a destination. Bounds come from the
buffers presented to `run`; only *fidelity* came from the survey.

The three shapes also answer #272's statistics question without an exception: a statistic
*is* a survey (§ 7). The issue calls statistics "the one exception" because a reduction
reads a tensor and produces no tensor; in this architecture that is not an exception, it is
one of the three shapes, and the crate needed it anyway to size a sparse buffer.

### 5. The caller allocates every destination

A run takes its inputs as one buffer list **per input tensor** and its outputs as one
mutable buffer list per output tensor, sized and aligned per the handles the plan published.
**The crate allocates no output.**

The per-tensor nesting is there from the start because § 1's arity generalisation demands
it: shard and composite split are one-to-many, merge and flatten are many-to-one, and
adding a nesting level to a shipped signature is the retrofit the issue says to decide now
rather than later. One-to-one conversions pass a one-element slice and pay nothing.

**Each output buffer is published as either produced or aliased**:

```text
BufferPlan::Produce { byte_size, alignment }   // the caller allocates; the run fills it
BufferPlan::Alias(input_buffer_index)          // byte-identical to an input buffer
```

§ 14c is what forces this, and it is not a micro-optimization. An exact quantization-scheme
promotion leaves the *storage buffer byte-identical* and changes only the scheme's metadata
buffers, so a `Produce`-only contract would copy a multi-gigabyte weight buffer in order to
add a few kilobytes of replicated scales. With aliasing, the caller allocates the small
buffers and shares the large one. The same arm covers every other no-op case the audit
turned up: an alignment the buffer already satisfies, a `byte_offset` already zero, a
target equal to the source.

The cost of the arm is one the caller must see: an aliased output buffer **shares memory
with its input**, so the output tensor cannot outlive the input's buffers and a caller that
needs an independent copy must ask for one. Publishing the arm per buffer is what makes that
visible before the run rather than surprising afterwards.

**A plan whose every output buffer is an `Alias` is `Ready` with zero work items, and is
not an error.** The first pass said both things in two places, and this is the one that
survives. Such a plan is what a target equal to the source produces, and an alignment the
buffer already satisfies, and a `byte_offset` already zero — a *materialisation* that
happens to need no byte moved. A caller performing layout negotiation reaches that case
routinely, and refusing it would push the comparison onto every caller, which is the
comparison this crate is best placed to make: it owns the arithmetic that decides whether
two descriptions imply the same bytes. So `run` is callable on it, writes nothing, and
reports nothing moved.

**That case is not corollary 1, and the distinction matters.** Corollary 1 is about
operations whose *definition* is a descriptor change — a reshape, a version bump, resolving
a dynamic dimension (§ 14f). Those are refused at plan time as unsupported pairs (§ 13) and
never produce a plan at all, because there is no representation change to ask for. An
all-alias plan is a supported conversion with an empty body. Conflating the two reads "I
asked for the representation I already have" as "I asked this crate to do something it does
not do", which is a different sentence and a worse error.

The precedent is in `hurray-core` already: `encode_into(&mut [u8])` with an
`encode_to_vec()` convenience over it. The same pair appears here, and the reasons it is
the right default are concrete:

- It is ADR-041's answer — payload memory is the edge's business — reached without
  importing ADR-041's `DestinationAlloc` trait, which transcoding does not need. "Tell me
  the sizes, then hand me the memory" is expressible as a sizing call plus `&mut [u8]`,
  with no trait and no type parameter.
- It makes the zero-copy cases possible at all: converting straight into a NumPy array, a
  pinned staging buffer, a `Bytes` the I/O layer will write, or an `AlignedBuf`. ADR-041's
  parked question of who allocates the aligned destination — and the GIL problem a NumPy
  allocator would have hit inside `py.detach` — does not arise when the caller has already
  allocated.
- It keeps the crate allocation-free in its hot path, which is the only way the
  parallel-edge split of § 6 means anything.

All runs are **out-of-place**. The single in-place operation in the crate is the byte swap
of § 12, which is in-place *only*.

### 6. A sync core; parallelism is opt-in at the edge

The crate is **synchronous, and its default build cannot spawn a thread.** `rayon` is an
optional feature, off by default, and it contributes exactly one thing: a `run_parallel`
that fans the plan's work items over `run_range`. No code inside a plan ever calls into
`rayon`.

A plan therefore publishes a partition of its work into **disjoint, order-independent
items**, with one invariant that is also the acceptance test:

> Running every work item exactly once, in any order, on any number of threads, produces
> byte-identical output to the serial run.

**Disjointness is a property of the output, not of the input.** § 14 is full of conversions
where one source byte feeds many work items — materialising a broadcast reads the same
element once per repeat, sparse → dense reads a shared index buffer, a block-paged source
may point two logical positions at one page, and an overlay's base is read by every item.
Items therefore partition the *output*, read the input freely, and the invariant above is
stated over writes. An input is never mutated, so nothing is lost by it.

This is the discipline ADR-041 § 1 imposed on I/O, for the same reason and with the same
payoff: `CLAUDE.md` forbids mixing `rayon` into async contexts, and a default-featured
`hurray-transcode` *structurally cannot* violate that. An async caller runs `run_range`
inline in its own task; a caller that wants the cores calls
`spawn_blocking(|| plan.run_parallel(..))`. Transcoding inside a streaming write stays open
as a possibility instead of being foreclosed by a dependency.

Two notes on partitioning, because the easy cases hide a hard one. Layout permutations,
casts, packing and quantization partition by output region trivially. Sparse → dense
partitions by output region, and CSR/CSC/CSF by row or fiber, cleanly. **Dense → sparse
does not partition without the survey**: each chunk's output offset is the exclusive prefix
sum of the per-chunk counts, so the survey that sizes the output is also what makes the run
parallel. One mechanism, two jobs — which is the argument for admitting it in § 4 rather
than hiding it.

**Finding, routed below:** `rayon` is today an *unconditional* dependency of
`hurray-core` and appears nowhere in `hurray-core/src`. This crate is where that dependency
was going, and it should arrive here optional rather than stay there unused.

### 7. Statistics are a survey, never a side effect

#272 asks whether a transcode produces a new descriptor with statistics added or updated.
**A new descriptor, always — statistics recomputed, never, unless asked for.** The rule has
three parts.

**(a) A new descriptor is the only possibility.** `TensorDescriptor` is an immutable value
and the output's type, layout, buffer table or alignment differ by construction. Nothing is
edited in place; a plan publishes the output descriptor before the run.

**(b) The default is conservative propagation, decided at plan time.** Every field in the
72-byte statistics section — `nnz`, `sparsity_ratio`, `value_min`/`max`/`abs_max`,
`value_mean`/`stddev`, `nm_n`/`nm_m`, `has_nan`/`has_inf` — is a property of the **logical
value**, not of the representation. `metadata.md` makes them advisory and, decisively,
requires that a writer which cannot compute a valid statistic **omit** it rather than emit
an invalid one. So propagation follows the plan's fidelity, and nothing else:

| Plan fidelity | Conversions | Statistics on the output |
|---|---|---|
| **exact** | any layout → any layout (broadcast materialisation and stride normalisation included), sparse ↔ dense, sparse → sparse, alignment, `byte_offset`, sub-byte packing, composite flatten/split, shard split/merge, integer and `bool` widening, the § 14c scheme promotions, dequantization to `float32`/`float64` | the source's section is carried through **unchanged, mask and all** |
| **target-declared loss** | float → narrower float, quantization to any scheme | the section is **dropped** (`HAS_STATISTICS` cleared) |

An exact plan preserves the index-to-value mapping, so every field keeps exactly the truth
value it had — propagating costs no pass over the data and launders nothing, since an
advisory field the source asserted is neither improved nor degraded by being carried. A
lossy plan can move *all six* groups: a small value rounds to zero and changes `nnz`; a
narrowing cast overflows to infinity and changes `has_inf`; mean, variance and the range
all shift. Clearing beats carrying a bound, because `metadata.md` has no encoding for
"approximately".

**The table above is the one-to-one rule, and § 1's arity generalisation splits it in
three.** This is the correction the second pass owes the first: "carry the section through
unchanged" is right for one-to-one and wrong the moment a tensor is split.

- **One-to-one, exact** — carry the section through unchanged, as above. Integer narrowing
  belongs here once its survey has passed: the values are unchanged, so the conversion is
  provably exact and not merely bounded.
- **One-to-many** (shard split, composite split) — **clear every field**, with three
  entailment exceptions. The minimum, maximum, mean and variance of a *part* are not those
  of the whole, and neither is its `nnz`. But three source values do entail the same value
  on every part: `has_nan = false`, `has_inf = false`, and `nnz = 0` (with
  `sparsity_ratio = 1.0`). A `false` survives where the matching `true` does not, which is
  exact, free, and worth getting right once rather than rediscovering per layout.
- **Many-to-one** (shard merge, composite flatten) — **merge the inputs' declared
  sections, with no pass over the data**: the minimum of the minima, the maximum of the
  maxima, `nnz` as a sum, `sparsity_ratio` recomputed from it, `has_nan`/`has_inf` OR'd,
  and mean and variance combined pairwise — which is `Survey::merge` below, applied to
  declared sections instead of scanned chunks. A field is carried only when **every** input
  declares it, and dropped otherwise. `nm_n`/`nm_m` are dropped on both split and merge: an
  N:M pattern is over groups of consecutive elements, and a shard boundary can cut one.

That the merge arm is free is the pleasant half; that the split arm is a *correction* is
the half that justifies re-running the pass.

Dequantization is exact because the spec fixes the arithmetic: per-tensor, per-channel,
per-block and NF4 all define the multiply in `float32`, and a `float32` or `float64` target
reproduces the defined value exactly. A `bfloat16`/`float16`/`float8` target is a narrowing
cast *after* dequantization, and is lossy by this table's second row.

**(c) Exactly one field is set automatically, and the spec forces it.** When the output is
sparse, `nnz` comes from the conversion's own count, never from the source's statistics —
`coo.md`, `csr.md`, `csc.md` and `csf.md` all make the layout descriptor's `nnz`
authoritative and require the statistics field to match it. The conversion counted the
entries to build the buffers, so `NNZ_VALID` and `SPARSITY_VALID` are set from facts in
hand, with no extra pass, even when the source had both bits clear. Sparse output is also
**canonical**: no explicitly stored zero is emitted, and one in a sparse source is dropped
rather than carried, which is value-preserving (a dropped stored zero decodes as an
implicit zero) and is what keeps the authoritative `nnz` and the statistical one the same
number.

**Recomputation is a survey the caller asks for, on the output.** It is never a side effect
of a run, because a layout permutation need not interpret a single element and making it
decode every one — through a dequantization, for a quantized tensor — would turn a
memcpy-class operation into a value pass nobody requested.

And the accumulator is the API, which is what #272 asks for when it says the surface must
expose the purpose and not the statistic:

```text
Survey::update(&mut self, chunk)     // feed bytes
Survey::merge(self, other)           // combine two partial surveys
Survey -> Statistics                 // the descriptor section
Survey -> quantization parameters    // § 8
```

`merge` is doing two jobs: it is what makes a survey parallel over the same work items as a
run, and it is the shape that lets a survey be fed incrementally later — "cheap now, a
rewrite afterwards", as the issue puts it. Mean and variance merge pairwise
(Chan–Golub–LeVeque) rather than by accumulating a running total, which is what keeps the
result numerically sound for a large tensor.

**It does not make the result bit-reproducible across partitions, and § 6's invariant does
not reach a survey.** Pairwise merging is associative in exact arithmetic and not in
floating point, so two different chunkings of the same data can differ in the last bits of
`value_mean` and `value_stddev`. Three guarantees, stated apart, because conflating them is
how an elegant invariant comes to cover something it does not:

- **A run's output bytes** are partition-independent — § 6's invariant, which is an
  acceptance test precisely because it is testable.
- **A survey's exact facts** are partition-independent too, and for the same reason the run
  is: `nnz`, `value_min`, `value_max`, `value_abs_max`, `has_nan` and `has_inf` are a count,
  a min, a max and two ORs, all associative in the arithmetic that computes them.
- **A survey's accumulated moments** — `value_mean`, `value_stddev` — are deterministic for
  a given partition and not bit-identical across partitions. They do not need to be:
  `metadata.md` makes every statistic advisory and forbids a reader from relying on one for
  correctness (§ 8 is this crate obeying that rule about its own inputs). A last-bit
  difference in an advisory field is not a defect, and buying bit-identity with a fixed
  reduction tree would cost the parallelism the accumulator exists to allow.

**The digest is the exception, and it is pinned rather than advisory.** § 10's digest is a
survey whose whole value is that two implementations agree, so its merge is fixed to
row-major index order — partial digests combine in index order only, never in completion
order. That is a constraint on the digest's `merge`, not on the accumulator in general, and
it is the reason § 10 defers the algorithm to a normative home rather than letting each
implementation pick a reduction.

The accumulator's *scalar getters are not public*. A `Survey` converts to a `Statistics`
section or to quantization parameters, and those are its only outputs. This is the
principled refusal #272 wanted: there is no `min(tensor)` to generalise into a sum along
axis 2, because a sum along axis 2 has no descriptor section to land in.

### 8. Quantization parameters never come from advisory statistics

`derive_quant_params` computes from the **data**, through a survey, even when the source
descriptor already carries `VALUE_RANGE_VALID`.

Statistics are advisory, and `metadata.md` says a reader MUST NOT rely on one for
correctness. A scale derived from an understated `value_abs_max` clips, clipping is a loss
the target does not declare, and § 1's rule is then broken by the crate that owns it — so
"trust the descriptor" is not an optimization here, it is a correctness bug with no symptom.
A caller who knows its own statistics are trustworthy can say so explicitly, through a
separate constructor that takes the parameters (or the `Statistics`) as an input; that is
the caller's policy, held at arm's length, exactly where § 3 puts policy.

### 9. Clipping has no flag: the provenance of the parameters decides

Quantizing with a range narrower than the data is standard practice (percentile
calibration clips outliers deliberately) and is also the silent-corruption case the
cookbook's external-conversion page describes. Both are true, and `CLAUDE.md` forbids
settling it with an option, so the answer comes from where the parameters came from:

- **Derived** by the crate from a survey: clipping cannot occur — `value_abs_max` covers
  the data by construction. A clip here is an internal invariant violation, not a caller
  error.
- **Supplied** by the caller: values saturate at the representable range, because a caller
  who chose the range chose the clipping, **and the run reports how many elements
  saturated.**

The count rides in a `#[non_exhaustive]` report returned by a run — facts about the run
that the output descriptor cannot carry. The dangerous case is not clipping; it is
*unobservable* clipping, and a count removes it without adding a knob.

### 10. The canonical digest is a survey over the logical value

The digest is a function of the shape, the element type, and the logical values in
row-major index order — and of nothing else. Layout, alignment, device tag, buffer
partitioning, section flags and statistics are all excluded, which is what makes it equal
across representations of the same tensor and usable as a dedup key, an integrity check, and
`DEBUG DIGEST` in #111's Redis module.

It is a survey by § 4's definition: it reads bytes and produces no tensor. Two tensors whose
element types or stored values differ — a `float32` tensor and its NF4 quantization, for
instance — have different digests, correctly, because they are different values.

**The algorithm must be pinned before the digest is exposed**, and possibly in the spec
rather than here: a "canonical" digest that two implementations compute differently is
worth nothing. Deferred below, because it is a format question, not a crate question.

### 11. Device support is movement-only (resolves #272's open question)

**Decided: staging only. On-device repacking is out of scope, permanently.** Host → device
and device → host movement and pinned-host staging are in; a CUDA kernel that repacks tiled
to row-major on the GPU is not, and never will be, because a per-vendor repacking kernel is
a kernel and the crate's name is the first argument against it. `DeviceTag` is already in
the buffer protocol; the movement is representation change by § 1's rule, the kernel is not.

**Staging is a separate API, not a field of the conversion target.** Three structural
differences, not one preference:

1. A device destination cannot be handed over as `&mut [u8]`, so § 5's run contract cannot
   describe it.
2. Staging has no element coding and no partition into CPU work items, so § 6's invariant
   has nothing to quantify over.
3. It cannot be verified in CI, which has no GPU.

So the crate splits into two modules whose asymmetry is visible in the tree: deterministic
host-side conversion that runs and is tested everywhere, and staging that is feature-gated,
off by default, compile-tested in CI, and whose tests **skip rather than fail** — the
precedent `torch` already sets in this repo. A plan additionally requires that every source
buffer be host-resident and refuses otherwise, which is what keeps the two halves apart at
run time and not merely in the directory listing.

**A device-to-device representation change is explicit composition**: device → host,
convert, host → device. The crate documents the sequence and does **not** wrap it in one
call — the same treatment § 1 gives requantization, for the same reason. One call hides
two transfers and a host pass behind a signature that looks like a conversion.

### 12. Endianness is an ingest/egress operation, not a plan

The format is little-endian throughout, so a big-endian host has no other route in. But a
byte swap produces a tensor whose descriptor is **identical** to its input's: the
distinguishing input is which side of the format boundary the bytes sit on, which no
descriptor records. It therefore fails § 1's "determined by the source and target
descriptors" clause, and modelling it as a plan would mean adding the mode flag the rest of
this design does not need.

So it is its own pair of operations — `to_native` / `from_native`, in place, over a
descriptor and its buffers — and on a little-endian host both are the identity. It swaps
**per buffer, each by its own element type**: a swap that touches only the payload leaves
the COO index buffers, the scale buffers and the zero-point buffers wrong, and each of
those has a width the layout or the scheme declares rather than the one `type_tag` does. Together
with § 11 these are the crate's only two non-plan data operations, and they are the two
whose parameterisation lives outside the descriptor pair: host byte order, and device
residency.

### 13. The error surface

`hurray_transcode::Error`, a `#[non_exhaustive]` `thiserror` enum, with a `Result<T>` alias
— the shape `hurray-core` and `hurray-io` already have — carrying
`Core(#[from] hurray_core::Error)` for everything `hurray-core` rejects, as `hurray-io`
does.

**There is no `std::io::Error` variant, and there will not be one.** The crate performs no
I/O, and the absence is a boundary marker in the type itself.

Variants fall into five families, and *when* each is raised is part of the contract:

| Family | Raised at | Examples |
|---|---|---|
| **Unsupported conversion** | plan | the (source, target) pair is not implementable or is out of scope — a requantization asked for directly, `float` → `int` with no scale. Must name both ends. |
| **Input mismatch** | survey / run | buffer count or byte length disagrees with the descriptor (`hurray-io`'s `MultiBufferLengthMismatch` and `BufferSizeMismatch`, mirrored), plus their output-side counterparts. |
| **Unusable destination** | run | a destination shorter than the plan published, or a base address that does not meet the declared alignment. |
| **Invalid data** | survey / run | the payload contradicts its descriptor: COO coordinates out of bounds or duplicated, `indptr` non-monotonic, a block table past its pool, a partition not covering its index space. The validation surface, and what a run raises instead of producing garbage. |
| **Conversion refused by survey** | survey | the data conforms and simply does not fit the requested target: an integer narrowing with an out-of-range value, a broadcast collapse whose repetition does not hold, a demotion whose scales are not uniform or whose zero points are not zero (§ 14a, § 14b, § 14c), a NaN or an infinity bound for a target that represents neither (§ 15). Distinct from invalid data on purpose — nothing is wrong with the input, and the caller's remedy is a different target, not a repair. |
| **Device** | staging | one always-present variant carrying a backend name and a detail string. |

The audit also filled out the plan-time refusals, which are now the crate's whole
precondition surface in one place: an unresolved dynamic dimension (§ 14f), a cast,
quantization or survey of an extension type (§ 14g), an incomplete or overlapping shard set
offered to a merge (§ 14d), and a source buffer that is not host-resident (§ 11). A request
whose plan is *all aliases* is **not** among them — it is a conversion with an empty body,
and § 5 says why it succeeds.

**What the destinations hold after an error.** A run that returns an error leaves every
`Produce` destination **unspecified but initialised**: the caller handed over `&mut [u8]`,
so there is no uninitialised memory and no unsoundness, but the contents are neither the
conversion's output nor necessarily the caller's original bytes. Inputs and aliased buffers
are never written at all (§ 6: items partition the output), and the plan itself remains
valid — a caller that repairs its input may call `run` again on the same destinations
without re-planning and without reallocating. The rule is worth stating because the
alternative is worse in both directions: a crate that promised all-or-nothing would have to
buffer or re-zero the output, and a crate that said nothing would leave callers guessing
whether a half-converted buffer is safe to hand on.

**A caller that needs all-or-nothing validates first.** Data-level validation is a survey
by § 4's definition — it reads bytes and produces facts — so the structural checks of the
*invalid data* family are available as their own call (§ 2's data-level counterpart to
`CompositeValidator`) and can be run to completion before any destination is touched. What
a run cannot do is hoist those checks into planning for free: a sparse → dense conversion
discovers an out-of-bounds coordinate at the element that carries it, mid-write, and
paying for a validation pass on every run would tax the conforming case to tidy up the
failing one. Separate call, caller's choice, stated cost.

Two deliberate omissions. **No "survey required" variant** — § 4's sum type makes the
illegal order unrepresentable. **No "value not representable" variant** — a narrowing cast
rounds, saturates or overflows to an infinity per § 15's contract, and that is precisely the
loss § 1 permits the target to declare. Two cases where the loss would *not* be the
target's are handled elsewhere rather than by a variant here: a caller-supplied
quantization range, which § 9 reports as a saturation count, and a value the target has no
vocabulary for at all — a NaN or an infinity bound for `float4_e2m1` or a `float6` — which
§ 15 refuses through a survey, before any byte is written, instead of discovering it
element by element mid-run.

The device variant is **not feature-gated**, so the enum has the same shape in every build.
A variant that appears and disappears with a feature makes error matching differ per build
configuration, which costs more than the one unused variant saves.

### 14. Scope completeness: an audit over the descriptor's own fields

The in-scope list cannot be checked by remembering hard enough. The descriptor **is** the
definition of "representation", so completeness is an audit: for every field and section
of `metadata.md`, can it differ between two descriptors of the same logical value? If it
can, there is a conversion; if it cannot, there is nothing to convert. That is a check a
future pass can re-run, which a list cannot.

| Descriptor field or section | Differs for one logical value? | Conversion |
|---|---|---|
| `magic`, `descriptor_length`, `flags` | no — fixed or derived | — |
| `version_major` / `version_minor` | yes, but no byte moves | **out** — descriptor-only, corollary 1 (#113) |
| `type_tag` | yes | float → float (#272), **int → int, `bool` → int** (§ 14a) |
| `layout_tag` + layout fields | yes | every layout and sparse conversion (#272), **plus the degenerate strides** (§ 14b) |
| `rank` | no — rank is logical | a rank change is a reshape: **out**, #113 |
| `shape` | no | — , but a **dynamic dimension sentinel is refused at plan time** (§ 14f) |
| `byte_offset` | yes | **rebasing a tensor within its buffer** (§ 14e) |
| buffer table — `byte_size` | derived | — |
| buffer table — `alignment` | yes | re-placement (#272) |
| buffer table — `device_tag`, `sync_mode`, `memory_class` | yes | staging, movement-only (§ 11) |
| Quantization section | yes | quantize / dequantize (#272), **plus the exact promotions** (§ 14c); lossy requantization stays out |
| Shard section | yes | **shard split and merge** (§ 14d) |
| Statistics section | yes | a survey (§ 7) |
| Extension Type section | yes | **movement and repacking yes, value casting no** (§ 14g) |
| Composite Member section | yes | flatten / split (#272), **and an overlay applies its `CombineOp`** (§ 14h) |
| *(not in the descriptor)* | — | byte order (§ 12) |

Eight entries the issue's list does not name. Five are in scope, two are refusals worth
recording because they look free, and one is a precondition.

**(a) Integer and `bool` casts.** #272's casting row covers float → float only. Integer
widening (`int8` → `int32`, `uint8` → `uint32`) is exact and belongs here, as does `bool` →
`int8`/`uint8` over the `{0, 1}` domain `metadata.md` already defines for `bool`
statistics. Integer **narrowing** is admitted only through a survey: unlike IEEE-754, where
overflow to infinity is the target's own declared behaviour, an integer type declares no
overflow semantics, so wrapping would change the value and saturating would be a path loss.
The survey establishes the data fits, and the run refuses it otherwise — the same shape
§ 9 gives a caller-supplied quantization range.

Two refusals, both of which look like free conversions: **signed ↔ unsigned at the same
width** is a bit-identical reinterpretation with a different logical value (`-1` and `255`
are not the same number), and **int → `bool`** is a predicate, not a representation. Both
are out, permanently.

**(b) The degenerate strides.** `strided.md` makes negative and zero strides valid and
mandates support for them, which puts two conversions inside "strided → row-major" that
have nothing in common with a general index remap. A **zero stride is a broadcast**:
materialising it expands the data, by a factor the descriptor states, so it is the one
conversion in the crate whose output can be arbitrarily larger than its input — and the
only one where a caller can ask for a terabyte by accident. The reverse — collapsing a
repeated axis back to a zero stride — is value-preserving only if the repetition actually
holds in the data, so it is survey-gated. A **negative stride reverses** an axis;
normalising it to a positive stride is a byte move, and it is the case where the source's
base address is not the lowest address it reads.

**(c) Exact quantization-scheme promotions are in; only *lossy* requantization is out.**
#272 rules out quantized → quantized because the loss compounds along the path. That
argument is exactly right about the lossy cases and does not reach a small, enumerable set
where **the storage buffer is byte-identical and only the scheme's metadata buffers
change**:

- per-tensor affine → per-channel affine, with the single scale and zero point replicated;
- per-channel affine → per-block affine, with each channel's scale replicated over its
  blocks;
- symmetric → asymmetric, with an all-zero zero-point buffer;
- and their reverses, which are survey-gated over the *scale and zero-point buffers* —
  admissible only if every replicated value is in fact equal, and every zero point is in
  fact zero.

"The storage buffer is byte-identical" is the test, and it is checkable mechanically. It
also states the requantization boundary better than fidelity does: **requantization
rewrites the data buffer; a promotion does not.** NF4 and MXFP have no exact affine
equivalent — a non-uniform grid and a power-of-two shared exponent are not an affine scale
— so they are not in this set in either direction. Scale-type conversion is not available
either: `scale_type_tag` MUST be `float32` in this version of every affine scheme.

This family is not academic. It is what a consumer that implements only per-block affine
needs in order to accept a per-tensor tensor at all, and it costs a metadata buffer rather
than a pass over the weights.

**(d) Shard split and merge.** #272 treats sharding as composite flatten/split, and that
covers the composite route — but ADR-004's **shard section** is a different mechanism: a
standalone tensor annotated with its `shard_offset` inside a `parent_shape`, with no head
descriptor binding the pieces together. Both routes exist in the format and they are not
interchangeable: a composite is self-describing as a whole, where shards are self-describing
*individually*. #226's parallel shard transfers and #111's cluster sharding both move
pieces independently, over separate streams and to separate nodes, so the shard route is
the one they need. Splitting a tensor into a shard grid, and merging a complete set back
into its parent, are in scope; a merge refuses an incomplete or overlapping set at plan
time, from the shard sections alone.

**(e) `byte_offset` rebasing.** A tensor may sit at a non-zero offset in its buffer, so
moving it to offset 0, or packing several tensors into one buffer at chosen offsets, is a
conversion in its own right. Mostly it falls out of alignment re-placement; it is named
here so that the `byte_offset` row of the audit is not blank, and because sub-byte types
constrain where a tensor may start.

**(f) A dynamic shape is refused at plan time.** `shape` may carry the dynamic-dimension
sentinel (`UINT64_MAX`), and `metadata.md` forbids computing buffer sizes, strides or
element counts for it until it is resolved. A plan is sizes, so it requires a fully
resolved shape and refuses one that is not. Resolving the sentinel is a descriptor-only
change and stays out by corollary 1.

**(g) Extension-typed tensors move and repack, but never cast.** ADR-001's extension
descriptor carries `bit_width`, `packing_factor` and `is_float` — enough to compute buffer
sizes, so every layout, alignment, packing and staging conversion works on a private type
and is not blocked by it. It is *not* enough to interpret a value, so casting, quantizing
and surveying a private type are refused. #207's foreign-file scanning will meet this
first, and the split is the useful one: the bytes can be rearranged for a consumer that
knows the type, by a crate that does not.

**(h) An overlay composite's materialisation applies `CombineOp`, and that is in scope.**
`CombineOp::Add` means the logical value at an index is the base plus the sum of every
covering correction, so flattening such a composite performs additions. That looks like
arithmetic, and the rule still admits it: the composite's logical value *is defined* as the
combination, so materialising it preserves that value exactly and computing something else
would be the violation. Two consequences. Floating-point addition is order-dependent, so
the materialisation follows ADR-027's validated layer order, and § 6's order-independence
invariant quantifies over work items, never over layers. And it is worth saying plainly:
`CombineOp::Add` is the only place in this crate where a value is computed rather than
moved or recoded, and it is admissible only because the format defines the result.

### 15. The numerical contract: what "a loss the target declares" is, in bytes

§ 1 permits a loss "the target representation itself declares", and § 13 declines a
*value not representable* error on the grounds that "a narrowing cast rounds or saturates
per the target type's own semantics". That sentence was doing work the target type does not
actually do. `element-types.md` pins bit patterns, pins the clamping of out-of-range values
for `float4_e2m1` and the `float6` pair, and requires that NaN payloads and subnormals
survive *zero-copy interchange* — it says nothing about the arithmetic of a **conversion**,
and the affine schemes go further and declare it out of scope explicitly: "A consumer MAY
further convert to `float64` or to a lower-precision float type; such conversion is out of
scope for this specification."

So the target declares the *destination*, and this crate must declare the *arithmetic*. It
is pinned here rather than left to the implementation, because an unpinned rounding mode
makes two conforming implementations produce different bytes for the same request, and
because a round-trip test cannot find it — two symmetric errors cancel.

- **Rounding is round-to-nearest, ties-to-even**, for every narrowing float cast and for
  the quantize direction's divide. It is IEEE 754's default, it is what the frameworks a
  tensor arrives from already use, and ties-to-away would bias a quantized weight
  distribution the one direction nobody wants.
- **Overflow follows the target's own vocabulary.** A magnitude above the target's finite
  range becomes an infinity where the target has one (`float16`, `bfloat16`,
  `float8_e5m2`) and saturates to the largest finite magnitude of the target's sign where
  it does not (`float8_e4m3`, `float4_e2m1`, `float6_e2m3`, `float6_e3m2`) — which is the
  clamping `element-types.md` already mandates for the sub-byte floats, applied to the
  whole family that lacks an infinity.
- **Subnormals are never flushed.** A source subnormal representable in the target is
  preserved exactly; one below the target's smallest subnormal rounds under the rule above,
  which is the target's own behaviour and not a flush.
- **Signed zero is preserved**, in both directions, including a value that rounds to zero
  from below reaching `-0.0`.
- **NaN survives as NaN, and its payload does not.** A NaN input becomes the target's
  canonical quiet NaN (`0x7F`/`0xFF` for `float8_e4m3`, per `element-types.md`); the
  payload bits are dropped, because preserving them across a mantissa narrowing is not
  generally possible and a partially preserved payload is worse than none. Infinity maps to
  the target's infinity where one exists.
- **A target with no NaN or infinity makes the cast survey-gated.** `float4_e2m1` and the
  `float6` pair have neither, so there is nothing for a NaN to become: clamping it to a
  finite number would substitute a value the target never declared, which is precisely the
  path loss § 1 forbids. The survey establishes that the data contains no NaN and no
  infinity, and refuses the conversion otherwise (§ 13's *conversion refused by survey*
  family, alongside integer narrowing). A statistics section claiming `has_nan = false` does
  not substitute for the survey — § 8's rule about advisory inputs applies here too.
- **Dequantization reproduces the spec's arithmetic exactly, not something better.** The
  subtraction is `int32`, the result is cast to `float32`, the multiply is `float32`, and a
  `float64` target widens that `float32` result rather than recomputing the multiply in
  `float64`. Recomputing would be *more* accurate and would produce different bytes from
  every other implementation, which is the wrong trade for an interchange crate. MXFP is
  pinned to `float32` on the same grounds (finding 2).
- **The arithmetic is reproducible across platforms and across serial and parallel runs.**
  No fused multiply-add contraction inside a conversion, no extended-precision
  accumulation, no fast-math. § 6's byte-identity invariant is unenforceable without this,
  and an acceptance test that depends on the host's FMA availability is not an acceptance
  test.

Two things this contract deliberately does not become. It does not become an *option* —
`CLAUDE.md`'s no-flag rule applies, and a caller who wants stochastic rounding or a
different saturation policy is asking for a quantization library, which this is not. And it
does not become a claim of agreement with any particular framework's cast: ties-to-even is
what the hardware does, but a framework that flushes subnormals or canonicalises NaN
differently will disagree in those cases, and the differential tests in Consequences are
written against IEEE 754 and the OCP specifications rather than against another library's
output.

Finding 5 routes the question of whether the spec should adopt this, which it should: the
rounding mode of a narrowing cast is no more implementation business than the NF4 lookup
table is.

### 16. What this crate does not add

**No coverage-matrix row and no new column.** The matrix's rows are *spec features* taken
from `compliance.md`, and the spec requires no implementation to transcode anything. The
existing `transcoding` and `layout_negotiation` rows belong to the Level 3 protocol and
stay `spec-only` until a Level 3 server exists (#226) — they describe a negotiated
capability, not the mechanism underneath it. This is the same reasoning that gave
`hurray.aio` no row in #264.

**No new `docs/impl/` conformance document**, for the same reason: `docs/impl/` states what
an implementation must do to conform, and conformance says nothing about transcoding. The
crate's contract is this ADR plus its rustdoc. `rust-reference.md` gains a paragraph
placing the crate in the stack.

**One new document that is not a conformance document: the conversion matrix**, at
`docs/impl/transcode-matrix.md`. § 14 is an audit of the *architectural* surface — what the
descriptor permits to differ, and therefore what a conversion crate could ever be asked to
do. It is deliberately not a statement about what the crate converts today, and a reader who
needs that answer should not have to infer it from an ADR. So the matrix is the versioned
counterpart: one row per conversion, with **status** (supported / survey-gated / refused /
deferred), the **crate version** a pair became supported in, and a pointer to the § of this
ADR that governs it. The refused rows carry their reason, because "signed ↔ unsigned at the
same width" (§ 14a) being permanently out is as much a part of the published surface as the
pairs that work.

It is created by the first implementation pass, seeded with every pair § 14 names, and
updated **in the same pass that adds, gates or refuses a conversion** — the discipline
`CLAUDE.md` already imposes for `website/coverage-matrix.toml`, for the same reason: a
matrix maintained after the fact is a matrix nobody trusts. It is hand-written for now;
whether it can be generated from the crate's own dispatch table, and drift-checked in CI
the way the coverage matrix is, is left to that pass.

**No new spec section** — with one possible exception, the digest (§ 10), which is deferred
below precisely because it is the one piece of this crate that two implementations would
need to agree on.

## Alternatives Considered

**Put it in `hurray-core`.** Rejected: `hurray-core`'s defining property is that it never
dereferences a payload, which is why `BufferHandle` has no pointer. Value-domain code there
makes every descriptor consumer — `hurray-inspect`, a binding that only reads metadata —
link conversion loops it will never call, and erases the one line in the workspace that is
easy to state and easy to check.

**Put it in `hurray-io`.** Rejected: it inverts the dependency the first consumer needs
(#226's server is `hurray-io` code calling transcoding), couples conversion to `bytes` and
to framing, and asserts a relationship that does not exist — repacking a tiled buffer has
nothing to do with moving bytes over a transport.

**Call it `hurray-compute`.** Rejected in #272 and endorsed here: the name is the boundary's
first line of defence, and a proposal to add a matmul should have to argue with it.

**One `transcode(source, target) -> owned tensor` call.** Rejected: it forecloses every
zero-copy destination, needs the allocator seam ADR-041 found awkward across the GIL, and
hides the data-dependent sizing of sparse outputs behind an allocator that must grow.

**On-device repacking kernels.** Rejected — § 11. Per-vendor kernels are a compute library
by any definition.

**Compute statistics on every run.** Rejected: it turns a byte move into a value pass,
emits advisory fields nobody asked for, and `metadata.md`'s omit-rather-than-invalidate rule
already points at propagate-or-clear as the useful default.

**A clipping policy option.** Rejected under `CLAUDE.md`'s no-flag rule; § 9 derives the
answer from the provenance of the parameters instead, which is information the crate already
has.

## Consequences

- The workspace gains a sixth published crate, and `rayon` moves to it as an optional
  dependency. `hurray-core` should drop the unconditional `rayon` it does not use, and
  `CLAUDE.md`'s crate-responsibility table needs the same edit.
- `hurray-transcode` is the first crate whose tests can be *numerically* wrong rather than
  structurally wrong. § 6's order-independence invariant and round-trip equality (convert
  and convert back, compare bytes) carry much of that weight, and frozen corpora remain the
  defence for anything the format pins — but **round-trip equality is not sufficient on its
  own**, because a pair of symmetric errors cancels: a wrong rounding mode applied in both
  directions, or a swapped stride in a permutation and its inverse, round-trips perfectly.
  So § 15's contract is tested against the external references that define it — IEEE 754
  and the OCP MX specification for the float formats, hand-computed vectors for the affine
  schemes — and the float edge cases are a named corpus of their own: NaN, both zeros, both
  infinities, subnormals at every boundary, and the ties that distinguish ties-to-even from
  ties-to-away. Sparse conversions get the same treatment with a corpus of degenerate
  structures (empty, all-zero, explicitly stored zeros, a single entry, a full tensor).
- The architectural surface and the implemented surface are deliberately different sizes,
  and § 16's `docs/impl/transcode-matrix.md` is where the difference is published. § 14's
  audit is the shape of the crate at completion; which pairs land in which pass is the
  planner's call, and no pair is claimed until it has a row.
- Two halves with different testability, by construction (§ 11). CI exercises the host half
  completely and compile-tests the other.
- `hurray-inspect` gains a backend it does not have today: data-level validation (§ 2) and
  a summary survey (§ 7). Its refactor is not part of this ADR.
- The Python and C surfaces are *not* part of this pass, and `CLAUDE.md`'s parity rule is
  written against `hurray-core` and `hurray-io`. Transcoding is the kind of mechanism a
  NumPy user wants directly, so parity is expected to apply — scheduled as its own pass
  once the Rust surface has stabilised, because a C surface means another ABI bump and the
  first consumers (#226, #111) are Rust-side.
- #226 and #111 stop needing their own conversion code, and #110 gets the mechanism it is
  the policy layer for.
- Broadcast materialisation (§ 14b) is the one conversion whose output can be arbitrarily
  larger than its input, by a factor the descriptor states. A plan publishes its output
  sizes before any byte moves, so a caller — a Level 3 server, say — can refuse it against
  a budget, which is exactly what `interchange.md`'s `TRANSCODE_LIMIT_EXCEEDED` is for.
  The budget is the caller's; the number is the plan's.

## Findings Routed Elsewhere

1. **(→ `format-spec-writer`, MEDIUM)** `coo.md` requires the statistics `nnz` to equal the
   COO descriptor's `nnz`, which is authoritative — but COO forbids only *duplicate*
   coordinates, not *explicitly stored zeros*. A source that stores an explicit zero makes
   the two counts genuinely different numbers, and the spec then forces a false statistical
   field. § 7(c) sidesteps it for this crate by emitting canonical sparse output; the spec
   should say which count `nnz` is, in `coo.md`, `csr.md`, `csc.md` and `csf.md` alike.
   **Wanted before this crate's sparse output ships**, not merely before 1.0: § 7(c) is the
   right answer under only one of the two readings, and a crate that has already shipped the
   other one becomes an argument the spec then has to accommodate.
2. **(→ `format-spec-writer`, MEDIUM)** `mxfp.md` § says dequantization arithmetic is
   performed in "`float32` (or higher) precision". "Or higher" makes the output bytes
   implementation-dependent, where the other three schemes pin `float32` exactly. A
   conversion crate must produce the same bytes as every other implementation; this one will
   pin `float32`, and the spec should too.
3. **(→ `format-spec-writer`, LOW)** `interchange.md` § On-the-Fly Transcoding should state
   that the `TRANSCODING` capability covers *layout* transcoding only, so a Level 3 server
   cannot read `ALLOW_TRANSCODE` as permission to cast or quantize on a client's behalf
   (§ 3).
4. **(→ a cleanup issue)** `rayon` is an unconditional dependency of `hurray-core` with no
   use anywhere in `hurray-core/src` (§ 6).
5. **(→ `format-spec-writer`, MEDIUM)** The spec pins no **conversion arithmetic**. It fixes
   every bit pattern, mandates clamping for the sub-byte floats, and requires NaN payloads
   and subnormals to survive zero-copy interchange — but it declares conversion *away* from
   the dequantized `float32` out of scope (`per-tensor-affine.md` § Dequantization Formula),
   and nowhere states a rounding mode, a NaN-payload rule or an overflow rule for a
   narrowing cast. Two conforming implementations can therefore transcode the same tensor
   to different bytes. § 15 pins the whole contract for this crate; the spec should adopt it
   — probably in `element-types.md`, since it is a property of the types rather than of a
   layout — on the same argument that put `NF4_LUT` in the spec instead of in an
   implementation. Related to finding 2, which is the same gap in one scheme.

## Open Questions Deferred

- **The digest algorithm, and whether it is normative.** Pinned before the digest is
  exposed; a spec section is the likely home (§ 10). Until then the digest is not published.
- **The Python and C surfaces**, and the ABI bump the C one implies (Consequences).
- **Whether `hurray-io` takes an optional dependency on `hurray-transcode`** for #226's
  `ALLOW_TRANSCODE`. The direction is settled (§ 3); the scheduling is not.
- **Whether transcoding ever runs inside a streaming write.** § 6 keeps it possible and
  nothing in this ADR commits to it.
- **Whether integer → float casts are in scope.** § 14's `type_tag` row names float → float,
  int → int and `bool` → int, and no pass has yet ruled on `int32` → `float32`. § 1 admits
  it — the rounding above 2^24 is IEEE's and § 15 now pins it — but it has not been decided,
  and the first implementation pass should accept or refuse it deliberately rather than
  inherit it from a rounding rule written for something else.
