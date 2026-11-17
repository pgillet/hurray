# ADR-044: `hurray-transcode` — one membership rule, and a plan that is a function of two descriptors

## Status

Proposed (2026-11-17)

The architect pass for [issue #272](https://github.com/pgillet/hurray/issues/272): the
crate boundary against `hurray-core`, the error surface, and the sync-core/parallel-edge
shape, decided before the first line is written. It also resolves that issue's one open
question — **device support is movement-only** (§ 11).

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

The rule is not new to this project. `interchange.md` § On-the-Fly Transcoding already
says it normatively for the protocol: "the logical tensor (the values at every index) MUST
be identical before and after transcoding. Only the memory layout changes." This ADR
generalises that sentence to element types and quantization, where the loss is declared by
the target, and gives it a mechanical form:

> **A conversion belongs in this crate iff its plan is a pure function of the source
> descriptor and the target descriptor.**

That is the rule as a type, not as prose. § 4 makes the plan a real object built from two
descriptors and nothing else, so an operation that cannot be planned from descriptors
alone cannot be expressed — a matmul has no plan, `arange` has no source descriptor, a
reshape annotation produces a plan with no work in it, and a requantization needs to know
which path the data took. The clause earns its keep twice: once as the paragraph a future
pass is measured against, and once as the signature that makes the measurement automatic.

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

The three shapes also answer #272's statistics question without an exception: a statistic
*is* a survey (§ 7). The issue calls statistics "the one exception" because a reduction
reads a tensor and produces no tensor; in this architecture that is not an exception, it is
one of the three shapes, and the crate needed it anyway to size a sparse buffer.

### 5. The caller allocates every destination

A run takes the input buffers as `&[&[u8]]` and the output buffers as `&mut [&mut [u8]]`,
sized and aligned per the handles the plan published. **The crate allocates no output.**

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
| **exact** | any layout → any layout, sparse ↔ dense, sparse → sparse, alignment, sub-byte packing, composite flatten/split, dequantization to `float32`/`float64` | the source's section is carried through **unchanged, mask and all** |
| **target-declared loss** | float → narrower float, quantization to any scheme | the section is **dropped** (`HAS_STATISTICS` cleared) |

An exact plan preserves the index-to-value mapping, so every field keeps exactly the truth
value it had — propagating costs no pass over the data and launders nothing, since an
advisory field the source asserted is neither improved nor degraded by being carried. A
lossy plan can move *all six* groups: a small value rounds to zero and changes `nnz`; a
narrowing cast overflows to infinity and changes `has_inf`; mean, variance and the range
all shift. Clearing beats carrying a bound, because `metadata.md` has no encoding for
"approximately".

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
(Chan–Golub–LeVeque), not by accumulating a running total, so the result does not depend on
the chunking.

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
descriptor and its buffers — and on a little-endian host both are the identity. Together
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
| **Device** | staging | one always-present variant carrying a backend name and a detail string. |

Two deliberate omissions. **No "survey required" variant** — § 4's sum type makes the
illegal order unrepresentable. **No "value not representable" variant** — a narrowing cast
rounds or saturates per the target type's own semantics, and that is precisely the loss § 1
permits the target to declare; the one case where loss is *not* the target's is
caller-supplied quantization range, and § 9 reports it as a count rather than an error.

The device variant is **not feature-gated**, so the enum has the same shape in every build.
A variant that appears and disappears with a feature makes error matching differ per build
configuration, which costs more than the one unused variant saves.

### 14. What this crate does not add

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
  structurally wrong. Round-trip equality (convert and convert back, compare bytes) and
  § 6's order-independence invariant are the two properties that carry most of that weight;
  frozen corpora remain the defence for anything the format pins.
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

## Findings Routed Elsewhere

1. **(→ `format-spec-writer`, MEDIUM)** `coo.md` requires the statistics `nnz` to equal the
   COO descriptor's `nnz`, which is authoritative — but COO forbids only *duplicate*
   coordinates, not *explicitly stored zeros*. A source that stores an explicit zero makes
   the two counts genuinely different numbers, and the spec then forces a false statistical
   field. § 7(c) sidesteps it for this crate by emitting canonical sparse output; the spec
   should say which count `nnz` is, in `coo.md`, `csr.md`, `csc.md` and `csf.md` alike.
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

## Open Questions Deferred

- **The digest algorithm, and whether it is normative.** Pinned before the digest is
  exposed; a spec section is the likely home (§ 10). Until then the digest is not published.
- **The Python and C surfaces**, and the ABI bump the C one implies (Consequences).
- **Whether `hurray-io` takes an optional dependency on `hurray-transcode`** for #226's
  `ALLOW_TRANSCODE`. The direction is settled (§ 3); the scheduling is not.
- **Whether transcoding ever runs inside a streaming write.** § 6 keeps it possible and
  nothing in this ADR commits to it.
