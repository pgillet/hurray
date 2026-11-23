# ADR-045: Target distinctness — a proved trichotomy, and what the format says about aliasing strides

## Status

Accepted (2026-11-22), proposed 2026-11-22

> Accepted after three review passes, recorded in the revision notes below. No
> architectural decision moved: the conservative predicate, the three-way verdict, the
> error split, the `hurray-core` placement and the format-versus-transcoder boundary are as
> first proposed. What the reviews changed was the strength of the claims used to justify
> them — two arguments were withdrawn as overreaching (the ecosystem prediction, and
> consistency of a verdict across extents), two statements were narrowed to what is
> actually true (an aliasing destination cannot represent *arbitrary* values losslessly;
> recognising a witness is constant time while materialising one is O(rank)), and the
> verification obligations became binding.

Arises from the pass that landed `hurray-transcode`'s dense layout family
([#272](https://github.com/pgillet/hurray/issues/272), **ADR-044**) and the adversarial
review of it. Refines **ADR-044** § 13's error surface — one variant becomes two — and
pins the two matrix rows § 14 left to `docs/impl/transcode-matrix.md`. **ADR-044** § 6's
soundness argument stands unchanged: widening the set of targets proved distinct never
weakens it.

Requires an amendment to `docs/spec/layouts/strided.md` § Stride Semantics. Nothing here
is a wire-format or C ABI change.

**Revision 4 (2026-11-23).** A § *Prior Art* section was added, surveying eight libraries
at source where possible. It changed no decision. It supplies three things the ADR was
missing: a precedent for § 4's sentence (MLIR states the property; its UB remedy is the
part to avoid), the argument that predicate strength is what licenses refusing the unknown
(PyTorch cannot, because its proof is density-only), and the evidence that exactness is
practical (NumPy ships a bounded solver), which moves the weight of § 1 onto consumer
absence. It also retracts an originality claim never made in this document but made around
it: the honest three-valued naming is PyTorch's prior art, not ours. One new finding is
routed below, on DLPack export.

**Revision 3 (2026-11-22), after review.** One mathematical precision fix in § 2:
*recognising* an exit is constant time, *materialising* its witness is O(rank), and the
`magnitude == reach` case is where the two part company — its colliding index places every
smaller axis at its last position. The predicate only recognises; nothing in it builds a
witness.

**Revision 2 (2026-11-22), after review.** Five clarifications, no change to the
decisions. The "one stride family must get one answer" argument is withdrawn — a verdict
depends on the shape as much as on the strides, so differing verdicts at differing extents
are legitimate; § 2 justifies the equal-magnitude proof as a free constant-time exit
instead, and
*Alternatives Considered* § 3 rejects bounded enumeration on added complexity, a
shape-dependent performance threshold, and consumer absence. `AliasingTarget` no longer
claims the destination cannot hold the tensor's values: it aliases distinct logical indices
and cannot represent *arbitrary* values losslessly, and the refusal is this transcoder's,
unconditionally. § 4 states that an aliasing descriptor is not intrinsically invalid —
shared storage necessarily yields equal observed values — so the producer obligation is
about fidelity to the source, and valid serialized aliasing stays distinct from an unsafe
arbitrary-value target. The enum-stability commitment is stated as refinement without new
variants. § *Verification Obligations* is now MUST-level and carries the invariant that no
input may produce a false `Distinct`.

**Revision 1 (2026-11-22), after review.** Four changes, all narrowing claims rather than
changing the decision: § 1's cost argument no longer predicts what the ecosystem produces,
and rests on the negotiation surface plus the absence of a consumer; § 2 states the
commitment that licenses a closed enum, and pins the precedence of the arithmetic-free
proofs over span overflow; § 4 bounds the spec amendment to the emitted descriptor's own
value and keeps *representable* apart from *materialisable*; *Alternatives Considered* § 1
no longer implies the only alternative to today's test is a general solver. § *Verification
Obligations* was added and is binding on the implementation pass.

## Context

A tensor cannot in general be *written* through a layout whose strides send two logical
indices to
one stored element. Serially the second write silently overwrites a value the conversion
promised to preserve; in parallel the two writes can fall in different work items, which
is a data race through a raw pointer. `Remap::new` therefore refuses such a target, and
the refusal is the second of the two invariants `hurray-transcode/src/parallel.rs` rests
on — items partition the output cover, and distinct cover indices address distinct
elements.

The trap the check exists to close is that **a zero stride is not the only way to alias**.
Strides `[1, 1]` over shape `[2, 2]` send `[0, 1]` and `[1, 0]` both to offset 1, with no
zero stride anywhere. Before the check landed, that target was accepted.

The test in `hurray_core::strides_provably_distinct` is a sufficient condition — sort the
axes by stride magnitude ascending and require them strictly nested, each magnitude beyond
the total span of every smaller axis. It admits every layout § Contiguity calls dense,
every permutation of one, and every padded or reversed variant. It does not admit every
injective stride vector: `[2, 3]` over shape `[3, 2]` is injective and is refused.

Three things about that arrangement were left unsettled, and this ADR settles them.

1. The refusal is **conservative**, and nothing recorded why that is the right trade rather
   than a limitation to apologise for.
2. `Distinctness` is `Distinct | NotProven`, `#[non_exhaustive]`, with a comment
   anticipating a split into "aliases" and "not proven" on the grounds that the remedies
   differ. The anticipation is right and cheaper than it looks.
3. The format says nothing at all about a non-zero stride vector that aliases. It is legal
   by omission, which leaves a conforming consumer with no account of what such a tensor
   means.

## Decision

### 1. A write target must be *proved* distinct. The conservatism is a one-way door

The contract stands: `plan()` refuses any target the predicate does not prove distinct,
and refuses it at plan time.

The refusal is **unconditional, and deliberately coarser than the mathematics.** An
aliasing destination is not incapable of holding *some* tensors: it represents faithfully
exactly those whose values agree at the indices it aliases. What it cannot do is represent
arbitrary values losslessly — and a plan is built from descriptors, before any value has
been read, so "this caller's data happens to agree" is not something `plan()` is in a
position to know. The crate has exactly one mechanism for establishing such agreement, the
survey behind the broadcast collapse, and that case never reaches this check: its axes are
clamped out of the output cover first. Everything else is refused, and the refusal is a
property of this transcoder rather than of the format (§ 4).

The argument that makes this right is not the sufficiency of the test but the **direction
of its error**. Widening the predicate later turns an error into a success, which is
additive and breaks no caller. Admitting first and tightening later is the breaking
direction — and in the interval it ships silent value loss, which is the one failure this
crate must not have. Given a test that will certainly be imperfect, the imperfection
belongs on the side that can only ever be relaxed.

Two facts bound the cost of that choice, and a third rules out the obvious alternative.

**The refused set is unreachable through the format's own negotiation surface.**
`interchange.md` § Layout Entry Encoding makes a layout preference a single `layout_tag`
byte, plus opaque metadata for the private range. A peer cannot ask for `Strided([2, 3])`
— it cannot ask for a stride vector at all. So no negotiated transcode can land in the
refused set, whatever a caller's buffers look like.

**No supported negotiation path and no identified consumer of this crate requires such a
target, so a more complete injectivity test is not justified at this time.** That is the
whole of the claim, deliberately. Non-nested injective strides are *not* ecosystem
exotica: `numpy.lib.stride_tricks.as_strided` will build one on request, and an
interleaved view produced by a custom kernel can be perfectly injective without being
strictly nested. The decision rests on there being no consumer today, not on such layouts
being unrealistic — and § *Open Questions Deferred* names the trigger that would reopen
it.

**Accepting an unproven target instead is unsound.** Running it serially still loses a
value when the target genuinely aliases, and by construction the crate cannot tell which
case it holds — that is what the verdict says. Nor is the empirical test cheap: building
the offset set and checking it for duplicates costs a second full pass and a bitset over
the addressed extent, charged to every conversion. The space between that and today's
condition is not empty, and *Alternatives Considered* § 1 records what lives there — as
does § *Prior Art*, which notes that NumPy has shipped a bounded exact solver since 1.13,
so nothing here should be read as claiming exactness is impractical.

**And the strictness is affordable because the predicate is strong.** This is the argument
§ *Prior Art* supplies and it is the better defence of the refusal: PyTorch cannot refuse
its own unknown verdict, because its proof of distinctness requires true density, which
puts ordinary strided slices and every reversed view into the unknown. The strictly-nested
test with the sign reduction proves those distinct, so our unknown region holds only
interleavings nobody has asked for — and refusing a region that small costs a caller
nothing it can observe.

**No normative list of targets a writer must accept.** ADR-044 § 16 decided that
transcoding gets no conformance document and no coverage-matrix row, because the format
requires no implementation to transcode anything. A `SHOULD accept` list would contradict
that. What the format *should* say is a different sentence — § 4.

The obligation is a **write** obligation and nothing more, and the all-alias path in
`plan()` is the cleanest evidence: when `src_strides == dst_strides` the plan returns
`Ready` with an alias before `Remap::new` runs, so an aliasing source asked to "convert"
to its own layout succeeds. Nothing is written, no value is at risk, and the descriptor
is one the format permits. `hurray-core` must keep accepting these descriptors.

### 2. `Distinctness` becomes a trichotomy, and two of its three answers are proofs

The enum gains `Aliases`:

```rust
pub enum Distinctness {
    /// Proved: no two distinct indices of the shape address the same element.
    Distinct,
    /// Proved: two distinct indices address one element. A witness pair exists.
    Aliases,
    /// Not proved either way.
    NotProven,
}
```

This is worth doing because **the current negative exits are already proofs of aliasing**,
and reporting them as "could not prove" understates what the test knows. Three exits, each
**recognised** in constant time from quantities the scan already holds — so detection adds
nothing to the existing cost:

- **A zero magnitude on an axis of extent above 1.** Two indices differing only on that
  axis collide.
- **Two movable axes of equal magnitude.** `i_a = 1, i_b = 0` and `i_a = 0, i_b = 1`
  collide, at any extents.
- **`magnitude == reach`.** `reach` is `Σ |s_j| · (d_j − 1)` over the axes of strictly
  smaller magnitude, and that maximum is *attained* — set every one of those axes to its
  last index. So the index with `i_a = 1` and zeros elsewhere, and the index with
  `i_a = 0` and every smaller axis at `d_j − 1`, land on the same offset and differ in
  axis `a`. (`reach == 0` means no smaller axis, which forces `magnitude == 0`, already
  caught by the first exit.)

Recognising an exit and **materialising** its witness are different costs, and only the
first is constant. A witness is a pair of rank-length index vectors, so writing one out is
O(rank) whichever exit produced it — and the third exit is where the distinction shows:
`magnitude == reach` is recognised by one comparison against an accumulator the loop
already maintains, but the colliding index it certifies places *every* smaller axis at
`d_j − 1`, so the coordinates span the axes rather than sitting on one. The predicate only
ever recognises; nothing in it builds a witness, which is also why `Distinctness` stays
`Copy` (*Alternatives Considered* § 5).

The sign reduction carries each witness back to the original strides: `i_k ↦ d_k − 1 − i_k`
is a bijection of the index box, and it shifts both members of the pair by the same
constant.

Only `magnitude < reach` is a genuine unknown, and it is **two-sided** — `[2, 3]` over
shape `[3, 2]` is injective, `[2, 3]` over shape `[4, 3]` aliases (offset 6 twice). That
region is where the subset-sum problem lives, and the test stops at its edge.

The equal-magnitude exit is a correction to the shape the review proposed, not an extra.
A verdict is a property of the stride vector **and** the shape, and different extents may
legitimately earn different verdicts — so the argument for it is not consistency across
extents, it is that the proof is available and free. Without it, `[1, 1]` reports `Aliases`
at shape `[2, 2]`, where `magnitude == reach` fires, and falls back to `NotProven` at shape
`[3, 3]`, where it does not; both answers are *correct*, but the second discards a
collision the test can recognise with one comparison. An exit that converts an honest unknown into a
proof at no cost is worth taking wherever it exists, and this is one of the two places it
does.

The change is a handful of lines inside the existing loop and costs nothing at run time.

**The two arithmetic-free proofs are established before any span arithmetic.** A zero
magnitude and a duplicate magnitude are both decided by inspection, so they are settled in
a pre-pass over the sorted axes, ahead of the loop that accumulates `reach`. The
consequence is deliberate: a target that aliases is reported as `Aliases` even when its
span arithmetic would overflow `uint64` and today raises `AddressOverflow`. Aliasing is the
more specific and more permanent fact about such a target, and it is true independently of
whether the span is representable. `AddressOverflow` still fires on the paths that genuinely
need the span — the ones heading for `Distinct` or `NotProven`.

**`#[non_exhaustive]` comes off, and the commitment that licenses it is explicit.**
Proved-yes, proved-no, unknown is a complete trichotomy, but API exhaustiveness is a claim
about future evolution rather than about mathematics, so the claim is made in full:

> `Distinct`, `Aliases` and `NotProven` are **semantic verdicts about a stride vector and
> a shape**, not descriptions of proof algorithms. Proof strategy and proof diagnostics are
> deliberately outside the enum. A later pass that improves the proof algorithm **refines
> the classification without adding a variant**: targets migrate out of `NotProven` into
> whichever of the two proved verdicts is true of them, every existing `Distinct` and
> `Aliases` answer stands unchanged, and nothing about *which* argument proved a case
> belongs in the answer.

So a variant pair like `DistinctByStructure` / `DistinctByEnumeration` is not a future
possibility this enum is keeping room for; it is the proof strategy leaking into the
verdict, and a caller branching on it would be coupling to our internals. If diagnostics
are ever wanted — a witness pair, the axis that failed, the argument used — they arrive as
a separate return type beside the verdict, not as variants of it. On those terms the enum
is closed. The workspace has the precedent: `Planned` is deliberately exhaustive so that a
caller must handle every arm.

### 3. The error splits in two: a permanent refusal and a limitation

`Error::OverlappingTarget` is replaced by:

| Variant | Means | Family (ADR-044 § 13) |
|---|---|---|
| `AliasingTarget { strides, dims }` | The destination aliases distinct logical indices and cannot represent arbitrary tensor values losslessly. This transcoder rejects it unconditionally. | unsupported conversion |
| `TargetNotProvablyDistinct { strides, dims }` | This crate's test — magnitudes strictly nested — could not prove the target injective. It may well be. | unsupported conversion |

Two reasons, and the first is a defect rather than a preference.

**The current message is false for part of the set it describes.** It reads "strides … can
map two indices to one element", and for `[2, 3]` over `[3, 2]` that is not true; the
variant name asserts the same false thing. `Distinctness` was given a named verdict
precisely to stop a caller from reporting "this layout aliases" on an answer that never
made the claim — and the error reintroduced the misreading one layer up. This must be
fixed whether or not the enum splits.

**The split is a capability, not tidiness.** `docs/impl/transcode-matrix.md` already
publishes these as two rows with two statuses — `refused` and `deferred` — and the API
collapses them into one error, so a caller writing layout fallback policy cannot tell
"never retry this target" from "retry on a newer version". Layout fallback is
[#110](https://github.com/pgillet/hurray/issues/110), the first named consumer of this
crate. The matrix promises a distinction the error surface does not expose.

There is no compatibility cost. `hurray-transcode`, `strides_provably_distinct` and
`Distinctness` are all unreleased — the CHANGELOG carries them under `[Unreleased]`, and
the matrix dates the crate from 0.3.0.

### 4. The format defines what an aliasing stride vector *means*. It does not define the test

`strided.md` § Stride Semantics names the zero stride as a broadcast and mandates support
for it. It says nothing about a non-zero stride vector that maps two indices to one
element. § Buffer Size's arithmetic is indifferent to injectivity and § Element Address
requires only that every element be addressable, so `Strided([1, 1])` over shape `[2, 2]`
is **legal by omission** — and a consumer that meets one reads four logical elements out
of three stored ones, obtaining a tensor silently constrained to `t[0,1] == t[1,0]`, with
nothing in the format acknowledging that the constraint exists. A descriptor property that
changes what the data means, described nowhere, is the opaque-semantics failure in its
layout form.

The amendment (routed below) generalises aliasing, keeps the zero stride as its mandated
special case, and states the consequence normatively: a producer MUST NOT emit an aliasing
strided descriptor unless the logical values at the aliased indices are equal, and a
consumer MUST treat aliased indices as carrying one value.

**What that obligation is about is faithfulness, not validity.** An aliasing descriptor is
not intrinsically invalid, and cannot be: shared physical storage *necessarily* yields
equal observed values, so a consumer reading one always sees agreement at the aliased
indices, and the tensor it obtains is well defined. The only thing that can go wrong is
the producer's claim — if the source tensor's values at those indices differ, the
descriptor does not represent the tensor the producer says it is representing, and the
disagreement is unrecoverable by the time a consumer sees it. The requirement is therefore
on the producer's fidelity to its source, and the zero stride is the case where the format
has always relied on it: a broadcast axis is emitted precisely because the values along it
agree.

Four scope points must be unambiguous in the wording, because the obligation is narrower
than it first reads.

- **It is evaluated on the tensor the descriptor describes, at the moment the descriptor
  is emitted.** It is a statement about the represented value, not a guarantee about the
  buffer's future. The format already covers the rest: `buffer-protocol.md` § Zero-Copy
  Invariants (2) makes in-place mutation after handoff a protocol error outright, so an
  aliasing descriptor needs no concurrency rule of its own, and the amendment must not
  invent one.
- **It binds a descriptor, not a buffer.** One buffer may carry several descriptors with
  different strides; each is read on its own terms, and an aliasing descriptor says
  nothing about what any other descriptor over the same bytes means.
- **A serialized aliasing layout is valid; an arbitrary-value transcode target is not
  safe.** These are different objects and the wording must keep them apart. A descriptor
  that *arrives* with aliasing strides is conforming, readable, and carries a well-defined
  tensor — a reader MUST accept it. A caller-chosen aliasing layout used as the
  *destination* of a materialisation is the unsafe case, because the values to be written
  are arbitrary until they are known. The format permits the first and says nothing
  compelling about the second.
- **Representable is not materialisable.** Following from the above: the format places no
  obligation on a writer to materialise arbitrary values into an aliasing layout. The
  non-normative note should say so plainly — an implementation materialising a tensor into
  a caller-chosen strided layout cannot honour an aliasing target in general, and SHOULD
  refuse it — and that is a statement about writers, not a constraint on descriptors.

**There is a precedent for the sentence, and a counter-example for the remedy.** MLIR's
`memref` states exactly this property at the type level — "the layout must avoid internal
aliasing, i.e., two distinct tuples of in-bounds indices must be pointing to different
elements in memory" — which is evidence the property is the right thing for a format to
say. Its remedy, undefined behaviour, is the part not to copy: MLIR had to re-admit zero
strides in 2024 to support broadcast, and now carries both that UB clause and a
constructible aliasing layout. § *Prior Art* has the detail. A producer obligation plus a
writer refusal is what keeps the two requirements compatible.

**The spec must not state a decision procedure, and must not state this sufficient
condition.** Mandating a procedure forces every implementation into subset-sum. Mandating
our condition bakes one implementation's conservatism into the format, and would make both
a cleverer implementation (accepts `[2, 3]` over `[3, 2]`) and a simpler one (accepts only
canonical permutations) non-conforming. The format states the property; an implementation
chooses its proof. That separation is what makes § 1's one-way door a local decision
rather than a gap every other implementation has to reproduce.

### 5. The predicate stays in `hurray-core`, and nothing else calls it yet

Injectivity of an affine index map is a pure function of shape and strides —
descriptor-internal, no payload bytes, no `unsafe` — and it belongs beside
`dense_element_strides` and `element_offset_span`, which take the same inputs. ADR-044 § 2
is explicit that index-to-address arithmetic stays in `hurray-core` and that re-deriving it
in `hurray-transcode` is the failure to prevent. "It exists for `hurray-transcode`" is not
an argument against the placement: so does `dense_element_strides`.

The placement is also where this ADR differs from the two libraries that enforce the same
rule at descriptor creation, oneDNN and MLIR, and § *Prior Art* argues the difference is
forced: they are the only producer of the descriptors they read, while an interchange format
must accept conforming descriptors it would never emit. That is the reason the first row of
the table below says what it says.

Every other candidate call site is correctly silent, and stays silent:

| Candidate | Calls it? |
|---|---|
| `hurray-core` descriptor validation | **No.** The format permits aliasing strides and mandates the zero-stride case. Rejecting at construction would make the reference implementation refuse conforming descriptors and break reading conforming files. |
| `hurray-io` writers | **No, structurally.** The crate contains no stride arithmetic at all; ADR-041's cores move whole buffers and never address an element. There is no write-through-a-layout path to guard. |
| `hurray-ffi` | **No caller today.** Exposing the predicate would be a new ABI function serving nobody; it folds into ADR-044's open question on the C/Python transcode surface. |
| `hurray-python` authoring and DLPack | **No.** Importing from `np.broadcast_to` legitimately yields zero strides, and on export DLPack hands out a pointer and strides under DLPack's own contract — NumPy's `as_strided` carries the identical hazard and documents it rather than preventing it. |

## Prior Art: How Eight Libraries Handle Stride Aliasing

This section was written after the decision, not before it, and it changed none of it. It
is here because the decision's cost and placement arguments turn on what other systems do,
and those arguments were first made from memory — which is how two of the claims this ADR
had to withdraw got in. What follows is what was checked, how, and what it implies.

**Provenance.** NumPy was run (2.5.2); `ndarray` 0.16.1, PyTorch (`aten/src/ATen/MemoryOverlap.{h,cpp}`,
`c10/core/Contiguity.h`), oneDNN (`src/common/memory_desc.cpp`) and TensorStore
(`tensorstore/strided_layout.h`) were read at source; MLIR, cuTENSOR, XLA and DLPack come
from their published documentation and one merged LLVM pull request. Where a conclusion is
an inference rather than a quote, it says so.

| Library | Where the rule lives | Proves distinct | Proves aliasing | On "don't know" | Broadcast |
|---|---|---|---|---|---|
| **Hurray** | write target, at plan time | magnitudes strictly nested; padding and negatives admitted | zero stride, duplicate magnitude, `magnitude == reach` | **refuse** | legal descriptor; collapsed axis + survey |
| PyTorch | write target, at op dispatch | dense, minimum stride 1 | zero stride only | permit | zero stride, writes refused |
| `ndarray` (Rust) | write target, at construction | magnitudes strictly nested | — (one bool for both) | refuse | read-only views only |
| oneDNN | **descriptor creation** | each stride ≥ previous × extent | — (rejected as invalid) | refuse | explicit zero-stride exception |
| MLIR `memref` | **the type itself** (UB otherwise) | — (stated as a property, no test) | — | UB | zero strides re-admitted, 2024 |
| NumPy | nowhere, for destinations | exact Diophantine solver | the same solver | `TooHardError` | `writeable=False` |
| TensorStore | no notion of it | — | — | — | zero byte strides |
| DLPack | silent | — | — | — | not addressed |
| XLA / JAX, cuTENSOR | unrepresentable | n/a | n/a | n/a | tiling / mode omission |

### The same algorithm, four times

Four of the eight implement **our exact sufficient condition** — sort the axes by stride
magnitude, skip the axes of extent 1, require each stride to clear the span of all smaller
axes — and they did so independently.

`ndarray::dimension::dim_stride_overlap` is the closest: increasing `|stride|`, extent-1
axes skipped, empty shape means no overlap, and `s <= sum_prev_offsets` is the rejection.
It is consulted only when the check mode is not `ReadOnly`, and it fails with
`ShapeError::Unsupported`. Same condition, same read/write asymmetry, same error family.
Its one flaw is the one § 2 spent a paragraph avoiding: the function returns a `bool` named
"overlap", so it reports overlap for `[2, 3]` over `[3, 2]`, which is injective.

oneDNN's `memory_desc_strides_check` is the same walk with the broadcast exception written
into it — `if ((strides[d] == 0) || (md.padded_dims[d] == 1)) continue;`, under the comment
"Make an exception for `strides[d] == 0` as it has broadcast semantics" — rejecting
`strides[d] < min_stride` and then advancing `min_stride = strides[d] * blocks_ratio`. That
is marginally stricter than ours: it demands stride × extent where we demand
stride × (extent − 1) + 1, so it refuses a few layouts we prove distinct.

PyTorch's proof is the outlier, and the difference is load-bearing. `has_internal_overlap`
returns `No` only when `is_non_overlapping_and_dense_or_false()` holds, and
`_compute_non_overlapping_and_dense` requires **equality** against a running
`require_stride` that starts at 1. Padded layouts therefore fail it, and so does every
negative stride, since `require_stride` is positive. Both are cases our sign reduction and
strict inequality prove distinct.

### Where the rule lives follows from what the thing is

oneDNN and MLIR both put the obligation at **descriptor validity**: oneDNN returns
`status::invalid_arguments` at memory-descriptor creation, so an aliasing descriptor cannot
be constructed and `reorder` — its transcoder — needs no target check at all; MLIR states
it as a property of the `memref` type, and violating it is undefined behaviour.

That is the option § 5 rejected for `hurray-core`, and the contrast explains why rather
than weakening it. oneDNN only ever reads descriptors it created, so refusing at
construction costs it nothing. An interchange format must be able to **read** a conforming
descriptor that someone else produced — including the aliasing ones its own writer would
refuse to target — so the obligation cannot live at construction without making the
reference implementation reject conforming input. Same algorithm, different layer, and the
layer is determined by whether you are the only producer.

MLIR also supplies the cautionary tale for the UB remedy. Its `StridedLayoutAttr` originally
banned zero strides; LLVM PR #116463, merged 2024-11-21, removed the ban because a zero
stride is how broadcast reaches a multi-dimensional DMA through `memref.copy`. The argument
for removal is one this ADR knows well: the ban "didn't fully solve the problem anyway —
internal aliasing can still occur when the stride is less than the shape". That is an
independent discovery of the `[1, 1]`-over-`[2, 2]` trap, used to show that a
verifier-level ban cannot be the mechanism. MLIR now holds both "internal aliasing is
undefined behaviour" and "zero strides are constructible", which cannot both be clean. Our
split — the descriptor is legal, the *write* is refused — is what resolves that tension,
and § 4 is deliberately a producer obligation rather than a UB clause for this reason.

cuTENSOR takes the third route and removes the representation: "Each stride must be larger
than zero; to be precise, a stride of zero can be achieved by omitting this mode entirely."
Broadcast is expressed structurally, by the absence of a mode, which is our broadcast
collapse moved from the plan into the descriptor. XLA goes furthest — a `Layout` is a
`minor_to_major` permutation plus tiling, with no stride vector anywhere — which is the
same structural argument § 1 makes about our negotiation surface, obtained by not having
the feature.

### Predicate strength is what licenses the policy

PyTorch refuses only a proof: `assert_no_internal_overlap` is
`TORCH_CHECK(has_internal_overlap(t) != MemOverlap::Yes, ...)`, so `TooHard` is permitted,
and `[1, 1]` over `[2, 2]` is `TooHard`. The consequence is the NumPy failure mode reached
by another route: a silently lost value.

It could not easily choose otherwise. With a density-only proof, an ordinary strided slice
and every reversed view land in `TooHard`, so refusing the unknown would refuse a large
part of normal use. **Our stricter policy is affordable precisely because our predicate is
stronger** — the strictly-nested test with the sign reduction leaves a much smaller unknown
region, containing only interleavings nobody has asked for (§ 1). That is the defence of
the refusal, and it is stronger than the cost argument this ADR originally rested on.

What PyTorch does have, and what this ADR should not claim as original, is the honest
three-valued naming. `MemoryOverlap.h` documents it in the enum itself: "No: Absolutely no
memory overlap / Yes: Absolutely yes memory overlap / TooHard: There might be memory
overlap, but it was too expensive to compute." That is `Distinct` / `Aliases` /
`NotProven`, with the same reasoning, predating us. The trichotomy is shared prior art; the
refusal on the unknown verdict is the part that is ours.

### NumPy: the exact answer exists, and is spent elsewhere

NumPy's `mem_overlap.c` solves bounded Diophantine systems, exposed as
`np.shares_memory(..., max_work=...)` — documented to raise `TooHardError` past the work
limit — and as a self-overlap detector reachable from the test module. Called directly, it
answers `True` for `[1, 1]` over `[2, 2]`, `False` for a padded row-major slice, and
**`False` for `[2, 3]` over `[3, 2]`**: it decides exactly the case this ADR classifies
`NotProven`.

So exactness is not impractical, and *Alternatives Considered* § 1 must not be read as
claiming it is — a bounded exact solver has been in production since NumPy 1.13. Two things
keep the decision where it is. The solver's verdict depends on a work budget, which is the
shape-dependent threshold § 3 rejects, and NumPy's answer to that is to expose `max_work`
as a caller parameter — a knob on `plan()` of exactly the kind this project forbids. And
the targets it would unlock cannot be requested through negotiation at all. The rejection
rests on consumer absence, as revision 2 put it, and not on cost.

What NumPy does **not** do is guard a destination. `np.broadcast_to` returns
`writeable=False`, which handles the zero-stride case by flag rather than by analysis;
`as_strided` hands back a writeable `[1, 1]` view, and writing into one through `copyto`,
through a ufunc `out=`, or by assignment raises nothing and loses a value. Verified: writing
`[[10, 11], [12, 13]]` into such a view over a four-element buffer leaves the buffer
`[10, 12, 13, 3]`.

### TensorStore and DLPack say nothing

`tensorstore/strided_layout.h` carries no aliasing or overlap concept; the nearest thing is
`IsBroadcastScalar`, "checks if `layout` contains at most a single distinct element", beside
a `BroadcastStridedLayout` that *produces* zero byte strides. A code search for "injective"
across the repository returns nothing, and the write documentation describes which positions
are modified without addressing two input positions landing on one. (Checked: the layout
header and the public write documentation, not every internal copy path.)

DLPack documents `strides` as "in number of elements, not bytes" and, from v1.2, forbids
`NULL`; it says nothing about injectivity, zero strides, or two index tuples addressing one
element. This is the protocol Hurray exports tensors through, which makes the silence ours
to answer rather than merely to note — see *Findings Routed Elsewhere*.

## Alternatives Considered

1. **Decide injectivity exactly, in general.** Bounded subset-sum. Rejected: unbounded
   cost in the hot path of every plan, for a target with no consumer. The choice is *not*
   between today's condition and a general solver, and this ADR does not claim it is —
   the middle ground holds number-theoretic tests for the rank-2 case, recognition of
   specific named interleavings, and caching a proof per stride vector. Each is tractable
   and each is rejected only *for now*, on the same ground: no consumer. They are the
   first things to reach for when the trigger in § *Open Questions Deferred* fires, and
   they are cheaper than the general solver this item rejects. NumPy's `mem_overlap.c` is
   the existence proof that a bounded exact solver is practical (§ *Prior Art*); what it
   does not remove is the consumer-absence argument, or the work-budget knob that exactness
   with a cap requires.

2. **Check empirically — build the offset set, look for a duplicate.** A second full pass
   plus a bitset over the addressed extent, which for a sparsely addressing layout is
   enormous. Rejected for cost, and it answers a question no caller is asking.

3. **Enumerate exactly when the index space is small.** Would make `[2, 3]` over `[3, 2]`
   work. Rejected on three counts, none of them consistency across extents — a verdict
   depends on the shape as much as on the strides, so differing answers at differing
   extents are legitimate. First, it is a second algorithm with its own correctness burden
   and its own tests, bolted beside a predicate that is currently two lines. Second, the
   threshold is a performance cliff whose position is a guess: planning cost becomes a
   function of `Π dims`, and a caller gets a precise verdict or a conservative one
   depending on which side of an arbitrary constant its shape falls — a worse contract to
   document than a uniform sufficient condition. Third, no consumer has asked for it
   (§ 1), so the complexity buys nothing measurable today.

4. **Keep one error variant carrying a `Distinctness` field.** The error would carry a
   verdict that can never be `Distinct`, an unrepresentable state the workspace avoids
   elsewhere, and a caller writing policy would match on a field rather than a variant.

5. **Carry the collision witness in `Distinctness::Aliases`.** Tempting — `RepetitionNotHeld`
   sets the precedent of naming the offending index. Rejected: it makes the verdict
   allocate and stop being `Copy`, for a diagnostic any caller can reconstruct from the
   strides and dims the error already carries.

6. **A normative list of target layouts a writer MUST accept.** Contradicts ADR-044 § 16,
   and there is no normative hook for it: the only place the spec discusses a transcode
   target is § Server Layout Selection, where a target is a tag.

7. **Move the predicate into `hurray-transcode`.** Rejected per ADR-044 § 2 — it would be
   the second implementation of format arithmetic, and the one that drifts.

## Consequences

- `Distinctness::NotProven` shrinks to the region where its name is literally true, and
  every negative answer the crate gives is either a proof or an admitted unknown.
- A caller can write layout fallback policy: `AliasingTarget` is permanent,
  `TargetNotProvablyDistinct` may lift in a later release. This is the distinction the
  matrix publishes and the API has so far withheld.
- `hurray-core` grows no new public function, only a third variant and a sharper doc
  comment. The predicate's cost is unchanged.
- One answer changes shape: a target with duplicate or zero magnitudes whose span
  arithmetic overflows now returns `Ok(Aliases)` where it returned `Err(AddressOverflow)`.
  Deliberate, per § 2 — the caller learns the permanent fact instead of an arithmetic
  symptom — and sound, because the witness pair needs no span.
- The spec amendment gives the refusal a defined format property to reference, instead of
  one inferred from omission. Other implementations get the obligation without inheriting
  our proof.
- Widening the condition later remains purely additive, so no future pass that recognises
  more interleavings is a breaking change.
- `strides_provably_distinct` becomes harder to misread and marginally harder to explain:
  a reader now has to hold three outcomes rather than two. The doc comment carries the
  witness constructions so the third is justified in place.

## Verification Obligations

These are boundary conditions on the predicate, not suspected defects. Each one **MUST**
be covered by a named test before the implementation pass is complete, and a reviewer
should read this list as the acceptance criteria for § 2.

One invariant stands above the list and over every case in it: **no input may produce a
false `Distinct`.** `Distinct` is the only verdict a caller acts on destructively, so an
arithmetic failure must surface as `AddressOverflow`, and an unrecognised interleaving as
`NotProven` — never as a proof the predicate does not hold.

1. **Extent-one axes do not affect the verdict.** An axis of extent 1 cannot move, so it
   can neither create nor prevent a collision. Two cases matter: a zero magnitude on an
   extent-1 axis is `Distinct`, not `Aliases`; and an extent-1 axis sharing a magnitude
   with a movable axis must **not** trigger the duplicate-magnitude proof. The predicate
   handles both by dropping those axes before either proof runs, which is exactly what the
   tests must pin.
2. **Empty tensors are `Distinct`, vacuously.** Any zero extent means no two valid indices
   exist, so there is no pair to collide. This answer must not be `NotProven`: a caller
   writing an empty tensor through any strides is sound, and the plan's own cover is empty.
3. **Negative strides, and the minimum signed value.** The sign reduction must be
   exercised on every exit, not only on the `Distinct` one. `i64::MIN` is the specific case
   where `stride.abs()` would panic: `unsigned_abs` is load-bearing, and a test must hold
   it there so a later refactor cannot quietly swap it back.
4. **Large shapes, and `reach` overflow.** A shape and stride pair whose span exceeds
   `uint64` must raise `AddressOverflow` on the paths that need the span, and must return
   `Aliases` on the two pre-pass paths that do not (§ 2). Both directions get a test, and
   both are instances of the invariant above: **overflow must never yield `Distinct`.** The
   `Aliases` direction is also the behaviour change recorded under *Consequences*. Large
   extents that do *not* overflow belong here too — a span near `uint64::MAX` must still
   answer `Distinct` where it is nested, so the checked arithmetic is not over-eager.
5. **Equal magnitudes on several movable axes.** `Aliases` at any extents, including the
   extents where `magnitude == reach` does not fire. `[1, 1]` over shape `[3, 3]` is the
   case the duplicate-magnitude proof exists to catch, and it belongs in the suite for that
   reason — not because the pre-existing `NotProven` was wrong, which it was not (§ 2).
6. **The no-op alias path stays outside the write-target restriction.** When
   `src_strides == dst_strides`, `plan()` returns `Ready` with an aliased buffer before
   `Remap::new` is reached, so an *aliasing source* asked to convert to its own layout must
   still plan successfully — nothing is written, and the descriptor is one the format
   permits (§ 1). A test must pin that, because the natural implementation mistake is to
   hoist the distinctness check to the top of `plan()`, which would break a legal case and
   contradict § 5's placement audit.

Beyond the boundaries, two sweeps carry the correctness of § 2 and both are required:
`claimed_distinctness_is_always_real` must keep exhausting small index boxes against a
brute-force reference, and it gains a mirror — **every `Aliases` verdict must be a real
collision** — because `Aliases` is now a claim and not a shrug. The conservatism test
should pin at least one case on each side of the `magnitude < reach` region (`[2, 3]` over
`[3, 2]` injective, `[2, 3]` over `[4, 3]` aliasing) so the unknown region is documented as
genuinely two-sided.

## Required Documentation Amendments

In the implementation pass, per `CLAUDE.md`:

- `docs/impl/transcode-matrix.md` — the `refused` and `deferred` stride rows name the two
  errors, and the `deferred` row's reason gains the "unreachable through negotiation"
  clause, so `deferred` does not read as a scheduled promise.
- `hurray-transcode/src/error.rs` — the family table in the enum doc comment lists both
  variants, and neither variant's message or doc comment says an aliasing destination
  "cannot hold the tensor's values". The claim § 3 licenses is the narrower one: the
  destination aliases distinct logical indices, cannot represent arbitrary values
  losslessly, and is rejected unconditionally by this transcoder.
- `hurray-transcode/src/plan.rs` — `Target::layout`'s doc states the accepted set
  **positively** (every layout whose stride magnitudes are strictly nested: row-major,
  column-major, any permutation of either, any padded or reversed variant), and notes why
  the check cannot move to construction: distinctness is a property of strides *and*
  shape, and a `Target` carries no shape.
- `hurray-transcode/src/parallel.rs` — step 2 of the disjointness argument requires
  `Distinct` specifically, not "not non-distinct".
- `docs/SUMMARY.md` — the ADR index entry.

## Findings Routed Elsewhere

**→ `format-spec-writer`, MEDIUM — `strided.md` § Stride Semantics defines aliasing only
for the zero stride.** A non-zero stride vector may also map two indices to one element
(`[1, 1]` over shape `[2, 2]` addresses offsets `0, 1, 1, 2`); the spec neither permits,
forbids, nor explains it. Amend per § 4 above, with its stated non-scope. Should land
before `hurray-transcode` publishes in 0.3.0, so the crate's refusal cites a defined
property.

**→ `format-spec-writer`, LOW — `strided.md` § Contiguity conflates dense with injective.**
Density is defined as the stride magnitudes being a permutation of the row-major strides,
which is a strict subset of injective: a padded row-major layout is not dense and is
perfectly safe to write through. A reader can reasonably infer "not dense ⇒ not writable".
One sentence separating the two properties closes it, and it is the vocabulary the
amendment above needs.

**→ `format-spec-writer`, LOW (re-flag) — `strided.md` § Example contradicts § Buffer
Size.** For shape `[3, 4]` with strides `[-4, 1]`, § Buffer Size's own formula gives
`max_offset = 3`, `min_offset = -8`, `range_elements = 12`; the Example says "the full
range from offset -8 to 0 (9 elements)". `hurray-core`'s `element_offset_span` doc-test
already asserts `(-8, 3)`. Surfaced during PR #279 and still present.

**→ [issue #281](https://github.com/pgillet/hurray/issues/281), MEDIUM — `hurray-python`
exports an aliasing tensor as freely writeable.**
`strides_for_layout` passes a `Strided` layout's strides through to DLPack verbatim,
including zero strides and aliasing ones, and the capsule is built with `flags: 0` under a
comment stating that v1.0 defines no flags beyond the version. That comment is wrong: v1.0
defines the read-only bit, and NumPy uses it — a `broadcast_to` array exported through
DLPack arrives with `writeable=False`, while an aliasing array exported by us arrives
writeable, and writing into it loses a value. The fix is NumPy's: set the read-only flag
when the layout is not provably distinct. Note what that implies for § *Open Questions
Deferred* — it would be the first caller of the predicate outside `hurray-transcode`, which
is the trigger named there. It does not contradict § 5, whose audit row concerns *import*,
where zero strides must keep working.

## Open Questions Deferred

- **`LayoutDescriptor::addresses_distinctly(&self, shape)`** as a convenience over the free
  function. Not now: the transcode call site needs the free function anyway, because it
  passes a *modified* dims vector — the cover, with broadcast axes clamped to one element —
  and a method would owe a per-family answer for tiled, Morton, Hilbert and the sparse
  layouts, all injective by their own specs but by different arguments. **Trigger:** the
  first caller outside `hurray-transcode`.
- **A distinctness answer for the non-affine layouts.** Same trigger.
- **C and Python exposure of the predicate.** Folded into ADR-044's open question on the
  transcode surface for those bindings.
- **Widening the sufficient condition** past the strictly-nested test. **Trigger:** a
  consumer that needs an interleaved target. None exists, and negotiation cannot express
  one.
