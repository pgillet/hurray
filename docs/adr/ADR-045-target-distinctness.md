# ADR-045: Target distinctness — a proved trichotomy, and what the format says about aliasing strides

## Status

Proposed (2026-11-22)

Arises from the pass that landed `hurray-transcode`'s dense layout family
([#272](https://github.com/pgillet/hurray/issues/272), **ADR-044**) and the adversarial
review of it. Refines **ADR-044** § 13's error surface — one variant becomes two — and
pins the two matrix rows § 14 left to `docs/impl/transcode-matrix.md`. **ADR-044** § 6's
soundness argument stands unchanged: widening the set of targets proved distinct never
weakens it.

Requires an amendment to `docs/spec/layouts/strided.md` § Stride Semantics. Nothing here
is a wire-format or C ABI change.

## Context

A tensor cannot be *written* through a layout whose strides send two logical indices to
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

The argument that makes this right is not the sufficiency of the test but the **direction
of its error**. Widening the predicate later turns an error into a success, which is
additive and breaks no caller. Admitting first and tightening later is the breaking
direction — and in the interval it ships silent value loss, which is the one failure this
crate must not have. Given a test that will certainly be imperfect, the imperfection
belongs on the side that can only ever be relaxed.

Two facts make the cost of that choice close to zero.

**The refused set is unreachable through the format's own negotiation surface.**
`interchange.md` § Layout Entry Encoding makes a layout preference a single `layout_tag`
byte, plus opaque metadata for the private range. A peer cannot ask for `Strided([2, 3])`
— it cannot ask for a stride vector at all. An injective-but-unproven target is reachable
only from a Rust caller who hand-builds one, and no framework produces such strides:
NumPy, PyTorch and JAX all produce nested ones, and every padded, permuted and reversed
variant of those is already admitted.

**The alternatives are unsound or unbounded.** Accepting an unproven target and running
serially still loses a value when it genuinely aliases, and by construction the crate
cannot tell which case it holds — that is what the verdict says. Deciding injectivity
exactly is a bounded subset-sum problem. Building the offset set and checking it for
duplicates costs a second full pass and a bitset over the addressed extent, charged to
every conversion, to serve a target nobody has asked for.

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
with a witness constructible in O(1):

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

The sign reduction carries each witness back to the original strides: `i_k ↦ d_k − 1 − i_k`
is a bijection of the index box, and it shifts both members of the pair by the same
constant.

Only `magnitude < reach` is a genuine unknown, and it is **two-sided** — `[2, 3]` over
shape `[3, 2]` is injective, `[2, 3]` over shape `[4, 3]` aliases (offset 6 twice). That
region is where the subset-sum problem lives, and the test stops at its edge.

The equal-magnitude exit is a correction to the shape the review proposed, not an extra:
without it the canonical example answers differently at different extents — `[1, 1]` is
`Aliases` at shape `[2, 2]` (where `magnitude == reach`) but `NotProven` at shape `[3, 3]`,
though it obviously aliases at both. One stride family must get one answer; § 5 of
*Alternatives Considered* rejects the same inconsistency in another form.

The change is a handful of lines inside the existing loop and costs nothing at run time.

**`#[non_exhaustive]` comes off.** Proved-yes, proved-no, unknown is a complete trichotomy;
there is no fourth verdict to make room for. The workspace has the precedent — `Planned`
is deliberately exhaustive so a caller must handle every arm.

### 3. The error splits in two: a permanent refusal and a limitation

`Error::OverlappingTarget` is replaced by:

| Variant | Means | Family (ADR-044 § 13) |
|---|---|---|
| `AliasingTarget { strides, dims }` | The destination cannot hold the tensor's values. No version of this crate will accept it. | unsupported conversion |
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
consumer MUST treat aliased indices as carrying one value. A non-normative note observes
that an implementation materialising a tensor *into* a caller-chosen strided layout
therefore cannot honour an aliasing target in general, and SHOULD refuse it.

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

Every other candidate call site is correctly silent, and stays silent:

| Candidate | Calls it? |
|---|---|
| `hurray-core` descriptor validation | **No.** The format permits aliasing strides and mandates the zero-stride case. Rejecting at construction would make the reference implementation refuse conforming descriptors and break reading conforming files. |
| `hurray-io` writers | **No, structurally.** The crate contains no stride arithmetic at all; ADR-041's cores move whole buffers and never address an element. There is no write-through-a-layout path to guard. |
| `hurray-ffi` | **No caller today.** Exposing the predicate would be a new ABI function serving nobody; it folds into ADR-044's open question on the C/Python transcode surface. |
| `hurray-python` authoring and DLPack | **No.** Importing from `np.broadcast_to` legitimately yields zero strides, and on export DLPack hands out a pointer and strides under DLPack's own contract — NumPy's `as_strided` carries the identical hazard and documents it rather than preventing it. |

## Alternatives Considered

1. **Decide injectivity exactly.** Bounded subset-sum. Rejected: unbounded cost for a
   target with no consumer, in the hot path of every plan.

2. **Check empirically — build the offset set, look for a duplicate.** A second full pass
   plus a bitset over the addressed extent, which for a sparsely addressing layout is
   enormous. Rejected for cost, and it answers a question no caller is asking.

3. **Enumerate exactly when the index space is small.** Would make `[2, 3]` over `[3, 2]`
   work. Rejected twice over: it patches the specific case instead of the general one, and
   it makes one stride family answer differently at different extents — a less predictable
   contract than a uniform refusal.

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
- The spec amendment gives the refusal a defined format property to reference, instead of
  one inferred from omission. Other implementations get the obligation without inheriting
  our proof.
- Widening the condition later remains purely additive, so no future pass that recognises
  more interleavings is a breaking change.
- `strides_provably_distinct` becomes harder to misread and marginally harder to explain:
  a reader now has to hold three outcomes rather than two. The doc comment carries the
  witness constructions so the third is justified in place.

## Required Documentation Amendments

In the implementation pass, per `CLAUDE.md`:

- `docs/impl/transcode-matrix.md` — the `refused` and `deferred` stride rows name the two
  errors, and the `deferred` row's reason gains the "unreachable through negotiation"
  clause, so `deferred` does not read as a scheduled promise.
- `hurray-transcode/src/error.rs` — the family table in the enum doc comment lists both
  variants.
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
