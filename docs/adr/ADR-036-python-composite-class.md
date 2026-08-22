# ADR-036: A composite is a container of tensors, not a tensor

## Status

Proposed (2026-08-22)

Answers the question **ADR-031 § 5** deferred — whether composites belong on the
unified `hurray.Tensor` class. Amends **ADR-032 § 6** (the error a composite layout
raises) and **ADR-035 § 4** (the stream reader stops refusing composites).

## Context

Composites are the last capability `hurray-core` and `hurray-io` can express that
`hurray-python` cannot, and they are blocked in three places at once:

| Where | Today |
|---|---|
| authoring | `hurray.Tensor(..., layout=CompositeLayout(...))` raises `UnsupportedError` (ADR-032 § 6) |
| streaming | `StreamReader` raises `UnsupportedError` on a composite, by name (ADR-035 § 4) |
| files | `save` / `load` have no composite path at all |

All three trace to one root: **`hurray.Tensor` cannot represent a head that owns no
buffers.** One decision unblocks all of them, which is why this is a single ADR rather
than three.

The question ADR-031 left open is the one that has to be answered first. ADR-031 removed
`hurray.SparseTensor` on the grounds that sparse is a *layout*, not a *kind of object* —
a sparse tensor still has data, a shape, a dtype, and buffers, merely arranged
differently. Does that argument reach composites?

## Decision

### 1. `hurray.Composite` is its own class

It does not, and here is the difference. A sparse tensor **has** data. A composite
**contains** tensors:

- its head owns zero buffers — the format's own model calls it *virtual*, a fourth
  addressing category beside dense, sparse, and indirect (ADR-027)
- `len(composite.members)` has no meaning on a tensor
- there is no `composite.values`, no `__dlpack__`, no bytes to hand anyone — the data
  belongs to the members, each of which is an ordinary `hurray.Tensor`

ADR-031's rule was that a *layout* must not become a class. A composite is not a layout
applied to data; it is a grouping of tensors that happens to be introduced by a
descriptor. Giving it a class does not reopen `SparseTensor`, because nothing about a
composite is expressible as "a tensor whose bytes are arranged differently".

```python
composite = hurray.Composite(
    "partition",
    shape=[8, 8],
    dtype=hurray.float32,
    members=[tile0, tile1],
)

composite.members          # (Tensor, Tensor)
composite.layout           # CompositeLayout(composition_rule='partition', member_count=2)
composite.shape            # (8, 8)
```

### 2. The head is stated, never derived

`shape`, `dtype`, the composition rule, and the combine operation are **required** —
`member_count` is the one field taken from the members, because it is a count of what
was passed rather than a claim about it.

A partition's head shape could in principle be derived from its members' shards, and it
MUST NOT be. The same rule governs `layout=` on `hurray.Tensor` (ADR-032 § 4): the
descriptor is a declaration, the members are evidence, and they must agree. Deriving
would mean a caller who miscomputed a shard offset gets a head quietly reshaped to match
their mistake, and a consumer downstream reading a composite that is self-consistent and
wrong.

Validation is delegated entirely to `hurray-core`'s `CompositeValidator`, which already
enforces per-member checks, partition coverage, overlay ordering, and member count. The
binding MUST NOT grow a second copy of those rules.

### 3. Members may be composites

`members` accepts `hurray.Tensor` **or** `hurray.Composite`, because the format nests
(ADR-027 § Binding) and `hurray-io` already represents members as a tree. A nested
member MUST be validated by the same path as a top-level one, and the depth limit is
core's.

### 4. `hurray.Tensor` does not change

Constructing a tensor with a composite layout still raises `hurray.UnsupportedError`
(ADR-032 § 6). The rule is unchanged; only the message changes, to name
`hurray.Composite` instead of describing a gap.

This matters more than it looks. The alternative — letting `Tensor` hold zero buffers —
would put a second family of inapplicable accessors on the class ADR-031 had to justify
carefully: `values`, `buffer`, `buffer_count`, `__dlpack__`, `__hurray__`, `__array__`
would each need a composite branch, and `hasattr` would stop discriminating in the way
ADR-031 § 2 relies on.

### 5. Composites travel over both I/O paths

- `StreamWriter.write` MUST accept a `Composite` and emit it as head-then-members
  (`write_composite`).
- `StreamReader` MUST yield a `Composite` where it previously raised, amending
  ADR-035 § 4. Its iteration remains one item per `next()`, a composite counting as one.
- `save` MUST accept a `Composite` as a named entry, and `load` MUST return one.

A composite read back MUST equal what was written, member for member, so the round-trip
obligation ADR-032 § 4 states for descriptors extends to composite trees.

### 6. Composites stay out of the native protocol

`Composite` MUST NOT implement `__hurray__` or `__dlpack__`. The native protocol capsule
carries a buffer list and one descriptor (ADR-030 § 2, ADR-034); a composite is a tree,
and there is no honest way to flatten one into that shape without inventing wire
structure this ADR has no business inventing.

A caller who wants a composite across a process boundary uses the streaming or file
path, which the format already defines for exactly this. `hasattr(obj, "__hurray__")`
therefore keeps meaning what it means, rather than becoming true for an object the
protocol cannot actually carry.

## Alternatives Considered

**Let `hurray.Tensor` hold a buffer-less head.** The instinct ADR-031 established, and
the reason this question was deferred rather than answered there. Rejected under § 4: it
buys uniformity in the type and pays for it in every accessor, and it would make
`hasattr` — which ADR-031 § 2 chose deliberately over `UnsupportedError` — stop telling
the truth. It also asserts something the format does not: that a head is a thing you can
hold and use, when on the wire a head never appears without its members.

**Represent a composite as a tuple or a dict** — `(head, [members])`. Rejected for the
reason ADR-035 already rejected it for streaming: it invents a second, ad-hoc
representation that the next pass has to keep or break, and it gives the caller nothing
to validate against.

**Derive the head's shape and dtype from the members.** Rejected under § 2. Convenient
for the common partition case, wrong in exactly the case that matters.

**Expose composites read-only first, defer authoring.** Rejected: reading is the half
that is nearly free (the reader already decodes the tree and throws it away), and
authoring is the half that unblocks a Python producer. Shipping the easy half would
close none of the three gaps in the table above.

## Consequences

**Positive**

- The last capability gap between `hurray-python` and the Rust layers closes; all three
  blocked paths open on one decision.
- `hurray.Tensor` keeps its accessor discipline intact, and `hasattr` keeps
  discriminating.
- The distinction the class draws — container versus tensor — is the one the format
  draws, so `isinstance` teaches the reader something true about the wire.

**Negative**

- **A second top-level object.** Callers must now handle two kinds of thing coming out
  of a stream or a file, where before there was one. That is the format's shape, but it
  is still a branch every consumer has to write.
- **`Composite` cannot cross the native protocol**, so the fastest in-process path does
  not carry them. Deliberate (§ 6), and a real limitation.
- **The head's shape and dtype are the caller's to get right.** Core will reject a
  mismatch, but the caller must state something to be rejected — which is more work than
  deriving, and is the point.
- **Nesting makes `repr` a tree.** It should be depth-aware, as `TiledLayout`'s already
  is (ADR-032 § Consequences).

## Required Documentation Amendments

- `docs/impl/python-bindings.md` — a normative § Composites: the class, the required
  head parameters, nesting, the I/O paths, and the native-protocol exclusion.
- `docs/adr/ADR-031-*.md` § 5 — the deferral is answered; note pointing here.
- `docs/adr/ADR-032-*.md` § 6 — the `UnsupportedError` message now names `Composite`.
- `docs/adr/ADR-035-*.md` § 4 — the reader yields composites; note pointing here.
- `docs/cookbook/composite-tensors.md`, `composite-streaming.md`, `composite-file.md` —
  Python tabs beside the Rust recipes (#147).
- `hurray-python/examples/composites.py` — runnable.

## Open Questions Deferred

- **Whether a composite should be indexable** — `composite[0]` as sugar for
  `composite.members[0]`. Ergonomics; decide once there are users.
- **Whether `load` should return composites lazily**, reading members on access rather
  than eagerly with the head. Matters only for large trees, and the file reader would
  need to keep its handle open.
- **A native-protocol representation for trees** (§ 6), which would need wire structure
  the format does not currently define.
