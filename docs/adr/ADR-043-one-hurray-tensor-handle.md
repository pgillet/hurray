# ADR-043: One `HurrayTensor` handle — a capsule carries a tensor, not a pair

## Status

Accepted (2026-11-06), implemented 2026-11-08

Closes [issue #174](https://github.com/pgillet/hurray/issues/174) and both of
**ADR-034**'s deferred open questions. Amends **ADR-030** § 2 (the capsule pointer) and
**ADR-033** § 2 (the capsule name). Supersedes **ADR-034** § 1, whose type is deleted;
its §§ 2–4 are carried forward onto `HurrayTensor` rather than withdrawn, and § 5
stands. C ABI version 9 → 10.

This is a **breaking change to the C ABI**, deliberate and scheduled: pre-1.0 permits
one on a minor bump, so the window runs up to 1.0 and closes there
([#225](https://github.com/pgillet/hurray/issues/225) sequences it **before** the spec
flip).

## Context

A native-protocol capsule carries a tensor in two halves:

| Capsule part | Handle | Holds |
|---|---|---|
| pointer | `HurrayBufferList` (ADR-030 § 2) | the tensor's buffers, in descriptor order |
| context | `HurrayTensorContext` (ADR-034 § 1) | the encoded descriptor, the ABI version, the owner reference |

Nothing enforces the pairing. `PyCapsule_GetPointer` and `PyCapsule_GetContext` are
separate calls returning unrelated handles, and a consumer that reads one without the
other gets either bytes it cannot interpret or a description of bytes it does not have.
Teardown is two calls in a required order — `hurray_tensor_context_destroy` first,
because releasing the context drops the producer's reference.

ADR-034 recorded the better shape — one handle owning both halves — and deferred it:
ADR-030 § 2 had fixed the capsule pointer, and the change bought elegance rather than
capability. It was picked up alongside ADR-039 and deliberately put down again on
2026-10-11, because that pass landed purely additive and nothing else forced a break.

**What changed is that the handle now exists.** ADR-041 gave `hurray-io` a C surface,
and a stream read produces exactly a descriptor paired with its buffers, so step 3
introduced `HurrayTensor`: one handle owning a `HurrayDescriptor` and a
`HurrayBufferList`, with borrowed accessors and a single `hurray_tensor_destroy`. Two
consequences follow.

1. **ADR-034's second deferred question is answered by what shipped.** It asked whether
   streaming frames would reuse the capsule's representation or grow a parallel one.
   Streaming grew the tensor-shaped handle — the right one — and the question is now
   only whether the capsule joins it.
2. **The duplication ADR-034 predicted is real.** The ABI carries two shapes for one
   concept: `HurrayTensor` on the read side, `HurrayBufferList` + `HurrayTensorContext`
   on the capsule side. `docs/impl/c-ffi.md`'s handle table lists both, one line apart,
   and a binding author has to learn which paths produce which.

So the remaining work is no longer spending a break on ergonomics. It is deleting a
duplicate, and the break it costs is the same one it has always cost.

## Decision

### 1. `HurrayTensor` becomes the whole tensor, and gains a public constructor

The handle ADR-041 introduced keeps its shape and gains the three things the capsule
role needs and a stream read does not: the producing build's ABI version, and an opaque
owner with its release callback.

```c
typedef struct HurrayTensor HurrayTensor;

HurrayStatus hurray_tensor_new(uint32_t abi_version,
                               const uint8_t *descriptor_bytes,
                               uint64_t descriptor_len,
                               struct HurrayBufferList *buffers,   /* ownership transferred */
                               void *owner,
                               HurrayOwnerReleaseFn owner_release,
                               struct HurrayTensor **out_tensor);

HurrayStatus hurray_tensor_abi_version(const struct HurrayTensor *tensor, uint32_t *out);
HurrayStatus hurray_tensor_descriptor(const struct HurrayTensor *tensor,
                                      struct HurrayDescriptor **out_descriptor);
HurrayStatus hurray_tensor_buffers(const struct HurrayTensor *tensor,
                                   struct HurrayBufferList **out_buffers);
HurrayStatus hurray_tensor_destroy(struct HurrayTensor **tensor);
```

`hurray_tensor_descriptor` and `hurray_tensor_buffers` hand out **borrowed** pointers,
valid until the destroy and never to be destroyed individually — the discipline
`hurray_buffer_list_get` already follows. A tensor produced by a stream or file read
reports `HURRAY_C_ABI_VERSION` and carries no owner; one produced by
`hurray_tensor_new` reports what its producer passed.

ADR-034's §§ 2–4 move here unchanged in substance: the handle is opaque with accessors
rather than a `repr(C)` struct, the owner travels behind `void *` so no Python type
reaches the ABI, and the version is read before anything else is trusted (§ 5 below).

### 2. `HurrayTensorContext` is deleted

It exists only to be a capsule's context. Once `HurrayTensor` carries what it carried,
keeping it is surface with no capability behind it — and worse, a second way to say the
same thing. The deletion is what makes this an ABI break rather than an addition, and
it is the point of the exercise.

### 3. The capsule pointer is a `HurrayTensor *`

A capsule carries **one** pointer and **no context**. The pairing becomes structural: a
consumer that has the pointer has the descriptor and the buffers, because they are the
same object. ADR-030 § 2's buffer-list requirement is narrowed to what it was actually
protecting — a C consumer reaching the buffers through `hurray-ffi` accessors without
linking PyO3, which `hurray_tensor_buffers` preserves — and §§ 1, 3 and the rest of
ADR-030 stand untouched.

### 4. `HurrayBufferList` survives as the builder

A producer still accumulates buffers with `hurray_buffer_list_new` and
`hurray_buffer_list_push`; `hurray_tensor_new` consumes the finished list and takes
ownership of it. The list builds, the tensor is the finished product — which is already
how the reader path works internally.

### 5. The capsule is renamed once, to `hurray_tensor_handle`

| | Name |
|---|---|
| on creation | `"hurray_tensor_handle"` |
| after consumption | `"used_hurray_tensor_handle"` |

Without a rename, a pre-break and a post-break capsule share the name `"hurray_tensor"`
while pointing at different types. A stale consumer would `PyCapsule_GetPointer` a
`HurrayTensor *` and treat it as a `HurrayBufferList *`. The in-tree consumer happens to
survive — it null-checks the now-absent context and raises `BufferError` — but that is
luck, not a contract; a binding that reads the pointer before the context gets undefined
behaviour where it should get an error. A capsule name is a wire contract precisely so
that this is a hard `TypeError` in every consumer, in every language, with no
cooperation required.

This is **one** rename, not a counter that advances with the ABI. Two things make that
true, and both are normative:

- **`hurray_tensor_abi_version` is the stable first accessor.** It MUST be callable on
  any `HurrayTensor *` produced by any version of this ABI, and a consumer MUST call it
  before any other accessor and before trusting anything else about the handle. Its
  signature MUST NOT change. This is ADR-034 § 4's rule, re-targeted: it is what lets
  every future ABI version be dereferenced safely by any build, so additive growth never
  needs a new capsule name.
- **The break window closes at 1.0.** After the compatibility contract opens, a change
  that invalidated the promise above would be a major-version matter.

The suffix is `_handle` rather than `_v2` because a number would track nothing: the
capsule would read `v2` while the ABI reads `10`, two counters that never correspond
again. `_handle` names what the pointer *is*, which is the thing that changed, and
ADR-033 § 2's rule that no *version* suffix is added therefore still holds — the version
still travels in the payload where it can be checked and reported, now one dereference
deeper.

### 6. The descriptor is decoded at construction, and only the decoded form is kept

`hurray_tensor_new` decodes `descriptor_bytes` into the `HurrayDescriptor` the handle
owns, and does not retain the bytes. Three things follow.

- **Construction validates.** A producer can no longer emit a tensor whose descriptor
  does not parse; `hurray_tensor_new` returns the decode error instead. Today that
  failure is discovered by the consumer, one process or one machine later.
- **A null or zero-length descriptor is refused** with `HURRAY_ERR_NULL_POINTER`.
  `hurray_tensor_context_new` permits both today, and a doc example relies on it. A
  capsule with no descriptor is useless, and eager decoding rules it out anyway.
- **The encoded bytes are no longer reachable from C.** This honours ADR-039 § 8's
  promise to expose a borrowed `const HurrayDescriptor *` rather than a second accessor
  family, and it keeps one shape for both paths: a tensor from a stream read has no
  descriptor bytes to offer, because the sans-io cores decode out of their staging
  buffer and drop it. Every descriptor *field* stays readable through ADR-039 and
  ADR-040. What goes is verbatim byte forwarding — see Consequences.

### 7. One destroy

`hurray_tensor_destroy` releases the descriptor, every buffer the list owns, and the
owner reference, then nulls the caller's pointer. The two-call teardown and its required
order disappear: a consumer holds one object and ends it once.

A capsule consumer therefore keeps the handle for as long as it reads the buffers,
rather than taking the halves out and dropping the carrier.

### 8. C ABI version 9 → 10

`HURRAY_C_ABI_VERSION` becomes `10`. The native protocol's version check is exact
equality, so every capsule from a version-9 build is refused by a version-10 consumer
and vice versa — the rename in § 5 is for consumers that look at the pointer before the
version, not a substitute for the check.

## Alternatives Considered

**Keep the pair; document the pairing more loudly.** The status quo, and defensible
until ADR-041 shipped a second spelling of the same idea. Rejected now because the cost
is no longer a missing convenience but two types for one concept, each with its own
teardown rule, in an ABI whose compatibility contract is about to freeze.

**Keep the capsule name `"hurray_tensor"` and rely on the version check.** Tempting:
ADR-030 changed the capsule pointer's type at ABI 3 without renaming, and nothing broke.
That worked because the version then lived in the *context*, separate from the pointer,
so a consumer could check it before dereferencing. This change removes the context, so
the version now lives inside the thing whose type changed and a consumer must
dereference to read it. Rejected: the one protection ADR-030 had is exactly what this
change takes away.

**Rename to `"hurray_tensor_v2"`.** Rejected under § 5: a counter that corresponds to no
other version in the project, and an implied sequence that § 5's stability promise is
designed to prevent.

**Keep the encoded descriptor bytes in the handle, beside the decoded form.** Preserves
verbatim forwarding, at the cost of two sources of truth in one handle — and of plumbing
the bytes out of `hurray-io`'s sans-io cores, which decode from a staging buffer they
own, so that a read-produced tensor could offer them too. Rejected: the capability lost
is narrow, and the honest fix for it is an encode entry point on the descriptor handle,
which is additive and can land whenever it is actually wanted (Open Questions).

**Give streaming its own context-shaped type instead.** The mirror image of this ADR,
and the thing ADR-034's second question warned about. Rejected on sight: it doubles the
duplication rather than removing it.

## Consequences

**Positive**

- One type for one concept. A binding author learns `HurrayTensor` once, and every path
  that produces a tensor — capsule, stream read, file read — produces it.
- The pairing is structural. A capsule cannot carry buffers without their description,
  because there is no second handle to forget.
- Teardown has no order to get wrong, and no second call to omit on an error path.
- Producers fail at construction rather than exporting a descriptor that does not parse.
- `hurray-python`'s producer path stops round-tripping through encoded bytes: a Rust
  constructor takes the `TensorDescriptor` it already holds.

**Negative**

- **An ABI break**, with a rebuild required of every C consumer and a compatibility
  matrix entry. This is the known cost, scheduled rather than discovered.
- **Verbatim descriptor-byte forwarding is gone.** A C consumer that today copies a
  capsule's descriptor bytes into a container of its own must read fields instead.
  Nothing in tree does this, and no consumer outside the tree is known; the Open
  Question below is the route back if one appears.
- **Zero-length descriptors stop working**, breaking a call pattern that the ADR-034
  doc example itself uses.
- **A consumer must keep the handle alive** while it reads buffers, rather than taking
  ownership of a buffer list. For `hurray-python` this is no change in practice — the
  consumer already takes its own Python reference to the source object rather than
  relying on `owner_release` for liveness.

## Required Documentation Amendments

- `docs/impl/c-ffi.md` — § Tensor Context becomes § Tensor Handle; the opaque-handle
  table drops `HurrayTensorContext*`; the ABI version table gains rows `9` (the
  composite surface, missing today) and `10`.
- `docs/impl/python-bindings.md` — § Native Interchange Protocol (the capsule-pointer
  MUST), § Multi-buffer tensors, § Capsule lifetime (both names), § The capsule context
  (replaced), § ABI versioning (the accessor named in the ordering rule).
- `docs/adr/ADR-030-*.md` § 2 — amendment note: the capsule pointer is a
  `HurrayTensor *`; the accessor guarantee it protected survives.
- `docs/adr/ADR-033-*.md` § 2 — amendment note: the capsule is
  `"hurray_tensor_handle"`; the no-version-suffix rule stands.
- `docs/adr/ADR-034-*.md` — Status gains the supersession of § 1 and the note that
  §§ 2–4 move onto `HurrayTensor`; the first deferred question is marked resolved here,
  the second resolved by ADR-041 step 3 plus § 3 above.
- `docs/cookbook/layer-7-c-ffi.md` — the capsule recipe and the read-the-version-first
  note, in both language tabs.
- `hurray-python/COMPAT-MATRIX.md` — minimum `HURRAY_C_ABI_VERSION` 10, and the native
  protocol note (which today says `4` in its table, `5` in its prose, against a constant
  of `9`).
- `docs/SUMMARY.md`, `website/coverage-matrix.toml`, `CHANGELOG.md` (breaking).

## Open Questions Deferred

- **An encode entry point on the descriptor handle** —
  `hurray_descriptor_encode(const HurrayDescriptor *, …)` — which would restore verbatim
  forwarding uniformly for every tensor, whatever produced it, and would be the first
  piece of a C writer surface. Additive, so it needs no break and no decision now:
  deferred until a consumer wants it. Decoding is strict enough (length mismatch and
  reserved flag bits are rejected) that a decoded descriptor re-encodes faithfully.
- **Whether a capsule carrying a composite group keeps this name** or introduces a
  sibling — ADR-033's second deferred question, still open. `HurrayItem` is how a
  composite crosses the ABI today, and no capsule carries one.
- **The C writer surface (Compliance Level 2)** is unaffected by this ADR and remains
  deferred pending the async bridge design.
