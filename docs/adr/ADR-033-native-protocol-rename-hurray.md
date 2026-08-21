# ADR-033: The native protocol is named for the format — `__hurray__` / `from_hurray`

## Status

Proposed (2026-08-20)

Amends **ADR-023** § 1 (protocol name), § 7 (discovery), and § 8 (error semantics,
which names the capsule string), and **ADR-030** § 5 (one protocol, one probe). Every
other decision in both ADRs is unaffected.

## Context

The native interchange protocol is currently `hurray.Tensor.__hurray_buffer__()` and
`hurray.from_hurray_buffer()`. The name is wrong twice over.

**It is wrong about arity.** ADR-030 widened the capsule from a single
`HurrayBuffer` to a `HurrayBufferList`, because a tensor with per-channel scales,
sparse index arrays, or a page table has several buffers and every one of them has to
reach the consumer. The protocol has carried N buffers since then; the name still says
one.

**It is wrong about category, which matters more.** The capsule does not transport
buffers. It transports a whole tensor:

| Capsule part | Contents |
|---|---|
| pointer | `HurrayBufferList` — one `HurrayBuffer` per descriptor buffer index |
| context | the **encoded `TensorDescriptor`**, the ABI version, and a strong reference to the source |

The descriptor is what makes this protocol full-fidelity — element type, shape, layout,
quantization, statistics, shard — and it is precisely what DLPack cannot carry, which
is the protocol's reason to exist (ADR-023 § Context). Naming the thing after its
buffers hides the part that justifies it.

The clearest evidence is the signature: `from_hurray_buffer(obj) -> hurray.Tensor`. A
function named for buffers that returns a tensor is describing its payload wrongly.

**The original reasoning already pointed the other way.** ADR-023 § 1 justified the
name by analogy to `__dlpack__`, `__torch_function__`, and `__jax_array__` — and each
of those is named for a *format or project*, never for the bytes underneath.
`__dlpack__` does not describe what a DLPack capsule contains. The `_buffer` suffix was
drift from the precedent the decision itself cited.

## Decision

### 1. The protocol is named for the format

```python
hurray.Tensor.__hurray__(stream=None) -> PyCapsule
hurray.from_hurray(obj, /) -> hurray.Tensor
```

Naming the format rather than the payload is what keeps the protocol legible next to
the one it complements. The two are always read together, because choosing between
them is the decision a consumer actually makes:

```python
if hasattr(obj, "__hurray__"):        # full fidelity: every dtype, layout, device
    t = hurray.from_hurray(obj)
elif hasattr(obj, "__dlpack__"):      # universal, but lossy
    t = hurray.from_dlpack(obj)
```

It is also the only name that stays true as the protocol grows. A capsule that later
carries a composite group or a stream frame is still Hurray; it is no longer a tensor,
and would not be buffers either.

### 2. The capsule keeps a payload name

The PyCapsule is named `"hurray_tensor"`, and `"used_hurray_tensor"` once consumed.

This is deliberately *not* symmetric with the method name, and follows DLPack exactly:
`__dlpack__` returns a capsule named `"dltensor_versioned"`. The method name is
ergonomics — read by humans choosing a protocol. The capsule name is a wire contract —
read by `PyCapsule_IsValid` in a C consumer that never sees the Python method. Precision
belongs on the wire; brevity belongs in the call.

No version suffix is added, unlike DLPack's `_versioned`. DLPack needed one because
its v0.8 and v1.0 structs differ with no in-band version field. Hurray carries
`HURRAY_C_ABI_VERSION` in the capsule context, and `from_hurray` MUST verify it
(ADR-023 § 8), so the version travels in the payload where it can be checked and
reported rather than in a string that can only match or fail to match.

### 3. Discovery follows the method

`hasattr(obj, "__hurray__")` replaces `hasattr(obj, '__hurray_buffer__')` as the single
probe (ADR-023 § 7, ADR-030 § 5). One protocol, one probe, unchanged in substance.

### 4. The rename is clean — no aliases

The old names MUST be removed, not deprecated alongside the new ones. Keeping
`__hurray_buffer__` as an alias would mean two names for one protocol, two feature
probes a consumer must decide between, and a permanent second path through the
capsule code — the "flag to avoid making a decision" that `CLAUDE.md` § Guiding
Principles forbids.

Hurray is pre-`1.0`, where the versioning policy permits breaking changes precisely so
that mistakes like this one are fixed rather than carried. Renaming after `1.0` would
cost a deprecation cycle; renaming now costs a search and replace.

### 5. `hurray.from_hurray` stutters, and that is accepted

Inside the `hurray` module the name reads redundantly. That is the one genuine cost,
and it lands on the least important consumer: `hurray` → `hurray` is a round trip, not
the case the protocol was built for. Every other implementation reads correctly —
`torch.from_hurray(t)`, `mylib.from_hurray(t)` — exactly as `numpy.from_dlpack` does.
DLPack avoids the stutter only because its consumers live in other libraries; here one
of them happens to be us.

The status quo `hurray.from_hurray_buffer` stutters identically, so nothing is lost.

## Alternatives Considered

**`__hurray_tensor__` / `from_hurray_tensor`.** Accurate on both counts, symmetric with
`hurray.Tensor`, and matching Arrow's PyCapsule interface, which names the logical
object (`__arrow_c_array__`, `__arrow_c_stream__`) even though an Arrow array is itself
a schema plus several buffers. Rejected because the protocol's nearest neighbour is
DLPack, not Arrow: a consumer picks between `__hurray__` and `__dlpack__` in a single
`hasattr` chain, and the parallel is worth more there than descriptive precision. The
precision is not lost — it moves to the capsule name, per § 2.

**`__hurray_buffers__` / `from_hurray_buffers`.** The minimal edit: pluralise and stop.
Rejected because it corrects the arity error while preserving the category error, which
is the more misleading of the two. The capsule would still be named for the part that
does not distinguish it.

**Keep `__hurray_buffer__` and add `__hurray__` as an alias.** Rejected under § 4.

**Do nothing.** Rejected. The name is load-bearing documentation: it is the first thing
an implementer of a third-party binding reads, and it currently tells them the protocol
moves buffers when it moves tensors. That misdirection gets more expensive with every
binding written against it, and the cost of fixing it only rises after `1.0`.

## Consequences

**Positive**

- The name matches what the protocol transports, and stays true if the payload grows.
- `__hurray__` and `__dlpack__` read as the alternatives they are, in the one code
  shape — a `hasattr` chain — where a consumer chooses between them.
- The capsule string says `tensor` where a C consumer validates it, so the wire
  contract is more descriptive than before, not less.
- One protocol, one probe, one name: no alias, no second path, nothing to deprecate
  later.

**Negative**

- **The capsule name change is a protocol break.** A consumer keying on
  `"hurray_buffer"` via `PyCapsule_IsValid` stops recognising Hurray capsules. This is
  permitted pre-`1.0` and is the reason to do it now, but it is a real break and MUST
  be called out in the release notes rather than folded into a rename.
- **`hurray.from_hurray` stutters** (§ 5).
- **Roughly 150 references move**, across `hurray-python/src`, its tests and examples,
  the cookbook, the tutorials, `docs/impl/python-bindings.md`, and ADR-023/ADR-030.
  Most are prose; the mechanical surface is small.
- **Two ADRs gain amendment notes.** ADR-023 § 1, § 7, § 8 and ADR-030 § 5 keep their
  original text with a pointer here, matching how ADR-031 § 1 records its amendment by
  ADR-032.

## Required Documentation Amendments

- `docs/impl/python-bindings.md` — § Native Buffer Interchange Protocol: the method,
  the function, the capsule names, and the `hasattr` probe. The section title should
  lose "Buffer" too.
- `docs/adr/ADR-023-*.md` § 1, § 7, and § 8 — amendment note pointing here. § 8's
  rules are unchanged in substance; only the two names they quote move.
- `docs/adr/ADR-030-*.md` § 5 — amendment note; the "one protocol, one probe" rule is
  unchanged, only the spelling of the probe.
- `docs/cookbook/hurray-python-native-buffer.md` and
  `docs/tutorials/python-interop-paths.md` — worked examples.
- `hurray-python/examples/native_buffer.py` and `multi_buffer.py`.

No amendments under `docs/spec/` are required: the native protocol is
implementation-only (ADR-023 § 4), not part of the format specification.

## Open Questions Deferred

- ~~**Whether `hurray-ffi` should expose a matching C-level name.**~~ **Resolved by
  [ADR-034](ADR-034-c-readable-capsule-context.md) § 5: no rename.** Every C name was
  accurate for what it named — the C layer simply had no protocol type to misname.
  ADR-034 gives it one, `HurrayTensorContext`, named for what it carries.
- **Whether a future capsule carrying a composite group or a stream frame keeps the
  `"hurray_tensor"` capsule name** or introduces a sibling. § 1 makes the method name
  survive that change; the capsule name would not have to.
