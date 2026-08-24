# ADR-037: Buffer metadata is a read-only value object, and alignment is measured

## Status

Accepted (2026-08-23), implemented 2026-08-24 (§ 1–9, then § 6a)

**Correction (2026-08-24, from implementation):** § 6 specified `hurray.BufferError` for
`copy=False` on an under-aligned source. The implementation raises
`hurray.CopyRequiredError` instead — the binding already reserves that class for
"`copy=False` requested but a copy is needed", and `__array__` raises it for the same
reason, so a caller catching one should catch both. Both subclass `ValueError`.

Extends **ADR-032** to the last descriptor section without a Python representation, and
applies its § 4 declaration-versus-evidence rule to a field where, unusually, the
evidence is authoritative.

## Context

### The gap

`docs/cookbook/layer-1-buffer-protocol.md` is fourteen Rust blocks with no Python
counterpart, because none of it is reachable. `BufferHandle` in `hurray-core` carries
five fields; Python reaches three, two of them indirectly:

| Field | Per-buffer? | Reachable from Python |
|---|---|---|
| `byte_size` | yes | indirectly — `t.buffer(i).shape[0]` |
| `alignment` | yes | **no** |
| `sync_mode` | yes | **no** |
| `device_tag` | no — descriptor-wide | `t.device.kind` |
| `memory_class` | no — descriptor-wide | `t.device.memory_class` |

`buffer-protocol.md` § Device Colocation requires `device_tag` and `memory_class` to be
identical across every buffer of one descriptor, so `Tensor.device` being a single
device is correct and stays. The genuinely per-buffer gap is **alignment** and
**sync mode**.

Issue #147 makes this a gap by policy. Two findings make it more than that.

### Finding 1: the producer was declaring an alignment it did not provide

`hurray.Tensor` hardcoded `MIN_BUFFER_ALIGNMENT` (64) for every non-empty buffer, over
`Box<[u8]>` allocations made at `align_of::<u8>() == 1`. Measured:

```
owned,  256 bytes  -> address % 64 = 32
owned, 1024 bytes  -> address % 64 = 16
```

The spec makes a 64-byte-aligned base address a MUST and says a reader MAY rely on the
declared value, so every descriptor this binding produced invited a consumer's aligned
SIMD load against an address that usually was not. Fixed for **owned** buffers in
PR #179, which over-aligns the allocation so the declaration becomes true.

**Borrowed** buffers are not fixed, and cannot be without this decision. NumPy's
alignment, measured over twenty samples per size:

```
     64 bytes: 10/20 were 64-byte aligned
   1024 bytes:  0/20
   16384 bytes: 0/20
  4194304 bytes: 0/20
```

Essentially never — and it is worse than chance for exactly the arrays where copying
costs most. Above glibc's `MMAP_THRESHOLD` (128 KiB by default) an allocation is served
by `mmap`, which returns a page-aligned block; glibc then places a 16-byte header before
the pointer it hands back. Measured, and identical for plain `malloc`, so it is the
allocator rather than NumPy:

```
np.zeros(n) address % 4096, ten samples each
   262144 bytes -> [16]
  1048576 bytes -> [16]
  4194304 bytes -> [16]
 16777216 bytes -> [16]
```

A large NumPy array served by a **fresh** `mmap` is therefore exactly 16 bytes past a page
boundary, and never 64-byte aligned. Small arrays land on 64 about a quarter of the time,
by luck.

> **Correction (2026-08-24, from CI).** An earlier draft of this section said *every* large
> array is 16 bytes past a page, deterministically. That overstates it, and a test written
> on the strength of it failed on CI. The mechanism is real — 0/40 allocations of 1 MiB and
> above landed on 64 when the arrays were **held**, so a fresh `mmap` genuinely never
> qualifies — but glibc also **recycles freed chunks**, and a large allocation that lands
> in a recycled chunk inherits whatever offset the heap's history gives it, including 64.
> The honest claim is therefore: NumPy's alignment is *unpredictable*, reliably wrong for
> a fresh mmap and a matter of heap history otherwise. Nothing may be written that assumes
> either outcome for a particular array — including a test.

NumPy documents no alignment guarantee of its own: `numpy.org/devdocs/dev/alignment`
defines only "true" and "uint" alignment for its internal copy code, both derived from
`dtype.alignment` — 8 bytes for `float64`, never 64.

`BufferHandle::with_memory_class` rejects any declared alignment below 64, so the binding
cannot state the truth either. `hurray-python` therefore cannot
currently ingest a NumPy array zero-copy *and* be conformant. That is not a bug in the
binding; it is a collision between the format's alignment floor and what the Python
ecosystem allocates.

### Finding 2: an accessor is only worth shipping if the value behind it is true

Exposing `alignment` on top of a fabricated constant would convert a latent producer bug
into a documented API that lies to its caller, and would give the fabricated value users.
So the accessor and the measurement rule are one decision, not two.

### What the prior art already settled

- **ADR-031** — a layout is a *property of* a tensor, not a kind of object; and
  `AttributeError` over `UnsupportedError`, so `hasattr` keeps discriminating.
- **ADR-036** — a composite *contains* tensors, so it is a different kind of thing.
- **ADR-032** — layouts get a class, but as immutable value objects with **no
  back-reference to their tensor**, because a metadata accessor that pins buffer lifetime
  is a defect in a zero-copy format. Its rejection text carries the sentence that decides
  the shape here: *"The buffer table is a sibling section of the layout section, not a
  child."*

Layout, quantization, statistics, and shard each have a Python class. The buffer table
is the only descriptor section that does not.

## Decision

### 1. `hurray.BufferHandle` — one frozen value object per buffer-table entry

```python
class BufferHandle:          # frozen; NOT constructible from Python
    byte_size: int           # the declared size in bytes
    alignment: int           # power of two; >= 64 when byte_size > 0
    sync_mode: str           # "producer_synced" | "event" | "consumer_stream"
    device: Device           # descriptor-wide; the same object t.device returns
    is_empty: bool           # byte_size == 0
```

with value equality, hashing consistent with it, and a `repr` of the form
`BufferHandle(byte_size=1024, alignment=64, sync_mode='producer_synced', device=cpu)`.

`sync_mode` is a lowercase string per ADR-032 § 5's rule for small closed enumerations,
matching `hurray_core::SyncMode`'s `Display` output — which `hurray-inspect` also
prints — and produced by a single internal helper so the three cannot drift.

**A `BufferHandle` holds no reference to its tensor and none to any buffer.** It is five
scalars copied out of the descriptor, so `[h for t in stream for h in t.buffer_handles]`
pins nothing. This is ADR-032's rejection of layout back-references applied verbatim.

### 2. `t.buffer_handles` is a tuple property; `t.buffer(i)` is unchanged

```python
class Tensor:
    buffer_handles: tuple[BufferHandle, ...]   # descriptor order
    def buffer(self, index: int) -> Tensor: ...   # unchanged: 1-D uint8 data view
```

Metadata and data get separate accessors deliberately: reading how large or how aligned
a buffer is MUST NOT require materializing anything that references its bytes. On a CUDA
tensor, asking "how is buffer 2 aligned?" must not construct a view over device memory
the caller cannot read.

A property rather than a method, because the asymmetry is informative — `buffer(i)` does
work and hands back something that pins memory; `buffer_handles` is five scalars per
buffer. `len(t.buffer_handles) == t.buffer_count` is then self-evident, and `IndexError`
comes from tuple indexing rather than a second hand-written error path.

`hurray.Composite` MUST NOT expose `buffer_handles`, for the reason ADR-036 § 4 excluded
`buffer` and `values`: a head owns zero buffers, and an empty tuple answers a question
that has no meaning.

### 3. `BufferHandle.device` is the tensor's `Device` object, identically

`t.buffer_handles[i].device is t.device` MUST hold for every `i`.

The wire row has five fields and the Python image should show five, so a reader diffing
against `hurray-inspect` does not have to ask where two went. But colocation means the
per-handle device is a redundant encoding of one fact, and five separately constructed
`Device` objects would invite callers to compare them and branch on a difference the
format forbids. Returning the same object gives wire fidelity without the hazard.

### 4. Everything is read-only, and the constructor is not touched

No `alignment=`, no `sync_mode=`, no `buffer_handles=`. `hurray.BufferHandle` is **not
constructible from Python**, like the `hurray.Layout` base class.

This follows ADR-032 § 4 rather than departing from it. ADR-032 made `layout=` a
constructor argument because a layout is a declaration whose truth the buffers *cannot*
supply — `nnz` and `strides` are the author's intent, and a buffer of the right size is
consistent with many of them. A buffer handle has no field of that kind:

| Field | Where its value comes from |
|---|---|
| `byte_size` | `len(buffer)` — evidence, exact |
| `alignment` | the base address — evidence, exact (§ 5) |
| `device_tag`, `memory_class` | already the `device=` argument |
| `sync_mode` | fixed by what Python can do (§ 6) |

A `buffer_handles=` parameter would be a parameter with no free variables: its only
possible use is to contradict the buffers, and every contradiction must be rejected. A
parameter whose only reachable effect is to raise is not an API.

### 5. Alignment is measured, never asserted

This does not carve an exception out of "never infer". ADR-032's rule governs
*structural* claims — statements about what bytes mean, which bytes cannot settle.
Alignment is a physical property of an address that the binding can observe. Measuring
is observation; asserting 64 without looking is the actual violation, and it is what the
code did.

- **Empty buffers** declare `alignment = 1`, matching `BufferHandle::empty`.
- **Owned buffers** are allocated over-aligned to at least 64 and declare it (PR #179).
- **Borrowed buffers** MUST have their base address measured, and MUST declare the
  largest power of two the address actually satisfies, capped at `PAGE_ALIGNMENT`. A
  stronger true declaration is legal and useful to IPC and RDMA consumers, and free.

### 6. An under-aligned borrowed source is copied, and the caller can say otherwise

Because the spec's floor is 64 and NumPy essentially never provides it, `from_numpy` and
its siblings gain an explicit argument, matching the convention `__array__` already
follows in this binding:

```python
def from_numpy(array, *, copy: bool | None = None) -> Tensor: ...
# copy=None   copy into a 64-byte-aligned allocation only if the source is under-aligned
# copy=False  raise hurray.CopyRequiredError naming the measured alignment; never copy
# copy=True   always copy
```

`copy=None` is the default because the alternatives are worse: refusing by default breaks
`from_numpy` for essentially every array, and copying unconditionally gives up zero-copy
even when the source would have qualified.

**This is a real cost and it must be stated plainly rather than buried.** Zero-copy
NumPy ingest, which the binding appeared to offer, was never conformant; the honest
version of it copies for most arrays. `copy=False` exists so that a caller who needs the
guarantee gets an error instead of a silent memcpy, and so the cost is measurable rather
than mysterious.

Note where the cost falls: a large array freshly served by `mmap` never qualifies, so the
copy is all but unavoidable exactly where it costs most. It is not *certain* — a large
allocation that lands in a chunk glibc recycled may happen to be aligned — but a producer
cannot arrange for that, and unpredictability is no better than a copy. This is the
strongest argument for the escape hatch below.

### 6a. The escape hatch: allocate through NumPy's pluggable allocator

> **Resolved (2026-08-24), implemented.** Shipped as `hurray.aligned_allocator()`, a
> **context manager only** — no module-level install. The policy is thread-local, so
> "install once at startup" would quietly do nothing for arrays allocated on worker
> threads; one shape avoids teaching that footgun. No `alignment=` parameter either: 64 is
> the floor the format requires and the only reason the feature exists, and page alignment
> serves a different question (IPC/RDMA) that can be answered separately if it earns it.
>
> A prototype settled four things that the write-up below had left to assumption:
>
> - **The API slots must be called with the GIL held.** They read and write a
>   `ContextVar`; calling `PyDataMem_GetHandler` without the GIL segfaults immediately.
>   Free under PyO3, but it means these calls must never sit inside a `py.detach()` block.
> - **`np.zeros` goes through `calloc`, not `malloc`.** An implementation covering only
>   `malloc` looks correct and silently allocates nothing.
> - **`realloc` is exercised** (`ndarray.resize`) **and is handed only the new size**, so
>   the allocator carries a 64-byte header recording each block's size. That also keeps
>   every `alloc`/`dealloc` pair inside Rust, which is what the NEP's implementation notes
>   warn to preserve.
> - **Slots 304/305 verified** against the installed NumPy 2.5.2 headers rather than
>   recalled.


NumPy ≥ 1.22 lets an extension install a data-memory handler (NEP 49;
`numpy._core.multiarray.get_handler_name()` reports `default_allocator` today).

**This is the use case NEP 49 was written for.** Its Motivation lists "ensuring data
alignment" first, citing a 2005 numpy-discussion thread on SIMD alignment and issue
#5312, *"Use an aligned allocator for NumPy?"*, where 64-byte alignment produced a 40×
improvement in one reported case. NumPy considered guaranteeing alignment itself,
declined, and shipped the hook instead — so "bring your own allocator" is not a
workaround here, it is the ecosystem's answer to exactly this question. That also
strengthens § Alternatives: NumPy's own maintainers did not treat a 64-byte requirement
as unreasonable, they treated satisfying it as the consumer's job.

Three properties of the mechanism make a scoped installer safe, each verified against
NumPy's own tests and headers rather than assumed:

- **The handler is stored per array.** *"each `ndarray` carries with it the functions
  used at the time of its instantiation, and these will be used to reallocate or free
  the data memory of the instance."* An array allocated inside the block is therefore
  freed by the matching `free` long after the block exits.
- **It is thread- and context-local.** `numpy/_core/tests/test_mem_policy.py` asserts
  both: `test_thread_locality` requires that *"the policy is not affected by changes in
  parallel threads"*, and `test_context_locality` covers `asyncio`. Installing a handler
  cannot leak into unrelated code.
- **`PyDataMem_SetHandler` returns the previous handler**, and `NULL` restores the
  default, so save-and-restore is the intended usage.

With one gotcha that MUST be documented: **child threads do not inherit the policy.**
Arrays allocated by a worker thread started inside the block get the default allocator,
and will be copied on ingest like any other.

`hurray-python` SHOULD offer a handler that allocates 64-byte-aligned blocks:

```python
with hurray.aligned_allocator():
    weights = np.zeros(shape, dtype=np.float32)   # 64-byte aligned
t = hurray.from_numpy(weights, copy=False)        # genuinely zero-copy
```

This turns "Hurray always copies NumPy arrays" into "arrays allocated for Hurray are not
copied", which is a materially different bargain for a producer that controls its own
allocations — the case that matters for an inference pipeline writing checkpoints.

Deferred rather than decided here only because it is additive and independent: the
`copy` argument is needed regardless, for arrays the caller did not allocate.

Two implementation notes for whoever takes it: setting a handler is **C-API only** —
NumPy exposes `get_handler_name` and `get_handler_version` to Python but no setter — and
the `numpy` Rust crate this binding already depends on declares `PyDataMem_SetHandler`
and `PyDataMem_GetHandler` at API slots 304/305 but leaves them **commented out**, so
the binding must reach them itself. NEP 49's implementation PR also warns that mixing
allocators risks mismatched alloc/free pairs, and recommends a `PyCapsule` base when
taking ownership of data.

### 7. `sync_mode` is read-only, and byte-yielding paths refuse anything else

Read-only for a different reason than alignment, and the difference matters. Alignment is
a fact the binding can observe. Sync mode is a **promise the binding cannot keep**:
`SYNC_EVENT` means the producer recorded a device event *and* the consumer can retrieve
it through the C ABI, and no Python API supplies an event handle. A `sync_mode=` keyword
would let a caller emit a descriptor whose contract is unsatisfiable by construction —
the consumer does the correct thing, waits for an event that does not exist, and gets a
hard failure or a race. That is the same failure class ADR-030 and ADR-032 § 4 exist to
prevent: a descriptor that encodes and decodes cleanly and is wrong.

Anything Python constructs is `producer_synced`, which is a consequence rather than a
default: the interpreter cannot enqueue device work through this API.

Reading it carries an obligation. `buffer-protocol.md` § Consumer Requirement places a
normative duty on the *consumer* — inspect the field, and for `SYNC_EVENT` wait on the
producer's event before touching a byte. So:

> A tensor holding any buffer whose `sync_mode` is not `producer_synced` MUST refuse the
> paths that hand out its bytes — `buffer(i)`, `__array__`, `__array_interface__`,
> `to_torch`, `__dlpack__` — with `hurray.UnsupportedError` naming the mode, until the
> binding can honour the wait.

This is not new policy. `docs/impl/python-bindings.md` § Stream parameter semantics
already requires `BufferError` from `__dlpack__` for `Event` and `ConsumerStream`. This
extends the same discipline to every byte-yielding path, because the reason is the
buffer's contract rather than the protocol's — and it makes the refusal *explicable*,
since the caller can now read `t.buffer_handles[0].sync_mode` and see why.

`__hurray__` and `StreamWriter.write` continue to relay such tensors unchanged. Relaying
a declaration is not reading a byte.

### 8. Module constants

`hurray.MIN_BUFFER_ALIGNMENT = 64` and `hurray.PAGE_ALIGNMENT = 4096`, mirroring
`hurray-core`. The cookbook page's alignment sections cannot be translated without them.

### 9. Alignment is exempt from the round-trip obligation

ADR-032 § 4 requires that rebuilding a tensor from another's `layout`, `quantization`,
`statistics`, `shard`, and buffers produce an equal descriptor. That obligation does
**not** extend to `alignment`, and must not be made to: alignment describes an address,
and a rebuild that copies bytes has a different address. A tensor that arrived declaring
4096 and is rebuilt through Python `bytes` will honestly declare 64. Adding a settable
`alignment=` to force equality would restore the exact fiction § 5 removes.

## Alternatives Considered

**Parallel tuples on `Tensor`** — `t.alignments`, `t.sync_modes`. Rejected: it
decomposes one 16-byte wire row into unrelated columns the caller must re-zip, and it is
the shape ADR-032 already rejected as *"keeping `layout` as a string and adding separate
properties for the parameters"*. It leaves nothing to compare, hash, or diff against
`hurray-inspect`, and would make the buffer table the only descriptor section without a
class.

**Put the metadata on the view `t.buffer(i)` returns.** Rejected on three independent
grounds, the first fatal. `Tensor::buffer` returns a `hurray.Tensor` built by
`new_borrowed_view`, so this is literally "add `.alignment` to `hurray.Tensor`" — and
that view's descriptor carries a *freshly fabricated* handle, not the parent's
buffer-table entry, so `t.buffer(0).alignment` would report the view's value rather than
the parent's. Plumbing the parent's handle through would leave one attribute with two
referents. Second, it puts a per-buffer field on every tensor, so `t.alignment` exists on
a CSF tensor with nine buffers and answers about one. Third, it makes reading a number
require materializing a view over bytes — ADR-032's lifetime objection exactly.

**A constructible or settable `BufferHandle`.** Recorded as **rejected, not deferred**,
following ADR-032's treatment of layout back-references, because the alignment asymmetry
in § 9 makes it certain to be re-proposed. Every field is either evidence or an
unkeepable promise.

**Relax the spec's 64-byte MUST for foreign memory.** The tempting escape from § 6, and
rejected — though it deserves the full argument, because it is the one option that would
restore zero-copy NumPy ingest.

The case for it: DLPack imposes no alignment requirement at all, and a format whose most
important on-ramp must copy has an adoption problem — sharpened by the measurement above,
since the copy is certain for large arrays rather than occasional. The case against is stronger on this
project's own terms. `docs/prior-art.md` § 699 lists *"alignment guarantees — 64-byte
minimum for SIMD; page-aligned for GPU/IPC — expressed in the spec, not left to
convention"* among the gaps Hurray deliberately fixes; relaxing it gives away a stated
differentiator. The same document, at § 335, records that Arrow Flight loses alignment
through gRPC and that **receivers must copy to aligned memory** — so copying at an
unaligned boundary is the established remedy in the closest prior art, not a novel
penalty. And a MUST that consumers can rely on is worth more than one they must
defensively check, precisely because the consumer is the party that cannot see how the
buffer was made.

Recorded here rather than settled silently: this is a **format** question, and a binding
ADR cannot decide it. If the copy cost proves unacceptable in practice, the escalation
path is a spec amendment through `format-spec-writer`, not a quiet relaxation in
`hurray-python`.

**Naming it `BufferInfo` or `BufferSpec`.** Rejected: the spec calls it a buffer handle.
Inventing a third vocabulary is what ADR-032 § Consequences warned about for layout
names. Confusion with `hurray-ffi`'s `HurrayBuffer` is already borne by `hurray-core`,
which has both, and is answered with a sentence of documentation.

## Consequences

**Positive**

- The last descriptor section without a Python representation gets one, closing the
  buffer-protocol half of #147.
- The binding stops declaring an alignment guarantee it does not provide, so a consumer's
  aligned SIMD load stops being a coin flip.
- The consumer obligation `sync_mode` encodes becomes visible instead of suppressed, and
  the paths that cannot honour it fail loudly with a message the caller can verify.
- The house pattern holds across all five sections: layout, quantization, statistics,
  shard, and buffer table are each a frozen value object with value equality, string
  enums, and a `repr` that agrees with `hurray-inspect`.

**Negative**

- **`from_numpy` now copies for most arrays.** The largest cost in this ADR, and the one
  most likely to be reported as a regression. The previous behaviour was zero-copy and
  non-conformant; `copy=False` makes the difference diagnosable.
- **A sixth class in the namespace** whose instances most callers never inspect.
- **`alignment` does not round-trip** through a Python rebuild, by design (§ 9).
- **`t.buffer_handles` is not identity-stable** — a fresh tuple per access, like
  `t.layout`. Value equality mitigates it.
- **Refusing byte access on non-`producer_synced` buffers** converts a silent race into a
  visible failure, and will look like a regression to whoever meets it first.

## Required Documentation Amendments

- `docs/impl/python-bindings.md` — a normative § Buffer Handles covering the class, the
  tuple property, the identical-`Device` rule, non-constructibility, the measured
  alignment rule with `copy`, the `sync_mode` string set and byte-path refusal, and the
  § 9 exemption from the round-trip obligation.
- `docs/cookbook/layer-1-buffer-protocol.md` — Python tabs, plus
  `hurray-python/examples/buffer_protocol.py`.
- `hurray-python/src/buffer.rs` — the D2 design note describes an `Owned` variant that
  PR #179 already replaced.

## Open Questions Deferred

- ~~Shipping the NEP 49 aligned allocator of § 6a~~ — resolved and implemented
  2026-08-24; see the note on § 6a.
- **Requesting a stronger alignment at construction** — `hurray.empty(..., alignment=4096)`
  that *allocates* to the request and declares what it allocated. Explicitly not a
  reopening of § 4: an allocation request is an instruction to the allocator, not a
  declaration about memory the caller already holds.
- **Authoring `event` and `consumer_stream`**, which needs an event-handle type in
  Python and a CUDA-capable path. ADR-030's deferred per-buffer `stream` question belongs
  with it.
- **Honouring** a non-`producer_synced` buffer rather than refusing it (§ 7).
- **Private device tags are lossy in Python** — `device.rs` collapses every tag in
  `0xF0`–`0xFE` to `"private"` and cannot author one. Same cookbook page, separate
  decision; raised by the architect during this review.
- **`validate_colocation` has no Python surface.** Probably correct to leave unexposed,
  but the cookbook's § Device Colocation has nothing to translate to, so it wants an
  explicit decision rather than an omission.
