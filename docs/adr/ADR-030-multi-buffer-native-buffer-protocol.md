# ADR-030: Multi-buffer native buffer protocol — one capsule carrying a `HurrayBufferList`

## Status

Proposed (2026-08-11)

Resolves the **Multi-buffer tensors** open question deferred in **ADR-023**
(hurray-python Native Buffer Interchange Protocol) and supersedes its initial
recommendation of a separate `__hurray_sparse_buffer__` protocol.

## Context

`hurray-python` can hold, transport, save, and load exactly **one** buffer per tensor,
while the format, `hurray-core`, and `hurray-io` are all multi-buffer. Every format
feature whose descriptor references a second buffer is therefore unreachable from
Python (issue #146):

| Path | Current limit |
|---|---|
| `__hurray_buffer__` capsule | one `data_ptr` + `byte_size` |
| `hurray.save()` | passes exactly one buffer to `FileWriter::write_tensor` |
| `hurray.load()` | raises `UnsupportedError` on any tensor with more than one buffer |
| `SparseTensor` | does not implement `__hurray_buffer__` at all |

`hurray_io::FileWriter::write_tensor` already accepts a slice of buffers; only the
Python binding narrows it.

Three observations drive this decision.

1. **The descriptor channel is already lossless.** The capsule context carries the
   *encoded* `TensorDescriptor`, and `from_hurray_buffer` reconstructs it with
   `TensorDescriptor::decode`. Quantization, statistics, shard, and extension type all
   survive the hop today. The gap is the buffer channel alone.

2. **A single-buffer transport turns a valid descriptor into silent corruption.** A
   `PerChannelAffine`, `Nf4`, or `Mxfp` descriptor references a `scale_buffer_index`.
   Such a descriptor encodes and decodes perfectly well, so a consumer would receive a
   descriptor pointing at a scale buffer that was never transported — a dangling buffer
   index rather than a clean error.

3. **Multi-buffer is not a synonym for sparse.** ADR-023's initial recommendation
   (restrict `__hurray_buffer__` to dense tensors; add `__hurray_sparse_buffer__`
   later) assumed the two coincide. They do not: a *dense* per-channel-quantized tensor
   is multi-buffer, as are block-paged and composite tensors. A sparse-specific
   protocol would solve the narrower half of the problem and leave quantization
   unreachable.

This also stands against the standing requirement (issue #147) that `hurray-python`
must fully expose what `hurray-core` and `hurray-io` can express.

## Decision

### 1. One capsule carries every buffer

`__hurray_buffer__()` MUST return a **single** PyCapsule carrying all of the tensor's
buffers. Capsule names are unchanged: `"hurray_buffer"` on creation and
`"used_hurray_buffer"` after consumption. There is one deleter and one lifetime,
exactly as in ADR-023 § 5.

A single-buffer tensor is the `N = 1` case of this protocol, not a separate path.

### 2. The capsule pointer is a `HurrayBufferList`

`hurray-ffi` MUST gain an opaque `HurrayBufferList` handle holding an ordered
collection of `HurrayBuffer` handles, with C accessors:

| Function | Purpose |
|---|---|
| `hurray_buffer_list_len` | number of buffers in the list |
| `hurray_buffer_list_get` | borrow the `HurrayBuffer` at index `i` |
| `hurray_buffer_list_destroy` | destroy the list and every handle it owns |

The capsule pointer MUST be `*mut HurrayBufferList`. This preserves ADR-023's decision
**D-NB1** — a C consumer can call `hurray-ffi` accessors on the capsule pointer without
linking PyO3 — while extending it from one buffer to N.

`hurray_buffer_list_get` MUST return a **borrowed** handle: ownership stays with the
list, and the consumer MUST NOT call `hurray_buffer_destroy` on it. Destroying the list
destroys every handle it owns, exactly once.

### 3. Buffer order is descriptor order

Element `i` of the list MUST be the buffer at index `i` of the descriptor's buffer
table. Every buffer index appearing in a quantization descriptor
(`scale_buffer_index`, `zero_point_buffer_index`), a layout descriptor, or a composite
member therefore indexes the list directly.

A consumer MUST reject a capsule whose list length does not match the descriptor's
buffer count, and MUST validate that every referenced buffer index resolves — the
check `hurray_core::validate_buffer_placement` performs.

### 4. C ABI version becomes 3

`HURRAY_C_ABI_VERSION` MUST be raised from `2` to `3`. The capsule context MUST carry
the producer's version, and the consumer MUST verify it before dereferencing the
pointer, as already required by ADR-023 § 8. A consumer built against version 2 that
receives a version 3 capsule raises `hurray.UnsupportedError` rather than
misinterpreting a `HurrayBufferList` as a `HurrayBuffer`.

Pre-1.0, no compatibility guarantee applies (see `docs/spec/versioning.md`); the
version check exists so the mismatch is diagnosed rather than dereferenced.

### 5. `__hurray_sparse_buffer__` is not introduced

The separate sparse protocol floated in ADR-023 is superseded. `SparseTensor` MUST
implement `__hurray_buffer__` using this protocol, with its values and index buffers in
descriptor order. Consumers discover support exactly as before, via
`hasattr(obj, '__hurray_buffer__')` (ADR-023 § 7) — one protocol, one probe.

### 6. File I/O accepts multi-buffer tensors

`hurray.save()` MUST pass every buffer of a tensor to `FileWriter::write_tensor`, and
`hurray.load()` MUST accept multi-buffer tensors — the current buffer-count rejection is
removed. `load()` MUST validate buffer placement before handing a tensor to the caller.

### 7. Spec placement is unchanged

ADR-023 § 4 stands: the native buffer protocol remains documented **only** in
`docs/impl/python-bindings.md` and MUST NOT appear under `docs/spec/`. This decision
changes a Python-side transport and adds a C ABI type; it does not touch the wire
format, the binary descriptor encoding, or any layout definition. The
`HurrayBufferList` type is documented in `docs/impl/c-ffi.md`, which is authoritative
for the C ABI.

## Alternatives Considered

**A tuple of capsules, one per buffer.** Rejected. It admits partially-consumed states
where some buffers have been taken and others not, multiplies the deleter logic by N,
and forces the encoded descriptor to ride in one arbitrarily-chosen capsule, making
that capsule privileged and the others meaningless on their own.

**Keep the capsule pointer as `*mut HurrayBuffer` (buffer 0) and hide buffers 1..N in
the capsule context.** Rejected. It is source-compatible for existing consumers, which
is precisely the danger: a C consumer that reads only the capsule pointer — the access
pattern D-NB1 exists to support — silently sees a one-buffer tensor and reads a
quantized tensor as if it were unquantized. A silent wrong answer is worse than an ABI
bump.

**`__hurray_sparse_buffer__` as ADR-023 initially recommended.** Rejected, per Context
§ 3: it conflates multi-buffer with sparse, leaves dense quantized tensors unreachable,
and forces consumers to probe for two protocols and reconcile their semantics.

**Widen `HurrayBuffer` itself to hold N allocations.** Rejected. `HurrayBuffer` maps to
one allocation with one release callback, one device tag, and one sync mode; buffers in
a multi-buffer tensor may legitimately differ in device and sync mode. Collapsing them
into one handle would lose that per-buffer metadata.

## Consequences

**Positive**

- Quantization beyond per-tensor affine becomes expressible from Python, unblocking the
  descriptor-authoring work in issue #146.
- Sparse file I/O stops being rejected on buffer count.
- Block-paged and composite tensors gain a transport path for the same reason.
- One protocol and one probe for every tensor kind, dense or sparse.

**Negative**

- An ABI bump: any existing C consumer of the capsule must be updated. Pre-1.0 this
  costs nothing externally, but it is a real change to a published handle shape.
- `HurrayBufferList` is a new owning type whose lifetime discipline must be exactly
  right — the borrowed-handle rule in § 2 is the part most likely to be misused.
- `hurray.Tensor` grows from one `BufferStore` to a collection, touching every
  construction path in the bindings.

## Required Documentation Amendments

- `docs/impl/python-bindings.md` — the native buffer protocol section: one capsule, N
  buffers, descriptor order, list-length validation, and `SparseTensor` support.
- `docs/impl/c-ffi.md` — `HurrayBufferList` and its three accessors; ABI version 3.
- `docs/adr/ADR-023-*.md` — mark the *Multi-buffer tensors* open question resolved by
  this ADR.

No amendments under `docs/spec/` are required.

## Open Questions Deferred

- **Cross-process (IPC) variant.** Unchanged from ADR-023: a capsule only works
  in-process. Deferred.
- **Async counterpart.** Unchanged from ADR-023: no `__ahurray_buffer__`; `sync_mode`
  already carries the per-buffer synchronisation contract. Deferred.
- **Per-buffer `stream` handling.** ADR-023 § 6 defines `stream` for a single buffer.
  Where the buffers of one tensor sit on different devices, whether `stream` applies
  per-buffer or per-tensor is left open until a concrete multi-device case appears; the
  present implementation targets buffers sharing one device.
