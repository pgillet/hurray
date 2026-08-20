# ADR-023: hurray-python Native Buffer Interchange Protocol

## Status
Accepted

## Context

`hurray-python` interoperates with the broader Python ecosystem through three
buffer-sharing surfaces:

1. **DLPack (`__dlpack__` / `__dlpack_device__`)** — the cross-library protocol used
   by PyTorch, JAX, NumPy, CuPy, etc.
2. **NumPy (`__array__` / `from_numpy`)** — CPU-only, Tier 1 dtypes only.
3. **Python buffer protocol** — CPU-only, byte-level access.

DLPack is the right tool for cross-library zero-copy, but its device/memory model is
strictly less expressive than Hurray's. Per the device mapping table in
`docs/impl/python-bindings.md`, `hurray.UnsupportedError` is raised today in every
one of the following situations because DLPack v1.0 has no representation for them:

- **ROCm + `UNIFIED`** — `kDLROCMManaged` does not exist in the DLPack `DLDeviceType`
  enum.
- **`PEER` memory for any device** — DLPack has no flat enum value for peer-mapped
  memory.
- **Private device tags (`0xF0`–`0xFE`)** — by design, the format reserves these for
  implementation-private use; DLPack cannot represent them.

For hurray-to-hurray transfers — `hurray-python` ↔ `hurray-python`,
`hurray-python` ↔ `hurray-ffi` consumer, or `hurray-python` ↔ another binding built
on `hurray-ffi` — DLPack is not the correct protocol. The Hurray C ABI buffer handle
(`HurrayBuffer` in `hurray-ffi`) already carries the full descriptor: `device_tag`,
`memory_class`, `sync_mode`, `alignment`, `byte_size`, release callback, release
context. A native protocol can share all of that losslessly without flattening into
DLPack's `DLDeviceType` enum.

The proposal: an opt-in `__hurray_buffer__()` / `hurray.from_hurray_buffer()` PyCapsule
protocol. The capsule wraps a `HurrayBuffer` pointer from `hurray-ffi`, with the same
lifetime discipline as DLPack capsules (refcount on create, decrement on
consume/delete; capsule renamed after consumption to prevent double-free).

This ADR does **not** propose replacing DLPack. DLPack remains the only protocol
consumed by external libraries (PyTorch, JAX, NumPy). The native protocol exists
strictly to plug the holes the device mapping table calls out, and to give two
Hurray-aware peers a path that preserves the full descriptor.

### What was decided in OQ-A (ADR-022)

ADR-022 establishes that runtime compliance modes gate **exactly two** things: Tier 2 /
quantized dtype admission and `__array_namespace__` visibility on Tier 2 tensors. The
mode does **not** gate DLPack, `__array__`, error hierarchy, or buffer-protocol
semantics. That scope decision is load-bearing for this ADR: the native buffer protocol
is not an Array API construct and falls outside the modes' jurisdiction.

## Decision

### 1. Protocol name: `__hurray_buffer__` / `from_hurray_buffer`

> **Amended by [ADR-033](ADR-033-native-protocol-rename-hurray.md):** the protocol is
> named `__hurray__` / `from_hurray`, and the capsule is named `"hurray_tensor"` /
> `"used_hurray_tensor"`. The original text follows unchanged.

`hurray.Tensor` MUST expose a dunder method
`__hurray_buffer__(stream=None) -> PyCapsule`. The `hurray` module MUST expose
`hurray.from_hurray_buffer(obj, /) -> hurray.Tensor` that accepts any object whose
`__hurray_buffer__` returns a valid capsule.

The dunder name `__hurray_buffer__` mirrors the established pattern of `__dlpack__`,
`__array__`, and `__cuda_array_interface__`. The `hurray_` prefix is namespace-safe:
dunders prefixed with a project identifier are an accepted convention (PyTorch's
`__torch_function__`, JAX's `__jax_array__`).

### 2. Layer placement: deferred to Layer 8c (post-FFI Python exposure)

The protocol MUST NOT be implemented in Layer 8a. Layer 8a ships the core
`hurray.Tensor`, DLPack, NumPy interop, and strict mode only. The `hurray-ffi` C ABI
is not yet exposed to Python through any documented interface in Layer 8a.

The native buffer protocol logically belongs in a layer downstream of Layer 8b (file
I/O bridge) and downstream of the work that exposes `hurray-ffi` to Python. This ADR
designates that layer **Layer 8c — Native buffer protocol**. Layer 8c may run in
parallel with or after the relaxed-mode implementation; it has no dependency on relaxed
mode.

In Layer 8a and 8b:
- `__hurray_buffer__` MUST NOT be present on `hurray.Tensor`.
  `hasattr(t, '__hurray_buffer__')` MUST return `False`.
- `hurray.from_hurray_buffer` MUST NOT exist as a public name.

The names are **reserved** by this ADR to prevent third-party squatting.

### 3. Mode gating: neither strict nor relaxed; available unconditionally in Layer 8c

The native buffer protocol is **not** an Array API construct, and ADR-022's mode scope
explicitly excludes everything except `__array_namespace__` visibility and Tier 2 dtype
admission. Therefore:

- Once Layer 8c is shipped, `__hurray_buffer__` MUST be present on `hurray.Tensor`
  instances of **all** dtypes (Tier 1, Tier 2, quantized) in **both** strict and
  relaxed modes.
- The Array API conformance test suite (`array-api-tests`) does not probe
  `__hurray_buffer__` and is therefore unaffected.

A Hurray-aware consumer that needs a Tier 2 / quantized tensor's buffer in strict mode
still has a path: the native protocol. A non-Hurray-aware consumer that probes
`__array_namespace__` still sees `False` for Tier 2 in strict mode, as required by
ADR-022. The two surfaces are orthogonal.

### 4. Spec placement: implementation-only

The native buffer protocol MUST be documented **only** in
`docs/impl/python-bindings.md`. It MUST NOT appear in `docs/spec/buffer-protocol.md`
or any other file under `docs/spec/`.

Justification:
- The protocol is a Python-only transport — it does not affect the wire format, the
  binary descriptor, the C ABI, or any non-Python binding.
- The wire payload of the capsule is the `HurrayBuffer` pointer defined in
  `hurray-ffi`; the C FFI implementation guide is already authoritative for that handle.
- Putting the protocol in the format spec would create a normative obligation on all
  bindings (C, Java, JS, etc.) to expose a "native" protocol — out of scope for
  non-Python bindings that already pass `HurrayBuffer` directly through C.

### 5. Capsule lifetime: same discipline as DLPack, with Hurray-specific deleter

The PyCapsule lifetime rules MUST match DLPack semantics:

- **Capsule name on creation:** `"hurray_buffer"`.
- **Capsule name after consumption:** `"used_hurray_buffer"`. The consumer MUST rename
  the capsule before transferring ownership, exactly as DLPack consumers rename
  `"dltensor"` to `"used_dltensor"`.
- **Capsule destructor (producer side):** If the capsule is destroyed while its name is
  still `"hurray_buffer"` (the consumer did not take ownership), the destructor MUST
  call `hurray_buffer_destroy` on the wrapped `HurrayBuffer` pointer, which invokes the
  registered release callback exactly once.
- **Consumer-side responsibility:** Once the capsule has been renamed to
  `"used_hurray_buffer"`, the consuming `hurray.Tensor` owns the `HurrayBuffer` and
  MUST call `hurray_buffer_destroy` exactly once at its own finalisation.
- **Source `hurray.Tensor` reference counting:** When `__hurray_buffer__()` is called,
  the source `Tensor`'s Python refcount MUST be incremented and stored as the capsule
  context; the destructor MUST decrement it. This mirrors the rule required for
  `__dlpack__` in § Buffer Lifetime and Ownership.

> **Note (non-normative):** The Python refcount on the source `Tensor` and the
> `HurrayBuffer` internal release callback are independent. The Python refcount keeps
> the producer-side Python object alive while the capsule exists; the release callback
> governs the underlying buffer memory. `HurrayBuffer` is not internally refcounted at
> the C ABI; producers wanting multi-consumer fan-out MUST issue distinct
> `HurrayBuffer` handles per consumer.

### 6. `stream` parameter: same semantics as DLPack

`__hurray_buffer__(stream=None)` MUST accept an optional `stream` parameter with the
same semantics as `__dlpack__(stream=None)` in § Stream parameter semantics:

| `stream` | Requirement |
|---|---|
| `None` | The tensor MUST have `SyncMode::ProducerSynced`. |
| `-1` | The binding layer MUST perform a device-level synchronisation before returning the capsule. |
| Positive integer (stream handle) | If `ProducerSynced`, the stream is ignored. For `SyncMode::Event` or `SyncMode::ConsumerStream`, the binding MUST raise `BufferError`. |

### 7. Discovery: `hasattr(tensor, '__hurray_buffer__')`

> **Amended by [ADR-033](ADR-033-native-protocol-rename-hurray.md):** the protocol is
> named `__hurray__` / `from_hurray`, and the capsule is named `"hurray_tensor"` /
> `"used_hurray_tensor"`. The original text follows unchanged.

Consumers MUST discover support by probing `hasattr(obj, '__hurray_buffer__')`. There
MUST NOT be a capability flag on the `hurray` namespace. This matches the discovery
convention of `__dlpack__`, `__array__`, and `__cuda_array_interface__`. A consumer
that detects `__hurray_buffer__` MAY still fall back to `__dlpack__` if it does not
link `hurray-ffi`. The two probes are independent.

### 8. Error semantics

> **Amended by [ADR-033](ADR-033-native-protocol-rename-hurray.md):** the protocol is
> named `__hurray__` / `from_hurray`, and the capsule is named `"hurray_tensor"` /
> `"used_hurray_tensor"`. The original text follows unchanged.

- A consumer receiving an object lacking `__hurray_buffer__` MUST raise `TypeError`.
- If the wrapped `HurrayBuffer` pointer is null or the capsule name is not
  `"hurray_buffer"` (already consumed), `hurray.from_hurray_buffer` MUST raise
  `hurray.BufferError`.
- ABI version mismatch between producer and consumer MUST raise
  `hurray.UnsupportedError`. The capsule context MUST include `HURRAY_C_ABI_VERSION`
  from the producer; the consumer MUST verify it before dereferencing the handle.

## Alternatives Considered

**Option A: Extend DLPack upstream.** Lobby for `kDLROCMManaged`, peer-memory enum
values, and a private-tag escape hatch in DLPack itself. **Rejected:** upstream
evolution is slow; DLPack's structural flat-enum constraint cannot absorb Hurray's
full `(device_tag, memory_class, sync_mode)` space without losing the protocol's
cross-library simplicity.

**Option B: Reuse `__dlpack__` with a sentinel device-type integer.** Allocate a
private `DLDeviceType` value for "Hurray native" and embed the `HurrayBuffer` pointer
in `DLTensor.data`. **Rejected:** misuses a published cross-library protocol; silently
breaks DLPack consumers that do not recognise the sentinel value.

**Option C: Per-instance opt-in via a constructor flag
(`Tensor(..., expose_native=True)`).** Only tensors created with `expose_native=True`
carry `__hurray_buffer__`. **Rejected:** protocol availability belongs to the call
site, not the tensor instance — consistent with ADR-022's mode-scope decision. An
always-on dunder (once Layer 8c ships) is the uniform rule.

**Option D: Implement in Layer 8a.** **Rejected:** `hurray-ffi` is not yet reachable
from Python in Layer 8a. Implementing against a moving target adds risk with no
corresponding user value.

**Option E: Different capsule lifetime discipline.** For example refcount-only, no name
change. **Rejected:** the `"name" → "used_name"` rename is what makes DLPack's capsule
destructor safe under all consumption paths. Reinventing the discipline creates a
divergent mental model for binding authors.

**Option F: Capability flag on the `hurray` namespace** (e.g.,
`hurray.NATIVE_BUFFER_PROTOCOL_VERSION = 1`). **Rejected:** redundant atop the
`hasattr` probe; creates a synchronisation burden between the flag and the dunder.

## Consequences

- **Closes the DLPack representability gaps** for ROCm `UNIFIED`, all `PEER` memory,
  and private device tags `0xF0`–`0xFE`. Hurray-aware peers can transfer these buffers
  losslessly.
- **Preserves DLPack as the cross-library protocol.** Non-Hurray consumers (PyTorch,
  JAX, NumPy) continue to use `__dlpack__` unchanged.
- **Mode-orthogonal.** Tier 2 / quantized tensors gain a buffer-sharing path in strict
  mode without relaxing Array API conformance.
- **No spec churn.** `docs/spec/buffer-protocol.md` is untouched. Non-Python bindings
  are unaffected.
- **Layer 8c added to the roadmap.** Scope: one dunder, one constructor function, one
  capsule destructor, ABI version check, and associated tests. Small, contained, and
  user-approvable independently.
- **ABI version coupling.** Changes to `HurrayBuffer` in `hurray-ffi` affect the
  capsule payload. Mitigated by `HURRAY_C_ABI_VERSION` embedding and consumer-side
  verification.

## Required Spec Amendments

The following amendments to `docs/impl/python-bindings.md` are required as a follow-up
by `format-spec-writer`. Do NOT apply them here.

1. **New section `## Native Buffer Interchange Protocol`** (after
   `## DLPack Interoperability`, before `## Buffer Lifetime and Ownership`). Covers:
   dunder name, constructor, capsule names, lifetime rules, `stream` semantics,
   ABI versioning, mode independence, discovery, and reference to ADR-023.

2. **Amendment to `## DLPack Interoperability` § mapping table notes:** add a note
   pointing the `hurray.UnsupportedError` rows (ROCm `UNIFIED`, all `PEER`, private
   tags) to the native protocol as the recommended fallback for Hurray-aware consumers.

3. **Amendment to `### Layer 8a status`:** add bullets stating that
   `__hurray_buffer__` and `hurray.from_hurray_buffer` are reserved but not
   implemented in Layer 8a; `hasattr` returns `False` for the dunder.

4. **Amendment to `## Error Handling`:** add `hurray.BufferError` for null /
   already-consumed capsules passed to `from_hurray_buffer`; and `hurray.UnsupportedError`
   for ABI version mismatches.

5. **Amendment to `hurray-python/COMPAT-MATRIX.md`:** add a column recording, per
   release, whether the native buffer protocol is available and the minimum
   `HURRAY_C_ABI_VERSION` required.

## Open Questions Deferred

- **Multi-buffer tensors (`SparseTensor`).** Should `__hurray_buffer__` return a tuple
  of capsules or be restricted to dense tensors? **RESOLVED by ADR-030**: one capsule
  carries every buffer, wrapping a `HurrayBufferList`. The initial recommendation
  recorded here — restrict to dense and add `__hurray_sparse_buffer__` later — is
  superseded; it conflated multi-buffer with sparse, while dense per-channel-quantized,
  block-paged, and composite tensors are multi-buffer too.
- **Cross-process (IPC) variant.** A capsule only works in-process; cross-process
  Hurray-native exchange would need an OS-handle-based protocol. **Deferred** to a
  later layer.
- **Async counterpart.** Whether an `__ahurray_buffer__` async variant is needed.
  **Deferred.** Initial recommendation: no; `sync_mode` already handles the
  synchronisation contract per-buffer.
