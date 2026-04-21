# ADR-009: Release Callback is Normative; Reference Counting is an Implementation Detail

## Status

Accepted

## Context

OQ-1 in `docs/spec/buffer-protocol.md` asked whether the C ABI should expose a
normative `retain`/`release` pair for reference-counted buffer sharing, or
whether reference counting should remain an implementation detail with only the
single-consumer release callback being normative.

The core use case for normative reference counting would be: two independent
Hurray implementations simultaneously holding a reference to the same buffer,
each needing to signal its release so the memory is freed only when both are
done. This requires a shared, interoperable retain/release protocol at the C ABI
boundary.

DLPack, the closest existing tensor ABI, takes the simpler path: a single
`deleter` function on `DLManagedTensor`. The producer wraps its actual
deallocation inside the deleter; if the producer wants to share the buffer
across multiple consumers, it implements reference counting internally and
provides each consumer with a separate `DLManagedTensor` pointing to the same
data, each with a release-aware deleter. Consumers are not aware of the internal
reference count.

## Decision

**Reference counting is an implementation detail.** The only normative contract
at the C ABI level is:

- A buffer handle carries a release callback.
- The consumer MUST call the release callback exactly once.
- The release callback MUST be safe to call from any thread.

A producer that wishes to support multi-consumer sharing MUST implement
reference counting internally and hand each consumer a separate buffer handle
whose release callback decrements the internal count, invoking the actual
deallocation only when the count reaches zero. Consumers are not aware of this;
they call their release callback exactly once, as the normative contract requires.

No normative `retain` function is added to the C ABI.

## Alternatives Considered

**Normative `retain`/`release` pair at the C ABI level.**
Pros: makes cross-implementation multi-consumer sharing interoperable without
coordination — any consumer can extend a buffer's lifetime by retaining it.
Cons: forces every language binding (Python/GC, Rust/`Arc`, Go, Java) to bridge
between its own memory management idiom and the C ref-count, adding complexity.
The scenario requiring this — two *independent* implementations sharing a single
buffer region with no producer coordination — is effectively nonexistent in
inference-serving pipelines, which are sequential stage-to-stage rather than
parallel fan-out across runtimes. Rejected as over-engineering for the target
use case.

**No release callback; ownership is always transferred.**
Pros: simplest possible ABI — the consumer owns the memory and is responsible
for freeing it.
Cons: forces a specific allocator on the producer (the consumer must know how to
free memory allocated by the producer). Incompatible with GPU device memory,
shared memory segments, and arenas. Rejected.

## Consequences

- `docs/spec/buffer-protocol.md` OQ-1 is resolved. The section on reference
  counting ("Implementations MAY use reference counting…") is confirmed as
  non-normative description, not a requirement.
- `docs/impl/c-ffi.md` MUST define the release callback signature:
  `void (*release_fn)(void *user_data)` or equivalent, called exactly once,
  thread-safe. No `retain` function is defined.
- Language bindings (`hurray-python`, and future Go/Java bindings) implement
  their own lifetime management on top of the single release callback. For
  Python, the `__dlpack__` / `__dlpack_device__` protocol already provides the
  correct abstraction.
- The internal reference counting implementation in `hurray-core` (when
  multi-consumer sharing is needed within a single process) is an implementation
  detail of `hurray-ffi` and is not visible at the ABI boundary.
