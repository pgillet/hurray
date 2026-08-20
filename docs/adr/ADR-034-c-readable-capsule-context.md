# ADR-034: The capsule context becomes a C ABI handle

## Status

Proposed (2026-08-20)

Amends **ADR-023** § 5 (capsule lifetime, which owns the context) and § 8 (the ABI
version check), and resolves the C-level question **ADR-033** deferred.

## Context

The native protocol (`__hurray__` / `from_hurray`, ADR-023, ADR-030, ADR-033) exists
so that two Hurray-aware peers can exchange a tensor without the fidelity loss DLPack
imposes. ADR-023 § Context names three peer pairs it is meant to serve:

> `hurray-python` ↔ `hurray-python`, `hurray-python` ↔ `hurray-ffi` consumer, or
> `hurray-python` ↔ another binding built on `hurray-ffi`

**Only the first of those works.**

A capsule carries two things. The capsule *pointer* is a `HurrayBufferList` — a proper
`hurray-ffi` handle, reachable by anyone linking the C ABI. The capsule *context* holds
everything else:

```rust
struct NativeBufferContext {   // hurray-python/src/native_protocol.rs
    abi_version: u32,
    descriptor_bytes: Vec<u8>,
    tensor_ref: Py<PyAny>,
}
```

That struct is private to `hurray-python`, is not `#[repr(C)]`, contains a `Vec<u8>` and
a Python object reference, and is declared nowhere in `hurray.h`. Its layout is
unspecified and may change with a compiler version. A consumer that is not
`hurray-python` can call `PyCapsule_GetContext` and receive a pointer it has no legal
way to interpret.

So the buffers cross the boundary and the **descriptor does not** — and the descriptor
is the entire reason this protocol exists rather than DLPack. A Go or Julia binding on
`hurray-ffi` receives element bytes with no element type, no shape, no layout, no
quantization.

The gap also makes a normative rule unimplementable. `docs/impl/python-bindings.md`
§ ABI version requires:

> The capsule context MUST include the `HURRAY_C_ABI_VERSION` constant from the
> producing `hurray-ffi` build. […] the consumer MUST verify it before dereferencing
> the handle.

A non-Python consumer cannot verify the version it is required to verify, because the
version sits inside the struct it cannot read. The check exists precisely to stop a
consumer dereferencing handles from an incompatible build, and it is unavailable to
every consumer that is not the producer's twin.

This was found while resolving ADR-033's deferred question about C-level naming. That
question turns out to have a short answer — see § 5 — and this is the real defect
underneath it.

## Decision

### 1. The context becomes a `hurray-ffi` handle

`hurray-ffi` gains `HurrayTensorContext`: an opaque handle carrying what a capsule
needs beyond its buffers.

```c
typedef struct HurrayTensorContext HurrayTensorContext;

HurrayStatus hurray_tensor_context_new(uint32_t abi_version,
                                       const uint8_t *descriptor_bytes,
                                       uint64_t descriptor_len,
                                       void *owner,
                                       void (*owner_release)(void *owner),
                                       struct HurrayTensorContext **out);

HurrayStatus hurray_tensor_context_abi_version(const struct HurrayTensorContext *ctx,
                                               uint32_t *out);
HurrayStatus hurray_tensor_context_descriptor(const struct HurrayTensorContext *ctx,
                                               const uint8_t **out_bytes,
                                               uint64_t *out_len);
void hurray_tensor_context_destroy(struct HurrayTensorContext **ctx);
```

The handle owns a copy of the descriptor bytes; `hurray_tensor_context_destroy` frees
them and invokes `owner_release(owner)` exactly once, then nulls the caller's pointer —
the same discipline `hurray_buffer_destroy` already follows.

### 2. Opaque with accessors, not a public `repr(C)` struct

The obvious fix is to make the struct `#[repr(C)]` and declare its fields in
`hurray.h`, so a consumer reads them directly. This ADR rejects that.

Every other handle in the C ABI — `HurrayBuffer`, `HurrayBufferList`,
`HurrayDescriptor` — is opaque with accessor functions, and says so in its own
documentation: *"this struct is not `repr(C)`; its internal layout is an
implementation detail."* A single struct with a frozen public layout would be the one
exception, and it would freeze that layout for the life of the major version.

The usual argument for a public layout — that the consumer avoids linking the producing
library — does not apply. A consumer holding a capsule **already** links `hurray-ffi`;
it has to, because the capsule pointer is a `HurrayBufferList` and reading it requires
`hurray_buffer_list_get`. Nothing is saved by exposing this one type differently, and
consistency across the ABI is worth more.

### 3. The Python owner reference travels behind `void *`

The context must keep the source tensor alive while the capsule lives, and today it
does that with a `Py<PyAny>` — a type the C ABI must never see.

`owner` plus `owner_release` keeps it out. `hurray-python` boxes its strong reference,
passes it as `void *owner` with a release function that drops it, and the C ABI stores
two pointers it never interprets. This mirrors `release` / `release_context` on
`hurray_buffer_from_ptr`, which solves the same problem for buffer memory, so the
pattern is already the house idiom rather than a new invention.

It also puts the fix from PR #164 in one place: the release function `hurray-python`
supplies is the only code that touches Python, so it remains the only code that must
cope with running during interpreter finalization.

### 4. `abi_version` is read before anything else is trusted

`hurray_tensor_context_abi_version` MUST be callable on any context pointer produced by
any version of this ABI, and a consumer MUST call it first. Every other accessor MAY
assume the version has been checked.

This is what makes the handle extensible: fields added in a later ABI version are
reachable only through accessors added in that version, and a consumer that checked the
version knows which ones exist. Without that ordering rule, an opaque handle is as
frozen as a public struct.

### 5. The C ABI is not renamed

ADR-033 deferred "whether `hurray-ffi` should expose a matching C-level name". It should
not. `HurrayBuffer` is one buffer, `hurray_buffer_*` operates on one buffer,
`HurrayBufferList` is a list of buffers, `HurrayDescriptor` is a descriptor — every
name is accurate for what it names. The mistake ADR-033 corrected was a *protocol*
carrying a tensor while named for buffers, and the C layer had no protocol type to
misname.

It has one now, and it is named for what it carries. That closes the question.

### 6. C ABI version 3 → 4

New types and functions are additive, but a consumer must be able to tell whether a
context handle is available at all, so the version moves. `HURRAY_C_ABI_VERSION`
becomes `4`.

## Alternatives Considered

**`#[repr(C)]` public struct in `hurray.h`.** Rejected under § 2: inconsistent with
every other handle in the ABI, and freezes a layout for no benefit a consumer that
already links `hurray-ffi` can use.

**Put the descriptor bytes in the capsule *pointer* instead, as a combined
`HurrayTensor` handle owning both the buffer list and the descriptor.** Cleaner in the
abstract — one handle rather than a pointer/context pair — and worth revisiting. Rejected
here because ADR-030 § 2 fixed the capsule pointer as a `HurrayBufferList` and consumers
written against it would break for a change that buys elegance rather than capability.
Recorded as deferred below.

**Leave it, and document the protocol as `hurray-python` ↔ `hurray-python` only.**
Rejected. It would mean withdrawing a claim ADR-023 makes twice, and the protocol's
whole justification is preserving what DLPack cannot. A full-fidelity protocol that only
two instances of the same binding can speak is a private optimization, not an
interchange protocol — and the format's first principle is that it is language-agnostic.

**Expose the descriptor through the existing `HurrayDescriptor` handle instead of raw
bytes.** Attractive: the consumer would get a parsed descriptor rather than a byte
range. Rejected for now because it forces every producer to parse before sending and
every context to own a decoded structure, where today the encoded bytes are already in
hand and `hurray_descriptor_decode` is one call away for a consumer that wants one. The
bytes are the cheaper and more faithful thing to carry.

## Consequences

**Positive**

- The protocol's stated purpose becomes true: a non-Python binding on `hurray-ffi` can
  read the descriptor and the ABI version, not just the buffers.
- The `MUST verify the version` rule becomes implementable by every consumer rather
  than only by the producer's twin.
- The C ABI keeps one shape — opaque handles, accessor functions, explicit destroy —
  with no exception carved out for this type.
- The Python reference is confined behind `void *`, so the C ABI stays free of Python
  types and the finalization hazard stays in one function.

**Negative**

- **An ABI version bump**, with the compatibility-matrix and rebuild consequences every
  bump carries.
- **`hurray-python` no longer owns its context type**, and must construct it through
  `hurray-ffi`. That is the point, but it does mean the capsule path crosses one more
  boundary than before.
- **The descriptor bytes are copied into the context.** A borrow would avoid it, but
  would tie the context's validity to a buffer the producer might drop. A descriptor is
  small next to the tensor it describes.
- **No consumer exists to prove the design.** The first real non-Python binding may
  still find this insufficient; § 4's version-then-accessors rule is what leaves room to
  fix that without another break.

## Required Documentation Amendments

- `docs/impl/c-ffi.md` — `HurrayTensorContext`, its four functions, the ownership and
  version-check rules, and ABI version 4 in the version table.
- `docs/impl/python-bindings.md` — § Native Interchange Protocol: the capsule context is
  a `HurrayTensorContext`, and the version check is a documented C call rather than an
  internal detail.
- `docs/adr/ADR-023-*.md` § 5 and § 8 — amendment notes pointing here. The design
  note `D-NB2` in `hurray-python/src/native_protocol.rs` describes the context too and
  moves with the implementation.
- `docs/adr/ADR-033-*.md` § Open Questions Deferred — the C-level naming question is
  resolved by § 5.
- `hurray-python/COMPAT-MATRIX.md` — minimum `HURRAY_C_ABI_VERSION` 4.
- `docs/cookbook/layer-7-c-ffi.md` — a consumer-side example: check the version, read
  the descriptor, walk the buffer list.

## Open Questions Deferred

- **A combined `HurrayTensor` handle** owning both the buffer list and the descriptor,
  so a capsule carries one handle instead of a pointer/context pair. Better shape;
  breaks ADR-030 § 2's pointer contract. Worth doing at the next deliberate ABI break,
  not this one.
- **Whether `hurray-io`'s streaming frames should reuse `HurrayTensorContext`** as their
  C-level representation, rather than growing a parallel one when Layer 5 gains a C
  surface.
