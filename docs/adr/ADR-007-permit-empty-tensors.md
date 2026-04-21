# ADR-007: Empty Tensors Are Normatively Permitted

## Status

Accepted

## Context

OQ-2 in `docs/spec/data-model.md` asked whether a tensor with any dimension size
equal to `0` (an **empty tensor**) should be permitted by the format, or rejected
as invalid. The current spec text permits empty tensors ("a reader MUST accept an
empty tensor without treating it as an error"); the question was whether to
confirm that permissive rule or tighten it for v1.

Hurray's primary goal is faithful zero-copy interchange between runtimes and
languages. The producers Hurray is designed to wrap — PyTorch, NumPy, JAX, Apache
Arrow, DLPack — all permit zero-size dimensions. ML compiler IRs (XLA/StableHLO,
TorchDynamo, MLIR `tensor` dialect) produce them as valid intermediate results
from shape inference under dynamic batching, masked selection, and uniform
broadcasting. ONNX is the principal outlier; its restriction is a documented
source of friction between training and deployment toolchains.

Hurray already distinguishes an **empty** dimension (size `0`, fully resolved,
zero elements) from a **dynamic** dimension (the sentinel
`0xFFFFFFFFFFFFFFFF`, unresolved size to be supplied by the interchange
channel). The two concepts are orthogonal and must not be conflated.

The secondary concern — that zero-byte GPU allocations are implementation-defined
on some runtimes — is real but resolvable by specifying that an empty tensor's
data buffer carries `size = 0` and MAY be represented with a null pointer, so no
device allocation is required.

Compatibility asymmetry matters: permitting now and forbidding later is a
breaking change for v1 producers; forbidding now and permitting later is
non-breaking. That asymmetry argues for caution, but the cost of forbidding —
every runtime that permits empty tensors must insert a defensive check before
every Hurray export — exceeds the cost of potentially tightening via a future
stricter conformance profile.

## Decision

Empty tensors are **normatively permitted** in Hurray v1.

1. Any dimension size in the shape array MAY be `0`. A tensor with one or more
   zero-size dimensions is **empty** and has `element_count = 0`.
2. A writer MAY emit an empty tensor. A reader MUST accept an empty tensor
   without treating it as an error.
3. An empty tensor MUST carry a complete, valid descriptor: rank, shape,
   element type, layout tag, buffer table, and any applicable quantization
   descriptor. No descriptor fields are optional on account of emptiness.
4. The data buffer(s) of an empty tensor MUST have byte size `0`. The buffer
   pointer MAY be null. The 64-byte buffer alignment requirement does not
   apply to a zero-length buffer (there are no addressable bytes to align);
   a non-null zero-length buffer pointer MAY have any alignment.
5. The value `0` (resolved empty dimension) and the sentinel
   `0xFFFFFFFFFFFFFFFF` (dynamic, unresolved dimension) are distinct. A reader
   MUST NOT treat them as equivalent, and MUST NOT substitute one for the other
   when resolving dynamic dimensions.
6. For sparse layouts (COO, CSR, CSC, and any future sparse layout), `nnz = 0`
   is valid and is independent of whether any logical shape dimension is `0`.
   An empty sparse tensor has both `element_count = 0` implied by shape and
   `nnz = 0`.
7. For sub-byte element types (`bool`, `int4`, `uint4`, `int2`, `uint2`), an
   empty tensor occupies `0` bytes; no partial trailing byte is emitted.
8. A quantization scheme with a per-axis or per-block descriptor MUST accept
   a shape in which the quantization axis has size `0`: the scales and
   zero-point arrays are themselves empty (length `0`) in that case.

## Alternatives Considered

**Reject empty tensors at the format level (ONNX-style).**
Pros: matches a deployment-focused lineage; eliminates the zero-byte buffer
edge case in C FFI and device allocators.
Cons: breaks zero-copy import from PyTorch, NumPy, JAX, Apache Arrow, and
DLPack — every producer-to-Hurray handoff would need a defensive shape check
and a fallback path. Rejected because it violates Hurray's primary goal of
faithful zero-copy interchange with the ecosystem it targets.

**Reject in a strict conformance profile only.**
Pros: permits the core format to stay liberal while giving deployment pipelines
a way to enforce stricter rules.
Cons: profiles are a v2-and-later concern; introducing one prematurely adds
spec surface area before the compliance matrix has stabilised. Deferred — a
future stricter conformance profile MAY forbid empty tensors without altering
the core spec defined here.

**Conflate zero-size and dynamic dimensions.**
Pros: one sentinel to carry both "unknown" and "zero".
Cons: loses information. A producer that knows a batch is empty (a filter that
selected zero rows) has different downstream semantics than a producer that has
not yet resolved the batch size. Rejected as a clear information loss.

## Consequences

- Zero-copy import paths from PyTorch, NumPy, JAX, Arrow, and DLPack work
  without shape-gating. This is the intended outcome.
- `docs/spec/buffer-protocol.md` (to be written) MUST specify that zero-length
  buffers MAY have a null pointer, that the 64-byte alignment requirement is
  waived for zero-length regions, and that consumers MUST NOT dereference a
  zero-length buffer regardless of its pointer value.
- The C FFI layer (`docs/impl/c-ffi.md`) MUST treat a zero-length buffer handle
  as a valid input and MUST NOT issue a zero-byte device allocation. A null data
  pointer with `size = 0` is the canonical representation.
- `docs/impl/compliance.md` MUST require at least one round-trip test vector for
  an empty tensor: recommended cases are shape `[0]`, shape `[3, 0, 5]`, and an
  empty sparse CSR (`nnz = 0`).
- `docs/spec/data-model.md` OQ-2 is resolved and the marker MUST be removed.
- Any future stricter conformance profile that forbids empty tensors MUST be
  introduced as an additional constraint layered on top of this ADR, never by
  modifying the core rule.
