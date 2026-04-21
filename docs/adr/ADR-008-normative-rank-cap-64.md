# ADR-008: Normative Rank Cap of 64

## Status

Accepted

## Context

OQ-1 in `docs/spec/data-model.md` asked whether the specification should impose
a normative maximum rank (e.g., 64) or leave rank unconstrained with only a
SHOULD-level recommendation for implementations.

The `rank` field in the tensor descriptor is encoded as a `uint32`, which allows
values up to `0xFFFFFFFF`. Without a normative cap, a conforming reader would be
required to attempt to parse a shape array of up to ~4 billion `uint64` values
(32 GB) before determining whether the descriptor is valid. This is a latent
denial-of-service vector for any implementation that reads Hurray descriptors
from an untrusted source — IPC, cross-machine streaming, or file format.

In practice, no known ML workload uses tensors with rank above single digits.
PyTorch caps rank at 64 (MAX_DIMS). NumPy caps at 32 (NPY_MAXDIMS). TensorFlow
and Apache Arrow impose no formal cap, but neither targets the same
security-sensitive interchange contexts as Hurray.

A normative cap also has implementation benefits: shape arrays, stride arrays,
and per-dimension layout parameters can be stack-allocated with a fixed bound,
eliminating heap allocation on the hot path of descriptor parsing.

## Decision

The maximum rank of a Hurray tensor is **64**.

1. A writer MUST NOT emit a descriptor with `rank > 64`. A reader MUST reject a
   descriptor with `rank > 64`.
2. A conforming implementation MUST support tensors of rank `0` through `64`
   inclusive.
3. The cap applies to all layout descriptors: strides, tile shapes, Morton bits,
   sparse index arrays, and shard offsets are all bounded by `rank ≤ 64`.
4. The `uint32` encoding of `rank` is unchanged; values `65`–`0xFFFFFFFF` are
   reserved and MUST be rejected.

## Alternatives Considered

**Leave rank unconstrained (SHOULD-level recommendation only).**
Pros: maximum flexibility for scientific computing use cases with very high
dimensional data.
Cons: exposes every reader to a DoS vector — a four-byte field that instructs
the reader to consume up to 32 GB before rejecting the descriptor. For a
format designed for IPC and cross-machine streaming, this is unacceptable.
Also eliminates stack-allocation optimisations that simplify implementation.
Rejected.

**Cap at 32 (NumPy NPY_MAXDIMS).**
Pros: smaller stack footprint; matches NumPy.
Cons: rejects tensors that PyTorch (MAX_DIMS = 64) considers valid, breaking
round-trip fidelity at the upper edge of PyTorch's own range. Rejected in
favour of 64 to match the dominant ML framework.

**Cap at 8 or 16 (practical ML maximum).**
Pros: tightest possible bound; eliminates edge cases entirely.
Cons: overly restrictive — future architectures (e.g., multi-dimensional
attention with explicit head, sequence, and batch axes) may reach 8 naturally.
Rejected as unnecessarily limiting.

## Consequences

- All spec sections that reference per-dimension arrays (`shape`, `strides`,
  `tile_shape`, `morton_bits`, `outer_strides`, `inner_strides`, shard offsets)
  are implicitly bounded at 64 entries. No individual section needs to repeat
  the cap; a normative cross-reference to this ADR from `data-model.md` suffices.
- Implementations MAY use fixed-size stack arrays of length 64 for all
  per-dimension data, eliminating heap allocation from the descriptor-parsing
  hot path.
- `docs/spec/data-model.md` OQ-1 is resolved and the marker MUST be removed.
  The existing conformance text ("MUST support tensors of rank up to 64
  inclusive") is confirmed and extended with the rejection rule for `rank > 64`.
- `docs/impl/compliance.md` MUST include a test vector with a descriptor
  carrying `rank = 65` and verify that a conforming reader rejects it.
