# ADR-002: Multi-Buffer Descriptor for Sparse Layouts

## Status

Accepted

## Context

Sparse tensor formats (CSR, CSC, COO, BSR, ELLPACK) require multiple distinct
component arrays — values, index arrays, and pointer arrays — each with a different
element type and length. The original `memory-layout.md` draft assumed a single buffer
per tensor descriptor and deferred the question of how to handle this as OQ-1.

The options considered were:

1. **Single buffer with offsets**: pack all component arrays into one contiguous
   buffer, addressing each via `byte_offset`. Simple but problematic: the components
   have different element types, different alignment requirements, and sizes that are
   only known after computing `nnz`. Packing them together also breaks the zero-copy
   invariant when only one component needs to be shared.

2. **Nested tensor descriptors**: each component array is itself a full Hurray tensor
   descriptor. Clean but heavyweight — a CSR tensor would require three complete tensor
   descriptors, each with its own layout tag, shape, and framing overhead.

3. **Multi-buffer descriptor (buffer table)**: the tensor descriptor carries a `uint8`
   count followed by an ordered list of buffer handles. Dense tensors have count = 1;
   sparse layouts declare how many buffers they need and what each holds. This follows
   the same model Apache Arrow uses for arrays (up to three buffers per column: validity
   bitmap, offsets, data).

## Decision

A **buffer table** is introduced as a first-class field in every tensor descriptor.
The buffer table is a `uint8`-prefixed ordered list of buffer handles (as defined in
`buffer-protocol.md`). Dense tensors always have count `0x01`. Sparse layout tags
declare their required buffer count and the role of each buffer (values, indices,
pointers).

The existing `buffer_index` field in the general subpaving layout (`0x06`) already
anticipated this; the buffer table formalises what those indices reference.

## Alternatives Considered

- **Single buffer with offsets**: rejected. Different element types in one buffer
  breaks type-safe access and complicates zero-copy sharing of individual components.
- **Nested tensor descriptors**: rejected. Excessive framing overhead for what are
  logically sub-arrays of a single sparse tensor, not independent tensors.

## Consequences

- All tensor descriptors carry a `uint8` buffer count, costing one byte for every
  dense tensor on the wire. This is accepted in exchange for a uniform decoder path
  (no special-casing of dense vs. sparse).
- Sparse layout tags are now unblocked. They can be assigned and fully specified in
  a future revision of `memory-layout.md`.
- `buffer-protocol.md` must define the buffer handle encoding used in the table
  entries, including size, alignment, and device fields.
- The general subpaving layout's `buffer_index` field continues to work unchanged;
  it indexes into the new buffer table.
