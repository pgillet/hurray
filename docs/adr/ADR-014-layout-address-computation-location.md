# ADR-014: Layout Address Computation Lives in `hurray-core` as a Trait-Based Sub-Module

## Status
Accepted

## Context

Layer 3 has just landed descriptor-only types in `hurray-core/src/layout/` (`LayoutDescriptor`,
`StridedLayout`, `TiledLayout`, `CooLayout`, etc.). These structs carry the metadata fields
defined by the spec but contain zero address-computation logic. Layer 4 (tensor descriptor
encoding) is about to begin, and Layer 4 also requires `hurray-inspect` to drop its
self-contained parser and depend on `hurray-core` (CLAUDE.md, "Implementation rules").

The spec defines an element-address formula for every layout in `docs/spec/memory-layout.md`
and `docs/spec/layouts/*.md`. These formulas are normative. The question is where in the
workspace the implementation of those formulas lives.

**Key forces:**

1. **Spec fidelity.** Address formulas are part of the format contract. The closer the formula
   sits to the descriptor whose fields parameterise it, the harder it is for the two to drift.
2. **Reusability.** `hurray-inspect`, `hurray-ffi`, `hurray-python`, and the future array-database
   engine all need to compute addresses. None should need an extra crate dependency for this.
3. **`hurray-core` charter.** The crate already carries quantization descriptors, alignment
   validation, and `rayon`. "No I/O, no async" — not "no logic".
4. **Implementation complexity (Morton and Hilbert):**
   - **Morton**: a doubly-nested loop over `(bit_position, dimension)` — trivial integer bitops.
   - **Hilbert** (Skilling's algorithm): two nested loops with bit-XOR swaps. Non-trivial to
     derive, mechanically straightforward to transcribe from the spec pseudocode. No SIMD or
     lookup table needed for v1.
   Neither algorithm justifies a separate crate.
5. **Deferral cost.** `hurray-inspect`'s self-contained parser MUST be replaced in Layer 4.
   Without address computation in core, inspect cannot display element values, forcing it to
   keep its private addressing code — the drift risk the design is meant to prevent.

## Decision

Address computation MUST live in `hurray-core`, in a dedicated sub-module
`hurray-core/src/layout/addressing/` (one file per layout, mirroring the descriptor layout).
It MUST NOT live in a new crate.

The implementation is organised around two traits defined in
`hurray-core/src/layout/addressing/mod.rs`:

```rust
/// Implemented by every dense layout descriptor.
pub trait ElementAddress {
    /// Returns the linear element offset (in logical elements, not bytes)
    /// of the element at the given multi-dimensional index.
    fn element_offset(&self, index: &[u64]) -> Result<u64, Error>;
}

/// Implemented by sparse layout descriptors; requires borrow of index buffers.
pub(crate) trait SparseElementAddress {
    /// Returns the storage offset of the given index, or None if structurally absent.
    fn sparse_element_offset(&self, index: &[u64], buffers: &SparseBuffers<'_>) -> Result<Option<u64>, Error>;
}
```

**Key rules:**

1. Each dense layout file under `hurray-core/src/layout/addressing/` MUST carry a
   `// Spec: docs/spec/layouts/<name>.md § <section>` comment at the top of each impl block
   to maintain auditable spec-to-code traceability.
2. `LayoutDescriptor` gains a single dispatching method `element_offset(&self, index: &[u64])
   -> Result<u64, Error>` in `hurray-core/src/layout/mod.rs` via a `match` over the enum.
3. Sparse impls are `pub(crate)` initially; promoted to `pub` after the first consumer (Layer 4)
   validates the API shape.
4. A free function `byte_address_from_element_offset(element_offset, byte_offset, element_type)
   -> ByteAddress` in `mod.rs` handles the common whole-byte / sub-byte byte-address conversion
   from `memory-layout.md § Element Address Computation`, separating pure layout geometry from
   element-type byte arithmetic.
5. `unsafe` MUST NOT be used in addressing code in v1. SIMD / lookup-table optimisations are
   explicitly deferred and MUST be benchmark-gated with a separate ADR before introduction.

## Alternatives Considered

### New `hurray-access` crate (Option B)
Would keep `hurray-core` as a pure-types crate. Rejected: the total addressing code is
~600–800 LOC; a separate crate adds dependency overhead for every consumer while the
supposed purity boundary is already not held (quantization logic, rayon). The real
boundary is *reference CPU addressing* (core) vs *backend-optimised addressing* (future
backend crates).

### Methods directly on structs, no trait (Option A-flat)
Simpler, but callers cannot dispatch polymorphically on `LayoutDescriptor` without their
own `match`, and adding a new layout in v2 gives no compile-time reminder that addressing
is needed. The trait costs almost nothing and pays for itself the first time a new layout
is added.

### Defer (Option C)
Directly contradicts the Layer 4 obligation to refactor `hurray-inspect`. Every Layer 5+
consumer that needs addressing would write its own copy, multiplying drift risk.

## Consequences

**Positive:**
- `hurray-core` is the single normative source for descriptor structure *and* addressing semantics.
- `hurray-inspect` Layer 4 refactor drops its self-contained parser in one move.
- `hurray-ffi`, `hurray-python`, and the array-DB engine get addressing at no extra dependency cost.
- The trait creates a compiler-enforced coverage hook when new layouts are added.
- Sparse/dense distinction is type-safe (two traits, different return types).

**Obligations created:**
- `hurray-core` description in `Cargo.toml` SHOULD be updated to mention reference address computation.
- `Error` enum gains `IndexOutOfRange` and `IndexRankMismatch` variants (additive, non-breaking).
- `// Spec:` comment on every `ElementAddress` impl is a review-time obligation.
- Conformance-table tests for Morton and Hilbert MUST be reproduced as unit tests.

## Open Questions

- **OQ-014.1:** Promote `SparseElementAddress` to `pub` after Layer 4 validation? (Defer.)
- **OQ-014.2:** Should `byte_address_from_element_offset` return a single struct with
  `bit_offset: 0` for whole-byte types, or a sum type? (Defer to first FFI consumer.)
- **OQ-014.3:** Subpaving region lookup — are regions pre-sorted? Route to `format-spec-writer`
  before the `SubpavingLayout` impl is written.

## Layer 4 Impact

The Layer 4 plan gains a prerequisite sub-task before the `hurray-inspect` refactor:

1. `hurray-core/src/layout/addressing/mod.rs` — traits + `byte_address_from_element_offset`
2. One addressing impl file per dense layout under `hurray-core/src/layout/addressing/`
3. Sparse impls (`coo.rs`, `csr.rs`, `csc.rs`) as `pub(crate)`
4. Unit tests reproducing Morton and Hilbert conformance tables from the spec
5. `Error` enum additions
6. Runnable example in `hurray-core/examples/element_offset.rs` and cookbook entry

## Date
2026-05-05
