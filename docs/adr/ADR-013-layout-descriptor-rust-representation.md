# ADR-013: Rust Representation of `LayoutDescriptor`

## Status
Accepted

## Context

Layer 3 of `hurray-core` introduces the in-memory representation of layout descriptors.
The format spec (`docs/spec/memory-layout.md` and `docs/spec/layouts/*.md`) defines:

- A 1-byte layout tag with a partitioned tag space: core (`0x01`–`0x3F`), extended
  (`0x40`–`0x7F`), reserved, private extension (`0xF0`–`0xFE`), and two invalid
  sentinels (`0x00`, `0xFF`).
- Per-layout descriptor fields ranging from "none" (row-major, column-major, Morton)
  through fixed scalar fields (Hilbert `hilbert_order`, COO `nnz` + `is_sorted`,
  CSR/CSC `nnz`) to variable-length structures (Strided, Tiled with recursion up to 8
  levels, Subpaving region lists, Private Extension opaque payload).
- A `buffer_count` that depends on the layout (1 for dense non-quantized, 2 for COO, 3
  for CSR/CSC).
- A permissive-mode requirement: a reader MUST be able to hold an unrecognised layout
  tag without dereferencing its data.

The constraints that drive the choice are:

1. Permissive mode requires a non-fatal "unknown layout" representation.
2. Sparse layouts demand a deterministic mapping from descriptor to required `buffer_count`.
3. Strides, tile shapes, and shape are coupled with `rank`; mismatches must be
   detectable at construction time.
4. The common path (row-major, column-major, Morton) MUST NOT allocate.
5. Adding Tier 2 layouts in a future spec version should not be a breaking change to
   consumer code.
6. The Rust type SHOULD mirror the wire layout cleanly enough that decoding does not
   require pivot/translation logic.

## Decision

`LayoutDescriptor` is a **fat enum with a small-data discipline and an explicit
`Unknown` variant** for permissive mode. Layout-specific structs live in their own
files under `hurray-core/src/layout/` and are referenced from the enum.

```rust
// hurray-core/src/layout/mod.rs (illustrative sketch)
#[non_exhaustive]
pub enum LayoutDescriptor {
    RowMajor,                                 // 0x01 — no payload, no alloc
    ColMajor,                                 // 0x02 — no payload, no alloc
    Strided(StridedLayout),                   // 0x03
    Tiled(Box<TiledLayout>),                  // 0x04 — Box keeps enum small (recursive)
    Morton,                                   // 0x05 — no payload, no alloc
    Subpaving(SubpavingLayout),               // 0x06
    Coo(CooLayout),                           // 0x07
    Csr(CsrLayout),                           // 0x08
    Csc(CscLayout),                           // 0x09
    Hilbert(HilbertLayout),                   // 0x40
    PrivateExtension(PrivateExtensionLayout), // 0xF0..=0xFE
    Unknown(UnknownLayout),                   // permissive mode only
}
```

Key rules:

1. **`#[non_exhaustive]`** on `LayoutDescriptor` and on every payload struct that may
   grow fields. Adding a Tier 2 layout is then a non-breaking change at the source
   level.
2. **`Box<TiledLayout>`** because `TiledLayout` is recursive (`inner_layout` can be
   `Tiled` again, up to 8 levels). Boxing only the recursive variant keeps the enum
   size bounded.
3. **`Unknown(UnknownLayout)`** carries `{ tag: u8, raw_bytes: Vec<u8> }`. It is the
   only path for tags the reader does not recognise. Constructors for named variants
   reject `0x00`, `0xFF`, and any reserved tag. The wire decoder routes unknown-but-
   not-invalid tags through `Unknown` only in permissive mode; in strict mode it
   returns an error.
4. **`UnknownLayout` carries the layout-section raw bytes**, not just the tag. This
   preserves zero-copy forwarding: a relay in permissive mode can re-emit an
   unrecognised descriptor byte-for-byte.
5. **`buffer_count()` is a method on `LayoutDescriptor`** returning
   `Option<NonZeroU8>`. For `Unknown`, it returns `None`. Sparse-layout buffer counts
   are constants on the per-variant struct, exposed through this method.
6. **Layout tag is not stored** in the enum payload. The discriminant *is* the tag for
   known variants; `Unknown` carries it explicitly. A `pub fn tag(&self) -> u8`
   returns the canonical wire tag for any variant, eliminating any tag/params mismatch.
7. **Rank validation is explicit and external.** `LayoutDescriptor` does not store
   `rank`. A method `validate(&self, shape: &Shape) -> Result<(), Error>` is called
   at the tensor descriptor boundary (Layer 4), where shape and layout are assembled
   together. Per-variant constructors validate intra-descriptor invariants only (e.g.,
   `tile_shape` values > 0, `hilbert_order` > 0).
8. **One file per layout** under `hurray-core/src/layout/`:
   `row_major.rs`, `col_major.rs`, `strided.rs`, `tiled.rs`, `morton.rs`,
   `subpaving.rs`, `coo.rs`, `csr.rs`, `csc.rs`, `hilbert.rs`,
   `private_extension.rs`, `unknown.rs`. `mod.rs` declares the enum and re-exports.

## Alternatives Considered

### Tag + separate `LayoutParams` enum (Option B)
A `{ tag: LayoutTag, params: LayoutParams }` struct mirrors the wire format.
Rejected: admits invalid combinations (`LayoutTag::RowMajor` with
`LayoutParams::Strided(..)`) at the type level, forcing every consumer to handle
"should never happen" branches. The fat enum makes invalid states unrepresentable.

### Trait object — `Box<dyn Layout>` (Option C)
Heap allocation on the hot path (every row-major tensor) violates constraint 4.
Exhaustive matching is lost. Downcasting requires `TypeId`-based escape hatches.
Rejected.

### Storing `rank` inside `LayoutDescriptor`
Allows construction-time stride validation but replicates rank across the tensor
descriptor boundary, creating a second source of truth that can drift during reshape.
Validation is performed once at Layer 4 where rank and layout meet. Rejected.

## Consequences

- **Zero-copy:** `Unknown` retains raw layout-section bytes for permissive forwarding.
  Known variants own O(rank) or O(region_count) heap data — unavoidable given the wire
  format. Buffer data itself is never touched.
- **Spec stability:** `#[non_exhaustive]` makes adding a Tier 2 layout non-breaking at
  the source level. Old strict-mode readers correctly reject new tags per spec.
- **FFI (Layer 7):** The C ABI MUST NOT expose this enum directly. It exposes opaque
  handles plus a `hurray_layout_tag()` getter and per-layout typed accessors.
- **Quantization interop:** Layout and quantization remain orthogonal. The tensor
  descriptor (Layer 4) combines `layout.buffer_count()` with the quantization scheme's
  parameter-buffer count to verify the buffer table.
- **Follow-up:** Layer 4 MUST call `layout.validate(&shape)` during tensor construction.

## Open / Deferred

- Whether `RegionDescriptor` (Subpaving) recursion should be boxed: start with a flat
  `Vec<RegionDescriptor>`; box if benchmarks show enum size is a problem.
- `serde` derives on layout types: deferred; not required for Layer 3.
- ADR-011 numbering conflict (`ADR-011-file-format-random-access-container.md` and
  `ADR-011-server-device-selection.md` share a number): flag to format-spec-writer
  for renumbering.

## Date
2026-05-04
