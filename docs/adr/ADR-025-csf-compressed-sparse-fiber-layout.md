# ADR-025: CSF (Compressed Sparse Fiber) as the Rank-N Sparse Layout

## Status

Draft

## Context

Hurray already defines three sparse layouts: COO (`0x07`, any rank), CSR (`0x08`,
rank-2), and CSC (`0x09`, rank-2). CSR/CSC compress one mode with a dense outer
pointer array — compact and interop-canonical (SciPy, cuSPARSE, MKL, Eigen;
`hurray-python` already performs SciPy zero-copy sparse interop) — but they do not
generalise past rank 2. COO generalises to any rank but stores a full coordinate
tuple per non-zero with no hierarchical structure.

TACO uses **CSF** (Compressed Sparse Fiber): the rank-N generalisation of CSR/CSC. A
CSF tensor is a tree of `rank` levels, each compressing one mode with a `(pos, crd)`
pair, plus one shared values array. CSF is the natural higher-rank complement to COO
for structured sparsity — attention masks, sparse activations, and higher-order
factorisations. `TODO.md` names CSF, and tag `0x0A` was reserved for it in ADR-024 to
keep the sparse family contiguous (COO `0x07`, CSR `0x08`, CSC `0x09`, CSF `0x0A`).

The settled framing is that **CSR/CSC/COO are kept; CSF complements them**. CSR's
dense outer `row_ptr` is more compact and more interop-canonical for rank-2 matrices,
and an all-compressed CSF tree is a physically different layout. Writers SHOULD prefer
CSR/CSC for rank-2 data.

This ADR must respect the format's existing constraints:

- **Streamability / self-delimitation** (`README.md`, `interchange.md`): descriptors
  precede their data, all buffer lengths derive from the `pos` chain, and there are no
  back-references and no end-of-file index.
- **Single-owner buffers** (`buffer-protocol.md`; ADR-009).
- **Hyperrectangle shape model** (`data-model.md`): the logical shape is always a
  rectangular index space.
- **`byte_offset = 0` for sparse layouts** (`memory-layout.md`).
- **Reuse of the existing quantization schemes** (`quantization.md`).

This is the pre-1.0 draft period, so assigning the already-reserved tag `0x0A`
requires no `version_minor` bump, exactly as ADR-024 established for `0x0B`.

## Decision

### 1. All-compressed CSF, not the full per-mode model

Every level of the CSF tree is a **compressed** `(pos, crd)` pair. The TACO per-mode
dense/compressed model is **not** adopted in this version: dense-outer rank-2 cases
are already covered by CSR/CSC, and a per-level `mode_format[rank]` array is reader
burden Hurray does not need yet. This is forward-compatible: a future revision MAY add
a `mode_format: uint8[rank]` field to admit dense levels, and a v1 reader rejecting
that unknown field is intended versioning behaviour.

### 2. Explicit mode ordering

A CSF descriptor carries `mode_order: uint32[rank]`, a permutation of `0..rank-1`.
`mode_order[L]` is the logical dimension stored at level `L` (level `0` is outermost;
level `rank-1` is the leaf level directly above `values`). The bounding size for level
`L` is `shape[mode_order[L]]`. The logical shape is unchanged — `mode_order` affects
storage traversal only.

`mode_order` is a **performance knob**: it lets a writer match the tree's nesting to
its access pattern. A conforming reader MUST honour any valid `mode_order` permutation
for lookup and iteration, and MUST NOT reject a descriptor merely because `mode_order`
is non-identity — permutation support is mandatory for an interchange format. When a
writer has no access-pattern preference, the identity ordering
`[0, 1, ..., rank-1]` (row-major, outer-to-inner) is the RECOMMENDED default, for
reproducibility and cache-friendliness.

### 3. Buffer table = `2 × rank + 1`

`values` is at index `0` (consistent with CSR/CSC/COO). Each level `L` contributes a
`pos_L` buffer at index `2L + 1` and a `crd_L` buffer at index `2L + 2`. The top level
retains `pos_0 = [0, n_0]` (length 2) for structural uniformity; omitting the top
`pos` was rejected. All `pos`/`crd` buffers are `uint64`. Quantization-parameter
buffers, when present, occupy indices `2·rank + 1` and up.

### 4. Rank scope: `rank >= 3`

CSR, CSC, and COO own rank ≤ 2. A conforming reader MUST reject a CSF descriptor with
`rank < 3`. Rank remains capped at 64 by `data-model.md`.

### 5. Element lookup by per-level descent

Lookup descends the tree level by level, binary-searching each level's `crd` slice
delimited by the parent's `pos` entry. A coordinate not found at any level means the
element is implicitly **zero**.

### 6. Storage invariants mirror CSR, per level

For each level `L`: `pos_L[0] = 0`; `pos_L` is non-decreasing; the terminal `pos`
entry equals the level's stored count (`pos_0[1] = n_0`, `pos_L[n_{L-1}] = n_L`, with
`n_{rank-1} = nnz`); within each parent slice the `crd` values are strictly increasing
(no duplicate siblings); and `0 <= crd_L[i] < shape[mode_order[L]]`. In addition,
`mode_order` MUST be a valid permutation of `0..rank-1`.

### 7. Cross-cutting properties

`byte_offset` MUST be `0` (sparse). The layout is self-delimiting: every buffer length
derives from the `pos` chain, so no back-references or EOF index are needed.
Quantization decorates the `values` leaves (`values` is buffer 0); per-tensor,
per-channel, per-block, NF4, and MXFP schemes compose exactly as they do for COO/CSR,
with quant-parameter buffers at index `2·rank + 1` and up. Sharding is **forbidden**
in this version: the interaction between a shard's `shard_offset` and the per-level
`pos`/`crd` tree is unresolved. CSF is classified as **Sparse** (implicit zeros).

### 8. Tag assignment

CSF takes Tier-1 tag `0x0A`, classified Sparse. The `memory-layout.md` row for `0x0A`
is promoted from "(reserved — planned)" to a link to `layouts/csf.md`.

## Alternatives Considered

**Full per-mode dense/compressed model (TACO mode formats).** Rejected for v1:
dense-outer rank-2 cases are covered by CSR/CSC, and a per-level `mode_format` array is
reader burden Hurray does not need yet. Deferred and forward-compatible via a future
`mode_format` field.

**Replacing CSR/CSC with CSF.** Rejected (settled framing): CSR's dense outer pointer
is more compact and interop-canonical for rank-2, and all-compressed CSF is a
physically different layout.

**Omitting the top-level `pos_0`.** Rejected: structural uniformity across all levels
is worth the 16 bytes of a length-2 `pos_0`.

**Allowing CSF at rank 2 and up.** Rejected: it would duplicate CSR/CSC/COO. CSF is
restricted to `rank >= 3`.

**`uint32` indices.** Rejected for v1 to match the `uint64` indices used by
CSR/CSC/COO; an opt-in narrower index type may be added later.

**Non-Sparse classification.** Rejected: CSF has implicit zeros, so it is Sparse.

## Consequences

- `docs/spec/memory-layout.md`: the `0x0A` row is promoted from reserved to a link to
  `layouts/csf.md`, and the sparse buffer-table clause notes CSF's variable
  rank-dependent count of `2·rank + 1`.
- A new layout file `docs/spec/layouts/csf.md` is added.
- CSF is the first layout with a **variable, rank-dependent buffer count** and the
  first sparse layout with a **permutation descriptor field** (`mode_order`).
- A new reader validation surface is introduced: `mode_order` permutation validity,
  per-level `pos` monotonicity and terminal checks, and per-level `crd` bounds.
  Readers SHOULD validate these and MUST reject violations unless operating in
  permissive mode.
- The forward-reference note in `csr.md` (and the analogous mention) is now satisfied
  and updated to point at `csf.md` (editorial).
- Writers SHOULD prefer CSR/CSC for rank-2; CSF MUST NOT be used below rank 3.
- No `version_minor` bump (the tag was already reserved; pre-1.0 draft period).
- The `TODO.md` CSF entry can be marked done once `csf.md` lands.

Date: 2026-06-27.
