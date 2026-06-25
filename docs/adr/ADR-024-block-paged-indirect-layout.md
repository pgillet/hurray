# ADR-024: Block-Paged Layout as an Indirect-Dense Whole-Batch Snapshot

## Status

Draft

## Context

Disaggregated LLM inference is, at its core, a data-plane problem: the KV cache must
move between prefill and decode workers. The systems surveyed in `docs/prior-art.md`
§3 — DistServe, Mooncake, vLLM's KVConnector, NVIDIA Dynamo / TensorRT-LLM, llm-d, and
LMCache — all move the KV cache continuously, and all of them agree shape, dtype, paged
layout, and quantization *out-of-band*, shipping only opaque blocks plus IDs. §3.7
establishes this metadata gap; §3.8 names a Hurray block-paged layout as the planned
fix.

The KV cache is stored **paged** (PagedAttention): a flat pool of fixed-size pages plus
a per-sequence block table mapping logical token positions to physical page IDs.
Different sequences can share physical pages (prefix sharing / copy-on-write), which the
engines manage with internal reference counts.

Hurray's job is to describe the **static snapshot on the wire**, not live allocator
state. It must do so within the format's existing constraints:

- **Streamability** (`README.md`, `interchange.md`): descriptors precede their data, the
  format is self-delimiting, and there are no back-references and no end-of-file index.
- **Single-owner buffers** (`buffer-protocol.md`; ADR-009 keeps reference counting
  engine-internal and non-normative at the ABI boundary).
- **Hyperrectangle shape model** (`data-model.md`): the logical shape is always a
  rectangular index space.
- **Reuse of the existing quantization schemes** (`quantization.md`) rather than a
  paged-specific scheme.

This ADR records the design of a new `block-paged` layout that closes the §3.8 gap
within those constraints. Three open questions raised during design have been resolved
by the user and are folded into the Decision and Consequences below; no open questions
remain.

## Decision

### 1. A new addressing category: Indirect

`block-paged` introduces a third layout addressing category alongside Dense and Sparse:
**Indirect**. In an indirect layout every logical element exists (there are no
implicit zeros, unlike Sparse), but the mapping from a logical index to a physical
buffer position is **non-affine** — it is resolved through a block table rather than an
affine stride formula. The layout is assigned Tier-1 tag `0x0B` (tag `0x0A` is reserved
for the future CSF — Compressed Sparse Fiber — layout, keeping the sparse family
contiguous).

### 2. Whole-batch snapshot, not per-sequence descriptors

A single `block-paged` descriptor encodes one **whole batch**: one page pool, one flat
block table, and a CSR-style `seq_ptr` offset array delimiting each sequence's slice of
the block table. A per-sequence descriptor was rejected because cross-sequence prefix
sharing would then require a cross-tensor back-reference (one sequence's descriptor
pointing into another's pages), which the streamability contract forbids. Encoding the
whole batch turns prefix sharing into **internal aliasing** — two block-table entries
naming the same physical page ID — which is self-contained and streamable.

### 3. Ragged structure via `seq_ptr`; the shape stays a hyperrectangle

Per-sequence lengths are carried by `seq_ptr`, exactly as CSR carries ragged rows by
`row_ptr`. The logical shape `[total_tokens, num_heads, head_dim]` remains a
hyperrectangle, so the data model in `data-model.md` is unmodified. A padded dense shape
(`[num_seqs, max_seq_len, ...]`) was rejected: it wastes memory on padding and
misrepresents the snapshot. A conforming implementation MUST require `rank == 3` in this
version, mirroring CSR's rank-2-only restriction (see Consequences, OQ-3).

### 4. One descriptor per `{kv_role, layer_index}`

A full transformer KV cache (`[layers, 2, ...]`) is transmitted as a **stream of
descriptors**, one per key/value role per layer. This matches how the engines transfer
the cache — layer by layer (DistServe; vLLM's `save_kv_layer` / `wait_for_layer_load`) —
and keeps each descriptor a self-contained, streamable unit. A single whole-cache
descriptor was rejected because it would defeat layer-by-layer streaming.

### 5. Buffer table

`block-paged` is the first layout to mix a values buffer with two index/pointer buffers
and optional quantization-parameter buffers:

| Index | Buffer | Notes |
|-------|--------|-------|
| 0 | `page_pool` | flat pool of fixed-size pages |
| 1 | `block_table` | flat per-sequence concatenation of physical page IDs; `uint32` by default, `uint64` opt-in |
| 2 | `seq_ptr` | CSR-style offset array, `num_seqs + 1` entries |
| 3+ | quant-param buffers | present only when the tensor is quantized, per `quantization.md` placement rules |

Scalar descriptor fields: `page_size`, `num_pages`, `paged_axis` (= 0 in this version),
`num_seqs`, `kv_role`, `layer_index`, and `block_table_index_type`.

### 6. Quantization reuses existing schemes, paged through the same block table

Quantized paged KV caches reuse the schemes already defined in `quantization.md` — no
paged-specific scheme is introduced. Per-page-slot scales (and zero points, for asymmetric
schemes) are paged through the **same** block table as the values: a page carries its
own scales, so a shared page remains numerically coherent for every sequence that
aliases it. Quantization-parameter buffers occupy buffer indices 3 and up, per the
placement rules in `quantization.md`.

The composition rules are now **specified, not merely asserted** (see
`docs/spec/layouts/block-paged.md` § Quantization Compatibility): scales are per
page slot (`num_pages * page_size` entries, paged through `block_table`); the
scale-buffer size MUST be computed from the page structure rather than the standard
per-block-affine formula; per-tensor and per-channel (along `num_heads` or `head_dim`)
schemes compose normally; and per-block-affine (`scheme_tag = 0x03`) composes only when
`block_size == page_size` with the paged/token axis as the quantization axis.

### 7. Ownership is unchanged

`block-paged` requires no amendment to `buffer-protocol.md` or ADR-009. Aliasing is data
*inside* the `block_table` buffer; it does not create shared buffer ownership and
introduces no wire-level reference count. The engine-internal copy-on-write reference
counts that govern live page lifetime are out of scope: Hurray describes the snapshot,
not the allocator.

## Alternatives Considered

**Per-sequence descriptor (one descriptor per sequence).** Rejected: cross-sequence
prefix sharing would require a forbidden cross-tensor back-reference, violating the
streamability contract.

**Padded dense shape `[num_seqs, max_seq_len, num_heads, head_dim]`.** Rejected: wastes
memory on padding for ragged batches and cannot express page sharing across sequences.

**Classifying block-paged as a Sparse layout.** Rejected: a sparse layout implies
implicit zeros for unstored coordinates. A paged KV cache has no implicit zeros — every
logical element along the paged axis is materialised in some page. The mismatch would
mislead readers about the data model.

**A single whole-cache descriptor (`[layers, 2, ...]`).** Rejected: it defeats
layer-by-layer streaming, which is the dominant transfer pattern in disaggregated
serving.

**A paged-specific quantization scheme.** Rejected: it would duplicate the per-block
machinery already defined in `quantization.md`. Reusing the existing schemes and paging
the scales through the block table is sufficient.

**Wire-level reference counts on shared pages.** Rejected: it contradicts ADR-009
(reference counting is engine-internal and non-normative at the ABI) and the
snapshot framing — a snapshot has no live lifetime to count.

## Consequences

- `docs/spec/memory-layout.md` gains the `Indirect` type, the `0x0B` row in the Named
  Layout Tags table, and a buffer-table clause covering indirect layouts. Tag `0x0A`
  is reserved for the future CSF (Compressed Sparse Fiber) layout so the sparse family
  (COO `0x07`, CSR `0x08`, CSC `0x09`, CSF `0x0A`) stays contiguous.
- A new layout file `docs/spec/layouts/block-paged.md` is added.
- `block-paged` is the first layout to mix a values buffer with two index/pointer
  buffers plus optional quantization buffers — a new buffer-table shape for the format.
- A new reader validation surface is introduced: page-ID bounds checking
  (`0 <= block_table[k] < num_pages`) and `seq_ptr` monotonicity. Readers SHOULD
  validate these and MUST reject violations unless operating in permissive mode.
- **OQ-1 (partial trailing page) — resolved.** Slots beyond a sequence's valid token
  count are **left undefined**. A reader MUST NOT read past a sequence's valid token
  count (bounded by `seq_ptr`); a writer SHOULD zero unused slots when transferring
  across a trust or tenant boundary. No padding-value field is added.
- **OQ-2 (aliasing) — resolved.** Aliasing is **expressible, not validated** in this
  version. A reader validates page-ID bounds but is NOT required to verify that aliased
  pages hold identical content. No content-hash mechanism is added in v1.
- **OQ-3 (rank) — resolved.** `rank == 3` is **required** in this version
  (`[total_tokens, num_heads, head_dim]`), mirroring CSR's rank-2-only restriction.
  Generalisation is deferred to a future revision.
- **No ownership impact (confirmed).** `buffer-protocol.md` and ADR-009 are untouched;
  aliasing creates no shared buffer ownership and no wire-level reference count.
- **Sharding forbidden in v1.** A shard descriptor MUST NOT be applied to a block-paged
  tensor in this version (`docs/spec/layouts/block-paged.md` § Sharding). Tensor-parallel
  / multi-GPU sharding of a paged KV cache (e.g. along `num_heads`) is an **open
  question**: the interaction between a shard's `shard_offset` and the absolute `seq_ptr`
  offsets into the whole-batch `block_table` must be resolved before sharding can be
  permitted.

Date: 2026-06-23.
