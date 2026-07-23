# Block-Paged Layout — Hurray Format Specification

> **Status:** Draft

**Layout tag:** `0x0A` | **Tier:** 1 | **Type:** Indirect

> This section uses RFC 2119 key words: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT,
> SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

## Description

The block-paged format stores a tensor whose **paged axis** is divided into fixed-size
**pages** drawn from a shared **page pool**, with a **block table** mapping each logical
page position to a physical page ID. It is the interchange form of a PagedAttention KV
cache: one page pool plus a ragged set of per-sequence block tables, addressed CSR-style
by a `seq_ptr` offset array.

Block-paged is an **indirect-dense** layout: every logical element along the paged
axis exists (there are no implicit zeros, unlike sparse layouts), but the mapping from a
logical index to a physical buffer position is resolved through the block table rather
than by an affine stride formula. A block-paged descriptor describes a **static snapshot
of one whole batch** for one `{kv_role, layer_index}` pair. It carries no live allocator
state.

Prefix sharing across sequences is expressed as **static structure**: two block-table
entries MAY name the same physical page ID. The aliasing is internal to the block table;
it creates no shared buffer ownership (see § Prefix Sharing).

> **Note (non-normative):** Block-paged is defined only for **rank-3 tensors** in this
> version of the specification. Generalisation to other ranks is left to a future
> revision. The whole transformer KV cache (`[layers, 2, ...]`) is transmitted as a
> stream of descriptors, one per key/value role per layer (see ADR-024).

A conforming implementation MUST reject a block-paged descriptor whose `rank` is not 3.

> **Note (non-normative):** "Page" and "block" refer to the same fixed-size physical
> unit. "Page" names the storage slot in the `page_pool`; "block" names the entry in the
> index structure (`block_table`) that maps a logical page position to a physical page
> ID. This dual naming matches the vLLM convention (page pool, block table).

## Logical Shape

The logical shape is `[total_tokens, num_heads, head_dim]`, where `total_tokens` is the
sum of the per-sequence token counts across the batch. The ragged per-sequence structure
is carried by `seq_ptr` (see § Buffer Table), not by the shape, which remains a
hyperrectangle. The **paged axis** is the axis subdivided into pages; in this version it
MUST be axis 0 (`paged_axis = 0`).

## Buffer Table

A block-paged tensor descriptor MUST have `buffer_count >= 3` in the buffer table.
Buffers 0–2 are always present; buffers 3 and up are present only when the
`HAS_QUANTIZATION` flag is set, per the placement rules in `quantization.md`.

| Buffer index | Name | Element type | Length | Description |
|---|---|---|---|---|
| 0 | `page_pool` | tensor storage type | `num_pages * page_size * num_heads * head_dim` elements | Flat pool of fixed-size pages. Page `p` occupies the contiguous slice `[p * page_size * num_heads * head_dim, (p + 1) * page_size * num_heads * head_dim)`. |
| 1 | `block_table` | `uint32` or `uint64` (per `block_table_index_type`) | `total_logical_pages` elements | Concatenation of every sequence's page-ID list, in sequence order. `block_table[k]` is the physical page ID of the `k`-th logical page across the batch. |
| 2 | `seq_ptr` | `uint32` or `uint64` (per `block_table_index_type`) | `num_seqs + 1` elements | Offset array into `block_table`. Sequence `s` owns `block_table[seq_ptr[s]..seq_ptr[s+1])`. `seq_ptr[0] = 0` and `seq_ptr[num_seqs] = total_logical_pages`. |
| 3 | `scales` | quantization scale type | `num_pages * page_size` elements (per-token) | Present only when `HAS_QUANTIZATION` is set. Paged through the **same** `block_table` as the values. |
| 4 | `zero_points` | quantization zero-point type | same length as `scales` | Present only for an asymmetric scheme that uses a separate zero-point buffer (see `quantization.md` § Buffer Table Placement Rules). |

where `total_logical_pages = seq_ptr[num_seqs]`.

> **Note (non-normative):** Buffers 1 (`block_table`) and 2 (`seq_ptr`) share the same
> element type, selected by `block_table_index_type`.

> **Note (non-normative):** Empty batch (`num_seqs = 0`): `block_table` (buffer 1) is
> empty (`byte_size = 0`, and MAY be a null pointer per `buffer-protocol.md`), and
> `seq_ptr` (buffer 2) contains exactly one element, `seq_ptr[0] = 0`.

## Additional Descriptor Fields

| Field | Type | Description |
|-------|------|-------------|
| `page_size` | `uint32` | Tokens per page. MUST be >= 1. Common values are 16 and 32. |
| `num_pages` | `uint64` | Number of physical pages in `page_pool`. |
| `paged_axis` | `uint32` | The axis subdivided into pages. MUST be 0 in this version. |
| `num_seqs` | `uint32` | Number of sequences in the batch. MAY be 0 for an empty batch. |
| `kv_role` | `uint8` | `0x00` = key, `0x01` = value, `0x02` = fused / non-KV generic. |
| `layer_index` | `uint32` | Transformer layer index. `0xFFFFFFFF` indicates the tensor is not layer-scoped. |
| `block_table_index_type` | `uint8` | `0x00` = `uint32` (default), `0x01` = `uint64`. Governs the element type of buffers 1 and 2. |
| `_reserved` | `uint8[6]` | MUST be `0x00`. Readers MUST reject a descriptor with any non-zero reserved byte. |

All multi-byte fields in the block-paged descriptor MUST be little-endian.

> **Note (non-normative):** A `uint32` block table addresses up to `0xFFFFFFFF`
> (4294967295) pages. Larger pools MUST use `uint64` (`block_table_index_type = 0x01`).

## byte_offset

For block-paged tensors, `byte_offset` MUST be set to `0x0000000000000000`.

> **Note (non-normative):** The `byte_offset` field in the common descriptor header is
> not meaningful for block-paged — logical position 0 is located via the block table,
> not at a fixed offset, exactly as for sparse layouts.

## Storage Invariants

A conforming writer MUST ensure:

1. `seq_ptr[0] = 0` and `seq_ptr[num_seqs] = total_logical_pages`.
2. `seq_ptr` is non-decreasing: `seq_ptr[s] <= seq_ptr[s+1]` for all `s`.
3. All physical page IDs are within bounds: `0 <= block_table[k] < num_pages` for all
   `k`.
4. The `page_pool` buffer size is
   `num_pages * page_size * num_heads * head_dim * element_byte_width` bytes (or the
   sub-byte equivalent, `ceil(num_pages * page_size * num_heads * head_dim / packing_factor)`,
   where `packing_factor` is as defined in `memory-layout.md` § Sub-Byte Types).
5. When `HAS_QUANTIZATION` is set, `scales` (and `zero_points`, when present) are
   indexed by the **same** `block_table`, with one entry per page slot
   (`num_pages * page_size` entries).
6. `paged_axis = 0`.

A conforming reader SHOULD validate invariants (1)–(3) and (6) and MUST reject
descriptors that violate them, unless operating in permissive mode. A reader is **NOT**
required to validate that aliased page IDs carry identical content: aliasing is
expressible, not validated (see § Prefix Sharing).

**Partial trailing page.** A sequence's token count need not be a multiple of
`page_size`; the slots in its final page beyond the valid token count are **undefined**.
A reader MUST NOT read past a sequence's valid token count (bounded by `seq_ptr` and the
sequence's token count). A writer SHOULD zero those unused slots when transferring across
a trust or tenant boundary.

## Element Lookup

To retrieve the value at token `t` of sequence `s`, head `h`, dimension `d`:

```
page_in_seq    = t / page_size            (integer division)
offset_in_page = t mod page_size
phys_page      = block_table[seq_ptr[s] + page_in_seq]
flat           = ((phys_page * page_size + offset_in_page) * num_heads + h) * head_dim + d
value          = page_pool[flat]
```

When the tensor is quantized, dequantize `value` using
`scales[phys_page * page_size + offset_in_page]` (and the corresponding `zero_points`
entry for an asymmetric scheme), per the active scheme in `quantization.md`.

## Prefix Sharing

Two sequences share a prefix when their block-table slices name the same physical page
IDs for the shared leading positions. For example, with

```
seq 0 block-table slice: [12, 5, 7, 9]
seq 1 block-table slice: [12, 5, 7, 3]
```

sequences 0 and 1 share physical pages 12, 5, and 7. The aliasing is internal to the
`block_table` buffer: it creates no shared buffer ownership and no wire-level reference
count (see ADR-024 and ADR-009). The engine-internal copy-on-write reference counts that
govern live page lifetime are out of scope; a block-paged descriptor is a static
snapshot.

## Quantization Compatibility

Block-paged tensors MAY be quantized using the schemes defined in `quantization.md`. The
quantization-parameter buffers occupy buffer indices 3 and up, per `quantization.md`
§ Buffer Table Placement Rules. The following rules govern how those schemes compose with
the paged structure.

### Per-page-slot parameters

For a quantized block-paged tensor, scales (and zero-points, when present) are stored
**per page slot**: the scale array has exactly `num_pages * page_size` entries, and it is
paged through the **same** `block_table` as the values. The scale for the element at
physical page `p`, slot `i` lives at `scales[p * page_size + i]`. This is the indexing
already given in § Element Lookup: a reader dequantizes using
`scales[phys_page * page_size + offset_in_page]` (and the corresponding `zero_points`
entry for an asymmetric scheme). Zero-point parameters, when carried in a separate buffer,
follow the same per-page-slot layout.

> **Note (non-normative):** Per-page co-location is required so that a shared (aliased)
> page carries its own scales. Because two sequences may name the same physical page ID
> in their `block_table` slices (see § Prefix Sharing), storing scales per logical token
> would force a shared page's leading tokens to be dequantized with different scales
> depending on the aliasing sequence, breaking numerical coherence of the prefix. Paging
> the scales through the same `block_table` as the values keeps every aliased reference to
> a page numerically identical.

### Scale-buffer size

The standard per-block-affine scale-buffer-size formula in `quantization.md` (which yields
`shape[axis] / block_size` scale entries) does NOT apply to block-paged. A reader MUST
compute the scale-buffer size as `num_pages * page_size` entries from the page structure,
because scales are stored per physical page slot rather than per logical token. This is
what allows an aliased/shared page to carry its own scales and keeps prefix sharing
numerically coherent.

### Per-scheme composition

- **Per-tensor** schemes (`scheme_tag = 0x01`, see `quantization.md`) compose normally:
  one scale (and zero-point) applies to the whole tensor.
- **Per-channel** schemes (`scheme_tag = 0x02`) compose normally when the quantization
  axis is `num_heads` (axis 1) or `head_dim` (axis 2). The per-channel parameters are
  indexed by channel along that axis and are independent of the paged structure.
- **Per-block-affine** (`scheme_tag = 0x03`) composes only under the constraint that the
  quantization descriptor's `axis` field MUST be `0` (the paged / token axis) and
  `block_size` MUST equal `page_size`. Under this constraint the per-block parameters
  coincide exactly with the per-page-slot parameters described above. A reader MUST reject
  a block-paged descriptor using `scheme_tag = 0x03` whose `axis` field is not `0`, or
  whose `block_size` is not equal to `page_size`.

A writer MUST NOT quantize the paged / token axis (axis 0) with the per-channel scheme
(`scheme_tag = 0x02`); paged-axis quantization MUST use per-block-affine under the
`block_size == page_size` constraint above, so that scales remain per-page-slot and
aliasing stays coherent.

## Sharding

A shard descriptor (see `memory-layout.md` § Splittability and Sharding) MUST NOT be
applied to a block-paged tensor in this version of the specification. A reader MUST reject
a block-paged descriptor that also carries a shard descriptor.

> **[OQ-1]:** Tensor-parallel / multi-GPU sharding of a block-paged KV cache (for example,
> splitting along the `num_heads` axis) is deferred to a future revision and is under
> consideration. Before sharding can be permitted, the interaction between a shard
> descriptor's `shard_offset` and the absolute `seq_ptr` offsets into the `block_table`
> must be resolved, since `seq_ptr` indexes the whole-batch block table rather than a
> shard-local slice.

## Alignment

All buffers MUST satisfy the alignment requirements in `buffer-protocol.md` (at least
64 bytes; page-aligned for GPU, IPC, or RDMA). All buffers in a block-paged descriptor
MUST share the same `device_tag` and `memory_class`.

## Framework Compliance

> **Note (non-normative):** There is no formal PagedAttention specification. This layout
> is designed to be faithful to the dominant inference frameworks, principally vLLM.
>
> In vLLM's FlashAttention / FlashInfer backend, the per-layer KV cache is stored as one
> tensor per key and per value with shape `[num_blocks, block_size, num_kv_heads,
> head_dim]`, and the default `block_size` is 16. The block table is a per-sequence list
> of physical block numbers, and sequences may share physical blocks (copy-on-write
> prefix sharing). These map onto Hurray as follows:
>
> | vLLM concept | Hurray block-paged |
> |---|---|
> | `num_blocks` | `num_pages` |
> | `block_size` (default 16) | `page_size` |
> | `num_kv_heads` | `num_heads` |
> | `head_dim` | `head_dim` |
> | per-K / per-V tensor | one descriptor per `kv_role` per `layer_index` |
> | per-sequence block table | `block_table` slice delimited by `seq_ptr` |
> | shared physical block | aliased page ID in `block_table` |
>
> Hurray encodes the **portable interchange layout**
> (`[num_blocks, block_size, num_kv_heads, head_dim]` per role), **not** a
> kernel-internal reshaped layout. In particular, the older `paged_attention_v1`
> key-cache reshape `[num_blocks, num_kv_heads, head_dim/x, block_size, x]` is an engine
> implementation detail and is out of scope for this layout. A producer that stores its
> cache in a kernel-internal layout MUST transpose to the interchange layout before
> emitting a block-paged descriptor.

## Example

A KV cache for layer 3, key role, with 2 sequences, `page_size = 4`, `num_pages = 5`,
`num_heads = 2`, `head_dim = 8`, element type `float16`. Sequence 0 has 6 tokens,
sequence 1 has 3 tokens, and sequence 1 reuses page 0 (a shared prefix):

```
num_seqs    = 2
seq_ptr     (buffer 2): [0, 2, 3]
block_table (buffer 1): [0, 1, 0]
page_pool   (buffer 0): 5 * 4 * 2 * 8 = 320 float16 values

kv_role                = 0x00   (key)
layer_index            = 3
page_size              = 4
num_pages              = 5
paged_axis             = 0
block_table_index_type = 0x00   (uint32)
```

Sequence 0 occupies `block_table[0..2) = [0, 1]` (6 tokens span 2 pages: page 0 holds
tokens 0–3, page 1 holds tokens 4–5, with slots 6–7 of page 1 undefined). Sequence 1
occupies `block_table[2..3) = [0]` (3 tokens in page 0, slot 3 undefined), aliasing
page 0 — the shared prefix.
