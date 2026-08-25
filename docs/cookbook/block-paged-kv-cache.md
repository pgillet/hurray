# Block-Paged KV Cache

## Purpose

The **block-paged** layout (tag `0x0A`) is the interchange form of a PagedAttention-style
KV cache — the central data structure moved between prefill and decode workers in
disaggregated LLM inference. It is an *indirect* layout: every logical element exists, but
the mapping from a logical index to a physical buffer position is resolved through a
**block table** rather than an affine stride formula.

A block-paged descriptor is a static **snapshot** of one whole batch for one
`{kv_role, layer}` pair. It carries no live allocator state; prefix sharing across
sequences is expressed as static structure (two block-table entries naming the same
physical page). See `docs/spec/layouts/block-paged.md` and ADR-024.

## The three buffers

| Buffer | Name | Contents |
|--------|------|----------|
| 0 | `page_pool` | The flat pool of fixed-size pages (`num_pages × page_size × num_heads × head_dim` elements). |
| 1 | `block_table` | Physical page id of each logical page, concatenated across sequences. |
| 2 | `seq_ptr` | Offset array delimiting each sequence's slice of `block_table` (CSR-style). |

The logical shape is `[total_tokens, num_heads, head_dim]` — a hyperrectangle. The ragged
per-sequence structure lives in `seq_ptr`, not in the shape.

## Building a descriptor

<div class="lang-tabs">

```rust
use hurray_core::layout::{BlockPagedLayout, BlockTableIndexType, KvRole, LayoutDescriptor};
use hurray_core::Shape;

// One snapshot: keys for layer 3, a batch of 2 sequences, 4 tokens per page.
let layout = LayoutDescriptor::BlockPaged(BlockPagedLayout::new(
    4,                        // page_size (tokens per page)
    5,                        // num_pages (pool capacity)
    0,                        // paged_axis (MUST be 0 in this version)
    2,                        // num_seqs
    KvRole::Key,              // this descriptor holds keys (Value / Fused also exist)
    Some(3),                  // layer_index (None = not layer-scoped, wire 0xFFFFFFFF)
    BlockTableIndexType::U32, // 32-bit block_table / seq_ptr (U64 for huge pools)
));

assert_eq!(layout.tag(), 0x0A);
assert_eq!(layout.buffer_count().map(|n| n.get()), Some(3));

// Block-paged is rank-3 only.
let shape = Shape::new(vec![9, 2, 8]).unwrap(); // [total_tokens, num_heads, head_dim]
assert!(layout.validate_against_shape(&shape).is_ok());
assert!(layout.validate_against_shape(&Shape::new(vec![9, 2]).unwrap()).is_err());
```

```python
import hurray

# One snapshot: keys for layer 3, a batch of 2 sequences, 4 tokens per page.
layout = hurray.BlockPagedLayout(
    page_size=4,                        # tokens per page
    num_pages=5,                        # pool capacity
    paged_axis=0,                       # MUST be 0 in this version
    num_seqs=2,
    kv_role="key",                      # "value" and "fused" also exist
    layer_index=3,                      # None = not layer-scoped
    block_table_index_type="uint32",    # "uint64" for huge pools
)

assert layout.tag == 0x0A
assert layout.buffer_count == 3

# Block-paged is rank-3 only.
layout.validate_against_shape([9, 2, 8])   # [total_tokens, num_heads, head_dim]
try:
    layout.validate_against_shape([9, 2])
    raise AssertionError("rank-2 should be refused")
except hurray.InvalidDescriptorError:
    pass
```

</div>

Python names the enum-like parameters rather than importing them: `kv_role="key"`,
`block_table_index_type="uint32"`.

## Element lookup through the block table

Resolving a logical `(sequence, token, head, dim)` to a flat `page_pool` offset follows
the spec formula:

```text
page_in_seq    = token / page_size
offset_in_page = token % page_size
phys_page      = block_table[seq_ptr[seq] + page_in_seq]
flat           = ((phys_page * page_size + offset_in_page) * num_heads + head) * head_dim + dim
```

<div class="lang-tabs">

```rust
use hurray_core::layout::addressing::block_paged::element_offset_u32;

// seq 0 owns block_table[0..2] = [0, 1]; seq 1 owns block_table[2..3] = [0].
let seq_ptr: &[u32] = &[0, 2, 3];
let block_table: &[u32] = &[0, 1, 0];

// seq 0, token 4 → page 1 (4/4), offset 0 → phys_page = block_table[1] = 1.
// flat = ((1*4 + 0)*2 + 0)*8 + 0 = 64.
let flat = element_offset_u32(
    /*s=*/ 0, /*t=*/ 4, /*h=*/ 0, /*d=*/ 0,
    /*page_size=*/ 4, /*num_pages=*/ 5, /*num_heads=*/ 2, /*head_dim=*/ 8,
    block_table, seq_ptr,
)
.unwrap();
assert_eq!(flat, 64);
```

```python
import hurray

# Python has no element_offset: resolving one logical coordinate at a time is
# indexing, and Hurray is an interchange format rather than a compute library
# (and a Python loop over elements would be the wrong tool regardless). The
# formula above is what a consumer implements in its own kernel; what Python
# gives you is the descriptor that says how to read it.
layout = hurray.BlockPagedLayout(page_size=4, num_pages=5, paged_axis=0, num_seqs=2)
print(layout.page_size, layout.num_pages, layout.num_seqs)
```

</div>

## Prefix sharing (copy-on-write, zero copy)

Two sequences share a prefix when their block tables name the same physical page. The
aliasing is internal to `block_table`: both references resolve to the *same* `page_pool`
offset, so nothing is copied.

<div class="lang-tabs">

```rust
use hurray_core::layout::addressing::block_paged::element_offset_u32;

let seq_ptr: &[u32] = &[0, 2, 3];
let block_table: &[u32] = &[0, 1, 0]; // seq 1's page 0 aliases seq 0's page 0

let seq0 = element_offset_u32(0, 0, 0, 0, 4, 5, 2, 8, block_table, seq_ptr).unwrap();
let seq1 = element_offset_u32(1, 0, 0, 0, 4, 5, 2, 8, block_table, seq_ptr).unwrap();
assert_eq!(seq0, seq1); // shared prefix → identical physical slot
```

```python
import hurray

layout = hurray.BlockPagedLayout(page_size=4, num_pages=5, paged_axis=0, num_seqs=2)

seq_ptr = [0, 2, 3]
block_table = [0, 1, 0]     # seq 1's page 0 aliases seq 0's page 0

# Aliasing is not an error — it is the point. The validator accepts it.
layout.validate_index_buffers(seq_ptr=seq_ptr, block_table=block_table)
```

</div>

## Validating the storage invariants

`validate_index_buffers_u32` / `_u64` check the four storage invariants:
`seq_ptr[0] == 0`, `seq_ptr` non-decreasing, `seq_ptr[num_seqs] == block_table.len()`, and
every `block_table[k] < num_pages`. The empty batch (`num_seqs == 0`, `seq_ptr == [0]`) is
valid.

<div class="lang-tabs">

```rust
use hurray_core::layout::addressing::block_paged::validate_index_buffers_u32;

let seq_ptr: &[u32] = &[0, 2, 3];
let block_table: &[u32] = &[0, 1, 0];
assert!(validate_index_buffers_u32(/*num_pages=*/ 5, /*num_seqs=*/ 2, seq_ptr, block_table).is_ok());

// A page id outside [0, num_pages) is rejected.
let bad: &[u32] = &[0, 5, 0]; // page 5 == num_pages
assert!(validate_index_buffers_u32(5, 2, seq_ptr, bad).is_err());
```

```python
import hurray

layout = hurray.BlockPagedLayout(page_size=4, num_pages=5, paged_axis=0, num_seqs=2)

layout.validate_index_buffers(seq_ptr=[0, 2, 3], block_table=[0, 1, 0])

# A page id outside [0, num_pages) is rejected.
try:
    layout.validate_index_buffers(seq_ptr=[0, 2, 3], block_table=[0, 5, 0])
    raise AssertionError("page 5 == num_pages should be refused")
except hurray.InvalidDescriptorError as exc:
    print(exc)
```

</div>

`num_pages` and `num_seqs` come from the layout, so only the two buffers are passed. This
is worth calling before shipping a descriptor: the buffers' *contents* are not checked
when the descriptor is built — the descriptor describes them, it does not contain them —
so they are the only thing standing between a consumer and an out-of-bounds read.

## Quantization compatibility

KV caches are often fp8/int8. Block-paged reuses the existing quantization schemes, with
one rule: per-block-affine (`scheme_tag = 0x03`) requires the quantization `axis` to be `0`
and `block_size` to equal `page_size`, so scales stay per-page-slot and a shared page
carries its own scales.

<div class="lang-tabs">

```rust
use hurray_core::layout::{BlockPagedLayout, BlockTableIndexType, KvRole};

let bp = BlockPagedLayout::new(4, 5, 0, 2, KvRole::Key, Some(3), BlockTableIndexType::U32);

// per-block-affine: valid only when axis == 0 and block_size == page_size.
assert!(bp.validate_quantization_compatibility(0x03, 0, 4).is_ok());
assert!(bp.validate_quantization_compatibility(0x03, 0, 8).is_err()); // block_size != page_size
assert!(bp.validate_quantization_compatibility(0x02, 0, 4).is_err()); // per-channel on the paged axis
```

```python
import hurray

layout = hurray.BlockPagedLayout(page_size=4, num_pages=5, paged_axis=0, num_seqs=2)

# per-block-affine: valid only when axis == 0 and block_size == page_size.
layout.validate_quantization_compatibility(
    hurray.PerBlockAffine.symmetric(
        axis=0, block_size=4, scale_buffer_index=3, scale_type=hurray.float32
    )
)

for incompatible in (
    hurray.PerBlockAffine.symmetric(          # block_size != page_size
        axis=0, block_size=8, scale_buffer_index=3, scale_type=hurray.float32
    ),
    hurray.PerChannelAffine.symmetric(        # per-channel on the paged axis
        axis=0, scale_buffer_index=3
    ),
):
    try:
        layout.validate_quantization_compatibility(incompatible)
        raise AssertionError("should be refused")
    except hurray.InvalidDescriptorError:
        pass
```

</div>

The Python check takes the quantization object rather than a scheme tag: the caller
already has one, and `0x03` is a byte to look up rather than a thing to pass.

## Sharding

A block-paged descriptor MUST NOT carry a shard descriptor in this version;
`TensorDescriptor::new` rejects the combination. Tensor-parallel / multi-GPU sharding of a
paged KV cache is a deferred open question (see `block-paged.md` § Sharding).

## Runnable example

<div class="lang-tabs">

```text
cargo run --example block_paged_kv_cache
```

```text
python hurray-python/examples/block_paged.py
```

</div>
