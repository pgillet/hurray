"""A PagedAttention KV cache descriptor, from Python.

A block-paged tensor is three buffers: a flat pool of fixed-size pages, a
`block_table` naming the physical page behind each logical one, and a `seq_ptr`
delimiting each sequence's slice of the table. The logical shape stays a
hyperrectangle — `[total_tokens, num_heads, head_dim]` — and the ragged
per-sequence structure lives in `seq_ptr`, not in the shape.

Two sequences share a prefix when their block tables name the same page. Nothing
is copied; the aliasing is the point.

Run with:

    python hurray-python/examples/block_paged.py
"""

import struct

import hurray

PAGE_SIZE = 4  # tokens per page
NUM_PAGES = 5  # pool capacity
NUM_SEQS = 2
NUM_HEADS = 2
HEAD_DIM = 8

# ── The descriptor ────────────────────────────────────────────────────────────

print("=== The layout ===")

layout = hurray.BlockPagedLayout(
    page_size=PAGE_SIZE,
    num_pages=NUM_PAGES,
    paged_axis=0,  # MUST be 0 in this version
    num_seqs=NUM_SEQS,
    kv_role="key",  # "value" and "fused" also exist
    layer_index=3,  # None for a cache that is not layer-scoped
    block_table_index_type="uint32",
)

print(f"  {layout!r}")
print(f"  tag 0x{layout.tag:02X}, {layout.buffer_count} buffers")

# Block-paged is rank-3 only.
layout.validate_against_shape([9, NUM_HEADS, HEAD_DIM])
try:
    layout.validate_against_shape([9, NUM_HEADS])
except hurray.InvalidDescriptorError as exc:
    print(f"  rank-2 refused: {exc}")

# ── The three buffers ─────────────────────────────────────────────────────────

print("\n=== Three buffers ===")

# seq 0 owns block_table[0:2] = [0, 1]; seq 1 owns block_table[2:3] = [0].
seq_ptr = [0, 2, 3]
block_table = [0, 1, 0]  # seq 1's page 0 aliases seq 0's page 0

page_pool = bytes(NUM_PAGES * PAGE_SIZE * NUM_HEADS * HEAD_DIM * 2)  # float16

cache = hurray.Tensor(
    page_pool,
    hurray.float16,
    [9, NUM_HEADS, HEAD_DIM],
    aux_buffers=[
        struct.pack(f"{len(block_table)}I", *block_table),
        struct.pack(f"{len(seq_ptr)}I", *seq_ptr),
    ],
    layout=layout,
)

for index, handle in enumerate(cache.buffer_handles):
    name = ("page_pool", "block_table", "seq_ptr")[index]
    print(f"  [{index}] {name:12} {handle.byte_size:>5} bytes")

# ── Checking the index buffers ────────────────────────────────────────────────

print("\n=== The four storage invariants ===")

layout.validate_index_buffers(seq_ptr=seq_ptr, block_table=block_table)
print("  seq_ptr=[0, 2, 3], block_table=[0, 1, 0]: valid")
print("  (the repeated 0 is a shared prefix, not a mistake — aliasing is the point)")

for bad_seq_ptr, bad_table, why in (
    ([0, 2, 3], [0, 5, 0], "page id 5 is one past the end of a 5-page pool"),
    ([1, 2, 3], [0, 1, 0], "seq_ptr[0] must be 0"),
    ([0, 3, 2], [0, 1, 0], "seq_ptr must be non-decreasing"),
    ([0, 2, 9], [0, 1, 0], "seq_ptr[num_seqs] must equal len(block_table)"),
):
    try:
        layout.validate_index_buffers(bad_seq_ptr, bad_table)
        print(f"  MISSED: {why}")
    except hurray.InvalidDescriptorError:
        print(f"  refused: {why}")

print("\n  Worth calling before you ship a descriptor. The buffers' contents are")
print("  not checked when the descriptor is built — the descriptor describes them,")
print("  it does not contain them — so they are the only thing standing between a")
print("  consumer and an out-of-bounds read.")

empty = hurray.BlockPagedLayout(page_size=4, num_pages=5, paged_axis=0, num_seqs=0)
empty.validate_index_buffers([0], [])
print("\n  The empty batch (num_seqs=0, seq_ptr=[0]) is valid.")

# ── Quantization ──────────────────────────────────────────────────────────────

print("\n=== Quantization compatibility ===")

compatible = hurray.PerBlockAffine.symmetric(
    axis=0, block_size=PAGE_SIZE, scale_buffer_index=3, scale_type=hurray.float32
)
layout.validate_quantization_compatibility(compatible)
print(f"  per-block affine, axis 0, block_size {PAGE_SIZE}: compatible")

for incompatible, why in (
    (
        hurray.PerBlockAffine.symmetric(
            axis=0, block_size=8, scale_buffer_index=3, scale_type=hurray.float32
        ),
        "block_size 8 != page_size 4",
    ),
    (
        hurray.PerChannelAffine.symmetric(axis=0, scale_buffer_index=3),
        "per-channel on the paged axis",
    ),
):
    try:
        layout.validate_quantization_compatibility(incompatible)
        print(f"  MISSED: {why}")
    except hurray.InvalidDescriptorError:
        print(f"  refused: {why}")

print("\n  The rule: scales must stay per-page-slot, so a shared page carries its")
print("  own. block_size == page_size is what makes that true.")

# ── What Python does not do ───────────────────────────────────────────────────

print("\n=== Element lookup ===")

print("  There is no element_offset in Python. Resolving one logical coordinate")
print("  at a time is indexing, and Hurray is an interchange format rather than a")
print("  compute library — a Python loop over elements would be the wrong tool in")
print("  any case. The formula belongs in the consumer's kernel:")
print()
print("    page_in_seq    = token // page_size")
print("    offset_in_page = token % page_size")
print("    phys_page      = block_table[seq_ptr[seq] + page_in_seq]")
print("    flat = ((phys_page * page_size + offset_in_page) * num_heads + head)")
print("           * head_dim + dim")
print()
print("  What Python hands you is the descriptor that says how to read the bytes,")
print("  and the checks that say the index buffers are sound.")
