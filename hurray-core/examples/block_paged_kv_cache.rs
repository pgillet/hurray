//! Block-paged layout: a PagedAttention-style KV cache snapshot.
//!
//! Demonstrates the block-paged layout (tag `0x0A`) — the interchange form of a
//! paged KV cache used in disaggregated LLM inference. It shows:
//!
//! - building the descriptor for one `{kv_role, layer}` snapshot,
//! - resolving a logical `(sequence, token, head, dim)` to a flat `page_pool` offset
//!   through the `block_table` / `seq_ptr` indirection,
//! - prefix sharing: two sequences naming the same physical page resolve to the same
//!   offset (no copy),
//! - validating the storage invariants and the quantization-compatibility rule.
//!
//! Run with:
//!
//! ```text
//! cargo run --example block_paged_kv_cache
//! ```

use hurray_core::{
    layout::{
        addressing::block_paged::{element_offset_u32, validate_index_buffers_u32},
        BlockPagedLayout, BlockTableIndexType, KvRole, LayoutDescriptor,
    },
    Shape,
};

fn main() {
    println!("Block-Paged KV Cache Example\n");

    // The spec worked example (docs/spec/layouts/block-paged.md § Example):
    // layer 3, key role, 2 sequences, page_size = 4, num_pages = 5,
    // num_heads = 2, head_dim = 8. Sequence 0 has 6 tokens (2 pages), sequence 1
    // has 3 tokens (1 page) and REUSES sequence 0's first physical page (a shared
    // prompt prefix).
    let page_size = 4u32;
    let num_pages = 5u64;
    let num_heads = 2u64;
    let head_dim = 8u64;

    // One block-paged snapshot for {key, layer 3}.
    let layout = LayoutDescriptor::BlockPaged(BlockPagedLayout::new(
        page_size,
        num_pages,
        0, // paged_axis (the token axis)
        2, // num_seqs
        KvRole::Key,
        Some(3), // layer_index
        BlockTableIndexType::U32,
    ));

    // Logical shape is [total_tokens, num_heads, head_dim]; raggedness lives in seq_ptr.
    let shape = Shape::new(vec![9, num_heads, head_dim]).unwrap(); // 6 + 3 tokens
    println!(
        "layout: tag=0x{:02X}  buffer_count={:?}  valid_for_shape={}",
        layout.tag(),
        layout.buffer_count().map(|n| n.get()),
        layout.validate_against_shape(&shape).is_ok(),
    );

    // Index buffers (buffers 1 and 2 of the descriptor):
    //   seq_ptr     delimits each sequence's slice of the block table.
    //   block_table lists the physical page id of each logical page, batch-wide.
    let seq_ptr: &[u32] = &[0, 2, 3]; // seq 0 owns bt[0..2], seq 1 owns bt[2..3]
    let block_table: &[u32] = &[0, 1, 0]; // seq 1's page aliases seq 0's page 0

    // Storage invariants (seq_ptr monotonic + bounds, page ids < num_pages).
    validate_index_buffers_u32(num_pages, 2, seq_ptr, block_table)
        .expect("index buffers satisfy the storage invariants");
    println!("index buffers: valid\n");

    // Resolve a few logical coordinates to flat page_pool offsets.
    let lookups = [
        (0u64, 0u64, 0u64, 0u64), // seq 0, token 0, head 0, dim 0
        (0, 4, 0, 0),             // seq 0, token 4 -> page 1
        (0, 5, 1, 7),             // seq 0, last valid element of the sequence
        (1, 0, 0, 0),             // seq 1, token 0 -> aliases seq 0's page 0
    ];
    for (s, t, h, d) in lookups {
        let flat = element_offset_u32(
            s,
            t,
            h,
            d,
            page_size,
            num_pages,
            num_heads,
            head_dim,
            block_table,
            seq_ptr,
        )
        .expect("in-bounds lookup");
        println!("  (seq={s}, tok={t}, head={h}, dim={d}) -> page_pool[{flat}]");
    }

    // Prefix sharing: seq 0 and seq 1 token 0 resolve to the SAME physical slot.
    let a = element_offset_u32(
        0,
        0,
        0,
        0,
        page_size,
        num_pages,
        num_heads,
        head_dim,
        block_table,
        seq_ptr,
    )
    .unwrap();
    let b = element_offset_u32(
        1,
        0,
        0,
        0,
        page_size,
        num_pages,
        num_heads,
        head_dim,
        block_table,
        seq_ptr,
    )
    .unwrap();
    println!(
        "\nprefix sharing: seq0 and seq1 token 0 -> offsets {a} and {b} (equal = {})",
        a == b
    );

    // Quantization compatibility (spec § Quantization Compatibility):
    // per-block-affine (scheme 0x03) is only valid with axis == 0 and block_size == page_size.
    let bp = BlockPagedLayout::new(
        page_size,
        num_pages,
        0,
        2,
        KvRole::Key,
        Some(3),
        BlockTableIndexType::U32,
    );
    let ok = bp.validate_quantization_compatibility(0x03, 0, page_size); // axis 0, block_size == page_size
    let bad = bp.validate_quantization_compatibility(0x03, 0, page_size + 1); // block_size mismatch
    println!(
        "\nquant per-block-affine (block_size==page_size): {}",
        ok.is_ok()
    );
    println!(
        "quant per-block-affine (block_size!=page_size): rejected = {}",
        bad.is_err()
    );

    println!("\nExample completed successfully.");
}
