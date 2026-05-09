//! Tensor descriptor encode/decode example.
//!
//! Demonstrates building [`TensorDescriptor`] values, encoding them to bytes,
//! printing a hex dump, and decoding them back, covering the most common cases:
//!
//! 1. `float32 [3, 4]` row-major — the spec's worked example (61 bytes).
//! 2. `float16 [8, 8]` column-major with statistics (advisory metadata).
//! 3. `int4 [1024, 1024]` row-major shard of a larger `[4096, 1024]` parent.
//!
//! Run with:
//!
//! ```text
//! cargo run --example encode_decode_descriptor
//! ```

use hurray_core::{
    descriptor::{ShardDescriptor, Statistics, StatisticsMask, TensorDescriptor},
    layout::LayoutDescriptor,
    BufferHandle, DeviceTag, ElementType, Shape, MIN_BUFFER_ALIGNMENT,
};

fn hex_dump(label: &str, bytes: &[u8]) {
    println!("{label} ({} bytes):", bytes.len());
    for (i, chunk) in bytes.chunks(16).enumerate() {
        print!("  {:04x}  ", i * 16);
        for b in chunk {
            print!("{b:02x} ");
        }
        // Padding for short last row.
        for _ in chunk.len()..16 {
            print!("   ");
        }
        print!(" |");
        for b in chunk {
            let c = if b.is_ascii_graphic() || *b == b' ' {
                *b as char
            } else {
                '.'
            };
            print!("{c}");
        }
        println!("|");
    }
    println!();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── Example 1: spec worked example ───────────────────────────────────────
    // float32 [3, 4] row-major — spec §metadata "Worked Example" (61 bytes).

    let shape = Shape::new(vec![3u64, 4]).unwrap();
    let buffer = BufferHandle::new(192, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu)?;
    let desc = TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        shape,
        0,
        LayoutDescriptor::RowMajor,
        vec![buffer],
        None,
        None,
        None,
        None,
    )?;

    let encoded = desc.encode()?;
    hex_dump("Example 1 — float32 [3,4] row-major", &encoded);
    assert_eq!(encoded.len(), 61, "spec worked example must be 61 bytes");

    let decoded = TensorDescriptor::decode(&encoded)?;
    assert_eq!(decoded, desc);
    println!("  Round-trip OK: decoded descriptor matches original.\n");

    // ── Example 2: float16 [8, 8] col-major with statistics ──────────────────

    let shape2 = Shape::new(vec![8u64, 8]).unwrap();
    let buf2 = BufferHandle::new(128, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu)?;

    // Advisory statistics — value range is always valid here.
    let stats = Statistics {
        computed_mask: StatisticsMask(
            StatisticsMask::VALUE_RANGE_VALID | StatisticsMask::NAN_INF_VALID,
        ),
        nnz: 0,
        sparsity_ratio: 0.0,
        value_min: -1.0,
        value_max: 1.0,
        value_abs_max: 1.0,
        value_mean: 0.0,
        value_stddev: 0.5,
        nm_n: 0,
        nm_m: 0,
        has_nan: false,
        has_inf: false,
    };

    let desc2 = TensorDescriptor::new(
        1,
        0,
        ElementType::Float16,
        shape2,
        0,
        LayoutDescriptor::ColMajor,
        vec![buf2],
        None,
        None,
        Some(stats),
        None,
    )?;

    let encoded2 = desc2.encode()?;
    hex_dump(
        "Example 2 — float16 [8,8] col-major + statistics",
        &encoded2,
    );

    let decoded2 = TensorDescriptor::decode(&encoded2)?;
    assert_eq!(decoded2, desc2);

    let stats_back = decoded2.statistics.as_ref().unwrap();
    assert!(stats_back.computed_mask.value_range_valid());
    assert!(!stats_back.has_nan);
    println!(
        "  value_min={}, value_max={}",
        stats_back.value_min, stats_back.value_max
    );
    println!("  Round-trip OK: statistics section preserved.\n");

    // ── Example 3: int4 [1024, 1024] row-major shard of [4096, 1024] ─────────

    let shape3 = Shape::new(vec![1024u64, 1024]).unwrap();
    // int4: 1024 × 1024 elements = 1,048,576 values packed into 524,288 bytes.
    let buf3 = BufferHandle::new(524_288, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu)?;

    // This shard covers rows 2048..3071 of the full 4096-row parent.
    let shard = ShardDescriptor::new(
        vec![4096u64, 1024], // parent shape
        vec![2048u64, 0],    // shard origin in the parent
    );

    let desc3 = TensorDescriptor::new(
        1,
        0,
        ElementType::Int4,
        shape3,
        0,
        LayoutDescriptor::RowMajor,
        vec![buf3],
        None,
        Some(shard?),
        None,
        None,
    )?;

    let encoded3 = desc3.encode()?;
    hex_dump(
        "Example 3 — int4 [1024,1024] shard of [4096,1024]",
        &encoded3,
    );

    let decoded3 = TensorDescriptor::decode(&encoded3)?;
    assert_eq!(decoded3, desc3);

    let shard_back = decoded3.shard.as_ref().unwrap();
    println!("  parent shape: {:?}", shard_back.parent_shape);
    println!("  shard offset: {:?}", shard_back.shard_offset);
    println!("  Round-trip OK: shard section preserved.\n");

    // ── Summary ───────────────────────────────────────────────────────────────

    println!("All examples encoded and decoded successfully.");
    println!("  Example 1: {} bytes (spec worked example)", encoded.len());
    println!(
        "  Example 2: {} bytes (+ 72-byte statistics section)",
        encoded2.len()
    );
    println!("  Example 3: {} bytes (+ shard section)", encoded3.len());

    Ok(())
}
