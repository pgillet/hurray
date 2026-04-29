//! Demonstrates the element type system and shape model.
//!
//! Run with:
//!
//! ```text
//! cargo run --example element_types -p hurray-core
//! ```

use hurray_core::{buffer_size_bytes, ElementType, Shape, DYNAMIC};

fn main() {
    // ── Type properties ───────────────────────────────────────────────────────

    println!("=== Tier 1 floating-point types ===");
    let floats = [
        ElementType::Float16,
        ElementType::BFloat16,
        ElementType::Float32,
        ElementType::Float64,
    ];
    for ty in floats {
        println!(
            "  {:<12}  tag=0x{:02X}  bits={:3}  align={}  signed={}",
            ty,
            ty.tag(),
            ty.bit_width(),
            ty.element_alignment(),
            ty.is_signed(),
        );
    }

    println!("\n=== Sub-byte types ===");
    let sub_byte = [
        ElementType::Bool,
        ElementType::Int2,
        ElementType::Uint2,
        ElementType::Int4,
        ElementType::Uint4,
        ElementType::Float4E2M1,
        ElementType::Float6E2M3,
        ElementType::Float6E3M2,
    ];
    for ty in sub_byte {
        println!(
            "  {:<14}  tag=0x{:02X}  bits={}  sub_byte={}",
            ty,
            ty.tag(),
            ty.bit_width(),
            ty.is_sub_byte(),
        );
    }

    // ── Buffer size calculations ──────────────────────────────────────────────

    println!("\n=== Buffer sizes ===");

    // float32: 3×4×5 = 60 elements × 4 bytes = 240 bytes
    let shape = Shape::new(vec![3u64, 4, 5]).expect("valid shape");
    let n = shape.element_count().expect("no dynamic dimensions");
    println!(
        "  float32 {shape}: {n} elements → {} bytes",
        buffer_size_bytes(ElementType::Float32, n),
    );

    // int4: 7 elements → ceil(7/2) = 4 bytes
    println!(
        "  int4 [7]: {} bytes",
        buffer_size_bytes(ElementType::Int4, 7),
    );

    // bool: 9 elements → ceil(9/8) = 2 bytes
    println!(
        "  bool [9]: {} bytes",
        buffer_size_bytes(ElementType::Bool, 9),
    );

    // float6_e2m3: 5 elements → ceil(5/4)*3 = 6 bytes
    println!(
        "  float6_e2m3 [5]: {} bytes",
        buffer_size_bytes(ElementType::Float6E2M3, 5),
    );

    // ── Dynamic dimensions ────────────────────────────────────────────────────

    println!("\n=== Dynamic dimensions ===");
    let dyn_shape = Shape::new(vec![1u64, DYNAMIC, 768]).expect("valid shape");
    println!("  shape: {dyn_shape}");
    println!("  has_dynamic: {}", dyn_shape.has_dynamic());
    println!(
        "  element_count: {:?}  (None — unresolvable until dynamic dim is known)",
        dyn_shape.element_count()
    );

    // ── Empty tensor ──────────────────────────────────────────────────────────

    println!("\n=== Empty tensor ===");
    let empty = Shape::new(vec![3u64, 0, 5]).expect("valid shape");
    println!("  shape: {empty}");
    println!("  is_empty_tensor: {}", empty.is_empty_tensor());
    println!("  element_count: {:?}", empty.element_count());

    // ── Scalar tensor ─────────────────────────────────────────────────────────

    println!("\n=== Scalar tensor ===");
    let scalar = Shape::scalar();
    println!("  shape: {scalar}");
    println!("  rank: {}", scalar.rank());
    println!("  element_count: {:?}", scalar.element_count());

    // ── from_tag round-trip ───────────────────────────────────────────────────

    println!("\n=== Tag round-trip ===");
    let original = ElementType::Float8E4M3;
    let tag = original.tag();
    let recovered = ElementType::from_tag(tag).expect("known tag");
    println!("  {original} → tag 0x{tag:02X} → {recovered}");
    assert_eq!(original, recovered);

    println!("\n=== Tag validation ===");
    for bad_tag in [0x00u8, 0x47, 0x80, 0xFF] {
        println!(
            "  from_tag(0x{bad_tag:02X}) → {:?}",
            ElementType::from_tag(bad_tag),
        );
    }
}
