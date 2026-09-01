//! Private extension element types: describing a type the format does not standardize.
//!
//! Tags `0xF0`–`0xFE` are reserved for types the spec deliberately says nothing about.
//! The price of that freedom is that the tensor must describe the type well enough for a
//! stranger to size its buffers: bit width and packing, carried in the descriptor's
//! extension type section. A reader that has never heard of the type still knows how many
//! bytes to move, which is what an interchange format has to guarantee.
//!
//! Run with:
//!
//! ```text
//! cargo run --example extension_types
//! ```

use hurray_core::{
    buffer_size_bytes,
    descriptor::{ExtensionTypeDescriptor, TensorDescriptor},
    layout::LayoutDescriptor,
    BufferHandle, DeviceTag, ElementType, Error, Shape, SyncMode, DESCRIPTOR_VERSION_MAJOR,
    DESCRIPTOR_VERSION_MINOR, MIN_BUFFER_ALIGNMENT,
};

fn main() -> Result<(), Error> {
    // ── The tag alone says almost nothing ─────────────────────────────────────

    println!("=== The element type alone says almost nothing ===");

    let private = ElementType::from_tag(0xF2)?;
    println!("  {private}");
    println!("  tag        0x{:02X}", private.tag());
    println!(
        "  bit_width  {}   <- 0: the width is not in the type",
        private.bit_width()
    );
    println!(
        "  buffer_size_bytes(.., 10) = {}   <- and so this cannot answer",
        buffer_size_bytes(private, 10)
    );

    // ── The section is where the width lives ──────────────────────────────────

    println!("\n=== The section is where the width lives ===");

    // A private 24-bit signed integer. packing_factor is 1 for every whole-byte width.
    let int24 = ExtensionTypeDescriptor::new(24, 1, false, true, 0, 0, 0, 0, false, false)?;
    println!("  24-bit signed integer");
    println!("    packing_factor     {}", int24.packing_factor);
    println!(
        "    buffer_size_bytes  {} for 4 elements",
        int24.buffer_size_bytes(4)
    );

    // Sub-byte widths pack, and only 1, 2 and 4 bits are legal — the packing factor has
    // to be a whole number of elements per byte. 6-bit types pack 4-per-3-bytes and are
    // therefore built-in, not private.
    let nibble = ExtensionTypeDescriptor::new(4, 2, false, false, 0, 0, 0, 0, false, false)?;
    println!("  4-bit unsigned integer");
    println!(
        "    packing_factor     {}   <- 8 / 4",
        nibble.packing_factor
    );
    println!(
        "    buffer_size_bytes  {} for 7 elements (ceil(7 / 2))",
        nibble.buffer_size_bytes(7)
    );

    let six_bit = ExtensionTypeDescriptor::new(6, 1, false, false, 0, 0, 0, 0, false, false);
    println!("  6-bit               {}", err_text(six_bit));

    // ── Floats, and where their sign lives ────────────────────────────────────

    println!("\n=== A float's sign is sign_bits, never is_signed ===");

    let half = ExtensionTypeDescriptor::new(16, 1, true, false, 1, 5, 10, 15, true, true)?;
    println!(
        "  signed float:    is_signed {}  sign_bits {}",
        half.is_signed, half.sign_bits
    );

    // `is_signed` describes integers only. That is not a claim that floats are unsigned —
    // it is what makes an *unsigned* float expressible, the shape of the built-in
    // exponent-only float8_e8m0 block scale.
    let scale = ExtensionTypeDescriptor::new(8, 1, true, false, 0, 8, 0, 127, true, false)?;
    println!(
        "  unsigned float:  is_signed {}  sign_bits {}",
        scale.is_signed, scale.sign_bits
    );

    let both = ExtensionTypeDescriptor::new(16, 1, true, true, 1, 5, 10, 15, true, false);
    println!("  both set:        {}", err_text(both));

    // ── A descriptor of it ────────────────────────────────────────────────────

    println!("\n=== A descriptor carrying one ===");

    let shape = Shape::new(vec![4u64])?;
    let byte_len = int24.buffer_size_bytes(4);
    let buffer = BufferHandle::new(
        byte_len,
        MIN_BUFFER_ALIGNMENT,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )?;
    let descriptor = TensorDescriptor::new(
        DESCRIPTOR_VERSION_MAJOR,
        DESCRIPTOR_VERSION_MINOR,
        private,
        shape,
        0,
        LayoutDescriptor::RowMajor,
        vec![buffer],
        None,
        None,
        None,
        Some(int24.clone()),
    )?;
    println!("  buffer      {byte_len} bytes for 4 elements");
    println!("  encoded     {} bytes", descriptor.encode()?.len());

    // The flag and the tag are one fact, so the descriptor refuses to state half of it.
    let orphan = TensorDescriptor::new(
        DESCRIPTOR_VERSION_MAJOR,
        DESCRIPTOR_VERSION_MINOR,
        private,
        Shape::new(vec![4u64])?,
        0,
        LayoutDescriptor::RowMajor,
        vec![BufferHandle::new(
            byte_len,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )?],
        None,
        None,
        None,
        None,
    );
    println!("  no section  {}", err_text(orphan));

    // ── On the wire ───────────────────────────────────────────────────────────

    println!("\n=== It survives the round trip ===");

    let wire = descriptor.encode()?;
    let restored = TensorDescriptor::decode(&wire)?;
    let ext = restored
        .extension_type
        .as_ref()
        .expect("the descriptor was built with one");

    println!("  bit_width   {}", ext.bit_width);
    println!("  is_signed   {}", ext.is_signed);
    println!(
        "  a consumer that has never heard of type 0x{:02X} still knows it needs {} bytes",
        private.tag(),
        ext.buffer_size_bytes(4)
    );

    assert_eq!(restored, descriptor);
    assert_eq!(ext.buffer_size_bytes(4), byte_len);

    Ok(())
}

/// Renders the error a rejected construction produced, for printing.
fn err_text<T>(result: Result<T, Error>) -> String {
    match result {
        Ok(_) => "accepted".to_string(),
        Err(e) => format!("rejected: {e}"),
    }
}
