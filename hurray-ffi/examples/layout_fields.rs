//! Reading a layout's payload fields from the C ABI (ADR-039).
//!
//! `hurray_descriptor_decode` validates every layout `hurray-core` supports, but
//! the accessors beside it stop at the layout *tag*. Before these three functions
//! existed, a consumer handed a CSR descriptor read `layout_tag == 0x07` and
//! nothing further — not `nnz`, not which buffer held `row_ptr` — so it could not
//! address a single element.
//!
//! A field is named by a `HURRAY_LAYOUT_FIELD_*` identifier and read through one
//! of three typed functions: one for scalars, one for unsigned arrays, one for
//! signed arrays. This walks a sparse descriptor, a strided one, and a nested
//! tiled one — the shapes a real consumer meets. Written in Rust only because
//! that is what this repository builds; the sequence is what a Go or Julia
//! binding on the C ABI would follow.
//!
//! Run with:
//!
//!     cargo run -p hurray-ffi --example layout_fields

use hurray_core::layout::{CsrLayout, StridedLayout, TiledLayout};
use hurray_core::{
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode, TensorDescriptor,
    MIN_BUFFER_ALIGNMENT,
};
use hurray_ffi::{
    descriptor::{hurray_descriptor_destroy, hurray_descriptor_layout_tag},
    field::{
        hurray_descriptor_field_i64_array, hurray_descriptor_field_u64,
        hurray_descriptor_field_u64_array, HurrayDescriptorField, HURRAY_LAYOUT_FIELD_CSR_NNZ,
        HURRAY_LAYOUT_FIELD_STRIDED_STRIDES, HURRAY_LAYOUT_FIELD_TILED_DEPTH,
        HURRAY_LAYOUT_FIELD_TILED_TILE_SHAPE,
    },
    hurray_descriptor_decode, HurrayDescriptor, HURRAY_ERR_BUFFER_TOO_SMALL, HURRAY_OK,
};

/// Encodes a descriptor so this example has something to decode.
///
/// A real consumer receives these bytes over a stream, a file, or a capsule; it
/// never builds them.
fn encode(layout: LayoutDescriptor, dims: Vec<u64>, buffer_count: usize) -> Vec<u8> {
    let buffers = (0..buffer_count)
        .map(|_| {
            BufferHandle::new(
                256,
                MIN_BUFFER_ALIGNMENT,
                DeviceTag::Cpu,
                SyncMode::ProducerSynced,
            )
            .expect("buffer handle")
        })
        .collect();

    TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(dims).expect("shape"),
        0,
        layout,
        buffers,
        None,
        None,
        None,
        None,
    )
    .expect("descriptor")
    .encode()
    .expect("encode")
}

fn decode(bytes: &[u8]) -> *mut HurrayDescriptor {
    let mut handle: *mut HurrayDescriptor = std::ptr::null_mut();
    // SAFETY: bytes is valid for its own length, and handle is a live local.
    let status = unsafe { hurray_descriptor_decode(bytes.as_ptr(), bytes.len(), &mut handle) };
    assert_eq!(status, HURRAY_OK, "decode failed: {status}");
    handle
}

/// Reads a signed array field the way a C caller must: ask for the length, then
/// allocate and read.
///
/// The two-step shape is not ceremony — the caller cannot know a field's length
/// in advance, and the ABI never allocates on the caller's behalf.
fn read_i64_array(handle: *mut HurrayDescriptor, field: HurrayDescriptorField) -> Vec<i64> {
    let mut len: usize = 0;

    // Pass a null buffer with capacity 0 to learn the true length.
    // SAFETY: handle is live; out_values may be null in capacity-query mode.
    let status = unsafe {
        hurray_descriptor_field_i64_array(handle, field, 0, std::ptr::null_mut(), &mut len)
    };
    assert_eq!(status, HURRAY_ERR_BUFFER_TOO_SMALL);

    let mut values = vec![0i64; len];
    // SAFETY: values is writable for len elements, which is what *len says.
    let status = unsafe {
        hurray_descriptor_field_i64_array(handle, field, 0, values.as_mut_ptr(), &mut len)
    };
    assert_eq!(status, HURRAY_OK);
    values
}

/// The unsigned counterpart, with a `level` for nested tiled layouts.
fn read_u64_array(
    handle: *mut HurrayDescriptor,
    field: HurrayDescriptorField,
    level: u32,
) -> Vec<u64> {
    let mut len: usize = 0;
    // SAFETY: handle is live; out_values may be null in capacity-query mode.
    let status = unsafe {
        hurray_descriptor_field_u64_array(handle, field, level, std::ptr::null_mut(), &mut len)
    };
    assert_eq!(status, HURRAY_ERR_BUFFER_TOO_SMALL);

    let mut values = vec![0u64; len];
    // SAFETY: values is writable for len elements.
    let status = unsafe {
        hurray_descriptor_field_u64_array(handle, field, level, values.as_mut_ptr(), &mut len)
    };
    assert_eq!(status, HURRAY_OK);
    values
}

fn scalar(handle: *mut HurrayDescriptor, field: HurrayDescriptorField, level: u32) -> u64 {
    let mut value: u64 = 0;
    // SAFETY: handle is live and value is a writable local.
    let status = unsafe { hurray_descriptor_field_u64(handle, field, level, &mut value) };
    assert_eq!(status, HURRAY_OK, "field {field:#x} failed: {status}");
    value
}

fn layout_tag(handle: *mut HurrayDescriptor) -> u8 {
    let mut tag: u8 = 0;
    // SAFETY: handle is live and tag is a writable local.
    assert_eq!(
        unsafe { hurray_descriptor_layout_tag(handle, &mut tag) },
        HURRAY_OK
    );
    tag
}

fn main() {
    // ── A sparse descriptor: the case that motivated this ─────────────────────
    //
    // 137 stored non-zeros in an 8×8 matrix. Without the nnz read, a consumer
    // knows the buffers exist but not how much of them is meaningful.
    let bytes = encode(LayoutDescriptor::Csr(CsrLayout::new(137)), vec![8, 8], 3);
    let csr = decode(&bytes);

    println!("CSR descriptor");
    println!("  layout tag : {:#04x}", layout_tag(csr));
    println!(
        "  nnz        : {}",
        scalar(csr, HURRAY_LAYOUT_FIELD_CSR_NNZ, 0)
    );

    // SAFETY: csr was created by hurray_descriptor_decode and is destroyed once.
    assert_eq!(unsafe { hurray_descriptor_destroy(csr) }, HURRAY_OK);

    // ── Strides, where the sign carries meaning ───────────────────────────────
    //
    // A negative stride reverses a dimension and a zero stride broadcasts one,
    // which is why strides are read through the signed accessor. An unsigned
    // read would report -4 as 18446744073709551612.
    let bytes = encode(
        LayoutDescriptor::Strided(StridedLayout::new(vec![12, -4, 0])),
        vec![2, 3, 4],
        1,
    );
    let strided = decode(&bytes);

    println!("\nStrided descriptor");
    println!("  layout tag : {:#04x}", layout_tag(strided));
    println!(
        "  strides    : {:?}",
        read_i64_array(strided, HURRAY_LAYOUT_FIELD_STRIDED_STRIDES)
    );

    // SAFETY: created by decode above, destroyed exactly once.
    assert_eq!(unsafe { hurray_descriptor_destroy(strided) }, HURRAY_OK);

    // ── A nested tiled layout, where `level` earns its place ──────────────────
    //
    // Tiles can nest. `level` selects which to read: 0 is the outermost, and
    // TILED_DEPTH reports how many remain from there down.
    let inner = TiledLayout::new(vec![2, 2], 0x01, 0x01, None, None, None).expect("inner tile");
    let outer = TiledLayout::new(
        vec![4, 4],
        0x01,
        0x04, // inner_layout must be tiled for a nested tile to be valid
        None,
        None,
        Some(Box::new(inner)),
    )
    .expect("outer tile");
    let bytes = encode(LayoutDescriptor::Tiled(Box::new(outer)), vec![8, 8], 1);
    let tiled = decode(&bytes);

    let depth = scalar(tiled, HURRAY_LAYOUT_FIELD_TILED_DEPTH, 0);
    println!("\nTiled descriptor");
    println!("  layout tag : {:#04x}", layout_tag(tiled));
    println!("  depth      : {depth}");
    for level in 0..depth as u32 {
        println!(
            "  level {level}    : tile shape {:?}",
            read_u64_array(tiled, HURRAY_LAYOUT_FIELD_TILED_TILE_SHAPE, level)
        );
    }

    // SAFETY: created by decode above, destroyed exactly once.
    assert_eq!(unsafe { hurray_descriptor_destroy(tiled) }, HURRAY_OK);

    println!("\nAll reads returned HURRAY_OK.");
}
