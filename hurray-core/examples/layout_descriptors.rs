//! Layout descriptor examples.
//!
//! Demonstrates constructing every named layout variant, inspecting tag and
//! buffer-count, and running `validate_against_shape`.
//!
//! Run with:
//!
//! ```text
//! cargo run --example layout_descriptors
//! ```

use hurray_core::{
    layout::{
        CooLayout, CscLayout, CsrLayout, HilbertLayout, LayoutDescriptor, MortonLayout,
        OuterStrides, PrivateExtensionLayout, RegionDescriptor, StridedLayout, SubpavingLayout,
        TiledLayout, UnknownLayout,
    },
    Shape,
};

fn main() {
    println!("Hurray Layout Descriptor Examples\n");

    // ── Row-major (0x01) ──────────────────────────────────────────────────────
    let rm = LayoutDescriptor::RowMajor;
    let shape_2d = Shape::new(vec![3, 4]).unwrap();
    println!(
        "RowMajor:   tag=0x{:02X}  buffers={:?}  valid={}",
        rm.tag(),
        rm.buffer_count().map(|n| n.get()),
        rm.validate_against_shape(&shape_2d).is_ok(),
    );

    // ── Column-major (0x02) ───────────────────────────────────────────────────
    let cm = LayoutDescriptor::ColMajor;
    println!(
        "ColMajor:   tag=0x{:02X}  buffers={:?}  valid={}",
        cm.tag(),
        cm.buffer_count().map(|n| n.get()),
        cm.validate_against_shape(&shape_2d).is_ok(),
    );

    // ── Strided (0x03) — row-major strides + reversed first dim ──────────────
    let strided = LayoutDescriptor::Strided(StridedLayout::new(vec![-4, 1]));
    println!(
        "Strided:    tag=0x{:02X}  buffers={:?}  valid={}  (negative stride reverses dim 0)",
        strided.tag(),
        strided.buffer_count().map(|n| n.get()),
        strided.validate_against_shape(&shape_2d).is_ok(),
    );

    // ── Tiled (0x04) — 2×2 tiles, row-major outer, col-major inner ───────────
    let tiled = LayoutDescriptor::Tiled(Box::new(
        TiledLayout::new(vec![2, 2], 0x01, 0x02, None, None, None).expect("valid tiled layout"),
    ));
    println!(
        "Tiled:      tag=0x{:02X}  buffers={:?}  valid={}  (2×2 tiles, RM outer / CM inner)",
        tiled.tag(),
        tiled.buffer_count().map(|n| n.get()),
        tiled.validate_against_shape(&shape_2d).is_ok(),
    );

    // ── Tiled with explicit outer strides (strided tile grid) ─────────────────
    let tiled_strided_outer = LayoutDescriptor::Tiled(Box::new(
        TiledLayout::new(
            vec![2, 2],
            0x03, // strided outer
            0x01, // row-major inner
            Some(OuterStrides::new(vec![2, 1])),
            None,
            None,
        )
        .expect("valid tiled with outer strides"),
    ));
    println!(
        "Tiled(str): tag=0x{:02X}  buffers={:?}  valid={}  (strided tile grid)",
        tiled_strided_outer.tag(),
        tiled_strided_outer.buffer_count().map(|n| n.get()),
        tiled_strided_outer
            .validate_against_shape(&shape_2d)
            .is_ok(),
    );

    // ── Morton / Z-order (0x05) ───────────────────────────────────────────────
    let morton_shape = Shape::new(vec![4, 4]).unwrap(); // 4 <= 2^2
    let morton = LayoutDescriptor::Morton(MortonLayout::new(vec![2, 2]).expect("valid morton"));
    println!(
        "Morton:     tag=0x{:02X}  buffers={:?}  valid={}  (4×4 tensor, 2 bits/dim)",
        morton.tag(),
        morton.buffer_count().map(|n| n.get()),
        morton.validate_against_shape(&morton_shape).is_ok(),
    );

    // ── General Subpaving (0x06) — four 4×4 quadrants of an 8×8 tensor ───────
    let sp_shape = Shape::new(vec![8, 8]).unwrap();
    let regions = vec![
        RegionDescriptor::new(vec![0, 0], vec![4, 4], 0x01, 0, 0).unwrap(),
        RegionDescriptor::new(vec![0, 4], vec![4, 4], 0x01, 0, 64).unwrap(),
        RegionDescriptor::new(vec![4, 0], vec![4, 4], 0x01, 0, 128).unwrap(),
        RegionDescriptor::new(vec![4, 4], vec![4, 4], 0x01, 0, 192).unwrap(),
    ];
    let subpaving =
        LayoutDescriptor::Subpaving(SubpavingLayout::new(regions).expect("valid subpaving"));
    println!(
        "Subpaving:  tag=0x{:02X}  buffers={:?}  valid={}  (four 4×4 row-major quadrants)",
        subpaving.tag(),
        subpaving.buffer_count().map(|n| n.get()),
        subpaving.validate_against_shape(&sp_shape).is_ok(),
    );

    // ── COO sparse (0x07) — buffer_count = 2 ─────────────────────────────────
    let coo_shape = Shape::new(vec![4, 4]).unwrap();
    let coo = LayoutDescriptor::Coo(CooLayout::new(3, true));
    println!(
        "COO:        tag=0x{:02X}  buffers={:?}  valid={}  (nnz=3, sorted)",
        coo.tag(),
        coo.buffer_count().map(|n| n.get()),
        coo.validate_against_shape(&coo_shape).is_ok(),
    );

    // ── CSR sparse (0x08) — buffer_count = 3 ─────────────────────────────────
    let csr_shape = Shape::new(vec![4, 5]).unwrap();
    let csr = LayoutDescriptor::Csr(CsrLayout::new(5));
    println!(
        "CSR:        tag=0x{:02X}  buffers={:?}  valid={}  (nnz=5, rank-2 only)",
        csr.tag(),
        csr.buffer_count().map(|n| n.get()),
        csr.validate_against_shape(&csr_shape).is_ok(),
    );

    // ── CSC sparse (0x09) — buffer_count = 3 ─────────────────────────────────
    let csc_shape = Shape::new(vec![4, 5]).unwrap();
    let csc = LayoutDescriptor::Csc(CscLayout::new(5));
    println!(
        "CSC:        tag=0x{:02X}  buffers={:?}  valid={}  (nnz=5, rank-2 only)",
        csc.tag(),
        csc.buffer_count().map(|n| n.get()),
        csc.validate_against_shape(&csc_shape).is_ok(),
    );

    // ── Hilbert curve (0x40) — each dim must be exactly 2^order ──────────────
    let hilbert_shape = Shape::new(vec![4, 4]).unwrap(); // 4 == 2^2
    let hilbert = LayoutDescriptor::Hilbert(HilbertLayout::new(2, 2).expect("valid hilbert"));
    println!(
        "Hilbert:    tag=0x{:02X}  buffers={:?}  valid={}  (order=2, rank=2, shape 4×4)",
        hilbert.tag(),
        hilbert.buffer_count().map(|n| n.get()),
        hilbert.validate_against_shape(&hilbert_shape).is_ok(),
    );

    // ── Private extension (0xF0–0xFE) ────────────────────────────────────────
    let private = LayoutDescriptor::PrivateExtension(
        PrivateExtensionLayout::new(0xF0, 0xDEAD_BEEF_0000_0001, vec![0x01, 0x02, 0x03])
            .expect("valid private extension"),
    );
    println!(
        "Private:    tag=0x{:02X}  buffers={:?}  (buffer count unknown for private layouts)",
        private.tag(),
        private.buffer_count().map(|n| n.get()),
    );

    // ── Unknown (permissive mode passthrough) ─────────────────────────────────
    let unknown = LayoutDescriptor::Unknown(UnknownLayout::new(0x0A, vec![0xAB, 0xCD]).unwrap());
    println!(
        "Unknown:    tag=0x{:02X}  buffers={:?}  valid={}  (permissive mode — never dereference data)",
        unknown.tag(),
        unknown.buffer_count().map(|n| n.get()),
        unknown.validate_against_shape(&shape_2d).is_ok(),
    );

    // ── Error cases ───────────────────────────────────────────────────────────
    println!("\nError cases:");

    // CSR on a rank-3 tensor is rejected.
    let csr_bad = LayoutDescriptor::Csr(CsrLayout::new(0));
    let rank3 = Shape::new(vec![2, 3, 4]).unwrap();
    println!(
        "  CSR on rank-3: {:?}",
        csr_bad.validate_against_shape(&rank3).unwrap_err()
    );

    // COO on a scalar (rank-0) tensor is rejected.
    let coo_bad = LayoutDescriptor::Coo(CooLayout::new(0, false));
    println!(
        "  COO on scalar: {:?}",
        coo_bad
            .validate_against_shape(&Shape::scalar())
            .unwrap_err()
    );

    // Strided with wrong number of strides.
    let strided_bad = LayoutDescriptor::Strided(StridedLayout::new(vec![1])); // rank-1 strides for rank-2 shape
    println!(
        "  Strided rank mismatch: {:?}",
        strided_bad.validate_against_shape(&shape_2d).unwrap_err()
    );

    // Hilbert with non-power-of-two dimension.
    let hilbert_bad = LayoutDescriptor::Hilbert(HilbertLayout::new(2, 2).unwrap());
    let bad_hilbert_shape = Shape::new(vec![3, 4]).unwrap(); // 3 != 2^2
    println!(
        "  Hilbert dim mismatch: {:?}",
        hilbert_bad
            .validate_against_shape(&bad_hilbert_shape)
            .unwrap_err()
    );

    println!("\nAll examples completed successfully.");
}
