//! CSF (Compressed Sparse Fiber): a rank-N sparse tensor.
//!
//! Demonstrates the CSF layout (tag `0x09`) — the rank-N generalization of CSR/CSC.
//! It shows:
//!
//! - building the descriptor (`nnz` + a `mode_order` permutation),
//! - the `2·rank+1` buffer count derived from the rank,
//! - resolving a logical coordinate to a leaf `values` index by descending the
//!   per-level `pos`/`crd` tree (binary search per level),
//! - structural zeros: a coordinate that is not stored returns `None`.
//!
//! Uses the worked example from `docs/spec/layouts/csf.md`.
//!
//! Run with:
//!
//! ```text
//! cargo run --example csf_sparse_tensor
//! ```

use hurray_core::layout::addressing::csf::{element_offset, validate_index_buffers};
use hurray_core::layout::{CsfLayout, LayoutDescriptor};
use hurray_core::Shape;

fn main() {
    println!("CSF Sparse Tensor Example\n");

    // Spec § Example: rank-3 tensor [2, 3, 4], identity mode order, 4 non-zeros:
    //   (0,0,1)=1.0  (0,2,3)=2.0  (1,1,0)=3.0  (1,1,2)=4.0
    let mode_order: Vec<u32> = vec![0, 1, 2];
    let layout = LayoutDescriptor::Csf(CsfLayout::new(4, mode_order.clone()));
    let shape = Shape::new(vec![2, 3, 4]).unwrap();

    println!(
        "layout: tag=0x{:02X}  buffer_count={:?}  valid_for_shape={}",
        layout.tag(),
        layout.buffer_count().map(|n| n.get()), // 2*3 + 1 = 7
        layout.validate_against_shape(&shape).is_ok(),
    );

    // The per-level pos/crd tree (buffers 1.. of the descriptor):
    //   level 0 (dim 0): pos_0=[0,2]      crd_0=[0,1]
    //   level 1 (dim 1): pos_1=[0,2,3]    crd_1=[0,2,1]
    //   level 2 (dim 2): pos_2=[0,1,2,4]  crd_2=[1,3,0,2]
    //   values = [1.0, 2.0, 3.0, 4.0]
    let pos_0: &[u64] = &[0, 2];
    let crd_0: &[u64] = &[0, 1];
    let pos_1: &[u64] = &[0, 2, 3];
    let crd_1: &[u64] = &[0, 2, 1];
    let pos_2: &[u64] = &[0, 1, 2, 4];
    let crd_2: &[u64] = &[1, 3, 0, 2];

    let pos_levels: Vec<&[u64]> = vec![pos_0, pos_1, pos_2];
    let crd_levels: Vec<&[u64]> = vec![crd_0, crd_1, crd_2];

    // Storage invariants (per-level pos monotonicity / terminals, crd sortedness + bounds).
    let shape_dims: &[u64] = &[2, 3, 4];
    validate_index_buffers(4, &mode_order, shape_dims, &pos_levels, &crd_levels)
        .expect("index buffers satisfy the storage invariants");
    println!("index buffers: valid\n");

    // Resolve some logical coordinates to leaf `values` indices.
    let values = [1.0_f32, 2.0, 3.0, 4.0];
    let queries = [[0u64, 0, 1], [0, 2, 3], [1, 1, 0], [1, 1, 2]];
    for q in queries {
        match element_offset(&q, &mode_order, &pos_levels, &crd_levels).unwrap() {
            Some(leaf) => println!(
                "  ({}, {}, {}) -> values[{leaf}] = {}",
                q[0], q[1], q[2], values[leaf as usize]
            ),
            None => println!("  ({}, {}, {}) -> structural zero", q[0], q[1], q[2]),
        }
    }

    // A coordinate that is not stored is a structural (implicit) zero.
    let absent = element_offset(&[0, 1, 0], &mode_order, &pos_levels, &crd_levels).unwrap();
    println!("\n  (0, 1, 0) not stored -> {absent:?} (structural zero)");

    println!("\nExample completed successfully.");
}
