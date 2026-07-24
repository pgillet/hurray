//! Composite tensors: head + members, bound by forward stream adjacency.
//!
//! Demonstrates the composite / virtual layout (tag `0x0B`, ADR-027) — a head
//! descriptor that owns no data plus an ordered set of member tensor descriptors
//! that supply it. It builds the two worked examples from
//! `docs/spec/layouts/composite.md` § Example:
//!
//! - a **sealed overlay** (SpQR-style): an `int4` per-block-quantized base plus a
//!   `float16` COO outlier correction, both decoding to the head's `float16` view,
//! - a **partition**: two `[8, 4]` members exactly tiling a `[8, 8]` head with no
//!   gap and no overlap.
//!
//! It also shows [`CompositeValidator`] rejecting a malformed partition (a gap
//! left by an incorrect shard offset).
//!
//! Run with:
//!
//! ```text
//! cargo run --example composite_tensors
//! ```

use hurray_core::{
    buffer_size_bytes,
    composite::CompositeTensor,
    descriptor::{CompositeMemberDescriptor, MemberRole, TensorDescriptor},
    layout::{CombineOp, CompositeLayout, CompositionRule, CooLayout, LayoutDescriptor},
    BufferHandle, DeviceTag, ElementType, PerBlockAffine, QuantizationDescriptor, Shape,
    ShardDescriptor, SyncMode, MIN_BUFFER_ALIGNMENT,
};

fn buffer(byte_size: u64) -> BufferHandle {
    BufferHandle::new(
        byte_size,
        MIN_BUFFER_ALIGNMENT,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )
    .expect("valid buffer handle")
}

fn main() {
    println!("Composite Tensor Examples\n");
    sealed_overlay_example();
    println!();
    partition_example();
    println!();
    partition_gap_rejected_example();
}

// ── Sealed overlay (SpQR-style outlier quantization) ─────────────────────────

fn sealed_overlay_example() {
    println!("Sealed Overlay (SpQR-style int4 base + float16 outlier correction)");

    let shape = Shape::new(vec![4096u64, 4096]).unwrap();

    // Head: float16 logical view, overlay composed with replace semantics, 2 members.
    let head_layout = LayoutDescriptor::Composite(
        CompositeLayout::new(CompositionRule::Overlay(CombineOp::Replace), 2).unwrap(),
    );
    let head = TensorDescriptor::new(
        1,
        0,
        ElementType::Float16,
        shape.clone(),
        0,
        head_layout,
        vec![], // composite head owns no data (ADR-027 D-1)
        None,
        None,
        None,
        None,
    )
    .expect("valid composite head");
    println!(
        "  head:   shape={}  type=float16  layout=0x{:02X}  is_virtual={}",
        head.shape,
        head.layout.tag(),
        head.layout.is_virtual(),
    );

    // Member 0 (base): int4 storage, per-block-affine quantization along axis 1
    // (block_size = 128), decodes to float16 — the head's type_tag.
    let pba = PerBlockAffine::new_symmetric(1, 128, 1, ElementType::Float16)
        .expect("valid per-block-affine descriptor");
    let num_blocks = pba.num_blocks_per_axis(4096) * 4096; // blocks along axis 1 * rows
    let quant = QuantizationDescriptor::PerBlockAffine(pba);
    let base_data = buffer(buffer_size_bytes(ElementType::Int4, 4096 * 4096));
    let base_scales = buffer(buffer_size_bytes(ElementType::Float16, num_blocks));
    let base_shard = ShardDescriptor::new(vec![4096, 4096], vec![0, 0]).unwrap();
    let base = TensorDescriptor::new(
        1,
        0,
        ElementType::Int4,
        shape.clone(),
        0,
        LayoutDescriptor::RowMajor,
        vec![base_data, base_scales],
        Some(quant.encode_to_vec()),
        Some(base_shard),
        None,
        None,
    )
    .expect("valid base member")
    .with_composite_member(CompositeMemberDescriptor::new(MemberRole::Base));
    println!(
        "  member[0] (base):       type=int4 (per-block-affine -> float16)  shard_offset=[0, 0]"
    );

    // Member 1 (correction): float16 COO sparse outliers, scattered over the same
    // index space (shard spans it, but only `nnz` positions are actually stored).
    let nnz = 128u64;
    let coo_values = buffer(buffer_size_bytes(ElementType::Float16, nnz));
    let coo_indices = buffer(nnz * 2 /* rank */ * 8 /* uint64 */);
    let correction_shard = ShardDescriptor::new(vec![4096, 4096], vec![0, 0]).unwrap();
    let correction = TensorDescriptor::new(
        1,
        0,
        ElementType::Float16,
        shape,
        0,
        LayoutDescriptor::Coo(CooLayout::new(nnz, false)),
        vec![coo_values, coo_indices],
        None,
        Some(correction_shard),
        None,
        None,
    )
    .expect("valid correction member")
    .with_composite_member(CompositeMemberDescriptor::new(MemberRole::Correction));
    println!("  member[1] (correction): type=float16 (COO, nnz={nnz})  shard_offset=[0, 0]");

    let composite = CompositeTensor::new(head, vec![base, correction])
        .expect("overlay composite validates: base first + spans, correction follows");
    println!(
        "  -> CompositeTensor OK: rule={:?}  member_count={}",
        composite.rule(),
        composite.member_count(),
    );
}

// ── Partition (exact-cover, zero-copy tiling) ────────────────────────────────

fn partition_example() {
    println!("Partition ([8, 8] head split into two [8, 4] members)");

    let head_shape = Shape::new(vec![8u64, 8]).unwrap();
    let head_layout =
        LayoutDescriptor::Composite(CompositeLayout::new(CompositionRule::Partition, 2).unwrap());
    let head = TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        head_shape,
        0,
        head_layout,
        vec![],
        None,
        None,
        None,
        None,
    )
    .expect("valid composite head");

    let half = |offset: u64| {
        let data = buffer(buffer_size_bytes(ElementType::Float32, 8 * 4));
        let shard = ShardDescriptor::new(vec![8, 8], vec![0, offset]).unwrap();
        TensorDescriptor::new(
            1,
            0,
            ElementType::Float32,
            Shape::new(vec![8u64, 4]).unwrap(),
            0,
            LayoutDescriptor::RowMajor,
            vec![data],
            None,
            Some(shard),
            None,
            None,
        )
        .expect("valid partition member")
    };

    let composite = CompositeTensor::new(head, vec![half(0), half(4)])
        .expect("partition composite exactly covers [8, 8] with no overlap");
    println!(
        "  -> CompositeTensor OK: rule={:?}  member_count={}",
        composite.rule(),
        composite.member_count(),
    );
    println!("  element [3, 6] resolves to member 1 at local index [3, 2] (zero-copy)");
}

// ── A malformed partition is rejected ─────────────────────────────────────────

fn partition_gap_rejected_example() {
    println!("Malformed partition (shard offset leaves a gap) — rejected");

    let head_shape = Shape::new(vec![8u64, 8]).unwrap();
    let head_layout =
        LayoutDescriptor::Composite(CompositeLayout::new(CompositionRule::Partition, 2).unwrap());
    let head = TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        head_shape,
        0,
        head_layout,
        vec![],
        None,
        None,
        None,
        None,
    )
    .expect("valid composite head");

    let member_at = |offset: u64| {
        let data = buffer(buffer_size_bytes(ElementType::Float32, 8 * 4));
        let shard = ShardDescriptor::new(vec![8, 8], vec![0, offset]).unwrap();
        TensorDescriptor::new(
            1,
            0,
            ElementType::Float32,
            Shape::new(vec![8u64, 4]).unwrap(),
            0,
            LayoutDescriptor::RowMajor,
            vec![data],
            None,
            Some(shard),
            None,
            None,
        )
        .expect("valid partition member")
    };

    // Both members start at column 0 — they overlap entirely, and columns 4..8
    // are never covered (a gap). The overlap is what CompositeValidator reports
    // first (overlap is checked before coverage; see composite.rs docs).
    let err = CompositeTensor::new(head, vec![member_at(0), member_at(0)]).unwrap_err();
    println!("  -> rejected: {err}");
}
