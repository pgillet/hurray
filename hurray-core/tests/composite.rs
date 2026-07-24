//! Integration tests for composite / virtual tensors (ADR-027).
//!
//! These tests validate the public API contract of [`CompositeTensor`] /
//! [`CompositeValidator`] against `docs/spec/layouts/composite.md`, driving
//! everything through the public constructors (`TensorDescriptor::new`,
//! `.with_composite_member`, `TensorDescriptor::encode`/`decode`) — no crate
//! internals are accessed.
//!
//! No private items are accessed; everything is exercised through the public API.

use hurray_core::{
    buffer_size_bytes,
    composite::CompositeTensor,
    descriptor::{CompositeMemberDescriptor, MemberRole, TensorDescriptor},
    layout::{CombineOp, CompositeLayout, CompositionRule, CooLayout, LayoutDescriptor},
    BufferHandle, DeviceTag, ElementType, Error, PerBlockAffine, QuantizationDescriptor, Shape,
    ShardDescriptor, SyncMode, MIN_BUFFER_ALIGNMENT,
};

// ── Helpers ──────────────────────────────────────────────────────────────────

fn buf(byte_size: u64) -> BufferHandle {
    BufferHandle::new(
        byte_size,
        MIN_BUFFER_ALIGNMENT,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )
    .expect("valid buffer handle")
}

fn shard(parent: Vec<u64>, offset: Vec<u64>) -> ShardDescriptor {
    ShardDescriptor::new(parent, offset).expect("valid shard descriptor")
}

fn composite_head(
    rule: CompositionRule,
    member_count: u32,
    shape: Vec<u64>,
    ty: ElementType,
) -> TensorDescriptor {
    let layout = LayoutDescriptor::Composite(CompositeLayout::new(rule, member_count).unwrap());
    TensorDescriptor::new(
        1,
        0,
        ty,
        Shape::new(shape).unwrap(),
        0,
        layout,
        vec![], // composite head owns no data (ADR-027 D-1)
        None,
        None,
        None,
        None,
    )
    .expect("valid composite head")
}

/// Round-trips a [`TensorDescriptor`] through `encode()`/`decode()`, exactly as
/// a real reader/writer would when streaming a composite's head + members.
fn round_trip(desc: &TensorDescriptor) -> TensorDescriptor {
    let bytes = desc.encode().expect("encode should succeed");
    TensorDescriptor::decode(&bytes).expect("decode should succeed")
}

// ── Full round-trip: sealed overlay (SpQR-style) ──────────────────────────────

/// Byte-for-byte behavioral match of `docs/spec/layouts/composite.md` § Example
/// (sealed overlay) and `hurray-core/examples/composite_tensors.rs`: a
/// `float16` [4096, 4096] head, an `int4` per-block-affine base decoding to
/// `float16`, and a `float16` COO outlier correction.
#[test]
fn sealed_overlay_round_trips_through_encode_decode_and_validates() {
    let shape = Shape::new(vec![4096u64, 4096]).unwrap();

    let head = composite_head(
        CompositionRule::Overlay(CombineOp::Replace),
        2,
        vec![4096, 4096],
        ElementType::Float16,
    );

    let pba = PerBlockAffine::new_symmetric(1, 128, 1, ElementType::Float16).unwrap();
    let num_blocks = pba.num_blocks_per_axis(4096) * 4096;
    let quant = QuantizationDescriptor::PerBlockAffine(pba);
    let base = TensorDescriptor::new(
        1,
        0,
        ElementType::Int4,
        shape.clone(),
        0,
        LayoutDescriptor::RowMajor,
        vec![
            buf(buffer_size_bytes(ElementType::Int4, 4096 * 4096)),
            buf(buffer_size_bytes(ElementType::Float16, num_blocks)),
        ],
        Some(quant.encode_to_vec()),
        Some(shard(vec![4096, 4096], vec![0, 0])),
        None,
        None,
    )
    .unwrap()
    .with_composite_member(CompositeMemberDescriptor::new(MemberRole::Base));

    let nnz = 128u64;
    let correction = TensorDescriptor::new(
        1,
        0,
        ElementType::Float16,
        shape,
        0,
        LayoutDescriptor::Coo(CooLayout::new(nnz, false)),
        vec![
            buf(buffer_size_bytes(ElementType::Float16, nnz)),
            buf(nnz * 2 * 8),
        ],
        None,
        Some(shard(vec![4096, 4096], vec![0, 0])),
        None,
        None,
    )
    .unwrap()
    .with_composite_member(CompositeMemberDescriptor::new(MemberRole::Correction));

    // Round-trip the head and every member through the wire format, exactly as
    // a streaming reader/writer would (head precedes its members, per spec).
    let head_rt = round_trip(&head);
    let base_rt = round_trip(&base);
    let correction_rt = round_trip(&correction);

    assert_eq!(head_rt, head);
    assert_eq!(base_rt, base);
    assert_eq!(correction_rt, correction);

    let composite = CompositeTensor::new(head_rt, vec![base_rt, correction_rt])
        .expect("overlay composite validates: base first + spans, correction follows");
    assert_eq!(composite.member_count(), 2);
    assert_eq!(
        *composite.rule(),
        CompositionRule::Overlay(CombineOp::Replace)
    );
}

// ── Full round-trip: partition ────────────────────────────────────────────────

/// Byte-for-byte behavioral match of `docs/spec/layouts/composite.md` §
/// Example (partition): a `float32` [8, 8] head split into two [8, 4] members
/// exactly covering the index space with no overlap.
#[test]
fn partition_round_trips_through_encode_decode_and_validates() {
    let head = composite_head(
        CompositionRule::Partition,
        2,
        vec![8, 8],
        ElementType::Float32,
    );

    let member = |offset: u64| {
        TensorDescriptor::new(
            1,
            0,
            ElementType::Float32,
            Shape::new(vec![8u64, 4]).unwrap(),
            0,
            LayoutDescriptor::RowMajor,
            vec![buf(buffer_size_bytes(ElementType::Float32, 8 * 4))],
            None,
            Some(shard(vec![8, 8], vec![0, offset])),
            None,
            None,
        )
        .unwrap()
    };

    let head_rt = round_trip(&head);
    let m0_rt = round_trip(&member(0));
    let m1_rt = round_trip(&member(4));

    let composite = CompositeTensor::new(head_rt, vec![m0_rt, m1_rt])
        .expect("partition composite exactly covers [8, 8] with no overlap");
    assert_eq!(composite.member_count(), 2);
    assert_eq!(*composite.rule(), CompositionRule::Partition);
}

// ── Torn composite ────────────────────────────────────────────────────────────

/// A head declaring `member_count = 3` but only 2 members supplied is a torn
/// composite — `CompositeMemberCountMismatch`.
#[test]
fn torn_composite_member_count_mismatch() {
    let head = composite_head(
        CompositionRule::Partition,
        3,
        vec![8, 8],
        ElementType::Float32,
    );
    let member = |offset: u64| {
        TensorDescriptor::new(
            1,
            0,
            ElementType::Float32,
            Shape::new(vec![8u64, 4]).unwrap(),
            0,
            LayoutDescriptor::RowMajor,
            vec![buf(128)],
            None,
            Some(shard(vec![8, 8], vec![0, offset])),
            None,
            None,
        )
        .unwrap()
    };

    let err = CompositeTensor::new(head, vec![member(0), member(4)]).unwrap_err();
    assert!(matches!(
        err,
        Error::CompositeMemberCountMismatch {
            declared: 3,
            actual: 2,
        }
    ));
}

// ── Per-rule negative paths ───────────────────────────────────────────────────

/// Partition: a gap in the head's index space is rejected.
#[test]
fn partition_gap_rejected() {
    let head = composite_head(
        CompositionRule::Partition,
        2,
        vec![8, 8],
        ElementType::Float32,
    );
    // Columns [0,3) and [3,6) covered — columns [6,8) are never covered.
    let m0 = TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(vec![8u64, 3]).unwrap(),
        0,
        LayoutDescriptor::RowMajor,
        vec![buf(96)],
        None,
        Some(shard(vec![8, 8], vec![0, 0])),
        None,
        None,
    )
    .unwrap();
    let m1 = TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(vec![8u64, 3]).unwrap(),
        0,
        LayoutDescriptor::RowMajor,
        vec![buf(96)],
        None,
        Some(shard(vec![8, 8], vec![0, 3])),
        None,
        None,
    )
    .unwrap();
    let err = CompositeTensor::new(head, vec![m0, m1]).unwrap_err();
    assert!(matches!(err, Error::CompositePartitionGap));
}

/// Overlay: the base member MUST be first.
#[test]
fn overlay_base_not_first_rejected() {
    let head = composite_head(
        CompositionRule::Overlay(CombineOp::Replace),
        1,
        vec![4, 4],
        ElementType::Float32,
    );
    let correction = TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(vec![4u64, 4]).unwrap(),
        0,
        LayoutDescriptor::RowMajor,
        vec![buf(64)],
        None,
        Some(shard(vec![4, 4], vec![0, 0])),
        None,
        None,
    )
    .unwrap()
    .with_composite_member(CompositeMemberDescriptor::new(MemberRole::Correction));

    let err = CompositeTensor::new(head, vec![correction]).unwrap_err();
    assert!(matches!(err, Error::CompositeOverlayBaseNotFirst));
}

/// Group: no coverage/type check applies, so the only negative path shared
/// with the other rules is a `member_count` mismatch.
#[test]
fn group_member_count_mismatch_rejected() {
    let head = composite_head(CompositionRule::Group, 2, vec![1], ElementType::Float32);
    let m0 = TensorDescriptor::new(
        1,
        0,
        ElementType::Int8,
        Shape::new(vec![10u64]).unwrap(),
        0,
        LayoutDescriptor::RowMajor,
        vec![buf(16)],
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let err = CompositeTensor::new(head, vec![m0]).unwrap_err();
    assert!(matches!(
        err,
        Error::CompositeMemberCountMismatch {
            declared: 2,
            actual: 1,
        }
    ));
}

/// Group members accepted end-to-end through the public API: heterogeneous
/// shapes/types, no shard sections, no coverage/type check.
#[test]
fn group_heterogeneous_members_accepted() {
    let head = composite_head(CompositionRule::Group, 2, vec![1], ElementType::Float32);
    let m0 = TensorDescriptor::new(
        1,
        0,
        ElementType::Int8,
        Shape::new(vec![10u64]).unwrap(),
        0,
        LayoutDescriptor::RowMajor,
        vec![buf(16)],
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let m1 = TensorDescriptor::new(
        1,
        0,
        ElementType::Float64,
        Shape::new(vec![2u64, 2]).unwrap(),
        0,
        LayoutDescriptor::RowMajor,
        vec![buf(32)],
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let composite = CompositeTensor::new(head, vec![m0, m1]).unwrap();
    assert_eq!(composite.member_count(), 2);
}
