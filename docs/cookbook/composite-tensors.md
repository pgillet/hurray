# Composite Tensors

## Purpose

A **composite tensor** is a virtual descriptor (head) that owns no data, plus an ordered
set of member tensors, combined under a declared composition rule. This unifies three
previously distinct capabilities: partitioning an index space across heterogeneous
regions (sharding), overlaying a base tensor with scattered corrections (SpQR-style
outlier quantization), and grouping independent tensors under one logical identity.

The head presents one logical view — one `shape` and one `type_tag` — while the members
supply the actual data. Every member is an ordinary `TensorDescriptor` with its own
layout, buffers, quantization, and device placement. See
[`docs/spec/layouts/composite.md`](../spec/layouts/composite.md) for the full normative
specification (ADR-027).

## Quick reference

| Rule | Purpose | Base shape | Overlap | Corrections |
|------|---------|-----------|---------|-------------|
| Partition | Exact-cover tiling | Required | Not allowed | Not applicable |
| Overlay | Base + corrections | Whole space | Allowed | Yes, ordered |
| Group | Independent multi-output | Advisory | N/A | N/A |

## Partition: Zero-Copy Tiling

Members' shard boxes MUST exactly cover the head's index space with no gap and no
overlap. Logical index lookup is zero-copy: select the member whose box contains the
index, compute the local offset, apply that member's addressing.

Build a `[8, 8]` head split into two `[8, 4]` members:

```rust
use hurray_core::{
    composite::CompositeTensor,
    descriptor::TensorDescriptor,
    layout::{CompositeLayout, CompositionRule, LayoutDescriptor},
    BufferHandle, DeviceTag, ElementType, Shape, ShardDescriptor, SyncMode,
    MIN_BUFFER_ALIGNMENT,
};

// Head: float32 logical view, partition of 2 members.
let head_shape = Shape::new(vec![8u64, 8]).unwrap();
let head_layout = LayoutDescriptor::Composite(
    CompositeLayout::new(CompositionRule::Partition, 2).unwrap(),
);
let head = TensorDescriptor::new(
    1, 0, ElementType::Float32, head_shape, 0,
    head_layout, vec![], None, None, None, None,
).unwrap();

// Helper to build a member at a given shard offset.
let member = |offset: u64| {
    let buf = BufferHandle::new(128, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap();
    let shard = ShardDescriptor::new(vec![8, 8], vec![0, offset]).unwrap();
    TensorDescriptor::new(
        1, 0, ElementType::Float32, Shape::new(vec![8u64, 4]).unwrap(), 0,
        LayoutDescriptor::RowMajor, vec![buf],
        None, Some(shard), None, None,
    ).unwrap()
};

// Two [8, 4] members at columns 0..4 and 4..8 exactly tile the head.
let composite = CompositeTensor::new(head, vec![member(0), member(4)]).unwrap();
assert_eq!(composite.member_count(), 2);
```

Element `[3, 6]` (row 3, column 6) resolves to member 1 at local index `[3, 2]` (column
2 within that member's `[8, 4]` box). The read is zero-copy once the member is selected.

## Sealed Overlay: SpQR-Style Quantization

A base member spanning the whole index space plus corrections (outlier values) that
may overlap one another and the base. Precedence is stream order — later corrections
win. Two combine operations are supported:

- **Replace** (`0x01`): within a correction's box, the topmost member covering an index
  wins; outside every correction's box, the base shows through.
- **Add** (`0x02`): the value at an index is the base value plus the sum of all covering
  corrections' values at that index.

Example: `float16` logical view with `int4` per-block-affine quantized base and `float16`
COO sparse outlier correction:

```rust
use hurray_core::{
    composite::CompositeTensor,
    descriptor::{CompositeMemberDescriptor, MemberRole, TensorDescriptor},
    layout::{CombineOp, CompositeLayout, CompositionRule, CooLayout, LayoutDescriptor},
    BufferHandle, DeviceTag, ElementType, PerBlockAffine, QuantizationDescriptor,
    Shape, ShardDescriptor, SyncMode, MIN_BUFFER_ALIGNMENT, buffer_size_bytes,
};

let shape = Shape::new(vec![4096u64, 4096]).unwrap();

// Head: float16 logical view, overlay with replace combine, 2 members.
let head = TensorDescriptor::new(
    1, 0, ElementType::Float16, shape.clone(), 0,
    LayoutDescriptor::Composite(
        CompositeLayout::new(CompositionRule::Overlay(CombineOp::Replace), 2).unwrap()
    ),
    vec![], None, None, None, None,
).unwrap();

// Member 0 (base): int4 storage with per-block-affine quantization.
let pba = PerBlockAffine::new_symmetric(1, 128, 1, ElementType::Float16).unwrap();
let num_blocks = pba.num_blocks_per_axis(4096) * 4096;
let quant = QuantizationDescriptor::PerBlockAffine(pba);
let base_data = BufferHandle::new(
    buffer_size_bytes(ElementType::Int4, 4096 * 4096),
    MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced
).unwrap();
let base_scales = BufferHandle::new(
    buffer_size_bytes(ElementType::Float16, num_blocks),
    MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced
).unwrap();
let base_shard = ShardDescriptor::new(vec![4096, 4096], vec![0, 0]).unwrap();
let base = TensorDescriptor::new(
    1, 0, ElementType::Int4, shape.clone(), 0,
    LayoutDescriptor::RowMajor, vec![base_data, base_scales],
    Some(quant.encode_to_vec()),
    Some(base_shard),
    None, None,
).unwrap()
.with_composite_member(CompositeMemberDescriptor::new(MemberRole::Base));

// Member 1 (correction): float16 COO sparse outliers.
let nnz = 128u64;
let coo_values = BufferHandle::new(
    buffer_size_bytes(ElementType::Float16, nnz),
    MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced
).unwrap();
let coo_indices = BufferHandle::new(
    nnz * 2 /* rank */ * 8 /* uint64 */,
    MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced
).unwrap();
let correction_shard = ShardDescriptor::new(vec![4096, 4096], vec![0, 0]).unwrap();
let correction = TensorDescriptor::new(
    1, 0, ElementType::Float16, shape, 0,
    LayoutDescriptor::Coo(CooLayout::new(nnz, false)),
    vec![coo_values, coo_indices],
    None, Some(correction_shard),
    None, None,
).unwrap()
.with_composite_member(CompositeMemberDescriptor::new(MemberRole::Correction));

let composite = CompositeTensor::new(head, vec![base, correction]).unwrap();
assert_eq!(composite.member_count(), 2);
```

The merged logical view is: for each index, the correction's outlier value if present
(within its COO sparse structure), otherwise the dequantized base value. The consumer
computes this merge; it is not zero-copy at the composite level.

Contrast with `Add` combine for residual-correction overlays:

```rust
// Head with add combine instead of replace.
let head = TensorDescriptor::new(
    1, 0, ElementType::Float16, shape.clone(), 0,
    LayoutDescriptor::Composite(
        CompositeLayout::new(CompositionRule::Overlay(CombineOp::Add), 2).unwrap()
    ),
    vec![], None, None, None, None,
).unwrap();

// ... base and correction members as above ...
// The logical value is: base_value + correction_value (at indices
// where the correction is present; outside it, just the base).
```

## Group: Heterogeneous Multi-Output

Members are independent tensors under one head identity, with no spatial or ordering
semantics. Members MAY differ arbitrarily in rank, shape, element type, layout, and
device. Useful for weight collections, multi-head attention outputs, and other
use cases where multiple tensors are delivered together.

```rust
use hurray_core::{
    composite::CompositeTensor,
    descriptor::TensorDescriptor,
    layout::{CompositeLayout, CompositionRule, LayoutDescriptor},
    BufferHandle, DeviceTag, ElementType, Shape, SyncMode, MIN_BUFFER_ALIGNMENT,
};

// Head: shape and type_tag are advisory (members may differ).
let head = TensorDescriptor::new(
    1, 0, ElementType::Float32, Shape::new(vec![1u64]).unwrap(), 0,
    LayoutDescriptor::Composite(
        CompositeLayout::new(CompositionRule::Group, 2).unwrap()
    ),
    vec![], None, None, None, None,
).unwrap();

// Member 0: int8 vector, 100 elements.
let buf0 = BufferHandle::new(100, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap();
let member0 = TensorDescriptor::new(
    1, 0, ElementType::Int8, Shape::new(vec![100u64]).unwrap(), 0,
    LayoutDescriptor::RowMajor, vec![buf0],
    None, None, None, None,
).unwrap();

// Member 1: float64 3×3×3 tensor (completely different shape and type).
let buf1 = BufferHandle::new(216 * 8, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap();
let member1 = TensorDescriptor::new(
    1, 0, ElementType::Float64, Shape::new(vec![3u64, 3, 3]).unwrap(), 0,
    LayoutDescriptor::RowMajor, vec![buf1],
    None, None, None, None,
).unwrap();

// Both members grouped under one head.
let composite = CompositeTensor::new(head, vec![member0, member1]).unwrap();
assert_eq!(composite.member_count(), 2);
```

## Validation

[`CompositeTensor::new`] constructs a head + members set, driving a [`CompositeValidator`]
internally to perform per-member and close-time checks per the spec (§ Validation). Any
violation — shard coverage gap, overlap, type mismatch, missing/misplaced base in overlay,
member count mismatch — returns an error.

Example: a partition with a coverage gap is rejected:

```rust
use hurray_core::{
    composite::CompositeTensor,
    descriptor::TensorDescriptor,
    layout::{CompositeLayout, CompositionRule, LayoutDescriptor},
    BufferHandle, DeviceTag, ElementType, Error, Shape, ShardDescriptor, SyncMode,
    MIN_BUFFER_ALIGNMENT,
};

let head_shape = Shape::new(vec![8u64, 8]).unwrap();
let head = TensorDescriptor::new(
    1, 0, ElementType::Float32, head_shape, 0,
    LayoutDescriptor::Composite(
        CompositeLayout::new(CompositionRule::Partition, 2).unwrap()
    ),
    vec![], None, None, None, None,
).unwrap();

// Both members start at column 0: they overlap.
let member_at = |offset: u64| {
    let buf = BufferHandle::new(128, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap();
    let shard = ShardDescriptor::new(vec![8, 8], vec![0, offset]).unwrap();
    TensorDescriptor::new(
        1, 0, ElementType::Float32, Shape::new(vec![8u64, 4]).unwrap(), 0,
        LayoutDescriptor::RowMajor, vec![buf],
        None, Some(shard), None, None,
    ).unwrap()
};

let err = CompositeTensor::new(head, vec![member_at(0), member_at(0)]).unwrap_err();
assert!(matches!(err, Error::CompositePartitionOverlap { a: 0, b: 1 }));
```

The validator also returns
[`Error::CompositePartitionGap`] when members leave an uncovered region, and
[`Error::CompositeOverlayBaseNotFirst`] / [`Error::CompositeOverlayBaseNotSpanning`]
when overlay base rules are violated.

## Deferred features

This implementation covers v1.0 scope: partition, group, and sealed overlay only.
Reserved for a future ADR (versioned / open overlay and related work):

- **Versioned overlay** — an appendable, time-travel-capable overlay identified by an
  open-composite `member_count` sentinel (`0xFFFFFFFF`) and per-member version fields.
- **Cross-descriptor streaming and file binding** — Layers 5/6 will define how composites
  flow through IPC framing and file format sections; this pass is `hurray-core` type/codec/validator only.
- **Heterogeneous per-member device placement** — currently deferred; the spec reserves
  room for it.

See `docs/spec/layouts/composite.md` § Deferred for the full reserved wire-format list.

## Runnable example

```text
cargo run --example composite_tensors
```
