//! Integration tests for the layout-payload field accessors (ADR-039).
//!
//! Every test drives the public C ABI: encode a descriptor with `hurray-core`,
//! decode it through `hurray_descriptor_decode`, then read fields back the way a
//! C consumer would.

use hurray_core::layout::{
    BlockPagedLayout, BlockTableIndexType, CombineOp, CompositeLayout, CompositionRule, CooLayout,
    CscLayout, CsfLayout, CsrLayout, HilbertLayout, InnerStrides, KvRole, MortonLayout,
    OuterStrides, StridedLayout, TiledLayout,
};
use hurray_core::{
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode, TensorDescriptor,
    MIN_BUFFER_ALIGNMENT,
};
use hurray_ffi::{
    descriptor::hurray_descriptor_destroy, field::*, hurray_descriptor_decode, HurrayDescriptor,
    HURRAY_ERR_BUFFER_TOO_SMALL, HURRAY_ERR_FIELD_NOT_APPLICABLE, HURRAY_ERR_FIELD_NOT_PRESENT,
    HURRAY_ERR_INDEX_OUT_OF_BOUNDS, HURRAY_ERR_NULL_POINTER, HURRAY_OK,
};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn buffers(n: usize) -> Vec<BufferHandle> {
    (0..n)
        .map(|_| {
            BufferHandle::new(
                256,
                MIN_BUFFER_ALIGNMENT,
                DeviceTag::Cpu,
                SyncMode::ProducerSynced,
            )
            .expect("buffer handle")
        })
        .collect()
}

/// Encodes a descriptor with the given layout, shape and buffer count.
fn encode(layout: LayoutDescriptor, dims: Vec<u64>, buffer_count: usize) -> Vec<u8> {
    let desc = TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(dims).expect("shape"),
        0,
        layout,
        buffers(buffer_count),
        None,
        None,
        None,
        None,
    )
    .expect("descriptor");
    desc.encode().expect("encode")
}

/// Decodes bytes into a handle. The caller must destroy it.
fn decode(bytes: &[u8]) -> *mut HurrayDescriptor {
    let mut handle: *mut HurrayDescriptor = std::ptr::null_mut();
    let status = unsafe { hurray_descriptor_decode(bytes.as_ptr(), bytes.len(), &mut handle) };
    assert_eq!(status, HURRAY_OK, "decode failed with status {status}");
    handle
}

/// Reads a scalar field, asserting success.
fn scalar(handle: *mut HurrayDescriptor, field: HurrayDescriptorField) -> u64 {
    scalar_at(handle, field, 0)
}

fn scalar_at(handle: *mut HurrayDescriptor, field: HurrayDescriptorField, level: u32) -> u64 {
    let mut value: u64 = 0;
    let status = unsafe { hurray_descriptor_field_u64(handle, field, level, &mut value) };
    assert_eq!(status, HURRAY_OK, "field {field:#x} failed with {status}");
    value
}

/// Reads an unsigned array field with a two-step capacity query, as a C caller would.
fn u64_array(handle: *mut HurrayDescriptor, field: HurrayDescriptorField, level: u32) -> Vec<u64> {
    let mut len: usize = 0;
    let status = unsafe {
        hurray_descriptor_field_u64_array(handle, field, level, std::ptr::null_mut(), &mut len)
    };
    assert_eq!(status, HURRAY_ERR_BUFFER_TOO_SMALL);

    let mut out = vec![0u64; len];
    let status = unsafe {
        hurray_descriptor_field_u64_array(handle, field, level, out.as_mut_ptr(), &mut len)
    };
    assert_eq!(status, HURRAY_OK);
    assert_eq!(len, out.len());
    out
}

/// Reads a signed array field with a two-step capacity query.
fn i64_array(handle: *mut HurrayDescriptor, field: HurrayDescriptorField, level: u32) -> Vec<i64> {
    let mut len: usize = 0;
    let status = unsafe {
        hurray_descriptor_field_i64_array(handle, field, level, std::ptr::null_mut(), &mut len)
    };
    assert_eq!(status, HURRAY_ERR_BUFFER_TOO_SMALL);

    let mut out = vec![0i64; len];
    let status = unsafe {
        hurray_descriptor_field_i64_array(handle, field, level, out.as_mut_ptr(), &mut len)
    };
    assert_eq!(status, HURRAY_OK);
    assert_eq!(len, out.len());
    out
}

fn destroy(handle: *mut HurrayDescriptor) {
    assert_eq!(unsafe { hurray_descriptor_destroy(handle) }, HURRAY_OK);
}

// ── Sparse layouts ────────────────────────────────────────────────────────────

#[test]
fn reads_csr_nnz() {
    let bytes = encode(LayoutDescriptor::Csr(CsrLayout::new(137)), vec![8, 8], 3);
    let h = decode(&bytes);
    assert_eq!(scalar(h, HURRAY_LAYOUT_FIELD_CSR_NNZ), 137);
    destroy(h);
}

#[test]
fn reads_csc_nnz() {
    let bytes = encode(LayoutDescriptor::Csc(CscLayout::new(42)), vec![8, 8], 3);
    let h = decode(&bytes);
    assert_eq!(scalar(h, HURRAY_LAYOUT_FIELD_CSC_NNZ), 42);
    destroy(h);
}

#[test]
fn reads_coo_nnz_and_sorted_flag() {
    let bytes = encode(
        LayoutDescriptor::Coo(CooLayout::new(9, true)),
        vec![4, 4],
        2,
    );
    let h = decode(&bytes);
    assert_eq!(scalar(h, HURRAY_LAYOUT_FIELD_COO_NNZ), 9);
    assert_eq!(scalar(h, HURRAY_LAYOUT_FIELD_COO_IS_SORTED), 1);
    destroy(h);
}

#[test]
fn coo_unsorted_flag_reads_zero() {
    let bytes = encode(
        LayoutDescriptor::Coo(CooLayout::new(9, false)),
        vec![4, 4],
        2,
    );
    let h = decode(&bytes);
    assert_eq!(scalar(h, HURRAY_LAYOUT_FIELD_COO_IS_SORTED), 0);
    destroy(h);
}

#[test]
fn reads_csf_nnz_and_mode_order() {
    let layout = LayoutDescriptor::Csf(CsfLayout::new(31, vec![2, 0, 1]));
    // CSF buffer count is 2·rank + 1.
    let bytes = encode(layout, vec![3, 4, 5], 7);
    let h = decode(&bytes);
    assert_eq!(scalar(h, HURRAY_LAYOUT_FIELD_CSF_NNZ), 31);
    assert_eq!(
        u64_array(h, HURRAY_LAYOUT_FIELD_CSF_MODE_ORDER, 0),
        vec![2, 0, 1]
    );
    destroy(h);
}

// ── Dense layouts with variable-length payloads ───────────────────────────────

#[test]
fn reads_strides_including_negative_and_zero() {
    // Negative (reversed) and zero (broadcast) strides are both valid, which is
    // why the accessor is signed.
    let layout = LayoutDescriptor::Strided(StridedLayout::new(vec![12, -4, 0]));
    let bytes = encode(layout, vec![2, 3, 4], 1);
    let h = decode(&bytes);
    assert_eq!(
        i64_array(h, HURRAY_LAYOUT_FIELD_STRIDED_STRIDES, 0),
        vec![12, -4, 0]
    );
    destroy(h);
}

#[test]
fn reads_morton_bits() {
    let layout = LayoutDescriptor::Morton(MortonLayout::new(vec![3, 3]).expect("morton"));
    let bytes = encode(layout, vec![8, 8], 1);
    let h = decode(&bytes);
    assert_eq!(u64_array(h, HURRAY_LAYOUT_FIELD_MORTON_BITS, 0), vec![3, 3]);
    destroy(h);
}

#[test]
fn reads_hilbert_order_and_rank() {
    let layout = LayoutDescriptor::Hilbert(HilbertLayout::new(3, 2).expect("hilbert"));
    let bytes = encode(layout, vec![8, 8], 1);
    let h = decode(&bytes);
    assert_eq!(scalar(h, HURRAY_LAYOUT_FIELD_HILBERT_ORDER), 3);
    assert_eq!(scalar(h, HURRAY_LAYOUT_FIELD_HILBERT_RANK), 2);
    destroy(h);
}

// ── Tiled, including nesting ──────────────────────────────────────────────────

fn flat_tiled() -> LayoutDescriptor {
    LayoutDescriptor::Tiled(Box::new(
        TiledLayout::new(vec![4, 4], 0x01, 0x01, None, None, None).expect("tiled"),
    ))
}

#[test]
fn reads_tiled_scalars_and_tile_shape() {
    let bytes = encode(flat_tiled(), vec![8, 8], 1);
    let h = decode(&bytes);
    assert_eq!(scalar(h, HURRAY_LAYOUT_FIELD_TILED_OUTER_LAYOUT), 0x01);
    assert_eq!(scalar(h, HURRAY_LAYOUT_FIELD_TILED_INNER_LAYOUT), 0x01);
    assert_eq!(scalar(h, HURRAY_LAYOUT_FIELD_TILED_DEPTH), 1);
    assert_eq!(
        u64_array(h, HURRAY_LAYOUT_FIELD_TILED_TILE_SHAPE, 0),
        vec![4, 4]
    );
    destroy(h);
}

#[test]
fn reads_tiled_explicit_strides() {
    let layout = LayoutDescriptor::Tiled(Box::new(
        TiledLayout::new(
            vec![4, 4],
            // Explicit strides are only valid when the corresponding layout is
            // strided (0x03).
            0x03,
            0x03,
            Some(OuterStrides::new(vec![32, 4])),
            Some(InnerStrides::new(vec![4, 1])),
            None,
        )
        .expect("tiled"),
    ));
    let bytes = encode(layout, vec![8, 8], 1);
    let h = decode(&bytes);
    assert_eq!(
        i64_array(h, HURRAY_LAYOUT_FIELD_TILED_OUTER_STRIDES, 0),
        vec![32, 4]
    );
    assert_eq!(
        i64_array(h, HURRAY_LAYOUT_FIELD_TILED_INNER_STRIDES, 0),
        vec![4, 1]
    );
    destroy(h);
}

#[test]
fn absent_tiled_strides_report_not_present() {
    let bytes = encode(flat_tiled(), vec![8, 8], 1);
    let h = decode(&bytes);

    let mut len: usize = 0;
    let status = unsafe {
        hurray_descriptor_field_i64_array(
            h,
            HURRAY_LAYOUT_FIELD_TILED_OUTER_STRIDES,
            0,
            std::ptr::null_mut(),
            &mut len,
        )
    };
    // Not BUFFER_TOO_SMALL: the field is absent, so there is no length to report.
    assert_eq!(status, HURRAY_ERR_FIELD_NOT_PRESENT);
    destroy(h);
}

#[test]
fn reads_nested_tiled_levels() {
    let inner = TiledLayout::new(vec![2, 2], 0x01, 0x01, None, None, None).expect("inner");
    // A nested tile requires inner_layout to be tiled (0x04).
    let outer =
        TiledLayout::new(vec![4, 4], 0x01, 0x04, None, None, Some(Box::new(inner))).expect("outer");
    let bytes = encode(LayoutDescriptor::Tiled(Box::new(outer)), vec![8, 8], 1);
    let h = decode(&bytes);

    assert_eq!(scalar_at(h, HURRAY_LAYOUT_FIELD_TILED_DEPTH, 0), 2);
    assert_eq!(scalar_at(h, HURRAY_LAYOUT_FIELD_TILED_DEPTH, 1), 1);
    assert_eq!(
        u64_array(h, HURRAY_LAYOUT_FIELD_TILED_TILE_SHAPE, 0),
        vec![4, 4]
    );
    assert_eq!(
        u64_array(h, HURRAY_LAYOUT_FIELD_TILED_TILE_SHAPE, 1),
        vec![2, 2]
    );
    destroy(h);
}

#[test]
fn level_past_nesting_depth_is_out_of_bounds() {
    let bytes = encode(flat_tiled(), vec![8, 8], 1);
    let h = decode(&bytes);
    let mut value: u64 = 0;
    let status =
        unsafe { hurray_descriptor_field_u64(h, HURRAY_LAYOUT_FIELD_TILED_DEPTH, 1, &mut value) };
    assert_eq!(status, HURRAY_ERR_INDEX_OUT_OF_BOUNDS);
    destroy(h);
}

#[test]
fn nonzero_level_on_a_layout_that_does_not_nest_is_rejected() {
    let bytes = encode(LayoutDescriptor::Csr(CsrLayout::new(5)), vec![4, 4], 3);
    let h = decode(&bytes);
    let mut value: u64 = 0;
    let status =
        unsafe { hurray_descriptor_field_u64(h, HURRAY_LAYOUT_FIELD_CSR_NNZ, 1, &mut value) };
    assert_eq!(status, HURRAY_ERR_INDEX_OUT_OF_BOUNDS);
    destroy(h);
}

// ── Block-paged ───────────────────────────────────────────────────────────────

#[test]
fn reads_every_block_paged_scalar() {
    let layout = LayoutDescriptor::BlockPaged(BlockPagedLayout::new(
        16,
        1024,
        0,
        8,
        KvRole::Value,
        Some(7),
        BlockTableIndexType::U64,
    ));
    let bytes = encode(layout, vec![64, 8, 128], 3);
    let h = decode(&bytes);

    assert_eq!(scalar(h, HURRAY_LAYOUT_FIELD_BLOCK_PAGED_PAGE_SIZE), 16);
    assert_eq!(scalar(h, HURRAY_LAYOUT_FIELD_BLOCK_PAGED_NUM_PAGES), 1024);
    assert_eq!(scalar(h, HURRAY_LAYOUT_FIELD_BLOCK_PAGED_PAGED_AXIS), 0);
    assert_eq!(scalar(h, HURRAY_LAYOUT_FIELD_BLOCK_PAGED_NUM_SEQS), 8);
    assert_eq!(
        scalar(h, HURRAY_LAYOUT_FIELD_BLOCK_PAGED_KV_ROLE),
        KvRole::Value.wire_byte() as u64
    );
    assert_eq!(scalar(h, HURRAY_LAYOUT_FIELD_BLOCK_PAGED_LAYER_INDEX), 7);
    assert_eq!(
        scalar(h, HURRAY_LAYOUT_FIELD_BLOCK_PAGED_INDEX_TYPE),
        BlockTableIndexType::U64.wire_byte() as u64
    );
    destroy(h);
}

#[test]
fn absent_layer_index_reports_not_present() {
    let layout = LayoutDescriptor::BlockPaged(BlockPagedLayout::new(
        16,
        1024,
        0,
        8,
        KvRole::Key,
        None,
        BlockTableIndexType::U32,
    ));
    let bytes = encode(layout, vec![64, 8, 128], 3);
    let h = decode(&bytes);

    let mut value: u64 = 0;
    let status = unsafe {
        hurray_descriptor_field_u64(
            h,
            HURRAY_LAYOUT_FIELD_BLOCK_PAGED_LAYER_INDEX,
            0,
            &mut value,
        )
    };
    assert_eq!(status, HURRAY_ERR_FIELD_NOT_PRESENT);
    // Absence must not be reported as a zero the caller could read as data.
    assert_eq!(value, 0, "out_value is untouched on error");
    destroy(h);
}

// ── Composite ─────────────────────────────────────────────────────────────────

#[test]
fn reads_composite_rule_and_member_count() {
    let layout = LayoutDescriptor::Composite(
        CompositeLayout::new(CompositionRule::Overlay(CombineOp::Replace), 3).expect("composite"),
    );
    // A composite head owns no data: buffer_count is 0.
    let bytes = encode(layout, vec![16, 16], 0);
    let h = decode(&bytes);

    assert_eq!(
        scalar(h, HURRAY_LAYOUT_FIELD_COMPOSITE_RULE),
        CompositionRule::Overlay(CombineOp::Replace).rule_byte() as u64
    );
    assert_eq!(
        scalar(h, HURRAY_LAYOUT_FIELD_COMPOSITE_COMBINE_OP),
        CombineOp::Replace.wire_byte() as u64
    );
    assert_eq!(scalar(h, HURRAY_LAYOUT_FIELD_COMPOSITE_MEMBER_COUNT), 3);
    destroy(h);
}

// ── Applicability and argument errors ─────────────────────────────────────────

#[test]
fn field_from_another_layout_is_not_applicable() {
    let bytes = encode(LayoutDescriptor::Csr(CsrLayout::new(5)), vec![4, 4], 3);
    let h = decode(&bytes);
    let mut value: u64 = 0;
    // CSC's nnz, asked of a CSR descriptor.
    let status =
        unsafe { hurray_descriptor_field_u64(h, HURRAY_LAYOUT_FIELD_CSC_NNZ, 0, &mut value) };
    assert_eq!(status, HURRAY_ERR_FIELD_NOT_APPLICABLE);
    destroy(h);
}

#[test]
fn row_major_has_no_payload_fields() {
    let bytes = encode(LayoutDescriptor::RowMajor, vec![2, 3], 1);
    let h = decode(&bytes);
    let mut value: u64 = 0;
    let status =
        unsafe { hurray_descriptor_field_u64(h, HURRAY_LAYOUT_FIELD_CSR_NNZ, 0, &mut value) };
    assert_eq!(status, HURRAY_ERR_FIELD_NOT_APPLICABLE);
    destroy(h);
}

#[test]
fn unallocated_field_identifier_is_not_applicable() {
    let bytes = encode(LayoutDescriptor::Csr(CsrLayout::new(5)), vec![4, 4], 3);
    let h = decode(&bytes);
    let mut value: u64 = 0;
    // 0x0007 is an unallocated sub-object range; 0x0002_FF00 is the quantization
    // range with a discriminant byte that is not a scheme tag.
    for field in [0u32, 0x0001_07FF, 0x0002_FF00, 0x0007_0000, u32::MAX] {
        let status = unsafe { hurray_descriptor_field_u64(h, field, 0, &mut value) };
        assert_eq!(
            status, HURRAY_ERR_FIELD_NOT_APPLICABLE,
            "field {field:#x} should be rejected"
        );
    }
    destroy(h);
}

#[test]
fn a_scalar_field_is_not_readable_as_an_array() {
    let bytes = encode(LayoutDescriptor::Csr(CsrLayout::new(5)), vec![4, 4], 3);
    let h = decode(&bytes);
    let mut len: usize = 0;
    let status = unsafe {
        hurray_descriptor_field_u64_array(
            h,
            HURRAY_LAYOUT_FIELD_CSR_NNZ,
            0,
            std::ptr::null_mut(),
            &mut len,
        )
    };
    assert_eq!(status, HURRAY_ERR_FIELD_NOT_APPLICABLE);
    destroy(h);
}

#[test]
fn an_array_field_is_not_readable_as_a_scalar() {
    let layout = LayoutDescriptor::Strided(StridedLayout::new(vec![3, 1]));
    let bytes = encode(layout, vec![2, 3], 1);
    let h = decode(&bytes);
    let mut value: u64 = 0;
    let status = unsafe {
        hurray_descriptor_field_u64(h, HURRAY_LAYOUT_FIELD_STRIDED_STRIDES, 0, &mut value)
    };
    assert_eq!(status, HURRAY_ERR_FIELD_NOT_APPLICABLE);
    destroy(h);
}

#[test]
fn insufficient_capacity_reports_the_true_length() {
    let layout = LayoutDescriptor::Strided(StridedLayout::new(vec![12, 4, 1]));
    let bytes = encode(layout, vec![2, 3, 4], 1);
    let h = decode(&bytes);

    let mut out = [0i64; 1];
    let mut len: usize = 1; // capacity of 1, true length is 3
    let status = unsafe {
        hurray_descriptor_field_i64_array(
            h,
            HURRAY_LAYOUT_FIELD_STRIDED_STRIDES,
            0,
            out.as_mut_ptr(),
            &mut len,
        )
    };
    assert_eq!(status, HURRAY_ERR_BUFFER_TOO_SMALL);
    assert_eq!(len, 3, "the true length is written so the caller can retry");
    destroy(h);
}

#[test]
fn null_arguments_are_rejected() {
    let bytes = encode(LayoutDescriptor::Csr(CsrLayout::new(5)), vec![4, 4], 3);
    let h = decode(&bytes);

    let mut value: u64 = 0;
    assert_eq!(
        unsafe {
            hurray_descriptor_field_u64(
                std::ptr::null(),
                HURRAY_LAYOUT_FIELD_CSR_NNZ,
                0,
                &mut value,
            )
        },
        HURRAY_ERR_NULL_POINTER
    );
    assert_eq!(
        unsafe {
            hurray_descriptor_field_u64(h, HURRAY_LAYOUT_FIELD_CSR_NNZ, 0, std::ptr::null_mut())
        },
        HURRAY_ERR_NULL_POINTER
    );
    // out_len is required even in capacity-query mode, where out_values may be null.
    assert_eq!(
        unsafe {
            hurray_descriptor_field_u64_array(
                h,
                HURRAY_LAYOUT_FIELD_CSF_MODE_ORDER,
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        },
        HURRAY_ERR_NULL_POINTER
    );
    destroy(h);
}
