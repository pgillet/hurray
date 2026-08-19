//! Three-tier validation of a layout against the buffers supplied with it.
//!
//! **The layout object is a declaration; the buffers are evidence. They must agree.**
//!
//! | Tier | Check | Error |
//! |---|---|---|
//! | Shape | rank and shape constraints (CSR rank 2, CSF rank ≥ 3, `len(strides) == rank`) | `InvalidDescriptorError` |
//! | Buffer count | supplied buffers ≥ the layout's required count; quantization indices fall beyond them | `InvalidDescriptorError` |
//! | Buffer size | each buffer at least as large as the layout's parameters imply | `BufferError` |
//!
//! ## Never reinterpret, never infer
//!
//! `CooLayout(nnz=4)` supplied with a two-element values buffer raises. The
//! descriptor is not silently corrected to `nnz=2`, and it is not accepted as given:
//! it would encode and decode cleanly and hand the consumer an out-of-bounds read.
//! Over-sized buffers are allowed — alignment and padding slack are legitimate.
//!
//! ## The hole
//!
//! A private or unknown layout has no statically knowable buffer count, so the count
//! and size tiers cannot run for it at all. Nothing in such a descriptor says how
//! many buffers it needs or how large they should be; only its definer knows.

use pyo3::prelude::*;

use hurray_core::{
    buffer_size_bytes, ElementType, LayoutDescriptor, QuantizationDescriptor, Shape,
};

use crate::errors::{BufferError, InvalidDescriptorError};

use super::layout_name;

/// Bytes per `uint64` index element — the element type of every sparse index buffer.
const INDEX_ELEMENT_BYTES: u64 = 8;

/// Run all three tiers for `layout` against the supplied buffer lengths.
pub(crate) fn validate_layout(
    layout: &LayoutDescriptor,
    shape: &Shape,
    element_type: ElementType,
    buffer_lens: &[usize],
) -> PyResult<()> {
    // Tier 1 — shape. Core owns every rank and dimension rule; the binding must not
    // grow a second copy of them that can drift.
    layout.validate_against_shape(shape).map_err(|e| {
        InvalidDescriptorError::new_err(format!("layout does not match this tensor's shape: {e}"))
    })?;

    // Tier 2 — buffer count.
    if let Some(required) = layout.buffer_count() {
        let required = required.get() as usize;
        if buffer_lens.len() < required {
            return Err(InvalidDescriptorError::new_err(format!(
                "a {} layout needs {required} buffer(s), but {} were supplied; pass the \
                 rest via aux_buffers=[...]",
                layout_name(layout),
                buffer_lens.len(),
            )));
        }
    }

    // Tier 3 — buffer size.
    for (index, minimum) in buffer_minimums(layout, shape, element_type)
        .into_iter()
        .enumerate()
    {
        let (Some(minimum), Some(actual)) = (minimum, buffer_lens.get(index)) else {
            continue;
        };
        if (*actual as u64) < minimum {
            return Err(BufferError::new_err(format!(
                "buffer {index} ({}) is {actual} bytes, but this {} layout implies at \
                 least {minimum}",
                buffer_role(layout, index),
                layout_name(layout),
            )));
        }
    }

    Ok(())
}

/// Reject a quantization parameter buffer that overlaps the layout's own buffers.
///
/// Core already rejects an index that aliases the data buffer or runs off the end of
/// the table; it does not know that a CSR layout owns indices 1 and 2 as well, so a
/// scale index of 2 would silently designate the row-pointer buffer as scales.
pub(crate) fn validate_quantization_indices(
    layout: &LayoutDescriptor,
    quantization: &QuantizationDescriptor,
) -> PyResult<()> {
    let Some(owned) = layout.buffer_count() else {
        return Ok(());
    };
    let owned = owned.get() as u32;
    for index in quantization_buffer_indices(quantization) {
        if index < owned {
            return Err(InvalidDescriptorError::new_err(format!(
                "quantization buffer index {index} is inside the {} layout's own \
                 buffers (indices 0..{}); quantization parameters must come after them",
                layout_name(layout),
                owned - 1,
            )));
        }
    }
    Ok(())
}

/// The buffer indices a quantization scheme references.
fn quantization_buffer_indices(q: &QuantizationDescriptor) -> Vec<u32> {
    match q {
        // Per-tensor affine is the one scheme whose parameters are inline.
        QuantizationDescriptor::PerTensorAffine(_) => Vec::new(),
        QuantizationDescriptor::PerChannelAffine(x) => {
            [Some(x.scale_buffer_index()), x.zero_point_buffer_index()]
                .into_iter()
                .flatten()
                .collect()
        }
        QuantizationDescriptor::PerBlockAffine(x) => {
            [Some(x.scale_buffer_index()), x.zero_point_buffer_index()]
                .into_iter()
                .flatten()
                .collect()
        }
        QuantizationDescriptor::Nf4(x) => vec![x.scale_buffer_index()],
        QuantizationDescriptor::Mxfp(x) => vec![x.scale_buffer_index()],
    }
}

/// The minimum byte size of each buffer the layout implies.
///
/// `None` at an index means the size is not derivable from the descriptor alone —
/// a CSF interior level, or a block table whose length is `seq_ptr[num_seqs]`, a
/// value that lives in the data rather than in the descriptor.
fn buffer_minimums(
    layout: &LayoutDescriptor,
    shape: &Shape,
    element_type: ElementType,
) -> Vec<Option<u64>> {
    let rank = shape.rank() as u64;
    let values = |nnz: u64| Some(buffer_size_bytes(element_type, nnz));
    // Saturating rather than wrapping: an absurd nnz must produce an unsatisfiable
    // minimum, never a small one that lets an under-sized buffer through.
    let indices = |count: u64| Some(count.saturating_mul(INDEX_ELEMENT_BYTES));
    // A pointer array has one entry per slice along a dimension, plus one. A dynamic
    // dimension makes that count unknown, not enormous.
    let pointer_array = |dim: Option<&u64>| match dim {
        Some(&d) if d != hurray_core::DYNAMIC => {
            Some(d.saturating_add(1).saturating_mul(INDEX_ELEMENT_BYTES))
        }
        _ => None,
    };

    match layout {
        // Dense layouts hold every logical element in one buffer. A dynamic dimension
        // makes the count unknown, which is a None rather than a zero.
        LayoutDescriptor::RowMajor
        | LayoutDescriptor::ColMajor
        | LayoutDescriptor::Strided(_)
        | LayoutDescriptor::Tiled(_)
        | LayoutDescriptor::Morton(_)
        | LayoutDescriptor::Hilbert(_) => {
            vec![shape
                .element_count()
                .map(|n| buffer_size_bytes(element_type, n))]
        }

        LayoutDescriptor::Coo(l) => vec![values(l.nnz), indices(l.nnz.saturating_mul(rank))],

        // CSR: values, col_indices, row_ptr — row_ptr has one entry per row plus one.
        LayoutDescriptor::Csr(l) => vec![
            values(l.nnz),
            indices(l.nnz),
            pointer_array(shape.dims().first()),
        ],

        // CSC: values, row_indices, col_ptr.
        LayoutDescriptor::Csc(l) => vec![
            values(l.nnz),
            indices(l.nnz),
            pointer_array(shape.dims().get(1)),
        ],

        // CSF: values, then (pos_L, crd_L) per level. Only three of the 2·rank+1
        // sizes follow from the descriptor: pos_0 is always two entries, the leaf
        // crd holds one entry per non-zero, and the interior levels depend on counts
        // that live in the buffers themselves.
        LayoutDescriptor::Csf(l) => {
            let levels = l.mode_order.len();
            let mut minimums = vec![values(l.nnz)];
            for level in 0..levels {
                minimums.push(if level == 0 { indices(2) } else { None });
                minimums.push(if level + 1 == levels {
                    indices(l.nnz)
                } else {
                    None
                });
            }
            minimums
        }

        // Block-paged: the page pool covers every page; seq_ptr has one entry per
        // sequence plus one. The block table's length is seq_ptr[num_seqs], which is
        // data, not descriptor.
        LayoutDescriptor::BlockPaged(l) => {
            let per_page: Option<u64> = shape
                .dims()
                .iter()
                .enumerate()
                .filter(|(axis, _)| *axis != l.paged_axis as usize)
                .try_fold(1u64, |acc, (_, &dim)| acc.checked_mul(dim));
            let pool = per_page
                .and_then(|per| per.checked_mul(l.num_pages))
                .and_then(|n| n.checked_mul(l.page_size as u64))
                .map(|n| buffer_size_bytes(element_type, n));
            let index_bytes = l.block_table_index_type.element_bytes() as u64;
            vec![
                pool,
                None,
                Some((l.num_seqs as u64 + 1).saturating_mul(index_bytes)),
            ]
        }

        // A composite head owns no buffers; private and unknown layouts declare none
        // whose size this implementation could know.
        _ => Vec::new(),
    }
}

/// The spec's name for the buffer at `index` under this layout, for error messages.
fn buffer_role(layout: &LayoutDescriptor, index: usize) -> &'static str {
    match (layout, index) {
        (LayoutDescriptor::Coo(_), 0)
        | (LayoutDescriptor::Csr(_), 0)
        | (LayoutDescriptor::Csc(_), 0)
        | (LayoutDescriptor::Csf(_), 0) => "values",
        (LayoutDescriptor::Coo(_), 1) => "indices",
        (LayoutDescriptor::Csr(_), 1) => "col_indices",
        (LayoutDescriptor::Csr(_), 2) => "row_ptr",
        (LayoutDescriptor::Csc(_), 1) => "row_indices",
        (LayoutDescriptor::Csc(_), 2) => "col_ptr",
        (LayoutDescriptor::Csf(_), i) if i % 2 == 1 => "pos",
        (LayoutDescriptor::Csf(_), _) => "crd",
        (LayoutDescriptor::BlockPaged(_), 0) => "page_pool",
        (LayoutDescriptor::BlockPaged(_), 1) => "block_table",
        (LayoutDescriptor::BlockPaged(_), 2) => "seq_ptr",
        _ => "data",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use hurray_core::layout::{
        BlockPagedLayout, BlockTableIndexType, CooLayout, CscLayout, CsrLayout, KvRole,
    };
    use hurray_core::CsfLayout;

    fn shape(dims: &[u64]) -> Shape {
        Shape::new(dims.to_vec()).expect("valid shape")
    }

    #[test]
    fn dense_minimum_is_the_whole_element_grid() {
        let minimums = buffer_minimums(
            &LayoutDescriptor::RowMajor,
            &shape(&[2, 3]),
            ElementType::Float32,
        );
        assert_eq!(minimums, vec![Some(24)]);
    }

    #[test]
    fn coo_sizes_both_of_its_buffers() {
        let layout = LayoutDescriptor::Coo(CooLayout::new(3, false));
        // values: 3 * 4 bytes; indices: 3 * rank * 8 bytes.
        assert_eq!(
            buffer_minimums(&layout, &shape(&[2, 2]), ElementType::Float32),
            vec![Some(12), Some(48)]
        );
    }

    #[test]
    fn csr_row_ptr_has_one_entry_per_row_plus_one() {
        let layout = LayoutDescriptor::Csr(CsrLayout::new(4));
        assert_eq!(
            buffer_minimums(&layout, &shape(&[3, 5]), ElementType::Float32),
            vec![Some(16), Some(32), Some(32)]
        );
    }

    #[test]
    fn csc_col_ptr_uses_the_second_dimension() {
        let layout = LayoutDescriptor::Csc(CscLayout::new(4));
        assert_eq!(
            buffer_minimums(&layout, &shape(&[3, 5]), ElementType::Float32),
            vec![Some(16), Some(32), Some(48)]
        );
    }

    #[test]
    fn csf_sizes_only_what_the_descriptor_states() {
        let layout = LayoutDescriptor::Csf(CsfLayout::new(4, vec![0, 1, 2]));
        let minimums = buffer_minimums(&layout, &shape(&[2, 3, 4]), ElementType::Float32);

        assert_eq!(minimums.len(), 7, "2 * rank + 1 buffers");
        assert_eq!(minimums[0], Some(16), "values: nnz * 4 bytes");
        assert_eq!(minimums[1], Some(16), "pos_0 is always two entries");
        assert_eq!(minimums[6], Some(32), "leaf crd: one entry per non-zero");
        // The interior levels depend on counts that live in the buffers themselves.
        assert_eq!(minimums[2], None);
        assert_eq!(minimums[3], None);
        assert_eq!(minimums[4], None);
        assert_eq!(minimums[5], None);
    }

    #[test]
    fn block_paged_cannot_size_its_block_table() {
        let layout = LayoutDescriptor::BlockPaged(BlockPagedLayout::new(
            2,
            2,
            0,
            1,
            KvRole::Key,
            None,
            BlockTableIndexType::U32,
        ));
        let minimums = buffer_minimums(&layout, &shape(&[4, 2]), ElementType::Float32);

        // page pool: num_pages * page_size * (dims except the paged axis) * 4 bytes.
        assert_eq!(minimums[0], Some(32));
        // block_table's length is seq_ptr[num_seqs] — data, not descriptor.
        assert_eq!(minimums[1], None);
        // seq_ptr: (num_seqs + 1) uint32 entries.
        assert_eq!(minimums[2], Some(8));
    }

    #[test]
    fn an_extension_layout_declares_no_sizes() {
        let layout = LayoutDescriptor::Unknown(
            hurray_core::layout::UnknownLayout::new(0x0C, vec![]).expect("valid tag"),
        );
        assert!(buffer_minimums(&layout, &shape(&[4]), ElementType::Float32).is_empty());
    }

    #[test]
    fn buffer_roles_name_the_spec_fields() {
        let csr = LayoutDescriptor::Csr(CsrLayout::new(1));
        assert_eq!(buffer_role(&csr, 0), "values");
        assert_eq!(buffer_role(&csr, 1), "col_indices");
        assert_eq!(buffer_role(&csr, 2), "row_ptr");

        let csf = LayoutDescriptor::Csf(CsfLayout::new(1, vec![0, 1, 2]));
        assert_eq!(buffer_role(&csf, 1), "pos");
        assert_eq!(buffer_role(&csf, 2), "crd");
    }
}
