//! Dense layout conversion: the public API, end to end.
//!
//! The acceptance test ADR-044 § 6 names — every work item run exactly once, in any
//! order, produces byte-identical output to the serial run — is `items_in_any_order_*`
//! below. The rest covers the conversions `docs/impl/transcode-matrix.md` marks
//! `supported` or `survey-gated` for this version, and every refusal it names.

use hurray_core::descriptor::{Statistics, StatisticsMask, TensorDescriptor};
use hurray_core::layout::{MortonLayout, StridedLayout, TiledLayout};
use hurray_core::{
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode, MIN_BUFFER_ALIGNMENT,
};
use hurray_transcode::{
    plan, AlignedDestination, BufferPlan, Error, Planned, ReadyPlan, SurveyQuestion, Target,
};

// ── Fixtures ──────────────────────────────────────────────────────────────────

fn descriptor(
    ty: ElementType,
    dims: Vec<u64>,
    layout: LayoutDescriptor,
    byte_size: u64,
    byte_offset: u64,
) -> TensorDescriptor {
    let handle = BufferHandle::new(
        byte_size,
        MIN_BUFFER_ALIGNMENT,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )
    .unwrap();
    TensorDescriptor::new(
        1,
        0,
        ty,
        Shape::new(dims).unwrap(),
        byte_offset,
        layout,
        vec![handle],
        None,
        None,
        None,
        None,
    )
    .unwrap()
}

/// A 3×4 uint8 tensor holding 0..12 in row-major order.
fn rm_3x4() -> (TensorDescriptor, Vec<u8>) {
    (
        descriptor(
            ElementType::Uint8,
            vec![3, 4],
            LayoutDescriptor::RowMajor,
            12,
            0,
        ),
        (0u8..12).collect(),
    )
}

fn ready(source: &TensorDescriptor, target: &Target) -> ReadyPlan {
    match plan(source, target).expect("planning succeeds") {
        Planned::Ready(r) => r,
        Planned::NeedsSurvey(_) => panic!("expected a plan that needs no survey"),
    }
}

/// Runs a single-input, single-output plan and returns the destination bytes.
fn convert(ready: &ReadyPlan, source: &[u8]) -> Vec<u8> {
    let mut out = AlignedDestination::for_destination(ready.destinations(0)[0]).unwrap();
    let report = ready
        .run(&[&[source]], &mut [&mut [out.as_mut_slice()]])
        .expect("the run succeeds");
    assert_eq!(
        report.bytes_written(),
        report.elements_written() * (ready.outputs()[0].element_type.bit_width() as u64 / 8),
        "the report's two counts must agree on the element width"
    );
    out.as_slice().to_vec()
}

// ── Layout permutation ────────────────────────────────────────────────────────

#[test]
fn row_major_to_col_major() {
    let (source, data) = rm_3x4();
    let r = ready(&source, &Target::layout(LayoutDescriptor::ColMajor));
    // Column-major reads down the columns: 0,4,8 then 1,5,9 …
    assert_eq!(
        convert(&r, &data),
        vec![0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
    );
    assert_eq!(r.outputs()[0].layout, LayoutDescriptor::ColMajor);
    assert_eq!(r.outputs()[0].shape.dims(), &[3, 4]);
}

#[test]
fn col_major_back_to_row_major_is_the_original() {
    let (source, data) = rm_3x4();
    let to_cm = ready(&source, &Target::layout(LayoutDescriptor::ColMajor));
    let cm_bytes = convert(&to_cm, &data);

    let back = ready(
        &to_cm.outputs()[0].clone(),
        &Target::layout(LayoutDescriptor::RowMajor),
    );
    assert_eq!(convert(&back, &cm_bytes), data);
}

#[test]
fn rank3_permutation_round_trips() {
    let dims = vec![2u64, 3, 4];
    let data: Vec<u8> = (0u8..24).collect();
    let source = descriptor(
        ElementType::Uint8,
        dims.clone(),
        LayoutDescriptor::RowMajor,
        24,
        0,
    );
    let to_cm = ready(&source, &Target::layout(LayoutDescriptor::ColMajor));
    let cm = convert(&to_cm, &data);
    assert_ne!(cm, data, "a rank-3 permutation must move bytes");
    let back = ready(
        &to_cm.outputs()[0].clone(),
        &Target::layout(LayoutDescriptor::RowMajor),
    );
    assert_eq!(convert(&back, &cm), data);
}

#[test]
fn wider_element_types_move_whole_elements() {
    // float64: 8 bytes per element, so a permutation must move them as units.
    let values: Vec<f64> = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5];
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let source = descriptor(
        ElementType::Float64,
        vec![2, 3],
        LayoutDescriptor::RowMajor,
        48,
        0,
    );
    let r = ready(&source, &Target::layout(LayoutDescriptor::ColMajor));
    let out = convert(&r, &data);
    let got: Vec<f64> = out
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(got, vec![1.5, 4.5, 2.5, 5.5, 3.5, 6.5]);
}

#[test]
fn rank1_is_a_copy_with_a_different_layout_tag() {
    let data: Vec<u8> = (0u8..8).collect();
    let source = descriptor(
        ElementType::Uint8,
        vec![8],
        LayoutDescriptor::RowMajor,
        8,
        0,
    );
    // Rank-1 row-major and column-major address identically, so this is an alias.
    let r = ready(&source, &Target::layout(LayoutDescriptor::ColMajor));
    assert_eq!(
        r.buffer_plans(0),
        &[BufferPlan::Alias {
            input: 0,
            buffer: 0
        }]
    );
    assert_eq!(r.work_item_count(), 0);
    let report = r.run(&[&[&data]], &mut [&mut []]).unwrap();
    assert_eq!(report.bytes_written(), 0);
}

#[test]
fn scalar_tensor_has_one_element_and_no_work() {
    let data = vec![0xABu8];
    let source = descriptor(ElementType::Uint8, vec![], LayoutDescriptor::RowMajor, 1, 0);
    // A rank-0 tensor addresses offset 0 in every dense layout: nothing to permute.
    let r = ready(&source, &Target::layout(LayoutDescriptor::ColMajor));
    assert_eq!(
        r.buffer_plans(0),
        &[BufferPlan::Alias {
            input: 0,
            buffer: 0
        }]
    );
    assert_eq!(
        r.run(&[&[&data]], &mut [&mut []])
            .unwrap()
            .elements_written(),
        0
    );
}

#[test]
fn empty_tensor_produces_an_empty_buffer() {
    let source = descriptor(
        ElementType::Float32,
        vec![0, 4],
        LayoutDescriptor::RowMajor,
        0,
        0,
    );
    let r = ready(&source, &Target::layout(LayoutDescriptor::ColMajor));
    assert_eq!(r.destinations(0)[0].byte_size, 0);
    assert_eq!(r.outputs()[0].buffers[0].byte_size(), 0);
    assert_eq!(convert(&r, &[]), Vec::<u8>::new());
}

// ── Aliasing: two descriptions of the same bytes ──────────────────────────────

#[test]
fn identity_conversion_is_ready_with_no_work() {
    let (source, data) = rm_3x4();
    let r = ready(&source, &Target::layout(LayoutDescriptor::RowMajor));
    assert_eq!(r.work_item_count(), 0);
    assert_eq!(r.destinations(0), &[]);
    assert_eq!(
        r.buffer_plans(0),
        &[BufferPlan::Alias {
            input: 0,
            buffer: 0
        }]
    );
    // Callable, writes nothing, reports nothing moved — a conversion with an empty body.
    let report = r.run(&[&[&data]], &mut [&mut []]).unwrap();
    assert_eq!(report.elements_written(), 0);
    assert_eq!(report.bytes_written(), 0);
    // The output keeps the input's buffer handle and byte offset, because it *is* the
    // input's buffer.
    assert_eq!(r.outputs()[0].buffers[0], source.buffers[0]);
    assert_eq!(r.outputs()[0].byte_offset, source.byte_offset);
}

#[test]
fn row_major_and_its_strided_spelling_are_the_same_bytes() {
    let (source, _) = rm_3x4();
    let strided = LayoutDescriptor::Strided(StridedLayout::new(vec![4, 1]));
    let r = ready(&source, &Target::layout(strided.clone()));
    assert_eq!(
        r.buffer_plans(0),
        &[BufferPlan::Alias {
            input: 0,
            buffer: 0
        }]
    );
    // The output still carries the layout that was asked for.
    assert_eq!(r.outputs()[0].layout, strided);
}

#[test]
fn an_aliased_output_takes_no_destination() {
    let (source, data) = rm_3x4();
    let r = ready(&source, &Target::layout(LayoutDescriptor::RowMajor));
    let mut spare = AlignedDestination::zeroed(12, MIN_BUFFER_ALIGNMENT).unwrap();
    let err = r
        .run(&[&[&data]], &mut [&mut [spare.as_mut_slice()]])
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DestinationCount {
            tensor: 0,
            expected: 0,
            found: 1
        }
    ));
}

// ── Degenerate strides ────────────────────────────────────────────────────────

#[test]
fn negative_stride_is_normalised_to_row_major() {
    // Spec strided.md § Example: shape [3,4] strides [-4,1] is a row-major matrix with
    // its rows reversed, and `byte_offset` points at what would be row 2.
    let dims = vec![3u64, 4];
    let layout = LayoutDescriptor::Strided(StridedLayout::new(vec![-4, 1]));
    // Buffer holds rows in physical order 2,1,0; byte_offset 8 points at logical row 0.
    let data: Vec<u8> = vec![8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3];
    let source = descriptor(ElementType::Uint8, dims, layout, 12, 8);
    let r = ready(&source, &Target::layout(LayoutDescriptor::RowMajor));
    assert_eq!(r.outputs()[0].byte_offset, 0);
    assert_eq!(convert(&r, &data), (0u8..12).collect::<Vec<_>>());
}

#[test]
fn row_major_to_negative_stride_places_the_base_inside_the_buffer() {
    let (source, data) = rm_3x4();
    let reversed = LayoutDescriptor::Strided(StridedLayout::new(vec![-4, 1]));
    let r = ready(&source, &Target::layout(reversed));
    // Element [0,0] sits 8 elements in: the lowest address the tensor reads is row 2's.
    assert_eq!(r.outputs()[0].byte_offset, 8);
    assert_eq!(r.destinations(0)[0].byte_size, 12);
    assert_eq!(
        convert(&r, &data),
        vec![8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]
    );
}

#[test]
fn padded_strided_source_drops_its_gaps() {
    // Row stride 6 for 4-element rows: two padding elements per row, which a row-major
    // output does not keep.
    let layout = LayoutDescriptor::Strided(StridedLayout::new(vec![6, 1]));
    let data: Vec<u8> = vec![
        0, 1, 2, 3, 0xEE, 0xEE, // row 0 + padding
        4, 5, 6, 7, 0xEE, 0xEE, // row 1 + padding
        8, 9, 10, 11, // row 2 (the span stops at the last element)
    ];
    let source = descriptor(ElementType::Uint8, vec![3, 4], layout, 16, 0);
    let r = ready(&source, &Target::layout(LayoutDescriptor::RowMajor));
    assert_eq!(r.destinations(0)[0].byte_size, 12);
    assert_eq!(convert(&r, &data), (0u8..12).collect::<Vec<_>>());
}

#[test]
fn broadcast_is_materialised_and_the_output_can_be_larger_than_the_input() {
    // A zero stride on axis 0: four stored elements, read three times each.
    let layout = LayoutDescriptor::Strided(StridedLayout::new(vec![0, 1]));
    let data: Vec<u8> = vec![7, 8, 9, 10];
    let source = descriptor(ElementType::Uint8, vec![3, 4], layout, 4, 0);
    let r = ready(&source, &Target::layout(LayoutDescriptor::RowMajor));
    // The plan publishes the expanded size before any byte moves, which is what lets a
    // caller refuse it against a budget.
    assert_eq!(r.destinations(0)[0].byte_size, 12);
    assert_eq!(
        convert(&r, &data),
        vec![7, 8, 9, 10, 7, 8, 9, 10, 7, 8, 9, 10]
    );
}

// ── The survey ────────────────────────────────────────────────────────────────

#[test]
fn collapsing_an_axis_needs_a_survey_and_accepts_repeating_data() {
    let data: Vec<u8> = vec![7, 8, 9, 10, 7, 8, 9, 10, 7, 8, 9, 10];
    let source = descriptor(
        ElementType::Uint8,
        vec![3, 4],
        LayoutDescriptor::RowMajor,
        12,
        0,
    );
    let collapsed = LayoutDescriptor::Strided(StridedLayout::new(vec![0, 1]));
    let planned = plan(&source, &Target::layout(collapsed)).unwrap();
    let Planned::NeedsSurvey(survey) = planned else {
        panic!("collapsing a real axis is admissible only if the data repeats");
    };
    assert_eq!(
        survey.question(),
        &SurveyQuestion::BroadcastRepetition { axes: vec![0] }
    );
    // The output descriptor is known before the survey: this is an admissibility
    // question, not a sizing one.
    assert_eq!(survey.outputs()[0].buffers[0].byte_size(), 4);

    let r = survey.survey(&[&[&data]]).expect("the data repeats");
    assert_eq!(convert(&r, &data), vec![7, 8, 9, 10]);
}

#[test]
fn collapsing_refuses_data_that_does_not_repeat() {
    // Row 2 differs in its last element.
    let data: Vec<u8> = vec![7, 8, 9, 10, 7, 8, 9, 10, 7, 8, 9, 11];
    let source = descriptor(
        ElementType::Uint8,
        vec![3, 4],
        LayoutDescriptor::RowMajor,
        12,
        0,
    );
    let collapsed = LayoutDescriptor::Strided(StridedLayout::new(vec![0, 1]));
    let Planned::NeedsSurvey(survey) = plan(&source, &Target::layout(collapsed)).unwrap() else {
        panic!("expected a survey");
    };
    let err = survey.survey(&[&[&data]]).unwrap_err();
    match err {
        Error::RepetitionNotHeld { dim, index } => {
            assert_eq!(dim, 0);
            assert_eq!(index, vec![2, 3]);
        }
        other => panic!("expected a refusal, got {other}"),
    }
}

#[test]
fn an_already_broadcast_axis_needs_no_survey() {
    // Source and target both broadcast axis 0: nothing is collapsed, so there is no
    // repetition to establish.
    let from = LayoutDescriptor::Strided(StridedLayout::new(vec![0, 1]));
    let to = LayoutDescriptor::Strided(StridedLayout::new(vec![0, 2]));
    let source = descriptor(ElementType::Uint8, vec![3, 4], from, 4, 0);
    let r = ready(&source, &Target::layout(to));
    assert_eq!(r.destinations(0)[0].byte_size, 7); // offsets 0..=6 for strides [0,2]
}

// ── § 6: the order-independence invariant ─────────────────────────────────────

#[test]
// A megabyte of output is what it takes to span several work items, which Miri cannot
// interpret in reasonable time. `adversarial.rs` audits the same code path under Miri at a
// size it can manage.
#[cfg_attr(miri, ignore)]
fn items_in_any_order_produce_the_serial_output() {
    // A tensor large enough to be partitioned into several work items.
    // 1 MiB of output: several work items at this crate's item granularity.
    let dims = vec![512u64, 2048];
    let total = 512 * 2048;
    let data: Vec<u8> = (0..total).map(|i| (i % 251) as u8).collect();
    let source = descriptor(
        ElementType::Uint8,
        dims,
        LayoutDescriptor::RowMajor,
        total as u64,
        0,
    );
    let r = ready(&source, &Target::layout(LayoutDescriptor::ColMajor));
    assert!(
        r.work_item_count() > 1,
        "a {total}-byte output should partition into several items"
    );
    let serial = convert(&r, &data);

    let mut out = AlignedDestination::for_destination(r.destinations(0)[0]).unwrap();
    let mut report = hurray_transcode::RunReport::default();
    for item in (0..r.work_item_count()).rev() {
        report = report.combine(
            r.run_range(
                &[&[&data]],
                &mut [&mut [out.as_mut_slice()]],
                item..item + 1,
            )
            .unwrap(),
        );
    }
    assert_eq!(out.as_slice(), &serial[..]);
    assert_eq!(report.elements_written(), total as u64);
}

#[test]
#[cfg(feature = "rayon")]
#[cfg_attr(miri, ignore)]
fn parallel_output_is_byte_identical_to_serial() {
    let dims = vec![128u64, 512];
    let total = 128 * 512;
    let data: Vec<u8> = (0..total).map(|i| (i % 253) as u8).collect();
    let source = descriptor(
        ElementType::Uint8,
        dims,
        LayoutDescriptor::RowMajor,
        total as u64,
        0,
    );
    let r = ready(&source, &Target::layout(LayoutDescriptor::ColMajor));
    let serial = convert(&r, &data);

    let mut out = AlignedDestination::for_destination(r.destinations(0)[0]).unwrap();
    let report = r
        .run_parallel(&[&[&data]], &mut [&mut [out.as_mut_slice()]])
        .unwrap();
    assert_eq!(out.as_slice(), &serial[..]);
    assert_eq!(report.elements_written(), total as u64);
}

// ── Descriptor sections ───────────────────────────────────────────────────────

#[test]
fn an_exact_plan_carries_the_statistics_section_unchanged() {
    // ADR-044 § 7: an exact one-to-one plan preserves the index-to-value mapping, so
    // every advisory field keeps exactly the truth value it had — mask and all.
    let stats = Statistics {
        computed_mask: StatisticsMask(
            StatisticsMask::VALUE_RANGE_VALID | StatisticsMask::NAN_INF_VALID,
        ),
        nnz: 0,
        sparsity_ratio: 0.0,
        value_min: -2.0,
        value_max: 9.0,
        value_abs_max: 9.0,
        value_mean: 0.0,
        value_stddev: 0.0,
        nm_n: 0,
        nm_m: 0,
        has_nan: false,
        has_inf: true,
    };
    let handle = BufferHandle::new(
        12,
        MIN_BUFFER_ALIGNMENT,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )
    .unwrap();
    let source = TensorDescriptor::new(
        1,
        0,
        ElementType::Uint8,
        Shape::new(vec![3, 4]).unwrap(),
        0,
        LayoutDescriptor::RowMajor,
        vec![handle],
        None,
        None,
        Some(stats.clone()),
        None,
    )
    .unwrap();
    let r = ready(&source, &Target::layout(LayoutDescriptor::ColMajor));
    assert_eq!(r.outputs()[0].statistics, Some(stats));
}

// ── Refusals ──────────────────────────────────────────────────────────────────

#[test]
fn a_dynamic_dimension_is_refused_at_plan_time() {
    let source = descriptor(
        ElementType::Uint8,
        vec![3, hurray_core::DYNAMIC],
        LayoutDescriptor::RowMajor,
        12,
        0,
    );
    assert!(matches!(
        plan(&source, &Target::layout(LayoutDescriptor::ColMajor)),
        Err(Error::UnresolvedShape { dim: 1 })
    ));
}

#[test]
fn sub_byte_element_types_are_refused_by_name() {
    for ty in [
        ElementType::Int4,
        ElementType::Bool,
        ElementType::Float6E2M3,
    ] {
        let source = descriptor(ty, vec![3, 4], LayoutDescriptor::RowMajor, 16, 0);
        let err = plan(&source, &Target::layout(LayoutDescriptor::ColMajor)).unwrap_err();
        match err {
            Error::UnsupportedConversion { reason, .. } => {
                assert!(reason.contains("sub-byte"), "{reason}");
            }
            other => panic!("expected an unsupported-conversion refusal, got {other}"),
        }
    }
}

#[test]
fn layouts_without_an_affine_stride_form_are_refused() {
    let (source, _) = rm_3x4();
    let targets = [
        LayoutDescriptor::Tiled(Box::new(
            TiledLayout::new(vec![3, 2], 0x01, 0x01, None, None, None).unwrap(),
        )),
        LayoutDescriptor::Morton(MortonLayout::new(vec![2, 2]).unwrap()),
    ];
    for target in targets {
        let err = plan(&source, &Target::layout(target.clone())).unwrap_err();
        assert!(
            matches!(err, Error::UnsupportedConversion { .. }),
            "{target:?} gave {err}"
        );
    }
}

#[test]
fn a_device_resident_source_is_refused() {
    let handle = BufferHandle::new(12, 4096, DeviceTag::Cuda, SyncMode::Event).unwrap();
    let source = TensorDescriptor::new(
        1,
        0,
        ElementType::Uint8,
        Shape::new(vec![3, 4]).unwrap(),
        0,
        LayoutDescriptor::RowMajor,
        vec![handle],
        None,
        None,
        None,
        None,
    )
    .unwrap();
    assert!(matches!(
        plan(&source, &Target::layout(LayoutDescriptor::ColMajor)),
        Err(Error::NonHostBuffer {
            tensor: 0,
            buffer: 0,
            device_tag: DeviceTag::Cuda
        })
    ));
}

#[test]
fn a_source_buffer_too_small_for_its_own_layout_is_caught_at_plan_time() {
    // 3×4 uint8 needs 12 bytes; the handle declares 8.
    let source = descriptor(
        ElementType::Uint8,
        vec![3, 4],
        LayoutDescriptor::RowMajor,
        8,
        0,
    );
    assert!(matches!(
        plan(&source, &Target::layout(LayoutDescriptor::ColMajor)),
        Err(Error::SourceBufferTooSmall {
            needed: 12,
            declared: 8,
            ..
        })
    ));
}

#[test]
fn a_short_destination_is_refused_before_anything_is_written() {
    let (source, data) = rm_3x4();
    let r = ready(&source, &Target::layout(LayoutDescriptor::ColMajor));
    let mut out = AlignedDestination::zeroed(8, MIN_BUFFER_ALIGNMENT).unwrap();
    let err = r
        .run(&[&[&data]], &mut [&mut [out.as_mut_slice()]])
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DestinationTooShort {
            expected: 12,
            found: 8,
            ..
        }
    ));
    assert_eq!(out.as_slice(), &[0u8; 8], "nothing was written");
}

#[test]
fn a_misaligned_destination_is_refused() {
    let (source, data) = rm_3x4();
    let r = ready(&source, &Target::layout(LayoutDescriptor::ColMajor));
    // A Vec<u8> is aligned to 1: the declared 64-byte alignment is a promise to the
    // consumer, so writing into it would make the output descriptor lie.
    // A heap allocation, so the base address is the allocator's rather than the stack's.
    let mut backing = vec![0u8; 256].into_boxed_slice();
    let offset = 64 - (backing.as_ptr() as usize % 64) + 1;
    let misaligned = &mut backing[offset..offset + 12];
    let err = r.run(&[&[&data]], &mut [&mut [misaligned]]).unwrap_err();
    assert!(
        matches!(err, Error::DestinationMisaligned { expected: 64, .. }),
        "{err}"
    );
}

#[test]
fn an_input_whose_length_disagrees_with_its_handle_is_refused() {
    let (source, data) = rm_3x4();
    let r = ready(&source, &Target::layout(LayoutDescriptor::ColMajor));
    let mut out = AlignedDestination::for_destination(r.destinations(0)[0]).unwrap();
    let err = r
        .run(&[&[&data[..11]]], &mut [&mut [out.as_mut_slice()]])
        .unwrap_err();
    assert!(matches!(
        err,
        Error::InputBufferSize {
            expected: 12,
            found: 11,
            ..
        }
    ));
}

#[test]
fn the_wrong_number_of_input_tensors_is_refused() {
    let (source, data) = rm_3x4();
    let r = ready(&source, &Target::layout(LayoutDescriptor::ColMajor));
    let mut out = AlignedDestination::for_destination(r.destinations(0)[0]).unwrap();
    let err = r
        .run(&[&[&data], &[&data]], &mut [&mut [out.as_mut_slice()]])
        .unwrap_err();
    assert!(matches!(
        err,
        Error::InputTensorCount {
            expected: 1,
            found: 2
        }
    ));
}

#[test]
fn a_plan_is_reusable_across_conforming_buffers() {
    // A plan that came straight from `Planned::Ready` is a function of descriptors
    // alone, so the same plan converts any buffer matching them (ADR-044 § 4).
    let (source, data) = rm_3x4();
    let r = ready(&source, &Target::layout(LayoutDescriptor::ColMajor));
    assert_eq!(
        convert(&r, &data),
        vec![0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
    );
    let other: Vec<u8> = (100u8..112).collect();
    assert_eq!(
        convert(&r, &other),
        vec![100, 104, 108, 101, 105, 109, 102, 106, 110, 103, 107, 111]
    );
}
