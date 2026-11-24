//! Which targets can be written through, and whether every address is representable.
//!
//! Three questions, all of them about addressing rather than about any one layout:
//!
//! - **Injectivity.** Can two logical indices reach the same stored element? A target that
//!   aliases cannot represent arbitrary values — one of the two is overwritten — and in
//!   parallel the two writes can fall in different work items, which is a data race.
//!   Aliasing is *not* a zero-stride phenomenon: `[1, 1]` over shape `[2, 2]` is the
//!   counterexample that matters. The refusal comes in two forms, and they are kept apart
//!   here as they are in the API — `AliasingTarget` is proved and permanent,
//!   `TargetNotProvablyDistinct` is this crate's test giving up.
//! - **Arithmetic.** Do the byte strides, the base, and the extent stay representable, and
//!   is a descriptor that is merely unallocatable distinguished from one that overflows?
//! - **Order.** Does a run produce the same bytes serially, with its work items in reverse,
//!   and on a thread pool?
//!
//! Every case here was **accepted and wrong** before the planning-time injectivity check
//! existed; they arrived from an adversarial review of that pass. The oracle is deliberately
//! independent: `reference_convert` recomputes the expected bytes from the stride definition
//! in `docs/spec/layouts/strided.md § Element Address` using nothing from the crate, so a
//! defect in the production address helpers cannot hide behind a test that shares them.
//!
//! # Running these under Miri
//!
//! The default build is clean under Stacked Borrows:
//!
//! ```sh
//! cargo +nightly miri test -p hurray-transcode
//! ```
//!
//! The `rayon` build needs Tree Borrows and leak tolerance, for reasons that are both
//! outside this crate: `crossbeam-epoch` (a `rayon` dependency) violates Stacked Borrows,
//! which is still experimental, and `rayon`'s global pool never joins its threads.
//!
//! ```sh
//! MIRIFLAGS="-Zmiri-tree-borrows -Zmiri-ignore-leaks" \
//!   cargo +nightly miri test -p hurray-transcode --features rayon
//! ```
//!
//! Miri audits the unsafe write path; it does not prove the absence of a data race, since
//! it observes one interleaving. The disjointness argument is what rules races out — see
//! `parallel.rs` — and `every_accepted_plan_writes_each_output_byte_exactly_once` checks
//! its conclusion directly, independently of any schedule.

use std::collections::{HashMap, HashSet};

use hurray_core::descriptor::TensorDescriptor;
use hurray_core::layout::StridedLayout;
use hurray_core::{
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode, MIN_BUFFER_ALIGNMENT,
};
use hurray_transcode::{plan, AlignedDestination, Error, Planned, ReadyPlan, RunReport, Target};

// ── An independent reference implementation ───────────────────────────────────

/// Every logical index of `dims`, in row-major order.
fn indices(dims: &[u64]) -> Vec<Vec<u64>> {
    let mut out = vec![vec![]];
    for &d in dims {
        let mut next = Vec::new();
        for prefix in &out {
            for i in 0..d {
                let mut row = prefix.clone();
                row.push(i);
                next.push(row);
            }
        }
        out = next;
    }
    out
}

/// `offset = Σ index[k] × strides[k]`, straight from the spec, in `i128` so that the
/// oracle cannot overflow where the implementation might.
fn offset_of(index: &[u64], strides: &[i64]) -> i128 {
    index
        .iter()
        .zip(strides)
        .map(|(&i, &s)| i128::from(i) * i128::from(s))
        .sum()
}

/// What a conversion must produce: for every logical index, the source element read at the
/// source's strides, written at the target's strides.
///
/// Returns the destination buffer and its `byte_offset`, computed independently of the
/// crate. Only meaningful for targets that address each element distinctly — which is the
/// premise this test file is about.
fn reference_convert(
    dims: &[u64],
    src_strides: &[i64],
    dst_strides: &[i64],
    elem_bytes: usize,
    src_byte_offset: u64,
    src: &[u8],
) -> (Vec<u8>, u64) {
    let all = indices(dims);
    let dst_low = all
        .iter()
        .map(|i| offset_of(i, dst_strides))
        .min()
        .unwrap_or(0);
    let dst_high = all
        .iter()
        .map(|i| offset_of(i, dst_strides))
        .max()
        .unwrap_or(0);
    let base = (-dst_low) as u64 * elem_bytes as u64;
    let size = if dims.contains(&0) {
        0
    } else {
        ((dst_high - dst_low + 1) as u64) * elem_bytes as u64
    };
    let mut out = vec![0u8; size as usize];
    for index in &all {
        let from =
            (src_byte_offset as i128 + offset_of(index, src_strides) * elem_bytes as i128) as usize;
        let to = (base as i128 + offset_of(index, dst_strides) * elem_bytes as i128) as usize;
        out[to..to + elem_bytes].copy_from_slice(&src[from..from + elem_bytes]);
    }
    (out, base)
}

/// Does this stride vector send two distinct indices of `dims` to one element? Computed by
/// enumeration, not by the predicate under test.
fn actually_aliases(dims: &[u64], strides: &[i64]) -> bool {
    let mut seen = HashSet::new();
    indices(dims)
        .iter()
        .any(|i| !seen.insert(offset_of(i, strides)))
}

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

fn strided(strides: Vec<i64>) -> LayoutDescriptor {
    LayoutDescriptor::Strided(StridedLayout::new(strides))
}

fn ready(source: &TensorDescriptor, target: &Target) -> ReadyPlan {
    match plan(source, target).expect("planning succeeds") {
        Planned::Ready(r) => r,
        Planned::NeedsSurvey(_) => panic!("expected a plan that needs no survey"),
    }
}

fn convert(plan: &ReadyPlan, source: &[u8]) -> Vec<u8> {
    let mut out = AlignedDestination::for_destination(plan.destinations(0)[0]).unwrap();
    plan.run(&[&[source]], &mut [&mut [out.as_mut_slice()]])
        .expect("the run succeeds");
    out.as_slice().to_vec()
}

// ── 1 & 2. Overlapping destinations: the counterexample, and its family ───────

#[test]
fn the_review_counterexample_is_refused_at_plan_time() {
    // shape [2,2], target strides [1,1]: (0,1) and (1,0) both address element 1, so the
    // destination cannot hold all four values. Before the injectivity check this planned
    // as `Ready` and wrote [1, 3, 4] — value 2 silently overwritten by 3.
    let source = descriptor(
        ElementType::Uint8,
        vec![2, 2],
        LayoutDescriptor::RowMajor,
        4,
        0,
    );
    assert!(actually_aliases(&[2, 2], &[1, 1]), "premise of this test");

    let err = plan(&source, &Target::layout(strided(vec![1, 1]))).unwrap_err();
    match err {
        Error::AliasingTarget { strides, dims } => {
            assert_eq!(strides, vec![1, 1]);
            assert_eq!(dims, vec![2, 2]);
        }
        other => panic!("expected AliasingTarget, got {other}"),
    }
}

#[test]
fn an_overlapping_destination_is_refused_before_any_destination_is_touched() {
    // The refusal is a *planning* refusal, so there is no destination to mutate: a caller
    // cannot even reach a run, and the sentinel below can never be overwritten.
    let source = descriptor(
        ElementType::Uint8,
        vec![2, 2],
        LayoutDescriptor::RowMajor,
        4,
        0,
    );
    let mut sentinel = AlignedDestination::zeroed(64, MIN_BUFFER_ALIGNMENT).unwrap();
    sentinel.as_mut_slice().fill(0xA5);
    assert!(plan(&source, &Target::layout(strided(vec![1, 1]))).is_err());
    assert!(sentinel.as_slice().iter().all(|&b| b == 0xA5));
}

#[test]
fn overlapping_targets_are_refused_across_a_family_of_shapes() {
    // No zero stride anywhere in this table: aliasing is not a zero-stride phenomenon.
    let cases: &[(&[u64], &[i64])] = &[
        (&[2, 2], &[1, 1]),
        (&[3, 3], &[1, 2]),
        (&[4, 4], &[1, 3]),
        (&[2, 2, 2], &[1, 1, 1]),
        // The last two do *not* alias, and the table keeps them: a check that refused
        // every strided target would pass the rows above and still be wrong.
        (&[2, 3], &[3, 1]),
        (&[5, 5], &[5, 1]),
    ];
    for (dims, strides) in cases {
        let aliases = actually_aliases(dims, strides);
        let bytes: u64 = dims.iter().product();
        let source = descriptor(
            ElementType::Uint8,
            dims.to_vec(),
            LayoutDescriptor::RowMajor,
            bytes.max(1) * 4,
            0,
        );
        let outcome = plan(&source, &Target::layout(strided(strides.to_vec())));
        if aliases {
            // Every aliasing row above is within the predicate's reach — two equal
            // magnitudes, or a magnitude equal to the reach of the smaller axes — so the
            // proved error is the one to assert. An aliasing target outside that reach is
            // `TargetNotProvablyDistinct` instead; see the test below.
            assert!(
                matches!(outcome, Err(Error::AliasingTarget { .. })),
                "{dims:?} / {strides:?} aliases and must be refused: {outcome:?}"
            );
        } else {
            // A layout that does not alias must still be accepted — the check must not
            // turn into a blanket refusal of strided targets.
            assert!(
                outcome.is_ok(),
                "{dims:?} / {strides:?} does not alias and must be accepted"
            );
        }
    }
}

#[test]
fn an_overlap_that_survives_the_broadcast_clamp_is_refused() {
    // One axis is a genuine broadcast (clamped out of the cover), and the two that remain
    // still alias. The clamp must not be mistaken for a licence to skip the check.
    let source = descriptor(
        ElementType::Uint8,
        vec![2, 2, 2],
        LayoutDescriptor::RowMajor,
        8,
        0,
    );
    let err = plan(&source, &Target::layout(strided(vec![0, 1, 1]))).unwrap_err();
    assert!(
        matches!(err, Error::AliasingTarget { ref dims, .. } if dims == &vec![1, 2, 2]),
        "{err}"
    );
}

#[test]
fn a_supported_broadcast_collapse_is_still_planned() {
    // The one aliasing target the crate does support: the collapsed axis is clamped out of
    // the cover, so what remains is injective and the survey carries the value argument.
    let source = descriptor(
        ElementType::Uint8,
        vec![3, 4],
        LayoutDescriptor::RowMajor,
        12,
        0,
    );
    let planned = plan(&source, &Target::layout(strided(vec![0, 1]))).unwrap();
    assert!(matches!(planned, Planned::NeedsSurvey(_)));
}

#[test]
fn a_non_injective_source_is_materialised_rather_than_refused() {
    // Reading an aliasing layout is sound: the logical tensor *is* the one its strides
    // describe, repeats and all. Only writing through one loses values, so the check is
    // on the target alone.
    assert!(actually_aliases(&[2, 2], &[1, 1]));
    let source = descriptor(ElementType::Uint8, vec![2, 2], strided(vec![1, 1]), 3, 0);
    let data: Vec<u8> = vec![10, 20, 30];
    let r = ready(&source, &Target::layout(LayoutDescriptor::RowMajor));
    let (expected, _) = reference_convert(&[2, 2], &[1, 1], &[2, 1], 1, 0, &data);
    assert_eq!(convert(&r, &data), expected);
    assert_eq!(convert(&r, &data), vec![10, 20, 20, 30]);
}

#[test]
fn an_injective_but_unproved_target_is_a_limitation_not_a_refusal() {
    // Strides [2, 3] over shape [3, 2] address offsets 0, 3, 2, 5, 4, 7: injective, and
    // outside what the nested-magnitude proof can see. The two refusals must not be
    // conflated — this target is unsupported, not impossible, and a caller writing layout
    // fallback policy needs to tell "never retry" from "retry on a newer version".
    assert!(
        !actually_aliases(&[3, 2], &[2, 3]),
        "premise: this target does not alias"
    );
    let source = descriptor(
        ElementType::Uint8,
        vec![3, 2],
        LayoutDescriptor::RowMajor,
        8,
        0,
    );
    let err = plan(&source, &Target::layout(strided(vec![2, 3]))).unwrap_err();
    match err {
        Error::TargetNotProvablyDistinct { strides, dims } => {
            assert_eq!(strides, vec![2, 3]);
            assert_eq!(dims, vec![3, 2]);
        }
        other => panic!("expected TargetNotProvablyDistinct, got {other}"),
    }
}

#[test]
fn an_aliasing_source_asked_for_its_own_layout_still_plans() {
    // The write-target restriction must not reach the no-op path. When the stride vectors
    // are equal the plan aliases the input buffer and writes nothing, so there is no
    // destination to lose a value in — and the descriptor is one the format permits. The
    // natural implementation mistake is to hoist the distinctness check to the top of
    // `plan()`, which would refuse this legal case; this test is what catches that.
    for layout_strides in [vec![1i64, 1], vec![0, 1], vec![0, 0]] {
        let source = descriptor(
            ElementType::Uint8,
            vec![2, 2],
            strided(layout_strides.clone()),
            8,
            0,
        );
        let planned = plan(&source, &Target::layout(strided(layout_strides.clone())))
            .unwrap_or_else(|e| panic!("{layout_strides:?} must still plan: {e}"));
        let ready = match planned {
            Planned::Ready(r) => r,
            Planned::NeedsSurvey(_) => panic!("{layout_strides:?}: a no-op needs no survey"),
        };
        // Aliased, so nothing is written: no destination, and no work.
        assert!(ready.destinations(0).is_empty(), "{layout_strides:?}");
        assert_eq!(ready.work_item_count(), 0, "{layout_strides:?}");
    }
}

// ── 1. The parallel case: no two work items may write one byte ────────────────

#[test]
fn the_parallel_counterexample_cannot_be_planned() {
    // [2, 300_000] with target strides [1, 1] was the shape that made the race observable:
    // large enough that the colliding indices land in different work items. With the
    // planning check it never reaches a run, in either execution mode.
    let dims = vec![2u64, 300_000];
    let source = descriptor(
        ElementType::Uint8,
        dims.clone(),
        LayoutDescriptor::RowMajor,
        600_000,
        0,
    );
    assert!(actually_aliases(&[2, 4], &[1, 1]), "same family, small");
    assert!(matches!(
        plan(&source, &Target::layout(strided(vec![1, 1]))),
        Err(Error::AliasingTarget { .. })
    ));
}

#[test]
fn every_accepted_plan_writes_each_output_byte_exactly_once() {
    // The invariant stated directly, rather than inferred from serial/parallel equality:
    // walk the cover the way the engine does and count writes per byte. A target that
    // aliased would show a byte written twice — which is the race, independent of
    // scheduling.
    let cases: &[(&[u64], &[i64])] = &[
        (&[3, 4], &[4, 1]),
        (&[3, 4], &[1, 3]),
        (&[3, 4], &[-4, 1]),
        (&[3, 4], &[8, 1]),
        (&[2, 3, 4], &[1, 2, 6]),
        (&[3, 4], &[0, 1]),
    ];
    for (dims, dst_strides) in cases {
        let mut writes: HashMap<i128, usize> = HashMap::new();
        // The cover: a broadcast axis contributes one index, as the engine's `out_dims`.
        let cover: Vec<u64> = dims
            .iter()
            .zip(dst_strides.iter())
            .map(|(&d, &s)| if s == 0 { 1 } else { d })
            .collect();
        for index in indices(&cover) {
            *writes.entry(offset_of(&index, dst_strides)).or_default() += 1;
        }
        let total: u64 = cover.iter().product();
        assert_eq!(
            writes.len() as u64,
            total,
            "{dims:?} / {dst_strides:?}: {} distinct addresses for {total} writes",
            writes.len()
        );
        assert!(writes.values().all(|&n| n == 1));
    }
}

/// Rows and columns for the equivalence sweep. Miri interprets every element copy, so it
/// gets a tensor sized for an audit of the unsafe write path rather than for throughput;
/// the multi-item cases are covered natively in `dense_layouts.rs`.
#[cfg(feature = "rayon")]
const ROWS: u64 = if cfg!(miri) { 6 } else { 200 };
#[cfg(feature = "rayon")]
const COLS: u64 = if cfg!(miri) { 9 } else { 300 };

#[test]
#[cfg(feature = "rayon")]
fn serial_reverse_and_parallel_agree_on_every_supported_layout() {
    let count = (ROWS * COLS) as usize;
    let data: Vec<u8> = (0..count).map(|i| (i % 251) as u8).collect();
    let padded = COLS as i64 + 20;
    let cases: &[(&[u64], &[i64], &[i64])] = &[
        (&[ROWS, COLS], &[COLS as i64, 1], &[1, ROWS as i64]), // row-major → column-major
        (&[ROWS, COLS], &[COLS as i64, 1], &[-(COLS as i64), 1]), // → reversed rows
        (&[ROWS, COLS], &[1, ROWS as i64], &[COLS as i64, 1]), // column-major → row-major
        (&[ROWS, COLS], &[COLS as i64, 1], &[padded, 1]),      // → padded rows
    ];
    for (dims, src_strides, dst_strides) in cases {
        let source = descriptor(
            ElementType::Uint8,
            dims.to_vec(),
            strided(src_strides.to_vec()),
            count as u64,
            0,
        );
        let r = ready(&source, &Target::layout(strided(dst_strides.to_vec())));
        let serial = convert(&r, &data);

        // The independent oracle, for the same conversion.
        let (expected, base) = reference_convert(dims, src_strides, dst_strides, 1, 0, &data);
        assert_eq!(
            serial, expected,
            "{dims:?} {src_strides:?} → {dst_strides:?}"
        );
        assert_eq!(r.outputs()[0].byte_offset, base);

        // Items in reverse order.
        let mut out = AlignedDestination::for_destination(r.destinations(0)[0]).unwrap();
        for item in (0..r.work_item_count()).rev() {
            r.run_range(
                &[&[&data]],
                &mut [&mut [out.as_mut_slice()]],
                item..item + 1,
            )
            .unwrap();
        }
        assert_eq!(out.as_slice(), &serial[..], "reverse order");

        // And on the thread pool.
        let mut par = AlignedDestination::for_destination(r.destinations(0)[0]).unwrap();
        r.run_parallel(&[&[&data]], &mut [&mut [par.as_mut_slice()]])
            .unwrap();
        assert_eq!(par.as_slice(), &serial[..], "parallel");
    }
}

// ── Every execution path, on both destinations ────────────────────────────────

/// The engine has two execution paths — one copy per contiguous run, and one per element —
/// and two destinations: the safe slice, and the raw pointer the thread pool writes
/// through. Each combination is checked against the independent oracle here, so an
/// optimization that adds a path cannot land with only one of them covered (see the
/// maintenance notes in `remap.rs`).
#[test]
fn each_execution_path_matches_the_oracle_on_both_destinations() {
    // (name, source strides, target strides): the first two take the bulk-copy path, the
    // last two move single elements.
    let cases: &[(&str, &[i64], &[i64])] = &[
        ("bulk copy, padded rows", &[12, 1], &[16, 1]),
        ("bulk copy, reversed rows", &[12, 1], &[-12, 1]),
        ("per element, to column-major", &[12, 1], &[1, 8]),
        ("per element, strided last axis", &[12, 1], &[48, 4]),
    ];
    let dims = [8u64, 12];
    let count = 96usize;
    let data: Vec<u8> = (0..count).map(|i| (i * 7 % 251) as u8).collect();

    for (name, src_strides, dst_strides) in cases {
        let source = descriptor(
            ElementType::Uint8,
            dims.to_vec(),
            strided(src_strides.to_vec()),
            count as u64,
            0,
        );
        let r = ready(&source, &Target::layout(strided(dst_strides.to_vec())));
        let (expected, base) = reference_convert(&dims, src_strides, dst_strides, 1, 0, &data);
        assert_eq!(r.outputs()[0].byte_offset, base, "{name}");

        // The safe destination.
        assert_eq!(convert(&r, &data), expected, "{name}, serial");

        // Item by item, which is the same path driven by the caller.
        let mut stepped = AlignedDestination::for_destination(r.destinations(0)[0]).unwrap();
        for item in 0..r.work_item_count() {
            r.run_range(
                &[&[&data]],
                &mut [&mut [stepped.as_mut_slice()]],
                item..item + 1,
            )
            .unwrap();
        }
        assert_eq!(stepped.as_slice(), &expected[..], "{name}, by item");

        // The raw-pointer destination.
        #[cfg(feature = "rayon")]
        {
            let mut parallel = AlignedDestination::for_destination(r.destinations(0)[0]).unwrap();
            r.run_parallel(&[&[&data]], &mut [&mut [parallel.as_mut_slice()]])
                .unwrap();
            assert_eq!(parallel.as_slice(), &expected[..], "{name}, parallel");
        }
    }
}

// ── 3. Address arithmetic ─────────────────────────────────────────────────────

#[test]
fn a_byte_offset_above_the_signed_range_is_refused_not_wrapped() {
    // `byte_offset as i64` used to turn this into a negative base. It is now rejected as
    // part of the source's geometry, whatever the declared buffer size.
    let source = descriptor(
        ElementType::Uint8,
        vec![2, 2],
        LayoutDescriptor::RowMajor,
        u64::MAX,
        u64::MAX - 8,
    );
    let err = plan(&source, &Target::layout(LayoutDescriptor::ColMajor)).unwrap_err();
    assert!(
        matches!(
            err,
            Error::BeyondHostLimit { .. } | Error::Core(hurray_core::Error::AddressOverflow)
        ),
        "{err}"
    );
}

#[test]
fn a_stride_whose_byte_span_overflows_is_refused() {
    // Arithmetically representable in elements, not in bytes: float64 multiplies the span
    // by 8. The refusal must come from the arithmetic, not from a wrapped product.
    let handle = BufferHandle::new(
        64,
        MIN_BUFFER_ALIGNMENT,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )
    .unwrap();
    let source = TensorDescriptor::new(
        1,
        0,
        ElementType::Float64,
        Shape::new(vec![2, 2]).unwrap(),
        0,
        strided(vec![i64::MAX / 4, 1]),
        vec![handle],
        None,
        None,
        None,
        None,
    )
    .unwrap();
    assert!(plan(&source, &Target::layout(LayoutDescriptor::RowMajor)).is_err());
}

#[test]
fn a_destination_beyond_this_host_is_refused_at_plan_time() {
    // Arithmetically valid and physically impossible: the strides are representable, the
    // span is representable, and the destination would exceed the largest slice this host
    // can address. The distinction matters — this is not an overflow.
    let source = descriptor(
        ElementType::Uint8,
        vec![2, 2],
        LayoutDescriptor::RowMajor,
        4,
        0,
    );
    let err = plan(&source, &Target::layout(strided(vec![i64::MAX, 1]))).unwrap_err();
    match err {
        Error::BeyondHostLimit { what, limit, .. } => {
            assert!(what.contains("byte extent"), "{what}");
            assert_eq!(limit, isize::MAX as i128);
        }
        other => panic!("expected BeyondHostLimit, got {other}"),
    }
}

#[test]
fn a_negative_reach_below_the_declared_offset_is_refused() {
    // Strides reach eight elements below the base, and `byte_offset` is 4: the tensor
    // addresses bytes before its own buffer, which `strided.md` forbids.
    let source = descriptor(ElementType::Uint8, vec![3, 4], strided(vec![-4, 1]), 12, 4);
    let err = plan(&source, &Target::layout(LayoutDescriptor::RowMajor)).unwrap_err();
    assert!(
        matches!(
            err,
            Error::LayoutReachesBelowBuffer {
                byte_offset: 4,
                below: -4
            }
        ),
        "{err}"
    );
}

#[test]
// Miri treats a deliberately impossible allocation as resource exhaustion and stops the
// run, so this one case is excluded there; the rest of the file runs under Miri.
#[cfg_attr(miri, ignore)]
fn an_allocation_that_cannot_be_satisfied_is_an_error_not_an_abort() {
    // A plan may legitimately publish a destination no machine can provide. Asking for it
    // must fail, not abort the process — the size can come from a descriptor off a wire.
    let err = AlignedDestination::zeroed((isize::MAX as u64) / 2, 64).unwrap_err();
    assert!(matches!(err, Error::AllocationFailed { .. }), "{err}");
}

#[test]
fn extreme_but_valid_stride_arithmetic_still_converts() {
    // A padded row stride at the edge of what a small tensor can use: representable,
    // addressable, and correct against the oracle.
    let dims = [2u64, 2];
    let src_strides = [2i64, 1];
    let dst_strides = [1_000_000i64, 1];
    let data: Vec<u8> = vec![1, 2, 3, 4];
    let source = descriptor(
        ElementType::Uint8,
        dims.to_vec(),
        strided(src_strides.to_vec()),
        4,
        0,
    );
    let r = ready(&source, &Target::layout(strided(dst_strides.to_vec())));
    let (expected, base) = reference_convert(&dims, &src_strides, &dst_strides, 1, 0, &data);
    assert_eq!(r.destinations(0)[0].byte_size, 1_000_002);
    assert_eq!(r.outputs()[0].byte_offset, base);
    assert_eq!(convert(&r, &data), expected);
}

// ── Zero-sized dimensions, offsets, and the oracle on the ordinary cases ──────

#[test]
fn zero_sized_dimensions_convert_to_nothing_in_every_direction() {
    for dims in [vec![0u64, 4], vec![3, 0], vec![0, 0], vec![0]] {
        for target in [
            LayoutDescriptor::ColMajor,
            LayoutDescriptor::RowMajor,
            strided(vec![1; dims.len()]),
        ] {
            let source = descriptor(
                ElementType::Float32,
                dims.clone(),
                LayoutDescriptor::RowMajor,
                0,
                0,
            );
            // An empty tensor has no two indices to collide, so even strides that would
            // alias on a non-empty shape are accepted — and move nothing.
            let r = ready(&source, &Target::layout(target.clone()));
            assert_eq!(r.work_item_count(), 0, "{dims:?} → {target:?}");
            match r.destinations(0) {
                // Some of these targets address the source's own bytes, which makes the
                // plan an alias; the rest produce a buffer, and it is empty.
                [] => {
                    let report = r.run(&[&[&[]]], &mut [&mut []]).unwrap();
                    assert_eq!(report.bytes_written(), 0);
                }
                [destination] => {
                    assert_eq!(destination.byte_size, 0, "{dims:?} → {target:?}");
                    assert_eq!(convert(&r, &[]), Vec::<u8>::new());
                }
                other => panic!("one output buffer expected, got {}", other.len()),
            }
        }
    }
}

#[test]
fn a_nonzero_source_offset_inside_a_padded_allocation_reads_the_right_bytes() {
    // The tensor sits 64 bytes into its buffer, with trailing padding after it. Both are
    // the caller's business; the conversion must read exactly the tensor.
    let dims = [3u64, 4];
    let mut data = vec![0xEEu8; 128];
    for (i, slot) in data[64..76].iter_mut().enumerate() {
        *slot = i as u8;
    }
    let source = descriptor(
        ElementType::Uint8,
        dims.to_vec(),
        LayoutDescriptor::RowMajor,
        128,
        64,
    );
    let r = ready(&source, &Target::layout(LayoutDescriptor::ColMajor));
    let (expected, _) = reference_convert(&dims, &[4, 1], &[1, 3], 1, 64, &data);
    assert_eq!(convert(&r, &data), expected);
    assert_eq!(
        convert(&r, &data),
        vec![0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
    );
}

#[test]
fn the_oracle_agrees_on_every_element_width() {
    // Element size enters both the address arithmetic and the disjointness argument, so it
    // is swept rather than sampled.
    for (ty, eb) in [
        (ElementType::Uint8, 1usize),
        (ElementType::Float16, 2),
        (ElementType::Float32, 4),
        (ElementType::Float64, 8),
        (ElementType::Complex128, 16),
    ] {
        let dims = [3u64, 4];
        let bytes = 12 * eb as u64;
        let data: Vec<u8> = (0..bytes).map(|i| (i % 251) as u8).collect();
        let handle = BufferHandle::new(
            bytes,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )
        .unwrap();
        let source = TensorDescriptor::new(
            1,
            0,
            ty,
            Shape::new(dims.to_vec()).unwrap(),
            0,
            LayoutDescriptor::RowMajor,
            vec![handle],
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let r = ready(&source, &Target::layout(LayoutDescriptor::ColMajor));
        let (expected, _) = reference_convert(&dims, &[4, 1], &[1, 3], eb, 0, &data);
        assert_eq!(convert(&r, &data), expected, "{ty}");
    }
}

#[test]
fn a_report_combines_without_overflowing() {
    // A caller driving `run_range` in a loop combines many reports; the counts saturate
    // rather than panicking in a debug build.
    let mut report = RunReport::default();
    for _ in 0..4 {
        report = report.combine(report);
    }
    assert_eq!(report.elements_written(), 0);
}
