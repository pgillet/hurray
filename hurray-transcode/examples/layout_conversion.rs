//! Converting a tensor's memory layout, with `hurray-transcode`.
//!
//! Run with:
//!
//! ```sh
//! cargo run -p hurray-transcode --example layout_conversion
//! ```

use hurray_core::descriptor::TensorDescriptor;
use hurray_core::layout::StridedLayout;
use hurray_core::{
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode, MIN_BUFFER_ALIGNMENT,
};
use hurray_transcode::{plan, AlignedDestination, BufferPlan, Planned, SurveyQuestion, Target};

/// Builds a descriptor for a single-buffer host tensor.
fn descriptor(
    dims: Vec<u64>,
    layout: LayoutDescriptor,
    byte_size: u64,
    byte_offset: u64,
) -> hurray_transcode::Result<TensorDescriptor> {
    let handle = BufferHandle::new(
        byte_size,
        MIN_BUFFER_ALIGNMENT,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )?;
    Ok(TensorDescriptor::new(
        1,
        0,
        ElementType::Uint8,
        Shape::new(dims)?,
        byte_offset,
        layout,
        vec![handle],
        None,
        None,
        None,
        None,
    )?)
}

fn main() -> hurray_transcode::Result<()> {
    // A 3×4 tensor holding 0..12 in row-major order.
    let data: Vec<u8> = (0u8..12).collect();
    let source = descriptor(vec![3, 4], LayoutDescriptor::RowMajor, 12, 0)?;

    // ── 1. Row-major to column-major ─────────────────────────────────────────
    //
    // Planning reads descriptors only. The plan publishes the output descriptor and the
    // size of every buffer the caller must allocate *before* a byte moves.
    let Planned::Ready(ready) = plan(&source, &Target::layout(LayoutDescriptor::ColMajor))? else {
        unreachable!("a layout permutation needs no survey");
    };
    let destination = ready.destinations(0)[0];
    println!(
        "row-major → column-major: {} bytes at {}-byte alignment, {} work item(s)",
        destination.byte_size,
        destination.alignment,
        ready.work_item_count()
    );

    let mut out = AlignedDestination::for_destination(destination)?;
    let report = ready.run(&[&[&data]], &mut [&mut [out.as_mut_slice()]])?;
    println!(
        "  wrote {} elements: {:?}",
        report.elements_written(),
        out.as_slice()
    );

    // ── 2. A target that implies the same bytes ──────────────────────────────
    //
    // `[4, 1]` is how a strided layout spells row-major for this shape, so there is
    // nothing to move: the plan publishes the output buffer as an alias of the input's,
    // and the run writes nothing. A layout-negotiating server reaches this case routinely.
    let same = LayoutDescriptor::Strided(StridedLayout::new(vec![4, 1]));
    let Planned::Ready(aliased) = plan(&source, &Target::layout(same))? else {
        unreachable!()
    };
    assert_eq!(
        aliased.buffer_plans(0),
        &[BufferPlan::Alias {
            input: 0,
            buffer: 0
        }]
    );
    let report = aliased.run(&[&[&data]], &mut [&mut []])?;
    println!(
        "strided [4,1] is the same bytes: {} work item(s), {} bytes written",
        aliased.work_item_count(),
        report.bytes_written()
    );

    // ── 3. A reversed axis ───────────────────────────────────────────────────
    //
    // A negative stride reverses a dimension, so element [0,0] is no longer the lowest
    // address the tensor reads: the plan places the output's `byte_offset` inside its
    // buffer accordingly.
    let reversed = LayoutDescriptor::Strided(StridedLayout::new(vec![-4, 1]));
    let Planned::Ready(ready) = plan(&source, &Target::layout(reversed))? else {
        unreachable!()
    };
    let mut out = AlignedDestination::for_destination(ready.destinations(0)[0])?;
    ready.run(&[&[&data]], &mut [&mut [out.as_mut_slice()]])?;
    println!(
        "row-major → reversed rows: byte_offset {}, buffer {:?}",
        ready.outputs()[0].byte_offset,
        out.as_slice()
    );

    // ── 4. Materialising a broadcast ─────────────────────────────────────────
    //
    // A zero stride means every index along that axis reads one element. Materialising it
    // expands the data by a factor the descriptor states — the one conversion whose
    // output can be arbitrarily larger than its input, which is why the plan publishes
    // the size first.
    let broadcast = descriptor(
        vec![3, 4],
        LayoutDescriptor::Strided(StridedLayout::new(vec![0, 1])),
        4,
        0,
    )?;
    let row: Vec<u8> = vec![7, 8, 9, 10];
    let Planned::Ready(ready) = plan(&broadcast, &Target::layout(LayoutDescriptor::RowMajor))?
    else {
        unreachable!()
    };
    println!(
        "broadcast: 4 bytes in, {} bytes out",
        ready.destinations(0)[0].byte_size
    );
    let mut out = AlignedDestination::for_destination(ready.destinations(0)[0])?;
    ready.run(&[&[&row]], &mut [&mut [out.as_mut_slice()]])?;
    println!("  materialised: {:?}", out.as_slice());

    // ── 5. The reverse needs the data's permission ───────────────────────────
    //
    // Collapsing a repeated axis back into a broadcast preserves the value only if the
    // data actually repeats, which no descriptor records. So planning returns a survey,
    // and the survey can refuse.
    let full = descriptor(vec![3, 4], LayoutDescriptor::RowMajor, 12, 0)?;
    let collapse = Target::layout(LayoutDescriptor::Strided(StridedLayout::new(vec![0, 1])));
    let Planned::NeedsSurvey(survey) = plan(&full, &collapse)? else {
        unreachable!("collapsing an axis is never decidable from descriptors alone");
    };
    println!("collapse asks the data: {:?}", survey.question());
    assert!(matches!(
        survey.question(),
        SurveyQuestion::BroadcastRepetition { .. }
    ));

    // Three identical rows: admissible.
    let repeating: Vec<u8> = vec![7, 8, 9, 10, 7, 8, 9, 10, 7, 8, 9, 10];
    let ready = survey.survey(&[&[&repeating]])?;
    let mut out = AlignedDestination::for_destination(ready.destinations(0)[0])?;
    ready.run(&[&[&repeating]], &mut [&mut [out.as_mut_slice()]])?;
    println!("  repeating data collapses to {:?}", out.as_slice());

    // One row differs: refused, by name, with the index that disagrees.
    match survey.survey(&[&[&data]]) {
        Err(e) => println!("  the original data does not: {e}"),
        Ok(_) => unreachable!("0..12 does not repeat"),
    }

    Ok(())
}
