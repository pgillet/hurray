# Converting a Tensor's Layout

A consumer that reads row-major gets handed a column-major tensor. A kernel wants its
weights tiled. A server negotiates a layout a client did not send. All three need the same
thing: the same logical tensor, in a different memory layout.

[`hurray-transcode`](https://docs.rs/hurray-transcode) is the crate that does that, and
only that. One rule decides what belongs in it:

> An operation belongs in `hurray-transcode` **iff** it changes only the tensor's
> representation, never its logical value — the mapping from index to value is preserved
> exactly, or with a precision loss the **target representation itself declares** — and
> the conversion is fully determined by the source and target descriptors.

So a matmul is not here, and never will be. Neither is a reshape: a reshape moves no byte,
which makes it a descriptor change rather than a conversion. What *is* here is everything
the descriptor permits to differ between two descriptions of one logical tensor — layout,
element type, quantization scheme, alignment, placement. Which of those work today is the
[conversion matrix](../impl/transcode-matrix.md); this page covers the layout family.

## Plan first, then move bytes

Every conversion is two steps, and the split is the point. **Planning** reads descriptors
only, and publishes the output descriptor and the size of every buffer the caller must
allocate. **Running** moves the bytes.

```rust
use hurray_core::descriptor::TensorDescriptor;
use hurray_core::{
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode,
    MIN_BUFFER_ALIGNMENT,
};
use hurray_transcode::{plan, AlignedDestination, Planned, Target};

fn main() -> hurray_transcode::Result<()> {
    // A 3×4 uint8 tensor holding 0..12, row-major.
    let data: Vec<u8> = (0u8..12).collect();
    let handle = BufferHandle::new(
        12, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced,
    )?;
    let source = TensorDescriptor::new(
        1, 0, ElementType::Uint8, Shape::new(vec![3, 4])?, 0,
        LayoutDescriptor::RowMajor, vec![handle],
        None, None, None, None,
    )?;

    // Planning touches no payload byte.
    let Planned::Ready(ready) = plan(&source, &Target::layout(LayoutDescriptor::ColMajor))?
    else { unreachable!("a layout permutation needs no survey") };

    // The output descriptor is final before anything is converted, and so is the size.
    assert_eq!(ready.outputs()[0].layout, LayoutDescriptor::ColMajor);
    assert_eq!(ready.destinations(0)[0].byte_size, 12);

    let mut out = AlignedDestination::for_destination(ready.destinations(0)[0])?;
    let report = ready.run(&[&[&data]], &mut [&mut [out.as_mut_slice()]])?;

    assert_eq!(report.elements_written(), 12);
    assert_eq!(out.as_slice(), &[0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]);
    Ok(())
}
```

Publishing the size before the work is not ceremony. Materialising a broadcast expands the
data by a factor the descriptor states, so a conversion's output can be far larger than
its input — a server honouring `ALLOW_TRANSCODE` can refuse it against a budget and return
`TRANSCODE_LIMIT_EXCEEDED` instead of allocating it. The budget is the caller's; the number
is the plan's.

## The caller allocates every destination

The crate allocates no output. That is what makes the zero-copy destinations possible:
convert straight into a NumPy array, a pinned staging buffer, an mmap'd region, or a
`Bytes` the I/O layer will write.

It also means the destination must satisfy the alignment the plan published.
`buffer-protocol.md § Minimum Alignment` makes a buffer's declared alignment a promise —
every non-empty buffer is at least 64-byte aligned, and a consumer MAY rely on that for
aligned SIMD loads — and `Vec<u8>` does not keep it. A run refuses a misaligned
destination rather than producing a tensor whose descriptor lies, so
`AlignedDestination` is there for callers that have no aligned allocation of their own.

## When there is nothing to move

Two layouts can describe the same bytes. `Strided([4, 1])` is how the strided layout
spells row-major for a 3×4 tensor, and a layout-negotiating server meets that case
constantly. The plan says so per buffer: the output is an **alias** of the input's buffer,
and the run writes nothing.

```rust
# use hurray_core::descriptor::TensorDescriptor;
# use hurray_core::layout::StridedLayout;
# use hurray_core::{
#     BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode,
#     MIN_BUFFER_ALIGNMENT,
# };
use hurray_transcode::{plan, BufferPlan, Planned, Target};

fn main() -> hurray_transcode::Result<()> {
#   let data: Vec<u8> = (0u8..12).collect();
#   let handle = BufferHandle::new(
#       12, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced,
#   )?;
#   let source = TensorDescriptor::new(
#       1, 0, ElementType::Uint8, Shape::new(vec![3, 4])?, 0,
#       LayoutDescriptor::RowMajor, vec![handle],
#       None, None, None, None,
#   )?;
    let same_bytes = LayoutDescriptor::Strided(StridedLayout::new(vec![4, 1]));
    let Planned::Ready(ready) = plan(&source, &Target::layout(same_bytes))? else {
        unreachable!()
    };

    // Nothing to allocate, nothing to move — a supported conversion with an empty body.
    assert_eq!(ready.buffer_plans(0), &[BufferPlan::Alias { input: 0, buffer: 0 }]);
    assert_eq!(ready.destinations(0), &[]);
    assert_eq!(ready.work_item_count(), 0);

    let report = ready.run(&[&[&data]], &mut [&mut []])?;
    assert_eq!(report.bytes_written(), 0);
    Ok(())
}
```

An aliased output buffer **shares memory with its input**, so the output tensor cannot
outlive the input's buffers. The plan publishes the arm per buffer precisely so that is
visible before the run rather than surprising afterwards; a caller that needs an
independent copy asks for one.

This is not the same as asking for something the crate does not do. A reshape or a version
bump is refused outright, because there is no representation change to perform. An
all-alias plan is a conversion that succeeded and had no work in it.

## Reversed axes and broadcasts

`strided.md` makes negative and zero strides valid and requires an implementation to
support them, which puts two unusual conversions inside "strided to row-major".

A **negative** stride reverses an axis. It is the one case where element `[0, …, 0]` is not
the lowest address the tensor reads, so when it is the *target*, the plan places the
output's `byte_offset` inside its buffer:

```rust
# use hurray_core::descriptor::TensorDescriptor;
# use hurray_core::layout::StridedLayout;
# use hurray_core::{
#     BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode,
#     MIN_BUFFER_ALIGNMENT,
# };
use hurray_transcode::{plan, AlignedDestination, Planned, Target};

fn main() -> hurray_transcode::Result<()> {
    let data: Vec<u8> = (0u8..12).collect();
#   let handle = BufferHandle::new(
#       12, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced,
#   )?;
#   let source = TensorDescriptor::new(
#       1, 0, ElementType::Uint8, Shape::new(vec![3, 4])?, 0,
#       LayoutDescriptor::RowMajor, vec![handle],
#       None, None, None, None,
#   )?;
    let reversed = LayoutDescriptor::Strided(StridedLayout::new(vec![-4, 1]));
    let Planned::Ready(ready) = plan(&source, &Target::layout(reversed))? else {
        unreachable!()
    };

    // Logical element [0,0] sits eight elements in: row 2 is physically first.
    assert_eq!(ready.outputs()[0].byte_offset, 8);

    let mut out = AlignedDestination::for_destination(ready.destinations(0)[0])?;
    ready.run(&[&[&data]], &mut [&mut [out.as_mut_slice()]])?;
    assert_eq!(out.as_slice(), &[8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]);
    Ok(())
}
```

A **zero** stride is a broadcast: every index along that axis reads one stored element, and
the data is not physically replicated. Materialising it writes the repeats out.

Going the other way — collapsing a repeated axis back into a broadcast — is value-preserving
only if the data actually repeats, and no descriptor records that. This is the one case on
this page where the conversion's admissibility is not a function of the descriptors, so
planning hands back a **survey** instead of a plan, and the survey can refuse.

```rust
# use hurray_core::descriptor::TensorDescriptor;
# use hurray_core::layout::StridedLayout;
# use hurray_core::{
#     BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode,
#     MIN_BUFFER_ALIGNMENT,
# };
use hurray_transcode::{plan, AlignedDestination, Error, Planned, SurveyQuestion, Target};

fn main() -> hurray_transcode::Result<()> {
#   let handle = BufferHandle::new(
#       12, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced,
#   )?;
#   let source = TensorDescriptor::new(
#       1, 0, ElementType::Uint8, Shape::new(vec![3, 4])?, 0,
#       LayoutDescriptor::RowMajor, vec![handle],
#       None, None, None, None,
#   )?;
    let collapsed = LayoutDescriptor::Strided(StridedLayout::new(vec![0, 1]));
    let Planned::NeedsSurvey(survey) = plan(&source, &Target::layout(collapsed))? else {
        unreachable!("no descriptor says whether the data repeats")
    };
    assert_eq!(
        survey.question(),
        &SurveyQuestion::BroadcastRepetition { axes: vec![0] },
    );

    // Three identical rows: the collapse is exact, and the output is one row.
    let repeating: Vec<u8> = vec![7, 8, 9, 10, 7, 8, 9, 10, 7, 8, 9, 10];
    let ready = survey.survey(&[&[&repeating]])?;
    let mut out = AlignedDestination::for_destination(ready.destinations(0)[0])?;
    ready.run(&[&[&repeating]], &mut [&mut [out.as_mut_slice()]])?;
    assert_eq!(out.as_slice(), &[7, 8, 9, 10]);

    // Rows that differ: refused, with the index that disagrees. Nothing is wrong with
    // the data — the remedy is a different target, not a repair.
    let varied: Vec<u8> = (0u8..12).collect();
    match survey.survey(&[&[&varied]]) {
        Err(Error::RepetitionNotHeld { dim, index }) => {
            assert_eq!(dim, 0);
            assert_eq!(index, vec![1, 0]);
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
    Ok(())
}
```

A plan that came from `Planned::Ready` is a function of descriptors alone, so it may be run
any number of times on any buffers matching those descriptors. A plan that came from a
survey carries a fact about *particular bytes*, and is valid for those bytes only — mutating
them between the survey and the run voids the plan, and the crate cannot detect that it
happened: a buffer handle carries no pointer, and a plan keeps no digest of what it read.
It is a documented obligation rather than a type, because nothing in Rust can enforce it
for memory owned by NumPy or by a C caller.

## Work items, and parallelism at the edge

A plan partitions its work into items that partition the **output**, so running every item
exactly once, in any order, on any number of threads, produces byte-identical output to one
serial run. An input may be read by many items — a broadcast reads one element once per
repeat — which is why the invariant is about writes.

A default build of the crate cannot spawn a thread. That is deliberate: `hurray-io` is
async, and a crate that quietly called into a thread pool could not be used from inside a
streaming write. A caller that wants the cores asks for them:

```rust,ignore
// With the `rayon` feature: fans the plan's work items over the thread pool.
let report = ready.run_parallel(&[&[&data]], &mut [&mut [out.as_mut_slice()]])?;

// From async code, where blocking the executor is the thing to avoid:
let report = tokio::task::spawn_blocking(move || {
    ready.run_parallel(&[&[&data]], &mut [&mut [out.as_mut_slice()]])
}).await??;
```

(Not compiled here: the first line needs the `rayon` feature and the second a Tokio
runtime.)

A caller that wants a different granularity drives the partition itself with
`run_range(.., item..item + 1)`, which is what makes transcoding inside a streaming write
possible later.

## What it refuses, and why that is the useful part

Every refusal names what was refused. The ones you can reach from a layout conversion
today:

| Request | Outcome |
|---|---|
| A tiled, Morton, Hilbert, sparse, block-paged or composite layout | Unsupported: in the crate's scope, not implemented yet |
| A sub-byte element type (`int4`, `bool`, `float6_*`) | Unsupported: moving one element is a bit-field edit, which is the packing conversion |
| A quantized source | Unsupported: the scheme's metadata buffers belong to the quantization conversions |
| A target whose strides map two logical indices to one element, such as `[1, 1]` over shape `[2, 2]` | Refused as `AliasingTarget`: writing the tensor through it would lose a value (see below) |
| A target that is injective but not *provably* so, such as `[2, 3, 4]` over shape `[2, 2, 2]` | Refused as `TargetNotProvablyDistinct`: a limitation of the test, not a property of the layout (see below) |
| A tensor whose shape still carries the dynamic-dimension sentinel | Refused: a plan is sizes, and resolving the sentinel moves no byte |
| A device-resident buffer | Refused: device support is movement-only, and staging is a separate API |
| A reshape, a transpose as annotation, a version bump | Not expressible: a target names a representation, never a shape |
| A source buffer too small for the layout it declares | Refused at plan time, from the layout's own addressing span |
| A destination shorter or less aligned than the plan published | Refused before anything is written |

**Aliasing targets deserve their own paragraph, because the obvious test for them is
wrong.** A zero stride is not the only way for a layout to send two indices to one element:
strides `[1, 1]` over shape `[2, 2]` address elements `0, 1, 1, 2`, so `[0, 1]` and `[1, 0]`
share one. Written through, that layout loses a value — and in a parallel run the two writes
can land in different work items, which is a data race rather than merely a wrong answer. So
a plan admits a target only on a *proof* that its strides address every element of the
output cover distinctly, which is what makes § 6's "work items partition the output" true
rather than hoped for.

That proof is exact where at most two axes can move: a collision is then a solution of
`|s_a| · x = |s_b| · y` inside the index box, whose minimal solution is written down with a
`gcd`, so an interleaved target like `[2, 3]` over shape `[3, 2]` is admitted and `[2, 3]`
over shape `[4, 3]` is refused as aliasing. Beyond two movable axes it falls back to a
sufficient condition — sorted by magnitude, each stride must exceed the total span of all
smaller ones — which admits every dense layout, every permutation of one, and every padded
variant. **The two refusals are separate errors**, because they have different remedies:

- `AliasingTarget` is proved and permanent. The destination genuinely sends two indices to
  one element, so it can represent faithfully only those tensors whose values already agree
  there — and a plan sees descriptors, never values, so the refusal cannot be conditional on
  the data. Retrying it against a later version will not help.
- `TargetNotProvablyDistinct` is this crate's test giving up, which now takes three or more
  movable axes. `[2, 3, 4]` over shape `[2, 2, 2]` addresses `0, 4, 3, 7, 2, 6, 5, 9` and
  collides nowhere, yet comes back unproved. Such a target is unsupported rather than
  impossible, and a stronger proof would admit it with no change to the format or to the
  API — which is exactly what happened to the two-axis case this paragraph used to be
  written on.

Reading an aliasing layout is unaffected: a broadcast source is exactly that, and
materialising it duplicates the value, which is what the layout means.

A run that returns an error leaves every produced
destination **unspecified but initialised** — the caller handed over `&mut [u8]`, so there
is no uninitialised memory — while inputs and aliased buffers are never written at all, and
the plan stays valid for a retry.

## From Python

There is no Python surface for transcoding yet. The parity rule in this project is written
against `hurray-core` and `hurray-io`, and transcoding is the kind of mechanism a NumPy
user wants directly — so a Python surface is expected, as its own piece of work once the
Rust API has stabilised. Until then, a Python caller converts by handing buffers to Rust,
or waits.

## See also

- [Transcode Matrix](../impl/transcode-matrix.md) — every conversion, its status, and the
  reason for each permanent refusal
- [Layer 3: Layout Descriptors](layer-3-layout-descriptors.md) — the layouts themselves
- [Multi-Buffer Tensors](multi-buffer-tensors.md) — why a quantized or sparse tensor has
  more than one buffer, and what a conversion of one has to move
