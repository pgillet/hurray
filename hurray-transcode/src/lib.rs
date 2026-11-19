//! # hurray-transcode
//!
//! Representation conversion for the hurray tensor interchange format: the crate that
//! converts a tensor's **representation** and nothing else.
//!
//! ## What belongs here
//!
//! One rule decides membership, and it is the rule a future addition is measured against:
//!
//! > An operation belongs in this crate **iff** it changes only the tensor's
//! > representation, never its logical value — the mapping from index to value is
//! > preserved exactly, or with a precision loss the **target representation itself
//! > declares** — and the conversion is fully determined by the source and target
//! > descriptors.
//!
//! "Logical value" means the value *the format defines*: the element stored at an index
//! for a dense tensor, the dequantization arithmetic's result for a quantized one, the
//! `CombineOp` combination for an overlay composite. So this is not a compute library and
//! cannot become one: a matmul changes the value, `arange` has no source descriptor, and
//! a reshape moves no byte and is therefore a descriptor change rather than a conversion.
//!
//! The decided surface is [ADR-044]; which conversions this version actually performs is
//! the [conversion matrix].
//!
//! ## Three shapes
//!
//! | Shape | Reads | Produces |
//! |---|---|---|
//! | **plan** ([`plan`]) | descriptors only | an output descriptor, output buffer handles, and a partition of the work |
//! | **survey** ([`SurveyPlan::survey`]) | payload bytes | facts; never a tensor |
//! | **run** ([`ReadyPlan::run`]) | payload bytes | payload bytes |
//!
//! The caller allocates every destination, and a plan publishes the sizes before any byte
//! moves — so a conversion whose output would be too large can be refused against a
//! budget rather than discovered.
//!
//! ## What this version converts
//!
//! Between the layouts with an affine stride form — row-major, column-major and strided —
//! for whole-byte element types, including the two degenerate strides that
//! `docs/spec/layouts/strided.md` mandates support for: a **negative** stride (a reversed
//! axis) and a **zero** stride (a broadcast). Materialising a broadcast is a plain
//! conversion; collapsing a repeated axis back into one is admissible only if the data
//! repeats, so it is reached through a survey that can refuse.
//!
//! Everything else — tiled, Morton and Hilbert layouts, the sparse families, element
//! casting, quantization, sharding, composites, byte order and device staging — is in the
//! crate's scope and not implemented yet. Each is refused by name rather than silently
//! mishandled.
//!
//! ## Example
//!
//! Converting a row-major tensor to column-major:
//!
//! ```
//! use hurray_core::{
//!     BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode,
//!     MIN_BUFFER_ALIGNMENT, descriptor::TensorDescriptor,
//! };
//! use hurray_transcode::{plan, AlignedDestination, Planned, Target};
//!
//! // A 2×3 uint8 tensor holding 1..=6 in row-major order.
//! let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
//! let shape = Shape::new(vec![2, 3]).unwrap();
//! let handle = BufferHandle::new(
//!     6, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced,
//! ).unwrap();
//! let source = TensorDescriptor::new(
//!     1, 0, ElementType::Uint8, shape, 0, LayoutDescriptor::RowMajor, vec![handle],
//!     None, None, None, None,
//! ).unwrap();
//!
//! let Planned::Ready(ready) = plan(&source, &Target::layout(LayoutDescriptor::ColMajor))?
//! else { panic!("a layout permutation needs no survey") };
//!
// The plan says how much memory to allocate, and at what alignment, before
//! // anything is converted.
//! let mut out = AlignedDestination::for_destination(ready.destinations(0)[0])?;
//! let report = ready.run(&[&[&data]], &mut [&mut [out.as_mut_slice()]])?;
//!
//! assert_eq!(report.elements_written(), 6);
//! assert_eq!(out.as_slice(), &[1, 4, 2, 5, 3, 6]); // column-major order
//! # Ok::<(), hurray_transcode::Error>(())
//! ```
//!
//! ## Feature flags
//!
//! | Feature | Effect |
//! |---------|--------|
//! | `rayon` | Adds [`ReadyPlan::run_parallel`], which fans a plan's work items over `rayon`'s thread pool. Off by default: a default build cannot spawn a thread, so it stays callable from an async context. |
//!
//! [ADR-044]: https://github.com/pgillet/hurray/blob/main/docs/adr/ADR-044-hurray-transcode-crate.md
//! [conversion matrix]: https://github.com/pgillet/hurray/blob/main/docs/impl/transcode-matrix.md

// The modules are private and everything public is re-exported at the root: the crate's
// surface is one function and a handful of types, so a flat root is the better API — and
// `plan` can then be the function it should be, rather than a module shadowing it.
mod aligned;
mod dest;
mod error;
#[cfg(feature = "rayon")]
mod parallel;
mod plan;
mod remap;

pub use aligned::AlignedDestination;
pub use error::{Error, Result};
pub use plan::{
    plan, BufferPlan, Destination, Planned, ReadyPlan, RunReport, SurveyPlan, SurveyQuestion,
    Target, TensorBuffers, TensorDestinations,
};
