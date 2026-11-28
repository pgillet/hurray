//! Plan, survey, run: the three shapes every data-touching operation takes.
//!
//! A plan is built from descriptors alone, publishes its output descriptor and its output
//! buffer handles *before any byte moves*, and is the only way to reach a run. Where the
//! admissibility of a conversion depends on the data, planning returns
//! [`Planned::NeedsSurvey`] instead, and the survey is what produces the [`ReadyPlan`] —
//! so "ran a plan whose precondition was never established" is unrepresentable rather
//! than an error variant (ADR-044 § 4).

use std::ops::Range;

use hurray_core::descriptor::TensorDescriptor;
use hurray_core::{BufferHandle, DeviceTag, LayoutDescriptor, SyncMode};

use crate::remap::{destination_geometry, element_bytes, source_geometry, Remap};
use crate::{Error, Result};

/// The buffers of one input tensor, in its descriptor's buffer-table order.
pub type TensorBuffers<'a> = &'a [&'a [u8]];

/// The destinations for one output tensor: one per [`BufferPlan::Produce`] the plan
/// published, in buffer-table order. Aliased buffers take no destination.
pub type TensorDestinations<'d, 'b> = &'d mut [&'b mut [u8]];

/// What the caller asks for: the representation the output should have.
///
/// A target is **not** a [`TensorDescriptor`], and the reason is structural rather than
/// stylistic: a descriptor carries its buffer table, and the buffer sizes are what the
/// plan exists to compute. Asking a caller to supply them in order to request a
/// conversion would invert the dependency. A target carries only what may differ between
/// two descriptions of the same logical value — which is also what makes corollary 1
/// unrepresentable: there is no field here for a shape, a rank or a version, so a reshape
/// cannot be requested through this type at all (ADR-044 § 1, § 14f).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Target {
    layout: LayoutDescriptor,
}

impl Target {
    /// A target that changes the memory layout and nothing else.
    ///
    /// # Which layouts are accepted as a destination
    ///
    /// Every layout whose stride magnitudes are **strictly nested** — row-major,
    /// column-major, any axis permutation of either, and any padded or reversed variant of
    /// those — and, where at most **two axes can move** at the shape being planned, every
    /// injective stride vector, interleaved or not: that case is decided exactly rather
    /// than approximated. That is the set a conversion can write into while provably
    /// addressing each element once, and it is the whole accepted set: a target outside it
    /// is refused at plan time with [`Error::AliasingTarget`] or
    /// [`Error::TargetNotProvablyDistinct`] (ADR-045 § 1). A zero stride in the *target* is the one deliberate exclusion — that
    /// is a broadcast collapse, which is reached through the survey rather than by writing
    /// the repeats.
    ///
    /// The check cannot move here, to construction, and the reason is not convenience:
    /// distinctness is a property of the strides **and** the shape — `[1, 1]` aliases at
    /// shape `[2, 2]` and is merely unproved at `[3, 3]` — and a `Target` carries no shape.
    /// So the earliest point at which the question is even well-posed is `plan()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::LayoutDescriptor;
    /// use hurray_transcode::Target;
    ///
    /// let target = Target::layout(LayoutDescriptor::ColMajor);
    /// assert_eq!(target.requested_layout(), &LayoutDescriptor::ColMajor);
    /// ```
    pub fn layout(layout: LayoutDescriptor) -> Self {
        Self { layout }
    }

    /// The requested layout.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{LayoutDescriptor, layout::StridedLayout};
    /// use hurray_transcode::Target;
    ///
    /// let target = Target::layout(LayoutDescriptor::Strided(StridedLayout::new(vec![1, 3])));
    /// assert_eq!(target.requested_layout().tag(), 0x03);
    /// ```
    pub fn requested_layout(&self) -> &LayoutDescriptor {
        &self.layout
    }
}

/// How one output buffer comes about.
///
/// Publishing this per buffer is what makes the cost of a conversion visible before it
/// runs: an [`BufferPlan::Alias`] shares memory with an input, so the output tensor
/// cannot outlive that input's buffers, and a caller that needs an independent copy must
/// ask for one (ADR-044 § 5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferPlan {
    /// The caller allocates this buffer and the run fills it.
    Produce {
        /// Bytes the buffer must hold.
        byte_size: u64,
        /// Alignment the buffer's base address must meet.
        alignment: u32,
    },
    /// Byte-identical to a buffer of an input tensor; nothing is written.
    ///
    /// Names a tensor and a buffer within it rather than a flat index, because inputs
    /// nest per tensor: a flat index would make the plan's meaning depend on how the
    /// caller flattened its inputs, and would be ambiguous for a many-to-one conversion.
    Alias {
        /// Index of the input tensor.
        input: usize,
        /// Index of the buffer within that input tensor.
        buffer: usize,
    },
}

/// A buffer the caller must allocate, as `run` expects it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Destination {
    /// Index of this buffer in the output tensor's buffer table.
    pub buffer: usize,
    /// Bytes the destination must hold.
    pub byte_size: u64,
    /// Alignment the destination's base address must meet.
    pub alignment: u32,
}

/// What a run did, beyond what the output descriptor can carry.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct RunReport {
    elements: u64,
    bytes: u64,
}

impl RunReport {
    pub(crate) fn new(elements: u64, bytes: u64) -> Self {
        Self { elements, bytes }
    }

    /// Elements written. Zero for an all-alias plan, which moves nothing.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_transcode::RunReport;
    ///
    /// assert_eq!(RunReport::default().elements_written(), 0);
    /// ```
    pub fn elements_written(&self) -> u64 {
        self.elements
    }

    /// Bytes written.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_transcode::RunReport;
    ///
    /// assert_eq!(RunReport::default().bytes_written(), 0);
    /// ```
    pub fn bytes_written(&self) -> u64 {
        self.bytes
    }

    /// Combines two reports, for a caller that drove the work items itself.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_transcode::RunReport;
    ///
    /// let total = RunReport::default().combine(RunReport::default());
    /// assert_eq!(total.elements_written(), 0);
    /// ```
    pub fn combine(self, other: Self) -> Self {
        // Saturating: a caller driving `run_range` in a loop can combine arbitrarily many
        // reports, and a count that stops being exact is a better outcome than a panic in
        // a debug build. A single run cannot reach the ceiling — its totals are bounded by
        // the plan's byte extent.
        Self {
            elements: self.elements.saturating_add(other.elements),
            bytes: self.bytes.saturating_add(other.bytes),
        }
    }
}

/// The outcome of planning: either the plan is ready, or the data must be consulted first.
///
/// Deliberately **not** `#[non_exhaustive]`: there are two shapes and there will be two,
/// because both arms converge on one [`ReadyPlan`] with one run (ADR-044 § 4). A caller
/// is meant to match both, and forcing a wildcard arm on every one of them would cost
/// more than the breaking change a third arm would be — if a third arm were ever coming.
#[derive(Debug, Clone)]
pub enum Planned {
    /// Everything the conversion needs is known from the descriptors.
    Ready(ReadyPlan),
    /// The conversion is admissible only if the data says so. [`SurveyPlan::survey`] is
    /// the only way to a [`ReadyPlan`] from here.
    NeedsSurvey(SurveyPlan),
}

/// What a survey must establish before a conversion can run.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SurveyQuestion {
    /// Does the data repeat along each of these axes? Collapsing an axis to a broadcast
    /// stride preserves the value only if it does (ADR-044 § 14b).
    BroadcastRepetition {
        /// The axes the target collapses.
        axes: Vec<u32>,
    },
}

/// A plan whose admissibility depends on the bytes.
#[derive(Debug, Clone)]
pub struct SurveyPlan {
    plan: ReadyPlan,
    question: SurveyQuestion,
}

impl SurveyPlan {
    /// What the survey will establish — enough for a caller to decide whether to pay for
    /// a pass over the data before asking for one.
    pub fn question(&self) -> &SurveyQuestion {
        &self.question
    }

    /// The output descriptors this conversion will produce. Known already: this survey
    /// answers a question of admissibility, not of size.
    pub fn outputs(&self) -> &[TensorDescriptor] {
        self.plan.outputs()
    }

    /// Reads the data and either returns the ready plan or refuses the conversion.
    ///
    /// A refusal means the data conforms and simply does not fit the requested target, so
    /// the caller's remedy is a different target rather than a repair (ADR-044 § 13).
    ///
    /// The returned plan carries a fact about *these bytes*: it is valid for the buffers
    /// surveyed and no others. See [`ReadyPlan`] for the obligation that creates.
    pub fn survey(&self, inputs: &[TensorBuffers<'_>]) -> Result<ReadyPlan> {
        let src = self.plan.validate_inputs(inputs)?;
        if let Some(remap) = &self.plan.work {
            remap.verify_repetition(src)?;
        }
        Ok(self.plan.clone())
    }
}

/// A conversion that is ready to run.
///
/// # Reuse, and the one thing the compiler cannot check
///
/// A plan reached straight from [`Planned::Ready`] is a function of descriptors alone, so
/// it may be run any number of times on any buffers matching those descriptors. A plan
/// reached through [`SurveyPlan::survey`] carries a fact about the particular bytes the
/// survey read, and is valid for those bytes only: **mutating a surveyed input between
/// the survey and the run voids the plan, and this crate cannot detect that it happened.**
/// A [`BufferHandle`] carries no pointer and a plan keeps no digest of what it surveyed,
/// so there is nothing to compare against.
///
/// The borrow checker does not help here, by construction: a plan holds no borrow of any
/// input — the buffers arrive as arguments to [`ReadyPlan::run`] — so safe Rust will let a
/// caller mutate a surveyed buffer while holding a plan. Taking a borrow instead would
/// lock a multi-gigabyte weight buffer for the plan's whole life and still miss the cases
/// where the memory is owned outside Rust.
///
/// What the obligation does not extend to is soundness. Sizes are checked against the
/// destinations actually passed, so a caller who breaks the contract gets a wrong value or
/// an error — never a write outside a destination (ADR-044 § 4).
#[derive(Debug, Clone)]
pub struct ReadyPlan {
    inputs: Vec<TensorDescriptor>,
    outputs: Vec<TensorDescriptor>,
    buffer_plans: Vec<Vec<BufferPlan>>,
    destinations: Vec<Vec<Destination>>,
    /// `None` when every output buffer is an alias: a supported conversion with an empty
    /// body, which is what a target implying the same bytes as the source produces
    /// (ADR-044 § 5). Not an error, and not corollary 1 either — the request was for a
    /// representation the caller already has.
    work: Option<Remap>,
}

impl ReadyPlan {
    /// The output tensor descriptors. Final: nothing about the output's description
    /// depends on the data.
    pub fn outputs(&self) -> &[TensorDescriptor] {
        &self.outputs
    }

    /// How each buffer of output tensor `tensor` comes about.
    ///
    /// # Panics
    ///
    /// If `tensor` is not an output index of this plan.
    pub fn buffer_plans(&self, tensor: usize) -> &[BufferPlan] {
        &self.buffer_plans[tensor]
    }

    /// The buffers the caller must allocate for output tensor `tensor`, in the order
    /// [`ReadyPlan::run`] expects them.
    ///
    /// # Panics
    ///
    /// If `tensor` is not an output index of this plan.
    pub fn destinations(&self, tensor: usize) -> &[Destination] {
        &self.destinations[tensor]
    }

    /// How many work items this plan's work is partitioned into.
    ///
    /// Running every item exactly once, in any order, produces byte-identical output to
    /// one [`ReadyPlan::run`]. Items partition the **output**: an input may be read by
    /// many items (a broadcast reads one element once per repeat), which is why the
    /// invariant is stated over writes (ADR-044 § 6).
    pub fn work_item_count(&self) -> usize {
        self.work.as_ref().map_or(0, Remap::item_count)
    }

    /// Performs the conversion.
    ///
    /// An all-alias plan — a target that implies the same bytes as the source — writes
    /// nothing and reports nothing moved. That is a supported conversion with an empty
    /// body, not an error (ADR-044 § 5).
    pub fn run(
        &self,
        inputs: &[TensorBuffers<'_>],
        outputs: &mut [TensorDestinations<'_, '_>],
    ) -> Result<RunReport> {
        self.run_range(inputs, outputs, 0..self.work_item_count())
    }

    /// Performs work items `items` only, for a caller driving the partition itself.
    ///
    /// Inputs and destinations are validated on every call, so an out-of-range item range
    /// is an empty run rather than an error.
    pub fn run_range(
        &self,
        inputs: &[TensorBuffers<'_>],
        outputs: &mut [TensorDestinations<'_, '_>],
        items: Range<usize>,
    ) -> Result<RunReport> {
        let src = self.validate_inputs(inputs)?;
        let dst = self.validate_destinations(outputs)?;
        let (Some(mut dst), Some(remap)) = (dst, self.work.as_ref()) else {
            return Ok(RunReport::default());
        };
        Ok(remap.run_items(src, &mut dst, items))
    }

    /// The input buffer the remap reads. Single-input, single-buffer for every conversion
    /// this version plans; the nesting is in the signature because the arity
    /// generalisation is not (ADR-044 § 5).
    fn validate_inputs<'a>(&self, inputs: &[TensorBuffers<'a>]) -> Result<&'a [u8]> {
        if inputs.len() != self.inputs.len() {
            return Err(Error::InputTensorCount {
                expected: self.inputs.len(),
                found: inputs.len(),
            });
        }
        for (t, (buffers, descriptor)) in inputs.iter().zip(self.inputs.iter()).enumerate() {
            if buffers.len() != descriptor.buffers.len() {
                return Err(Error::InputBufferCount {
                    tensor: t,
                    expected: descriptor.buffers.len(),
                    found: buffers.len(),
                });
            }
            for (b, (bytes, handle)) in buffers.iter().zip(descriptor.buffers.iter()).enumerate() {
                if bytes.len() as u64 != handle.byte_size() {
                    return Err(Error::InputBufferSize {
                        tensor: t,
                        buffer: b,
                        expected: handle.byte_size(),
                        found: bytes.len(),
                    });
                }
            }
        }
        Ok(inputs[0][0])
    }

    /// The destination the remap writes, or `None` when the plan produces no buffer at
    /// all — the all-alias case.
    fn validate_destinations<'d>(
        &self,
        outputs: &'d mut [TensorDestinations<'_, '_>],
    ) -> Result<Option<&'d mut [u8]>> {
        if outputs.len() != self.outputs.len() {
            return Err(Error::OutputTensorCount {
                expected: self.outputs.len(),
                found: outputs.len(),
            });
        }
        for (t, (destinations, expected)) in
            outputs.iter().zip(self.destinations.iter()).enumerate()
        {
            if destinations.len() != expected.len() {
                return Err(Error::DestinationCount {
                    tensor: t,
                    expected: expected.len(),
                    found: destinations.len(),
                });
            }
            for (bytes, want) in destinations.iter().zip(expected.iter()) {
                if (bytes.len() as u64) < want.byte_size {
                    return Err(Error::DestinationTooShort {
                        tensor: t,
                        buffer: want.buffer,
                        expected: want.byte_size,
                        found: bytes.len(),
                    });
                }
                let addr = bytes.as_ptr() as usize;
                if want.byte_size > 0 && !addr.is_multiple_of(want.alignment as usize) {
                    return Err(Error::DestinationMisaligned {
                        tensor: t,
                        buffer: want.buffer,
                        expected: want.alignment,
                        // The largest power of two the address does satisfy.
                        actual: 1usize << addr.trailing_zeros().min(usize::BITS - 1),
                    });
                }
            }
        }
        if self.destinations[0].is_empty() {
            return Ok(None);
        }
        Ok(Some(&mut *outputs[0][0]))
    }
}

// ── Planning ──────────────────────────────────────────────────────────────────

/// Plans a conversion of `source` into the representation `target` asks for.
///
/// Reads descriptors only. The returned plan publishes the output descriptor and the
/// output buffer handles before any byte moves, so a caller can refuse a conversion
/// against a budget — which is what a broadcast materialisation needs, being the one
/// conversion whose output can be arbitrarily larger than its input (ADR-044 § 6).
///
/// # Errors
///
/// Every plan-time refusal is a precondition, and each names what it refused:
/// [`Error::UnresolvedShape`] for a dynamic dimension, [`Error::NonHostBuffer`] for device
/// memory, [`Error::SourceBufferTooSmall`] for a descriptor whose buffer cannot hold the
/// tensor it describes, and [`Error::UnsupportedConversion`] for a pair this version does
/// not implement — a sub-byte element type, a quantized source, a layout with no affine
/// stride form (tiled, Morton, Hilbert, sparse, block-paged, composite).
///
/// # Examples
///
/// ```
/// use hurray_core::{
///     BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode,
///     MIN_BUFFER_ALIGNMENT, descriptor::TensorDescriptor,
/// };
/// use hurray_transcode::{plan, Planned, Target};
///
/// // A 3×4 float32 row-major tensor.
/// let shape = Shape::new(vec![3, 4]).unwrap();
/// let buffer = BufferHandle::new(48, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap();
/// let source = TensorDescriptor::new(
///     1, 0, ElementType::Float32, shape, 0, LayoutDescriptor::RowMajor, vec![buffer],
///     None, None, None, None,
/// ).unwrap();
///
/// // Converting it to column-major is ready from the descriptors alone.
/// let planned = plan(&source, &Target::layout(LayoutDescriptor::ColMajor)).unwrap();
/// let Planned::Ready(ready) = planned else { panic!("no survey is needed") };
/// assert_eq!(ready.outputs()[0].layout, LayoutDescriptor::ColMajor);
/// assert_eq!(ready.destinations(0)[0].byte_size, 48);
/// ```
pub fn plan(source: &TensorDescriptor, target: &Target) -> Result<Planned> {
    let shape = &source.shape;
    if let Some(dim) = shape.dims().iter().position(|&d| d == hurray_core::DYNAMIC) {
        return Err(Error::UnresolvedShape { dim: dim as u32 });
    }
    if source.quantization.is_some() {
        return Err(Error::UnsupportedConversion {
            from: "a quantized tensor".to_string(),
            to: format!("layout 0x{:02X}", target.layout.tag()),
            reason: "a quantization scheme's metadata buffers are the quantization \
                     pass's business; dequantize first",
        });
    }
    if source.buffers.len() != 1 {
        return Err(Error::UnsupportedConversion {
            from: format!("a {}-buffer tensor", source.buffers.len()),
            to: format!("layout 0x{:02X}", target.layout.tag()),
            reason: "multi-buffer layouts are converted by the sparse and indirect passes",
        });
    }
    let handle = source.buffers[0];
    if handle.device_tag() != DeviceTag::Cpu {
        return Err(Error::NonHostBuffer {
            tensor: 0,
            buffer: 0,
            device_tag: handle.device_tag(),
        });
    }

    let elem_bytes = element_bytes(source.element_type, source.extension_type.as_ref())?;

    // Both layouts must be valid for this shape before their strides mean anything.
    source.layout.validate_against_shape(shape)?;
    target.layout.validate_against_shape(shape)?;

    let src_strides = affine_strides(&source.layout, source, target)?;
    let dst_strides = affine_strides(&target.layout, source, target)?;

    // The source's byte geometry, validated: every address its layout reaches is proved
    // representable and inside its own buffer here, before a run reads a byte. This is
    // also where a descriptor whose buffer cannot hold the tensor it describes is caught.
    let src_geometry = source_geometry(shape.dims(), &src_strides, source.byte_offset, elem_bytes)?;
    if src_geometry.extent() > handle.byte_size() {
        return Err(Error::SourceBufferTooSmall {
            tensor: 0,
            buffer: 0,
            needed: src_geometry.extent(),
            declared: handle.byte_size(),
        });
    }

    // Two descriptions imply the same bytes exactly when they address every index
    // identically, which for this family is one comparison: the stride vectors. The
    // comparison belongs here because this crate owns the arithmetic that decides it —
    // pushing it onto every caller is what § 5 refuses.
    if src_strides == dst_strides {
        let output = output_descriptor(source, target, handle, source.byte_offset)?;
        return Ok(Planned::Ready(ReadyPlan {
            inputs: vec![source.clone()],
            outputs: vec![output],
            buffer_plans: vec![vec![BufferPlan::Alias {
                input: 0,
                buffer: 0,
            }]],
            destinations: vec![vec![]],
            work: None,
        }));
    }

    // The destination's geometry, by the same checked arithmetic. It also chooses the
    // output's `byte_offset`: a negative stride reverses an axis, so element [0, …, 0] is
    // not the lowest address the tensor reads, and the base has to sit that far in.
    let dst_geometry = destination_geometry(shape.dims(), &dst_strides, elem_bytes)?;
    let dst_byte_size = dst_geometry.extent();
    let dst_byte_offset = dst_geometry.byte_offset();
    // The alignment is carried from the source: this version converts the layout and
    // nothing else, and re-placing a buffer at a different alignment is its own
    // conversion (ADR-044 § 14).
    let alignment = handle.alignment();
    // The caller allocates this buffer, so the plan cannot know its memory class:
    // carrying the source's would make the output descriptor claim pinned or unified
    // memory for an ordinary allocation. Standard is the honest default, and asking for
    // a different class is a placement conversion of its own.
    let out_handle = BufferHandle::new(
        dst_byte_size,
        alignment,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )?;
    let remap = Remap::new(
        shape,
        src_strides,
        dst_strides,
        elem_bytes,
        src_geometry,
        dst_geometry,
    )?;
    let output = output_descriptor(source, target, out_handle, dst_byte_offset)?;
    let ready = ReadyPlan {
        inputs: vec![source.clone()],
        outputs: vec![output],
        buffer_plans: vec![vec![BufferPlan::Produce {
            byte_size: dst_byte_size,
            alignment,
        }]],
        destinations: vec![vec![Destination {
            buffer: 0,
            byte_size: dst_byte_size,
            alignment,
        }]],
        work: Some(remap),
    };

    let collapsed = ready
        .work
        .as_ref()
        .map(Remap::collapsed_axes)
        .unwrap_or_default();
    if collapsed.is_empty() {
        Ok(Planned::Ready(ready))
    } else {
        Ok(Planned::NeedsSurvey(SurveyPlan {
            plan: ready,
            question: SurveyQuestion::BroadcastRepetition { axes: collapsed },
        }))
    }
}

/// The stride vector of a layout, or a refusal naming the pair.
fn affine_strides(
    layout: &LayoutDescriptor,
    source: &TensorDescriptor,
    target: &Target,
) -> Result<Vec<i64>> {
    layout
        .dense_element_strides(&source.shape)?
        .ok_or_else(|| Error::UnsupportedConversion {
            from: format!("layout 0x{:02X}", source.layout.tag()),
            to: format!("layout 0x{:02X}", target.layout.tag()),
            reason: "this version converts between layouts with an affine stride form \
                     (row-major, column-major, strided) only",
        })
}

/// Builds the output descriptor through `hurray-core`'s own constructor, so it is valid
/// by construction and no output can be more permissive than a decoded one.
///
/// Every section is carried through unchanged. All conversions in this version are exact
/// one-to-one plans, and ADR-044 § 7 carries an exact plan's statistics section through
/// mask and all: the index-to-value mapping is preserved, so every advisory field keeps
/// exactly the truth value it had.
fn output_descriptor(
    source: &TensorDescriptor,
    target: &Target,
    buffer: BufferHandle,
    byte_offset: u64,
) -> Result<TensorDescriptor> {
    let mut output = TensorDescriptor::new(
        source.version_major,
        source.version_minor,
        source.element_type,
        source.shape.clone(),
        byte_offset,
        target.layout.clone(),
        vec![buffer],
        None,
        source.shard.clone(),
        source.statistics.clone(),
        source.extension_type.clone(),
    )?;
    if let Some(member) = source.composite_member {
        output = output.with_composite_member(member);
    }
    Ok(output)
}

impl ReadyPlan {
    /// Runs every work item, fanning them over `rayon`'s thread pool.
    ///
    /// Byte-identical to [`ReadyPlan::run`] by construction: work items partition the
    /// output, so no two items write the same byte (ADR-044 § 6).
    ///
    /// Available with the `rayon` feature. A default build cannot spawn a thread, which
    /// is what keeps the crate callable from an async context — an async caller that wants
    /// the cores calls this inside `spawn_blocking`.
    #[cfg(feature = "rayon")]
    pub fn run_parallel(
        &self,
        inputs: &[TensorBuffers<'_>],
        outputs: &mut [TensorDestinations<'_, '_>],
    ) -> Result<RunReport> {
        use rayon::prelude::*;

        let src = self.validate_inputs(inputs)?;
        let dst = self.validate_destinations(outputs)?;
        let (Some(dst), Some(remap)) = (dst, self.work.as_ref()) else {
            return Ok(RunReport::default());
        };
        let shared = crate::parallel::SharedDestination::new(dst);
        Ok((0..self.work_item_count())
            .into_par_iter()
            .map(|item| {
                let mut writer = shared;
                remap.run_items(src, &mut writer, item..item + 1)
            })
            .reduce(RunReport::default, RunReport::combine))
    }
}
