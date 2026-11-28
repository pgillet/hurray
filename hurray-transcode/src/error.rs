//! The crate's error type.
//!
//! One `thiserror` enum with a [`Result`] alias, the shape `hurray-core` and `hurray-io`
//! already have. There is **no** `std::io::Error` variant and there will not be one: the
//! crate performs no I/O, and the absence is a boundary marker in the type itself
//! (ADR-044 § 13).

use hurray_core::{DeviceTag, ElementType};

/// The result type every fallible operation in this crate returns.
pub type Result<T> = std::result::Result<T, Error>;

/// Everything this crate can refuse, and when.
///
/// The variants fall into families, and *when* each is raised is part of the contract
/// (ADR-044 § 13):
///
/// | Family | Raised at | Variants |
/// |---|---|---|
/// | unsupported conversion | plan | [`Error::UnsupportedConversion`], [`Error::UnresolvedShape`], [`Error::NonHostBuffer`], [`Error::AliasingTarget`], [`Error::TargetNotProvablyDistinct`] |
/// | input mismatch | plan / survey / run | [`Error::InputTensorCount`], [`Error::InputBufferCount`], [`Error::InputBufferSize`], [`Error::SourceBufferTooSmall`], [`Error::LayoutReachesBelowBuffer`] |
/// | not representable here | plan | [`Error::BeyondHostLimit`], [`Error::AllocationFailed`] |
/// | unusable destination | run | [`Error::OutputTensorCount`], [`Error::DestinationCount`], [`Error::DestinationTooShort`], [`Error::DestinationMisaligned`] |
/// | conversion refused by survey | survey | [`Error::RepetitionNotHeld`] |
/// | anything `hurray-core` rejects | any | [`Error::Core`] |
///
/// The enum is `#[non_exhaustive]`: the remaining families of § 13 — invalid data, and
/// the device variant — arrive with the conversions that raise them.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The (source, target) pair is not implementable or is out of scope. Names both
    /// ends, because a refusal that does not say what was refused is a dead end.
    #[error("unsupported conversion from {from} to {to}: {reason}")]
    UnsupportedConversion {
        /// The source representation, as the error should name it.
        from: String,
        /// The target representation, as the error should name it.
        to: String,
        /// Why this pair is refused.
        reason: &'static str,
    },

    /// The shape carries the dynamic-dimension sentinel. A plan is sizes, so it requires
    /// a fully resolved shape (ADR-044 § 14f); resolving the sentinel is a descriptor-only
    /// change and is not this crate's job.
    #[error(
        "dimension {dim} is dynamic: a plan computes sizes, so the shape must be fully \
         resolved before a conversion can be planned"
    )]
    UnresolvedShape {
        /// Index of the first unresolved dimension.
        dim: u32,
    },

    /// A source buffer is not host-resident. Device support is movement-only and is a
    /// separate API (ADR-044 § 11), so a conversion plan refuses device memory rather
    /// than appearing to convert it.
    #[error(
        "input tensor {tensor} buffer {buffer} is on device {device_tag}: conversion \
         requires host-resident buffers (stage to the host first)"
    )]
    NonHostBuffer {
        /// Index of the offending input tensor.
        tensor: usize,
        /// Index of the offending buffer within that tensor.
        buffer: usize,
        /// The buffer's device tag.
        device_tag: DeviceTag,
    },

    /// The number of input tensors handed to a survey or a run disagrees with the plan.
    #[error("input tensor count mismatch: the plan takes {expected}, got {found}")]
    InputTensorCount {
        /// What the plan takes.
        expected: usize,
        /// What was passed.
        found: usize,
    },

    /// An input tensor was handed a different number of buffers than its descriptor
    /// declares.
    #[error(
        "input tensor {tensor}: buffer count mismatch: the descriptor declares \
         {expected}, got {found}"
    )]
    InputBufferCount {
        /// Index of the offending input tensor.
        tensor: usize,
        /// What the descriptor declares.
        expected: usize,
        /// What was passed.
        found: usize,
    },

    /// An input buffer's byte length disagrees with its handle's `byte_size`.
    #[error(
        "input tensor {tensor} buffer {buffer}: {found} bytes passed, but the buffer \
         handle declares {expected}"
    )]
    InputBufferSize {
        /// Index of the offending input tensor.
        tensor: usize,
        /// Index of the offending buffer within that tensor.
        buffer: usize,
        /// What the handle declares.
        expected: u64,
        /// What was passed.
        found: usize,
    },

    /// A source descriptor's buffer is too small for the tensor it describes. Detected at
    /// plan time, from the layout's own addressing span — before any byte is read.
    #[error(
        "input tensor {tensor} buffer {buffer}: the layout addresses {needed} bytes but \
         the buffer handle declares {declared}"
    )]
    SourceBufferTooSmall {
        /// Index of the offending input tensor.
        tensor: usize,
        /// Index of the offending buffer within that tensor.
        buffer: usize,
        /// Bytes the layout addresses, counting from the buffer's start.
        needed: u64,
        /// Bytes the handle declares.
        declared: u64,
    },

    /// The number of output tensors handed to a run disagrees with the plan.
    #[error("output tensor count mismatch: the plan produces {expected}, got {found}")]
    OutputTensorCount {
        /// What the plan produces.
        expected: usize,
        /// What was passed.
        found: usize,
    },

    /// The number of destinations handed over for one output tensor disagrees with the
    /// number of [`crate::BufferPlan::Produce`] entries the plan published. Aliased
    /// buffers take no destination — the caller does not allocate them.
    #[error(
        "output tensor {tensor}: the plan produces {expected} buffer(s), got {found} \
         destination(s) (aliased buffers take no destination)"
    )]
    DestinationCount {
        /// Index of the offending output tensor.
        tensor: usize,
        /// How many destinations the plan expects.
        expected: usize,
        /// How many were passed.
        found: usize,
    },

    /// A destination is shorter than the plan published.
    #[error(
        "output tensor {tensor} buffer {buffer}: destination is {found} bytes, the plan \
         published {expected}"
    )]
    DestinationTooShort {
        /// Index of the offending output tensor.
        tensor: usize,
        /// Index of the offending buffer within that tensor.
        buffer: usize,
        /// What the plan published.
        expected: u64,
        /// What was passed.
        found: usize,
    },

    /// A destination's base address does not meet the alignment the plan published. The
    /// declared alignment is a promise a consumer may rely on for aligned SIMD loads
    /// (`buffer-protocol.md`), so writing into a less-aligned allocation would produce a
    /// tensor whose descriptor lies.
    #[error(
        "output tensor {tensor} buffer {buffer}: destination base address is \
         {actual}-byte aligned, the plan published {expected}"
    )]
    DestinationMisaligned {
        /// Index of the offending output tensor.
        tensor: usize,
        /// Index of the offending buffer within that tensor.
        buffer: usize,
        /// The alignment the plan published.
        expected: u32,
        /// The largest power of two the destination's base address is aligned to.
        actual: usize,
    },

    /// The target's strides **alias**: two distinct logical indices address one stored
    /// element. This transcoder refuses such a target unconditionally.
    ///
    /// A layout whose strides alias is legal in the format and is sometimes the point — a
    /// zero stride is a broadcast, and `docs/spec/layouts/strided.md § Aliasing` requires a
    /// consumer to accept one. What it cannot serve as is the *write* destination for
    /// an arbitrary tensor: strides `[1, 1]` over shape `[2, 2]` send `[0, 1]` and `[1, 0]`
    /// to the same element, so whichever is written second wins and the other value is
    /// gone. In parallel it is worse — the two writes can fall in different work items,
    /// which is a data race.
    ///
    /// Such a destination does hold, faithfully, exactly those tensors whose values already
    /// agree at the aliased indices. But a plan is built from descriptors and never sees
    /// values, so the refusal cannot be conditional on the data: it is unconditional, and
    /// it is *this transcoder's* refusal rather than a judgement the format makes
    /// (ADR-045 § 1).
    ///
    /// The broadcast *collapse* this crate does support is not this error: its axes are
    /// clamped out of the cover before the check, and the survey is what establishes that
    /// the discarded indices carried the same value.
    #[error(
        "target layout aliases: strides {strides:?} over the output cover {dims:?} map two \
         distinct indices to one element, so writing a tensor through it would lose a value"
    )]
    AliasingTarget {
        /// The target's strides.
        strides: Vec<i64>,
        /// The index space they were checked against — the shape with broadcast axes
        /// clamped to one element.
        dims: Vec<u64>,
    },

    /// The target's strides could not be *proved* to address every element distinctly.
    /// They may well do so; this crate's test did not establish it.
    ///
    /// That test is a sufficient condition once three or more axes can move — stride
    /// magnitudes strictly nested — so a few exotic interleavings that are genuinely
    /// injective are refused here as well: strides `[2, 3, 4]` over shape `[2, 2, 2]`
    /// address offsets `0, 4, 3, 7, 2, 6, 5, 9` and collide nowhere, yet come back
    /// unproved. Two movable axes leave no such gap — that case is decided exactly — so
    /// reaching this error takes a target with three. This crate will not write through a
    /// target it cannot prove distinct, because that proof is what the parallel path's
    /// disjointness argument rests on.
    ///
    /// Unlike [`Error::AliasingTarget`], this is a limitation and not a verdict: a stronger
    /// proof would admit these targets with no change to the format or to this API. See
    /// `hurray_core::strides_provably_distinct`.
    #[error(
        "target layout is not provably distinct: strides {strides:?} over the output cover \
         {dims:?} were not proved to address every element distinctly, so this crate \
         declines to write through them"
    )]
    TargetNotProvablyDistinct {
        /// The target's strides.
        strides: Vec<i64>,
        /// The index space they were checked against — the shape with broadcast axes
        /// clamped to one element.
        dims: Vec<u64>,
    },

    /// A layout addresses bytes before the start of its own buffer: its `byte_offset` is
    /// smaller than the distance its negative strides reach below the base.
    ///
    /// `docs/spec/layouts/strided.md § Element Address` requires `byte_offset` to be set
    /// so that every element is addressable within the buffer, so this is an invalid
    /// source descriptor rather than an unsupported conversion.
    #[error(
        "layout reaches {below} bytes from the start of the buffer, below its own \
         byte_offset of {byte_offset}: every element must be addressable within the buffer"
    )]
    LayoutReachesBelowBuffer {
        /// The declared `byte_offset`.
        byte_offset: u64,
        /// The lowest byte the layout addresses, relative to the buffer's start. Negative.
        below: i128,
    },

    /// The conversion is arithmetically well defined, but one of its quantities cannot be
    /// represented on this host.
    ///
    /// This is the line between a descriptor whose arithmetic overflows — which is
    /// [`Error::Core`] carrying `AddressOverflow` — and one that is perfectly consistent
    /// but asks for more than the machine can address. A tensor whose byte extent exceeds
    /// `isize::MAX` cannot be handed to a run as `&mut [u8]` at any allocation, and a work
    /// item count beyond `usize::MAX` cannot be indexed.
    #[error("{what} is {value}, beyond this host's limit of {limit}")]
    BeyondHostLimit {
        /// Which quantity exceeded the limit.
        what: &'static str,
        /// The value computed for it.
        value: i128,
        /// The largest value this host can represent for that quantity.
        limit: i128,
    },

    /// An [`crate::AlignedDestination`] could not be allocated.
    ///
    /// Returned rather than aborting, because the size and alignment come from a
    /// descriptor, which may have come off a wire: a process that aborts on a number a
    /// peer chose is a denial of service.
    #[error("could not allocate {byte_size} bytes at {alignment}-byte alignment")]
    AllocationFailed {
        /// Bytes requested.
        byte_size: u64,
        /// Alignment requested.
        alignment: u32,
    },

    /// A survey refused the conversion: collapsing a broadcast axis requires that the
    /// data actually repeat along it, and it does not.
    ///
    /// Nothing is wrong with the input — the caller's remedy is a different target, not a
    /// repair (ADR-044 § 13, *conversion refused by survey*).
    #[error(
        "broadcast collapse refused: the data does not repeat along dimension {dim} \
         (first difference at index {index:?})"
    )]
    RepetitionNotHeld {
        /// The collapsed dimension whose repetition does not hold.
        dim: u32,
        /// The first logical index at which the data differs from the element the
        /// collapsed layout would keep.
        index: Vec<u64>,
    },

    /// Everything `hurray-core` rejects: an invalid descriptor, an invalid layout for a
    /// shape, address arithmetic that overflows.
    #[error(transparent)]
    Core(#[from] hurray_core::Error),
}

impl Error {
    /// Builds an [`Error::UnsupportedConversion`] for an element type this pass cannot
    /// move.
    pub(crate) fn unsupported_element_type(ty: ElementType, reason: &'static str) -> Self {
        Self::UnsupportedConversion {
            from: format!("element type {ty}"),
            to: format!("element type {ty}"),
            reason,
        }
    }
}
