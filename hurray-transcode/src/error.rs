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
/// | unsupported conversion | plan | [`Error::UnsupportedConversion`], [`Error::UnresolvedShape`], [`Error::NonHostBuffer`] |
/// | input mismatch | plan / survey / run | [`Error::InputTensorCount`], [`Error::InputBufferCount`], [`Error::InputBufferSize`], [`Error::SourceBufferTooSmall`] |
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
