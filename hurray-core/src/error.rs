/// Crate-level error type for `hurray-core`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// An element type is not supported or recognized.
    #[error("unsupported element type: {0}")]
    UnsupportedElementType(String),

    /// A shape or stride value is invalid.
    #[error("invalid shape or stride: {0}")]
    InvalidShape(String),

    /// A buffer alignment requirement was not satisfied.
    #[error("buffer alignment error: expected {expected}-byte alignment, got {actual}")]
    AlignmentError { expected: usize, actual: usize },

    /// A quantization descriptor is malformed.
    #[error("invalid quantization descriptor: {0}")]
    InvalidQuantization(String),
}

/// Convenience alias for `Result` with [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
