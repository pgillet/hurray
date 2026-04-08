/// Crate-level error type for `hurray-io`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// An I/O error from the underlying stream or file.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A core format error.
    #[error("format error: {0}")]
    Core(#[from] hurray_core::Error),

    /// The stream ended before a complete record was read.
    #[error("unexpected end of stream")]
    UnexpectedEof,

    /// A frame or record header is malformed.
    #[error("invalid frame header: {0}")]
    InvalidHeader(String),
}

/// Convenience alias for `Result` with [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
