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

    /// The number of supplied buffer slices does not match the descriptor's buffer table.
    #[error("buffer count mismatch: descriptor declares {declared}, got {actual}")]
    MultiBufferLengthMismatch { declared: usize, actual: usize },

    /// A supplied buffer's byte length does not match the corresponding handle's `byte_size`.
    #[error("buffer[{index}] size mismatch: declared {declared}, got {actual}")]
    BufferSizeMismatch {
        index: usize,
        declared: u64,
        actual: u64,
    },

    /// A descriptor or buffer exceeded the configured size limit.
    #[error("frame too large: {kind} = {value}, limit = {limit}")]
    FrameTooLarge {
        kind: &'static str,
        value: u64,
        limit: u64,
    },

    /// A buffer handle's `sync_mode` is invalid for a cross-machine transport.
    #[error(
        "buffer[{index}] sync_mode 0x{actual:02X} is invalid for cross-machine transport \
         (must be 0x00 ProducerSynced)"
    )]
    InvalidCrossMachineSyncMode { index: usize, actual: u8 },
}

/// Convenience alias for `Result` with [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
