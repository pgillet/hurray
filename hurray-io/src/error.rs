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

    /// A composite head's members were truncated: the stream ended before the head's
    /// declared `member_count` members had been read.
    #[error("torn composite: head declares {declared} member(s), stream ended after {actual}")]
    TornComposite { declared: u32, actual: u32 },

    /// Composite nesting exceeded the configured maximum depth. Guards against stack
    /// exhaustion from a maliciously deep composite on an untrusted stream.
    #[error("composite nesting too deep: exceeded limit of {limit}")]
    CompositeNestingTooDeep { limit: usize },

    // ── File format ───────────────────────────────────────────────────────────
    /// The file header magic is not `HRRYFILE`.
    #[error("invalid file magic: expected HRRYFILE")]
    InvalidFileMagic,

    /// The trailer magic is not `HRRY`.
    #[error("invalid trailer magic")]
    InvalidTrailerMagic,

    /// The container major version is higher than this reader supports.
    #[error("unsupported container version {major}.x (reader supports 1.x)")]
    UnsupportedContainerVersion { major: u8 },

    /// A tensor name was requested but is not in the file index.
    #[error("tensor not found: {0:?}")]
    TensorNotFound(String),

    /// Two tensors in the same file share a name.
    #[error("duplicate tensor name: {0:?}")]
    DuplicateTensorName(String),

    /// A tensor name is empty.
    #[error("tensor name must not be empty")]
    TensorNameEmpty,

    /// A tensor name exceeds the 65 535-byte `uint16` limit.
    #[error("tensor name too long: {len} bytes (max 65535)")]
    TensorNameTooLong { len: usize },

    /// The index CRC-32C does not match the stored value.
    #[error("index CRC-32C mismatch: stored 0x{stored:08X}, computed 0x{computed:08X}")]
    IndexCrc32cMismatch { stored: u32, computed: u32 },

    /// Two KV entries share the same key.
    #[error("duplicate KV key: {0:?}")]
    DuplicateKvKey(String),

    /// A KV key is empty.
    #[error("KV key must not be empty")]
    KvKeyEmpty,

    /// A KV key exceeds the 65 535-byte `uint16` limit.
    #[error("KV key too long: {len} bytes (max 65535)")]
    KvKeyTooLong { len: usize },

    /// A reserved flag bit was set in the file header.
    #[error("file header has reserved flag bits set: 0x{flags:08X}")]
    ReservedFileFlagBits { flags: u32 },

    /// The index section overlaps the trailer.
    #[error("index section overruns trailer boundary")]
    IndexOverrunsTrailer,
}

/// Convenience alias for `Result` with [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
