pub(crate) const FILE_MAGIC: &[u8; 8] = b"HRRYFILE";
pub(crate) const TRAILER_MAGIC: &[u8; 4] = b"HRRY";
pub(crate) const FILE_HEADER_SIZE: u64 = 64;
pub(crate) const TRAILER_SIZE: u64 = 40;
pub(crate) const SUPPORTED_CONTAINER_VERSION_MAJOR: u8 = 1;
pub(crate) const MIN_DATA_BUFFER_ALIGNMENT: u32 = 4096;
pub(crate) const MAX_DATA_BUFFER_ALIGNMENT: u32 = 2 * 1024 * 1024;

pub(crate) mod file_flags {
    pub const HAS_KV_METADATA: u32 = 1 << 0;
    pub const SORTED_INDEX: u32 = 1 << 1;
    pub const HAS_INDEX_CRC32C: u32 = 1 << 2;
    pub const RESERVED_MASK: u32 = !0x07u32;
}

mod sans_io;
pub mod types;

// Only the drivers need Tokio; the cores and the types they work on do not (ADR-041 § 1).
#[cfg(feature = "tokio")]
pub mod reader;
#[cfg(feature = "tokio")]
pub mod writer;

pub use sans_io::{
    flatten_composite, CompositeWrite, FileReaderCore, FileStep, FileWriteOp, FileWriterCore,
};
pub use types::{
    FileComposite, FileCompositeNode, FileItem, FileTensor, FileWriterOptions, IndexEntry, KvValue,
    DEFAULT_MAX_COMPOSITE_DEPTH,
};

#[cfg(feature = "tokio")]
pub use reader::FileReader;
#[cfg(feature = "tokio")]
pub use writer::FileWriter;
