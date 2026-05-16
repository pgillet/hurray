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

pub mod reader;
pub mod types;
pub mod writer;

pub use reader::{FileReader, FileTensor};
pub use types::{FileWriterOptions, IndexEntry, KvValue};
pub use writer::FileWriter;
