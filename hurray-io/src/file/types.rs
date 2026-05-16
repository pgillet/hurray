use crate::file::{MAX_DATA_BUFFER_ALIGNMENT, MIN_DATA_BUFFER_ALIGNMENT};

/// Options for [`FileWriter`][super::FileWriter].
pub struct FileWriterOptions {
    /// Alignment (bytes) applied to every data buffer within the file.
    ///
    /// MUST be a power of two in the range `[4096, 2097152]`. The default (4096)
    /// enables page-aligned mmap on all common operating systems.
    pub data_buffer_alignment: u32,
    /// When `true`, the footer index is sorted by UTF-8 byte order of tensor
    /// name, enabling binary search by readers.
    pub sorted_index: bool,
}

impl Default for FileWriterOptions {
    fn default() -> Self {
        Self {
            data_buffer_alignment: MIN_DATA_BUFFER_ALIGNMENT,
            sorted_index: false,
        }
    }
}

impl FileWriterOptions {
    pub(crate) fn validate(&self) -> Result<(), String> {
        let a = self.data_buffer_alignment;
        if !a.is_power_of_two()
            || !(MIN_DATA_BUFFER_ALIGNMENT..=MAX_DATA_BUFFER_ALIGNMENT).contains(&a)
        {
            return Err(format!(
                "data_buffer_alignment {a} is invalid \
                 (must be a power of two in [{MIN_DATA_BUFFER_ALIGNMENT}, {MAX_DATA_BUFFER_ALIGNMENT}])"
            ));
        }
        Ok(())
    }
}

/// A typed value stored in the file-level KV metadata section.
///
/// The `Array` variant holds a homogeneous list of scalar values; its elements
/// MUST NOT be `Array` themselves.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum KvValue {
    /// UTF-8 string. Wire tag `0x01`.
    String(std::string::String),
    /// Signed 64-bit integer. Wire tag `0x02`.
    Int64(i64),
    /// Unsigned 64-bit integer. Wire tag `0x03`.
    Uint64(u64),
    /// IEEE 754 binary64. Wire tag `0x04`.
    Float64(f64),
    /// Boolean. Wire tag `0x05`.
    Bool(bool),
    /// Opaque byte sequence. Wire tag `0x06`.
    Bytes(Vec<u8>),
    /// Homogeneous array of scalar values. Wire tag `0x07`.
    ///
    /// Elements MUST all have the same type and MUST NOT be `Array`.
    Array(Vec<KvValue>),
}

/// One entry in the footer index of a Hurray file.
#[derive(Debug, Clone)]
pub struct IndexEntry {
    /// UTF-8 tensor name, unique within the file.
    pub name: std::string::String,
    /// Absolute byte offset of the tensor descriptor from the file start.
    pub descriptor_offset: u64,
    /// Byte length of the tensor descriptor.
    pub descriptor_length: u32,
    /// Absolute byte offset of the first data buffer from the file start.
    pub data_offset: u64,
    /// Total byte length of all data buffers (including inter-buffer alignment
    /// padding, excluding trailing descriptor-alignment padding).
    pub data_length: u64,
}
