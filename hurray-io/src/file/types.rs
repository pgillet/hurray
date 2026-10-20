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

/// Default maximum composite nesting depth accepted when reading a composite.
pub const DEFAULT_MAX_COMPOSITE_DEPTH: usize = 64;

/// A tensor read from a Hurray file: descriptor plus zero-copy buffer views.
#[derive(Debug)]
pub struct FileTensor {
    /// The tensor's name as recorded in the file index.
    pub name: std::string::String,
    /// The decoded tensor descriptor.
    pub descriptor: hurray_core::TensorDescriptor,
    /// Raw buffer bytes, one [`Bytes`][bytes::Bytes] per buffer handle.
    pub buffers: Vec<bytes::Bytes>,
}

/// One member of a composite read from a file: a plain tensor or a nested composite.
#[derive(Debug)]
pub enum FileItem {
    /// A single (non-composite) tensor.
    Tensor(FileTensor),
    /// A nested composite.
    Composite(FileComposite),
}

impl FileItem {
    /// The item's governing descriptor: the tensor's descriptor, or the composite head.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let item: hurray_io::file::FileItem = unimplemented!();
    /// println!("{:?}", item.descriptor().element_type);
    /// # Ok(())
    /// # }
    /// ```
    pub fn descriptor(&self) -> &hurray_core::TensorDescriptor {
        match self {
            FileItem::Tensor(t) => &t.descriptor,
            FileItem::Composite(c) => &c.head,
        }
    }
}

/// A composite tensor read from a file: its head plus its ordered members.
///
/// Membership was recovered from the head's `member_count` and file-offset adjacency, then
/// validated with [`CompositeValidator`][hurray_core::CompositeValidator] (member count,
/// partition coverage, overlay ordering). Members are ordered as written; each may itself
/// be a composite (nesting).
#[derive(Debug)]
pub struct FileComposite {
    /// The head's name, as recorded in the file index.
    pub name: std::string::String,
    /// The composite head descriptor (owns no data buffers).
    pub head: hurray_core::TensorDescriptor,
    /// The members, in write order. Each may itself be a composite.
    pub members: Vec<FileItem>,
}

/// A named node to write as part of a composite: either a plain tensor or a nested
/// composite. Every node carries a name because every tensor — head and members alike —
/// gets its own footer-index entry (ADR-027 § Binding).
pub enum FileCompositeNode<'a> {
    /// A single (non-composite) tensor member.
    Tensor {
        /// Unique tensor name for this member's index entry.
        name: &'a str,
        /// The member's tensor descriptor.
        descriptor: &'a hurray_core::TensorDescriptor,
        /// One byte-slice per buffer handle in `descriptor.buffers`.
        buffers: &'a [&'a [u8]],
    },
    /// A nested composite member: its head plus its own members.
    Composite {
        /// Unique tensor name for the nested head's index entry.
        name: &'a str,
        /// The nested composite's head descriptor.
        head: &'a hurray_core::TensorDescriptor,
        /// The nested composite's members, in order.
        members: &'a [FileCompositeNode<'a>],
    },
}

impl FileCompositeNode<'_> {
    /// The node's governing descriptor: the tensor's descriptor, or the nested head.
    pub(crate) fn descriptor(&self) -> &hurray_core::TensorDescriptor {
        match self {
            FileCompositeNode::Tensor { descriptor, .. } => descriptor,
            FileCompositeNode::Composite { head, .. } => head,
        }
    }
}
