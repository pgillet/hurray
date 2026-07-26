mod frame;
pub mod reader;
pub mod writer;

pub use reader::{
    StreamComposite, StreamItem, StreamReader, StreamReaderOptions, StreamTensor,
    DEFAULT_MAX_COMPOSITE_DEPTH, DEFAULT_MAX_DESCRIPTOR_BYTES,
};
pub use writer::{CompositeNode, StreamWriter};
