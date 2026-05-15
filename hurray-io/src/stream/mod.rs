mod frame;
pub mod reader;
pub mod writer;

pub use reader::{StreamReader, StreamReaderOptions, StreamTensor, DEFAULT_MAX_DESCRIPTOR_BYTES};
pub use writer::StreamWriter;
