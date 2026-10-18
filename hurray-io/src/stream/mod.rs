// Private modules with a flat re-export: one canonical path per item, so
// `hurray_io::stream::StreamReaderCore` is the only way to spell it.
mod sans_io;
mod types;

// Only the drivers need Tokio; the core and the types it works on do not (ADR-041 § 1).
#[cfg(feature = "tokio")]
mod frame;
#[cfg(feature = "tokio")]
pub mod reader;
#[cfg(feature = "tokio")]
pub mod writer;

pub use sans_io::{
    Destination, DestinationAlloc, DestinationRequest, ReadStep, StreamReaderCore,
    StreamWriterCore, SystemDestinations, Unusable, WriteOp, WriteStep,
};
pub use types::{
    CompositeNode, StreamComposite, StreamItem, StreamReaderOptions, StreamTensor,
    DEFAULT_MAX_COMPOSITE_DEPTH, DEFAULT_MAX_DESCRIPTOR_BYTES,
};

#[cfg(feature = "tokio")]
pub use reader::StreamReader;
#[cfg(feature = "tokio")]
pub use writer::StreamWriter;
