//! Blocking stream read and write over a caller-supplied transport (ADR-041 § 8).
//!
//! `hurray-io` is async and generic over Tokio's traits; the C ABI is blocking and returns
//! status codes. What connects them is not a bridge but a shared foundation: `hurray-io`'s
//! sans-io core is a state machine over bytes, and these handles are a second driver for
//! it — the first being the Tokio one. No runtime is linked into a C consumer's binary.
//!
//! The division of labour:
//!
//! - the **core** decides what bytes it needs next and never blocks,
//! - the **transport** ([`HurrayTransport`]) moves those bytes and MAY block,
//! - these handles carry counts between the two.
//!
//! That is what lets the blocking contract live entirely in the caller's callbacks.
//!
//! Scope: flat tensors. A composite head reads back as the buffer-less tensor it is on the
//! wire, and writing a composite from C is not supported yet. File containers are
//! [`crate::file`]'s: they are addressed rather than consumed, so they drive the
//! random-access core over the vtable's `read_at` and `size` callbacks.

mod reader;
// `pub(crate)`: the file surface in `crate::file` is a second driver over the same vtable
// and hands out the same tensor handle, so it shares these two internals.
pub(crate) mod tensor;
pub(crate) mod transport;
mod writer;

pub use reader::{hurray_reader_destroy, hurray_reader_new, hurray_reader_next_tensor};
pub use reader::{HurrayReader, HurrayReaderOptions};
pub use tensor::HurrayTensor;
pub use tensor::{hurray_tensor_buffers, hurray_tensor_descriptor, hurray_tensor_destroy};
pub use transport::{
    HurrayTransport, HurrayTransportReadAtFn, HurrayTransportReadFn, HurrayTransportReleaseFn,
    HurrayTransportSizeFn, HurrayTransportWriteFn,
};
pub use writer::{hurray_writer_destroy, hurray_writer_new, hurray_writer_write_tensor};
pub use writer::{HurrayIoSlice, HurrayWriter, HurrayWriterOptions};
