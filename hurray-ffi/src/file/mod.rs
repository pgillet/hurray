//! Blocking `HRRYFILE` reads and writes over a caller-supplied transport (ADR-041 § 8).
//!
//! The file container is *addressed*, not consumed: its trailer sits at `size - 40`, its
//! index is located from there, and tensors are found by name. So the reader here drives
//! `hurray-io`'s random-access core and asks the caller's
//! [`read_at`][crate::HurrayTransport::read_at] callback for absolute ranges, while the
//! writer emits one forward pass through `write` and never seeks back.
//!
//! The division of labour is the stream surface's, unchanged:
//!
//! - the **core** decides which bytes it needs and never blocks,
//! - the **transport** moves them and MAY block,
//! - these handles carry counts between the two.
//!
//! Scope: flat tensors, composites, and the KV metadata section. A composite is addressed
//! by its head's name, read back whole by
//! [`hurray_file_reader_read_composite`], and written from the preorder node array
//! [`HurrayCompositeNode`][crate::HurrayCompositeNode] describes.

// `pub(crate)`: its pointer helpers (`bytes_from_c`, `str_from_c`) are the boundary's,
// not the KV section's, and `crate::composite` needs the same ones.
pub(crate) mod kv;
mod reader;
mod writer;

pub use kv::{
    HurrayKvEntry, HurrayKvValue, HURRAY_KV_ARRAY, HURRAY_KV_BOOL, HURRAY_KV_BYTES,
    HURRAY_KV_FLOAT64, HURRAY_KV_INT64, HURRAY_KV_STRING, HURRAY_KV_UINT64,
};
pub use reader::{
    hurray_file_reader_destroy, hurray_file_reader_kv_count, hurray_file_reader_kv_entry,
    hurray_file_reader_new, hurray_file_reader_read_composite, hurray_file_reader_read_descriptor,
    hurray_file_reader_read_tensor, hurray_file_reader_tensor_count,
    hurray_file_reader_tensor_name, HurrayFileReader,
};
pub use writer::{
    hurray_file_writer_destroy, hurray_file_writer_finish, hurray_file_writer_new,
    hurray_file_writer_write_composite, hurray_file_writer_write_tensor, HurrayFileWriter,
    HurrayFileWriterOptions,
};
