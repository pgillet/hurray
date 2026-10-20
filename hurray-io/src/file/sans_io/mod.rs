//! A sans-io core for the file container: a state machine over bytes, with no file,
//! socket or runtime anywhere in it (ADR-041 § 1).
//!
//! Two machines again, and for the same reason as the stream's — the directions are not
//! mirror images — but neither is a copy of its stream counterpart, because the file is
//! *addressed* rather than consumed:
//!
//! - [`FileReaderCore`] answers in **ranges**. It asks for bytes `[offset, offset + len)`
//!   and is told how many arrived (§ 7): the trailer sits at `size - 40`, the index is
//!   located from it, and tensors are found by name. Nothing is read twice and nothing is
//!   read that a request did not ask for.
//! - [`FileWriterCore`] emits **one forward pass**. The footer index is accumulated as
//!   tensors go out and written last, so no offset is ever patched and the sink never has
//!   to be seekable.
//!
//! The allocation seam is shared with the stream core rather than duplicated:
//! [`DestinationAlloc`][crate::stream::DestinationAlloc] and
//! [`Destination`][crate::stream::Destination] are one contract for both, because § 2's
//! "the core owns the destination" and § 5's declared alignment are one promise about
//! payload memory, not two.

mod read;
mod sections;
mod write;

pub use read::{FileReaderCore, FileStep};
pub use write::{flatten_composite, CompositeWrite, FileWriteOp, FileWriterCore};
