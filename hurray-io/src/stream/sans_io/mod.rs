//! A sans-io core for the stream wire format: a state machine over bytes, with no file,
//! socket or runtime anywhere in it (ADR-041 § 1).
//!
//! The format already has this shape. Descriptors precede their buffers, the encoding is
//! self-delimiting, back-references are forbidden, and nothing is ever re-read — so a
//! reader never looks backwards, and a reader that never looks backwards wants to be a
//! state machine.
//!
//! Two machines, because the two directions are not mirror images:
//! [`StreamReaderCore`] *owns* the memory each payload lands in (§ 2), while
//! [`StreamWriterCore`] borrows the caller's payloads and emits them one slice at a time
//! without ever assembling them (§ 3).
//!
//! Both are driven the same way: ask with `step`, transfer the bytes, report the count.
//! Neither call blocks, which is what lets one core serve a blocking C driver and an
//! event loop alike (§ 4).
//!
//! Composites are assembled a layer up, by [`CompositeAssembler`]. Rebuilding the tree
//! needs no bytes — the tensors it works on are already decoded — so it is bookkeeping,
//! not I/O, and keeping it out of the byte-level machine is what lets one core serve both
//! the flat and the assembled view of the same stream.

mod composite;
mod dest;
mod read;
mod write;

pub use composite::CompositeAssembler;
pub use dest::{
    AlignedDestination, Destination, DestinationAlloc, DestinationRequest, SystemDestinations,
};
pub use read::{ReadStep, StreamReaderCore};
pub use write::{StreamWriterCore, WriteOp, WriteStep};

use std::fmt;

/// Why a core became unusable: a transport moved bytes and their count was lost, so the
/// core's position no longer matches the transport's (ADR-041 § 6).
///
/// No inspection recovers it — `0x00` is valid tensor data — so the state is named and
/// reported rather than guessed at. "Probably still fine" is not a state.
///
/// # Examples
///
/// ```
/// use hurray_io::stream::StreamReaderCore;
/// use hurray_io::Error;
///
/// let mut core = StreamReaderCore::new();
/// core.mark_unusable();
/// assert!(matches!(core.step(), Err(Error::CoreUnusable(_))));
/// ```
// `non_exhaustive`: the reasons a core can be retired are open-ended, and a caller
// matching on them exhaustively would break on every future one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Unusable {
    /// A read moved bytes into the core's destination and the count never arrived. Only
    /// this core's position is lost; the transport is untouched.
    LostReadCount,
    /// A write put an unknown number of payload bytes on the wire, so the position is
    /// *unknown*. Narrower than [`Self::IncompleteOutput`], which is a known-incomplete
    /// item: a stream cannot be un-sent, so the sink is poisoned too — the reader at the
    /// far end will see a truncated item, and a fresh writer over the same connection is
    /// not a recovery.
    LostWriteCount,
    /// An item was begun and cannot be completed, so the output is incomplete. The count
    /// is *known*, unlike [`Self::LostWriteCount`]; what is lost is the ability to produce
    /// a conforming output, because a forward-only writer has no way back to the item's
    /// start (§ 6b).
    IncompleteOutput,
}

impl fmt::Display for Unusable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LostReadCount => f.write_str("a read's byte count was lost"),
            Self::LostWriteCount => {
                f.write_str("a write's byte count was lost; the sink is poisoned")
            }
            // Worded to cover both reachable cases: an item torn mid-emission, and a sink
            // that closed before the item's first byte. "Begun and not finished" would be
            // false for the second.
            Self::IncompleteOutput => {
                f.write_str("the output is incomplete and cannot be completed")
            }
        }
    }
}
