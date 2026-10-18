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

mod dest;
mod read;
mod write;

pub use dest::{Destination, DestinationAlloc, DestinationRequest, SystemDestinations};
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unusable {
    /// A read moved bytes into the core's destination and the count never arrived. Only
    /// this core's position is lost; the transport is untouched.
    LostReadCount,
    /// A write put an unknown number of payload bytes on the wire. A stream cannot be
    /// un-sent, so the sink is poisoned too: the reader at the far end will see a
    /// truncated item, and a fresh writer over the same connection is not a recovery.
    LostWriteCount,
}

impl fmt::Display for Unusable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LostReadCount => f.write_str("a read's byte count was lost"),
            Self::LostWriteCount => {
                f.write_str("a write's byte count was lost; the sink is poisoned")
            }
        }
    }
}
