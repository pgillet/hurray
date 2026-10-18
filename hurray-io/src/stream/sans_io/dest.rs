//! Where the bytes of a read land.
//!
//! ADR-041 § 2 puts the destination on the core's side of the boundary: the core hands the
//! driver somewhere to fill and never accepts payload bytes the driver placed elsewhere.
//! The transport's write into that destination is then the only copy.
//!
//! § 5 makes the destination's alignment the one the buffer handle *declares*, and ADR-041
//! left open whether a binding may supply the allocation. It may: [`DestinationAlloc`] is
//! that seam. A Python read can land directly in a NumPy array without the state machine
//! knowing, which is what `hurray-python/src/numpy_allocator.rs` already does elsewhere.

use bytes::Bytes;

use crate::aligned::AlignedBuf;
use crate::{Error, Result};

/// What the core needs for one buffer of the tensor it is reading.
///
/// # Examples
///
/// ```
/// use hurray_io::stream::DestinationRequest;
///
/// let request = DestinationRequest { index: 0, byte_size: 256, alignment: 64 };
/// assert_eq!(request.byte_size % u64::from(request.alignment), 0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DestinationRequest {
    /// Index of the buffer in the descriptor's buffer table, for diagnostics.
    pub index: usize,
    /// Exactly how many bytes the handle declares.
    pub byte_size: u64,
    /// The alignment the handle declares: a power of two, and at least 64 when
    /// `byte_size > 0` (`buffer-protocol.md § Minimum Alignment`).
    pub alignment: u32,
}

/// One buffer's worth of memory, filled by a driver and then handed to the caller.
///
/// Implement this alongside [`DestinationAlloc`] to make reads land in memory your runtime
/// owns. The address MUST NOT move between [`as_mut_slice`][Destination::as_mut_slice]
/// calls: a driver may hold a pointer into it across a suspension (ADR-041 § 6a).
///
/// # Examples
///
/// ```
/// use bytes::Bytes;
/// use hurray_io::stream::Destination;
///
/// /// A destination that ignores alignment — fine for a test, not for a reader.
/// struct VecDest(Vec<u8>);
///
/// impl Destination for VecDest {
///     fn as_mut_slice(&mut self) -> &mut [u8] {
///         &mut self.0
///     }
///     fn freeze(self: Box<Self>) -> Bytes {
///         Bytes::from(self.0)
///     }
/// }
///
/// let mut dest = VecDest(vec![0u8; 4]);
/// dest.as_mut_slice().copy_from_slice(b"hurr");
/// assert_eq!(Box::new(dest).freeze(), Bytes::from_static(b"hurr"));
/// ```
pub trait Destination: Send {
    /// The bytes to fill, exactly `byte_size` long.
    fn as_mut_slice(&mut self) -> &mut [u8];

    /// Hands the filled memory out as [`Bytes`], without copying it.
    fn freeze(self: Box<Self>) -> Bytes;
}

/// Allocates the destinations a [`StreamReaderCore`][super::StreamReaderCore] fills.
///
/// # Examples
///
/// ```
/// use hurray_io::stream::{DestinationAlloc, SystemDestinations, DestinationRequest};
///
/// let mut alloc = SystemDestinations;
/// let mut dest = alloc
///     .allocate(DestinationRequest { index: 0, byte_size: 128, alignment: 4096 })?;
/// assert_eq!(dest.as_mut_slice().len(), 128);
/// # Ok::<(), hurray_io::Error>(())
/// ```
pub trait DestinationAlloc: Send {
    /// Allocates one destination, zero-initialised, at the requested alignment.
    ///
    /// # Errors
    ///
    /// [`Error::UnallocatableBuffer`] when the request cannot be satisfied — a `byte_size`
    /// beyond the address space, or an `alignment` this allocator cannot meet. A
    /// descriptor off the wire can ask for either, so this MUST NOT panic.
    fn allocate(&mut self, request: DestinationRequest) -> Result<Box<dyn Destination>>;
}

/// The default allocator: one host allocation per buffer at the declared alignment.
///
/// # Examples
///
/// ```
/// use hurray_io::stream::{DestinationAlloc, DestinationRequest, SystemDestinations};
///
/// let mut dest = SystemDestinations
///     .allocate(DestinationRequest { index: 0, byte_size: 64, alignment: 64 })?;
/// assert_eq!(dest.as_mut_slice().len(), 64);
/// assert_eq!(dest.as_mut_slice().as_ptr() as usize % 64, 0);
/// # Ok::<(), hurray_io::Error>(())
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemDestinations;

impl DestinationAlloc for SystemDestinations {
    fn allocate(&mut self, request: DestinationRequest) -> Result<Box<dyn Destination>> {
        let buf = AlignedBuf::zeroed(request.byte_size, request.alignment).ok_or(
            Error::UnallocatableBuffer {
                index: request.index,
                byte_size: request.byte_size,
                alignment: request.alignment,
            },
        )?;
        Ok(Box::new(buf))
    }
}

impl Destination for AlignedBuf {
    fn as_mut_slice(&mut self) -> &mut [u8] {
        AlignedBuf::as_mut_slice(self)
    }

    fn freeze(self: Box<Self>) -> Bytes {
        AlignedBuf::freeze(*self)
    }
}
