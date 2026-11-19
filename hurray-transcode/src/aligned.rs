//! An owned destination that satisfies a plan's declared alignment.
//!
//! The crate allocates no output (ADR-044 § 5), but a caller cannot allocate a conforming
//! one with `Vec<u8>`: `docs/spec/buffer-protocol.md § Minimum Alignment` makes every
//! non-empty buffer's base address at least 64-byte aligned, and a consumer MAY rely on
//! that declaration for aligned SIMD loads — while `Vec<u8>` allocates at
//! `align_of::<u8>()`. A run therefore refuses a misaligned destination
//! ([`crate::Error::DestinationMisaligned`]), and this is the allocation that satisfies it.
//!
//! It is a convenience, not a policy: a caller converting straight into a NumPy array, a
//! pinned staging buffer or an mmap'd region passes that memory instead and never
//! constructs this type.

use std::alloc::{self, Layout};
use std::ptr::NonNull;

use crate::{Destination, Error, Result};

/// An owned, zero-initialised allocation whose base address meets a declared alignment.
///
/// # Examples
///
/// ```
/// use hurray_transcode::AlignedDestination;
///
/// let mut buffer = AlignedDestination::zeroed(96, 64).unwrap();
/// assert_eq!(buffer.as_mut_slice().len(), 96);
/// assert_eq!(buffer.as_slice().as_ptr() as usize % 64, 0);
/// ```
pub struct AlignedDestination {
    /// Dangling when `len == 0`, which allocates nothing and is never dereferenced.
    ptr: NonNull<u8>,
    layout: Layout,
    len: usize,
}

impl AlignedDestination {
    /// Allocates the buffer a plan asked for.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{
    ///     BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode,
    ///     MIN_BUFFER_ALIGNMENT, descriptor::TensorDescriptor,
    /// };
    /// use hurray_transcode::{plan, AlignedDestination, Planned, Target};
    ///
    /// let shape = Shape::new(vec![2, 3]).unwrap();
    /// let handle = BufferHandle::new(
    ///     6, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced,
    /// ).unwrap();
    /// let source = TensorDescriptor::new(
    ///     1, 0, ElementType::Uint8, shape, 0, LayoutDescriptor::RowMajor, vec![handle],
    ///     None, None, None, None,
    /// ).unwrap();
    ///
    /// let Planned::Ready(ready) = plan(&source, &Target::layout(LayoutDescriptor::ColMajor))?
    /// else { unreachable!() };
    /// let buffer = AlignedDestination::for_destination(ready.destinations(0)[0])?;
    /// assert_eq!(buffer.as_slice().len(), 6);
    /// # Ok::<(), hurray_transcode::Error>(())
    /// ```
    pub fn for_destination(destination: Destination) -> Result<Self> {
        Self::zeroed(destination.byte_size, destination.alignment)
    }

    /// Allocates `byte_size` zeroed bytes at `alignment`.
    ///
    /// Zero-initialised rather than uninitialised: the destination is handed to a run as
    /// `&mut [u8]`, and a slice over uninitialised memory is undefined behaviour.
    ///
    /// # Errors
    ///
    /// - [`hurray_core::Error::AlignmentNotPowerOfTwo`] — `alignment` is not a power of two.
    /// - [`hurray_core::Error::AddressOverflow`] — `byte_size` does not form a valid host
    ///   allocation layout. Both are reachable from a descriptor off the wire, so neither
    ///   panics.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::Error as CoreError;
    /// use hurray_transcode::{AlignedDestination, Error};
    ///
    /// // An empty buffer is legal at any power of two and allocates nothing.
    /// assert!(AlignedDestination::zeroed(0, 1).unwrap().as_slice().is_empty());
    ///
    /// // A page-aligned destination, as a GPU or RDMA producer declares.
    /// let page = AlignedDestination::zeroed(8192, 4096).unwrap();
    /// assert_eq!(page.as_slice().as_ptr() as usize % 4096, 0);
    ///
    /// assert!(matches!(
    ///     AlignedDestination::zeroed(64, 3),
    ///     Err(Error::Core(CoreError::AlignmentNotPowerOfTwo { alignment: 3 }))
    /// ));
    /// ```
    pub fn zeroed(byte_size: u64, alignment: u32) -> Result<Self> {
        if !alignment.is_power_of_two() {
            return Err(hurray_core::Error::AlignmentNotPowerOfTwo { alignment }.into());
        }
        let len = usize::try_from(byte_size).map_err(|_| hurray_core::Error::AddressOverflow)?;
        if len == 0 {
            // An empty buffer stays legal at any declared power of two
            // (`buffer-protocol.md § Empty Buffers`) and allocates nothing.
            return Ok(Self {
                ptr: NonNull::dangling(),
                layout: Layout::from_size_align(0, 1).map_err(layout_error)?,
                len: 0,
            });
        }
        let layout = Layout::from_size_align(len, alignment as usize).map_err(layout_error)?;
        // SAFETY: `layout` has non-zero size, checked immediately above.
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        // Allocation failure aborts, as it does for `Vec`: there is no fallible path a
        // caller could usefully take for a size it chose.
        let ptr = NonNull::new(ptr).unwrap_or_else(|| alloc::handle_alloc_error(layout));
        Ok(Self { ptr, layout, len })
    }

    /// The destination to fill.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_transcode::AlignedDestination;
    ///
    /// let mut buffer = AlignedDestination::zeroed(4, 64).unwrap();
    /// buffer.as_mut_slice()[0] = 0xAB;
    /// assert_eq!(buffer.as_slice(), &[0xAB, 0, 0, 0]);
    /// ```
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.len == 0 {
            return &mut [];
        }
        // SAFETY: `ptr` is a live allocation of `len` zeroed (therefore initialised)
        // bytes, owned exclusively by `self`, and `&mut self` prevents any aliasing slice.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// The buffer's contents.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_transcode::AlignedDestination;
    ///
    /// assert_eq!(AlignedDestination::zeroed(2, 64).unwrap().as_slice(), &[0, 0]);
    /// ```
    pub fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: as in `as_mut_slice`; `&self` grants shared read access only.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

fn layout_error(_: std::alloc::LayoutError) -> Error {
    hurray_core::Error::AddressOverflow.into()
}

impl Drop for AlignedDestination {
    fn drop(&mut self) {
        if self.len > 0 {
            // SAFETY: `ptr` came from `alloc_zeroed` with exactly this layout, and `Drop`
            // runs once per allocation.
            unsafe { alloc::dealloc(self.ptr.as_ptr(), self.layout) };
        }
    }
}

impl AsRef<[u8]> for AlignedDestination {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsMut<[u8]> for AlignedDestination {
    fn as_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

impl std::fmt::Debug for AlignedDestination {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // EXPRESSION reprs are for values a reader can retype; an allocation is not one.
        f.debug_struct("AlignedDestination")
            .field("len", &self.len)
            .field("alignment", &self.layout.align())
            .finish()
    }
}

// SAFETY: the allocation is owned exclusively, and a slice is handed out only through
// `&self` / `&mut self`, so the usual borrow rules govern every access.
unsafe impl Send for AlignedDestination {}
// SAFETY: as above — shared access yields `&[u8]` only.
unsafe impl Sync for AlignedDestination {}
