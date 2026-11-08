//! A tensor handle: a descriptor and its buffers, paired structurally.
//!
//! A tensor *is* a descriptor plus its buffers, and the C ABI used to make callers keep
//! the two halves paired by convention — a consumer that fetched one without the other
//! got either bytes it could not interpret or a description of bytes it did not have
//! ([#174](https://github.com/pgillet/hurray/issues/174)). [`HurrayTensor`] owns both, so
//! the pairing is structural: a caller holding the handle holds the description of the
//! bytes it points at.
//!
//! Every path that produces a tensor produces this handle (ADR-043): a stream read, a
//! file read, and a native-protocol capsule, which carries one `HurrayTensor *` and no
//! context. One destroy call, and its internal order cannot be got wrong.
//!
//! ## Reading one
//!
//! Check the version first, always — see [`hurray_tensor_abi_version`]. Every other
//! accessor assumes a caller that has done so.
//!
//! ## The owner
//!
//! A producer's buffers point into memory something else owns, and that something is
//! passed in as an opaque `owner` pointer with an `owner_release` callback. This crate
//! never interprets either, which is what lets `hurray-python` park a Python reference
//! there without any Python type reaching the C ABI. A tensor from a read carries
//! neither: it owns its buffers outright.

use std::ffi::c_void;

use bytes::Bytes;
use hurray_core::TensorDescriptor;

use crate::{
    buffer::HurrayBuffer,
    buffer_list::{hurray_buffer_list_destroy, HurrayBufferList},
    descriptor::{hurray_descriptor_destroy, HurrayDescriptor},
    panic::{catch, null_check},
    status::{
        status_from_core_error, HurrayStatus, HURRAY_ERR_INTERNAL, HURRAY_ERR_NULL_POINTER,
        HURRAY_OK,
    },
};

// ── Debug sentinel constants ──────────────────────────────────────────────────

/// Sentinel placed in `HurrayTensor.sentinel` on allocation (debug builds only).
#[cfg(debug_assertions)]
const SENTINEL_LIVE: u64 = 0x7E43_0000_7E43_0000;

/// Sentinel written before the allocation is freed, to detect a double destroy in debug
/// builds.
#[cfg(debug_assertions)]
const SENTINEL_DEAD: u64 = 0x7E43_DEAD_7E43_DEAD;

// ── Callback type ─────────────────────────────────────────────────────────────

/// Callback invoked exactly once by [`hurray_tensor_destroy`] to release whatever keeps
/// the tensor's memory alive.
///
/// The argument is the `owner` pointer passed to [`hurray_tensor_new`], with the exact
/// value provided at construction time.
///
/// Implementations MUST NOT call back into the Hurray C ABI from within this callback;
/// doing so may cause deadlocks or use-after-free.
pub type HurrayOwnerReleaseFn = Option<unsafe extern "C" fn(*mut c_void)>;

// ── HurrayTensor ──────────────────────────────────────────────────────────────

/// Opaque handle to a tensor: its descriptor and its buffers.
///
/// Not `#[repr(C)]`; the layout is an implementation detail. The handle **owns** both
/// halves. [`hurray_tensor_descriptor`] and [`hurray_tensor_buffers`] hand out *borrowed*
/// pointers — valid until [`hurray_tensor_destroy`], and never to be destroyed
/// individually.
pub struct HurrayTensor {
    /// Double-destroy sentinel (debug builds only).
    #[cfg(debug_assertions)]
    sentinel: u64,
    /// ABI version of the build that produced this tensor.
    abi_version: u32,
    /// Owned; nulled as soon as it is destroyed so a re-entrant destroy cannot free twice.
    descriptor: *mut HurrayDescriptor,
    /// Owned, in descriptor buffer-table order.
    buffers: *mut HurrayBufferList,
    /// Opaque pointer to whatever owns the tensor's memory. Never interpreted.
    owner: *mut c_void,
    /// Callback that releases `owner`, invoked exactly once on destroy.
    owner_release: HurrayOwnerReleaseFn,
}

// SAFETY: mirrors the HurrayBuffer and HurrayBufferList contract — the C ABI is
// single-thread-at-a-time per handle, so Send only permits moving the box across threads
// during construction and destruction, never concurrent access.
unsafe impl Send for HurrayTensor {}

impl HurrayTensor {
    /// Takes ownership of a descriptor handle and a buffer list.
    pub(crate) fn from_parts(
        abi_version: u32,
        descriptor: *mut HurrayDescriptor,
        buffers: *mut HurrayBufferList,
        owner: *mut c_void,
        owner_release: HurrayOwnerReleaseFn,
    ) -> Box<Self> {
        Box::new(Self {
            #[cfg(debug_assertions)]
            sentinel: SENTINEL_LIVE,
            abi_version,
            descriptor,
            buffers,
            owner,
            owner_release,
        })
    }

    /// Builds a handle from an already-decoded descriptor, for a Rust producer.
    ///
    /// The Rust-side twin of [`hurray_tensor_new`], which decodes bytes a C caller
    /// supplied. `hurray-python` holds a `TensorDescriptor` already, so encoding it only
    /// for the constructor to decode it straight back would be a round trip through bytes
    /// neither side wants (ADR-043).
    ///
    /// # Safety
    ///
    /// - `buffers` MUST be a live list from
    ///   [`hurray_buffer_list_new`](crate::hurray_buffer_list_new) that nothing else owns;
    ///   ownership transfers to the returned handle.
    /// - `owner` MUST remain valid until `owner_release` is invoked.
    /// - The returned pointer MUST be destroyed exactly once, with
    ///   [`hurray_tensor_destroy`].
    pub unsafe fn from_core(
        abi_version: u32,
        descriptor: TensorDescriptor,
        buffers: *mut HurrayBufferList,
        owner: *mut c_void,
        owner_release: HurrayOwnerReleaseFn,
    ) -> *mut HurrayTensor {
        let descriptor = Box::into_raw(HurrayDescriptor::from_descriptor(descriptor));
        Box::into_raw(Self::from_parts(
            abi_version,
            descriptor,
            buffers,
            owner,
            owner_release,
        ))
    }

    /// Borrows the decoded descriptor as the `hurray-core` type, for a Rust consumer.
    ///
    /// The Rust-side twin of [`hurray_tensor_descriptor`], which hands a C caller the
    /// opaque handle to read fields through. A Rust consumer — `hurray-python` rebuilding
    /// its own tensor — wants the type itself, and re-encoding it to decode it again
    /// would be the round trip [`HurrayTensor::from_core`] exists to avoid.
    ///
    /// Returns `None` only for a handle mid-destruction, which a caller holding a live
    /// tensor never has.
    pub fn core_descriptor(&self) -> Option<&TensorDescriptor> {
        if self.descriptor.is_null() {
            return None;
        }
        // SAFETY: non-null and owned by this handle, which the caller holds alive.
        Some(unsafe { (*self.descriptor).inner() })
    }
}

// ── Constructor ───────────────────────────────────────────────────────────────

/// Creates a [`HurrayTensor`] from an encoded descriptor and a finished buffer list.
///
/// `abi_version` MUST be the producing build's
/// [`HURRAY_C_ABI_VERSION`](crate::HURRAY_C_ABI_VERSION); a consumer compares it against
/// its own before trusting anything else about the handle.
///
/// `descriptor_bytes` is **decoded during this call** and only the decoded form is kept,
/// so a producer cannot export a tensor whose descriptor does not parse: bytes that fail
/// to decode are rejected with the status the decode failed with. A null pointer or a
/// `descriptor_len` of `0` is rejected with
/// [`HURRAY_ERR_NULL_POINTER`] — a tensor with no
/// descriptor is a carrier for bytes nothing can interpret.
///
/// On success `buffers` is **owned by the tensor**, and the caller MUST NOT push into it,
/// destroy it, or use the handle at all. On failure ownership stays with the caller, who
/// MUST destroy it — the discipline
/// [`hurray_buffer_list_push`](crate::buffer_list::hurray_buffer_list_push) already
/// follows.
///
/// `owner` and `owner_release` are optional and opaque. When `owner_release` is non-null
/// it is invoked exactly once, with `owner`, during [`hurray_tensor_destroy`].
///
/// # Safety
///
/// - `out_tensor` MUST be a valid, non-null, writable pointer.
/// - `descriptor_bytes` MUST point to at least `descriptor_len` readable bytes.
/// - `buffers` MUST be a live list from
///   [`hurray_buffer_list_new`](crate::hurray_buffer_list_new) that nothing else owns.
/// - `owner` MUST remain valid until `owner_release` is invoked.
///
/// # Examples
///
/// ```
/// use hurray_core::{
///     BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode,
///     TensorDescriptor, MIN_BUFFER_ALIGNMENT,
/// };
/// use hurray_ffi::tensor::{hurray_tensor_destroy, hurray_tensor_new};
/// use hurray_ffi::{
///     hurray_buffer_list_new, HurrayBufferList, HurrayTensor, HURRAY_C_ABI_VERSION, HURRAY_OK,
/// };
///
/// let handle = BufferHandle::new(
///     16,
///     MIN_BUFFER_ALIGNMENT,
///     DeviceTag::Cpu,
///     SyncMode::ProducerSynced,
/// )?;
/// let descriptor = TensorDescriptor::new(
///     1,
///     0,
///     ElementType::Float32,
///     Shape::new(vec![4])?,
///     0,
///     LayoutDescriptor::RowMajor,
///     vec![handle],
///     None,
///     None,
///     None,
///     None,
/// )?;
/// let encoded = descriptor.encode()?;
///
/// // An empty list is enough to show the ownership transfer; a real producer pushes
/// // one buffer per descriptor buffer-table entry first.
/// let mut buffers: *mut HurrayBufferList = std::ptr::null_mut();
/// assert_eq!(unsafe { hurray_buffer_list_new(1, &mut buffers) }, HURRAY_OK);
///
/// let mut tensor: *mut HurrayTensor = std::ptr::null_mut();
/// assert_eq!(
///     unsafe {
///         hurray_tensor_new(
///             HURRAY_C_ABI_VERSION,
///             encoded.as_ptr(),
///             encoded.len() as u64,
///             buffers, // ownership transfers on success
///             std::ptr::null_mut(),
///             None,
///             &mut tensor,
///         )
///     },
///     HURRAY_OK,
/// );
///
/// // One call ends the whole tensor: descriptor, buffers, and the owner reference.
/// assert_eq!(unsafe { hurray_tensor_destroy(&mut tensor) }, HURRAY_OK);
/// assert!(tensor.is_null()); // destroy nulls the caller's pointer
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[no_mangle]
pub unsafe extern "C" fn hurray_tensor_new(
    abi_version: u32,
    descriptor_bytes: *const u8,
    descriptor_len: u64,
    buffers: *mut HurrayBufferList,
    owner: *mut c_void,
    owner_release: HurrayOwnerReleaseFn,
    out_tensor: *mut *mut HurrayTensor,
) -> HurrayStatus {
    catch(|| {
        null_check!(out_tensor, descriptor_bytes, buffers);

        // A zero-length descriptor is refused rather than stored: the handle keeps only
        // the decoded form, and there is nothing to decode (ADR-043 § 6).
        if descriptor_len == 0 {
            return HURRAY_ERR_NULL_POINTER;
        }

        // SAFETY: caller guarantees descriptor_len readable bytes at descriptor_bytes.
        let slice = std::slice::from_raw_parts(descriptor_bytes, descriptor_len as usize);
        let descriptor = match TensorDescriptor::decode(slice) {
            Ok(descriptor) => descriptor,
            // The list is not destroyed here: ownership only transfers on success, so a
            // producer that got the descriptor wrong still holds its buffers.
            Err(e) => return status_from_core_error(&e),
        };

        // SAFETY: the caller's contract on `buffers` and `owner` is this function's own.
        *out_tensor =
            HurrayTensor::from_core(abi_version, descriptor, buffers, owner, owner_release);
        HURRAY_OK
    })
}

// ── Accessors ─────────────────────────────────────────────────────────────────

/// Reads the ABI version recorded by the producing build.
///
/// **Call this first.** It is the one accessor guaranteed to work on a tensor produced by
/// any version of this ABI, and every other accessor assumes a caller that has already
/// compared this value against its own
/// [`HURRAY_C_ABI_VERSION`](crate::HURRAY_C_ABI_VERSION) and found it equal (ADR-043
/// § 5). That ordering is what lets later versions add accessors without breaking
/// consumers compiled against an earlier header.
///
/// A tensor from a stream or file read reports this library's version; one from
/// [`hurray_tensor_new`] reports what its producer passed.
///
/// # Safety
///
/// `handle` and `out` MUST be valid, non-null pointers.
///
/// # Examples
///
/// ```
/// # use hurray_core::{
/// #     BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode,
/// #     TensorDescriptor, MIN_BUFFER_ALIGNMENT,
/// # };
/// use hurray_ffi::tensor::{
///     hurray_tensor_abi_version, hurray_tensor_destroy, hurray_tensor_new,
/// };
/// use hurray_ffi::{
///     hurray_buffer_list_new, HurrayBufferList, HurrayTensor, HURRAY_C_ABI_VERSION, HURRAY_OK,
/// };
/// # let handle = BufferHandle::new(16, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced)?;
/// # let descriptor = TensorDescriptor::new(
/// #     1, 0, ElementType::Float32, Shape::new(vec![4])?, 0, LayoutDescriptor::RowMajor,
/// #     vec![handle], None, None, None, None,
/// # )?;
/// # let encoded = descriptor.encode()?;
/// # let mut buffers: *mut HurrayBufferList = std::ptr::null_mut();
/// # unsafe { hurray_buffer_list_new(1, &mut buffers) };
/// # let mut tensor: *mut HurrayTensor = std::ptr::null_mut();
/// # unsafe {
/// #     hurray_tensor_new(
/// #         HURRAY_C_ABI_VERSION, encoded.as_ptr(), encoded.len() as u64, buffers,
/// #         std::ptr::null_mut(), None, &mut tensor,
/// #     )
/// # };
/// let mut version: u32 = 0;
/// assert_eq!(
///     unsafe { hurray_tensor_abi_version(tensor, &mut version) },
///     HURRAY_OK,
/// );
/// if version != HURRAY_C_ABI_VERSION {
///     // Producer and consumer disagree — refuse rather than dereference further.
///     return Ok(());
/// }
/// # unsafe { hurray_tensor_destroy(&mut tensor) };
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[no_mangle]
pub unsafe extern "C" fn hurray_tensor_abi_version(
    handle: *const HurrayTensor,
    out: *mut u32,
) -> HurrayStatus {
    catch(|| {
        null_check!(handle, out);
        // SAFETY: handle is non-null and points to a live HurrayTensor.
        *out = (*handle).abi_version;
        HURRAY_OK
    })
}

/// Borrows the tensor's descriptor handle.
///
/// The returned handle carries every accessor in [`crate::descriptor`] and
/// [`crate::field`]. It is **borrowed**: the caller MUST NOT call
/// [`hurray_descriptor_destroy`] on it, and MUST keep the tensor alive while it reads.
///
/// # Safety
///
/// `handle` and `out_descriptor` MUST be valid, non-null pointers.
///
/// # Examples
///
/// ```
/// # use hurray_core::{
/// #     BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode,
/// #     TensorDescriptor, MIN_BUFFER_ALIGNMENT,
/// # };
/// use hurray_ffi::descriptor::hurray_descriptor_rank;
/// use hurray_ffi::tensor::{hurray_tensor_descriptor, hurray_tensor_destroy, hurray_tensor_new};
/// use hurray_ffi::{
///     hurray_buffer_list_new, HurrayBufferList, HurrayDescriptor, HurrayTensor,
///     HURRAY_C_ABI_VERSION, HURRAY_OK,
/// };
/// # let handle = BufferHandle::new(16, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced)?;
/// # let descriptor = TensorDescriptor::new(
/// #     1, 0, ElementType::Float32, Shape::new(vec![4])?, 0, LayoutDescriptor::RowMajor,
/// #     vec![handle], None, None, None, None,
/// # )?;
/// # let encoded = descriptor.encode()?;
/// # let mut buffers: *mut HurrayBufferList = std::ptr::null_mut();
/// # unsafe { hurray_buffer_list_new(1, &mut buffers) };
/// # let mut tensor: *mut HurrayTensor = std::ptr::null_mut();
/// # unsafe {
/// #     hurray_tensor_new(
/// #         HURRAY_C_ABI_VERSION, encoded.as_ptr(), encoded.len() as u64, buffers,
/// #         std::ptr::null_mut(), None, &mut tensor,
/// #     )
/// # };
/// let mut descriptor: *mut HurrayDescriptor = std::ptr::null_mut();
/// assert_eq!(
///     unsafe { hurray_tensor_descriptor(tensor, &mut descriptor) },
///     HURRAY_OK,
/// );
///
/// let mut rank: u32 = 0;
/// unsafe { hurray_descriptor_rank(descriptor, &mut rank) };
/// assert_eq!(rank, 1);
/// # unsafe { hurray_tensor_destroy(&mut tensor) };
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[no_mangle]
pub unsafe extern "C" fn hurray_tensor_descriptor(
    handle: *const HurrayTensor,
    out_descriptor: *mut *mut HurrayDescriptor,
) -> HurrayStatus {
    catch(|| {
        null_check!(handle, out_descriptor);
        // SAFETY: handle is non-null and points to a live HurrayTensor.
        *out_descriptor = (*handle).descriptor;
        HURRAY_OK
    })
}

/// Borrows the tensor's buffer list.
///
/// Element `i` of the list is buffer `i` of the descriptor's buffer table. The list is
/// **borrowed**: the caller MUST NOT call [`hurray_buffer_list_destroy`] on it, and MUST
/// keep the tensor alive while it reads the buffers.
///
/// # Safety
///
/// `handle` and `out_buffers` MUST be valid, non-null pointers.
///
/// # Examples
///
/// ```
/// # use hurray_core::{
/// #     BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode,
/// #     TensorDescriptor, MIN_BUFFER_ALIGNMENT,
/// # };
/// use hurray_ffi::buffer_list::hurray_buffer_list_len;
/// use hurray_ffi::tensor::{hurray_tensor_buffers, hurray_tensor_destroy, hurray_tensor_new};
/// use hurray_ffi::{
///     hurray_buffer_list_new, HurrayBufferList, HurrayTensor, HURRAY_C_ABI_VERSION, HURRAY_OK,
/// };
/// # let handle = BufferHandle::new(16, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced)?;
/// # let descriptor = TensorDescriptor::new(
/// #     1, 0, ElementType::Float32, Shape::new(vec![4])?, 0, LayoutDescriptor::RowMajor,
/// #     vec![handle], None, None, None, None,
/// # )?;
/// # let encoded = descriptor.encode()?;
/// # let mut buffers: *mut HurrayBufferList = std::ptr::null_mut();
/// # unsafe { hurray_buffer_list_new(1, &mut buffers) };
/// # let mut tensor: *mut HurrayTensor = std::ptr::null_mut();
/// # unsafe {
/// #     hurray_tensor_new(
/// #         HURRAY_C_ABI_VERSION, encoded.as_ptr(), encoded.len() as u64, buffers,
/// #         std::ptr::null_mut(), None, &mut tensor,
/// #     )
/// # };
/// let mut borrowed: *mut HurrayBufferList = std::ptr::null_mut();
/// assert_eq!(
///     unsafe { hurray_tensor_buffers(tensor, &mut borrowed) },
///     HURRAY_OK,
/// );
///
/// let mut len: u64 = 0;
/// unsafe { hurray_buffer_list_len(borrowed, &mut len) };
/// # unsafe { hurray_tensor_destroy(&mut tensor) };
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[no_mangle]
pub unsafe extern "C" fn hurray_tensor_buffers(
    handle: *const HurrayTensor,
    out_buffers: *mut *mut HurrayBufferList,
) -> HurrayStatus {
    catch(|| {
        null_check!(handle, out_buffers);
        // SAFETY: handle is non-null and points to a live HurrayTensor.
        *out_buffers = (*handle).buffers;
        HURRAY_OK
    })
}

// ── Destructor ────────────────────────────────────────────────────────────────

/// Destroys a tensor handle — its descriptor, every buffer it owns, and its owner
/// reference — then writes null through `handle`.
///
/// One call rather than two, so the teardown order is not the caller's problem. A
/// non-null `owner_release` is invoked exactly once, with the `owner` pointer given at
/// construction.
///
/// # Safety
///
/// - `handle` MUST be a valid, non-null pointer to a `*mut HurrayTensor` produced by
///   [`hurray_tensor_new`], by [`hurray_reader_next_tensor`][crate::hurray_reader_next_tensor],
///   by [`hurray_file_reader_read_tensor`][crate::hurray_file_reader_read_tensor], or
///   taken out of a native-protocol capsule.
/// - That tensor MUST NOT have been destroyed already.
///
/// # Examples
///
/// ```
/// # use hurray_core::{
/// #     BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode,
/// #     TensorDescriptor, MIN_BUFFER_ALIGNMENT,
/// # };
/// use hurray_ffi::tensor::{hurray_tensor_destroy, hurray_tensor_new};
/// use hurray_ffi::{
///     hurray_buffer_list_new, HurrayBufferList, HurrayTensor, HURRAY_C_ABI_VERSION,
///     HURRAY_ERR_INTERNAL, HURRAY_OK,
/// };
/// # let handle = BufferHandle::new(16, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced)?;
/// # let descriptor = TensorDescriptor::new(
/// #     1, 0, ElementType::Float32, Shape::new(vec![4])?, 0, LayoutDescriptor::RowMajor,
/// #     vec![handle], None, None, None, None,
/// # )?;
/// # let encoded = descriptor.encode()?;
/// # let mut buffers: *mut HurrayBufferList = std::ptr::null_mut();
/// # unsafe { hurray_buffer_list_new(1, &mut buffers) };
/// # let mut tensor: *mut HurrayTensor = std::ptr::null_mut();
/// # unsafe {
/// #     hurray_tensor_new(
/// #         HURRAY_C_ABI_VERSION, encoded.as_ptr(), encoded.len() as u64, buffers,
/// #         std::ptr::null_mut(), None, &mut tensor,
/// #     )
/// # };
/// assert_eq!(unsafe { hurray_tensor_destroy(&mut tensor) }, HURRAY_OK);
/// assert!(tensor.is_null());
///
/// // A second destroy finds the caller's variable dead and says so, rather than
/// // freeing anything twice.
/// assert_eq!(
///     unsafe { hurray_tensor_destroy(&mut tensor) },
///     HURRAY_ERR_INTERNAL,
/// );
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[no_mangle]
pub unsafe extern "C" fn hurray_tensor_destroy(handle: *mut *mut HurrayTensor) -> HurrayStatus {
    catch(|| {
        null_check!(handle);
        let tensor = *handle;
        if tensor.is_null() {
            return HURRAY_ERR_INTERNAL;
        }

        #[cfg(debug_assertions)]
        {
            // SAFETY: tensor is non-null and points to a live HurrayTensor. Read and mark
            // through the raw pointer before Box::from_raw, so the dead marker lands in
            // the heap slot a second destroy would inspect.
            if (*tensor).sentinel == SENTINEL_DEAD {
                return HURRAY_ERR_INTERNAL;
            }
            (*tensor).sentinel = SENTINEL_DEAD;
        }

        // SAFETY: produced by HurrayTensor::from_parts via Box::into_raw; the caller
        // guarantees this is the first and only destroy call.
        let mut boxed = Box::from_raw(tensor);

        // Taken before anything is freed, so a re-entrant or panicking release callback
        // cannot reach a handle that is mid-destruction.
        let release = boxed.owner_release.take();
        let owner = std::mem::replace(&mut boxed.owner, std::ptr::null_mut());

        let descriptor = std::mem::replace(&mut boxed.descriptor, std::ptr::null_mut());
        if !descriptor.is_null() {
            // SAFETY: owned by this tensor and destroyed exactly once.
            hurray_descriptor_destroy(descriptor);
        }
        let mut buffers = std::mem::replace(&mut boxed.buffers, std::ptr::null_mut());
        if !buffers.is_null() {
            // SAFETY: owned by this tensor and destroyed exactly once; the list destroys
            // every handle it owns.
            hurray_buffer_list_destroy(&mut buffers);
        }

        drop(boxed);
        *handle = std::ptr::null_mut();

        // Last, and after the buffers: a producer's buffers point into memory the owner
        // keeps alive, so releasing the owner first would hand the buffer release
        // callbacks freed memory.
        if let Some(release) = release {
            // SAFETY: the caller guaranteed at construction time that owner stays valid
            // until this call, and this is the only call.
            release(owner);
        }
        HURRAY_OK
    })
}

// ── Handing a decoded tensor to C ─────────────────────────────────────────────

/// Releases one buffer's backing `Bytes` when its [`HurrayBuffer`] is destroyed.
///
/// # Safety
///
/// `context` MUST be the `Box<Bytes>` leaked by [`tensor_handle`] for this buffer, and
/// MUST NOT have been released already.
unsafe extern "C" fn release_bytes(_data: *mut c_void, context: *mut c_void) {
    if !context.is_null() {
        // SAFETY: context is the Box<Bytes> leaked for this buffer; hurray_buffer_destroy
        // invokes a release callback exactly once.
        drop(Box::from_raw(context as *mut Bytes));
    }
}

/// Wraps a decoded tensor in the handles a C caller can hold.
///
/// No payload is copied: each `HurrayBuffer` points into the destination the core
/// allocated and the transport filled, and owns a reference to it. Shared by the stream
/// and file readers — a decoded tensor is the same thing whichever core produced it.
///
/// A read-produced tensor reports this build's ABI version and carries no owner: it owns
/// its buffers outright, so there is nothing external to keep alive (ADR-043 § 1).
pub(crate) fn tensor_handle(
    descriptor: TensorDescriptor,
    buffers: Vec<Bytes>,
) -> Box<HurrayTensor> {
    let mut handles: Vec<*mut HurrayBuffer> = Vec::with_capacity(buffers.len());
    for (bytes, handle) in buffers.into_iter().zip(descriptor.buffers.iter().copied()) {
        let boxed = Box::new(bytes);
        let data = boxed.as_ptr() as *mut c_void;
        let context = Box::into_raw(boxed) as *mut c_void;
        handles.push(Box::into_raw(HurrayBuffer::from_parts(
            data,
            handle,
            Some(release_bytes),
            context,
        )));
    }

    let list = Box::into_raw(HurrayBufferList::from_handles(handles));
    let descriptor = Box::into_raw(HurrayDescriptor::from_descriptor(descriptor));
    HurrayTensor::from_parts(
        crate::HURRAY_C_ABI_VERSION,
        descriptor,
        list,
        std::ptr::null_mut(),
        None,
    )
}
