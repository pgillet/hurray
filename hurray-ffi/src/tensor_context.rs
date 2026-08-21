//! Opaque carrier for everything a native-protocol capsule holds beyond its
//! buffers (ADR-034).
//!
//! A capsule carries two things: its pointer is a
//! [`HurrayBufferList`](crate::HurrayBufferList), and its context is a
//! [`HurrayTensorContext`]. The list holds the bytes; the context holds the
//! encoded tensor descriptor that says what those bytes *are* — element type,
//! shape, layout, quantization — plus the ABI version of the build that produced
//! them.
//!
//! Before ADR-034 the context was a Rust struct private to `hurray-python`, so
//! only `hurray-python` could read it. The buffers crossed the language boundary
//! and the descriptor did not, which made the protocol's full-fidelity promise
//! true between two Python peers and false for every other binding.
//!
//! ## Reading one
//!
//! Check the version first, always — see [`hurray_tensor_context_abi_version`].
//! Every other accessor assumes a caller that has done so.
//!
//! ## The owner
//!
//! A context keeps its producer's tensor alive: the buffers point into memory
//! something else owns. That something is passed in as an opaque `owner` pointer
//! with an `owner_release` callback, and this crate never interprets either —
//! which is what lets `hurray-python` park a Python reference there without any
//! Python type reaching the C ABI.

use std::ffi::c_void;

use crate::{
    panic::{catch, null_check},
    status::{HurrayStatus, HURRAY_ERR_INTERNAL, HURRAY_ERR_NULL_POINTER, HURRAY_OK},
};

// ── Debug sentinel constants ──────────────────────────────────────────────────

/// Sentinel placed in `HurrayTensorContext.sentinel` on allocation (debug builds only).
#[cfg(debug_assertions)]
const SENTINEL_LIVE: u64 = 0x1157_C7C0_0000_0000;

/// Sentinel written before the allocation is freed, to detect a double destroy.
#[cfg(debug_assertions)]
const SENTINEL_DEAD: u64 = 0x1157_C7C0_DEAD_DEAD;

// ── Callback type ─────────────────────────────────────────────────────────────

/// Callback invoked exactly once by [`hurray_tensor_context_destroy`] to release
/// whatever keeps the tensor's memory alive.
///
/// The argument is the `owner` pointer passed to [`hurray_tensor_context_new`],
/// with the exact value provided at construction time.
///
/// Implementations MUST NOT call back into the Hurray C ABI from within this
/// callback; doing so may cause deadlocks or use-after-free.
pub type HurrayOwnerReleaseFn = Option<unsafe extern "C" fn(*mut c_void)>;

// ── HurrayTensorContext ───────────────────────────────────────────────────────

/// Opaque handle to a capsule's tensor context: descriptor bytes, ABI version,
/// and an owner reference.
///
/// Like every other handle in this ABI, this struct is **not** `#[repr(C)]`; its
/// layout is an implementation detail and callers MUST treat it as a black-box
/// pointer. Publishing the layout would buy nothing — a caller holding a capsule
/// already links this library to read its buffer list — and would freeze the
/// layout for the life of the major version (ADR-034 § 2).
pub struct HurrayTensorContext {
    /// Double-destroy sentinel (debug builds only).
    #[cfg(debug_assertions)]
    sentinel: u64,
    /// ABI version of the build that produced this context.
    abi_version: u32,
    /// Owned copy of the encoded tensor descriptor.
    ///
    /// Copied rather than borrowed: a borrow would tie this handle's validity to
    /// a buffer the producer may drop, and a descriptor is small beside the
    /// tensor it describes.
    descriptor: Vec<u8>,
    /// Opaque pointer to whatever owns the tensor's memory. Never interpreted.
    owner: *mut c_void,
    /// Callback that releases `owner`, invoked exactly once on destroy.
    owner_release: HurrayOwnerReleaseFn,
}

// SAFETY: mirrors the `HurrayBuffer` contract — the C ABI is single-thread-at-a-time
// per handle, so `Send` only permits moving the box across threads during
// construction and destruction, never concurrent access.
unsafe impl Send for HurrayTensorContext {}

// ── Constructor ───────────────────────────────────────────────────────────────

/// Creates a [`HurrayTensorContext`] owning a copy of `descriptor_bytes`.
///
/// `abi_version` MUST be the producing build's `HURRAY_C_ABI_VERSION`; a consumer
/// compares it against its own before trusting anything else about the capsule.
///
/// `owner` and `owner_release` are optional and opaque. When `owner_release` is
/// non-null it is invoked exactly once, with `owner`, during
/// [`hurray_tensor_context_destroy`].
///
/// The caller owns the returned handle and MUST destroy it exactly once.
///
/// # Safety
///
/// - `out_ctx` MUST be a valid, non-null, writable pointer.
/// - `descriptor_bytes` MUST point to at least `descriptor_len` readable bytes,
///   unless `descriptor_len` is `0`, in which case it MAY be null.
/// - `owner` MUST remain valid until `owner_release` is invoked.
///
/// # Examples
///
/// ```
/// use hurray_ffi::tensor_context::{
///     hurray_tensor_context_destroy, hurray_tensor_context_new,
/// };
/// use hurray_ffi::{HurrayTensorContext, HURRAY_C_ABI_VERSION, HURRAY_OK};
///
/// let descriptor = [0x48u8, 0x52, 0x52, 0x59]; // "HRRY"
/// let mut ctx: *mut HurrayTensorContext = std::ptr::null_mut();
/// assert_eq!(
///     unsafe {
///         hurray_tensor_context_new(
///             HURRAY_C_ABI_VERSION,
///             descriptor.as_ptr(),
///             descriptor.len() as u64,
///             std::ptr::null_mut(),
///             None,
///             &mut ctx,
///         )
///     },
///     HURRAY_OK,
/// );
///
/// assert_eq!(unsafe { hurray_tensor_context_destroy(&mut ctx) }, HURRAY_OK);
/// assert!(ctx.is_null()); // destroy nulls the caller's pointer
/// ```
#[no_mangle]
pub unsafe extern "C" fn hurray_tensor_context_new(
    abi_version: u32,
    descriptor_bytes: *const u8,
    descriptor_len: u64,
    owner: *mut c_void,
    owner_release: HurrayOwnerReleaseFn,
    out_ctx: *mut *mut HurrayTensorContext,
) -> HurrayStatus {
    catch(|| {
        null_check!(out_ctx);

        // A null pointer is only meaningful for an empty descriptor; anything
        // else is a caller bug that would otherwise become an unsound read.
        if descriptor_bytes.is_null() && descriptor_len != 0 {
            return HURRAY_ERR_NULL_POINTER;
        }

        let descriptor = if descriptor_len == 0 {
            Vec::new()
        } else {
            // SAFETY: caller guarantees descriptor_len readable bytes at descriptor_bytes.
            std::slice::from_raw_parts(descriptor_bytes, descriptor_len as usize).to_vec()
        };

        let boxed = Box::new(HurrayTensorContext {
            #[cfg(debug_assertions)]
            sentinel: SENTINEL_LIVE,
            abi_version,
            descriptor,
            owner,
            owner_release,
        });
        // SAFETY: just allocated; ownership transfers to the caller, who must
        // call hurray_tensor_context_destroy exactly once.
        *out_ctx = Box::into_raw(boxed);
        HURRAY_OK
    })
}

// ── Accessors ─────────────────────────────────────────────────────────────────

/// Reads the ABI version recorded by the producing build.
///
/// **Call this first.** It is the one accessor guaranteed to work on a context
/// produced by any version of this ABI; every other accessor assumes a caller
/// that has already compared this value against its own
/// [`HURRAY_C_ABI_VERSION`](crate::HURRAY_C_ABI_VERSION) and found it
/// compatible (ADR-034 § 4). That ordering is what lets later versions add
/// accessors without breaking older consumers.
///
/// # Safety
///
/// `ctx` MUST be a live handle from [`hurray_tensor_context_new`], and `out`
/// MUST be a valid, writable pointer.
///
/// # Examples
///
/// ```
/// use hurray_ffi::tensor_context::{
///     hurray_tensor_context_abi_version, hurray_tensor_context_destroy,
///     hurray_tensor_context_new,
/// };
/// use hurray_ffi::{HurrayTensorContext, HURRAY_C_ABI_VERSION, HURRAY_OK};
///
/// let mut ctx: *mut HurrayTensorContext = std::ptr::null_mut();
/// unsafe {
///     hurray_tensor_context_new(
///         HURRAY_C_ABI_VERSION, std::ptr::null(), 0, std::ptr::null_mut(), None, &mut ctx,
///     );
/// }
///
/// let mut version: u32 = 0;
/// assert_eq!(
///     unsafe { hurray_tensor_context_abi_version(ctx, &mut version) },
///     HURRAY_OK,
/// );
/// assert_eq!(version, HURRAY_C_ABI_VERSION);
///
/// unsafe { hurray_tensor_context_destroy(&mut ctx) };
/// ```
#[no_mangle]
pub unsafe extern "C" fn hurray_tensor_context_abi_version(
    ctx: *const HurrayTensorContext,
    out: *mut u32,
) -> HurrayStatus {
    catch(|| {
        null_check!(ctx, out);
        // SAFETY: ctx is non-null and points to a live HurrayTensorContext.
        *out = (*ctx).abi_version;
        HURRAY_OK
    })
}

/// Borrows the encoded tensor descriptor.
///
/// The returned pointer is owned by `ctx` and is valid until the context is
/// destroyed; the caller MUST NOT free it. Decode it with
/// [`hurray_descriptor_decode`](crate::descriptor::hurray_descriptor_decode) to
/// read the element type, shape, layout, and optional sections.
///
/// `out_len` is `0` — and `out_bytes` null — for a context created without a
/// descriptor.
///
/// # Safety
///
/// `ctx` MUST be a live handle from [`hurray_tensor_context_new`], and both out
/// pointers MUST be valid and writable.
///
/// # Examples
///
/// ```
/// use hurray_ffi::tensor_context::{
///     hurray_tensor_context_descriptor, hurray_tensor_context_destroy,
///     hurray_tensor_context_new,
/// };
/// use hurray_ffi::{HurrayTensorContext, HURRAY_C_ABI_VERSION, HURRAY_OK};
///
/// let descriptor = [0x48u8, 0x52, 0x52, 0x59];
/// let mut ctx: *mut HurrayTensorContext = std::ptr::null_mut();
/// unsafe {
///     hurray_tensor_context_new(
///         HURRAY_C_ABI_VERSION,
///         descriptor.as_ptr(),
///         descriptor.len() as u64,
///         std::ptr::null_mut(),
///         None,
///         &mut ctx,
///     );
/// }
///
/// let mut bytes: *const u8 = std::ptr::null();
/// let mut len: u64 = 0;
/// assert_eq!(
///     unsafe { hurray_tensor_context_descriptor(ctx, &mut bytes, &mut len) },
///     HURRAY_OK,
/// );
/// assert_eq!(len, 4);
/// assert_eq!(unsafe { std::slice::from_raw_parts(bytes, len as usize) }, &descriptor);
///
/// unsafe { hurray_tensor_context_destroy(&mut ctx) };
/// ```
#[no_mangle]
pub unsafe extern "C" fn hurray_tensor_context_descriptor(
    ctx: *const HurrayTensorContext,
    out_bytes: *mut *const u8,
    out_len: *mut u64,
) -> HurrayStatus {
    catch(|| {
        null_check!(ctx, out_bytes, out_len);
        // SAFETY: ctx is non-null and points to a live HurrayTensorContext.
        let descriptor = &(*ctx).descriptor;
        // An empty descriptor reports a null pointer rather than a dangling
        // one-past-the-end address, so a caller that ignores the length cannot
        // read from it.
        *out_bytes = if descriptor.is_empty() {
            std::ptr::null()
        } else {
            descriptor.as_ptr()
        };
        *out_len = descriptor.len() as u64;
        HURRAY_OK
    })
}

// ── Destructor ────────────────────────────────────────────────────────────────

/// Destroys a [`HurrayTensorContext`], invoking its owner-release callback, and
/// nulls the caller's pointer.
///
/// Destroying a null handle is a no-op that returns [`HURRAY_OK`], so the call
/// is safe to make unconditionally on a cleanup path.
///
/// # Safety
///
/// - `ctx` MUST be a valid, writable pointer to a handle slot.
/// - The handle it points to MUST come from [`hurray_tensor_context_new`] and
///   MUST NOT have been destroyed already.
///
/// # Examples
///
/// ```
/// use hurray_ffi::tensor_context::{hurray_tensor_context_destroy, hurray_tensor_context_new};
/// use hurray_ffi::{HurrayTensorContext, HURRAY_C_ABI_VERSION, HURRAY_OK};
///
/// let mut ctx: *mut HurrayTensorContext = std::ptr::null_mut();
/// unsafe {
///     hurray_tensor_context_new(
///         HURRAY_C_ABI_VERSION, std::ptr::null(), 0, std::ptr::null_mut(), None, &mut ctx,
///     );
/// }
///
/// assert_eq!(unsafe { hurray_tensor_context_destroy(&mut ctx) }, HURRAY_OK);
/// assert!(ctx.is_null());
/// // Destroying again is a no-op, not a double free.
/// assert_eq!(unsafe { hurray_tensor_context_destroy(&mut ctx) }, HURRAY_OK);
/// ```
#[no_mangle]
pub unsafe extern "C" fn hurray_tensor_context_destroy(
    ctx: *mut *mut HurrayTensorContext,
) -> HurrayStatus {
    catch(|| {
        null_check!(ctx);

        // SAFETY: ctx is a valid, writable pointer to a handle slot.
        let ptr = *ctx;
        if ptr.is_null() {
            return HURRAY_OK;
        }

        #[cfg(debug_assertions)]
        {
            // SAFETY: ptr is non-null and points to a live HurrayTensorContext.
            if (*ptr).sentinel == SENTINEL_DEAD {
                return HURRAY_ERR_INTERNAL;
            }
            // Written through the raw pointer before Box::from_raw so the freed
            // slot carries the dead marker; writing after would only update the
            // moved-out local copy.
            (*ptr).sentinel = SENTINEL_DEAD;
        }

        // Take the callback and its argument before freeing, then null the
        // caller's slot, so a re-entrant or panicking callback cannot reach a
        // handle that is mid-destruction.
        // SAFETY: ptr is non-null and points to a live HurrayTensorContext.
        let release = (*ptr).owner_release.take();
        let owner = std::mem::replace(&mut (*ptr).owner, std::ptr::null_mut());

        // SAFETY: created by hurray_tensor_context_new via Box::into_raw; this is
        // the first and only destroy.
        drop(Box::from_raw(ptr));
        *ctx = std::ptr::null_mut();

        if let Some(release) = release {
            // SAFETY: the caller guaranteed at construction time that owner stays
            // valid until this call, and this is the only call.
            release(owner);
        }
        HURRAY_OK
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a context over `bytes`, runs `f`, and destroys it.
    fn with_context(bytes: &[u8], f: impl FnOnce(*mut HurrayTensorContext)) {
        let mut ctx: *mut HurrayTensorContext = std::ptr::null_mut();
        let status = unsafe {
            hurray_tensor_context_new(
                crate::HURRAY_C_ABI_VERSION,
                bytes.as_ptr(),
                bytes.len() as u64,
                std::ptr::null_mut(),
                None,
                &mut ctx,
            )
        };
        assert_eq!(status, HURRAY_OK);
        f(ctx);
        assert_eq!(
            unsafe { hurray_tensor_context_destroy(&mut ctx) },
            HURRAY_OK
        );
        assert!(ctx.is_null());
    }

    #[test]
    fn descriptor_survives_the_round_trip() {
        let bytes = [1u8, 2, 3, 4, 5];
        with_context(&bytes, |ctx| {
            let mut out: *const u8 = std::ptr::null();
            let mut len: u64 = 0;
            assert_eq!(
                unsafe { hurray_tensor_context_descriptor(ctx, &mut out, &mut len) },
                HURRAY_OK
            );
            assert_eq!(len, 5);
            // SAFETY: the accessor returned a live borrow of the context's copy.
            assert_eq!(
                unsafe { std::slice::from_raw_parts(out, len as usize) },
                &bytes
            );
        });
    }

    #[test]
    fn the_descriptor_is_copied_not_borrowed() {
        let mut ctx: *mut HurrayTensorContext = std::ptr::null_mut();
        {
            // Dropped before the context is read: a borrow would dangle here.
            // Heap, not an array: freeing the allocation is what makes a borrow a
            // real use-after-free rather than a stale stack read that may pass.
            #[allow(clippy::useless_vec)]
            let transient = vec![9u8, 8, 7];
            unsafe {
                hurray_tensor_context_new(
                    crate::HURRAY_C_ABI_VERSION,
                    transient.as_ptr(),
                    transient.len() as u64,
                    std::ptr::null_mut(),
                    None,
                    &mut ctx,
                );
            }
        }

        let mut out: *const u8 = std::ptr::null();
        let mut len: u64 = 0;
        unsafe { hurray_tensor_context_descriptor(ctx, &mut out, &mut len) };
        // SAFETY: the context owns its copy, which outlives `transient`.
        assert_eq!(
            unsafe { std::slice::from_raw_parts(out, len as usize) },
            &[9, 8, 7]
        );
        unsafe { hurray_tensor_context_destroy(&mut ctx) };
    }

    #[test]
    fn abi_version_round_trips() {
        with_context(&[0u8], |ctx| {
            let mut version = 0u32;
            assert_eq!(
                unsafe { hurray_tensor_context_abi_version(ctx, &mut version) },
                HURRAY_OK
            );
            assert_eq!(version, crate::HURRAY_C_ABI_VERSION);
        });
    }

    #[test]
    fn an_empty_descriptor_reports_null_and_zero() {
        let mut ctx: *mut HurrayTensorContext = std::ptr::null_mut();
        assert_eq!(
            unsafe {
                hurray_tensor_context_new(
                    crate::HURRAY_C_ABI_VERSION,
                    std::ptr::null(),
                    0,
                    std::ptr::null_mut(),
                    None,
                    &mut ctx,
                )
            },
            HURRAY_OK
        );

        let mut out: *const u8 = std::ptr::null();
        let mut len: u64 = 1;
        unsafe { hurray_tensor_context_descriptor(ctx, &mut out, &mut len) };
        assert!(out.is_null(), "no dangling one-past-the-end pointer");
        assert_eq!(len, 0);
        unsafe { hurray_tensor_context_destroy(&mut ctx) };
    }

    #[test]
    fn a_null_descriptor_with_a_nonzero_length_is_rejected() {
        let mut ctx: *mut HurrayTensorContext = std::ptr::null_mut();
        assert_eq!(
            unsafe {
                hurray_tensor_context_new(
                    crate::HURRAY_C_ABI_VERSION,
                    std::ptr::null(),
                    8,
                    std::ptr::null_mut(),
                    None,
                    &mut ctx,
                )
            },
            HURRAY_ERR_NULL_POINTER
        );
        assert!(ctx.is_null());
    }

    #[test]
    fn null_arguments_are_rejected_rather_than_dereferenced() {
        assert_eq!(
            unsafe {
                hurray_tensor_context_new(
                    crate::HURRAY_C_ABI_VERSION,
                    std::ptr::null(),
                    0,
                    std::ptr::null_mut(),
                    None,
                    std::ptr::null_mut(),
                )
            },
            HURRAY_ERR_NULL_POINTER
        );
        let mut version = 0u32;
        assert_eq!(
            unsafe { hurray_tensor_context_abi_version(std::ptr::null(), &mut version) },
            HURRAY_ERR_NULL_POINTER
        );
        assert_eq!(
            unsafe { hurray_tensor_context_destroy(std::ptr::null_mut()) },
            HURRAY_ERR_NULL_POINTER
        );
    }

    // ── Owner release ─────────────────────────────────────────────────────────

    static RELEASE_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    unsafe extern "C" fn count_release(owner: *mut c_void) {
        assert_eq!(owner as usize, 0xABCD, "owner arrives unchanged");
        RELEASE_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }

    #[test]
    fn the_owner_is_released_exactly_once() {
        RELEASE_COUNT.store(0, std::sync::atomic::Ordering::SeqCst);
        let mut ctx: *mut HurrayTensorContext = std::ptr::null_mut();
        unsafe {
            hurray_tensor_context_new(
                crate::HURRAY_C_ABI_VERSION,
                std::ptr::null(),
                0,
                0xABCD as *mut c_void,
                Some(count_release),
                &mut ctx,
            );
        }
        assert_eq!(RELEASE_COUNT.load(std::sync::atomic::Ordering::SeqCst), 0);

        unsafe { hurray_tensor_context_destroy(&mut ctx) };
        assert_eq!(RELEASE_COUNT.load(std::sync::atomic::Ordering::SeqCst), 1);

        // The handle is null now, so a second destroy must not release again.
        unsafe { hurray_tensor_context_destroy(&mut ctx) };
        assert_eq!(RELEASE_COUNT.load(std::sync::atomic::Ordering::SeqCst), 1);
    }
}
