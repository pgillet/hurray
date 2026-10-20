//! The caller-supplied transport vtable (ADR-041 § 8).
//!
//! Hurray never opens a socket, a pipe or a file for a C caller. It is handed four
//! things — a context pointer and up to three callbacks — and asks them for bytes. That
//! is what keeps an async runtime out of a C consumer's binary, and it composes with
//! anything the caller can read from or write to.
//!
//! The callbacks are **blocking by contract**, and that contract belongs here rather than
//! to the sans-io core: a callback that blocks until it has bytes is correct and expected,
//! precisely because the core's own progress API never blocks.

use std::ffi::c_void;

use crate::status::{HurrayStatus, HURRAY_ERR_NULL_POINTER, HURRAY_ERR_TRANSPORT, HURRAY_OK};

// ── Callback types ────────────────────────────────────────────────────────────

/// Reads up to `len` bytes into `dst`, reporting the count through `out_filled`.
///
/// - Returns [`HURRAY_OK`] and writes the number of bytes transferred, which MAY be fewer
///   than `len`. A short read is normal and is not an error.
/// - Writing `0` with [`HURRAY_OK`] means **end of input**.
/// - On any other return value the callback MUST have transferred **no bytes** and MUST
///   write `0` through `out_filled`. A transport that partially succeeds and then fails
///   MUST report the partial count with [`HURRAY_OK`] and surface the failure on the next
///   call — exactly what POSIX `read` does, and the one place a hand-written adapter
///   tends to go wrong.
///
/// A callback MUST NOT call any Hurray function on the handle that invoked it.
pub type HurrayTransportReadFn = Option<
    unsafe extern "C" fn(
        ctx: *mut c_void,
        dst: *mut u8,
        len: usize,
        out_filled: *mut usize,
    ) -> HurrayStatus,
>;

/// Writes up to `len` bytes from `src`, reporting the count through `out_written`.
///
/// - Returns [`HURRAY_OK`] and writes the number of bytes accepted, which MAY be fewer
///   than `len`.
/// - Writing `0` with [`HURRAY_OK`] is **not** success: it means the sink will accept no
///   more bytes. Hurray reports that as [`HURRAY_ERR_SINK_CLOSED`][crate::HURRAY_ERR_SINK_CLOSED]
///   when an item is in flight.
/// - On any other return value the callback MUST have transferred **no bytes** and MUST
///   write `0` through `out_written`, per the rule on [`HurrayTransportReadFn`].
///
/// A callback MUST NOT call any Hurray function on the handle that invoked it.
pub type HurrayTransportWriteFn = Option<
    unsafe extern "C" fn(
        ctx: *mut c_void,
        src: *const u8,
        len: usize,
        out_written: *mut usize,
    ) -> HurrayStatus,
>;

/// Releases the transport's context pointer when the handle that owns it is destroyed.
///
/// Hurray never frees `ctx` itself — it belongs to the caller — so this callback is the
/// only place a caller's transport object is torn down. Symmetric with the buffer release
/// callbacks that already cross this ABI (ADR-009).
pub type HurrayTransportReleaseFn = Option<unsafe extern "C" fn(ctx: *mut c_void)>;

// ── The vtable ────────────────────────────────────────────────────────────────

/// A caller-supplied byte transport: a context pointer plus the callbacks that move bytes.
///
/// Not a path and not a file descriptor. A vtable composes with a socket, a pipe, a memory
/// span, or the caller's own object, and file descriptors are a portability wart of their
/// own.
///
/// The struct is copied on construction, so the caller MAY free the `HurrayTransport`
/// itself as soon as the handle is created; `ctx` must outlive the handle, and is released
/// through [`release`][Self::release] when the handle is destroyed.
///
/// Thread affinity: **one handle, one thread at a time.** Hurray runs no threads of its
/// own, so this is a rule rather than a threading model.
#[repr(C)]
pub struct HurrayTransport {
    /// Opaque caller context, passed verbatim to every callback. MAY be null.
    pub ctx: *mut c_void,
    /// Required by a reader; MAY be null for a write-only transport.
    pub read: HurrayTransportReadFn,
    /// Required by a writer; MAY be null for a read-only transport.
    pub write: HurrayTransportWriteFn,
    /// Invoked once when the owning handle is destroyed. MAY be null.
    pub release: HurrayTransportReleaseFn,
}

// ── Internal owned copy ───────────────────────────────────────────────────────

/// A handle's own copy of the vtable, which releases `ctx` when the handle drops.
pub(crate) struct Transport {
    ctx: *mut c_void,
    read: HurrayTransportReadFn,
    write: HurrayTransportWriteFn,
    release: HurrayTransportReleaseFn,
}

impl Transport {
    /// Copies a caller-supplied vtable.
    ///
    /// # Safety
    ///
    /// `vtable` MUST point to a readable [`HurrayTransport`] for the duration of the call.
    pub(crate) unsafe fn from_c(vtable: *const HurrayTransport) -> Self {
        Self {
            ctx: (*vtable).ctx,
            read: (*vtable).read,
            write: (*vtable).write,
            release: (*vtable).release,
        }
    }

    /// Whether this transport can read.
    pub(crate) fn can_read(&self) -> bool {
        self.read.is_some()
    }

    /// Whether this transport can write.
    pub(crate) fn can_write(&self) -> bool {
        self.write.is_some()
    }

    /// Fills up to `dst.len()` bytes, returning how many arrived.
    ///
    /// `Err(status)` means no bytes moved and the caller's stream position is intact.
    pub(crate) fn read(&self, dst: &mut [u8]) -> Result<usize, HurrayStatus> {
        let Some(f) = self.read else {
            return Err(HURRAY_ERR_NULL_POINTER);
        };
        let mut filled: usize = 0;
        // SAFETY: `dst` is a live slice, so its pointer is valid and writable for
        // `dst.len()` bytes; `filled` is a live local. The callback's own contract covers
        // `ctx`, which Hurray only ever passes through.
        let status = unsafe { f(self.ctx, dst.as_mut_ptr(), dst.len(), &mut filled) };
        if status != HURRAY_OK {
            return Err(status);
        }
        // A count beyond what was offered is a broken callback. Refusing it here keeps the
        // core's position provably correct instead of handing it a report it must reject.
        if filled > dst.len() {
            return Err(HURRAY_ERR_TRANSPORT);
        }
        Ok(filled)
    }

    /// Writes from `src`, returning how many bytes the sink accepted.
    ///
    /// `Err(status)` means no bytes reached the sink.
    pub(crate) fn write(&self, src: &[u8]) -> Result<usize, HurrayStatus> {
        let Some(f) = self.write else {
            return Err(HURRAY_ERR_NULL_POINTER);
        };
        let mut written: usize = 0;
        // SAFETY: `src` is a live slice, readable for `src.len()` bytes; `written` is a
        // live local. `ctx` is passed through untouched.
        let status = unsafe { f(self.ctx, src.as_ptr(), src.len(), &mut written) };
        if status != HURRAY_OK {
            return Err(status);
        }
        if written > src.len() {
            return Err(HURRAY_ERR_TRANSPORT);
        }
        Ok(written)
    }
}

impl Drop for Transport {
    fn drop(&mut self) {
        if let Some(f) = self.release.take() {
            // SAFETY: invoked exactly once, when the owning handle is destroyed. The
            // caller guaranteed `ctx` valid until this callback returns.
            unsafe { f(self.ctx) };
        }
    }
}
