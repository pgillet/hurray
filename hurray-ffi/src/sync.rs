//! Synchronization mode handoff cross-checks for the Hurray C ABI.
//!
//! Before consuming a buffer produced by another party, the consumer MUST
//! call the appropriate handoff function to verify that the buffer's declared
//! [`SyncMode`] matches the payload they are providing. This prevents silent
//! race conditions caused by producer/consumer sync-mode disagreement.
//!
//! See `docs/spec/buffer-protocol.md § Synchronization Mode` and ADR-018.

use std::ffi::c_void;

use hurray_core::SyncMode;

use crate::{
    buffer::HurrayBuffer,
    panic::{catch, null_check},
    status::{HurrayStatus, HURRAY_ERR_SYNC_MODE_MISMATCH, HURRAY_OK},
};

// ── C-visible payload types ───────────────────────────────────────────────────

/// Optional callback invoked to release a sync event handle.
///
/// The first argument is the event handle; the second is the
/// `event_release_context` pointer. MUST NOT call back into the Hurray C ABI.
pub type HurrayEventReleaseFn = Option<unsafe extern "C" fn(*mut c_void, *mut c_void)>;

/// Payload for the `Event` sync mode handoff.
///
/// The consumer supplies this struct when calling
/// [`hurray_buffer_handoff_event`] to provide the device event that the
/// producer recorded and on which the consumer will wait.
#[repr(C)]
pub struct HurraySyncEventPayload {
    /// Non-null device-event handle (e.g., `cudaEvent_t`).
    pub sync_handle: *mut c_void,
    /// Wire byte of the device on which `sync_handle` was recorded.
    pub sync_handle_device_tag: u8,
    /// Release callback for `sync_handle`; MUST be non-null.
    pub event_release_fn: HurrayEventReleaseFn,
    /// Opaque context forwarded to `event_release_fn`; MAY be null.
    pub event_release_context: *mut c_void,
}

// SAFETY: Single-thread-at-a-time per the C ABI contract; the payload is only
// passed as a const pointer and is not mutated across threads.
unsafe impl Send for HurraySyncEventPayload {}

/// Payload for the `ConsumerStream` sync mode handoff.
///
/// The consumer supplies this struct when calling
/// [`hurray_buffer_handoff_consumer_stream`] to declare the stream on which
/// they intend to use the buffer.
#[repr(C)]
pub struct HurraySyncConsumerStreamPayload {
    /// Non-null consumer stream handle (e.g., `cudaStream_t`).
    pub consumer_stream: *mut c_void,
    /// Wire byte of the device that owns `consumer_stream`.
    pub consumer_stream_device_tag: u8,
}

// SAFETY: Single-thread-at-a-time per the C ABI contract; same rationale as
// HurraySyncEventPayload.
unsafe impl Send for HurraySyncConsumerStreamPayload {}

// ── Handoff functions ─────────────────────────────────────────────────────────

/// Validates an `Event`-mode sync handoff from producer to consumer.
///
/// Checks that:
/// - The buffer's declared sync mode is [`SyncMode::Event`].
/// - `payload.sync_handle` is non-null.
/// - `payload.event_release_fn` is non-null.
/// - `payload.sync_handle_device_tag` matches the buffer's declared device tag.
///
/// Returns [`HURRAY_OK`] if all checks pass; [`HURRAY_ERR_SYNC_MODE_MISMATCH`]
/// otherwise.
///
/// # Safety
///
/// `buffer` and `payload` MUST be valid, non-null pointers pointing to live
/// objects.
#[no_mangle]
pub unsafe extern "C" fn hurray_buffer_handoff_event(
    buffer: *const HurrayBuffer,
    payload: *const HurraySyncEventPayload,
) -> HurrayStatus {
    catch(|| {
        null_check!(buffer, payload);

        // SAFETY: buffer and payload are non-null and point to live objects.
        let sync_mode = (*buffer).handle.sync_mode();
        if sync_mode != SyncMode::Event {
            return HURRAY_ERR_SYNC_MODE_MISMATCH;
        }

        if (*payload).sync_handle.is_null() {
            return HURRAY_ERR_SYNC_MODE_MISMATCH;
        }

        if (*payload).event_release_fn.is_none() {
            return HURRAY_ERR_SYNC_MODE_MISMATCH;
        }

        // Device tag on the event must match the buffer's declared device.
        let buffer_device_byte = (*buffer).handle.device_tag().to_byte();
        if (*payload).sync_handle_device_tag != buffer_device_byte {
            return HURRAY_ERR_SYNC_MODE_MISMATCH;
        }

        HURRAY_OK
    })
}

/// Validates a `ConsumerStream`-mode sync handoff from producer to consumer.
///
/// Checks that:
/// - The buffer's declared sync mode is [`SyncMode::ConsumerStream`].
/// - `payload.consumer_stream` is non-null.
/// - `payload.consumer_stream_device_tag` matches the buffer's declared device
///   tag.
///
/// Returns [`HURRAY_OK`] if all checks pass; [`HURRAY_ERR_SYNC_MODE_MISMATCH`]
/// otherwise.
///
/// # Safety
///
/// `buffer` and `payload` MUST be valid, non-null pointers pointing to live
/// objects.
#[no_mangle]
pub unsafe extern "C" fn hurray_buffer_handoff_consumer_stream(
    buffer: *const HurrayBuffer,
    payload: *const HurraySyncConsumerStreamPayload,
) -> HurrayStatus {
    catch(|| {
        null_check!(buffer, payload);

        // SAFETY: buffer and payload are non-null and point to live objects.
        let sync_mode = (*buffer).handle.sync_mode();
        if sync_mode != SyncMode::ConsumerStream {
            return HURRAY_ERR_SYNC_MODE_MISMATCH;
        }

        if (*payload).consumer_stream.is_null() {
            return HURRAY_ERR_SYNC_MODE_MISMATCH;
        }

        // Device tag on the consumer stream must match the buffer's declared device.
        let buffer_device_byte = (*buffer).handle.device_tag().to_byte();
        if (*payload).consumer_stream_device_tag != buffer_device_byte {
            return HURRAY_ERR_SYNC_MODE_MISMATCH;
        }

        HURRAY_OK
    })
}

/// Validates a `ProducerSynced`-mode sync handoff.
///
/// Checks that the buffer's declared sync mode is
/// [`SyncMode::ProducerSynced`]. This mode requires no payload — calling
/// this function is itself the assertion that the producer has already issued
/// a host-side wait.
///
/// Returns [`HURRAY_OK`] if the buffer is `ProducerSynced`;
/// [`HURRAY_ERR_SYNC_MODE_MISMATCH`] otherwise.
///
/// # Safety
///
/// `buffer` MUST be a valid, non-null pointer to a live [`HurrayBuffer`].
#[no_mangle]
pub unsafe extern "C" fn hurray_buffer_handoff_producer_synced(
    buffer: *const HurrayBuffer,
) -> HurrayStatus {
    catch(|| {
        null_check!(buffer);

        // SAFETY: buffer is non-null and points to a live HurrayBuffer.
        if (*buffer).handle.sync_mode() != SyncMode::ProducerSynced {
            return HURRAY_ERR_SYNC_MODE_MISMATCH;
        }

        HURRAY_OK
    })
}
