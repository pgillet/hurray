# C FFI Layer Requirements — Hurray Implementation Requirements

## Overview

The `hurray-ffi` crate exposes a stable C ABI that allows non-Rust runtimes to
consume and produce Hurray tensors without depending on Rust tooling. It is the
foundation for all non-Python language bindings.

## ABI Stability

- All public symbols MUST use the `#[no_mangle]` attribute and `extern "C"` linkage.
- The ABI MUST be declared stable across patch versions and SHOULD be stable across
  minor versions. Breaking ABI changes require a major version bump.
- All struct layouts exposed across the FFI boundary MUST be `#[repr(C)]`.
- Enums exposed across the FFI boundary MUST be `#[repr(u8)]` or `#[repr(i32)]` as
  appropriate, never `#[repr(Rust)]`.

## C ABI Version

The C ABI carries a single `uint32` version identifier exposed via a constant
`HURRAY_C_ABI_VERSION` and a runtime accessor `hurray_c_abi_version()`. The
current version is `4`:

| Version | Changes |
|---------|---------|
| `1` | Initial C ABI: opaque handles, buffer release callbacks, panic-safe error returns. |
| `2` | Per-mode buffer handoff sync payloads (`SYNC_PRODUCER_SYNCED`, `SYNC_EVENT`, `SYNC_CONSUMER_STREAM`) and the event-release callback. See [Buffer Handoff Synchronisation](#buffer-handoff-synchronisation). |
| `3` | `HurrayBufferList` for multi-buffer tensors, and the native protocol capsule now wraps a list rather than a single `HurrayBuffer` (ADR-030). See [Buffer Lists](#buffer-lists). |
| `4` | `HurrayTensorContext`, so a consumer in any language can read a capsule's descriptor and ABI version (ADR-034). See [Tensor Context](#tensor-context). |

A consumer of the C ABI MUST query `hurray_c_abi_version()` before invoking any
function whose contract changed in a later version. A consumer compiled against
version `1` of the ABI that links against a runtime providing version `2` will
receive buffer handles whose `sync_mode` is `SYNC_PRODUCER_SYNCED` (`0x00`) by
default, which is the safe fallback: a version-`1` consumer that does not
inspect `sync_mode` will still observe the strongest synchronisation guarantee
and will not race against the producer's device writes.

## Opaque Handles

All Hurray objects crossing the FFI boundary MUST be represented as **opaque pointer
handles**. Callers MUST NOT dereference or inspect the pointed-to memory directly.

| Handle type | Represents |
|---|---|
| `HurrayDescriptor*` | A parsed tensor descriptor |
| `HurrayBuffer*` | A buffer handle (data + metadata) |
| `HurrayBufferList*` | An ordered, owning collection of buffer handles |
| `HurrayTensorContext*` | A capsule's descriptor bytes, ABI version, and owner reference |
| `HurrayReader*` | A streaming tensor reader |
| `HurrayWriter*` | A streaming tensor writer |

Each handle is obtained from a `hurray_*_create` function and MUST be released by
the corresponding `hurray_*_destroy` function. Double-free and use-after-free are
undefined behaviour on the caller side; the implementation MUST detect them in debug
builds (e.g., via a poisoned sentinel).

## Buffer Lists

A tensor whose descriptor references more than one buffer — per-channel / NF4 /
MXFP quantization, sparse layouts, block-paged, composite — needs all of its
buffers to travel together. `HurrayBufferList` is that carrier (ADR-030).

| Function | Contract |
|---|---|
| `hurray_buffer_list_new(capacity, out_list)` | Creates an empty list. `capacity` is a hint. |
| `hurray_buffer_list_push(list, buffer)` | Appends `buffer`, **transferring ownership** to the list on success. On failure ownership stays with the caller. |
| `hurray_buffer_list_len(list, out_len)` | Number of buffers in the list. |
| `hurray_buffer_list_get(list, index, out_buffer)` | Borrows the handle at `index`. Returns `HURRAY_ERR_INDEX_OUT_OF_BOUNDS` if `index >= len`. |
| `hurray_buffer_list_destroy(list)` | Destroys the list and every handle it owns, then writes null through `list`. |

The following rules are normative.

- A list **owns** every handle pushed into it. A handle obtained from
  `hurray_buffer_list_get` is **borrowed**: the caller MUST NOT call
  `hurray_buffer_destroy` on it. It remains valid until the list is destroyed.
- Buffers MUST be pushed in descriptor buffer-table order: element `i` of the
  list is buffer index `i` of the descriptor.
- `hurray_buffer_list_destroy` takes a **pointer to the caller's handle
  variable** and MUST write null through it, so the caller's variable is
  observably dead and a repeated destroy is a no-op. This is the sound half of
  the "release marks the structure released" discipline: the list allocation is
  freed, so only memory the caller owns can carry the marker.
- Destroying a list MUST destroy every owned handle exactly once, and MUST null
  each owned slot as it goes, so that a release callback which panics or
  re-enters cannot cause a double free.

## Tensor Context

A native-protocol capsule carries two things: a `HurrayBufferList` as its pointer,
and a `HurrayTensorContext` as its context (ADR-034). The list holds the bytes; the
context holds the **encoded tensor descriptor** that says what those bytes are, plus
the ABI version of the build that produced them.

Before ADR-034 the context was a structure private to `hurray-python`, so the buffers
crossed the language boundary and the descriptor did not. A consumer in any other
language received element bytes with no element type, shape, layout, or quantization.

```c
HurrayStatus hurray_tensor_context_new(uint32_t abi_version,
                                       const uint8_t *descriptor_bytes,
                                       uint64_t descriptor_len,
                                       void *owner,
                                       HurrayOwnerReleaseFn owner_release,
                                       struct HurrayTensorContext **out_ctx);

HurrayStatus hurray_tensor_context_abi_version(const struct HurrayTensorContext *ctx,
                                               uint32_t *out);
HurrayStatus hurray_tensor_context_descriptor(const struct HurrayTensorContext *ctx,
                                              const uint8_t **out_bytes,
                                              uint64_t *out_len);
HurrayStatus hurray_tensor_context_destroy(struct HurrayTensorContext **ctx);
```

- A consumer MUST call `hurray_tensor_context_abi_version` **first** and compare the
  result against its own `HURRAY_C_ABI_VERSION` before calling any other accessor.
  That ordering is what allows later ABI versions to add accessors without breaking
  older consumers: a consumer that checked the version knows which ones exist.
- `hurray_tensor_context_descriptor` returns a **borrow** owned by the context, valid
  until the context is destroyed. A caller that needs it longer MUST copy it. An empty
  descriptor reports a null pointer and a zero length.
- The context owns a copy of the descriptor bytes; a borrow would tie its validity to
  a buffer the producer may drop.
- `owner` and `owner_release` are opaque and never interpreted. They exist so a
  producer can keep whatever owns the tensor's memory alive for the capsule's
  lifetime — `hurray-python` parks a Python object reference there, which is how a
  Python type is kept out of the C ABI entirely. `owner_release` is invoked exactly
  once, during `hurray_tensor_context_destroy`.
- Destroying a null handle is a no-op returning `HURRAY_OK`, so cleanup paths may call
  it unconditionally. Destroy nulls the caller's pointer.

## Panic Safety

Rust panics MUST NOT propagate across the FFI boundary. Every `extern "C"` function
that calls Rust code MUST wrap the call in `std::panic::catch_unwind`. If a panic is
caught, the function MUST:

1. Log or store the panic message (implementation-defined).
2. Return a well-defined error code (e.g., `HURRAY_ERR_INTERNAL_PANIC`).
3. Leave no partially-constructed state visible to the caller.

## Error Handling

All fallible FFI functions MUST return an error code of type `HurrayStatus`
(`int32`). The value `0` (`HURRAY_OK`) indicates success. All other values indicate
errors.

```c
typedef int32_t HurrayStatus;

#define HURRAY_OK                    0
#define HURRAY_ERR_INVALID_MAGIC    -1
#define HURRAY_ERR_VERSION_MISMATCH -2
#define HURRAY_ERR_INVALID_LAYOUT   -3
#define HURRAY_ERR_INVALID_TYPE     -4
#define HURRAY_ERR_BUFFER_TOO_SMALL -5
#define HURRAY_ERR_NULL_POINTER     -6
#define HURRAY_ERR_INTERNAL_PANIC   -7
/* ... */
```

Functions MUST return `HURRAY_ERR_NULL_POINTER` for any required pointer argument
that is `NULL`, without invoking undefined behaviour.

## Buffer Release Callbacks

Buffer handles carry a **release callback** to support zero-copy buffer sharing with
non-Rust runtimes. When `hurray-ffi` wraps an externally-owned buffer, the caller
provides a release function and a context pointer:

```c
typedef void (*HurrayReleaseCallback)(void* buffer, void* context);

HurrayStatus hurray_buffer_from_ptr(
    void*                 data,
    uint64_t              byte_size,
    uint32_t              alignment,
    uint8_t               device_tag,
    HurrayReleaseCallback release,
    void*                 release_context,
    HurrayBuffer**        out_handle
);
```

The release callback MUST be called exactly once when the buffer's reference count
reaches zero. The implementation MUST NOT call the release callback from a destructor
that runs on a foreign thread without the caller's consent.

## Thread Safety

- All handles MUST be safe to use from a single thread at a time (i.e., `Send` but
  not `Sync` in Rust terms).
- Concurrent access to the same handle from multiple threads is undefined behaviour
  unless documented otherwise.
- Reference counting for shared buffer handles MUST be performed with atomic
  operations (`std::sync::atomic`).

## Naming Conventions

All public symbols MUST be prefixed with `hurray_`. Type names use `Hurray` prefix
with PascalCase. Error codes use `HURRAY_ERR_` prefix with SCREAMING_SNAKE_CASE.

## Header Generation

A C header file (`hurray.h`) MUST be generated from the Rust source using `cbindgen`
as part of the build process. The generated header MUST be checked into the repository
and kept in sync with the Rust source. CI MUST fail if the generated header differs
from the committed one.

## Buffer Handoff Synchronisation

The buffer protocol's `sync_mode` field (see `docs/spec/buffer-protocol.md`
§ Stream and Event Synchronisation) declares one of three producer-side
synchronisation mechanisms in the binary descriptor. The C ABI carries the
corresponding **payload** out of band; the discriminant itself is read from the
buffer handle's binary descriptor and is NOT duplicated in the ABI struct.

The ABI layer MUST cross-check the `sync_mode` declared in the descriptor
against the payload provided at handoff time. A mismatch is a producer-side
bug; the ABI MUST reject the handoff with `HURRAY_ERR_SYNC_MODE_MISMATCH`
before returning a buffer handle to the consumer.

### Per-Mode Payloads

#### `sync_mode = SYNC_PRODUCER_SYNCED` (`0x00`)

No additional fields beyond the buffer pointer, byte size, alignment, device
tag, release callback, and release context defined in [Buffer Release
Callbacks](#buffer-release-callbacks).

The producer MUST have issued a host-side wait on the device stream(s) that
wrote the buffer before calling the handoff function. For CPU buffers
(`device_tag == 0x00`), the producer MUST have issued a host memory fence if
concurrent host-side writes exist.

#### `sync_mode = SYNC_EVENT` (`0x01`)

The handoff struct carries an opaque event handle and an event-release callback
in addition to the buffer fields:

| Field | Type | Description |
|-------|------|-------------|
| `sync_handle` | opaque pointer (`void*`) | Device-vendor-specific event handle (e.g., `cudaEvent_t`, `hipEvent_t`, `MTLSharedEvent`, `VkSemaphore`). Opaque to the ABI layer. |
| `sync_handle_device_tag` | `uint8` | Device tag identifying the event handle's driver context. MUST equal the buffer's `device_tag`. The ABI MUST reject the handoff if it does not. |
| `event_release_fn` | `void (*)(void* sync_handle, void* context)` | Callback the consumer calls exactly once after issuing its stream-wait. Thread-safe. |
| `event_release_context` | opaque pointer (`void*`) | Context pointer passed to `event_release_fn`. |

The producer MUST record the event on the writing stream(s) before calling the
handoff function. The producer MUST NOT defer event recording past the
handoff; doing so would allow the consumer to issue a stream-wait on an
unrecorded event and deadlock.

The consumer MUST issue a device-stream-wait on `sync_handle` on every stream
that will access the buffer before enqueuing any work that touches the
buffer's bytes. The consumer MUST call `event_release_fn(sync_handle,
event_release_context)` exactly once after all stream-waits have been issued
(typically immediately after handoff). `event_release_fn` MUST be safe to call
from any thread.

The event-release callback is **separate from** the buffer-release callback
defined in [Buffer Release Callbacks](#buffer-release-callbacks): a consumer
in `SYNC_EVENT` mode makes two release calls per buffer, with independent
lifetimes.

#### `sync_mode = SYNC_CONSUMER_STREAM` (`0x02`)

At handoff request time, the consumer supplies an opaque stream handle to the
producer:

| Field | Type | Description |
|-------|------|-------------|
| `consumer_stream` | opaque pointer (`void*`) | Device-vendor-specific stream handle (e.g., `cudaStream_t`, `hipStream_t`, `id<MTLCommandQueue>`). Opaque to the ABI layer. Supplied by the consumer at handoff request time. |
| `consumer_stream_device_tag` | `uint8` | Device tag identifying the stream handle's driver context. MUST equal the buffer's `device_tag`. The ABI MUST reject the handoff if it does not. |

The producer MUST issue a device-side ordering dependency from its writing
stream(s) onto `consumer_stream` before the handoff function returns a buffer
handle. The producer-side stream is not blocked; the ordering is established
on the device.

The consumer MAY access the buffer on `consumer_stream` after handoff returns,
but MUST NOT access the buffer on any other stream until it has issued an
inter-stream wait on that other stream.

`SYNC_CONSUMER_STREAM` does not introduce a second release callback: there is
no event handle whose lifetime must be managed.

### ABI Cross-Check

For every buffer handed off through the C ABI, the implementation MUST:

1. Read the `sync_mode` byte at offset 13 of the buffer handle binary
   descriptor.
2. Verify that the payload provided by the producer at handoff time matches
   the declared `sync_mode`:
   - `SYNC_PRODUCER_SYNCED` MUST NOT carry a `sync_handle` or `consumer_stream`.
   - `SYNC_EVENT` MUST carry a non-NULL `sync_handle`, an `event_release_fn`,
     and a `sync_handle_device_tag` equal to the buffer's `device_tag`.
   - `SYNC_CONSUMER_STREAM` MUST carry a non-NULL `consumer_stream` and a
     `consumer_stream_device_tag` equal to the buffer's `device_tag`.
3. Reject any reserved `sync_mode` value (`0x03`–`0xFE`) and the invalid value
   `0xFF` with `HURRAY_ERR_INVALID_SYNC_MODE`.
4. Reject a mismatch between descriptor and payload with
   `HURRAY_ERR_SYNC_MODE_MISMATCH`.

### Backward Compatibility for Pre-Version-2 Consumers

A consumer compiled against C ABI version `1` that does not inspect `sync_mode`
will receive buffers whose `sync_mode == SYNC_PRODUCER_SYNCED` by default. This
is the safe fallback: the strongest synchronisation guarantee, no per-mode
payload required, and behaviour identical to the version-`1` contract. A
producer that wishes to interoperate with version-`1` consumers MUST set
`sync_mode = SYNC_PRODUCER_SYNCED` for every buffer it hands off and MUST
issue the corresponding host-side wait before handoff.
