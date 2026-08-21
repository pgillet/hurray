# Layer 7 — C FFI Cookbook

The `hurray-ffi` crate exposes a stable C ABI over `hurray-core` types.
All functions return a `HurrayStatus` integer (`0` = OK, negative = error).
All handles are opaque — never inspect their internals.

## ABI version check

Always verify the ABI version at startup so mismatched builds are caught early.

**C:**
```c
#include "hurray.h"
#include <assert.h>

void startup_check(void) {
    uint32_t v = hurray_c_abi_version();
    assert(v == 2 && "unexpected Hurray C ABI version");
}
```

**Rust (via FFI):**
```rust
use hurray_ffi::{hurray_c_abi_version, HURRAY_C_ABI_VERSION};

assert_eq!(unsafe { hurray_c_abi_version() }, HURRAY_C_ABI_VERSION);
```

## Creating a buffer with a release callback

The release callback is called exactly once by `hurray_buffer_destroy`.
Use it to free or unmap the underlying memory.

**C:**
```c
#include "hurray.h"
#include <stdlib.h>
#include <stdio.h>

static void my_release(void *data, void *ctx) {
    (void)ctx;
    free(data);
    printf("buffer freed\n");
}

HurrayBuffer *create_cpu_buffer(size_t n_bytes) {
    void *data = aligned_alloc(64, n_bytes);
    if (!data) return NULL;

    HurrayBuffer *handle = NULL;
    HurrayStatus s = hurray_buffer_from_ptr(
        data, (uint64_t)n_bytes,
        /*alignment=*/64,
        /*device_tag=*/0x00,    /* CPU */
        /*sync_mode=*/0x00,     /* ProducerSynced */
        /*memory_class=*/0x00,  /* Standard */
        my_release, /*release_context=*/NULL,
        &handle
    );
    if (s != HURRAY_OK) { free(data); return NULL; }
    return handle;
}
```

**Rust:**
```rust
use hurray_ffi::{hurray_buffer_from_ptr, HurrayBuffer, HurrayReleaseCallback, HURRAY_OK};
use hurray_core::{DeviceTag, MemoryClass, SyncMode, MIN_BUFFER_ALIGNMENT};
use std::ffi::c_void;

unsafe extern "C" fn release(data: *mut c_void, _ctx: *mut c_void) {
    drop(Vec::from_raw_parts(data as *mut u8, 4096, 4096));
}

let mut storage = vec![0u8; 4096];
let ptr = storage.as_mut_ptr() as *mut c_void;
std::mem::forget(storage);

let mut handle: *mut HurrayBuffer = std::ptr::null_mut();
let status = unsafe {
    hurray_buffer_from_ptr(
        ptr, 4096, MIN_BUFFER_ALIGNMENT,
        DeviceTag::Cpu.to_byte(),
        SyncMode::ProducerSynced.to_byte(),
        MemoryClass::Standard.to_byte(),
        Some(release) as HurrayReleaseCallback,
        std::ptr::null_mut(),
        &mut handle,
    )
};
assert_eq!(status, HURRAY_OK);
```

## Destroying a buffer

```c
HurrayStatus s = hurray_buffer_destroy(handle);
assert(s == HURRAY_OK);
/* handle is invalid after this point — do not dereference */
```

In debug builds, a second call to `hurray_buffer_destroy` on the same handle
returns `HURRAY_ERR_INTERNAL` (sentinel-based double-free detection).

## Decoding a tensor descriptor

```c
#include "hurray.h"

HurrayDescriptor *decode_descriptor(const uint8_t *bytes, size_t len) {
    HurrayDescriptor *desc = NULL;
    HurrayStatus s = hurray_descriptor_decode(bytes, len, &desc);
    if (s != HURRAY_OK) return NULL; /* inspect s for the specific error */
    return desc;
}

void inspect(HurrayDescriptor *desc) {
    uint32_t rank;
    hurray_descriptor_rank(desc, &rank);

    uint64_t dims[64];
    size_t capacity = rank;
    hurray_descriptor_shape(desc, dims, &capacity);
    /* capacity now holds the true rank; dims[0..rank] are the dimension sizes */

    hurray_descriptor_destroy(desc);
}
```

### Shape capacity/query pattern

`hurray_descriptor_shape` uses an in/out `out_rank` parameter:

1. Set `*out_rank` to the number of `uint64_t` slots in `out_dims`.
2. If the function returns `HURRAY_ERR_BUFFER_TOO_SMALL`, `*out_rank` has been
   updated to the true rank — allocate that many slots and retry.

```c
size_t cap = 0;
/* Query-only call: out_dims=NULL forces BUFFER_TOO_SMALL, writes true rank */
hurray_descriptor_shape(desc, NULL, &cap);

uint64_t *dims = malloc(cap * sizeof(uint64_t));
hurray_descriptor_shape(desc, dims, &cap);
```

## Sync mode handoff cross-check

Before consuming a GPU buffer, call the matching handoff function to verify
that the producer's declared sync mode matches your payload.

```c
/* Event mode: producer recorded a CUDA event */
HurraySyncEventPayload payload = {
    .sync_handle           = cuda_event,
    .sync_handle_device_tag = 0x01, /* CUDA */
    .event_release_fn      = my_event_release,
    .event_release_context = NULL,
};
HurrayStatus s = hurray_buffer_handoff_event(buffer, &payload);
if (s == HURRAY_ERR_SYNC_MODE_MISMATCH) { /* handle disagreement */ }
```

```c
/* ConsumerStream mode: consumer declares its target stream */
HurraySyncConsumerStreamPayload sp = {
    .consumer_stream            = my_cuda_stream,
    .consumer_stream_device_tag = 0x01, /* CUDA */
};
HurrayStatus s = hurray_buffer_handoff_consumer_stream(buffer, &sp);
```

```c
/* ProducerSynced mode: producer issued a host-side wait; no payload needed */
HurrayStatus s = hurray_buffer_handoff_producer_synced(buffer);
```

## Reading a capsule from outside Python

A native-protocol capsule carries a `HurrayBufferList` as its pointer and a
`HurrayTensorContext` as its context. The list holds the bytes; the context holds the
descriptor that says what those bytes *are* (ADR-034). Without it a consumer gets
element data with no element type, shape, or layout — which is what every non-Python
consumer got before ADR-034.

**Check the version first.** It is the one accessor guaranteed to work across ABI
versions, and every other accessor assumes a caller that has already checked:

```c
uint32_t abi_version = 0;
if (hurray_tensor_context_abi_version(ctx, &abi_version) != HURRAY_OK) return -1;
if (abi_version != HURRAY_C_ABI_VERSION) {
    /* Producer and consumer disagree — refuse rather than dereference. */
    return -1;
}
```

Then borrow the descriptor and decode it:

```c
const uint8_t *bytes = NULL;
uint64_t len = 0;
if (hurray_tensor_context_descriptor(ctx, &bytes, &len) != HURRAY_OK) return -1;

HurrayDescriptor *descriptor = NULL;
if (hurray_descriptor_decode(bytes, (uintptr_t)len, &descriptor) != HURRAY_OK) return -1;

uint8_t type_tag = 0;
uint32_t rank = 0;
hurray_descriptor_element_type_tag(descriptor, &type_tag);
hurray_descriptor_rank(descriptor, &rank);
```

The pointer `bytes` is **borrowed** — owned by the context, valid until the context is
destroyed. Copy it if you need it longer. An empty descriptor reports a null pointer
and a zero length.

Now the buffers mean something, because the descriptor said what they hold:

```c
uint64_t count = 0;
hurray_buffer_list_len(list, &count);
for (uint64_t i = 0; i < count; i++) {
    HurrayBuffer *borrowed = NULL;          /* owned by the list — do not destroy */
    hurray_buffer_list_get(list, i, &borrowed);

    void *ptr = NULL;
    uint64_t size = 0;
    hurray_buffer_data_ptr(borrowed, &ptr);
    hurray_buffer_byte_size(borrowed, &size);
}
```

Destroying the context runs its `owner_release` callback, which is how the producer
learns it can let the tensor go:

```c
hurray_descriptor_destroy(descriptor);
hurray_tensor_context_destroy(&ctx);       /* runs owner_release exactly once */
hurray_buffer_list_destroy(&list);         /* destroys every handle it owns */
```

Runnable version, in Rust because that is what this repository builds — but the
sequence is the one any language follows:

```
cargo run -p hurray-ffi --example tensor_context
```

## Key takeaways

- **No panics cross the boundary.** Every function returns `HURRAY_OK` or a
  negative error code. `HURRAY_ERR_INTERNAL_PANIC` means the library panicked
  internally — the handle is in an undefined state and MUST NOT be reused.
- **Opaque handles.** `HurrayBuffer`, `HurrayDescriptor`, `HurrayReader`, and
  `HurrayWriter` are opaque; never dereference or cast their pointers.
- **`HURRAY_ERR_NULL_POINTER` for null required arguments.** Every function
  checks its required pointer arguments and returns this code immediately if
  any is null. Optional context pointers (e.g., `release_context`) MAY be null.
- **Exactly one destroy per create.** Each handle created by a `*_from_ptr`,
  `*_decode`, or `*_new` function MUST be destroyed exactly once.
- **Read the ABI version before anything else.** For a `HurrayTensorContext` this is
  normative, not advisory: it is what lets later versions add accessors without
  breaking consumers compiled against an earlier header.
