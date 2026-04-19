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

## Opaque Handles

All Hurray objects crossing the FFI boundary MUST be represented as **opaque pointer
handles**. Callers MUST NOT dereference or inspect the pointed-to memory directly.

| Handle type | Represents |
|---|---|
| `HurrayDescriptor*` | A parsed tensor descriptor |
| `HurrayBuffer*` | A buffer handle (data + metadata) |
| `HurrayReader*` | A streaming tensor reader |
| `HurrayWriter*` | A streaming tensor writer |

Each handle is obtained from a `hurray_*_create` function and MUST be released by
the corresponding `hurray_*_destroy` function. Double-free and use-after-free are
undefined behaviour on the caller side; the implementation MUST detect them in debug
builds (e.g., via a poisoned sentinel).

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
