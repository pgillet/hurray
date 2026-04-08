---
paths:
  - "hurray-ffi/**/*.rs"
  - "hurray-core/src/unsafe_*.rs"
  - "hurray-core/src/ffi*.rs"
---
# Unsafe and FFI Conventions

## SAFETY Comments

Every `unsafe` block MUST be preceded by a `// SAFETY:` comment explaining why the operation is sound.

```rust
// SAFETY: ptr is non-null and aligned to T, and the caller guarantees
// exclusive access for the lifetime of this reference.
let r = unsafe { &*ptr };
```

No `unsafe` block without a `// SAFETY:` comment will pass review.

## No Panics Across the FFI Boundary

The `hurray-ffi` crate MUST NOT let panics propagate across the C ABI boundary.
Every exported `extern "C"` function MUST wrap its body in `std::panic::catch_unwind`.

```rust
#[no_mangle]
pub extern "C" fn hurray_foo(handle: *mut HurrayHandle) -> HurrayStatus {
    match std::panic::catch_unwind(|| {
        // implementation
    }) {
        Ok(result) => result,
        Err(_) => HurrayStatus::InternalError,
    }
}
```

## No unwrap / expect in FFI or Unsafe Modules

`unwrap()` and `expect()` MUST NOT appear in `hurray-ffi` or in any module that contains `unsafe` blocks.
Use `?`, `match`, or explicit error returns instead.

## Opaque Handle Convention

Public C ABI handles MUST be opaque pointer types. Never expose Rust struct layout across the boundary.

```rust
// Correct: opaque handle
pub struct HurrayTensor(Box<TensorInner>);

// Wrong: layout exposed
#[repr(C)]
pub struct HurrayTensor { pub data: *mut u8, pub len: usize }
```

## Error Return Convention

All exported functions MUST return a `HurrayStatus` integer code (never `bool`, never a pointer that encodes error state).
Null output pointers indicate allocation failure, not logic errors.

## Alignment

Raw buffer pointers passed across the FFI boundary MUST be aligned to at least 64 bytes (SIMD minimum).
Page alignment (4096 bytes) is REQUIRED for GPU/IPC buffers.
The caller's alignment guarantee MUST be documented in the `// SAFETY:` comment.

## Isolation

`unsafe` code in `hurray-core` MUST be confined to dedicated modules named `unsafe_<purpose>.rs`
(e.g., `unsafe_buffer.rs`, `unsafe_cast.rs`). Do not scatter `unsafe` blocks across general modules.
