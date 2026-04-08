//! # hurray-ffi
//!
//! C ABI layer for hurray language bindings.
//!
//! This crate exposes opaque handles, a function table, and buffer release
//! callbacks via a stable C ABI. It is the foundation for all non-Rust language
//! bindings.
//!
//! ## Safety contract
//!
//! - No panics may propagate across the FFI boundary. All `extern "C"` functions
//!   wrap their bodies in `std::panic::catch_unwind`.
//! - All `unsafe` blocks carry a `// SAFETY:` comment explaining soundness.
//! - Buffer pointers MUST be aligned to at least 64 bytes.
