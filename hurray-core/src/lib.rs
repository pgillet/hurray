//! # hurray-core
//!
//! Core types for the hurray tensor interchange format.
//!
//! This crate provides the format types, tensor descriptor, buffer handle, and
//! quantization descriptors. It has no I/O and no async dependencies — it is the
//! foundation for all other hurray crates.
//!
//! ## Feature flags
//!
//! | Feature | Effect |
//! |---------|--------|
//! | `serde` | Derives `serde::Serialize` / `serde::Deserialize` for all public types |

pub mod error;

pub use error::{Error, Result};
