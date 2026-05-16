//! # hurray-io
//!
//! Async streaming and file format read/write for the hurray tensor interchange format.
//!
//! This crate builds on [`hurray-core`] and provides async I/O via Tokio.
//!
//! ## Feature flags
//!
//! | Feature | Effect |
//! |---------|--------|
//! | `tokio` | Enables async read/write using `tokio::io` traits |

pub mod error;

#[cfg(feature = "tokio")]
pub mod file;

#[cfg(feature = "tokio")]
pub mod stream;

pub use error::{Error, Result};
