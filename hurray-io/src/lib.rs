//! # hurray-io
//!
//! Async streaming and file format read/write for the hurray tensor interchange format.
//!
//! This crate builds on [`hurray_core`] and provides async I/O via Tokio.
//!
//! ## Feature flags
//!
//! | Feature | Effect |
//! |---------|--------|
//! | `tokio` | Enables async read/write using `tokio::io` traits |

pub mod error;

// Not behind the `tokio` feature: the allocation contract belongs to the format, not to
// the async drivers, and the sans-io core (ADR-041 § 2) needs it too.
mod aligned;

pub mod file;

// The stream module carries the sans-io core, which needs no runtime; only its drivers
// are behind the `tokio` feature (ADR-041 § 1).
pub mod stream;

pub use error::{Error, Result};
