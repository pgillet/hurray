//! Hurray is a tensor **interchange format**. This crate is its name, not its code.
//!
//! There is nothing to call here. The implementation is split by responsibility, so
//! depend on the piece you need:
//!
//! | Crate | What it gives you |
//! |---|---|
//! | [`hurray-core`](https://docs.rs/hurray-core) | Format types: tensor descriptor, element types, layouts, quantization, buffer handles. No I/O. |
//! | [`hurray-io`](https://docs.rs/hurray-io) | Async streaming interchange and the `HRRYFILE` container. |
//! | [`hurray-ffi`](https://docs.rs/hurray-ffi) | The C ABI, for bindings in other languages. |
//! | [`hurray-inspect`](https://crates.io/crates/hurray-inspect) | A CLI that prints a descriptor as an annotated hex table. |
//!
//! From Python, install [`pyhurray`](https://pypi.org/project/pyhurray/) — the
//! distribution name differs because `hurray` on PyPI is an unrelated package, but the
//! module is imported as `hurray`.
//!
//! ```sh
//! cargo add hurray-core
//! ```
//!
//! Specification, cookbook and API reference: <https://www.pascalgillet.net/hurray/>
//!
//! # Why this crate exists
//!
//! It holds the umbrella name alongside the crates that do the work, so the obvious name
//! for the project cannot come to mean something else. It deliberately re-exports
//! nothing: a facade would be a second public API to keep in step with `hurray-core` and
//! would fix the name to one crate's shape permanently. If an umbrella that pulls the
//! pieces together behind feature flags is ever wanted, it can grow here — that direction
//! is additive, and the reverse would be a breaking removal.
