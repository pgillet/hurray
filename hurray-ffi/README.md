# hurray-ffi

The C ABI layer for **[Hurray](https://www.pascalgillet.net/hurray/)** — a zero-copy,
streamable, language-agnostic tensor interchange format for AI/ML inference pipelines
and scientific arrays.

This crate exposes [`hurray-core`](https://crates.io/crates/hurray-core) and
[`hurray-io`](https://crates.io/crates/hurray-io) over a stable C ABI — opaque handles, a
function table, buffer release callbacks, and blocking stream read/write over a transport
vtable the caller supplies — so any language that can call C can produce and consume
Hurray tensors without going through Rust. No panics cross the FFI boundary, and no async
runtime is linked: the stream decoding is a state machine, and the blocking happens inside
the caller's own callbacks.

## Documentation

- API docs: <https://docs.rs/hurray-ffi>
- C FFI requirements and the format specification: <https://www.pascalgillet.net/hurray/>
- Source: <https://github.com/pgillet/hurray>

## License

Licensed under either of MIT or Apache-2.0 at your option.
