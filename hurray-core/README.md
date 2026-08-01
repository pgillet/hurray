# hurray-core

Core types for **[Hurray](https://www.pascalgillet.net/hurray/)** — a zero-copy,
streamable, language-agnostic tensor interchange format for AI/ML inference.

This crate is the no-I/O, no-async foundation: the tensor descriptor, element-type system,
buffer handle, quantization descriptors, and memory-layout vocabulary (dense, strided,
tiled, Morton/Hilbert, sparse COO/CSR/CSC/CSF, block-paged, composite). Higher layers build
on it:

- [`hurray-io`](https://crates.io/crates/hurray-io) — async streaming + file format
- [`hurray-ffi`](https://crates.io/crates/hurray-ffi) — C ABI for language bindings

## Documentation

- API docs: <https://docs.rs/hurray-core>
- Format specification, cookbook, and guides: <https://www.pascalgillet.net/hurray/>
- Source: <https://github.com/pgillet/hurray>

## License

Licensed under either of MIT or Apache-2.0 at your option.
