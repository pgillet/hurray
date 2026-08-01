# hurray-io

Async streaming and file I/O for **[Hurray](https://www.pascalgillet.net/hurray/)** — a
zero-copy, streamable, language-agnostic tensor interchange format for AI/ML inference.

Built on [`hurray-core`](https://crates.io/crates/hurray-core), this crate provides:

- **Streaming format** — self-delimiting, no-seek runtime interchange (in-process, IPC,
  cross-machine), including composite tensors.
- **File format** — the `HRRYFILE` container with named tensors, a footer index for random
  access, mmap-friendly alignment, and typed key-value metadata.

Async via Tokio.

## Documentation

- API docs: <https://docs.rs/hurray-io>
- Format specification, cookbook, and guides: <https://www.pascalgillet.net/hurray/>
- Source: <https://github.com/pgillet/hurray>

## License

Licensed under either of MIT or Apache-2.0 at your option.
