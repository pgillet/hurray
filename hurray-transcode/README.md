# hurray-transcode

Representation conversion for **[Hurray](https://www.pascalgillet.net/hurray/)** — a
zero-copy, streamable, language-agnostic tensor interchange format for AI/ML inference
pipelines and scientific arrays.

Built on [`hurray-core`](https://crates.io/crates/hurray-core), this crate converts a
tensor's **representation** and nothing else: the mapping from index to value is preserved
exactly, or with a precision loss the target representation itself declares. It is not a
compute library and cannot become one — a matmul changes the value, and a reshape moves no
byte at all.

This version converts between the layouts with an affine stride form — row-major,
column-major and strided, including reversed axes and broadcasts — for whole-byte element
types. Every other conversion in the crate's scope is refused by name rather than
mishandled.

- **Plan before you move** — a plan is built from descriptors alone and publishes the
  output descriptor and buffer sizes before any byte moves, so a conversion can be refused
  against a budget.
- **The caller allocates** — convert straight into a NumPy array, a pinned staging buffer
  or an mmap'd region.
- **Sync core, parallelism at the edge** — a default build cannot spawn a thread, so it
  stays callable from an async context; the `rayon` feature fans a plan's work items over
  the thread pool.

## Documentation

- API docs: <https://docs.rs/hurray-transcode>
- What it converts today: [conversion matrix](https://github.com/pgillet/hurray/blob/main/docs/impl/transcode-matrix.md)
- Format specification, cookbook, and guides: <https://www.pascalgillet.net/hurray/>
- Source: <https://github.com/pgillet/hurray>

## License

Licensed under either of MIT or Apache-2.0 at your option.
