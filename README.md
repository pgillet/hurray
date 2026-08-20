<p align="center">
  <img src="website/site/static/img/hurray-logo.svg" alt="Hurray" width="384">
</p>

<p align="center">
  A language-agnostic, zero-copy tensor format for AI/ML inference pipelines.<br>
  <a href="https://pgillet.github.io/hurray">pgillet.github.io/hurray</a>
</p>

---

Hurray defines two binary formats that share one tensor descriptor encoding:

- **Streaming format** — runtime interchange: in-process pointer passing, IPC, and
  cross-machine streaming. Self-delimiting, no seek required.
- **File format** — on-disk model storage: named tensors, footer index for random
  access, mmap-based zero-copy loading.

A descriptor parser written once works for both.

## Core properties

The contract of the format, most load-bearing first. Every section of the
[specification](https://pgillet.github.io/hurray/docs/stable/) must be consistent with
all of them.

- **Zero-copy first** — tensor data is shared by buffer handle, never by copy. 64-byte
  minimum alignment for SIMD, page-aligned for GPU and mmap.
- **Rich layout vocabulary** — twelve layouts, from row-major to Morton, Hilbert,
  sparse COO/CSR/CSC/CSF, block-paged KV caches, and composite tensors. Strides are in
  logical elements; negative and zero strides are valid.
- **First-class quantization** — five normative schemes (per-tensor, per-channel and
  per-block affine, NF4, MXFP) with normative dequantization formulas. Storage type and
  quantization scheme are orthogonal.
- **Streamable by design** — a descriptor always precedes its data, so a reader can
  start before the payload finishes arriving. No back-references, no buffering of the
  whole input.
- **Language-agnostic** — a stable C ABI and a binary spec written in generic type
  names. Any language that can read a struct from a buffer can implement it.
- **Device-aware** — every descriptor records where its data lives: device tag, memory
  class, and synchronization mode. Placement survives interchange between runtimes.
- **Inference-optimized types** — `float16` through `float64`, `int4` through `int64`,
  `bfloat16`, and the `float8` variants current accelerators want.
- **Self-describing** — a descriptor carries its own length in its first 10 bytes, and
  optional sections are flag-gated and length-prefixed, so a reader can skip what it
  does not understand instead of rejecting the tensor.
- **Built to evolve** — within `1.x`, tag values are never rebound and readers tolerate
  additive change. Private ranges (`0xF0`–`0xFE`) are reserved permanently.
- **Multi-transport** — one interchange protocol across in-process, shared memory, and
  the network, with layout negotiation built in.
- **An array-database foundation** — tiled, Morton, and Hilbert layouts preserve
  locality for sub-array queries; the file footer index is designed to extend to
  spatial and dimension-range indexes.

### Design invariants

- Descriptor precedes its data buffer; no back-references within a tensor
- Little-endian throughout — no endianness negotiation
- Strides in logical elements, not bytes
- Storage type (`type_tag`) is orthogonal to quantization scheme (`scheme_tag`)
- Streaming: self-delimiting, no end-of-file index, no seek
- File: footer index for random access, 4 KiB-aligned buffers for mmap

## Status and versioning

**Hurray is in beta.** No release before `1.0` guarantees forward or backward
compatibility with the format, and breaking changes are permitted until then: tags may
be renumbered, sections may change shape, and files written today may not be readable
by a later pre-`1.0` build.

From `1.0` onward the [versioning policy](docs/spec/versioning.md) applies in full —
backward compatibility within a major version, additive forward compatibility, and no
rebinding of allocated tags.

## Documentation

- [Website and specification](https://pgillet.github.io/hurray)
- [Prior art survey](docs/prior-art.md) — the formats and protocols that informed the
  design, and where Hurray differs
- [Architecture decisions](docs/adr/) — every non-obvious choice, with its alternatives
- [Project structure](PROJECT_STRUCTURE.md)
- Open questions are marked inline in the spec: `grep -rn "OQ-" docs/spec/`
- Ideas and tasks live in [GitHub issues](https://github.com/pgillet/hurray/issues)

## AI full disclosure

Hurray is developed with strong assistance from Claude (Anthropic), via Claude Code,
with a human leading the direction, the design decisions, and the review. Every
architectural choice is recorded as an ADR in [`docs/adr/`](docs/adr/), so the reasoning
behind the format is auditable regardless of who or what typed it.

We say this openly because it shaped how the project was built. If you would rather not
use AI-developed code, this is not the project for you.

## Acknowledgments

Hurray owes its shape to **[Apache Arrow](https://arrow.apache.org/)**, which
demonstrated that a language-agnostic, zero-copy columnar format with a shared in-memory
specification could become common ground for an entire ecosystem. Hurray applies that
idea to tensors: same instinct — one descriptor, buffers shared rather than copied,
implementations in every language — aimed at the layout diversity and quantization that
inference workloads require. See [prior art](docs/prior-art.md) for the wider survey.

## License

Dual-licensed under [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at your option.
