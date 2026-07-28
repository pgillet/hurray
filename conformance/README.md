# Hurray conformance corpus

A language-neutral set of **golden test vectors** — byte-exact encoded artifacts plus a JSON
manifest of their expected decoded properties — used to validate Hurray implementations and
bindings against a fixed source of truth.

> The reference bindings are **validated against a shared golden corpus**; they are not
> independent re-implementations of the decoder (`hurray-python` wraps `hurray-core`/`-io`).
> An independent, non-Rust decoder remains the gold standard for the language-agnostic
> guarantee and is future work.

## Layout

```
conformance/
├── vectors/
│   ├── descriptors/*.bin   encoded TensorDescriptors (one per vector)
│   ├── files/*.hrry        HRRYFILE container vectors
│   └── manifest.json       expected decoded properties for every vector
├── src/                    corpus definitions + the generator
└── tests/verify.rs         Rust conformance check (decode + round-trip vs manifest)
```

## Regenerating the corpus

The corpus is deterministic and committed. Regenerate after an intentional format change:

```sh
cargo run -p hurray-conformance --bin generate-vectors
```

Then review the diff under `conformance/vectors/` and commit it.

## What it validates

| Level | Exercised by |
|-------|--------------|
| **Writer** — encode descriptors and a file | the generator (`generate-vectors`) |
| **Reader** — decode descriptors, open files, read tensors + KV | `tests/verify.rs` (Rust), and the Python binding suite |
| **Round-trip** — bytes → decode → observed properties == manifest | `tests/verify.rs` |

Streaming (network-transport) conformance is covered by the `hurray-io` stream round-trip
tests.

## Coverage (current)

Descriptor vectors: dense (row-major, strided, Morton) across `float32` / `float16` /
`int8` / `bool`; an empty tensor; a shard section; a sparse **CSR** matrix; a **composite**
partition head; and a **per-block-affine int4** quantized tensor. File vector: two dense
tensors with string + int64 KV metadata.

> Expansion candidates: tiled and block-paged descriptor vectors, and per-tensor /
> per-channel / NF4 / MXFP quantization vectors.

## Consumers

- **Rust:** `cargo test -p hurray-conformance`.
- **Python:** the `hurray-python` test suite loads `files/*.hrry` and asserts against the
  same `manifest.json`.
