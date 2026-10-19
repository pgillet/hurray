# Frozen stream wire corpus

These bytes are the Hurray streaming wire format as the 0.1.0 `StreamWriter` emitted it.
They exist so ADR-041's driver refactor can be checked for byte-for-byte compatibility
against something that is *not* the refactored code.

## Provenance

Generated on 2026-10-19 from commit **91a99ae** (`feat(io): a sans-io core for the stream
wire format (ADR-041 step 1)`), by driving the then-current `StreamWriter` — the recursive,
`write_all`-based implementation that predates the sans-io core — over the descriptors
built in `../wire_compat.rs`. The generator was a throwaway: the descriptors and payloads
in `wire_compat.rs` reproduce these streams exactly, so the test is the specification of
what each file contains.

| File | Contents |
|------|----------|
| `row_major.bin` | One `[8, 8]` float32 row-major tensor, single 256-byte buffer |
| `coo.bin` | One COO tensor, two buffers declaring 128- and 4096-byte alignment |
| `quantized.bin` | One int4 tensor carrying per-block-affine quantization and shard sections |
| `nested_composite.bin` | A group of two whose first member is a partition of two (nesting depth 2) |
| `cross_machine.bin` | Two tensors written back to back through `StreamWriter::cross_machine` |

## Do not regenerate

Regenerating these files from current code makes the compatibility test vacuous — it would
then assert only that the code agrees with itself. A test here failing means either a real
wire-format regression or a deliberate format change; in the latter case the spec and
`docs/spec/interchange.md` change first, and the new bytes get their own commit explaining
why, with the format version bumped.
