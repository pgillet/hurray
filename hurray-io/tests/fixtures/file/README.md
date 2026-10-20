# Frozen file container corpus

These bytes are the Hurray file container as the pre-sans-io `FileWriter` emitted it. They
exist so ADR-041's file driver refactor can be checked for byte-for-byte compatibility
against something that is *not* the refactored code.

## Provenance

Generated on 2026-10-20 from commit **4647b64** (`feat(ffi): the C stream surface drives
the sans-io core (ADR-041 step 3)`), by driving the then-current `FileWriter` — the
recursive, `write_all`-based implementation that predates the file core — over the
descriptors built in `../../file_wire_compat.rs`. The generator was a throwaway: the
descriptors, payloads and writer options in `file_wire_compat.rs` reproduce these files
exactly, so the test is the specification of what each one contains.

| File | Contents |
|------|----------|
| `row_major.hrry` | One `[8, 8]` float32 tensor, default options (4096-byte data alignment, unsorted index), no KV |
| `multi_buffer.hrry` | One COO tensor whose two buffers declare 128- and 4096-byte alignment, so the data region carries inter-buffer padding |
| `kv_sorted.hrry` | Two tensors written out of byte order with `sorted_index`, 8192-byte data alignment, and a KV section covering every wire tag including an array |
| `nested_composite.hrry` | A group of two whose first member is a partition of two (nesting depth 2), five index entries |

Between them they cover the 64-byte header, the 40-byte trailer, the index section with and
without sorting, CRC-32C verification, the KV section's seven tags, descriptor-to-data
padding, inter-buffer padding, and composite recovery by file-offset adjacency.

## Do not regenerate

Regenerating these files from current code makes the compatibility test vacuous — it would
then assert only that the code agrees with itself. A test here failing means either a real
container-format regression or a deliberate format change; in the latter case the spec and
`docs/spec/metadata.md` change first, and the new bytes get their own commit explaining
why, with the container version bumped.
