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

One file has moved since: `nested_composite.hrry` was re-emitted on 2026-10-25 for issue
#243, and its 0.1.0 bytes live on beside it as `nested_composite_padded_head.hrry` — see
[§ The one exception](#the-one-exception-nested_compositehrry).

| File | Contents |
|------|----------|
| `row_major.hrry` | One `[8, 8]` float32 tensor, default options (4096-byte data alignment, unsorted index), no KV |
| `multi_buffer.hrry` | One COO tensor whose two buffers declare 128- and 4096-byte alignment, so the data region carries inter-buffer padding |
| `kv_sorted.hrry` | Two tensors written out of byte order with `sorted_index`, 8192-byte data alignment, and a KV section covering every wire tag including an array |
| `nested_composite.hrry` | A group of two whose first member is a partition of two (nesting depth 2), five index entries |
| `nested_composite_padded_head.hrry` | The same five tensors as 0.1.0 wrote them, with the forbidden head padding still in place — 8 KiB larger, read-only (see below) |

Between them they cover the 64-byte header, the 40-byte trailer, the index section with and
without sorting, CRC-32C verification, the KV section's seven tags, descriptor-to-data
padding, inter-buffer padding, and composite recovery by file-offset adjacency.

## Do not regenerate

Regenerating these files from current code makes the compatibility test vacuous — it would
then assert only that the code agrees with itself. A test here failing means either a real
container-format regression or a deliberate format change; in the latter case the spec and
`docs/spec/metadata.md` change first, and the new bytes get their own commit explaining
why, with the container version bumped.

### The one exception: `nested_composite.hrry`

Regenerated on 2026-10-25 for issue #243. The 0.1.0 writer padded a composite head's
descriptor to `data_buffer_alignment` before discovering the head declares no buffer —
4 KiB per head, which `file-format.md § Composite Tensors` forbids outright. Fixing the
writer moved the bytes, so this one file was re-emitted (20 742 → 12 550 bytes).

The container version is **not** bumped: nothing about the format changed. The old writer
was emitting padding the spec already forbade, and a reader takes every offset from the
footer index, so neither the old bytes nor the new ones require anything of it.

That last claim is what `nested_composite_padded_head.hrry` is for. It is the original
0.1.0 output, byte for byte, kept so the corpus can still show that a composite file
written before the fix reads correctly today — membership, payloads and all. It is never
written, only read, which is why regenerating it would be meaningless rather than merely
unwise.
