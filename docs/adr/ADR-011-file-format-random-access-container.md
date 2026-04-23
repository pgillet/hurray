# ADR-011: File Format — Random-Access Container for Hurray Tensors

## Status

Accepted

## Context

Hurray defines a streaming IPC format (descriptor + data, back-to-back, no seek)
suitable for sockets, pipes, and RDMA. For on-disk model distribution, cold-start
inference, and multi-tensor archives, a random-access container is needed: the
ability to open a file, enumerate named tensors, and mmap any tensor's data without
reading the rest of the file.

Prior art (SafeTensors, GGUF, Arrow IPC file) converges on a footer-based index with
optional typed key-value metadata. SafeTensors uses a JSON header; GGUF uses a custom
binary header. Arrow IPC file uses a FlatBuffers footer.

Both formats share the same tensor descriptor encoding (`metadata.md`); the file
format is purely a container layer on top. This is the key design constraint: the
file format MUST NOT redefine how individual tensors are described.

## Decision

Define a Hurray **file format** as a named-tensor container wrapping the existing
tensor descriptor encoding, with the following design:

1. **Magic:** 8-byte `HRRYFILE` (`0x48 0x52 0x52 0x59 0x46 0x49 0x4C 0x45`).
   Distinguishes the file format from the streaming format (which begins with `HRRY`
   followed by a tensor descriptor). An `HRRYFILE` magic byte sequence is never a
   valid streaming tensor descriptor.

2. **Container version:** Independent `container_version_major` / `container_version_minor`
   (`uint8` each) in the file header, separate from the tensor descriptor's version.
   Current: `0x01` / `0x00`.

3. **Tensor names:** UTF-8, length-prefixed with `uint16` (max 65 535 bytes), no
   null terminator. Names MUST be unique within a file (case-sensitive, byte-exact).
   Names MUST NOT be empty. No hierarchical semantics are assigned to any character,
   including `/`; names are opaque identifiers at the spec level.

4. **Index position:** Footer only. The index is written after all tensor data; a
   fixed-size 32-byte trailer at the end of the file locates the index. This is the
   only design compatible with single-pass streaming writes.

5. **Trailer:** Fixed 32 bytes at `file_size - 32`. Contains: `index_offset` (uint64),
   `index_length` (uint64), `kv_offset` (uint64), `kv_length` (uint32), and
   `trailer_magic` (`HRRY_END`, 4 bytes ASCII).

6. **Alignment:** Tensor data buffers MUST be aligned to a page boundary within the
   file. The default page size is 4096 bytes; the file header MAY declare a larger
   alignment (up to 2 MiB) for huge-page environments. Tensor descriptors MUST be
   aligned to 8 bytes. All padding bytes MUST be `0x00`.

7. **KV metadata:** Optional file-level key-value metadata section with typed values
   (UTF-8 string, int64, uint64, float64, bool, byte sequence, typed array). This is
   required for ecosystem adoption: model architecture, tokenizer parameters, and
   quantization configuration cannot be expressed as tensor descriptors.

8. **Streaming write:** MUST be supported. A writer tracks byte offsets as it writes
   each tensor inline, then appends KV metadata (if any), then the index, then the
   32-byte trailer. No backward seeks are required.

9. **Streaming read (no seek):** OPTIONAL. Tensor descriptors and data appear inline
   in file order; a non-seeking reader MAY consume them sequentially without the
   footer. This is a courtesy for pipe-based pipelines, not a required reader mode.

10. **Shared descriptor encoding:** Both formats use the tensor descriptor encoding
    from `metadata.md` verbatim. The file format index caches `descriptor_length` per
    entry (for O(index) fast enumeration without parsing descriptors), but MUST NOT
    cache shape, dtype, layout, or any other descriptor field.

## Alternatives Considered

**Header index (offsets written before data).** Rejected: a streaming writer cannot
know tensor offsets before writing them. Requires full buffering or two-pass writes.

**Dual index (header + footer).** Rejected: doubles write cost; creates a reconciliation
problem on disagreement.

**FlatBuffers for the index.** Rejected: adds an external dependency and a schema
parser. The index schema is static and narrow; a hand-rolled binary format is simpler
and fully specifiable in RFC 2119 terms.

**Reuse Arrow IPC file format.** Rejected: Arrow is columnar and cannot express Hurray's
layout diversity (tiled, Morton, sparse) or quantization descriptors. Wrapping Hurray
descriptors in Arrow's framing would add complexity without benefit.

**No KV metadata in v1.** Rejected: GGUF's widespread adoption is driven by typed KV
metadata. Without it, producers must store model metadata out-of-band, creating a
fragmented ecosystem.

**Compress tensor data within the file.** Rejected: compression and zero-copy mmap are
mutually exclusive. Hurray is zero-copy-first.

**Content-addressable/chunked storage (Zarr-style).** Rejected: Zarr's strength is
cloud storage with per-chunk compression. Incompatible with zero-copy mmap.

## Consequences

- Hurray now covers the full lifecycle: runtime interchange (streaming format) and
  model storage / distribution (file format).
- The file format enables SafeTensors/GGUF replacement with zero-copy mmap and rich
  layout/quantization metadata.
- README.md Core Property 2 ("no end-of-file index") must be scoped to the streaming
  format. The file format explicitly uses a footer index.
- `hurray-io` must implement both a streaming writer/reader and a file writer/reader.
- `hurray-inspect` should print the format kind on open (`stream` vs `file`).
- The C FFI API should expose separate entry points (`hurray_file_open` vs
  `hurray_stream_open`) to prevent format confusion.
- A new spec file `docs/spec/file-format.md` is the normative reference for the
  container layer.
