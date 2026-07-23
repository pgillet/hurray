# File Format — Hurray Format Specification

> **Status:** Draft

> This section uses RFC 2119 key words: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT,
> SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

## Scope

This section defines the **Hurray file format**: a random-access container for one or
more named tensors, intended for on-disk model storage, distribution, and mmap-based
zero-copy loading. It is the companion to the streaming IPC format defined in
`interchange.md`.

The two formats are **complementary, not competing**:

| | Streaming format | File format |
|-|-----------------|-------------|
| Transport | Socket, pipe, RDMA, IPC | Seekable file or mmap-able region |
| Tensor count | Unbounded, unknown upfront | Fixed, enumerable via index |
| Tensor names | None | Required, unique UTF-8 |
| Random access | No | Yes — seek to any tensor by name |
| Writer constraint | Single-pass, no seek required | Single-pass, sequential write + footer |
| Reader start | Immediately (streaming) | After file is complete (index in footer) |
| Prior art | Arrow IPC stream, gRPC | SafeTensors, GGUF, Arrow IPC file |

Both formats share the **same tensor descriptor encoding** defined in `metadata.md`.
The file format adds a container layer; it does not redefine how individual tensors
are described.

See `docs/adr/ADR-011-file-format-random-access-container.md` for the design decisions.

---

## File Layout

A Hurray file has the following structure, in order:

```
[ File header          ]   64 bytes, fixed
[ Padding              ]   0x00 bytes, to first_descriptor_offset
[ Tensor region        ]   repeated: descriptor → padding → data buffer(s) → padding
[ KV metadata section  ]   optional, located by trailer
[ Index section        ]   located by trailer
[ Trailer              ]   40 bytes, fixed, at file_size - 40
```

All multi-byte fields MUST be encoded in little-endian byte order.

---

## File Header

The file header occupies the first 64 bytes of the file.

| Offset | Field | Type | Description |
|--------|-------|------|-------------|
| 0 | `magic` | `uint8[8]` | MUST be `0x48 0x52 0x52 0x59 0x46 0x49 0x4C 0x45` (ASCII `HRRYFILE`). |
| 8 | `container_version_major` | `uint8` | Container format major version. Current: `0x01`. |
| 9 | `container_version_minor` | `uint8` | Container format minor version. Current: `0x00`. |
| 10 | `_reserved` | `uint8[2]` | MUST be `0x00`. |
| 12 | `file_flags` | `uint32` | Bitmask of file-level flags. See [§ File Flags](#file-flags). |
| 16 | `data_buffer_alignment` | `uint32` | Alignment (bytes) applied to all tensor data buffers within the file. MUST be a power of two and MUST be at least `4096`. MUST NOT exceed `2097152` (2 MiB). |
| 20 | `first_descriptor_offset` | `uint64` | Absolute byte offset of the first tensor descriptor. MUST be `>= 64`. Typically `64` (immediately after the header). |
| 28 | `tensor_count_hint` | `uint64` | Number of tensors in the file, if known at write time. Writers that do not know this upfront MUST set this field to `0xFFFFFFFFFFFFFFFF`. Readers MUST NOT rely on this field for correctness; use the index entry count instead. |
| 36 | `_reserved_header` | `uint8[28]` | MUST be `0x00`. Reserved for future use. |

Total: 64 bytes.

A reader MUST reject a file whose `magic` does not equal `HRRYFILE`. A reader MUST
reject a file whose `container_version_major` exceeds the highest major version the
reader supports.

### File Flags

| Bit | Name | Meaning |
|-----|------|---------|
| 0 | `HAS_KV_METADATA` | KV metadata section is present; `kv_offset` and `kv_length` in the trailer are non-zero. |
| 1 | `SORTED_INDEX` | Index entries are sorted by UTF-8 byte order of `name`, enabling binary search. A writer MUST NOT set this flag unless the index is actually sorted. |
| 2 | `HAS_INDEX_CRC32C` | The `index_crc32c` field in the trailer carries a valid CRC-32C of the index section bytes. When this flag is set, a reader MUST verify the CRC and MUST reject the file if it does not match. Writers SHOULD always set this flag and populate `index_crc32c`. When this flag is not set, the `index_crc32c` field MUST be `0x00000000` and readers MUST NOT perform checksum verification. |
| 3–31 | (reserved) | MUST be `0`. A reader MUST reject a file with reserved flag bits set. |

---

## Tensor Region

Following the file header (and any padding to `first_descriptor_offset`), tensors are
written sequentially. Each tensor occupies the following region:

```
[ Tensor descriptor    ]   as defined in metadata.md (begins with HRRY magic)
[ Padding              ]   0x00 bytes to align next data buffer to data_buffer_alignment
[ Data buffer 0        ]   tensor data, byte_size bytes
[ Padding              ]   0x00 bytes to align next item to data_buffer_alignment
[ Data buffer 1        ]   (if buffer_count > 1)
[ Padding              ]
  ...
[ Padding              ]   0x00 bytes to align next descriptor to 8 bytes
```

### Descriptor Placement

Each tensor descriptor MUST begin at a byte offset that is a multiple of `8`. The
first descriptor begins at `first_descriptor_offset` (which MUST be 8-byte aligned).
Subsequent descriptors begin at the next 8-byte-aligned offset after the previous
tensor's last data buffer (including padding).

### Data Buffer Placement

The first data buffer of a tensor MUST begin at a byte offset that is a multiple of
`data_buffer_alignment`. If the tensor descriptor ends at a byte offset that is not
a multiple of `data_buffer_alignment`, the writer MUST insert `0x00` padding bytes
to reach the next `data_buffer_alignment` boundary.

If a tensor has multiple data buffers (e.g., dense data + quantization parameter
buffers), each buffer MUST begin at a `data_buffer_alignment`-aligned offset. Padding
bytes between buffers MUST be `0x00`.

### Empty Tensors

An empty tensor (any dimension size `0`) has data buffer byte size `0`. Its data
buffer region occupies `0` bytes. The writer MUST still insert sufficient padding
after the descriptor to satisfy alignment requirements for the *next* descriptor.

---

## KV Metadata Section

The KV metadata section, when present (`HAS_KV_METADATA` flag set), is located by
the trailer's `kv_offset` and `kv_length` fields. It MUST be aligned to an 8-byte
boundary.

### KV Section Encoding

| Field | Type | Description |
|-------|------|-------------|
| `kv_count` | `uint32` | Number of key-value entries. |
| `kv_entries[kv_count]` | (variable) | Sequentially encoded KV entries. |

Each KV entry is encoded as:

| Field | Type | Description |
|-------|------|-------------|
| `key_length` | `uint16` | Length of the key in bytes. MUST be at least `1`. |
| `key` | `uint8[key_length]` | UTF-8 key bytes, no null terminator. |
| `value_tag` | `uint8` | Type tag for the value (see [§ KV Value Types](#kv-value-types)). |
| `value` | (variable) | Value payload, encoded per value_tag. |

Keys MUST be unique within the KV section (case-sensitive, byte-exact comparison).
A reader MUST reject a file with duplicate KV keys. Keys MUST be valid UTF-8.

### KV Value Types

| Tag | Type | Payload encoding |
|-----|------|-----------------|
| `0x01` | `utf8 string` | `uint32` byte length, then UTF-8 bytes. |
| `0x02` | `int64` | 8 bytes, little-endian. |
| `0x03` | `uint64` | 8 bytes, little-endian. |
| `0x04` | `float64` | 8 bytes, little-endian IEEE 754 binary64. |
| `0x05` | `bool` | 1 byte. `0x00` = false, `0x01` = true. All other values MUST be rejected. |
| `0x06` | `byte sequence` | `uint32` byte length, then opaque bytes. |
| `0x07` | `array` | `uint8` element type tag (MUST be `0x01`–`0x06`), then `uint32` element count, then element payloads concatenated. |
| `0x08`–`0xEF` | (reserved) | MUST NOT be used. A reader MUST reject a file containing a reserved value tag. |
| `0xF0`–`0xFE` | (extension) | Implementation-private. MUST NOT appear in files exchanged between independent implementations unless agreed out of band. |
| `0xFF` | (invalid) | MUST NOT be used. |

> **Note (non-normative):** The KV section is intended for model-level metadata:
> architecture name, quantization configuration identifier, tokenizer vocabulary
> size, etc. It is not a substitute for per-tensor metadata; per-tensor fields
> belong in the tensor descriptor.

---

## Index Section

The index section is located by the trailer's `index_offset` and `index_length`
fields. It MUST be aligned to an 8-byte boundary.

### Index Encoding

| Field | Type | Description |
|-------|------|-------------|
| `index_entry_count` | `uint64` | Number of index entries. MUST equal the number of tensors in the file. |
| `index_entries[index_entry_count]` | (variable) | Sequentially encoded index entries. |

Each index entry is encoded as:

| Field | Type | Description |
|-------|------|-------------|
| `name_length` | `uint16` | Length of the tensor name in bytes. MUST be at least `1`. |
| `name` | `uint8[name_length]` | UTF-8 tensor name bytes, no null terminator. |
| `descriptor_offset` | `uint64` | Absolute byte offset of the tensor descriptor from the start of the file. |
| `descriptor_length` | `uint32` | Length of the tensor descriptor in bytes. MUST equal the descriptor's own internal `descriptor_length` field; a reader MUST reject a file where they disagree. |
| `data_offset` | `uint64` | Absolute byte offset of the first data buffer from the start of the file. |
| `data_length` | `uint64` | Total byte length of all data buffers for this tensor (sum of all buffer `byte_size` values, plus inter-buffer padding within the tensor's data region). |
| `flags` | `uint32` | Reserved for future use. MUST be `0x00000000`. |

Index entry names MUST be unique within the index (case-sensitive, byte-exact).
Index entries are written in tensor write order by default. If the `SORTED_INDEX`
file flag is set, entries MUST be sorted by UTF-8 byte order of `name` (strict
byte comparison, no Unicode normalisation).

> **Note (non-normative):** The `descriptor_length` field in the index duplicates the
> descriptor's own internal length to enable fast enumeration: a reader that wants to
> list all tensor names, shapes, and dtypes can parse the index without touching any
> descriptor byte, at O(index size) cost instead of O(file size).

> **Note (non-normative):** `data_length` covers the full data region including
> inter-buffer padding. A reader that wants to mmap the entire data region of a tensor
> can use `data_offset` and `data_length` without parsing the buffer table.

---

## Composite Tensors

A composite tensor (head + members; see `layouts/composite.md`) is written as a **head**
descriptor followed by its `member_count = N` members, all as **consecutive tensors** in
the tensor region, in that order. The head is an ordinary tensor with `layout_tag = 0x0B`
and `buffer_count = 0`, so its data region occupies `0` bytes (its `data_length` in the
index is `0`); each member is written as an ordinary tensor with its own descriptor and
data buffers. Every tensor — head and members alike — gets its own index entry.

**No `data_buffer_alignment` padding after a head.** § Data Buffer Placement's alignment
rule applies to a tensor's data buffer(s); a `buffer_count = 0` head has none — unlike an
*empty* tensor (§ Empty Tensors), which still has one buffer of `byte_size = 0` and is
therefore still aligned as a buffer. A writer MUST NOT insert `data_buffer_alignment`
padding after a composite head's descriptor. The next descriptor (the head's first
member) follows at the ordinary `8`-byte descriptor-aligned offset (§ Descriptor
Placement).

Membership is recovered from the head's `member_count` plus **descriptor-offset order**:
the members are the next `N` tensors, ordered by ascending `descriptor_offset`, following
the head. This recovery rule uses `descriptor_offset` order (write order in the tensor
region), which is preserved **regardless of whether the `SORTED_INDEX` file flag is set**.
Setting `SORTED_INDEX` reorders only the *index entries* (by tensor name, for binary
search); it does not move tensors within the tensor region, so `descriptor_offset` order
still recovers head→member adjacency.

The composite **closes** at the Nth member. A reader MUST run the close-time validation for
the composition rule (`layouts/composite.md` § Validation) once all N members are located.
A file in which fewer than N members follow a head (a torn composite) is invalid: a strict
reader MUST reject it; a permissive reader MAY treat the arrived members as independent
shard tensors but MUST NOT present the composite as complete.

Nested composites are permitted: a member MAY itself be a head, whose own members are the
immediately following tensors (pre-order), subject to the depth limit in
`layouts/composite.md` § Binding.

> **Note (non-normative):** Because membership is positional, composite members are not
> required to carry distinguishing names beyond the file format's per-tensor uniqueness
> rule. A reader that only wants the merged logical view resolves it from the head plus its
> N following members; a reader indexing by name still sees each member as an addressable
> tensor.

---

## Trailer

The trailer occupies the last 40 bytes of the file (bytes `file_size - 40` through
`file_size - 1`).

| Offset from trailer start | Field | Type | Description |
|---------------------------|-------|------|-------------|
| 0 | `index_offset` | `uint64` | Absolute byte offset of the index section from the start of the file. |
| 8 | `index_length` | `uint64` | Byte length of the index section. |
| 16 | `kv_offset` | `uint64` | Absolute byte offset of the KV metadata section. `0` if no KV section. |
| 24 | `kv_length` | `uint32` | Byte length of the KV metadata section. `0` if no KV section. |
| 28 | `index_crc32c` | `uint32` | CRC-32C of the index section bytes (the `index_length` bytes starting at `index_offset`). Valid only when the `HAS_INDEX_CRC32C` file flag (bit 2) is set. MUST be `0x00000000` when the flag is not set. |
| 32 | `_reserved` | `uint8[4]` | MUST be `0x00`. Reserved for future trailer fields. |
| 36 | `trailer_magic` | `uint8[4]` | MUST be `0x48 0x52 0x52 0x59` (ASCII `HRRY`). |

Total: 40 bytes.

> **Note (non-normative):** `kv_length` is `uint32` (4-byte), capping the KV
> metadata section at approximately 4 GiB. This is intentional: KV metadata
> holds model-level annotations (architecture, tokenizer config, quantization
> settings) and is not expected to exceed this limit in practice. Tensor
> data, which can be arbitrarily large, is addressed by `uint64` offsets in
> the index.

A reader locates the trailer by seeking to `file_size - 40`. It MUST verify
`trailer_magic` before trusting any other trailer field.

A reader MUST check the `HAS_INDEX_CRC32C` file flag before interpreting `index_crc32c`. If the flag is set, the reader MUST verify the CRC-32C against the index section bytes and MUST reject the file on mismatch. If the flag is not set, `index_crc32c` MUST be `0x00000000`; a reader that finds a non-zero value with the flag unset MUST reject the file.

A reader MUST reject a file whose `index_offset + index_length` extends into the
trailer (i.e., `index_offset + index_length > file_size - 40`).

---

## Reader Protocol

### Random-Access (Seek-Capable) Reader

1. Read bytes 0–7. MUST equal `HRRYFILE`.
2. Read bytes 8–63 (remainder of file header). Check `container_version_major`.
3. Seek to `file_size - 40`. Read trailer. Verify `trailer_magic`. If `HAS_INDEX_CRC32C` is set in `file_flags`, record `index_crc32c` for verification after step 4.
4. Seek to `index_offset`. Read `index_length` bytes. Parse index.
5. Optionally seek to `kv_offset`. Read KV metadata.
6. For each requested tensor: seek to `descriptor_offset`, parse descriptor;
   seek to `data_offset`, mmap or read `data_length` bytes.

### Sequential Reader (No Seek)

A reader that cannot seek MAY consume the file sequentially. Such a reader MUST
track the running byte offset from the start of the file at every step and use
offset arithmetic — never inspection of data buffer content — to determine
section boundaries.

1. Read and verify the 64-byte file header. Set the running offset to `64`.
2. Read padding bytes until the running offset equals `first_descriptor_offset`.
3. For each tensor in the tensor region:
   a. Read the tensor descriptor's first 10 bytes to obtain `descriptor_length`
      (bytes 6–9), then read the remaining `descriptor_length - 10` bytes of the
      descriptor. Advance the running offset by `descriptor_length`.
   b. Read padding bytes until the running offset is a multiple of
      `data_buffer_alignment` (the value declared in the file header).
   c. For each entry in the descriptor's buffer table, read `byte_size` bytes of
      data and then read padding bytes until the running offset is a multiple of
      `data_buffer_alignment`. Advance the running offset by the buffer size and
      the padding.
   d. Read padding bytes until the running offset is a multiple of `8`
      (alignment for the next descriptor).
   e. The reader has now reached the start of either the next tensor descriptor,
      the KV metadata section, or the index section.
4. To determine whether step (3) should be repeated, the reader cannot inspect
   data content (a tensor's data buffer may legitimately contain the byte
   sequence `0x48 0x52 0x52 0x59`, and the KV / index sections do not begin
   with `HRRY`). The end of the tensor region MUST be detected by one of:
   - **Trailer probe (preferred):** a sequential reader that has buffered the
     entire file in memory after the fact MAY locate the trailer at
     `file_size - 40` and use `index_offset` (and `kv_offset` if present) to
     determine the tensor region's end offset.
   - **Tensor count hint:** if the file header's `tensor_count_hint` is not
     `0xFFFFFFFFFFFFFFFF`, the reader MAY iterate exactly that many tensors.
     The reader MUST then verify that the running offset corresponds to a
     declared section boundary.
   - **EOF:** the reader MAY consume bytes until end-of-file and treat the
     terminal 40 bytes as the trailer; it MUST NOT treat that 40 bytes as
     a tensor region.

A sequential reader cannot perform random access and cannot look up tensors by
name without reading the entire file in order. This mode is OPTIONAL to
implement and is significantly more constrained than the random-access reader
defined above; conforming implementations SHOULD prefer the random-access path.

---

## Writer Protocol

A conforming streaming writer produces a valid Hurray file in a single forward pass:

1. Write the file header. Set `tensor_count_hint` to `0xFFFFFFFFFFFFFFFF` if the
   count is not known.
2. For each tensor, in any order:
   a. Record the current byte offset as `descriptor_offset`.
   b. Write the tensor descriptor.
   c. Insert `0x00` padding to the next `data_buffer_alignment` boundary.
   d. Record the current byte offset as `data_offset`.
   e. Write each data buffer; insert padding between buffers and after the last
      one to maintain `data_buffer_alignment`.
   f. Insert `0x00` padding to the next 8-byte boundary.
   g. Record `descriptor_length` and `data_length` for the index.
3. Write the KV metadata section (if any). Record its offset and length.
4. Write the index section. Record its offset and length.
5. Compute CRC-32C over the index section bytes. Set `HAS_INDEX_CRC32C` in `file_flags`. Write the 40-byte trailer with `index_crc32c` populated.

The writer MUST NOT seek backward at any point. All offset information is tracked
in memory as a list of `(name, descriptor_offset, descriptor_length, data_offset,
data_length)` tuples, which is the only in-memory state required beyond the tensors
themselves.

---

## Alignment and Padding Summary

| Item | Required alignment | Padding fill |
|------|--------------------|--------------|
| File header | n/a (byte 0) | — |
| First tensor descriptor | `first_descriptor_offset` (≥ 8-byte aligned) | `0x00` |
| Subsequent tensor descriptors | 8-byte | `0x00` |
| Data buffers (all) | `data_buffer_alignment` (≥ 4096) | `0x00` |
| KV metadata section | 8-byte | `0x00` |
| Index section | 8-byte | `0x00` |
| Trailer | last 40 bytes of file | — |

---

## Relationship to Other Sections

- **`metadata.md`** defines the tensor descriptor encoding used verbatim within
  tensor regions. The `descriptor_length` field in the index caches the value of
  the descriptor's own internal length field.
- **`interchange.md`** defines the streaming IPC format. The two formats are
  complementary: file magic `HRRYFILE` distinguishes the file format from a stream
  that begins with an `HRRY` descriptor.
- **`buffer-protocol.md`** defines alignment rules. The file format's
  `data_buffer_alignment` (minimum 4096 bytes) enables zero-copy mmap of tensor
  data. The same device-tag and ownership rules apply to mmapped buffers.
- **`versioning.md`** defines descriptor versioning. Container versioning
  (`container_version_major` / `container_version_minor`) is independent of
  descriptor versioning.

---

## Open Questions

> **[OQ-1]:** ~~Should single-tensor files be required to use a specific tensor name?~~ **Resolved:** No required name. Tensor names are always meaningful and left to the producer. For the array database use case (Core Property 10), mandating a generic name like `"data"` would erase the semantic identity of the tensor. Readers that need a single-tensor API SHOULD use the sole entry in the footer index without reference to its name.

> **[OQ-2]:** ~~Should a future `SORTED_INDEX_NFC` flag be defined for NFC-normalised or case-folded comparisons?~~ **Resolved:** Strict UTF-8 byte order is sufficient. Tensor names in practice are ASCII identifiers; NFC normalisation adds implementation complexity with no practical benefit for the target use case. A new flag can be defined if a future use case genuinely requires Unicode-aware sorting.

> **[OQ-3]:** ~~Should the trailer carry a CRC-32C of the footer index for integrity verification?~~ **Resolved:** Added. The `index_crc32c` field (uint32, offset 28 in the trailer) carries the CRC-32C of the index section bytes. Writers SHOULD always populate it; a value of `0x00000000` signals "not computed" and readers MAY accept without verification. The trailer grows from 32 to 40 bytes accordingly. Related `metadata.md` OQ-1 (descriptor checksum) was resolved as "no" — file-at-rest integrity is a stronger argument than in-flight descriptor integrity.
