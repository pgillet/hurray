# C FFI Layer Requirements — Hurray Implementation Requirements

## Overview

The `hurray-ffi` crate exposes a stable C ABI that allows non-Rust runtimes to
consume and produce Hurray tensors without depending on Rust tooling. It is the
foundation for all non-Python language bindings.

## ABI Stability

- All public symbols MUST use the `#[no_mangle]` attribute and `extern "C"` linkage.
- The ABI MUST be declared stable across patch versions and SHOULD be stable across
  minor versions. Breaking ABI changes require a major version bump.
- All struct layouts exposed across the FFI boundary MUST be `#[repr(C)]`.
- Enums exposed across the FFI boundary MUST be `#[repr(u8)]` or `#[repr(i32)]` as
  appropriate, never `#[repr(Rust)]`.

## C ABI Version

The C ABI carries a single `uint32` version identifier exposed via a constant
`HURRAY_C_ABI_VERSION` and a runtime accessor `hurray_c_abi_version()`. The
current version is `8`:

| Version | Changes |
|---------|---------|
| `1` | Initial C ABI: opaque handles, buffer release callbacks, panic-safe error returns. |
| `2` | Per-mode buffer handoff sync payloads (`SYNC_PRODUCER_SYNCED`, `SYNC_EVENT`, `SYNC_CONSUMER_STREAM`) and the event-release callback. See [Buffer Handoff Synchronisation](#buffer-handoff-synchronisation). |
| `3` | `HurrayBufferList` for multi-buffer tensors, and the native protocol capsule now wraps a list rather than a single `HurrayBuffer` (ADR-030). See [Buffer Lists](#buffer-lists). |
| `4` | `HurrayTensorContext`, so a consumer in any language can read a capsule's descriptor and ABI version (ADR-034). See [Tensor Context](#tensor-context). |
| `5` | Layout-payload field reads, so a consumer can interpret every Tier 1 layout it can decode (ADR-039). See [Descriptor Field Reads](#descriptor-field-reads). |
| `6` | The field identifier space grew to cover all five optional descriptor sections, which renamed the three ADR-039 accessors and added two (ADR-040). See [Descriptor Field Reads](#descriptor-field-reads). |
| `7` | `HurrayReader` and `HurrayWriter` became real handles over a transport vtable, and `HurrayTensor` pairs a descriptor with its buffers (ADR-041). See [Streaming Over a Transport Vtable](#streaming-over-a-transport-vtable). |
| `8` | The transport vtable grew `read_at` and `size`, and `HurrayFileReader` / `HurrayFileWriter` read and write the `HRRYFILE` container (ADR-041 § 8). See [File Containers Over a Transport Vtable](#file-containers-over-a-transport-vtable). |

A consumer of the C ABI MUST query `hurray_c_abi_version()` before invoking any
function whose contract changed in a later version. A consumer compiled against
version `1` of the ABI that links against a runtime providing version `2` will
receive buffer handles whose `sync_mode` is `SYNC_PRODUCER_SYNCED` (`0x00`) by
default, which is the safe fallback: a version-`1` consumer that does not
inspect `sync_mode` will still observe the strongest synchronisation guarantee
and will not race against the producer's device writes.

## Opaque Handles

All Hurray objects crossing the FFI boundary MUST be represented as **opaque pointer
handles**. Callers MUST NOT dereference or inspect the pointed-to memory directly.

| Handle type | Represents |
|---|---|
| `HurrayDescriptor*` | A parsed tensor descriptor |
| `HurrayBuffer*` | A buffer handle (data + metadata) |
| `HurrayBufferList*` | An ordered, owning collection of buffer handles |
| `HurrayTensorContext*` | A capsule's descriptor bytes, ABI version, and owner reference |
| `HurrayTensor*` | A tensor: a descriptor and its buffers, owned together |
| `HurrayReader*` | A blocking stream reader over a caller-supplied transport |
| `HurrayWriter*` | A blocking stream writer over a caller-supplied transport |

Each handle is obtained from a `hurray_*_create` function and MUST be released by
the corresponding `hurray_*_destroy` function. Double-free and use-after-free are
undefined behaviour on the caller side; the implementation MUST detect them in debug
builds (e.g., via a poisoned sentinel).

## Buffer Lists

A tensor whose descriptor references more than one buffer — per-channel / NF4 /
MXFP quantization, sparse layouts, block-paged, composite — needs all of its
buffers to travel together. `HurrayBufferList` is that carrier (ADR-030).

| Function | Contract |
|---|---|
| `hurray_buffer_list_new(capacity, out_list)` | Creates an empty list. `capacity` is a hint. |
| `hurray_buffer_list_push(list, buffer)` | Appends `buffer`, **transferring ownership** to the list on success. On failure ownership stays with the caller. |
| `hurray_buffer_list_len(list, out_len)` | Number of buffers in the list. |
| `hurray_buffer_list_get(list, index, out_buffer)` | Borrows the handle at `index`. Returns `HURRAY_ERR_INDEX_OUT_OF_BOUNDS` if `index >= len`. |
| `hurray_buffer_list_destroy(list)` | Destroys the list and every handle it owns, then writes null through `list`. |

The following rules are normative.

- A list **owns** every handle pushed into it. A handle obtained from
  `hurray_buffer_list_get` is **borrowed**: the caller MUST NOT call
  `hurray_buffer_destroy` on it. It remains valid until the list is destroyed.
- Buffers MUST be pushed in descriptor buffer-table order: element `i` of the
  list is buffer index `i` of the descriptor.
- `hurray_buffer_list_destroy` takes a **pointer to the caller's handle
  variable** and MUST write null through it, so the caller's variable is
  observably dead and a repeated destroy is a no-op. This is the sound half of
  the "release marks the structure released" discipline: the list allocation is
  freed, so only memory the caller owns can carry the marker.
- Destroying a list MUST destroy every owned handle exactly once, and MUST null
  each owned slot as it goes, so that a release callback which panics or
  re-enters cannot cause a double free.

## Tensor Context

A native-protocol capsule carries two things: a `HurrayBufferList` as its pointer,
and a `HurrayTensorContext` as its context (ADR-034). The list holds the bytes; the
context holds the **encoded tensor descriptor** that says what those bytes are, plus
the ABI version of the build that produced them.

Before ADR-034 the context was a structure private to `hurray-python`, so the buffers
crossed the language boundary and the descriptor did not. A consumer in any other
language received element bytes with no element type, shape, layout, or quantization.

```c
HurrayStatus hurray_tensor_context_new(uint32_t abi_version,
                                       const uint8_t *descriptor_bytes,
                                       uint64_t descriptor_len,
                                       void *owner,
                                       HurrayOwnerReleaseFn owner_release,
                                       struct HurrayTensorContext **out_ctx);

HurrayStatus hurray_tensor_context_abi_version(const struct HurrayTensorContext *ctx,
                                               uint32_t *out);
HurrayStatus hurray_tensor_context_descriptor(const struct HurrayTensorContext *ctx,
                                              const uint8_t **out_bytes,
                                              uint64_t *out_len);
HurrayStatus hurray_tensor_context_destroy(struct HurrayTensorContext **ctx);
```

- A consumer MUST call `hurray_tensor_context_abi_version` **first** and compare the
  result against its own `HURRAY_C_ABI_VERSION` before calling any other accessor.
  That ordering is what allows later ABI versions to add accessors without breaking
  older consumers: a consumer that checked the version knows which ones exist.
- `hurray_tensor_context_descriptor` returns a **borrow** owned by the context, valid
  until the context is destroyed. A caller that needs it longer MUST copy it. An empty
  descriptor reports a null pointer and a zero length.
- The context owns a copy of the descriptor bytes; a borrow would tie its validity to
  a buffer the producer may drop.
- `owner` and `owner_release` are opaque and never interpreted. They exist so a
  producer can keep whatever owns the tensor's memory alive for the capsule's
  lifetime — `hurray-python` parks a Python object reference there, which is how a
  Python type is kept out of the C ABI entirely. `owner_release` is invoked exactly
  once, during `hurray_tensor_context_destroy`.
- Destroying a null handle is a no-op returning `HURRAY_OK`, so cleanup paths may call
  it unconditionally. Destroy nulls the caller's pointer.

## Descriptor Field Reads

Decoding a descriptor yields its fixed header — rank, element type tag, layout tag,
byte offset, buffer table. Everything with internal structure behind that header is
read separately: the layout payload, and the five optional sections (quantization,
shard, statistics, extension type, composite member).

An implementation MUST expose every layout-specific field of the Tier 1 layouts
(`0x01`–`0x09`), because Compliance Level 1 requires a reader to correctly interpret
those fields and a consumer that cannot read them cannot address an element. The
optional sections are not required at any conformance level, but an implementation that
decodes one SHOULD expose its fields: a section a caller cannot read is a section the
producer wrote for nothing.

Fields are named by a `HurrayDescriptorField` identifier and read through five typed
accessors (ADR-039, ADR-040):

| Accessor | Reads |
|---|---|
| `hurray_descriptor_field_u64` | every unsigned or boolean scalar |
| `hurray_descriptor_field_i64` | signed scalars: per-tensor affine `zero_point` |
| `hurray_descriptor_field_f64` | real scalars: per-tensor affine `scale`, the `float64` statistics |
| `hurray_descriptor_field_u64_array` | `tile_shape`, `morton_bits`, `mode_order`, `parent_shape`, `shard_offset` |
| `hurray_descriptor_field_i64_array` | `strides`, `outer_strides`, `inner_strides` |

### Field Identifiers

`HurrayDescriptorField` MUST be a `uint32` with named constants, and MUST NOT be a C
enum: a caller may pass any integer, and an out-of-range value materialised as a Rust
enum is undefined behaviour.

The identifier encoding is:

```text
field_id = (range << 16) | (discriminant << 8) | field_index
```

| Range | Sub-object | Discriminant byte |
|---|---|---|
| `0x0001` | layout payload | layout tag |
| `0x0002` | quantization | scheme tag; `0x00` addresses the section itself |
| `0x0003` | statistics | `0x00` |
| `0x0004` | shard | `0x00` |
| `0x0005` | extension type | `0x00` |
| `0x0006` | composite member | `0x00` |

The embedded discriminant makes applicability a comparison rather than a table lookup,
and the layout range inherits the layout tag space's reserved and private ranges.
`0x0000_0000` MUST NOT be a valid identifier, so that a zeroed variable is rejected
rather than read as the first field.

Quantization is the only sub-object with an inner sum type, so it is the only section
that uses the discriminant byte. `0x0002_0000` MUST read the quantization scheme tag
itself, which a caller MUST read before selecting any scheme-specific identifier. The
remaining sections are flat and carry `0x00` in that byte.

Allocation follows the anti-rebind rule in
[ADR-019](../adr/ADR-019-format-evolvability-as-core-property.md): a retired identifier
MUST NOT be reused for a different field.

### Field Semantics

- Every scalar field in the format is unsigned or boolean but two, so almost all of
  them widen losslessly into `uint64`. A boolean MUST read as `0` or `1`, and a wire tag
  byte MUST read as its documented value.
- A signed scalar MUST be read through the signed accessor, and a real scalar through
  the real accessor. A `zero_point` is routinely negative, and an unsigned read would
  report it as a value near `2^64`.
- Stride fields MUST be read through the signed array accessor: a negative stride
  reverses a dimension and a zero stride broadcasts one, both of which an unsigned read
  would misreport.
- Array accessors MUST use the capacity/length in-out protocol that
  `hurray_descriptor_shape` uses: `*out_len` carries the caller's capacity in and the
  true length out, and a null or insufficient buffer returns
  `HURRAY_ERR_BUFFER_TOO_SMALL` with the true length written.
- An absent optional field MUST return `HURRAY_ERR_FIELD_NOT_PRESENT` and MUST NOT
  write to the output: absence is a state this format expresses through a cleared flag
  bit, not a value, and reporting it as a zero would be indistinguishable from data.
  This covers an absent section, an unset optional field, and a symmetric quantization
  scheme's zero-point buffer index.
- A statistics field whose guard bit in `computed_mask` is clear MUST return
  `HURRAY_ERR_FIELD_NOT_PRESENT`. The statistics fields are physically present whatever
  the mask says, so an implementation that returned the stored bytes would let a caller
  read an uncomputed statistic by omitting the mask check the spec requires.
  `0x0003_0000` MUST read the raw mask, for a caller that branches on it directly.
- A field identifier that does not belong to the descriptor's layout or quantization
  scheme MUST return `HURRAY_ERR_FIELD_NOT_APPLICABLE`. An identifier outside any
  allocated range, or passed to an accessor of the wrong type, MUST return the same
  code.
- Every accessor takes a `level`, which selects the tile to read in a nested tiled
  layout. `level` MUST be `0` for every layout that does not nest and for every section
  field, and a non-zero value MUST be rejected with
  `HURRAY_ERR_INDEX_OUT_OF_BOUNDS` rather than ignored.

## Streaming Over a Transport Vtable

A reader and a writer are **blocking** and take their bytes from callbacks the caller
supplies at construction. The implementation MUST NOT open a path or own a file
descriptor on the caller's behalf, and MUST NOT link an async runtime into the C library:
a vtable composes with a socket, a pipe, a memory span, or a caller's own object, and
file descriptors are not portable across the platforms this ABI targets.

```c
typedef HurrayStatus (*HurrayTransportReadFn)(
    void* ctx, uint8_t* dst, size_t len, size_t* out_filled);
typedef HurrayStatus (*HurrayTransportWriteFn)(
    void* ctx, const uint8_t* src, size_t len, size_t* out_written);
typedef void (*HurrayTransportReleaseFn)(void* ctx);
typedef HurrayStatus (*HurrayTransportReadAtFn)(
    void* ctx, uint64_t offset, uint8_t* dst, size_t len, size_t* out_filled);
typedef HurrayStatus (*HurrayTransportSizeFn)(void* ctx, uint64_t* out_size);

typedef struct HurrayTransport {
    void*                    ctx;
    HurrayTransportReadFn    read;     /* required by a stream reader */
    HurrayTransportWriteFn   write;    /* required by any writer */
    HurrayTransportReleaseFn release;  /* optional */
    HurrayTransportReadAtFn  read_at;  /* required by a file reader */
    HurrayTransportSizeFn    size;     /* required by a file reader */
} HurrayTransport;
```

One struct serves both surfaces, and a caller populates only the callbacks its transport
has. `read_at` and `size` are appended rather than grouped beside `read` so that a caller
using positional initialisation gets null for them instead of a release callback where a
positional read is expected.

The struct is copied at construction; `ctx` MUST stay valid until the owning handle is
destroyed, at which point `release` is invoked exactly once.

### Transport Rules

These are ABI surface, and none of them may be left to inference.

- **Partial transfers are legal.** A callback MUST report the number of bytes it moved.
  Fewer than requested is a short transfer: not an error, and not end of input.
- **Zero means different things in each direction.** A read reporting `0` with
  `HURRAY_OK` is end of input. A write reporting `0` with `HURRAY_OK` is **not**
  completion — it means the sink will accept nothing more, and is reported as
  `HURRAY_ERR_SINK_CLOSED` when an item is in flight.
- **EOF, error, and closed are three distinct statuses**, never one.
- **A failing callback MUST have moved no bytes** and MUST write `0` through its count
  out-parameter. A transport that partially succeeds and *then* fails MUST report the
  partial count with `HURRAY_OK` and surface the failure on the next call — what POSIX
  `read` and `write` already do. This rule is what makes a failed call recoverable: the
  handle's stream position still matches the transport's, so the call MAY be retried and
  will resume rather than restart.
- **A range request is absolute.** `read_at` MUST read from the offset it is given and
  MUST NOT rely on, or be relied upon to advance, any cursor of its own; a short range
  read is re-requested at the offset that follows. Because the request carries its own
  position, a failed `read_at` is always retryable — re-issuing it reads the same bytes.
- **`size` is read once**, when a file reader is created. A container is located from its
  end, so its size is a precondition of opening it and never something a read discovers:
  `read_at` reporting `0` means the file is shorter than its own structure claims, which
  is `HURRAY_ERR_UNEXPECTED_EOF` and not an end-of-input signal.
- **No reentrancy.** A callback MUST NOT call any Hurray function on the handle that
  invoked it.
- **The context pointer belongs to the caller.** The implementation MUST NOT free it;
  `release` is the only teardown hook.
- **One handle, one thread at a time.** The absence of an internal runtime makes this a
  rule rather than a threading model.

### Reader and Writer Contracts

| Function | Contract |
|---|---|
| `hurray_reader_new(transport, options, out_reader)` | `transport->read` MUST be non-null. `options` MAY be `NULL` for the documented defaults. |
| `hurray_reader_next_tensor(reader, out_tensor)` | Returns `HURRAY_OK` with a new `HurrayTensor*`, `HURRAY_END_OF_STREAM` at a clean tensor boundary, `HURRAY_ERR_UNEXPECTED_EOF` if input ended inside a descriptor or buffer, or the read callback's own status. |
| `hurray_reader_destroy(&reader)` | Destroys the reader, releases the transport context, writes `NULL` through the argument. Tensors already produced are unaffected. |
| `hurray_writer_new(transport, options, out_writer)` | `transport->write` MUST be non-null. `options` MAY be `NULL`. |
| `hurray_writer_write_tensor(writer, descriptor, payloads, count)` | `payloads` MUST have one `HurrayIoSlice` per buffer in the descriptor's buffer table, in that order, each of the declared `byte_size`. The tensor is validated before any byte reaches the sink. |
| `hurray_writer_destroy(&writer)` | Destroys the writer and releases the transport context. There is nothing to flush. |

A writer whose sink failed **after** the first byte of a tensor MUST be poisoned: a
truncated item is on the wire, a stream cannot be un-sent, and a fresh writer over the
same connection is not a recovery. A writer whose sink failed **before** the first byte
MUST stay usable.

Readers MUST allocate every buffer at the alignment its buffer handle *declares*, and the
transport MUST write into that allocation directly: a consumer is entitled to rely on the
declared alignment for SIMD loads, and no copy of a payload may happen after the transport
writes it.

A `HurrayTensor` owns its descriptor and its buffers. `hurray_tensor_descriptor` and
`hurray_tensor_buffers` hand back **borrowed** handles, valid until
`hurray_tensor_destroy`, which releases both halves in one call so the teardown order is
not the caller's problem.

> **Note (non-normative):** composite tensors are not reachable from this surface yet. A
> composite head arrives as the buffer-less tensor it is on the wire, and writing one is
> refused. File containers need random access and are a separate surface.

## File Containers Over a Transport Vtable

A `HRRYFILE` container is **addressed**, not consumed: its trailer sits at `size - 40`,
its footer index is located from there, and a tensor is read by name without the bytes
between being touched. A file reader therefore asks for absolute ranges through `read_at`
and learns the file's length through `size`, while a file writer emits one forward pass
through `write` alone and never seeks back.

### Reader and Writer Contracts

| Function | Contract |
|---|---|
| `hurray_file_reader_new(transport, out_reader)` | `transport->read_at` and `transport->size` MUST both be non-null. Reads the header, trailer, index and KV section. A failed open is terminal: there is no index to start from, so no handle is produced and the transport context is released. |
| `hurray_file_reader_tensor_count(reader, out_count)` | Number of footer-index entries. Every tensor is individually addressable, composite heads and members included. |
| `hurray_file_reader_tensor_name(reader, index, out_name, out_len)` | Borrows entry `index`'s name as UTF-8 bytes, **not** NUL-terminated, valid until the reader is destroyed. `HURRAY_ERR_INDEX_OUT_OF_BOUNDS` past the end. |
| `hurray_file_reader_read_descriptor(reader, name, name_len, out_descriptor)` | Decodes one descriptor and reads no buffer data. The descriptor handle is **owned** by the caller. |
| `hurray_file_reader_read_tensor(reader, name, name_len, out_tensor)` | Reads a descriptor and every buffer, as an owned `HurrayTensor*`. `HURRAY_ERR_NOT_FOUND` if the name is not in the index. |
| `hurray_file_reader_kv_count(reader, out_count)` / `hurray_file_reader_kv_entry(reader, index, out_entry)` | The file-level KV section, in file order. An entry's pointers borrow from the reader. |
| `hurray_file_reader_destroy(&reader)` | Destroys the reader, releases the transport context, writes `NULL`. Tensors already produced are unaffected; borrowed name and KV pointers are not. |
| `hurray_file_writer_new(transport, options, out_writer)` | `transport->write` MUST be non-null. Writes the 64-byte header during the call. `options` MAY be `NULL` for 4096-byte data alignment and an index in write order. |
| `hurray_file_writer_write_tensor(writer, name, name_len, descriptor, payloads, count)` | As the stream writer's, plus the name that becomes the footer-index entry: non-empty, at most 65 535 bytes, unique within the file. |
| `hurray_file_writer_finish(writer, kv, kv_count)` | Writes the KV section, the footer index and the trailer. `kv` MAY be `NULL` with a count of `0`. Until this returns `HURRAY_OK` the bytes are not a container. Calling it twice **after it returned `HURRAY_OK`** MUST fail; a footer that failed before its first byte MAY be retried. |
| `hurray_file_writer_destroy(&writer)` | Destroys the writer and releases the transport context. Any flush or close belongs to the transport's release callback. |

**The file writer's failure contract is the stream writer's.** The rule stated above for
`hurray_writer_write_tensor` is normative for both surfaces: a write that fails **after**
the first byte of its item MUST poison the writer, and one that fails **before** it MUST
leave the writer usable, the item's name unconsumed, and the file completable. The status
distinguishes the two — `HURRAY_ERR_SINK_CLOSED` after, the callback's own status before.

`hurray_file_writer_finish` is an item under the same rule, which makes it the one retry
worth having here: the data is already on the sink and only the index is missing, so a
footer that failed before its first byte MAY be written again.

### KV Metadata

One tagged value type serves reads and writes, so a KV section read from one file can be
handed straight to another file's writer without being transcribed:

```c
#define HURRAY_KV_STRING   0x01
#define HURRAY_KV_INT64    0x02
#define HURRAY_KV_UINT64   0x03
#define HURRAY_KV_FLOAT64  0x04
#define HURRAY_KV_BOOL     0x05
#define HURRAY_KV_BYTES    0x06
#define HURRAY_KV_ARRAY    0x07

typedef struct HurrayKvValue {
    uint8_t  kind;
    bool     bool_value;
    int64_t  i64_value;
    uint64_t u64_value;
    double   f64_value;
    const uint8_t* data;       /* STRING (UTF-8, no NUL) or BYTES */
    size_t         data_len;
    const struct HurrayKvValue* items;  /* ARRAY */
    size_t                      item_count;
} HurrayKvValue;

typedef struct HurrayKvEntry {
    const uint8_t* key;        /* UTF-8, non-empty, no NUL terminator */
    size_t         key_len;
    HurrayKvValue  value;
} HurrayKvEntry;
```

The `kind` values are the KV section's own wire tags, not a second numbering. Fields the
tag does not select MUST be zero on a read and are ignored on a write. An array's elements
MUST be scalars — the format forbids an array of arrays, and a nested one MUST be refused
with `HURRAY_ERR_INVALID_FILE` rather than flattened.

> **Note (non-normative):** composite tensors are not reachable from this surface yet. A
> composite head reads back as the buffer-less tensor it is in the file, and writing one is
> refused.

## Panic Safety

Rust panics MUST NOT propagate across the FFI boundary. Every `extern "C"` function
that calls Rust code MUST wrap the call in `std::panic::catch_unwind`. If a panic is
caught, the function MUST:

1. Log or store the panic message (implementation-defined).
2. Return a well-defined error code (e.g., `HURRAY_ERR_INTERNAL_PANIC`).
3. Leave no partially-constructed state visible to the caller.

## Error Handling

All fallible FFI functions MUST return an error code of type `HurrayStatus`
(`int32`). The value `0` (`HURRAY_OK`) indicates success. All other values indicate
errors.

```c
typedef int32_t HurrayStatus;

#define HURRAY_OK                    0
#define HURRAY_ERR_INVALID_MAGIC    -1
#define HURRAY_ERR_VERSION_MISMATCH -2
#define HURRAY_ERR_INVALID_LAYOUT   -3
#define HURRAY_ERR_INVALID_TYPE     -4
#define HURRAY_ERR_BUFFER_TOO_SMALL -5
#define HURRAY_ERR_NULL_POINTER     -6
#define HURRAY_ERR_INTERNAL_PANIC   -7
#define HURRAY_ERR_INVALID_SYNC_MODE    -8
#define HURRAY_ERR_SYNC_MODE_MISMATCH   -9
#define HURRAY_ERR_INTERNAL            -10
#define HURRAY_ERR_INDEX_OUT_OF_BOUNDS -11
#define HURRAY_ERR_FIELD_NOT_PRESENT   -12
#define HURRAY_ERR_FIELD_NOT_APPLICABLE -13
#define HURRAY_ERR_TRANSPORT           -14
#define HURRAY_ERR_SINK_CLOSED         -15
#define HURRAY_ERR_UNEXPECTED_EOF      -16
#define HURRAY_ERR_CORE_UNUSABLE       -17
#define HURRAY_ERR_LIMIT_EXCEEDED      -18
#define HURRAY_ERR_INVALID_STREAM      -19
#define HURRAY_ERR_NOT_FOUND           -20
#define HURRAY_ERR_INVALID_FILE        -21

#define HURRAY_END_OF_STREAM             1
```

Functions MUST return `HURRAY_ERR_NULL_POINTER` for any required pointer argument
that is `NULL`, without invoking undefined behaviour.

Positive status values report a normal outcome that is not plain success.
`HURRAY_END_OF_STREAM` is the only one defined: a stream ending is not a failure, so
consumers MUST test for errors with `status < 0` rather than `status != HURRAY_OK`.

## Buffer Release Callbacks

Buffer handles carry a **release callback** to support zero-copy buffer sharing with
non-Rust runtimes. When `hurray-ffi` wraps an externally-owned buffer, the caller
provides a release function and a context pointer:

```c
typedef void (*HurrayReleaseCallback)(void* buffer, void* context);

HurrayStatus hurray_buffer_from_ptr(
    void*                 data,
    uint64_t              byte_size,
    uint32_t              alignment,
    uint8_t               device_tag,
    HurrayReleaseCallback release,
    void*                 release_context,
    HurrayBuffer**        out_handle
);
```

The release callback MUST be called exactly once when the buffer's reference count
reaches zero. The implementation MUST NOT call the release callback from a destructor
that runs on a foreign thread without the caller's consent.

## Thread Safety

- All handles MUST be safe to use from a single thread at a time (i.e., `Send` but
  not `Sync` in Rust terms).
- Concurrent access to the same handle from multiple threads is undefined behaviour
  unless documented otherwise.
- Reference counting for shared buffer handles MUST be performed with atomic
  operations (`std::sync::atomic`).

## Naming Conventions

All public symbols MUST be prefixed with `hurray_`. Type names use `Hurray` prefix
with PascalCase. Error codes use `HURRAY_ERR_` prefix with SCREAMING_SNAKE_CASE.

## Header Generation

A C header file (`hurray.h`) MUST be generated from the Rust source using `cbindgen`
as part of the build process. The generated header MUST be checked into the repository
and kept in sync with the Rust source. CI MUST fail if the generated header differs
from the committed one.

## Buffer Handoff Synchronisation

The buffer protocol's `sync_mode` field (see `docs/spec/buffer-protocol.md`
§ Stream and Event Synchronisation) declares one of three producer-side
synchronisation mechanisms in the binary descriptor. The C ABI carries the
corresponding **payload** out of band; the discriminant itself is read from the
buffer handle's binary descriptor and is NOT duplicated in the ABI struct.

The ABI layer MUST cross-check the `sync_mode` declared in the descriptor
against the payload provided at handoff time. A mismatch is a producer-side
bug; the ABI MUST reject the handoff with `HURRAY_ERR_SYNC_MODE_MISMATCH`
before returning a buffer handle to the consumer.

### Per-Mode Payloads

#### `sync_mode = SYNC_PRODUCER_SYNCED` (`0x00`)

No additional fields beyond the buffer pointer, byte size, alignment, device
tag, release callback, and release context defined in [Buffer Release
Callbacks](#buffer-release-callbacks).

The producer MUST have issued a host-side wait on the device stream(s) that
wrote the buffer before calling the handoff function. For CPU buffers
(`device_tag == 0x00`), the producer MUST have issued a host memory fence if
concurrent host-side writes exist.

#### `sync_mode = SYNC_EVENT` (`0x01`)

The handoff struct carries an opaque event handle and an event-release callback
in addition to the buffer fields:

| Field | Type | Description |
|-------|------|-------------|
| `sync_handle` | opaque pointer (`void*`) | Device-vendor-specific event handle (e.g., `cudaEvent_t`, `hipEvent_t`, `MTLSharedEvent`, `VkSemaphore`). Opaque to the ABI layer. |
| `sync_handle_device_tag` | `uint8` | Device tag identifying the event handle's driver context. MUST equal the buffer's `device_tag`. The ABI MUST reject the handoff if it does not. |
| `event_release_fn` | `void (*)(void* sync_handle, void* context)` | Callback the consumer calls exactly once after issuing its stream-wait. Thread-safe. |
| `event_release_context` | opaque pointer (`void*`) | Context pointer passed to `event_release_fn`. |

The producer MUST record the event on the writing stream(s) before calling the
handoff function. The producer MUST NOT defer event recording past the
handoff; doing so would allow the consumer to issue a stream-wait on an
unrecorded event and deadlock.

The consumer MUST issue a device-stream-wait on `sync_handle` on every stream
that will access the buffer before enqueuing any work that touches the
buffer's bytes. The consumer MUST call `event_release_fn(sync_handle,
event_release_context)` exactly once after all stream-waits have been issued
(typically immediately after handoff). `event_release_fn` MUST be safe to call
from any thread.

The event-release callback is **separate from** the buffer-release callback
defined in [Buffer Release Callbacks](#buffer-release-callbacks): a consumer
in `SYNC_EVENT` mode makes two release calls per buffer, with independent
lifetimes.

#### `sync_mode = SYNC_CONSUMER_STREAM` (`0x02`)

At handoff request time, the consumer supplies an opaque stream handle to the
producer:

| Field | Type | Description |
|-------|------|-------------|
| `consumer_stream` | opaque pointer (`void*`) | Device-vendor-specific stream handle (e.g., `cudaStream_t`, `hipStream_t`, `id<MTLCommandQueue>`). Opaque to the ABI layer. Supplied by the consumer at handoff request time. |
| `consumer_stream_device_tag` | `uint8` | Device tag identifying the stream handle's driver context. MUST equal the buffer's `device_tag`. The ABI MUST reject the handoff if it does not. |

The producer MUST issue a device-side ordering dependency from its writing
stream(s) onto `consumer_stream` before the handoff function returns a buffer
handle. The producer-side stream is not blocked; the ordering is established
on the device.

The consumer MAY access the buffer on `consumer_stream` after handoff returns,
but MUST NOT access the buffer on any other stream until it has issued an
inter-stream wait on that other stream.

`SYNC_CONSUMER_STREAM` does not introduce a second release callback: there is
no event handle whose lifetime must be managed.

### ABI Cross-Check

For every buffer handed off through the C ABI, the implementation MUST:

1. Read the `sync_mode` byte at offset 13 of the buffer handle binary
   descriptor.
2. Verify that the payload provided by the producer at handoff time matches
   the declared `sync_mode`:
   - `SYNC_PRODUCER_SYNCED` MUST NOT carry a `sync_handle` or `consumer_stream`.
   - `SYNC_EVENT` MUST carry a non-NULL `sync_handle`, an `event_release_fn`,
     and a `sync_handle_device_tag` equal to the buffer's `device_tag`.
   - `SYNC_CONSUMER_STREAM` MUST carry a non-NULL `consumer_stream` and a
     `consumer_stream_device_tag` equal to the buffer's `device_tag`.
3. Reject any reserved `sync_mode` value (`0x03`–`0xFE`) and the invalid value
   `0xFF` with `HURRAY_ERR_INVALID_SYNC_MODE`.
4. Reject a mismatch between descriptor and payload with
   `HURRAY_ERR_SYNC_MODE_MISMATCH`.

### Backward Compatibility for Pre-Version-2 Consumers

A consumer compiled against C ABI version `1` that does not inspect `sync_mode`
will receive buffers whose `sync_mode == SYNC_PRODUCER_SYNCED` by default. This
is the safe fallback: the strongest synchronisation guarantee, no per-mode
payload required, and behaviour identical to the version-`1` contract. A
producer that wishes to interoperate with version-`1` consumers MUST set
`sync_mode = SYNC_PRODUCER_SYNCED` for every buffer it hands off and MUST
issue the corresponding host-side wait before handoff.
