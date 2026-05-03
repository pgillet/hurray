# Buffer Protocol — Hurray Format Specification

> **Status:** Draft

> This section uses RFC 2119 key words: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT,
> SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

## Scope

This section defines the **buffer protocol**: the rules governing how tensor data
buffers and quantization-parameter buffers are represented, aligned, located on
a device, owned, and released. It is the normative reference for all other
sections that reference buffer handles, device tags, or alignment requirements.

The buffer protocol is independent of the interchange transport (in-process,
IPC, or cross-machine); `interchange.md` defines the transport-level framing.

---

## Buffer Handle

A **buffer handle** is the unit by which a tensor descriptor references a
contiguous region of memory. Each handle appears as a 16-byte entry in the
buffer table of the tensor descriptor (see `metadata.md` § Buffer Table).

| Offset | Field | Type | Description |
|--------|-------|------|-------------|
| 0 | `byte_size` | `uint64` | Size of the buffer in bytes. `0` denotes an empty buffer. |
| 8 | `alignment` | `uint32` | Minimum alignment of the buffer's base address in bytes. MUST be a power of two; MUST be at least `64` for non-empty buffers (`byte_size > 0`); any power-of-two value (including `1`) is valid for empty buffers (`byte_size == 0`). See [§ Empty Buffers](#empty-buffers). |
| 12 | `device_tag` | `uint8` | Device where this buffer resides. See [§ Device Tags](#device-tags). |
| 13 | `_reserved` | `uint8[3]` | MUST be `0x00`. Readers MUST reject a descriptor with non-zero reserved bytes. |

All multi-byte fields MUST be encoded in little-endian byte order.

> **Note (non-normative):** The buffer handle in the descriptor is a *declaration*
> of a buffer's properties, not a pointer. The actual pointer (or shared-memory
> handle, or RDMA registration) is communicated out-of-band via the interchange
> protocol or, for in-process use, via the C ABI defined in `docs/impl/c-ffi.md`.

---

## Alignment

### Minimum Alignment

The base address of every non-empty buffer MUST be aligned to at least **64
bytes**. This ensures compatibility with all current SIMD instruction sets
(AVX-512, NEON, SVE) without requiring per-operation alignment negotiation.

A writer MUST set `alignment` to the actual alignment it guarantees, which MUST
be a power of two and MUST be at least `64`. A writer MAY set `alignment` to a
larger value (e.g., a page boundary of `4096` or `65536`) to communicate a
stronger guarantee.

A reader MAY rely on the declared alignment for SIMD loads. A reader MUST NOT
rely on alignment stronger than what is declared in the `alignment` field.

### Page Alignment for GPU and IPC

Buffers shared across process boundaries (IPC) or placed in device memory (GPU)
SHOULD be aligned to the host page size, which is typically `4096` bytes.
Writers targeting GPU or IPC transport MUST set `alignment` to at least `4096`.

Buffers intended for RDMA transfer SHOULD be aligned to the RDMA provider's
minimum pinnable unit, which is typically `4096` bytes. Writers targeting RDMA
MUST set `alignment` to at least `4096`.

### Empty Buffers

A buffer with `byte_size = 0` is an **empty buffer**. The 64-byte minimum
alignment requirement does not apply to an empty buffer: there are no addressable
bytes to align. A writer MAY set `alignment` to any power-of-two value (including
`0x00000001`) for an empty buffer. A reader MUST NOT dereference the pointer of
an empty buffer.

In C ABI contexts, an empty buffer MAY be represented by a null pointer. A
non-null pointer for an empty buffer is also valid; readers MUST handle both.

---

## Device Tags

The `device_tag` field identifies the memory space in which the buffer resides.

| Value | Device |
|-------|--------|
| `0x00` | CPU host memory |
| `0x01` | CUDA device memory |
| `0x02` | ROCm device memory |
| `0x03` | Metal device memory (Apple Silicon unified memory) |
| `0x04`–`0xEF` | Reserved for future specification versions |
| `0xF0`–`0xFE` | Implementation-private device types |
| `0xFF` | Reserved (invalid) |

A reader MUST reject a buffer handle whose `device_tag` is `0xFF`.

Tags in the range `0x04`–`0xEF` MUST NOT be used by any implementation; they
are reserved for future specification versions.

Tags in the range `0xF0`–`0xFE` MAY be used by implementations for
private or experimental device types. Descriptors carrying private device tags
MUST NOT be exchanged between independent implementations unless both parties
have agreed on the semantics out of band.

### Device Colocation

All buffers referenced by a single tensor descriptor (data buffer + all
quantization-parameter buffers) MUST share the same `device_tag`. A reader
MUST reject a descriptor whose buffers carry different `device_tag` values.

For `TENSOR_PUT` transfers (see `interchange.md`), the client unilaterally
declares the destination `device_tag` in the descriptor; the server MAY reject
the transfer with `DEVICE_UNAVAILABLE` but MUST NOT silently place buffers on a
different device.

> **Note (non-normative):** Device colocation ensures that quantized tensor
> kernels can dereference both the data and the quantization parameters without
> triggering cross-device transfers. A writer that needs quantization parameters
> on a different device must emit a separate tensor descriptor.

When buffer handles are exchanged across machines, the device selection rules
in `interchange.md` § Device Negotiation govern which device tag is valid for a
given transfer.

---

## Buffer Ownership and Lifetime

### Ownership Model

At any instant, exactly one entity — the **owner** — is responsible for the
buffer's memory. Ownership may be transferred between a producer and a consumer
as part of the interchange protocol, but it is never shared: concurrent
read/write access to the same buffer by multiple owners is a protocol error.

#### In-Process

In in-process exchange, the producer creates the buffer and holds ownership
until the consumer signals that it has retained a reference (via the release
callback mechanism described below). The consumer then owns the buffer for the
duration of its use and MUST release it exactly once when done.

#### IPC

In IPC exchange via shared memory, the producer creates and owns the shared
memory segment. The consumer maps the segment into its own address space. The
producer MUST NOT unmap or destroy the segment until all consumers have
unmapped it. The IPC channel MUST convey a release signal so that the producer
knows when it may reclaim the segment.

#### Cross-Machine

In cross-machine exchange, the sender owns the source buffer and the receiver
owns the destination buffer. There is no shared buffer; data is copied (or
RDMA-written) from sender to receiver. See `interchange.md` for framing details.

### Release Callback

For in-process and IPC exchange, the buffer handle is augmented at the ABI level
with a **release callback**: a function pointer that the consumer calls exactly
once when it has finished using the buffer. The release callback is not encoded
in the binary descriptor; it is supplied by the producer at handoff time via the
C ABI (see `docs/impl/c-ffi.md`).

A consumer MUST call the release callback exactly once. A consumer MUST NOT
access the buffer after calling the release callback. A producer's release
callback MUST be safe to call from any thread.

### Reference Counting

Reference counting is an **implementation detail** — not a normative contract
(see `docs/adr/ADR-009-release-callback-not-normative-refcount.md`). A producer
that wishes to support multiple simultaneous consumers of the same buffer MUST
implement reference counting internally. Each consumer receives a separate buffer
handle whose release callback decrements the internal count; the actual
deallocation occurs only when the count reaches zero. Consumers are unaware of
this; they call their release callback exactly once as the normative contract
requires.

> **Note (non-normative):** This is the same model used by DLPack's
> `DLManagedTensor.deleter`. It keeps the ABI surface minimal and allows each
> language binding to use its own lifetime management idiom (Python GC,
> Rust `Arc`, etc.) without bridging to a C reference count.

---

## Zero-Copy Invariants

The buffer protocol is designed to preserve zero-copy access across language
and runtime boundaries. The following invariants MUST hold at all times:

1. **No implicit copies.** Neither the producer nor the consumer MAY copy the
   buffer contents as part of the handoff. Copies are only permitted when
   explicitly requested by the interchange protocol (e.g., layout transcoding
   in response to a `TENSOR_REQUEST` that specifies a different layout tag).
2. **No in-place mutation after handoff.** Once a producer has handed off a
   buffer to a consumer, the producer MUST NOT modify the buffer's contents.
   Mutating a buffer that is held by a consumer is a protocol error.
3. **Quantization parameter buffers are immutable.** Scale and zero-point
   buffers MUST NOT be modified after the tensor descriptor is emitted. They
   are part of the tensor's logical value and MUST be treated as read-only by
   all consumers.
4. **Pointer stability.** The base address of a buffer MUST NOT change for the
   duration of the consumer's hold. Buffer defragmentation or garbage collection
   that moves the buffer is the producer's responsibility to prevent while any
   consumer holds a reference.

---

## Relationship to Other Sections

- **`metadata.md`** defines the binary encoding of buffer handles in the buffer
  table. The `alignment` and `device_tag` fields are declared there; this file
  defines the normative rules they must satisfy.
- **`data-model.md`** defines empty tensors (zero-size dimensions). This file
  specifies the corresponding empty-buffer rules (null pointer allowed,
  alignment waived, no dereference).
- **`quantization.md`** defines quantization-parameter buffers. This file's
  device-colocation and immutability rules apply to those buffers.
- **`interchange.md`** defines in-process, IPC, and cross-machine transport.
  This file's alignment and ownership rules are prerequisites for all three
  transport modes.
- **`docs/impl/c-ffi.md`** defines the C ABI for buffer handle handoff,
  including the release callback signature.

---

## Open Questions

All open questions in this section are resolved. See
`docs/adr/ADR-009-release-callback-not-normative-refcount.md` (OQ-1).
