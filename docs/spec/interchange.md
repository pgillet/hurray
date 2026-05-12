# Interchange — Hurray Format Specification

> **Status:** Draft

> This section uses RFC 2119 key words: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT,
> SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

## Scope

This section defines how Hurray tensors are exchanged between producers and consumers.
Three interchange modes are in scope:

| Mode | Description |
|------|-------------|
| In-process | Shared memory within a single address space; zero-copy by pointer passing |
| IPC | Cross-process on a single machine; shared memory segments or Unix domain sockets |
| Network transport | Cross-machine over a network interface; client-server streaming protocol |

This section focuses primarily on the **network transport** mode, as it is the most
complex and the most relevant to distributed inference pipelines. In-process and IPC
modes are covered in [In-Process and IPC](#in-process-and-ipc).

A Hurray stream MAY contain zero or more tensors. Back-to-back concatenation of
self-delimiting descriptor+data pairs is the canonical multi-tensor encoding: a
reader processes one tensor at a time, advancing to the next descriptor after the
current tensor's data is consumed. No container header, index, or tensor names are
defined at this level (see `docs/adr/ADR-010-multi-tensor-collections-deferred.md`).

> **Note (non-normative):** The design of the Hurray network transport protocol draws
> inspiration from Apache Arrow Flight, but differs in several key respects: it is
> tensor-focused (not columnar), it supports layout and device negotiation, it defines
> on-the-fly transcoding, and its data plane is designed for large buffer transfers
> where gRPC framing overhead is impractical.

---

## In-Process and IPC

### In-Process

Within a single address space, tensor interchange is accomplished by passing a tensor
descriptor (see `metadata.md`) and a buffer handle (see `buffer-protocol.md`) by
value. No serialization is required. The buffer handle carries a release callback (see `buffer-protocol.md` § Buffer Ownership and Lifetime and ADR-009); the receiver MUST retain the handle for the duration of its use and call the release callback exactly once when done.

### IPC

For cross-process interchange on a single machine, two mechanisms are supported:

1. **Shared memory**: the producer maps a shared memory segment and places the data
   buffer there. The tensor descriptor is transmitted over any IPC channel (pipe, Unix
   domain socket, etc.). The `byte_offset` field in the descriptor identifies the
   buffer's position within the shared segment.
2. **Unix domain socket streaming**: the producer serializes the tensor using the
   network transport framing defined below, transmitted over a Unix domain socket.
   This is slower than shared memory but requires no shared-memory setup.

In both cases, buffer alignment requirements from `buffer-protocol.md` apply.

---

## Network Transport Protocol

### Overview

The Hurray network transport protocol is a **client-server streaming protocol** for
transferring tensor data over a reliable, ordered byte stream (e.g., TCP). It consists
of:

- A **control plane**: session establishment, capability negotiation, tensor requests,
  and error signalling. Uses message framing defined in this section.
- A **data plane**: tensor descriptor and buffer transmission. Uses the same message
  framing, but data frame payloads MAY be transferred over a separate high-throughput
  channel (e.g., RDMA) by prior agreement during session establishment.

The protocol is **half-duplex per stream**: within a single stream, the client sends a
request and the server responds with a sequence of messages. Multiple streams MAY be
multiplexed over a single connection.

> **Note (non-normative):** The `stream_id` field is an opaque per-stream identifier.
> Implementations MAY use one TCP connection per stream, or MAY multiplex multiple
> streams over a single connection by demultiplexing on `stream_id`. Messages are
> self-framing (fixed-size header + `payload_length` bytes), so a receiver can always
> advance to the next message regardless of `stream_id`. Normative multiplexing rules
> are not defined in this version of the spec.

### Message Framing

Every message on the wire consists of a **message header** followed by a **payload**.
All multi-byte fields are little-endian.

**Message header** (12 bytes):

| Offset | Field | Type | Description |
|--------|-------|------|-------------|
| 0 | `message_type` | `uint32` | Message type tag (see [Message Types](#message-types)) |
| 4 | `stream_id` | `uint32` | Stream identifier. `0x00000000` is reserved. |
| 8 | `payload_length` | `uint32` | Length of the payload in bytes, not including the header. |

The payload follows immediately after the header. A receiver MUST read exactly
`payload_length` bytes as the payload. A receiver MUST reject messages whose
`payload_length` exceeds the receiver's configured maximum message size, and MUST
send an `ERROR` message in response.

### Message Types

| Tag | Name | Direction | Description |
|-----|------|-----------|-------------|
| `0x00000001` | `CLIENT_HELLO` | Client → Server | Session initiation and capability advertisement |
| `0x00000002` | `SERVER_HELLO` | Server → Client | Session acceptance and capability advertisement |
| `0x00000003` | `TENSOR_REQUEST` | Client → Server | Request a tensor by key, with layout and device preferences |
| `0x00000004` | `TENSOR_DESCRIPTOR` | Server → Client | Tensor descriptor, precedes data frames |
| `0x00000005` | `TENSOR_DATA` | Server → Client | Tensor data frame (partial or complete) |
| `0x00000006` | `TENSOR_DATA_END` | Server → Client | Signals that all data frames for a tensor have been sent |
| `0x00000007` | `TENSOR_PUT` | Client → Server | Push a tensor to the server (descriptor + data) |
| `0x00000008` | `TENSOR_PUT_ACK` | Server → Client | Acknowledges receipt of a pushed tensor |
| `0x00000009` | `ERROR` | Either | Error response; terminates the stream |
| `0x0000000A` | `PING` | Either | Keepalive request |
| `0x0000000B` | `PONG` | Either | Keepalive response |
| `0x0000000C` | `RDMA_REGISTER` | Either | RDMA memory region registration: shares rkey and remote address |
| `0x0000000D` | `RDMA_READY` | Either | Acknowledges RDMA_REGISTER; signals readiness for the RDMA transfer |
| `0x000000F0`–`0x000000FE` | (private extension) | — | Reserved for implementation-private extensions |
| `0x000000FF` | (invalid) | — | Reserved; MUST NOT be used |

A receiver that encounters an unrecognised `message_type` MUST send an `ERROR`
message and close the stream.

Message type values in the range `0x000000F0`–`0x000000FE` are reserved for
implementation-private extensions. Implementations MAY use these values for
private extension message types; messages using private extension types MUST
NOT be sent to a peer that has not agreed to the extension out of band. The
value `0x000000FF` is reserved and MUST NOT be used.

---

## Session Establishment

### CLIENT_HELLO Payload

| Field | Type | Description |
|-------|------|-------------|
| `protocol_version` | `uint32` | Protocol version. Current version: `0x00000001`. |
| `max_message_size` | `uint32` | Maximum payload size (bytes) the client will accept. |
| `capability_flags` | `uint64` | Bitmask of client capabilities (see [Capability Flags](#capability-flags)). |
| `supported_layouts` | `layout_entry[layout_count]` | Layout entries the client can consume, in preference order (most preferred first). Preceded by a `uint16` count field. Each entry is encoded as defined in [Layout Entry Encoding](#layout-entry-encoding). |
| `supported_devices` | `uint8[device_count]` | Device tags the client can accept (see [Device Tags](#device-tags)). Preceded by a `uint16` count field. |

### SERVER_HELLO Payload

| Field | Type | Description |
|-------|------|-------------|
| `protocol_version` | `uint32` | Protocol version the server will use. MUST be `<=` the client's version. |
| `max_message_size` | `uint32` | Maximum payload size (bytes) the server will accept. |
| `capability_flags` | `uint64` | Bitmask of server capabilities. |
| `supported_layouts` | `layout_entry[layout_count]` | Layout entries the server can produce (possibly via transcoding). Preceded by a `uint16` count field. Each entry is encoded as defined in [Layout Entry Encoding](#layout-entry-encoding). |
| `supported_devices` | `uint8[device_count]` | Device tags the server can target. Preceded by a `uint16` count field. |

A client MUST send a `CLIENT_HELLO` as the first message on every new connection.
The server MUST respond with a `SERVER_HELLO` before any other message. If the server
cannot satisfy the minimum requirements (e.g., protocol version mismatch), it MUST
respond with an `ERROR` message instead and close the connection.

### Capability Flags

| Bit | Name | Meaning |
|-----|------|---------|
| 0 | `TRANSCODING` | Sender can transcode tensors to a requested layout on the fly |
| 1 | `PARALLEL_STREAMS` | Sender supports multi-stream parallel shard transfer |
| 2 | `RDMA_DATA_PLANE` | Sender supports RDMA for the data plane |
| 3 | `RDMA_GPUDIRECT` | Sender supports GPUDirect-style writes into a receiver-registered destination memory region. MUST imply `RDMA_DATA_PLANE` (bit 2). |
| 4–63 | (reserved) | MUST be 0 |

### Device Tags

> The canonical definition of device tags lives in `buffer-protocol.md` § Device
> Tags. The table below is reproduced for transport-protocol convenience and
> MUST stay consistent with `buffer-protocol.md`.

| Tag | Device |
|-----|--------|
| `0x00` | CPU host memory |
| `0x01` | CUDA device memory |
| `0x02` | ROCm device memory |
| `0x03` | Metal device memory (Apple Silicon unified memory) |
| `0x04` | Vulkan device memory |
| `0x05` | WebGPU device memory |
| `0x06` | Qualcomm Hexagon (HVX/HMX) memory |
| `0x07` | Intel Level Zero / oneAPI device memory |
| `0x08` | OpenCL device memory |
| `0x09`–`0xEF` | Reserved for future specification versions |
| `0xF0`–`0xFE` | Implementation-private device types |
| `0xFF` | Reserved (invalid) |

---

## Layout Entry Encoding

Layout lists appear in `CLIENT_HELLO`, `SERVER_HELLO`, and `TENSOR_REQUEST`. Each list
is preceded by a `uint16` count, followed by that many **layout entries**. A layout
entry is variable-length and encoded as follows:

1. **`layout_tag`** (`uint8`): the layout tag as defined in `memory-layout.md`.
2. If `layout_tag` is in the **extension range** (`0xF0`–`0xFE`):
   - **`ext_metadata_length`** (`uint16`): byte length of the opaque metadata that
     follows. MAY be `0`.
   - **`ext_metadata`** (`byte sequence`): opaque hardware- or implementation-specific
     metadata of `ext_metadata_length` bytes.
3. If `layout_tag` is **not** in the extension range, the entry consists of the single
   `layout_tag` byte only. No length or metadata fields follow.

A reader MUST skip any extension entry whose `ext_metadata` it does not understand,
using `ext_metadata_length` to advance past it. A reader MUST NOT reject a layout list
solely because it contains unrecognised extension entries.

> **Note (non-normative):** The primary use case for extension layout entries in
> negotiation is hardware-specific panel/pack formats for BLAS kernels. A client
> advertising such a format would include an extension tag with opaque metadata
> encoding its hardware parameters (e.g. panel width, register block dimensions,
> SIMD width, cache line size). The server either recognises the profile and transcodes
> accordingly, or skips the entry and falls back to the next preference. The packed
> buffer travels to the client and is handed directly to the BLAS kernel — it is never
> forwarded or reinterpreted by generic tensor code.

---

## Layout Negotiation

### Request

When sending a `TENSOR_REQUEST`, the client specifies its layout preferences. The
server MUST honor the negotiation rules below.

### TENSOR_REQUEST Payload

| Field | Type | Description |
|-------|------|-------------|
| `tensor_key` | `utf8 string` | Identifier of the requested tensor. Encoded as a `uint32` byte length followed by UTF-8 bytes. |
| `preferred_layouts` | `layout_entry[layout_count]` | Layout entries in preference order (most preferred first). Preceded by a `uint16` count field. `0` count means no preference. Each entry is encoded as defined in [Layout Entry Encoding](#layout-entry-encoding). |
| `preferred_device` | `uint8` | Preferred device tag for the response buffer. |
| `min_alignment` | `uint32` | Minimum buffer alignment (bytes) the client requires. MUST be a power of two. |
| `request_flags` | `uint32` | Bitmask of request flags (see below). |

**Request flags:**

| Bit | Name | Meaning |
|-----|------|---------|
| 0 | `ALLOW_TRANSCODE` | Client permits the server to transcode to a preferred layout |
| 1 | `PARALLEL_OK` | Client supports receiving the tensor as multiple parallel shards |
| 2–31 | (reserved) | MUST be 0 |

### Server Layout Selection

Upon receiving a `TENSOR_REQUEST`, the server MUST select a layout for the response
according to the following rules, in order:

1. If the client supplied a non-empty `preferred_layouts` list, the server MUST iterate
   through the list in order and select the **first** layout tag that satisfies one of:
   a. The tensor is already stored in that layout (no transcoding needed), or
   b. `ALLOW_TRANSCODE` is set and the server has `TRANSCODING` capability for that
      layout tag.
2. If no preferred layout can be satisfied (list exhausted or empty), the server MUST
   serve the tensor in its native stored layout.

The server MUST indicate the chosen layout in the `TENSOR_DESCRIPTOR` message. The
client MUST be prepared to receive any layout that appeared in the server's
`supported_layouts` advertisement, even if it was not among the client's preferences.

### On-the-Fly Transcoding

When the server transcodes a tensor to satisfy a layout preference, the transcoding
MUST be element-preserving: the logical tensor (same rank, same shape, same element
values at every index) MUST be identical before and after transcoding. Only the memory
layout differs.

A server MUST NOT transcode if doing so would require materialising a buffer larger
than the server's configured transcoding memory limit. In that case the server MUST
fall through to the next preferred layout or the native layout.

> **Note (non-normative):** Transcoding is inherently a memory and compute cost on the
> server side. Servers SHOULD document their transcoding capabilities and limits.
> Clients SHOULD list lightweight layouts (e.g. row-major `0x01`) later in their
> preference list as a fallback, rather than requiring the server to transcode into a
> complex layout first.

---

## Device Negotiation

The `preferred_device` field of `TENSOR_REQUEST` drives device selection for the
response buffer. Device selection is distinct from layout negotiation: device
selection has no ordered preference list on the wire (a single tag is supplied),
and the rules below define the server's response when the preferred device cannot
be served.

### Server Device Selection

Upon receiving a `TENSOR_REQUEST`, the server MUST select a placement device for
the response buffer according to the following rules, in order:

1. If `preferred_device` appears in the server's `supported_devices` list (from
   `SERVER_HELLO`) and the server has resources to satisfy it at request time,
   the server MUST place the response buffer on `preferred_device`.
2. If `preferred_device == 0x00` (CPU) and the server cannot satisfy CPU, the
   server MUST send `ERROR` with `error_code = DEVICE_UNAVAILABLE` and close the
   stream. CPU is the universal fallback; if a server cannot satisfy CPU, no
   silent fallback is meaningful.
3. If the preferred device is a non-CPU device the server cannot serve, the
   server MUST send `ERROR` with `error_code = DEVICE_UNAVAILABLE` and close the
   stream. The server MUST NOT silently fall back to a different device.
4. Exception to rule 3: if `preferred_device` was advertised by the server in
   `SERVER_HELLO` but is transiently unavailable (e.g., out of device memory),
   the server MAY fall back to CPU (`0x00`) if and only if the client also
   advertised CPU in its `CLIENT_HELLO supported_devices` list. This is the only
   permitted silent fallback.
5. The server MUST report the actual placement device in the `device_tag` field
   of every buffer handle in `TENSOR_DESCRIPTOR`. Per `buffer-protocol.md`
   § Device Colocation, all buffers of a single tensor MUST share the same
   `device_tag`.
6. A client that receives a `TENSOR_DESCRIPTOR` whose buffer `device_tag`
   differs from its `preferred_device` MUST be prepared to either accept the
   placement (if the tag is in its `supported_devices` list) or close the
   stream with `ERROR`.

> **Note (non-normative):** The "no silent fallback" design is motivated by the
> performance characteristics of inference workloads: a model running on CUDA
> that silently lands on CPU produces correct output but performance collapses
> catastrophically (often by orders of magnitude). The narrow CPU-fallback
> exception in rule 4 is provided only as a graceful degradation path for
> clients that explicitly opt in by advertising CPU in their
> `supported_devices` list.

> **Note (non-normative):** Clients that want graceful CPU fallback should
> advertise CPU (`0x00`) in `CLIENT_HELLO supported_devices`. The
> `DEVICE_UNAVAILABLE` error gives the client enough information to retry with
> a different `preferred_device` if it has alternatives available.

---

## Streaming: Tensor Descriptor and Data Frames

### Ordering Invariant

For every tensor transferred, the server MUST send messages in the following order:

```
TENSOR_DESCRIPTOR
TENSOR_DATA  (one or more frames)
TENSOR_DATA_END
```

A receiver MUST NOT attempt to interpret data frames before receiving the
`TENSOR_DESCRIPTOR`. This invariant holds for each shard in a parallel transfer.

### TENSOR_DESCRIPTOR Payload

The payload is a serialized tensor descriptor as defined in `metadata.md`, followed by
the following transport-specific fields:

| Field | Type | Description |
|-------|------|-------------|
| `total_data_bytes` | `uint64` | Total number of bytes that will follow in `TENSOR_DATA` frames for this tensor (or shard). |
| `shard_index` | `uint32` | Index of this shard in a parallel transfer. `0` for non-parallel transfers. |
| `total_shards` | `uint32` | Total number of shards in a parallel transfer. `1` for non-parallel transfers. |

The `sync_mode` field of each buffer handle inside `TENSOR_DESCRIPTOR` declares
the producer's synchronisation guarantee for that buffer. Consumers MUST observe
the `sync_mode` value and apply the matching rule per `buffer-protocol.md`
§ Stream and Event Synchronisation before accessing the buffer's bytes.

### TENSOR_DATA Payload

| Field | Type | Description |
|-------|------|-------------|
| `byte_offset_in_buffer` | `uint64` | Byte offset within the tensor's data buffer where this frame's bytes begin. |
| `data` | `byte sequence` | Raw tensor data bytes. Length is `payload_length` minus 8 (the `byte_offset_in_buffer` field). |

Data frames for a single tensor MUST be sent in ascending `byte_offset_in_buffer`
order with no gaps and no overlaps. The sum of all data frame lengths MUST equal the
`total_data_bytes` declared in the `TENSOR_DESCRIPTOR`.

### TENSOR_DATA_END Payload

The `TENSOR_DATA_END` message has an empty payload (`payload_length = 0`). It signals
that all data frames for the current tensor (or shard) have been sent on this stream.

---

## Parallel Transfers

### Overview

A tensor MAY be transferred as a set of **shards** delivered simultaneously over
multiple independent streams (e.g., multiple TCP connections or RDMA queue pairs).
Each shard is a rectangular sub-region of the logical tensor, described by the shard
descriptor mechanism defined in `memory-layout.md` (fields `parent_shape` and
`shard_offset`).

The client indicates willingness to receive parallel shards by setting the
`PARALLEL_OK` flag in the `TENSOR_REQUEST`. The server indicates parallel transfer
support via the `PARALLEL_STREAMS` capability flag in `SERVER_HELLO`.

### Parallel Transfer Flow

1. The client sends a single `TENSOR_REQUEST` with `PARALLEL_OK` set.
2. The server selects a sharding strategy (number of shards, shard boundaries) and
   responds on **N** separate streams, one per shard. Each stream carries an
   independent `TENSOR_DESCRIPTOR` → `TENSOR_DATA` → `TENSOR_DATA_END` sequence.
3. Each `TENSOR_DESCRIPTOR` payload MUST include:
   - A tensor descriptor with a shard descriptor (`parent_shape`, `shard_offset`, and
     the shard's own `shape`) embedded as defined in `metadata.md`.
   - `shard_index` and `total_shards` fields in the transport header.
4. The client reassembles the logical tensor by placing each shard at the position
   indicated by its `shard_offset` within a buffer sized for `parent_shape`.

### Shard Consistency

All shards of a single tensor MUST share the same:
- `parent_shape`
- element type
- layout tag (outer layout; the shard's inner layout MAY vary in future extensions)

The union of all shard bounding boxes (defined by `shard_offset` and `shape`) MUST
exactly cover the full `parent_shape` without overlap, satisfying the coverage and
non-overlap constraints from `memory-layout.md`.

A client MUST validate shard consistency upon receiving all `TENSOR_DESCRIPTOR`
messages. A client MUST reject a parallel transfer where any shard descriptor
violates these constraints.

> **Note (non-normative):** Sharding along the batch dimension (dimension 0) is the
> simplest and most common case — each shard is a contiguous slice of rows. The
> protocol does not restrict sharding to any particular dimension or sharding scheme.

---

## RDMA Data Plane

### Overview

When both client and server advertise the `RDMA_DATA_PLANE` capability flag during
session establishment, the data plane for individual tensor transfers MAY use RDMA
rather than TCP `TENSOR_DATA` frames. The control plane (TCP) is still used for all
session management, descriptor exchange, and completion signalling.

> **Note (non-normative):** The RDMA data plane bypasses TCP framing for the tensor
> buffer itself. For GB-scale tensors this eliminates CPU copies and TCP serialisation
> overhead, achieving near-line-rate GPU-to-GPU transfer via GPUDirect RDMA. The
> underlying RDMA operations are performed by an RDMA library such as UCX
> (`ucp_put_nb` / `ucp_get_nb`) or libibverbs (`ibv_post_send`). The Hurray protocol
> specifies the handshake messages; it does not mandate a specific RDMA library.

### Handshake Flow (Server → Client Tensor Transfer)

When both parties have advertised `RDMA_DATA_PLANE`, the server MAY substitute the
`TENSOR_DATA` / `TENSOR_DATA_END` sequence with an RDMA handshake. The client MUST
be prepared to handle either path.

```
Client                          Server
  |                               |
  |--- TENSOR_REQUEST ----------->|
  |<-- TENSOR_DESCRIPTOR ---------|
  |<-- RDMA_REGISTER -------------|  (server registers source buffer, shares rkey + addr)
  |--- RDMA_REGISTER ------------>|  (client registers destination buffer; only if RDMA_GPUDIRECT)
  |--- RDMA_READY --------------->|  (client is ready; RDMA operation may begin)
  |                               |
  |   [RDMA Write executes outside the TCP control plane]
  |                               |
  |<-- TENSOR_DATA_END -----------|  (server signals that buffer is ready on client side)
```

The client's `RDMA_REGISTER` step is OPTIONAL and only occurs when both peers
advertised the `RDMA_GPUDIRECT` capability flag in their respective `HELLO`
messages. See [GPUDirect Destination Registration](#gpudirect-destination-registration).

The server MUST send `TENSOR_DESCRIPTOR` before `RDMA_REGISTER`. The `RDMA_REGISTER`
message MUST be sent on the same stream as the corresponding `TENSOR_DESCRIPTOR`.

### RDMA_REGISTER Payload

The `RDMA_REGISTER` message is sent in one or two roles per transfer:

- The **source-buffer owner** sends `RDMA_REGISTER` to declare its source memory
  region. This is the server for `TENSOR_REQUEST` transfers and the client for
  `TENSOR_PUT` transfers.
- When both peers advertised the `RDMA_GPUDIRECT` capability flag, the
  **destination-buffer owner** MAY also send `RDMA_REGISTER` to declare a
  pre-pinned destination memory region. This is the client for `TENSOR_REQUEST`
  transfers and the server for `TENSOR_PUT` transfers. See
  [GPUDirect Destination Registration](#gpudirect-destination-registration).

The payload table is identical in both roles; the role is determined by the
sender and the position of the message in the handshake sequence.

| Field | Type | Description |
|-------|------|-------------|
| `remote_addr` | `uint64` | Virtual address of the registered memory region on the sender's side. Little-endian. |
| `length` | `uint64` | Size of the memory region in bytes. MUST equal `total_data_bytes` from the preceding `TENSOR_DESCRIPTOR`. Little-endian. |
| `rkey` | `byte sequence` | Opaque RDMA memory key. Encoded as a `uint32` byte-length prefix followed by that many bytes. The format is RDMA-library-specific (e.g., a UCX packed rkey blob, or a 4-byte IB verbs `rkey`). |

A receiver that cannot complete RDMA setup (e.g., memory pinning failed, no RDMA
hardware available on the required path) MUST respond with an `ERROR` message instead
of `RDMA_READY`. The source-buffer owner MUST then fall back to transmitting the
tensor via `TENSOR_DATA` frames on the control plane.

### GPUDirect Destination Registration

When both peers advertised the `RDMA_GPUDIRECT` capability flag in their
respective `HELLO` messages, the destination-buffer owner (the receiver) MAY
pre-register a destination memory region and share its rkey with the sender
via a second `RDMA_REGISTER` message. This enables the sender to write directly
into the receiver's device memory (e.g., GPU memory), eliminating an
intermediate host-to-device copy.

1. The receiver MUST send its `RDMA_REGISTER` (destination) after the sender's
   `RDMA_REGISTER` (source) and before `RDMA_READY`, if it intends to use
   GPUDirect.
2. The `length` field of the destination `RDMA_REGISTER` MUST equal
   `total_data_bytes` from the preceding `TENSOR_DESCRIPTOR`.
3. The receiver MUST register a buffer whose device type is consistent with the
   `device_tag` reported in `TENSOR_DESCRIPTOR`. If the server fell back to a
   different device per [Device Negotiation](#device-negotiation), the receiver
   MUST NOT send a destination `RDMA_REGISTER` for a buffer on the originally
   requested device. The receiver MUST either accept the fallback and proceed
   without GPUDirect, or close the stream with `ERROR`.
4. If the receiver does NOT send `RDMA_REGISTER`, the sender MUST use its own
   buffer management strategy for the RDMA destination — typically landing data
   in the receiver's host memory via the RDMA library's managed staging area.
5. A sender that receives a destination `RDMA_REGISTER` but cannot perform a
   GPUDirect write (e.g., topology mismatch) MUST send `ERROR` instead of
   `RDMA_READY`.

> **Note (non-normative):** GPUDirect requires the NIC and GPU to share a PCIe
> root complex (or to be NVLink-connected). The protocol cannot validate this
> topology; the receiver is responsible for ensuring it before advertising
> `RDMA_GPUDIRECT` and before sending a destination `RDMA_REGISTER`.

### RDMA_READY Payload

The `RDMA_READY` message has an empty payload (`payload_length = 0`). It signals
that the receiver has processed `RDMA_REGISTER` and is ready for the RDMA transfer
to begin.

### Completion

After the RDMA operation completes on the sender's side, the sender MUST send
`TENSOR_DATA_END` over the control plane. The receiver MUST NOT read from the
transferred buffer before receiving `TENSOR_DATA_END`.

For cross-machine transport, the sender MUST set
`sync_mode = SYNC_PRODUCER_SYNCED` (`0x00`) in every buffer handle within the
transmitted `TENSOR_DESCRIPTOR`. `TENSOR_DATA_END` is the cross-machine
equivalent of `SYNC_PRODUCER_SYNCED` per `buffer-protocol.md` § Stream and
Event Synchronisation: the receiver MUST NOT read from the transferred buffer
before receiving `TENSOR_DATA_END`, and upon receipt of `TENSOR_DATA_END` the
buffer is considered producer-synced. The `sync_mode` values `SYNC_EVENT`
(`0x01`) and `SYNC_CONSUMER_STREAM` (`0x02`) are FORBIDDEN on cross-machine
transports because device event and stream handles are not valid in a
different driver context on a different host. A receiver MUST reject a
cross-machine `TENSOR_DESCRIPTOR` whose buffer handle declares any other mode.

> **Note (non-normative):** RDMA Write completion on the sender does not imply that
> the receiver has observed the data without an explicit memory fence or signal.
> `TENSOR_DATA_END` serves as that authoritative "buffer is ready" signal. The sender
> MUST ensure the RDMA operation has completed (e.g., via a completion queue event)
> before sending `TENSOR_DATA_END`.

### TENSOR_PUT with RDMA

For `TENSOR_PUT` (client → server), roles are reversed relative to the
server-to-client flow: the client is the source-buffer owner and the server is
the receiver.

```
Client                          Server
  |                               |
  |--- TENSOR_PUT --------------->|  (tensor key + descriptor)
  |--- RDMA_REGISTER ------------>|  (client registers source buffer, shares rkey + addr)
  |<-- RDMA_REGISTER -------------|  (server registers destination buffer; only if RDMA_GPUDIRECT)
  |<-- RDMA_READY ----------------|  (server is ready; RDMA operation may begin)
  |                               |
  |   [RDMA Write executes outside the TCP control plane]
  |                               |
  |--- TENSOR_DATA_END ---------->|  (client signals data placed)
  |<-- TENSOR_PUT_ACK ------------|  (server confirmed receipt)
```

The server's `RDMA_REGISTER` (destination) step is OPTIONAL and only occurs
when both peers advertised the `RDMA_GPUDIRECT` capability flag in their
respective `HELLO` messages.

In `TENSOR_PUT`, the client unilaterally declares the destination `device_tag`
in the descriptor; the server has no negotiation opportunity (there is no
analogue of `preferred_device` for PUT transfers). If the server cannot accept
the tensor on the declared device, it MUST send `ERROR` with
`error_code = DEVICE_UNAVAILABLE` before any RDMA exchange and close the stream.

The rules in [GPUDirect Destination Registration](#gpudirect-destination-registration)
apply symmetrically with roles inverted: the server is the destination-buffer
owner, and its destination `RDMA_REGISTER` declares a region whose device type
MUST match the `device_tag` declared by the client in the descriptor.

If the client (source-buffer owner) cannot perform a GPUDirect write after
receiving the server's destination `RDMA_REGISTER` (e.g., topology mismatch),
the client MUST send `ERROR` and close the stream before `TENSOR_DATA_END`.

---

## TENSOR_PUT

### Overview

`TENSOR_PUT` allows a client to push a tensor to the server. The protocol defines
the wire exchange only — the server-side storage model (lifetime, eviction, collision
handling) is an implementation concern and is intentionally out of scope.

### TENSOR_PUT Flow

```
Client                          Server
  |--- TENSOR_PUT --------------->|  (tensor key + descriptor)
  |--- TENSOR_DATA (one or more)->|  (data frames)
  |--- TENSOR_DATA_END ---------->|
  |<-- TENSOR_PUT_ACK ------------|  (server confirmed receipt)
```

With the RDMA data plane, `TENSOR_DATA` frames are replaced by the RDMA handshake
described in [TENSOR_PUT with RDMA](#tensor_put-with-rdma).

### TENSOR_PUT Payload

| Field | Type | Description |
|-------|------|-------------|
| `tensor_key` | `utf8 string` | Identifier for the pushed tensor. Encoded as a `uint32` byte length followed by UTF-8 bytes. |
| `descriptor` | `byte sequence` | Serialized tensor descriptor as defined in `metadata.md`. Encoded as a `uint32` byte length followed by the descriptor bytes. |
| `total_data_bytes` | `uint64` | Total number of bytes that will follow in `TENSOR_DATA` frames. |

### TENSOR_PUT_ACK Payload

The `TENSOR_PUT_ACK` message has an empty payload (`payload_length = 0`). It
signals that the server has received and accepted the complete tensor buffer. It
does not imply anything about how the server stores, forwards, or uses the tensor.

If the server cannot accept the tensor for any reason (e.g., policy rejection, resource
exhaustion), it MUST send an `ERROR` message instead of `TENSOR_PUT_ACK` and close
the stream.

> **Note (non-normative):** Server-side storage semantics — including tensor lifetime,
> eviction policy, and key collision handling — are deliberately unspecified. A server
> implementation is free to store the tensor for the session, forward it immediately to
> another peer, or discard it after use. The protocol's role is delivery confirmation,
> not storage coordination.

---

## Error Handling

### ERROR Payload

| Field | Type | Description |
|-------|------|-------------|
| `error_code` | `uint32` | Error code (see below). |
| `message` | `utf8 string` | Human-readable error description. `uint32` length prefix followed by UTF-8 bytes. |

| Code | Name | Meaning |
|------|------|---------|
| `0x00000001` | `PROTOCOL_VERSION_MISMATCH` | Incompatible protocol versions |
| `0x00000002` | `UNKNOWN_TENSOR` | Requested tensor key not found |
| `0x00000003` | `LAYOUT_UNAVAILABLE` | No acceptable layout could be served |
| `0x00000004` | `TRANSCODE_LIMIT_EXCEEDED` | Transcoding refused: buffer too large |
| `0x00000005` | `DEVICE_UNAVAILABLE` | Requested device not available |
| `0x00000006` | `INVALID_MESSAGE` | Malformed message received |
| `0x00000007` | `MESSAGE_TOO_LARGE` | `payload_length` exceeds receiver's limit |
| `0x00000008` | `SHARD_MISMATCH` | Parallel shard descriptors are inconsistent |
| `0x000000F0`–`0x000000FE` | (implementation-defined) | |

Upon sending or receiving an `ERROR` message, both parties MUST close the affected
stream. The connection MAY remain open for other streams.

---

## Open Questions Summary

> **[OQ-1]:** ~~Endianness negotiation: should the transport protocol allow a client to request big-endian wire encoding?~~ **Resolved:** No endianness negotiation. The wire format is always little-endian; big-endian clients MUST byte-swap on receipt. Rationale: all AI/ML inference hardware targeted by Hurray is little-endian; negotiation would add protocol complexity with zero practical benefit. Consistent with Arrow, DLPack, and SafeTensors.

> **[OQ-2]:** ~~RDMA data plane handshake.~~ **Resolved:** `RDMA_REGISTER` (`0x0000000C`) and `RDMA_READY` (`0x0000000D`) message types are now defined. The party owning the source buffer registers its memory region and sends `RDMA_REGISTER` (rkey + remote address + length) over the control plane; the peer responds with `RDMA_READY`; the RDMA operation executes out-of-band; `TENSOR_DATA_END` is sent over the control plane as the authoritative completion signal. See [RDMA Data Plane](#rdma-data-plane).

> **[OQ-3]:** ~~Multiplexing scheme.~~ **Resolved:** The `stream_id` field is defined as an opaque per-stream identifier. Implementations MAY multiplex multiple streams over a single TCP connection using `stream_id` for demultiplexing, but the protocol does not mandate a normative multiplexing scheme. Each stream MAY equivalently run on its own connection. Normative multiplexing rules are deferred to a future revision once the format is stable.

> **[OQ-4]:** ~~`TENSOR_PUT` semantics.~~ **Resolved:** Server-side storage model is explicitly out of scope. `TENSOR_PUT_ACK` means "received and accepted"; it carries no implication about persistence, lifetime, or collision handling. Those are implementation concerns. The server sends `ERROR` to reject a PUT for any reason. See [TENSOR_PUT](#tensor_put).

> **[OQ-5]:** ~~Device negotiation and GPUDirect RDMA destination registration.~~ **Resolved:** A normative [Device Negotiation](#device-negotiation) section defines server device selection rules. GPUDirect destination registration uses bidirectional `RDMA_REGISTER` gated by the new `RDMA_GPUDIRECT` capability flag (bit 3). See `docs/adr/ADR-011-server-device-selection.md` (server device selection) and `docs/adr/ADR-012-gpudirect-rdma.md` (GPUDirect RDMA).

---

## Interaction with Other Sections

- **Memory Layout (`memory-layout.md`)**: defines the layout tag space used in
  capability advertisement and layout negotiation. Shard descriptors (`parent_shape`,
  `shard_offset`) are the logical basis for parallel transfers.
- **Metadata (`metadata.md`)**: defines the binary encoding of the tensor descriptor
  transmitted in `TENSOR_DESCRIPTOR` messages.
- **Buffer Protocol (`buffer-protocol.md`)**: defines buffer alignment requirements
  that the client expresses via `min_alignment` in `TENSOR_REQUEST`, and device memory
  semantics relevant to the `preferred_device` field.
- **Element Types (`element-types.md`)**: defines the little-endian wire encoding of
  tensor element data, which is the data transmitted in `TENSOR_DATA` frames.
