# ADR-018: GPU Stream and Event Synchronisation at Buffer Handoff

## Status

Proposed (revised)

## Context

The buffer protocol (`docs/spec/buffer-protocol.md`) defines how buffer handles carry size, alignment, and device tag, and how ownership transfer is signalled via a release callback (ADR-009). It is silent on **GPU stream and event synchronisation**: when a producer hands off a CUDA/ROCm/Metal/Vulkan/Level-Zero/OpenCL/WebGPU buffer to a consumer, nothing in the spec guarantees that the producer's previously-enqueued device writes are visible when the consumer dereferences the buffer.

The interchange protocol (`docs/spec/interchange.md`) inherits the same gap. `TENSOR_DATA_END` is described as the authoritative "buffer is ready" signal for the RDMA path (the receiver "MUST NOT read from the transferred buffer before receiving `TENSOR_DATA_END`"), but for in-process and IPC handoff there is no such signal; for the RDMA GPUDirect path, the spec relies on the sender's RDMA completion queue to ensure the write retired before `TENSOR_DATA_END` is sent — which gives a producer-side fence but does not establish a consumer-side device-stream ordering relationship.

Concretely, three failure modes exist today:

1. **In-process CUDA handoff.** Producer issues a kernel on stream A that writes the buffer, then hands the buffer to a consumer. Consumer enqueues a kernel on stream B that reads the buffer. Streams A and B are not ordered. The consumer's kernel may execute before the producer's kernel retires. Data race.
2. **IPC GPU handoff.** Same as above, across processes, with a CUDA IPC handle. The handle carries no ordering information.
3. **Cross-machine GPUDirect.** Sender's NIC has DMA-written into the receiver's GPU memory. Receiver's first read on a GPU stream may not observe the write without a device-side fence, depending on the RDMA provider and the GPU vendor's PCIe ordering guarantees.

DLPack issue #176 is the closest prior-art discussion. DLPack currently treats producer-side stream synchronisation as advisory; the issue debates four options (producer-MUST-sync, pass producer stream handle, pass event handle, sync-version field). The Python array API has converged on passing a consumer stream handle to `__dlpack__(stream=...)` so the producer can record an event on its own stream and make the consumer stream wait — pushing the cost onto the producer but localising it to a single stream-wait rather than a full stream sync.

The choice has direct implications for Hurray:

- The buffer handle is a 16-byte fixed binary record (`buffer-protocol.md` § Buffer Handle). Three bytes at offsets 13–15 are currently reserved.
- Whatever rule is chosen MUST be implementable by a non-Rust reader from the spec alone (interoperability invariant).
- The rule MUST NOT close the door on additional device types (`device_tag` 0x09–0xEF reserved) whose synchronisation primitives may not yet exist or may differ structurally from CUDA events.
- The rule MUST integrate cleanly with ADR-009: the release callback signals end-of-use, not start-of-use; the new synchronisation rule occupies the symmetric position at the start of consumer access.

Open questions resolved by this ADR:

- **[OQ-A]** Producer-side requirement: nothing, sync stream, or record event? → Three tiers, producer's choice.
- **[OQ-B]** Consumer-side requirement: nothing, sync stream, or wait on event? → Depends on producer's chosen tier, declared in the buffer handle.
- **[OQ-C]** Wire-format change: extend buffer handle, extend interchange message, or neither? → Discriminant in the binary buffer handle (offset 13); handle in the C ABI out-of-band.
- **[OQ-D]** Per-transport variation: should in-process, IPC, and cross-machine differ? → Yes; see § Per-transport rules.

## Decision

Hurray defines a normative **handoff-completed-before-handoff-observed** rule, stated as a behavioural contract on the producer. The buffer handle binary layout is extended by one byte to carry a `sync_mode` discriminant (consuming one of the three existing reserved bytes); the synchronisation primitive itself (CUDA stream, CUDA event, `hipEvent_t`, `MTLSharedEvent`, `VkSemaphore`, etc.) is **not** carried in the binary descriptor and is exchanged out of band via the C ABI for in-process and IPC, and is subsumed by `TENSOR_DATA_END` for cross-machine.

The rule has three tiers, one per producer mechanism, and per-transport rules govern which tiers are valid on which transport.

### 1. Producer requirement (all transports)

A producer of a non-CPU buffer MUST ensure that, at the instant ownership of the buffer is transferred to the consumer, **all device-side writes enqueued by the producer that affect the buffer's bytes have reached a point at which a properly-synchronised consumer access on the same device will observe them**. The producer satisfies this requirement by exactly one of the following mechanisms; the choice is the producer's, and the choice MUST be declared in the buffer handle's `sync_mode` field (see §3):

- **(P1) `SYNC_PRODUCER_SYNCED = 0x00`.** Producer issues a host-side wait on the device stream(s) that wrote the buffer, ensuring all preceding work has completed before handoff. Strongest guarantee, highest cost. This is the universal default and the only mode valid on cross-machine transports.
- **(P2) `SYNC_EVENT = 0x01`.** Producer records a device event on the stream(s) that wrote the buffer and provides the event handle to the consumer through the C ABI. The producer-side stream is not blocked. Valid in-process; valid for IPC only when the device supports IPC-exportable events.
- **(P3) `SYNC_CONSUMER_STREAM = 0x02`.** When the consumer has advertised its target stream at handoff time via the C ABI, the producer issues a device-side wait that orders the consumer's stream after the producer's writing stream(s). The producer-side stream is not blocked; no event handle crosses the boundary.

Values `0x03`–`0xFE` are reserved for future specification versions. Value `0xFF` is reserved (invalid). A reader MUST reject a buffer handle whose `sync_mode` is not one of the values defined for the format version it implements.

For CPU buffers (`device_tag = 0x00`) `sync_mode` MUST be `SYNC_PRODUCER_SYNCED` (`0x00`); the producer MUST still ensure that any concurrent host-side writes are sequenced before handoff using a host memory fence (a release-store or equivalent).

### 2. Consumer requirement (all transports)

A consumer that has received a non-CPU buffer MUST inspect the buffer handle's `sync_mode` field and apply the matching rule:

- If `sync_mode == SYNC_PRODUCER_SYNCED`, the consumer MAY access the buffer immediately on any stream — the producer's host-side wait (or, for cross-machine, the producer-side fence preceding `TENSOR_DATA_END`) established a host-program-order point.
- If `sync_mode == SYNC_EVENT`, the consumer MUST retrieve the producer's event handle from the C ABI handoff structure and MUST issue a device-stream-wait on it on every stream that will access the buffer, before enqueuing any work that touches it. The consumer MUST release the event handle exactly once via the event-release callback defined in §4.
- If `sync_mode == SYNC_CONSUMER_STREAM`, the consumer MAY access the buffer on the stream(s) it declared at handoff, but MUST NOT access the buffer on any other stream until it has issued an inter-stream wait.

Because exactly one mode is declared per buffer handle, a consumer MUST NOT need to negotiate or guess the producer's chosen mechanism. A consumer that does not recognise the declared `sync_mode` value MUST reject the descriptor.

### 3. Wire format: `sync_mode` added to the buffer handle

The buffer handle binary layout in `buffer-protocol.md` § Buffer Handle is updated by reallocating one byte of the existing `_reserved` field to a `sync_mode` discriminant:

| Offset | Field | Type | Description |
|--------|-------|------|-------------|
| 0 | `byte_size` | `uint64` | Size of the buffer in bytes (little-endian). `0` denotes an empty buffer. |
| 8 | `alignment` | `uint32` | Minimum alignment of the buffer's base address in bytes (little-endian). |
| 12 | `device_tag` | `uint8` | Device where this buffer resides. |
| 13 | `sync_mode` | `uint8` | Producer-side synchronisation mechanism in effect. See §1. |
| 14 | `_reserved` | `uint8[2]` | MUST be `0x00`. Readers MUST reject a descriptor with non-zero reserved bytes. |

Total size: 16 bytes. All multi-byte fields MUST be encoded in little-endian byte order.

The opaque event handle (for `SYNC_EVENT`) and the consumer stream handle (for `SYNC_CONSUMER_STREAM`) are **not** carried in the binary descriptor. They are exchanged out of band via the C ABI (see §4) because event and stream handles are pointers / opaque IDs valid only in the producer's driver context and have a transient lifetime that does not match the descriptor's. The `sync_mode` field is a *declaration* of a buffer's synchronisation properties; the synchronisation handle is a transport detail, not a buffer property.

> **Note (non-normative):** `hurray-inspect` and any other static inspector can surface `sync_mode` from the descriptor alone, without reaching into the C ABI layer.

No synchronisation field is added to `TENSOR_DESCRIPTOR`, `RDMA_REGISTER`, `RDMA_READY`, or `TENSOR_DATA_END`. The buffer handle inside `TENSOR_DESCRIPTOR` already carries `sync_mode`, so the interchange protocol inherits it for free.

### 4. C ABI

The C ABI handoff structure carries the *handle* corresponding to the `sync_mode` declared in the buffer handle. `docs/impl/c-ffi.md` MUST define:

- For `SYNC_PRODUCER_SYNCED`: no additional fields. The C ABI handoff carries only the buffer pointer and release callback.
- For `SYNC_EVENT`: an opaque `sync_handle` pointer (device-vendor-specific event), a `sync_handle_device_tag` (which MUST equal the buffer's `device_tag`), and an `event_release_fn` callback the consumer MUST call exactly once after it has issued its stream-wait. The lifetime model mirrors ADR-009: a single normative release, thread-safe, with internal refcounting at the producer's discretion.
- For `SYNC_CONSUMER_STREAM`: at handoff request time, the consumer supplies an opaque `consumer_stream` handle of type matching the buffer's `device_tag`; the producer issues the stream wait before returning the buffer handle.

The C ABI MUST validate that the handle provided at handoff time matches the `sync_mode` declared in the buffer handle. A mismatch is a producer-side bug that the ABI layer MUST reject before returning the handle to the consumer.

The C ABI no longer carries a `sync_mode` discriminant of its own — the descriptor is the single source of truth; the ABI provides only the payload for whichever mode is declared.

The exact C struct layout is the C ABI layer's concern; this ADR fixes only the abstract contract and the binary-descriptor field.

### 5. Relationship to the release callback (ADR-009)

The release callback and the synchronisation contract are **independent and symmetric**:

- The synchronisation contract governs the **start** of consumer access (when is it safe to read?).
- The release callback governs the **end** of consumer access (when does the producer learn the consumer is done?).

A consumer that has issued device work using the buffer MUST NOT call the release callback until that device work has completed on the device. The consumer is free to satisfy this by host-side waiting, by recording its own completion event and waiting on it, or by deferring the release callback to a completion callback registered on its stream. This is symmetric to the producer's obligation in §1 and is an existing implication of ADR-009's "consumer MUST NOT access the buffer after calling the release callback" rule; this ADR makes the implication explicit.

The event-release callback (for `SYNC_EVENT` mode) is **separate from the buffer release callback**. A consumer therefore makes two release calls in `SYNC_EVENT` mode: one for the event handle (called after the consumer has issued its stream-wait — typically immediately after handoff) and one for the buffer (called after all device work on the buffer is complete). Conflating the two would force the producer to keep the event alive for the buffer's entire lifetime, which defeats the purpose of using events instead of full stream sync.

### 6. Per-transport rules

#### In-process

All three `sync_mode` values are valid. Event handles in `SYNC_EVENT` are valid because both parties share the same device context. Stream handles in `SYNC_CONSUMER_STREAM` are valid for the same reason.

#### IPC (same machine, different processes)

- `SYNC_PRODUCER_SYNCED` (P1) is always valid.
- `SYNC_EVENT` (P2) is valid only if the device supports IPC-exportable events (CUDA `cudaIpcEventHandle_t`, ROCm `hipIpcEventHandle_t`). The C ABI MUST expose a query that tells a binding which `sync_mode` values are available for a given `device_tag` and IPC channel.
- `SYNC_CONSUMER_STREAM` (P3) is valid only if the device supports IPC-exportable streams (not universally available).

A producer that cannot offer any valid mode for an IPC GPU buffer handoff MUST fall back to a host-staged copy. This degradation is not negotiated on the wire; it is the producer's local decision.

#### Cross-machine (network transport)

A producer MUST set `sync_mode = SYNC_PRODUCER_SYNCED` for every buffer handle transmitted over a network transport. `SYNC_EVENT` and `SYNC_CONSUMER_STREAM` are **forbidden across machines** because device event and stream handles are not valid in a different driver context on a different host. A receiver MUST reject a cross-machine `TENSOR_DESCRIPTOR` whose buffer handle declares any other mode.

The existing `TENSOR_DATA_END` message is the authoritative producer-synced signal:

- For the non-RDMA path (`TENSOR_DATA` frames), the sender MUST ensure all device-side writes to the source buffer are visible to host reads before sending the first `TENSOR_DATA` frame whose `byte_offset_in_buffer` covers those bytes.
- For the RDMA path **without** GPUDirect, the sender MUST ensure source-buffer writes have retired on the device before posting the RDMA send; the receiver MUST NOT read the buffer before `TENSOR_DATA_END`.
- For the RDMA path **with** GPUDirect (ADR-012), the sender MUST ensure: (a) producer-side device writes to the source buffer have retired before posting the RDMA send, and (b) the RDMA completion has been observed via the sender's completion queue before sending `TENSOR_DATA_END`. The receiver, upon receiving `TENSOR_DATA_END`, MAY assume the destination GPU buffer is visible to subsequent device-stream reads on the receiver's GPU, provided the receiver is using a CUDA/ROCm/etc. driver version that establishes PCIe-write-ordering between an inbound NIC DMA and a subsequent kernel launch on the same device — the spec MUST state this as a receiver-side responsibility, not a producer-side one, because it depends entirely on the receiver's hardware topology.

`TENSOR_DATA_END` is the cross-machine equivalent of `SYNC_PRODUCER_SYNCED`.

### 7. Versioning

Because Hurray has not yet shipped a 1.0 wire format and has no external users, the buffer handle layout change lands in the still-draft 1.0 spec without a version bump; no migration path is required.

Forward extensibility:

- New `sync_mode` values MAY be added in future format revisions by allocating from the reserved range `0x03`–`0xFE`. Adding a value is a wire-incompatible change (a v1 reader that does not understand the new value MUST reject the descriptor) and therefore requires a format version bump.
- A future device type added in `device_tag` `0x09`–`0xEF` whose synchronisation model does not fit any of P1/P2/P3 MUST motivate either a new `sync_mode` value or a separate extension ADR. P1 remains the universal fallback because every device supports host-side wait.

The C ABI version MUST be bumped to expose the `sync_handle` / `event_release_fn` / `consumer_stream` payloads defined in §4.

## Alternatives Considered

### Alternative A — Normative "producer MUST sync" rule, no negotiation

Mandate `SYNC_PRODUCER_SYNCED` always; the producer MUST host-wait on its device stream before handoff. Simplest spec, simplest consumer.

- **Pros**: zero ABI surface, zero negotiation, trivially correct, matches what most CPU code expects of GPU buffers.
- **Cons**: full stream synchronisation is the single most expensive thing one can do on a CUDA stream; it serialises the entire inference pipeline. For the inference workloads Hurray targets, this collapses pipeline parallelism.
- **Rejected because**: the cost is unacceptable for the primary use case. Hurray's value proposition is zero-copy zero-stall handoff; mandating a full stream sync defeats both halves.

### Alternative B — Carry both the discriminant AND a sync-object handle in the buffer handle binary layout

Burn 8 bytes of `_reserved` (or extend the buffer handle from 16 to 24 bytes) for an opaque sync-object handle, in addition to the discriminant.

- **Pros**: fully self-describing on the wire; no out-of-band C ABI handle needed for in-process use.
- **Cons**: (1) event and stream handles are not portable across processes (need IPC export) or machines (meaningless); the same field would carry incompatible payloads or be null across transports. (2) Bakes a pointer-shaped value into a format that is otherwise pointer-free, breaking the rule that the binary descriptor is a declaration of properties, not a transport detail. (3) The handle has a lifetime that does not match the descriptor's — the descriptor may be inspected long after the event has been released, leaving a dangling pointer encoded in the format. (4) Closes the door on future devices whose synchronisation primitives do not fit a single 8-byte handle (e.g., a 16-byte UUID, or a handle + context pair).
- **Rejected because**: handles are transport-layer state, not format-layer state. The discriminant belongs in the format; the handle does not. See Alternative F for the chosen hybrid.

### Alternative C — Add a new `BUFFER_SYNC` interchange message

Define a new `0x0000000E BUFFER_SYNC` message carrying a sync-object payload, sent after `TENSOR_DESCRIPTOR` and before `TENSOR_DATA_END`.

- **Pros**: clean separation; opt-in via capability flag; does not pollute the buffer handle.
- **Cons**: meaningless for in-process and IPC (which do not use the wire framing for handoff at all) and meaningless for cross-machine (where event handles do not cross host boundaries).
- **Rejected because**: solves a wire-format problem where the problem is not wire-format.

### Alternative D — Defer entirely, leave to bindings

Make no normative statement; each binding defines its own rule.

- **Pros**: zero spec surface.
- **Cons**: directly violates the language-agnostic interoperability invariant. Two implementations both claiming Hurray 1.0 compliance could fail to share GPU buffers safely.
- **Rejected because**: synchronisation is the single most common source of silent data corruption in GPU code. Leaving it implementation-defined is a guarantee of interoperability failures.

### Alternative E — DLPack-style: consumer passes a stream at request time, producer waits on it

Adopt only `SYNC_CONSUMER_STREAM` (P3); forbid P1 and P2.

- **Pros**: matches the Python array API; pushes the wait onto the producer's device side (cheap) rather than the host side (expensive); no event handle lifetime to manage.
- **Cons**: forces every consumer to have a designated stream before requesting a buffer. Some consumers (CLI inspectors, disk writers) have no GPU stream; they would have to construct one purely to satisfy the protocol. Also fails the IPC case where stream handles are not shareable across drivers without IPC-stream support.
- **Rejected because**: too restrictive. P3 is offered as one option among three; mandating it loses the host-side use case and the cross-driver IPC use case.

### Alternative F (chosen) — Hybrid: `sync_mode` discriminant in the binary descriptor, sync handle in the C ABI

Add a single `uint8 sync_mode` field at offset 13 of the buffer handle (consuming one of the three existing reserved bytes, leaving two), and keep the actual sync object handle (event pointer, stream pointer) in the C ABI handoff structure.

- **Pros**:
  - **Self-describing at the binary level**: a static inspector (`hurray-inspect`) can show the synchronisation contract for any buffer without reaching into the C ABI.
  - **Zero byte cost**: uses an already-reserved byte; the buffer handle stays 16 bytes; CPU buffers naturally encode `sync_mode = SYNC_PRODUCER_SYNCED = 0x00`, which is the natural zero value.
  - **No device-specific primitives baked into the format**: the field carries a discriminant, not a vendor-specific handle. Future devices can allocate a new `sync_mode` value without restructuring the descriptor.
  - **Transports stay clean**: across machines, the discriminant is `SYNC_PRODUCER_SYNCED` and no handle is needed; in-process and IPC carry the handle out of band in the C ABI.
  - **C ABI simplifies**: the discriminant moves from the C ABI handoff struct into the descriptor, leaving the C ABI to carry only the handle payload matching the declared mode.
  - **Forward-compatible**: two reserved bytes remain at offsets 14–15 for future use.
- **Cons**:
  - Wire-incompatible with any earlier draft that assumed offset 13 was reserved. Acceptable because Hurray 1.0 has not shipped and has no external users.
  - The discriminant and the handle live in two different places; the C ABI layer MUST cross-check them at handoff time.
- **Accepted because**: it captures the discoverability benefit of Alternative B without inheriting its cost (a non-portable pointer baked into the format). The user's explicit removal of backward-compatibility pressure is what made this option reachable.

## Consequences

### Positive

- The buffer handle gains a `sync_mode` discriminant in a byte previously reserved; the handle stays 16 bytes; CPU-only workloads naturally encode `SYNC_PRODUCER_SYNCED = 0x00` and are unaffected.
- The producer-consumer synchronisation contract is normative, closing a silent-data-corruption gap before 1.0 freeze.
- All three sync mechanisms are available; producers and consumers can choose the cheapest valid option for their workload.
- Cross-machine GPUDirect inherits `TENSOR_DATA_END` as the producer-synced signal — no new message types, no new fields, full alignment with ADR-012.
- Event-release callback is symmetric with the buffer-release callback (ADR-009), so the C ABI gains one mechanism shaped like one the bindings already implement.
- `hurray-inspect` can surface `sync_mode` without reaching into the C ABI.

### Negative

- The buffer handle layout is updated in-draft (offset 13 moves from `_reserved` to `sync_mode`). Must land before 1.0 freeze; readers MUST validate `sync_mode`.
- The C ABI surface grows: per-mode payloads and an event-release callback are added. The C ABI version MUST be bumped.
- Bindings MUST implement three code paths for the three sync modes.
- IPC GPU handoff requires producers to know whether the target device supports exportable events.
- Two-callback handoff (event-release + buffer-release) is more error-prone than one; binding authors MUST be guided clearly.

### Risks

- A producer using `SYNC_EVENT` that fails to record the event before handoff could cause a deadlock. Mitigation: producers MUST record the event before calling the handoff function; the C ABI contract MUST state this.
- A future device type (≥ 0x09) may have a synchronisation model that fits none of P1/P2/P3. Mitigation: P1 is universal; a fourth `sync_mode` can be added in a later ABI version.
- The receiver-side PCIe ordering assertion for GPUDirect is hardware-dependent. Must be stated as a receiver responsibility and documented in `c-ffi.md` or a hardware-compatibility note.

### Compatibility impact

- **Wire format**: buffer handle layout updated in-draft. No backward-compatibility hazard — Hurray 1.0 has not shipped.
- **C ABI**: requires a version bump. The default `sync_mode` for legacy clients MUST be `SYNC_PRODUCER_SYNCED`, preserving safe-but-slow behaviour for pre-ADR consumers.
- **Forward compatibility**: a future ADR MAY add `sync_mode` values without affecting existing values.

## Handoff

- `format-spec-writer`: update `buffer-protocol.md` § Buffer Handle table — split offset 13 from `_reserved` to `sync_mode` (uint8); shrink `_reserved` to `uint8[2]` at offsets 14–15. Add a new normative subsection **§ Stream and Event Synchronisation** covering §1, §2, §5, §6 of this decision, including the `sync_mode` enumeration (`SYNC_PRODUCER_SYNCED = 0x00`, `SYNC_EVENT = 0x01`, `SYNC_CONSUMER_STREAM = 0x02`, reserved `0x03`–`0xFE`, invalid `0xFF`). Add a cross-reference in `interchange.md` § RDMA Data Plane and § Streaming: Tensor Descriptor and Data Frames stating that cross-machine senders MUST set `sync_mode = SYNC_PRODUCER_SYNCED` and that `TENSOR_DATA_END` is the cross-machine producer-synced signal.
- `format-spec-writer`: update `docs/impl/c-ffi.md` to define the per-mode handoff payloads (`sync_handle` + `event_release_fn` for `SYNC_EVENT`; `consumer_stream` for `SYNC_CONSUMER_STREAM`) and the ABI-side cross-check against the descriptor's `sync_mode`. The C ABI no longer carries a `sync_mode` discriminant of its own. Bump the documented C ABI version.
- `rust-developer`: `hurray-core`'s `BufferHandle` gains a `sync_mode: SyncMode` field at byte offset 13. Add a `SyncMode` enum with three named values plus a `Reserved(u8)` carrier for forward-compat rejection (decoders reject reserved values; producers cannot construct them). The `hurray-inspect` refactor onto `hurray-core` (already scheduled for Layer 4) MUST surface `sync_mode` in the per-buffer view.
- `architect`: revisit if a device type added in the reserved range cannot express its synchronisation model via P1/P2/P3; that would prompt an extension ADR rather than amending this one.
- `spec-checker`: audit the new buffer-protocol subsection and the interchange cross-references for RFC 2119 correctness and consistency with ADR-009 (release callback) and ADR-012 (GPUDirect RDMA) once the spec edits land. Verify that `sync_mode = SYNC_PRODUCER_SYNCED` is stated normatively as the constraint for both CPU buffers and cross-machine transports.

## Open Questions Deferred

- Per-direction GPUDirect sync semantics for asymmetric NIC-GPU topologies (deferred with ADR-012's per-direction `RDMA_GPUDIRECT` flag).
- Whether `SYNC_CONSUMER_STREAM` can be carried over IPC for drivers that support stream IPC export (CUDA's `cuStreamGetCtx` is not exportable; ROCm has experimental support). Resolution depends on driver capability evolution; not 1.0 blocking.

## Date

2026-05-12
