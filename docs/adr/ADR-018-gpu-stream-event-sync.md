# ADR-018: GPU Stream and Event Synchronisation at Buffer Handoff

## Status

Proposed

## Context

The buffer protocol (`docs/spec/buffer-protocol.md`) defines how buffer handles carry size, alignment, and device tag, and how ownership transfer is signalled via a release callback (ADR-009). It is silent on **GPU stream and event synchronisation**: when a producer hands off a CUDA/ROCm/Metal/Vulkan/Level-Zero/OpenCL/WebGPU buffer to a consumer, nothing in the spec guarantees that the producer's previously-enqueued device writes are visible when the consumer dereferences the buffer.

The interchange protocol (`docs/spec/interchange.md`) inherits the same gap. `TENSOR_DATA_END` is described as the authoritative "buffer is ready" signal for the RDMA path (the receiver "MUST NOT read from the transferred buffer before receiving `TENSOR_DATA_END`"), but for in-process and IPC handoff there is no such signal; for the RDMA GPUDirect path, the spec relies on the sender's RDMA completion queue to ensure the write retired before `TENSOR_DATA_END` is sent — which gives a producer-side fence but does not establish a consumer-side device-stream ordering relationship.

Concretely, three failure modes exist today:

1. **In-process CUDA handoff.** Producer issues a kernel on stream A that writes the buffer, then hands the buffer to a consumer. Consumer enqueues a kernel on stream B that reads the buffer. Streams A and B are not ordered. The consumer's kernel may execute before the producer's kernel retires. Data race.
2. **IPC GPU handoff.** Same as above, across processes, with a CUDA IPC handle. The handle carries no ordering information.
3. **Cross-machine GPUDirect.** Sender's NIC has DMA-written into the receiver's GPU memory. Receiver's first read on a GPU stream may not observe the write without a device-side fence, depending on the RDMA provider and the GPU vendor's PCIe ordering guarantees.

DLPack issue #176 is the closest prior-art discussion. DLPack currently treats producer-side stream synchronisation as advisory; the issue debates four options (producer-MUST-sync, pass producer stream handle, pass event handle, sync-version field). The Python array API has converged on passing a consumer stream handle to `__dlpack__(stream=...)` so the producer can record an event on its own stream and make the consumer stream wait — pushing the cost onto the producer but localising it to a single stream-wait rather than a full stream sync.

The choice has direct implications for Hurray:

- The buffer handle is a 16-byte fixed binary record (`buffer-protocol.md` § Buffer Handle). Adding a "sync object" field there would burn part of the reserved space and bake a synchronisation primitive into the on-wire format.
- Whatever rule is chosen MUST be implementable by a non-Rust reader from the spec alone (interoperability invariant).
- The rule MUST NOT close the door on additional device types (`device_tag` 0x09–0xEF reserved) whose synchronisation primitives may not yet exist or may differ structurally from CUDA events.
- The rule MUST integrate cleanly with ADR-009: the release callback signals end-of-use, not start-of-use; the new synchronisation rule occupies the symmetric position at the start of consumer access.

Open questions resolved by this ADR:

- **[OQ-A]** Producer-side requirement: nothing, sync stream, or record event? → Three tiers, producer's choice.
- **[OQ-B]** Consumer-side requirement: nothing, sync stream, or wait on event? → Depends on producer's chosen tier.
- **[OQ-C]** Wire-format change: extend buffer handle, extend interchange message, or neither? → Neither; contract lives in the C ABI.
- **[OQ-D]** Per-transport variation: should in-process, IPC, and cross-machine differ? → Yes; see § Per-transport rules.

## Decision

Hurray defines a normative **handoff-completed-before-handoff-observed** rule, stated as a behavioural contract on the producer with no wire-format change. The rule has three tiers, one per transport mode. The synchronisation primitive itself (CUDA stream, CUDA event, ROCm `hipEvent_t`, Metal `MTLSharedEvent`, Vulkan `VkSemaphore`, etc.) is **not carried in the binary descriptor** and **not carried in any interchange wire message**. It is exchanged out of band via the C ABI for in-process / IPC, and is subsumed by the existing `TENSOR_DATA_END` completion signal for cross-machine.

### 1. Producer requirement (all transports)

A producer of a non-CPU buffer MUST ensure that, at the instant ownership of the buffer is transferred to the consumer, **all device-side writes enqueued by the producer that affect the buffer's bytes have reached a point at which a properly-synchronised consumer access on the same device will observe them**. The producer satisfies this requirement by any of the following mechanisms; the choice is the producer's:

- **(P1) Full stream synchronisation.** Producer issues a host-side wait on the device stream(s) that wrote the buffer, ensuring all preceding work has completed before handoff. Strongest guarantee, highest cost.
- **(P2) Event record + handoff via C ABI.** Producer records a device event on the stream(s) that wrote the buffer and provides the event handle to the consumer through the C ABI extension defined below. The producer-side stream is not blocked.
- **(P3) Receiver-stream wait.** When the consumer has advertised its target stream at handoff time (see § Consumer requirement), the producer issues a device-side wait that orders the consumer's stream after the producer's writing stream(s). The producer-side stream is not blocked; no event handle crosses the boundary.

For CPU buffers (`device_tag = 0x00`) no device synchronisation is required; the producer MUST still ensure that any concurrent host-side writes are sequenced before handoff using a host memory fence (a release-store or equivalent).

### 2. Consumer requirement (all transports)

A consumer that has received a non-CPU buffer MUST NOT issue a device read or write that depends on the buffer's contents until it has observed one of the producer's synchronisation acts described in §1. Specifically:

- If the producer used **(P1)**, the consumer MAY access the buffer immediately on any stream — the producer's host-side wait established a host-program-order point.
- If the producer used **(P2)**, the consumer MUST issue a device-stream-wait on the producer-supplied event on every stream that will access the buffer, before enqueuing any work that touches it. The consumer MUST release the event handle exactly once via the event-release callback defined below.
- If the producer used **(P3)**, the consumer MAY access the buffer on the stream(s) it declared at handoff, but MUST NOT access the buffer on any other stream until it has issued an inter-stream wait.

A consumer that does not know which mechanism the producer used MUST behave as if the producer used the weakest mechanism supplied by the C ABI handoff structure. The C ABI handoff (see §3) MUST make this unambiguous: exactly one of (P1, P2, P3) is in effect per handoff.

### 3. Wire format: no change

The buffer handle binary layout in `buffer-protocol.md` § Buffer Handle is unchanged. The 3-byte `_reserved` field stays reserved and zero. No synchronisation field is added to `TENSOR_DESCRIPTOR`, `RDMA_REGISTER`, `RDMA_READY`, or `TENSOR_DATA_END`.

The synchronisation contract lives in the **C ABI**, not the wire format. `docs/impl/c-ffi.md` will define an extension to the buffer handoff structure:

- A `sync_mode` enum: `SYNC_PRODUCER_SYNCED` (P1), `SYNC_EVENT` (P2), `SYNC_CONSUMER_STREAM` (P3).
- For `SYNC_EVENT`: an opaque `sync_handle` pointer (device-vendor-specific event), a `sync_handle_device_tag` (which MUST equal the buffer's `device_tag`), and an `event_release_fn` callback the consumer MUST call exactly once after it has issued its stream-wait. The lifetime model mirrors ADR-009: a single normative release, thread-safe, with internal refcounting at the producer's discretion.
- For `SYNC_CONSUMER_STREAM`: at handoff request time, the consumer supplies an opaque `consumer_stream` handle of type matching the buffer's `device_tag`; the producer issues the stream wait before returning the handle.

The exact C struct layout is the C ABI layer's concern; this ADR only fixes the abstract contract.

### 4. Relationship to the release callback (ADR-009)

The release callback and the synchronisation contract are **independent and symmetric**:

- The synchronisation contract governs the **start** of consumer access (when is it safe to read?).
- The release callback governs the **end** of consumer access (when does the producer learn the consumer is done?).

A consumer that has issued device work using the buffer MUST NOT call the release callback until that device work has completed on the device. The consumer is free to satisfy this by host-side waiting, by recording its own completion event and waiting on it, or by deferring the release callback to a completion callback registered on its stream. This is symmetric to the producer's obligation in §1 and is an existing implication of ADR-009's "consumer MUST NOT access the buffer after calling the release callback" rule; this ADR makes the implication explicit.

The event-release callback (for `SYNC_EVENT` mode) is **separate from the buffer release callback**. A consumer therefore makes two release calls in `SYNC_EVENT` mode: one for the event handle (called after the consumer has issued its stream-wait — typically immediately after handoff) and one for the buffer (called after all device work on the buffer is complete). Conflating the two would force the producer to keep the event alive for the buffer's entire lifetime, which defeats the purpose of using events instead of full stream sync.

### 5. Per-transport rules

#### In-process

The full §1–§4 contract applies. All three producer mechanisms (P1, P2, P3) are valid. The C ABI handoff structure carries the `sync_mode` discriminant. This is the only transport where event handles cross the producer/consumer boundary; the handles are valid because both parties share the same device context.

#### IPC (same machine, different processes)

The full §1–§4 contract applies with one restriction: **`SYNC_EVENT` (P2) requires an IPC-exportable event primitive on the device in question**. CUDA supports this via `cudaIpcEventHandle_t`; ROCm supports it via `hipIpcEventHandle_t`. Devices without exportable events MUST use `SYNC_PRODUCER_SYNCED` (P1) or, if the consumer has shared an IPC-importable stream handle, `SYNC_CONSUMER_STREAM` (P3). The C ABI MUST expose a query that tells a binding which sync modes are available for a given `device_tag` and IPC channel.

A producer that cannot offer any of P1/P2/P3 (e.g., a device whose driver supports neither exportable events nor stream sharing) MUST NOT initiate an IPC GPU buffer handoff and MUST fall back to a host-staged copy. This degradation is not negotiated on the wire; it is the producer's local decision.

#### Cross-machine (network transport)

`SYNC_EVENT` and `SYNC_CONSUMER_STREAM` are **forbidden across machines**: device event handles and stream handles are not valid in a different driver context on a different host. The only valid producer mechanism is `SYNC_PRODUCER_SYNCED` (P1), and the existing `TENSOR_DATA_END` message is the authoritative producer-synced signal.

Specifically:

- For the non-RDMA path (`TENSOR_DATA` frames), the sender MUST ensure all device-side writes to the source buffer are visible to host reads before sending the first `TENSOR_DATA` frame whose `byte_offset_in_buffer` covers those bytes. This is already implied by the wire encoding (the bytes are read by the host NIC stack); this ADR makes it explicit.
- For the RDMA path **without** GPUDirect, the sender MUST ensure source-buffer writes have retired on the device before posting the RDMA send; the receiver MUST NOT read the buffer before `TENSOR_DATA_END`.
- For the RDMA path **with** GPUDirect (ADR-012), the sender MUST ensure: (a) producer-side device writes to the source buffer have retired before posting the RDMA send, and (b) the RDMA completion has been observed via the sender's completion queue before sending `TENSOR_DATA_END`. The receiver, upon receiving `TENSOR_DATA_END`, MAY assume the destination GPU buffer is visible to subsequent device-stream reads on the receiver's GPU, provided the receiver is using a CUDA/ROCm/etc. driver version that establishes PCIe-write-ordering between an inbound NIC DMA and a subsequent kernel launch on the same device — the spec MUST state this as a receiver-side responsibility, not a producer-side one, because it depends entirely on the receiver's hardware topology.

`TENSOR_DATA_END` therefore retains and extends its existing role: it is the cross-machine equivalent of `SYNC_PRODUCER_SYNCED`.

### 6. Versioning

This ADR adds normative behaviour but no wire-format field. A Hurray 1.0 reader that does not know about §1–§5 will be wire-compatible with a Hurray 1.0 writer; the synchronisation contract is enforced at the C ABI / binding layer, which is versioned separately. The C ABI version MUST be bumped when the `sync_mode` discriminant is added; existing ABI consumers continue to receive `SYNC_PRODUCER_SYNCED` by default, which is the safe choice because it preserves the pre-ADR behaviour of "producer must do the work".

## Alternatives Considered

### Alternative A — Normative "producer MUST sync" rule, no negotiation

Mandate `SYNC_PRODUCER_SYNCED` always; the producer MUST host-wait on its device stream before handoff. Simplest spec, simplest consumer.

- **Pros**: zero ABI surface, zero negotiation, trivially correct, matches what most CPU code expects of GPU buffers.
- **Cons**: full stream synchronisation is the single most expensive thing one can do on a CUDA stream; it serialises the entire inference pipeline. For the inference workloads Hurray targets (where a model graph hands tensor outputs to the next stage every few milliseconds), this collapses pipeline parallelism.
- **Rejected because**: the cost is unacceptable for the primary use case. Hurray's value proposition is zero-copy zero-stall handoff; mandating a full stream sync defeats both halves.

### Alternative B — Carry a sync object handle in the buffer handle binary layout

Burn 8 bytes of the `_reserved` field (or extend the buffer handle from 16 to 24 bytes) for an opaque sync-object handle.

- **Pros**: explicit on-wire description; uniform across in-process / IPC / network.
- **Cons**: (1) event handles are not portable across processes (need IPC export) or machines (meaningless); the same field would carry incompatible payloads in different transports. (2) Bakes a CUDA-flavored primitive into a language- and device-agnostic format. (3) Closes the door on future devices whose synchronisation models do not match the CUDA event abstraction. (4) Bloats every descriptor — even CPU descriptors that need no synchronisation — by 8 bytes.
- **Rejected because**: violates the language-agnostic and device-extensibility invariants of the format. Synchronisation is a transport / runtime concern, not a tensor descriptor property.

### Alternative C — Add a new `BUFFER_SYNC` interchange message

Define a new `0x0000000E BUFFER_SYNC` message carrying a sync-object payload, sent after `TENSOR_DESCRIPTOR` and before `TENSOR_DATA_END`.

- **Pros**: clean separation; opt-in via capability flag; does not pollute the buffer handle.
- **Cons**: meaningless for in-process and IPC (which do not use the wire framing for handoff at all) and meaningless for cross-machine (where event handles do not cross host boundaries). The message would only ever fire in degenerate scenarios.
- **Rejected because**: solves a wire-format problem where the problem is not wire-format.

### Alternative D — Defer entirely, leave to bindings

Make no normative statement; each binding (PyO3, the C ABI, future Go/Java) defines its own rule.

- **Pros**: zero spec surface.
- **Cons**: directly violates the language-agnostic interoperability invariant. Two implementations both claiming Hurray 1.0 compliance could fail to share GPU buffers safely. Worse, both would appear to work on small workloads (where streams happen to drain between operations) and fail probabilistically under load.
- **Rejected because**: synchronisation is the single most common source of silent data corruption in GPU code. Leaving it implementation-defined is a guarantee of interoperability failures.

### Alternative E — DLPack-style: consumer passes a stream at request time, producer waits on it

Adopt only `SYNC_CONSUMER_STREAM` (P3); forbid P1 and P2.

- **Pros**: matches the Python array API; pushes the wait onto the producer's device side (cheap) rather than the host side (expensive); no event handle lifetime to manage.
- **Cons**: forces every consumer to have a designated stream before requesting a buffer. Some consumers (e.g., a CLI inspecting a tensor, a writer streaming to disk) have no GPU stream; they would have to construct one purely to satisfy the protocol. Also fails the IPC case where stream handles are not shareable across drivers without IPC-stream support, which is not universally available.
- **Rejected because**: too restrictive. P3 is offered as one option among three; mandating it loses the host-side use case and the cross-driver IPC use case.

## Consequences

### Positive

- The format wire layout (buffer handle, all message payloads) is unchanged; no backward-compatibility hazard for early adopters.
- The producer-consumer synchronisation contract is normative, closing a silent-data-corruption gap before 1.0 freeze.
- All three sync mechanisms (producer-sync, event, consumer-stream) are available; producers and consumers can choose the cheapest valid option for their workload.
- Cross-machine GPUDirect inherits `TENSOR_DATA_END` as the producer-synced signal — no new message types, no new fields, full alignment with ADR-012.
- Event-release callback is symmetric with the buffer-release callback (ADR-009), so the C ABI gains one mechanism shaped like one the bindings already implement.

### Negative

- The C ABI surface grows: a `sync_mode` discriminant, an opaque sync handle, and an event-release callback are added. The C ABI version MUST be bumped before this is exposed.
- Bindings (PyO3, future Go/Java) MUST implement three code paths for the three sync modes. PyO3 will likely map `SYNC_EVENT` and `SYNC_CONSUMER_STREAM` to `__dlpack__(stream=...)` and `SYNC_PRODUCER_SYNCED` to the no-stream form.
- IPC GPU handoff requires producers to know whether the target device supports exportable events; querying this adds a small amount of complexity to `hurray-ffi`.
- Two-callback handoff (event-release + buffer-release) is more error-prone than one; binding authors MUST be guided clearly. This is the only meaningful new failure mode introduced.

### Risks

- A producer using `SYNC_EVENT` that fails to retire the event before the buffer release callback could cause the consumer's stream-wait to deadlock on a never-recorded event. Mitigation: producers MUST record the event before calling the handoff function; the C ABI contract MUST state this and bindings MUST validate it where cheap.
- A future device type (`device_tag` ≥ 0x09) may have a synchronisation model that fits none of P1/P2/P3 cleanly. Mitigation: P1 is universal (every device supports host-side wait), so P1 is always available as a fallback. The reserved-range owner can add a fourth `sync_mode` in a later ABI version without affecting existing modes.
- A consumer running on a CPU stream (no GPU) that requests a `device_tag = 0x01` (CUDA) buffer via `TENSOR_REQUEST` would currently receive a CUDA buffer; the spec does not say what stream it should synchronise on. Mitigation: the device-colocation rule already requires the consumer to handle the device or close the stream; the synchronisation rule inherits this by reference.
- The receiver-side PCIe ordering assertion for GPUDirect (§5, cross-machine) is hardware-dependent. We must state it as a receiver responsibility, not a producer guarantee, and document the supported NIC/GPU combinations in `docs/impl/c-ffi.md` or a hardware-compatibility note. A receiver that advertises `RDMA_GPUDIRECT` on incompatible hardware is silently broken.

### Compatibility impact

- **Wire format**: unchanged. A 1.0 reader and a 1.0 writer remain wire-compatible regardless of whether either is aware of this ADR.
- **C ABI**: requires a version bump. The default `sync_mode` for legacy clients MUST be `SYNC_PRODUCER_SYNCED`, preserving safe-but-slow behaviour for pre-ADR consumers.
- **Forward compatibility**: a future ADR MAY add `sync_mode` values without affecting existing values, because the discriminant is at the C ABI layer, not in the binary descriptor.

## Handoff

- `format-spec-writer`: add a new normative subsection **§ Stream and Event Synchronisation** to `docs/spec/buffer-protocol.md` covering §1, §2, §4, §5 of this decision. Add a corresponding cross-reference in `docs/spec/interchange.md` § RDMA Data Plane and § Streaming: Tensor Descriptor and Data Frames stating that `TENSOR_DATA_END` is the cross-machine producer-synced signal. No wire-format edits.
- `format-spec-writer`: update `docs/impl/c-ffi.md` to define the `sync_mode` discriminant, the opaque `sync_handle`, the `event_release_fn` callback, and the consumer-stream handoff variant. Bump the documented C ABI version.
- `architect`: revisit if a device type added in the reserved range cannot express its synchronisation model via P1/P2/P3; that would prompt an extension ADR rather than amending this one.
- `spec-checker`: audit the new buffer-protocol subsection and the interchange cross-references for RFC 2119 correctness and consistency with ADR-009 (release callback) and ADR-012 (GPUDirect RDMA) once the spec edits land.

## Open Questions Deferred

- Per-direction GPUDirect sync semantics for asymmetric NIC-GPU topologies (deferred with ADR-012's per-direction RDMA_GPUDIRECT flag).
- Whether `SYNC_CONSUMER_STREAM` can be carried over IPC for drivers that support stream IPC export (CUDA's `cuStreamGetCtx` is not exportable; ROCm has experimental support). Resolution depends on driver capability evolution; not 1.0 blocking.

## Date

2026-05-09
