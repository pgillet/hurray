# ADR-012: GPUDirect RDMA via Bidirectional RDMA_REGISTER

## Status

Accepted

## Context

The RDMA data plane handshake (defined for OQ-2) only covered the
source-buffer owner registering its memory region. For GPUDirect — where the
sender writes directly into the receiver's GPU memory, eliminating the
host-to-device copy — the receiver must also register a destination GPU memory
region and share its rkey with the sender. No message or capability existed
for this.

The existing `RDMA_REGISTER` message tag (`0x0000000C`) is defined as "Either"
direction but the prose only described its use by the source-buffer owner.

## Decision

Bidirectional `RDMA_REGISTER`: the destination-buffer owner sends a second
`RDMA_REGISTER` (same message tag, same payload shape) after the
source-buffer owner's `RDMA_REGISTER`, before `RDMA_READY`. This is gated by a
new capability flag `RDMA_GPUDIRECT` (bit 3 of `capability_flags`), which MUST
imply `RDMA_DATA_PLANE` (bit 2).

The mechanism is fully symmetric: the same rules apply to `TENSOR_REQUEST`
(server is source, client is destination) and `TENSOR_PUT` (client is source,
server is destination) with roles inverted.

For `TENSOR_PUT`, the client unilaterally declares the destination
`device_tag` in the descriptor; the server rejects with `DEVICE_UNAVAILABLE`
before the RDMA handshake if it cannot honor that device.

## Alternatives Considered

**Option A — Extend `RDMA_READY` to carry an optional destination rkey.**
Rejected — conflates an acknowledgement message with a registration message;
creates an asymmetry between `TENSOR_REQUEST` (client sends `RDMA_READY` with
rkey) and `TENSOR_PUT` (server sends `RDMA_READY`, which direction carries
the rkey?).

**Option B — New `RDMA_REGISTER_DST` message type.** Rejected — adds a new
tag that is semantically identical to `RDMA_REGISTER`; the only distinction
is "which direction"; that distinction is already captured by the message
sequence and the capability flag.

**Option C (chosen) — Bidirectional `RDMA_REGISTER`.** Reuses the existing
tag; the payload is role-agnostic (rkey + remote_addr + length); the sequence
position and capability flag make the role unambiguous.

## Consequences

- New capability flag `RDMA_GPUDIRECT` (bit 3). Reserved range updated to
  4–63.
- No new message type tags.
- `RDMA_REGISTER` payload prose updated to describe both source-owner and
  destination-owner roles.
- New normative subsection: § GPUDirect Destination Registration.
- `TENSOR_PUT with RDMA` subsection rewritten to include full diagram and
  GPUDirect path.
- Receivers advertising `RDMA_GPUDIRECT` are responsible for ensuring
  NIC-GPU topology compatibility (PCIe root complex colocation or NVLink);
  the protocol cannot validate this.
- Per-direction `RDMA_GPUDIRECT` capability flags (TX vs RX) are deferred to
  a future revision.
