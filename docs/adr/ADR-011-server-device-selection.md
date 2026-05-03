# ADR-011: Server Device Selection Algorithm

## Status

Accepted

## Context

The Hurray network transport protocol advertises device capabilities in
`CLIENT_HELLO` and `SERVER_HELLO` (`supported_devices` lists) and accepts a
`preferred_device` tag in `TENSOR_REQUEST`. The spec overview claimed "device
negotiation" support, but no normative selection algorithm existed. Without
one, the `DEVICE_UNAVAILABLE` error code and `preferred_device` field had no
defined semantics.

The core problem is that device selection, unlike layout selection, has no
natural ordered preference list on the wire (`preferred_device` is a single
tag). A silent fallback to a different device (e.g., CUDA → CPU) produces
correct results but may cause catastrophic performance regressions in
inference workloads.

## Decision

The server follows a strict ordered algorithm upon receiving
`preferred_device`:

1. Serve on `preferred_device` if available.
2. If `preferred_device` is CPU and unavailable: `DEVICE_UNAVAILABLE` error.
3. If `preferred_device` is non-CPU and unavailable: `DEVICE_UNAVAILABLE`
   error (no silent fallback).
4. Single exception: if `preferred_device` was advertised but is transiently
   unavailable, the server MAY fall back to CPU if and only if the client also
   advertised CPU. This is the only permitted silent fallback.

The actual device is reported in the buffer handle `device_tag` fields of
`TENSOR_DESCRIPTOR`.

## Alternatives Considered

**Silent fallback to CPU always.** Rejected — hides performance collapses in
inference workloads.

**Client supplies a preference list (ordered).** Rejected for v1 — expands the
`TENSOR_REQUEST` wire format and conflates device selection with layout
negotiation. The `DEVICE_UNAVAILABLE` error gives the client enough
information to retry with a different `preferred_device`.

**Never fallback, always error.** Rejected — the narrow CPU-fallback exception
provides a useful graceful-degradation path for clients that advertise CPU
support, at zero protocol complexity cost.

## Consequences

- `DEVICE_UNAVAILABLE` (`0x00000005`) is now a normative response to an
  unsatisfiable `preferred_device`.
- Clients that want graceful CPU fallback MUST advertise CPU in
  `supported_devices`.
- The `hurray-io` Layer 5 implementation will need a server-side
  device-availability hook (not a spec concern).
- `buffer-protocol.md` § Device Colocation gains a cross-reference noting that
  `TENSOR_PUT` `device_tag` is binding on the receiver.
