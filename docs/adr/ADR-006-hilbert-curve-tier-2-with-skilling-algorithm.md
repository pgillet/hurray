# ADR-006: Hilbert Curve Is a Named Tier 2 Layout with the Skilling (2004) Algorithm

## Status

Accepted

## Context

The Hilbert curve layout (tag `0x40`) was placed in Tier 2 as a provisional entry
with no normative index mapping. OQ-5 asked whether to confirm the Tier 2 placement
(requiring a complete normative algorithm) or move it to the implementation-private
extension range (`0xF0`–`0xFE`) where no normative definition would be needed.

## Decision

The Hilbert curve is confirmed as a **named Tier 2 layout**. The normative index
mapping is the algorithm from Skilling (2004), reproduced verbatim in
`memory-layout.md`. Both directions — `CoordsToHilbert` and `HilbertToCoords` — are
specified as normative pseudocode.

## Alternatives Considered

- **Extension range only**: rejected. A layout tag in the extension range requires
  out-of-band agreement between producer and consumer, providing no interoperability
  benefit. If the mapping is not normative, there is no point including it in the spec.
- **Drop entirely**: rejected. The Hilbert curve has a meaningful advantage over
  Morton for 2D/3D spatial tensors (no large jumps across quadrant boundaries), and a
  clean normative algorithm exists. Dropping it would leave a gap for spatial-locality
  use cases with no standard answer.

## Consequences

- Conforming implementations that support layout tag `0x40` MUST implement the
  Skilling (2004) algorithm exactly. Two compliant implementations will produce
  identical index mappings for identical inputs.
- The algorithm imposes the constraint that all tensor dimensions equal `2^hilbert_order`.
  Non-power-of-two spatial tensors must be padded or use a different layout.
- The Skilling algorithm has O(r * p) time complexity per element (r = rank, p = order),
  which is more expensive than Morton's O(r * p) bit-interleave but with higher
  constant factors due to the rotation/Gray-code state machine. This is acceptable for
  a reference implementation; performance-critical paths may cache the mapping.
- The conformance example table in `memory-layout.md` (16 entries for the 4×4 case)
  gives implementations a concrete test vector.
