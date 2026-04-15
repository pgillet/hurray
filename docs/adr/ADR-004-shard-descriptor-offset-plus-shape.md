# ADR-004: Shard Descriptor Uses Offset + Shape (Axis-Aligned Box)

## Status

Accepted

## Context

A Hurray tensor MAY carry a shard descriptor indicating its position within a larger
logical parent tensor. OQ-3 in `memory-layout.md` asked whether the shard descriptor
should use the current offset + shape model (an axis-aligned hyperrectangle / box) or
a more general subpaving region descriptor that could express non-rectangular or
non-contiguous shards.

## Decision

The shard descriptor retains the **offset + shape** design. A shard is an axis-aligned
box in the parent tensor's index space, fully described by `parent_shape`,
`shard_offset`, and the shard's own `shape`. For each dimension `k` the shard covers
indices `[shard_offset[k], shard_offset[k] + shape[k])`.

## Alternatives Considered

- **General subpaving region descriptor**: would reuse the machinery of layout tag
  `0x06` to allow non-rectangular or non-contiguous shards. Rejected for two reasons:
  1. Conflates distinct concepts — the subpaving layout describes how elements are
     arranged in memory; the shard descriptor describes the logical position of a
     sub-tensor within a parent. Merging these adds implementation complexity without
     benefit.
  2. No identified use case: all practical sharding patterns in ML inference
     (batch splitting, tensor parallelism, pipeline stages) produce axis-aligned boxes.

## Consequences

- The shard descriptor is simple to encode, decode, and validate (one bounds check per
  dimension).
- The parallel transfer protocol in `interchange.md` can rely on box semantics for
  coverage and non-overlap validation.
- If a future use case for non-rectangular shards is identified, a general shard
  descriptor MAY be added alongside the current one as an optional field; the offset +
  shape form would remain the default.
