# ADR-010: Multi-Tensor Collections Deferred; Streams Are Sequences of Self-Delimiting Tensors

## Status

Accepted

## Context

The Hurray spec defines one tensor per descriptor. The question arose whether
Hurray should define a normative multi-tensor collection format — a file or stream
that groups multiple tensors under shared framing, names, or an index.

Use cases that motivate this question:

- **Multi-output inference**: a model server returning multiple output tensors
  in a single response (e.g., logits + hidden states + attention weights).
- **Model weight storage**: a collection of named weight tensors stored in a
  file for distribution and loading (analogous to SafeTensors or GGUF).
- **Batch streaming**: a stream of tensors produced by a pipeline stage,
  consumed incrementally by the next.

Three options were evaluated:

- **Option A** — Sequential stream of self-delimiting descriptor+data pairs.
  No index, no names, pure streaming. Falls out of the existing invariants for
  free.
- **Option B** — Named tensor map: a header with a name→offset index, followed
  by data blobs. Like SafeTensors but zero-copy at runtime.
- **Option C** — Both: a streaming mode (no index) for runtime RPC, and an
  indexed file mode for model storage.

## Decision

**Option A is the normative multi-tensor encoding for Hurray v1. Options B and
C are out of scope for v1.**

A Hurray stream MAY contain zero or more tensors. Back-to-back concatenation of
self-delimiting descriptor+data pairs is the canonical multi-tensor encoding.
A reader processes tensors one at a time, advancing to the next descriptor after
the current tensor's data is consumed. No new framing bytes, no container header,
no names are defined.

A named/indexed tensor collection format (analogous to SafeTensors or GGUF) is
deferred to a future `hurray-archive` sibling specification, not this document.

## Rationale

**Option A is free.** The self-delimiting invariant (a reader can determine each
descriptor's total byte length from its first 10 bytes) already makes sequential
concatenation parse-able. Apache Arrow IPC uses exactly this model: a stream is
zero or more record batches, each self-delimiting. Hurray adopts the same idiom
for tensors.

**Options B and C are premature.** Introducing naming and indexing commits the
spec to a set of decisions that are orthogonal to runtime interchange:

- A string encoding and uniqueness policy for tensor names.
- A namespace model (flat, hierarchical, Zarr-like groups?).
- A lookup mechanism that is in tension with the no-back-references invariant:
  a header index requires knowing all descriptor and buffer sizes before writing
  begins (breaks streamable writers); a footer index requires scanning to
  end-of-file before reading begins (breaks streamable readers).
- Eventual pressure to add key-value metadata (model provenance, quantization
  config, tokenizer parameters) — a parallel type system.

Making these decisions in v1, before any implementation experience, risks locking
in choices that prove wrong in practice.

**Ecosystem positioning.** Hurray's differentiator is zero-copy at runtime, not
better model storage. SafeTensors and GGUF are mature, ecosystem-supported, and
solve the at-rest model-distribution problem. A `hurray-archive` format, if
designed, should be informed by the experience of running the runtime format in
production — not designed up front in parallel.

**Compatibility.** A future indexed format is additive: it would use a distinct
magic byte sequence and would not alter the v1 tensor descriptor encoding.
Deferral is safe.

## Alternatives Considered

**SafeTensors-style header (JSON name→offset map + flat data).** Breaks the
streamable-writer invariant: a writer must know all tensor sizes before it can
write the header. Rejected for v1.

**GGUF-style KV metadata + named tensors.** Commits to a key-value metadata
schema and a type system for metadata values. Scope is much larger than runtime
interchange. Rejected for v1.

**Zarr-style hierarchical namespace.** Commits to a group/array namespace model
with storage-backend abstraction. Orthogonal to runtime interchange. Rejected.

## Consequences

- `docs/spec/interchange.md` MUST add one normative statement: a stream MAY
  contain zero or more tensors; back-to-back concatenation of self-delimiting
  descriptor+data pairs is the canonical multi-tensor encoding.
- `TODO.md` MUST record `hurray-archive` as a future exploration item.
- No other spec files require changes.
- A future `hurray-archive` specification may define a named/indexed format as
  a separate document. It will use a distinct magic byte sequence and will not
  modify the v1 tensor descriptor encoding. It may be introduced as a minor
  version increment (additive) or as a sibling specification, to be decided when
  the design is ready.
