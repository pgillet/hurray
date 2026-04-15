# ADR-003: Panel/Pack Formats via Extension Layout Tags and Content Negotiation

## Status

Accepted

## Context

Panel/pack formats are internal buffer layouts used by BLAS/BLIS libraries (and
equivalents such as cuBLAS, oneDNN) when preparing matrix inputs for GEMM kernels.
Before a multiply, inputs are repacked into a layout tuned to the target hardware's
cache hierarchy, SIMD register width, and panel dimensions. The repacked buffer is
consumed immediately by the kernel and then discarded.

OQ-2 in `memory-layout.md` asked whether panel/pack should be a named Tier 1 or Tier 2
layout tag, or explicitly out of scope.

The initial analysis favoured "out of scope" on the grounds that these formats are
implementation-specific and not portable. However, the design of content negotiation
in `interchange.md` changed the calculus: a client can advertise its hardware profile
to the server, the server transcodes and packs on the fly, and the client hands the
result directly to the BLAS kernel. The packed buffer never crosses an incompatible
boundary; portability is not required.

The remaining question was whether to define a named Tier 2 layout tag with normative
hardware-parameter fields, or to use the existing extension layout mechanism.

## Decision

Panel/pack formats are **not** given a named layout tag. They are explicitly supported
via the **extension layout mechanism** (`0xF0`–`0xFE`) combined with the transport
protocol's **content negotiation**.

The layout entry encoding in `interchange.md` is extended so that extension layout tags
in `preferred_layouts` (in `TENSOR_REQUEST`) and `supported_layouts` (in `CLIENT_HELLO`
/ `SERVER_HELLO`) MAY carry opaque metadata (`ext_metadata`) alongside the tag byte.
For panel/pack, this metadata encodes the client's hardware profile. The server either
recognises the profile and transcodes, or skips to the next preference.

## Alternatives Considered

- **Named Tier 2 layout tag with normative hardware-parameter fields**: rejected.
  BLIS, OpenBLAS, cuBLAS, and oneDNN do not agree on the relevant parameters or their
  semantics. Any normative definition would either be too narrow (tied to one library's
  model) or too abstract to be actionable. Named layouts must be interpretable by any
  conforming reader; panel/pack cannot meet that bar without locking in specific library
  internals.

- **Explicitly out of scope**: rejected. The content negotiation mechanism makes
  panel/pack tractable without requiring portability. Saying "out of scope" would miss
  a real use case that the extension mechanism already handles cleanly.

## Consequences

- The layout entry encoding in `interchange.md` is variable-length: core layout tags
  are a single byte; extension tags carry an additional `uint16` length and opaque
  metadata. Decoders MUST be able to skip unrecognised extension entries using the
  length field.
- Panel/pack is explicitly documented in `memory-layout.md` as the canonical use case
  for extension layouts via content negotiation.
- No central registry of extension layout identifiers is defined. Producers and
  consumers must agree on the `extension_layout_id` and `ext_metadata` schema out of
  band (e.g. via a shared library or published profile specification).
- NVIDIA Tensor Core fragment layouts and other hardware-internal formats remain out of
  scope even under this decision, as they are not intended for interchange at all.
