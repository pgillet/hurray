# ADR-001: Private Extension Types Must Carry Inline Descriptors

## Status

Accepted

## Context

The element-types spec defines a `uint8` type tag space with a private extension range (`0xF0`–`0xFE`) reserved for implementation-specific types. The question (OQ-3 in `element-types.md`) was whether these extension tags should be opaque identifiers or carry enough inline metadata for any conforming reader to handle them gracefully — specifically, to compute buffer sizes and refuse cleanly rather than corrupting memory or crashing.

A related question arose: should the core type system be replaced entirely by a parameterized float descriptor (sign bits, exponent width, mantissa width, exponent bias, NaN/Inf flags) rather than an enumerated tag space? This would make the format self-describing for any IEEE 754-family format without requiring spec updates.

## Decision

Private extension type tags (`0xF0`–`0xFE`) MUST carry an inline type descriptor in the tensor metadata. The descriptor MUST include at minimum:

- A human-readable name (utf8 string, for diagnostics)
- Bit width of a single logical element
- Whether the type is sub-byte packed (bool), and if so the packing factor
- Whether the type is a floating-point type (bool)
- For floating-point types: sign bits (uint8), exponent bits (uint8), mantissa bits (uint8), exponent bias (uint16), and a flags field encoding NaN/Inf semantics

The core enumerated type tag space (Tier 1 and Tier 2) is retained as-is. The parameterized descriptor lives only in the extension mechanism, not in the core type system.

## Alternatives Considered

**Fully parameterized type system (no enumerated tags):** Replace all type tags with inline descriptors. Rejected because:
1. Structural parameters alone (sign, exponent, mantissa widths) cannot fully describe floating-point semantics — NaN patterns, infinity availability, and rounding conventions differ between formats that are structurally identical (e.g., OCP OFP8 `float8_e4m3` vs. a hypothetical IEEE 754 binary8 with the same bit widths but different NaN/Inf rules).
2. Every reader would need to parse a descriptor in hot-path interchange code instead of switching on a single byte.
3. Enumerated tags allow the spec to make unambiguous semantic commitments for each named type.

**Opaque extension tags (no inline descriptor):** Extension tags carry no metadata. Rejected because readers cannot compute buffer sizes for unknown types, making safe rejection impossible without risking out-of-bounds memory access.

## Consequences

- The `metadata.md` spec section must define the inline type descriptor binary encoding for extension type tags.
- Readers that encounter an unknown extension tag MUST parse the inline descriptor to determine buffer size, then reject the tensor (or skip it in permissive mode). They MUST NOT attempt to interpret the data buffer.
- OQ-3 in `element-types.md` is resolved and can be removed when `metadata.md` is written.
- The parameterized float descriptor format defined here for extensions may also serve as the canonical way to describe `float4` and `float6` variants (OQ-4, OQ-5) if those are added to Tier 2 before being formally standardized.
