---
paths:
  - "docs/spec/**/*.md"
  - "docs/adr/**/*.md"
---
# Spec and ADR Writing Conventions

## Normative Language (RFC 2119)

All normative requirements MUST use RFC 2119 key words in UPPERCASE:

| Word | Meaning |
|------|---------|
| `MUST` / `REQUIRED` / `SHALL` | Absolute requirement |
| `MUST NOT` / `SHALL NOT` | Absolute prohibition |
| `SHOULD` / `RECOMMENDED` | Strong preference; deviation requires justification |
| `SHOULD NOT` / `NOT RECOMMENDED` | Strong discouragement |
| `MAY` / `OPTIONAL` | Permitted but not required |

Every spec file that uses normative language MUST include this notice near the top:

> This section uses RFC 2119 key words: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT,
> SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

## Non-Normative Content

Non-normative notes MUST be prefixed:

```markdown
> **Note (non-normative):** <content>
```

Never use normative key words inside non-normative blocks.

## Open Questions

Unresolved design questions MUST be marked inline and tracked:

```markdown
> **[OQ-N]:** <question text>
```

Where `N` is sequential per file, starting at 1. Open questions are resolved by the `architect` agent and removed once a decision is recorded in an ADR.

## Type Names

All types MUST use language-agnostic names. Never use Rust, C, or Python type names in the spec.

| Use | Never use |
|-----|-----------|
| `int8`, `uint8` | `i8`, `u8`, `char` |
| `int32`, `uint32` | `i32`, `u32`, `int` |
| `int64`, `uint64` | `i64`, `u64`, `usize` |
| `float32`, `float64` | `f32`, `f64`, `double` |
| `float16`, `bfloat16` | `f16`, `bf16` |
| `utf8 string` | `String`, `str`, `&str` |
| `byte sequence` | `Vec<u8>`, `bytes`, `[]byte` |

## Byte Values

All byte values and bit patterns MUST use hex literals: `0x00`, `0xFF`, `0x1A2B`.
Never use decimal for byte-level values.

## Endianness

All multi-byte fields are little-endian. Specs MUST state byte order explicitly for every multi-byte field.

## Strides

Strides are ALWAYS expressed in logical elements, not bytes. Specs MUST not use byte-stride without an explicit conversion note.

## ADR Format

```markdown
# ADR-NNN: <Title>

## Status
Draft | Accepted | Superseded by ADR-MMM

## Context
<what situation or problem drove this decision>

## Decision
<what was decided, in present tense>

## Alternatives Considered
<what else was evaluated and why it was rejected>

## Consequences
<trade-offs, follow-up work, constraints this decision imposes>
```
