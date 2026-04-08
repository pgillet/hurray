---
name: spec-section
description: Scaffold a new normative spec section in docs/spec/ with RFC 2119 template, non-normative note format, and open question markers.
---

# Spec Section Scaffolder

Use the `format-spec-writer` agent to create a new normative spec section for the hurray format.

## Arguments

`$ARGUMENTS` — the section name and topic (e.g., `memory-layout`, `quantization`, `buffer-protocol`)

## What to Do

1. Derive a kebab-case filename from `$ARGUMENTS` (e.g., `docs/spec/memory-layout.md`).
2. Create the file using the template below.
3. Add an entry to `docs/spec/README.md` under the table of contents linking to the new file.

## Template

```markdown
# <Title>

> **Status:** Draft

## Scope

This section defines <one-line scope statement>.

> **Note (non-normative):** <optional context for readers>

## Normative Requirements

> This section uses RFC 2119 key words: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT,
> SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

<!-- Write normative content here -->

## Open Questions

<!-- Mark unresolved design questions as:
> **[OQ-N]:** <question text>
-->
```

## Conventions to Enforce

- Normative language: `MUST`, `MUST NOT`, `SHOULD`, `SHOULD NOT`, `MAY` (uppercase, RFC 2119)
- Non-normative content: prefix block with `> **Note (non-normative):**`
- Open questions: `> **[OQ-N]:** ...` where N is sequential per file
- All types use language-agnostic names: `int32`, `uint64`, `utf8 string` — never `i32`, `usize`, `String`
- All byte values use hex literals: `0x00`, `0xFF`
- Strides expressed in logical elements, not bytes
