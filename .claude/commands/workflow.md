---
name: workflow
description: Run the full hurray feature development pipeline — researcher → architect → format-spec-writer → rust-developer → rust-test-writer → rust-reviewer — for a given feature or design question.
---

# Feature Development Workflow

Trigger the full hurray development pipeline for a new feature or design decision.

## Arguments

`$ARGUMENTS` — description of the feature, type, or design question to implement
(e.g., `int4 packed memory layout`, `per-channel quantization descriptor`, `C FFI handle lifecycle`)

## Pipeline

Execute the following agents **in order**, passing each agent's output as context to the next:

### 1. researcher
Survey prior art, existing formats, hardware constraints, or academic literature relevant to `$ARGUMENTS`.
Produce: a structured findings report with references.

### 2. architect
Consume researcher output. Make and document the design decision via an ADR in `docs/adr/`.
Produce: an ADR file and a clear spec mandate for the format-spec-writer.

### 3. format-spec-writer
Consume the ADR. Write or update the relevant normative spec section(s) in `docs/spec/`.
Produce: updated spec markdown with RFC 2119 language.

### 4. rust-developer
Implement `$ARGUMENTS` against the spec. Touch only files under `hurray-*/src/`.
Produce: implementation code. Does not write tests.

### 5. rust-test-writer
Write tests for the public API surface implemented in step 4.
Touch only `hurray-*/tests/` and `#[cfg(test)]` modules.
Produce: test code covering happy path, edge cases, and property-based tests where applicable.

### 6. rust-reviewer
Review both implementation and tests for correctness, spec fidelity, Rust idioms, and safety.
Produce: review report. Block merge on CRITICAL issues; flag HIGH issues.

## Notes

- If the researcher finds that no prior art exists, the architect may proceed directly from first principles.
- If the spec already covers `$ARGUMENTS`, skip steps 1–3 and start from rust-developer.
- If `cargo check` or `cargo clippy` fails after step 4, invoke `rust-build-resolver` before proceeding to step 5.
