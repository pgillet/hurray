# Hurray Implementation Requirements

This directory contains requirements for **implementations** of the Hurray format.
These requirements are distinct from the format specification (`docs/spec/`):

| | `docs/spec/` | `docs/impl/` |
|---|---|---|
| **Scope** | What the binary encoding and protocol must look like | What a conforming implementation must provide |
| **Language** | Language-agnostic | Specific to each implementation layer |
| **Audience** | Anyone writing a Hurray reader/writer | Authors of the Hurray implementations themselves |

The format spec is the source of truth for the wire format. These documents define
the API contracts, binding conventions, compliance criteria, and quality requirements
that the reference implementation and language bindings must satisfy.

## Implementations in Scope

| Implementation | Language | Crate / Module |
|---|---|---|
| Reference implementation | Rust | `hurray-core`, `hurray-io` |
| C FFI layer | C (via Rust) | `hurray-ffi` |
| Python bindings | Python (via PyO3) | `hurray-python` |

## Documents

| Document | Description |
|---|---|
| [compliance](compliance.md) | Conformance levels, mandatory vs optional feature support, test surface |
| [rust-reference](rust-reference.md) | Requirements for `hurray-core` and `hurray-io` |
| [c-ffi](c-ffi.md) | C ABI layer requirements: opaque handles, function table, panic safety |
| [python-bindings](python-bindings.md) | Python Array API compliance, DLPack interop, NumPy/PyTorch zero-copy |
