# Hurray — Agent Configuration

> For the human-readable project overview, format contract, and key properties, see **`README.md`** at the repo root.

## Live Information Sources

Agents MUST read from project files directly rather than relying on memory for project content:

| What | Where |
|------|-------|
| Project overview and format contract | `README.md` |
| Pending ideas and tasks | `TODO.md` |
| Open questions | `grep -rn "OQ-" docs/spec/` |
| Architectural decisions | `docs/adr/` |
| Prior art survey | `docs/prior-art.md` |
| Format specification | `docs/spec/` |
| Implementation requirements | `docs/impl/` |

## Current Phase

The project is in the **specification and requirements phase**. Work is focused on `docs/spec/`, `docs/impl/`, and `docs/prior-art.md`. Rust implementation work is deferred until the spec is stable. Do not produce implementation code unless explicitly asked.

## Project Structure

```
hurray/
├── CLAUDE.md                   # This file
├── TODO.md                     # Running list of ideas and future tasks (reviewed periodically)
├── Cargo.toml                  # Workspace root
├── docs/
│   ├── prior-art.md            # Research snapshot: formats, protocols, libraries
│   ├── spec/                   # Format specification (source of truth)
│   │   ├── README.md           # Scope, goals, RFC 2119 notice, versioning
│   │   ├── data-model.md       # Shape/dimension model
│   │   ├── element-types.md    # Element type system (int, float, quantized, custom)
│   │   ├── quantization.md     # Quantization schemes: per-tensor, per-channel, per-block
│   │   ├── memory-layout.md    # Layout index and overview
│   │   ├── layouts/            # Per-layout spec files
│   │   │   ├── row-major.md
│   │   │   ├── column-major.md
│   │   │   ├── strided.md
│   │   │   ├── tiled.md
│   │   │   ├── morton.md
│   │   │   ├── hilbert.md
│   │   │   ├── subpaving.md
│   │   │   ├── coo.md          # Sparse: Coordinate list
│   │   │   ├── csr.md          # Sparse: Compressed Sparse Row
│   │   │   └── csc.md          # Sparse: Compressed Sparse Column
│   │   ├── buffer-protocol.md  # Zero-copy semantics, alignment, device memory
│   │   ├── metadata.md         # Tensor descriptor binary encoding
│   │   ├── interchange.md      # Runtime interchange: in-process, IPC, cross-machine
│   │   ├── versioning.md       # Format version field, compatibility policy
│   │   └── references.md       # Normative references
│   ├── impl/                   # Implementation requirements (not the spec itself)
│   │   ├── README.md           # Overview of implementation requirement docs
│   │   ├── compliance.md       # Compliance checklist for implementors
│   │   ├── rust-reference.md   # Rust reference implementation guide
│   │   ├── c-ffi.md            # C FFI implementation guide
│   │   └── python-bindings.md  # Python bindings implementation guide
│   └── adr/                    # Architecture Decision Records
│       └── ADR-NNN-*.md
├── hurray-core/                # Core types, no I/O, no async
├── hurray-io/                  # Async I/O: streaming + file format (tokio)
├── hurray-ffi/                 # C ABI layer for language bindings
├── hurray-python/              # Python bindings (PyO3)
└── hurray-inspect/             # CLI hex viewer for Hurray descriptor files
```

## Guiding Principles

- **Spec is the source of truth.** The Rust implementation follows the spec. When they conflict, fix the implementation, not the spec.
- **Zero-copy first.** Data must be shareable across runtimes without copying whenever possible.
- **Streamable.** Both readers and writers must be able to process tensor data incrementally. A reader must be able to start processing without buffering the entire input; a writer must be able to emit tensors one at a time without buffering the entire output. Tensor descriptors always precede their data buffers; the format is self-delimiting; back-references and end-of-file indexes are forbidden.
- **Language-agnostic.** No Rust-isms leak into the format design or the C FFI boundary.
- **Correctness before performance.** This is a reference implementation. Optimize only when explicitly asked.
- **Inference-optimized.** Layout diversity, quantization, and device memory are first-class concerns.

## Crate Responsibilities

| Crate | Responsibility | Key dependencies |
|-------|---------------|-----------------|
| `hurray-core` | Format types, tensor descriptor, buffer handle, quantization descriptors. No I/O, no async. | `thiserror`, `serde` (feature-gated), `half`, `rayon` |
| `hurray-io` | Async streaming and file format read/write. | `hurray-core`, `tokio`, `bytes` |
| `hurray-ffi` | C ABI: opaque handles, function table, buffer release callbacks. No panics across FFI. | `hurray-core` |
| `hurray-python` | Python bindings with NumPy/PyTorch zero-copy interop via `__dlpack__`. | `hurray-ffi` or `hurray-core`, `pyo3` |

## Key Technical Decisions

- **Endianness**: little-endian throughout (all multi-byte fields)
- **Alignment**: minimum 64-byte buffer alignment (SIMD); page-aligned for GPU/IPC
- **Strides**: expressed in logical elements, not bytes; negative and zero strides are valid
- **Sub-byte packing**: `int4`/`bool` packing order defined in spec (see `memory-layout.md`)
- **Error handling**: crate-level `Error` enum via `thiserror`; no `unwrap()`/`expect()` in library code
- **Unsafe**: isolated to dedicated modules with `// SAFETY:` comments on every block

## Agent Roles

| Agent | Owns |
|-------|------|
| `researcher` | State-of-the-art surveys, prior art analysis, hardware constraint research. Maintains `docs/prior-art.md`. Runs before major design decisions. |
| `architect` | Design decisions, trade-off analysis, ADRs (`docs/adr/`). Consumes researcher output. Resolves open questions escalated by `spec-checker`. |
| `format-spec-writer` | All files under `docs/spec/` and `docs/impl/`. Resolves ambiguities and contradictions reported by `spec-checker` or implementation agents. |
| `spec-checker` | Read-only audit of the full spec corpus (`docs/spec/`, `docs/impl/`). Reports contradictions, gaps, redundant definitions, unclosed open questions [OQ-N], and RFC 2119 misuse. Never edits files directly — findings go to `format-spec-writer` (editorial fixes) or `architect` (design questions). Invoked periodically or before major spec milestones. |
| `planner` | Breaks complex features into concrete, phased implementation steps. Runs before `rust-developer` for non-trivial work. |
| `rust-developer` | All files under `hurray-*/src/`. Implements what the spec defines. Does not write tests. |
| `rust-test-writer` | All files under `hurray-*/tests/` and `#[cfg(test)]` modules. Tests the public API, not internals. |
| `rust-reviewer` | Reviews implementation and tests for correctness, idioms, and spec fidelity. |
| `rust-build-resolver` | Resolves `cargo check` / `cargo build` failures. |
| `performance-optimizer` | Profiling and optimization passes. Only invoked explicitly. |
| `refactor-cleaner` | Code cleanup and refactoring. Only invoked explicitly. |
| `doc-updater` | Keeps `///` doc comments and `docs/` in sync with implementation changes. |

## Development Workflow

### Spec phase (current)

```
researcher          (surveys prior art, updates docs/prior-art.md)
    ↓
architect           (makes design decision, writes ADR)
    ↓
format-spec-writer  (writes/updates docs/spec/ and docs/impl/)
    ↓
spec-checker        (audits full corpus for consistency, reports findings)
    ↓
format-spec-writer  (applies editorial fixes from spec-checker report)
    ↑
architect           (resolves design-level findings from spec-checker)
```

### Implementation phase (future)

```
planner             (breaks feature into steps)
    ↓
rust-developer      (implements against the spec)
    ↓
rust-test-writer    (writes tests against the public API)
    ↓
rust-reviewer       (reviews both)
    ↓
rust-build-resolver (fixes any compile errors)
    ↓
doc-updater         (syncs doc comments and docs/ with implementation)
```

## Spec Writing Conventions

- Normative language follows RFC 2119: `MUST`, `MUST NOT`, `SHOULD`, `SHOULD NOT`, `MAY`
- Non-normative content is prefixed: `> **Note (non-normative):**`
- Open questions are marked inline: `> **[OQ-N]:** ...`
- All types use language-agnostic names: `int32`, `uint64`, `utf8 string` — never `i32`, `usize`, `String`
- All byte examples use hex literals: `0x00`, `0xFF`

## Rust Conventions

- No `unwrap()` or `expect()` in library code — propagate with `?`
- All public items have `///` doc comments with at least one example
- `unsafe` code is isolated in dedicated modules; every block has a `// SAFETY:` comment
- `cargo clippy -- -D warnings` must pass before any code is considered complete
- Feature flags: `serde` for serialization support, `tokio` for async I/O, `python` for bindings
- Do not mix `rayon` thread pool calls directly in async contexts — use `tokio::task::spawn_blocking`

## Prior Art

See `docs/prior-art.md` for the full survey. A summary table is in `README.md`.
