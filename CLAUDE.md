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

The project is in the **implementation phase**. The spec (still in Draft status) is the source of truth, but implementation feedback MAY reveal gaps or ambiguities that require spec corrections — this is expected and permitted during the draft period. When the implementation contradicts the spec, fix the implementation first; if the spec itself is ambiguous or wrong, report it and wait for user approval before amending it.

### Implementation rules

- **One sub-topic at a time.** Each development pass covers exactly one layer of the dependency stack (see workflow below). Do not advance to the next layer until the user approves the current one.
- **Always explain before coding.** Before writing any code, describe what you are about to implement, which spec sections govern it, and any design choices you are making. Wait for the user to confirm.
- **Every pass ships five things:** (1) implementation code, (2) unit tests (`#[cfg(test)]` modules or `tests/`), (3) a doc-comment example on every public item, (4) a runnable example in `<crate>/examples/<name>.rs` (Rust "script" with a `main` entry point, runnable via `cargo run --example <name>`), (5) a new entry in `docs/cookbook/` demonstrating the feature in context.
- **`cargo clippy -- -D warnings` and `cargo test` must pass** before a pass is considered complete.
- **Spec amendments are allowed.** If an implementation pass surfaces a genuine spec ambiguity or error, open a finding (like a spec-checker finding) and route it to `format-spec-writer` or `architect` before proceeding. The spec is not frozen.
- **hurray-inspect depends on hurray-core.** The existing self-contained implementation in `hurray-inspect/src/main.rs` must be replaced once the relevant `hurray-core` types are available. Do not add to the self-contained version; schedule its refactor as part of the descriptor encoding pass.

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
- **Document design decisions in code.** When a non-obvious implementation choice was made over a considered alternative, add a brief inline comment explaining the WHY — not what the code does, but why this specific approach was chosen. Examples: `// PartialEq only: f32 NaN semantics make Eq unsound.` or `// Reject private tags: unconstrained wire format gives callers nothing useful.` Keep it to one line; no multi-line blocks.
- **Correctness first, but performance is a first-class concern.** The implementation must be correct above all, but must also aim for performance from the start — choose efficient algorithms, avoid unnecessary allocations, and design for zero-copy and SIMD-friendly layouts. The `performance-optimizer` agent may be invoked proactively, not only on explicit request.
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

### Spec phase (complete)

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

### Implementation phase (current)

One layer at a time, user-approved before advancing:

```
Layer 0 — Element types + Data model     (hurray-core: ElementType, Shape)
Layer 1 — Buffer protocol                (hurray-core: BufferHandle, DeviceTag, alignment)
Layer 2 — Quantization descriptors       (hurray-core: scheme types, per-tensor/channel/block/NF4/MXFP)
Layer 3 — Layout descriptors             (hurray-core: layout tags, per-layout structs, sparse multi-buffer)
Layer 4 — Tensor descriptor encoding     (hurray-core: TensorDescriptor, binary encode/decode)
           → refactor hurray-inspect to use hurray-core (replaces self-contained parser)
Layer 5 — Streaming interchange          (hurray-io: IPC framing, async reader/writer)
Layer 6 — File format                    (hurray-io: HRRYFILE container, footer index, KV section)
Layer 7 — C FFI                          (hurray-ffi: opaque handles, function table, release callbacks)
Layer 8 — Python bindings                (hurray-python: PyO3 + __dlpack__ zero-copy)
```

Each layer's pipeline:
```
planner             (breaks layer into steps, identifies spec sections)
    ↓
[user approval]
    ↓
rust-developer      (implements against the spec)
    ↓
rust-test-writer    (unit tests + integration tests)
    ↓
rust-reviewer       (correctness, idioms, spec fidelity)
    ↓
rust-build-resolver (fixes any compile errors)
    ↓
doc-updater         (syncs /// doc comments + adds docs/cookbook/ entry)
    ↓
[user approval → next layer]
```

Spec feedback loop (runs in parallel when needed):
```
rust-developer finds ambiguity
    ↓
format-spec-writer (editorial) or architect (design)
    ↓
spec-checker (targeted re-audit of affected section)
    ↓
resume rust-developer
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
