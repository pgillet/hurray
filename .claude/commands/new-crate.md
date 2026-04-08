---
name: new-crate
description: Scaffold a new hurray-* crate with standard Cargo.toml, src/lib.rs, src/error.rs, and correct feature flags per CLAUDE.md.
---

# New Crate Scaffolder

Scaffold a new `hurray-*` crate with the standard structure defined in CLAUDE.md.

## Arguments

`$ARGUMENTS` — the crate name (e.g., `hurray-core`, `hurray-io`, `hurray-ffi`, `hurray-python`)

## What to Do

1. Create `$ARGUMENTS/Cargo.toml` with the dependencies appropriate for the crate (see table below).
2. Create `$ARGUMENTS/src/lib.rs` with module declarations and a crate-level `//!` doc comment.
3. Create `$ARGUMENTS/src/error.rs` with a crate-level `Error` enum using `thiserror` (except `hurray-python` which uses `pyo3::PyErr`).
4. Add `"$ARGUMENTS"` to the `members` list in the workspace `Cargo.toml`.

## Crate Dependencies (from CLAUDE.md)

| Crate | Dependencies | Feature flags |
|-------|-------------|---------------|
| `hurray-core` | `thiserror`, `half`, `rayon` | `serde` (gates `serde` + `serde_derive`) |
| `hurray-io` | `hurray-core`, `tokio`, `bytes`, `thiserror` | `tokio` (gates async I/O) |
| `hurray-ffi` | `hurray-core` | none |
| `hurray-python` | `hurray-core`, `pyo3` | `extension-module` (pyo3) |

## Conventions

- No `unwrap()` or `expect()` in library code — propagate with `?`
- All public items must have `///` doc comments with at least one example
- `unsafe` code goes in a dedicated submodule with `// SAFETY:` on every block
- Feature flags: use `#[cfg(feature = "...")]` guards consistently
- `cargo clippy -- -D warnings` must pass
