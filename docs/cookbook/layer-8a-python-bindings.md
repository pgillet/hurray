# Layer 8a — Python Bindings: Phase 8a.0 Scaffolding

This entry covers the initial scaffolding for the `hurray` Python package: the
Maturin build wiring, the runtime compliance mode API, and the smoke-test
example that exercises it.

## What was built

| Artifact | Purpose |
|----------|---------|
| `hurray-python/Cargo.toml` | `extension-module` feature flag off by default so `cargo test` can link libpython; `abi3-py310` for a single wheel per OS/arch |
| `hurray-python/pyproject.toml` | Maturin build config; activates `extension-module` for wheel builds |
| `hurray-python/src/lib.rs` | `#[pymodule]` root; registers `__version__`, `set_strict`, `is_strict`, `StrictCtx`, `RelaxedCtx` |
| `hurray-python/src/modes.rs` | Runtime compliance mode API (ADR-022) |
| `hurray-python/COMPAT-MATRIX.md` | Living document tracking Array API / DLPack / CPython version support |
| `hurray-python/examples/00_hello_hurray.py` | Runnable smoke test (requires `maturin develop`) |

## Runtime compliance modes (ADR-022)

`hurray` ships in strict mode by default. Strict mode enforces full Python Array
API Standard compliance — only Tier 1 element types are admitted through the
Array API surface. Relaxed mode (Tier 2 / quantized types) is reserved for a
future release.

```python
import hurray

hurray.is_strict()       # True — always in this version
hurray.set_strict(True)  # no-op

try:
    hurray.set_strict(False)  # raises NotImplementedError (reserved)
except NotImplementedError as e:
    print(e)

with hurray.StrictCtx():  # no-op context manager, reserves the name
    pass

try:
    with hurray.RelaxedCtx():  # raises NotImplementedError (reserved)
        pass
except NotImplementedError as e:
    print(e)
```

## Building the wheel

```bash
# Development install (builds in-place, no wheel file)
cd hurray-python
maturin develop

# Run the smoke-test example
python examples/00_hello_hurray.py

# Build a release wheel
maturin build --release
```

## Running the Rust tests

```bash
# Unit tests link libpython directly (extension-module feature must be OFF)
cargo test -p hurray-python

# Build as a Python extension (extension-module ON)
cargo build -p hurray-python --features extension-module
```

## Key design notes

- **`extension-module` off by default** — PyO3's `extension-module` feature
  removes the `-lpython` linker flag needed by test binaries. Maturin enables
  it at wheel-build time via `pyproject.toml`; `cargo test` leaves it off.

- **`abi3-py310` stable ABI** — one wheel per OS/arch runs on CPython 3.10+;
  no per-minor-version rebuilds required.

- **`#![allow(clippy::useless_conversion)]` in `modes.rs`** — PyO3 0.22
  macro expansion emits a redundant `.into()` on `PyErr` in functions
  returning `PyResult<()>`; this is a known false positive across the whole
  module.

## Spec references

- `docs/impl/python-bindings.md` — Runtime modes section
- `docs/adr/ADR-022-hurray-python-runtime-compliance-modes.md`
- `hurray-python/COMPAT-MATRIX.md`
