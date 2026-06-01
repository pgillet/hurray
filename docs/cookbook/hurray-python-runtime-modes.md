# hurray-python Runtime Modes

`hurray` ships in **strict mode** by default. Strict mode enforces full Python
Array API Standard compliance — only Tier 1 element types are admitted through
the Array API surface. Relaxed mode (Tier 2 / quantized types) is reserved for
a future release.

## Checking and setting the mode

```python
import hurray

hurray.is_strict()       # True
hurray.set_strict(True)  # no-op

try:
    hurray.set_strict(False)  # raises NotImplementedError (not yet implemented)
except NotImplementedError as e:
    print(e)
```

## Context managers

`hurray.StrictCtx` and `hurray.RelaxedCtx` let you scope mode changes to a
`with` block:

```python
import hurray

with hurray.StrictCtx():  # no-op context manager, reserves the name
    pass

try:
    with hurray.RelaxedCtx():  # raises NotImplementedError (not yet implemented)
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
python examples/hello_hurray.py

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
