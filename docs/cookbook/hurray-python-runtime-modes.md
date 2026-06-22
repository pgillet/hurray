# hurray-python Runtime Modes

`hurray` ships in **strict mode** by default. Strict mode enforces full Python
Array API Standard compliance — only Tier 1 element types are admitted through
the Array API surface. Relaxed mode allows Tier 2 / quantized types through.

## Checking and setting the mode

```python
import hurray

hurray.is_strict()        # True (default)
hurray.set_strict(False)  # switch to relaxed
hurray.is_strict()        # False
hurray.set_strict(True)   # back to strict
```

## Context managers

`hurray.StrictCtx` / `hurray.RelaxedCtx` scope mode changes to a `with` block.
The prior mode is restored exactly on exit, even if an exception is raised:

```python
import hurray

hurray.set_strict(False)

with hurray.StrictCtx():
    assert hurray.is_strict() == True   # strict inside

assert hurray.is_strict() == False  # restored

# Nesting works correctly.
with hurray.RelaxedCtx():
    with hurray.StrictCtx():
        assert hurray.is_strict() == True
    assert hurray.is_strict() == False  # back to relaxed
```

## Factory helpers

`hurray.strict()` and `hurray.relaxed()` return the same context managers in a
slightly more readable form:

```python
import hurray

with hurray.relaxed():
    # Tier 2 / quantized types are allowed here.
    pass

with hurray.strict():
    # Full Array API compliance enforced.
    pass
```

## asyncio Task isolation

The mode is backed by a `contextvars.ContextVar`, so each asyncio Task inherits
an independent copy. Changes in one task do not affect others:

```python
import asyncio
import hurray

async def task_a():
    hurray.set_strict(False)
    await asyncio.sleep(0)          # yield to task_b
    assert hurray.is_strict() == False

async def task_b():
    assert hurray.is_strict() == True  # unaffected by task_a

asyncio.run(asyncio.gather(task_a(), task_b()))
```

> **Note:** threads created with `threading.Thread` always start in strict mode
> (the ContextVar default), regardless of the spawning thread's current mode.

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

- **`ContextVar` over `thread_local!`** — asyncio Tasks share an OS thread but
  have independent `contextvars` copies; a Rust `thread_local!` would bleed
  mode changes across Tasks.

## Spec references

- `docs/impl/python-bindings.md` — Runtime modes section
- `docs/adr/ADR-022-hurray-python-runtime-compliance-modes.md`
- `hurray-python/COMPAT-MATRIX.md`
