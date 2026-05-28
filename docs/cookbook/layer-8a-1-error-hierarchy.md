# Layer 8a.1 — Python Bindings: Error Hierarchy + Panic Conversion

This entry covers the `hurray` Python exception hierarchy and the `catch_panic`
utility for converting Rust panics to typed Python exceptions.

## Exception class tree

```
ValueError
├── hurray.InvalidDescriptorError  — parse / validation errors
└── hurray.BufferError             — buffer size / alignment errors

NotImplementedError
└── hurray.UnsupportedError        — unsupported element type or layout

RuntimeError
└── hurray.InternalError           — unexpected Rust panics
```

> **Note:** `hurray.BufferError` (subclass of `ValueError`) is distinct from the
> Python built-in `builtins.BufferError`. The built-in is used by `__dlpack__` for
> element types outside the DLPack type enum (per the Array API Standard).
> `hurray.BufferError` is used for buffer size and alignment errors from the Rust core.

## Catching hurray exceptions

```python
import hurray

# Catch a specific hurray error
try:
    # ... operation that may fail ...
    pass
except hurray.InvalidDescriptorError as exc:
    print(f"bad descriptor: {exc}")

# Catch by base class (when the exact subtype doesn't matter)
try:
    pass
except ValueError as exc:          # catches InvalidDescriptorError and BufferError
    print(f"value error: {exc}")
except NotImplementedError as exc:  # catches UnsupportedError
    print(f"unsupported: {exc}")
except RuntimeError as exc:        # catches InternalError
    print(f"internal: {exc}")
```

## `catch_panic` — converting Rust panics to InternalError

The `errors::catch_panic` helper wraps a closure in `std::panic::catch_unwind`
and converts any panic to `hurray.InternalError`. Use it in Rust code that calls
into potentially-panicking operations:

```rust
use hurray::errors::catch_panic;

#[pyfunction]
fn risky_operation(py: Python<'_>) -> PyResult<i64> {
    catch_panic(|| {
        // ... call into Rust core that might panic on bad input ...
        Ok(compute_something())
    })
}
```

PyO3 already catches panics in `#[pyfunction]` wrappers and raises `RuntimeError`,
but `catch_panic` ensures the more specific `hurray.InternalError` subclass is raised
with the panic message embedded.

## Build notes

### `[lints.rust]` for PyO3 0.22 `create_exception!`

PyO3 0.22's `create_exception!` macro emits `cfg(feature = "gil-refs")` into the
destination crate's scope. Without explicit configuration this triggers an
`unexpected_cfgs` warning that `-D warnings` promotes to an error. The fix in
`hurray-python/Cargo.toml`:

```toml
[lints.rust]
unexpected_cfgs = { level = "warn", check-cfg = ['cfg(feature, values("gil-refs"))'] }
```

## Spec references

- `docs/impl/python-bindings.md` — § Error Handling
