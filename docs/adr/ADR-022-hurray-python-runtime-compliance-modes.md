# ADR-022: hurray-python Runtime Compliance Modes

## Status
Superseded by ADR-029

> **Note:** The runtime compliance modes described here existed solely to gate
> `__array_namespace__` visibility by dtype tier. ADR-029 drops the Array API
> conformance claim and removes both `__array_namespace__` and these modes. This ADR
> is retained for historical context.

## Context

`hurray-python` targets two audiences simultaneously:

1. **Array API consumers** — libraries (NumPy, PyTorch, JAX, SciPy, scikit-learn,
   Xarray, …) that accept any Python Array API Standard-conformant array. For these
   consumers, `hurray.Tensor` must behave exactly as the standard requires, including
   the requirement that `__array_namespace__` MUST NOT be implemented on tensors with
   non-Array-API dtypes (Tier 2, quantized).

2. **Hurray-native consumers** — code that deliberately uses the full Hurray type
   system: `int4`, `float8` variants, quantized types, non-standard memory layouts.
   These consumers do not need Array API conformance; they need access to the complete
   Hurray feature set without hitting artificial walls.

The Python Array API Standard clearly delineates the two cases — Tier 1 types are
Array-API-conformant; Tier 2 and quantized types are not — but does not prescribe how
an implementation must behave when a user tries to cross that line. Different
implementations have made different choices (raise, return a special object, do
nothing), creating interoperability confusion.

The project TODO records the need for: "hurray-python runtime modes: support two modes
— strict Array API compliance mode (Tier 1 types only, all Array API invariants
enforced) and standard-free mode (Tier 2 / quantized types exposed, Array API
constraints relaxed)."

An open question during Layer 8a planning (OQ-A) asked: for `hurray.Tensor` instances
with Tier 2 / quantized dtypes, should `__array_namespace__` be (a) absent (raises
`AttributeError`, `hasattr` returns `False`) or (b) present but raises `TypeError`?
The correct answer depends on the runtime-modes design: in strict mode (a) is correct;
in relaxed mode (b) is wrong — it must be present and functional.

A third concern is thread safety: `hurray-python` is used in multi-threaded inference
pipelines (Torch `DataLoader` workers, Triton server thread pools, asyncio coroutine
groups). A plain global flag (`hurray.config.strict = False`) would be unsafe across
threads.

## Decision

### Mode carrier: `contextvars.ContextVar`

The compliance mode is stored in a module-level `contextvars.ContextVar[bool]` named
`_strict_mode`, with a default value of `True` (strict). This is thread-safe and
coroutine-safe: each OS thread and each asyncio `Task` inherits a copy of the context
on spawn; changes in one thread do not affect others.

> **Note (non-normative):** Threads created with the raw `threading.Thread` API inherit
> the *default value* of the ContextVar (`True`, strict), not the caller's current
> context. This is standard Python `contextvars` behavior. Document it prominently in
> the user guide.

### Public API (reserved in Layer 8a, fully implemented in a later layer)

Four names are reserved in the `hurray` module namespace from Layer 8a onward:

| Name | Signature | Behaviour in Layer 8a |
|---|---|---|
| `hurray.set_strict` | `set_strict(strict: bool) -> None` | `set_strict(True)` is a no-op; `set_strict(False)` raises `NotImplementedError` |
| `hurray.is_strict` | `is_strict() -> bool` | always returns `True` |
| `hurray.strict` | context manager | always a no-op (already in strict mode) |
| `hurray.relaxed` | context manager | raises `NotImplementedError` |

The full implementation (allowing `set_strict(False)` and `relaxed()` to actually
switch mode) is deferred to a later layer. Reserving the names now prevents users or
third-party packages from squatting on them.

### OQ-A resolution: `__array_namespace__` visibility

| Tensor dtype | Mode | `__array_namespace__` | `hasattr(t, '__array_namespace__')` |
|---|---|---|---|
| Tier 1 | strict | present, returns `hurray` namespace | `True` |
| Tier 1 | relaxed | present, returns `hurray` namespace | `True` |
| Tier 2 / quantized | strict | **absent** | **`False`** |
| Tier 2 / quantized | relaxed | present, returns `hurray` namespace | `True` |

In strict mode, Tier 2 / quantized tensors MUST NOT expose `__array_namespace__`.
This satisfies the Array API Standard literally (the attribute does not exist) and
ensures `array-api-tests` conformance checks pass correctly.

In relaxed mode, `__array_namespace__` is present on all tensors and returns the
`hurray` module, which acts as a non-conformant extended namespace for Tier 2 types.
The user has explicitly opted out of Array API conformance guarantees for that scope.

### Implementation: `__getattribute__` override on `Tensor`

To make `hasattr(tier2_tensor, '__array_namespace__')` return `False` in strict mode
while keeping a single `Tensor` class, `Tensor` MUST override `__getattribute__` in
PyO3:

```python
# Pseudo-code; actual implementation is in Rust via PyO3 #[pymethods]
def __getattribute__(self, name):
    if name == '__array_namespace__' and not is_tier1_dtype(self.dtype):
        if is_strict():
            raise AttributeError(
                f"hurray.Tensor with dtype {self.dtype.name!r} is a Tier 2 type "
                f"and does not implement __array_namespace__ in strict mode. "
                f"Use `with hurray.relaxed(): ...` to access Hurray-native features."
            )
    return type(self).__getattribute__(self, name)
```

The fast-path early return (`name not in GATED_ATTRIBUTES`) MUST be placed before the
mode check to minimise overhead on non-gated attribute accesses. `GATED_ATTRIBUTES` is
a small frozen set; initially `{'__array_namespace__'}`.

A private helper `is_tier1_dtype(dtype) -> bool` MUST be factored out from the gate
check. Layer 8b+ will call it from multiple sites.

### Scope of the mode: narrow

The compliance mode gates **exactly two things**:

1. **Tier 2 / quantized dtype admission** through Array-API-shaped construction APIs
   (`hurray.zeros`, `hurray.ones`, `hurray.asarray`, etc.). In strict mode these raise
   `hurray.UnsupportedError` for non-Tier-1 dtypes. In relaxed mode they succeed.
2. **`__array_namespace__` visibility** on Tier 2 / quantized tensor instances, per
   the table above.

The mode does **not** affect:

- `size` returning `None` for dynamic dimensions (correct Array API behavior, not a
  constraint to relax).
- `T` raising `ValueError` for non-rank-2 tensors (Array API specification).
- `shape` returning `Tuple[Optional[int], ...]` (Array API specification).
- DLPack `BufferError` for element types outside the DLPack type enum (structural
  limitation of DLPack, not a compliance constraint).
- Error hierarchy or exception semantics.

These invariants hold in both modes. They are properties of well-formed operations,
not of compliance scope.

The namespace object returned by `__array_namespace__()` for Tier 1 tensors is the
same `hurray` module in both modes. It is not "extended" or restricted based on mode.
Tier 2 entry is through bare `hurray` module functions (e.g., `hurray.zeros`), not
through the namespace returned by `__array_namespace__`.

## Alternatives Considered

**Option 1a: Plain module-level global flag.** `hurray.config.strict = False` sets a
global boolean. Rejected: not thread-safe. Two Torch DataLoader worker threads can
race on the flag. In a Triton inference server, one request handler's mode flip leaks
to concurrent handlers. Unsafe by default in all multi-threaded ML environments.

**Option 2: Tensor subclass (`hurray.RawTensor`).** Two classes: `hurray.Tensor`
(strict, Tier 1, full Array API) and `hurray.RawTensor` (standard-free, all types).
Rejected: contradicts the user's intent for a simple global setting, and requires
every consumer library to widen `isinstance` checks. Makes `from_numpy` ambiguous
when the dtype is `int4`.

**Option 3: Per-instance `__getattr__` dispatch.** The method is absent from the
class definition; `__getattr__` raises `AttributeError` for Tier 2 in strict mode.
Rejected: `__getattr__` only fires on attribute-not-found; it cannot conditionally
expose an attribute that exists on the class. Furthermore, the mode belongs to the
*call site*, not the *tensor instance* — a Tier 2 tensor created in relaxed mode
should not carry `__array_namespace__` for its entire lifetime. `__getattribute__`
is the correct hook.

**`(b)` for OQ-A: `__array_namespace__` present, raises `TypeError` for Tier 2.**
Rejected: the Array API conformance test suite (`array-api-tests`) probes
`hasattr(t, '__array_namespace__')`. Returning `True` and raising on call makes the
tensor claim Array API capability and then crash the consumer at the worst moment.
Worse than absent.

## Consequences

- **Layer 8a ships strict-mode only.** `set_strict(False)` and `relaxed()` raise
  `NotImplementedError`. This is documented in `docs/impl/python-bindings.md`.
- **`Tensor.__getattribute__` is overridden from day one.** The strict-mode gate for
  `__array_namespace__` is active; the relaxed-mode branch is a no-op placeholder.
  Adding relaxed mode later requires no public API change.
- **`is_tier1_dtype(dtype) -> bool` is a first-class internal helper.** It is
  factored out in Layer 8a and reused in Layer 8b+ wherever Tier-2 behavior diverges.
- **TODO.md item "hurray-python runtime modes"** is resolved by this ADR for
  architecture; the implementation of the relaxed path is a separate tracked layer.
- `docs/impl/python-bindings.md` § "Tier 2 and quantized types" MUST be amended to
  reference this ADR and describe strict vs relaxed behavior. A new § "Runtime modes"
  MUST be added.
- Threads created with raw `threading.Thread` always enter strict mode regardless of
  the spawning thread's mode. This SHOULD be documented in the user guide.
