# ADR-029: hurray-python is interchange-first — drop the Array API conformance claim

## Status

Proposed (2026-08-06)

Supersedes **ADR-022** (hurray-python Runtime Compliance Modes).

## Context

`hurray-python` currently declares itself a **strict reference implementation of the
Python Array API Standard** for Tier 1 element types (see `docs/impl/python-bindings.md`
and ADR-022). A review of the actual surface and of the standard's conformance model
shows this claim is not tenable and does not serve the format's purpose:

1. **The claim is unmet.** `hurray-python` implements the Array API's *creation* and
   *inspection* surface (and returns the `hurray` module from `__array_namespace__`
   advertising version `2025.12`), but implements **none** of the mandatory compute
   core — no elementwise functions, reductions, manipulation, linear algebra,
   searching/sorting/set functions, indexing (`__getitem__`), or operator dunders.
   A consumer that obtains the namespace via `__array_namespace__()` and calls, e.g.,
   `xp.reshape`, `xp.sum`, `xp.matmul`, or uses `x[0]`, fails.

2. **Partial implementation is only sanctioned along designated seams.** The standard
   permits omitting the *optional extensions* (`linalg`, `fft` — "Each array library
   supporting this standard may, but is not required to, implement an extension") and
   *negotiating capabilities/dtypes/devices* (`__array_namespace_info__().capabilities()`).
   It does **not** sanction dropping the mandatory core while still presenting a
   conforming namespace. Conformance is defined operationally by the `array-api-tests`
   suite, which exercises the full specified surface.

3. **Exposing `__array_namespace__` without the core is actively harmful.**
   Array-agnostic consumers (scikit-learn, SciPy's array-api support, einops, …) treat
   `__array_namespace__` as the promise that the full core exists; a `hurray.Tensor`
   breaks them at runtime rather than being cleanly rejected.

4. **Compute is not hurray's purpose.** `hurray-python` is the **Python face of the
   Hurray interchange format** — a codec and zero-copy bridge (produce / consume /
   hand off), in the same spirit as the Python packages of other data formats. It is
   not, and should not become, a numerical library. Numerical work belongs to the
   frameworks the buffer is handed to (NumPy, PyTorch, JAX, …).

5. **DLPack is not the Array API.** DLPack is an independent, header-only ABI +
   PyCapsule protocol that the Array API merely adopts. `from_dlpack` works on any
   object exposing `__dlpack__` **without** requiring `__array_namespace__`. The
   valuable zero-copy interop hook and the conformance claim were never coupled;
   dropping the claim does not cost us DLPack reach.

6. **The runtime modes exist only to serve the claim.** ADR-022's strict/relaxed
   modes (`set_strict`/`is_strict`/`strict`/`relaxed`, `modes.rs`) exist for the sole
   purpose of gating `__array_namespace__` visibility by dtype tier (Tier 1 vs Tier 2 /
   quantized). With the claim removed, they gate nothing.

## Decision

`hurray-python` is positioned as **interchange-first: a codec and zero-copy bridge for
the Hurray format, not an Array API implementation.**

1. **Drop the Array API conformance claim.** `hurray-python` MUST NOT describe itself
   as an Array API implementation, reference implementation, or conforming namespace.

2. **Remove `__array_namespace__`.** `hurray.Tensor` MUST NOT implement
   `__array_namespace__` (for any dtype tier). The `hurray` module is not an Array API
   namespace.

3. **Remove the runtime compliance modes.** `set_strict`, `is_strict`, `strict`,
   `relaxed`, `StrictCtx`, `RelaxedCtx`, and the `_strict_mode` carrier are removed.
   ADR-022 is superseded. (Pre-1.0, no compatibility guarantee applies; see
   `docs/spec/versioning.md`.)

4. **Keep the interchange and producer/consumer surface**, which never depended on the
   claim:
   - Zero-copy interop protocols: `__dlpack__` / `__dlpack_device__`, `__array__` /
     `__array_interface__`, and the native `__hurray_buffer__` / `from_hurray_buffer`.
   - Structural/inspection surface on `Tensor`: `shape`, `dtype`, `device`, `ndim`,
     `size`, `T`.
   - Construction and ingest: `zeros`/`ones`/`full`/`empty`(+`_like`),
     `arange`/`linspace`/`eye`, `asarray`, `from_dlpack`, `from_numpy`, `from_torch`,
     `to_torch`, `from_scipy`, `save`/`load`.
   These are framed as **standalone interop protocols**, not as Array API surface.

5. **Dtype identity.** Tier 1 and Tier 2 / quantized dtypes remain first-class
   `hurray.dtype.*` objects. They are no longer described in terms of "Array API dtype"
   mapping; the NumPy/DLPack dtype correspondence is documented purely as an *interop*
   detail (what a given Hurray type becomes when handed to NumPy/PyTorch, and which
   types cannot cross a given bridge — e.g. `bool` over DLPack).

6. **Validation.** The conformance anchor for `hurray-python` is the shared **golden
   test-vector corpus** (`conformance/vectors/`, cross-checked Rust ↔ Python) plus the
   binding's own unit/integration tests. `array-api-tests` is **not** used — it targets
   a whole conforming namespace and presupposes the compute core, which is the wrong
   shape for a producer/consumer-only surface.

## Consequences

**Positive**

- The binding's advertised behaviour matches its actual behaviour; no consumer is
  misled by `__array_namespace__`.
- Smaller, more coherent surface; the modes machinery and its thread-safety caveats
  disappear.
- DLPack / `__array__` reach to NumPy/PyTorch/JAX/CuPy is fully retained.
- Positioning is honest and defensible: "Array-API-*interoperable* (via DLPack), not
  Array-API-*implementing*."

**Negative / cost**

- User-facing API removal: `__array_namespace__` and the `set_strict`/`relaxed` family
  are gone. Acceptable pre-1.0 (no compatibility guarantee), but must be called out in
  the changelog.
- A consumer that wants an Array API namespace from Hurray data must first hand the
  buffer to a real backend (zero-copy), e.g. `xp = array_namespace(np.from_dlpack(t))`.
  This is documented as the recommended pattern.

**Follow-up work (sequenced; each user-approved)**

1. This ADR.
2. Spec / impl docs: `docs/impl/python-bindings.md` (rewrite the normative Array API
   sections), `docs/impl/README.md`, `docs/spec/element-types.md` and
   `docs/spec/README.md` (reframe Tier-1 "Array API dtype" language as interop),
   `docs/SUMMARY.md`.
3. Code: remove `__array_namespace__` and `modes.rs`; keep interop; drop dependent
   tests.
4. Cookbook / examples: retire or rewrite `docs/cookbook/hurray-python-array-api.md`
   and `docs/cookbook/hurray-python-runtime-modes.md`, and the `examples/array_api.py`
   example; refresh `hurray-python/COMPAT-MATRIX.md`.

## Notes (non-normative)

The `docs/impl/python-bindings.md` **Rationale** section (added 2026-08-06) already
states the interchange-first motivation; this ADR makes it normative and reconciles the
surrounding requirements.
