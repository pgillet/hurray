# ADR-031: One Python `Tensor` class for every layout — retire `SparseTensor`

## Status

Proposed (2026-08-15). § 2 amended 2026-08-16: inapplicable accessors raise `AttributeError`, not `UnsupportedError` — see the reasoning in that section.

Amends the **Sparse Tensor Support** section of `docs/impl/python-bindings.md`, which
currently requires COO / CSR / CSC tensors to be exposed as a distinct
`hurray.SparseTensor` class.

## Context

`hurray-python` exposes two tensor classes: `hurray.Tensor` for dense tensors and
`hurray.SparseTensor` for COO, CSR, and CSC. The split is mandated normatively in
`docs/impl/python-bindings.md` § Sparse Tensor Support, but **no ADR records a reason
for it**. It was asserted rather than decided.

Four observations argue against keeping it.

### 1. The format does not make this distinction

There is no sparse tensor descriptor. Sparse is a `layout_tag` inside the ordinary
`TensorDescriptor` — same element type, same shape, same buffer table, same optional
quantization / shard / statistics sections. `hurray-core` models every layout as one
`LayoutDescriptor` enum.

A binding whose stated purpose is to expose what the format can express (issue #147)
should not invent a type boundary the wire format does not have. The split makes
`hurray-python` the only place in the stack where "sparse" is a *kind of object*
rather than a *property of a tensor*.

### 2. The taxonomy is already incomplete

`hurray.SparseFormat` models exactly three layouts: COO, CSR, CSC. But COO is not the
only multi-buffer layout, and sparse is not the only non-dense one:

| Layout | Buffers | Python class today |
|---|---|---|
| RowMajor, ColMajor, Strided, Tiled, Morton, Hilbert | 1 | `Tensor` |
| COO | 2 | `SparseTensor` |
| CSR, CSC | 3 | `SparseTensor` |
| CSF | `2·rank+1` | *(none — falls back to `Tensor`)* |
| BlockPaged | 3 | *(none — falls back to `Tensor`)* |
| Composite | 0 | *(none)* |

CSF and block-paged are as far from row-major as CSR is, yet they land in `Tensor`.
The boundary does not track any property of the format; it tracks which layouts
happened to be bound first. Every new layout re-opens the question of which class it
belongs to — a question that would not exist if there were one class.

### 3. The split leaks into every API

Because the two types are unrelated, each new capability must be threaded through
twice or make a dispatch decision:

- `hurray.save()` accepts `Tensor` only and raises `UnsupportedError` for
  `SparseTensor` — the whole of issue #156.
- `hurray.load()` must decide which class to construct from a decoded descriptor.
- `hurray.from_hurray_buffer` returns `Tensor`, so a sparse tensor that travels over
  the native protocol comes back as the wrong type (a leftover from #146).
- `__hurray_buffer__` is implemented twice, once per class.

The fix proposed for #156 was a shared "reconstruct into the right class" function.
That function is pure overhead created by this decision; unifying deletes it rather
than writing it.

### 4. The nearest analogue in this domain already unified

The ecosystem is genuinely split on this question:

| Library | Design |
|---|---|
| PyTorch | **one** `torch.Tensor` with a `.layout` attribute (`torch.strided`, `torch.sparse_coo`, `torch.sparse_csr`) |
| TensorFlow | separate `tf.SparseTensor` |
| SciPy | separate `scipy.sparse` matrix types |
| Apache Arrow | one `Array` hierarchy; layout is a property of the type |

SciPy's split is the one `hurray-python` currently mirrors, and it is the least
applicable: SciPy's sparse types are a *matrix* library with their own arithmetic,
not an interchange surface. PyTorch — the closest analogue, an ML tensor library where
layout is metadata rather than a different kind of object — unified, and calling
`.values()` on a dense tensor simply raises. The unified design is proven in practice.

## Decision

**`hurray-python` exposes exactly one tensor class, `hurray.Tensor`, for every
layout.** `hurray.SparseTensor` is removed.

### 1. Layout becomes a property

`hurray.Tensor` MUST expose a `layout` property reporting the descriptor's layout as a
string: `"row_major"`, `"col_major"`, `"strided"`, `"tiled"`, `"morton"`, `"hilbert"`,
`"coo"`, `"csr"`, `"csc"`, `"csf"`, `"block_paged"`, `"composite"`.

This replaces `SparseTensor.format`, which reported only the three sparse cases.

### 2. Layout-specific accessors live on the one class and raise `AttributeError`

`values`, `indices`, `col_indices`, `row_ptr`, `row_indices`, `col_ptr`, and `nnz`
MUST be available on `hurray.Tensor` and MUST raise **`AttributeError`** when the
tensor's layout does not define them.

This extends design decision **D10** — already applied within `SparseTensor`, where a
CSR tensor raises `AttributeError` for `.indices` — from the three sparse formats to
every layout.

`AttributeError` rather than `hurray.UnsupportedError`, because it is the only choice
that keeps `hasattr` honest:

```python
hasattr(coo_tensor, "row_ptr")     # False — genuinely not available
hasattr(csr_tensor, "row_ptr")     # True
```

`UnsupportedError` subclasses `NotImplementedError`, so it is raised *after* attribute
lookup succeeds; `hasattr` would return `True` for every accessor on every tensor and
callers would be forced to switch on `layout` instead. Feature detection matters more
under a unified class, not less: once the type no longer tells you what a tensor
supports, `hasattr` is what remains.

> **Note (non-normative):** PyTorch raises `RuntimeError` here, so it is not a model
> to follow on this point — its users check `.layout`. The cost of the unified class
> is that `dense.row_ptr` fails at call time rather than being absent from the type;
> `AttributeError` recovers as much of the "absent" behaviour as Python allows.

### 3. Dense-only protocols reject non-dense layouts by layout, not by type

`__dlpack__`, `__array__`, `__array_interface__`, and `to_torch` MUST raise
`hurray.UnsupportedError` (or `BufferError` where the protocol requires it) for
layouts they cannot represent. Previously the type system enforced this by making the
methods absent from `SparseTensor`; now the check is explicit and states the layout in
the message.

### 4. Constructors and interop keep their names

`hurray.sparse_coo`, `hurray.from_scipy`, and `to_scipy` remain, returning and
accepting `hurray.Tensor`. They are named for what they *do*, not for the class they
produce.

### 5. One protocol implementation

`__hurray_buffer__` is implemented once. `hurray.from_hurray_buffer`,
`hurray.load()`, and `hurray.save()` handle every layout uniformly, which closes
issue #156 and the sparse half of the #146 follow-up without any per-class dispatch.

## Alternatives Considered

**Keep both classes and give `SparseTensor` its own file I/O** (the original #156
plan). Rejected: it fixes one symptom and leaves the cause. Every future capability —
streaming (#157), extension types, new layouts — pays the same tax again, and the
class boundary still does not correspond to anything in the format.

**Keep both, and add classes for the missing layouts** (`CsfTensor`,
`BlockPagedTensor`, …). Rejected: it multiplies the problem. Each new layout in the
spec would require a new Python class and another round of protocol implementations,
and consumers would have to switch on type to do anything generic.

**One class, with layout-specific accessors on a namespace object**
(`t.sparse.row_ptr`). Rejected as a middle road that costs an extra concept without
removing the failure mode: `t.sparse` still has to raise for a dense tensor.

**Subclass: `SparseTensor(Tensor)`.** Rejected. It would fix the "`save()` rejects
sparse" symptom via inheritance while keeping the taxonomy question ("which layouts
get a subclass?") permanently open, and `isinstance` checks in user code would quietly
encode a boundary the format does not have.

## Consequences

**Positive**

- The Python object model matches the format's: a tensor has a layout, rather than a
  layout implying a type.
- Issue #156 is closed by construction — `save`/`load`/`from_hurray_buffer` stop
  caring what layout a tensor has.
- CSF, block-paged, and composite tensors gain the same accessors and protocol support
  as everything else, instead of falling back to a partial `Tensor`.
- One implementation of `__hurray_buffer__`, one of `__repr__`, one of the property
  surface.

**Negative**

- **A visible API break.** `hurray.SparseTensor` disappears; `isinstance(x,
  hurray.SparseTensor)` and `x.format` stop working. Pre-1.0 this carries no
  compatibility guarantee (see `docs/spec/versioning.md`), but it is a real change for
  anyone already using the sparse API.
- **Static analysis loses a signal.** `dense.row_ptr` still raises `AttributeError`,
  so `hasattr` and `getattr` behave as before, but a type checker or IDE can no longer
  tell from the class alone which accessors a given tensor supports — every accessor
  exists on every `Tensor`.
- **A wider class surface.** One class carries every layout's accessors, most of which
  raise for any given instance. Documentation must be explicit about which apply where.

## Required Documentation Amendments

- `docs/impl/python-bindings.md` § Sparse Tensor Support — rewritten around one class
  with a `layout` property; the `hurray.SparseTensor` requirement removed.
- `docs/cookbook/hurray-python-sparse-scipy.md` — updated for the unified class.
- `docs/tutorials/python-interop-paths.md` — the note recording that `save()` rejects
  `SparseTensor` is removed, since it no longer will.

No amendments under `docs/spec/` are required: this decision changes only the Python
binding's object model, not the format.

## Open Questions Deferred

- **Composite tensors.** A composite head owns no buffers and is a container rather
  than a tensor. Whether it belongs on the unified class at all, or needs its own
  representation, is left open — it is the one case where a separate type may be
  genuinely justified, and nothing in Python constructs composites today.
- **Layout-specific construction.** `hurray.sparse_coo` stays, but whether every
  layout eventually gets a matching constructor, or a single generic one taking a
  layout argument, is left to whoever binds the next layout.
