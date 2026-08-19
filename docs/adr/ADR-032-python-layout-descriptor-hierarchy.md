# ADR-032: A Python layout descriptor hierarchy — `t.layout` returns an object, not a string

## Status

Proposed (2026-08-18)

Extends **ADR-031** (One Python `Tensor` class for every layout). Amends its § 1 —
`layout` returns a `hurray.Layout` instance rather than a string — and leaves § 2–§ 5
in force.

## Context

ADR-031 unified the Python tensor classes and gave `hurray.Tensor` a `layout`
property. That property returns a **string**, which is lossy: `hurray-core` models
layout as an enum whose variants carry data, and none of that data survives the
translation.

| Layout | Fields in `hurray-core` | Reachable from Python |
|---|---|---|
| COO | `nnz`, `is_sorted` | no |
| CSR / CSC | `nnz` | no |
| CSF | `nnz`, `mode_order` | no |
| Strided | `strides` | no |
| Tiled | `tile_shape`, outer/inner strides and layouts | no |
| Morton | `morton_bits` | no |
| BlockPaged | `page_size`, `num_pages`, `paged_axis`, `num_seqs`, `kv_role` | no |
| Composite | composition rule, combine op, member count | no |
| PrivateExtension | `extension_layout_id`, `extension_data` | no |
| Unknown | tag, raw bytes | no |

A string cannot carry any of it. Under the standing requirement that `hurray-python`
expose what `hurray-core` and `hurray-io` can express (issue #147), this is a hole.

It is also an inconsistency. Every *other* optional descriptor section already has a
Python class — the five quantization schemes, `Statistics`, `Shard` — accepted by the
`Tensor` constructor and returned by the matching getter. Layout is the only section
of the tensor descriptor with no Python representation of its own.

> **Note (non-normative):** This does not re-open `SparseTensor`. `CsrLayout` is a
> *descriptor*, not a tensor: it has no buffers, no protocols, and no `save()`.
> `isinstance(t.layout, CsrLayout)` encodes a distinction the wire format genuinely
> makes — the layout tag — where `isinstance(t, SparseTensor)` encoded one it does
> not. ADR-031 removed the hierarchy from the tensor; this ADR puts one where the
> format actually has it.

## Decision

### 1. A class hierarchy with a data-carrying base

```
hurray.Layout                        # base: holds the core descriptor
├── RowMajorLayout   ColMajorLayout
├── StridedLayout    TiledLayout     MortonLayout    HilbertLayout
├── CooLayout        CsrLayout       CscLayout       CsfLayout
├── BlockPagedLayout
├── CompositeLayout
├── PrivateExtensionLayout
└── UnknownLayout
```

The base MUST store the core `LayoutDescriptor` and implement `tag`, `name`,
`buffer_count`, `is_dense`, and `is_virtual` once; subclasses are typed façades
reading their own fields off it. This mirrors how `Statistics` and the quantization
schemes already wrap core types.

The base class also gives the binding a **legal fallback object**. `LayoutDescriptor`
is `#[non_exhaustive]`: when core gains a variant this build has not bound, `t.layout`
MUST return a bare `Layout` carrying `tag` and `name`. It MUST NOT be reported as
`UnknownLayout`, which would claim the tag is unrecognised when it is merely unbound —
destroying the signal a permissive reader depends on.

All layout classes MUST be immutable, with value equality and hashing, and a `__repr__`
of the form `CsrLayout(nnz=4)`.

### 2. `t.layout` returns an instance; the string moves to `.name`

`t.layout == "csr"` no longer works; `t.layout.name == "csr"` replaces it. A single
internal helper MUST produce that string, so `.name` and the layout named in error
messages (`__dlpack__`, `__array__`, `to_scipy`) cannot drift.

Layout classes MUST NOT define equality against strings. It would break the
hash/equality contract and keep a lossy comparison path alive indefinitely.

`layout` is read-only: assigning a layout would silently reinterpret existing buffers.

### 3. Component views stay on `Tensor`, plus a generic accessor

`values`, `indices`, `row_ptr`, `col_indices`, `row_indices`, `col_ptr`, and `nnz`
remain on `hurray.Tensor` exactly as ADR-031 § 2 defines them, including the
`AttributeError` discipline.

`hurray.Tensor` MUST additionally expose `buffer(index)`, returning a 1-D `uint8` view
of exactly the declared byte size of the buffer at that descriptor index. CSF has
`2·rank+1` buffers and block-paged has three; without a generic accessor their
parameters would become reachable while their buffers stayed unreachable, leaving
issue #147 unsatisfied for those layouts. `uint8` is the only honest element type for a
generic view — buffers within one tensor differ (values take the tensor's dtype, index
buffers are `uint64`, MXFP scales are `e8m0`) — and it cannot misreport a dtype. The
layout object tells the caller what each index means.

### 4. Authoring: `layout=` is accepted, and never inferred

`hurray.Tensor` MUST accept a `layout` keyword holding a `hurray.Layout` instance,
mirroring `quantization=`. Omitting it means row-major, as today. A non-`Layout` value
MUST raise `TypeError`.

A layout **string** MUST NOT be accepted. A string cannot carry `nnz` or `strides`, so
`layout="csr"` is a request that cannot be honoured; accepting it would create a second,
lossy authoring path.

**The layout object is a declaration; the buffers are evidence. They MUST agree**, and
the constructor MUST check three things:

| Tier | Check | Error |
|---|---|---|
| Shape | rank and shape constraints (CSR rank 2, CSF rank ≥ 3, `len(strides) == rank`, …) | `InvalidDescriptorError` |
| Buffer count | supplied buffers ≥ the layout's required count; quantization indices fall beyond it | `InvalidDescriptorError` |
| Buffer size | each buffer at least as large as the layout's parameters imply | `BufferError` |

**Never reinterpret, never infer.** `CooLayout(nnz=4)` supplied with a two-element
values buffer MUST raise — the descriptor MUST NOT be silently corrected to `nnz=2`,
and MUST NOT be accepted as given. Such a descriptor encodes and decodes cleanly and
hands the consumer an out-of-bounds read: the same class of failure the existing
quantization buffer-placement check exists to prevent. Over-sized buffers are permitted
(alignment and padding slack are legitimate); under-sized are not.

Consequently `nnz` MUST be a required argument on the sparse layout constructors.
Inference belongs to the array-shaped constructors — `hurray.sparse_coo`,
`hurray.from_scipy` — which are handed the arrays and can derive it. That yields two
clear paths: high-level constructors infer and are ergonomic; `Tensor(...)` requires the
descriptor to be stated and validates it.

**Round-trip obligation.** For every constructible layout, rebuilding a tensor from a
tensor's own `layout`, `quantization`, `statistics`, `shard`, and buffers MUST produce
an equal descriptor.

### 5. Units and enumerations

- Strides — `StridedLayout.strides` and the tiled layouts' outer and inner strides —
  are **in logical elements, signed, and may be negative or zero**. This MUST be stated
  in the binding documentation: a reader arriving from NumPy will otherwise assume
  bytes.
- Small closed enumerations (`kv_role`, block-table index type, tiled inner and outer
  layout tags) are exposed as lowercase strings, matching `device.kind` and
  `layout.name`. No new Python enum classes.
- `CompositeLayout` exposes the composition rule and the combine operation as **two**
  properties, with the combine operation `None` for non-overlay rules. They MUST NOT be
  flattened into one string, and the raw combine byte MUST NOT be exposed for rules
  where it means "not applicable".

### 6. Private, unknown, and composite layouts

`PrivateExtensionLayout` and `UnknownLayout` MUST be **separate** classes. "A private
layout I can identify by its extension id" and "a tag from a newer spec version I could
not parse" are different facts, and merging them erases the signal a permissive relay
needs.

- `PrivateExtensionLayout` — exposes tag, extension id, and extension data; buffer count
  is unknown. Constructible and accepted for authoring. Because the buffer count is
  unknown, the size tier of § 4 cannot run; that hole MUST be documented rather than
  left implicit.
- `UnknownLayout` — exposes tag and raw bytes; buffer count unknown. Constructible, so
  that a permissive relay can reconstruct a descriptor it decoded and write it back
  out. The Python constructor MUST reject any tag for which a named variant exists.
- `CompositeLayout` — readable in full: a composite head decoded from a stream MUST NOT
  have its layout misreported. Constructing a tensor with a composite layout MUST raise
  `hurray.UnsupportedError`; a composite head owns no buffers, which the Python `Tensor`
  cannot yet represent. ADR-031's deferral of *whether composites belong on the unified
  class* is unaffected; this answers only what `t.layout` reports.

> **Resolved (2026-08-19, issue #162):** core now rejects named and private tags in
> `UnknownLayout::new`, via a new `is_named_tag` helper shared with
> `validate_layout_tag_strict`. The finding as originally written follows.
>
> **Finding for `hurray-core`:** `UnknownLayout::new` currently rejects only the
> reserved tags `0x00` and `0xFF`, so a caller can construct an "unknown" layout on a
> tag that has a named variant — smuggling an unvalidated descriptor past every rank
> and buffer check. The Python constructor closes this per § 6; core should be
> tightened to the same rule.

## Alternatives Considered

**Layout objects holding a back-reference to their tensor, so `t.layout.row_ptr`
returns a view.** Rejected, and recorded here as rejected rather than deferred because
it is otherwise certain to be re-proposed.

It turns a metadata accessor into a buffer-lifetime anchor. `t.layout` would hold a
strong reference to the tensor, which holds the buffers, so

```python
layouts = {name: t.layout for name, t in stream}   # "just collecting metadata"
```

pins every tensor's buffers for the lifetime of that dictionary. In a format whose
premise is that the consumer decides when a zero-copy buffer is released, a *descriptor*
attribute that silently extends buffer lifetime is a defect, and nothing in its name
warns the caller. The `Tensor → layout → Tensor` reference cycle is a second cost of the
same choice: uncollectable without GC traversal support on every layout class, and
clearing the back-reference leaves a partially-dead object whose getters must then
raise.

It also asserts a containment the wire format does not have. The buffer table is a
sibling section of the layout section, not a child — which is precisely why quantization
descriptors reference buffers by index into that shared table. `t.quantization` returns
a descriptor carrying `scale_buffer_index` and *not* the scale buffer; a layout that
owned views would be the only descriptor section in the binding behaving differently.

Finally it does not deliver its own headline benefit without a second, worse change:
`dir(t.layout)` lists only what applies **only if** the accessors are removed from
`Tensor`, which repeals ADR-031 § 2 and takes `hasattr` with it. And `t.layout.values`
reads as though the layout has values. The tensor has values; the layout describes how
they are arranged.

**Keeping `layout` as a string and adding separate properties for the parameters**
(`t.nnz`, `t.strides`, `t.page_size`, …). Rejected: it moves every layout's fields onto
`Tensor`, multiplying exactly the `AttributeError`-guarded surface ADR-031 already had
to justify, and leaves no object to pass to `layout=` for authoring.

**Thirteen independent classes with no common base.** Rejected: `tag`, `name`,
`buffer_count`, `is_dense`, and `is_virtual` would be written thirteen times, the
`layout=` keyword would need a downcast chain instead of one type check, and — decisively
— there would be no legal object to return when core gains a variant the binding has not
yet bound.

**Accepting a layout string for authoring, alongside objects.** Rejected under § 4: a
string cannot carry the parameters, so it can only produce a descriptor that is wrong.

## Consequences

**Positive**

- Every layout parameter in `hurray-core` becomes reachable from Python, closing the
  layout half of issue #147.
- Layout joins quantization, statistics, and shard as a descriptor section with a Python
  class, so authoring and inspection read the same way across all four.
- `isinstance(t.layout, hurray.CsrLayout)` is a stronger discriminator than a string
  comparison, and `dir(t.layout)` is fully honest about parameters.
- The `layout=` keyword plus the three validation tiers make an inconsistent descriptor
  unauthorable, rather than merely undetected.

**Negative**

- **`t.layout` is not identity-stable**: each access builds a fresh object, so
  `t.layout is t.layout` is false. Value equality and hashing mitigate it. Caching one
  object per tensor would restore identity without a cycle (tensor → layout is safe;
  only layout → tensor is not) and is recorded below as deferred.
- **A large mechanical surface**: thirteen classes, each with a constructor, getters,
  `__repr__`, equality, a doc example, and a stub entry. The shared base is what keeps
  this mechanical rather than combinatorial.
- **`TiledLayout` is recursive** — a tiled layout may nest another — so the Python class
  is self-referential and its `__repr__` nests. Core bounds the depth, but the
  representation should be depth-aware rather than naively recursive.
- **`Layout.name` strings become public API.** They are already public through the
  current string property, so this is not new surface, but pinning them to a per-class
  property makes them harder to change later. They MUST be frozen in the binding
  documentation and MUST match the specification's layout names, so a third naming
  vocabulary does not emerge alongside the spec's and `hurray-inspect`'s.
- **Private and unknown layouts skip buffer-size validation**, because their buffer count
  is unknowable. This is a genuine hole in an otherwise complete check.

## Required Documentation Amendments

- `docs/impl/python-bindings.md` — a normative section defining the class hierarchy, the
  `layout=` keyword, the three validation tiers, the frozen string sets for `name` and
  the enumeration-valued properties, the statement that strides are in logical elements
  and signed, and the private/unknown size-validation hole.
- `docs/adr/ADR-031-*.md` § 1 — amended by reference: `layout` returns a `Layout`
  instance; the string is `layout.name`. § 2 through § 5 are unchanged.
- `docs/cookbook/hurray-python-sparse-scipy.md`, `docs/cookbook/composite-streaming.md`,
  and `docs/tutorials/python-interop-paths.md` — updated for `t.layout.name`.
- `hurray-inspect` — its layout rendering and the Python `repr` of a layout should agree
  field for field; any field `hurray-inspect` prints that the Python class omits is a
  remaining issue #147 gap.

No amendments under `docs/spec/` are required: this changes the Python binding's object
model, not the format.

## Open Questions Deferred

- **Named CSF and block-paged component accessors** (`pos` / `crd`, page table). The
  generic `buffer(index)` accessor covers them for now, and the layout object explains
  what each index holds.
- **Composite authoring from Python**, which requires representing a tensor that owns no
  buffers.
- **Caching the layout object per tensor** for identity stability.
- **Per-layout high-level constructors** beyond `sparse_coo` — whether each layout
  eventually gets an array-shaped constructor that infers its parameters.
