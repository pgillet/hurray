# Python: Layout Descriptors

`t.layout` returns a `hurray.Layout` object, not a string (ADR-032). The wire
format models a layout as a tag plus that layout's parameters — `nnz`, `strides`,
`page_size`, `mode_order` — and a string carries the tag while throwing the rest
away. The object carries all of it, compares by value, and can be handed straight
back to the `hurray.Tensor` constructor.

The string is still there, as `layout.name`.

## Reading a layout

```python
import hurray

t = hurray.Tensor(bytes(16), hurray.float32, [4])

print(repr(t.layout))            # RowMajorLayout()
print(t.layout.name)             # row_major
print(hex(t.layout.tag))         # 0x1
print(t.layout.buffer_count)     # 1
print(t.layout.is_dense)         # True

isinstance(t.layout, hurray.RowMajorLayout)   # True
isinstance(t.layout, hurray.Layout)           # True — every layout shares a base
```

`isinstance` is the discriminator, and it encodes a distinction the wire format
genuinely makes: the layout tag.

## The class hierarchy

```
hurray.Layout                        # base: tag, name, buffer_count, is_dense, is_virtual
├── RowMajorLayout   ColMajorLayout
├── StridedLayout    TiledLayout     MortonLayout    HilbertLayout
├── CooLayout        CsrLayout       CscLayout       CsfLayout
├── BlockPagedLayout
├── CompositeLayout
├── PrivateExtensionLayout
└── UnknownLayout
```

`hurray.Layout` itself is not constructible — there is no layout that is only "a
layout". It is returned directly in exactly one case: a layout tag this build of
`hurray` does not yet bind, where `tag` and `name` still work. That is
deliberately **not** `UnknownLayout`, which means "the tag was unrecognised" — a
different and load-bearing fact for a permissive reader.

## Authoring: `layout=`

```python
import struct
import hurray

# A 2x2 CSR matrix holding [[5.0, 0.0], [0.0, 7.0]].
csr = hurray.Tensor(
    struct.pack("2f", 5.0, 7.0),          # buffer 0 — values
    hurray.float32,
    [2, 2],
    aux_buffers=[
        struct.pack("2Q", 0, 1),           # buffer 1 — col_indices
        struct.pack("3Q", 0, 1, 2),        # buffer 2 — row_ptr
    ],
    layout=hurray.CsrLayout(nnz=2),
)

csr.layout == hurray.CsrLayout(nnz=2)     # True
csr.nnz                                   # 2
```

Omitting `layout` means row-major, as before.

### The layout is a declaration; the buffers are evidence

They must agree. The constructor checks three tiers:

| Tier | Check | Error |
|---|---|---|
| Shape | rank and shape constraints (CSR rank 2, CSF rank ≥ 3, `len(strides) == rank`) | `hurray.InvalidDescriptorError` |
| Buffer count | enough buffers for the layout; quantization indices fall beyond them | `hurray.InvalidDescriptorError` |
| Buffer size | each buffer at least as large as the layout's parameters imply | `hurray.BufferError` |

Nothing is inferred and nothing is reinterpreted:

```python
hurray.Tensor(
    struct.pack("2f", 5.0, 7.0),          # two values...
    hurray.float32,
    [2, 2],
    aux_buffers=[struct.pack("8Q", *range(8))],
    layout=hurray.CooLayout(nnz=4),       # ...but the layout declares four
)
# hurray.BufferError: buffer 0 (values) is 8 bytes, but this coo layout implies at least 16
```

The descriptor is not quietly corrected to `nnz=2`, and it is not accepted as
given — it would encode and decode cleanly and hand the consumer an
out-of-bounds read. Over-sized buffers *are* allowed: alignment and padding slack
are legitimate.

This is why `nnz` is a required argument on the sparse layout constructors.
Inference belongs to the array-shaped constructors — `hurray.sparse_coo`,
`hurray.from_scipy` — which are handed the arrays and can derive it honestly.

### A layout string is not accepted

```python
hurray.Tensor(bytes(16), hurray.float32, [4], layout="csr")
# TypeError: layout must be a hurray.Layout instance (e.g. hurray.CsrLayout(nnz=4)), got str
```

A string cannot carry `nnz` or `strides`, so `layout="csr"` is a request that
cannot be honoured. Accepting it would open a second, lossy authoring path.

## Value semantics

Layout objects are immutable, compare by value, and hash:

```python
hurray.CsrLayout(nnz=4) == hurray.CsrLayout(nnz=4)     # True
hurray.CsrLayout(nnz=4) == hurray.CsrLayout(nnz=5)     # False
len({hurray.CooLayout(nnz=1), hurray.CooLayout(nnz=1)})  # 1

t.layout is t.layout                                    # False — a fresh object
t.layout == t.layout                                    # True
t.layout == "row_major"                                 # False — always
```

`t.layout` is read-only: assigning one would silently reinterpret the buffers the
tensor already holds. And a layout never equals a string — keeping that
comparison alive as a special case would break the hash/equality contract and
leave the lossy path open indefinitely.

## The parameters a string could not carry

```python
hurray.StridedLayout([4, 1]).strides            # (4, 1)
hurray.MortonLayout([3, 3]).morton_bits         # (3, 3)
hurray.HilbertLayout(3, 2).hilbert_order        # 3
hurray.CooLayout(nnz=7, is_sorted=True).is_sorted   # True
hurray.CsfLayout(nnz=5, mode_order=[2, 0, 1]).mode_order   # (2, 0, 1)
```

**Strides are in logical elements, signed, and may be negative or zero** — not in
bytes, as NumPy's are. That applies to `StridedLayout.strides` and to the tiled
layouts' `outer_strides` and `inner_strides`.

Small closed enumerations are lowercase strings, matching `device.kind` and
`layout.name`:

```python
paged = hurray.BlockPagedLayout(
    page_size=16, num_pages=64, paged_axis=0, num_seqs=2,
    kv_role="key", layer_index=3, block_table_index_type="uint32",
)
paged.kv_role                    # 'key'
paged.block_table_index_type     # 'uint32'
```

A composite head keeps its rule and its combine operation as two properties:

```python
overlay = hurray.CompositeLayout("overlay", member_count=3, combine_op="add")
overlay.composition_rule                            # 'overlay'
overlay.combine_op                                  # 'add'
hurray.CompositeLayout("partition", 2).combine_op    # None — it does not apply
```

They are not flattened into one string, because for a partition or a group the
operation is not merely unset: it has no meaning.

## Reaching buffers that have no named accessor

`values`, `indices`, `row_ptr`, `col_indices`, `row_indices` and `col_ptr` cover
COO, CSR and CSC. CSF has `2 * rank + 1` buffers and block-paged has three, none
of them named. `t.buffer(index)` reaches any of them, and the layout object says
what each index holds:

```python
csf = hurray.Tensor(
    struct.pack("4f", 1.0, 2.0, 3.0, 4.0),
    hurray.float32,
    [2, 3, 4],
    aux_buffers=[
        struct.pack("2Q", 0, 2),        # pos_0
        struct.pack("2Q", 0, 1),        # crd_0
        struct.pack("3Q", 0, 2, 3),     # pos_1
        struct.pack("3Q", 0, 2, 1),     # crd_1
        struct.pack("4Q", 0, 1, 2, 4),  # pos_2
        struct.pack("4Q", 1, 3, 0, 2),  # crd_2
    ],
    layout=hurray.CsfLayout(nnz=4, mode_order=[0, 1, 2]),
)

csf.buffer_count          # 7  (2 * rank + 1)
csf.buffer(6).shape       # (32,) — the leaf crd, 4 uint64 entries
csf.buffer(6).dtype       # hurray.uint8
```

The view is 1-D `uint8` covering exactly the buffer's declared byte size. `uint8`
is the only honest element type for a generic view: the buffers of one tensor do
not share a dtype — values take the tensor's dtype, index buffers are `uint64`,
MXFP scales are `e8m0` — and `uint8` cannot misreport any of them.

## Private, unknown, and composite

`PrivateExtensionLayout` and `UnknownLayout` are separate classes. "A private
layout I can identify by its extension id" and "a tag from a newer spec version I
could not parse" are different facts, and a permissive relay needs both.

```python
private = hurray.PrivateExtensionLayout(0xF0, extension_layout_id=7, extension_data=b"\x01")
unknown = hurray.UnknownLayout(0x0C, b"\x01\x02")

private.name == unknown.name == "extension"   # True — isinstance separates them
private.extension_layout_id                   # 7
unknown.raw_bytes                             # b'\x01\x02'
```

`UnknownLayout` is constructible so a relay can rebuild what it decoded and write
it back out. Its constructor rejects any tag that has a named class:

```python
hurray.UnknownLayout(0x07)
# ValueError: tag 0x07 is the csr layout, not an unknown one; use hurray.CsrLayout instead
```

Calling a known tag "unknown" would smuggle a descriptor past every rank and
buffer check the named class applies.

> **Known gap:** because a private or unknown layout's buffer count is not
> knowable, the buffer-count and buffer-size tiers cannot run for it. Nothing in
> such a descriptor says how many buffers it needs or how large they should be.

A `CompositeLayout` is readable in full, so a composite head decoded from a
stream reports its own layout truthfully. Building a `hurray.Tensor` with one
raises, because a composite head owns no buffers:

```python
hurray.Tensor(bytes(16), hurray.float32, [4], layout=hurray.CompositeLayout("group", 2))
# hurray.UnsupportedError: a composite layout cannot be given to hurray.Tensor:
# a composite head owns no buffers, which this class cannot represent
```

## Round-tripping a descriptor

A tensor's own layout goes straight back into the constructor, which is what lets
a relay read a descriptor and write an equal one:

```python
rebuilt = hurray.Tensor(
    values_bytes,
    original.dtype,
    list(original.shape),
    aux_buffers=[col_indices_bytes, row_ptr_bytes],
    layout=original.layout,
    quantization=original.quantization,
    statistics=original.statistics,
    shard=original.shard,
)
rebuilt.layout == original.layout    # True
```

## Runnable example

```
python hurray-python/examples/layouts.py
```

## See also

- [Python: Sparse Tensors and SciPy](hurray-python-sparse-scipy.md) — the
  array-shaped constructors that *do* infer `nnz`.
- [Python: Tensor Construction](hurray-python-construction.md)
- [Layout Descriptors (Rust)](layer-3-layout-descriptors.md) — the core types
  these classes wrap.
