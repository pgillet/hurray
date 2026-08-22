# Integrating a Python Library with Hurray

You maintain a Python library with its own array type — a sparse linear algebra
package, an inference runtime, a columnar store. Someone asks it to read and write
Hurray tensors. There are four ways to do that, and they differ enormously in cost.

This tutorial is about **choosing**, not about mechanism. Each individual mechanism
is already documented in the [Cookbook](../cookbook/quickstart.md); what is harder to
find is which one your library should adopt, and what you sign up for by doing so.

Throughout, the running example is `sparselib` — a fictional stand-in for a
SciPy-shaped library. It has its own storage types, cares about sparse layouts, and
would rather not take a hard dependency on a format it merely supports. That makes it
a good lens: it exercises the trade-offs instead of the happy path.

## Choose a path

Answer four questions:

| Question | If yes | If no |
|---|---|---|
| Do you need **zero-copy**? | paths 2, 3 | path 1 is fine |
| Are producer and consumer **in the same process**? | paths 2, 3 | paths 1, 4 |
| Do you handle **Tier 2 or quantized** dtypes, or non-standard devices? | path 3 | path 2 suffices |
| Can you take a **dependency on `hurray`**? | paths 1, 2, 3 | path 4 |

Which gives:

| | Path | You write | You depend on | Zero-copy |
|---|---|---|---|---|
| 1 | Import `hurray` | Python | `hurray` | on the buffers, yes |
| 2 | Speak DLPack | nothing Hurray-specific | nothing | yes |
| 3 | Implement `__hurray__` | C/C++/Rust extension | `hurray-ffi` | yes, full fidelity |
| 4 | Parse the bytes | a reader/writer | nothing | your choice |

> **Note (non-normative):** Most integrations should start at path 2, discover it
> covers their case, and stop. Path 3 exists for what DLPack cannot express; path 4
> for when a dependency is unacceptable.

---

## Path 1 — Import `hurray`

The direct route. Your library converts between its own type and `hurray.Tensor`,
and uses `hurray` for file I/O.

For `sparselib`, whose arrays are already SciPy-compatible, conversion is nearly
free — `hurray.from_scipy` wraps CSR/CSC/COO component arrays without copying, and
`Tensor.to_scipy()` converts back:

```python
import hurray

sparse = hurray.from_scipy(matrix)     # zero-copy over the component arrays
assert sparse.layout.name == "csr"
assert sparse.nnz == matrix.nnz

matrix_again = sparse.to_scipy()       # back to scipy.sparse
```

Dense arrays go through `hurray.from_numpy` and `numpy.asarray`. Either kind
persists the same way — `save()` writes every buffer a tensor has, so a sparse
tensor's index arrays travel with its values:

```python
tensor = hurray.from_numpy(dense_array)
hurray.save("out.hrry", {"a": tensor})

loaded = hurray.load("out.hrry")["a"]
```

**What you get.** The full descriptor — quantization, statistics, shard — and the
file format, or `hurray.StreamWriter` / `hurray.StreamReader` for the streaming one
(see [Python: Streaming](../cookbook/hurray-python-streaming.md)).

**What it costs.** A hard dependency on `hurray`, and your users install a compiled
extension. For a library whose Hurray support is one feature among many, that is the
main objection — and path 2 exists precisely to avoid it.

---

## Path 2 — Speak DLPack, and write no Hurray code at all

DLPack is an independent, header-only ABI plus a `PyCapsule` protocol. It is not
part of Hurray, and Hurray does not own it — it is the same protocol NumPy, PyTorch,
JAX, and CuPy already implement.

If `sparselib`'s dense array type exposes `__dlpack__` and `__dlpack_device__`, then
**Hurray can already consume it** and `sparselib` ships nothing Hurray-specific:

```python
# In hurray-aware code, elsewhere — sparselib itself needs no changes.
import hurray

tensor = hurray.from_dlpack(sparselib_array)   # zero-copy, no copy of your data
hurray.save("out.hrry", {"a": tensor})
```

Going the other way, a `hurray.Tensor` exposes `__dlpack__` too, so your library
consumes it with the machinery you already have:

```python
arr = sparselib.asarray(tensor)     # via your existing from_dlpack support
```

**What you get.** Zero-copy in both directions, no dependency, no build changes.
Implementing `__dlpack__` is worth doing regardless of Hurray — it buys
interoperability with the whole array ecosystem at once.

**What it costs.** DLPack's type and device enums are narrower than Hurray's
descriptor. It cannot represent:

- Tier 2 element types — `int4`, `float8` variants, sub-byte packed types
- quantized tensors and their scale buffers
- `UNIFIED` / `PEER` / private memory classes and device tags
- sparse layouts, block-paged layouts, composite tensors

`hurray.Tensor.__dlpack__` raises `BufferError` for those rather than lying about
the payload. If your library only ever handles dense Tier 1 data on CPU or CUDA,
none of that matters and you are done.

---

## Path 3 — Implement `__hurray__`

When DLPack is too narrow, implement Hurray's native protocol on your own
type. This is for libraries with a compiled extension: you link `hurray-ffi` and hand
back a `PyCapsule`.

> This path is **Python-plus-C**, not Rust — there is no Rust counterpart to show,
> because a Rust producer would use `hurray-core` types directly rather than crossing
> a Python capsule boundary.

The contract, in full:

1. Expose `__hurray__(stream=None)` returning a `PyCapsule` named
   `"hurray_tensor"`.
2. The capsule pointer is a `HurrayBufferList` — build it with
   `hurray_buffer_list_new` and one `hurray_buffer_list_push` per buffer, in
   descriptor buffer-table order.
3. The capsule context carries your encoded `TensorDescriptor` and
   `HURRAY_C_ABI_VERSION`.
4. Consumers rename the capsule to `"used_hurray_tensor"` on consumption; your
   destructor calls `hurray_buffer_list_destroy` if it was never consumed.

Discovery is duck-typed. Nothing registers anything:

```python
if hasattr(obj, "__hurray__"):
    tensor = hurray.from_hurray(obj)
elif hasattr(obj, "__dlpack__"):
    tensor = hurray.from_dlpack(obj)        # narrower, but widely available
else:
    raise TypeError("no supported interchange protocol")
```

For `sparselib`, this is the path that makes its **sparse** types first-class: a COO
matrix is values plus coordinates, two buffers, which DLPack cannot express as one
object but a `HurrayBufferList` carries natively.

**What you get.** Full fidelity — every element type, every device tag and memory
class, quantization with its scale buffers, multi-buffer layouts.

**What it costs.** A compiled extension linking `hurray-ffi`, and the lifetime
discipline that comes with it. Two rules do most of the work:

- A handle from `hurray_buffer_list_get` is **borrowed**. Do not destroy it; the list
  owns it.
- `hurray_buffer_list_destroy` takes a pointer to your pointer and nulls it. Destroy
  the list exactly once; it frees every handle it holds.

See [Multi-Buffer Tensors](../cookbook/multi-buffer-tensors.md) for worked code and
[the C FFI guide](../impl/c-ffi.md) for the normative rules.

---

## Path 4 — Parse the bytes yourself

Nothing obliges you to link anything. The format is designed to be re-implemented:

- little-endian throughout
- self-delimiting — every section states its own length
- no back-references and no end-of-file index, so a reader can start work before it
  has seen the whole input

A pure-Python `.hrry` reader over `struct` and `numpy` is a reasonable weekend
project, and for a library that refuses new binary dependencies it may be the only
acceptable option.

**What you get.** Zero dependencies, full control, and the ability to read Hurray
data anywhere Python runs.

**What it costs.** You own conformance. The
[compliance checklist](../impl/compliance.md) is the contract, and the
[conformance vectors](https://github.com/pgillet/hurray/tree/main/conformance) are
how you check yourself against the reference implementation. You also inherit every
future format addition.

`hurray-inspect` is the tool to develop against — it decodes any descriptor field by
field, so you can compare your parser's interpretation against the reference one byte
at a time:

```bash
hurray-inspect weights.hrry
```

---

## Worked example: `sparselib` picks a path

`sparselib` handles float32 and float64 sparse matrices, has a compiled extension
already, and does not want a hard `hurray` dependency for a feature only some users
need.

**Ruling out.** Path 1 is rejected on the dependency. Path 4 is rejected as
disproportionate — it is a lot of surface to own for one feature.

**The choice.** Path 2 for dense arrays, since `sparselib` already implements
`__dlpack__` and it costs nothing. Path 3 for the sparse types, because a COO matrix
is inherently multi-buffer and DLPack has nowhere to put the coordinate array.

**The seam.** `hurray` becomes an optional dependency, imported lazily inside the
conversion functions:

```python
def to_hurray(matrix):
    try:
        import hurray
    except ImportError as exc:
        raise RuntimeError(
            "Hurray support requires the 'hurray' package: pip install hurray"
        ) from exc
    return hurray.from_scipy(matrix)
```

Users who never touch Hurray never install it; the sparse fast path stays zero-copy
for those who do.

## Where to go next

- [Quickstart](../cookbook/quickstart.md) — the smallest end-to-end example
- [Framework Interop](../cookbook/framework-interop.md) — NumPy, PyTorch, DLPack in detail
- [Python: Sparse Tensors with SciPy](../cookbook/hurray-python-sparse-scipy.md) — path 1 and 2 for sparse
- [Multi-Buffer Tensors](../cookbook/multi-buffer-tensors.md) — the protocol behind path 3
- [Authoring Quantized Tensors](../cookbook/authoring-quantized-tensors.md) — building and reading quantized descriptors
- [Compliance Checklist](../impl/compliance.md) — the contract for path 4
