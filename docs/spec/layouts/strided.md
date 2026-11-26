# Strided Layout — Hurray Format Specification

**Layout tag:** `0x03` | **Tier:** 1

> This section uses RFC 2119 key words: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT,
> SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

## Description

The strided layout generalises both row-major and column-major by allowing an
arbitrary stride value per dimension. It is the most general of the simple
(non-tiled, non-curve) layouts.

## Additional Descriptor Fields

| Field | Type | Description |
|-------|------|-------------|
| `strides` | `int64[rank]` | Stride of each dimension in **logical elements**. |

## Stride Semantics

- A **positive** stride advances forward through the buffer.
- A **negative** stride advances backward. A negative stride on dimension `k` reverses
  that dimension: logical index 0 maps to the highest physical offset along that axis.
- A **zero** stride on dimension `k` means all indices along `k` map to the same
  physical element — a **broadcast** (virtual) dimension. Data is not physically
  replicated.

Negative and zero strides are valid. A conforming implementation MUST support them.

### Aliasing

A stride vector is **injective** for a given shape when no two distinct indices of that
shape share an element offset (§ Element Address). Otherwise the stride vector
**aliases**, and each set of indices sharing one offset is an **aliased index set**.

A zero stride on a dimension of extent greater than 1 is the mandated case: every index
along that axis belongs to one aliased set, which is exactly what makes the dimension a
broadcast. It is not the only case. A stride vector with no zero stride anywhere may still
alias — strides `[1, 1]` over shape `[2, 2]` address offsets `0, 1, 1, 2`, so indices
`[0, 1]` and `[1, 0]` share an element.

A producer MUST NOT emit an aliasing strided descriptor unless the values of the tensor it
describes are equal across every aliased index set. A consumer MUST accept an aliasing
descriptor, and MUST treat an aliased index set as carrying a single value.

The producer's obligation is one of fidelity, not of validity. An aliasing descriptor is
never invalid: shared physical storage *necessarily* yields equal observed values, so a
consumer always reads agreement across an aliased set, and the tensor it obtains is well
defined. The only thing that can be wrong is the producer's claim — if the source tensor's
values at those indices differ, the descriptor does not represent the tensor the producer
says it represents, and the disagreement is unrecoverable by the time a consumer sees it.
The zero stride is the case where the format has always relied on this: a broadcast axis is
emitted precisely because the values along it agree.

Three limits bound that obligation:

- It is evaluated on the tensor the descriptor describes, at the moment the descriptor is
  emitted. It is not a guarantee about the buffer's future; `buffer-protocol.md`
  § Zero-Copy Invariants already makes in-place mutation after handoff a protocol error,
  and this section adds no rule of its own about concurrent access.
- It binds a descriptor, not a buffer. One buffer MAY carry several descriptors with
  different strides. Each is read on its own terms, and an aliasing descriptor says nothing
  about what any other descriptor over the same bytes means.
- It says nothing about writing. This specification places no obligation on an
  implementation to materialise arbitrary values into an aliasing layout, because the
  values at an aliased index set are arbitrary until they are known. An implementation
  that materialises a tensor into a caller-chosen strided layout SHOULD refuse an aliasing
  destination.

This specification does not define how injectivity is decided. Conforming implementations
need not agree on which stride vectors they can prove injective.

> **Note (non-normative):** deciding injectivity over an index box is a bounded subset-sum
> problem in the general case, so an implementation will typically apply a sufficient
> condition and treat what it cannot prove as unproven rather than as aliasing. Mandating
> one such condition here would make both a more capable implementation and a simpler one
> non-conforming.

## Element Address

The linear element offset of element `[i_0, i_1, ..., i_{r-1}]` is:

```text
offset = sum(i_k * strides[k] for k = 0, ..., r - 1)
```

When negative strides are present, the offset may be negative relative to the base
address at `byte_offset`. The `byte_offset` field MUST be set such that element
`[0, 0, ..., 0]` is addressable within the buffer. The physical address of every
valid element MUST lie within the buffer's bounds.

## Buffer Size

The minimum buffer size must cover every addressable element:

```text
max_offset = sum(max(0, strides[k] * (shape[k] - 1)) for all k)
min_offset = sum(min(0, strides[k] * (shape[k] - 1)) for all k)
range_elements = max_offset - min_offset + 1
```

Buffer size in bytes depends on the element type (see `memory-layout.md`
§ Element Address Computation).

## Validity Constraints

This layout MUST NOT be used for rank-0 (scalar) tensors. See
`data-model.md` § Scalar Tensors.

## Contiguity

A strided tensor is **dense** (contiguous with no gaps) when its elements occupy every
offset in `min_offset ..= max_offset` (§ Buffer Size), each exactly once — equivalently,
when the strides are injective (§ Aliasing) and `range_elements` equals the tensor's
element count.

After discarding every dimension of extent 1 — such a dimension addresses one element, so
its stride is unobservable — a tensor with no zero extent is dense if and only if the
absolute values of the remaining strides form a permutation of the row-major strides of the
remaining extents. A tensor with a zero extent addresses no elements and is dense
vacuously. Implementations SHOULD NOT assume density without verifying this condition.

> **Note (non-normative):** density and injectivity are different properties, and a density
> check substitutes for an injectivity check in neither direction. Strides `[8, 1]` over
> shape `[3, 4]` are injective but not dense: the rows are padded, every element still has
> its own offset, and nothing is lost by writing through the layout — "not dense" does not
> imply "not writable". Strides `[1, 1]` over shape `[2, 2]` leave no gap in the offset
> range `0 ..= 2` and nevertheless alias, which is why a gapless offset range alone is not
> density.

## Example

A rank-2 tensor with shape `[3, 4]` and strides `[-4, 1]` represents a row-major
matrix with the row order reversed. `byte_offset` points to what would be element
`[2, 0]` in a non-reversed matrix. Element `[2, 3]` is at offset
`2 * (-4) + 3 * 1 = -5`. § Buffer Size gives `max_offset = 3` and `min_offset = -8`, so
the buffer MUST be large enough to cover the full range from offset -8 to 3 — 12 elements,
which is the tensor's element count because this layout is dense.
