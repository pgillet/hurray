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

## Element Address

The linear element offset of element `[i_0, i_1, ..., i_{r-1}]` is:

```
offset = sum(i_k * strides[k] for k = 0, ..., r - 1)
```

When negative strides are present, the offset may be negative relative to the base
address at `byte_offset`. The `byte_offset` field MUST be set such that element
`[0, 0, ..., 0]` is addressable within the buffer. The physical address of every
valid element MUST lie within the buffer's bounds.

## Buffer Size

The minimum buffer size must cover every addressable element:

```
max_offset = sum(max(0, strides[k] * (shape[k] - 1)) for all k)
min_offset = sum(min(0, strides[k] * (shape[k] - 1)) for all k)
range_elements = max_offset - min_offset + 1
```

Buffer size in bytes depends on the element type (see `memory-layout.md`
§ Element Address Computation).

## Contiguity

A strided tensor is **dense** (contiguous with no gaps) if and only if the absolute
values of its strides form a permutation of the row-major strides for the same shape.
Implementations SHOULD NOT assume density without verifying this condition.

## Example

A rank-2 tensor with shape `[3, 4]` and strides `[-4, 1]` represents a row-major
matrix with the row order reversed. `byte_offset` points to what would be element
`[2, 0]` in a non-reversed matrix. Element `[2, 3]` is at offset
`2 * (-4) + 3 * 1 = -5`. The buffer MUST be large enough to cover the full range
from offset -8 to 0 (9 elements).
