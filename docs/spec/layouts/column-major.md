# Column-Major (Fortran Order) Layout — Hurray Format Specification

**Layout tag:** `0x02` | **Tier:** 1

> This section uses RFC 2119 key words: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT,
> SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

## Description

In column-major layout, elements are stored with the **first dimension varying
fastest**. Strides are implicit and MUST NOT be present in the descriptor for this
layout tag.

## Implicit Strides

```
strides[0] = 1
strides[i] = shape[i - 1] * strides[i - 1]    for i = 1, ..., rank - 1
```

All strides are in **logical elements**.

## Element Address

The linear element offset of element `[i_0, i_1, ..., i_{r-1}]` is computed
identically to row-major using the column-major strides above.

The byte address is computed from the element offset using the rules in
`memory-layout.md` § Element Address Computation.

## Buffer Size

Identical to row-major: `num_elements * element_byte_width` for whole-byte types,
or `ceil(num_elements / packing_factor)` for sub-byte types.

## Additional Descriptor Fields

None. This layout has no layout-specific fields in the tensor descriptor.

## Example

A rank-2 tensor with shape `[3, 4]` has implicit strides `[1, 3]`.
Element `[1, 2]` is at linear offset `1 * 1 + 2 * 3 = 7`.
