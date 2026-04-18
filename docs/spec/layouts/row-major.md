# Row-Major (C Order) Layout — Hurray Format Specification

**Layout tag:** `0x01` | **Tier:** 1

> This section uses RFC 2119 key words: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT,
> SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

## Description

In row-major layout, elements are stored with the **last dimension varying fastest**.
Strides are implicit and MUST NOT be present in the descriptor for this layout tag.

## Implicit Strides

```
strides[rank - 1] = 1
strides[i] = shape[i + 1] * strides[i + 1]    for i = rank - 2, ..., 0
```

All strides are in **logical elements**.

## Element Address

The linear element offset of element `[i_0, i_1, ..., i_{r-1}]` in a row-major
tensor of rank `r` is:

```
offset = sum(i_k * strides[k] for k = 0, ..., r - 1)
```

The byte address is computed from the element offset using the rules in
`memory-layout.md` § Element Address Computation.

## Buffer Size

For a contiguous row-major tensor, the minimum buffer size is
`num_elements * element_byte_width` for whole-byte types, or
`ceil(num_elements / packing_factor)` for sub-byte types, where `num_elements` is
the product of all dimension sizes.

## Additional Descriptor Fields

None. This layout has no layout-specific fields in the tensor descriptor.

## Example

A rank-2 tensor with shape `[3, 4]` has implicit strides `[4, 1]`.
Element `[1, 2]` is at linear offset `1 * 4 + 2 * 1 = 6`.
