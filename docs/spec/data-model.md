# Data Model — Hurray Format Specification

> **Status:** Draft

> This section uses RFC 2119 key words: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT,
> SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

## Scope

This section defines the **logical data model** of a Hurray tensor: the abstract
description of its shape and element structure, independent of how it is stored
in memory or encoded on the wire. All other sections of the specification
operate on the model defined here.

---

## Tensor

A **tensor** is a multi-dimensional array of elements, all of the same element
type. A tensor is characterised by:

- a **rank** — the number of dimensions,
- a **shape** — the size of each dimension,
- an **element type** — defined in `element-types.md`.

The element type and shape fully determine the logical content of the tensor.
How the elements are mapped to memory is defined separately by the tensor's
**memory layout** (`memory-layout.md`).

---

## Rank

The **rank** of a tensor is a non-negative integer that counts the number of
dimensions. It is encoded as a `uint32` in the tensor descriptor (`metadata.md`).

- A rank-0 tensor is a **scalar**: it contains exactly one element and has no
  shape array.
- A rank-1 tensor is a **vector**.
- A rank-2 tensor is a **matrix**.
- Higher ranks are used for batched inputs, activation maps, and weight tensors
  in neural networks.

The maximum rank is **64**. A writer MUST NOT emit a descriptor with `rank > 64`.
A reader MUST reject a descriptor with `rank > 64`. A conforming implementation
MUST support tensors of rank `0` through `64` inclusive (see
`docs/adr/ADR-008-normative-rank-cap-64.md`).

> **Note (non-normative):** The `uint32` rank field can encode values above 64,
> but those values are reserved and will be rejected. The cap matches PyTorch's
> MAX_DIMS = 64 and bounds the shape-array size at 512 bytes (64 × 8), enabling
> stack allocation on the descriptor-parsing hot path.

---

## Shape

The **shape** of a tensor is an ordered sequence of `rank` dimension sizes,
indexed from `0`. Dimension `0` is the outermost (slowest-varying) dimension
in row-major order; dimension `rank − 1` is the innermost (fastest-varying).

Each dimension size is encoded as a `uint64`. A dimension size:

- MUST be greater than or equal to `0`.
- A size of `0` denotes an **empty dimension**: the tensor contains no elements
  along that axis. The total element count of a tensor with any zero-size
  dimension is `0`.
- The value `0xFFFFFFFFFFFFFFFF` (`UINT64_MAX`) is the **dynamic dimension
  sentinel** — see [§ Dynamic Dimensions](#dynamic-dimensions) below.
- Any other value in the range `[1, 0xFFFFFFFFFFFFFFFE]` is a **static**
  dimension size.

For a scalar tensor (`rank = 0`), the shape is the empty sequence; no shape
bytes are present in the descriptor.

### Total Element Count

For a tensor with no dynamic dimensions, the total number of logical elements
is:

```
element_count = product(shape[i] for i in 0 .. rank-1)
```

This product is `1` for a scalar. For a tensor with one or more zero-size
dimensions, `element_count = 0`.

A reader MUST NOT compute `element_count` for a tensor that has any dynamic
dimension without first resolving all dynamic dimensions to concrete values.

---

## Dynamic Dimensions

A dimension whose size is `0xFFFFFFFFFFFFFFFF` is **dynamic**: its concrete
value is not known at descriptor-write time and MUST be supplied by the reader
or the interchange protocol before the tensor's data buffer can be safely
accessed.

Rules for dynamic dimensions:

1. A reader MUST NOT compute buffer sizes, strides, or element counts for a
   tensor containing a dynamic dimension without first resolving it to a
   concrete value.
2. A writer that sets a dimension to `0xFFFFFFFFFFFFFFFF` MUST ensure the
   interchange channel provides a mechanism to communicate the resolved value
   before the data buffer is transferred (see `interchange.md`).
3. A quantization scheme that requires a statically known dimension size along
   its quantization axis (`axis`) MUST reject a descriptor whose `shape[axis]`
   is `0xFFFFFFFFFFFFFFFF`. The specific constraint is documented per scheme in
   `quantization/`.
4. Shard descriptors MUST NOT use dynamic dimensions (see `metadata.md`
   § Shard Section).

> **Note (non-normative):** Dynamic dimensions are intended for streaming and
> just-in-time dispatch scenarios, where a model producer does not know the
> batch size or sequence length at graph-compilation time. They are not a
> substitute for shape polymorphism at the type level.

---

## Element Type

The **element type** of a tensor is identified by a `uint8` `type_tag` defined
in `element-types.md`. All elements in a tensor share the same type tag.

The element type describes the **storage** representation of each element in
the data buffer. When a tensor is quantized (`HAS_QUANTIZATION` flag set), the
storage type is an integer or float8 type and the quantization descriptor
defines the mapping to real-valued elements. The storage type is always
orthogonal to the quantization scheme: `type_tag` never encodes quantization
semantics.

> **Note (non-normative):** A tensor MAY carry an optional Compound
> Annotation (see [`compound-types.md`](compound-types.md)) that groups its
> innermost dimension into a named or unnamed fixed-size tuple. The
> annotation does not change the tensor's bytes, shape, strides, or storage
> type; it only changes the consumer-facing view.

---

## Scalar Tensors

A scalar tensor has `rank = 0`. Its shape is empty. It contains exactly one
element. The data buffer size is `sizeof(element_type)` bytes (or a partial
byte for sub-byte types; see `element-types.md` § Sub-Byte Packing).

A scalar tensor MUST NOT carry a strided, tiled, or sparse layout descriptor;
only row-major (`0x01`) is permitted for scalars.

> **Note (non-normative):** Scalar tensors arise as outputs of reduction
> operations (e.g., loss values) and as single-element configuration parameters.

---

## Empty Tensors

A tensor is **empty** if `element_count = 0`. This occurs when at least one
dimension has size `0`. An empty tensor is valid; its data buffer has size `0`
bytes.

An empty tensor MUST still carry a complete, valid descriptor with a correct
element type, layout tag, buffer table, and any applicable quantization
descriptor. A reader MUST accept an empty tensor without treating it as an
error. A zero-length data buffer MAY be represented with a null pointer; the
64-byte alignment requirement does not apply to a zero-length buffer.

> **Note (non-normative):** The decision to permit empty tensors is recorded in
> `docs/adr/ADR-007-permit-empty-tensors.md`. The primary motivation is
> round-trip fidelity with PyTorch, NumPy, JAX, Apache Arrow, and DLPack, all
> of which permit zero-size dimensions.

---

## Relationship to Other Sections

- **`element-types.md`** defines the `type_tag` values, sub-byte packing rules,
  and the Tier 1 / Tier 2 type classification.
- **`memory-layout.md`** defines how the logical index space described by the
  shape is mapped to a linear buffer: strides, block shapes, sparse indices.
- **`metadata.md`** defines the binary encoding of rank, shape, and all other
  descriptor fields on the wire.
- **`quantization.md`** defines how storage elements are mapped to real-valued
  elements when `HAS_QUANTIZATION` is set.
- **`interchange.md`** defines how dynamic dimensions are resolved during
  runtime tensor exchange.

---

## Open Questions

All open questions in this section are resolved. See `docs/adr/ADR-007-permit-empty-tensors.md` (OQ-2) and `docs/adr/ADR-008-normative-rank-cap-64.md` (OQ-1).
