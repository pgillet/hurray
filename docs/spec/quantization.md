# Quantization — Hurray Format Specification

> **Status:** Draft

> This section uses RFC 2119 key words: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT,
> SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

## Scope

This section defines the binary encoding of the **quantization descriptor**: the
opaque payload that appears inside the Quantization Section of a tensor descriptor
(see `metadata.md` § Quantization Section). It specifies the set of supported
quantization schemes, the wire format for each scheme's parameters, and the
normative dequantization formula that a reader MUST apply to recover real-valued
elements from the quantized storage buffer.

A Hurray **quantized tensor** has two parts:

1. A **storage type** — an integer or float8 type tag defined in `element-types.md`
   that describes how each element is encoded in the tensor's data buffer.
2. A **quantization descriptor** — the subject of this file, which describes how
   to map those storage values to real-valued elements.

> **Note (non-normative):** The tensor descriptor's `type_tag` field always refers
> to the storage type. Hurray does not allocate separate type tags for quantized
> formats; a tensor is quantized if and only if the `HAS_QUANTIZATION` flag
> (bit 0 of `flags`) is set in the tensor descriptor. This separation keeps the
> type system orthogonal to the quantization scheme space.

---

## Relationship to Other Sections

- **`metadata.md`** defines the Quantization Section framing. The section is
  present if and only if the `HAS_QUANTIZATION` flag (bit 0) is set in the tensor
  descriptor. It consists of a `uint32` `quantization_length` prefix followed by
  `quantization_length` bytes of payload. This file defines the contents of those
  payload bytes.
- **`element-types.md`** defines the storage type tags (`type_tag` values) that
  may appear in a quantized tensor descriptor. Each scheme below lists the set of
  storage types it accepts.
- **`memory-layout.md`** defines the buffer table: an ordered list of buffer
  handles indexed from `0`. Several schemes store their per-block scale and
  zero-point arrays in buffers other than buffer 0; those schemes reference those
  buffers by index.

---

## Descriptor Header

Every quantization descriptor begins with a **fixed 4-byte header**.

| Offset | Field | Type | Description |
|--------|-------|------|-------------|
| 0 | `scheme_tag` | `uint8` | Identifies the quantization scheme. Values are assigned below. |
| 1 | `scheme_version` | `uint8` | Version of the scheme-specific encoding. Current value for all schemes defined here: `0x01`. |
| 2 | `flags` | `uint16` | Scheme-specific flags bitmask. Reserved bits MUST be `0`. |

All multi-byte fields in the quantization descriptor MUST be encoded in
little-endian byte order.

Immediately following the 4-byte header, **scheme-specific fields** are encoded.
The total length of the descriptor (header plus scheme-specific fields, plus any
trailing padding chosen by the writer) MUST equal the `quantization_length`
prefix defined in `metadata.md`.

A reader MUST dispatch on `scheme_tag` after reading the first byte. A reader
that does not recognise `scheme_tag` MUST reject the tensor descriptor, unless
operating in permissive mode. In permissive mode, the reader MAY skip past the
descriptor using the `quantization_length` prefix but MUST NOT dereference the
tensor data buffer.

A reader MUST reject a descriptor whose `scheme_version` exceeds the highest
version defined in this specification for the given `scheme_tag`.

A reader MUST reject a descriptor with any reserved `flags` bit set.

A reader MUST NOT read beyond `quantization_length` bytes when parsing the
descriptor.

---

## Scheme Tag Space

| Range | Allocation |
|-------|------------|
| `0x00` | Reserved (invalid) |
| `0x01` – `0x3F` | Tier 1 schemes (MUST be supported by conforming implementations that advertise quantization support) |
| `0x40` – `0x7F` | Tier 2 schemes (OPTIONAL); range `0x60`–`0x7F` reserved for future nested/composite schemes |
| `0x80` – `0xEF` | Reserved for future specification versions |
| `0xF0` – `0xFE` | Implementation-private extension schemes |
| `0xFF` | Reserved (invalid) |

A reader MUST reject a descriptor whose `scheme_tag` is `0x00` or `0xFF`.

Tags in the range `0x80` – `0xEF` MUST NOT be used by any implementation; they
are reserved for future specification versions.

Tags in the range `0xF0` – `0xFE` MAY be used by implementations for private
schemes. Tensors using private scheme tags MUST NOT be exchanged between
independent implementations unless both parties have agreed on the semantics
out of band.

### Assigned Scheme Tags

| Scheme | Tag | Tier | File |
|--------|-----|------|------|
| Per-tensor affine | `0x01` | 1 | [`quantization/per-tensor-affine.md`](quantization/per-tensor-affine.md) |
| Per-channel affine | `0x02` | 1 | [`quantization/per-channel-affine.md`](quantization/per-channel-affine.md) |
| Per-block affine | `0x03` | 1 | [`quantization/per-block-affine.md`](quantization/per-block-affine.md) |
| NF4 (NormalFloat4) | `0x04` | 2 | [`quantization/nf4.md`](quantization/nf4.md) |
| MXFP (OCP Microscaling) | `0x05` | 2 | [`quantization/mxfp.md`](quantization/mxfp.md) |

---

## Buffer Table Placement Rules

Every scheme that references a buffer (per-channel, per-block, NF4, MXFP) adds
entries to the tensor descriptor's buffer table beyond the buffers used by the
layout itself. The following rules apply across all quantization schemes:

1. The tensor data buffer always occupies a buffer table index determined by
   the layout (typically `0` for dense layouts). Quantization-parameter buffers
   MUST occupy distinct indices.
2. A quantization-parameter buffer MUST NOT be shared with the tensor data
   buffer. A reader MUST reject a descriptor that violates this rule.
3. Two quantization-parameter buffers (e.g., scale and zero-point) MAY reside
   in the same buffer table entry if and only if the entry's `byte_size`
   accommodates both arrays laid out end-to-end, and both arrays start at
   byte offset `0`. Since each parameter descriptor field specifies its own
   `*_buffer_index`, this case is expressed by writing the same index into
   both fields; readers MUST then interpret the buffer as the concatenation
   `[scales | zero_points]`. Writers SHOULD prefer distinct buffers for
   clarity; sharing is permitted only as an optimization.
4. A quantization-parameter buffer's `device_tag` (see `metadata.md` buffer
   handle format) MUST match the tensor data buffer's `device_tag`. A reader
   MUST reject a descriptor that violates this rule.

> **Note (non-normative):** The device-colocation rule ensures that quantized
> tensor kernels can dereference both the data and the quantization parameters
> without triggering cross-device transfers. A writer that needs to materialize
> quantization parameters on a different device must emit a separate tensor.

---

## Extension Schemes (`0xF0` – `0xFE`)

Implementations MAY define private quantization schemes using scheme tags in
`0xF0` – `0xFE`. The binary encoding of an extension scheme descriptor is
unconstrained beyond the fixed 4-byte header; the writer and reader MUST agree
on the payload format out of band.

> **Note (non-normative):** Extension schemes are the mechanism for
> implementation-specific or experimental quantization formats (e.g., GPTQ,
> AWQ group quantization with per-group permutations, or hardware-vendor-
> specific packings). Schemes that prove broadly useful SHOULD be proposed for
> assignment in the Tier 1 or Tier 2 range through a specification revision.

---

## Version Compatibility

Adding a new scheme tag MUST be accompanied by a minor version increment of the
overall format (`version_minor` in the fixed header; see `metadata.md`). A
reader at an earlier minor version will encounter an unrecognised `scheme_tag`
and reject the descriptor per the rules above, which is the intended behaviour.

Adding a new field to an existing scheme (or repurposing a reserved byte) MUST
be accompanied by a `scheme_version` increment for that scheme. A reader MUST
compare `scheme_version` against the highest version it supports for the given
`scheme_tag` and reject the descriptor if the version is newer than supported.

Removing a scheme or changing the dequantization formula for an existing
`(scheme_tag, scheme_version)` pair is a backward-incompatible change and MUST
be accompanied by a major version increment.

---

## Open Questions

> **[OQ-1]:** Should this file define a normative encoding for double quantization (quantizing the scale buffer itself, as in bitsandbytes QLoRA "nested" quantization)? **Deferred.** Rationale: the base quantization encoding must be validated through implementation before a recursive or two-level descriptor can be specified safely. Double quantization is primarily a weight-storage optimization rather than a runtime interchange primitive. Scheme tags `0x60`–`0x7F` are reserved for future nested/composite schemes. A conforming implementation MAY express double quantization today by representing the scale tensor as a separate quantized tensor using existing scheme tags, with the relationship conveyed by application-layer convention.
>
> **Note (non-normative):** The intended future occupant of the `0x60`–`0x7F` range is a nested-scale scheme compatible with bitsandbytes-style double quantization (NF4 data with quantized `float8` scales and a `float32` super-scale). This will be specified in a future revision once implementation experience is available.

> **[OQ-2]:** ~~Should MXFP `block_size` be fixed at 32 or parameterised?~~ **Resolved:** `block_size` is parameterised. The field already exists in the binary encoding; the constraint is: MUST be a power of two in `[16, 2048]`. The lower bound of 16 excludes values with no hardware Tensor Core support. The OCP MX v1.0 canonical value of `32` is documented as the default. Rationale: avoids scheme tag proliferation for what is effectively one scheme with a size variation; future OCP revisions and hardware variants can use different block sizes under the same scheme tag.

> **[OQ-3]:** ~~Should per-channel affine support lower-precision scale types?~~ **Resolved:** Per-channel scales remain locked to `float32` (`scale_type_tag = 0x03`). Rationale: storage saving is negligible (~8–16 KB per layer) while accuracy cost is real for precision-critical per-channel weight quantization. A `scale_type_tag` field (+ 3 reserved bytes) has been added to the per-channel affine binary encoding at offset 16 to allow future relaxation without a wire-format break.

> **[OQ-4]:** ~~Should the descriptor include an explicit `num_blocks` field?~~ **Resolved:** `num_blocks` remains derived: `num_blocks = shape[axis] / block_size`. Rationale: the value is fully determined by fields already present in the descriptor; an explicit field would be redundant and introduce a new mismatch failure mode.

> **[OQ-5]:** ~~Should the NF4 lookup table be duplicated into a normative reference appendix?~~ **Resolved:** The table remains inline in `quantization/nf4.md`. Rationale: duplication would risk the two copies diverging; a single source of truth is safer. Spec audits (spec-checker) are the guard against accidental mutation.
