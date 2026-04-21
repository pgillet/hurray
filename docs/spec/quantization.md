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
| `0x40` – `0x7F` | Tier 2 schemes (OPTIONAL) |
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

> **[OQ-1]:** Should this file define a normative encoding for double
> quantization (quantizing the scale buffer itself, as in bitsandbytes QLoRA
> "nested" quantization)? The current design makes the scale buffer a flat
> `float32` array. Double quantization would require either a recursive
> descriptor or a separate scheme tag. Deferred until a concrete interop need
> arises.

> **[OQ-2]:** The MXFP scheme currently fixes `block_size = 32` per the OCP MX
> v1.0 specification. Future OCP revisions (or NVIDIA-specific variants) may
> define alternative block sizes. Should `block_size` be left parameterised at
> the scheme level now to avoid a new scheme tag later, or should the fixed
> constraint be enforced to ensure strict OCP conformance? Current text enforces
> `block_size = 32`.

> **[OQ-3]:** Per-channel and per-block schemes currently limit the scale type
> to `float32` (per-channel) or `float16/bfloat16/float32` (per-block). Should
> per-channel support lower-precision scale types as well? The storage saving
> is modest (a few KB per layer) and the precision cost is non-trivial for
> per-channel weight quantization. Deferred.

> **[OQ-4]:** Should the descriptor include an explicit `num_blocks` field for
> the block-based schemes, or is deriving it from `shape[axis]` and
> `block_size` sufficient? Current text derives it. Deriving is simpler but
> requires a shape resolution step on the reader side; an explicit field would
> be redundant but self-contained. Deferred.

> **[OQ-5]:** NF4 decoding requires the fixed 16-entry lookup table. Should
> this table be duplicated into a normative reference appendix (to insulate
> against accidental mutation of the main table in future edits), or is the
> inline specification sufficient? Current text keeps it inline in
> `quantization/nf4.md`.
