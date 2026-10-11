# ADR-039: Structured descriptor sub-objects cross the C ABI as typed field reads

## Status

Accepted (2026-10-08), implemented 2026-10-11 (PR #228)

Resolves the design question deferred to `architect` by
[issue #200](https://github.com/pgillet/hurray/issues/200). Applies **ADR-019**'s
evolution rules to the C ABI surface; amends no earlier decision.

## Context

`hurray_descriptor_decode` parses and validates a descriptor through `hurray-core`, so
the *reading* is complete for every layout the core supports. The C surface then
exposes almost none of the result. Its 29 symbols cover scalars — rank, element type
tag, layout tag, byte offset, buffer count — and exactly one variable-length field,
`hurray_descriptor_shape`. Nothing with internal structure crosses the boundary.

The consequence is that a C consumer handed a CSR descriptor learns
`layout_tag == 0x07` and nothing else: not `nnz`, not which buffer is `row_ptr`. It
cannot address a single element. Compliance Level 1 requires an implementation to
correctly interpret *layout-specific fields for all Tier 1 layouts* (`0x01`–`0x09`),
which is why the generated status page reports `hurray-ffi` as a partial Level 1 reader.

**The layout payload is not the only gap, and that is what shapes this decision.** The
five optional descriptor sections — quantization, shard, statistics, extension type,
composite member — each appear in `website/coverage-matrix.toml` with a `c_symbols`
entry, and not one of those symbols exists in `hurray.h`. Six structured sub-objects are
unreachable from C, for the same reason, and whatever pattern is chosen for layouts will
be the pattern each of the other five inherits. The question is therefore not "how does
a C caller read CSR `nnz`" but "how does a structured descriptor sub-object cross the C
ABI" — asked once, answered once.

### The field inventory

What a complete layout payload surface must carry, read from `hurray-core/src/layout/`:

| Layout | Tag | Scalar fields | Variable-length fields |
|---|---|---|---|
| Row-major | `0x01` | — | — |
| Column-major | `0x02` | — | — |
| Strided | `0x03` | — | `strides` (signed) |
| Tiled | `0x04` | `outer_layout`, `inner_layout`, nesting depth | `tile_shape`, `outer_strides`*, `inner_strides`* |
| Morton | `0x05` | — | `morton_bits` |
| COO | `0x06` | `nnz`, `is_sorted` | — |
| CSR | `0x07` | `nnz` | — |
| CSC | `0x08` | `nnz` | — |
| CSF | `0x09` | `nnz` | `mode_order` |
| Block-paged | `0x0A` | `page_size`, `num_pages`, `paged_axis`, `num_seqs`, `kv_role`, `layer_index`*, `block_table_index_type` | — |
| Composite | `0x0B` | `rule`, `combine_op`, `member_count` | — |
| Hilbert | `0x40` | `hilbert_order`, `hilbert_rank` | — |

`*` optional — absent on a conforming descriptor that does not carry it.

Three properties of that table drive the decision:

1. **It is about 26 fields**, against an existing ABI of 29 symbols. One accessor per
   field nearly doubles the surface for one sub-object out of six.
2. **Every scalar is unsigned or boolean.** There is no signed scalar payload field.
3. **`Tiled` is recursive** (`inner_tiled: Option<Box<TiledLayout>>`, bounded by
   `MAX_TILED_DEPTH`), so any design must address a nested level or deliberately refuse
   to.

### Constraints that bind the answer

- `.claude/rules/unsafe-ffi.md`: public handles MUST be opaque, Rust struct layout MUST
  NOT cross the boundary, and every exported function MUST return a `HurrayStatus`.
- **ADR-019**: the format evolves by allocating tag values in reserved ranges behind
  flag bits and length prefixes — never by adding a fixed-offset field that an older
  reader mis-parses. A retired tag value is never rebound. The C ABI should be
  disciplined the same way, or the two evolution models diverge.
- `hurray_descriptor_shape` already establishes the house idiom for a variable-length
  read: the caller passes a capacity, receives the true length, and retries after
  `HURRAY_ERR_BUFFER_TOO_SMALL`.

## Decision

### 1. Three typed field accessors, keyed by a field identifier

```c
HurrayStatus hurray_descriptor_layout_field_u64(
    const HurrayDescriptor *handle, HurrayLayoutField field,
    uint32_t level, uint64_t *out_value);

HurrayStatus hurray_descriptor_layout_field_u64_array(
    const HurrayDescriptor *handle, HurrayLayoutField field,
    uint32_t level, uint64_t *out_values, size_t *out_len);

HurrayStatus hurray_descriptor_layout_field_i64_array(
    const HurrayDescriptor *handle, HurrayLayoutField field,
    uint32_t level, int64_t *out_values, size_t *out_len);
```

Three symbols, not twenty-six. Two array functions rather than one because the stride
fields are the only signed arrays in the format and widening them into an unsigned
accessor would misreport the negative strides the spec explicitly permits.

### 2. `HurrayLayoutField` is a `uint32_t` tag space, not a C enum

```rust
pub type HurrayLayoutField = u32;
```

Named constants, exactly as `HurrayStatus` is done. **Not** a `#[repr(C)]` enum: a C
caller can pass any integer, and an out-of-range value materialised as a Rust enum is
undefined behaviour. A `u32` parameter validated in Rust is sound for every input, which
is the only form that holds at an ABI boundary where the caller is not trusted.

### 3. A field identifier encodes the layout it belongs to

```text
field_id = 0x0001_0000 | (layout_tag << 8) | field_index
```

| Constant | Value |
|---|---|
| `HURRAY_LAYOUT_FIELD_STRIDED_STRIDES` | `0x0001_0300` |
| `HURRAY_LAYOUT_FIELD_TILED_TILE_SHAPE` | `0x0001_0400` |
| `HURRAY_LAYOUT_FIELD_TILED_OUTER_LAYOUT` | `0x0001_0401` |
| `HURRAY_LAYOUT_FIELD_CSR_NNZ` | `0x0001_0700` |
| `HURRAY_LAYOUT_FIELD_BLOCK_PAGED_PAGE_SIZE` | `0x0001_0A00` |
| `HURRAY_LAYOUT_FIELD_HILBERT_ORDER` | `0x0001_4000` |

The embedded layout tag makes § 6's applicability check a comparison rather than a
lookup table, and makes the field space inherit the layout tag space's existing
reserved and private ranges for free. `0x0000_0000` is never a valid identifier, so a
zeroed variable is caught rather than silently read as the first field.

The high half-word is the **sub-object range**. Layout fields occupy `0x0001_xxxx`;
each of the five optional sections takes its own range when it is implemented
(quantization `0x0002_xxxx`, and so on). Allocation follows ADR-019's anti-rebind rule:
a retired identifier is never reused for a different field.

### 4. Scalars widen to `uint64_t`

Every scalar payload field is unsigned or boolean (§ Context), so widening is lossless:
`bool` reads as 0 or 1, and the `u8` wire bytes of `kv_role`, `block_table_index_type`,
`rule` and `combine_op` read as their documented tag values. One accessor replaces one
per width. Should a signed scalar field ever be specified, it gets its own accessor
then — additively, without touching these three.

### 5. Arrays reuse the `hurray_descriptor_shape` capacity idiom

`*out_len` carries the caller's capacity in and the true length out; a short or null
buffer returns `HURRAY_ERR_BUFFER_TOO_SMALL` with the true length written, so the
caller can allocate and retry. Being consistent with the one variable-length accessor
that already exists matters more than any marginal improvement on it.

### 6. Absence and inapplicability are distinct statuses, never sentinel values

Two new codes:

| Code | Value | Meaning |
|---|---|---|
| `HURRAY_ERR_FIELD_NOT_PRESENT` | `-12` | The field is valid for this layout but this descriptor does not carry it — an absent optional field (`layer_index`, `outer_strides`, `inner_strides`). |
| `HURRAY_ERR_FIELD_NOT_APPLICABLE` | `-13` | The field belongs to a different layout than this descriptor's. |

Absence is a state in this format, not a value — it is what a cleared flag bit means —
so it must not be reported as a zero a caller could mistake for data. Keeping the two
apart tells a caller whether it asked a reasonable question about a tensor that lacks
the field, or asked the wrong question entirely.

`HURRAY_ERR_FIELD_NOT_APPLICABLE` is the honest cost of § 1: a C compiler cannot check
that a field belongs to the layout in hand, so the check moves to runtime. Making it a
named status rather than a generic error is what keeps that cost diagnosable.

### 7. The `level` parameter is present from the outset

Only `Tiled` nests. `level` MUST be `0` for every other layout, and a non-zero `level`
against a non-nesting layout returns `HURRAY_ERR_INDEX_OUT_OF_BOUNDS`.
`HURRAY_LAYOUT_FIELD_TILED_DEPTH` reports how many levels exist, so a caller can walk
them.

It is carried now, unused by nine of the twelve layouts, because adding a parameter
later is an ABI break while an unused parameter costs a register. Exposing only the
outermost level and deferring nesting — the option #200 raises — buys nothing and
spends a future break.

### 8. The accessors stay on `HurrayDescriptor`

Not on the combined handle [#174](https://github.com/pgillet/hurray/issues/174) will
introduce: a consumer may legitimately hold descriptor bytes and no buffers at all, and
that consumer needs these reads. `HurrayTensor` will expose its descriptor as a borrowed
`const HurrayDescriptor *` rather than duplicating the accessor family, which also makes
this decision independent of that one.

### 9. C ABI version 4 → 5

Additive — new symbols, new constants, no change to an existing signature. #174's
breaking change bumps again when it lands, rather than sharing this one: two accurate
bumps describe the surface better than one that conflates an addition with a break, and
a consumer that only needs these reads can gate on `>= 5`.

## Alternatives Considered

**One named accessor per field** — `hurray_descriptor_csr_nnz`, and twenty-five more.
What #200 proposed, and the most discoverable option: a C programmer gets autocompletion,
the right type, and a compile error for a field that does not belong to the layout in
hand. Rejected on how it scales. Twenty-six symbols for one sub-object out of six means
the surface grows linearly in fields forever, and each of the five remaining sections
re-grows it — quantization alone has five schemes. It also needs twenty-six near-identical
layout-tag guards. The decisive argument is ADR-019: that document's thesis is that this
format evolves by allocating tag values, not by adding fixed members, and § 3 is that
thesis applied to the ABI. The cost of rejecting it is real and is recorded in § 6 and
§ Consequences.

**One `#[repr(C)]` payload struct or union** — the other option #200 raises. Rejected on
three independent grounds, any one sufficient: it publishes Rust struct layout across the
boundary, which `.claude/rules/unsafe-ffi.md` forbids; it cannot carry the six
variable-length fields without embedding pointers whose validity is tied to the
descriptor handle, which is a use-after-free waiting for a caller who keeps the struct
and drops the handle; and it cannot express recursive `Tiled` at all. It would also
change shape on every new layout — an ABI break per layout, inverting ADR-019 exactly.

**An opaque layout sub-handle** — `hurray_descriptor_layout` returning a
`HurrayLayout *` carrying its own accessors. Rejected because it reduces nothing: the
per-field accessors still have to exist on the sub-handle, and it adds a second handle
plus a borrow-lifetime question against the parent descriptor.

**Return the encoded layout payload as bytes** and let the caller parse it. Rejected
because it makes every binding reimplement descriptor parsing, which is the specific
thing `hurray-ffi` exists to prevent. It would also re-expose the wire format to
consumers that should only see decoded values.

## Consequences

- `hurray-ffi`'s Level 1 cell flips from ◐ to ✅ once every Tier 1 layout's fields are
  reachable. Level 2 stays ❌: there is no encode entry point in C at all, which is a
  separate deferred decision, not a missing accessor.
- The five optional sections become additive work against a settled pattern — the same
  three functions and a new sub-object range, with no further design.
- A field-to-layout mismatch surfaces at runtime, not compile time. Higher-level
  bindings should wrap these reads in typed accessors; `hurray-python` already has the
  layout object hierarchy (ADR-032) to hang them on.
- The header is less self-documenting than named functions would be. Mitigated by
  prefixing every constant with its layout and by a worked consumer example in the
  cookbook, not by the signatures themselves.
- Test surface: every field of every layout in the table, the capacity-retry path, both
  new statuses, nested `Tiled` at more than one level, and a `level != 0` rejection.

## Required Documentation Amendments

- `docs/impl/c-ffi.md` — the accessor family, the field identifier scheme, and the
  sub-object range allocation.
- `docs/cookbook/layer-7-c-ffi.md` — read a CSR descriptor's `nnz` and a strided
  descriptor's `strides` from C, including the capacity retry.
- `website/coverage-matrix.toml` — the `level-1` row's `ffi = "partial:ffi-level-1"`
  override is replaced by a `c_symbols` list, and the `ffi-level-1` note is deleted.
- `docs/SUMMARY.md` — add this ADR, and ADR-038, which is on disk but absent from the
  navigation.

## Open Questions Deferred

- **The writer side.** `HurrayWriter` is a typedef with nothing behind it and there is
  no encode entry point, deferred pending the async bridge design. Not an accessor
  question, and untouched here.
- **Whether the five optional sections share one identifier enum** with reserved ranges,
  as § 3 assumes, **or take one type each.** Settled when the first of them is
  implemented; the range discipline in § 3 leaves both open.
