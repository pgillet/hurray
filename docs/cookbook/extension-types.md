# Private Extension Element Types

Element type tags `0xF0`–`0xFE` are reserved for types the format deliberately says
nothing about — a private posit, a block scale your runtime invented, an experiment that
has not earned a standard tag. Nothing about their numeric semantics is specified, and two
implementations that both use `0xF2` need not mean the same thing by it.

The price of that freedom is one obligation: **the tensor must describe the type well
enough for a stranger to size its buffers.** That description is the descriptor's
extension type section — a fixed 20 bytes carrying the bit width, the packing, and the
floating-point parameters. A reader that has never heard of your type still knows exactly
how many bytes to move, which is what an interchange format has to guarantee.

The flag and the tag are one fact: a descriptor whose type tag is in `0xF0`–`0xFE` MUST
carry the section, and one whose tag is anything else MUST NOT. Neither half stands alone.

## The tag alone says almost nothing

An extension type reports a `bit_width` of `0`. That is a sentinel, not a claim: the real
width lives in the section, so the generic buffer-size helper has nothing to compute from.

<div class="lang-tabs">

```rust
use hurray_core::{buffer_size_bytes, ElementType};

fn main() -> Result<(), hurray_core::Error> {
    let private = ElementType::from_tag(0xF2)?;

    assert_eq!(private.tag(), 0xF2);
    assert_eq!(private.bit_width(), 0);          // sentinel: ask the section
    assert_eq!(buffer_size_bytes(private, 10), 0);

    Ok(())
}
```

```python
import hurray

private = hurray.Dtype.from_tag(0xF2)

assert private.tag == 0xF2
assert private.bit_width == 0            # sentinel: ask the section

# Python refuses rather than returning a plausible 0 — as a byte count that is
# wrong, not unknown, and would size a buffer to nothing.
try:
    hurray.buffer_size_bytes(private, 10)
except hurray.InvalidDescriptorError as exc:
    assert "ExtensionType" in str(exc)
```

</div>

## Describing the type

`packing_factor` is how many elements fit in a byte. Whole-byte widths pack one; sub-byte
widths pack `8 / bit_width`, and only `1`, `2` and `4` bits are legal — anything else
would need a fractional number of elements per byte. Non-power-of-two sub-byte widths are
reserved to the built-in type space, which is where the 6-bit floats live with their
4-elements-per-3-bytes packing.

The Python constructor **derives** `packing_factor` rather than asking for it: the spec
leaves exactly one legal value per width, so restating it could only produce an error.

<div class="lang-tabs">

```rust
use hurray_core::descriptor::ExtensionTypeDescriptor;

fn main() -> Result<(), hurray_core::Error> {
    // A private 24-bit signed integer: whole-byte, so packing_factor is 1.
    let int24 = ExtensionTypeDescriptor::new(24, 1, false, true, 0, 0, 0, 0, false, false)?;
    assert_eq!(int24.buffer_size_bytes(4), 12);

    // A private 4-bit type: two per byte, rounding up.
    let nibble = ExtensionTypeDescriptor::new(4, 2, false, false, 0, 0, 0, 0, false, false)?;
    assert_eq!(nibble.buffer_size_bytes(7), 4);

    // 6-bit is reserved to the built-in types.
    assert!(ExtensionTypeDescriptor::new(6, 1, false, false, 0, 0, 0, 0, false, false).is_err());

    Ok(())
}
```

```python
import hurray

int24 = hurray.ExtensionType(bit_width=24, is_signed=True)
assert int24.packing_factor == 1
assert int24.buffer_size_bytes(4) == 12

nibble = hurray.ExtensionType(bit_width=4)
assert nibble.packing_factor == 2        # derived: 8 / 4
assert nibble.buffer_size_bytes(7) == 4  # ceil(7 / 2)

try:
    hurray.ExtensionType(bit_width=6)
except hurray.InvalidDescriptorError as exc:
    assert "1, 2 or 4" in str(exc)
```

</div>

## Where a float's sign lives

A float carries its sign in `sign_bits`; `is_signed` describes **integer** types only.
Setting both would give a reader two answers to one question, so a float with `is_signed`
set is rejected.

This is not a claim that float extension types are unsigned — it is what makes an
*unsigned* float expressible. The built-in `float8_e8m0` (`0x42`) is exactly that shape:
8 exponent bits, no sign, no mantissa, used as an MX block scale. A private analogue sets
`is_float` with `sign_bits = 0`.

<div class="lang-tabs">

```rust
use hurray_core::descriptor::ExtensionTypeDescriptor;

fn main() -> Result<(), hurray_core::Error> {
    // Signed 16-bit float: 1-5-10, bias 15.
    let half = ExtensionTypeDescriptor::new(16, 1, true, false, 1, 5, 10, 15, true, true)?;
    assert_eq!(half.sign_bits, 1);
    assert!(!half.is_signed);

    // Unsigned float — the float8_e8m0 shape.
    let scale = ExtensionTypeDescriptor::new(8, 1, true, false, 0, 8, 0, 127, true, false)?;
    assert_eq!(scale.sign_bits, 0);

    // Both set: rejected.
    assert!(ExtensionTypeDescriptor::new(16, 1, true, true, 1, 5, 10, 15, true, false).is_err());

    Ok(())
}
```

```python
import hurray

half = hurray.ExtensionType(
    bit_width=16, is_float=True,
    sign_bits=1, exponent_bits=5, mantissa_bits=10, exponent_bias=15,
    has_nan=True, has_inf=True,
)
assert half.sign_bits == 1
assert half.is_signed is False

# Unsigned float — the float8_e8m0 shape.
scale = hurray.ExtensionType(
    bit_width=8, is_float=True, exponent_bits=8, exponent_bias=127
)
assert scale.sign_bits == 0

try:
    hurray.ExtensionType(bit_width=16, is_float=True, is_signed=True, sign_bits=1)
except hurray.InvalidDescriptorError as exc:
    assert "is_signed" in str(exc)
```

</div>

## Authoring a tensor

The section travels with the descriptor and is what sizes the buffer check. Without it the
check would compute zero bytes and wave an empty buffer through for a tensor of any
length.

<div class="lang-tabs">

```rust
use hurray_core::{
    descriptor::{ExtensionTypeDescriptor, TensorDescriptor},
    layout::LayoutDescriptor,
    BufferHandle, DeviceTag, ElementType, Shape, SyncMode, DESCRIPTOR_VERSION_MAJOR,
    DESCRIPTOR_VERSION_MINOR, MIN_BUFFER_ALIGNMENT,
};

fn main() -> Result<(), hurray_core::Error> {
    let int24 = ExtensionTypeDescriptor::new(24, 1, false, true, 0, 0, 0, 0, false, false)?;
    let buffer = BufferHandle::new(
        int24.buffer_size_bytes(4),
        MIN_BUFFER_ALIGNMENT,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )?;

    let descriptor = TensorDescriptor::new(
        DESCRIPTOR_VERSION_MAJOR,
        DESCRIPTOR_VERSION_MINOR,
        ElementType::from_tag(0xF2)?,
        Shape::new(vec![4u64])?,
        0,
        LayoutDescriptor::RowMajor,
        vec![buffer],
        None,
        None,
        None,
        Some(int24),
    )?;

    // Round-trips: a consumer that has never heard of 0xF2 still learns the width.
    let restored = TensorDescriptor::decode(&descriptor.encode()?)?;
    let ext = restored.extension_type.as_ref().expect("carries a section");
    assert_eq!(ext.buffer_size_bytes(4), 12);

    Ok(())
}
```

```python
import hurray

private = hurray.Dtype.from_tag(0xF2)
int24 = hurray.ExtensionType(bit_width=24, is_signed=True)

tensor = hurray.Tensor(
    bytes(int24.buffer_size_bytes(4)), private, [4], extension_type=int24
)
assert tensor.extension_type.bit_width == 24

# Round-trips: a consumer that has never heard of 0xF2 still learns the width.
restored = hurray.Descriptor.decode(tensor.descriptor.encode())
assert restored == tensor.descriptor
assert restored.extension_type.buffer_size_bytes(4) == 12
```

</div>

Both directions of the pairing are refused, at authoring time:

```python
import hurray

private = hurray.Dtype.from_tag(0xF2)
int24 = hurray.ExtensionType(bit_width=24, is_signed=True)

# An extension dtype must describe itself.
try:
    hurray.Tensor(bytes(12), private, [4])
except hurray.InvalidDescriptorError as exc:
    assert "must describe itself" in str(exc)

# And the section cannot stand alone.
try:
    hurray.Tensor(bytes(16), hurray.float32, [4], extension_type=int24)
except hurray.InvalidDescriptorError as exc:
    assert "not a private extension" in str(exc)

# The section sizes the buffer check.
try:
    hurray.Tensor(bytes(4), private, [4], extension_type=int24)
except hurray.BufferError as exc:
    assert "need at least 12 bytes" in str(exc)
```

## What this does not buy you

Portability. The spec is explicit: tensors using private extension tags MUST NOT be
exchanged between independent implementations unless both parties have agreed on the
semantics out of band. The section lets a stranger *move* your bytes, not interpret them.
If you need a type both ends understand without a side agreement, request a built-in tag
through the specification governance process instead.

## See also

- [Layer 0: Element Types and Shape](layer-0-element-types-and-shape.md) — the built-in
  type system the extension range sits beside
- [Layer 4: Tensor Descriptor Encoding](layer-4-tensor-descriptor-encoding.md) — where the
  optional sections sit in the wire format
- `docs/spec/metadata.md` § Extension Type Section — the normative field table
- ADR-001 — why extension tags carry an inline descriptor at all
