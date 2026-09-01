"""Private extension element types: describing a type the format does not standardize.

Tags `0xF0`–`0xFE` are reserved for types the spec deliberately says nothing about —
your own posit, your own block scale, whatever your runtime needs. The format's price
for that freedom is that the tensor must describe the type well enough for a stranger
to size its buffers: bit width and packing, carried in the descriptor's extension type
section.

That is the whole contract. A reader that has never heard of your type still knows how
many bytes to move, which is what an interchange format has to guarantee.

Run with:

    python hurray-python/examples/extension_types.py
"""

import hurray

# ── Naming the type ───────────────────────────────────────────────────────────

print("=== The dtype alone says almost nothing ===")

private = hurray.Dtype.from_tag(0xF2)
print(f"  {private!r}")
print(f"  name      {private.name}")
print(f"  tag       0x{private.tag:02X}")
print(f"  bit_width {private.bit_width}   <- 0: the width is not in the dtype")

# Asking the generic helper is a question it cannot answer, so it refuses rather
# than returning a plausible 0.
try:
    hurray.buffer_size_bytes(private, 10)
except hurray.InvalidDescriptorError as exc:
    print(f"\n  hurray.buffer_size_bytes(private, 10) -> {type(exc).__name__}")
    print(f"    {exc}")

# ── Describing it ─────────────────────────────────────────────────────────────

print("\n=== The section is where the width lives ===")

int24 = hurray.ExtensionType(bit_width=24, is_signed=True)
print(f"  {int24!r}")
print(f"  packing_factor    {int24.packing_factor}   <- derived, not supplied")
print(f"  buffer_size_bytes {int24.buffer_size_bytes(4)} for 4 elements")

# Sub-byte widths pack, and only 1, 2 and 4 bits are legal: the packing factor has to
# be a whole number of elements per byte. 6-bit types pack 4-per-3-bytes and are
# therefore built-in, not private.
nibble = hurray.ExtensionType(bit_width=4)
print(f"\n  {nibble!r}")
print(f"  packing_factor    {nibble.packing_factor}   <- 8 / 4")
print(f"  buffer_size_bytes {nibble.buffer_size_bytes(7)} for 7 elements (ceil(7 / 2))")

try:
    hurray.ExtensionType(bit_width=6)
except hurray.InvalidDescriptorError as exc:
    print(f"\n  bit_width=6 -> {type(exc).__name__}: {exc}")

# ── Floats, and where their sign lives ────────────────────────────────────────

print("\n=== A float's sign is sign_bits, never is_signed ===")

half = hurray.ExtensionType(
    bit_width=16,
    is_float=True,
    sign_bits=1,
    exponent_bits=5,
    mantissa_bits=10,
    exponent_bias=15,
    has_nan=True,
    has_inf=True,
)
print(f"  signed float:   {half!r}")
print(f"    is_signed {half.is_signed}   sign_bits {half.sign_bits}")

# `is_signed` describes integers only. That is not a claim that floats are unsigned —
# it is what makes an *unsigned* float expressible, the shape of the built-in
# exponent-only float8_e8m0 block scale.
scale = hurray.ExtensionType(
    bit_width=8, is_float=True, exponent_bits=8, exponent_bias=127
)
print(f"  unsigned float: {scale!r}")
print(f"    is_signed {scale.is_signed}   sign_bits {scale.sign_bits}")

try:
    hurray.ExtensionType(bit_width=16, is_float=True, is_signed=True, sign_bits=1)
except hurray.InvalidDescriptorError as exc:
    print(f"\n  is_float + is_signed -> {type(exc).__name__}: {exc}")

# ── A tensor of it ────────────────────────────────────────────────────────────

print("\n=== Authoring a tensor ===")

values = bytes(int24.buffer_size_bytes(4))
tensor = hurray.Tensor(values, private, [4], extension_type=int24)
print(f"  {tensor.descriptor!r}")
print(f"  extension_type  {tensor.extension_type!r}")

# The two must agree: the section describes the dtype, so neither stands alone.
try:
    hurray.Tensor(values, private, [4])
except hurray.InvalidDescriptorError as exc:
    print(f"\n  extension dtype, no section -> {exc}")

try:
    hurray.Tensor(bytes(16), hurray.float32, [4], extension_type=int24)
except hurray.InvalidDescriptorError as exc:
    print(f"  section, standard dtype     -> {exc}")

# And the section is what sizes the buffer check. Without it this would compute 0
# bytes and accept anything.
try:
    hurray.Tensor(bytes(4), private, [4], extension_type=int24)
except hurray.BufferError as exc:
    print(f"  buffer too small            -> {exc}")

# ── On the wire ───────────────────────────────────────────────────────────────

print("\n=== It survives the round trip ===")

wire = tensor.descriptor.encode()
restored = hurray.Descriptor.decode(wire)
plain = hurray.Tensor(bytes(16), hurray.float32, [4]).descriptor

print(f"  encoded          {len(wire)} bytes")
print(f"  without section  {plain.encoded_len} bytes  (the section is a fixed 20)")
print(f"  restored         {restored.extension_type!r}")

assert restored == tensor.descriptor
assert restored.extension_type.bit_width == 24
assert restored.extension_type.is_signed is True
assert len(wire) == plain.encoded_len + 20

print("\n  A consumer that has never heard of type 0xF2 still knows it needs")
print(f"  {restored.extension_type.buffer_size_bytes(4)} bytes for 4 elements.")
