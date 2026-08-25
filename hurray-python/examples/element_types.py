"""The element type system and the shape model, from Python.

A Hurray element type is more than a name. It has a wire tag that identifies it
in an encoded descriptor, a bit width that may be smaller than a byte, and a
packing rule that decides how many bytes N of them actually occupy. A shape may
leave a dimension unresolved.

Run with:

    python hurray-python/examples/element_types.py
"""

import hurray

# ── A type describes itself ───────────────────────────────────────────────────

print("=== What a dtype knows ===")

for dtype in (hurray.float32, hurray.dtype.int4, hurray.bool):
    print(
        f"  {dtype.name:<8} tag=0x{dtype.tag:02X}  {dtype.bit_width:>2} bits  "
        f"tier {dtype.tier}  element_alignment={dtype.element_alignment}  "
        f"sub_byte={dtype.is_sub_byte}"
    )

print("\n  element_alignment is the element's own, not the buffer's: a float32")
print(f"  buffer starts on a {hurray.MIN_BUFFER_ALIGNMENT}-byte boundary, but its elements are")
print(f"  {hurray.float32.element_alignment}-aligned within it. A packed element reports 1 —")
print("  two int4s share a byte, so neither has an address of its own.")

# ── Tags are the wire ─────────────────────────────────────────────────────────

print("\n=== The wire tag ===")

tag = hurray.dtype.float8_e4m3.tag
print(f"  float8_e4m3 -> 0x{tag:02X} -> {hurray.Dtype.from_tag(tag).name}")
print(f"  and it is the same object back: {hurray.Dtype.from_tag(tag) is hurray.dtype.float8_e4m3}")

for bad, why in ((0x00, "permanently invalid"), (0xFF, "permanently invalid"), (0x7E, "reserved")):
    try:
        hurray.Dtype.from_tag(bad)
    except hurray.InvalidDescriptorError as exc:
        print(f"  0x{bad:02X} ({why}): {exc}")

print("\n  A reserved tag is refused rather than guessed at: it may mean the")
print("  producer is newer than this reader, or that the descriptor is corrupt,")
print("  and inventing a type for it would turn either into silent wrong data.")

# ── Sizing a buffer ───────────────────────────────────────────────────────────

print("\n=== How many bytes? ===")

print("  count * bit_width // 8 is wrong for every sub-byte type:")
for dtype, count in (
    (hurray.float32, 100),
    (hurray.dtype.int4, 7),
    (hurray.bool, 9),
    (hurray.dtype.float6_e2m3, 100),
    (hurray.dtype.int2, 5),
):
    naive = count * dtype.bit_width // 8
    actual = hurray.buffer_size_bytes(dtype, count)
    flag = "" if naive == actual else f"   <- naive says {naive}"
    print(f"    {dtype.name:<12} x {count:>3} = {actual:>3} bytes{flag}")

print("\n  The difference is one byte short of the last element, which the")
print("  constructor rejects — better to ask than to find out:")

count = 7
needed = hurray.buffer_size_bytes(hurray.dtype.int4, count)
hurray.Tensor(bytes(needed), hurray.dtype.int4, [count])
print(f"    {needed} bytes holds {count} int4 elements")

try:
    hurray.Tensor(bytes(needed - 1), hurray.dtype.int4, [count])
except hurray.BufferError as exc:
    print(f"    {needed - 1} does not: {exc}")

# ── A dimension that is not known yet ─────────────────────────────────────────

print("\n=== Dynamic dimensions ===")

signature = hurray.Tensor(b"", hurray.float32, [None, 512, 768])
print(f"  {signature!r}")
print(f"  shape={signature.shape}  ndim={signature.ndim}  size={signature.size}")
print("  None is a dimension whose extent is not known when the descriptor is")
print("  written — a batch size resolved at inference time. size is None to match:")
print("  an unknown extent means an unknown element count.")

print("\n  It is spelled None rather than a sentinel constant because that is what")
print("  .shape gives you back, so the pair round trips:")
rebuilt = hurray.Tensor(b"", signature.dtype, list(signature.shape))
print(f"    rebuilt from its own shape: {rebuilt.shape == signature.shape}")

resolved = hurray.Tensor(bytes(4 * 8 * 512 * 768), hurray.float32, [8, 512, 768])
print(f"\n  once the batch size is known: shape={resolved.shape} size={resolved.size}")

print("\n  Anything that allocates refuses it — you cannot allocate an unknown")
print("  number of bytes — and says so by name:")
try:
    hurray.zeros([None, 512])
except hurray.InvalidDescriptorError as exc:
    print(f"    {exc}")

# ── Empty and scalar ──────────────────────────────────────────────────────────

print("\n=== The edges ===")

scalar = hurray.Tensor(bytes(4), hurray.float32, [])
empty = hurray.Tensor(b"", hurray.float32, [5, 0, 10])

print(f"  a scalar has rank 0 and one element:  shape={scalar.shape} size={scalar.size}")
print(f"  an empty tensor has a shape but no elements: shape={empty.shape} size={empty.size}")
print(f"  and its buffer declares alignment {empty.buffer_handles[0].alignment} — nothing to align")
