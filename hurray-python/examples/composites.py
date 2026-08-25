"""
Composite tensors from Python (ADR-036).

A composite presents one logical tensor over several real ones: a weight matrix
sharded across devices, a base plus corrections, a group that simply travels
together. The head owns no data — it is a *view* — and the members carry the
bytes.

That is why a composite is its own class rather than a Tensor with a composite
layout. A composite contains tensors; a sparse tensor has data.

Run with:

    python hurray-python/examples/composites.py
"""

import os
import tempfile

import hurray


def tile(offset: int) -> hurray.Tensor:
    """One half of an 8x8 matrix: an 8x4 tile starting at column `offset`."""
    return hurray.Tensor(
        bytes(128), hurray.float32, [8, 4], shard=hurray.Shard([8, 8], [0, offset])
    )


# ── A partition: members tile the head exactly ────────────────────────────────

print("=== A partition ===")

weight = hurray.Composite(
    "partition",
    shape=[8, 8],
    dtype=hurray.float32,
    members=[tile(0), tile(4)],
)

print(f"  {weight!r}")
print(f"  head presents shape={weight.shape} dtype={weight.dtype}")
print(f"  members: {weight.member_count}, each {weight.members[0].shape}")
print(f"  layout: {weight.layout!r}")

# ── The head is stated, never derived ─────────────────────────────────────────

print("\n=== Stated, not derived ===")

try:
    hurray.Composite(
        "partition", shape=[8, 8], dtype=hurray.float32, members=[tile(0)]
    )
except hurray.InvalidDescriptorError as exc:
    print(f"  one tile cannot cover an 8x8 head: {exc}")

print("  (the head is a declaration and the members are evidence — if they")
print("   disagree you hear about it, rather than the head being reshaped to fit)")

# ── A composite is not a tensor ───────────────────────────────────────────────

print("\n=== Not a tensor ===")

print(f"  isinstance(weight, hurray.Tensor): {isinstance(weight, hurray.Tensor)}")
print(f"  has .values:    {hasattr(weight, 'values')}")
print(f"  has __dlpack__: {hasattr(weight, '__dlpack__')}")
print("  the head owns no buffers, so there is nothing to hand a consumer —")
print("  the data belongs to the members, each an ordinary hurray.Tensor")

# ── Streaming ─────────────────────────────────────────────────────────────────

print("\n=== Over a stream ===")

plain = hurray.Tensor(bytes(16), hurray.float32, [4])

with hurray.StreamWriter() as writer:
    writer.write(plain)
    writer.write(weight)
    writer.write(plain)

items = list(hurray.StreamReader(writer.getvalue()))
print(f"  wrote 1 tensor, 1 composite, 1 tensor — read back {len(items)} items:")
for index, item in enumerate(items):
    print(f"    {index}: {type(item).__name__}")
print("  (the composite is ONE item; its members do not arrive separately)")

# ── Files ─────────────────────────────────────────────────────────────────────

print("\n=== In a file ===")

path = os.path.join(tempfile.mkdtemp(), "model.hrry")
hurray.save(path, {"weight": weight, "bias": plain})

loaded = hurray.load(path)
print(f"  keys: {sorted(loaded)}")
print(f"  weight is a {type(loaded['weight']).__name__}, equal to what we wrote: "
      f"{loaded['weight'] == weight}")
print("  (members get index entries named weight.0 and weight.1, but they do not")
print("   come back as top-level entries — on the wire they belong to the head)")

member = hurray.load(path, names=["weight.0"])["weight.0"]
print(f"  asked for weight.0 by name: {type(member).__name__} shape={member.shape}")

# ── Nesting ───────────────────────────────────────────────────────────────────

print("\n=== Nesting ===")

group = hurray.Composite(
    "group",
    shape=[4],
    dtype=hurray.float32,
    members=[hurray.Tensor(bytes(16), hurray.float32, [4]) for _ in range(2)],
)
outer = hurray.Composite("group", shape=[4], dtype=hurray.float32, members=[group])

print(f"  {outer!r}")

with hurray.StreamWriter() as writer:
    writer.write(outer)
(back,) = list(hurray.StreamReader(writer.getvalue()))
print(f"  nested tree survives a round trip: {back == outer}")

# ── An overlay: a base plus corrections ───────────────────────────────────────

print("\n=== An overlay ===")

SIZE = 32

# The base spans the whole index space. In an SpQR-style model this is the
# quantized weight matrix; here it is plain float32 to keep the example short.
base = hurray.Tensor(
    bytes(4 * SIZE * SIZE),
    hurray.float32,
    [SIZE, SIZE],
    shard=hurray.Shard([SIZE, SIZE], [0, 0]),
)

# The correction is sparse: the outliers quantization would have ruined.
correction = hurray.Tensor(
    bytes(4 * 16),
    hurray.float32,
    [SIZE, SIZE],
    aux_buffers=[bytes(16 * 2 * 8)],       # packed [nnz, rank] uint64 indices
    layout=hurray.CooLayout(nnz=16, is_sorted=True),
    shard=hurray.Shard([SIZE, SIZE], [0, 0]),
)

weights = hurray.Composite(
    "overlay",
    shape=[SIZE, SIZE],
    dtype=hurray.float32,
    members=[base, correction],
    combine_op="replace",
)

print(f"  {weights!r}")
print(f"  roles: {weights.member_roles}")
print(f"  combine_op: {weights.layout.combine_op}")

print("\n  You do not say which member is the base. The format fixes the roles by")
print("  position — member 0 is the base and must span the index space, the rest")
print("  are corrections — so stating them would only be a way to get them wrong.")

print("\n  The roles are read from the composite, not from a member: a tensor has")
print("  no role; a tensor inside an overlay does.")
print(f"    the base is still your own object: {weights.members[0] is base}")
print(f"    and it has no role of its own:     {not hasattr(base, 'member_role')}")

with hurray.StreamWriter() as writer:
    writer.write(weights)
(returned,) = list(hurray.StreamReader(writer.getvalue()))

print(f"\n  survives a round trip: {returned == weights}")
print(f"  with its roles intact: {returned.member_roles}")

# "add" combines by summation instead of precedence.
summed = hurray.Composite(
    "overlay",
    shape=[SIZE, SIZE],
    dtype=hurray.float32,
    members=[base, correction],
    combine_op="add",
)
print(f"\n  combine_op='add': the value is base + correction where the correction")
print(f"  is present, and just the base outside it ({summed.layout.combine_op})")
