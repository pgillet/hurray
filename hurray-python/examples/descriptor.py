"""The descriptor: what a tensor declares, apart from its bytes.

A Hurray descriptor is self-delimiting and travels ahead of the buffers it
describes. That is what lets a reader consume a stream without framing, and it is
what lets you carry a Hurray tensor inside a container of your own — a message, a
cache entry, a column of something else — by putting the descriptor where you
have room and the buffers beside it.

Run with:

    python hurray-python/examples/descriptor.py
"""

import numpy as np

import hurray

# ── A descriptor comes from a tensor ──────────────────────────────────────────

print("=== What it says ===")

tensor = hurray.Tensor(bytes(192), hurray.float32, [3, 4])
descriptor = tensor.descriptor

print(f"  {descriptor!r}")
print(f"  dtype        {descriptor.dtype.name}")
print(f"  shape        {descriptor.shape}   size={descriptor.size}")
print(f"  layout       {descriptor.layout!r}")
print(f"  buffers      {descriptor.buffer_count}")
print(f"  byte_offset  {descriptor.byte_offset}")
print(f"  version      {descriptor.version}")
print(f"  device       {descriptor.device.kind}")

print("\n  It is the same thing the tensor reports, because it is the same")
print("  descriptor — a Tensor is a descriptor plus buffers:")
print(f"    descriptor.dtype is tensor.dtype:   {descriptor.dtype is tensor.dtype}")
print(f"    descriptor.layout == tensor.layout: {descriptor.layout == tensor.layout}")

# ── On the wire ───────────────────────────────────────────────────────────────

print("\n=== On the wire ===")

wire = descriptor.encode()
print(f"  encoded: {len(wire)} bytes — the spec's worked example is 61")
print(f"  first 16: {wire[:16].hex(' ')}")
print(f"  encoded_len agrees: {descriptor.encoded_len == len(wire)}")

back = hurray.Descriptor.decode(wire)
print(f"  decodes back to an equal descriptor: {back == descriptor}")

print("\n  Self-delimiting: bytes 6-9 hold the total length, so trailing bytes")
print("  are not an error — in a stream, what follows is the data.")
print(f"    with 64 junk bytes appended: {hurray.Descriptor.decode(wire + bytes(64)) == descriptor}")

# ── Carrying a tensor in a container of your own ──────────────────────────────

print("\n=== Your own envelope ===")

payload = {
    "descriptor": tensor.descriptor.encode(),
    "buffers": [
        np.asarray(tensor.buffer(i)).tobytes() for i in range(tensor.buffer_count)
    ],
}

print(f"  descriptor: {len(payload['descriptor'])} bytes")
print(f"  buffers:    {[len(b) for b in payload['buffers']]}")

received = hurray.Descriptor.decode(payload["descriptor"])
print(f"\n  the far side learns what the bytes are before touching them:")
print(f"    {received.dtype.name} {received.shape}, {received.buffer_handles[0].byte_size} bytes,")
print(f"    {received.buffer_handles[0].alignment}-byte aligned, {received.buffer_handles[0].sync_mode}")

# ── A composite head: the descriptor that is only a descriptor ────────────────

print("\n=== A head owns no bytes ===")

composite = hurray.Composite(
    "group",
    shape=[4],
    dtype=hurray.float32,
    members=[hurray.Tensor(bytes(16), hurray.float32, [4])],
)
head = composite.descriptor

print(f"  {head!r}")
print(f"  buffer_count: {head.buffer_count}  (a head declares, its members carry)")
print(f"  round trips:  {hurray.Descriptor.decode(head.encode()) == head}")

# ── Quantization sections stand alone too ─────────────────────────────────────

print("\n=== A quantization section on its own ===")

for scheme in (
    hurray.PerTensorAffine(0.015625, 128),
    hurray.PerChannelAffine.symmetric(axis=0, scale_buffer_index=1),
    hurray.NF4(axis=0, block_size=64, scale_buffer_index=1),
    hurray.MXFP(axis=0, block_size=32, scale_buffer_index=1),
):
    encoded = scheme.encode()
    decoded = hurray.decode_quantization(encoded)
    print(f"  {type(scheme).__name__:18} {len(encoded):>2} bytes -> {type(decoded).__name__}")

print("\n  decode_quantization returns whichever class the section's scheme tag")
print("  names, so you read the bytes without knowing what wrote them. That is")
print("  what a tagged section is for.")
