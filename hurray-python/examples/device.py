#!/usr/bin/env python3
"""Smoke test: hurray.Device — device constants and construction."""
import hurray

# Well-known constant
cpu = hurray.device.cpu
assert cpu.kind == "cpu"
assert cpu.device_id == 0
assert cpu.memory_class == "standard"

# Custom construction
gpu1 = hurray.Device("cuda", 1)
assert gpu1.kind == "cuda"
assert gpu1.device_id == 1

# Equality
assert hurray.Device("cpu") == hurray.device.cpu
assert hurray.Device("cuda", 0) != hurray.Device("cuda", 1)

# Hashable (frozen)
device_map = {hurray.device.cpu: "cpu0"}
assert device_map[hurray.Device("cpu")] == "cpu0"

# Unknown kind raises
try:
    hurray.Device("tpu")
    assert False, "should have raised"
except hurray.InvalidDescriptorError:
    pass

print("03_device.py: all assertions passed")

# ── Vendor devices ────────────────────────────────────────────────────────────

print("\n=== Private tags ===")

# 0xF0-0xFE is the private range: an agreement between one producer and one
# consumer. The spec gives them no names, so the tag is the identity.
custom = hurray.Device(0xF2)

print(f"  {custom!r}")
print(f"  kind={custom.kind!r}  tag=0x{custom.tag:02X}  is_private={custom.is_private}")

other = hurray.Device(0xF5)
print(f"\n  two vendor devices are different things:")
print(f"    {custom!r}")
print(f"    {other!r}")
print(f"    equal: {custom == other}")
print("  (kind is 'private' for both, which is why repr carries the tag —")
print("   without it a reader could not tell which device it was looking at)")

# The memory class has a private range too.
vendor_memory = hurray.Device(0xF2, 0, memory_class=0xF1)
print(f"\n  a vendor memory class on it: {vendor_memory!r}")
print(f"    memory_class_tag = 0x{vendor_memory.memory_class_tag:02X}")

tensor = hurray.Tensor(bytes(4096), hurray.float32, [1024], device=vendor_memory)
decoded = hurray.Descriptor.decode(tensor.descriptor.encode())
print(f"\n  both tags survive the wire: "
      f"0x{decoded.device.tag:02X} / 0x{decoded.device.memory_class_tag:02X}")

# What is not accepted, and why.
print("\n  Reserved bytes are refused rather than guessed at:")
for tag, why in ((0x09, "reserved for a future version"), (0xFF, "permanently invalid")):
    try:
        hurray.Device(tag)
    except hurray.InvalidDescriptorError:
        print(f"    0x{tag:02X}: {why}")

print("\n  A private tag means an agreement you either hold or do not. A reader")
print("  without it should refuse the tensor rather than guess — which is what")
print("  hurray.layout_tag_kind() is for on the layout side.")
