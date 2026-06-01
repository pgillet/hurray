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
