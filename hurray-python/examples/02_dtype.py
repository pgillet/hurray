#!/usr/bin/env python3
"""Phase 8a.2 smoke test: hurray.Dtype"""
import hurray

# Tier 1 at top level
assert hurray.float32.name == "float32"
assert hurray.float32.is_array_api
assert hurray.float32.bit_width == 32

# Same object via hurray.dtype
assert hurray.float32 is hurray.dtype.float32

# Tier 2 only on hurray.dtype
assert hurray.dtype.int4.bit_width == 4
assert not hurray.dtype.int4.is_array_api
assert not hasattr(hurray, "int4")  # NOT at top level

# Works as dict key (hashable)
dtypes = {hurray.float32: "fp32", hurray.dtype.int4: "i4"}
assert dtypes[hurray.float32] == "fp32"

# from_name round-trip
assert hurray.Dtype.from_name("float32") == hurray.float32

# Unknown name raises
try:
    hurray.Dtype.from_name("not_a_type")
    assert False, "should have raised"
except hurray.InvalidDescriptorError:
    pass

print("02_dtype.py: all assertions passed")
