"""
Smoke test: verify the hurray exception hierarchy.

Run after `maturin develop`:
    python examples/01_errors.py
"""

import hurray

# --- Exception classes exist and have correct base types ---

assert issubclass(hurray.InvalidDescriptorError, ValueError), \
    "InvalidDescriptorError must subclass ValueError"
assert issubclass(hurray.BufferError, ValueError), \
    "hurray.BufferError must subclass ValueError"
assert issubclass(hurray.UnsupportedError, NotImplementedError), \
    "UnsupportedError must subclass NotImplementedError"
assert issubclass(hurray.InternalError, RuntimeError), \
    "InternalError must subclass RuntimeError"
print("Exception class hierarchy: ok")

# --- hurray.BufferError is distinct from builtins.BufferError ---

assert hurray.BufferError is not BufferError, \
    "hurray.BufferError must be distinct from builtins.BufferError"
assert not issubclass(hurray.BufferError, BufferError), \
    "hurray.BufferError must NOT subclass builtins.BufferError"
print("hurray.BufferError vs builtins.BufferError: distinct (ok)")

# --- Exceptions can be raised and caught by base class ---

try:
    raise hurray.InvalidDescriptorError("bad header magic")
except ValueError as exc:
    print(f"InvalidDescriptorError caught as ValueError: {exc}")

try:
    raise hurray.UnsupportedError("int4 not supported via DLPack")
except NotImplementedError as exc:
    print(f"UnsupportedError caught as NotImplementedError: {exc}")

try:
    raise hurray.InternalError("unexpected Rust panic: index out of bounds")
except RuntimeError as exc:
    print(f"InternalError caught as RuntimeError: {exc}")

print("All error hierarchy checks passed.")
