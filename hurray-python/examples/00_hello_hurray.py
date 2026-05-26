"""
Smoke test: import hurray, check version, and verify runtime mode API.

Run after `maturin develop`:
    python examples/00_hello_hurray.py
"""

import hurray

print(f"hurray version : {hurray.__version__}")
print(f"strict mode    : {hurray.is_strict()}")

# set_strict(True) is a no-op in strict-mode-only builds
hurray.set_strict(True)
print("set_strict(True): ok")

# set_strict(False) is reserved until a future release
try:
    hurray.set_strict(False)
except NotImplementedError as exc:
    print(f"set_strict(False): NotImplementedError (expected) — {exc}")

# strict() context manager is a no-op
with hurray.StrictCtx():
    print("StrictCtx: entered and exited cleanly")

# relaxed() context manager is reserved
try:
    with hurray.RelaxedCtx():
        pass
except NotImplementedError as exc:
    print(f"RelaxedCtx: NotImplementedError (expected) — {exc}")
