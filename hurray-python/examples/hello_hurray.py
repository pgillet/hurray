"""
Smoke test: import hurray, check version, and verify runtime mode API.

Run after `maturin develop`:
    python examples/hello_hurray.py
"""

import hurray

print(f"hurray version : {hurray.__version__}")
print(f"strict mode    : {hurray.is_strict()}")

# Toggle mode globally.
hurray.set_strict(False)
print(f"after set_strict(False): {hurray.is_strict()}")
hurray.set_strict(True)
print(f"after set_strict(True) : {hurray.is_strict()}")

# StrictCtx: temporarily enter strict mode.
hurray.set_strict(False)
with hurray.StrictCtx():
    print(f"inside StrictCtx       : {hurray.is_strict()}")
print(f"after StrictCtx        : {hurray.is_strict()}")
hurray.set_strict(True)

# RelaxedCtx: temporarily enter relaxed mode.
with hurray.RelaxedCtx():
    print(f"inside RelaxedCtx      : {hurray.is_strict()}")
print(f"after RelaxedCtx       : {hurray.is_strict()}")

# Factory helpers: hurray.strict() / hurray.relaxed()
with hurray.relaxed():
    print(f"inside relaxed()       : {hurray.is_strict()}")
print(f"after relaxed()        : {hurray.is_strict()}")
