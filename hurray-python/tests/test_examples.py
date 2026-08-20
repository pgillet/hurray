"""Every example in ``hurray-python/examples/`` must still run.

The examples are documentation people copy, and they rot silently: nothing imports
them, so a renamed attribute or a changed protocol breaks them without breaking a
build. Four of them were broken at once — a stale ``hasattr`` assertion, a dtype
that had moved to the ``hurray.dtype`` submodule, int32 indices where the format
requires uint64, and a NumPy idiom removed in 2.1 — and none of it was visible.

This is a smoke test, not a unit test: it asserts that each script exits zero.
Most of the examples already assert their own claims, so running them checks
rather more than "it did not crash".

Each example is a separate case, so a failure names the file that broke.
"""

import pathlib
import subprocess
import sys

import pytest

EXAMPLES_DIR = pathlib.Path(__file__).resolve().parent.parent / "examples"

# Collected at import time so a new example is picked up with no edit here.
EXAMPLES = sorted(p for p in EXAMPLES_DIR.glob("*.py") if not p.name.startswith("_"))


def test_there_are_examples_to_run():
    """Guards against the glob silently matching nothing and vacuously passing."""
    assert EXAMPLES, f"no examples found in {EXAMPLES_DIR}"


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_example_runs(example):
    result = subprocess.run(
        [sys.executable, str(example)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"{example.name} exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
