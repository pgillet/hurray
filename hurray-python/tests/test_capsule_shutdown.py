"""An unconsumed native-buffer capsule must not take the process down with it.

A ``__hurray_buffer__`` capsule that is never consumed is finalized by CPython — and
that can happen *during interpreter shutdown*, after the interpreter is no longer
initialized. The destructor must release its reference to the source tensor without
asking PyO3 for a ``Python`` token there: a panic inside an ``extern "C"`` finalizer
cannot unwind, so it aborts the process instead of exiting cleanly.

These run in a subprocess because the failure is the exit status, not an exception.
"""

import subprocess
import sys
import textwrap


def _run(body):
    """Run a snippet in a fresh interpreter; return the CompletedProcess."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_an_unconsumed_capsule_exits_cleanly():
    result = _run(
        """
        import hurray

        t = hurray.Tensor(bytes(16), hurray.float32, [4])
        capsule = t.__hurray_buffer__()   # never consumed, still alive at exit
        print("built")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "built" in result.stdout
    assert "panic" not in result.stderr


def test_an_unconsumed_multi_buffer_capsule_exits_cleanly():
    """The sparse case: several buffers in the list, all owned by the capsule."""
    result = _run(
        """
        import numpy as np
        import hurray

        values = np.array([5.0, 7.0], dtype=np.float32)
        indices = np.array([[0, 0], [1, 1]], dtype=np.uint64)
        t = hurray.sparse_coo(values, indices, [2, 2])
        capsule = t.__hurray_buffer__()   # two buffers, never consumed
        print("built")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "built" in result.stdout


def test_a_consumed_capsule_exits_cleanly():
    """The consumer already freed the context; the destructor must not double-free."""
    result = _run(
        """
        import hurray

        t = hurray.Tensor(bytes(16), hurray.float32, [4])
        received = hurray.from_hurray_buffer(t)
        assert received.shape == (4,)
        print("consumed")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "consumed" in result.stdout


def test_dropping_a_capsule_before_exit_still_works():
    """The ordinary GC path, which was never broken — pinned so it stays that way."""
    result = _run(
        """
        import gc
        import hurray

        t = hurray.Tensor(bytes(16), hurray.float32, [4])
        capsule = t.__hurray_buffer__()
        del capsule
        gc.collect()
        print("collected")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "collected" in result.stdout
