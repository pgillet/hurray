"""Allocating NumPy arrays that Hurray can share without copying (ADR-037 § 6a).

`hurray.from_numpy` copies most arrays. Not out of caution — the format requires
every buffer to start on a 64-byte boundary and NumPy does not promise one. A
large array served by a fresh mmap never has it (glibc puts a 16-byte chunk
header before the pointer); any other array's address is a matter of heap
history. Either way it is not something a producer can arrange.

NEP 49 is the way out, and it is NumPy's own: alignment is the first motivation
the NEP lists. NumPy considered guaranteeing it, declined, and shipped a
pluggable data-memory allocator instead. So "bring your own allocator" is the
sanctioned answer to exactly this question.

Run with:

    python hurray-python/examples/aligned_allocator.py
"""

import threading

import numpy as np
import numpy._core.multiarray as multiarray

import hurray

BIG = 1 << 20  # 4 MiB of float32 — comfortably past glibc's mmap threshold


def alignment_of(array: np.ndarray) -> int:
    """The largest power of two the array's base address is a multiple of."""
    address = array.__array_interface__["data"][0]
    alignment = 1
    while alignment < hurray.PAGE_ALIGNMENT and address % (alignment * 2) == 0:
        alignment *= 2
    return alignment


# ── The problem ───────────────────────────────────────────────────────────────

print("=== Without the allocator ===")

ordinary = np.zeros(BIG, dtype=np.float32)
print(f"  a {ordinary.nbytes // 1024} KiB array is {alignment_of(ordinary)}-byte aligned")
print(f"  the format needs {hurray.MIN_BUFFER_ALIGNMENT}")

# A slice offset by one element is under-aligned on every machine, which a freshly
# allocated array is not quite: it usually misses 64, but a recycled heap chunk can
# land on it. That unpredictability is the point — you cannot plan around it.
under_aligned = ordinary[1:]

try:
    hurray.from_numpy(under_aligned, copy=False)
except hurray.CopyRequiredError as exc:
    print(f"  so copy=False refuses it: {exc}")

print("  and the default, copy=None, copies instead — silently, on every ingest")

# ── The allocator ─────────────────────────────────────────────────────────────

print("\n=== With it ===")

print(f"  handler before: {multiarray.get_handler_name()}")

with hurray.aligned_allocator():
    print(f"  handler inside: {multiarray.get_handler_name()}")
    weights = np.zeros(BIG, dtype=np.float32)
    small = np.arange(16, dtype=np.float64)

print(f"  handler after:  {multiarray.get_handler_name()}")
print(f"  the {weights.nbytes // 1024} KiB array is now {alignment_of(weights)}-byte aligned")
print(f"  the 128-byte one too: {alignment_of(small)}")

tensor = hurray.from_numpy(weights, copy=False)
print(f"\n  from_numpy(copy=False) accepted it, declaring {tensor.buffer_handles[0].alignment}")

weights[0] = 42.0
print(f"  and the buffer really is shared: {np.asarray(tensor)[0] == 42.0}")
print("  (one allocation, no memcpy — which is what the format promised all along)")

# ── What it does not cover ────────────────────────────────────────────────────

print("\n=== Scope ===")

print(f"  an array allocated outside: {alignment_of(np.zeros(BIG, dtype=np.float32))}-byte aligned")
print("  (whatever that number happens to be — outside the block it is the heap's")
print("   business, not ours; inside, it is always at least 64)")
print("  the handler is per array and thread-local, so it never leaks into code")
print("  that did not ask for it. Each array remembers the handler it was born")
print("  under, and is freed through that one — so an array outlives its block:")
print(f"    weights was allocated inside, and is still here: {weights.shape}")
print(f"    its handler: {multiarray.get_handler_name(weights)}")

# The one sharp edge, worth meeting here rather than in a data loader.
seen = {}


def allocate_on_a_worker():
    seen["handler"] = multiarray.get_handler_name(np.zeros(BIG, dtype=np.float32))


with hurray.aligned_allocator():
    worker = threading.Thread(target=allocate_on_a_worker)
    worker.start()
    worker.join()

print(f"\n  a thread started inside the block: {seen['handler']}")
print("  thread-local means exactly that — child threads do not inherit the")
print("  policy, so arrays they allocate are copied on ingest like any other.")
print("  Enter the block on the thread that allocates.")
