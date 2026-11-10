# ADR-042: The asyncio surface is a Python driver over the stream cores

## Status

Accepted (2026-11-01)

Implements **ADR-041 § 9** and § 11 step 5b, and records the three shape decisions
§ 9 left open. Closes ADR-035 § 1's deferral of an `asyncio` surface — § 1 gains an
amendment note rather than being worked around — and closes
[issue #252](https://github.com/pgillet/hurray/issues/252).

The semantics are not decided here. ADR-041 §§ 6, 6a and 6b fixed cancellation,
resumability and the terminal state before any of this was written, deliberately, so that
the hard part was settled while the cores were being built rather than discovered while
being driven. This ADR is about *where the driver lives* and *what it may hold*.

## Context

ADR-035 § 1 chose a blocking Python surface and deferred `asyncio` with a condition
attached: revisit when a real multiplexing consumer appears, rather than work around it.
ADR-041 § 9 is that revisit. It also removes the reason the deferral was comfortable — the
binding no longer owns a `tokio` runtime at all, so "async in Python" no longer means
bridging two executors. It means driving a state machine from the loop that is already
running.

What § 9 fixed: the sync surface does not change, the async surface is driven by Python's
own event loop with no Rust runtime involved, and **the async surface MUST NOT drive the
blocking transport vtable of § 8**. One core, two drivers, no blocking call inside a
coroutine.

What § 9 did not fix: which language the driver is written in, what the boundary between
Python and Rust carries, and how a borrowed-payload write operation survives a suspension.
Those are below.

## Decision

### 1. The driver is Python source, not `pyo3`

`hurray-python` becomes a mixed Rust/Python package: `python-source` in `pyproject.toml`,
the `__init__.py` maturin was generating checked in as-is, and `hurray/aio.py` beside it.

The readiness logic is `loop.add_reader` / `add_writer`, a future per wait, and
`remove_reader` on the way out — about 120 lines that read as what they are in Python, and
as hand-built future plumbing in `pyo3`. ADR-041 § 6a's own pseudo-code is Python-level;
writing the driver in Python keeps the thing being reviewed the same shape as the thing
that was decided.

This is not a restructure: maturin already shipped `hurray/` as a package with a generated
`__init__.py` doing `from .hurray import *`, so the import path, the module identity, and
every `__module__` are unchanged. Nothing moves except what the wheel copies in.

**Consequence accepted:** the package now has Python that CI must exercise in its own
right. The repr contract (`test_reprs.py`) grew a submodule walk and switched from class
*names* to class *identity* — `hurray.StreamReader` and `hurray.aio.StreamReader` are
different classes with the same name, and a name-keyed guard would have let either stand
in for the other.

### 2. The surface is descriptor-only, and the transfer stays in Rust

Python awaits readiness. One synchronous call into the extension then performs the
`read(2)` or `write(2)` **and** reports the byte count to the core before it returns.

Two things follow, and both are the point:

- ADR-041 § 6a's case 2 — bytes landed, count never reported — is unreachable by
  construction. There is no suspension point between the syscall and the report, so no
  cancellation can land there.
- No `memoryview` into the core's destination ever reaches Python. One that outlived a
  single step would be a use-after-free, because `step()` replaces the destination when a
  buffer completes. Keeping the transfer on the Rust side is what prevents such a view
  from existing, rather than documenting that callers must not keep one.

**Consequences stated rather than hidden:**

- **Unix-only** — the same boundary as the sync surface's `fileno()` support
  ([issue #215](https://github.com/pgillet/hurray/issues/215)). A path or `bytes` has no
  readiness to await; a regular file is refused outright, because `epoll` cannot poll one
  and a file read does not block anyway.
- **No async file API.** `hurray.load` and `hurray.save` get no `hurray.aio` counterpart.
  asyncio has no asynchronous random access, so one would be ADR-041 step 5a's API wearing
  a coroutine — which `asyncio.to_thread` already expresses at a cost the caller can see.
- **The descriptor is put in non-blocking mode**, which `dup` shares with the caller's own
  object. Any descriptor an event loop drives must be in that mode regardless.

### 3. `WriteOp<'a>` is held across suspension by a self-referential holder

A `WriteOp` borrows the core, the descriptors and the payload slices for as long as it
lives, and an async writer has to hold one across a suspension. Neither alternative
survives inspection:

- **Drop the op when the sink would block.** Mid-item that sets the core's `torn` flag and
  poisons the writer by ADR-041 § 6b — so a stream larger than a socket buffer could never
  be written at all.
- **Buffer the item's wire bytes and write them later.** That is the second,
  tensor-sized payload copy ADR-041 § 2 exists to forbid: stop-condition territory, not a
  trade-off.

So one `#[pyclass]` owns a boxed `StreamWriterCore`, the pinned payload, and the op with
its lifetime erased to `'static`. The unsafe is one `erase` helper with its invariants
written out, and the type's field order is load-bearing: `op` is declared first because
Rust drops fields in declaration order and the op borrows every field below it.

**`WriteOp::park()` / `core.resume(..)` in `hurray-io` was considered and rejected.** It
would publish public surface for exactly one consumer, and parking a composite means
rebuilding the frame stack from a node path on resume — moving the hard part into the
crate that is supposed to have no opinion about drivers.

### 4. A composite's borrowed node tree is pinned in a flat arena

`composite.rs` builds the borrowed tree for a *synchronous* composite write by
continuation-passing: each node's `&[&[u8]]` lives in the stack frame that built it, so
every node is constructed around the call that consumes it. That is safe, and it cannot
survive an await.

The async writer therefore flattens the tree into an arena of boxed descriptors, boxed
slice vectors and boxed node vectors, children before parents. Boxing each level is the
mechanism: appending to one of the arena's `Vec`s may reallocate the `Vec`, but it never
moves a `Box`'s pointee, so references already handed out stay valid — including across
the move that returns the arena to its owner.

**The two builders are deliberately not merged.** Pointing the synchronous writer at the
arena would trade its safe, stack-scoped builder for an erased-lifetime one in order to
delete duplication, which is the wrong way round: the duplication is two different
answers to two different questions, and only one of them needs `unsafe`. A comment on each
says so, because "unify these" is the obvious wrong move for a future reader.

Async composite *writes* were briefly scoped out of this pass, to be added later behind a
clear error. That was rejected on the grounds that it is the same work either way, plus a
temporary refusal, an issue, and a second pass through the same file to delete both —
while shipping a surface we already knew we would change.

## Consequences

- `hurray.aio.StreamReader` and `hurray.aio.StreamWriter` exist, and the parity rule in
  `CLAUDE.md` is satisfied in the one direction it never has been: anything the sync
  surface can stream, the async one can stream, composites included.
- The bytes are identical. `hurray.aio.StreamWriter` and `hurray.StreamWriter` emit the
  same stream for the same items, which is asserted rather than assumed, and the frozen
  corpus in `hurray-io/tests/fixtures/` now has a fourth reader.
- Cancellation is observable and specified: a cancelled read resumes, a cancelled write
  poisons if and only if the item had already emitted a byte. Both are tested, and the
  poisoned writer's error names the one recovery that is *not* available — a fresh writer
  over the same connection.
- `hurray-io` gained nothing, and needed to be asked for less than expected. The driver
  does not evaluate § 6b's predicate at all: `StreamWriterCore` sets its own `torn` flag at
  an item's first accepted byte and clears it when the item completes, and reports it
  through `is_unusable()` — so *dropping* the cancelled operation is the whole of the
  driver's obligation, and `poisoned()` answers the question afterwards.
  `WriteOp::emitted()` is public and would have made the same decision explicitly; using
  it would have put a second copy of the predicate in the binding, to be kept in step with
  the core's for no new answer. Unlike step 5a, which had to publish
  `FileWriterCore::poison`, this pass needed no new surface in the core crate at all.
- No new coverage-matrix rows. Async is a driver style, not a format capability; the matrix
  describes what the format can express, and the same honesty applied to `level-2` in
  [#240](https://github.com/pgillet/hurray/issues/240).

  > **Affirmed by [#264](https://github.com/pgillet/hurray/issues/264):** this bullet was
  > re-opened as a missing row and upheld, so the two mechanical reasons are recorded here
  > rather than re-derived. `check-coverage-matrix.py` raises when two rows claim one
  > capability, and the Rust probe already proves `stream_read`/`stream_write` *through the
  > tokio async driver* — async streaming in Rust is not an unlisted capability, it is how
  > the listed rows are proven. And neither the `hurray-ffi` cell nor the Unix-only
  > restriction has an honest glyph: ❌ would assert a C gap that § 8 of ADR-041 denies,
  > and ◐ means "partly implemented" where this surface is fully implemented and
  > conditionally *available* — a restriction the table already carries unlabelled for the
  > sync reader's `fileno()` transport. What the matrix gained instead is one sentence in
  > its interchange intro: rows are framings, not driver styles.

## Alternatives Considered

**`loop.sock_recv_into` / `sock_sendall`.** The obvious primitives, and ADR-041 § 6a rules
them out: both perform the transfer inside a loop callback and then resolve a future, so a
cancellation landing between those two events loses the byte count — unrecoverably, since
`0x00` is valid tensor data.

**`asyncio.StreamReader` / `StreamWriter` and the protocol API.** Avoids the lost count
entirely, because cancelled data stays in asyncio's own buffer. It hands back a `bytes`
that must then be copied into the core's destination, which is the copy § 2 forbids.
Resumability cannot be bought with a payload copy.

**A thread-backed façade** — `asyncio.to_thread` around the blocking surface, presented as
`hurray.aio`. Rejected as dishonest packaging: it is the sync API with a thread behind it,
which callers can write in one line and see the cost of. It is also what an async *file*
API would have to be, which is why there isn't one.

**Generating the driver with `pyo3-asyncio` or an equivalent.** Rejected: it exists to
bridge a Rust executor to Python's, and after ADR-041 there is no Rust executor to bridge.

## Open Questions Deferred

- **Whether the readiness wait should use a protocol/transport instead of
  `add_reader`.** `loop.add_reader` is unavailable on Windows' proactor loop, so a
  `Protocol`-based driver is the shape a Windows port would need — along with an answer to
  issue #215's handle-space problem, which is the actual blocker. Revisit if Windows
  support is scheduled.
- **Whether `hurray.aio` should offer a bounded-concurrency helper** — a reader pool over
  many peers. Not until something needs it; the loop already composes with
  `asyncio.gather`, and a helper that only wraps `gather` is surface without capability.
