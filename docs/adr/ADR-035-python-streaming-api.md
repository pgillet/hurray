# ADR-035: The Python streaming API is blocking, and iterates

## Status

Proposed (2026-08-22)

Resolves the design questions in issue #157. Extends `docs/impl/python-bindings.md`
with a streaming section; no existing ADR is amended.

## Context

`hurray-io` implements the streaming interchange format — `StreamWriter::write_tensor`
/ `write_composite` / `finish`, and `StreamReader::next_tensor` / `next_item`. None of
it is reachable from Python, which exposes `load` and `save` only.

So a Python producer can write **files** but not **streams**, and the format's headline
property is missing from the language most of the ecosystem uses: a reader that starts
before the whole input has arrived, and a writer that emits tensors one at a time
without buffering the output. `hurray.StreamError` has been defined and registered
since Layer 8b — an exception for an API that does not exist.

Four questions have to be settled before any code, and the first one settles the rest.

## Decision

### 1. Blocking, with an owned runtime — not `asyncio`

`hurray.StreamReader` and `hurray.StreamWriter` each own a current-thread `tokio`
runtime for their lifetime and release the GIL around every call into it.

`load` and `save` already do this per call, building a disposable runtime and wrapping
the work in `py.detach`. Streaming is long-lived, so the runtime moves from the call to
the object; nothing else about the bridge changes.

An `asyncio` surface is **deferred, not rejected**. It is a second API with its own
integration layer, and the case for it is concurrency across *many* streams — a Python
process multiplexing dozens of peers. Nothing needs that yet, and a Python caller who
does can run blocking readers on threads, which is what the GIL release is for. When a
real multiplexing consumer appears, this ADR should be revisited rather than worked
around; the note in § Open Questions records what evidence would justify it.

Shipping both surfaces now was considered and rejected outright: two APIs for one
protocol is the shape `CLAUDE.md` § Guiding Principles forbids, doubling the surface to
avoid choosing.

### 2. The reader is an iterator; the writer is a context manager

```python
with hurray.StreamWriter(path) as writer:
    for tensor in tensors:
        writer.write(tensor)

for tensor in hurray.StreamReader(path):
    ...                                    # each tensor as it arrives
```

The reader implements `__iter__` / `__next__`, raising `StopIteration` at clean EOF.
That maps exactly onto `next_tensor` returning `Ok(None)`, and it is the shape a Python
caller expects from something that yields values incrementally. It also composes with
everything that consumes an iterator, at no cost.

The writer is a context manager because `finish` **flushes**, and a caller who forgets
it loses however much of the stream was still buffered. Making the close automatic
means that cannot happen silently on the happy path. `finish()` remains available
explicitly for callers who cannot use `with`, and MUST be idempotent so both paths
compose. A writer used after finishing MUST raise `hurray.StreamError`.

> **Correction (2026-08-22):** this section first said `finish` writes a *terminator*.
> It does not — `StreamWriter::finish` flushes and returns the sink. The Hurray stream
> format is self-delimiting per frame and has no end marker; a stream ends at EOF,
> which is the same property that forbids end-of-file indexes. The decision is
> unchanged and the reason is if anything stronger: a forgotten `finish` truncates the
> stream rather than merely leaving it unterminated.
>
> One consequence follows and is worth stating: a stream truncated **exactly** at a
> frame boundary is indistinguishable from a complete one, because EOF is the only end
> marker there is. Truncation mid-frame raises `hurray.StreamError`; truncation on a
> boundary yields a short stream and no error. That is a property of the format, not of
> this binding.

The reader MUST also be usable as a context manager, so a caller can release the
transport deterministically rather than waiting for garbage collection.

### 3. Transports: a path, a file descriptor, or bytes

| Source | Reader | Writer |
|---|---|---|
| filesystem path | `StreamReader(path)` | `StreamWriter(path)` |
| anything with `fileno()` — sockets, pipes, open files | `StreamReader(obj)` | `StreamWriter(obj)` |
| in-memory | `StreamReader(data: bytes)` | `StreamWriter()` → `getvalue()` |

`hurray-io` needs `AsyncRead` / `AsyncWrite`, and a Python file object is neither. Two
bridges were possible and only one is sound.

**Rejected: implementing `AsyncRead` over a Python object's `.read()`.** It would need
the GIL *inside* the poll loop, on a thread that deliberately released it — legal, but
it reintroduces the contention the release exists to avoid, and every read becomes a
reacquisition. It would also make the transport's failure modes Python exceptions
raised from inside a Rust poll.

**Chosen: `fileno()`.** A file descriptor is already what `tokio` wants, and it covers
sockets, pipes, and real files — which is the pipeline case the streaming format exists
for. Objects with no descriptor (`io.BytesIO`) are served by the bytes path, which
covers the rest of what a Python caller would reach for.

The descriptor MUST be duplicated (`dup`) so the stream owns its own and closing one
side does not invalidate the caller's object. The stream MUST close its duplicate on
`finish` or on drop.

### 4. Composites are rejected by name, not skipped

`StreamReader::next_item` yields a tensor **or** a composite. A composite head owns no
buffers, and `hurray.Tensor` cannot represent that — ADR-031 and ADR-032 both deferred
composite authoring from Python for exactly this reason, and ADR-032 § 6 already makes
constructing a tensor with a composite layout raise `hurray.UnsupportedError`.

The Python reader therefore iterates **tensors**, and MUST raise
`hurray.UnsupportedError` naming the composite when it meets one. It MUST NOT skip it:
silently dropping a composite would hand the caller a stream that decoded "successfully"
while losing data, which is worse than refusing.

This is a real gap and is recorded as such rather than papered over. It closes when
composite authoring does.

### 5. `hurray.StreamError` finally means something

Framing errors — a truncated frame, a bad magic, a length that overruns — MUST surface
as `hurray.StreamError`, the exception registered since Layer 8b and unused since. I/O
failures on the transport keep raising `hurray.FileError` (an `OSError` subclass), and
descriptor-level problems keep raising `hurray.InvalidDescriptorError`, so the three
stay distinguishable.

## Alternatives Considered

**An `asyncio` API instead of a blocking one.** Rejected for now under § 1: it is the
right answer only for a multiplexing consumer, which does not exist yet, and it costs an
integration layer that would have to be maintained through every `pyo3` upgrade.

**A callback API** — `hurray.read_stream(src, on_tensor=fn)`. Rejected: it inverts
control for no gain, cannot be composed with `itertools` or a `for` loop, and makes
early exit awkward. The iterator gives the same incrementality in the shape Python
already has.

**Reading the whole stream into a list** — `hurray.load_stream(src) -> list[Tensor]`.
Rejected as the primary API: it buffers the entire input, which is the exact property
the streaming format exists to avoid, and would make the Python surface a worse version
of `load`. It may be added later as a convenience *on top of* the iterator, where its
cost is explicit in its name.

**Exposing `next_item` and returning composites as tuples or dicts.** Rejected: it
would invent a second, ad-hoc representation of a composite in the binding, which the
next pass on composite authoring would then have to keep or break. Refusing is honest
and costs nothing to undo.

## Consequences

**Positive**

- The format's defining property becomes available in Python: incremental in, incremental
  out, no whole-input buffering on either side.
- The API is the one a Python caller would guess — a `for` loop and a `with` block.
- `fileno()` makes sockets and pipes work without a Python-object bridge, so the
  pipeline story is real rather than file-only.
- One protocol surface, not two; the async question stays open without an API standing
  in for its answer.

**Negative**

- **A runtime per stream object.** A current-thread runtime is cheap, but a caller
  holding hundreds of open streams pays for hundreds of them. That is the same caller
  who will want the async API, which is the signal § 1 asks for.
- **Composites are unreadable from Python**, and a stream containing one fails rather
  than degrading. Deliberate, and the alternative is worse.
- **`io.BytesIO` is not accepted directly**, only its `getvalue()`. A caller must know
  which of the two paths their object takes, which is a wart on an otherwise uniform
  constructor.
- **A duplicated descriptor is a resource the caller cannot see.** It is closed on
  `finish` or drop, but a leaked reader leaks an fd until collection.

## Required Documentation Amendments

- `docs/impl/python-bindings.md` — a normative § Streaming section: the two classes,
  the transports, the iterator and context-manager protocols, the composite rejection,
  and the exception mapping.
- `docs/tutorials/python-interop-paths.md` — currently says "the streaming format has no
  Python API yet; it is `hurray-io`, Rust only". That becomes false.
- `docs/cookbook/` — a Python streaming recipe beside the Rust ones in
  `ipc-streaming.md` and `layer-5-streaming-interchange.md`, per issue #147.
- `hurray-python/examples/streaming.py` — runnable producer and consumer.

## Open Questions Deferred

- **An `asyncio` surface.** Revisit when a consumer needs to multiplex many streams in
  one process — that is the case a blocking API cannot serve by adding threads.
- **Composite streaming**, which unblocks when `hurray.Tensor` can represent a
  buffer-less head.
- **Whether the writer should accept anything buffer-like** rather than only
  `hurray.Tensor` — a NumPy array or a DLPack producer could be converted on the way in.
  Convenience, not capability; decide once the core API has users.
