# ADR-041: `hurray-io` splits into a sans-io core; every binding drives it

## Status

Proposed (2026-10-17)

Resolves the bridge question `HurrayReader` and `HurrayWriter` have carried as bare
typedefs since C ABI 1, and which **ADR-034** deferred alongside "does streaming reuse
this handle". Addresses [issue #233](https://github.com/pgillet/hurray/issues/233).

Unblocks three parked things: Level 2 for the C surface, the seven streaming and file
rows that are ❌ for `hurray-ffi`, and the half of
[#174](https://github.com/pgillet/hurray/issues/174) that waits on a Layer 5 C surface —
which puts this upstream of the 1.0 gate in
[#225](https://github.com/pgillet/hurray/issues/225).

## Context

`hurray-io` is generic over `tokio::io::AsyncRead`/`AsyncWrite` and every entry point is
an `async fn`: `StreamReader::next_tensor`, `StreamReader::next_item`,
`FileReader::read_tensor`, `FileWriter::finish`. The C ABI is six versions of blocking,
`HurrayStatus`-returning functions. Nothing connects them, so the generated status page
shows seven consecutive ❌ in the `hurray-ffi` column — stream write, stream read, stream
composites, file write, file read, file KV metadata, file composites — against ✅ in both
other columns. A C, Go or Julia consumer can decode a descriptor handed to it and cannot
obtain one.

### What the bindings do today

`hurray-python` has already answered this question, and its answer is the thing to look
at before choosing another one. Every I/O entry point in the bindings is

```rust
py.detach(|| runtime.block_on(/* … */))
```

— the GIL is released so other Python threads keep running, but the calling coroutine is
blocked for the duration. There is **no `async` method in the Python surface at all**.

The cost is recorded in the code, at the point where it was paid
(`hurray-python/src/file_io.rs`):

```rust
// One block_on per step rather than one around the whole loop: a composite's
// borrowed node tree is built by a *synchronous* continuation (see
// with_file_nodes), and block_on cannot be called from inside another.
```

A nested `block_on` panics, so the composite file writer is structured around the bridge
rather than around the problem. That is one wart today; it is the shape of every future
wart in the same place.

This matters more than it looks, because `hurray-python` is not a courtesy binding. The
parity rule in `CLAUDE.md` says anything the Rust layers can express about a tensor, the
Python bindings must express too — and Python is the canonical runtime with an event
loop, which is exactly the audience the one async interface in this domain was built for.

### What comparable projects chose

Surveyed for #233. The useful axis is not sync versus async but **who drives progress**:
whose thread runs the work, and who owns the file handle or socket.

| Implementation | Whose thread | Whose I/O | What the C caller must own |
|---|---|---|---|
| Arrow C stream interface | caller, blocked | producer | nothing |
| GGUF / llama.cpp | caller, blocked | library | nothing |
| zenoh-c | library | library | a callback, a drop policy |
| rustls-ffi | caller | caller | sockets, buffers |
| quiche | caller | caller | sockets, buffers, timers |
| gRPC core | caller, polling | library | threads, a poll loop |
| UniFFI | caller's executor | library | an event loop |
| Arrow async device stream handler | producer | producer | an event loop |

Four findings bear on the decision.

1. **Nothing in this domain exposes async over a C ABI.** Arrow — whose C data interface
   Hurray's is modelled on — documents `ArrowArrayStream::get_next` as a *blocking
   pull-style iteration function*, with `errno` codes and an in-band end-of-stream
   (`get_next` succeeds and returns an already-released array). GGUF is blunter:
   `llama_model_load_from_file` takes a path and blocks, mmapping tensor data.
2. **The one async interface in the domain was built for event-loop languages, not C.**
   Arrow's `ArrowAsyncDeviceStreamHandler` is consumer-allocated and producer-pushed, and
   its stated motivation is that the synchronous interface is difficult to use with
   anything resembling Python `async`/`await` — the targets named being R, Python and
   Ruby. It remains **experimental** years after the blocking interface stabilised.
3. **The sans-io entries answer a different question.** rustls-ffi and quiche are not C
   API shapes; they are architectures from which a small C API falls out. rustls-ffi's own
   claim is the point: *"Rustls doesn't do any I/O on its own … because of that it can be
   used equally well in a blocking or non-blocking I/O context."* quiche goes further, and
   makes the application own the event loop and its timers too, because QUIC has a clock.
4. **The alternatives that give C real concurrency charge for it.** gRPC core creates no
   threads of its own and requires the application to poll a completion queue
   *"preferably in a tight loop"* or no I/O progress happens. UniFFI hands futures across
   by making the foreign side supply the executor, does not support cancellation at all,
   and assumes an event loop exists on the other side — which, for C, it does not.

So the C surface should be blocking. That much every comparable project agrees on, and it
was never really in doubt. The open question is what sits behind it.

## Decision

### 1. `hurray-io` splits into a sans-io core and an async shell

The core is a state machine that consumes and produces bytes and touches no file, socket
or runtime. The Tokio `StreamReader`/`StreamWriter`/`FileReader`/`FileWriter` become thin
drivers over it, keeping their current public signatures.

This is the layer boundary rustls and quiche are built on, and it is the format's own
shape: descriptors precede their buffers, the encoding is self-delimiting, back-references
are forbidden, and the file writer emits in a single forward pass. A format designed so a
reader never looks backwards is a format whose reader wants to be a state machine.

Half of it already exists. `TensorDescriptor::decode` is pure and lives in `hurray-core`;
`stream/frame.rs` is 71 lines that read a 10-byte prefix, take `descriptor_length`, read
that many bytes and hand them to the core. That is a state machine, written in `async`.

### 2. Zero-copy is a precondition, not a consequence

The core MUST hand the driver an **aligned, correctly sized destination to fill**, and
MUST NOT accept bytes the driver already holds elsewhere. `rustls_connection_read_tls`
is the model: the caller supplies a callback that writes *into* rustls's own buffer.

This is stated as a decision rather than an implementation note because the naive
byte-fed core — caller buffer in, library buffer out — costs a copy per tensor, and
"zero-copy first" is a project principle. A sans-io core that copies is a net loss and
MUST NOT be merged. § 7 makes this the gate on the whole effort.

### 3. Two drivers, because the format has two access patterns

The sequential driver feeds bytes forward: stream read and write, and the file writer's
single pass. The random-access driver answers **range requests** — the core asks for bytes
`[a, b)` and is given them — because the file reader seeks: the trailer sits at
`file_size - 40`, the index is located from it, and tensors are read by name.

Whether that is one state machine with two drivers or two machines is left to § Open
Questions; what is decided here is that a sequential byte feed alone does not serve
`FileReader`, so the core's interface admits both.

### 4. The C ABI is blocking, over a caller-supplied transport vtable

`HurrayReader` and `HurrayWriter` become real handles. Every call is synchronous and
returns a `HurrayStatus`. Transport is a vtable of synchronous callbacks — read, write,
seek, size — supplied by the caller at construction.

Not a path and not a file descriptor: `FileReader` must seek, file descriptors are
already a portability wart of their own
([#215](https://github.com/pgillet/hurray/issues/215)), and a vtable composes with a
socket, a pipe, a memory span, or a caller's own object. It is the structural choice Arrow
makes when it hands a consumer four producer callbacks.

**No Tokio reaches a C consumer.** With the core sans-io there is no runtime inside the
handle, no `block_on`, no library-owned thread pool to document, no ambiguity about which
thread a buffer release callback fires on — and release callbacks already cross this ABI —
and the handle's thread-affinity rule is one sentence.

### 5. `hurray-python` keeps its synchronous API and gains an asynchronous one

The existing sync surface does not change; its `py.detach(|| block_on(…))` calls become
`py.detach(|| /* drive the core */)`. Alongside it, an `async` surface driven by Python's
own event loop — `await reader.next_item()` — with no Rust runtime involved. This is
UniFFI's "the foreign bindings supply the executor" without anything being generated, and
it is the parity rule satisfied in the one direction it never has been.

### 6. Composite traversal becomes an explicit stack

`StreamReader::read_item` currently recurses through nested composites via
`Box::pin(self.read_item(depth + 1))`. A state machine cannot, so the nesting becomes an
explicit stack. This is an improvement to be collected rather than a cost to be absorbed:
`max_composite_depth` is currently enforced against an implicit call depth, and becomes a
bound on a visible structure that a C caller can also walk.

### 7. Sequencing, and the condition that stops it

1. **Sequential stream core.** The clearest shape and the smallest file. Prove the
   fill-my-buffer contract of § 2 keeps reads zero-copy *here*, with a test that fails if
   a copy appears.
2. **Rust drivers over it**, with `hurray-io`'s public API and its full test suite
   unchanged. If that suite needs edits beyond imports, the split is wrong.
3. **The C surface**: handles, the transport vtable, stream read and write. ABI version
   bump, decided in its own pass.
4. **Random-access driver and the file core**, then the C file surface.
5. **Python async surface.**
6. **Composites across both**, last, because it needs § 6 and the C sum-type answer.

**Stop condition:** if step 1 cannot preserve zero-copy, the effort stops there. The
fallback is then the C-only blocking façade in § Alternatives, which this ADR rejects on
architecture rather than on feasibility, and which step 1 leaves untouched.

## Alternatives Considered

**A blocking façade over a hidden runtime, for C only.** Each `HurrayReader` owns a
current-thread Tokio runtime and `block_on`s one operation per call; `hurray-io` is
unchanged. This was the first recommendation for #233 and it works — it is what Arrow's C
consumers live with. Rejected because it builds a second copy of `hurray-python`'s wall
instead of removing the first, leaves the nested-`block_on` constraint in place for every
future binding, and puts a Tokio runtime inside a C consumer's binary to do work that
needs no concurrency. Retained explicitly as the fallback under § 7's stop condition.

**A completion queue (gRPC core's model).** Buys genuine concurrency, at the price of a C
consumer that must donate threads and poll in a tight loop or see no progress at all. No
consumer has asked for concurrency; several will want a single blocking call.

**Futures across the C boundary (UniFFI's model).** Requires an executor on the far side,
and C does not come with one. Note that this is precisely what § 5 does for *Python*,
where an executor is a given — the model is right for one binding and wrong for the other,
which is itself an argument for putting the split below both.

**A library-owned background runtime with completion callbacks (zenoh-c's shape).** Puts
a thread-safety contract on every handle and makes buffer-release-callback timing the
hardest thing in the documentation to state correctly.

**Leave the C ABI without I/O.** The status quo, and defensible for exactly as long as
nobody writes a non-Python binding. It also leaves #174 — and therefore the 1.0
compatibility contract — blocked on a question nobody is answering.

## Consequences

- The seven streaming and file rows flip for `hurray-ffi`, and Level 2 for the C surface
  stops being "deferred pending the async bridge design" because the bridge exists.
- `hurray-python` gains an async surface. An asyncio application stops needing a thread
  pool to read a Hurray stream.
- Surgery on shipped, tested code: 1,879 lines across `hurray-io/src/stream/` and
  `hurray-io/src/file/`, released in 0.1.0. The public API is intended to survive
  unchanged, which § 7 step 2 turns into a check rather than a hope.
- `hurray-io` grows an internal API with its own tests and documentation. This only breaks
  even if the Tokio layer genuinely shrinks to a driver; if both layers end up
  substantial, the result is net new surface, which `CLAUDE.md` calls slop. Reviewers
  should hold step 2 to that standard.
- A future WASM, Go, or Node binding inherits a core it can drive, rather than arriving
  with its own bridge question.
- The C surface still has no cancellation, and this ADR does not invent one. Arrow's has
  none either and UniFFI declines to support it where the host can.

## Required Documentation Amendments

- `docs/impl/c-ffi.md` — the reader and writer handles, the transport vtable, the
  blocking contract, and the thread-affinity rule.
- `docs/impl/rust-reference.md` — the core/driver split, and the fill-my-buffer contract
  as a requirement on an implementation rather than an internal detail.
- `docs/impl/python-bindings.md` — the async surface and its relationship to the sync one.
- `docs/cookbook/` — a C consumer reading a stream through the vtable, and the Python
  async counterpart on the same page, per the parity rule.
- `website/coverage-matrix.toml` — the seven streaming and file rows' `c_symbols`, and the
  `level-2` row.
- `docs/SUMMARY.md` — add this ADR.

## Open Questions Deferred

- **One state machine or two.** Whether the sequential and random-access drivers front one
  core or two. Settled by step 1 and step 4, not before: the sequential core's shape is
  what makes the answer obvious.
- **The read result versus #174.** `next_tensor` yields a descriptor plus its buffers,
  exactly the pairing #174 folds into one `HurrayTensor`. Whichever lands first constrains
  the other; this ADR deliberately does not decide it, but it does remove the reason #174
  was postponed, so #174 should be re-examined when step 3 is scheduled.
- **Composites as a C sum type.** `StreamItem` is a tensor or a composite head with
  members, and the C surface has no way to express a sum type. § 6's explicit stack has to
  produce something C can walk.
- **Cancellation on the Python async path.** Unexpressible in C and therefore unanswered
  there, but an `await` that cannot be cancelled is a surprise in Python. Needs a stated
  answer in step 5.
- **A late `finish` failure.** The file writer's footer index is written last, so a writer
  that fails after the data leaves a file with no index. What both surfaces promise about
  that state is unstated today and should not stay that way.
