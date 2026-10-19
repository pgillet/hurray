# ADR-041: `hurray-io` splits into a sans-io core; every binding drives it

## Status

Accepted (2026-10-19)

> Direction approved on review; the invariants in §§ 2–6 were tightened in response to it.
> Accepted once the first implementation pass landed ([#237](https://github.com/pgillet/hurray/pull/237)),
> which answered the destination-allocator question and settled on two state machines
> rather than one. Step 2 followed in
> [#239](https://github.com/pgillet/hurray/pull/239), closing issue
> [#238](https://github.com/pgillet/hurray/issues/238): the Tokio readers and writers became
> drivers over the core, and composite reassembly moved out of the read core into a
> synchronous `CompositeAssembler` — reassembly consumes no bytes, so it is not the byte
> machine's work, and keeping them apart is what lets one core serve both the flat and the
> assembled view of a stream. Steps 3–6 remain.

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

### What the two paths already do, which is not symmetrical

The read path **allocates**: `read_buffers` creates one `BytesMut` per buffer, resizes it,
reads into it and freezes it. The write path **borrows**: `write_tensor(&desc, buffers:
&[&[u8]])` validates each slice's length against its handle, writes the encoded
descriptor, then `write_all`s each caller-owned slice in turn. It never assembles a
contiguous output buffer, and it never copies a payload.

Any core that serves both has to preserve both properties, and they are different
properties. § 2 and § 3 state them separately for that reason.

### An alignment gap that exists today

`buffer-protocol.md` is normative twice over: the base address of every non-empty buffer
MUST be at least 64-byte aligned, and *a reader MAY rely on the declared alignment for
SIMD loads*. `BytesMut` allocates through `Vec<u8>`, so what both readers hand out carries
the allocator's alignment for a 1-byte type — 8 or 16 in practice — while the descriptor
beside it declares 64, or 4096 for a GPU/IPC writer.

That is a live conformance bug, filed as
[#235](https://github.com/pgillet/hurray/issues/235) and fixable without this refactor. It
is recorded here because it is the reason § 5 states alignment as a tested guarantee
rather than assuming the current behaviour is already correct.

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

### 2. On the read path the core supplies the destination

The core MUST hand the driver an **aligned, correctly sized destination to fill**, and
MUST NOT accept payload bytes the driver has already placed somewhere else.
`rustls_connection_read_tls` is the model: the caller supplies a callback that writes
*into* rustls's own buffer.

**What this does and does not promise.** It promises **no payload copy after the transport
writes into the destination** — the transport's write *is* the only copy, and for a socket
that copy is unavoidable. It is not "end-to-end zero-copy", which only an mmap or a shared
memory handle can give and which is a separate path, not this one. Keeping those two
claims distinct matters, because the naive design this rules out — driver buffer in, core
buffer out — adds a *second*, tensor-sized copy on every tensor, and that is what makes it
unmergeable rather than merely suboptimal.

The destination is owned by the core for the lifetime of the item being read, and is
handed to the caller only when that item completes (§ 6).

### 3. On the write path the core stays borrowed and scatter/gather

Symmetrically, and this is not the same requirement restated: the core MUST accept
**borrowed, non-contiguous** payload slices and MUST NOT assemble them into one output
buffer. It emits the encoded descriptor and then yields each caller-owned slice in turn
for the driver to write, which is exactly what `write_tensor` does today with `write_all`
per buffer.

A core that concatenated would copy every payload on the way out, which is the same defect
as § 2's in the other direction, and it would also lose the vectored-write opportunity the
current shape leaves open.

**Borrowed payloads need a stated lifetime, because an async binding cannot assume one.**
A Python coroutine cannot hold arbitrary borrowed memory across a suspension point, so:

- The caller MUST keep every payload slice valid until the core reports that item written.
  The core MUST expose that point explicitly — a write is complete when the core says so,
  not when the last `write` callback returns.
- A binding that cannot make that promise for caller memory MUST pin it for the duration:
  `hurray-python` holds a reference to the backing object (a NumPy array, a buffer handle)
  from the moment a write begins until the core releases it, and the release is what drops
  the reference.
- The core MUST NOT retain a payload pointer after reporting the item written.

### 4. The progress contract is explicit, and partial progress never loses position

The core's driver-facing API is a step function. One request is outstanding at a time, and
the driver answers it before the core produces another:

```text
NeedInput  { dst: &mut [u8] }   driver fills, then reports how many bytes
NeedOutput { src: &[u8] }       driver writes, then reports how many bytes
Item(…)                         one tensor or composite node is complete
Finished                        clean end, at an item boundary
Error(…)                        see § 6 for what survives it
```

The rules that make short and zero-length progress unambiguous:

- A **short read is not EOF.** Reporting `0 < n < dst.len()` is normal: the core keeps its
  position and re-offers the remainder. Only `n == 0` means the transport reached end of
  input — and then end *at an item boundary* is `Finished`, while end *mid-item* is an
  error. `stream/frame.rs` already draws exactly this line ("a 0-byte read before any
  descriptor byte is a clean EOF"), and the core MUST preserve it.
- A **short write is not completion, and a zero-length write is not success.** Reporting
  `n < src.len()` re-offers the remainder. Reporting `n == 0` MUST NOT be treated as
  progress; a driver whose sink is closed MUST say so through a distinct signal, which is
  an error if an item is in flight.
- **The progress API never blocks and never waits.** It does not know what a transport is.
  A driver that cannot proceed without blocking returns to its own executor, and the core
  holds position until the driver comes back. This is the property that lets one core serve
  both a blocking C driver and a Python event loop (§ 8, § 9).
- Byte counts are reported for every `NeedInput` and `NeedOutput`. Nothing is inferred from
  a callback's return value alone.

### 5. Alignment is a declared guarantee with a test that can fail

The destination of § 2 MUST satisfy **the alignment the buffer handle declares** — not 64
as a constant, and not whatever the allocator happens to return. The spec requires that
declaration to be a power of two and at least 64 for a non-empty buffer, and permits a
reader to rely on it for SIMD loads, so the declaration is the contract.

This is stated as a decision because today's implementation does not keep it (#235), and
because `BytesMut` is the obvious thing to reach for and is wrong. An empty buffer
(`byte_size == 0`) stays legal at any declared power of two.

**The test that proves it:** read tensors whose handles declare 64, 128 and 4096 and assert
`ptr as usize % declared == 0` for every buffer handed out, across the multi-buffer and
composite paths. It must fail if the allocation path ever regresses to a plain `Vec`-backed
allocation. `hurray-python/src/numpy_allocator.rs` and `tests/test_aligned_allocator.py`
are the in-repo precedent for both the technique and the test.

### 6. Cancellation, resumability, and the unusable state

Cancellation interrupts the driver's transport operation. It never interrupts the core,
which has no I/O to interrupt.

- **The core is resumable across cancellation if and only if the byte position is known.**
  A driver MUST perform a transport operation and report its byte count as one
  uninterruptible step; cancellation is observable only at a step boundary. Then a
  cancelled `await` leaves the core exactly where the last report left it, and the next
  call continues.
- **Otherwise the core enters an explicit unusable state.** If bytes moved and could not be
  reported, the core MUST be marked unusable by name — a state its API exposes and every
  subsequent call reports — never left to diverge silently from the transport. "Probably
  still fine" is not a state.
- **A partially filled destination is never observable.** It belongs to the core until the
  item completes, so a cancelled read cannot hand anyone a half-written or partially
  initialised buffer.
- **Caller memory outlives the operation that uses it.** A buffer the binding owns —
  Python's included — MUST remain valid until every outstanding native operation has
  released it. The core releases on item completion or on entering the unusable state, and
  the binding drops its reference then, not before.
- **In Python:** a cancelled `await reader.next_item()` leaves the reader usable and the
  next `await` resumes the same tensor. A reader that became unusable raises a distinct,
  named error on next use rather than an arbitrary one.

Deciding this now rather than at step 5 is deliberate: it constrains the core's state
transitions and its ownership model, which are built in step 1.

#### 6a. What the atomic step costs in asyncio, and the shape it forces

The atomicity requirement above is cheap in a blocking C driver and is the hardest thing
in this ADR to honour in Python, because asyncio can deliver `CancelledError` at an
`await` *after* the operation behind it has already moved bytes. Writing the three cases
out is what makes the constraint visible:

```python
dst = core.need_input()            # a memoryview into the core's destination
n = await transport.readinto(dst)  # cancellation point
core.report_filled(n)              # synchronous, never blocks (§ 4)
```

1. Cancelled before any byte lands — nothing moved, position known, **resumable**.
2. Cancelled after bytes landed in `dst` but before the count is reported — the bytes are
   in the core's buffer and the core believes none arrived. Position unknown, and no
   inspection can recover it, because `0x00` is valid tensor data. **Unusable.**
3. The await returned and the report has not run yet — unreachable, *provided* the report
   is a synchronous call with no await between it and the transfer. § 4's non-blocking
   progress API is what guarantees that, so this case is closed by construction.

Case 2 is reachable with exactly the primitive this design wants —
`loop.sock_recv_into(sock, buf)` performs the `recv` in a loop callback and then resolves
a future, and a cancellation that lands between those two events loses the count.

**The apparent escape is a trap.** Using `asyncio.StreamReader.read(n)` avoids case 2
entirely, because cancelled data stays in the StreamReader's own buffer and never reaches
ours — but it hands back a `bytes` object that must then be copied into the core's
destination. That is precisely the second, tensor-sized copy § 2 exists to forbid. A
Python async driver cannot buy resumability with a copy; it would be trading the reason
for the refactor against one of its requirements.

**So the async driver MUST be readiness-driven, not transfer-await-driven.** The `await`
covers *readiness only* (`loop.add_reader`/`add_writer`, or a protocol's
`data_received`); the transfer and its byte-count report happen together inside one
synchronous callback, where no cancellation can interleave. Then case 2 cannot arise and
§ 6's resumable branch is the normal path rather than a theoretical one.

**The write side is not symmetrical, and is worse.** A cancelled read with an unknown
count damages only our own state. A cancelled write with an unknown count has already put
an unknown number of payload bytes on the wire, and a stream cannot be un-sent: the
reader at the far end will see a truncated item. So when the count is unknown on the write
path, the core MUST become unusable *and* the sink MUST be treated as poisoned — a fresh
writer over the same connection is not a recovery. `sock_sendall` is unusable here for
that reason; the same readiness-driven shape is required.

This is step 5's hardest requirement and step 1 has to leave room for it: the report API
must be callable from inside an event-loop callback, which § 4 already requires by making
it non-blocking.

### 7. Two drivers, because the format has two access patterns

The sequential driver feeds bytes forward: stream read and write, and the file writer's
single pass. The random-access driver answers **range requests** — the core asks for bytes
`[a, b)` and is given them — because the file reader seeks: the trailer sits at
`file_size - 40`, the index is located from it, and tensors are read by name.

Whether that is one state machine with two drivers or two machines is left to § Open
Questions; what is decided here is that a sequential byte feed alone does not serve
`FileReader`, so the core's interface admits both.

### 8. The C ABI: a blocking surface over a transport vtable, with stated rules

`HurrayReader` and `HurrayWriter` become real handles. Every call is synchronous and
returns a `HurrayStatus`. Transport is a vtable of callbacks — read, write, seek, size —
supplied by the caller at construction.

Not a path and not a file descriptor: `FileReader` must seek, file descriptors are already
a portability wart of their own
([#215](https://github.com/pgillet/hurray/issues/215)), and a vtable composes with a
socket, a pipe, a memory span, or a caller's own object. It is the structural choice Arrow
makes when it hands a consumer four producer callbacks.

**The vtable is blocking by contract — and that contract belongs to the driver, not the
core.** A C callback that blocks until it has bytes is correct and expected here, precisely
because § 4's progress API is independently non-blocking.

The rules the vtable needs, all of which are ABI surface and none of which may be left to
inference:

- **Partial transfers are legal.** The read callback returns the number of bytes written
  into the destination; fewer than requested is a short read, not an error, and not EOF.
- **Zero is not success on the write side.** The read callback returning `0` means end of
  input. The write callback returning `0` is *not* completion — a closed or failing sink is
  reported through a distinct status, which is an error if an item is in flight.
- **EOF, error and closed are three statuses**, never one.
- **No reentrancy.** A callback MUST NOT call any Hurray function on the handle that
  invoked it. The alternative is a reentrancy and thread-safety contract on every handle,
  bought for a use case nobody has.
- **The context pointer belongs to the caller.** Hurray never frees it. The vtable carries
  an optional release callback for it, symmetric with the buffer release callbacks that
  already cross this ABI (ADR-009).
- **One handle, one thread at a time**, which the absence of an internal runtime makes a
  one-sentence rule rather than a threading chapter.

### 9. Python keeps its synchronous API and gains an asynchronous one

The existing sync surface does not change; its `py.detach(|| block_on(…))` calls become
`py.detach(|| /* drive the core */)`. Alongside it, an `async` surface driven by Python's
own event loop — `await reader.next_item()` — with no Rust runtime involved. This is
UniFFI's "the foreign bindings supply the executor" without anything being generated, and
it is the parity rule satisfied in the one direction it never has been.

**The async surface MUST NOT drive the blocking vtable of § 8.** Python's loop drives the
core directly through § 4's progress API and performs its own awaits for transport. One
core, two drivers, no blocking call inside a coroutine.

### 10. Composite traversal becomes an explicit stack

`StreamReader::read_item` currently recurses through nested composites via
`Box::pin(self.read_item(depth + 1))`, and `StreamWriter::write_composite_unchecked`
recurses the same way. A state machine cannot, so the nesting becomes an explicit stack.

This is an improvement to be collected rather than a cost to be absorbed:
`max_composite_depth` is currently enforced against an implicit call depth, and becomes a
bound on a visible structure that a C caller can also walk.

### 11. Sequencing, and the condition that stops it

1. **Sequential stream core**, proving the contracts that the rest depends on:
   § 2's destination ownership, § 3's borrowed scatter/gather output, § 4's partial-progress
   rules (short read, zero read at a boundary and mid-item, short write, zero write),
   § 5's alignment test, and § 6's resumability — **including at least one nested-composite
   case.** Composite state is where suspension gets interesting, and a core validated only
   on single tensors can be elegant there and awkward the moment a composite must survive
   being suspended mid-member.
2. **Rust drivers over it.** Acceptance is behavioural, not textual: unchanged public
   signatures, unchanged observable behaviour, byte-for-byte wire compatibility on a
   corpus, and the existing integration tests passing. New or adapted test *harnesses* are
   legitimate and expected; what is not legitimate is a changed signature or a changed
   byte.
3. **The C surface**: handles, the transport vtable and its § 8 rules, stream read and
   write. ABI version bump, decided in its own pass.
4. **Random-access driver and the file core**, then the C file surface.
5. **Python async surface**, implementing the semantics § 6 already fixes — readiness-driven
   per § 6a, with a test that cancels mid-tensor and asserts the reader resumes rather than
   poisons, and one that cancels mid-write and asserts it poisons rather than silently
   truncating.
6. **Composites across both surfaces**, including the C sum-type answer. Full C composite
   support lands here; the core's composite handling was proven in step 1.

**Stop condition:** stop if the contract in §§ 2–4 **cannot** be satisfied without an
additional payload copy — a flaw in the abstraction, demonstrated by argument and not by a
single disappointing benchmark. A first prototype that copies once and can be improved is
not that. The fallback is then the C-only blocking façade in § Alternatives, which this ADR
rejects on architecture rather than on feasibility, and which step 1 leaves untouched.

## Alternatives Considered

**A blocking façade over a hidden runtime, for C only.** Each `HurrayReader` owns a
current-thread Tokio runtime and `block_on`s one operation per call; `hurray-io` is
unchanged. This was the first recommendation for #233 and it works — it is what Arrow's C
consumers live with. Rejected because it builds a second copy of `hurray-python`'s wall
instead of removing the first, leaves the nested-`block_on` constraint in place for every
future binding, and puts a Tokio runtime inside a C consumer's binary to do work that
needs no concurrency. Retained explicitly as the fallback under § 11's stop condition.

**A completion queue (gRPC core's model).** Buys genuine concurrency, at the price of a C
consumer that must donate threads and poll in a tight loop or see no progress at all. No
consumer has asked for concurrency; several will want a single blocking call.

**Futures across the C boundary (UniFFI's model).** Requires an executor on the far side,
and C does not come with one. Note that this is precisely what § 9 does for *Python*,
where an executor is a given — the model is right for one binding and wrong for the other,
which is itself an argument for putting the split below both.

**A library-owned background runtime with completion callbacks (zenoh-c's shape).** Puts
a thread-safety contract on every handle and makes buffer-release-callback timing the
hardest thing in the documentation to state correctly.

**A `feed(bytes)` core that returns assembled buffers.** The conventional sans-io shape,
and the reason §§ 2–3 are written as requirements. It copies each payload twice over the
round trip — once into the driver's buffer, once into the core's — which for a format whose
first principle is zero-copy converts the refactor from a gain into a regression.

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
  unchanged, which § 11 step 2 turns into a behavioural check.
- The invariants in §§ 2–6 are binding on *both* possible core shapes, which is what makes
  leaving the one-versus-two-machines question open safe rather than vague.
- `hurray-io` grows an internal API with its own tests and documentation. This only breaks
  even if the Tokio layer genuinely shrinks to a driver; if both layers end up
  substantial, the result is net new surface, which `CLAUDE.md` calls slop. Reviewers
  should hold step 2 to that standard.
- The risk this trades into: complexity moves out of Tokio and into a cross-language memory
  and progress protocol. §§ 2–6 exist to make that protocol specified rather than emergent,
  and they are the part of this ADR to review hardest.
- A future WASM, Go, or Node binding inherits a core it can drive, rather than arriving
  with its own bridge question.
- The C surface still has no cancellation mid-call, and this ADR does not invent one.
  Arrow's has none either and UniFFI declines to support it where the host can. § 6 gives
  Python real cancellation semantics because Python can express them.
- The atomicity rule in § 6 constrains the *shape* of the Python async driver, not just its
  internals: it must be readiness-driven (§ 6a). A driver written the obvious way — `await`
  a transport method that transfers into our buffer — is either unsound under cancellation
  or buys soundness with the copy § 2 forbids. This is the single most likely place for
  this design to be implemented wrongly and still look correct.

## Required Documentation Amendments

- `docs/impl/c-ffi.md` — the reader and writer handles, the transport vtable and every § 8
  rule, the blocking contract, and the thread-affinity rule.
- `docs/impl/rust-reference.md` — the core/driver split, and §§ 2–6 as requirements on an
  implementation rather than internal details.
- `docs/impl/python-bindings.md` — the async surface, its relationship to the sync one, and
  § 6's cancellation and buffer-lifetime semantics.
- `docs/cookbook/` — a C consumer reading a stream through the vtable, and the Python async
  counterpart on the same page, per the parity rule.
- `website/coverage-matrix.toml` — the seven streaming and file rows' `c_symbols`, and the
  `level-2` row.
- `docs/SUMMARY.md` — add this ADR.

## Open Questions Deferred

- **One state machine or two.** Whether the sequential and random-access drivers front one
  core or two. Settled by step 1 and step 4, not before: the sequential core's shape is
  what makes the answer obvious. §§ 2–6 bind either way.
- ~~**Who allocates the aligned destination.**~~ **Answered: a binding may.** The core
  supplies the destination (§ 2) at the declared alignment (§ 5), and `DestinationAlloc` is
  the seam a binding implements so a Python read can land directly in a NumPy array, as
  `numpy_allocator.rs` already does elsewhere. Step 1 made it a `Box<dyn DestinationAlloc>`
  handing out `Box<dyn Destination>`; step 2 made it an associated type on a core generic
  over the allocator instead. The reason is not taste: a boxed trait object erased `Sync`,
  `UnwindSafe` and `RefUnwindSafe` from the core and so from every published type
  containing it, which `cargo semver-checks` caught against 0.1.0's `StreamReader`. The
  generic form keeps those traits, drops a heap allocation per buffer, and lets `freeze`
  take `self` by value. Type erasure is no longer available by default, which costs
  nothing yet: a C allocator is a concrete struct holding the caller's vtable, so step 3
  instantiates the core over that rather than erasing it.
- **The read result versus #174.** `next_tensor` yields a descriptor plus its buffers,
  exactly the pairing #174 folds into one `HurrayTensor`. Whichever lands first constrains
  the other; this ADR deliberately does not decide it, but it does remove the reason #174
  was postponed, so #174 should be re-examined when step 3 is scheduled.
- **Composites as a C sum type.** `StreamItem` is a tensor or a composite head with
  members, and the C surface has no way to express a sum type. § 10's explicit stack has to
  produce something C can walk.
- **A late `finish` failure.** The file writer's footer index is written last, so a writer
  that fails after the data leaves a file with no index. What both surfaces promise about
  that state is unstated today and should not stay that way.
