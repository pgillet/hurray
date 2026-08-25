"""
The streaming interchange format from Python (ADR-035).

A file is written once and read back whole. A *stream* is different: the writer
emits tensors one at a time without buffering the output, and the reader gets
each tensor as soon as its bytes have arrived. That is the property the format
exists for, and until now it was reachable only from Rust.

Run with:

    python hurray-python/examples/streaming.py
"""

import socket
import struct
import threading

import hurray


def tensor(seed: float) -> hurray.Tensor:
    return hurray.Tensor(
        struct.pack("4f", seed, seed + 1, seed + 2, seed + 3), hurray.float32, [4]
    )


# ── In memory: the shortest possible round trip ───────────────────────────────

print("=== In memory ===")

with hurray.StreamWriter() as writer:
    for i in range(3):
        writer.write(tensor(float(i)))

encoded = writer.getvalue()
print(f"  wrote 3 tensors as {len(encoded)} bytes")

for index, received in enumerate(hurray.StreamReader(encoded)):
    print(f"    tensor {index}: shape={received.shape} dtype={received.dtype}")

# ── Incremental: a tensor before the stream ends ──────────────────────────────

print("\n=== Incremental ===")

reader = hurray.StreamReader(encoded)
first = next(reader)
print(f"  first tensor available immediately: shape={first.shape}")
print(f"  and {len(list(reader))} more still to come")
print("  (nothing buffered the whole sequence to get that first one)")

# ── Over a socket: the case the format is for ─────────────────────────────────

print("\n=== Over a socket ===")

producer, consumer = socket.socketpair()


def produce():
    """Emit tensors one at a time, as a real producer would."""
    with hurray.StreamWriter(producer) as sender:
        for i in range(4):
            sender.write(tensor(float(i * 10)))
    producer.shutdown(socket.SHUT_WR)


thread = threading.Thread(target=produce)
thread.start()

count = 0
for received in hurray.StreamReader(consumer):
    count += 1
    print(f"  received tensor {count}: shape={received.shape}")

thread.join()
print(f"  {count} tensors crossed the socket, none of them buffered whole")
producer.close()
consumer.close()

# ── The stream owns its own descriptor ────────────────────────────────────────

print("\n=== Descriptor ownership ===")

mine, theirs = socket.socketpair()
with hurray.StreamWriter(mine) as sender:
    sender.write(tensor(1.0))

# The writer duplicated the descriptor, so finishing did not close this socket.
# A send on a closed socket would raise OSError; this one does not.
mine.send(b"still mine")
print("  send succeeded after the stream finished — the socket is still ours")
print(f"  (the peer's queue holds the stream, then those bytes: {len(theirs.recv(4096))} read)")
mine.close()
theirs.close()

# ── Multi-buffer tensors travel whole ─────────────────────────────────────────

print("\n=== A sparse tensor keeps all three buffers ===")

csr = hurray.Tensor(
    struct.pack("2f", 5.0, 7.0),
    hurray.float32,
    [2, 2],
    aux_buffers=[struct.pack("2Q", 0, 1), struct.pack("3Q", 0, 1, 2)],
    layout=hurray.CsrLayout(nnz=2),
)

with hurray.StreamWriter() as writer:
    writer.write(csr)

(back,) = list(hurray.StreamReader(writer.getvalue()))
print(f"  layout={back.layout!r} buffers={back.buffer_count} nnz={back.nnz}")

# ── What a truncated stream does ──────────────────────────────────────────────

print("\n=== Truncation ===")

try:
    list(hurray.StreamReader(encoded[:-10]))
except hurray.StreamError as exc:
    print(f"  mid-frame cut: {exc}")

print("  (a cut exactly on a frame boundary reads as a shorter stream instead —")
print("   frames are self-delimiting and EOF is the only end marker there is)")

# ── Reading a stream you did not produce ──────────────────────────────────────

print("\n=== Limits ===")

tensor = hurray.Tensor(bytes(4096), hurray.float32, [1024])
with hurray.StreamWriter() as writer:
    writer.write(tensor)
wire = writer.getvalue()

print(f"  a {len(wire)}-byte stream carrying one 4 KiB tensor")

# A descriptor's length field is read before its contents, so a hostile stream
# can ask a reader to allocate whatever it likes. The limits bound one frame.
try:
    list(hurray.StreamReader(wire, max_buffer_bytes=1024))
except hurray.StreamError as exc:
    print(f"  max_buffer_bytes=1024: {exc}")

try:
    list(hurray.StreamReader(wire, max_descriptor_bytes=8))
except hurray.StreamError as exc:
    print(f"  max_descriptor_bytes=8: {exc}")

guarded = hurray.StreamReader(
    wire,
    max_descriptor_bytes=1 << 20,   # 1 MiB
    max_buffer_bytes=512 << 20,     # 512 MiB
    cross_machine=True,
)
print(f"  sensible limits: {len(list(guarded))} tensor read")

print("\n  max_descriptor_bytes defaults to 16 MiB; max_buffer_bytes is unbounded,")
print("  because a legitimate tensor can be enormous. Set it when the peer is")
print("  not one you control.")

# ── Across a machine boundary ─────────────────────────────────────────────────

print("\n=== cross_machine ===")

with hurray.StreamWriter(cross_machine=True) as strict:
    strict.write(tensor)

print(f"  writer accepted it: {len(strict.getvalue())} bytes")
print(f"  reader accepted it: {len(list(hurray.StreamReader(wire, cross_machine=True)))}")
print("\n  A device event or a stream handle means nothing on the far side of a")
print("  network, so a descriptor carrying one across is wrong. Both ends assert")
print("  producer_synced rather than leaving a consumer to wait on an event that")
print("  does not exist. Everything this binding builds is producer_synced already,")
print("  so it costs a Python producer nothing — it states the assumption.")

# getvalue is repeatable, like io.BytesIO.getvalue().
assert writer.getvalue() == wire
