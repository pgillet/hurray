"""
Layout descriptors: ``t.layout`` is an object, not a string (ADR-032).

The wire format models a layout as a tag plus that layout's parameters — ``nnz``,
``strides``, ``page_size``. A string carries the tag and throws the rest away, so
``t.layout`` returns a ``hurray.Layout``: an immutable value object with a class
per layout tag, which you can inspect, compare, hash, and hand straight back to
the ``hurray.Tensor`` constructor.

A layout object describes; it does not contain. It holds no reference to its
tensor and owns no buffers — reach those through the tensor, with the named
accessors or the generic ``t.buffer(index)``.

Run with:

    python hurray-python/examples/layouts.py
"""

import struct

import hurray

# ── Reading a layout ──────────────────────────────────────────────────────────

dense = hurray.Tensor(bytes(16), hurray.float32, [4])

print("=== A dense tensor's layout ===")
print(f"  repr:         {dense.layout!r}")
print(f"  name:         {dense.layout.name}")
print(f"  tag:          0x{dense.layout.tag:02X}")
print(f"  buffer_count: {dense.layout.buffer_count}")
print(f"  is_dense:     {dense.layout.is_dense}")
print(f"  isinstance:   {isinstance(dense.layout, hurray.RowMajorLayout)}")

# ── Authoring with layout= ────────────────────────────────────────────────────

# A 2x2 CSR matrix holding [[5.0, 0.0], [0.0, 7.0]]:
#   buffer 0 — values        [5.0, 7.0]
#   buffer 1 — col_indices   [0, 1]
#   buffer 2 — row_ptr       [0, 1, 2]
csr = hurray.Tensor(
    struct.pack("2f", 5.0, 7.0),
    hurray.float32,
    [2, 2],
    aux_buffers=[struct.pack("2Q", 0, 1), struct.pack("3Q", 0, 1, 2)],
    layout=hurray.CsrLayout(nnz=2),
)

print("\n=== Authoring a CSR tensor ===")
print(f"  layout:       {csr.layout!r}")
print(f"  nnz:          {csr.layout.nnz}")
print(f"  buffer_count: {csr.buffer_count}")
print(f"  row_ptr:      shape {csr.row_ptr.shape}")

# ── The layout is a declaration; the buffers are evidence ─────────────────────

print("\n=== The two must agree ===")
try:
    hurray.Tensor(
        struct.pack("2f", 5.0, 7.0),  # only two values...
        hurray.float32,
        [2, 2],
        aux_buffers=[struct.pack("8Q", *range(8))],
        layout=hurray.CooLayout(nnz=4),  # ...but the layout declares four
    )
except hurray.BufferError as exc:
    print(f"  rejected: {exc}")
    print("  (nnz is never inferred: the descriptor is not quietly rewritten to 2)")

try:
    hurray.Tensor(bytes(16), hurray.float32, [4], layout="csr")
except TypeError as exc:
    print(f"  rejected: {exc}")
    print("  (a string could not carry nnz, so it can only produce a wrong descriptor)")

# ── Value semantics ───────────────────────────────────────────────────────────

print("\n=== Layouts are values ===")
print(f"  equal by value:      {hurray.CsrLayout(nnz=2) == csr.layout}")
print(f"  differ by parameter: {hurray.CsrLayout(nnz=3) != csr.layout}")
print(f"  hashable:            {len({hurray.CooLayout(nnz=1), hurray.CooLayout(nnz=1)})}")
print(f"  fresh object:        t.layout is t.layout -> {csr.layout is csr.layout}")
print(f"  never a string:      t.layout == 'csr'    -> {csr.layout == 'csr'}")

# ── Every parameter is reachable ──────────────────────────────────────────────

print("\n=== Parameters a string could not carry ===")
strided = hurray.StridedLayout([4, 1])
print(f"  {strided!r}")
print("    strides are in logical *elements*, signed — not bytes as in NumPy")

tiled = hurray.TiledLayout([64, 64], inner_layout="tiled", inner_tiled=hurray.TiledLayout([8, 8]))
print(f"  {tiled!r}")
print(f"    nested tile shape: {tiled.inner_tiled.tile_shape}")

paged = hurray.BlockPagedLayout(
    page_size=16, num_pages=64, paged_axis=0, num_seqs=2, kv_role="key", layer_index=3
)
print(f"  {paged!r}")
print(f"    kv_role={paged.kv_role!r} layer_index={paged.layer_index}")

overlay = hurray.CompositeLayout("overlay", member_count=3, combine_op="add")
print(f"  {overlay!r}")
print(f"    rule and op stay separate: combine_op={overlay.combine_op!r}")
print(f"    for a partition it does not apply: {hurray.CompositeLayout('partition', 2).combine_op}")

# ── Buffers with no named accessor ────────────────────────────────────────────

# CSF stores a rank-3 tensor as 2*rank+1 buffers: values, then a (pos, crd) pair
# per level. They have no named accessors — t.buffer(index) reaches them, and the
# layout says what each index holds.
csf = hurray.Tensor(
    struct.pack("4f", 1.0, 2.0, 3.0, 4.0),
    hurray.float32,
    [2, 3, 4],
    aux_buffers=[
        struct.pack("2Q", 0, 2),  # pos_0
        struct.pack("2Q", 0, 1),  # crd_0
        struct.pack("3Q", 0, 2, 3),  # pos_1
        struct.pack("3Q", 0, 2, 1),  # crd_1
        struct.pack("4Q", 0, 1, 2, 4),  # pos_2
        struct.pack("4Q", 1, 3, 0, 2),  # crd_2
    ],
    layout=hurray.CsfLayout(nnz=4, mode_order=[0, 1, 2]),
)

print("\n=== The generic buffer accessor ===")
print(f"  layout:       {csf.layout!r}")
print(f"  buffer_count: {csf.buffer_count}  (2 * rank + 1)")
for index in range(csf.buffer_count):
    view = csf.buffer(index)
    print(f"    buffer({index}): {view.shape[0]:>2} bytes, dtype={view.dtype}")
print("  (uint8 is the only honest dtype: these buffers do not share one)")

# ── Unknown layouts survive a relay ───────────────────────────────────────────

print("\n=== A tag this build does not know ===")
unknown = hurray.UnknownLayout(0x0C, b"\x01\x02")
relayed = hurray.Tensor(bytes(16), hurray.float32, [4], layout=unknown)
print(f"  {relayed.layout!r}")
print(f"  raw_bytes preserved: {relayed.layout.raw_bytes!r}")
try:
    hurray.UnknownLayout(0x07)
except ValueError as exc:
    print(f"  rejected: {exc}")
    print("  (calling a known tag 'unknown' would skip every check CsrLayout applies)")
