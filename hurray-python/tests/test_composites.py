"""A composite is a container of tensors, not a tensor (ADR-036).

Composites were the last thing `hurray-core` and `hurray-io` could express that
`hurray-python` could not, and they were blocked in three places at once: authoring,
streaming, and files. These tests cover all three, plus the reason the class exists —
a composite head owns no buffers, so it is not a `hurray.Tensor` and does not pretend
to be one.
"""

import struct

import pytest

import hurray


def _tile(offset: int):
    """One half of an 8x8 partition: an 8x4 tile at column `offset`."""
    return hurray.Tensor(
        bytes(128), hurray.float32, [8, 4], shard=hurray.Shard([8, 8], [0, offset])
    )


def _partition():
    return hurray.Composite(
        "partition",
        shape=[8, 8],
        dtype=hurray.float32,
        members=[_tile(0), _tile(4)],
    )


def _group(count: int = 2):
    return hurray.Composite(
        "group",
        shape=[4],
        dtype=hurray.float32,
        members=[hurray.Tensor(bytes(16), hurray.float32, [4]) for _ in range(count)],
    )


# ── The class ─────────────────────────────────────────────────────────────────


def test_a_composite_reports_its_head_and_members():
    composite = _partition()
    assert composite.member_count == 2
    assert len(composite.members) == 2
    assert composite.shape == (8, 8)
    assert composite.ndim == 2
    assert composite.dtype == hurray.float32
    assert composite.layout == hurray.CompositeLayout("partition", 2)


def test_a_composite_is_not_a_tensor():
    """The distinction the class exists to draw."""
    composite = _group()
    assert not isinstance(composite, hurray.Tensor)
    for absent in ("values", "buffer", "buffer_count", "__dlpack__", "__array__"):
        assert not hasattr(composite, absent), absent


def test_a_composite_stays_out_of_the_native_protocol():
    """ADR-036 § 6: the capsule carries a buffer list and one descriptor, not a tree."""
    assert not hasattr(_group(), "__hurray__")


def test_members_are_the_objects_that_were_passed():
    tiles = [_tile(0), _tile(4)]
    composite = hurray.Composite(
        "partition", shape=[8, 8], dtype=hurray.float32, members=tiles
    )
    assert composite.members[0] is tiles[0]
    assert composite.members[1] is tiles[1]


def test_composites_compare_by_value():
    assert _partition() == _partition()
    assert _group(2) != _group(3)
    assert _group() != "not a composite"


def test_repr_names_the_rule_and_shape():
    assert repr(_partition()).startswith("hurray.Composite(rule='partition', shape=(8, 8)")


# ── Validation is core's, not the binding's ───────────────────────────────────


def test_a_partition_that_does_not_cover_is_refused():
    """One 8x4 tile cannot cover an 8x8 head."""
    with pytest.raises(hurray.InvalidDescriptorError) as exc:
        hurray.Composite("partition", shape=[8, 8], dtype=hurray.float32, members=[_tile(0)])
    assert "cover" in str(exc.value)


def test_overlapping_partition_members_are_refused():
    with pytest.raises(hurray.InvalidDescriptorError):
        hurray.Composite(
            "partition",
            shape=[8, 8],
            dtype=hurray.float32,
            members=[_tile(0), _tile(0)],
        )


def test_an_overlay_needs_its_combine_op():
    with pytest.raises(ValueError):
        hurray.Composite("overlay", shape=[8, 8], dtype=hurray.float32, members=[_tile(0)])


def test_a_combine_op_on_a_partition_is_refused():
    with pytest.raises(ValueError):
        hurray.Composite(
            "partition",
            shape=[8, 8],
            dtype=hurray.float32,
            members=[_tile(0), _tile(4)],
            combine_op="add",
        )


def test_a_member_that_is_neither_tensor_nor_composite_is_refused():
    with pytest.raises(TypeError) as exc:
        hurray.Composite("group", shape=[4], dtype=hurray.float32, members=[object()])
    assert "members[0]" in str(exc.value)


def test_the_head_is_stated_not_derived():
    """A wrong head shape must be rejected, not silently corrected to fit."""
    with pytest.raises(hurray.InvalidDescriptorError):
        hurray.Composite(
            "partition",
            shape=[8, 16],  # the tiles cover 8x8, not 8x16
            dtype=hurray.float32,
            members=[_tile(0), _tile(4)],
        )


# ── Nesting ───────────────────────────────────────────────────────────────────


def test_a_composite_can_be_a_member():
    inner = _group()
    outer = hurray.Composite("group", shape=[4], dtype=hurray.float32, members=[inner])
    assert outer.member_count == 1
    assert outer.members[0] == inner


def test_a_nested_repr_shows_the_tree():
    inner = _group()
    outer = hurray.Composite("group", shape=[4], dtype=hurray.float32, members=[inner])
    # The inner composite appears inside the outer one's members, not as "Tensor".
    assert repr(outer).count("hurray.Composite") == 2
    assert "Tensor, Tensor" in repr(outer)


# ── hurray.Tensor still refuses a composite layout ────────────────────────────


def test_a_tensor_with_a_composite_layout_points_at_the_right_class():
    with pytest.raises(hurray.UnsupportedError) as exc:
        hurray.Tensor(
            bytes(16), hurray.float32, [4], layout=hurray.CompositeLayout("group", 1)
        )
    assert "hurray.Composite" in str(exc.value)


# ── Streaming ─────────────────────────────────────────────────────────────────


def test_a_composite_round_trips_through_a_stream():
    original = _partition()
    with hurray.StreamWriter() as writer:
        writer.write(original)

    (back,) = list(hurray.StreamReader(writer.getvalue()))
    assert isinstance(back, hurray.Composite)
    assert back == original


def test_a_composite_is_one_item_not_three():
    """The bug the reader's use of next_item avoids: a head plus two members would
    otherwise arrive as three separate tensors, with the composition lost and no error."""
    with hurray.StreamWriter() as writer:
        writer.write(_partition())

    items = list(hurray.StreamReader(writer.getvalue()))
    assert len(items) == 1
    assert items[0].member_count == 2


def test_a_stream_can_mix_tensors_and_composites():
    plain = hurray.Tensor(struct.pack("4f", 1, 2, 3, 4), hurray.float32, [4])
    with hurray.StreamWriter() as writer:
        writer.write(plain)
        writer.write(_partition())
        writer.write(plain)

    items = list(hurray.StreamReader(writer.getvalue()))
    assert [type(i).__name__ for i in items] == ["Tensor", "Composite", "Tensor"]


def test_a_nested_composite_round_trips_through_a_stream():
    inner = _group()
    outer = hurray.Composite("group", shape=[4], dtype=hurray.float32, members=[inner])
    with hurray.StreamWriter() as writer:
        writer.write(outer)

    (back,) = list(hurray.StreamReader(writer.getvalue()))
    assert back == outer
    assert back.members[0].member_count == 2


def test_writing_something_that_is_neither_is_refused():
    with hurray.StreamWriter() as writer:
        with pytest.raises(TypeError):
            writer.write("not a tensor")


# ── Files ─────────────────────────────────────────────────────────────────────


def test_a_composite_round_trips_through_a_file(tmp_path):
    path = str(tmp_path / "m.hrry")
    original = _partition()
    hurray.save(path, {"weight": original})

    loaded = hurray.load(path)
    assert list(loaded) == ["weight"]
    assert loaded["weight"] == original


def test_a_composites_members_do_not_also_appear_at_the_top_level(tmp_path):
    """Every tensor gets an index entry, head and member alike — so without the
    membership check a member would come back twice."""
    path = str(tmp_path / "m.hrry")
    hurray.save(path, {"weight": _partition()})

    loaded = hurray.load(path)
    assert len(loaded) == 1, f"members leaked into the top level: {sorted(loaded)}"


def test_a_file_can_mix_tensors_and_composites(tmp_path):
    path = str(tmp_path / "m.hrry")
    plain = hurray.Tensor(bytes(16), hurray.float32, [4])
    hurray.save(path, {"weight": _partition(), "bias": plain})

    loaded = hurray.load(path)
    assert sorted(loaded) == ["bias", "weight"]
    assert isinstance(loaded["weight"], hurray.Composite)
    assert isinstance(loaded["bias"], hurray.Tensor)


def test_a_nested_composite_round_trips_through_a_file(tmp_path):
    path = str(tmp_path / "m.hrry")
    inner = _group()
    outer = hurray.Composite("group", shape=[4], dtype=hurray.float32, members=[inner])
    hurray.save(path, {"tree": outer})

    loaded = hurray.load(path)
    assert loaded["tree"] == outer


def test_a_member_can_still_be_loaded_by_name(tmp_path):
    """Member names are derived as '{head}.{index}' for the file index."""
    path = str(tmp_path / "m.hrry")
    hurray.save(path, {"weight": _partition()})

    member = hurray.load(path, names=["weight.0"])["weight.0"]
    assert isinstance(member, hurray.Tensor)
    assert member.shape == (8, 4)


def test_save_still_refuses_anything_else(tmp_path):
    path = str(tmp_path / "m.hrry")
    with pytest.raises(hurray.UnsupportedError) as exc:
        hurray.save(path, {"x": "not a tensor"})
    assert "hurray.Composite" in str(exc.value)


# ── Overlays (the SpQR case) ──────────────────────────────────────────────────


def _base(size: int = 32):
    return hurray.Tensor(
        bytes(4 * size * size),
        hurray.float32,
        [size, size],
        shard=hurray.Shard([size, size], [0, 0]),
    )


def _correction(nnz: int = 16, size: int = 32):
    return hurray.Tensor(
        bytes(4 * nnz),
        hurray.float32,
        [size, size],
        aux_buffers=[bytes(nnz * 2 * 8)],  # packed [nnz, rank] uint64 indices
        layout=hurray.CooLayout(nnz=nnz, is_sorted=True),
        shard=hurray.Shard([size, size], [0, 0]),
    )


def _overlay(combine_op: str = "replace"):
    return hurray.Composite(
        "overlay",
        shape=[32, 32],
        dtype=hurray.float32,
        members=[_base(), _correction()],
        combine_op=combine_op,
    )


def test_an_overlay_can_be_authored():
    """It could not be, before: an overlay's members carry a role, and Python had no
    way to state one, so core's validator refused the rule outright."""
    overlay = _overlay()
    assert overlay.member_count == 2
    assert overlay.layout.composition_rule == "overlay"
    assert overlay.layout.combine_op == "replace"


def test_the_roles_are_positional():
    """The format fixes them — member 0 is the base and spans the index space, the rest
    are corrections — so the binding attaches them rather than asking."""
    assert _overlay().member_roles == ("base", "correction")


def test_a_rule_without_roles_reports_none():
    assert _group().member_roles == (None, None)
    assert _partition().member_roles == (None, None)


def test_the_roles_belong_to_the_composite_not_the_tensors():
    """A tensor has no role; a tensor inside an overlay does. The caller's own objects
    are handed back unchanged, which is why the role is read from the composite."""
    base = _base()
    overlay = hurray.Composite(
        "overlay",
        shape=[32, 32],
        dtype=hurray.float32,
        members=[base, _correction()],
        combine_op="replace",
    )
    assert overlay.members[0] is base
    assert not hasattr(base, "member_role")


def test_a_base_that_does_not_span_is_refused():
    """The first member must cover the whole index space; core checks it."""
    with pytest.raises(hurray.InvalidDescriptorError):
        hurray.Composite(
            "overlay",
            shape=[32, 32],
            dtype=hurray.float32,
            members=[
                hurray.Tensor(
                    bytes(4 * 16 * 32),
                    hurray.float32,
                    [16, 32],
                    shard=hurray.Shard([32, 32], [0, 0]),
                ),
                _correction(),
            ],
            combine_op="replace",
        )


@pytest.mark.parametrize("combine_op", ["replace", "add"])
def test_an_overlay_round_trips_through_a_stream(combine_op):
    original = _overlay(combine_op)
    with hurray.StreamWriter() as writer:
        writer.write(original)

    (back,) = list(hurray.StreamReader(writer.getvalue()))
    assert back == original, "an overlay must equal its own round trip"
    assert back.member_roles == ("base", "correction")
    assert back.layout.combine_op == combine_op


def test_an_overlay_round_trips_through_a_file(tmp_path):
    path = str(tmp_path / "m.hrry")
    original = _overlay()
    hurray.save(path, {"weight": original})

    loaded = hurray.load(path)["weight"]
    assert loaded == original
    assert loaded.member_roles == ("base", "correction")


def test_the_written_bytes_carry_the_roles():
    """The write path must use the composite's wire-ready descriptors, not the members'
    own: emitting a member without its role produces a stream a conformant reader
    rejects, and the only place that shows up is on the far side."""
    with hurray.StreamWriter() as writer:
        writer.write(_overlay())

    # cross_machine is irrelevant here; a strict reader is simply another decoder.
    (back,) = list(hurray.StreamReader(writer.getvalue(), cross_machine=True))
    assert back.member_roles == ("base", "correction")


def test_a_nested_overlay_keeps_its_roles():
    inner = _overlay()
    outer = hurray.Composite(
        "group", shape=[32, 32], dtype=hurray.float32, members=[inner]
    )
    with hurray.StreamWriter() as writer:
        writer.write(outer)

    (back,) = list(hurray.StreamReader(writer.getvalue()))
    assert back == outer
    assert back.members[0].member_roles == ("base", "correction")

