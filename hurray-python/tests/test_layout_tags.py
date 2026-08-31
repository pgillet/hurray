"""Classifying a layout tag, and checking a layout against a shape (#147, layer 3).

ADR-032 gave Python a class per layout, so most of the layer-3 cookbook page translated
without new API. Two sections did not: a tag this build does not recognise could reach
Python — `Layout.tag` and `UnknownLayout` both exist — but nothing could say *what kind*
of unrecognised it was, and nothing could check a layout against a shape without building
a whole tensor around it.

The kind matters because the three unrecognised cases call for three different reactions:
relay it (reserved, the producer is probably newer), consult the out-of-band agreement
(private), or reject the input (invalid).
"""

import pytest

import hurray

# ── layout_tag_kind ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "tag, kind",
    [
        (0x01, "named"),  # row_major
        (0x02, "named"),  # col_major
        (0x07, "named"),  # csr
        (0x09, "named"),  # csf
        (0x0A, "named"),  # block_paged
        (0x0B, "named"),  # composite
        (0x40, "named"),  # hilbert
        (0x10, "reserved"),
        (0x7F, "reserved"),
        (0xF0, "private"),
        (0xF3, "private"),
        (0xFE, "private"),
        (0x00, "invalid"),
        (0xFF, "invalid"),
    ],
)
def test_a_tag_is_classified(tag, kind):
    assert hurray.layout_tag_kind(tag) == kind


def test_the_four_kinds_partition_the_whole_byte_space():
    """No tag is unclassified and none is two things at once — which is what lets one
    call replace the four predicates hurray-core exposes."""
    kinds = {hurray.layout_tag_kind(tag) for tag in range(256)}
    assert kinds == {"named", "reserved", "private", "invalid"}


def test_every_layout_class_reports_a_named_tag():
    layouts = [
        hurray.RowMajorLayout(),
        hurray.ColMajorLayout(),
        hurray.StridedLayout([1]),
        hurray.TiledLayout([2]),
        hurray.MortonLayout([2]),
        hurray.HilbertLayout(3, 3),
        hurray.CooLayout(1, False),
        hurray.CsrLayout(1),
        hurray.CscLayout(1),
        hurray.CsfLayout(1, [0, 1, 2]),
        hurray.CompositeLayout("group", 1),
    ]
    for layout in layouts:
        assert hurray.layout_tag_kind(layout.tag) == "named", layout.name


def test_a_private_layouts_tag_classifies_as_private():
    private = hurray.PrivateExtensionLayout(0xF0, 1, b"")
    assert hurray.layout_tag_kind(private.tag) == "private"


def test_an_unknown_layouts_tag_classifies_as_reserved():
    """The pairing that makes the function useful: UnknownLayout only accepts tags that
    are genuinely unassigned, so what it wraps is what `layout_tag_kind` calls reserved."""
    unknown = hurray.UnknownLayout(0x10, b"")
    assert hurray.layout_tag_kind(unknown.tag) == "reserved"
    assert unknown.buffer_count is None


def test_a_tag_outside_a_byte_is_refused():
    with pytest.raises(OverflowError):
        hurray.layout_tag_kind(256)


# ── validate_against_shape ────────────────────────────────────────────────────


def test_a_rank_2_layout_accepts_a_rank_2_shape():
    hurray.CsrLayout(nnz=5).validate_against_shape([4, 5])
    hurray.CscLayout(nnz=5).validate_against_shape([4, 5])


def test_a_rank_2_layout_refuses_anything_else():
    for shape in ([2, 3, 4], [4], []):
        with pytest.raises(hurray.InvalidDescriptorError) as exc:
            hurray.CsrLayout(nnz=5).validate_against_shape(shape)
        assert "csr" in str(exc.value)


def test_csf_is_the_rank_3_and_up_case():
    csf = hurray.CsfLayout(nnz=4, mode_order=[0, 1, 2])
    csf.validate_against_shape([2, 3, 4])
    with pytest.raises(hurray.InvalidDescriptorError):
        csf.validate_against_shape([3, 4])


def test_morton_bits_must_cover_the_extents():
    morton = hurray.MortonLayout([2, 2])
    morton.validate_against_shape([4, 4])  # 4 <= 2**2
    with pytest.raises(hurray.InvalidDescriptorError):
        morton.validate_against_shape([8, 4])  # 8 > 2**2


def test_hilbert_wants_power_of_two_extents():
    hilbert = hurray.HilbertLayout(hilbert_order=3, hilbert_rank=3)
    hilbert.validate_against_shape([8, 8, 8])
    with pytest.raises(hurray.InvalidDescriptorError):
        hilbert.validate_against_shape([8, 8, 7])


def test_a_dense_layout_takes_any_shape():
    for shape in ([], [4], [2, 3, 4, 5]):
        hurray.RowMajorLayout().validate_against_shape(shape)


def test_a_dynamic_dimension_is_nothing_to_check_yet():
    """An unresolved extent cannot violate an extent constraint."""
    hurray.CsrLayout(nnz=5).validate_against_shape([None, 5])
    hurray.MortonLayout([2, 2]).validate_against_shape([None, 4])


def test_the_check_matches_what_the_constructor_enforces():
    """Same rule, two entry points: if validate_against_shape passes, the constructor
    accepts the pair, and if it fails, the constructor refuses it."""
    layout = hurray.CsrLayout(nnz=2)
    buffers = [bytes(16), bytes(24)]

    layout.validate_against_shape([2, 2])
    hurray.Tensor(bytes(8), hurray.float32, [2, 2], aux_buffers=buffers, layout=layout)

    with pytest.raises(hurray.InvalidDescriptorError):
        layout.validate_against_shape([2, 2, 2])
    with pytest.raises(hurray.InvalidDescriptorError):
        hurray.Tensor(
            bytes(8), hurray.float32, [2, 2, 2], aux_buffers=buffers, layout=layout
        )

