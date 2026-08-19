"""A layout is an object, not a string (ADR-032).

`t.layout` returns a `hurray.Layout` carrying that layout's parameters — `nnz`,
`strides`, `page_size` — none of which a string could hold. These tests pin the
hierarchy, the value semantics, the three validation tiers that keep a layout
declaration and its buffers in agreement, and the round-trip obligation.
"""

import struct

import numpy as np
import pytest

import hurray


def _dense():
    return hurray.Tensor(bytes(16), hurray.float32, [4])


def _csr_tensor():
    """A 2x2 CSR tensor with two stored values."""
    return hurray.Tensor(
        struct.pack("2f", 5.0, 7.0),
        hurray.float32,
        [2, 2],
        aux_buffers=[struct.pack("2Q", 0, 1), struct.pack("3Q", 0, 1, 2)],
        layout=hurray.CsrLayout(nnz=2),
    )


def _coo_tensor():
    values = np.array([5.0, 7.0], dtype=np.float32)
    indices = np.array([[0, 0], [1, 1]], dtype=np.uint64)
    return hurray.sparse_coo(values, indices, [2, 2])


# ── The hierarchy ─────────────────────────────────────────────────────────────


def test_layout_is_an_object_with_a_name():
    layout = _dense().layout
    assert isinstance(layout, hurray.Layout)
    assert isinstance(layout, hurray.RowMajorLayout)
    assert layout.name == "row_major"
    assert layout.tag == 0x01


def test_every_layout_class_shares_the_base():
    for layout in (
        hurray.RowMajorLayout(),
        hurray.ColMajorLayout(),
        hurray.StridedLayout([4, 1]),
        hurray.TiledLayout([4, 4]),
        hurray.MortonLayout([3, 3]),
        hurray.HilbertLayout(3, 2),
        hurray.CooLayout(nnz=2),
        hurray.CsrLayout(nnz=2),
        hurray.CscLayout(nnz=2),
        hurray.CsfLayout(nnz=2, mode_order=[0, 1, 2]),
        hurray.BlockPagedLayout(16, 64, 0, 2),
        hurray.CompositeLayout("group", 2),
        hurray.PrivateExtensionLayout(0xF0, 7, b""),
        hurray.UnknownLayout(0x0C),
    ):
        assert isinstance(layout, hurray.Layout), layout


def test_the_base_is_not_constructible():
    """There is no layout that is only 'a layout'."""
    with pytest.raises(TypeError):
        hurray.Layout()


def test_buffer_count_separates_virtual_from_unknown():
    assert hurray.RowMajorLayout().buffer_count == 1
    assert hurray.CooLayout(nnz=2).buffer_count == 2
    assert hurray.CsrLayout(nnz=2).buffer_count == 3
    # CSF is rank-dependent: 2 * 3 + 1.
    assert hurray.CsfLayout(nnz=2, mode_order=[0, 1, 2]).buffer_count == 7
    # A composite head owns a *known* zero buffers...
    assert hurray.CompositeLayout("group", 2).buffer_count == 0
    assert hurray.CompositeLayout("group", 2).is_virtual
    # ...where an extension's count is genuinely unknown.
    assert hurray.UnknownLayout(0x0C).buffer_count is None
    assert hurray.PrivateExtensionLayout(0xF0, 7, b"").buffer_count is None


def test_is_dense_tracks_the_addressing_category():
    assert hurray.RowMajorLayout().is_dense
    assert hurray.StridedLayout([4, 1]).is_dense
    assert not hurray.CsrLayout(nnz=2).is_dense
    assert not hurray.BlockPagedLayout(16, 64, 0, 2).is_dense


# ── Value semantics ───────────────────────────────────────────────────────────


def test_layouts_compare_by_value():
    assert hurray.CsrLayout(nnz=4) == hurray.CsrLayout(nnz=4)
    assert hurray.CsrLayout(nnz=4) != hurray.CsrLayout(nnz=5)
    assert hurray.CsrLayout(nnz=4) != hurray.CscLayout(nnz=4)


def test_layouts_hash_consistently_with_equality():
    assert len({hurray.CsrLayout(nnz=4), hurray.CsrLayout(nnz=4)}) == 1
    assert len({hurray.CsrLayout(nnz=4), hurray.CsrLayout(nnz=5)}) == 2
    assert {hurray.CooLayout(nnz=1): "a"}[hurray.CooLayout(nnz=1)] == "a"


def test_a_layout_never_equals_a_string():
    """The lossy comparison this hierarchy replaces stays dead."""
    layout = _dense().layout
    assert layout != "row_major"
    assert "row_major" != layout
    assert layout.name == "row_major"


def test_layout_is_a_fresh_object_but_an_equal_one():
    t = _dense()
    assert t.layout is not t.layout
    assert t.layout == t.layout


def test_layout_is_read_only():
    """Assigning a layout would silently reinterpret the existing buffers."""
    t = _dense()
    with pytest.raises(AttributeError):
        t.layout = hurray.ColMajorLayout()


# ── Every parameter is reachable ──────────────────────────────────────────────


def test_sparse_parameters():
    assert hurray.CooLayout(nnz=7, is_sorted=True).nnz == 7
    assert hurray.CooLayout(nnz=7, is_sorted=True).is_sorted
    assert hurray.CooLayout(nnz=7).is_sorted is False
    assert hurray.CsrLayout(nnz=4).nnz == 4
    assert hurray.CscLayout(nnz=4).nnz == 4

    csf = hurray.CsfLayout(nnz=5, mode_order=[2, 0, 1])
    assert csf.nnz == 5
    assert csf.mode_order == (2, 0, 1)


def test_dense_parameters():
    assert hurray.StridedLayout([4, 1]).strides == (4, 1)
    # Strides are in logical elements and may be negative or zero.
    assert hurray.StridedLayout([0, -1]).strides == (0, -1)
    assert hurray.MortonLayout([3, 3]).morton_bits == (3, 3)
    assert hurray.HilbertLayout(3, 2).hilbert_order == 3
    assert hurray.HilbertLayout(3, 2).hilbert_rank == 2


def test_tiled_parameters_including_nesting():
    flat = hurray.TiledLayout([4, 4], inner_layout="col_major")
    assert flat.tile_shape == (4, 4)
    assert flat.outer_layout == "row_major"
    assert flat.inner_layout == "col_major"
    assert flat.inner_tiled is None
    assert flat.outer_strides is None

    nested = hurray.TiledLayout(
        [64, 64], inner_layout="tiled", inner_tiled=hurray.TiledLayout([8, 8])
    )
    assert nested.inner_tiled.tile_shape == (8, 8)
    assert "inner_tiled=TiledLayout(tile_shape=(8, 8)" in repr(nested)


def test_block_paged_parameters():
    layout = hurray.BlockPagedLayout(
        page_size=16,
        num_pages=64,
        paged_axis=0,
        num_seqs=2,
        kv_role="fused",
        layer_index=3,
        block_table_index_type="uint64",
    )
    assert layout.page_size == 16
    assert layout.num_pages == 64
    assert layout.paged_axis == 0
    assert layout.num_seqs == 2
    assert layout.kv_role == "fused"
    assert layout.layer_index == 3
    assert layout.block_table_index_type == "uint64"


def test_composite_keeps_rule_and_combine_separate():
    overlay = hurray.CompositeLayout("overlay", member_count=3, combine_op="add")
    assert overlay.composition_rule == "overlay"
    assert overlay.combine_op == "add"
    assert overlay.member_count == 3

    # For a non-overlay rule the operation does not apply — it is not merely unset.
    assert hurray.CompositeLayout("partition", 2).combine_op is None
    assert hurray.CompositeLayout("group", 2).combine_op is None


def test_composite_rejects_a_combine_that_cannot_apply():
    with pytest.raises(ValueError):
        hurray.CompositeLayout("group", 2, combine_op="add")
    with pytest.raises(ValueError):
        hurray.CompositeLayout("overlay", 2)


def test_private_and_unknown_are_different_facts():
    private = hurray.PrivateExtensionLayout(0xF0, extension_layout_id=7, extension_data=b"\x01")
    unknown = hurray.UnknownLayout(0x0C, b"\x00\x01")

    assert isinstance(private, hurray.PrivateExtensionLayout)
    assert not isinstance(private, hurray.UnknownLayout)
    assert private.extension_layout_id == 7
    assert private.extension_data == b"\x01"

    assert unknown.tag == 0x0C
    assert unknown.raw_bytes == b"\x00\x01"
    # Both report the same name; isinstance is what tells them apart.
    assert private.name == unknown.name == "extension"


def test_unknown_rejects_a_tag_that_has_a_named_class():
    """Calling a known tag 'unknown' would smuggle a descriptor past every check."""
    with pytest.raises(ValueError) as exc:
        hurray.UnknownLayout(0x07)
    assert "CsrLayout" in str(exc.value)
    with pytest.raises(ValueError):
        hurray.UnknownLayout(0xF0)


# ── Repr ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("layout", "expected"),
    [
        (hurray.RowMajorLayout(), "RowMajorLayout()"),
        (hurray.CsrLayout(nnz=4), "CsrLayout(nnz=4)"),
        (hurray.CooLayout(nnz=2), "CooLayout(nnz=2, is_sorted=False)"),
        (hurray.StridedLayout([4, 1]), "StridedLayout(strides=(4, 1))"),
        (hurray.MortonLayout([3, 3]), "MortonLayout(morton_bits=(3, 3))"),
        (hurray.HilbertLayout(3, 2), "HilbertLayout(hilbert_order=3, hilbert_rank=2)"),
        (
            hurray.CsfLayout(nnz=5, mode_order=[0, 1, 2]),
            "CsfLayout(nnz=5, mode_order=(0, 1, 2))",
        ),
        (
            hurray.CompositeLayout("group", 2),
            "CompositeLayout(composition_rule='group', member_count=2)",
        ),
    ],
)
def test_repr_names_the_class_and_its_fields(layout, expected):
    assert repr(layout) == expected


# ── Authoring ─────────────────────────────────────────────────────────────────


def test_layout_keyword_builds_the_declared_layout():
    csr = _csr_tensor()
    assert csr.layout == hurray.CsrLayout(nnz=2)
    assert csr.nnz == 2
    assert csr.buffer_count == 3


def test_a_layout_string_is_not_accepted():
    """A string cannot carry nnz, so layout='csr' is a request that cannot be honoured."""
    with pytest.raises(TypeError) as exc:
        hurray.Tensor(bytes(16), hurray.float32, [4], layout="csr")
    assert "hurray.Layout" in str(exc.value)


def test_a_non_layout_object_is_rejected():
    with pytest.raises(TypeError):
        hurray.Tensor(bytes(16), hurray.float32, [4], layout=object())


def test_omitting_layout_means_row_major():
    assert _dense().layout == hurray.RowMajorLayout()


# ── Tier 1: shape ─────────────────────────────────────────────────────────────


def test_csr_rejects_a_rank_that_is_not_two():
    with pytest.raises(hurray.InvalidDescriptorError):
        hurray.Tensor(
            bytes(16),
            hurray.float32,
            [2, 2, 2],
            aux_buffers=[bytes(16), bytes(24)],
            layout=hurray.CsrLayout(nnz=2),
        )


def test_strides_must_match_the_rank():
    with pytest.raises(hurray.InvalidDescriptorError):
        hurray.Tensor(bytes(16), hurray.float32, [2, 2], layout=hurray.StridedLayout([1]))


# ── Tier 2: buffer count ──────────────────────────────────────────────────────


def test_too_few_buffers_for_the_layout():
    with pytest.raises(hurray.InvalidDescriptorError) as exc:
        hurray.Tensor(
            struct.pack("2f", 5.0, 7.0),
            hurray.float32,
            [2, 2],
            aux_buffers=[struct.pack("2Q", 0, 1)],  # missing row_ptr
            layout=hurray.CsrLayout(nnz=2),
        )
    assert "3 buffer" in str(exc.value)


def test_quantization_may_not_claim_a_layout_buffer():
    """A scale index of 2 on CSR would designate the row-pointer buffer as scales."""
    with pytest.raises(hurray.InvalidDescriptorError) as exc:
        hurray.Tensor(
            struct.pack("2b", 1, 2),
            hurray.int8,
            [2, 2],
            aux_buffers=[struct.pack("2Q", 0, 1), struct.pack("3Q", 0, 1, 2)],
            layout=hurray.CsrLayout(nnz=2),
            quantization=hurray.PerChannelAffine.symmetric(axis=0, scale_buffer_index=2),
        )
    assert "quantization buffer index 2" in str(exc.value)


# ── Tier 3: buffer size ───────────────────────────────────────────────────────


def test_nnz_is_never_inferred_from_a_short_buffer():
    """CooLayout(nnz=4) with two values raises; it does not quietly become nnz=2."""
    with pytest.raises(hurray.BufferError) as exc:
        hurray.Tensor(
            struct.pack("2f", 5.0, 7.0),
            hurray.float32,
            [2, 2],
            aux_buffers=[struct.pack("8Q", *range(8))],
            layout=hurray.CooLayout(nnz=4),
        )
    assert "values" in str(exc.value)


def test_an_index_buffer_that_is_too_small_is_rejected():
    with pytest.raises(hurray.BufferError) as exc:
        hurray.Tensor(
            struct.pack("2f", 5.0, 7.0),
            hurray.float32,
            [2, 2],
            aux_buffers=[struct.pack("2Q", 0, 1), struct.pack("2Q", 0, 1)],  # row_ptr short
            layout=hurray.CsrLayout(nnz=2),
        )
    assert "row_ptr" in str(exc.value)


def test_over_sized_buffers_are_allowed():
    """Alignment and padding slack are legitimate; only under-sized is a defect."""
    t = hurray.Tensor(
        struct.pack("8f", *range(8)),
        hurray.float32,
        [2, 2],
        aux_buffers=[struct.pack("8Q", *range(8)), struct.pack("8Q", *range(8))],
        layout=hurray.CsrLayout(nnz=2),
    )
    assert t.nnz == 2


def test_a_dense_tensor_still_checks_its_element_count():
    with pytest.raises(hurray.BufferError):
        hurray.Tensor(bytes(4), hurray.float32, [4])


# ── Composite authoring ───────────────────────────────────────────────────────


def test_a_composite_layout_cannot_be_given_to_tensor():
    with pytest.raises(hurray.UnsupportedError) as exc:
        hurray.Tensor(bytes(16), hurray.float32, [4], layout=hurray.CompositeLayout("group", 2))
    assert "owns no buffers" in str(exc.value)


# ── The generic buffer accessor ───────────────────────────────────────────────


def test_buffer_returns_a_uint8_view_of_the_declared_size():
    csr = _csr_tensor()
    assert csr.buffer(0).shape == (8,)  # 2 float32 values
    assert csr.buffer(1).shape == (16,)  # 2 uint64 column indices
    assert csr.buffer(2).shape == (24,)  # 3 uint64 row pointers
    assert csr.buffer(0).dtype == hurray.uint8


def test_buffer_reaches_a_csf_tensors_levels():
    """CSF has 2*rank+1 buffers and no named accessors for them."""
    nnz = 4
    values = struct.pack("4f", 1.0, 2.0, 3.0, 4.0)
    pos_0 = struct.pack("2Q", 0, 2)
    crd_0 = struct.pack("2Q", 0, 1)
    pos_1 = struct.pack("3Q", 0, 2, 3)
    crd_1 = struct.pack("3Q", 0, 2, 1)
    pos_2 = struct.pack("4Q", 0, 1, 2, 4)
    crd_2 = struct.pack("4Q", 1, 3, 0, 2)

    t = hurray.Tensor(
        values,
        hurray.float32,
        [2, 3, 4],
        aux_buffers=[pos_0, crd_0, pos_1, crd_1, pos_2, crd_2],
        layout=hurray.CsfLayout(nnz=nnz, mode_order=[0, 1, 2]),
    )
    assert t.layout.mode_order == (0, 1, 2)
    assert t.buffer_count == 7
    assert t.buffer(6).shape == (32,)  # leaf crd: nnz uint64 entries


def test_buffer_rejects_an_index_that_does_not_exist():
    with pytest.raises(IndexError):
        _dense().buffer(1)


# ── Round-trip obligation ─────────────────────────────────────────────────────


def test_a_tensor_can_be_rebuilt_from_its_own_layout():
    """Handing a tensor's own layout back to the constructor reproduces it."""
    values = struct.pack("2f", 5.0, 7.0)
    col_indices = struct.pack("2Q", 0, 1)
    row_ptr = struct.pack("3Q", 0, 1, 2)
    original = _csr_tensor()

    rebuilt = hurray.Tensor(
        values,
        original.dtype,
        list(original.shape),
        aux_buffers=[col_indices, row_ptr],
        layout=original.layout,
        quantization=original.quantization,
        statistics=original.statistics,
        shard=original.shard,
    )
    assert rebuilt.layout == original.layout
    assert rebuilt.nnz == original.nnz
    assert rebuilt.buffer_count == original.buffer_count


def test_a_coo_tensor_round_trips_through_its_layout():
    original = _coo_tensor()
    assert original.layout == hurray.CooLayout(nnz=2, is_sorted=original.layout.is_sorted)

    rebuilt = hurray.Tensor(
        struct.pack("2f", 5.0, 7.0),
        original.dtype,
        list(original.shape),
        aux_buffers=[struct.pack("4Q", 0, 0, 1, 1)],
        layout=original.layout,
    )
    assert rebuilt.layout == original.layout


def test_an_unknown_layout_survives_a_rebuild():
    """A permissive relay must be able to write back what it decoded."""
    layout = hurray.UnknownLayout(0x0C, b"\x01\x02")
    t = hurray.Tensor(bytes(16), hurray.float32, [4], layout=layout)
    assert t.layout == layout
    assert t.layout.raw_bytes == b"\x01\x02"
