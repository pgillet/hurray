"""Checking a paged KV cache before shipping it (#147, block-paged).

A block-paged descriptor describes its `block_table` and `seq_ptr`; it does not contain
them. Nothing about their contents is checked when the descriptor is built, so they are
the only thing standing between a consumer and an out-of-bounds read — and until now a
Python producer had no way to check them.

The other half is quantization: a paged cache is usually fp8 or int8, and per-block-affine
only works on one when its blocks line up with the pages.
"""

import struct

import pytest

import hurray

PAGE_SIZE = 4
NUM_PAGES = 5
NUM_SEQS = 2


def _layout(**overrides) -> hurray.BlockPagedLayout:
    kwargs = {
        "page_size": PAGE_SIZE,
        "num_pages": NUM_PAGES,
        "paged_axis": 0,
        "num_seqs": NUM_SEQS,
    }
    kwargs.update(overrides)
    return hurray.BlockPagedLayout(**kwargs)


# ── The four storage invariants ───────────────────────────────────────────────


def test_a_sound_pair_of_index_buffers_passes():
    _layout().validate_index_buffers(seq_ptr=[0, 2, 3], block_table=[0, 1, 0])


def test_a_shared_prefix_is_not_an_error():
    """Two sequences naming the same page is how prefix sharing is represented. It is
    the point of the layout, not a mistake to catch."""
    _layout().validate_index_buffers(seq_ptr=[0, 2, 3], block_table=[0, 1, 0])
    _layout().validate_index_buffers(seq_ptr=[0, 1, 2], block_table=[3, 3])


@pytest.mark.parametrize(
    "seq_ptr, block_table, why",
    [
        ([1, 2, 3], [0, 1, 0], "seq_ptr[0] must be 0"),
        ([0, 3, 2], [0, 1, 0], "seq_ptr must be non-decreasing"),
        ([0, 2, 9], [0, 1, 0], "seq_ptr[num_seqs] must equal len(block_table)"),
        ([0, 2, 3], [0, 5, 0], "a page id must be < num_pages"),
        ([0, 2, 3], [0, 99, 0], "a page id far out of range"),
    ],
)
def test_a_broken_invariant_is_refused(seq_ptr, block_table, why):
    with pytest.raises(hurray.InvalidDescriptorError):
        _layout().validate_index_buffers(seq_ptr, block_table)


def test_the_empty_batch_is_valid():
    _layout(num_seqs=0).validate_index_buffers(seq_ptr=[0], block_table=[])


def test_the_page_id_bound_comes_from_the_layout():
    """num_pages and num_seqs are the layout's, so only the buffers are passed — the
    caller cannot accidentally check against a different pool than the descriptor
    declares."""
    _layout(num_pages=100).validate_index_buffers([0, 2, 3], [0, 50, 99])
    with pytest.raises(hurray.InvalidDescriptorError):
        _layout(num_pages=100).validate_index_buffers([0, 2, 3], [0, 50, 100])


def test_a_wide_index_type_is_checked_the_same_way():
    """The declared index type governs how the buffer is encoded, not whether the
    invariants hold — those are about values."""
    wide = _layout(block_table_index_type="uint64")
    wide.validate_index_buffers([0, 2, 3], [0, 1, 0])
    with pytest.raises(hurray.InvalidDescriptorError):
        wide.validate_index_buffers([0, 2, 3], [0, 5, 0])


# ── Quantization compatibility ────────────────────────────────────────────────


def _per_block(block_size: int, axis: int = 0):
    return hurray.PerBlockAffine.symmetric(
        axis=axis,
        block_size=block_size,
        scale_buffer_index=3,
        scale_type=hurray.float32,
    )


def test_per_block_affine_fits_when_its_blocks_are_the_pages():
    _layout().validate_quantization_compatibility(_per_block(PAGE_SIZE))


def test_per_block_affine_with_a_different_block_size_is_refused():
    """Scales must stay per-page-slot so a shared page carries its own; that is only
    true when block_size == page_size."""
    with pytest.raises(hurray.InvalidDescriptorError):
        _layout().validate_quantization_compatibility(_per_block(PAGE_SIZE * 2))


def test_per_channel_on_the_paged_axis_is_refused():
    with pytest.raises(hurray.InvalidDescriptorError):
        _layout().validate_quantization_compatibility(
            hurray.PerChannelAffine.symmetric(axis=0, scale_buffer_index=3)
        )


def test_per_channel_off_the_paged_axis_is_fine():
    for axis in (1, 2):
        _layout().validate_quantization_compatibility(
            hurray.PerChannelAffine.symmetric(axis=axis, scale_buffer_index=3)
        )


def test_per_tensor_affine_is_unconstrained():
    """It has no axis and no block size, so no paging rule bears on it."""
    _layout().validate_quantization_compatibility(hurray.PerTensorAffine(0.5, 0))


def test_the_check_takes_the_object_not_a_tag():
    """A caller already has the quantization descriptor; 0x03 would be a byte to look
    up rather than a thing to pass."""
    with pytest.raises(hurray.InvalidDescriptorError) as exc:
        _layout().validate_quantization_compatibility(0x03)
    assert "PerBlockAffine" in str(exc.value)


# ── The descriptor these checks are for ───────────────────────────────────────


def test_the_layout_builds_a_real_tensor():
    layout = _layout()
    seq_ptr, block_table = [0, 2, 3], [0, 1, 0]
    layout.validate_index_buffers(seq_ptr, block_table)

    cache = hurray.Tensor(
        bytes(NUM_PAGES * PAGE_SIZE * 2 * 8 * 2),  # pool of float16
        hurray.float16,
        [9, 2, 8],
        aux_buffers=[
            struct.pack(f"{len(block_table)}I", *block_table),
            struct.pack(f"{len(seq_ptr)}I", *seq_ptr),
        ],
        layout=layout,
    )

    assert cache.buffer_count == 3
    assert cache.layout == layout
    assert cache.layout.kv_role == "key"


def test_block_paged_is_rank_3_only():
    layout = _layout()
    layout.validate_against_shape([9, 2, 8])
    with pytest.raises(hurray.InvalidDescriptorError):
        layout.validate_against_shape([9, 2])

