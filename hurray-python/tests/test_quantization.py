"""Authoring quantization, statistics, and shard descriptors from Python.

Covers the surface added in ADR-030 Pass 2: the five quantization schemes, the
statistics section with its derived validity mask, and shard descriptors — plus
the rule that a buffer index which resolves to nothing is rejected at construction
rather than handed to a consumer as a dangling reference.
"""

import struct

import pytest

import hurray


def _scales(n=2):
    """A float32 scale buffer with `n` entries."""
    return struct.pack(f"{n}f", *([0.02] * n))


# ── Scheme construction ───────────────────────────────────────────────────────


def test_per_tensor_affine_is_inline():
    q = hurray.PerTensorAffine(0.02, 128)
    assert q.scale == pytest.approx(0.02)
    assert q.zero_point == 128


def test_per_tensor_affine_rejects_a_zero_scale():
    with pytest.raises(hurray.InvalidDescriptorError):
        hurray.PerTensorAffine(0.0, 0)


def test_per_channel_symmetric_has_no_zero_point():
    q = hurray.PerChannelAffine.symmetric(axis=0, scale_buffer_index=1)
    assert q.axis == 0
    assert q.scale_buffer_index == 1
    assert q.zero_point_buffer_index is None


def test_per_channel_asymmetric_carries_both_indices():
    q = hurray.PerChannelAffine.asymmetric(
        axis=0, scale_buffer_index=1, zero_point_buffer_index=2
    )
    assert q.scale_buffer_index == 1
    assert q.zero_point_buffer_index == 2


def test_per_block_affine_carries_block_size_and_scale_type():
    q = hurray.PerBlockAffine.symmetric(1, 32, 1, hurray.float32)
    assert q.block_size == 32
    assert q.scale_type == hurray.float32


def test_nf4_and_mxfp():
    assert hurray.NF4(1, 64, 1).block_size == 64
    assert hurray.MXFP(1, 32, 1).scale_buffer_index == 1


# ── Attaching a scheme to a tensor ────────────────────────────────────────────


def test_per_channel_tensor_needs_its_scale_buffer():
    t = hurray.Tensor(
        bytes(8),
        hurray.int8,
        [2, 4],
        aux_buffers=[_scales()],
        quantization=hurray.PerChannelAffine.symmetric(0, 1),
    )
    assert t.shape == (2, 4)
    assert t.dtype == hurray.int8


def test_dangling_scale_buffer_index_is_rejected():
    """The failure the multi-buffer transport exists to prevent must be unauthorable."""
    with pytest.raises(hurray.InvalidDescriptorError):
        hurray.Tensor(
            bytes(8),
            hurray.int8,
            [2, 4],
            quantization=hurray.PerChannelAffine.symmetric(0, 1),  # no aux buffer
        )


def test_per_tensor_affine_stays_single_buffer():
    t = hurray.Tensor(
        bytes(8), hurray.int8, [2, 4], quantization=hurray.PerTensorAffine(0.02, 0)
    )
    assert t.shape == (2, 4)


def test_a_non_scheme_object_is_rejected():
    with pytest.raises(hurray.InvalidDescriptorError):
        hurray.Tensor(bytes(8), hurray.int8, [2, 4], quantization="per-channel")


def test_quantized_tensor_round_trips_through_a_file(tmp_path):
    path = tmp_path / "weights.hrry"
    t = hurray.Tensor(
        bytes(8),
        hurray.int8,
        [2, 4],
        aux_buffers=[_scales()],
        quantization=hurray.PerChannelAffine.symmetric(0, 1),
    )
    hurray.save(str(path), {"w": t})

    back = hurray.load(str(path))["w"]
    assert back.shape == (2, 4)
    assert back.dtype == hurray.int8


# ── Statistics ────────────────────────────────────────────────────────────────


def test_statistics_mask_is_derived_from_what_you_pass():
    s = hurray.Statistics(nnz=1024)
    assert s.nnz == 1024
    assert s.computed_mask == 1  # NNZ_VALID only
    assert s.sparsity_ratio is None
    assert s.value_min is None


def test_statistics_empty_claims_nothing():
    s = hurray.Statistics()
    assert s.computed_mask == 0
    assert s.nnz is None


def test_statistics_value_range_is_all_or_nothing():
    ok = hurray.Statistics(value_min=-1.0, value_max=1.0, value_abs_max=1.0)
    assert ok.value_abs_max == 1.0
    with pytest.raises(hurray.InvalidDescriptorError):
        hurray.Statistics(value_min=-1.0)  # partial group


def test_statistics_paired_fields_are_all_or_nothing():
    assert hurray.Statistics(value_mean=0.0, value_stddev=1.0).value_mean == 0.0
    with pytest.raises(hurray.InvalidDescriptorError):
        hurray.Statistics(value_mean=0.0)
    with pytest.raises(hurray.InvalidDescriptorError):
        hurray.Statistics(nm_n=2)
    with pytest.raises(hurray.InvalidDescriptorError):
        hurray.Statistics(has_nan=True)


def test_statistics_attach_to_a_tensor():
    t = hurray.Tensor(
        bytes(24),
        hurray.float32,
        [2, 3],
        statistics=hurray.Statistics(nnz=6, sparsity_ratio=0.0),
    )
    assert t.size == 6


# ── Shard ─────────────────────────────────────────────────────────────────────


def test_shard_records_position_in_the_parent():
    s = hurray.Shard(parent_shape=[1024, 512], shard_offset=[512, 0])
    assert s.parent_shape == (1024, 512)
    assert s.shard_offset == (512, 0)


def test_shard_rejects_mismatched_lengths():
    with pytest.raises(hurray.InvalidDescriptorError):
        hurray.Shard(parent_shape=[8, 8], shard_offset=[0])


def test_shard_attaches_to_a_tensor():
    t = hurray.Tensor(
        bytes(24),
        hurray.float32,
        [2, 3],
        shard=hurray.Shard([4, 3], [2, 0]),
    )
    assert t.shape == (2, 3)
