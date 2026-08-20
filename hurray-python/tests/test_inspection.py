"""Reading descriptor sections back from Python (ADR-030 Pass 3).

Authoring landed in Pass 2; this covers the other direction — a consumer that
receives a tensor and wants to know how it is quantized, what statistics travelled
with it, and whether it is a shard of something larger.
"""

import struct

import hurray


def _scales(n=2):
    return struct.pack(f"{n}f", *([0.02] * n))


def _per_channel_tensor():
    return hurray.Tensor(
        bytes(8),
        hurray.int8,
        [2, 4],
        aux_buffers=[_scales()],
        quantization=hurray.PerChannelAffine.symmetric(0, 1),
    )


# ── Quantization ──────────────────────────────────────────────────────────────


def test_quantization_getter_returns_the_scheme():
    q = _per_channel_tensor().quantization
    assert isinstance(q, hurray.PerChannelAffine)
    assert q.axis == 0
    assert q.scale_buffer_index == 1
    assert q.zero_point_buffer_index is None


def test_unquantized_tensor_reports_none():
    t = hurray.Tensor(bytes(16), hurray.float32, [4])
    assert t.quantization is None
    assert t.statistics is None
    assert t.shard is None


def test_per_tensor_affine_reports_inline_parameters():
    t = hurray.Tensor(
        bytes(8), hurray.int8, [2, 4], quantization=hurray.PerTensorAffine(0.02, 128)
    )
    q = t.quantization
    assert isinstance(q, hurray.PerTensorAffine)
    assert q.zero_point == 128


def test_every_scheme_survives_the_round_trip():
    cases = [
        (hurray.PerChannelAffine.symmetric(0, 1), hurray.PerChannelAffine),
        (hurray.PerBlockAffine.symmetric(0, 32, 1, hurray.float32), hurray.PerBlockAffine),
        (hurray.NF4(0, 64, 1), hurray.NF4),
        (hurray.MXFP(0, 32, 1), hurray.MXFP),
    ]
    for scheme, cls in cases:
        t = hurray.Tensor(
            bytes(64), hurray.int8, [2, 32], aux_buffers=[_scales()], quantization=scheme
        )
        assert isinstance(t.quantization, cls)


def test_an_inspected_scheme_can_be_reused():
    """The getter returns what the constructor accepts, so it composes."""
    scheme = _per_channel_tensor().quantization
    again = hurray.Tensor(
        bytes(8), hurray.int8, [2, 4], aux_buffers=[_scales()], quantization=scheme
    )
    assert again.quantization.scale_buffer_index == 1


# ── The consumer path: read it back off disk ──────────────────────────────────


def test_quantization_survives_a_file_round_trip(tmp_path):
    path = tmp_path / "weights.hrry"
    hurray.save(str(path), {"w": _per_channel_tensor()})

    loaded = hurray.load(str(path))["w"]
    assert loaded.buffer_count == 2

    q = loaded.quantization
    assert isinstance(q, hurray.PerChannelAffine)
    assert q.axis == 0
    assert q.scale_buffer_index == 1


def test_quantization_survives_the_native_protocol():
    received = hurray.from_hurray(_per_channel_tensor())
    assert received.buffer_count == 2
    assert received.quantization.scale_buffer_index == 1


# ── Statistics and shard ──────────────────────────────────────────────────────


def test_statistics_getter_reports_only_what_was_claimed():
    t = hurray.Tensor(
        bytes(24),
        hurray.float32,
        [2, 3],
        statistics=hurray.Statistics(nnz=6, value_min=-1.0, value_max=1.0, value_abs_max=1.0),
    )
    s = t.statistics
    assert s.nnz == 6
    assert s.value_abs_max == 1.0
    assert s.value_mean is None  # never supplied, so never claimed


def test_shard_getter_reports_position():
    t = hurray.Tensor(
        bytes(24), hurray.float32, [2, 3], shard=hurray.Shard([4, 3], [2, 0])
    )
    assert t.shard.parent_shape == (4, 3)
    assert t.shard.shard_offset == (2, 0)


def test_statistics_and_shard_survive_a_file_round_trip(tmp_path):
    path = tmp_path / "shard.hrry"
    t = hurray.Tensor(
        bytes(24),
        hurray.float32,
        [2, 3],
        statistics=hurray.Statistics(nnz=6),
        shard=hurray.Shard([4, 3], [2, 0]),
    )
    hurray.save(str(path), {"piece": t})

    back = hurray.load(str(path))["piece"]
    assert back.statistics.nnz == 6
    assert back.shard.shard_offset == (2, 0)


# ── Buffer count ──────────────────────────────────────────────────────────────


def test_buffer_count_distinguishes_quantized_tensors():
    assert hurray.Tensor(bytes(16), hurray.float32, [4]).buffer_count == 1
    assert _per_channel_tensor().buffer_count == 2
