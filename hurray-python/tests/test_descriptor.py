"""The descriptor as a thing you can hold and send (#147, layers 2 and 4).

`hurray-python` could hold a descriptor — a `Tensor` is one, plus buffers — but could not
produce the artifact itself. For an interchange format that was the central gap: a Python
program could not put a Hurray tensor into a container of its own, nor read a descriptor
that arrived out of band.

`hurray.Descriptor` is that artifact. It is deliberately not constructible: a constructor
would duplicate `hurray.Tensor`'s whole parameter list to build the half of it that
carries no data.
"""

import numpy
import pytest

import hurray


def _tensor(n: int = 4) -> hurray.Tensor:
    return hurray.Tensor(bytes(4 * n), hurray.float32, [n])


# ── The class ─────────────────────────────────────────────────────────────────


def test_a_tensor_hands_back_its_descriptor():
    descriptor = _tensor().descriptor
    assert descriptor.dtype is hurray.float32
    assert descriptor.shape == (4,)
    assert descriptor.ndim == 1
    assert descriptor.size == 4
    assert descriptor.layout == hurray.RowMajorLayout()
    assert descriptor.buffer_count == 1
    assert descriptor.byte_offset == 0
    assert descriptor.version == (1, 0)


def test_it_agrees_with_the_tensor_it_came_from():
    """Two views of one thing; they must not be able to disagree."""
    tensor = hurray.Tensor(bytes(128), hurray.float16, [8, 8])
    descriptor = tensor.descriptor

    assert descriptor.dtype is tensor.dtype
    assert descriptor.shape == tensor.shape
    assert descriptor.ndim == tensor.ndim
    assert descriptor.size == tensor.size
    assert descriptor.layout == tensor.layout
    assert descriptor.buffer_count == tensor.buffer_count
    assert descriptor.buffer_handles == tensor.buffer_handles


def test_it_is_not_constructible():
    with pytest.raises(TypeError):
        hurray.Descriptor()


def test_descriptors_compare_by_value():
    assert _tensor().descriptor == _tensor().descriptor
    assert _tensor(4).descriptor != _tensor(8).descriptor
    assert _tensor().descriptor != "not a descriptor"


def test_repr_names_what_it_declares():
    text = repr(_tensor().descriptor)
    assert text.startswith("hurray.Descriptor(")
    assert "float32" in text and "row_major" in text


# ── Encoding ──────────────────────────────────────────────────────────────────


def test_the_spec_worked_example_is_61_bytes():
    """float32 [3, 4] row-major, one CPU buffer — the number `metadata.md` states."""
    descriptor = hurray.Tensor(bytes(192), hurray.float32, [3, 4]).descriptor
    assert len(descriptor.encode()) == 61
    assert descriptor.encoded_len == 61


def test_encode_decode_round_trips():
    original = _tensor().descriptor
    assert hurray.Descriptor.decode(original.encode()) == original


@pytest.mark.parametrize(
    "tensor_factory",
    [
        lambda: hurray.Tensor(bytes(16), hurray.float32, [4]),
        lambda: hurray.Tensor(b"", hurray.float32, [None, 512]),
        lambda: hurray.Tensor(bytes(16), hurray.float32, [4], shard=hurray.Shard([8], [0])),
        lambda: hurray.Tensor(
            bytes(16),
            hurray.float32,
            [4],
            statistics=hurray.Statistics(has_nan=False, has_inf=True),
        ),
        lambda: hurray.Tensor(
            bytes(16),
            hurray.dtype.int8,
            [16],
            quantization=hurray.PerTensorAffine(0.5, 0),
        ),
        lambda: hurray.Tensor(
            bytes(8),
            hurray.float32,
            [2, 2],
            aux_buffers=[bytes(16), bytes(24)],
            layout=hurray.CsrLayout(nnz=2),
        ),
    ],
)
def test_every_optional_section_survives(tensor_factory):
    original = tensor_factory().descriptor
    assert hurray.Descriptor.decode(original.encode()) == original


def test_trailing_bytes_are_ignored():
    """A descriptor is self-delimiting: what follows it in a stream is the buffers it
    describes, not part of it."""
    descriptor = _tensor().descriptor
    assert hurray.Descriptor.decode(descriptor.encode() + b"\xff" * 64) == descriptor


def test_truncated_bytes_are_refused():
    wire = _tensor().descriptor.encode()
    with pytest.raises(hurray.InvalidDescriptorError):
        hurray.Descriptor.decode(wire[:-8])


def test_nonsense_is_refused():
    with pytest.raises(hurray.InvalidDescriptorError):
        hurray.Descriptor.decode(b"not a descriptor at all")


def test_the_encoded_length_is_the_wires_own():
    """`encoded_len` is what the descriptor's length field carries, which is what makes
    it self-delimiting."""
    for tensor in (_tensor(), _tensor(1024), hurray.Tensor(b"", hurray.float32, [0])):
        descriptor = tensor.descriptor
        assert descriptor.encoded_len == len(descriptor.encode())


# ── A composite head: the descriptor that is only a descriptor ────────────────


def test_a_composite_head_has_no_buffers():
    composite = hurray.Composite(
        "group", shape=[4], dtype=hurray.float32, members=[_tensor()]
    )
    head = composite.descriptor

    assert head.buffer_count == 0
    assert head.buffer_handles == ()
    assert head.layout == hurray.CompositeLayout("group", 1)


def test_a_composite_head_round_trips():
    composite = hurray.Composite(
        "group", shape=[4], dtype=hurray.float32, members=[_tensor()]
    )
    head = composite.descriptor
    assert hurray.Descriptor.decode(head.encode()) == head


# ── The point of it all ───────────────────────────────────────────────────────


def test_a_descriptor_can_travel_in_a_container_of_your_own():
    """The capability this exists for: the descriptor goes wherever you have room, and
    the buffers travel beside it."""
    tensor = hurray.Tensor(bytes(48), hurray.float32, [3, 4])

    envelope = {
        "descriptor": tensor.descriptor.encode(),
        "buffers": [
            numpy.asarray(tensor.buffer(i)).tobytes() for i in range(tensor.buffer_count)
        ],
    }

    decoded = hurray.Descriptor.decode(envelope["descriptor"])
    assert decoded.dtype is hurray.float32
    assert decoded.shape == (3, 4)
    assert decoded.buffer_handles[0].byte_size == len(envelope["buffers"][0])


# ── Standalone quantization sections (layer 2) ────────────────────────────────


def _schemes():
    return [
        hurray.PerTensorAffine(0.015625, 128),
        hurray.PerChannelAffine.symmetric(axis=0, scale_buffer_index=1),
        hurray.PerChannelAffine.asymmetric(
            axis=1, scale_buffer_index=1, zero_point_buffer_index=2
        ),
        hurray.PerBlockAffine.symmetric(
            axis=0, block_size=64, scale_buffer_index=1, scale_type=hurray.float32
        ),
        hurray.NF4(axis=0, block_size=64, scale_buffer_index=1),
        hurray.MXFP(axis=0, block_size=32, scale_buffer_index=1),
    ]


@pytest.mark.parametrize("scheme", _schemes())
def test_a_scheme_round_trips_on_its_own(scheme):
    assert hurray.decode_quantization(scheme.encode()) == scheme


def test_the_scheme_tag_is_what_decode_dispatches_on():
    """A caller reads the bytes without knowing which scheme wrote them."""
    for scheme in _schemes():
        assert type(hurray.decode_quantization(scheme.encode())) is type(scheme)


def test_per_tensor_affine_is_16_bytes():
    assert len(hurray.PerTensorAffine(0.5, 0).encode()) == 16


def test_schemes_compare_by_value():
    """They were the only descriptor value objects in the binding that did not — so
    `tensor.quantization == hurray.PerTensorAffine(0.5, 0)` was False even when the two
    said exactly the same thing."""
    assert hurray.PerTensorAffine(0.5, 0) == hurray.PerTensorAffine(0.5, 0)
    assert hurray.PerTensorAffine(0.5, 0) != hurray.PerTensorAffine(0.25, 0)
    assert hurray.NF4(axis=0, block_size=64, scale_buffer_index=1) == hurray.NF4(
        axis=0, block_size=64, scale_buffer_index=1
    )
    assert hurray.PerTensorAffine(0.5, 0) != "not a scheme"


def test_a_tensors_scheme_equals_an_equal_one():
    tensor = hurray.Tensor(
        bytes(16), hurray.dtype.int8, [16], quantization=hurray.PerTensorAffine(0.5, 0)
    )
    assert tensor.quantization == hurray.PerTensorAffine(0.5, 0)
    assert tensor.descriptor.quantization == hurray.PerTensorAffine(0.5, 0)


def test_an_integer_scheme_is_hashable_and_a_float_one_is_not():
    """Per-tensor affine carries an f32 scale, and NaN would break the hash/equality
    contract — so it defines __eq__ without __hash__, which makes it unhashable. That is
    the right answer for a float-carrying value, not an oversight."""
    assert len({hurray.NF4(axis=0, block_size=64, scale_buffer_index=1)}) == 1
    with pytest.raises(TypeError):
        {hurray.PerTensorAffine(0.5, 0)}


def test_a_malformed_section_is_refused():
    with pytest.raises(hurray.InvalidDescriptorError):
        hurray.decode_quantization(b"\x00\x00\x00\x00")


# ── The pages ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "page_name, minimum",
    [
        ("layer-2-quantization-descriptors.md", 7),
        ("layer-4-tensor-descriptor-encoding.md", 3),
        ("hurray-inspect-cli.md", 1),
    ],
)
def test_every_python_block_on_the_page_runs(page_name, minimum):
    import pathlib
    import re

    page = pathlib.Path(__file__).parents[2] / "docs/cookbook" / page_name
    if not page.exists():
        pytest.skip("cookbook not present")

    blocks = re.findall(r"```python\n(.*?)```", page.read_text(), re.S)
    assert len(blocks) >= minimum, "the page lost its Python tabs"

    # One block writes a file; run from a scratch directory so it does not litter
    # the working tree.
    import os
    import tempfile

    previous = os.getcwd()
    os.chdir(tempfile.mkdtemp())
    try:
        for index, block in enumerate(blocks):
            exec(compile(block, f"{page_name}#python[{index}]", "exec"), {})
    finally:
        os.chdir(previous)
