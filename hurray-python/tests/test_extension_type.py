"""Private extension element types from Python (#199).

The format reserves tags `0xF0`–`0xFE` for types it does not standardize, and requires a
descriptor using one to carry an extension type section describing its width. Python could
*name* such a type — `Dtype.from_tag(0xF2)` has always worked — but could neither read the
section off a descriptor nor author one, which left `hurray-python` unable to express
something `hurray-core` round-trips.

The section is also what sizes the buffer: an extension `Dtype` reports a `bit_width` of 0,
because the real width lives here.
"""

import pytest

import hurray

PRIVATE_TAG = 0xF2


def _int24() -> hurray.ExtensionType:
    return hurray.ExtensionType(bit_width=24, is_signed=True)


def _tensor(ext: hurray.ExtensionType | None = None, n: int = 4) -> hurray.Tensor:
    ext = ext or _int24()
    return hurray.Tensor(
        bytes(ext.buffer_size_bytes(n)),
        hurray.Dtype.from_tag(PRIVATE_TAG),
        [n],
        extension_type=ext,
    )


# ── The class ─────────────────────────────────────────────────────────────────


def test_fields_round_trip_through_the_getters():
    ext = hurray.ExtensionType(
        bit_width=16,
        is_float=True,
        sign_bits=1,
        exponent_bits=5,
        mantissa_bits=10,
        exponent_bias=15,
        has_nan=True,
        has_inf=True,
    )
    assert ext.bit_width == 16
    assert ext.packing_factor == 1
    assert ext.is_float is True
    assert ext.is_signed is False
    assert ext.sign_bits == 1
    assert ext.exponent_bits == 5
    assert ext.mantissa_bits == 10
    assert ext.exponent_bias == 15
    assert ext.has_nan is True
    assert ext.has_inf is True


def test_defaults_describe_an_unsigned_integer():
    ext = hurray.ExtensionType(bit_width=8)
    assert (ext.is_float, ext.is_signed, ext.sign_bits) == (False, False, 0)
    assert (ext.exponent_bits, ext.mantissa_bits, ext.exponent_bias) == (0, 0, 0)


def test_is_frozen():
    with pytest.raises(AttributeError):
        _int24().bit_width = 8


# ── packing_factor is derived, not asked for ──────────────────────────────────


@pytest.mark.parametrize(
    ("bit_width", "packing_factor"),
    [(1, 8), (2, 4), (4, 2), (8, 1), (16, 1), (24, 1), (64, 1)],
)
def test_packing_factor_is_derived_from_bit_width(bit_width, packing_factor):
    assert hurray.ExtensionType(bit_width=bit_width).packing_factor == packing_factor


def test_packing_factor_is_not_an_argument():
    # The spec leaves exactly one legal value per bit_width, so restating it could only
    # ever produce an error.
    with pytest.raises(TypeError):
        hurray.ExtensionType(bit_width=8, packing_factor=1)


@pytest.mark.parametrize("bit_width", [3, 5, 6, 7])
def test_non_power_of_two_sub_byte_widths_are_rejected(bit_width):
    # Reserved to the built-in type space: 6-bit floats pack 4-per-3-bytes, which the
    # generic "elements per byte" model cannot express.
    with pytest.raises(hurray.InvalidDescriptorError, match="1, 2 or 4"):
        hurray.ExtensionType(bit_width=bit_width)


def test_zero_bit_width_is_rejected():
    with pytest.raises(hurray.InvalidDescriptorError, match="greater than 0"):
        hurray.ExtensionType(bit_width=0)


# ── Sign fields ───────────────────────────────────────────────────────────────


def test_float_may_not_set_is_signed():
    # A float's sign is sign_bits; is_signed describes integers. Setting both would give
    # a reader two answers to one question.
    with pytest.raises(hurray.InvalidDescriptorError, match="is_signed"):
        hurray.ExtensionType(bit_width=16, is_float=True, is_signed=True, sign_bits=1)


def test_unsigned_float_is_expressible():
    # The float8_e8m0 shape: 8 exponent bits, no sign, no mantissa. If floats were
    # assumed signed, this type could not be described.
    scale = hurray.ExtensionType(
        bit_width=8, is_float=True, exponent_bits=8, exponent_bias=127
    )
    assert scale.is_float is True
    assert scale.sign_bits == 0


def test_signed_float_is_expressible():
    half = hurray.ExtensionType(
        bit_width=16, is_float=True, sign_bits=1, exponent_bits=5, mantissa_bits=10
    )
    assert half.sign_bits == 1
    assert half.is_signed is False


def test_sign_bits_above_one_is_rejected():
    with pytest.raises(hurray.InvalidDescriptorError, match="sign_bits"):
        hurray.ExtensionType(bit_width=16, is_float=True, sign_bits=2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sign_bits": 1},
        {"exponent_bits": 5},
        {"mantissa_bits": 10},
        {"exponent_bias": 15},
    ],
)
def test_integer_may_not_set_float_fields(kwargs):
    with pytest.raises(hurray.InvalidDescriptorError):
        hurray.ExtensionType(bit_width=16, **kwargs)


# ── Buffer sizing ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("bit_width", "count", "expected"),
    [
        (24, 10, 30),
        (8, 37, 37),
        (4, 7, 4),  # ceil(7 / 2)
        (2, 5, 2),  # ceil(5 / 4)
        (1, 9, 2),  # ceil(9 / 8)
        (16, 0, 0),
    ],
)
def test_buffer_size_bytes(bit_width, count, expected):
    assert hurray.ExtensionType(bit_width=bit_width).buffer_size_bytes(count) == expected


def test_generic_buffer_size_bytes_refuses_an_extension_dtype():
    # Core answers 0 for an extension type, which as a byte count is wrong rather than
    # unknown — and would silently size a buffer to nothing.
    with pytest.raises(hurray.InvalidDescriptorError, match="ExtensionType"):
        hurray.buffer_size_bytes(hurray.Dtype.from_tag(PRIVATE_TAG), 10)


# ── Authoring a tensor ────────────────────────────────────────────────────────


def test_tensor_carries_the_section():
    tensor = _tensor()
    assert tensor.extension_type.bit_width == 24
    assert tensor.descriptor.extension_type.bit_width == 24


def test_a_standard_dtype_has_no_section():
    tensor = hurray.Tensor(bytes(16), hurray.float32, [4])
    assert tensor.extension_type is None
    assert tensor.descriptor.extension_type is None


def test_extension_dtype_without_a_section_is_rejected():
    with pytest.raises(hurray.InvalidDescriptorError, match="must describe itself"):
        hurray.Tensor(bytes(12), hurray.Dtype.from_tag(PRIVATE_TAG), [4])


def test_section_without_an_extension_dtype_is_rejected():
    with pytest.raises(hurray.InvalidDescriptorError, match="not a private extension"):
        hurray.Tensor(bytes(16), hurray.float32, [4], extension_type=_int24())


def test_buffer_is_sized_from_the_section():
    # Without the section this check computes 0 and waves an empty buffer through.
    with pytest.raises(hurray.BufferError, match="need at least 12 bytes"):
        hurray.Tensor(
            bytes(4), hurray.Dtype.from_tag(PRIVATE_TAG), [4], extension_type=_int24()
        )


def test_sub_byte_extension_tensor_packs():
    ext = hurray.ExtensionType(bit_width=4)
    tensor = hurray.Tensor(
        bytes(4), hurray.Dtype.from_tag(PRIVATE_TAG), [7], extension_type=ext
    )
    assert tensor.extension_type.packing_factor == 2


# ── The wire ──────────────────────────────────────────────────────────────────


def test_descriptor_round_trips():
    original = _tensor().descriptor
    restored = hurray.Descriptor.decode(original.encode())
    assert restored == original
    assert restored.extension_type.bit_width == 24
    assert restored.extension_type.is_signed is True


def test_section_adds_twenty_bytes_to_the_descriptor():
    plain = hurray.Tensor(bytes(16), hurray.float32, [4]).descriptor
    assert _tensor().descriptor.encoded_len == plain.encoded_len + 20


@pytest.mark.parametrize("tag", [0xF0, 0xF7, 0xFE])
def test_every_tag_in_the_private_range_works(tag):
    tensor = hurray.Tensor(
        bytes(12), hurray.Dtype.from_tag(tag), [4], extension_type=_int24()
    )
    assert tensor.descriptor.dtype.tag == tag


def test_survives_a_stream():
    with hurray.StreamWriter() as writer:
        writer.write(_tensor())
    (restored,) = list(hurray.StreamReader(writer.getvalue()))
    assert restored.extension_type.bit_width == 24


def test_survives_a_file(tmp_path):
    path = tmp_path / "private.hrry"
    hurray.save(str(path), {"w": _tensor()})
    restored = hurray.load(str(path))["w"]
    assert restored.dtype.tag == PRIVATE_TAG
    assert restored.extension_type.bit_width == 24


def test_extension_dtype_reprs_as_the_call_that_builds_it():
    assert repr(hurray.Dtype.from_tag(PRIVATE_TAG)) == "hurray.Dtype.from_tag(0xF2)"
