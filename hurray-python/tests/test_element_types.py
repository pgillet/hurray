"""The element type system and the shape model, from Python (#147, layer 0).

Four things the binding could not say before, each of which the layer-0 cookbook page is
largely about: a type's wire tag, its element alignment, how many bytes a packed buffer
needs, and a dimension whose extent is not known yet.

The packing rules are the reason `buffer_size_bytes` exists rather than being left to the
caller: `count * bit_width // 8` is wrong for every sub-byte type, and wrong in a way that
produces a buffer one byte short of the last element.
"""

import pytest

import hurray

# ── Wire tags ─────────────────────────────────────────────────────────────────


def test_a_dtype_reports_its_wire_tag():
    assert hurray.float32.tag == 0x03
    assert hurray.dtype.int4.tag == 0x48
    assert hurray.bool.tag == 0x20


def test_a_tag_round_trips():
    for dtype in (
        hurray.float32,
        hurray.float64,
        hurray.int8,
        hurray.uint64,
        hurray.bool,
        hurray.dtype.int4,
        hurray.dtype.float8_e4m3,
        hurray.dtype.float6_e2m3,
        hurray.dtype.complex128,
    ):
        assert hurray.Dtype.from_tag(dtype.tag) is dtype, dtype.name


def test_tags_are_unique_across_every_type():
    """They index the wire format; a collision would be a decoder that guesses."""
    names = [n for n in dir(hurray.dtype) if not n.startswith("_")]
    tags = [getattr(hurray.dtype, n).tag for n in names]
    assert len(set(tags)) == len(tags)


def test_the_permanently_invalid_sentinels_are_refused():
    for tag in (0x00, 0xFF):
        with pytest.raises(hurray.InvalidDescriptorError):
            hurray.Dtype.from_tag(tag)


def test_a_reserved_tag_is_refused():
    """A tag this version assigns to nothing — possibly a newer producer, possibly
    corruption, but not something to guess at."""
    with pytest.raises(hurray.InvalidDescriptorError):
        hurray.Dtype.from_tag(0x7E)


def test_the_tag_agrees_with_hash_and_identity():
    assert hurray.Dtype.from_tag(hurray.float32.tag) is hurray.float32


# ── Element alignment ─────────────────────────────────────────────────────────


def test_element_alignment_is_the_elements_own_not_the_buffers():
    assert hurray.float32.element_alignment == 4
    assert hurray.float64.element_alignment == 8
    assert hurray.int16.element_alignment == 2
    assert hurray.dtype.complex128.element_alignment == 8  # its component's, not its own


def test_a_packed_element_has_no_alignment_of_its_own():
    for name in ("int4", "uint4", "int2", "uint2", "float4_e2m1"):
        assert getattr(hurray.dtype, name).element_alignment == 1, name
    assert hurray.bool.element_alignment == 1


# ── Buffer sizing ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "dtype_name, count, expected",
    [
        ("float32", 100, 400),  # whole-byte: count × width
        ("float64", 3, 24),
        ("int4", 7, 4),  # ceil(7 / 2)
        ("int4", 8, 4),
        ("uint4", 1, 1),
        ("bool", 9, 2),  # ceil(9 / 8)
        ("bool", 8, 1),
        ("float6_e2m3", 100, 75),  # ceil(100 / 4) × 3
        ("float6_e3m2", 4, 3),
        ("int2", 5, 2),  # ceil(5 / 4)
    ],
)
def test_buffer_size_bytes_follows_the_packing_rules(dtype_name, count, expected):
    dtype = getattr(hurray.dtype, dtype_name)
    assert hurray.buffer_size_bytes(dtype, count) == expected


def test_no_elements_needs_no_bytes():
    for name in ("float64", "int4", "bool"):
        assert hurray.buffer_size_bytes(getattr(hurray.dtype, name), 0) == 0


def test_the_size_is_what_the_constructor_demands():
    """The reason to expose this at all: it answers 'how big a buffer do I pass?'."""
    count = 7
    needed = hurray.buffer_size_bytes(hurray.dtype.int4, count)

    hurray.Tensor(bytes(needed), hurray.dtype.int4, [count])
    with pytest.raises(hurray.BufferError):
        hurray.Tensor(bytes(needed - 1), hurray.dtype.int4, [count])


# ── Dynamic dimensions ────────────────────────────────────────────────────────


def test_a_dimension_can_be_unknown():
    t = hurray.Tensor(b"", hurray.float32, [None, 512, 768])
    assert t.shape == (None, 512, 768)
    assert t.ndim == 3
    assert t.size is None, "an unknown extent means an unknown element count"


def test_the_shape_round_trips_through_python():
    """ADR-032 § 4: what you read back has to rebuild the same descriptor. Before this,
    `shape` returned None for a dynamic dimension and nothing accepted None back."""
    original = hurray.Tensor(b"", hurray.float32, [None, 512])
    rebuilt = hurray.Tensor(b"", original.dtype, list(original.shape))
    assert rebuilt.shape == original.shape


def test_a_resolved_shape_has_a_size_again():
    assert hurray.Tensor(bytes(4 * 8 * 512), hurray.float32, [8, 512]).size == 4096


def test_every_dimension_may_be_dynamic():
    assert hurray.Tensor(b"", hurray.float32, [None, None]).shape == (None, None)


def test_a_dynamic_tensor_does_not_print_as_an_empty_one():
    """NumPy reads the DYNAMIC extent as -1, i.e. 'infer', so a naive render would show
    an unresolved batch dimension as []."""
    text = repr(hurray.Tensor(b"", hurray.float32, [None, 512]))
    assert "None" in text and "512" in text
    assert not text.endswith("([], dtype=float32)")


def test_a_negative_dimension_still_points_at_None():
    with pytest.raises(hurray.InvalidDescriptorError) as exc:
        hurray.Tensor(b"", hurray.float32, [-1, 512])
    assert "None" in str(exc.value)


# ── What cannot take one ──────────────────────────────────────────────────────


@pytest.mark.parametrize("fn", ["zeros", "ones", "empty"])
def test_allocating_functions_refuse_a_dynamic_dimension(fn):
    """You cannot allocate an unknown number of bytes. The refusal names the call, so
    the caller is not left with a bare TypeError from argument conversion."""
    with pytest.raises(hurray.InvalidDescriptorError) as exc:
        getattr(hurray, fn)([None, 4])

    message = str(exc.value)
    assert f"hurray.{fn}()" in message
    assert "index 0" in message


def test_full_refuses_one_too():
    with pytest.raises(hurray.InvalidDescriptorError):
        hurray.full([2, None], 1.0)


def test_a_composite_head_cannot_be_dynamic():
    """A head is checked against what its members cover; nothing covers an unknown
    extent."""
    member = hurray.Tensor(bytes(16), hurray.float32, [4])
    with pytest.raises(hurray.InvalidDescriptorError) as exc:
        hurray.Composite("group", shape=[None], dtype=hurray.float32, members=[member])
    assert "hurray.Composite()" in str(exc.value)


def test_sparse_coo_cannot_be_dynamic():
    numpy = pytest.importorskip("numpy")
    values = numpy.array([5.0], dtype=numpy.float32)
    indices = numpy.array([[0, 0]], dtype=numpy.uint64)

    with pytest.raises(hurray.InvalidDescriptorError) as exc:
        hurray.sparse_coo(values, indices, [None, 2])
    assert "hurray.sparse_coo()" in str(exc.value)


# ── The page itself ───────────────────────────────────────────────────────────


def test_every_python_block_in_the_layer_0_cookbook_runs():
    """The tabs are documentation people copy, and nothing else executes them. Running
    them here caught a Rust block on the same page asserting that tag 0xF0 is an error,
    when the private-extension range has been valid all along."""
    import pathlib
    import re

    page = pathlib.Path(__file__).parents[2] / "docs/cookbook/layer-0-element-types-and-shape.md"
    if not page.exists():  # installed without the repo alongside
        pytest.skip("cookbook not present")

    blocks = re.findall(r"```python\n(.*?)```", page.read_text(), re.S)
    assert len(blocks) >= 7, "the page lost its Python tabs"
    for index, block in enumerate(blocks):
        exec(compile(block, f"{page.name}#python[{index}]", "exec"), {})
