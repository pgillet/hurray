"""Every repr this package prints is Python, and says what the object holds.

Two things go wrong with a PyO3 ``__repr__`` and neither breaks a build, because
nothing reads a repr except a person:

- **It leaks Rust.** ``PerBlockAffine`` printed ``zero_point_buffer_index=Some(2)`` for
  as long as it had a repr — ``{:?}`` on an ``Option``. Rust's ``{}`` on a ``bool`` is
  the same trap: it writes ``true``, which Python cannot read.
- **It prints an internal instead of the value.** ``Statistics`` printed
  ``computed_mask=0x4`` — the wire bitmask, and none of the numbers that bit gates.

So every repr is classified here. An ``EXPRESSION`` repr must evaluate back to an equal
object; a ``SUMMARY`` repr is prose for a human and only has to stay out of Rust. The
last test asserts the two tables between them name every class in ``hurray`` that
defines ``__repr__``, so a new one cannot arrive without that decision being made.
"""

import inspect
import re

import pytest

import hurray


def _tensor(shape=(4,), nbytes=16):
    return hurray.Tensor(bytes(nbytes), hurray.float32, list(shape))


def _tile(offset: int) -> hurray.Tensor:
    """One half of an 8x8 matrix — a partition member needs its shard section."""
    return hurray.Tensor(
        bytes(128), hurray.float32, [8, 4], shard=hurray.Shard([8, 8], [0, offset])
    )


# ── The two tables ────────────────────────────────────────────────────────────

# repr(x) is a Python expression that rebuilds an equal x.
EXPRESSION = {
    "PerTensorAffine": hurray.PerTensorAffine(0.5, 3),
    "PerChannelAffine.symmetric": hurray.PerChannelAffine.symmetric(0, 1),
    "PerChannelAffine.asymmetric": hurray.PerChannelAffine.asymmetric(0, 1, 2),
    "PerBlockAffine.symmetric": hurray.PerBlockAffine.symmetric(1, 64, 1, hurray.float32),
    "PerBlockAffine.asymmetric": hurray.PerBlockAffine.asymmetric(
        1, 64, 1, 2, hurray.bfloat16
    ),
    "NF4": hurray.NF4(axis=0, block_size=64, scale_buffer_index=1),
    "MXFP": hurray.MXFP(axis=0, block_size=32, scale_buffer_index=1),
    "Statistics.empty": hurray.Statistics(),
    "Statistics.range": hurray.Statistics(
        nnz=1024, value_min=-1.0, value_max=1.0, value_abs_max=1.0
    ),
    "Statistics.flags": hurray.Statistics(
        value_mean=0.5, value_stddev=0.25, has_nan=False, has_inf=True
    ),
    "Shard": hurray.Shard([1024, 512], [512, 0]),
    "ExtensionType.integer": hurray.ExtensionType(bit_width=24, is_signed=True),
    "ExtensionType.float": hurray.ExtensionType(
        bit_width=16,
        is_float=True,
        sign_bits=1,
        exponent_bits=5,
        mantissa_bits=10,
        exponent_bias=15,
        has_nan=True,
        has_inf=True,
    ),
    # Sub-byte: packing_factor is derived, so the repr must not print it.
    "ExtensionType.sub_byte": hurray.ExtensionType(bit_width=4),
    "Device": hurray.Device(kind="cpu"),
    "Dtype.tier1": hurray.float32,
    "Dtype.tier2": hurray.dtype.int4,
    "Dtype.extension": hurray.Dtype.from_tag(0xF2),
    "RowMajorLayout": hurray.RowMajorLayout(),
    "ColMajorLayout": hurray.ColMajorLayout(),
    "StridedLayout": hurray.StridedLayout([4, 1]),
    "TiledLayout": hurray.TiledLayout([2, 2]),
    "MortonLayout": hurray.MortonLayout([4, 4]),
    "HilbertLayout": hurray.HilbertLayout(4, 2),
    "CooLayout": hurray.CooLayout(nnz=4),
    "CsrLayout": hurray.CsrLayout(nnz=4),
    "CscLayout": hurray.CscLayout(nnz=4),
    "CsfLayout": hurray.CsfLayout(nnz=4, mode_order=[0, 1]),
    "CompositeLayout": hurray.CompositeLayout("partition", 2),
}

# repr(x) describes x for a human and is not claimed to be an expression.
SUMMARY = {
    # Prints the buffer contents, the way numpy's does.
    "Tensor": _tensor(),
    # A view over the wire artifact; the constructor takes bytes, not these fields.
    "Descriptor": _tensor().descriptor,
    # Describes a buffer the caller does not own and cannot re-create by value.
    "BufferHandle": _tensor().buffer_handles[0],
    # Owns members; printing them all is what __repr__ must not do.
    "Composite": hurray.Composite(
        "partition", shape=[8, 8], dtype=hurray.float32, members=[_tile(0), _tile(4)]
    ),
    # Carries a raw byte count, deliberately: the tag is unrecognised, so there is no
    # constructor call that means anything more.
    "UnknownLayout": hurray.UnknownLayout(0x42),
    "PrivateExtensionLayout": hurray.PrivateExtensionLayout(0xF0, 7),
    "BlockPagedLayout": hurray.BlockPagedLayout(16, 4, 1, 2),
}

# Classes that define __repr__ but are never instantiated as themselves.
NOT_INSTANTIABLE = {
    # Abstract base: every layout is one of its subclasses, all covered above.
    "Layout": "base class — `hurray.Layout()` raises, subclasses carry the repr",
}


def _namespace():
    """What `eval` sees: the module, plus its public names unqualified.

    The quantization and layout reprs print bare constructor calls
    (`CsrLayout(nnz=4)`), while Device, Dtype and Tensor print qualified ones
    (`hurray.Device(...)`). Both spellings resolve here.
    """
    ns = {"hurray": hurray}
    ns.update({n: getattr(hurray, n) for n in dir(hurray) if not n.startswith("_")})
    return ns


# ── Rust must not leak ────────────────────────────────────────────────────────

# `Some(`/`Ok(`/`Err(` come from {:?} on an Option or Result; `true`/`false` from {} on
# a bool; `::` and the generic spellings from a type name reaching a format string.
RUST_ISMS = ("Some(", "Ok(", "Err(", "::", "Vec<", "Option<", "->")
RUST_BOOLS = re.compile(r"\b(true|false)\b")


@pytest.mark.parametrize("label", sorted({**EXPRESSION, **SUMMARY}))
def test_no_repr_leaks_a_rust_ism(label):
    text = repr({**EXPRESSION, **SUMMARY}[label])
    for ism in RUST_ISMS:
        assert ism not in text, f"{label}: repr contains {ism!r} — {text}"
    assert not RUST_BOOLS.search(text), f"{label}: Rust bool spelling — {text}"


@pytest.mark.parametrize("label", sorted({**EXPRESSION, **SUMMARY}))
def test_every_repr_is_printable_ascii_python(label):
    """A repr goes into tracebacks and logs; a stray newline or tab breaks both."""
    text = repr({**EXPRESSION, **SUMMARY}[label])
    assert text == text.strip()
    assert "\n" not in text and "\t" not in text


# ── Expression reprs rebuild their object ─────────────────────────────────────


@pytest.mark.parametrize("label", sorted(EXPRESSION))
def test_expression_repr_evaluates_back_to_an_equal_object(label):
    original = EXPRESSION[label]
    text = repr(original)
    rebuilt = eval(text, _namespace())  # noqa: S307 - the string is our own repr
    assert repr(rebuilt) == text, f"{label}: {text} rebuilt as {rebuilt!r}"


def test_a_statistics_repr_names_the_values_not_the_mask():
    """The bug this test exists for: `Statistics(computed_mask=0x4)` was the whole repr,
    so the numbers the bit gates were invisible."""
    text = repr(hurray.Statistics(nnz=1024, value_min=-1.0, value_max=1.0, value_abs_max=1.0))
    assert "computed_mask" not in text
    assert "nnz=1024" in text and "value_min=-1.0" in text


def test_a_scheme_repr_names_the_constructor_that_rebuilds_it():
    """`PerBlockAffine(...)` named a constructor that does not exist — the class is built
    through `.symmetric` / `.asymmetric`, which is also what makes the zero point
    unambiguous."""
    asymmetric = hurray.PerBlockAffine.asymmetric(1, 64, 1, 2, hurray.float32)
    assert repr(asymmetric).startswith("PerBlockAffine.asymmetric(")
    assert "zero_point_buffer_index=2" in repr(asymmetric)

    symmetric = hurray.PerBlockAffine.symmetric(1, 64, 1, hurray.float32)
    assert repr(symmetric).startswith("PerBlockAffine.symmetric(")
    assert "zero_point_buffer_index" not in repr(symmetric)


def test_a_dtype_repr_is_the_way_you_write_that_dtype():
    """It printed `hurray.Dtype('float32')`, which reads as a constructor call. `Dtype`
    has none — the types are module singletons, tier 2 on the submodule only."""
    assert repr(hurray.float32) == "hurray.float32"
    assert repr(hurray.dtype.int4) == "hurray.dtype.int4"


def test_a_summary_repr_still_identifies_its_class():
    for label, obj in SUMMARY.items():
        assert type(obj).__name__ in repr(obj), f"{label}: {repr(obj)}"


# ── Completeness ──────────────────────────────────────────────────────────────


def test_every_repr_bearing_class_is_classified():
    """The guard that keeps this file honest: a class that defines `__repr__` and appears
    in neither table fails here, rather than shipping an unexamined repr."""
    declared = {
        name
        for name, cls in inspect.getmembers(hurray, inspect.isclass)
        if cls.__repr__ is not object.__repr__ and not issubclass(cls, BaseException)
    }
    classified = {type(obj).__name__ for obj in (*EXPRESSION.values(), *SUMMARY.values())}
    classified |= set(NOT_INSTANTIABLE)

    assert not declared - classified, (
        "these classes define __repr__ but are in no table: "
        f"{sorted(declared - classified)}"
    )


def test_the_tables_name_no_class_that_is_gone():
    """The other direction — a table entry for a class the module no longer exposes."""
    exposed = {name for name, _ in inspect.getmembers(hurray, inspect.isclass)}
    assert not set(NOT_INSTANTIABLE) - exposed
