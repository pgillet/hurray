"""The native protocol is named for the format, and only for the format (ADR-033).

`__hurray_buffer__` / `from_hurray_buffer` said "buffer" twice over wrongly: singular
where the capsule carries N buffers, and "buffer" where it carries a whole tensor —
the buffer list *and* the encoded descriptor.

ADR-033 § 4 requires a clean rename with no alias, because two names for one protocol
means two feature probes a consumer has to choose between. A rename that quietly leaves
the old name working is the failure mode these tests exist to catch: everything else in
the suite would keep passing.
"""

import struct

import pytest

import hurray


def _tensor():
    return hurray.Tensor(struct.pack("4f", 1.0, 2.0, 3.0, 4.0), hurray.float32, [4])


# ── The new names work ────────────────────────────────────────────────────────


def test_the_protocol_is_named_for_the_format():
    t = _tensor()
    assert hasattr(t, "__hurray__")
    assert callable(hurray.from_hurray)


def test_the_protocol_round_trips():
    t = _tensor()
    back = hurray.from_hurray(t)
    assert back.shape == t.shape
    assert back.dtype == t.dtype


# ── The old names are gone, not aliased ───────────────────────────────────────


def test_the_old_method_name_is_gone():
    assert not hasattr(_tensor(), "__hurray_buffer__")


def test_the_old_function_name_is_gone():
    assert not hasattr(hurray, "from_hurray_buffer")


def test_the_probe_a_consumer_writes_is_the_only_one_that_works():
    """A third-party binding feature-detects with one hasattr; there must be one."""
    t = _tensor()
    probes = [name for name in ("__hurray__", "__hurray_buffer__") if hasattr(t, name)]
    assert probes == ["__hurray__"]


# ── The capsule carries the payload name ──────────────────────────────────────


def test_the_capsule_is_named_for_its_payload():
    """The wire contract a C consumer validates with PyCapsule_IsValid.

    Deliberately not symmetric with the method name (ADR-033 § 2): brevity in the
    call, precision on the wire — the same split DLPack makes with ``__dlpack__``
    returning a ``"dltensor_versioned"`` capsule.
    """
    capsule = _tensor().__hurray__()
    assert "hurray_tensor" in repr(capsule)
    assert "hurray_buffer" not in repr(capsule)


# ── A non-participant is still rejected by type ───────────────────────────────


def test_an_object_without_the_protocol_raises_type_error():
    with pytest.raises(TypeError):
        hurray.from_hurray(object())
