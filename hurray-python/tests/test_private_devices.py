"""Vendor devices and vendor memory classes (#147, the last coverage gap).

`0xF0`–`0xFE` is the private range in both the device-tag and memory-class spaces: an
agreement between one producer and one consumer, with no name the spec can give it. Python
could neither author one nor tell two apart — `device.rs` reported `"private"` for every
tag and refused the name on the way in.

The fix follows the one the extension element types got: the name stays `"private"`,
because that is what the spec says these are, and the *tag* is what identifies them.
"""

import pytest

import hurray


# ── Authoring ─────────────────────────────────────────────────────────────────


def test_a_private_device_can_be_built_from_its_tag():
    device = hurray.Device(0xF2)
    assert device.kind == "private"
    assert device.tag == 0xF2
    assert device.is_private


def test_every_tag_in_the_private_range_works():
    for tag in range(0xF0, 0xFF):
        assert hurray.Device(tag).tag == tag


def test_a_private_memory_class_can_be_built_too():
    device = hurray.Device("cuda", 0, memory_class=0xF1)
    assert device.kind == "cuda"
    assert device.memory_class == "private"
    assert device.memory_class_tag == 0xF1


def test_both_at_once():
    device = hurray.Device(0xF2, 3, memory_class=0xF1)
    assert (device.tag, device.device_id, device.memory_class_tag) == (0xF2, 3, 0xF1)


def test_named_kinds_still_take_names():
    device = hurray.Device("cuda", 1, "unified")
    assert device.kind == "cuda"
    assert device.tag == 0x01
    assert device.memory_class == "unified"
    assert device.memory_class_tag == 0x02
    assert not device.is_private


def test_a_named_kind_can_also_be_given_as_a_byte():
    """One field, two spellings — the byte is what the wire carries either way."""
    assert hurray.Device(0x01) == hurray.Device("cuda")
    assert hurray.Device("cpu", 0, 0x02) == hurray.Device("cpu", 0, "unified")


# ── Telling them apart ────────────────────────────────────────────────────────


def test_two_private_devices_are_not_equal():
    assert hurray.Device(0xF2) != hurray.Device(0xF5)


def test_two_private_devices_do_not_print_the_same():
    """The hole this closes: both said `kind='private'` and nothing else, so a repr
    could not tell a caller which vendor device it was looking at."""
    assert repr(hurray.Device(0xF2)) != repr(hurray.Device(0xF5))
    assert "0xF2" in repr(hurray.Device(0xF2))


def test_a_private_memory_class_shows_its_tag_too():
    assert "0xF1" in repr(hurray.Device("cuda", 0, memory_class=0xF1))


def test_a_named_device_prints_as_before():
    """The tag is only added where the name does not identify the value."""
    text = repr(hurray.Device("cuda", 1, "unified"))
    assert text == "hurray.Device(kind='cuda', device_id=1, memory_class='unified')"


def test_private_devices_hash_by_their_tag():
    devices = {hurray.Device(0xF2), hurray.Device(0xF5), hurray.Device(0xF2)}
    assert len(devices) == 2


# ── What is refused ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("tag", [0x09, 0x10, 0x7F, 0xEF])
def test_a_reserved_tag_is_refused(tag):
    """Reserved for a future version of the format — not something to invent a device
    for, since a later reader would disagree about what it means."""
    with pytest.raises(hurray.InvalidDescriptorError):
        hurray.Device(tag)


def test_the_permanently_invalid_tag_is_refused():
    with pytest.raises(hurray.InvalidDescriptorError):
        hurray.Device(0xFF)


def test_an_unknown_name_still_names_the_alternatives():
    with pytest.raises(hurray.InvalidDescriptorError) as exc:
        hurray.Device("tpu")

    message = str(exc.value)
    assert "cuda" in message
    assert "wire byte" in message, "the error should mention the other spelling"


def test_a_nonsense_kind_is_refused():
    with pytest.raises(hurray.InvalidDescriptorError):
        hurray.Device(3.5)


def test_a_reserved_memory_class_is_refused():
    with pytest.raises(hurray.InvalidDescriptorError):
        hurray.Device("cuda", 0, memory_class=0x40)


# ── On a tensor, and on the wire ──────────────────────────────────────────────


def test_a_tensor_can_live_on_a_private_device():
    tensor = hurray.Tensor(bytes(16), hurray.float32, [4], device=hurray.Device(0xF2))
    assert tensor.device.tag == 0xF2
    assert tensor.device.is_private


def test_the_tag_survives_the_wire():
    tensor = hurray.Tensor(
        bytes(16), hurray.float32, [4], device=hurray.Device(0xF2, 0, memory_class=0xF1)
    )
    decoded = hurray.Descriptor.decode(tensor.descriptor.encode())

    assert decoded.device.tag == 0xF2
    assert decoded.device.memory_class_tag == 0xF1


def test_the_buffer_handles_agree():
    """Colocation: one device per descriptor, so the handle reports the same private
    tag rather than a second copy of it."""
    tensor = hurray.Tensor(bytes(16), hurray.float32, [4], device=hurray.Device(0xF2))
    assert tensor.buffer_handles[0].device is tensor.device
    assert tensor.buffer_handles[0].device.tag == 0xF2
