"""Reading a stream you did not produce, and reading back what you wrote (#147, layers 5–6).

Three things the layer-5 and layer-6 pages document that Python could not do. Two are
defences — a descriptor's length field is read before its contents, so a hostile stream
can ask a reader to allocate whatever it likes — and one was a plain hole: `save(kv=...)`
wrote a metadata section that nothing in the binding could read back.
"""

import pytest

import hurray


def _tensor(n: int = 1024) -> hurray.Tensor:
    return hurray.Tensor(bytes(4 * n), hurray.float32, [n])


def _stream(*tensors: hurray.Tensor) -> bytes:
    with hurray.StreamWriter() as writer:
        for tensor in tensors:
            writer.write(tensor)
    return writer.getvalue()


# ── Reader limits ─────────────────────────────────────────────────────────────


def test_a_stream_reads_without_limits_by_default():
    assert len(list(hurray.StreamReader(_stream(_tensor(), _tensor())))) == 2


def test_a_buffer_over_the_limit_is_refused():
    data = _stream(_tensor(1024))  # 4 KiB of buffer
    with pytest.raises(hurray.StreamError) as exc:
        list(hurray.StreamReader(data, max_buffer_bytes=1024))

    message = str(exc.value)
    assert "4096" in message and "1024" in message, message


def test_a_descriptor_over_the_limit_is_refused():
    with pytest.raises(hurray.StreamError):
        list(hurray.StreamReader(_stream(_tensor()), max_descriptor_bytes=8))


def test_a_generous_limit_lets_the_stream_through():
    data = _stream(_tensor())
    assert len(list(hurray.StreamReader(data, max_buffer_bytes=1 << 30))) == 1


def test_the_limit_applies_per_frame_not_per_stream():
    """Ten tensors of 4 KiB each pass a 8 KiB per-buffer limit: the bound is on what a
    single frame may claim, which is the allocation a reader is asked to make."""
    data = _stream(*[_tensor(1024) for _ in range(10)])
    assert len(list(hurray.StreamReader(data, max_buffer_bytes=8192))) == 10


def test_a_composite_depth_limit_is_available():
    inner = hurray.Composite(
        "group", shape=[4], dtype=hurray.float32, members=[_tensor(4)]
    )
    outer = hurray.Composite("group", shape=[4], dtype=hurray.float32, members=[inner])
    with hurray.StreamWriter() as writer:
        writer.write(outer)

    assert len(list(hurray.StreamReader(writer.getvalue(), max_composite_depth=8))) == 1
    with pytest.raises(hurray.StreamError):
        list(hurray.StreamReader(writer.getvalue(), max_composite_depth=1))


# ── cross_machine ─────────────────────────────────────────────────────────────


def test_cross_machine_accepts_what_this_binding_produces():
    """Everything hurray-python builds is producer_synced, so the flag costs a Python
    producer nothing — its value is on the reading end, and in stating the assumption."""
    data = _stream(_tensor())
    assert len(list(hurray.StreamReader(data, cross_machine=True))) == 1

    with hurray.StreamWriter(cross_machine=True) as writer:
        writer.write(_tensor())
    assert writer.getvalue() == data


def test_cross_machine_is_off_by_default():
    """A stream between two processes on one machine may legitimately carry a sync mode
    that means something there, so the check is opt-in."""
    with hurray.StreamWriter() as writer:
        writer.write(_tensor())
    assert list(hurray.StreamReader(writer.getvalue()))


def test_the_limits_compose():
    data = _stream(_tensor())
    assert len(
        list(
            hurray.StreamReader(
                data,
                max_descriptor_bytes=1 << 20,
                max_buffer_bytes=512 << 20,
                max_composite_depth=8,
                cross_machine=True,
            )
        )
    ) == 1


# ── getvalue ──────────────────────────────────────────────────────────────────


def test_getvalue_is_repeatable():
    """It was destructive. The second call returned b"", so
    `list(StreamReader(writer.getvalue()))` after any earlier getvalue() read an empty
    stream and reported a *clean* end of it — data loss wearing a success. Every existing
    test happened to call it exactly once, which is why nothing caught it."""
    with hurray.StreamWriter() as writer:
        writer.write(_tensor(4))

    first = writer.getvalue()
    second = writer.getvalue()

    assert first == second
    assert len(first) > 0
    assert len(list(hurray.StreamReader(writer.getvalue()))) == 1
    assert len(list(hurray.StreamReader(writer.getvalue()))) == 1


def test_getvalue_still_refuses_a_writer_with_a_destination(tmp_path):
    writer = hurray.StreamWriter(str(tmp_path / "s.hrry"))
    writer.write(_tensor(4))
    writer.finish()
    with pytest.raises(hurray.StreamError):
        writer.getvalue()


# ── load_kv ───────────────────────────────────────────────────────────────────


def test_the_kv_section_round_trips(tmp_path):
    """The hole this closes: save(kv=...) wrote metadata nothing could read back."""
    path = str(tmp_path / "m.hrry")
    kv = {
        "model": "demo-v1",
        "layers": 12,
        "lr": 0.001,
        "quantized": False,
        "signature": b"\x01\x02\x03",
        "shape": [1, 2, 3],
        "tags": ["a", "b"],
    }
    hurray.save(path, {"w": _tensor(4)}, kv=kv)

    assert hurray.load_kv(path) == kv


@pytest.mark.parametrize(
    "value",
    ["text", 42, -1, 3.5, True, False, b"bytes", [1, 2, 3], ["a"], [True, False], [1.5]],
)
def test_each_value_type_survives(tmp_path, value):
    path = str(tmp_path / "m.hrry")
    hurray.save(path, {"w": _tensor(4)}, kv={"v": value})

    back = hurray.load_kv(path)["v"]
    assert back == value
    assert type(back) is type(value)


def test_a_file_with_no_kv_section_reads_as_empty(tmp_path):
    path = str(tmp_path / "m.hrry")
    hurray.save(path, {"w": _tensor(4)})
    assert hurray.load_kv(path) == {}


def test_the_kv_section_does_not_need_the_tensors(tmp_path):
    """A footer read, not a scan: reading metadata must not cost the buffers."""
    path = str(tmp_path / "m.hrry")
    hurray.save(path, {f"t{i}": _tensor(4096) for i in range(8)}, kv={"n": 8})

    assert hurray.load_kv(path) == {"n": 8}


def test_a_missing_file_raises(tmp_path):
    with pytest.raises(hurray.FileError):
        hurray.load_kv(str(tmp_path / "does-not-exist.hrry"))


def test_bool_does_not_arrive_as_an_int(tmp_path):
    """Python's bool is a subclass of int, so the writer checks it first — and the
    reader has to give it back as a bool, not as 1."""
    path = str(tmp_path / "m.hrry")
    hurray.save(path, {"w": _tensor(4)}, kv={"flag": True, "count": 1})

    back = hurray.load_kv(path)
    assert back["flag"] is True
    assert type(back["count"]) is int and back["count"] == 1


# ── The pages themselves ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "page_name, minimum",
    [
        ("layer-5-streaming-interchange.md", 5),
        ("layer-6-file-format.md", 4),
    ],
)
def test_the_cookbook_pages_keep_their_python_tabs(page_name, minimum):
    """These blocks are not executed here — they write real files and open real pipes —
    but their presence is checked, so a page cannot quietly lose a tab."""
    import pathlib
    import re

    page = pathlib.Path(__file__).parents[2] / "docs/cookbook" / page_name
    if not page.exists():
        pytest.skip("cookbook not present")

    blocks = re.findall(r"```python\n(.*?)```", page.read_text(), re.S)
    assert len(blocks) >= minimum
    for index, block in enumerate(blocks):
        compile(block, f"{page_name}#python[{index}]", "exec")
