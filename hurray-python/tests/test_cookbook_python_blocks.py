"""Every ```python block in ``docs/cookbook/`` is executed, or says why it is not.

The Rust half of the cookbook has had this since #187: ``website/check-rust-blocks.py``
runs ``rustdoc --test`` over every page in CI. The Python half was assumed to have it —
#187 said so — but what existed was seven copies of the same fifteen-line ``exec`` loop,
scattered through unrelated test modules, each with its own hardcoded page list. They
covered 11 pages of 29. The other 18 held 100 blocks that nothing read.

The classification lives here rather than in the fence because it cannot live in the
fence: ``lang-tabs.js`` groups blocks by language, so writing ```python,ignore`` would
change that tab's identity and break the page.

Two tiers, mirroring the Rust convention (``rust`` / ``rust,ignore`` + a reason):

- ``RUN`` — every block executes, in order, in one namespace, in a scratch directory.
  A page is a document read top to bottom, so a later block may use an earlier one's
  names and files.
- ``COMPILE`` — the blocks are parsed but not run, and the entry states why. Reserved
  for pages whose snippets are deliberately fragments: a socket, a path, a list of
  tensors the reader supplies.

``test_every_page_is_classified`` fails on a page in neither table, so a new cookbook
page cannot arrive without that decision being made.
"""

import os
import pathlib
import re
import tempfile

import pytest

COOKBOOK = pathlib.Path(__file__).resolve().parents[2] / "docs" / "cookbook"

# Pages whose blocks all execute. The value is the set of imports that may be missing
# from the environment: a block that fails on one of these is skipped rather than
# failed, and any other ImportError is still a failure.
RUN: dict[str, frozenset[str]] = {
    "authoring-quantized-tensors.md": frozenset(),
    "block-paged-kv-cache.md": frozenset(),
    "composite-tensors.md": frozenset(),
    "converting-external-quantized-tensors.md": frozenset(),
    # torch, jax and cupy are each shown handing a buffer across; CI installs none.
    "framework-interop.md": frozenset({"torch", "jax", "cupy"}),
    "hurray-inspect-cli.md": frozenset(),
    "hurray-python-construction.md": frozenset(),
    "hurray-python-display.md": frozenset(),
    "hurray-python-dlpack-numpy.md": frozenset({"torch"}),
    "hurray-python-error-handling.md": frozenset(),
    "hurray-python-file-io.md": frozenset(),
    "hurray-python-layouts.md": frozenset(),
    "hurray-python-native-buffer.md": frozenset(),
    "hurray-python-sparse-scipy.md": frozenset(),
    "hurray-python-tensor-basics.md": frozenset(),
    "ipc-streaming.md": frozenset(),
    "layer-0-element-types-and-shape.md": frozenset(),
    "layer-1-buffer-protocol.md": frozenset(),
    "layer-2-quantization-descriptors.md": frozenset(),
    "layer-3-layout-descriptors.md": frozenset(),
    "layer-4-tensor-descriptor-encoding.md": frozenset(),
    "layer-5-streaming-interchange.md": frozenset(),
    "layer-6-file-format.md": frozenset(),
    "multi-buffer-tensors.md": frozenset(),
    "quantized-inference.md": frozenset(),
    "quickstart.md": frozenset(),
}

# Pages parsed but not executed, and why. A reason is required: "it does not run" is
# a property of the snippet, and if nobody can say which property, it is rot.
COMPILE: dict[str, str] = {
    "composite-file.md": (
        "both blocks take a composite the reader already built; assembling one here "
        "would be six lines of setup around two lines of subject"
    ),
    "composite-streaming.md": (
        "the sink and source are the reader's — the page is about what crosses them"
    ),
    "hurray-python-streaming.md": (
        "idiom fragments: a list of tensors, a socket, a path, a BytesIO the caller owns"
    ),
}


def _blocks(page: pathlib.Path) -> list[str]:
    return re.findall(r"```python\n(.*?)```", page.read_text(), re.S)


def _pages_with_python() -> dict[str, pathlib.Path]:
    return {p.name: p for p in sorted(COOKBOOK.glob("*.md")) if _blocks(p)}


# ── Completeness ──────────────────────────────────────────────────────────────


def test_the_cookbook_is_present():
    """Guards against every test below passing vacuously on a bad path."""
    assert COOKBOOK.is_dir(), f"no cookbook at {COOKBOOK}"
    assert _pages_with_python(), "no cookbook page has a Python block"


def test_every_page_is_classified():
    """A page with Python tabs and no entry in either table fails here."""
    unclassified = set(_pages_with_python()) - set(RUN) - set(COMPILE)
    assert not unclassified, (
        "these cookbook pages have Python blocks but are in neither RUN nor COMPILE: "
        f"{sorted(unclassified)}"
    )


@pytest.mark.parametrize("page_name", sorted({**RUN, **COMPILE}))
def test_a_classified_page_still_has_python_blocks(page_name):
    """The other direction: an entry for a page that has lost its tabs, or its file."""
    page = COOKBOOK / page_name
    assert page.exists(), f"{page_name} is classified but does not exist"
    assert _blocks(page), f"{page_name} is classified but has no Python block left"


def test_no_page_is_in_both_tables():
    assert not set(RUN) & set(COMPILE)


# ── Running ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("page_name", sorted(RUN))
def test_every_python_block_on_the_page_runs(page_name):
    page = COOKBOOK / page_name
    blocks = _blocks(page)
    optional = RUN[page_name]

    # One namespace for the page: blocks build on each other, the way a reader reads
    # them. One scratch directory: several blocks write files, and a leftover from a
    # previous run once made a block that *reads* one pass without its writer.
    namespace: dict = {}
    previous = os.getcwd()
    os.chdir(tempfile.mkdtemp())
    try:
        for index, block in enumerate(blocks):
            where = f"{page_name}#python[{index}]"
            try:
                exec(compile(block, where, "exec"), namespace)
            except ImportError as exc:
                if exc.name in optional or any(m in str(exc) for m in optional):
                    continue
                raise
    finally:
        os.chdir(previous)


@pytest.mark.parametrize("page_name", sorted(COMPILE))
def test_every_python_block_on_the_page_parses(page_name):
    """Not executed — see the reason in COMPILE — but a syntax error is still caught."""
    page = COOKBOOK / page_name
    for index, block in enumerate(_blocks(page)):
        compile(block, f"{page_name}#python[{index}]", "exec")


def test_every_compile_only_page_states_a_reason():
    for page_name, reason in COMPILE.items():
        assert reason.strip(), f"{page_name} is compile-only with no reason"
