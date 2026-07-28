"""Conformance test: the Python binding validates against the shared golden corpus.

Loads the committed HRRYFILE vectors under ``conformance/vectors/`` and asserts their
tensors match ``manifest.json`` — the same corpus the Rust conformance test checks.

This proves the language-agnostic interchange path end-to-end for the reference Python
binding. Note: ``hurray-python`` wraps ``hurray-core``/``-io`` (it does not re-implement the
decoder), so this validates the binding layer + round-trip fidelity against a shared golden
corpus — not an independent re-implementation.
"""

import json
from pathlib import Path

import pytest

import hurray

VECTORS_DIR = Path(__file__).resolve().parents[2] / "conformance" / "vectors"
MANIFEST_PATH = VECTORS_DIR / "manifest.json"


def _load_manifest():
    with MANIFEST_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def test_corpus_is_present():
    assert MANIFEST_PATH.is_file(), f"missing corpus manifest at {MANIFEST_PATH}"
    manifest = _load_manifest()
    assert manifest["format_version"] == "1.0"
    assert manifest["files"], "corpus has no file vectors"


def _file_vector_ids():
    return [fv["file"] for fv in _load_manifest()["files"]]


@pytest.mark.parametrize("file_vector", _load_manifest()["files"], ids=_file_vector_ids())
def test_file_vector_tensors_match_manifest(file_vector):
    """Every tensor in a committed .hrry loads with the manifest's name/shape/dtype."""
    path = VECTORS_DIR / file_vector["file"]
    tensors = hurray.load(str(path))

    # Names, in index order (dict preserves the file's tensor order).
    expected_names = [t["name"] for t in file_vector["tensors"]]
    assert list(tensors.keys()) == expected_names

    for expected in file_vector["tensors"]:
        tensor = tensors[expected["name"]]
        assert tuple(tensor.shape) == tuple(expected["shape"]), expected["name"]
        assert tensor.dtype.name == expected["element_type"], expected["name"]


# Note: hurray.load() does not currently expose the file's key-value metadata to Python,
# so the manifest's `kv` expectations are validated by the Rust conformance test only.
# Exposing file KV in the Python binding is tracked as a follow-up.
