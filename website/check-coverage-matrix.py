#!/usr/bin/env python3
"""Generate the Implementation Status page from the real artifacts, and fail on drift.

The site documents what the format specifies and what the reference implementation does,
but nothing stated which spec features are actually implemented, by which implementation,
at which conformance level (issue #197). A reader had to go and read the crates.

A hand-maintained matrix answers that question until the day it silently stops being true,
which is worse than not having one. So this page is derived, never written:

    conformance/src/bin/support_probe.rs   what the Rust crates support, proven by
                                           round-trips and by walking the whole tag space
    the compiled `hurray` module           introspected in this interpreter
    hurray-ffi/include/hurray.h            the exported C symbols
    website/coverage-matrix.toml           the canonical feature list (rows) and columns

Everything the probes cannot mechanically detect — the network transport, the C layer's
tag pass-through — carries an explicit declared status with a required justification, so a
gap reads as a deliberate statement rather than a silent absence.

Run from the repo root:

    python3 website/check-coverage-matrix.py            # verify, exit 1 on drift
    python3 website/check-coverage-matrix.py --write    # regenerate the page

The interpreter must be one with the `hurray` extension installed (`maturin develop`),
otherwise the Python column cannot be probed and the check refuses to run.
"""

import argparse
import difflib
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

CONFIG = Path("website/coverage-matrix.toml")
PAGE = Path("docs/impl/implementation-status.md")
C_HEADER = Path("hurray-ffi/include/hurray.h")

# Status → cell glyph. `no` is the only status that needs no justification: it is the only
# one that cannot overstate what an implementation does.
GLYPHS = {
    "yes": "✅",
    "no": "❌",
    "partial": "◐",
    "n/a": "➖",
    "spec-only": "📄",
}
NEEDS_NOTE = {"yes", "partial", "n/a", "spec-only"}
LEGEND = [
    ("yes", "Implemented"),
    ("partial", "Partly implemented — see the note"),
    ("no", "Not implemented"),
    ("n/a", "Not this layer's to implement — see the note"),
    ("spec-only", "Specified, implemented nowhere — see the note"),
]

# Which recipes each probe backend can answer.
RECIPES = {
    "rust": {"element-type", "layout-tag", "quant-scheme", "capability"},
    "python": {"element-type", "python-class", "python-attr"},
    "c-header": {"c-symbol"},
}


class ConfigError(Exception):
    """The feature list and the probes disagree, or the feature list is malformed."""


# ── Probes ───────────────────────────────────────────────────────────────────────────


def probe_rust() -> dict:
    """Run the conformance support probe and return its report."""
    proc = subprocess.run(
        ["cargo", "run", "-q", "-p", "hurray-conformance", "--bin", "support-probe"],
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise ConfigError("the Rust support probe failed to run")
    return json.loads(proc.stdout)


def probe_python() -> dict:
    """Introspect the installed `hurray` extension module.

    Element types are read out of `Dtype.from_tag` one tag at a time, the same walk the
    Rust probe does. Everything else is surface presence: whether the binding *behaves*
    correctly is what its pytest suite is for, and duplicating that here would give the
    page a second, weaker copy of an answer it already has.
    """
    try:
        import hurray
    except ImportError as exc:
        raise ConfigError(
            f"cannot import the `hurray` extension module ({exc}).\n"
            "Build it into this interpreter first:\n"
            "    maturin develop -m hurray-python/Cargo.toml"
        ) from exc

    element_types = {}
    for tag in range(256):
        try:
            element_types[f"0x{tag:02X}"] = hurray.Dtype.from_tag(tag).name
        except Exception:  # noqa: BLE001 — any refusal means "not this tag"
            continue
    return {"module": hurray, "element_types": element_types}


def probe_c_header() -> set:
    """Return the `hurray_*` functions the generated C header declares.

    Comments are stripped first: the header's doc comments cross-reference symbols by
    name, and a mention is not an export.
    """
    source = C_HEADER.read_text(encoding="utf-8")
    source = re.sub(r"/\*.*?\*/", " ", source, flags=re.DOTALL)
    source = re.sub(r"//[^\n]*", " ", source)
    return set(re.findall(r"\b(hurray_[a-z0-9_]+)\s*\(", source))


def python_attr_exists(module, path: str) -> bool:
    """Resolve a dotted attribute path such as `Descriptor.decode` against the module."""
    target = module
    for part in path.split("."):
        target = getattr(target, part, None)
        if target is None:
            return False
    return True


# ── Cell resolution ──────────────────────────────────────────────────────────────────


def parse_cell(value: str, where: str) -> tuple:
    """Split a cell value into (recipe, status, note_id). Exactly one of recipe/status."""
    head, _, note = value.partition(":")
    note = note or None
    if head in GLYPHS:
        if head in NEEDS_NOTE and not note:
            raise ConfigError(f"{where}: status '{head}' requires a note id")
        if head not in NEEDS_NOTE and note:
            raise ConfigError(f"{where}: status '{head}' takes no note id")
        return None, head, note
    if note:
        raise ConfigError(f"{where}: detection recipe '{head}' takes no note id")
    return head, None, None


def resolve(recipe: str, row: dict, impl: dict, probes: dict, where: str) -> str:
    """Run one detection recipe and return the resulting status."""
    backend = impl.get("probe")
    if backend not in RECIPES:
        raise ConfigError(f"{where}: column '{impl['id']}' has no probe, so it must declare a status")
    if recipe not in RECIPES[backend]:
        raise ConfigError(f"{where}: recipe '{recipe}' is not answerable by the '{backend}' probe")

    def field(name: str) -> list:
        value = row.get(name)
        if not value:
            raise ConfigError(f"{where}: recipe '{recipe}' needs a non-empty `{name}`")
        return value

    if recipe == "element-type":
        return yes_no(row_tag(row, where) in probes[backend]["element_types"])
    if recipe == "layout-tag":
        return yes_no(row_tag(row, where) in probes["rust"]["layout_tags"])
    if recipe == "quant-scheme":
        return yes_no(row_tag(row, where) in probes["rust"]["quantization_scheme_tags"])
    if recipe == "capability":
        name = row.get("capability")
        if name not in probes["rust"]["capabilities"]:
            raise ConfigError(f"{where}: no capability '{name}' in the Rust probe report")
        return yes_no(probes["rust"]["capabilities"][name])
    if recipe == "python-class":
        module = probes["python"]["module"]
        return yes_no(
            all(isinstance(getattr(module, name, None), type) for name in field("python_class"))
        )
    if recipe == "python-attr":
        module = probes["python"]["module"]
        return yes_no(all(python_attr_exists(module, p) for p in field("python_attrs")))
    if recipe == "c-symbol":
        return yes_no(set(field("c_symbols")) <= probes["c-header"])
    raise ConfigError(f"{where}: unknown detection recipe '{recipe}'")


def yes_no(supported: bool) -> str:
    return "yes" if supported else "no"


def row_tag(row: dict, where: str) -> str:
    """A row's wire tag, formatted the way the probes report tags."""
    if "tag" not in row:
        raise ConfigError(f"{where}: this row is matched by wire tag, so it needs a `tag`")
    return f"0x{row['tag']:02X}"


# ── Cross-checks ─────────────────────────────────────────────────────────────────────


def cross_check(config: dict, probes: dict) -> None:
    """Fail if the feature list and the probes have drifted apart.

    This is what makes the page self-maintaining in the direction that matters: a type,
    layout, scheme, or capability that the implementation gains and the matrix does not
    mention turns CI red instead of quietly going unlisted.
    """
    by_recipe = {"element-type": {}, "layout-tag": {}, "quant-scheme": {}}
    capabilities = {}
    for section in config["sections"]:
        for row in section["rows"]:
            where = f"section '{section['id']}', row '{row['id']}'"
            recipe = row.get("rust", section.get("detect", {}).get("rust", ""))
            recipe = recipe.split(":")[0]
            if recipe in by_recipe:
                by_recipe[recipe][row_tag(row, where)] = row["id"]
            elif recipe == "capability":
                name = row.get("capability")
                if name in capabilities:
                    raise ConfigError(
                        f"capability '{name}' is claimed by both rows "
                        f"'{capabilities[name]}' and '{row['id']}'"
                    )
                capabilities[name] = row["id"]

    for tag, name in probes["rust"]["element_types"].items():
        listed = by_recipe["element-type"].get(tag)
        if listed is None:
            raise ConfigError(
                f"hurray-core decodes element type {tag} ({name}) but no row lists it — "
                f"add one to {CONFIG}"
            )
        if listed != name:
            raise ConfigError(f"element type {tag} is '{name}' in hurray-core, '{listed}' in {CONFIG}")

    for tag in probes["rust"]["layout_tags"]:
        if tag not in by_recipe["layout-tag"]:
            raise ConfigError(f"hurray-core knows layout tag {tag} but no row lists it")
    for tag in probes["rust"]["quantization_scheme_tags"]:
        if tag not in by_recipe["quant-scheme"]:
            raise ConfigError(f"hurray-core knows quantization scheme tag {tag} but no row lists it")
    for name in probes["rust"]["capabilities"]:
        if name not in capabilities:
            raise ConfigError(f"the Rust probe reports capability '{name}' but no row uses it")

    # The Python binding must agree with the reference implementation about type *names*,
    # not just about which tags exist: a renamed dtype is a compatibility break that would
    # otherwise show up as two green cells.
    for tag, name in probes["python"]["element_types"].items():
        expected = probes["rust"]["element_types"].get(tag)
        if expected is not None and expected != name:
            raise ConfigError(f"element type {tag} is '{expected}' in Rust but '{name}' in Python")


# ── Rendering ────────────────────────────────────────────────────────────────────────


def place_notes(cells: dict, row_count: int, implementations: list, notes: dict) -> tuple:
    """Decide where each note is written: once per table, once per column, or per cell.

    A note that holds for every cell it could hold for says nothing extra when repeated in
    each one — the whole `hurray-ffi` element-type column carries the same sentence, and a
    marker beside all twenty-six rows only makes the table harder to read. So a note is
    lifted to the widest scope that covers all of its cells.

    Returns `({(row, impl): marker}, [footnote lines])`.
    """
    scope = {}
    for (r, i), (_, note) in cells.items():
        if note:
            scope.setdefault(note, set()).add((r, i))

    all_cells = {(r, i) for r in range(row_count) for i in range(len(implementations))}
    markers = {key: "" for key in all_cells}
    footnotes = []
    letters = {}

    for note, covered in scope.items():
        text = " ".join(notes[note].split())
        if covered == all_cells:
            footnotes.append(f"- **Every column** — {text}")
            continue
        columns = {i for _, i in covered}
        if len(columns) == 1:
            column = columns.pop()
            if covered == {(r, column) for r in range(row_count)}:
                footnotes.append(f"- **{implementations[column]['label']}** — {text}")
                continue
        letters[note] = chr(ord("a") + len(letters))
        for key in covered:
            markers[key] = f" ({letters[note]})"
        footnotes.append(f"- **({letters[note]})** {text}")

    return markers, footnotes


def render(config: dict, probes: dict) -> str:
    implementations = config["implementations"]
    notes = config["notes"]
    used_notes = set()
    out = []

    out.append("<!-- GENERATED by website/check-coverage-matrix.py — do not edit by hand.")
    out.append("     Rows come from website/coverage-matrix.toml; cells come from probing the")
    out.append("     crates, the Python module, and the C header. Run the script with --write. -->")
    out.append("")
    out.append("# Implementation Status")
    out.append("")
    out.append(
        "Which spec features each Hurray implementation actually provides. Rows are format "
        "features, taken from [Compliance](compliance.md); columns are implementations."
    )
    out.append("")
    out.append("| Implementation | Covers |")
    out.append("|---|---|")
    for impl in implementations:
        out.append(f"| {impl['label']} | {impl['summary']} |")
    out.append("")
    out.append(
        "This page is generated, never hand-written. Tag coverage is read back out of the "
        "decoders one byte at a time, each capability is proven by an encode → decode "
        "round-trip, the Python column is introspected from the compiled module, and the C "
        "column is the set of symbols the generated header exports. CI regenerates the page "
        "and fails if it differs from the committed copy, so a feature an implementation "
        "gains — or loses — cannot pass unreported."
    )
    out.append("")
    out.append(
        "What no probe can detect carries a declared status and a justification, listed under "
        "the table it appears in."
    )
    out.append("")
    out.append("**Legend**")
    out.append("")
    for status, meaning in LEGEND:
        out.append(f"- {GLYPHS[status]} &nbsp;{meaning}")
    out.append("")
    out.append(
        "Adding a third-party implementation is a column, not a redesign: append an "
        "`[[implementations]]` block to `website/coverage-matrix.toml` with `default = "
        '"no"`, then override that per section or per row as the implementation covers '
        "more. No row changes shape."
    )

    for section in config["sections"]:
        out.append("")
        out.append(f"## {section['title']}")
        out.append("")
        intro = section.get("intro", "").strip()
        if intro:
            out.append(intro)
            out.append("")

        columns = section.get("columns", [])
        header = ["Feature"] + [c.capitalize() for c in columns] + [i["label"] for i in implementations]
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "---|" * len(header))

        # (row index, implementation index) -> (status, note id or None)
        cells = {}
        for r, row in enumerate(section["rows"]):
            for i, impl in enumerate(implementations):
                where = f"section '{section['id']}', row '{row['id']}', column '{impl['id']}'"
                # Row override, then the section default, then the column's own fallback —
                # which is what lets a third-party implementation be added as one
                # `[[implementations]]` block with `default = "no"`, and refined row by row
                # afterwards, instead of touching every section first.
                declared = row.get(impl["id"])
                if declared is None:
                    declared = section.get("detect", {}).get(impl["id"])
                if declared is None:
                    declared = impl.get("default")
                if declared is None:
                    raise ConfigError(f"{where}: no detection recipe and no declared status")
                recipe, status, note = parse_cell(declared, where)
                if recipe is not None:
                    status = resolve(recipe, row, impl, probes, where)
                if note and note not in notes:
                    raise ConfigError(f"{where}: unknown note '{note}'")
                cells[(r, i)] = (status, note)
        used_notes.update(note for _, note in cells.values() if note)

        markers, footnotes = place_notes(cells, len(section["rows"]), implementations, notes)
        for r, row in enumerate(section["rows"]):
            line = [row.get("name", f"`{row['id']}`")]
            for column in columns:
                value = row.get(column)
                line.append(f"`0x{value:02X}`" if column == "tag" else str(value))
            for i in range(len(implementations)):
                line.append(GLYPHS[cells[(r, i)][0]] + markers[(r, i)])
            out.append("| " + " | ".join(line) + " |")

        if footnotes:
            out.append("")
            out.extend(footnotes)

    unused = set(notes) - used_notes
    if unused:
        raise ConfigError(f"unused note(s) in {CONFIG}: {', '.join(sorted(unused))}")

    out.append("")
    return "\n".join(out)


# ── Entry point ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="regenerate the page instead of verifying it"
    )
    args = parser.parse_args()

    if not CONFIG.is_file():
        print(f"error: {CONFIG} not found (run from the repo root)", file=sys.stderr)
        return 2

    config = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    try:
        probes = {
            "rust": probe_rust(),
            "python": probe_python(),
            "c-header": probe_c_header(),
        }
        cross_check(config, probes)
        page = render(config, probes)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.write:
        PAGE.write_text(page, encoding="utf-8")
        print(f"wrote {PAGE}")
        return 0

    current = PAGE.read_text(encoding="utf-8") if PAGE.is_file() else ""
    if current == page:
        print(f"{PAGE} is up to date with the implementations. ✓")
        return 0

    diff = difflib.unified_diff(
        current.splitlines(keepends=True),
        page.splitlines(keepends=True),
        fromfile=f"{PAGE} (committed)",
        tofile=f"{PAGE} (probed)",
    )
    sys.stdout.writelines(diff)
    print(
        f"\nerror: {PAGE} does not match what the implementations support.\n"
        "Run `python3 website/check-coverage-matrix.py --write` and commit the result.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
