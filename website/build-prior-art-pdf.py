#!/usr/bin/env python3
"""Render docs/prior-art.md to docs/prior-art.pdf.

The review is published in two forms: Markdown (GitHub and the documentation site) and
PDF (a paginated, citable artifact). The Markdown is the source of truth; this script is
the only way the PDF is produced, so the two never diverge by hand-editing.

Requires two single-binary tools on PATH, neither of which needs root to install:

    pandoc >= 3.0   https://github.com/jgm/pandoc/releases   (Markdown -> Typst)
    typst  >= 0.12  https://github.com/typst/typst/releases  (Typst -> PDF)

Usage:  python3 website/build-prior-art-pdf.py [--check]

--check verifies the tools are present and the Markdown parses, without writing the PDF.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "docs" / "prior-art.md"
OUTPUT = REPO / "docs" / "prior-art.pdf"

# The Markdown uses emoji for the strength/limitation/implication bullets. They render on
# GitHub and the site but need a colour emoji font no PDF toolchain here carries, so they
# are swapped for typographic equivalents that any serif font has.
GLYPHS = {"✅": "✓", "❌": "✗", "\U0001f539": "→"}

# Typst show rules applied to the whole document. The breakable-block rule is the load
# bearing one: pandoc wraps every table in a figure, and an unbreakable figure taller than
# a page silently overprints its own rows.
HEADER = """\
#show figure: set block(breakable: true)
#show figure.where(kind: table): set figure.caption(position: bottom)
#show table: set text(size: 8.2pt, hyphenate: false)
#show table: set par(justify: false, leading: 0.5em)
#set par(justify: true)
"""


def split_front_matter(text: str) -> tuple[str, str, str]:
    """Return (title, subtitle, body). The title block becomes PDF metadata."""
    lines = text.split("\n")
    if not lines[0].startswith("# "):
        sys.exit("prior-art.md must open with a level-1 heading")
    title = lines[0][2:].strip()

    try:
        rule = lines.index("---")
    except ValueError:
        sys.exit("prior-art.md must separate its title block with a horizontal rule")

    head = "\n".join(lines[1:rule])
    subtitle = " ".join(re.findall(r"^\*(.+?)\*$", head, re.S | re.M)).replace("\n", " ")
    return title, re.sub(r"\s+", " ", subtitle).strip(), "\n".join(lines[rule + 1 :])


def promote_table_captions(body: str) -> str:
    """Turn a `**Table N.** …` paragraph above a table into that table's caption.

    Left as an ordinary paragraph the label floats free of the table it names and can land
    on the previous page; as a pandoc caption it is bound to the table.
    """
    lines = body.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        if re.match(r"^\*\*Table \d+\.\*\*", lines[i]):
            stop = i
            while stop < len(lines) and lines[stop]:
                stop += 1
            # Drop the "Table N." label itself: Typst numbers and prefixes the caption.
            caption = re.sub(r"^\*\*Table \d+\.\*\*\s*", "", " ".join(lines[i:stop])).strip()
            start = stop
            while start < len(lines) and not lines[start]:
                start += 1
            if start < len(lines) and lines[start].startswith("|"):
                end = start
                while end < len(lines) and lines[end].startswith("|"):
                    end += 1
                out.extend(lines[start:end])
                out.append(": " + caption)
                out.append("")
                i = end
                while i < len(lines) and not lines[i]:
                    i += 1
                continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def require_tools() -> None:
    missing = [tool for tool in ("pandoc", "typst") if shutil.which(tool) is None]
    if missing:
        sys.exit(
            f"missing required tool(s): {', '.join(missing)}\n"
            "Both ship as single static binaries needing no root:\n"
            "  pandoc  https://github.com/jgm/pandoc/releases\n"
            "  typst   https://github.com/typst/typst/releases"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify inputs, write nothing")
    args = parser.parse_args()

    require_tools()
    source = SOURCE.read_text(encoding="utf-8")
    title, subtitle, body = split_front_matter(source)
    for emoji, replacement in GLYPHS.items():
        body = body.replace(emoji, replacement)
    body = promote_table_captions(body)

    revision = re.search(r"\*\*Revision:\*\*\s*(.+)", source)
    # The header's "also available as PDF" pointer is meaningless inside the PDF itself.
    date = re.split(r"\s*·\s*Also available", revision.group(1))[0].strip() if revision else ""

    if args.check:
        print(f"ok: {SOURCE.relative_to(REPO)} parses; title {title!r}, revision {date!r}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "body.md"
        staged.write_text(body, encoding="utf-8")
        subprocess.run(
            [
                "pandoc", str(staged),
                "-o", str(OUTPUT),
                # implicit_figures off: the figures carry their own bold caption paragraph,
                # and leaving it on would emit the alt text as a second caption.
                "-f", "markdown-implicit_figures",
                "--pdf-engine=typst",
                "--resource-path", str(SOURCE.parent),
                "-V", "papersize=a4",
                "-V", "fontsize=10pt",
                "-V", "margin-x=2.1cm",
                "-V", "margin-y=2.1cm",
                "-V", f"header-includes={HEADER}",
                "-M", f"title={title}",
                "-M", f"subtitle={subtitle}",
                "-M", f"date={date}",
            ],
            check=True,
        )
    print(f"wrote {OUTPUT.relative_to(REPO)} ({OUTPUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
