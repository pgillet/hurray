#!/usr/bin/env python3
"""Validate internal Markdown cross-links so they work on BOTH GitHub and the mdBook site.

The docs use relative `.md` links between spec/impl/cookbook/ADR files. GitHub resolves
those against the raw `.md` files; mdBook renders each `.md` to `.html` and rewrites the
links. For a link to work in both places it must:

  1. stay inside `docs/` (out-of-book targets can't be rendered — use an absolute URL);
  2. point at a file that exists;
  3. point at a file that is part of the book (listed in `docs/SUMMARY.md`), so mdBook
     renders it; and
  4. not target a `README.md` — mdBook renders `README.md` to `index.html` but rewrites
     the link to `README.html` (a 404). Link to the directory (`some/dir/`) instead, which
     resolves to the folder's README on GitHub and to `index.html` on the site.

Exits non-zero and prints every offending link if any rule is violated.

Run from the repo root: `python3 website/check-doc-links.py`
"""

import re
import sys
from pathlib import Path

DOCS = Path("docs").resolve()
SUMMARY = DOCS / "SUMMARY.md"
LINK_RE = re.compile(r"\]\(\s*([^)\s]+?)\s*\)")
# Full inline link, capturing visible text + URL — used to lint internal link labels.
TEXT_RE = re.compile(r"\[([^\]]*)\]\(\s*([^)\s]+)\s*\)")


def rendered_targets() -> set:
    """Absolute paths of every `.md` file listed in SUMMARY.md (i.e. rendered by mdBook)."""
    out = set()
    for m in re.finditer(r"\]\(([^)]+\.md)\)", SUMMARY.read_text(encoding="utf-8")):
        out.add((DOCS / m.group(1)).resolve())
    return out


def main() -> int:
    if not SUMMARY.is_file():
        print(f"error: {SUMMARY} not found (run from the repo root)", file=sys.stderr)
        return 2

    rendered = rendered_targets()
    problems = []

    for md in sorted(DOCS.rglob("*.md")):
        if md.name == "SUMMARY.md":
            continue
        rel_src = md.relative_to(DOCS)
        content = md.read_text(encoding="utf-8")

        # Link labels should read as titles, not file artifacts: no ".md" extension, and
        # no bare repo path (a "/" with no surrounding spaces, e.g. `layouts/row-major`) —
        # the repo tree doesn't match the site's chapter structure. Real titles with a
        # slash keep spaces around it ("Tiled / Blocked"), so they are allowed.
        for tm in TEXT_RE.finditer(content):
            label, url = tm.group(1), tm.group(2)
            # Only lint labels of INTERNAL links; external links may show a URL as text.
            if url.startswith(("http://", "https://", "mailto:")):
                continue
            if ".md" in label:
                problems.append(
                    (rel_src, label, "link text shows a .md extension (drop it; keep .md only in the URL)")
                )
            elif "/" in label and not any(c.isspace() for c in label):
                problems.append(
                    (rel_src, label, "link text is a repo path; use the chapter title instead")
                )

        for m in LINK_RE.finditer(content):
            raw = m.group(1)
            if raw.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path_part = raw.split("#", 1)[0]
            if not path_part.endswith(".md"):
                continue  # directory links / assets are not our concern here
            target = (md.parent / path_part).resolve()
            try:
                target.relative_to(DOCS)
            except ValueError:
                problems.append((rel_src, raw, "points outside docs/ (use an absolute URL)"))
                continue
            if target.name == "README.md":
                problems.append(
                    (rel_src, raw, "targets a README.md (mdBook -> index.html); link the directory instead")
                )
                continue
            if not target.exists():
                problems.append((rel_src, raw, "target does not exist (dead link)"))
            elif target not in rendered:
                problems.append((rel_src, raw, "target not in SUMMARY.md (won't render on the site)"))

    if problems:
        print(f"Found {len(problems)} broken doc cross-link(s):\n")
        for src, raw, why in problems:
            print(f"  docs/{src}: `{raw}` — {why}")
        print("\nFix so links work on both GitHub and the generated site.")
        return 1

    print("All internal Markdown cross-links resolve on both GitHub and the site. ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
