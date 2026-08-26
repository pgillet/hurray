#!/usr/bin/env python3
"""Compile (and run) every Rust code block in the cookbook against the real crates.

CI runs every program under `<crate>/examples/`, and the cookbook's Python tabs are
executed by tests written alongside them — but a fenced ```rust block on a cookbook page
was inert text. Three had rotted unnoticed before this check existed (see issue #187).

Mechanism: `rustdoc --test <page>.md` treats a Markdown file as a doctest carrier. It
extracts the ```rust fences, compiles each one, and runs it unless the fence says
otherwise. Fences in any other language are skipped, so the Python/C/text blocks on the
same page are left alone.

The fence records intent, and CI enforces exactly what it says:

    ```rust           compiled AND run — its assertions are real
    ```rust,no_run    compiled, not run (writes a file, needs a runtime, …)
    ```rust,ignore    not compiled — must carry a comment saying why

Scope is `docs/cookbook/` only. ADRs also hold ```rust blocks, but those are point-in-time
sketches of a decision as it was made (ADR-013's still names the retired `Subpaving`
layout); compiling them would force history to be rewritten whenever the API moves.

Run from the repo root: `python3 website/check-rust-blocks.py [page.md ...]`
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

COOKBOOK = Path("docs/cookbook")
EDITION = "2021"
RUST_FENCE = re.compile(r"^\s*```rust\b", re.MULTILINE)


def run(cmd: list, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def cargo_build() -> str:
    """Build the workspace with every feature on; return cargo's JSON message stream."""
    # --all-features: the async blocks need hurray-io's `tokio` feature, and a page should
    # never fail the check because a feature happened to be off.
    proc = run(
        ["cargo", "build", "--workspace", "--all-features", "--message-format=json"],
    )
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        sys.exit("error: `cargo build --workspace --all-features` failed")
    return proc.stdout


def extern_paths(build_json: str) -> dict:
    """Map crate name -> .rlib path, for the workspace crates and their direct deps.

    Paths come from cargo's own artifact messages rather than a glob over target/debug/deps:
    stale hashed rlibs accumulate there, and only cargo knows which one this build produced.

    The map is deliberately NOT every transitive rlib — a snippet must not compile against a
    crate the workspace does not itself depend on.
    """
    meta_proc = run(["cargo", "metadata", "--format-version", "1", "--no-deps"])
    if meta_proc.returncode != 0:
        print(meta_proc.stderr, file=sys.stderr)
        sys.exit("error: `cargo metadata` failed")
    meta = json.loads(meta_proc.stdout)

    wanted = set()
    for pkg in meta["packages"]:
        wanted.add(pkg["name"].replace("-", "_"))
        for dep in pkg["dependencies"]:
            if dep["kind"] is None:  # normal dependency (not dev-/build-only)
                wanted.add(dep["name"].replace("-", "_"))

    externs = {}
    for line in build_json.splitlines():
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("reason") != "compiler-artifact":
            continue
        name = msg["target"]["name"].replace("-", "_")
        if name not in wanted:
            continue
        for path in msg.get("filenames", []):
            if path.endswith(".rlib"):
                externs.setdefault(name, path)
    return externs


def check_page(page: Path, externs: dict) -> tuple:
    """rustdoc --test one page. Returns (ok, combined output)."""
    # --test-run-directory puts each page's compiled doctests in a throwaway CWD while
    # rustdoc itself still runs from the repo root, so failures keep their relative path.
    # Blocks that write a file ("model.hrry") would otherwise litter the repo, and — worse —
    # a leftover file made the block that *reads* it pass on a rerun, so a snippet that
    # depends on an earlier one looked self-contained when it was not.
    with tempfile.TemporaryDirectory(prefix="hurray-doc-") as scratch:
        cmd = ["rustdoc", "--test", str(page), "--edition", EDITION,
               "--test-run-directory", scratch,
               "-L", "dependency=target/debug/deps"]
        for name, path in sorted(externs.items()):
            cmd += ["--extern", f"{name}={path}"]
        proc = run(cmd)
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def main() -> int:
    if not COOKBOOK.is_dir():
        print(f"error: {COOKBOOK} not found (run from the repo root)", file=sys.stderr)
        return 2

    if sys.argv[1:]:
        pages = [Path(a) for a in sys.argv[1:]]
    else:
        pages = sorted(p for p in COOKBOOK.glob("*.md")
                       if RUST_FENCE.search(p.read_text(encoding="utf-8")))

    externs = extern_paths(cargo_build())
    missing = [c for c in ("hurray_core", "hurray_io", "hurray_ffi") if c not in externs]
    if missing:
        print(f"error: cargo produced no rlib for {', '.join(missing)}", file=sys.stderr)
        return 2

    failed = []
    for page in pages:
        ok, output = check_page(page, externs)
        status = "ok" if ok else "FAILED"
        summary = next((ln for ln in output.splitlines() if ln.startswith("test result:")), "")
        print(f"{status:6}  {page}  {summary}")
        if not ok:
            failed.append((page, output))

    if failed:
        for page, output in failed:
            print(f"\n{'=' * 70}\n{page}\n{'=' * 70}\n{output}")
        print(f"\n{len(failed)} of {len(pages)} cookbook page(s) have Rust blocks that "
              f"do not compile or fail at runtime.")
        return 1

    print(f"\nAll Rust blocks on {len(pages)} cookbook page(s) compile and pass. ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
