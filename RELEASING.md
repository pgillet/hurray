# Releasing Hurray

Hurray ships as **one workspace version** — all crates and the Python package are released
together under a single `MAJOR.MINOR.PATCH` tag.

> **Pre-1.0:** breaking changes are allowed; bump the **minor** (`0.x` convention). The
> `1.x` compatibility contract begins at `1.0.0`. Published versions are **immutable** on
> crates.io and PyPI (you can only *yank*, never overwrite) — so double-check before you
> publish.

## What goes where

| Artifact | Registry | How |
|----------|----------|-----|
| `hurray-core`, `hurray-io`, `hurray-ffi`, `hurray-inspect` | [crates.io](https://crates.io) (source) | `cargo publish`, in dependency order |
| `hurray` (from `hurray-python`) | [PyPI](https://pypi.org) (wheels) | `maturin` |
| Documentation site `/docs/<tag>/` | GitHub Pages | automatic on tag (`docs.yml`) |

`conformance` is `publish = false` (internal tooling) and is never published.

**Crate dependency order** (publish parents before children):

```
hurray-core  →  hurray-io , hurray-ffi  →  hurray-inspect
```

## One-time setup

- **crates.io:** a maintainer account with an API token, or configure crates.io
  [Trusted Publishing](https://crates.io/docs/trusted-publishing) (GitHub Actions OIDC — no
  stored token).
- **PyPI:** create the `hurray` project and configure
  [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (recommended) or an API
  token.
- Install tooling: `cargo install cargo-release` and `pipx install maturin` (or
  `pip install maturin`).

## Release checklist

1. **Green `main`.** CI (fmt, clippy `--all-targets`, tests, Python conformance, docs
   checks) passes.
2. **Changelog.** Move items from `## [Unreleased]` into a new `## [X.Y.Z] - YYYY-MM-DD`
   section in [`CHANGELOG.md`](CHANGELOG.md).
3. **Version bump.** Update `version` in `[workspace.package]` **and** the `version` fields
   of the inter-crate deps in `[workspace.dependencies]` (`hurray-core`, `hurray-io`,
   `hurray-ffi`) to the same number. `cargo release` (below) does both.
4. **Tag.** Commit, then tag `X.Y.Z` (no leading `v`) and push the tag. Pushing the tag
   triggers the docs deploy, which builds `/docs/X.Y.Z/` from that tag and makes it
   `stable`.
5. **Publish crates** to crates.io in dependency order.
6. **Publish the Python wheel(s)** to PyPI.
7. **GitHub Release.** Create a release for the tag with the changelog section as notes.

## Commands

Rust crates (dry run first — `cargo release` bumps versions, updates the inter-crate deps,
commits, tags, and publishes in dependency order):

```sh
# Dry run — shows exactly what it would do, changes nothing:
cargo release minor        # or: patch / X.Y.Z

# Execute (bumps, tags, publishes core → io/ffi → inspect):
cargo release minor --execute
```

Python wheel to PyPI (start simple; add a multi-platform matrix later with
[`cibuildwheel`](https://cibuildwheel.pypa.io/) or `maturin-action` in CI):

```sh
maturin publish -m hurray-python/Cargo.toml
```

## Notes

- **Docs are automatic.** The tag push rebuilds the site: `/docs/X.Y.Z/` is generated from
  that tag's Markdown, the version dropdown updates, and `stable` resolves to the highest
  non-prerelease tag. No manual docs step.
- **Prereleases** (e.g. `0.2.0-rc.1`) publish as their own version but are never selected as
  `stable`.
- **If a publish is wrong**, you cannot delete it — `cargo yank --version X.Y.Z <crate>` and
  the PyPI *yank* hide it from new resolutions; then release a fixed version.
