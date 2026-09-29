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
| `pyhurray` (from `hurray-python`) | [PyPI](https://pypi.org) (wheels) | `maturin` |
| Documentation site `/docs/<tag>/` | GitHub Pages | automatic on tag (`docs.yml`) |

`conformance` is `publish = false` (internal tooling) and is never published.

The Python distribution is `pyhurray` (the name `hurray` is taken on PyPI); the module it
installs is imported as `hurray`.

**Crate dependency order** (publish parents before children):

```
hurray-core  →  hurray-io , hurray-ffi  →  hurray-inspect
```

## One-time setup

Publishing is a **human act** — nothing here uploads on a tag push. Both registries
therefore need an **API token**; [Trusted
Publishing](https://docs.pypi.org/trusted-publishers/) does not apply, because it
authenticates a GitHub Actions job rather than a person. (crates.io could not use it for a
first release anyway: it can only be configured on a crate that already exists.)

- **crates.io:** a maintainer account with an API token (`cargo login`).
- **PyPI:** an account with an API token. There is nothing to create in advance — **a
  distribution name is claimed by its first upload, not by registering**. Registering an
  account does not hold `pyhurray`, and a PyPI *pending publisher* explicitly does not
  either.
- Install tooling: `cargo install cargo-release` and `pipx install maturin` (or
  `pip install maturin`), plus `pipx install twine` to upload the wheels.

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
6. **Build the Python distributions** — run the **Build Python wheels** workflow against
   the tag, then download the `pyhurray-dist` artifact and **upload it yourself**.
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

Python distributions to PyPI. The build runs in CI and the upload does not: `pyproject.toml`
builds `abi3-py310`, which is one wheel *per platform*, and one machine cannot
cross-compile macOS and Windows — while
[`python-bindings.md` §Packaging](docs/impl/python-bindings.md) requires wheels for Linux
(x86_64, aarch64), macOS (x86_64, arm64) and Windows (x86_64), and forbids needing a Rust
toolchain at install time. So CI builds the artifacts and a human ships them:

```sh
# 1. Actions tab → "Build Python wheels" → Run workflow → pick the tag.
#    (Or: gh workflow run build-wheels.yml --ref X.Y.Z)
# 2. Download the single bundled artifact once the run is green:
gh run download --name pyhurray-dist --dir dist

# 3. Check what you are about to make permanent, then upload it:
twine check dist/*
twine upload dist/*
```

The workflow only ever *builds* — it has no upload step and no registry credentials.

## Notes

- **Docs are automatic.** The tag push rebuilds the site: `/docs/X.Y.Z/` is generated from
  that tag's Markdown, the version dropdown updates, and `stable` resolves to the highest
  non-prerelease tag. No manual docs step.
- **Prereleases** (e.g. `0.2.0-rc.1`) publish as their own version but are never selected as
  `stable`.
- **If a publish is wrong**, you cannot delete it — `cargo yank --version X.Y.Z <crate>` and
  the PyPI *yank* hide it from new resolutions; then release a fixed version.
