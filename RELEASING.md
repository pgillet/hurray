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
| `hurray` | [crates.io](https://crates.io) (source) | `cargo publish`, any time — it depends on nothing |
| `pyhurray` (from `hurray-python`) | [PyPI](https://pypi.org) (wheels) | `maturin` |
| Documentation site `/docs/<tag>/` | GitHub Pages | `docs.yml`, **dispatched on `main`** after tagging |

`conformance` and `hurray-python` are `publish = false` and never go to crates.io — the
first is internal tooling, the second ships to PyPI as `pyhurray`.

`hurray` is the umbrella name holding the namespace next to the crates that do the work.
It contains no code and depends on nothing, so it has no place in the dependency order —
but it **does** share the workspace version, so it is released along with everything else.

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

- **crates.io:** a maintainer account with a scoped API token — see below.
- **PyPI:** an account with an API token. There is nothing to create in advance — **a
  distribution name is claimed by its first upload, not by registering**. Registering an
  account does not hold `pyhurray`, and a PyPI *pending publisher* explicitly does not
  either. Scope the token to the `pyhurray` project once it exists; the very first upload
  has to use an account-wide token, which is the other reason to revoke afterwards.
- Install tooling: `cargo install cargo-release`, and **twine ≥ 6.1** to upload the
  distributions. maturin is *not* needed here — CI builds the wheels (see below), so the
  release machine only uploads them.

  The twine floor is not cosmetic: maturin writes `Metadata-Version: 2.4`, and twine 5.x
  rejects it as *"missing required fields: Name, Version"* — a parser limit wearing the
  costume of a broken package.

  On a distribution that marks its Python as externally managed (PEP 668), a plain
  `pip install twine` refuses with `error: externally-managed-environment`. Use a tool
  installer or a virtualenv rather than `--break-system-packages`:

  ```sh
  pipx install 'twine>=6.1'          # or: uv tool install 'twine>=6.1'
  # or, with no pipx: python3 -m venv ~/.venvs/release
  #                   ~/.venvs/release/bin/pip install 'twine>=6.1'
  #                   then call ~/.venvs/release/bin/twine
  ```

### The crates.io token

Create it at [crates.io/settings/tokens](https://crates.io/settings/tokens). A token is a
bearer credential: anything holding it can act as you, within its scopes, until it
expires. Give it the least that still completes a release.

| Field | Value | Why |
|---|---|---|
| **Endpoint scopes** | `publish-new`, `publish-update` | `publish-new` creates crates that do not exist yet — `publish-update` cannot, so a first release needs both. `publish-update` alone would make the token useless for a new crate; omitting it would make the token single-use. |
| | *not* `yank` | Yanking is rare, recoverable and doable from the web UI. A publish token that can also retract published versions is strictly worse. |
| | *not* `change-owners` | Ownership is the one thing a leaked token should never be able to change. |
| | *never* `legacy` | Every endpoint, which is the opposite of scoping. |
| **Crate scope** | `hurray-*` **and** `hurray` | Patterns glob only with a trailing `*`, so `hurray-*` does **not** match the bare name `hurray`. Scopes cover present *and future* crates matching them, so this keeps working for crates not yet written. |
| **Expiry** | 30 days | Short enough to bound a leak, long enough to finish a release and a follow-up patch. |

Then `cargo login` and paste it. **Revoke it when the release is done** rather than waiting
for the expiry: the token lives in `~/.cargo/credentials.toml` in plain text and in your
shell history if you passed it as an argument, and there is no reason for a publish
credential to outlive the publish. Revoking does not affect anything already published.

> **The long-term fix is not a better token.**
> [Trusted Publishing](https://crates.io/docs/trusted-publishing) replaces the stored
> credential with a short-lived OIDC exchange from a GitHub Actions job. It is unavailable
> for a crate's *first* release — it can only be configured on a crate that already exists
> — and it authenticates a workflow, not a person, so it only becomes relevant if
> publishing ever moves into CI. Until then, scoped and revoked is the standard to hold.

## Release checklist

1. **Green `main`.** CI (fmt, clippy `--all-targets`, tests, Python conformance, docs
   checks) passes.
2. **Changelog.** Move items from `## [Unreleased]` into a new `## [X.Y.Z] - YYYY-MM-DD`
   section in [`CHANGELOG.md`](CHANGELOG.md).
3. **Version bump.** Update `version` in `[workspace.package]` **and** the `version` fields
   of the inter-crate deps in `[workspace.dependencies]` (`hurray-core`, `hurray-io`,
   `hurray-ffi`) to the same number. `cargo release` (below) does both.
4. **Tag.** Commit, then tag `X.Y.Z` (no leading `v`) and push the tag. The tag push starts
   a docs run that builds `/docs/X.Y.Z/` — but **does not publish it**; see step 5.
5. **Publish the docs.** Dispatch `docs.yml` **on `main`**, and check that
   `/docs/X.Y.Z/` serves and `versions.json` reports `"stable": "X.Y.Z"`:

   ```sh
   gh workflow run docs.yml --ref main
   # then, once it is green:
   curl -s https://www.pascalgillet.net/hurray/versions.json | head -4
   ```

   A docs deploy created from a **tag** ref is recorded and reports success, and is then
   never served — only a run on the default branch is promoted to the live site
   ([#268](https://github.com/pgillet/hurray/issues/268)). The tag run going green is not
   evidence that anything published. Dispatching on `main` is what publishes, and it
   builds the new version correctly because `build-site.sh` enumerates every tag in the
   repository rather than only the ref it was called on.
6. **Publish crates** to crates.io in dependency order.
7. **Build the Python distributions** — run the **Build Python wheels** workflow against
   the tag, then download the `pyhurray-dist` artifact and **upload it yourself**.
8. **GitHub Release.** Create a release for the tag with the changelog section as notes.

## Commands

Rust crates. Two routes, and they are **alternatives, not steps** — `cargo release` does
the bump, the commit, the tag *and* the crate publish as one act, so it replaces checklist
steps 3, 4 and 6 rather than following them. It does **not** do step 5: the docs still have
to be published by a `main` dispatch, and `cargo release` finishing is not evidence that
they were. Pick one route before you start:

**A — `cargo release`** (it has not yet done a release for this project):

```sh
# Dry run — shows exactly what it would do, changes nothing:
cargo release minor        # or: patch / X.Y.Z

# Execute (bumps, tags, publishes core → io/ffi → inspect):
cargo release minor --execute
```

**B — the tag already exists**, because you bumped and tagged by hand. Do **not** then run
`cargo release`: it would bump *again*, to the next version. Publish each crate in
dependency order, waiting for each to appear in the index before the crate that depends on
it — `cargo publish` blocks on this by default, but a `--dry-run` of a downstream crate
fails until its dependency is actually live:

```sh
cargo publish -p hurray-core
cargo publish -p hurray-io
cargo publish -p hurray-ffi
cargo publish -p hurray-inspect
cargo publish -p hurray          # no dependencies; order does not matter
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

- **Docs need one manual step.** `/docs/X.Y.Z/` is generated from that tag's Markdown, the
  version dropdown updates, and `stable` resolves to the highest non-prerelease tag — but
  the run that *publishes* all of that has to be dispatched on `main` (step 5). A docs run
  triggered by the tag push builds the version and silently fails to serve it
  ([#268](https://github.com/pgillet/hurray/issues/268)).

  The failure mode is worth recognising, because it hides: the tag's run is **green**, and
  the missing version appears on its own at some arbitrary later time, when the next
  ordinary push to `main` republishes the site with the tag already in place. So a release
  whose docs step was skipped looks fine in the Actions tab and wrong on the website, and
  then quietly corrects itself days later.
- **Prereleases** (e.g. `0.2.0-rc.1`) publish as their own version but are never selected as
  `stable`.
- **If a publish is wrong**, you cannot delete it — `cargo yank --version X.Y.Z <crate>` and
  the PyPI *yank* hide it from new resolutions; then release a fixed version.
