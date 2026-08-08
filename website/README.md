# Hurray documentation website

Sources for the public documentation site. Design and structure are specified in
[`docs/website/README.md`](../docs/website/README.md) (ADR-028).

- `book/` — [mdBook](https://rust-lang.github.io/mdBook/) config and theme. The book is a
  view over the Markdown under `docs/`, driven by `docs/SUMMARY.md`.
- `site/` — [Zola](https://www.getzola.org/) project for the outer shell (landing, FAQ,
  blog, community).
- `build-site.sh` — reconstructs the full deployable tree into `website/public/`.

## Prerequisites

- `mdbook` (0.4.x), `zola` (0.19.x), and a Rust toolchain (`cargo`, for the API reference).

## Build the whole site

```sh
bash website/build-site.sh          # full tree, all release tags, into website/public/
DEV_ONLY=1 bash website/build-site.sh   # fast: shell + dev book only (skips tags)
```

Open `website/public/index.html`, or serve the directory to follow the version dropdown and
cross-links between the shell and the books.

## Work on one piece

```sh
cd website/book && mdbook serve      # live-reload the technical book
cd website/site && zola serve        # live-reload the outer shell
```

Build output (`book/book/`, `site/public/`, `public/`) is git-ignored.

## Canonical URLs (single sources)

If the project moves (e.g. to a GitHub organization) or the site's canonical address
changes, update these — each is the single source for its scope. Nothing else needs to
change except prose links in READMEs / CHANGELOG (listed last).

**Site URL** (`https://www.pascalgillet.net/hurray/`)
- `Cargo.toml` → `[workspace.package].homepage` — inherited by every published crate.
- `website/site/config.toml` → `base_url` — local/fallback only; the deploy overrides it
  from the GitHub Pages URL (`pages.outputs.base_url`, coerced to https in
  `website/build-site.sh`), so the deployed site follows Pages automatically.

**Repository URL** (`https://github.com/pgillet/hurray`)
- `Cargo.toml` → `[workspace.package].repository` — inherited by every published crate.
- `website/site/config.toml` → `[extra].github_url` — the shell's GitHub links.
- `website/book/book.toml` → `git-repository-url` and `edit-url-template` — the book's
  repo + "edit this page" links.
- `hurray-ffi/cbindgen.toml` → generated C header banner.

**Prose links** (not parameterizable — plain Markdown): the crate `README.md` files,
root `README.md`, `CHANGELOG.md`, and `CLAUDE.md` reference the repo/site URLs in text.
GitHub auto-redirects the old repo path after a transfer, so these keep working; update
them at leisure.

> Note: GitHub Pages serves a custom domain only for the account that owns it. Moving the
> repo to an org drops it from `www.pascalgillet.net/hurray` (that domain belongs to the
> user site) back to `<org>.github.io/hurray` unless a domain/subdomain is attached to the
> org's Pages.
