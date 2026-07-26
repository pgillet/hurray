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
