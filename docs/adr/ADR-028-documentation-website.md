# ADR-028: Documentation Website — mdBook + Zola, Per-Tag Versioning, GitHub Pages

## Status

Accepted.

## Context

Hurray needs a public HTML site, in the mould of `arrow.apache.org`: a mostly-technical
site targeting reference-implementation users, format implementers, and ML/inference
engineers. It must carry two kinds of content:

1. **Versioned technical documentation** — the format specification (`docs/spec/`),
   implementation requirements (`docs/impl/`), cookbook (`docs/cookbook/`), tutorials,
   and the Rust API reference (`cargo doc`) for the `hurray-*` crates.
2. **An unversioned outer site** — landing/overview, FAQ, a blog (posts by core
   contributors), and a community section (contributing guidelines, code of conduct,
   governance, mailing lists).

Confirmed requirements from the project owner:

- **Versioning is by spec version, which git tags/releases follow, and the full history
  must be browsable** — not just the two or three most recent versions.
- **Fully automated in CI, and simple.** Content must be easy to reorganize and extend.
- **GitHub Pages** initially; a custom domain will come later (none is owned yet); the
  host may change — the pipeline must stay portable.
- **Default landing** on the latest stable release, with the unreleased development
  version reachable but clearly marked.
- **Per-version search** is sufficient for v1.
- **The Rust API reference is in scope** and published alongside each version.

The project is a Rust workspace with a strong "single binary, no supply chain" ethos.
At the time of writing there are **zero release tags** and the spec is `0.1.0-draft`.

## Decision

Build the site from **two Rust single-binary generators**, deployed together to GitHub
Pages as one static tree:

- **mdBook** renders the versioned technical book (spec + impl + cookbook + tutorials).
  The book is a *view* over the existing in-repo Markdown (authored in place under
  `docs/`), curated by a maintained `SUMMARY.md`, so reorganizing content stays a
  Markdown-and-table-of-contents edit.
- **Zola** renders the unversioned outer shell (landing, FAQ, blog, community). Markdown
  plus a folder-per-section layout keeps adding content trivial.

**Versioning strategy — build per git tag.** CI reconstructs the full multi-version tree
on every deploy: for each release tag it checks out that tag and builds its book and API
reference into a version-scoped path; `main` is additionally built as the `dev` version.
A generated `versions.json` manifest drives a version-selector dropdown injected into the
book. This maps one-to-one onto "spec versions which git tags follow, full history," keeps
`main` free of frozen doc snapshots, and renders every version faithfully from its own tag.

**Rust API reference** is produced by `cargo doc` per version and published under that
version's path, linked from the book's navigation.

**Hosting** is GitHub Pages via GitHub Actions. Because the generators emit a plain static
tree and nothing depends on Pages-specific features, the same artifact deploys to any
static host later.

Concrete URL scheme, directory layout, CI pipeline stages, the `versions.json` schema, and
the content model are specified in [`docs/website/README`](../website/).

## Alternatives Considered

- **Docusaurus (single tool for everything).** Docs + native versioning + blog + search +
  landing under one theme is the most turnkey option and was the strongest single-tool
  candidate. Rejected as the default for two reasons: (1) it requires a Node/npm toolchain,
  against the project's single-binary, minimal-supply-chain ethos; (2) its versioning
  *snapshots* docs into the main branch (`versioned_docs/`), which bloats `main` as history
  grows and does **not** build from tags — a poor fit for "versions are git tags, full
  history." It remains the fallback if a single unified theme ever outweighs tag-faithful
  history.

- **Starlight (Astro).** Similar unified-theme appeal, but still a Node toolchain, and its
  versioning is a third-party plugin that is also snapshot-based. No advantage over
  Docusaurus for our constraints.

- **mdBook only, minimal (defer blog/community/FAQ).** Simplest, but does not deliver the
  Arrow-style outer site the owner asked for. Rejected as under-scoped.

- **Zola only (book as a Zola section).** One tool, single binary, but re-implements the
  book's sidebar/nav/search UX that mdBook gives for free and would require fully custom
  versioning. More work for a worse book experience.

- **Snapshot-in-repo versioning (regardless of tool).** Copy current docs into a
  per-version folder committed to `main` on each release. Rejected: bloats `main`, makes
  historical edits awkward, and is less faithful than rebuilding from the tag itself.

- **Incremental version builds (only build the new/changed version each deploy).** Faster
  than full rebuild-from-tags as history grows, but stateful (must carry prior builds
  forward). Deferred as a future optimization; v1 rebuilds all versions each deploy for a
  stateless, simpler pipeline.

## Consequences

- **Two themes to keep visually consistent.** The mdBook book and the Zola shell are themed
  independently; a modest, ongoing investment is needed to keep them coherent. Apache Arrow
  accepts the same split (Jekyll site + Sphinx docs); this is a known, tolerable trade-off.

- **Build time grows with tag count.** Rebuilding every tag's book and `cargo doc` on each
  deploy is O(number of releases). Acceptable at current scale; the incremental-build
  optimization above is the escape hatch when it hurts.

- **No stable version until the first tag.** With zero release tags today, only `dev`
  exists initially; `/docs/stable/` and the default "Docs" entry point MUST fall back to
  `dev` until the first release tag is cut. The spec defines this fallback.

- **Search is per-version.** mdBook's built-in search covers a single book; there is no
  cross-version or whole-site search in v1. Revisit if users need it.

- **A new authoring surface appears:** `docs/tutorials/` and a maintained book `SUMMARY.md`.
  The ADRs (`docs/adr/`) and `docs/prior-art.md` are published as a book appendix in v1.
  Release tags use `MAJOR.MINOR.PATCH` with no leading `v`. The doc-site spec owns these
  conventions.

- **New CI responsibility.** A GitHub Actions workflow builds and deploys the whole tree;
  it needs full git history/tags (`fetch-depth: 0`) and Pages deploy permissions.

- **Relationship to the array-database vision.** A browsable full-history spec site is
  directly useful to the long-term versioned-data direction; nothing here forecloses it.
