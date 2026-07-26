# Hurray Documentation Website — Specification

> **Status:** Accepted. Decision recorded in
> [ADR-028](../adr/ADR-028-documentation-website.md). This document specifies the concrete
> structure of the website; it is infrastructure spec, not part of the normative format
> specification under `docs/spec/`.

> This document uses RFC 2119 key words: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT,
> SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

## 1. Scope and goals

The website is the public face of Hurray, modelled on `arrow.apache.org`: a mostly-technical
site for reference-implementation users, format implementers, and ML/inference engineers.

Goals, in priority order:

1. **Faithful, full-history versioned docs.** Every released spec version is browsable,
   built from its own git tag.
2. **Simple, fully automated CI pipeline** with no Node/npm toolchain.
3. **Easy to reorganize and extend** — content is Markdown authored in place.
4. **Portable** — a plain static tree deployable to GitHub Pages today and any static host
   later.

## 2. Toolchain

| Concern | Tool | Notes |
|---------|------|-------|
| Versioned technical book | **mdBook** | Spec + impl + cookbook + tutorials. Single Rust binary. Built-in per-book search. |
| Outer site shell | **Zola** | Landing, FAQ, blog, community. Single Rust binary. |
| Rust API reference | **`cargo doc`** | Per version, per crate; published under the version path. |
| CI / deploy | **GitHub Actions → GitHub Pages** | Full static-tree deploy. |

Both generators are prebuilt binaries pinned to explicit versions in the workflow. The
build MUST NOT require Node, npm, or any package-manager network install beyond fetching
the two pinned binaries and the Rust toolchain already used by the workspace.

## 3. Deployed URL scheme

The site deploys as a single static tree. Paths (relative to the Pages site root):

```
/                         Landing / overview            (Zola)
/faq/                     FAQ                            (Zola)
/blog/                    Blog index + posts            (Zola)
/community/               Contributing, CoC, governance, mailing lists  (Zola)
/docs/                    → redirects to /docs/stable/
/docs/stable/             Latest stable release book     (mdBook; copy of the stable tag build)
/docs/dev/                Book built from `main`         (mdBook)
/docs/<version>/          Book built from tag <version>  (mdBook)   e.g. /docs/0.1.0/
/docs/<version>/api/      cargo doc for that version     (rustdoc)
/docs/stable/api/         cargo doc for the stable release
/docs/dev/api/            cargo doc for `main`
/versions.json            Version manifest (drives the dropdown)
```

- Version path segments MUST be the exact git tag name. The tag convention is
  `MAJOR.MINOR.PATCH` with **no leading `v`** (e.g. `0.1.0`); release tags MUST follow it.
- The API reference for a version MUST live under that version's `api/` subpath so a single
  version prefix scopes both the book and its API docs.
- `/docs/` MUST redirect to `/docs/stable/` (an emitted `index.html` meta-refresh is
  acceptable, since GitHub Pages does not honour symlinks).

## 4. Versioning policy

- A **version** is a spec version. Release tags follow the spec version. The set of
  published versions is exactly the set of matching git tags, plus the special `dev`
  (built from `main`).
- **`stable`** is the highest non-prerelease semantic-version tag. Its build MUST be copied
  to `/docs/stable/` (a copy, not a symlink) on each deploy.
- **Default entry point.** The site navigation "Docs" link and `/docs/` MUST resolve to
  `/docs/stable/`. The `dev` version MUST be reachable and MUST be clearly labelled as
  unreleased in the version dropdown and via an in-page banner on every `dev` page.
- **Bootstrap fallback.** Until the first release tag exists, there is no stable build;
  `/docs/`, `/docs/stable/`, and the "Docs" nav link MUST fall back to `/docs/dev/`. The
  workflow MUST detect "no release tags" and emit this fallback rather than a broken link.
- **Prerelease tags** (e.g. `0.2.0-rc.1`) MAY be published as their own version entry but
  MUST NOT be selected as `stable`.
- **Immutability.** A published version path, once deployed for a given tag, MUST reflect
  that tag's content; version builds MUST come from `git checkout <tag>`, never from `main`.

### 4.1 `versions.json` schema

A single manifest at the site root drives the version dropdown. It is regenerated on every
deploy. Shape:

```json
{
  "stable": "0.1.0",
  "dev": "dev",
  "versions": [
    { "id": "dev",   "label": "dev (unreleased)", "path": "/docs/dev/",   "released": null,         "prerelease": false, "dev": true },
    { "id": "0.1.0", "label": "0.1.0",            "path": "/docs/0.1.0/", "released": "2026-08-01", "prerelease": false, "dev": false }
  ]
}
```

- `versions` MUST be ordered newest-first, with `dev` first.
- `stable` MUST name the `id` of the stable version, or be `null` before the first release.
- Consumers (the dropdown script) MUST treat an absent/`null` `stable` by pointing at `dev`.

## 5. Repository layout

New site sources live under `website/`; published content stays authored in place under
`docs/`.

```
website/
├── book/
│   └── book.toml            # mdBook config; src points at the curated doc tree
├── site/                    # Zola project
│   ├── config.toml
│   ├── content/
│   │   ├── _index.md        # landing
│   │   ├── faq/
│   │   ├── blog/
│   │   └── community/
│   ├── templates/
│   ├── sass/ or static/
│   └── static/
└── theme/                   # shared tokens (colors, fonts) used to keep book + site coherent

docs/
├── SUMMARY.md               # mdBook table of contents — the single reorganization surface
├── spec/                    # (existing) format specification
├── impl/                    # (existing) implementation requirements
├── cookbook/                # (existing) cookbook entries
├── tutorials/               # (new) longer-form guided tutorials
├── adr/                     # (existing) architecture decision records — published as appendix
└── prior-art.md             # (existing) prior-art survey — published as appendix
```

- The book is a **view** over `docs/`. `book.toml` sets `src` to the doc tree and the book's
  navigation is defined solely by `docs/SUMMARY.md`. Reorganizing the book is editing
  `SUMMARY.md`; adding a page is adding a Markdown file and one `SUMMARY.md` line.
- The ADRs (`docs/adr/`) and `docs/prior-art.md` MUST be published as a book appendix in v1,
  listed under an "Appendix" section in `SUMMARY.md`.
- The Zola shell MUST NOT duplicate versioned technical content; it links into `/docs/`.

## 6. Visual coherence

- The mdBook book and the Zola shell are themed independently (see ADR-028 Consequences).
- A shared set of design tokens (brand colors, typography, logo) under `website/theme/`
  SHOULD be applied to both so navigation between shell and book feels like one site.
- Both themes MUST support light and dark modes and MUST be responsive.

## 7. CI/CD pipeline

A GitHub Actions workflow builds and deploys the entire tree. It is stateless: it
reconstructs all versions from git on every run.

**Triggers:** push to `main`, push of a release tag (semver `MAJOR.MINOR.PATCH[-prerelease]`,
no leading `v`), and manual dispatch.

**Stages (in order):**

1. **Checkout** with full history and tags (`fetch-depth: 0`).
2. **Install pinned tools:** the Rust toolchain, `mdbook` (pinned version), `zola` (pinned
   version).
3. **Build the shell:** `zola build` from `website/site/` into the output root
   (`public/`).
4. **Build `dev`:** from the current `main` tree, `mdbook build` → `public/docs/dev/`, then
   `cargo doc --no-deps --workspace` → `public/docs/dev/api/`. Stamp the "unreleased" banner.
5. **Build each release version:** for every release tag (semver, no leading `v`), in a
   detached worktree at that tag, `mdbook build` → `public/docs/<tag>/` and `cargo doc` →
   `public/docs/<tag>/api/`.
6. **Resolve stable:** compute the highest non-prerelease tag; copy its build to
   `public/docs/stable/`. If no release tag exists, make `stable` fall back to `dev`.
7. **Emit** `public/versions.json` and the `public/docs/` redirect to `stable`.
8. **Deploy** `public/` to GitHub Pages (`actions/deploy-pages`).

The workflow MUST fail the build (not silently skip) if a tagged version fails to build, so
history stays trustworthy. Pull-request builds SHOULD build the shell + `dev` only (no full
history) for fast preview.

> **Note (non-normative):** Stage 5 is O(number of release tags). This is acceptable at
> current scale (zero tags today). When it becomes slow, switch to incremental builds that
> carry prior version outputs forward and rebuild only new/changed tags — see ADR-028.

## 8. Search

- mdBook's built-in search is enabled per book, giving per-version search for free.
- Cross-version and whole-site search are out of scope for v1 (ADR-028).

## 9. Content model

| Section | Source | Owner |
|---------|--------|-------|
| Landing / overview | `website/site/content/_index.md` | core |
| FAQ | `website/site/content/faq/` | core |
| Blog | `website/site/content/blog/` | core contributors only |
| Community (contributing, CoC, governance, mailing lists) | `website/site/content/community/` | core |
| Spec / impl / cookbook / tutorials book | `docs/` via `docs/SUMMARY.md` | per existing agent ownership |
| Appendix: ADRs + prior-art | `docs/adr/`, `docs/prior-art.md` via `docs/SUMMARY.md` | per existing agent ownership |
| API reference | `cargo doc` output | generated |

## 10. Open questions

> **[OQ-1]:** *Resolved.* Tag naming is `MAJOR.MINOR.PATCH` with no leading `v` (e.g.
> `0.1.0`). See §4.

> **[OQ-2]:** *Resolved.* ADRs and `prior-art.md` are published as a book appendix in v1.
> See §5, §9.

> **[OQ-3]:** Branding — logo, color palette, and typography for the shared theme tokens.
> Deferred; to be decided before the theme is built.
