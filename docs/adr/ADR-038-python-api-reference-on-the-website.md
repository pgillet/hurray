# ADR-038: Publish the Python API reference on the docs website with pdoc

## Status

Accepted (2026-08-27).

Extends **ADR-028**, which put the Rust API reference (`cargo doc`) on the site but said
nothing about Python. Amends `docs/website/README.md` § 2, § 3, § 7, and § 9.

## Context

The site publishes `cargo doc` for every version at `/docs/<version>/api/`. Python gets
nothing. A reader who wants to know what `hurray.Tensor.buffer(i)` returns, or which
keyword arguments `hurray.save` takes, has three options: the cookbook pages (task-shaped,
not exhaustive), `docs/impl/python-bindings.md` (normative requirements, not an API
reference), or `help()` in a REPL after installing the wheel. For a binding whose stated
job is to expose everything the Rust layers can express (issue #147, closed 2026-08-26),
having no browsable API surface is the last hole of that work.

Three facts make this cheap to close.

**The content is already written.** Every `#[pyclass]`, `#[pymethods]` item, and
`#[pyfunction]` carries a `///` comment — CLAUDE.md requires one on every public item —
and PyO3 puts those verbatim into `__doc__`. They are Markdown, including the
`## Examples (Python)` blocks and the parameter tables. A generator that renders Markdown
docstrings gets the whole binding documented on day one.

**Signatures come for free.** PyO3 0.29 synthesises `__text_signature__` from the Rust
argument list, including keyword-only markers and defaults:

```text
>>> hurray.Tensor.__text_signature__
(buffer, dtype, shape, device=None, *, aux_buffers=None, layout=None,
 quantization=None, statistics=None, shard=None)
>>> hurray.arange.__text_signature__
(start, stop=None, step=None, *, dtype=None, device=None)
```

No `text_signature` attributes have to be added — there are none in the tree today, and
none are needed.

**The submodules survive introspection.** `hurray.dtype` and `hurray.device` are module
objects added as attributes rather than importable packages; a generator driven by
introspection rather than by source files documents them as pages of their own.

### What the obstacle actually is

`docs/website/README.md` § 2 says the build "MUST NOT require Node, npm, or any
package-manager network install beyond fetching the two pinned binaries and the Rust
toolchain already used by the workspace." Every Python documentation generator is a pip
install, so the rule as written forbids all of them. The rule exists to keep the supply
chain small and the pipeline free of a Node toolchain (ADR-028), not to forbid Python —
and CI already pip-installs `maturin`, `numpy`, `scipy`, and `pytest` in the
`python-conformance` job. The constraint needs to be stated as what it means.

There is a second, harder constraint that no amount of wording removes: **a Python doc
generator has to import the module**, and the module is a compiled extension. Generating
the reference for a version means building that version's wheel first. The docs pipeline
does not touch Python today.

### Two defects found while scoping this

- **`/docs/<version>/api/` is a 404.** `cargo doc --no-deps --workspace` on a
  multi-crate workspace writes `target/doc/<crate>/index.html` for each crate and no
  `target/doc/index.html`, so the directory the spec advertises has no landing page.
- **Nothing links to the API reference.** ADR-028 says it is "linked from the book's
  navigation"; no such link was ever added, to the book, the shell nav, or any page.

Both are in scope here: publishing a second API tree next to a broken, unreachable first
one would not be finishing the job.

## Decision

### 1. pdoc, pinned, at `/docs/<version>/python-api/`

`pdoc` generates the Python API reference. It is one pip package with no configuration
file, it renders Markdown docstrings (`--docformat markdown`), it works by importing the
module — so a compiled PyO3 extension is no special case — and it emits a self-contained
static tree with its own search index, which is exactly the shape the deploy already
handles for `cargo doc`.

Output goes to `/docs/<version>/python-api/`, alongside `/docs/<version>/api/`, so one
version prefix continues to scope everything about that version. `pdoc` writes an
`index.html` that redirects to `hurray.html`, so the directory URL resolves.

### 2. Every version, not just `dev` and `stable`

The Python reference is generated for each version the site builds, exactly as `cargo doc`
is, and skipped for a version whose tree has no `hurray-python` — the same shape as the
existing "no `website/book`, skip this version" rule.

Restricting it to `dev` plus the stable tag was considered and rejected. `stable` is a
*copy* of a tag's build, so producing it requires knowing which tag is stable before
building — an ordering dependency in the script that buys nothing today, when the repo has
zero release tags. Uniformity with `cargo doc` costs less than the special case. If tag
count ever makes this slow, the escape hatch is ADR-028's: incremental builds that carry
prior version outputs forward.

### 3. The version's own wheel, built from the version's own tree

`build-site.sh` builds each version's wheel with `maturin` from that version's source tree
and installs it before running `pdoc`. A version's API reference therefore describes that
tag's binding, which is the same immutability rule § 4 of the website spec already states
for books.

The wheel is a debug build: the docs are made from docstrings and signatures, and
`--release` would multiply the build time of every version for no change in output.

### 4. `build-site.sh` requires the Python tooling, and says so

The script already hard-requires `git`, `zola`, `mdbook`, and `cargo`; `python3` with
`maturin` and `pdoc` joins that list. Both documentation workflows install them at pinned
versions, the same way they install the two pinned binaries. The script does not silently
skip the Python reference when the tools are missing — a version that half-builds is worse
than one that fails, which is the rule the website spec already sets for tagged builds.

### 5. One "API Reference" page in the book, linking both

A new book page lists the Rust crate docs and the Python module docs with relative links
that resolve inside the version's own path. This is the navigation entry ADR-028 called for
and never got, and it is version-correct by construction: the `0.2.0` book links to
`0.2.0`'s API docs.

`build-site.sh` additionally emits `api/index.html` as a redirect to `hurray_core`, so the
advertised `/docs/<version>/api/` path resolves instead of 404ing.

### 6. Docstrings are user-facing text now

Publishing them makes every `///` comment on a Python-visible item part of the public
documentation. Two consequences apply immediately:

- The `#[pymodule]` doc comment is the landing page of the reference. It contained an
  internal "Module layout" table keyed by development phase (`8a.1`, `8a.2`, `8c`) and
  issue numbers. It is rewritten as a user-facing overview, in line with the standing rule
  that phase references do not appear in user-facing files.
- Rustdoc shortcut links (`` [`Descriptor::decode`] ``) render as literal brackets in a
  Python docstring. Where such a link appears on a Python-visible item it is rewritten as
  prose naming the Python attribute. Rust-internal items keep theirs.

## Alternatives Considered

- **Sphinx + `autodoc`.** The standard in the Python ecosystem, and what NumPy, SciPy, and
  PyTorch use; it also offers intersphinx cross-linking to NumPy's docs. Rejected: it
  brings `conf.py`, a theme dependency, a `docs/` source tree of `.rst` stubs listing every
  class, and a build step per version — a whole authoring surface for output we would
  immediately reduce back to "one page per module." The intersphinx benefit is small for a
  binding whose public API is thirty-odd classes in one flat namespace.

- **`mkdocstrings` (MkDocs).** Same objection as Sphinx plus a second site generator, when
  the project already runs two.

- **Hand-written Markdown API pages in the book.** No new tooling at all, and it would live
  next to the cookbook. Rejected: it duplicates the docstrings, and a duplicate drifts. The
  docstrings are the thing CI already forces to exist; the reference should be generated
  from them.

- **Generate `.pyi` stubs and document those.** Would also give type checkers something to
  read, which is real value. Rejected *here*: stubs are a separate deliverable with their
  own correctness question (they can disagree with the extension), and hanging the API
  reference off them makes the docs depend on that correctness. Stub generation stays an
  open idea on its own merits.

- **Link `help()` and publish nothing.** Rejected: it requires installing the wheel to read
  the docs, and it is not reachable from a search engine.

## Consequences

- **The docs build now compiles `hurray-python` once per version.** Cost is one debug
  `cargo build` per version on top of the existing per-version `cargo doc`. With zero
  release tags it is one build; it grows linearly with tags, and shares the workspace cache
  with the `cargo doc` step for the same tree.

- **The website spec's toolchain rule is narrower than it was.** § 2 now bans the Node/npm
  toolchain and unpinned installs specifically, rather than all package managers. This is
  the rule ADR-028 intended; the previous wording over-reached.

- **Two API doc UIs.** rustdoc and pdoc look nothing alike, and neither looks like mdBook.
  This is the same trade-off ADR-028 already accepted for mdBook plus Zola, at a place
  readers cross less often.

- **Docstrings are now published.** Every Python-visible `///` comment is user-facing text
  and should be written that way. `rust-developer` and `doc-updater` own this; the two
  cleanups in § 6 are the first application of it, not a one-off.

- **A reader can now find the API reference.** The book has a navigation entry for it, and
  `/docs/<version>/api/` resolves — neither was true before this ADR.
