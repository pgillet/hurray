#!/usr/bin/env bash
#
# Build the full Hurray documentation site into website/public/ (ADR-028).
#
# Layout produced:
#   public/                     Zola shell (landing, FAQ, blog, community)
#   public/docs/dev/            book built from the current tree (main)
#   public/docs/dev/api/        cargo doc for the current tree
#   public/docs/<tag>/[api/]    book + cargo doc built from each release tag
#   public/docs/stable/         copy of the highest non-prerelease tag (or a redirect to
#                               dev before the first release exists)
#   public/docs/index.html      redirect to stable
#   public/versions.json        manifest that drives the in-book version dropdown
#
# The pipeline is stateless: every version is rebuilt from git on each run, so main never
# carries frozen doc snapshots and each version renders faithfully from its own tag.
#
# Environment:
#   BASE_URL   optional; passed to `zola build --base-url` (CI sets it from the Pages URL).
#   DEV_ONLY   if non-empty, skip release-tag builds (fast PR preview builds).
#
# Requires: git, zola, mdbook, cargo. Run from anywhere inside the repo.

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
OUT="$ROOT/website/public"
BASE_URL="${BASE_URL:-}"
DEV_ONLY="${DEV_ONLY:-}"

# GitHub Pages can report an http:// base_url for a custom domain even though the site is
# served over https. Zola bakes base_url into every absolute asset URL, so an http base_url
# yields http:// CSS/JS/logo links that an https page blocks as mixed content (the shell then
# renders completely unstyled). Force https so assets load. (mdBook uses relative links and is
# unaffected either way.)
if [ -n "$BASE_URL" ]; then
  BASE_URL="${BASE_URL/#http:/https:}"
fi

# Release tags follow the spec version: MAJOR.MINOR.PATCH with an optional -prerelease
# suffix, no leading "v" (ADR-028, OQ-1).
TAG_RE='^[0-9]+\.[0-9]+\.[0-9]+([-+].*)?$'

echo "==> Cleaning $OUT"
rm -rf "$OUT"
mkdir -p "$OUT/docs"

# --- 1. Outer shell (Zola) -------------------------------------------------------------
echo "==> Building shell (zola)"
if [ -n "$BASE_URL" ]; then
  ( cd "$ROOT/website/site" && zola build --base-url "$BASE_URL" --output-dir "$OUT" --force )
else
  ( cd "$ROOT/website/site" && zola build --output-dir "$OUT" --force )
fi

# --- helper: build one version's book + API docs from a source tree ---------------------
# $1 = source directory (a git worktree or the repo root); $2 = version id (dir name).
build_version() {
  local srcdir="$1" name="$2" dest="$OUT/docs/$2"
  if [ ! -f "$srcdir/website/book/book.toml" ]; then
    echo "==> Skipping '$name': no website/book (predates the docs site)"
    return 0
  fi
  echo "==> Building docs version '$name'"
  mkdir -p "$dest"
  ( cd "$srcdir/website/book" && mdbook build --dest-dir "$dest" )
  ( cd "$srcdir" && cargo doc --no-deps --workspace --quiet )
  if [ -d "$srcdir/target/doc" ]; then
    mkdir -p "$dest/api"
    cp -R "$srcdir/target/doc/." "$dest/api/"
  fi
}

# --- 2. dev (current tree) -------------------------------------------------------------
build_version "$ROOT" "dev"

# --- 3. each release tag (from a detached worktree) ------------------------------------
tags=()
while IFS= read -r t; do [ -n "$t" ] && tags+=("$t"); done < <(git -C "$ROOT" tag | grep -E "$TAG_RE" || true)

if [ -z "$DEV_ONLY" ] && [ "${#tags[@]}" -gt 0 ]; then
  for tag in "${tags[@]}"; do
    wt="$(mktemp -d)"
    git -C "$ROOT" worktree add --detach "$wt" "refs/tags/$tag" >/dev/null
    build_version "$wt" "$tag"
    git -C "$ROOT" worktree remove --force "$wt"
  done
fi

# --- 4. resolve stable + redirects -----------------------------------------------------
stable_id=""
releases=()
while IFS= read -r t; do [ -n "$t" ] && releases+=("$t"); done \
  < <(printf '%s\n' "${tags[@]:-}" | grep -Ev '[-+]' | sort -V || true)
if [ "${#releases[@]}" -gt 0 ]; then
  candidate="${releases[${#releases[@]}-1]}"
  if [ -d "$OUT/docs/$candidate" ]; then stable_id="$candidate"; fi
fi

if [ -n "$stable_id" ]; then
  echo "==> stable = $stable_id"
  rm -rf "$OUT/docs/stable"
  cp -R "$OUT/docs/$stable_id" "$OUT/docs/stable"
else
  echo "==> No release tag yet: stable falls back to dev"
  mkdir -p "$OUT/docs/stable"
  printf '<!doctype html><meta http-equiv="refresh" content="0; url=../dev/">\n' > "$OUT/docs/stable/index.html"
fi
printf '<!doctype html><meta http-equiv="refresh" content="0; url=stable/">\n' > "$OUT/docs/index.html"

# --- 5. versions.json ------------------------------------------------------------------
echo "==> Writing versions.json"
{
  echo '{'
  if [ -n "$stable_id" ]; then echo "  \"stable\": \"$stable_id\","; else echo '  "stable": null,'; fi
  echo '  "dev": "dev",'
  echo '  "versions": ['
  echo '    { "id": "dev", "label": "dev (unreleased)", "path": "/docs/dev/", "released": null, "prerelease": false, "dev": true }'
  if [ "${#tags[@]}" -gt 0 ]; then
    while IFS= read -r tag; do
      [ -n "$tag" ] || continue
      [ -d "$OUT/docs/$tag" ] || continue
      date="$(git -C "$ROOT" log -1 --format=%as "refs/tags/$tag" 2>/dev/null || true)"
      if printf '%s' "$tag" | grep -q '[-+]'; then pre=true; else pre=false; fi
      if [ -n "$date" ]; then rel="\"$date\""; else rel=null; fi
      echo "    ,{ \"id\": \"$tag\", \"label\": \"$tag\", \"path\": \"/docs/$tag/\", \"released\": $rel, \"prerelease\": $pre, \"dev\": false }"
    done < <(printf '%s\n' "${tags[@]}" | sort -Vr)
  fi
  echo '  ]'
  echo '}'
} > "$OUT/versions.json"

echo "==> Done. Site in $OUT"
