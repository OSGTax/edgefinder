#!/bin/sh
# Publish the playable build to the gh-pages branch, which GitHub Pages serves at
# https://osgtax.github.io/edgefinder/ — usage: npm run deploy [-- "commit message"]
set -e
cd "$(dirname "$0")/.."
npm run build:pages
msg="${1:-Playable web build from $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)}"
git fetch -q origin gh-pages
tmp=$(mktemp -d)
git worktree add -q --detach "$tmp" origin/gh-pages
find "$tmp" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp -r docs/. "$tmp"/
touch "$tmp/.nojekyll"
git -C "$tmp" add -A
if git -C "$tmp" commit -q -m "$msg"; then git -C "$tmp" push -q origin HEAD:gh-pages; echo "Published."; else echo "Live site already up to date."; fi
git worktree remove --force "$tmp"
