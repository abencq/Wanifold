#!/usr/bin/env bash
# scripts/sync-upstream.sh -- bring upstream's main into this branch, and carry the pages with it.
#
#   git sync-upstream            fetch origin, merge origin/main, refresh the pages
#   git sync-upstream --dry-run  fetch and report what the merge would do, touching nothing
#
# A sync has three things to settle, and this script settles them in this order:
#
#   1. the code. Merged by git and by nobody else. Any conflict in a real file stops the script with the
#      list of files and the merge left open: the script never guesses what a person meant to keep.
#   2. README.md, this fork's generated front page, and the one file upstream rewrites about fifty times
#      a year for other people's features. Two cases, and neither is left to chance. When both sides
#      changed it, the rule in .git/info/attributes -- README.md merge=ours, installed by
#      scripts/install-fork-hooks.sh -- settles it without a conflict. When only upstream changed it, git
#      takes theirs without consulting any driver, and this script puts the fork's page back into the
#      merge. Nothing is lost either way, because the page is derived: it is written by
#      scripts/fork-pr-table.sh from the commit this branch ends up sitting on.
#   3. nothing else. This repository keeps no copy of upstream's landing page, so a sync leaves no third
#      file to reconcile: upstream's page is upstream's, at the commit the merge lands on, and that
#      commit is in this repository.
#
# Nothing is committed beyond the merge itself. The refreshed pages land in the working tree for
# git publish-main to take out with whatever else is pending.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

UP_REMOTE=${UP_REMOTE:-origin}     # scripts/fork-checks points it at a bare repository in /tmp
UP="$UP_REMOTE/main"
case "${1:-}" in
  '') ;;
  --dry-run|-n) DRY=1 ;;
  -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "usage: $(basename "$0") [--dry-run]" >&2; exit 2 ;;
esac

say() { echo 'sync-upstream:' "$*"; }

if ! git diff --quiet || ! git diff --cached --quiet; then
  say 'the tree is not clean: commit or stash what is pending before syncing' >&2
  git status --short | sed -n '1,10p' | sed 's/^/    /' >&2
  exit 1
fi

say "fetching $UP"
git fetch -q "$UP_REMOTE"

behind=$(git rev-list --count "HEAD..$UP")
if [ "$behind" = 0 ]; then
  say "already on upstream $(git rev-parse --short "$UP"): nothing to merge"
  exit 0
fi
say "$behind upstream commit(s) to take, newest $(git log -1 --format='%h %s' "$UP")"

if [ -n "${DRY:-}" ]; then
  # What git would report without touching the working tree. An empty list is a clean merge: README.md
  # disappears from it entirely once the merge rule is in place, which is the point of the rule.
  out=$(git merge-tree --write-tree --name-only HEAD "$UP" | sed -n '2,40p' || true)
  if [ -z "$out" ]; then
    say '--dry-run: the merge would be clean'
  else
    say '--dry-run: the merge would report these files:'
    printf '%s\n' "$out" | sed 's/^/    /'
  fi
  exit 0
fi

before=$(git rev-parse HEAD)
# --no-ff, because a fast-forward is not a merge: it moves the branch and applies no merge driver, so
# upstream's README would simply replace this fork's page and the rule above would never be consulted.
# A sync is an event worth a commit of its own anyway, and that is what this branch's history says.
if ! git merge --no-edit --no-ff "$UP"; then
  unresolved=$(git diff --name-only --diff-filter=U)
  real=$(printf '%s\n' "$unresolved" | grep -v '^README\.md$' | grep -v '^$' || true)
  if [ -n "$real" ]; then
    say 'the merge stopped on files that are not the front page:' >&2
    printf '    %s\n' $real >&2
    say 'nothing is decided for you here: finish the merge, then run git fork-refresh' >&2
    exit 1
  fi
  if [ -z "$unresolved" ]; then
    say 'the merge stopped without reporting a conflict -- see git status' >&2
    exit 1
  fi
  say "README.md conflicted: keeping this fork's page, which is regenerated below"
  git checkout --ours -- README.md
  git add -- README.md
  git commit --no-edit
fi

# A merge driver is only consulted when *both* sides changed the file. When only upstream rewrote its
# README — which is most of the fifty times a year it does — git takes theirs without asking, and the
# fork's front page is simply gone. So the page is put back here, from the commit the branch was on a
# minute ago, and written into the merge itself: a checkout of this merge has to find a front page the
# generator can rewrite, not upstream's, and the refresh below rewrites it against the new sync commit.
took_theirs=''
if ! grep -qs '^<!-- BEGIN GENERATED' README.md; then
  took_theirs=1
  say "README.md came out of the merge as upstream wrote it: putting this fork's page back"
  git checkout "$before" -- README.md
  git add -- README.md
  git commit --amend --no-edit -q
fi

say 'refreshing the pages from the merged tree'
# The pages speak of the upstream this branch was merged with, so the generator has to be told which ref
# that was: the rehearsal in scripts/fork-checks points it at a bare repository instead of origin.
FORK_UPSTREAM_REF="refs/remotes/$UP_REMOTE/main" bash scripts/fork-pr-table.sh

say "this branch now sits on upstream $(git rev-parse --short "$(git merge-base "$UP" HEAD)")"
if [ -n "$took_theirs" ]; then
  say 'upstream had rewritten its own README: the merge took it, and this fork'"'"'s page went back in'
elif git diff --quiet "$before" HEAD -- README.md; then
  say 'README.md came out of the merge as it went in'
fi
say 'README.md and docs/FORK.md are refreshed in the working tree'
say 'read them, then git publish-main takes them out with the rest'
