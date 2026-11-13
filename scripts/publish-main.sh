#!/usr/bin/env bash
# scripts/publish-main.sh -- publish this fork's main branch with its pages up to date.
#
#   scripts/publish-main.sh            regenerate the pages, commit them, push
#   scripts/publish-main.sh --dry-run  regenerate, show what would be committed, push nothing
#   scripts/publish-main.sh --fast     skip the read-back of the committed pages (one API pass less)
#
# The order is the whole point. scripts/fork-pr-table.sh rewrites README.md and docs/FORK.md in the
# working tree, so a page that has just been refreshed is an uncommitted change: pushing first
# publishes the page of the previous run. So the sequence is look at the remote, generate, commit
# what the generator touched, read the pages back, push.
#
# Only the fork is ever pushed: the script refuses a remote whose URL names the upstream owner,
# because in this clone `origin` is upstream (deepbeepmeep) and `fork` is mine (abencq).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

FORK_REMOTE="${FORK_REMOTE:-fork}"
FORK_BRANCH="${FORK_BRANCH:-main}"
UPSTREAM_OWNER="${UPSTREAM_OWNER:-deepbeepmeep}"
notes='scripts/fork-pr-notes.json'
[ -f "$notes" ] && UPSTREAM_OWNER=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['upstream_repo'].split('/')[0])" "$notes")

DRY=0
FAST=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    --fast) FAST=1 ;;
    -h|--help) echo 'usage: publish-main [--dry-run] [--fast]'; exit 0 ;;
    *) echo "usage: publish-main [--dry-run] [--fast]" >&2; exit 2 ;;
  esac
done

url=$(git remote get-url "$FORK_REMOTE" 2>/dev/null || echo '')
if [ -z "$url" ]; then
  echo "publish-main: remote '$FORK_REMOTE' does not exist — see git remote -v" >&2
  exit 1
fi
case "$url" in
  *"$UPSTREAM_OWNER"*)
    echo "publish-main: '$FORK_REMOTE' points at $url, which is upstream. Nothing is pushed there." >&2
    exit 1 ;;
esac
LOCAL_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Look at the remote before writing anything. A page commit made on top of a branch that has fallen
# behind its remote ends in a rejected push and a page commit stranded in the local history.
echo "publish-main: looking at $FORK_REMOTE/$FORK_BRANCH"
remote_tip=''
if git fetch --quiet "$FORK_REMOTE" "$FORK_BRANCH" 2>/dev/null; then
  remote_tip=$(git rev-parse --verify --quiet "refs/remotes/$FORK_REMOTE/$FORK_BRANCH" || echo '')
  if [ -n "$remote_tip" ] && ! git merge-base --is-ancestor "$remote_tip" HEAD; then
    echo "publish-main: $FORK_REMOTE/$FORK_BRANCH holds $(git log --oneline -1 "$remote_tip")," >&2
    echo "publish-main: which $LOCAL_BRANCH does not contain. Merge it first; nothing is generated, committed or pushed." >&2
    if [ "$DRY" = 1 ]; then exit 0; fi
    exit 1
  fi
  if [ -n "$remote_tip" ]; then
    echo "publish-main: $FORK_REMOTE/$FORK_BRANCH is at $(git rev-parse --short "$remote_tip"), an ancestor of $LOCAL_BRANCH"
  fi
else
  echo "publish-main: $FORK_REMOTE cannot be reached, so the remote state is unknown; the push will tell the truth" >&2
fi

echo "publish-main: regenerating README.md and docs/FORK.md"
scripts/fork-pr-table.sh

# generated_only FILE — true when every changed line of FILE sits in a generated region: the block
# between the BEGIN/END GENERATED markers, or the sentence of the licence record between the inline
# SYNCED markers. A hunk outside both is a hand-written sentence, and that belongs in a commit of its
# own, so scripts/page-hunk-check.sh answers the question and is tested by the fixture test.
generated_only() { scripts/page-hunk-check.sh "$1"; }

dirty=()
for f in README.md docs/FORK.md; do
  git diff --quiet -- "$f" && continue
  if ! generated_only "$f"; then
    echo "publish-main: $f holds edits outside the generated block." >&2
    echo "publish-main: commit those words yourself — publish-main commits only what the generator wrote." >&2
    exit 1
  fi
  dirty+=("$f")
done
if [ ${#dirty[@]} -gt 0 ]; then
  echo "publish-main: the generator changed ${dirty[*]}"
  if [ "$DRY" = 1 ]; then
    git --no-pager diff --stat -- "${dirty[@]}"
  else
    git add -- "${dirty[@]}"
    {
      echo 'Refresh the front page and the pull-request ledger'
      echo
      echo 'The generator counted this fork against upstream and read the state of every pull'
      echo 'request from the GitHub API, so the front page and the licence record say what is'
      echo 'true today. The words of the front page come from docs/fork-features.md and the'
      echo 'rows from scripts/fork-pr-notes.json; nothing here was typed by hand.'
    } | git commit -q -F -
    echo "publish-main: committed the regenerated pages"
  fi
else
  echo "publish-main: the pages were already up to date"
fi

if [ "$DRY" = 1 ]; then
  echo "publish-main: --dry-run, nothing pushed. Would push $LOCAL_BRANCH to $FORK_REMOTE/$FORK_BRANCH"
  git --no-pager log --oneline -3
  exit 0
fi

if ! git diff --quiet; then
  echo "publish-main: note: the working tree still has changes of its own; they stay uncommitted" >&2
fi

# Read the committed pages back from the data. A generator that cannot reproduce its own output has a
# rule that depends on something it did not read, and that is cheaper to find here than on GitHub.
if [ "$FAST" = 0 ]; then
  echo "publish-main: reading the pages back"
  if ! scripts/fork-pr-table.sh --check; then
    echo "publish-main: the committed pages do not reproduce from git and the GitHub API." >&2
    echo "publish-main: the page commit sits in the local history; run git fork-refresh, look at the" >&2
    echo "publish-main: diff, and publish again. Nothing is pushed." >&2
    exit 1
  fi
fi

echo "publish-main: pushing $LOCAL_BRANCH to $FORK_REMOTE/$FORK_BRANCH"
git push "$FORK_REMOTE" "$LOCAL_BRANCH:$FORK_BRANCH"

python3 - "$FORK_REMOTE" "$FORK_BRANCH" <<'PY'
import pathlib, re, subprocess, sys
t = pathlib.Path('README.md').read_text(encoding='utf-8')
m = re.search(r'<!-- front page: regenerated \d{4}-\d\d-\d\d · (\d+) files \+(\d+) lines · (\d+)'
              r' authored commits · (\d+) open, (\d+) merged', t)
remote, branch = sys.argv[1], sys.argv[2]
sha = subprocess.run(['git', 'rev-parse', '--short', '%s/%s' % (remote, branch)],
                     capture_output=True, text=True).stdout.strip()
if m:
    files, adds, commits, open_, merged = m.groups()
    said = '%s files, +%s lines, %s authored commits, %s pull requests open upstream, %s merged' % (
        files, adds, commits, open_, merged)
else:
    said = 'the page stamp is unreadable'
print('publish-main: %s/%s is now %s — %s' % (remote, branch, sha, said))
PY
