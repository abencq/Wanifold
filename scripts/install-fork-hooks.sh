#!/usr/bin/env bash
# scripts/install-fork-hooks.sh -- wire this fork's own tooling into a clone.
#
# Run it once after cloning. It sets:
#   git config core.hooksPath scripts/git-hooks     the pre-push reminder, versioned in the tree
#   git config alias.publish-main, alias.fork-refresh, alias.sync-upstream
#   git config merge.ours.driver true, with "README.md merge=ours" in .git/info/attributes
#     README.md is this fork's generated front page and nothing else, while upstream rewrites its own
#     README about fifty times a year. Keeping ours on a merge loses nothing: upstream's page is not
#     merged here at all, and no copy of it is kept in this repository -- the upstream commit this branch
#     sits on carries it, and that copy cannot go stale. One caveat, and scripts/sync-upstream.sh handles
#     it: a merge driver is consulted only when both sides changed the file, so when upstream alone
#     rewrote it git takes theirs, and the script puts this fork's page back into the merge. The rule is
#     git state of this clone rather than a committed .gitattributes, so it cannot collide with upstream's
#     own attributes -- and it is silently absent from a clone that never ran this script, which is why
#     scripts/fork-checks/scratch-clone.sh looks for it.
#   git config pull.rebase false
#     a rebase swaps ours and theirs, so the rule above would then throw away this fork's page and keep
#     upstream's edit of it. The rule is written for a merge, and git sync-upstream does a merge.
#   a symlink from docs/fork-skills/<name> into the editor's skills directory
#   the fork clauses appended to the commit-message, pr-message and changelog skills
#
# The hook, the aliases and the skill texts live in the repository exactly so that a fresh clone
# keeps them; none of this is git's own state, so it has to be re-established per clone. The script
# is idempotent, and it removes the old global publish-main alias that regenerated the pages and
# then pushed the commit from before the regeneration.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

chmod +x scripts/git-hooks/pre-push scripts/git-hooks/pre-push.py \
         scripts/fork-pr-table.sh scripts/publish-main.sh scripts/sync-upstream.sh 2>/dev/null || true

git config core.hooksPath scripts/git-hooks
git config alias.publish-main '!cd "$(git rev-parse --show-toplevel)" && scripts/publish-main.sh'
git config alias.fork-refresh '!cd "$(git rev-parse --show-toplevel)" && scripts/fork-pr-table.sh && git --no-pager diff --stat -- README.md docs/FORK.md'
git config alias.sync-upstream '!cd "$(git rev-parse --show-toplevel)" && scripts/sync-upstream.sh'
echo 'install: hooksPath = ' "$(git config core.hooksPath)"
echo 'install: aliases   = publish-main, fork-refresh, sync-upstream (repository only)'

# Upstream's landing page against this fork's front page -- see the header. Both halves are git state of
# this clone, so neither is ever pushed, and a clone that skips this script merges README.md the hard way.
git config merge.ours.driver true
attrs="$(git rev-parse --absolute-git-dir)/info/attributes"
mkdir -p "$(dirname "$attrs")"
if grep -qs '^README\.md[[:space:]]\+merge=ours$' "$attrs"; then
  echo 'install: README.md already keeps this fork'"'"'s page on a merge'
else
  printf '%s\n' 'README.md merge=ours' >> "$attrs"
  echo "install: README.md merge=ours added to $attrs"
fi
git config pull.rebase false
echo 'install: merge    = README.md keeps ours, pull.rebase = false (a rebase would swap the two)'

# A hook left under .git/hooks is inert once core.hooksPath points into the tree, and an inert copy
# of an old reminder is worse than nothing: it reads like the ritual and never runs. Ask for the git
# dir, not for `git rev-parse --git-path hooks`, which answers with core.hooksPath and would name the
# live hook.
old_hook=$(git rev-parse --absolute-git-dir)/hooks/pre-push
if [ -e "$old_hook" ] && [ ! -L "$old_hook" ] \
   && [ "$(realpath "$old_hook")" != "$(realpath scripts/git-hooks/pre-push)" ]; then
  rm -f "$old_hook"
  echo "install: removed the superseded hook at $old_hook — scripts/git-hooks/pre-push is the one that runs"
fi

if git config --global --get-regexp '^alias\.publish-main' >/dev/null 2>&1; then
  git config --global --unset alias.publish-main
  echo 'install: removed the old global alias publish-main, which pushed the page of the previous run'
fi

skills_dir="${SKILLS_DIR:-$HOME/.config/Code/User/skills}"
[ -d "$skills_dir" ] || { echo "install: no skills directory at $skills_dir — skills left alone"; exit 0; }

for s in docs/fork-skills/*; do
  [ -f "$s/SKILL.md" ] || continue       # clauses/ holds fragments, not skills
  name=$(basename "$s")
  target="$skills_dir/$name"
  here=$(pwd)/"$s"
  if [ -L "$target" ] && [ "$(readlink -f "$target")" = "$(readlink -f "$here")" ]; then
    echo "install: skill $name already linked"
  elif [ -e "$target" ]; then
    echo "install: $target exists and is not this repository's $name — left alone"
  else
    ln -s "$here" "$target"
    echo "install: skill $name linked into $skills_dir"
  fi
done

begin='<!-- BEGIN fork clause (scripts/install-fork-hooks.sh) -->'
end='<!-- END fork clause -->'
for frag in docs/fork-skills/clauses/*.md; do
  name=$(basename "$frag" .md)
  skill="$skills_dir/$name/SKILL.md"
  [ -f "$skill" ] || { echo "install: no $name skill installed — clause skipped"; continue; }
  python3 - "$skill" "$frag" "$begin" "$end" <<'PY'
import pathlib, sys
skill, frag = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
begin, end = sys.argv[3], sys.argv[4]
if skill.is_symlink() and skill.resolve().is_relative_to(pathlib.Path('.').resolve()):
    sys.exit('install: %s is this repository\'s own skill — clause skipped' % skill)
clause = '\n'.join([begin, frag.read_text(encoding='utf-8').rstrip(), end]) + '\n'
text = skill.read_text(encoding='utf-8')
if begin in text:
    head = text.split(begin)[0]
    tail = text.split(end, 1)[1] if end in text else ''
    text = head + clause + tail.lstrip('\n')
else:
    text = text.rstrip('\n') + '\n\n' + clause
skill.write_text(text, encoding='utf-8')
print('install: fork clause in place in %s' % skill.name)
PY
done
echo 'install: done'
