#!/usr/bin/env bash
# scripts/fork-table-fixture-test.sh -- run the page generator against a GitHub API that is not called.
#
#   scripts/fork-table-fixture-test.sh
#
# The paths that matter on the day upstream accepts a pull request — the accepted row, the release it
# went into, the claim of the front page, the licence sentence of docs/FORK.md — are the paths nothing
# can be tried at home today, because nothing of mine is merged upstream. So the answers are written
# down: FORK_TABLE_FIXTURES points the generator at a JSON map of API path to response, and this script
# builds a scratch repository around it. The scratch upstream is a fake whose wgp.py carries
# WanGP_version, which is what an official release of this application is named by: upstream puts no
# release tag on the app, only on its manager.
#
# The scenarios asserted: an acceptance shipped in a release, an acceptance not yet in one and a
# closure, everything open as it is today, every feature shipped, the status callout that rides with a
# feature this fork is still reworking and leaves when upstream takes it whole, the closed pull request
# upstream answered by writing the same fix inside a larger change of its own, the front page holding
# nothing but this fork (no copy of upstream's landing page is kept anywhere), the long form of a row,
# which the ledger keeps and the front page leaves off, and the sentence naming where upstream has got to
# once the fetched ref sits past the commit this branch is merged with. Nothing is fetched, nothing is
# pushed, and the repository this script lives in is not written to.
set -uo pipefail
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SCRATCH=${SCRATCH:-/tmp/fork-fixture-$$}
FAILED=0
ok()  { printf 'ok   %s\n' "$1"; }
bad() { printf 'FAIL %s\n' "$1"; FAILED=1; }
flat()   { tr '\n' ' ' < "$1" | tr -s ' '; }        # generated prose is wrapped; compare it unwrapped
want()   { flat "$1" | grep -qF -- "$(printf '%s' "$2" | tr -s ' ')" && ok "$3" \
           || { bad "$3"; echo "  missing from $1: $2"; }; }
wantnt() { flat "$1" | grep -qF -- "$(printf '%s' "$2" | tr -s ' ')" && { bad "$3"; echo "  present in $1: $2"; } \
           || ok "$3"; }
gen()    { ( cd "$SCRATCH/fork" && FORK_TABLE_FIXTURES="$SCRATCH/api" scripts/fork-pr-table.sh ) > "$SCRATCH/gen.log" 2>&1; }
check()  { ( cd "$SCRATCH/fork" && FORK_TABLE_FIXTURES="$SCRATCH/api" scripts/fork-pr-table.sh --check ) > "$SCRATCH/check.log" 2>&1; }

rm -rf "$SCRATCH"; mkdir -p "$SCRATCH/fork" "$SCRATCH/api"
git init -q -b up "$SCRATCH/fork"
cd "$SCRATCH/fork" || { echo 'ABORT: no scratch repository'; exit 9; }
top=$(git rev-parse --show-toplevel 2>/dev/null)
[ "$top" = "$SCRATCH/fork" ] || { echo "ABORT: not the scratch repository ($top)"; exit 9; }
git config user.name abencq
git config user.email abencq@users.noreply.github.com

# The fake upstream and the acceptances, all reachable so that `git show <sha>:wgp.py` answers:
# v18.10, an acceptance inside it, v19.02 as origin/main, an acceptance inside that, and an acceptance
# whose tree carries no version at all.
printf 'WanGP_version = "18.10"\n' > wgp.py
printf '# third party notices\n' > THIRD.md
git add -A && git commit -qm 'upstream at v18.10'
printf '# 2292 accepted\n' > naming.py && git add -A && git commit -qm 'merge of #2292, in v18.10'
M18=$(git rev-parse HEAD)
# Upstream's own landing page exists at the sync commit, because that is the case the fork has to live
# with: upstream keeps writing this file, about fifty times a year, for other people's features.
printf '# WanGP\n\nThe landing page upstream writes and rewrites.\n\nMade with love by DeepBeepMeep\n' > README.md
printf 'WanGP_version = "19.02"\n' > wgp.py && git add -A && git commit -qm 'upstream at v19.02'
U2=$(git rev-parse HEAD)
printf '# latent io\n' > latent_io.py && git add -A && git commit -qm 'merge of #2441, in v19.02'
M19=$(git rev-parse HEAD)
git rm -q wgp.py && git commit -qm 'merge of #2290, whose tree names no version'
MNR=$(git rev-parse HEAD)
git checkout -q "$U2"

# The fork's own commit: the generator, its words, the two pages it writes.
mkdir -p scripts/git-hooks docs
cp "$SRC/scripts/fork-pr-table.sh" "$SRC/scripts/fork-pr-notes.json" scripts/
cp "$SRC/scripts/git-hooks/pre-push" "$SRC/scripts/git-hooks/pre-push.py" scripts/git-hooks/
cp "$SRC/scripts/publish-main.sh" "$SRC/scripts/page-hunk-check.sh" "$SRC/scripts/install-fork-hooks.sh" scripts/
cp "$SRC/docs/fork-features.md" "$SRC/docs/FORK.md" docs/
cp "$SRC/README.md" README.md
git add -A && git commit -qm 'the fork: the pages and the scripts that write them'
git checkout -q -B local
git update-ref refs/remotes/origin/main "$U2"
SHORT=$(git rev-parse --short "$U2")

# The scenario builder is a script of its own, so the scratch-clone and the publish suite can ask it
# for the same written-down answers without reaching for the network.
cp "$SRC/scripts/fork-api-fixture.py" "$SCRATCH/build.py"

scenario() {                     # $1 = merged map json, $2 = closed list json
  FIX_MERGED="$1" FIX_CLOSED="$2" UP_TIP="$U2" API="$SCRATCH/api" NOTES="$SRC/scripts/fork-pr-notes.json" \
    python3 "$SCRATCH/build.py" || { bad 'the fixture could not be built'; return; }
  gen
  [ $? = 0 ] && ok "the generator writes from the fixture ($3)" || { bad "generator, $3"; sed 's/^/  /' "$SCRATCH/gen.log"; }
}
echo '=== the scratch repository ==='
echo "  fork $(git rev-parse --abbrev-ref HEAD) at $(git rev-parse --short HEAD), upstream $SHORT"


echo '=== 1. an acceptance in a release, an acceptance with no version, nothing closed ==='
scenario "{\"2441\": \"$M19\", \"2290\": \"$MNR\"}" '[]' 'an acceptance'
want docs/FORK.md 'accepted 2026-11-10 in v19.02' 'the accepted row names the release it went into'
want docs/FORK.md 'accepted 2026-11-10, not in a release yet' 'an acceptance with no version says so'
wantnt docs/FORK.md '*(none yet' 'the merged table left its empty note behind'
want docs/FORK.md '*(none — nothing of mine has been closed without merge' 'the closed table says none'
want docs/FORK.md "synced with upstream \`$SHORT\` (WanGP v19.02)" 'the licence sentence names the commit and its version'
want README.md '1 of the 3 features below are in an official WanGP release' 'the headline counts the shipped feature'
want README.md '(v19.02 and later), **2 features are not**' 'the headline counts the rest'
want README.md 'on 2026-11-10 in v19.02, so this fork is not needed for it' 'the shipped feature says the fork is not needed'
want README.md 'on 2026-11-10, not in a release yet' 'the front page says so too when no version is known'
want README.md '[PR #2292](https://github.com/deepbeepmeep/WanGP/pull/2292)' 'a feature with two pull requests keeps both of them named'
check && ok 'the pages read back' || { bad 'the read-back'; sed 's/^/  /' "$SCRATCH/check.log"; }

# The licence sentence sits in the hand-written part of docs/FORK.md, so publish-main's question — is
# every changed line generated? — has to know the inline region too.
scripts/page-hunk-check.sh README.md && ok 'a regenerated front page counts as generated' || bad 'the front page is refused'
scripts/page-hunk-check.sh docs/FORK.md && ok 'a regenerated licence sentence counts as generated' || bad 'the licence sentence is refused'
python3 - <<'PY'
import pathlib
p = pathlib.Path('docs/FORK.md')
t = p.read_text(encoding='utf-8')
p.write_text(t.replace('change dates from **2025-04-16**.',
                       'change dates from **2025-04-16**, and this sentence was typed by hand.', 1),
             encoding='utf-8')
PY
scripts/page-hunk-check.sh docs/FORK.md && bad 'a hand-typed sentence passes as generated' \
  || ok 'a hand-typed sentence is refused its own commit'
git checkout -- docs/FORK.md
gen

# A fetch moves origin/main; the licence statement is true of the commit the branch carries, so the
# fetch must not move it, and that a sync is due has to be said out loud. The commit upstream moved to
# is built with plumbing, because the working tree of this fixture is dirty by design.
blob=$(printf 'WanGP_version = "19.13"\n' | git hash-object -w --stdin)
tree=$(git ls-tree "$U2" | awk -v b="$blob" '$4=="wgp.py"{$3=b} {printf "%s %s %s\t%s\n", $1, $2, $3, $4}' | git mktree)
NEW=$(git commit-tree "$tree" -p "$U2" -m 'upstream moves to v19.13, ahead of this branch')
[ "$(git show "$NEW:wgp.py" | sed -n 's/^WanGP_version = "\(.*\)".*/\1/p')" = 19.13 ] \
  && ok 'the scratch upstream moves to v19.13' || bad 'the scratch upstream commit is wrong'
NEWSHORT=$(git rev-parse --short "$NEW")
git update-ref refs/remotes/origin/main "$NEW"
gen
want docs/FORK.md "synced with upstream \`$SHORT\` (WanGP v19.02)" 'a fetch leaves the licence statement on the commit the branch carries'
wantnt docs/FORK.md "\`$NEWSHORT\`" 'the fetched commit is claimed by neither page'
want README.md "synced_with_upstream-$SHORT" 'the front-page badge stays on the sync point'
want README.md "[\`$SHORT\`](https://github.com/deepbeepmeep/WanGP/commit/$U2) (WanGP v19.02)" \
  'the headline links the commit this branch is merged with, and names its release'
flat "$SCRATCH/gen.log" | grep -qF 'past the sync point' && ok 'the generator says the sync is due' \
  || { bad 'the generator stayed quiet about the sync'; sed 's/^/  /' "$SCRATCH/gen.log"; }
printf 'refs/heads/local %s refs/heads/main %s\n' "$(git rev-parse HEAD)" "$(git rev-parse HEAD)" \
  | scripts/git-hooks/pre-push fork url > "$SCRATCH/hook.log" 2>&1
flat "$SCRATCH/hook.log" | grep -qF 'past the sync point' && ok 'the reminder says the sync is due' \
  || { bad 'the reminder stayed quiet about the sync'; sed 's/^/  /' "$SCRATCH/hook.log"; }
flat "$SCRATCH/hook.log" | grep -qF 'licence statement is behind' && bad 'the reminder blames the page for a fetch' \
  || ok 'the page is not blamed for a fetch'
git update-ref refs/remotes/origin/main "$U2"
gen
check && ok 'the pages read back at the sync point' \
  || { bad 'the read-back at the sync point'; sed 's/^/  /' "$SCRATCH/check.log"; }

python3 - "$SCRATCH/fork/docs/FORK.md" "$SHORT" <<'PY'
import pathlib, sys
p, sha = pathlib.Path(sys.argv[1]), sys.argv[2]
t = p.read_text(encoding='utf-8')
p.write_text(t.replace('synced with upstream `%s`' % sha, 'synced with upstream `deadbeef`', 1),
             encoding='utf-8')
PY
check && bad 'a hand-typed licence sha passes the read-back' || ok 'a hand-typed licence sha fails the read-back'
out=$(printf 'refs/heads/local %s refs/heads/main %s\n' "$(git rev-parse HEAD)" "$(git rev-parse origin/main)" \
       | scripts/git-hooks/pre-push fork url 2>&1)
echo "$out" | grep -q 'licence statement is behind' && ok 'the reminder names the licence sha' \
  || { bad 'the reminder on the licence sha'; echo "  $out"; }
gen
check && ok 'the sentence writes itself back' || bad 'regeneration after the tamper'

echo '=== 2. everything open, which is where things stand today ==='
scenario '{}' '[]' 'all open'
want README.md 'the 3 features below are **not in an official WanGP release today**' 'the headline claims no release'
wantnt README.md 'in an official WanGP release** (' 'no version is named when nothing is shipped'
want docs/FORK.md '*(none yet' 'the merged table keeps its empty note'
want README.md ', so today this needs the fork' 'an open feature says the fork is needed'
wantnt README.md "## What's new in this fork" 'the front page keeps no separate list of new work'
flat "$SCRATCH/gen.log" | grep -qF 'described twice' && bad 'a change is described twice on the page' \
  || ok 'nothing is described twice on the front page'
want README.md 'Not covered' 'the latents section keeps the release-note detail'
want README.md 'This feature is **being reworked**: refactored, improved and fixed' \
  'a feature still being reworked says so under its own heading'
rework=$(grep -cF 'This feature is **being reworked**' README.md)
[ "$rework" = 1 ] && ok 'only the feature under rework carries the status callout' \
                  || bad "the status callout appears $rework times, expected once"
wantnt README.md 'LoRA paths' "the LoRA entry, upstream's own, is not on the page"
check && ok 'the pages read back' || { bad 'the read-back'; sed 's/^/  /' "$SCRATCH/check.log"; }

echo '=== 3. every feature accepted and shipped ==='
scenario "{\"2441\": \"$M19\", \"2292\": \"$M18\", \"2290\": \"$M19\", \"2535\": \"$M19\"}" '[]' 'all shipped'
want README.md '**all 3 features below are in an official WanGP release** (v18.10, v19.02 and later)' \
  'the headline says all are released and names the versions'
want README.md 'in v18.10, v19.02, so this fork is not needed for it' 'a feature shipped twice names both'
wantnt README.md 'This feature is **being reworked**' 'a feature upstream took whole drops the rework line'
wantnt README.md '~~' 'an accepted pull request is not struck through'
check && ok 'the pages read back' || { bad 'the read-back'; sed 's/^/  /' "$SCRATCH/check.log"; }

echo '=== 4. a pull request closed without merge ==='
scenario '{}' '[2286, 2565]' 'a closure'
want docs/FORK.md 'not picked up; the builder script stays in this fork' 'the closed row keeps the hand-written reason'
want README.md '| Silent startup |' 'a closed pull request upstream answered returns to the front page'
want README.md "in upstream's own 3D update of 2026-11-03" 'a fix with no commit of its own is placed by date'
want docs/FORK.md "as part of a larger update, in upstream's own 3D update of 2026-11-03" \
  'the ledger row carries the reason and the placement together'
want docs/FORK.md '~~[#2286](https://github.com/deepbeepmeep/WanGP/pull/2286)~~' \
  'a closed pull request is struck through in the ledger'
want README.md '~~[#2565](https://github.com/deepbeepmeep/WanGP/pull/2565)~~' \
  'the front page strikes the closed pull request it still lists'
wantnt README.md '~~[#2526]' 'an open pull request keeps its link whole'
# A whole feature's pull request can be closed too — it is the case that comes for the latents if upstream
# ever refuses them — and then the strike belongs on the bullet that closes the feature section, not just
# on a table row.
scenario '{}' '[2290]' 'a feature whose pull request was closed'
want README.md '~~[PR #2290](https://github.com/deepbeepmeep/WanGP/pull/2290)~~' \
  'the bullet of a feature whose pull request is closed is struck through'
want docs/FORK.md '~~[#2290](https://github.com/deepbeepmeep/WanGP/pull/2290)~~' \
  'the same pull request is struck in the ledger as well'
want README.md 'closed without merge on' 'and the feature says so in words next to it'
wantnt docs/FORK.md '*(none — nothing of mine has been closed without merge' 'the closed table filled in'
want README.md 'the 3 features below are **not in an official WanGP release today**' 'a closure moves no release claim'
check && ok 'the pages read back' || { bad 'the read-back'; sed 's/^/  /' "$SCRATCH/check.log"; }

echo '=== 5. the front page holds only this fork, and a change is said once ==='
# README.md is this fork's own front page and nothing else. No copy of upstream's landing page is kept in
# this repository, so the note that used to point at one must not come back, and nothing at all may follow
# the block the generator wrote — not a paragraph, not a blank line.
check && ok 'the pages read back' || { bad 'the read-back'; sed 's/^/  /' "$SCRATCH/check.log"; }
wantnt README.md "Upstream's own page." 'the front page ends without a note about upstream and its copy'
wantnt README.md 'docs/UPSTREAM-README.md' 'no page points at a copy this fork does not keep'
wantnt README.md "## What's new in this fork" 'the front page carries no second list of changes'
[ "$(tail -n 1 README.md)" = '<!-- END GENERATED -->' ] \
  && ok 'the front page ends at its generated block' || bad 'something follows the generated block'

# The long form of a row belongs to the ledger, so a change is described once on the page a reader lands
# on. The scratch copy of the notes is edited, not the one in the source tree.
python3 - <<'PY'
import json, pathlib
p = pathlib.Path('scripts/fork-pr-notes.json')
d = json.loads(p.read_text(encoding='utf-8'))
d['prs']['2441']['detail'] = 'A long form that belongs to the ledger alone.'
p.write_text(json.dumps(d, indent=2) + '\n', encoding='utf-8')
PY
gen
want docs/FORK.md 'A long form that belongs to the ledger alone' 'the ledger carries the long form'
wantnt README.md 'A long form that belongs to the ledger alone' 'the front page keeps only the short form'

# Two descriptions of one pull request are two sentences that can disagree, so the generator says so. The
# short title is what the front-page table prints, so that is where a second description would appear.
python3 - <<'PY'
import json, pathlib
p = pathlib.Path('scripts/fork-pr-notes.json')
d = json.loads(p.read_text(encoding='utf-8'))
d['prs']['2565']['short'] += ' ([PR #2565](https://github.com/deepbeepmeep/WanGP/pull/2565))'
p.write_text(json.dumps(d, indent=2) + '\n', encoding='utf-8')
PY
gen
flat "$SCRATCH/gen.log" | grep -qF '#2565 described twice' && ok 'a change described twice is said out loud' \
  || { bad 'the duplicate mention goes unheard'; sed 's/^/  /' "$SCRATCH/gen.log"; }
git checkout -- scripts/fork-pr-notes.json
gen
flat "$SCRATCH/gen.log" | grep -qF 'described twice' && bad 'the warning outlives the duplicate' \
  || ok 'the warning goes when the duplicate does'
check && ok 'the pages read back' || { bad 'the read-back'; sed 's/^/  /' "$SCRATCH/check.log"; }

echo '=== 6. upstream moving past the sync point ==='
# The headline sentence names the commit this branch is merged with. Once the fetched ref sits past it, the
# page names the commit upstream has got to as well: a reader who cannot see the distance cannot tell a
# synced branch from a stale one, and the licence sentence must go on naming the commit the branch carries.
# Catching up here is a rewind of the ref rather than a merge — what is under test is the sentence, and the
# merge itself is rehearsed by scripts/fork-checks/scratch-clone.sh.
git checkout -q -- README.md docs/FORK.md
git checkout -q "$U2"
printf 'WanGP_version = "19.03"\n' > wgp.py
printf '# WanGP\n\nUpstream rewrote its own landing page again.\n' > README.md
git add -A && git commit -qm 'upstream at v19.03, past the sync point'
U3=$(git rev-parse HEAD)
U3SHORT=$(git rev-parse --short "$U3")
git checkout -q local
git update-ref refs/remotes/origin/main "$U3"
gen
flat "$SCRATCH/gen.log" | grep -qF 'past the sync point' && ok 'the generator says the sync is due' \
  || { bad 'the generator stays quiet about a branch fetched past its sync point'; sed 's/^/  /' "$SCRATCH/gen.log"; }
want README.md "[\`$U3SHORT\`](https://github.com/deepbeepmeep/WanGP/commit/$U3)" \
  'the page links the commit upstream has got to'
want README.md 'upstream is 1 commit further on at' 'the page says how far upstream has moved'
want README.md '(WanGP v19.03)' 'the commit upstream has got to is named by its release too'
want README.md "[\`$SHORT\`](https://github.com/deepbeepmeep/WanGP/commit/$U2)" \
  'the commit this branch is merged with keeps being named'
want docs/FORK.md "synced with upstream \`$SHORT\`" 'the licence sentence does not run ahead of the merge'
check && ok 'the pages read back with upstream ahead' \
  || { bad 'the read-back with upstream ahead'; sed 's/^/  /' "$SCRATCH/check.log"; }
git update-ref refs/remotes/origin/main "$U2"
gen
wantnt README.md 'further on at' 'the sentence goes when the branch is up to date again'
check && ok 'the pages read back once more' || { bad 'the read-back'; sed 's/^/  /' "$SCRATCH/check.log"; }

cd / && rm -rf "$SCRATCH"
echo
[ "$FAILED" = 0 ] && echo 'THE FIXTURE TESTS PASSED' || echo 'SOME FIXTURE CHECKS FAILED'
exit "$FAILED"
