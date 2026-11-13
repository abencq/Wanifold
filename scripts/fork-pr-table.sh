#!/usr/bin/env bash
# scripts/fork-pr-table.sh -- regenerate this fork's own pages from git and from the GitHub API.
#
#   scripts/fork-pr-table.sh          rewrite README.md and docs/FORK.md
#   scripts/fork-pr-table.sh --check  exit 1 when one of the two is not up to date
#
# docs/FORK.md             the ledger: the state of my pull requests, in tables, between the markers.
# README.md                the front page, written whole: the size of the fork from git, the features of
#                          docs/fork-features.md with the live state of their pull requests, and the
#                          table of the pull requests that are open upstream. Everything on it is about
#                          this fork; no word of upstream's landing page is mixed into it, and none of it
#                          is copied here either. Upstream's page stays upstream's, at the commit the
#                          front page names: this repository carries that commit, so the copy a reader
#                          wants is one `git show` away, and it cannot be a stale second hand of it.
#
# The hand-written wording is scripts/fork-pr-notes.json (the rows, the features, the short titles)
# and docs/fork-features.md (the sentences of the front page). New pull requests, closures and
# upstream acceptances appear by themselves; a pull request with no note appears as a loud TODO row.
# The size of the fork is computed, never typed: git diff and git rev-list against origin/main, with
# authored commits counted by --no-merges so a sync with upstream cannot inflate them, and without
# the commits that only refresh these pages so that publishing a page is not counted as work.
# The sentence in docs/FORK.md that names the upstream commit this fork sits on — the licence record
# of section 4.1 — is written here too, between the inline <!-- SYNCED --> markers, because a dated
# statement of what was modified is only true of the commit it names.
# What a pull request shipped in is read from upstream's own version string in wgp.py at the merge
# commit, since upstream publishes no release tags for the app: that is what makes the front page
# claim "in a release" per feature rather than for the fork as a whole.
# Needs an authenticated gh, and a fetched origin/main for the "behind" counts. With
# FORK_TABLE_FIXTURES=dir the API is read from dir/api.json (a map of API path to response) and no
# network is touched: scripts/fork-table-fixture-test.sh uses that to exercise acceptances offline.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
MODE="${1:-write}"
case "$MODE" in
  write|check) ;;
  --check) MODE=check ;;
  -h|--help) echo "usage: $(basename "$0") [--check]"; exit 0 ;;
  *) echo "usage: $(basename "$0") [--check]" >&2; exit 2 ;;
esac
if [ "$MODE" = write ] && [ -n "${FORK_TABLE_SKIP:-}" ]; then
  exit 0                      # set by git during a merge, where rewriting is pointless
fi
export FORK_TABLE_MODE="$MODE"
exec python3 - <<'PY'
import collections, datetime, json, os, re, subprocess, sys, textwrap

root = subprocess.run(['git', 'rev-parse', '--show-toplevel'],
                      capture_output=True, text=True).stdout.strip()
notes_file = os.path.join(root, 'scripts', 'fork-pr-notes.json')
features_file = os.path.join(root, 'docs', 'fork-features.md')
ledger = os.path.join(root, 'docs', 'FORK.md')
front = os.path.join(root, 'README.md')
# The two pages this script writes are not this fork's work: the size on the front page counts the code
# the fork added, not the generated prose that describes it, and it stopped counting upstream's page the
# day the fork stopped copying it.
NOT_FORK_WORK = ('README.md', 'docs/FORK.md')
mode = os.environ['FORK_TABLE_MODE']
fixtures = os.environ.get('FORK_TABLE_FIXTURES')   # a dir holding api.json: then no network at all
BEGIN, END = '<!-- BEGIN GENERATED', '<!-- END GENERATED -->'
SYNC_BEGIN, SYNC_END = '<!-- SYNCED -->', '<!-- /SYNCED -->'
GREEN = 'https://img.shields.io/badge/merged-brightgreen'
AMBER = 'https://img.shields.io/badge/addressed%20upstream-orange'
REF = 'refs/remotes/origin/main'
# scripts/fork-checks rehearses a sync against a bare repository in /tmp, and points both this script
# and scripts/sync-upstream.sh at the ref the rehearsal calls upstream.
REF = os.environ.get('FORK_UPSTREAM_REF') or REF

def gh(path):
    """One GitHub API call, or one entry of a fixture map when FORK_TABLE_FIXTURES is set."""
    if fixtures:
        api = os.path.join(fixtures, 'api.json')
        if not os.path.exists(api):
            sys.exit('fork-pr-table: no fixture file at %s' % api)
        data = json.load(open(api, encoding='utf-8'))
        if path not in data:
            sys.exit('fork-pr-table: the fixture has no answer for %s (it answers %s)'
                     % (path, ', '.join(sorted(data))[:400]))
        return data[path]
    r = subprocess.run(['gh', 'api', path], capture_output=True, text=True)
    if r.returncode != 0:
        first = (r.stderr or '').strip().splitlines()
        sys.exit('fork-pr-table: gh api %s failed: %s'
                 % (path, first[0] if first else 'no output'))
    return json.loads(r.stdout)

def git(*args):
    r = subprocess.run(('git',) + args, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ''


def upstream_version(at):
    """Upstream's own version string in wgp.py at a commit, as '19.02'.

    Upstream puts no release tag on the application — the only tags in this clone are the manager's —
    so the version string in the tree is what an official release is named by, and it is readable at
    any commit: the version at a merge commit is the release that merge went into.
    """
    m = re.search(r'^WanGP_version\s*=\s*["\']([^"\']+)["\']', git('show', '%s:wgp.py' % at), re.M)
    return m.group(1) if m else ''


def vkey(v):
    return [int(x) if x.isdigit() else x for x in re.split(r'[._-]', v)]


def versions(nums):
    """'v18.10, v19.02', the releases the given pull requests went into, oldest first."""
    return ', '.join('v%s' % v for v in sorted({info[n]['release'] for n in nums if info[n]['release']},
                                               key=vkey))

notes = json.load(open(notes_file, encoding='utf-8'))
up, author = notes['upstream_repo'], notes['author']
areas, prs = notes['areas'], notes['prs']
closed_reason, fork_only = notes.get('closed_reason', {}), notes.get('fork_only', [])
# A pull request upstream answered by writing the same fix itself is closed rather than merged, and the
# fork no longer needs it. It stays on the front page's table with that state, because "upstream has it"
# is the answer a reader of this page came for; the note is also the reason the ledger's closed row gives.
addressed = {int(k): v for k, v in notes.get('addressed_upstream', {}).items()}
addressed_commit = {int(k): v for k, v in notes.get('addressed_upstream_commit', {}).items()}
# A fix upstream wrote inside a larger change of its own has no commit that carries only it. Such a row
# names that change instead: the same honesty told at a coarser grain, and never a badge left unexplained.
addressed_when = {int(k): v for k, v in notes.get('addressed_upstream_when', {}).items()}


def addressed_where(n):
    """'in WanGP v19.25 `0e0a06f6`' — the release of the app that first carried upstream's own copy.

    The release is read from upstream's own version string at that commit and never typed in the notes:
    a version typed in two places is a version that can disagree with the commit it claims. A commit this
    clone cannot read (a scratch repository built from fixtures) simply goes unversioned. A fix upstream
    wrote inside a larger change of its own has no commit to name and is placed by the notes instead.
    """
    sha = addressed_commit.get(int(n), '')
    if not sha:
        return addressed_when.get(int(n), '')
    link = '[`%s`](https://github.com/%s/commit/%s)' % (sha, up, sha)
    rel = upstream_version(sha)
    if rel:
        return 'in WanGP v%s %s' % (rel, link)
    print('fork-pr-table: %s is not a commit this clone can read: no release is named for it' % sha,
          file=sys.stderr)
    return 'upstream in %s' % link
features = notes.get('features', [])

q = 'repo:%s+is:pr+author:%s' % (up, author)
open_nums = [i['number'] for i in gh('search/issues?q=%s+state:open&per_page=200' % q)['items']]
merged_nums = [i['number'] for i in gh('search/issues?q=%s+is:merged&per_page=200' % q)['items']]
dead_nums = [i['number'] for i in gh('search/issues?q=%s+is:closed+is:unmerged&per_page=200' % q)['items']]

info, supersedes = {}, {}
for n in sorted(set(open_nums) | set(merged_nums) | set(dead_nums)):
    d = gh('repos/%s/pulls/%d' % (up, n))
    base_behind = git('rev-list', '--count', '%s..%s' % (d['base']['sha'], REF))
    release = ''
    if d.get('merged_at') and d.get('merge_commit_sha'):
        release = upstream_version(d['merge_commit_sha'])
    info[n] = {
        'title': d['title'], 'url': d['html_url'],
        'state': 'merged' if d.get('merged_at') else ('open' if d['state'] == 'open' else 'closed'),
        'commits': d['commits'], 'files': d['changed_files'],
        'add': d['additions'], 'del': d['deletions'],
        'mergeable': d.get('mergeable_state') or '?',
        'behind': base_behind if base_behind.isdigit() else '?',
        'created': d['created_at'][:10],
        'closed': (d.get('closed_at') or '')[:10],
        'merged': (d.get('merged_at') or '')[:10],
        'release': release,
    }
    for m in re.finditer(r'supersedes and closes #(\d+)|supersedes #(\d+)', d.get('body') or '', re.I):
        supersedes[int(m.group(1) or m.group(2))] = n

unknown = sorted(int(k) for k in prs if int(k) not in info)
for f in features:
    missing = [n for n in f['prs'] if n not in info]
    if missing:
        sys.exit('fork-pr-table: feature %s names pull request(s) %s, which are not mine upstream'
                 % (f['id'], ', '.join(map(str, missing))))
    if not os.path.exists(os.path.join(root, f.get('image', '.'))):
        f.pop('image', None)          # never publish a link to a picture that is not there

def label(n):
    e = info[n]
    if e['state'] == 'open':
        return 'open · %s · %s behind' % (e['mergeable'], e['behind'])
    if e['state'] == 'merged':
        return 'accepted %s%s' % (e['merged'], in_release(e))
    return 'closed %s' % e['closed']


def in_release(e):
    """' in v19.02' when the acceptance reached an official release, the truth when it has not."""
    return ' in v%s' % e['release'] if e['release'] else ', not in a release yet'

def sup(n):
    return ' (supersedes #%d)' % supersedes[n] if n in supersedes else ''

def pr_link(n, text=None):
    """A pull request the way the pages name it: a link, struck through once upstream closed it
    without merging.

    The strike says the request is dead, not the change: several of the closed ones carry a fix upstream
    wrote itself and shipped, which is why the state beside them still reads addressed upstream. The link
    stays clickable under the strike, because why it was closed is written on the page it links to.
    """
    t = '[#%d](%s)' % (n, info[n]['url']) if text is None else text
    return '~~%s~~' % t if info[n]['state'] == 'closed' else t

def row(area, text, n):
    e = info[n]
    return '| %s | %s | %s | %s |' % (area, text, pr_link(n), label(n))

order = list(areas)
def rank(n):
    a = prs.get(str(n), {}).get('area', '')
    return (order.index(a) if a in order else len(order), -n)

today = datetime.date.today().isoformat()
# "synced with upstream X" is true of the newest upstream commit this branch carries, and a fetch moves
# origin/main without moving the branch. So the sentence is named by the merge base, and a branch that
# has been fetched past its own sync point says so out loud: the sync is due, and the page may not
# claim one. The counts of how far behind each pull request sits are measured against the fetched tip,
# because that is the distance a reviewer sees.
sync = git('merge-base', REF, 'HEAD')
tip = git('rev-parse', '--short', sync) or git('rev-parse', '--short', REF) or '?'
tip_version = upstream_version(sync)
behind = int(git('rev-list', '--count', 'HEAD..%s' % REF) or 0)
if behind:
    print('fork-pr-table: origin/main is %d commit%s past the sync point %s: the branch is not synced'
          % (behind, '' if behind == 1 else 's', tip), file=sys.stderr)
L = ['<!-- BEGIN GENERATED by scripts/fork-pr-table.sh -- do not edit between the markers;',
     '     the hand-written wording is in scripts/fork-pr-notes.json -->',
     '*current as of %s · upstream `%s` · %d open · %d merged · %d closed without merge*'
     % (today, tip, len(open_nums), len(merged_nums), len(dead_nums)), '',
     '### In this fork, submitted upstream',
     '| Area | What it does | PR | State |', '|---|---|---|---|']
for n in sorted(open_nums, key=rank):
    note = prs.get(str(n))
    area = areas.get(note['area'], note['area']) if note else '**TODO**'
    text = note['note'] if note else '**add a note for this PR in scripts/fork-pr-notes.json** (%s)' % info[n]['title']
    if note and note.get('wip'):    # still being reworked here: the ledger says it in the front page's words
        text = '**Being reworked: %s**, and not verified against current upstream. %s' % (note['wip'], text)
    if note and note.get('detail'):  # the long form of a row: the ledger carries it, the front page the short
        text += ' ' + note['detail']
    L.append(row(area, text + sup(n), n))
L.append('')

L += ['### Only in this fork, not submitted', '| Area | What it does |', '|---|---|']
for e in fork_only:
    L.append('| %s | %s |' % (areas.get(e['area'], e['area']), e['note']))
L.append('')

L += ['### Merged upstream — removed from this fork', '| Area | What it does | PR | State |', '|---|---|---|---|']
if merged_nums:
    for n in sorted(merged_nums, key=rank):
        note = prs.get(str(n))
        L.append(row(areas.get(note['area'], note['area']) if note else info[n]['title'],
                     note['note'] if note else info[n]['title'], n))
else:
    del L[-2:]            # keep the heading, drop the empty table
    L.append('*(none yet — nothing of mine has been accepted upstream as of %s. When a pull request'
             ' is merged, its row moves here by itself and leaves the list above.)*' % today)
L.append('')

L += ['### Closed without merge', '| PR | Why |', '|---|---|']
if dead_nums:
    for n in sorted(dead_nums, reverse=True):
        why = closed_reason.get(str(n)) or addressed.get(n) or 'not picked up'
        for other, who in supersedes.items():
            if other == n:
                w = info.get(who, {})
                when = w.get('closed') or w.get('merged') or ''
                why = 'superseded by #%d (%s%s)' % (who, w.get('state', 'open'),
                                                    ' on ' + when if when else '')
        L.append('| %s | %s — %s%s |'
                 % (pr_link(n), info[n]['title'], why,
                    (', ' + addressed_where(n)) if n in addressed else ''))
else:
    del L[-2:]            # keep the heading, drop the empty table
    L.append('*(none — nothing of mine has been closed without merge as of %s)*' % today)
L.append('<!-- END GENERATED -->')
block = '\n'.join(L)

# ---- the front page block of README.md ---------------------------------------------------
def num(n):
    return format(int(n), ',').replace(',', ' ')

def fork_size():
    """The size of this fork against upstream, from git alone: a file count is never typed again.

    The commits are counted without merges, so a sync with upstream cannot inflate them, and
    without the commits that only refresh the two generated pages: publishing the page is not
    work, and a headline number that moves every time it is published is never true of anything.
    The two generated pages are excluded by name: the fork claims no authorship of prose it
    regenerates, and a headline that moved every time the pages were refreshed would measure nothing.
    """
    keep = ['--', '.'] + [':(exclude)%s' % p for p in NOT_FORK_WORK]
    names = [l for l in git('diff', '--name-only', '%s...HEAD' % REF, *keep).splitlines() if l.strip()]
    adds = dels = 0
    for line in git('diff', '--numstat', '%s...HEAD' % REF, *keep).splitlines():
        p = line.split('\t')
        if len(p) > 1:
            adds += int(p[0]) if p[0].isdigit() else 0
            dels += int(p[1]) if p[1].isdigit() else 0
    commits = git('rev-list', '--count', '--author=%s' % author, '--no-merges', '%s...HEAD' % REF,
                  '--', '.', ':(exclude)README.md', ':(exclude)docs/FORK.md')
    return len(names), adds, dels, (commits or '?')

def wrap(text, width=100):
    """A generated sentence, wrapped like the prose around it; a bullet keeps its indent."""
    bullet = text.startswith('- ')
    return textwrap.fill(text, width=width, subsequent_indent='  ' if bullet else '')

def quote(text, width=98):
    """A generated sentence set as a blockquote, so every one of its lines carries its '> ' marker."""
    return textwrap.fill(text, width=width, initial_indent='> ', subsequent_indent='> ')

def sections():
    """docs/fork-features.md, split on its own directives: intro, each feature, closing."""
    out, key, buf = {}, None, []
    for line in open(features_file, encoding='utf-8').read().splitlines():
        s = line.strip()
        if s == '<!-- end -->':
            if key:
                out[key] = '\n'.join(buf).strip('\n')
            key, buf = None, []
        elif key is None and s.startswith('<!--') and s.endswith('-->'):
            key, buf = s[5:-3].strip(), []
        elif key:
            buf.append(line)
    if key:
        out[key] = '\n'.join(buf).strip('\n')
    return out

def state_phrase(n):
    e = info[n]
    if e['state'] == 'merged':
        return '**Merged** ![merged](%s) on %s%s' % (GREEN, e['merged'], in_release(e))
    if e['state'] == 'closed':
        return 'closed without merge on %s' % e['closed']
    return 'open, not accepted'

def merged_phrase(nums):
    """'on 2026-11-10 in v19.02' — the dates of a feature's acceptances and the releases they went into."""
    dates = ', '.join(sorted({info[n]['merged'] for n in nums}))
    vs = versions(nums)
    return 'on %s in %s' % (dates, vs) if vs else 'on %s, not in a release yet' % dates


def feature_bullets(f):
    """The bullets the front page writes for itself: the size of the change, the doc, the state."""
    out = []
    if f.get('stat'):
        files = sum(info[n]['files'] for n in f['prs'])
        adds = sum(info[n]['add'] for n in f['prs'])
        which = ' · '.join('PR #%d' % n for n in f['prs'])
        out.append(wrap('- %s files, +%s lines%s (%s)'
                        % (num(files), num(adds), f.get('stat_extra', ''), which)))
    if f.get('doc_note'):
        out.append(wrap('- ' + f['doc_note']))
    links = []
    for n in f['prs']:
        short = prs.get(str(n), {}).get('short', '')
        links.append('**%s**%s' % (pr_link(n, '[PR #%d](%s)' % (n, info[n]['url'])),
                                   (' (%s)' % short) if short else sup(n)))
    states = {info[n]['state'] for n in f['prs']}
    if len(states) == 1:
        st = states.pop()
        if st == 'merged':
            # every pull request of the feature is accepted: name the releases they went into once,
            # and draw the conclusion the reader came for
            tail = ', so this fork is not needed for it' \
                if all(info[n]['release'] for n in f['prs']) else ''
            out.append(wrap('- upstream: %s — **Merged** ![merged](%s) %s%s'
                            % (' and '.join(links), GREEN, merged_phrase(f['prs']), tail)))
        elif st == 'open':
            out.append(wrap('- upstream: %s — %s, so today this needs the fork'
                            % (' and '.join(links), state_phrase(f['prs'][0]))))
        else:
            out.append(wrap('- upstream: %s — %s — the feature lives in this fork'
                            % (' and '.join(links), state_phrase(f['prs'][0]))))
        return out
    for n, link in zip(f['prs'], links):
        out.append(wrap('- upstream: %s — %s' % (link, state_phrase(n))))
    return out

sec = sections()
files_n, adds_n, dels_n, commits_n = fork_size()
claim = notes.get('front_claim_sentence',
                  'Everything general enough is submitted upstream, which means')
def commit_link(short, full):
    """A commit the way the page names it: the short sha, clickable on the upstream repository."""
    return '[`%s`](https://github.com/%s/commit/%s)' % (short, up, full)


# The sentence names two commits when it may: the one this branch is merged with, and — once the fetched
# upstream ref sits past it — the one upstream has got to. A reader who cannot see the distance cannot
# tell a synced branch from a stale one, and a sync is worth asking about while it is due. The ref is a
# local one, so the distance is the one the last fetch or `git sync-upstream` left behind.
synced = 'synced with upstream %s%s' % (
    commit_link(tip, sync), ' (WanGP v%s)' % tip_version if tip_version else '')
if behind:
    head_tip = git('rev-parse', '--short', REF)
    head_full = git('rev-parse', REF)
    head_version = upstream_version(head_full)
    synced += '; upstream is %s commit%s further on at %s%s' % (
        num(behind), '' if behind == 1 else 's', commit_link(head_tip, head_full),
        ' (WanGP v%s)' % head_version if head_version else '')


def release_claim():
    """What the front page may say about official releases, feature by feature.

    The claim used to be about the fork as a whole, which is true only until upstream accepts the
    first pull request: from then on the accepted feature is in a release and the rest are not. So
    the sentence is counted feature by feature, from the upstream version each acceptance went into.
    """
    def feats(n):
        return '%d %s' % (n, 'feature' if n == 1 else 'features')
    shipped = [f for f in features if f['prs'] and all(info[n]['release'] for n in f['prs'])]
    left = len(features) - len(shipped)
    if not shipped:
        return 'the %s below are **not in an official WanGP release today**' % feats(len(features))
    vs = versions([n for f in shipped for n in f['prs']])
    if not left:
        return '**all %s below are in an official WanGP release** (%s and later)' % (
            feats(len(features)), vs)
    return ('**%d of the %s below are in an official WanGP release** (%s and later), '
            '**%s %s not**' % (len(shipped), feats(len(features)), vs, feats(left),
                               'is' if left == 1 else 'are'))

SHIELDS_OPEN = ('![upstream PRs open](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2F'
                'api.github.com%2Fsearch%2Fissues%3Fq%3Drepo%3Adeepbeepmeep%2FWanGP%2Bis%3Apr%2B'
                'author%3Aabencq%2Bis%3Aopen&query=%24.total_count&label=upstream%20PRs%20open&color=blue)')
SHIELDS_MERGED = ('![upstream PRs merged](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2F'
                  'api.github.com%2Fsearch%2Fissues%3Fq%3Drepo%3Adeepbeepmeep%2FWanGP%2Bis%3Apr%2B'
                  'author%3Aabencq%2Bis%3Amerged&query=%24.total_count&label=upstream%20PRs%20merged&color=blue)')

F = ['<!-- BEGIN GENERATED (front page) by scripts/fork-pr-table.sh -- do not edit between the markers;',
     '     the words are docs/fork-features.md and scripts/fork-pr-notes.json, the size comes from',
     "     git and the states from the GitHub API. Everything on this page is this fork's own: upstream",
     '     writes its own landing page upstream and this repository copies none of it, so the upstream',
     '     commit the branch is merged with is named below, and it carries the only copy that can be',
     '     trusted to say what upstream said then -->',
     '# Wan2GP — unofficial fork of WanGP', '',
     '[![fork of WanGP](https://img.shields.io/badge/fork_of-WanGP-blue)](https://github.com/deepbeepmeep/WanGP)',
     '![synced with upstream](https://img.shields.io/badge/synced_with_upstream-%s-orange)' % tip,
     SHIELDS_OPEN, SHIELDS_MERGED, '', sec['intro'], '',
     wrap('The `main` branch here is an integration branch: current upstream, plus my own work — '
          '**%s files, +%s lines**, %s authored commits, %s. %s %s.'
          % (num(files_n), num(adds_n), commits_n, synced, claim.rstrip('.'), release_claim())), '']
def wip_callout(f):
    """The line a section carries while its feature is still being reworked in this fork.

    The state sentence closing a section is about upstream; this one is about how far this fork's own copy
    can be trusted today, so it goes under the heading, where the reader meets it before using the feature.
    A feature upstream has taken whole is no longer this fork's to call unfinished, so the line goes with
    the acceptance; a rework that merely ends is retired from scripts/fork-pr-notes.json like any note.
    """
    if all(info[n]['state'] == 'merged' for n in f['prs']):
        return []
    wips = [prs[str(n)]['wip'] for n in f['prs'] if prs.get(str(n), {}).get('wip')]
    if not wips:
        return []
    return [quote('**Status.** This feature is **being reworked**: %s. It is **not verified against current '
                  'upstream** (`%s`), so treat it as this fork\'s development branch rather than a released '
                  'feature; where it stands upstream is given at the end of this section.'
                  % (', '.join(wips), tip)), '']

for f in features:
    F += ['## ' + f['title'], '']
    F += wip_callout(f)
    for ln in sec.get('feature: ' + f['id'], '').splitlines():
        s = ln.strip()
        if s == '<!-- image -->':
            if 'image' in f:
                F.append('![%s](%s)' % (f.get('alt', f['title']), f['image']))
        elif s == '<!-- bullets -->':
            bullets = feature_bullets(f)
            if bullets and F and F[-1] == '' and len(F) > 1 and F[-2].startswith('- '):
                F.pop()           # keep a bullet list tight
            F += bullets
        else:
            F.append(ln)
    F.append('')

claimed = {n for f in features for n in f['prs']}
# A pull request noted as addressed upstream that the API still calls open (or merged) is a note ahead of
# the truth: the page may not claim a fix upstream has not written, so it is left out and said out loud.
for n in sorted(addressed):
    if n in info and info[n]['state'] != 'closed':
        print('fork-pr-table: #%d is noted as addressed upstream but reads %s on the API: the note is ahead'
              % (n, info[n]['state']), file=sys.stderr)
addressed_nums = [n for n in sorted(addressed) if n in info and info[n]['state'] == 'closed']
F += ['## Also in this fork', '', '| Feature | What it does | Upstream pull request |', '|---|---|---|']
for n in [x for x in sorted(open_nums, key=rank) if x not in claimed] \
        + [x for x in sorted(merged_nums, key=rank) if x not in claimed] \
        + [x for x in sorted(addressed_nums, key=rank) if x not in claimed]:
    note = prs.get(str(n)) or {}
    title = note.get('short') or '**add a short title in scripts/fork-pr-notes.json** (%s)' % info[n]['title']
    what = note.get('note') or info[n]['title']
    link = pr_link(n)
    if info[n]['state'] == 'merged':
        state = link + ' **Merged** ![merged](%s) %s' % (GREEN, info[n]['merged'])
    elif n in addressed_nums:
        state = link + (' **addressed upstream, not merged** ![addressed upstream](%s) %s'
                        % (AMBER, addressed_where(n)))
    else:
        state = link + ' open'
    F.append('| %s | %s | %s |' % (title, what, state))
F += ['', wrap('The table lists what is **open upstream**, what upstream has **accepted** — the latter as '
               '**Merged** ![merged](%s) — and the pull requests upstream answered by writing the same fix '
               'itself, which are closed rather than left hanging as **addressed upstream, not merged** '
               '![addressed upstream](%s), named by the upstream commit and the WanGP release that carry '
               'that copy, or, when upstream wrote it inside a larger change of its own, by that change. '
               'Where the request itself is closed its number is struck through — the request is dead even '
               'where the fix is not — and the link stays clickable, because why upstream closed it is '
               'written on the page it links to. '
               'Any other pull request closed without merging is not a feature of this fork, so '
               'it is not listed here; [docs/FORK.md](docs/FORK.md) keeps those with their reason, as the '
               'licence record asks, and it is regenerated from the GitHub API on every publish.'
               % (GREEN, AMBER)), '']
F += [sec['closing'], '']
F += ['<!-- front page: regenerated %s · %s files +%s lines · %s authored commits · %d open, %d merged upstream -->'
      % (today, files_n, adds_n, commits_n, len(open_nums), len(merged_nums)),
      '<!-- END GENERATED -->']
front_block = '\n'.join(F)

# One change is described once on the front page — in a feature section, in a row of the table, or in the
# licence note, not in two of them. Two descriptions of one pull request are two sentences written at
# different times that can disagree, and the reader cannot tell which one the fork still stands behind.
mentioned = collections.Counter(re.findall(r'/pull/(\d+)', front_block))
twice = sorted(n for n, times in mentioned.items() if times > 1)
if twice:
    print('fork-pr-table: %s described twice on the front page: one change, one description'
          % ' and '.join('#%s' % n for n in twice), file=sys.stderr)

def sync_paragraph():
    """The sentence of the licence record that names the commit this fork sits on.

    Section 4.1 of the licence asks for a dated statement of the modifications, and that statement is
    only true of the upstream commit it names. It used to be typed by hand, and it drifted: a sync that
    moved the branch left the record naming a commit the branch had left behind. So the sentence is
    written here, from git, between the inline markers in docs/FORK.md.
    """
    base = 'The branch is currently synced with upstream `%s`' % tip
    if tip_version:
        base += ' (WanGP v%s)' % tip_version
    return wrap(base + ', and the tables on this page are regenerated from the GitHub API on each '
                       'publish.')


def rewrite_sync(path, cur):
    """Replace the generated sentence inside the hand-written part of a file, between the inline markers."""
    i, j = cur.find(SYNC_BEGIN), cur.find(SYNC_END)
    if i < 0 and j < 0:
        return cur                                   # a file without the markers is left as it is
    if i < 0 or j < 0 or j < i:
        sys.exit('fork-pr-table: %s has one of the two %s markers, or they are the wrong way round'
                 % (os.path.relpath(path, root), SYNC_BEGIN))
    return cur[:i + len(SYNC_BEGIN)] + '\n' + sync_paragraph() + '\n' + cur[j:]


def rewrite(path, new_block):
    cur = open(path, encoding='utf-8').read()
    i, j = cur.find(BEGIN), cur.find(END)
    if i < 0 or j < 0:
        sys.exit('fork-pr-table: the BEGIN/END GENERATED markers are missing from %s'
                 % os.path.relpath(path, root))
    if path == front and cur[:i].strip():
        sys.exit('fork-pr-table: the front page block must start at the top of README.md')
    if path == front:
        # The front page is written whole, and nothing follows the block: upstream's own page used to sit
        # below it, reconciled by hand at every sync, and no copy of it is kept in this repository now.
        if cur[j + len(END):].strip():
            print("fork-pr-table: what is below the block of README.md is dropped: upstream writes its "
                  'own landing page upstream, and this fork copies none of it', file=sys.stderr)
        return cur, cur[:i] + new_block + '\n'
    tail = cur[j + len(END):].lstrip('\n')
    while tail.startswith(END):
        tail = tail[len(END):].lstrip('\n')     # the block brings its own END
    return cur, rewrite_sync(path, cur[:i] + new_block + '\n\n' + tail)

def clone_reminder():
    """A fresh clone has no hook, no alias and no skill link until scripts/install-fork-hooks.sh ran.

    The generator is the first thing anyone runs in a new clone, so it is the right place to say so:
    the pre-push reminder cannot speak before it is installed.
    """
    missing = []
    if git('config', '--get', 'core.hooksPath') != 'scripts/git-hooks':
        missing.append('core.hooksPath')
    for a in ('publish-main', 'fork-refresh', 'sync-upstream'):
        if not git('config', '--get', 'alias.' + a):
            missing.append('alias.' + a)
    skills = os.path.expanduser(os.environ.get('SKILLS_DIR', '~/.config/Code/User/skills'))
    if os.path.isdir(skills) and not os.path.islink(os.path.join(skills, 'fork-publish')):
        missing.append('the fork-publish skill')
    return missing

stale = []
wrote = []
for path, new_block in ((ledger, block), (front, front_block)):
    cur, new = rewrite(path, new_block)
    if new == cur:
        continue
    if mode == 'check':
        stale.append(os.path.relpath(path, root))
    else:
        open(path, 'w', encoding='utf-8').write(new)
        wrote.append(os.path.relpath(path, root))
        print('fork-pr-table: %s rewritten' % wrote[-1])
pages = 'README.md and docs/FORK.md'
if mode == 'check':
    if stale:
        sys.exit('fork-pr-table: out of date: %s — run scripts/fork-pr-table.sh' % ', '.join(stale))
    print('fork-pr-table: %s are up to date' % pages)
elif not wrote:
    print('fork-pr-table: %s are already up to date' % pages)

missing = clone_reminder()
if missing:
    print('fork-pr-table: this clone is missing %s — run scripts/install-fork-hooks.sh'
          % ', '.join(missing))
PY
