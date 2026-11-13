## When the repository keeps its own ledger of pull requests

When the repository carries `scripts/fork-pr-notes.json`, the front page and the licence record list
every pull request on their own, from that file. The note is written in the same session as the pull
request, before the session is forgotten:

- Add the entry — `area`, `short`, `note` — on the personal branch, and add the number to `features`
  when the pull request carries a feature the front page presents.
- `note` is one sentence about what the change does for the person who runs the app, in the voice of
  a commit message. The row is read years later by someone deciding whether this fork is worth
  cloning, and by the licence record.
- `wip` marks a feature this fork is still reworking — "refactored, improved and fixed", say. The front
  page prints it as a **Status** line under that feature's own heading and the ledger repeats it in the
  row, so nobody trials a half-finished feature believing it current. Retire it when the rework ends; it
  goes on its own once upstream takes the whole feature, a shipped feature being no longer ours to call
  unfinished.
- `detail` is the long form of the same row: what the change replaces, the mechanism behind it, the
  option it adds. The ledger carries it and the front page leaves it off, so a change is described once
  on the page a reader lands on and in full where the record is kept. Reach for it instead of a second
  paragraph on the front page: two descriptions of one pull request are two sentences written at
  different times, and the reader cannot tell which one the fork still stands behind.
- Run `git fork-refresh` and commit the regenerated pages on the personal branch. A pull request
  without a note appears as a loud **TODO** row in both pages.
- A pull request that upstream closes without merging keeps its row in the ledger with its reason. It
  leaves the front page, a closed request being no feature of this fork — except the ones upstream
  answered by writing the same fix itself: those stay on the front page as **addressed upstream, not
  merged**, named by upstream's commit and the release that carries it, because "upstream has it already"
  is the answer a reader weighing this fork came for. They live in `addressed_upstream`, with the commit
  in `addressed_upstream_commit` and its release read from upstream's own version string. A fix upstream
  wrote inside a larger change of its own has no commit carrying it alone: name the change instead in
  `addressed_upstream_when`, so the row says when rather than nothing.

## What a comment on an upstream pull request says

A comment left there speaks to that repository alone, to the person who has to decide about the change.
It names only what they can check: the commit, the release whose `wgp.py` carries it, the file and the
line, the shape a function returns. It never mentions this fork's ledger, its front page, or how the
change is recorded here — where a pull request stands in this fork is this repository's own business,
and its table is where that is said.
