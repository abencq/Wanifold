# This WanGP fork

**This is a personal, unofficial fork of [WanGP](https://github.com/deepbeepmeep/WanGP), the project of
[DeepBeepMeep](https://github.com/deepbeepmeep).** The application, its name, its documentation, its models
and its licence are the upstream project's, and upstream is the canonical project — see
[wangp.ai](https://wangp.ai). This repository is not affiliated with, sponsored or endorsed by the upstream
author, and the software here is still called WanGP.

## Statement of modifications

Required by section 4.1 of the [WanGP Community License 2.0](../LICENSE.txt), which is kept unchanged in this
repository.

The `main` branch of this repository is the source of `deepbeepmeep/WanGP`, **modified by abencq**. My first
change dates from **2025-04-16**.

<!-- SYNCED -->
The branch is currently synced with upstream `c8d47da2` (WanGP v19.291), and the tables on this page
are regenerated from the GitHub API on each publish.
<!-- /SYNCED -->

The upstream licence ([LICENSE.txt](../LICENSE.txt)), the in-tree notices and the third-party licence texts
([docs/third_party_licenses/](third_party_licenses/)) are preserved, and every modification of the upstream
code is listed in the tables below, by area and by pull request. My own files carry the WanGP copyright
header, not a claim of separate ownership.

## How this fork is kept

- **`main`** (the default branch here) is an integration branch: current upstream, plus my work, merged
  from upstream as it moves. Its size against upstream — files, lines, authored commits — is counted
  from git on the [front page](../README.md), never typed here; commits that only refresh these pages
  are not counted as work.
- **`pr-*`** branches are the thin branches that carry my pull requests to upstream, each one cut from an
  upstream snapshot so that its diff is only that change.
- Anything general enough is submitted upstream as a pull request, under the project CLA. When upstream
  accepts one, the feature is removed from the list below, and the row moves to the "merged upstream"
  table on its own.
- The pages write themselves. [scripts/fork-pr-table.sh](../scripts/fork-pr-table.sh) counts this fork
  against upstream, reads the state of every pull request from the GitHub API, and assembles the tables
  below and the front page of the [README](../README.md). The wording of a row is in
  [scripts/fork-pr-notes.json](../scripts/fork-pr-notes.json), the sentences of the front page in
  [docs/fork-features.md](fork-features.md), the What's new entries of the front page in
  [docs/fork-new.md](fork-new.md). Nothing between the `BEGIN GENERATED` markers of the
  two pages is written by hand, and a pull request with no note shows up as a loud **TODO** row.
- The sentence above naming the upstream commit this fork sits on is written by that same script,
  between the inline `<!-- SYNCED -->` markers, because a statement of what was modified is only true
  of the commit it names. The commit it names is the newest upstream commit this branch carries, and
  not the tip of `origin/main`: a fetch moves the tip, a merge moves the branch, and only the second
  is a sync. So is the claim on the front page of which of these features are in an official release:
  upstream tags no release of the application, so the release an accepted pull request went into is
  the `WanGP_version` in `wgp.py` at its merge commit, and a feature counts as released only when
  every pull request behind it is accepted into one.
- [scripts/fork-table-fixture-test.sh](../scripts/fork-table-fixture-test.sh) runs the generator in a
  scratch repository against written-down answers of the GitHub API, offline: an accepted pull request,
  a shipped release, an acceptance not yet in one, a closure. Run it after changing the generator —
  `--check` proves the pages are true today, and the fixture proves they stay true in the states that
  have not happened here yet.
- Publish with `git publish-main`: look at `fork/main`, regenerate the pages, commit what the
  generator wrote (and only that: the generated block and the licence sentence, checked by
  [scripts/page-hunk-check.sh](../scripts/page-hunk-check.sh)), read the pages back from git and the
  API, push. The order is the point — a regenerated page is an uncommitted change, so pushing first
  publishes the page of the previous run. `git fork-refresh` regenerates and shows the diff, and
  pushes nothing.
- After a fresh clone, run [scripts/install-fork-hooks.sh](../scripts/install-fork-hooks.sh): it wires
  the `pre-push` reminder, the two aliases and the skill texts of `docs/fork-skills/` into the clone,
  and removes a hook left behind under `.git/hooks/`. The reminder is offline and never stops a push;
  it says when the front page and the commits disagree, when a page is older than a week, when the two
  pages carry different dates, when the words behind the page changed after the page, when the upstream
  commit named in the licence statement is not the one this fork carries, and when `origin/main` has
  moved past that commit, which is a sync to do and not a page to fix. The generator
  repeats the reminder in a clone where the installer has not run.

## Use it

The installation and the usage are the upstream ones — see the
[installation](INSTALLATION.md) and the [getting started](GETTING_STARTED.md) guides:

```bash
git clone https://github.com/abencq/Wan2GP.git
cd Wan2GP
python wgp.py            # Windows: run.bat
```

Your models, your settings and your outputs are laid out exactly as upstream puts them, so a switch between
this fork and upstream keeps them. Nothing is installed here that upstream does not already offer: the
extra installers in `scripts/` (SpargeAttn, flash-attn, DLSS runtime, the Linux DLSS Frame Generation
worker) are the ones the upstream documentation describes, made to work on Linux and CUDA 13.

## What is in this fork

<!-- BEGIN GENERATED by scripts/fork-pr-table.sh -- do not edit between the markers;
     the hand-written wording is in scripts/fork-pr-notes.json -->
*current as of 2026-11-13 · upstream `c8d47da2` · 14 open · 0 merged · 5 closed without merge*

### In this fork, submitted upstream
| Area | What it does | PR | State |
|---|---|---|---|
| Latents | Save, decode and branch latents for Wan, LongCat, LTX-2, MiniMax H3, Magi Human and Ovi; pick a latent file straight from a video | [#2441](https://github.com/deepbeepmeep/WanGP/pull/2441) | open · dirty · 14 behind |
| Output filenames | A default output filename for every model, with a template checked before the generation starts | [#2292](https://github.com/deepbeepmeep/WanGP/pull/2292) | open · clean · 39 behind |
| Output filenames | Effective-output keywords and Python format specs for the Output Filename box | [#2290](https://github.com/deepbeepmeep/WanGP/pull/2290) | open · clean · 39 behind |
| Linux DLSS | DLSS temporal upscaling at x3-x4, up to x6 on RTX 50, with the native Linux Frame Generation worker and installer | [#2535](https://github.com/deepbeepmeep/WanGP/pull/2535) | open · clean · 14 behind |
| Attention installs | SpargeAttn install on Linux with CUDA 13 for RTX 30XX-50XX | [#2284](https://github.com/deepbeepmeep/WanGP/pull/2284) | open · clean · 23 behind |
| LTX-2 | Keep LTX-2 on CUDA Graph and stop the minutes-long first steps of PyTorch compilation; the Inductor cache of a session is kept and its shapes capped. The copy in this fork carries the measured RTX 5090 timings in docs/MODELS.md; its code lines are identical | [#2551](https://github.com/deepbeepmeep/WanGP/pull/2551) | open · clean · 23 behind |
| LTX-2 | Optional Sol-Attn sparse self-attention for native LTX-2 inference (experimental) | [#2289](https://github.com/deepbeepmeep/WanGP/pull/2289) | open · clean · 29 behind |
| Prompt enhancer | Prompt enhancer crash with Auto Speculative Decoding on GGUF | [#2553](https://github.com/deepbeepmeep/WanGP/pull/2553) | open · dirty · 16 behind |
| Console | Time stamps at the start of the console lines WanGP prints, on the lines the terminal draws only | [#2568](https://github.com/deepbeepmeep/WanGP/pull/2568) | open · clean · 27 behind |
| Video | Video continuation task with an empty Source Video dividing the frame count by zero after the model loaded, instead of asking for the file | [#2639](https://github.com/deepbeepmeep/WanGP/pull/2639) | open · clean · 0 behind |
| Video | Sliding Window sizes up to 999 frames for every video model | [#2288](https://github.com/deepbeepmeep/WanGP/pull/2288) | open · clean · 37 behind |
| Video | x264 MJPEG (intra-frame-only) video codec option | [#2287](https://github.com/deepbeepmeep/WanGP/pull/2287) | open · clean · 176 behind |
| Audio | “Continue Video” audio joins when the soundtrack ends early | [#2478](https://github.com/deepbeepmeep/WanGP/pull/2478) | open · clean · 0 behind |
| Shell & UI | Shell prompt left dead after Ctrl+C in a WanGP run | [#2526](https://github.com/deepbeepmeep/WanGP/pull/2526) | open · clean · 0 behind |

### Only in this fork, not submitted
| Area | What it does |
|---|---|
| Linux DLSS | Local fixes around the DLSS entries: the Frame Generation multipliers offered on a multi-GPU host, and a saved Frame Generation choice that the menus used to reset (part of #2535) |

### Merged upstream — removed from this fork
*(none yet — nothing of mine has been accepted upstream as of 2026-11-13. When a pull request is merged, its row moves here by itself and leaves the list above.)*

### Closed without merge
| PR | Why |
|---|---|
| [#2600](https://github.com/deepbeepmeep/WanGP/pull/2600) | Fix the Audio Guidance setting ignored by Ovi — the fix is upstream's own: it renames audio_cfg_scalecale to audio_cfg_scale in the same three lines of models/wan/ovi_fusion_engine.py, so the Audio Guidance setting reaches Ovi, in WanGP v19.25 [`0e0a06f6`](https://github.com/deepbeepmeep/WanGP/commit/0e0a06f6) |
| [#2565](https://github.com/deepbeepmeep/WanGP/pull/2565) | Fix the SyntaxWarning printed at startup on the About tab credits line — the fix is upstream's own: the 3D update of 2026-11-03 rewrote that credits line with the escape |
| [#2482](https://github.com/deepbeepmeep/WanGP/pull/2482) | Fix the crash when saving audio that came out empty — the fix is upstream's own: it normalises an empty track in _prepare_audio_array to the frame-first shape this pull request gave it in write_wav_file, which leaves the guard added here unreachable, in WanGP v18.19 [`16bab9e2`](https://github.com/deepbeepmeep/WanGP/commit/16bab9e2) |
| [#2295](https://github.com/deepbeepmeep/WanGP/pull/2295) | Linux: add support for DLSS Frame Generation (temporal upscaling only) — superseded by #2535 (open) |
| [#2286](https://github.com/deepbeepmeep/WanGP/pull/2286) | Linux: build script for flash-attn 2.8.3.post1 with RTX 50xx (Blackwell) support — not picked up; the builder script stays in this fork |
<!-- END GENERATED -->

## Licence and credit

The software remains under the [WanGP Community License 2.0](../LICENSE.txt) of DeepBeepMeep; the fork adds
no further restriction, and third-party components keep their own licences. Credit for the project goes to
its author and to the contributors of the upstream repository; the features listed above are mine, and they
are offered as they are, without warranty.
