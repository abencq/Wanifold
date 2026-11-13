<!-- BEGIN GENERATED (front page) by scripts/fork-pr-table.sh -- do not edit between the markers;
     the words are docs/fork-features.md and scripts/fork-pr-notes.json, the size comes from
     git and the states from the GitHub API. Everything on this page is this fork's own: upstream
     writes its own landing page upstream and this repository copies none of it, so the upstream
     commit the branch is merged with is named below, and it carries the only copy that can be
     trusted to say what upstream said then -->
# Wan2GP — unofficial fork of WanGP

[![fork of WanGP](https://img.shields.io/badge/fork_of-WanGP-blue)](https://github.com/deepbeepmeep/WanGP)
![synced with upstream](https://img.shields.io/badge/synced_with_upstream-38758d3c-orange)
![upstream PRs open](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.github.com%2Fsearch%2Fissues%3Fq%3Drepo%3Adeepbeepmeep%2FWanGP%2Bis%3Apr%2Bauthor%3Aabencq%2Bis%3Aopen&query=%24.total_count&label=upstream%20PRs%20open&color=blue)
![upstream PRs merged](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.github.com%2Fsearch%2Fissues%3Fq%3Drepo%3Adeepbeepmeep%2FWanGP%2Bis%3Apr%2Bauthor%3Aabencq%2Bis%3Amerged&query=%24.total_count&label=upstream%20PRs%20merged&color=blue)

**This is a personal fork of [WanGP](https://github.com/deepbeepmeep/WanGP) by
[DeepBeepMeep](https://github.com/deepbeepmeep)**. It started on 2025-04-16, when the upstream project was
still named Wan2GP, and this repository keeps that name. The app, its name, its docs, its models and its
licence are the upstream project's, and upstream is the canonical project: see [wangp.ai](https://wangp.ai).
This repository is **not affiliated with, sponsored or endorsed** by the upstream author.

**Tested under Linux.** This fork is run and tested on Linux, and on no other platform. Upstream's
Windows and macOS paths are upstream's and I do not test them, so a problem that shows itself only there
is a report I cannot reproduce. It reaches the work I submit upstream too: the frame generation worker
here builds natively for Linux instead of running the Windows binaries under Wine, and the SpargeAttn
build it offers is a Linux source build — those pull requests are Linux-specific by what they are, not by
oversight.

The `main` branch here is an integration branch: current upstream, plus my own work — **115 files,
+17 089 lines**, 138 authored commits, synced with upstream
[`38758d3c`](https://github.com/deepbeepmeep/WanGP/commit/38758d3c73366422efcf609abb63cab8385f3552)
(WanGP v19.292). Everything general enough is submitted upstream, which means the 3 features below
are **not in an official WanGP release today**.

## Latent saving, decoding and branching

> **Status.** This feature is **being reworked**: refactored, improved and fixed. It is **not
> verified against current upstream** (`38758d3c`), so treat it as this fork's development branch
> rather than a released feature; where it stands upstream is given at the end of this section.

**What it is.** Keep the latents a generation produced, saved as a companion latent file along the
traditional video and audio output, and reuse them: decode them again, or continue from them.
Wan 2.1/2.2, LongCat, LTX-2, MiniMax H3, Magi Human and Ovi.

**Why it is worth having.** The diffusion is the expensive part, and WanGP discards the internal
latent representation of the video and audio once the compressed output file is written, so a
continuation can only start from that file. With the latent file on disk, the same generation can
be re-decoded with different output choices, without running the diffusion again: the model work
is spent once and the output can be tuned. And a shot you like stops being a one-off: the
*Continue Video* and *Continue Last Video* menus already start a new generation from the video,
and the latent file is their second starting point, the exact latent the model ended on. From
the video, the branch re-encodes the part you keep, and the quality steps down a little with every
continuation built on the last. From the latent file, the kept part comes out at the quality of the
original generation, seed and work preserved. For LTX2, MiniMax H3, Magi Human and Ovi the file
also holds the audio latent, so the audio is re-decoded with it too.

![A generation, and its latent file: re-decode it, or continue at the quality of the original](docs/images/fork/latent-flow.svg)

The latent file is a plain `torch.save` payload, so a generation you did not plan for
is scriptable too:

```
my_shot.mp4
my_shot_latent.pt        python -c "import torch; d = torch.load('my_shot_latent.pt'); print(d.keys())"
```

- **Save Latents:** the new config option makes Wan, LongCat, LTX2, MiniMax H3, Magi Human and Ovi
  save the latent right before the final decode, as a `*_latent.pt` file next to each video.
  Sliding-window videos save one latent per window, and LTX2, MiniMax H3, Magi Human and Ovi
  also save their audio latent, so the file holds the video the way the model saw it. A
  *Save Latents* choice in the *Misc.* tab of the generate form overrides the option for a single
  generation: *On* writes the latent file, *Off* skips it, and the default *Configuration* option
  follows the *Configuration* tab, which is also what Deepy and API jobs use. The choice is stored
  with the generation's settings, so it keeps its value across form refreshes.

- **Latent Decode:** the new process in *Media Flow* takes a video, finds its `*_latent.pt` file,
  and turns the file back into video without loading the diffusion model. For LTX2, MiniMax H3,
  Magi Human and Ovi it also rebuilds the soundtrack and puts it back into the video.

- **From WanGP latent file:** a new option in *Continue Video* and *Continue Last Video* starts the
  video generation from the saved latent file instead of re-encoding the part you keep. The file is
  picked in the new file control, or it is detected from the video. The branch goes ahead from the
  frames recorded in the file, so it works on a machine that no longer has the video. The *keep
  frames* field becomes the branch point, and the part you keep comes out at the quality of the
  original generation. You can keep branching from the same file. The frame count keeps the classic
  continuation's meaning: it is the new segment generated after the cut, and the job log prints the
  planned result (kept, new, total) before the first window generates. If the length recorded in
  the file ever disagrees with the video, *Latent Decode* still works, but a branch asks you to
  re-run the original generation first. The kept part of a branch is rebuilt from the latent file and
  never copied from the video — the video is only a viewing copy of what the file holds — so when
  that file cannot be decoded for the job, the branch stops and says why instead of continuing on
  the video's frames. Run the job without the latent file to continue from the video.

- **Not covered:** the older LTX (0.9 / 1.0) builds its video in two stages, and its decode may add
  noise, so a re-decode would only be an approximation of the original. Hunyuan is not covered,
  because a branch on its image-driven models would not keep the starting image exactly.

- 26 files, +8 433 lines, `shared/utils/latent_io.py` alone being 2 276 of them (PR #2441)
- the settings are documented in [docs/SETTINGS.md](docs/SETTINGS.md)
- upstream: **[PR #2441](https://github.com/deepbeepmeep/WanGP/pull/2441)** — open, not accepted, so
  today this needs the fork

## Output filenames you control

**What it is.** One default filename template per model, honoured at generation time and checked before the
generation starts.

**Why it is worth having.** A folder of `ltx2_1280x720_1234.mp4` says nothing a week later. The template writes
in what produced the file, using the *effective* output values: a task upsampled to 1440p is named 1440p, not
the resolution you clicked. A template that cannot be honoured is reported when you generate, not halfway
through the video.

![The template keywords and the filename they produce](docs/images/fork/filename-anatomy.svg)

- 20 keywords — `{model}`, `{date(…)}`, `{width}`, `{height}`, `{fps}`, `{scale}`, `{steps}`, `{seed}`,
  `{prompt(40)}`, `{teacache}`, `{temporal_upsampling}` … — and any of them takes a Python format spec,
  `{fps:.0f}`, `{scale:.2f}`.

- upstream: **[PR #2292](https://github.com/deepbeepmeep/WanGP/pull/2292)** (the default filename)
  and **[PR #2290](https://github.com/deepbeepmeep/WanGP/pull/2290)** (the keywords, the format
  specs, the precheck) — open, not accepted, so today this needs the fork

## Linux DLSS: x3–x6, with native Frame Generation

**What it is.** Temporal upscaling beyond x2 on Linux — x3/x4 on RTX 40 and RTX 50, up to x6 on RTX 50 — plus
Frame Generation whose worker builds natively on Linux: no Windows, no Wine.

**Why it is worth having.** On Linux the *Temporal Upsampling* menu stopped at x2, and the frame-generation
binaries existed only for Windows, so the same graphics cards gave smooth high-resolution output under Windows
and not under Linux. They do now, from the same menu, and on a two-GPU host the run prints which card makes the
frames.

| | upstream | this fork |
|---|---|---|
| temporal upscaling on Linux | up to x2 | x3, x4, up to x6 on RTX 50 |
| frame generation on Linux | Windows binaries only | built natively, `scripts/build_dlssg_worker_linux.sh` |
| the DLSS SDK runtime files | fetched by hand | `scripts/install_dlss5.sh` |
| a saved choice in the menu | reset by the model menus | kept |
| a two-GPU host | the worker follows noone | prints the card that makes the frames |

- the worker is `scripts/dlssg_worker_linux/`, 1 196 lines of C++ (a Vulkan port, MIT, notice kept), built on
  your machine because the NVIDIA runtime licence forbids redistributing it.

- native Linux frame generation is a normal choice again after the v18.21 merge pinned it to x2
  behind a switch set before starting. WanGP lists the Linux worker as its own entry in the
  *Temporal Upsampling* dropdown, next to the upstream worker and named with the multipliers it
  supports on the current card, for instance, `DLSS Frame Generation (x2-x6 supported)`. Both
  workers can sit in one install, an entry that cannot run says so in the same bracket, and x2 is
  listed once when the upstream worker already offers it.

- upstream: **[PR #2535](https://github.com/deepbeepmeep/WanGP/pull/2535)** — open, not accepted, so
  today this needs the fork

## Also in this fork

| Feature | What it does | Upstream pull request |
|---|---|---|
| SpargeAttn source build | SpargeAttn builds from source on Linux with CUDA 13 for the Python versions the prebuilt cp311 wheel does not cover; upstream wheel stays | [#2284](https://github.com/deepbeepmeep/WanGP/pull/2284) open |
| LTX-2 compilation fix | Keep LTX-2 on CUDA Graph and stop the minutes-long first steps of PyTorch compilation; the Inductor cache of a session is kept and its shapes capped. The copy in this fork carries the measured RTX 5090 timings in docs/MODELS.md; its code lines are identical | [#2551](https://github.com/deepbeepmeep/WanGP/pull/2551) open |
| Sol-Attn for LTX-2 | Optional Sol-Attn sparse self-attention for native LTX-2 inference (experimental) | [#2289](https://github.com/deepbeepmeep/WanGP/pull/2289) open |
| Prompt enhancer on GGUF | Prompt enhancement keeps loading when Auto cannot size the drafter: where WanGP cannot read the drafter, Auto decides on the model's own size instead of the prompt enhancer failing to start | [#2553](https://github.com/deepbeepmeep/WanGP/pull/2553) open |
| Console time stamps | Time stamps at the start of the console lines WanGP prints, on the lines the terminal draws only | [#2568](https://github.com/deepbeepmeep/WanGP/pull/2568) open |
| Blank Source Video | Video continuation task with an empty Source Video dividing the frame count by zero after the model loaded, instead of asking for the file | [#2639](https://github.com/deepbeepmeep/WanGP/pull/2639) open |
| Sliding window up to 999 frames | Sliding Window sizes up to 999 frames for every video model | [#2288](https://github.com/deepbeepmeep/WanGP/pull/2288) open |
| x264 MJPEG codec | x264 MJPEG (intra-frame-only) video codec option | [#2287](https://github.com/deepbeepmeep/WanGP/pull/2287) open |
| “Continue Video” joins | “Continue Video” audio joins when the soundtrack ends early | [#2478](https://github.com/deepbeepmeep/WanGP/pull/2478) open |
| Helper programs and the terminal | ffmpeg, ffprobe and the other helper programs WanGP starts are given no keyboard and a process group of their own, so a program that reads the keyboard cannot leave the shell prompt dead | [#2640](https://github.com/deepbeepmeep/WanGP/pull/2640) open |
| Live shell after Ctrl+C | Shell prompt left dead after Ctrl+C in a WanGP run | [#2526](https://github.com/deepbeepmeep/WanGP/pull/2526) open |
| Ovi Audio Guidance | Audio Guidance setting ignored by Ovi generations, which ran at a fixed 4.0 | ~~[#2600](https://github.com/deepbeepmeep/WanGP/pull/2600)~~ **addressed upstream, not merged** ![addressed upstream](https://img.shields.io/badge/addressed%20upstream-orange) in WanGP v19.25 [`0e0a06f6`](https://github.com/deepbeepmeep/WanGP/commit/0e0a06f6) |
| Empty-audio crash | Crash when saving audio that came out empty | ~~[#2482](https://github.com/deepbeepmeep/WanGP/pull/2482)~~ **addressed upstream, not merged** ![addressed upstream](https://img.shields.io/badge/addressed%20upstream-orange) in WanGP v18.19 [`16bab9e2`](https://github.com/deepbeepmeep/WanGP/commit/16bab9e2) |
| Silent startup | SyntaxWarning printed at startup by the About tab credits line | ~~[#2565](https://github.com/deepbeepmeep/WanGP/pull/2565)~~ **addressed upstream, not merged** ![addressed upstream](https://img.shields.io/badge/addressed%20upstream-orange) in upstream's own 3D update of 2026-11-03 |

The table lists what is **open upstream**, what upstream has **accepted** — the latter as **Merged**
![merged](https://img.shields.io/badge/merged-brightgreen) — and the pull requests upstream answered
by writing the same fix itself, which are closed rather than left hanging as **addressed upstream,
not merged** ![addressed upstream](https://img.shields.io/badge/addressed%20upstream-orange), named
by the upstream commit and the WanGP release that carry that copy, or, when upstream wrote it inside
a larger change of its own, by that change. Where the request itself is closed its number is struck
through — the request is dead even where the fix is not — and the link stays clickable, because why
upstream closed it is written on the page it links to. Any other pull request closed without merging
is not a feature of this fork, so it is not listed here; [docs/FORK.md](docs/FORK.md) keeps those
with their reason, as the licence record asks, and it is regenerated from the GitHub API on every
publish.

**Install:** as upstream — `git clone https://github.com/abencq/Wan2GP.git && cd Wan2GP && python wgp.py`
(Windows: `run.bat`, which I have not tested). Your models, settings and outputs are laid out as upstream
puts them, so switching either way keeps them.

> Per section 4.1 of the [WanGP Community License 2.0](LICENSE.txt), this fork states its modifications and
> their dates in **[docs/FORK.md](docs/FORK.md)**, and keeps the upstream licence, the in-tree notices and the
> [third-party licence texts](docs/third_party_licenses/).

<!-- front page: regenerated 2026-11-13 · 115 files +17089 lines · 138 authored commits · 15 open, 0 merged upstream -->
<!-- END GENERATED -->
