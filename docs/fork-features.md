# Front page of this fork — the words

The front page block of `README.md` is generated: the numbers come from git, the state of every pull
request comes from the GitHub API, and the words come from this file. `scripts/fork-pr-table.sh`
assembles the three. Edit this file, run `git fork-refresh`, and the front page follows.

Sections are delimited by `<!-- intro -->`, `<!-- feature: id -->` and `<!-- closing -->`, each ended by
`<!-- end -->`. Inside a feature section, `<!-- image -->` is where the diagram goes (the generator
skips it when the file is missing, so no broken image is ever published) and `<!-- bullets -->` is where
the generated bullets are appended: the size of the change, the documentation link and the state upstream.

The sentence that says how much of this fork is in an official WanGP release is not written here. It is
counted from the state of every pull request, and from the upstream release each acceptance went into,
because a typed claim of that kind goes false the day upstream accepts the first one.

<!-- intro -->
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
<!-- end -->

<!-- feature: latents -->
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

<!-- image -->

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

<!-- bullets -->
<!-- end -->

<!-- feature: filenames -->
**What it is.** One default filename template per model, honoured at generation time and checked before the
generation starts.

**Why it is worth having.** A folder of `ltx2_1280x720_1234.mp4` says nothing a week later. The template writes
in what produced the file, using the *effective* output values: a task upsampled to 1440p is named 1440p, not
the resolution you clicked. A template that cannot be honoured is reported when you generate, not halfway
through the video.

<!-- image -->

- 20 keywords — `{model}`, `{date(…)}`, `{width}`, `{height}`, `{fps}`, `{scale}`, `{steps}`, `{seed}`,
  `{prompt(40)}`, `{teacache}`, `{temporal_upsampling}` … — and any of them takes a Python format spec,
  `{fps:.0f}`, `{scale:.2f}`.

<!-- bullets -->
<!-- end -->

<!-- feature: dlss -->
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

<!-- bullets -->
<!-- end -->

<!-- closing -->
**Install:** as upstream — `git clone https://github.com/abencq/Wan2GP.git && cd Wan2GP && python wgp.py`
(Windows: `run.bat`, which I have not tested). Your models, settings and outputs are laid out as upstream
puts them, so switching either way keeps them.

> Per section 4.1 of the [WanGP Community License 2.0](LICENSE.txt), this fork states its modifications and
> their dates in **[docs/FORK.md](docs/FORK.md)**, and keeps the upstream licence, the in-tree notices and the
> [third-party licence texts](docs/third_party_licenses/).
<!-- end -->
