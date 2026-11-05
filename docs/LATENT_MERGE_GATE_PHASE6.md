# Phase 6 — Merge Gate: latent merge into the fork

## Phase 6 — Merge Gate: results

### Merge
- `pr-latent` merged into `local` as `4d57fbf6` (5 files, +3847/−3426; `wgp.py` −3591 lines — the latent engine moved to `shared/utils/latent_io.py`, `latent_jobs.py`, `latent_ui.py`).
- 28 conflict hunks in `wgp.py` (all pure refactor regions; `pr-latent` side taken) + `plugins/configuration/plugin.py` (rerere auto-resolve).
- Fork-specific lines verified present post-merge: `precheck_output_filename`, extended `scale/width/height/fps` filename-template inputs, 6-arg `audio_overlap_repair_seconds`, fork fps/outpainting/sliding-window ranges, the extended Output-Filename block with saved-template chain, `[WanGP]` console prefix.
- `diff local pr-latent -- wgp.py` contains only the fork additions (no lost refactor hunks).

### Tests (headless, merged tree)
- `tests/test_latent_io.py` + `tests/test_latent_jobs.py` (updated/copied from `pr-latent`): **87/87 pass**.
- Full non-latent console suite: **68/68 pass**.

### Smoke tests (live, GPU 1, `--process` CLI driver)
Model set: Wan 2.1 14B T2V int8 + 1.3B T2V (transformer/VAE/umT5 in `ckpts/`).

1. **T2V + Save Latents (1.3B, 832x480, 33f, 10 steps, seed 42)** — `phase6_A_*.mp4` + `*_latent.pt` sidecar written next to the video; window VAE decode 45 s; queue 1/1 in 16 m 01 s (machine was heavily loaded; generation window < 2 min). Sidecar verified: per-window `prompt`/`seed`/`frame_count`/`head_trim`/`tail_trim`, VAE id + sha256, stride 4, base `t2v_1.3B`.
2. **Latent Decode (media-flow `edit_latent`)** — `phase6_A_*_latent_decode.mp4`, 33/33 frames, decoded from the sidecar in 5.8 s (no diffusion model loaded).
3. **Branch job from latent** — see findings below.
4. **Queue save/load round-trip with `branch_latent_file`** — `wgp._save_queue_to_zip` → `queue.json` manifest + embedded `task*_branch_latent_file_*.pt` + companion video; `--process <zip> --dry-run` loads, extracts the sidecar to the media cache, and validates `[OK]`.
5. **Deepy/API job** — `plugin_data.api` via zip manifest (`return_video_uint8`), `client_id` artifact assertion.

### Finding: branch jobs need a Continue-capable model
`t2v_1.3B` has `image_prompt_types_allowed = ""` (no V/L Continue location), so `fix_settings` clears `image_prompt_type` and the branch gate in `generate_media` (`filter_letters(image_prompt_type, "VL")`) never engages: a settings JSON with `branch_from_latent: true` silently degrades to a plain T2V (observed in job C: 17 frames, random seed, factory prompt, no `From latent:` log lines). This matches the UI design (the *From latent* checkbox is only reachable from the Continue locations) — it is a model-capability gate, not a merge regression. The live branch test therefore runs on **Wan 2.1 T2V 14B** (`image_prompt_types_allowed = "TVL"`), whose sidecar base (`t2v`) matches the branch job's base.

### Smoke test detail (live runs, `--process` CLI driver, GPU 1)
- **Job A** — `t2v_1.3B`, 832x480, 33f, 10 steps, seed 42: `phase6_A_2026-11-05-16h04m00s_42_A_red_balloon_drifting_over_a.mp4` (1.35 MB) + `_latent.pt` sidecar (2.15 MB). Window: VAE decode 45.1 s; queue 1/1 in 16 m 01 s (machine heavily loaded by a co-tenant; the generation window itself was under 2 min).
- **Job B** — `edit_latent` (Latent Decode media flow) on job A's sidecar: `phase6_A_..._latent_decode.mp4`, 33/33 frames, 5.8 s end-to-end, no diffusion model loaded.
- **Job C** — branch settings on `t2v_1.3B`: silently ran as plain T2V (see finding). Output `phase6_C_..._415709519_..._17@16...mp4`, 17 frames, random seed — evidence for the finding, not a pass.
- **Job E** — `t2v` (Wan 2.1 14B T2V int8), 832x480, 33f, 10 steps, seed 42, `save_latents_choice: 1`: `phase6_E_2026-11-05-17h53m36s_42_832x480x33@16_scale1.mp4` (2.04 MB) + `_latent.pt` (2.20 MB). First attempt with `save_latents_choice: "yes"` wrote no sidecar — the Misc. choice is the numeric radio (0 = Configuration option, 1 = On, 2 = Off); non-numeric values fall back to 0 (unit-tested). Queue 1/1 in 12 m 48 s (≈ 12 min of it model pinning on the loaded machine; generation window ≈ 2 m 05 s + VAE decode).
- **Job F** — branch from job E's sidecar: keep 17, same prompt/seed (the UI's sidecar prefill is a form callback; headless settings carry the values explicitly), `image_prompt_type: "V"`, `save_latents_choice` unset (first run — see below). **Result: pass.**
  - All five fail-fast receipts printed before the model load, e.g. `From latent: using saved latent file … (33 frames recorded; cut at frame 17).` and the plan line `From latent: plan: kept 17 (verbatim) + 20 new (2 window(s)) = 37 frames` — the requested 17 new frames quantize to 16 (the cut window) + 4 (the tail window's 4-frame extent), so the output is **37 frames**, not 34 (window-grid rounding at the tail).
  - Final output (last window save): 37 frames. Kept region 0..16 vs job E: mean abs diff **0.70/255** (mp4 re-encode noise — the kept latents are copied unchanged, not re-extracted); new region 17..36: mean abs diff **74/255** (genuinely generated content).
  - The run wrote one mp4 per window (an intermediate 33-frame window file, then the final 37-frame stitched file); the last `Video file saved` is the job output.
  - No sidecar was written because the template left `save_latents_choice` unset (→ Configuration option → off on this machine); that is the test driver's omission, not a defect — the branch-sidecar path is unit-tested (`test_recorder_all_latent_multi_window`: carried layout, per-window prompts, branch point against the accumulated layout) and was verified live as **job F2** (same job + `save_latents_choice: 1`): the final window's sidecar records **37 committed frames** (matching the 37-frame video, no length-mismatch flag) and its branch points resolve — at frame 17 `kept_frames: 17, mid_latent: False` (the original cut) and at frame 30 `mid_latent: True` (mid-window). The sidecar chain (source sidecar → branch job → new branchable sidecar) works end-to-end.

- **Job D** — API smoke (`plugin_data.api: {return_video_uint8: true}`, client_id `phase6_api_smoke`, `t2v_1.3B` 17f): **pass** in 34.0 s (hot page cache — the T5 weights were just loaded by job F2). `process_tasks_cli` returned True; the artifact store keyed by client_id returned the job's output (`media_type: video`, `fps: 16.0`, path exists on disk) and the response carried the video as a **uint8 tensor `(3, 17, 480, 832)`**; `side_files: []` as expected (1.3B writes no audio side file and the latent sidecar is deliberately excluded from API responses).

### Findings / follow-ups
1. **Branch gate is model-capability gated (by design).** `t2v_1.3B` has `image_prompt_types_allowed = ""` (no V/L Continue location); `fix_settings` clears `image_prompt_type`, the `generate_media` branch gate (`filter_letters(image_prompt_type, "VL")`) never engages, and a headless settings file with `branch_from_latent: true` degrades silently to a plain job (job C). The UI only exposes *From latent* from the Continue locations, so this cannot happen from the form; headless (CLI/API) callers on a non-Continue model get a plain job with no warning. Consider a console note for API/CLI branch requests that are dropped by the gate.
2. **Robustness fix (applied in this phase):** `validate_settings` and `generate_media` called `len(keep_frames_video_source)`, which raises `TypeError` when a headless settings file sends the keep-frames as a JSON number (the UI text box always sends a string). Both call sites now coerce with `str(... or "")` (`wgp.py` ~1562, ~8028). Discovered by job F's dry-run.
3. **`save_latents_choice`** is the numeric radio (0/1/2); `effective_save_latents` maps non-numeric → 0 (follows the Configuration option) — covered by `tests/test_latent_io.py::test_effective_save_latents`.
4. **Sidecar layout:** prompt/seed/trim are per-window (`windows[i]`); the VAE is identified by file name + sha256; the branch family check requires the same `base_model_type` (VAE-specific latents never cross families).
5. **The latent sidecar is deliberately excluded from API response side-files** (`wgp.py` ~10138: "it is never echoed in API responses") — API jobs return other side files; the latent file is a disk companion only.
