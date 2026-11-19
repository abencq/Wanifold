"""Parity harness: the latent branch's geometry against the classic continuation's.

Run from the repository root:

    venv/bin/python -m pytest tests/test_branch_parity.py -v

This is the branch work's golden baseline. Everything it compares is geometry -
frame counts, latent counts, window boundaries, audio sample counts - so it
needs no GPU, no weights and no ``wgp`` import. The classic side is the planner
the classic path itself calls, ``frame_scheduler.build_default_window_plan``, fed
the inputs ``wgp.generate_media`` derives for a job that continues a video
(``wgp.py:8622-8623``): the kept run as the first window's available overlap, the
family's reuse as the default overlap, and one shared frame charged for the frame
the continuation stands on. The branch side is the real seam:
``latent_jobs._prepare_branch_prefix`` followed by
``latent_jobs.plan_branch_head_window``, the pair ``wgp`` runs for a branch job.

How to read a failure:

* a branch row that no longer matches its expectation is a latent-side bug;
* a ``classic`` value that no longer matches the planner (the first test) is a
  classic-side change - escalate it, do not bend the branch or this table to it;
* the divergences this harness found are pinned one per test, by name: DIV-1 ..
  DIV-4 (plan shape, the frame grid, the legacy length, the trimmed chained cut)
  and AUD-1 / AUD-2 (the audio payload's latent layout, and what a head window's
  audio block covers). They are recorded, not approved: each test says what it
  would take to close the divergence and which side owns it.

The fixture arithmetic is the LTX2 layout's own (8/1: t latents decode to
1 + 8*(t-1) frames, a 24 fps video, the audio VAE at 25 latent steps a second of
640 samples). The family's real planner geometry is used (``frames_minimum`` 17,
``frames_steps`` 8, overlap grid offset 1, overlap max 97, window 241).
"""

import os
import pathlib
import sys
import threading

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from shared.utils import latent_branch_head, latent_io, latent_jobs
from shared.utils.frame_scheduler import (align_overlap, build_default_window_plan, continuation_source_shortfall,
                                          continuation_window_overlap, continuation_window_shortfall,
                                          job_continues_video, resolve_continuation_handoff)
from shared.utils.utils import continuation_color_reference, continuation_color_window_anchor

APP_VERSION = "17.17"
LTX2 = "ltx2_22B"
LTX2_VAE = "ltx2_vae.safetensors"
LTX2_VAE_SHA = "cd34"

STRIDE = 8                      # the LTX2 temporal layout: 8/1
FRAME_OFFSET = 1
FRAMES_MINIMUM = 17             # models/ltx2/ltx2_handler.py
FRAMES_STEPS = 8
WINDOW_SIZE = 241
OVERLAP = 17                    # 1 + 8*2, on the family's overlap grid
OVERLAP_MAX = 97                # ltx2_handler.py: sliding_window_defaults
FPS = 24.0
AUDIO_STEPS_PER_SECOND = 25     # the LTX2 audio VAE
SAMPLES_PER_AUDIO_STEP = 640
SAMPLE_RATE = 16000


def decoded_frames(latents):
    """The frames a run of ``latents`` decodes to on the 8/1 layout."""
    return 1 + STRIDE * (int(latents) - 1)


# ---------------------------------------------------------------------------
# The classic side: the continuation's own window plan, frozen
# ---------------------------------------------------------------------------

def classic_plan(request, *, window_size=WINDOW_SIZE, overlap=OVERLAP, kept,
                 steps=FRAMES_STEPS, minimum=FRAMES_MINIMUM):
    """The window plan the classic path builds for the same job continued from
    the video file (``wgp.py:8607`` and ``wgp.py:8622``): the kept run is the
    first window's available overlap, one frame of the request is the frame the
    continuation stands on, the rest is the planner's own grid arithmetic.

    The handoff is cut by the latent stride, not by the family's shortest
    generation: ``wgp.py:8607`` keeps ``min(window - latent_size, overlap)``,
    because the inherited frames live inside the window's own block and the
    window still adds up to its full length. The shortest generation bounds a run
    of *new* frames, which is a different quantity; reading it here made the
    baseline disagree with the program for any handoff left within a stride of
    the window size, which is what the boundary test pins now.
    """
    reuse = min(window_size - STRIDE, overlap)
    first_overlap = min(max(1, reuse), int(kept))
    return build_default_window_plan(
        total_frames=int(request), window_size=window_size, default_overlap=reuse,
        discard_last_frames=0, minimum=minimum, step=steps, frame_offset=FRAME_OFFSET,
        overlap_offset=1, max_overlap=OVERLAP_MAX,
        first_window_overlap=first_overlap, first_window_available_overlap=int(kept),
        initial_shared_frames=1 if first_overlap > 0 else 0)




def shape(plan):
    """A plan as (output_frames, overlap_frames) per window - the geometry rows
    of the table below."""
    return [(int(w["output_frames"]), int(w["overlap_frames"])) for w in plan]


# ---------------------------------------------------------------------------
# The branch side, headless
# ---------------------------------------------------------------------------

def stub_ctx(save_path):
    """A JobContext with dummy registry lookups (the branch paths never touch
    the real model catalog)."""
    return latent_jobs.JobContext(
        server_config={}, save_path=str(save_path), WanGP_version=APP_VERSION,
        verbose_level=0, lock=threading.Lock(), primary_settings={},
        model_types_handlers={}, transformer_quantization="", text_encoder_quantization="",
        get_model_def=lambda model_type: {},
        get_model_record=lambda *a, **k: "Latent Branch",
        get_model_config_groups=lambda *a, **k: None,
        get_base_model_type=lambda model_type: model_type,
        get_available_filename=lambda *a, **k: "",
        clear_status=lambda *a, **k: None,
        get_gen_info=lambda state: {},
        resolve_last_video=lambda state: None)


def ltx2_latents(t, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(128, t, 64, 36, generator=gen, dtype=torch.float16)


_SIDECARS = {}


def ltx2_sidecar(latent_counts, head_trims=(), fps=FPS):
    """A source sidecar as the job reads it: one clean t2v window per latent
    run on the 8/1 layout, with the given per-window head trims.

    Memoized on its arguments: building one costs a VAE-sized random tensor
    per window, and every parametrized test in this module rebuilds the same
    handful of files. The branch paths only read it (they clone what they
    stack), so the cached object is never handed out in a mutable state.
    """
    key = (tuple(latent_counts), tuple(head_trims), fps)
    if key not in _SIDECARS:
        _SIDECARS[key] = _build_ltx2_sidecar(latent_counts, head_trims, fps)
    return _SIDECARS[key]


def _build_ltx2_sidecar(latent_counts, head_trims, fps):
    blocks = []
    for i, count in enumerate(latent_counts):
        blocks.append({
            "prompt": f"window {i + 1}",
            "seed": 7,
            "steps": 40,
            "denoising_strength": 1.0,
            "head_trim": int(head_trims[i]) if i < len(head_trims) else 0,
            "tail_trim": 0,
            "frame_count": decoded_frames(count),
        })
    runs = [ltx2_latents(c) for c in latent_counts]
    return latent_io.build_latent_sidecar(
        runs, blocks, app_version=APP_VERSION, model_type=LTX2, base_model_type=LTX2,
        model_filename="ltx2.safetensors", loras=(),
        vae_file=LTX2_VAE, vae_sha256=LTX2_VAE_SHA,
        vae_z_dim=128, vae_dtype="16", width=512, height=512, fps=fps,
        latent_stride=STRIDE, frame_offset=FRAME_OFFSET, is_hdr=False)


def ltx2_block(latents, frames=None, seed=7, prompt="window 1"):
    """One window's latent payload (the model's saved pre-decode run)."""
    if frames is None:
        frames = decoded_frames(int(latents.shape[1]))
    return latent_io.build_latent_payload(
        latents, app_version=APP_VERSION, model_type=LTX2, base_model_type=LTX2,
        model_filename="ltx2.safetensors", vae_file=LTX2_VAE, vae_sha256=LTX2_VAE_SHA,
        vae_z_dim=128, vae_dtype="16", frames=frames, fps=FPS, seed=seed, prompt=prompt,
        denoising_strength=1.0, steps=40, latent_stride=STRIDE, frame_offset=FRAME_OFFSET)


MEL_BINS = 16                   # the LTX2 audio VAE's mel bins per latent step


def _audio_values(rows, cols, kind="randn"):
    """The latents the harness writes as a window's audio.

    ``"randn"`` is enough for the geometry rows. ``"ramp"`` encodes each latent
    time step in its own mel frames - every column of step ``j`` carries the
    value ``j`` - so a slice of a stored block can be read back as the steps it
    came from, which is what proves a slice stops at the cut and not a bridge
    before it.
    """
    if kind == "ramp":
        step = torch.arange(cols, dtype=torch.float16) // MEL_BINS
        return step.unsqueeze(0).repeat(rows, 1)
    return torch.randn(rows, cols, dtype=torch.float16)


def ltx2_audio_payload(t_steps=251, values="randn"):
    """The tagged audio payload of the 241-frame source window: 25 latent steps
    a second, 640 samples per step, 8 channels x 16 mel bins at 16 kHz."""
    samples = t_steps * SAMPLES_PER_AUDIO_STEP
    return latent_io.build_audio_latent_payload(
        [_audio_values(8, t_steps * MEL_BINS, values)],
        [{"window_no": 1, "start_sample": 0, "samples": samples}],
        app_version=APP_VERSION, model_type=LTX2, base_model_type=LTX2,
        model_filename="ltx2.safetensors", loras=(),
        audio_vae_file="ltx2_audio_vae.safetensors", audio_vae_sha256="ef56",
        audio_vae_dtype="16", sample_rate=SAMPLE_RATE, channels=8, samples=samples,
        seed=7, prompt="window 1", denoising_strength=1.0, steps=40,
        total_windows=1, vocoder_file="", mel_bins=MEL_BINS)


def window_args(**overrides):
    """The per-window bookkeeping the recorder needs (a plain window 1)."""
    args = dict(sliding_window=False, latent_windows=[], latent_audio_windows=[],
                reuse_frames=0, post_decode_pre_trim=0, trim_first_frames=0,
                source_video_overlap_frames_count=0, discard_last_frames=0,
                automatic_trim_last_frames=0, window_prefix_is_image=False,
                sample_is_hdr=False, image_start_tensor=None, written_frame_count=0)
    args.update(overrides)
    return args


def branch_prefix(sidecar_obj, cut, request, *, overlap=OVERLAP, mode="pixel_rule",
                  audio_entry=None):
    """The prefix half of a branch job at ``cut`` - what ``wgp`` computes
    before the window plan is built. Returns ``(prefix, point, error)``."""
    entry = latent_io.latent_payload(sidecar_obj)
    point = latent_io.branch_point_for_frame(entry, cut)
    assert point is not None, f"no branch point at frame {cut}"
    carried = latent_io.carry_over_latent_windows(sidecar_obj, point)
    prefix, error = latent_jobs._prepare_branch_prefix(
        entry, point, carried, base_model_type=LTX2, video_length=int(request),
        reuse_frames=overlap, audio_entry=audio_entry,
        vae_file=LTX2_VAE, vae_sha256=LTX2_VAE_SHA, head_mode=mode)
    return prefix, point, error


def branch_job(sidecar_obj, cut, request, *, overlap=OVERLAP, mode="pixel_rule",
               window_size=WINDOW_SIZE, audio_entry=None, ctx=None):
    """The branch seam ``wgp`` runs: the prefix, then the head-window plan,
    with the default window template built exactly the way the classic path
    builds it for the same job. Returns ``(prefix, point, error, plan_out)``."""
    prefix, point, error = branch_prefix(sidecar_obj, cut, request, overlap=overlap,
                                         mode=mode, audio_entry=audio_entry)
    if prefix is None:
        return None, point, error, None
    template = [dict(w) for w in classic_plan(request, window_size=window_size,
                                              overlap=overlap, kept=prefix["kept_frames"])]
    out = latent_jobs.plan_branch_head_window(
        ctx or stub_ctx("/tmp"),
        branch_latent_info={"prefill_seed": 42, "prefill_prompt": "source prompt",
                            "entry": latent_io.latent_payload(sidecar_obj),
                            "video_present": True},
        branch_prefix=prefix, branch_decode=None, branch_carried={}, prefix_error=None,
        decode_reason=None, all_latent_job=True, base_model_type=LTX2,
        scheduler_active=False, seed=None, prompt="", prompts=["p"],
        video_length=int(request), current_video_length=int(request), video_source=None,
        video_continuation_frames=0, sliding_window_discard_last_frames=0,
        sliding_window=True, frame_scheduler={"active": False},
        scheduled_windows_template=[], default_windows_template=template,
        window_plan=latent_jobs.WindowPlan(
            sliding_window_size=window_size, default_reuse_frames=overlap,
            frames_minimum=FRAMES_MINIMUM, frames_steps=FRAMES_STEPS,
            frames_offset=FRAME_OFFSET,
            sliding_window_defaults={"overlap_offset": 1, "overlap_max": OVERLAP_MAX},
            any_sliding_window=True, model_def={}))
    return prefix, point, error, out


# ---------------------------------------------------------------------------
# The geometry table
#
# Each row is one job: a source file, a cut, a request. ``classic`` is the
# window plan of the same job continued from the video file (frozen here, and
# re-checked against the planner by the first test); ``branch`` is the branch's
# own plan. ``expect`` names which length rule the row is a case of:
#   "request" - the head window holds the whole request, so the job delivers the
#               request minus the frame the cut stands on;
#   "classic" - the head window cannot hold it and the tail windows are the
#               classic planner's own, grid rounding included, so the branch
#               delivers exactly what the classic job delivers;
#   "legacy"  - the full freeze's own shape (see DIV-3).
# ---------------------------------------------------------------------------

CASES = [
    {
        "tag": "window-end request fills one grown head window",
        "latents": (31,), "cut": 241, "request": 240,
        "kept": 241, "shared": 1, "head": (257, 239, 1),
        "classic": [(224, 17), (16, 17)], "branch": [(239, 17)],
        "delivered": 239, "expect": "request",
    },
    {
        "tag": "window-end request below the window budget",
        "latents": (31,), "cut": 241, "request": 100,
        "kept": 241, "shared": 1, "head": (121, 99, 5),
        "classic": [(96, 17)], "branch": [(99, 17)],
        "delivered": 99, "expect": "request",
    },
    {
        "tag": "mid-window request inside the containing window",
        "latents": (31,), "cut": 41, "request": 100,
        "kept": 41, "shared": 0, "head": (121, 100, 4),
        "classic": [(96, 17)], "branch": [(100, 17)],
        "delivered": 100, "expect": "request",
    },
    {
        "tag": "mid-window request past the containing window",
        "latents": (31,), "cut": 41, "request": 300,
        "kept": 41, "shared": 0, "head": (241, 224, 0),
        "classic": [(224, 17), (72, 17)], "branch": [(224, 17), (72, 17)],
        "delivered": 296, "expect": "classic",
    },
    {
        "tag": "kept run shorter than the overlap clamps to 9",
        "latents": (31,), "cut": 9, "request": 100,
        "kept": 9, "shared": 0, "head": (113, 100, 4),
        "classic": [(96, 9)], "branch": [(100, 9)],
        "delivered": 100, "expect": "request",
    },
    {
        "tag": "request past the family window budget",
        "latents": (31,), "cut": 241, "request": 1000,
        "kept": 241, "shared": 1, "head": (1017, 999, 1),
        "classic": [(224, 17), (224, 17), (224, 17), (224, 17), (104, 17)],
        "branch": [(999, 17)], "delivered": 999, "expect": "request",
    },
    {
        "tag": "small window budget against a long request",
        "latents": (31,), "cut": 241, "request": 200, "window": 81,
        "kept": 241, "shared": 1, "head": (217, 199, 1),
        "classic": [(64, 17), (64, 17), (64, 17), (8, 17)], "branch": [(199, 17)],
        "delivered": 199, "expect": "request",
    },
    {
        "tag": "overlap on the grid at 33",
        "latents": (31,), "cut": 241, "request": 240, "overlap": 33,
        "kept": 241, "shared": 1, "head": (273, 239, 1),
        "classic": [(208, 33), (32, 33)], "branch": [(239, 33)],
        "delivered": 239, "expect": "request",
    },
    {
        "tag": "request of one new block",
        "latents": (31,), "cut": 241, "request": 8,
        "kept": 241, "shared": 1, "head": (25, 7, 1),
        "classic": [(8, 17)], "branch": [(7, 17)],
        "delivered": 7, "expect": "request",
    },
    {
        "tag": "chained file, cut inside the second window",
        "latents": (31, 31), "head_trims": (0, 17), "cut": 300, "request": 240,
        "kept": 297, "shared": 0, "head": (257, 240, 0),
        "classic": [(224, 17), (16, 17)], "branch": [(240, 17)],
        "delivered": 240, "expect": "request",
    },
    {
        "tag": "legacy full freeze at the same cut",
        "latents": (31,), "cut": 241, "request": 240, "mode": "full_freeze",
        "kept": 241, "shared": 0, "cut_end": True, "head": None,
        "classic": [(224, 17), (16, 17)], "branch": [(1, 241), (224, 17), (16, 17)],
        "delivered": 241, "expect": "legacy",
    },
]


def case_row(tag):
    for row in CASES:
        if row["tag"] == tag:
            return row
    raise KeyError(tag)


def run_case(row, ctx=None):
    """The table row as the branch runs it: ``(sidecar, prefix, plan_out)``."""
    sidecar = ltx2_sidecar(row["latents"], row.get("head_trims", ()))
    prefix, point, error, out = branch_job(
        sidecar, row["cut"], row["request"], overlap=row.get("overlap", OVERLAP),
        mode=row.get("mode", "pixel_rule"), window_size=row.get("window", WINDOW_SIZE),
        ctx=ctx)
    assert error == "" and prefix is not None and out is not None, error
    return sidecar, prefix, out


def total_delivered(plan):
    return sum(int(w["output_frames"]) for w in plan)


# ---------------------------------------------------------------------------
# The golden baseline, and the branch against it
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("row", CASES, ids=[c["tag"] for c in CASES])
def test_the_classic_baseline_is_still_what_the_classic_planner_builds(row):
    # The frozen classic column is the contract with the classic path. If this
    # fails, the classic planner moved under the branch: escalate, do not
    # re-freeze the number to make the row pass.
    plan = classic_plan(row["request"], window_size=row.get("window", WINDOW_SIZE),
                        overlap=row.get("overlap", OVERLAP), kept=row["kept"])
    assert shape(plan) == row["classic"]


@pytest.mark.parametrize("row", CASES, ids=[c["tag"] for c in CASES])
def test_the_branch_plan_matches_its_frozen_geometry(row):
    _sidecar, _prefix, out = run_case(row)
    assert out["refusal"] is None
    assert shape(out["default_windows_template"]) == row["branch"]


@pytest.mark.parametrize("row", CASES, ids=[c["tag"] for c in CASES])
def test_the_frame_the_cut_stands_on_belongs_to_the_window_end(row):
    # One frame of the request is the frame the cut stands on when the cut ends
    # a window; a cut inside a window spends nothing (its length is the
    # containing window's own committed end). The full freeze is the exception
    # the pixel rule replaced: it stands on the same window-end cut and spends
    # nothing, which is the extra frame of DIV-3.
    _sidecar, prefix, _out = run_case(row)
    assert int(prefix["shared_cut_frame"]) == row["shared"]
    assert int(prefix["kept_frames"]) == row["kept"]
    assert bool(prefix["cut_at_window_end"]) == row.get("cut_end", row["shared"] == 1)


@pytest.mark.parametrize("row", CASES, ids=[c["tag"] for c in CASES])
def test_the_job_delivers_what_its_case_promises(row, capsys):
    # "request" rows: the head window holds the whole request, so the job lands
    # on the request minus the frame the cut stands on. "classic" rows: the tail
    # windows are the classic planner's own, so the job lands where the classic
    # job lands, frame for frame.
    _sidecar, prefix, out = run_case(row)
    plan = out["default_windows_template"]
    delivered = total_delivered(plan)
    assert delivered == row["delivered"]
    if row["expect"] == "request":
        assert delivered == row["request"] - row["shared"]
    elif row["expect"] == "classic":
        classic = classic_plan(row["request"], window_size=row.get("window", WINDOW_SIZE),
                              overlap=row.get("overlap", OVERLAP), kept=row["kept"])
        assert delivered == total_delivered(classic)
    # The plan receipt the job log prints is the same arithmetic.
    logged = capsys.readouterr().out
    assert f"+ {delivered} new in {len(plan)} window" in logged
    assert f"= {row['kept'] + delivered} frames" in logged


@pytest.mark.parametrize("row", [c for c in CASES if c.get("mode", "pixel_rule") == "pixel_rule"],
                         ids=[c["tag"] for c in CASES if c.get("mode", "pixel_rule") == "pixel_rule"])
def test_the_head_window_keeps_the_family_grid_and_its_parts_add_up(row):
    # Every window the branch plans decodes on the family's grid (1 + 8k), and
    # the head window's frame count is exactly its frozen overlap plus what it
    # delivers plus what it trims - the identity the writer's committed counts
    # and the audio bridge both rest on.
    _sidecar, prefix, out = run_case(row)
    plan = out["default_windows_template"]
    for window in plan:
        frame_num = int(window["frame_num"])
        assert (frame_num - FRAME_OFFSET) % STRIDE == 0, window
    head = plan[0]
    assert int(head["frame_num"]) == (int(head["overlap_frames"]) + int(head["output_frames"])
                                      + int(head.get("trim_last_frames") or 0)
                                      + int(head.get("discard_last_frames") or 0))
    if row["head"] is not None:
        assert (int(prefix["head_frame_num"]), int(prefix["head_output"]),
                int(prefix["head_trim_new"])) == row["head"]
        assert int(head["overlap_frames"]) == int(prefix["assembly_overlap"])


# ---------------------------------------------------------------------------
# The written file: the plan, the blocks it produces and the cut read back
# ---------------------------------------------------------------------------

def write_branch_job(tmp_path, row, *, audio=False, ctx=None, audio_values="randn"):
    """Run the row's job the production way - the engine's own recorder - and
    return the saved sidecar object (plus the pieces the tests reference).

    The head window's payload is what the model would hand the writer: a run of
    ``head_frame_num`` decoded frames' worth of latents, and for the audio rows
    the window's audio with the bridge the model froze in front of it.
    ``audio_values="ramp"`` writes position-encoded audio, so a later slice can
    be checked against the frames it should cover.
    """
    sidecar = ltx2_sidecar(row["latents"], row.get("head_trims", ()))
    if audio:
        # The job reads its source as one object: the video entry plus the
        # soundtrack the source job carried.
        sidecar = {**sidecar, **ltx2_audio_payload(values=audio_values)}
    audio_entry = latent_io.audio_latent_payload(sidecar) if audio else None
    prefix, point, error, out = branch_job(
        sidecar, row["cut"], row["request"], overlap=row.get("overlap", OVERLAP),
        mode=row.get("mode", "pixel_rule"), window_size=row.get("window", WINDOW_SIZE),
        audio_entry=audio_entry, ctx=ctx or stub_ctx(tmp_path))
    assert error == "" and prefix is not None and out is not None, error
    head = out["default_windows_template"][0]
    head_frames = int(head["frame_num"])
    head_latents = 1 + (head_frames - FRAME_OFFSET) // STRIDE
    video_block = latent_io.branch_cut_window_block(sidecar, point)
    audio_block = latent_io.branch_cut_audio_window_block(sidecar, point) if audio else None
    kwargs = window_args(
        sliding_window=True,
        source_video_overlap_frames_count=int(prefix["assembly_overlap"]),
        automatic_trim_last_frames=int(head.get("trim_last_frames") or 0),
        written_frame_count=row["kept"] + total_delivered(out["default_windows_template"]))
    obj = dict(ltx2_block(ltx2_latents(head_latents, seed=7), frames=head_frames))
    if audio:
        bridge = int(prefix.get("audio_bridge_latents") or 0)
        steps = max(bridge + 1, round(head_frames / FPS * AUDIO_STEPS_PER_SECOND))
        head_audio = latent_io.build_audio_latent_payload(
            [_audio_values(8, steps * MEL_BINS, audio_values)],
            [{"window_no": 1, "start_sample": 0, "samples": steps * SAMPLES_PER_AUDIO_STEP}],
            app_version=APP_VERSION, model_type=LTX2, base_model_type=LTX2,
            audio_vae_file="ltx2_audio_vae.safetensors", audio_vae_sha256="ef56",
            audio_vae_dtype="16", sample_rate=SAMPLE_RATE, channels=8,
            samples=steps * SAMPLES_PER_AUDIO_STEP, mel_bins=MEL_BINS)
        obj.update(head_audio)
        carried = latent_io.build_audio_latent_payload(
            [audio_block["latents"]],
            [{"start_sample": audio_block["start_sample"], "samples": audio_block["samples"]}],
            app_version=APP_VERSION, model_type=LTX2, base_model_type=LTX2,
            sample_rate=SAMPLE_RATE, channels=8, samples=audio_block["samples"],
            mel_bins=MEL_BINS)[latent_io.LATENT_FORMAT_ID_AUDIO]
        kwargs["latent_audio_windows"] = [carried]
    rec = latent_jobs.LatentWindowRecorder(
        ctx or stub_ctx(tmp_path), model_def={}, model_type=LTX2, base_model_type=LTX2,
        branch_prefix=prefix, branch_cut_video_block=video_block,
        branch_cut_audio_block=audio_block)
    media = str(tmp_path / f"parity-{abs(hash(row['tag']))}.mp4")
    rec.on_window(obj, {}, media, 1, **kwargs)
    saved = latent_io.load_latent_file(latent_io.latent_sidecar_path(media))
    return saved, sidecar, prefix, point, out


# The shapes the writer is exercised on: the window-end cut, a cut inside the
# window, and the clamped kept run that is shorter than the overlap.
WRITER_ROWS = ["window-end request fills one grown head window",
               "mid-window request inside the containing window",
               "kept run shorter than the overlap clamps to 9"]


@pytest.mark.parametrize("tag", WRITER_ROWS)
def test_the_written_file_commits_what_the_plan_promised(tag, tmp_path):
    # The plan's arithmetic has to survive the writer: the cut block plus the
    # head block commit the kept region plus what the windows deliver, the file
    # verifies its own length, and a branch point at its end reads back the same
    # kept count - the file the next cut will be taken from.
    row = case_row(tag)
    saved, _sidecar, prefix, _point, out = write_branch_job(tmp_path, row)
    entry = saved[latent_io.LATENT_FORMAT_ID]
    delivered = total_delivered(out["default_windows_template"])
    specs = latent_io.entry_window_specs(entry)
    assert latent_io.sidecar_total_frames(entry) == row["kept"] + delivered
    assert entry.get("length_verified") is not False
    committed = 0
    for spec in specs:
        frames = decoded_frames(int(spec["latents"].shape[1]))
        committed += frames - int(spec["head_trim"]) - int(spec["tail_trim"])
    assert committed == row["kept"] + delivered
    again = latent_io.branch_point_for_frame(entry, row["kept"] + delivered)
    # A cut at the file's own end resolves to the last boundary the file can
    # offer: the head window's tail trim moves the final boundary off the very
    # last frame, so the resolvable cut sits within one latent step of it.
    assert again is not None
    assert row["kept"] + delivered - STRIDE < int(again["kept_frames"]) <= row["kept"] + delivered


def fake_decode_window(spec):
    """A family decoder that answers with the window's own frame count - the
    geometry the branch job's kept-region decode cares about, no VAE."""
    frames = decoded_frames(int(spec["latents"].shape[1]))
    return torch.zeros(3, frames, 4, 4, dtype=torch.uint8)


@pytest.mark.parametrize("tag", WRITER_ROWS)
def test_the_kept_region_decodes_to_exactly_the_cut(tag, tmp_path):
    # The kept region of the output video is a decode of the source's kept
    # windows, sliced at the cut - and the file the job writes decodes back to
    # its own length, which is what lets the next cut be taken from it.
    row = case_row(tag)
    saved, sidecar, _prefix, point, out = write_branch_job(tmp_path, row)
    source_entry = latent_io.latent_payload(sidecar)
    specs, kept, is_i2v = latent_jobs._branch_kept_window_specs(source_entry, point)
    assert specs is not None and kept == row["kept"]
    assembled = latent_io.decode_kept_windows(fake_decode_window, specs, kept, is_i2v=is_i2v)
    assert int(assembled.shape[1]) == kept
    # The written file, branched again at its own end: the cut it offers is
    # within a latent step of its length, and it decodes out of its own windows.
    entry = saved[latent_io.LATENT_FORMAT_ID]
    delivered = total_delivered(out["default_windows_template"])
    total = row["kept"] + delivered
    end = latent_io.branch_point_for_frame(entry, total)
    assert end is not None
    assert total - STRIDE < int(end["kept_frames"]) <= total
    again_specs, again_kept, again_i2v = latent_jobs._branch_kept_window_specs(entry, end)
    again = latent_io.decode_kept_windows(fake_decode_window, again_specs, again_kept,
                                          is_i2v=again_i2v)
    assert int(again.shape[1]) == again_kept


@pytest.mark.parametrize("tag", WRITER_ROWS)
def test_the_soundtrack_joins_at_the_cut_and_is_recorded_once(tag, tmp_path):
    # The kept region keeps its own soundtrack, the head window keeps the part
    # the model generated, and the bridge between them is written exactly once:
    # the head window's recorded audio starts after the steps the model froze,
    # and its position follows the kept region on the new track. The track is at
    # least as long as the video - the tail the video trims is not trimmed out of
    # the audio, so the soundtrack can outlive the picture by the head window's
    # rounding overshoot, never the other way round.
    row = case_row(tag)
    saved, sidecar, prefix, point, out = write_branch_job(tmp_path, row, audio=True)
    total = row["kept"] + total_delivered(out["default_windows_template"])
    head_frames = int(out["default_windows_template"][0]["frame_num"])
    audio = latent_io.audio_latent_payload(saved)
    assert audio is not None
    kept_win, head_win = audio["windows"]
    cut_audio = latent_io.branch_cut_audio_window_block(sidecar, point)
    assert cut_audio is not None
    assert int(kept_win["samples"]) == int(cut_audio["samples"])
    # The time layout is contiguous: the head window's audio starts exactly
    # where the kept region's ends, so the decode assembles one track with no
    # gap and no overlap. (The *latent* layout of the stacked payload is not -
    # see AUD-1 below.)
    assert int(head_win["start_sample"]) == (int(kept_win["start_sample"])
                                             + int(kept_win["samples"]))
    bridge = int(prefix["audio_bridge_latents"])
    head_steps = round(head_frames / FPS * AUDIO_STEPS_PER_SECOND)
    assert bridge >= 1 and int(head_win["samples"]) == (head_steps - bridge) * SAMPLES_PER_AUDIO_STEP
    # The first block of the payload is the cut's own audio, verbatim.
    specs = latent_io.audio_window_specs(audio)
    torch.testing.assert_close(specs[0]["latents"].float(), cut_audio["latents"].float(),
                               rtol=0, atol=0)
    track_frames = sum(int(w["samples"]) for w in audio["windows"]) / SAMPLE_RATE * FPS
    assert total <= track_frames <= total + FRAMES_STEPS
    # The recorder trims a head audio chunk by min(its head trim, the entry's
    # recorded sliding-window overlap). No writer has ever recorded that field,
    # which is what keeps the bridge from being trimmed a second time - if a
    # writer starts, this row is where it shows up.
    for window in audio["windows"]:
        assert "sliding_window_overlap" not in window


def test_a_file_written_without_the_pixel_rule_still_decodes_and_branches(tmp_path):
    # Phase D retires the full freeze. A file it wrote is an ordinary file - the
    # legacy writer recorded no head mode at all - so it must keep decoding and
    # keep branching under the rule. This is the backward-compatibility guard:
    # the reader has no mode concept, and nothing here looks one up.
    ctx = stub_ctx(tmp_path)
    rec = latent_jobs.LatentWindowRecorder(ctx, model_def={}, model_type=LTX2,
                                           base_model_type=LTX2)
    media = str(tmp_path / "legacy.mp4")
    rec.on_window(ltx2_block(ltx2_latents(31)), {}, media, 1,
                  **window_args(written_frame_count=241))
    saved = latent_io.load_latent_file(latent_io.latent_sidecar_path(media))
    entry = saved[latent_io.LATENT_FORMAT_ID]
    assert "branch_head_mode" not in entry
    assert all("branch_head_mode" not in w for w in entry["windows"])
    assert latent_io.sidecar_total_frames(entry) == 241
    point = latent_io.branch_point_for_frame(entry, 241)
    specs, kept, is_i2v = latent_jobs._branch_kept_window_specs(entry, point)
    assert kept == 241
    assembled = latent_io.decode_kept_windows(fake_decode_window, specs, kept, is_i2v=is_i2v)
    assert int(assembled.shape[1]) == 241
    # The same job the table describes, taken from the legacy file instead.
    row = case_row("window-end request fills one grown head window")
    _prefix, _point, error, out = branch_job(saved, row["cut"], row["request"], ctx=ctx)
    assert error == "" and out is not None and out["refusal"] is None
    assert shape(out["default_windows_template"]) == row["branch"]
    assert total_delivered(out["default_windows_template"]) == row["delivered"]


# ---------------------------------------------------------------------------
# The divergences this harness found
#
# Pinned with their numbers so they cannot quietly grow, and named so the next
# phase can close them one at a time. None of them is approved: each says what
# it would take, and which side owns it.
# ---------------------------------------------------------------------------

LTX2_WINDOW_MAX = 501           # ltx2_handler.py: sliding_window_defaults window_max


def test_div1_the_pixel_rule_head_window_grows_past_the_window_budget():
    # DIV-1 (latent side): the pixel rule puts the whole request on top of the
    # frozen overlap, so the head window is overlap + request however long the
    # request is - one window of 1017 frames where the classic job slides five
    # windows of 241. The family's own ceiling (501) is not consulted, and the
    # request is only bounded by the video-length control. Closing it means
    # capping the head window at the budget and letting the tail windows make up
    # the rest, which changes the shapes the live run accepted: latent-side work,
    # needs the acceptance run.
    row = case_row("request past the family window budget")
    _sidecar, prefix, out = run_case(row)
    plan = out["default_windows_template"]
    classic = classic_plan(row["request"], kept=row["kept"])
    assert len(plan) == 1 and len(classic) == 5
    assert int(plan[0]["frame_num"]) == 1017 > LTX2_WINDOW_MAX
    assert int(prefix["head_frame_num"]) == 1017
    assert total_delivered(plan) == row["request"] - row["shared"] == 999
    assert total_delivered(classic) == row["request"] == 1000
    # The same holds at a smaller budget: the budget is not the ceiling it looks
    # like, because the branch does not read it.
    small = case_row("small window budget against a long request")
    _s2, _p2, out2 = run_case(small)
    assert shape(out2["default_windows_template"]) == [(199, 17)]
    assert len(classic_plan(small["request"], window_size=small["window"], kept=small["kept"])) == 4


def test_div2_the_two_paths_put_the_frame_grid_on_different_sides():
    # DIV-2 (classic side - escalate, do not fix here): the classic planner
    # rounds the window's frame count onto the family grid and lets the output
    # count follow, so a 100-frame request under a 17-frame overlap delivers 96.
    # The pixel rule fixes the output at the request and moves the overshoot into
    # the window's tail trim, delivering 99. Same job, same cut, three frames
    # apart - only the frame count, never a frame of content. Making the two
    # agree would change every classic job's length, classic-side work.
    row = case_row("window-end request below the window budget")
    _sidecar, prefix, out = run_case(row)
    head = out["default_windows_template"][0]
    classic = classic_plan(row["request"], kept=row["kept"])
    assert total_delivered([head]) == 99 == row["request"] - row["shared"]
    assert total_delivered(classic) == 96 == row["request"] - row["shared"] - 3
    assert int(classic[0]["frame_num"]) == 113 and int(classic[0]["trim_last_frames"] or 0) == 0
    assert int(head["frame_num"]) == 121 and int(head["trim_last_frames"]) == 5
    assert int(head["overlap_frames"]) == int(classic[0]["overlap_frames"]) == OVERLAP
    assert int(prefix["head_trim_new"]) == 5


def test_div3_the_full_freeze_delivers_one_frame_more_than_either_neighbour():
    # DIV-3 (legacy, Phase D removes it): the full freeze spends no frame for the
    # frame the cut stands on and re-generates the containing window's shape from
    # the frozen kept run, so it lands one frame above the classic job and two
    # above the rule at the same cut. Its head window is the whole kept run as
    # the overlap, which is also why its window count is the classic plan's plus
    # the frozen head.
    legacy = case_row("legacy full freeze at the same cut")
    _sidecar, prefix, out = run_case(legacy)
    plan = out["default_windows_template"]
    classic = classic_plan(legacy["request"], kept=legacy["kept"])
    rule = case_row("window-end request fills one grown head window")
    _s2, _p2, out2 = run_case(rule)
    assert total_delivered(plan) == 241
    assert total_delivered(classic) == 240
    assert total_delivered(out2["default_windows_template"]) == 239
    assert int(plan[0]["overlap_frames"]) == legacy["kept"] == 241
    assert int(prefix["shared_cut_frame"]) == 0
    assert shape(plan) == [legacy["branch"][0]] + shape(classic)


def test_div4_a_cut_at_the_end_of_a_trimmed_window_in_a_chained_file_is_accepted():
    # DIV-4 (latent side). Phase 3.1 lifted the K > 1 window-end refusal for the
    # pixel rule: the frozen head is the kept run's tail N, decoupled from the
    # containing window's own decoded head, so a window whose head was trimmed
    # (head trim 17) cuts cleanly at its committed end. The assembly splices the
    # window's committed run, which already excludes that head share. The classic
    # job continues from the video file and never asks. The same file still cuts
    # a step inside the window. A tail trim (not the head share) is what stays
    # refused - it would drop committed frames from the kept decode.
    sidecar = ltx2_sidecar((31, 31), (0, 17))
    prefix, point, error, out = branch_job(sidecar, 465, 240)
    assert error == "" and prefix is not None and out is not None
    assert int(point["kept_frames"]) == 465 and int(prefix["kept_frames"]) == 465
    assert int(prefix["shared_cut_frame"]) == 1 and prefix["cut_at_window_end"] is True
    assert int(prefix["assembly_overlap"]) == int(prefix["guide_frames"]) == OVERLAP
    plan = out["default_windows_template"]
    assert shape(plan) == [(239, OVERLAP)]
    assert total_delivered(plan) == 239                      # the request minus the shared frame
    # One latent step inside the same window the same job plans normally.
    inside, _point, inner_error, inner = branch_job(sidecar, 457, 240)
    assert inner_error == "" and inside is not None and inner is not None
    assert int(inside["kept_frames"]) == 457
    assert total_delivered(inner["default_windows_template"]) == 240


def test_the_soundtrack_stack_holes_are_the_padded_slots(tmp_path):
    # What AUD-1 turned out to be. The silence between two stacked audio blocks
    # is the stack's own padded slot - every block is padded to the longest one,
    # the layout the video stack uses and the offsets the reader walks back - not
    # a hole the writer left behind the bridge. The payload is therefore bigger
    # than the track it holds by design, the blocks still read back exactly what
    # was written, and the bridge's steps are simply not in the head block (the
    # kept region carries them, recorded once).
    row = case_row("mid-window request inside the containing window")
    saved, _sidecar, prefix, _point, _out = write_branch_job(
        tmp_path, row, audio=True, audio_values="ramp")
    audio = latent_io.audio_latent_payload(saved)
    kept_win, head_win = audio["windows"]
    specs = latent_io.audio_window_specs(audio)
    stacked = latent_io.payload_latents(audio)
    slot = max(int(kept_win["latent_len"]), int(head_win["latent_len"]))
    # The head block sits at the kept block's padded slot and the payload is as
    # long as the slots, so the gap is padding and nothing but padding.
    assert int(head_win["latent_start"]) == slot
    assert int(stacked.shape[1]) == slot + int(head_win["latent_len"])
    assert float(stacked[:, int(kept_win["latent_len"]):slot].abs().sum()) == 0.0
    # Each block reads back as its own contiguous run of steps, and the values
    # say where in the window they came from: the kept block carries the source
    # window's steps from its first, the head block carries the new window's
    # steps *from the bridge on* - the steps the model froze are not stored, the
    # kept region carries them, recorded once.
    bridge = int(prefix["audio_bridge_latents"])
    for spec, win, first_step in zip(specs, audio["windows"], (0, bridge)):
        block = spec["latents"]
        steps = int(win["latent_len"]) // MEL_BINS
        assert int(block.shape[1]) == steps * MEL_BINS
        assert int(block[0, 0]) == first_step and int(block[0, -1]) == first_step + steps - 1
    # The head block is the window's generated audio without the bridge the
    # model froze, so it holds fewer steps than the window has.
    window_steps = round(int(prefix["head_frame_num"]) / FPS * AUDIO_STEPS_PER_SECOND)
    assert bridge >= 1 and int(head_win["latent_len"]) // MEL_BINS == window_steps - bridge


def test_a_second_cut_inside_a_head_window_carries_the_audio_up_to_the_cut(tmp_path):
    # AUD-2, closed. The head window of a pixel-rule branch stores its audio
    # after the bridge the model froze, so its block starts ``overlap_decoded``
    # frames into the window while a cut inside that window is measured from the
    # window's decoded start. The slice the next branch resolves has to stop at
    # the cut: stop a bridge short of it and the next window's soundtrack starts
    # early by the bridge - 17 frames, 0.7 s on this shape. The block's recorded
    # duration is not the problem, it describes the steps the block stores; the
    # frame the block's first step sits at is, and the file already records it.
    row = case_row("mid-window request inside the containing window")
    saved, _sidecar, prefix, _point, _out = write_branch_job(
        tmp_path, row, audio=True, audio_values="ramp")
    entry = saved[latent_io.LATENT_FORMAT_ID]
    audio = latent_io.audio_latent_payload(saved)
    video_spec = latent_io.entry_window_specs(entry)[-1]
    head_win = audio["windows"][-1]
    block = latent_io.audio_window_specs(audio)[-1]["latents"]
    bridge = int(video_spec["overlap_decoded"])
    audio_bridge = int(prefix["audio_bridge_latents"])     # the same span in audio steps
    assert video_spec["branch_head_mode"] == "pixel_rule"
    assert bridge == int(prefix["overlap_decoded"]) == int(prefix["assembly_overlap"]) == OVERLAP
    steps = int(block.shape[1]) // MEL_BINS
    decoded = decoded_frames(int(video_spec["latents"].shape[1]))
    covered = decoded - bridge
    # The block covers the window from the bridge on: what it records is the
    # duration of that run, and what it stores is the steps of that run.
    assert abs(int(head_win["samples"]) / int(audio["sample_rate"]) * FPS - covered) <= 1.0
    for offset in (50, 90):
        point = latent_io.branch_point_for_frame(entry, row["kept"] + offset)
        assert point is not None and int(point["window_no"]) == 2
        sl = latent_io._branch_audio_prefix_slice(audio, entry, point, bridge_frames=bridge)
        assert sl is not None and int(sl["origin_frames"]) == bridge
        # The step of the block that stands at the cut, and the slice that ends
        # there - not one a bridge earlier.
        at_cut = min(steps, round(steps * (int(point["decoded_prefix"]) - bridge) / covered))
        assert abs(int(sl["end"]) - at_cut) <= 1
        assert int(sl["end"]) > at_cut - bridge          # pre-fix, it landed here
        assert int(sl["end"]) - int(sl["start"]) == int(sl["count"])
        # The content agrees with the numbers. Every column of the block stores
        # the step of the window's generated audio it came from, and the block
        # starts after the bridge's steps - so the slice's own values say which
        # frames of the window it holds, and the last of them is the frame right
        # before the cut.
        assert int(sl["flat"][0, 0]) == int(sl["start"]) + audio_bridge
        assert int(sl["flat"][0, -1]) == int(sl["end"]) - 1 + audio_bridge
        cut_step = round((int(point["decoded_prefix"]) - 1) / FPS * AUDIO_STEPS_PER_SECOND)
        assert abs(int(sl["flat"][0, -1]) - cut_step) <= 1
        # The same cut without a bridge is the kept region's audio: the block
        # from its first step to the cut.
        whole = latent_io._branch_audio_prefix_slice(audio, entry, point)
        assert int(whole["start"]) == 0 and int(whole["end"]) == int(sl["end"])
        assert int(whole["flat"][0, 0]) == audio_bridge


def test_the_bridge_head_only_moves_cuts_inside_the_head_window(tmp_path):
    # The origin is a property of the window the cut falls in, not of the file:
    # a kept block that came from an ordinary window still starts at its
    # window's first step, and only a block written under the pixel rule carries
    # the head that moves a slice. Without that distinction the fix would slide
    # every second cut's soundtrack by a bridge in the other direction.
    row = case_row("mid-window request inside the containing window")
    saved, _sidecar, prefix, _point, _out = write_branch_job(
        tmp_path, row, audio=True, audio_values="ramp")
    entry = saved[latent_io.LATENT_FORMAT_ID]
    audio = latent_io.audio_latent_payload(saved)
    specs = latent_io.entry_window_specs(entry)
    bridge = int(prefix["overlap_decoded"])
    audio_bridge = int(prefix["audio_bridge_latents"])     # the same span in audio steps
    assert "branch_head_mode" not in specs[0] and "overlap_decoded" not in specs[0]
    assert specs[-1]["branch_head_mode"] == "pixel_rule"
    # A cut inside the first window: measured from that window's decoded start,
    # which is where its block starts too.
    first = latent_io.branch_point_for_frame(entry, row["kept"] - 10)
    assert first is not None and int(first["window_no"]) == 1
    sl_first = latent_io._branch_audio_prefix_slice(audio, entry, first, bridge_frames=bridge)
    assert sl_first is not None and int(sl_first["origin_frames"]) == 0
    assert int(sl_first["flat"][0, 0]) == int(sl_first["start"])
    # A cut inside the head window: the block's head moves with it.
    second = latent_io.branch_point_for_frame(entry, row["kept"] + 50)
    assert second is not None and int(second["window_no"]) == 2
    sl_second = latent_io._branch_audio_prefix_slice(audio, entry, second, bridge_frames=bridge)
    assert sl_second is not None and int(sl_second["origin_frames"]) == bridge
    assert int(sl_second["flat"][0, 0]) == int(sl_second["start"]) + audio_bridge


# ---------------------------------------------------------------------------
# The Wan ports: the 4/1 grid's own numbers
#
# The plain-headed Wan classes take the pixel rule: the guide-door classes (i2v,
# i2v_2_2, ti2v_2_2) declare "guide_pixels" and the text-only classes (t2v,
# t2v_1.3B, t2v_2_2 and the shotplan pair) declare "encoded_head"
# (wan_handler.py). Both build their head from the decode of their latent file's
# kept tail and differ only in where the encode goes - the class's own guide door,
# or the frozen head of the denoise tensor. Their plan arithmetic is the
# family-neutral one these rows already drove, so what they pin is the contract
# between the plan and the engine: the plan hands the engine an overlap whose
# encode turns back into the latents the plan counted.
# ---------------------------------------------------------------------------
WAN = "t2v"
WAN_VAE, WAN_VAE_SHA = "Wan2.1_VAE.safetensors", "ab12"
WAN_STRIDE, WAN_OFFSET = 4, 1               # the Wan 2.1 video VAE's temporal layout
WAN_MINIMUM, WAN_STEPS = 5, 4               # wan_handler.py: frames_minimum / frames_steps
WAN_WINDOW, WAN_OVERLAP = 129, 5            # get_factory_settings("t2v_1.3B")
# The family's own ceiling on a window-to-window overlap. It is a different
# quantity from the capacity a window leaves for a handoff (which is cut by the
# stride, see classic_plan): the planner only ever *raises* its limit with it.
WAN_OVERLAP_MAX = WAN_WINDOW - WAN_MINIMUM
WAN_FPS = 16.0                              # wan_handler.py: the 14B and 1.3B classes
WAN_CHANNELS = 16                           # the Wan 2.1 VAE's latent channels
WAN_LATENTS = 33                            # one clean 129-frame window
WAN_GUIDE_DOOR = {"branch_pixel_rule": "guide_pixels"}   # wan_handler.py query_model_def
WAN_ENCODE_HEAD = {"branch_pixel_rule": "encoded_head"}  # the text-only classes


def wan_frames(latents):
    """The frames a run of ``latents`` decodes to on the 4/1 layout."""
    return WAN_OFFSET + WAN_STRIDE * (int(latents) - 1)


def wan_sidecar():
    """A one-window 129-frame t2v source file on the 4/1 grid (memoized)."""
    if "wan" not in _SIDECARS:
        run = torch.zeros(WAN_CHANNELS, WAN_LATENTS, 2, 2, dtype=torch.float16)
        _SIDECARS["wan"] = latent_io.build_latent_sidecar(
            [run], [{"prompt": "window 1", "seed": 7, "steps": 20,
                     "denoising_strength": 1.0, "head_trim": 0, "tail_trim": 0,
                     "frame_count": wan_frames(WAN_LATENTS)}],
            app_version=APP_VERSION, model_type="t2v_1.3B", base_model_type=WAN,
            model_filename="wan.safetensors", loras=(),
            vae_file=WAN_VAE, vae_sha256=WAN_VAE_SHA, vae_z_dim=WAN_CHANNELS,
            vae_dtype="32", width=256, height=256, fps=WAN_FPS,
            latent_stride=WAN_STRIDE, frame_offset=WAN_OFFSET, is_hdr=False)
    return _SIDECARS["wan"]


def wan_classic_plan(request, kept, *, window=WAN_WINDOW, overlap=WAN_OVERLAP):
    """The plan of the same job continued from the video file, on Wan's grid.

    The handoff is cut by ``WAN_STRIDE``, the way ``wgp.py:8607`` cuts it: the
    inherited frames live inside the window, which still adds up to its full
    length. ``WAN_MINIMUM`` bounds a run of new frames, not that room.
    """
    reuse = min(window - WAN_STRIDE, overlap)
    first = min(max(1, reuse), int(kept))
    return build_default_window_plan(
        total_frames=int(request), window_size=window, default_overlap=reuse,
        discard_last_frames=0, minimum=WAN_MINIMUM, step=WAN_STEPS,
        frame_offset=WAN_OFFSET, overlap_offset=1, max_overlap=WAN_OVERLAP_MAX,
        first_window_overlap=first, first_window_available_overlap=int(kept),
        initial_shared_frames=1 if first > 0 else 0)


def wan_rule(cut, request, *, overlap=WAN_OVERLAP, mode="pixel_rule"):
    """The head window the rule plans for a Wan file at ``cut``.

    ``branch_head_window`` takes the layout from the sidecar, so this is the
    same planner the LTX2 rows use - the family gate in ``_prepare_branch_prefix``
    is the separate question the last row asks.
    """
    entry = latent_io.latent_payload(wan_sidecar())
    point = latent_io.branch_point_for_frame(entry, cut)
    assert point is not None, f"no branch point at frame {cut}"
    return point, latent_io.branch_head_window(
        entry, point, int(request), stride=WAN_STRIDE, offset=WAN_OFFSET,
        head_mode=mode, overlap_frames=overlap)


def wan_branch_prefix(sidecar_obj, cut, request, *, overlap=WAN_OVERLAP, mode="pixel_rule", model_def=None):
    """The prefix half of a Wan branch job, as ``wgp`` computes it."""
    entry = latent_io.latent_payload(sidecar_obj)
    point = latent_io.branch_point_for_frame(entry, cut)
    assert point is not None, f"no branch point at frame {cut}"
    carried = latent_io.carry_over_latent_windows(sidecar_obj, point)
    prefix, error = latent_jobs._prepare_branch_prefix(
        entry, point, carried, base_model_type=WAN, video_length=int(request),
        reuse_frames=overlap, audio_entry=None, model_def=model_def,
        vae_file=WAN_VAE, vae_sha256=WAN_VAE_SHA, head_mode=mode)
    assert error == "" and prefix is not None, error
    return prefix, point, error


def wan_door_prefix(cut, request, *, overlap=WAN_OVERLAP, mode="pixel_rule"):
    """The same prefix for a class that declares the guide-door rule."""
    prefix, point, _error = wan_branch_prefix(
        wan_sidecar(), cut, request, overlap=overlap, mode=mode, model_def=WAN_GUIDE_DOOR)
    return prefix


def wan_encode_prefix(cut, request, *, overlap=WAN_OVERLAP, mode="pixel_rule"):
    """The same prefix for a class that declares the encoded-head rule."""
    prefix, point, _error = wan_branch_prefix(
        wan_sidecar(), cut, request, overlap=overlap, mode=mode, model_def=WAN_ENCODE_HEAD)
    return prefix


def wan_guide_frames(prefix):
    """The kept run as a decode, and the guide the door is handed from it."""
    run = torch.arange(3 * int(prefix["kept_frames"]) * 2 * 2,
                       dtype=torch.uint8).reshape(3, int(prefix["kept_frames"]), 2, 2)
    return run, latent_branch_head.guide_tail(run, int(prefix["guide_frames"]))


def test_the_bridge_arithmetic_lands_on_the_wan_grid():
    # The rule's frozen head and re-denoised bridge come from
    # tail_latents_for_overlap, which takes the layout as an argument: an overlap
    # that already sits on the family's 1 + 4k grid is met exactly, and one that
    # does not overshoots by less than frames_steps - the same contract the LTX2
    # rows pin for 8/1, so nothing in the rule is LTX2-shaped.
    assert latent_io.tail_latents_for_overlap(5, WAN_STRIDE, WAN_OFFSET) == (2, 5)
    assert latent_io.tail_latents_for_overlap(13, WAN_STRIDE, WAN_OFFSET) == (4, 13)
    assert latent_io.tail_latents_for_overlap(17, WAN_STRIDE, WAN_OFFSET) == (5, 17)
    for wanted in range(1, 40):
        n, o = latent_io.tail_latents_for_overlap(wanted, WAN_STRIDE, WAN_OFFSET)
        assert o >= wanted, (wanted, n, o)
        assert o - wanted < WAN_STEPS, (wanted, n, o)
        assert wan_frames(n) == o, (wanted, n, o)


def test_the_rule_on_the_wan_grid_keeps_the_request_it_was_given():
    # The window-end cut on a 129-frame file: the rule puts the request on top of
    # the 5-frame bridge (2 latents frozen), trims the grid overshoot, and keeps
    # the frame the cut stands on - the same shape as the LTX2 rows, with the
    # 4/1 grid's own numbers.
    point, head = wan_rule(wan_frames(WAN_LATENTS), 100)
    assert int(point["kept_frames"]) == WAN_WINDOW and head["cut_at_window_end"]
    assert head["assembly_overlap"] == WAN_OVERLAP
    assert head["head_prefix_latents"] == 2 and head["overlap_decoded"] == WAN_OVERLAP
    assert (head["frame_num"], head["new_frames"], head["tail_trim_new"]) == (105, 99, 1)
    assert (head["frame_num"] - WAN_OFFSET) % WAN_STRIDE == 0
    assert head["frame_num"] == (head["assembly_overlap"] + head["new_frames"]
                                 + head["tail_trim_new"])
    assert head["shared_cut_frame"] == 1
    # The classic job on the same file and cut delivers the request itself: the
    # DIV-2 length gap is one frame on the 4/1 grid (up to three), where LTX2's
    # 8/1 grid can put it at seven.
    assert total_delivered(wan_classic_plan(100, WAN_WINDOW)) == 100


def test_the_rule_defers_to_the_classic_plan_inside_a_wan_window():
    # A cut inside the window: the rule completes the containing window, so the
    # branch and the classic job land on the same length - the parity relation
    # that holds wherever the rule does not override the planner.
    point, head = wan_rule(21, 100)
    assert int(point["kept_frames"]) == 21 and not head["cut_at_window_end"]
    assert (head["frame_num"], head["new_frames"], head["tail_trim_new"]) == (105, 100, 0)
    assert head["frame_num"] <= wan_frames(WAN_LATENTS)
    assert head["new_frames"] == total_delivered(wan_classic_plan(100, 21))
    # Past the window budget the tail windows are the classic planner's own.
    _point, long_head = wan_rule(21, 300)
    assert long_head["frame_num"] == WAN_WINDOW
    assert long_head["new_frames"] == wan_classic_plan(300, 21)[0]["output_frames"]


def test_the_handler_decides_which_wan_classes_re_encode_their_head():
    # The declaration is the gate, read from the handler itself: the guide-door classes
    # and the plain text-to-video classes build their head from the kept region's
    # decode. Every other Wan class - a control video, a second stream, a model of its
    # own - is not that head yet and says so by declaring nothing.
    from models.wan.wan_handler import family_handler
    defs = {name: family_handler.query_model_def(name, {})
            for name in family_handler.query_supported_types()}
    assert {n for n, d in defs.items() if d.get("branch_pixel_rule") == "guide_pixels"} == {
        "i2v", "i2v_2_2", "ti2v_2_2"}
    assert {n for n, d in defs.items() if d.get("branch_pixel_rule") == "encoded_head"} == {
        "t2v", "t2v_1.3B", "t2v_2_2", "shotplan_t2v", "shotplan_t2v_2_2"}
    assert latent_jobs.branch_head_takes_rule("i2v", defs["i2v"])
    assert latent_jobs.branch_head_takes_rule("t2v_1.3B", defs["t2v_1.3B"])
    assert not latent_jobs.branch_head_takes_rule("vace_14B", defs["vace_14B"])


def test_a_wan_class_that_declares_nothing_keeps_the_frozen_prefix():
    # The same call with no declaration: the whole kept run is still the head overlap
    # and no bridge is planned - the shape every family keeps until its own port lands.
    prefix, point, error = wan_branch_prefix(wan_sidecar(), wan_frames(WAN_LATENTS), 100)
    assert error == "" and prefix is not None
    assert prefix["head_mode"] == "full_freeze"
    assert prefix["assembly_overlap"] == int(point["kept_frames"]) == WAN_WINDOW
    assert "overlap_decoded" not in prefix and "audio_bridge_latents" not in prefix


def test_the_wan_guide_door_takes_the_rule_and_keeps_the_kept_run():
    # The same call on a class that declares the rule: the head overlap becomes the
    # bridge, while kept_frames stays the whole kept run - the sidecar and the output's
    # prefix still cover it, and only the window's frozen head shrinks.
    declared = wan_door_prefix(wan_frames(WAN_LATENTS), 100)
    assert declared["head_mode"] == "pixel_rule"
    assert (declared["assembly_overlap"] == declared["guide_frames"]
            == declared["overlap_decoded"] == WAN_OVERLAP)
    assert declared["kept_frames"] == WAN_WINDOW > declared["assembly_overlap"]
    assert int(declared["prefix_latents"].shape[1]) == declared["head_prefix_latents"]
    # Wan writes no audio latents, so the rule plans no audio bridge for them.
    assert declared["audio_bridge_latents"] == 0


def test_the_encoded_head_is_the_same_plan_under_the_other_door():
    # A text-only class on the same file and cut: the plan's numbers are the door's
    # numbers, because the two heads differ inside the engine - one is fed to a guide
    # door, one is frozen - and not in what the plan asks the engine for.
    encoded = wan_encode_prefix(wan_frames(WAN_LATENTS), 100)
    door = wan_door_prefix(wan_frames(WAN_LATENTS), 100)
    assert encoded["head_mode"] == door["head_mode"] == "pixel_rule"
    for key in ("assembly_overlap", "guide_frames", "overlap_decoded", "overlap_latents",
                "head_prefix_latents", "kept_frames", "head_frame_num", "head_output",
                "head_trim_new", "cut_at_window_end", "shared_cut_frame",
                "audio_bridge_latents", "prefix_source"):
        assert encoded[key] == door[key], key
    assert int(encoded["prefix_latents"].shape[1]) == encoded["head_prefix_latents"]


def test_the_wan_plan_hands_the_encode_a_head_it_can_build():
    # Both heads are an encode of the plan's overlap: one latent for the first frame
    # and one per stride frames after it. That is the count the plan froze wherever the
    # cut sits and wherever the bridge clamps to the kept run, which is the check the
    # engine makes before it decodes anything
    # (models/wan/any2video.py branch_encode_head).
    for cut, request, overlap in ((WAN_WINDOW, 100, WAN_OVERLAP), (WAN_WINDOW, 100, 13),
                                  (21, 100, WAN_OVERLAP), (45, 200, 9), (WAN_WINDOW, 8, 5),
                                  (5, 100, 13)):
        for build in (wan_door_prefix, wan_encode_prefix):
            prefix = build(cut, request, overlap=overlap)
            head = 1 + (int(prefix["guide_frames"]) - WAN_OFFSET) // WAN_STRIDE
            assert head == prefix["head_prefix_latents"] == int(prefix["prefix_latents"].shape[1]), \
                (cut, overlap, prefix["guide_frames"], prefix["head_prefix_latents"])
            assert prefix["assembly_overlap"] == prefix["guide_frames"] <= int(prefix["kept_frames"])


def test_the_door_gets_the_bridges_own_frames_from_the_kept_run():
    # One decode serves the guide and the output's kept prefix: the guide is the tail
    # the plan asked for, at the levels a guide is handed over in, and the run it came
    # from is untouched so the assembly can still be built from it.
    prefix = wan_door_prefix(WAN_WINDOW, 100, overlap=13)
    run, guide = wan_guide_frames(prefix)
    assert tuple(guide.shape)[1] == prefix["overlap_decoded"] == 13
    assert guide.dtype == torch.float32
    assert torch.equal(guide, run[:, -13:].float().div_(127.5).sub_(1.0))
    assert run.dtype == torch.uint8 and int(run.shape[1]) == WAN_WINDOW


def test_the_rule_follows_the_capability_declaration_and_not_the_family_name():
    # The gate asks the model def how the head takes pixels, so a family joins the rule
    # by declaring it and not by being named in the planner.
    assert latent_jobs.branch_pixel_rule_capability(LTX2, None) == "guide_pixels"
    assert latent_jobs.branch_pixel_rule_capability(
        LTX2, WAN_ENCODE_HEAD) == "encoded_head"
    assert latent_jobs.branch_pixel_rule_capability(WAN, WAN_ENCODE_HEAD) == "encoded_head"
    assert latent_jobs.branch_pixel_rule_capability(WAN, WAN_GUIDE_DOOR) == "guide_pixels"
    assert latent_jobs.branch_pixel_rule_capability(WAN, {}) is None
    assert latent_jobs.branch_head_takes_rule(LTX2, None)
    assert latent_jobs.branch_head_takes_rule(WAN, WAN_ENCODE_HEAD)
    assert not latent_jobs.branch_head_takes_rule(WAN, {})
    # An unknown wording is not a capability: the family keeps frozen latents.
    assert latent_jobs.branch_pixel_rule_capability(
        WAN, {"branch_pixel_rule": "yes please"}) is None


def test_the_declared_capability_reaches_the_branch_plan():
    # A family that declares the capability gets the rule from the same call that
    # leaves an undeclared family on frozen latents: the declaration is the gate.
    point, _head = wan_rule(wan_frames(WAN_LATENTS), 100)
    entry = latent_io.latent_payload(wan_sidecar())
    carried = latent_io.carry_over_latent_windows(wan_sidecar(), point)
    declared, _err = latent_jobs._prepare_branch_prefix(
        entry, point, carried, base_model_type=WAN, video_length=100,
        reuse_frames=WAN_OVERLAP, audio_entry=None, vae_file=WAN_VAE,
        vae_sha256=WAN_VAE_SHA, head_mode="pixel_rule",
        model_def=WAN_ENCODE_HEAD)
    assert declared is not None and declared["head_mode"] == "pixel_rule"
    assert declared["overlap_decoded"] == WAN_OVERLAP
    plain, _err = latent_jobs._prepare_branch_prefix(
        entry, point, carried, base_model_type=WAN, video_length=100,
        reuse_frames=WAN_OVERLAP, audio_entry=None, vae_file=WAN_VAE,
        vae_sha256=WAN_VAE_SHA, head_mode="pixel_rule")
    assert plain["head_mode"] == "full_freeze"


def test_the_head_helper_encodes_the_bridges_own_frames():
    # The shared head builder is the port's other half: it encodes exactly the frames
    # of the bridge and hands back the latents the plan froze - counted by the encode's
    # rule, so a bridge clamped to a short kept run is met at its own length instead of
    # being rounded back up onto a grid it no longer starts on.
    frames = torch.arange(3 * 60 * 2 * 2, dtype=torch.uint8).reshape(3, 60, 2, 2)
    for stride, offset, overlap in ((4, 1, 13), (4, 1, 16), (4, 1, 3), (8, 1, 13), (8, 1, 8)):
        frozen = latent_branch_head.head_latent_count(overlap, stride, offset)
        given = {}

        def _encode(run):
            given["run"] = run
            return torch.zeros(
                4, latent_branch_head.head_latent_count(int(run.shape[1]), stride, offset), 1, 1)

        head = latent_branch_head.head_latents(
            _encode, frames, overlap, frozen, stride=stride, offset=offset)
        assert tuple(head.shape) == (4, frozen, 1, 1)
        assert torch.equal(given["run"], frames[:, -overlap:, :, :])


def test_the_head_helper_refuses_a_head_it_cannot_build():
    # The rule is only honoured when the encode lands on the plan's latent count;
    # anything else stops the job instead of freezing a head of the wrong span.
    frames = torch.zeros(3, 20, 2, 2, dtype=torch.uint8)
    with pytest.raises(latent_branch_head.BranchRefusal):
        latent_branch_head.head_latents(
            lambda run: torch.zeros(4, 9, 1, 1), frames, 13, 2, stride=4, offset=1)
    with pytest.raises(latent_branch_head.BranchRefusal):
        latent_branch_head.head_latents(
            lambda run: None, frames, 13, 4, stride=4, offset=1)
    with pytest.raises(latent_branch_head.BranchRefusal):
        latent_branch_head.head_latents(
            lambda run: torch.zeros(4, 4, 1, 1), None, 13, 4, stride=4, offset=1)


def test_the_decode_helper_refuses_a_kept_region_that_is_not_there():
    # The kept region has one source, so a decode that fails or comes up short is
    # a refusal: the job never continues on frames read from somewhere else.
    def _broken_window(spec):
        raise ValueError("window 1 could not be decoded")

    with pytest.raises(latent_branch_head.BranchRefusal):
        latent_branch_head.kept_frames(_broken_window, [{"window_no": 1}], 8)

    def _short_window(spec):
        return torch.zeros(3, 2, 2, 2, dtype=torch.uint8)

    with pytest.raises(latent_branch_head.BranchRefusal):
        latent_branch_head.kept_frames(_short_window, [{"window_no": 1}], 8)


# ---------------------------------------------------------------------------
# The other three sub-architectures: LongCat, Magi Human and Ovi
#
# Each of them brings its own door to the head window, so the declaration says
# which one the rule uses: LongCat's video class hands the kept decode's tail to
# the door a video continuation already reads ("guide_pixels"), while Magi Human
# and Ovi freeze one run of latents at the tensor head and have no multi-frame
# door, so the encode of those frames is that head ("encoded_head") - the shape
# the plain Wan text-to-video classes use. LongCat's two avatar classes are driven
# by an audio guide and an identity image and declare nothing yet. The gate is the
# one the Wan rows read, so these rows pin that the declaration and not the family
# name decides, on each family's own layout: Magi's 48-channel 16x Turbo VAED
# latent, LongCat's 16-channel 8x one.
# ---------------------------------------------------------------------------
MAGI = "magi_human"
MAGI_VAE, MAGI_VAE_SHA = "turbo_vaed.safetensors", "cd44"
MAGI_STRIDE, MAGI_OFFSET = 4, 1               # the Turbo VAED's temporal layout
MAGI_CHANNELS = 48                            # 48 channels, 16x spatial
MAGI_LATENTS = 26                             # one clean 101-frame window
MAGI_FPS = 25.0                               # magi_human_handler.py
MAGI_OVERLAP = 5                              # 1 + 4*1, on the family's overlap grid
MAGI_ENCODE_HEAD = {"branch_pixel_rule": "encoded_head"}    # magi_human_handler.py

LONGCAT = "longcat_video"
LONGCAT_AVATAR = "longcat_avatar"
LONGCAT_VAE, LONGCAT_VAE_SHA = WAN_VAE, WAN_VAE_SHA   # the Wan 2.1 8x 16-channel VAE
LONGCAT_FPS = 15.0                            # longcat_handler.py
LONGCAT_GUIDE_DOOR = {"branch_pixel_rule": "guide_pixels"}  # longcat_handler.py


def magi_frames(latents):
    """The frames a run of ``latents`` decodes to on Magi's 4/1 layout."""
    return MAGI_OFFSET + MAGI_STRIDE * (int(latents) - 1)


def magi_sidecar():
    """A one-window 101-frame Magi Human source file (memoized)."""
    if "magi" not in _SIDECARS:
        run = torch.zeros(MAGI_CHANNELS, MAGI_LATENTS, 16, 16, dtype=torch.float16)
        _SIDECARS["magi"] = latent_io.build_latent_sidecar(
            [run], [{"prompt": "window 1", "seed": 7, "steps": 8,
                     "denoising_strength": 1.0, "head_trim": 0, "tail_trim": 0,
                     "frame_count": magi_frames(MAGI_LATENTS)}],
            app_version=APP_VERSION, model_type="magi_human", base_model_type=MAGI,
            model_filename="magi.safetensors", loras=(),
            vae_file=MAGI_VAE, vae_sha256=MAGI_VAE_SHA, vae_z_dim=MAGI_CHANNELS,
            vae_dtype="32", width=256, height=256, fps=MAGI_FPS,
            latent_stride=MAGI_STRIDE, frame_offset=MAGI_OFFSET, is_hdr=False)
    return _SIDECARS["magi"]


def magi_prefix(cut, request, *, overlap=MAGI_OVERLAP, mode="pixel_rule", model_def=None):
    """The prefix half of a Magi Human branch job, as ``wgp`` computes it."""
    entry = latent_io.latent_payload(magi_sidecar())
    point = latent_io.branch_point_for_frame(entry, cut)
    assert point is not None, f"no branch point at frame {cut}"
    carried = latent_io.carry_over_latent_windows(magi_sidecar(), point)
    prefix, error = latent_jobs._prepare_branch_prefix(
        entry, point, carried, base_model_type=MAGI, video_length=int(request),
        reuse_frames=overlap, audio_entry=None, model_def=model_def,
        vae_file=MAGI_VAE, vae_sha256=MAGI_VAE_SHA, head_mode=mode)
    assert error == "" and prefix is not None, error
    return prefix


def longcat_sidecar():
    """A one-window 129-frame LongCat source file (memoized)."""
    if "longcat" not in _SIDECARS:
        run = torch.zeros(WAN_CHANNELS, WAN_LATENTS, 2, 2, dtype=torch.float16)
        _SIDECARS["longcat"] = latent_io.build_latent_sidecar(
            [run], [{"prompt": "window 1", "seed": 7, "steps": 20,
                     "denoising_strength": 1.0, "head_trim": 0, "tail_trim": 0,
                     "frame_count": wan_frames(WAN_LATENTS)}],
            app_version=APP_VERSION, model_type="longcat_video", base_model_type=LONGCAT,
            model_filename="longcat.safetensors", loras=(),
            vae_file=LONGCAT_VAE, vae_sha256=LONGCAT_VAE_SHA, vae_z_dim=WAN_CHANNELS,
            vae_dtype="32", width=256, height=256, fps=LONGCAT_FPS,
            latent_stride=WAN_STRIDE, frame_offset=WAN_OFFSET, is_hdr=False)
    return _SIDECARS["longcat"]


def longcat_prefix(base_model_type, cut, request, *, overlap=WAN_OVERLAP,
                   mode="pixel_rule", model_def=None):
    """The prefix half of a LongCat branch job on either of its two classes."""
    entry = latent_io.latent_payload(longcat_sidecar())
    point = latent_io.branch_point_for_frame(entry, cut)
    assert point is not None, f"no branch point at frame {cut}"
    carried = latent_io.carry_over_latent_windows(longcat_sidecar(), point)
    prefix, error = latent_jobs._prepare_branch_prefix(
        entry, point, carried, base_model_type=base_model_type, video_length=int(request),
        reuse_frames=overlap, audio_entry=None, model_def=model_def,
        vae_file=LONGCAT_VAE, vae_sha256=LONGCAT_VAE_SHA, head_mode=mode)
    assert error == "" and prefix is not None, error
    return prefix


def test_the_three_new_families_declare_their_head_door():
    # Read from the handlers themselves: LongCat's video class names the door a
    # continuation reads, its two avatar classes say nothing yet, and both Magi arches
    # name the frozen encode.
    from models.longcat.longcat_handler import family_handler as longcat_family_handler
    from models.magi_human.magi_human_handler import family_handler as magi_family_handler
    longcat_defs = {name: longcat_family_handler.query_model_def(name, {})
                    for name in longcat_family_handler.query_supported_types()}
    magi_defs = {name: magi_family_handler.query_model_def(name, {})
                 for name in magi_family_handler.query_supported_types()}
    assert {n for n, d in longcat_defs.items() if d.get("branch_pixel_rule")} == {LONGCAT}
    assert longcat_defs[LONGCAT]["branch_pixel_rule"] == "guide_pixels"
    assert {n for n, d in magi_defs.items() if d.get("branch_pixel_rule")} == {
        "magi_human", "magi_human_distill"}
    assert all(d["branch_pixel_rule"] == "encoded_head" for d in magi_defs.values())
    assert latent_jobs.branch_head_takes_rule(LONGCAT, longcat_defs[LONGCAT])
    assert latent_jobs.branch_head_takes_rule("magi_human_distill", magi_defs["magi_human_distill"])
    assert not latent_jobs.branch_head_takes_rule(LONGCAT_AVATAR, longcat_defs[LONGCAT_AVATAR])


def test_ovi_declares_the_encoded_head_like_the_plain_wan_classes():
    # Ovi shares the Wan 2.2 5B VAE and freezes its video prefix the same way, so it
    # names the same door - and its handler sits beside the Wan one.
    from models.wan.ovi_handler import family_handler as ovi_family_handler
    ovi_def = ovi_family_handler.query_model_def("ovi", {})
    assert ovi_def.get("branch_pixel_rule") == "encoded_head"
    assert latent_jobs.branch_head_takes_rule("ovi", ovi_def)


def test_a_magi_job_takes_the_rule_only_when_the_family_declares_it():
    # The same file and cut, twice: with the declaration the head is the bridge and its
    # encode, without one the whole kept run is still frozen. The family's 4/1 layout
    # decides the numbers, not its name.
    declared = magi_prefix(wan_frames(MAGI_LATENTS), 100, model_def=MAGI_ENCODE_HEAD)
    assert declared["head_mode"] == "pixel_rule"
    assert (declared["assembly_overlap"] == declared["guide_frames"]
            == declared["overlap_decoded"] == MAGI_OVERLAP)
    assert declared["head_prefix_latents"] == int(declared["prefix_latents"].shape[1]) == 2
    assert declared["kept_frames"] == magi_frames(MAGI_LATENTS) > declared["guide_frames"]
    # The soundtrack is the kept run's own: with no audio entry the rule plans no audio
    # bridge, and with one it slices the flat block to these same frames.
    assert declared["audio_bridge_latents"] == 0

    plain = magi_prefix(wan_frames(MAGI_LATENTS), 100)
    assert plain["head_mode"] == "full_freeze"
    assert plain["assembly_overlap"] == int(plain["kept_frames"])
    assert "overlap_decoded" not in plain


def test_a_longcat_video_job_takes_the_door_and_an_avatar_job_does_not():
    # One handler, two answers: the video class names the guide door and gets the bridge,
    # the audio-and-identity avatar class declares nothing and keeps the kept run frozen.
    video = longcat_prefix(LONGCAT, WAN_WINDOW, 100, model_def=LONGCAT_GUIDE_DOOR)
    avatar = longcat_prefix(LONGCAT_AVATAR, WAN_WINDOW, 100)
    assert video["head_mode"] == "pixel_rule"
    assert (video["assembly_overlap"] == video["guide_frames"]
            == video["overlap_decoded"] == WAN_OVERLAP)
    assert int(video["prefix_latents"].shape[1]) == video["head_prefix_latents"]
    assert avatar["head_mode"] == "full_freeze"
    assert avatar["assembly_overlap"] == int(avatar["kept_frames"]) == WAN_WINDOW
    assert video["head_frame_num"] != avatar["head_frame_num"]


# ---------------------------------------------------------------------------
# The Continuation Overlap: the handoff a job gets when it continues a run
# ---------------------------------------------------------------------------

PINNED_OVERLAP = 1     # wan_handler.py: the Wan 5B ti2v overlap grid is pinned to one frame
LIFTED_OVERLAP = 33    # the same setting raised, on this family's own grid (1 + 8*4)


def test_the_continuation_overlap_lifts_the_handoff_of_a_continuation_only():
    # The setting is a maximum, never a reduction, and it speaks only to the jobs
    # that continue a run of frames: everything else keeps the family's own value.
    assert continuation_window_overlap(PINNED_OVERLAP, LIFTED_OVERLAP, continuing=True) == LIFTED_OVERLAP
    assert continuation_window_overlap(PINNED_OVERLAP, LIFTED_OVERLAP, continuing=False) == PINNED_OVERLAP
    assert continuation_window_overlap(PINNED_OVERLAP, 0, continuing=True) == PINNED_OVERLAP
    assert continuation_window_overlap(OVERLAP, 5, continuing=True) == OVERLAP
    assert continuation_window_overlap(5, None, continuing=True) == 5
    assert continuation_window_overlap(5, "13", continuing=True) == 13


def test_a_start_image_is_no_continuation_to_inherit_motion_from():
    # A start image is a single frame: there is no speed to carry on, so the
    # setting stays silent. A video source and a latent branch both own a run.
    assert job_continues_video("/tmp/source.mp4") is True
    assert job_continues_video(None, branch_from_latent=True) is True
    assert job_continues_video([None]) is False
    assert job_continues_video("   ") is False
    assert job_continues_video(None) is False


def test_the_lifted_handoff_widens_the_classic_and_the_branch_head_together():
    # The two paths hand the model the same run of frames at the same cut: the
    # classic planner takes it as the first window's overlap, the branch head as
    # its guide. The pinned one-frame handoff is the case the setting exists for:
    # the new window reads the last frame's appearance and none of its motion.
    row = case_row("overlap on the grid at 33")
    sidecar = ltx2_sidecar(row["latents"], row.get("head_trims", ()))
    narrow_prefix, _point, narrow_error = branch_prefix(
        sidecar, row["cut"], row["request"], overlap=PINNED_OVERLAP)
    wide_prefix, _point, wide_error = branch_prefix(
        sidecar, row["cut"], row["request"], overlap=LIFTED_OVERLAP)
    assert narrow_error == "" and wide_error == ""
    assert narrow_prefix is not None and wide_prefix is not None
    assert int(narrow_prefix["guide_frames"]) == PINNED_OVERLAP
    assert int(wide_prefix["guide_frames"]) == LIFTED_OVERLAP
    assert LIFTED_OVERLAP == continuation_window_overlap(PINNED_OVERLAP, LIFTED_OVERLAP, continuing=True)
    narrow = classic_plan(row["request"], kept=row["kept"], overlap=PINNED_OVERLAP)
    wide = classic_plan(row["request"], kept=row["kept"], overlap=LIFTED_OVERLAP)
    assert int(narrow[0]["overlap_frames"]) == PINNED_OVERLAP
    assert int(wide[0]["overlap_frames"]) == LIFTED_OVERLAP


# ---------------------------------------------------------------------------
# The handoff's latent grid: the number the job runs and the number the job log
# announces must be one number, not two
# ---------------------------------------------------------------------------

WAN_GRID = dict(step=4, offset=1)     # Wan 2.2: latent_size 4, overlap_offset 1 -> 1, 5, 9, 13 ...
ZERO_GRID = dict(step=4, offset=0)    # SVI2 Pro: overlap_offset 0 -> whole latent frames from zero


def test_a_handoff_exists_only_in_whole_latent_frames():
    # The frames a window inherits are re-denoised inside the window's latent
    # block, so a handoff of 15 frames on a 4-frame grid is not a handoff of 15
    # frames. The slider, the planner and the job log all owe this one rule.
    assert [align_overlap(one, 4, 1) for one in (0, 1, 5, 9, 13, 17, 97)] == [0, 1, 5, 9, 13, 17, 97]
    assert [align_overlap(one, 4, 1) for one in (2, 4, 6, 14, 15, 20, 24, 96)] == [1, 1, 5, 13, 13, 17, 21, 93]
    assert [align_overlap(one, 4, 0) for one in (3, 4, 8, 15, 97)] == [0, 4, 8, 12, 96]
    assert align_overlap(15, 8, 1) == 9          # LTX2 counts its handoffs by 8
    assert align_overlap(30, 17, 5) == 22        # the family whose series starts at 5


def test_the_handoff_the_job_runs_is_the_one_the_log_announces():
    # The receipt used to print the setting and the planner to floor it afterwards,
    # so a Continuation Overlap of 15 promised 15 and ran 13.
    lifted = resolve_continuation_handoff(1, 15, continuing=True, **WAN_GRID)
    assert lifted["head"] == 13 and lifted["snapped"] and not lifted["vacuous"]
    on_grid = resolve_continuation_handoff(1, 17, continuing=True, **WAN_GRID)
    assert on_grid["head"] == 17 and not on_grid["snapped"]
    assert resolve_continuation_handoff(1, 97, continuing=True, **WAN_GRID)["head"] == 97
    assert resolve_continuation_handoff(5, "13", continuing=True, **WAN_GRID)["head"] == 13
    assert resolve_continuation_handoff(6, 15, continuing=True, **WAN_GRID)["head"] == 13
    assert resolve_continuation_handoff(15, 15, continuing=True, **WAN_GRID)["head"] == 0
    assert resolve_continuation_handoff(1, 17, continuing=True, **ZERO_GRID)["head"] == 16
    # A job that continues nothing has no handoff to widen.
    quiet = resolve_continuation_handoff(5, 33, continuing=False, **WAN_GRID)
    assert quiet["head"] == 0 and quiet["window_overlap"] == 5 and not quiet["vacuous"]


def test_a_lift_the_grid_undoes_says_so_instead_of_claiming_a_wider_handoff():
    # The rendered arm D: a Continuation Overlap of 15 against a Windows Frames
    # Overlap of 13 both land on 13, so the setting changes nothing.
    vacuous = resolve_continuation_handoff(13, 15, continuing=True, **WAN_GRID)
    assert vacuous["head"] == 0 and vacuous["vacuous"] is True and vacuous["window_overlap"] == 13
    assert resolve_continuation_handoff(13, 17, continuing=True, **WAN_GRID)["vacuous"] is False


def test_a_handoff_wider_than_the_footage_or_the_window_is_reported():
    # Two clamps the setting cannot see, because neither is a property of the model:
    # how many frames the continued video holds once its frame rate is matched to
    # the job's, and how much room the window leaves for frames it must re-denoise.
    assert continuation_source_shortfall(21, 17) == 17
    for head, available in ((17, 17), (17, 81), (0, 17), (21, 0), (21, None), (None, 17)):
        assert continuation_source_shortfall(head, available) is None
    assert continuation_window_shortfall(65, 69, 4) is None     # the window leaves exactly one block
    assert continuation_window_shortfall(66, 69, 4) == 65
    assert continuation_window_shortfall(69, 69, 4) == 65
    assert continuation_window_shortfall(81, 69, 4) == 65
    for handoff, window in ((17, 69), (17, 0), (0, 69), (None, 69)):
        assert continuation_window_shortfall(handoff, window, 4) is None


def test_the_announced_handoff_is_the_handoff_the_plan_runs():
    # The four rendered arms, on their own geometry: Wan 2.2 (a 4-frame stride from
    # frame 1), a 69-frame window and 81 frames of footage to inherit. The receipt's
    # number and the first window's overlap are one number now - including the arm
    # whose setting changed nothing, where the honest announcement is that there is
    # nothing to announce and the plan keeps the window overlap it already had.
    for window_overlap, requested, announced, arm in ((1, 5, 5, "A and B"), (1, 17, 17, "C"), (13, 15, 13, "D")):
        handoff = resolve_continuation_handoff(window_overlap, requested, continuing=True, **WAN_GRID)
        if requested != window_overlap:
            assert (handoff["head"] or window_overlap) == announced, arm
        ran = min(69 - 4, handoff["head"] if handoff["head"] else window_overlap)   # wgp.py:8607
        plan = build_default_window_plan(
            total_frames=80, window_size=69, default_overlap=ran, discard_last_frames=0,
            minimum=5, step=4, frame_offset=1, overlap_offset=1, max_overlap=None,
            first_window_overlap=min(max(1, ran), 81), first_window_available_overlap=81,
            initial_shared_frames=1)
        assert int(plan[0]["overlap_frames"]) == ran, arm
    # A handoff wider than the footage, and one wider than the window: both were
    # clamped in silence, and both are the sort of thing a slider cannot warn about.
    assert continuation_source_shortfall(21, 81) is None and continuation_source_shortfall(97, 81) == 81
    assert continuation_window_shortfall(81, 69, 4) == 65
    assert len(build_default_window_plan(
        total_frames=80, window_size=69, default_overlap=65, discard_last_frames=0, minimum=5,
        step=4, frame_offset=1, overlap_offset=1, max_overlap=None,
        first_window_overlap=65, first_window_available_overlap=81, initial_shared_frames=1)) == 20


def test_the_classic_reuse_is_cut_by_the_stride_not_the_shortest_generation():
    # ``wgp.py:8607`` bounds the handoff by the window minus one latent frame: the
    # inherited frames live inside the window, which still adds up to its full
    # size. The family's shortest generation bounds a run of new frames instead,
    # and the baseline used to read it here - which only shows up within a stride
    # of the window size, so that is exactly where the rule is pinned, on both
    # families the harness models.
    wide = shape(classic_plan(200, overlap=230, kept=250))
    assert wide[:2] == [(16, 225), (8, 9)]          # 230 floored onto the 1+8k grid
    assert wide[0][1] > WINDOW_SIZE - FRAMES_MINIMUM
    wan_wide = shape(wan_classic_plan(100, 129, overlap=126))
    assert wan_wide[:2] == [(4, 125), (4, 1)]       # 126 -> the window minus one Wan latent frame
    assert wan_wide[0][1] > WAN_WINDOW - WAN_MINIMUM
    # The frozen rows sit far from the bound, which is why no golden number moved.
    assert shape(classic_plan(200, kept=250)) == [(200, 17)]
    assert shape(wan_classic_plan(100, 129)) == [(100, 5)]


def test_the_continuation_slider_offers_only_what_the_model_can_denoise():
    from shared import extra_settings
    wan = {"frames_steps": 4, "sliding_window_defaults": {
        "overlap_min": 1, "overlap_max": 1, "overlap_step": 0, "overlap_default": 1}}
    assert extra_settings.continuation_overlap_grid(wan) == (1, 97, 4)
    assert [extra_settings.continuation_overlap_value(wan, one) for one in (0, 1, 15, 24, 105)] == [1, 1, 13, 21, 97]
    ltx2 = {"frames_steps": 8, "sliding_window_defaults": {
        "overlap_min": 1, "overlap_max": 97, "overlap_step": 8, "overlap_default": 9}}
    assert extra_settings.continuation_overlap_grid(ltx2) == (1, 97, 8)
    svi = {"frames_steps": 4, "sliding_window_defaults": {
        "overlap_min": 4, "overlap_max": 4, "overlap_step": 0, "overlap_default": 4, "overlap_offset": 0}}
    # 97 is not a handoff on a family that counts 4, 8, 12 from zero: the ceiling of the
    # control is brought onto the grid too, so the top of the slider is a value the job
    # runs instead of one it rewrites. This row moved from (4, 97, 4) with that rule.
    assert extra_settings.continuation_overlap_grid(svi) == (4, 96, 4)
    assert extra_settings.continuation_overlap_value(svi, 9) == 8
    assert extra_settings.continuation_overlap_value(svi, 97) == 96
    assert extra_settings.continuation_overlap_grid(None) == (1, 97, 4)
    # Ovi divides a window in blocks of 120 frames and pins its own overlap to one, so one
    # handoff is all the grid leaves under the ceiling. The control says exactly that: a
    # range of one value, rather than 1..97 in strides of 120 that the job floors to 1.
    ovi = {"latent_size": 120, "sliding_window_defaults": {
        "overlap_min": 1, "overlap_max": 1, "overlap_step": 0, "overlap_default": 1}}
    assert extra_settings.continuation_overlap_grid(ovi) == (1, 1, 120)
    assert extra_settings.continuation_overlap_value(ovi, 21) == 1
    # Magi Human declares an overlap stride of 1 - the stride of its *Windows Frames Overlap*
    # control - and divides a window in blocks of 4. The handoff ladder is the block, because
    # the block is what the planner floors by: a control stepping by 1 would offer 2, and the
    # job would run 1 and have to say so.
    magi = {"latent_size": 4, "sliding_window_defaults": {
        "overlap_min": 1, "overlap_max": 1, "overlap_step": 1, "overlap_default": 1}}
    assert extra_settings.continuation_overlap_grid(magi) == (1, 97, 4)
    assert extra_settings.continuation_overlap_value(magi, 2) == 1
    assert extra_settings.continuation_overlap_value(magi, 15) == 13

    # The control's floor and the planner's floor are one rule restated twice (that
    # module stays free of engine imports), so pin them against each other.
    for value in range(1, 98):
        assert extra_settings.continuation_overlap_value(wan, value) == align_overlap(value, 4, 1)
        assert extra_settings.continuation_overlap_value(svi, value) == align_overlap(value, 4, 0) or value < 4


def test_the_job_log_reports_the_resolved_handoff_rather_than_the_setting():
    # A wiring check, because the bug was in what got printed rather than in the
    # arithmetic: the receipt must speak the resolver's number, and the two clamps
    # the setting cannot see have to be spoken about at all.
    text = _source("wgp.py")
    block = text[text.index("resolve_continuation_handoff("):][:2200]
    assert 'continuation_head_frames = _handoff["head"]' in block
    assert "now inherits {continuation_head_frames}" in block
    assert 'elif _handoff["vacuous"]' in block
    assert "{sliding_window_continuation_overlap}" not in block
    assert "continuation_source_shortfall(continuation_head_frames, first_window_available_overlap)" in text
    assert "continuation_window_shortfall(continuation_head_frames, sliding_window_size, latent_size)" in text
    assert "align_overlap(sliding_window_overlap, latent_size, overlap_offset)" in text
    # A queued job has its window plan validated by a mirror of this arithmetic, so the
    # mirror is floored the same way rather than checking a plan the job will not build.
    assert "job_overlap = align_overlap(job_overlap, latent_size," in text


def test_wgp_imports_every_frame_scheduler_helper_it_calls():
    # The handoff rule moved into a shared helper, and a call left behind by that move
    # fails only when a real job reaches the line. This catches it from the module's own
    # list of helpers, so it keeps working as helpers are added.
    import shared.utils.frame_scheduler as scheduler
    text = _source("wgp.py")
    header = text.split("from shared.utils.frame_scheduler import", 1)[1].split("\n", 1)[0]
    imported = {one.strip() for one in header.split(",")}
    helpers = {one for one, value in vars(scheduler).items()
               if callable(value) and not one.startswith("_")
               and getattr(value, "__module__", "") == scheduler.__name__}
    called = {one for one in helpers if (one + "(") in text and ("def " + one + "(") not in text}
    assert called <= imported, f"wgp.py calls {sorted(called - imported)} without importing them"


def _source(*parts):
    """The text of a file of the repository, for the checks that read a wiring rather than a value."""
    return pathlib.Path(REPO_ROOT, *parts).read_text(encoding="utf-8")


def test_a_continuation_is_matched_to_the_frame_it_continues():
    # A continuation hands the model a run of frames and claims no reference to match the new
    # window to: that is why the Color Correction Strength slider has no effect on a join. The
    # reference is the last frame of that run and nothing else - a path that already owns a
    # reference keeps it, and an off strength must not ask for a frame at all, so that a plain
    # generation decodes nothing extra on the way to a correction it will not apply.
    kept = torch.arange(3 * 9 * 2 * 2, dtype=torch.uint8).reshape(3, 9, 2, 2)
    ref = continuation_color_reference(1.0, None, kept)
    assert ref is not None and ref.dtype == torch.float32 and tuple(ref.shape) == (3, 1, 2, 2)
    assert torch.equal(ref, kept[:, -1:].to(torch.float32).div_(127.5).sub_(1.0))
    assert float(ref.min()) >= -1.0 and float(ref.max()) <= 1.0
    claimed = torch.zeros(3, 1, 2, 2)
    assert continuation_color_reference(1.0, claimed, kept) is claimed      # the start image wins
    assert continuation_color_reference(0, None, kept) is None              # off asks for nothing
    assert continuation_color_reference(None, None, kept) is None
    assert continuation_color_reference(1.0, None, None) is None            # no run to continue
    assert continuation_color_reference(1.0, None, torch.zeros(3, 0, 2, 2, dtype=torch.uint8)) is None
    assert continuation_color_reference("5", None, kept) is not None        # a slider may hand a string
    assert continuation_color_reference("off", None, kept) is None          # and must not crash on it


def test_matching_a_continuation_leaves_the_frames_it_hands_over_untouched():
    # The last kept frame is read out of a tensor the job still owns: the branch's cached decode of
    # the source's kept windows, the classic path's prefix video. A match that retouched those
    # frames would move the footage the job continues instead of bringing the new segment to it, so
    # the reference is a copy in guide levels - and the kept run stays uint8, byte for byte.
    kept = torch.arange(3 * 5 * 2 * 2, dtype=torch.uint8).reshape(3, 5, 2, 2)
    before = kept.clone()
    ref = continuation_color_reference(1.0, None, kept)
    assert kept.dtype == torch.uint8 and torch.equal(kept, before)
    assert ref.data_ptr() != kept.data_ptr()
    ref.fill_(0.25)
    assert torch.equal(kept, before) and kept.dtype == torch.uint8
    guide = torch.linspace(-1.0, 1.0, 3 * 4 * 2 * 2).reshape(3, 4, 2, 2)
    again = continuation_color_reference(1.0, None, guide)                  # already in guide levels
    assert torch.equal(again, guide[:, -1:]) and again.data_ptr() != guide.data_ptr()
    again.mul_(0.0)
    assert torch.equal(guide, torch.linspace(-1.0, 1.0, 3 * 4 * 2 * 2).reshape(3, 4, 2, 2))


def test_the_continuation_match_reaches_the_engine_and_survives_the_metadata():
    # The name carries no sliding_window_ prefix on purpose - the join is not about the cadence of
    # the windows, which is the whole reason a second slider exists - so what ties it to its group
    # is membership in _SLIDING, whose capability decides whether a key lives on in a job's
    # metadata. In the engine it reads as the sibling of color_correction_strength, and it reaches
    # every window: which frame each one is pulled onto is the anchor's question, not the strength's.
    from shared import extra_settings
    from shared.settings_metadata import _SLIDING

    assert "continuation_color_strength" in _SLIDING
    assert "continuation_color_strength" in extra_settings.iter_defs(None)
    assert extra_settings.get_def("continuation_color_strength", {"color_correction": True}).visible
    assert not extra_settings.get_def("continuation_color_strength", {}).visible

    app = _source("wgp.py")
    assert "    continuation_color_strength,\n" in app                       # a generate_media parameter
    assert "continuation_color = float(continuation_color_strength or 0)" in app
    assert "if not job_continues_video(video_source, branch_from_latent):" in app
    assert "continuation_color_correction_strength = continuation_color," in app
    assert "continuation_color_lock = continuation_color_lock," in app
    assert "continuation_color_anchor_frame = continuation_color_anchor_frame," in app
    assert app.index("continuation_color = float(continuation_color_strength or 0)") < app.index(
        "continuation_color_correction_strength = continuation_color,")

    engine = _source("models", "wan", "any2video.py")
    assert "continuation_color_correction_strength = 0," in engine           # other engines swallow it
    assert "_kept_region() if prefix_kept_windows is not None else prefix_video" in engine
    taken = engine.index("_cont_cc_ref = continuation_color_window_anchor(")
    paint = engine.index("elif _cont_cc_ref is not None:")
    assert taken < paint < engine.index("elif color_reference_frame is not None:")
    assert engine.index("videos = videos[0] # return only first video") < paint
    assert paint < engine.index("videos = videos.clamp_(-1, 1)")
    # the hand-off frames are skipped, so the footage the job continues is never repainted
    assert "videos[:, _cc_head:]" in engine and "torch.cat([videos[:, :_cc_head], _cc_made]" in engine
    # and a path that cannot be matched says so, instead of silently ignoring the setting
    assert "Color correction: nothing to match this window to" in engine


def test_the_anchor_says_which_frame_a_window_is_matched_to():
    # Two frames can supply the same six numbers, and the two only part company from the second
    # window on: a chained anchor follows the run wherever its light goes, a locked one keeps every
    # window matched to the frame the join started from. The rule is small enough to say once, so it
    # is said once and the engine merely asks it - which also means the choice is testable without
    # a GPU, a model or a queue.
    from shared.utils.utils import continuation_color_lock_anchor

    footage = torch.zeros(3, 1, 2, 2)
    previous = torch.ones(3, 1, 2, 2)
    assert continuation_color_lock_anchor(0, footage, previous) is previous    # chained: the neighbour
    assert continuation_color_lock_anchor(1, footage, previous) is footage     # locked: the footage
    assert continuation_color_lock_anchor(1, None, previous) is previous       # window 1 has no past
    assert continuation_color_lock_anchor(0, None, previous) is previous
    assert continuation_color_lock_anchor("1", footage, previous) is footage   # a widget may hand text
    assert continuation_color_lock_anchor("off", footage, previous) is previous
    assert continuation_color_lock_anchor(None, footage, previous) is previous # and hands it empty

    from shared import extra_settings
    from shared.settings_metadata import _SLIDING

    assert "continuation_color_anchor" in _SLIDING                             # lives on in metadata
    assert "continuation_color_anchor" in extra_settings.iter_defs(None)
    assert extra_settings.get_def("continuation_color_anchor", {"color_correction": True}).visible
    assert not extra_settings.get_def("continuation_color_anchor", {}).visible # rides with the match

    app = _source("wgp.py")
    assert "    continuation_color_anchor,\n" in app                           # a generate_media parameter
    assert "continuation_color_lock = 1 if int(continuation_color_anchor or 0) else 0" in app
    assert "continuation_color_anchor = setting_choice(" in app                # words, not a 0-1 slider
    assert '("The last frame of the previous window", 0),' in app
    assert '("The last frame of the video being continued", 1),' in app
    assert "_cont_cc_anchor.detach().cpu()" in app                             # never held in VRAM

    engine = _source("models", "wan", "any2video.py")
    assert "continuation_color_lock = 0," in engine                            # other engines swallow it
    assert "continuation_color_anchor_frame = None," in engine
    assert "continuation_color_window_anchor(" in engine                       # one decision, not three
    assert 'ret["continuation_color_anchor"] = _cont_cc_ref' in engine         # the frame goes back

    # The decision must be taken where the hand-off already exists as pixels: an anchor chosen
    # before the decode cannot see the frames the window was conditioned on, and that is how a
    # chained run came to promise a match it never made.
    anchor_at = engine.index("continuation_color_window_anchor(")
    assert anchor_at > engine.index("videos = self.vae.decode_to_cpu_uint8")
    assert "videos[:, :_cc_head] if _cc_head > 0 else None" in engine          # the hand-off is offered
    assert "videos[:, _cc_head:]" in engine                                    # and left unretouched


def test_a_window_after_the_first_still_finds_a_frame_to_match_to():
    # A rendered job caught the chained choice doing nothing past its first window: the classic
    # continuation hands the second window no footage, so the reference resolved to nothing, the
    # whole match was skipped, and the run printed a promise it had not kept. What a later window
    # continues is the hand-off it carries in as its own prefix - the previous window as it was
    # actually delivered - so that stands in for the footage, and a locked job still prefers the
    # frame carried down the loop to both of them.
    def last(frames):
        return frames[:, -1:].to(torch.float32).div_(127.5).sub_(1.0)

    footage = torch.arange(3 * 6 * 2 * 2, dtype=torch.uint8).reshape(3, 6, 2, 2)
    handoff = torch.full((3, 4, 2, 2), 200, dtype=torch.uint8)

    first = continuation_color_window_anchor(0.5, None, footage, 0, None, handoff)
    assert torch.equal(first, last(footage))                     # window 1: the footage answers
    second = continuation_color_window_anchor(0.5, None, None, 0, None, handoff)
    assert torch.equal(second, last(handoff))                    # window 2: so does the hand-off
    assert not torch.equal(second, last(footage))                # and they are not the same frame

    carried = torch.zeros(3, 1, 2, 2)
    assert continuation_color_window_anchor(0.5, None, None, 1, carried, handoff) is carried
    assert continuation_color_window_anchor(0.5, None, footage, 1, carried, handoff) is carried

    assert continuation_color_window_anchor(0.5, None, None, 0, None, None) is None
    assert continuation_color_window_anchor(0, None, None, 0, None, handoff) is None     # off asks nothing
    assert continuation_color_window_anchor(0, None, footage, 0, None, handoff) is None
    claimed = torch.zeros(3, 1, 2, 2)
    # A path that owns a reference keeps it, and a locked job still overrides it with the frame the
    # join was matched to - which is that same reference, since it is the frame window 1 answered
    # with. Locked means "hold the light the job started from", not "fall back to the neighbour".
    assert continuation_color_window_anchor(0.5, claimed, footage, 0, None, handoff) is claimed
    assert continuation_color_window_anchor(0.5, claimed, footage, 1, carried, handoff) is carried
    assert continuation_color_window_anchor(0.5, claimed, footage, 1, None, handoff) is claimed

    before = handoff.clone()
    continuation_color_window_anchor(0.5, None, None, 0, None, handoff)
    assert handoff.dtype == torch.uint8 and torch.equal(handoff, before)     # the hand-off is not retouched










