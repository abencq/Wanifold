"""Tests for the moved latent branch engine (shared/utils/latent_jobs.py).

Run from the repository root:

    venv/bin/python -m pytest tests/test_latent_jobs.py -v

The engine is tested headless: its JobContext is stubbed (the model
registry lookups are dummies) and the sidecars are produced through the
engine's own writer (LatentWindowRecorder) into a temp directory, so no
wgp import and no model weights are needed. The branch-point and
multi-window values asserted here are the 4/1 linear layout's own
arithmetic: an 8-latent window decodes to 29 frames, its latent
boundaries sit at frames 1, 5, 9, ..., 29, and a sliding window 2 that
reuses 16 frames commits 13 new ones (29 + 13 = 42).
"""

import os
import sys
import threading
import types

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from shared.utils import latent_io, latent_jobs

APP_VERSION = "17.17"
WAN = "wan_14_14B"
LTX2 = "ltx2_22B"


def stub_ctx(tmp_path, arch="t2v", handlers=None, settings=None):
    """A JobContext with dummy registry lookups (the engine's family and
    bookkeeping paths never touch the real model catalog)."""
    return latent_jobs.JobContext(
        server_config={}, save_path=str(tmp_path), WanGP_version=APP_VERSION, verbose_level=0,
        lock=threading.Lock(), primary_settings=settings or {},
        model_types_handlers=handlers or {},
        transformer_quantization="", text_encoder_quantization="",
        get_model_def=lambda model_type: {"architecture": arch},
        get_model_record=lambda *args, **kwargs: "Latent Decode",
        get_model_config_groups=lambda *args, **kwargs: None,
        get_base_model_type=lambda model_type: model_type,
        get_available_filename=lambda *args, **kwargs: "",
        clear_status=lambda *args, **kwargs: None,
        get_gen_info=lambda state: {},
        resolve_last_video=lambda state: None)


def make_block(seed=7, prompt="a prompt"):
    """One window's latent payload: 8 latents = a 29-frame window (4/1)."""
    return latent_io.build_latent_payload(
        torch.randn(16, 8, 64, 36, dtype=torch.float16),
        app_version=APP_VERSION, model_type=WAN, base_model_type=WAN,
        model_filename="wan.safetensors", vae_file="Wan2.1_VAE.pth", vae_sha256="ab12",
        vae_z_dim=16, vae_dtype="16", frames=29, fps=16.0, seed=seed, prompt=prompt,
        denoising_strength=1.0, steps=30, latent_stride=4, frame_offset=1)


def make_ltx2_block(seed=7, prompt="a prompt"):
    """One LTX2 window: 3 latents = a 17-frame window (8/1)."""
    return latent_io.build_latent_payload(
        torch.randn(128, 3, 4, 6, dtype=torch.float16),
        app_version=APP_VERSION, model_type=LTX2, base_model_type=LTX2,
        model_filename="ltx2.safetensors", vae_file="ltx2_vae.safetensors", vae_sha256="cd34",
        vae_z_dim=128, vae_dtype="16", frames=17, fps=24.0, seed=seed, prompt=prompt,
        denoising_strength=1.0, steps=30, latent_stride=8, frame_offset=1)


def window_args(**overrides):
    """The per-window bookkeeping of a plain (non-branch) window 1."""
    args = dict(sliding_window=False, latent_windows=[], latent_audio_windows=[],
                reuse_frames=0, post_decode_pre_trim=0, trim_first_frames=0,
                source_video_overlap_frames_count=0, discard_last_frames=0,
                automatic_trim_last_frames=0, window_prefix_is_image=False,
                sample_is_hdr=False, image_start_tensor=None, written_frame_count=29)
    args.update(overrides)
    return args


def write_sidecar(ctx, tmp_path, stem, *, base_model_type=WAN, model_type=WAN,
                  model_def=None, windows=1, reuse=0, per_window=29,
                  written_frame_count=None, block=None):
    """Produce a sidecar the production way (the engine's own recorder)
    and return its path."""
    rec = latent_jobs.LatentWindowRecorder(ctx, model_def=model_def or {},
                                           model_type=model_type, base_model_type=base_model_type)
    media = str(tmp_path / f"{stem}.mp4")
    args = window_args()
    committed = 0
    for window_no in range(1, windows + 1):
        if window_no > 1:
            args["sliding_window"] = True
            args["reuse_frames"] = reuse
        committed += per_window - (reuse if window_no > 1 else 0)
        args["written_frame_count"] = (written_frame_count
                                       if written_frame_count is not None else committed)
        rec.on_window(block if block is not None else make_block(), {}, media, window_no, **args)
    return latent_io.latent_sidecar_path(media)


def load_entry(path):
    return latent_io.latent_payload(latent_io.load_latent_file(path))


@pytest.fixture
def sidecar(tmp_path):
    """A 29-frame single-window Wan sidecar, written by the engine."""
    return write_sidecar(stub_ctx(tmp_path), tmp_path, "out")


# ---------------------------------------------------------------------------
# The model-family predicates (the all-latent writer's gate)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("base", ["ltx2_22B", "joyai_echo"])
def test_ltx2_family_base_types(base):
    assert latent_jobs._is_ltx2_family(base, {})


@pytest.mark.parametrize("base", ["minimax_h3_fl2va", "minimax_h3_fl2va_pruned", "viggle_animate"])
def test_h3_family_base_types(base):
    assert latent_jobs._is_h3_family(base, {})


@pytest.mark.parametrize("base", ["magi_human", "magi_human_distill"])
def test_magi_family_base_types(base):
    assert latent_jobs._is_magi_human_family(base, {})


def test_ovi_family_base_type():
    assert latent_jobs._is_ovi_family("ovi", {})


@pytest.mark.parametrize("base", ["t2v", "i2v_2_2", "multitalk", "standin"])
def test_wan_family_base_types(base, tmp_path):
    assert latent_jobs._is_wan_family(stub_ctx(tmp_path), base, {})


@pytest.mark.parametrize("base", ["longcat_video", "longcat_avatar_v1_5"])
def test_longcat_family_base_types(base, tmp_path):
    assert latent_jobs._is_longcat_family(stub_ctx(tmp_path), base, {})


def test_family_predicates_architecture_fallback(tmp_path):
    # a base type the sets do not know falls back to the architecture name
    # (wan / longcat additionally ask the model registry, so they need a ctx)
    ctx = stub_ctx(tmp_path)
    assert latent_jobs._is_ltx2_family(None, {"architecture": "ltx2_22B"})
    assert latent_jobs._is_h3_family(None, {"architecture": "minimax_h3_ref2va"})
    assert latent_jobs._is_magi_human_family(None, {"architecture": "magi_human_distill"})
    assert latent_jobs._is_ovi_family(None, {"architecture": "ovi"})
    assert latent_jobs._is_wan_family(ctx, None, {"architecture": "wan"})
    assert latent_jobs._is_wan_family(ctx, None, {"architecture": "t2v"})
    assert latent_jobs._is_longcat_family(ctx, None, {"architecture": "longcat_video"})


@pytest.mark.parametrize("arch", ["hunyuan_video", "ltx_19B", "wan_x", ""])
def test_family_predicates_negatives(arch, tmp_path):
    ctx = stub_ctx(tmp_path)
    assert not latent_jobs._is_ltx2_family(None, {"architecture": arch})
    assert not latent_jobs._is_h3_family(None, {"architecture": arch})
    assert not latent_jobs._is_magi_human_family(None, {"architecture": arch})
    assert not latent_jobs._is_ovi_family(None, {"architecture": arch})
    assert not latent_jobs._is_wan_family(ctx, None, {"architecture": arch})
    assert not latent_jobs._is_longcat_family(ctx, None, {"architecture": arch})
    assert not latent_jobs._is_all_latent_family(ctx, None, {"architecture": arch})


def test_all_latent_family_union(tmp_path):
    ctx = stub_ctx(tmp_path)
    for base in ("ltx2_22B", "t2v", "longcat_video", "minimax_h3_fl2va", "magi_human"):
        assert latent_jobs._is_all_latent_family(ctx, base, {}), base


def test_all_latent_family_exclusions(tmp_path):
    ctx = stub_ctx(tmp_path)
    # the old LTX builds and the image-driven models never write a sidecar
    for base in ("hunyuan_video", "ltx_19B"):
        assert not latent_jobs._is_all_latent_family(ctx, base, {}), base


def test_wan_family_handler_module_fallback(tmp_path):
    # a base type outside the set is decided by the model registry: a
    # handler living in the models.wan package is a Wan-family model
    class _WanHandler:
        __module__ = "models.wan.some_handler"
    ctx = stub_ctx(tmp_path, handlers={"custom_type": _WanHandler()})
    assert latent_jobs._is_wan_family(ctx, "custom_type", {})
    assert latent_jobs._is_all_latent_family(ctx, "custom_type", {})


def test_longcat_family_handler_module_fallback(tmp_path):
    handler = types.ModuleType("models.longcat.longcat_handler")
    ctx = stub_ctx(tmp_path, handlers={"custom_type": handler})
    assert latent_jobs._is_longcat_family(ctx, "custom_type", {})


def test_all_latent_family_ovi_via_handler(tmp_path):
    # Ovi rides the models.wan package: without the registry it is not a
    # Wan base type, with it the all-latent gate opens for it
    assert not latent_jobs._is_all_latent_family(stub_ctx(tmp_path), "ovi", {})
    class _OviHandler:
        __module__ = "models.wan.ovi_handler"
    ctx = stub_ctx(tmp_path, handlers={"ovi": _OviHandler()})
    assert latent_jobs._is_all_latent_family(ctx, "ovi", {})


def test_branch_latent_search_dirs(tmp_path):
    ctx = stub_ctx(tmp_path)
    assert latent_jobs._branch_latent_search_dirs(ctx) == [
        str(tmp_path), os.path.join(str(tmp_path), "videos")]


def test_branch_latent_search_dirs_includes_the_setting(tmp_path):
    ctx = stub_ctx(tmp_path, settings={"branch_latent_search_dirs": ["/a", "", " /b "]})
    assert latent_jobs._branch_latent_search_dirs(ctx) == [
        str(tmp_path), os.path.join(str(tmp_path), "videos"),
        "/a", "/a/videos", "/b", "/b/videos"]


# ---------------------------------------------------------------------------
# resolve_latent_branch (the branch point resolution)
# ---------------------------------------------------------------------------

def _resolve(ctx, keep_frames, sidecar_path, fit=False, model_type=WAN,
             base_model_type=WAN, model_def=None):
    return latent_jobs.resolve_latent_branch(ctx, "V", "", None, model_type,
                                             keep_frames=keep_frames, fit=fit,
                                             base_model_type=base_model_type,
                                             model_def=model_def or {"architecture": "t2v"},
                                             latent_file=sidecar_path)


def test_resolve_branch_at_end(tmp_path, sidecar):
    info, error = _resolve(stub_ctx(tmp_path), "", sidecar)
    assert error == "" and info is not None
    assert info["branch_frame"] == 29
    assert info["branch"]["kept_frames"] == 29
    assert info["branch"]["latent_index"] == 8
    assert info["branch"]["mid_latent"] is False
    assert info["video_present"] is False
    assert info["prefill_prompt"] == "a prompt"
    assert info["prefill_seed"] == 0
    assert any("no companion video" in w for w in info["warnings"])


def test_resolve_branch_mid_latent_default(tmp_path, sidecar):
    info, error = _resolve(stub_ctx(tmp_path), "14", sidecar)
    assert error == "" and info is not None
    assert info["branch_frame"] == 14
    # the default keeps up to the latent boundary before the cut: frame 14
    # lies inside latent 4 (frames 13..16), so the kept run ends at 13 and
    # the whole containing latent is re-generated
    assert info["branch"]["kept_frames"] == 13
    assert info["branch"]["latent_index"] == 4
    assert info["branch"]["mid_latent"] is True


def test_resolve_branch_fit_rounds_up(tmp_path, sidecar):
    info, error = _resolve(stub_ctx(tmp_path), "14", sidecar, fit=True)
    assert error == "" and info is not None
    assert info["branch"]["kept_frames"] == 17
    assert info["branch"]["latent_index"] == 5
    assert info["branch"]["fit"] is True
    assert info["branch"]["mid_latent"] is False


def test_resolve_branch_on_boundary_is_its_own_cut(tmp_path, sidecar):
    for fit in (False, True):
        info, error = _resolve(stub_ctx(tmp_path), "13", sidecar, fit=fit)
        assert error == "" and info is not None
        assert info["branch"]["kept_frames"] == 13
        assert info["branch"]["latent_index"] == 4
        assert info["branch"]["mid_latent"] is False
        assert info["branch"]["fit"] is False


def test_resolve_branch_negative_counts_from_the_end(tmp_path, sidecar):
    info, error = _resolve(stub_ctx(tmp_path), "-3", sidecar)
    assert error == "" and info is not None
    assert info["branch_frame"] == 26
    assert info["branch"]["kept_frames"] == 25
    assert info["branch"]["mid_latent"] is True




@pytest.mark.parametrize("keep, fragment", [
    ("abc", "not a valid frame count"),
    ("999", "outside the range recorded in the latent file (1..29)"),
    ("0", "outside the range recorded in the latent file (1..29)"),
])
def test_resolve_branch_bad_keep_frames(tmp_path, sidecar, keep, fragment):
    info, error = _resolve(stub_ctx(tmp_path), keep, sidecar)
    assert info is None and fragment in error


def test_resolve_branch_refuses_a_wrong_family(tmp_path):
    ctx = stub_ctx(tmp_path)
    ltx = write_sidecar(ctx, tmp_path, "ltx", base_model_type="ltx2_22B",
                        model_type="ltx2_22B", per_window=57)
    info, error = _resolve(ctx, "", ltx)
    assert info is None and "different model family (ltx2_22B)" in error
    longcat = write_sidecar(ctx, tmp_path, "lc", base_model_type="longcat_video",
                            model_type="longcat_video")
    info, error = _resolve(ctx, "", longcat)
    assert info is None and "saved by model family 'longcat_video'" in error
    # the same family passes the gate
    info, error = _resolve(ctx, "", longcat, model_type="longcat_video",
                           base_model_type="longcat_video",
                           model_def={"architecture": "longcat_video"})
    assert error == "" and info is not None
    assert info["branch"]["kept_frames"] == 29


def test_resolve_branch_missing_inputs(tmp_path, sidecar):
    ctx = stub_ctx(tmp_path)
    # the gate: the mode requires a Continue Video / Last Video location
    info, error = latent_jobs.resolve_latent_branch(ctx, "T", "", None, WAN,
        base_model_type=WAN, model_def={"architecture": "t2v"}, latent_file=sidecar)
    assert info is None
    assert error == "requires Continue Video / Continue Last Video to be selected"
    # no file, no video
    info, error = latent_jobs.resolve_latent_branch(ctx, "V", "", None, WAN,
        base_model_type=WAN, model_def={"architecture": "t2v"})
    assert info is None and error == "no source video was found for the branch job"
    # a named file that is gone, and no video to look its sidecar up
    info, error = latent_jobs.resolve_latent_branch(ctx, "V", "", None, WAN,
        base_model_type=WAN, model_def={"architecture": "t2v"},
        latent_file=str(tmp_path / "gone_latent.pt"))
    assert info is None and "could not be found (gone_latent.pt)" in error
    # a file that is not a WanGP latent file
    bad = tmp_path / "bad_latent.pt"
    bad.write_bytes(b"not a latent file")
    info, error = latent_jobs.resolve_latent_branch(ctx, "V", "", None, WAN,
        base_model_type=WAN, model_def={"architecture": "t2v"}, latent_file=str(bad))
    assert info is None
    assert error.startswith("the latent file bad_latent.pt could not be read")


def test_resolve_branch_refuses_a_flagged_sidecar(tmp_path):
    ctx = stub_ctx(tmp_path)
    flagged = write_sidecar(ctx, tmp_path, "flag", written_frame_count=99)
    info, error = _resolve(ctx, "", flagged)
    assert info is None
    assert "recorded length (29 frames) does not match the video it was written for (99 frames)" in error


# ---------------------------------------------------------------------------
# validate_latent_branch_task (the queue-time check)
# ---------------------------------------------------------------------------

def _task_inputs(sidecar_path):
    return {"branch_from_latent": True, "image_prompt_type": "VL", "video_source": "",
            "keep_frames_video_source": "", "branch_fit_latent": False,
            "branch_latent_file": sidecar_path}


def test_validate_task_dormant(tmp_path, sidecar):
    ctx = stub_ctx(tmp_path)
    inputs = _task_inputs(sidecar)
    # the checkbox is off: the whole check is a no-op
    assert latent_jobs.validate_latent_branch_task(ctx, None, WAN,
                                                   dict(inputs, branch_from_latent=False)) == ""
    # an image type with no video-letter branch source: nothing to check
    assert latent_jobs.validate_latent_branch_task(ctx, None, WAN,
                                                   dict(inputs, image_prompt_type="T")) == ""


def test_validate_task_dormant_for_a_non_latent_family(tmp_path, sidecar):
    # the family gate: a model family the app never writes a sidecar for
    # has nothing to refuse (the all-latent check is not reachable from here)
    ctx = stub_ctx(tmp_path, arch="hunyuan_video")
    assert latent_jobs.validate_latent_branch_task(ctx, None, "hunyuan_video",
                                                   _task_inputs(sidecar)) == ""


def test_validate_task_requires_a_source(tmp_path):
    ctx = stub_ctx(tmp_path)
    assert latent_jobs.validate_latent_branch_task(ctx, None, WAN, _task_inputs("")) == \
        "From latent: select a source video or a latent file"


def test_validate_task_ok_and_refusal(tmp_path, sidecar):
    ctx = stub_ctx(tmp_path)
    assert latent_jobs.validate_latent_branch_task(ctx, None, WAN, _task_inputs(sidecar)) == ""
    bad = tmp_path / "bad_latent.pt"
    bad.write_bytes(b"not a latent file")
    error = latent_jobs.validate_latent_branch_task(ctx, None, WAN,
                                                    _task_inputs(str(bad)))
    assert error.startswith("From latent: the latent file bad_latent.pt")
def test_validate_task_refuses_start_image(tmp_path, sidecar):
    ctx = stub_ctx(tmp_path)
    inputs = dict(_task_inputs(sidecar), image_start=object())
    error = latent_jobs.validate_latent_branch_task(ctx, None, WAN, inputs)
    assert "Start Image cannot be used on a branch" in error


def test_validate_task_refuses_reference_images_on_a_non_ltx2_branch(tmp_path, sidecar):
    ctx = stub_ctx(tmp_path)
    inputs = dict(_task_inputs(sidecar), image_refs=[object()])
    error = latent_jobs.validate_latent_branch_task(ctx, None, WAN, inputs)
    assert "does not condition the new segment" in error


def test_validate_task_refuses_control_video_on_a_non_ltx2_branch(tmp_path, sidecar):
    ctx = stub_ctx(tmp_path)
    inputs = dict(_task_inputs(sidecar), video_guide="control.mp4")
    error = latent_jobs.validate_latent_branch_task(ctx, None, WAN, inputs)
    assert "does not condition the new segment" in error


def test_conditioning_gate_accepts_ltx2_reference_images(tmp_path):
    ctx = stub_ctx(tmp_path, arch="ltx2_22B")
    inputs = dict(_task_inputs("unused.pt"), image_refs=[object()], video_prompt_type="")
    error = latent_jobs._branch_conditioning_conflict(
        ctx, LTX2, inputs, branch_info={"branch": {"kept_frames": 17}})
    assert error == ""


def test_conditioning_gate_accepts_ltx2_control_video(tmp_path):
    ctx = stub_ctx(tmp_path, arch="ltx2_22B")
    inputs = dict(_task_inputs("unused.pt"), video_guide="control.mp4", video_prompt_type="")
    error = latent_jobs._branch_conditioning_conflict(
        ctx, LTX2, inputs, branch_info={"branch": {"kept_frames": 17}})
    assert error == ""


def test_conditioning_gate_refuses_control_and_refs_together(tmp_path):
    ctx = stub_ctx(tmp_path, arch="ltx2_22B")
    inputs = dict(_task_inputs("unused.pt"), video_guide="control.mp4",
                  image_refs=[object()], video_prompt_type="")
    error = latent_jobs._branch_conditioning_conflict(
        ctx, LTX2, inputs, branch_info={"branch": {"kept_frames": 17}})
    assert "Control Video and reference images cannot both condition" in error


def test_conditioning_gate_refuses_mask_without_control(tmp_path):
    ctx = stub_ctx(tmp_path, arch="ltx2_22B")
    inputs = dict(_task_inputs("unused.pt"), video_mask=object(), video_prompt_type="")
    error = latent_jobs._branch_conditioning_conflict(
        ctx, LTX2, inputs, branch_info={"branch": {"kept_frames": 17}})
    assert "Video Mask needs the Control Video" in error


def test_conditioning_gate_refuses_audio_from_control_video(tmp_path):
    ctx = stub_ctx(tmp_path, arch="ltx2_22B")
    inputs = dict(_task_inputs("unused.pt"), video_guide="control.mp4",
                  audio_prompt_type="2", video_prompt_type="")
    error = latent_jobs._branch_conditioning_conflict(
        ctx, LTX2, inputs, branch_info={"branch": {"kept_frames": 17}})
    assert "audio-from-control-video" in error


def test_conditioning_gate_refuses_heavy_branch_family(tmp_path):
    ctx = stub_ctx(tmp_path, arch="ltx2_22B_msr")
    inputs = dict(_task_inputs("unused.pt"), image_refs=[object()], video_prompt_type="")
    error = latent_jobs._branch_conditioning_conflict(
        ctx, "ltx2_22B_msr", inputs, branch_info={"branch": {"kept_frames": 17}})
    assert "rebuilds a per-window guide" in error


def test_conditioning_gate_refuses_start_image_on_ltx2(tmp_path):
    ctx = stub_ctx(tmp_path, arch="ltx2_22B")
    inputs = dict(_task_inputs("unused.pt"), image_start=object(), video_prompt_type="")
    error = latent_jobs._branch_conditioning_conflict(
        ctx, LTX2, inputs, branch_info={"branch": {"kept_frames": 17}})
    assert "Start Image cannot be used on a branch" in error


def test_branch_reference_position_one_is_the_first_new_frame():
    # wgp resolves a branch's reference positions with the kept prefix as the
    # timeline shift: position 1 lands after the kept run, not at source frame 0.
    from shared.prompt_enhancer import images
    windows = [{"frame_num": 345, "overlap_frames": 241,
                "discard_last_frames": 0, "trim_last_frames": 0}]
    resolved = images.resolve_injected_positions(
        "1", 1, windows=windows, source_frames=241, source_overlap=241,
        window_size=241, discard_frames=0, reuse_frames=17, alignment_shift=241)
    assert resolved == [241]


def test_conditioning_gate_refuses_zero_frame_position(tmp_path):
    ctx = stub_ctx(tmp_path, arch="ltx2_22B")
    inputs = dict(_task_inputs("unused.pt"), image_refs=[object()],
                  frames_positions="0", video_prompt_type="")
    error = latent_jobs._branch_conditioning_conflict(
        ctx, LTX2, inputs, branch_info={"branch": {"kept_frames": 17}})
    assert "must start at 1 on a branch" in error


def test_conditioning_gate_refuses_aligned_to_source_start(tmp_path):
    ctx = stub_ctx(tmp_path, arch="ltx2_22B")
    inputs = dict(_task_inputs("unused.pt"), image_refs=[object()],
                  video_prompt_type="T")
    error = latent_jobs._branch_conditioning_conflict(
        ctx, LTX2, inputs, branch_info={"branch": {"kept_frames": 17}})
    assert "Aligned to the beginning of the Source Video" in error


def test_validate_task_refuses_heavy_conditioning_model(tmp_path):
    ctx = stub_ctx(tmp_path, arch="ltx2_22B_edit_anything")
    ltx = write_sidecar(ctx, tmp_path, "ltx_edit",
                        base_model_type="ltx2_22B_edit_anything",
                        model_type="ltx2_22B_edit_anything", per_window=57)
    error = latent_jobs.validate_latent_branch_task(ctx, None, "ltx2_22B_edit_anything",
                                                   _task_inputs(ltx))
    assert "per-window guide" in error







# ---------------------------------------------------------------------------
# LatentWindowRecorder (the per-window sidecar writer)
# ---------------------------------------------------------------------------

def _recorder(ctx):
    return latent_jobs.LatentWindowRecorder(ctx, model_def={}, model_type=WAN,
                                            base_model_type=WAN)


def test_recorder_all_latent_single_window(tmp_path):
    ctx = stub_ctx(tmp_path)
    media = str(tmp_path / "one.mp4")
    args = window_args()
    _recorder(ctx).on_window(make_block(), {}, media, 1, **args)
    path = latent_io.latent_sidecar_path(media)
    assert os.path.isfile(path)
    entry = load_entry(path)
    assert latent_io.is_all_latent_entry(entry)
    assert latent_io.sidecar_total_frames(entry) == 29
    assert len(latent_io.entry_window_specs(entry)) == 1
    # the window's block was accumulated for the next window's handoff
    assert len(args["latent_windows"]) == 1
    assert latent_io.sidecar_length_mismatch(entry) is None


def test_recorder_all_latent_multi_window(tmp_path):
    ctx = stub_ctx(tmp_path)
    rec = _recorder(ctx)
    media = str(tmp_path / "two.mp4")
    first = window_args()
    rec.on_window(make_block(), {}, media, 1, **first)
    second = window_args(sliding_window=True, latent_windows=first["latent_windows"],
                         reuse_frames=16, written_frame_count=42)
    rec.on_window(make_block(seed=8, prompt="second window prompt"), {}, media, 2, **second)
    entry = load_entry(latent_io.latent_sidecar_path(media))
    # 29 + (29 - the 16 reused frames) = 42 committed frames
    assert latent_io.sidecar_total_frames(entry) == 42
    specs = latent_io.entry_window_specs(entry)
    assert len(specs) == 2
    assert [s.get("head_trim") for s in specs] == [0, 16]
    assert [latent_io.entry_window_prompt(entry, s["window_no"]) for s in specs] == \
        ["a prompt", "second window prompt"]
    assert latent_io.sidecar_length_mismatch(entry) is None
    # the branch point resolves against the accumulated layout: frame 35
    # lies in window 2 (committed frames 30..42), mid-latent at 34
    point = latent_io.branch_point_for_frame(entry, 35)
    assert point["window_no"] == 2 and point["kept_frames"] == 34
    assert point["mid_latent"] is True


def test_recorder_flags_a_length_mismatch(tmp_path):
    ctx = stub_ctx(tmp_path)
    media = str(tmp_path / "flag.mp4")
    _recorder(ctx).on_window(make_block(), {}, media, 1,
                             **window_args(written_frame_count=99))
    entry = load_entry(latent_io.latent_sidecar_path(media))
    assert entry.get("length_verified") is False
    assert latent_io.sidecar_length_mismatch(entry) == (29, 99)


def test_recorder_classic_sidecar_write(tmp_path):
    # a non-all-latent job hands its classic bytes through unchanged and
    # the side_files dict is emptied (the writer owned the bytes now)
    ctx = stub_ctx(tmp_path)
    media = str(tmp_path / "classic.mp4")
    side_files = {latent_io.LATENT_SIDECAR_SUFFIX: b"classic-bytes"}
    _recorder(ctx).on_window(None, side_files, media, 1, **window_args())
    path = latent_io.latent_sidecar_path(media)
    with open(path, "rb") as fh:
        assert fh.read() == b"classic-bytes"
    assert side_files == {}


def test_recorder_without_video_entry_writes_nothing(tmp_path):
    # an audio-only job (no video latent block) must not create a file and
    # must not claim the caller's classic bytes
    ctx = stub_ctx(tmp_path)
    media = str(tmp_path / "audio.mp4")
    side_files = {latent_io.LATENT_SIDECAR_SUFFIX: b"keep-me"}
    _recorder(ctx).on_window({latent_io.LATENT_FORMAT_ID_AUDIO: {"windows": []}},
                             side_files, media, 1, **window_args())
    assert not os.path.isfile(latent_io.latent_sidecar_path(media))
    assert side_files == {latent_io.LATENT_SIDECAR_SUFFIX: b"keep-me"}


# ---------------------------------------------------------------------------
# Branch head window bookkeeping (the "From latent" sidecar)
# ---------------------------------------------------------------------------

LTX2 = "ltx2_22B"


def _ltx2_block(seed=7, prompt="an ltx2 prompt", latents=8):
    """One LTX2 window's latent payload: 8 latents = 57 frames (8/1)."""
    return latent_io.build_latent_payload(
        torch.randn(128, latents, 64, 36, dtype=torch.float16),
        app_version=APP_VERSION, model_type=LTX2, base_model_type=LTX2,
        model_filename="ltx2.safetensors", vae_file="ltx2_vae.safetensors",
        vae_sha256="cd34", vae_z_dim=128, vae_dtype="16",
        frames=8 * latents - 7, fps=24.0, seed=seed, prompt=prompt,
        denoising_strength=1.0, steps=40, latent_stride=8, frame_offset=1)


def _branch_prefix(kept_latents=4, **overrides):
    """An accepted branch prefix whose kept region is exactly the head
    window's frozen run (K = 1, non-i2v, window-end cut): 4 latents =
    25 kept frames, 25 frames of assembly overlap."""
    prefix = dict(prefix_latents=torch.randn(128, kept_latents, 64, 36, dtype=torch.float16),
                  assembly_overlap=8 * kept_latents - 7,
                  kept_frames=8 * kept_latents - 7,
                  containing_window_head_trim=0,
                  cut_at_window_end=True)
    prefix.update(overrides)
    return prefix


def _ltx2_recorder(ctx, prefix):
    return latent_jobs.LatentWindowRecorder(ctx, model_def={}, model_type=LTX2,
                                            base_model_type=LTX2, branch_prefix=prefix)


def test_recorder_branch_head_commits_kept_region(tmp_path):
    # K = 1 window-end cut: the kept latents are frozen inside the head
    # window's own run, so the head block commits them - head trim 0, no
    # stand-alone cut block, and the committed total equals the written
    # video (the reported 241->481 case, scaled to 25->57).
    ctx = stub_ctx(tmp_path)
    rec = _ltx2_recorder(ctx, _branch_prefix())
    media = str(tmp_path / "branch.mp4")
    args = window_args(source_video_overlap_frames_count=25, written_frame_count=57)
    rec.on_window(_ltx2_block(), {}, media, 1, **args)
    entry = load_entry(latent_io.latent_sidecar_path(media))
    specs = latent_io.entry_window_specs(entry)
    assert len(specs) == 1
    assert specs[0].get("head_trim") == 0
    assert latent_io.sidecar_total_frames(entry) == 57
    assert latent_io.sidecar_length_mismatch(entry) is None


def test_recorder_branch_head_keeps_assembly_overlap(tmp_path):
    # The kept run is longer than the window's frozen head (K > 1 carried
    # windows, or an i2v image frame outside the overlap): the head block
    # keeps recording the assembly overlap - today's behavior, still
    # length-flagged until the cut-block follow-up lands.
    ctx = stub_ctx(tmp_path)
    rec = _ltx2_recorder(ctx, _branch_prefix(kept_frames=40))
    media = str(tmp_path / "branch_k2.mp4")
    args = window_args(source_video_overlap_frames_count=25, written_frame_count=32)
    rec.on_window(_ltx2_block(), {}, media, 1, **args)
    entry = load_entry(latent_io.latent_sidecar_path(media))
    specs = latent_io.entry_window_specs(entry)
    assert specs[0].get("head_trim") == 25
    assert latent_io.sidecar_total_frames(entry) == 32


def test_recorder_branch_mid_window_cut_unchanged(tmp_path):
    # A mid-window cut records the containing window's own head trim
    # regardless of kept/overlap (the kept partial run commits from the
    # head block's frozen latents, as before).
    ctx = stub_ctx(tmp_path)
    rec = _ltx2_recorder(ctx, _branch_prefix(cut_at_window_end=False, kept_frames=40,
                                             containing_window_head_trim=1))
    media = str(tmp_path / "branch_mid.mp4")
    args = window_args(source_video_overlap_frames_count=25, written_frame_count=56)
    rec.on_window(_ltx2_block(), {}, media, 1, **args)
    entry = load_entry(latent_io.latent_sidecar_path(media))
    specs = latent_io.entry_window_specs(entry)
    assert specs[0].get("head_trim") == 1
    assert latent_io.sidecar_total_frames(entry) == 56
    assert latent_io.sidecar_length_mismatch(entry) is None


def test_recorder_branch_carries_i2v_image(tmp_path):
    # A branch of an LTX2 i2v sidecar keeps the source's frame 0 image:
    # the writer re-embeds it verbatim on the branch's own sidecar, and
    # the committed total still matches the written video.
    ctx = stub_ctx(tmp_path)
    png = b"\x89PNG\r\n\x1a\ncarried-image"
    prefix = _branch_prefix(containing_window_head_trim=1,
                            carried_image={"kind": "image_png", "data": png})
    rec = _ltx2_recorder(ctx, prefix)
    media = str(tmp_path / "branch_i2v.mp4")
    args = window_args(source_video_overlap_frames_count=25, written_frame_count=25)
    rec.on_window(_ltx2_block(latents=4), {}, media, 1, **args)
    entry = load_entry(latent_io.latent_sidecar_path(media))
    image = latent_io.entry_i2v_image(entry)
    assert image is not None and image["data"] == png
    assert latent_io.sidecar_total_frames(entry) == 25
    assert latent_io.sidecar_length_mismatch(entry) is None



# ---------------------------------------------------------------------------
# plan_branch_head_window (the head-window plan + prefill)
# ---------------------------------------------------------------------------

def window_plan():
    return latent_jobs.WindowPlan(
        sliding_window_size=81, default_reuse_frames=16, frames_minimum=1,
        frames_steps=1, frames_offset=1,
        sliding_window_defaults={"overlap_offset": 1, "overlap_max": None},
        any_sliding_window=True, model_def={})


def plan_case(ctx, **overrides):
    base = dict(
        branch_latent_info={"prefill_seed": 42, "prefill_prompt": "source prompt",
                            "entry": {"latent_stride": 4}, "video_present": True},
        branch_prefix=None, branch_decode=None, branch_carried=None,
        prefix_error=None, decode_reason=None, all_latent_job=False,
        base_model_type=WAN, scheduler_active=False,
        seed=None, prompt="", prompts=["p"], video_length=100, current_video_length=100,
        video_source=None, video_continuation_frames=0,
        sliding_window_discard_last_frames=0, sliding_window=False,
        frame_scheduler={"active": False}, scheduled_windows_template=[],
        default_windows_template=[{"output_frames": 100, "overlap_frames": 0,
                                   "discard_last_frames": 0, "trim_last_frames": 0,
                                   "frame_num": 100}],
        window_plan=window_plan())
    base.update(overrides)
    return latent_jobs.plan_branch_head_window(ctx, **base)


def _head_window_bp(**overrides):
    bp = {"prefix_latents": torch.zeros(16, 16, 1, 1), "cut_at_window_end": False,
          "head_output": 40, "assembly_overlap": 61, "head_trim_new": 2,
          "head_frame_num": 45, "kept_frames": 61, "guide_frames": 61}
    bp.update(overrides)
    return bp


def test_plan_case_a_head_window(tmp_path):
    out = plan_case(stub_ctx(tmp_path), branch_prefix=_head_window_bp(), branch_carried={})
    # the source job's prompt and seed prefill this one
    assert out["seed"] == 42 and out["prompt"] == "source prompt"
    assert out["prompts"] == ["source prompt"]
    head = out["default_windows_template"][0]
    # the head window reshapes: keep the overlap, trim the spliced head,
    # emit only the new frames
    assert (head["overlap_frames"], head["trim_last_frames"], head["output_frames"]) == (61, 2, 40)
    # the whole job still spans its full length
    assert sum(int(w["output_frames"]) for w in out["default_windows_template"]) == 100
    assert out["branch_prefix"] is not None
    assert out["sliding_window"] is True
    assert out["refusal"] is None

def test_plan_refuses_a_start_image_on_a_branch(tmp_path):
    # A start image would condition frame 0 of the branch, which is the
    # kept run's own frame 0; the planner refuses it instead of reshaping
    # the head around the image.
    out = plan_case(stub_ctx(tmp_path), branch_prefix=_head_window_bp(),
                    branch_carried={}, image_start=object())
    assert out["branch_prefix"] is not None
    assert out["refusal"].startswith("From latent: a Start Image cannot be used on a branch")




def test_plan_case_b_rounded_cut(tmp_path):
    # The full freeze leaves the head window's geometry to the caller: the
    # standard window keeps its length, its overlap becomes the frozen run's
    # decoded length, and the tail window stretches to cover the total.
    bp = _head_window_bp(head_output=None, head_frame_num=None, head_trim_new=0,
                         assembly_overlap=25, cut_at_window_end=True)
    out = plan_case(stub_ctx(tmp_path), branch_prefix=bp, branch_carried={},
                    default_windows_template=[{"output_frames": 80, "overlap_frames": 0,
                                               "discard_last_frames": 0, "trim_last_frames": 0,
                                               "frame_num": 80}])
    head = out["default_windows_template"][0]
    # a cut on a window boundary is a fresh window: no trim, the overlap
    # guides it, and the tail window stretches to cover the total
    assert (head["overlap_frames"], head["trim_last_frames"], head["output_frames"]) == (25, 0, 55)
    assert sum(int(w["output_frames"]) for w in out["default_windows_template"]) == 100
    assert out["branch_prefix"] is not None


def test_plan_case_b_pixel_rule_head_geometry(tmp_path):
    # The pixel rule's window-end plan carries its own head geometry (the
    # re-denoised overlap plus the request rounded up to the latent grid, the
    # overshoot trimmed) and the caller must use it: the request rides on top
    # of the kept run, so the head window covers it and no tail window is due.
    bp = _head_window_bp(prefix_latents=torch.zeros(16, 16, 1, 3), head_output=100,
                         head_frame_num=117, head_trim_new=0, assembly_overlap=17,
                         cut_at_window_end=True)
    out = plan_case(stub_ctx(tmp_path), branch_prefix=bp, branch_carried={},
                    default_windows_template=[{"output_frames": 100, "overlap_frames": 0,
                                               "discard_last_frames": 0, "trim_last_frames": 0,
                                               "frame_num": 100}])
    head = out["default_windows_template"][0]
    assert (head["frame_num"], head["overlap_frames"], head["trim_last_frames"],
            head["output_frames"]) == (117, 17, 0, 100)
    assert len(out["default_windows_template"]) == 1
    assert sum(int(w["output_frames"]) for w in out["default_windows_template"]) == 100
    assert out["branch_prefix"] is not None


def test_plan_case_b_plain_mode_grows_one_window(tmp_path):
    # A single-window job (no sliding window, no frame scheduler) writes one
    # window as one video, so the head window grows to hold the frozen run plus
    # the whole request instead of shipping a stub or spilling into a second
    # window (which would be saved as a second video). The frozen head decodes
    # to 61 frames and the layout is 4/1: a 100-frame request adds 25 latents,
    # so the window becomes 161 frames and commits exactly 100 new ones.
    bp = _head_window_bp(head_output=None, head_frame_num=None, head_trim_new=0,
                         assembly_overlap=61, cut_at_window_end=True)
    plain_plan = latent_jobs.WindowPlan(
        sliding_window_size=81, default_reuse_frames=16, frames_minimum=1,
        frames_steps=1, frames_offset=1,
        sliding_window_defaults={"overlap_offset": 1, "overlap_max": None},
        any_sliding_window=False, model_def={})
    out = plan_case(stub_ctx(tmp_path), branch_prefix=bp, branch_carried={},
                    window_plan=plain_plan,
                    default_windows_template=[{"output_frames": 61, "overlap_frames": 0,
                                               "discard_last_frames": 0, "trim_last_frames": 0,
                                               "frame_num": 61}])
    head = out["default_windows_template"][0]
    assert len(out["default_windows_template"]) == 1
    assert head["frame_num"] == 161
    assert (head["overlap_frames"], head["trim_last_frames"], head["output_frames"]) == (61, 0, 100)
    assert sum(int(w["output_frames"]) for w in out["default_windows_template"]) == 100
    assert out["branch_prefix"] is not None
    assert out["sliding_window"] is True


def test_plan_case_b_grown_window_trims_the_overshoot(tmp_path):
    # The grown window lands on the latent grid, so a request that is not a
    # whole number of latents trims the overshoot: 102 frames need 26 latents
    # (104 frames), and 2 frames go to the tail trim -- the window commits
    # exactly the requested 102 new frames on top of the kept 61.
    bp = _head_window_bp(head_output=None, head_frame_num=None, head_trim_new=0,
                         assembly_overlap=61, cut_at_window_end=True)
    plain_plan = latent_jobs.WindowPlan(
        sliding_window_size=81, default_reuse_frames=16, frames_minimum=1,
        frames_steps=1, frames_offset=1,
        sliding_window_defaults={"overlap_offset": 1, "overlap_max": None},
        any_sliding_window=False, model_def={})
    out = plan_case(stub_ctx(tmp_path), branch_prefix=bp, branch_carried={},
                    window_plan=plain_plan, video_length=102, current_video_length=102,
                    default_windows_template=[{"output_frames": 61, "overlap_frames": 0,
                                               "discard_last_frames": 0, "trim_last_frames": 0,
                                               "frame_num": 61}])
    head = out["default_windows_template"][0]
    assert len(out["default_windows_template"]) == 1
    assert head["frame_num"] == 165
    assert (head["overlap_frames"], head["trim_last_frames"], head["output_frames"]) == (61, 2, 102)


def test_plan_case_b_rejected_too_small_window(tmp_path):
    bp = _head_window_bp(head_output=None, head_frame_num=None, head_trim_new=0,
                         assembly_overlap=25, cut_at_window_end=True)
    # a 40-frame window cannot absorb a 55-frame head: the whole prefix
    # is rejected and the job runs without it
    out = plan_case(stub_ctx(tmp_path), branch_prefix=bp, branch_carried={},
                    default_windows_template=[{"output_frames": 40, "overlap_frames": 0,
                                               "discard_last_frames": 0, "trim_last_frames": 0,
                                               "frame_num": 40}])
    assert out["branch_prefix"] is None
    assert out["default_windows_template"][0]["frame_num"] == 40


def test_plan_gate1_refusal(tmp_path):
    # a canvas mismatch (no prefix could be prepared at all) is the one
    # refusal that stops the job instead of dropping the prefix
    out = plan_case(stub_ctx(tmp_path), branch_carried={"video_windows": []},
                    all_latent_job=True, prefix_error="the canvas does not match")
    assert out["refusal"].startswith("From latent: the canvas does not match")
    assert "or run the job without the latent file" in out["refusal"]


def test_plan_passthrough_without_a_branch(tmp_path):
    tpl = [{"output_frames": 100, "overlap_frames": 0, "discard_last_frames": 0,
            "trim_last_frames": 0, "frame_num": 100}]
    out = plan_case(stub_ctx(tmp_path), seed=5, prompt="mine", prompts=["mine"],
                    default_windows_template=tpl)
    assert out["seed"] == 5 and out["prompt"] == "mine" and out["prompts"] == ["mine"]
    assert out["default_windows_template"] is tpl
    assert out["branch_prefix"] is None and out["refusal"] is None



# ---------------------------------------------------------------------------
# Latent continuation by the pixel rule: the head-window plan
# (branch_head_window / _prepare_branch_prefix), the freeze shape the setting
# selects, the frozen audio bridge, and the sidecar writer specs.
#
# The arithmetic is the LTX2 8/1 layout's own: N tail latents decode to
# 1 + 8 * (N - 1) frames, so the 17-frame re-denoised overlap is N = 3,
# O = 17. The f241 source is one clean t2v window of 31 latents (241
# frames); its window-end branch keeps all 31 and re-denoises the tail
# 3 (17 frames) around the 241-frame kept prefix.
# ---------------------------------------------------------------------------

LTX2_VAE = "ltx2_vae.safetensors"
LTX2_VAE_SHA = "cd34"


def _ltx2_latents(t, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(128, t, 64, 36, generator=gen, dtype=torch.float16)


def _ltx2_entry(window_latents, head_trims=(), tail_trims=(), fps=24.0):
    """The current all-latent LTX2 sidecar (8/1) as its format-tagged
    object: one clean t2v window per latent run, with the given per-window
    trims."""
    windows = []
    for i, lat in enumerate(window_latents):
        windows.append({
            "latents": lat,
            "prompt": f"window {i + 1}",
            "seed": 7,
            "steps": 40,
            "denoising_strength": 1.0,
            "head_trim": int(head_trims[i]) if i < len(head_trims) else 0,
            "tail_trim": int(tail_trims[i]) if i < len(tail_trims) else 0,
            "frame_count": 1 + 8 * (int(lat.shape[1]) - 1),
        })
    return latent_io.build_latent_sidecar(
        [w["latents"] for w in windows], windows,
        app_version=APP_VERSION, model_type=LTX2, base_model_type=LTX2,
        model_filename="ltx2.safetensors", loras=(),
        vae_file=LTX2_VAE, vae_sha256=LTX2_VAE_SHA,
        vae_z_dim=128, vae_dtype="16", width=512, height=512, fps=fps,
        latent_stride=8, frame_offset=1, is_hdr=False)


def _ltx2_entry_block(latents, frames=None, seed=7, prompt="window 1"):
    """One window's LTX2 latent payload (the model's saved pre-decode run)."""
    if frames is None:
        frames = 8 * int(latents.shape[1]) - 7
    return latent_io.build_latent_payload(
        latents, app_version=APP_VERSION, model_type=LTX2, base_model_type=LTX2,
        model_filename="ltx2.safetensors", vae_file=LTX2_VAE, vae_sha256=LTX2_VAE_SHA,
        vae_z_dim=128, vae_dtype="16", frames=frames, fps=24.0, seed=seed, prompt=prompt,
        denoising_strength=1.0, steps=40, latent_stride=8, frame_offset=1)


def _ltx2_audio_payload(t_k=251):
    """The tagged ``wgp_latent/audio/1`` payload over the 241-frame 24 fps
    video window (the LTX2 audio VAE: 8 channels, 16 mel bins, 16 kHz, 25
    latent steps a second, so the window's 241 frames are its 10.04 s and 251
    steps) — the source's soundtrack."""
    samples = t_k * 640                        # 640 samples per audio latent step
    return latent_io.build_audio_latent_payload(
        [torch.randn(8, t_k * 16, dtype=torch.float16)],
        [{"window_no": 1, "start_sample": 0, "samples": samples}],
        app_version=APP_VERSION, model_type=LTX2, base_model_type=LTX2,
        model_filename="ltx2.safetensors", loras=(),
        audio_vae_file="ltx2_audio_vae.safetensors", audio_vae_sha256="ef56",
        audio_vae_dtype="16", sample_rate=16000, channels=8, samples=samples,
        seed=7, prompt="window 1", denoising_strength=1.0, steps=40,
        total_windows=1, vocoder_file="", mel_bins=16)


def _ltx2_audio_entry(t_k=251):
    """That payload's audio entry, unwrapped as
    ``latent_io.audio_latent_payload`` returns it in the job."""
    return latent_io.audio_latent_payload(_ltx2_audio_payload(t_k))


def _ltx2_info(entry):
    """The planner's ``branch_latent_info`` for the LTX2 fixture sidecars."""
    return {"prefill_seed": 42, "prefill_prompt": "source prompt",
            "entry": entry, "video_present": True}


def _ltx2_branch(sidecar_obj, frame, video_length=240, *, head_mode="pixel_rule",
                 reuse_frames=17, audio_entry=None, conditioning=False):
    """The ``_prepare_branch_prefix`` run for the LTX2 branch at ``frame``
    (returns ``(prefix, branch_point, error)``)."""
    entry = latent_io.latent_payload(sidecar_obj)
    branch = latent_io.branch_point_for_frame(entry, frame)
    assert branch is not None
    carried = latent_io.carry_over_latent_windows(sidecar_obj, branch)
    prefix, error = latent_jobs._prepare_branch_prefix(
        entry, branch, carried, base_model_type=LTX2,
        video_length=video_length, reuse_frames=reuse_frames,
        audio_entry=audio_entry, vae_file=LTX2_VAE, vae_sha256=LTX2_VAE_SHA,
        head_mode=head_mode, conditioning=conditioning)
    return prefix, branch, error


def _ltx2_plan(sidecar_obj, branch, requested, *, head_mode="pixel_rule", reuse_frames=17):
    return latent_io.branch_head_window(
        latent_io.latent_payload(sidecar_obj), branch, requested,
        stride=8, offset=1, head_mode=head_mode, overlap_frames=reuse_frames)


def test_pixel_rule_f241_end_cut_shape():
    # 9.1-1: the 241 -> 481 job. N = 1 + ceil((17 - 1) / 8) = 3 tail
    # latents, O = 1 + 8 * (3 - 1) = 17 frames. The plan keeps the kept
    # latents' meaning (prefix_latents = 31, R3) and hands the model the
    # tail 3; assembly_overlap / guide_frames are O.
    sidecar = _ltx2_entry([_ltx2_latents(31)])
    prefix, branch, error = _ltx2_branch(sidecar, 241)
    assert error == "" and prefix is not None
    assert branch["latent_index"] == 31 and branch["kept_frames"] == 241
    plan = _ltx2_plan(sidecar, branch, 240)
    assert plan["head_mode"] == "pixel_rule" and plan["prefix_source"] == "tail"
    assert plan["prefix_latents"] == 31          # the kept latents of K (R3)
    assert plan["head_prefix_latents"] == 3      # the latents handed to the model
    assert plan["overlap_latents"] == 3
    assert plan["overlap_decoded"] == 17
    assert plan["assembly_overlap"] == 17 and plan["guide_frames"] == 17
    assert int(prefix["prefix_latents"].shape[1]) == 3
    assert prefix["head_mode"] == "pixel_rule"
    assert prefix["head_prefix_latents"] == 3 and prefix["overlap_latents"] == 3
    assert prefix["overlap_decoded"] == 17
    assert prefix["assembly_overlap"] == 17 and prefix["guide_frames"] == 17
    assert prefix["cut_at_window_end"] is True
    assert prefix["kept_frames"] == 241
    # The prefix tensor is the tail slice of the kept run
    source = latent_io.payload_latents(latent_io.latent_payload(sidecar))
    assert torch.equal(prefix["prefix_latents"].float(), source[:, -3:].float())


def test_pixel_rule_k2_window_end_accepted():
    # 9.1-2: the K > 1 window end. The pixel rule decouples the frozen head
    # (the kept run's tail N) from the containing window's own decoded head,
    # so a trimmed window end (head trim 17) is accepted: the assembly splices
    # the window's committed run, which already excludes that head share. The
    # guide and the re-denoised overlap are both O, kept_frames is the full
    # committed run, and only the model's head is the tail 3. A tail trim would
    # still reject (it drops committed frames from the kept decode).
    sidecar = _ltx2_entry([_ltx2_latents(31), _ltx2_latents(31, seed=1)],
                          head_trims=(0, 17))
    prefix, branch, error = _ltx2_branch(sidecar, 465)
    assert branch is not None and branch["window_no"] == 2
    assert branch["latent_index"] == 31 and branch["kept_frames"] == 465
    assert error == "" and prefix is not None
    assert prefix["cut_at_window_end"] is True
    assert prefix["assembly_overlap"] == 17 == prefix["guide_frames"]
    assert prefix["kept_frames"] == 465
    assert prefix["containing_window_head_trim"] == 17
    assert prefix["head_prefix_latents"] == 3
    assert int(prefix["prefix_latents"].shape[1]) == 3
    assert prefix["head_mode"] == "pixel_rule"
    # The frozen head is the tail 3 of the kept run (the second window's tail).
    source = latent_io.payload_latents(latent_io.latent_payload(sidecar))
    assert torch.equal(prefix["prefix_latents"].float(), source[:, -3:].float())


def test_pixel_rule_k2_window_end_tail_trim_rejected():
    # The other half of the gate: a tail trim on the window end still rejects
    # under the pixel rule. Dropping the window's committed tail would leave
    # committed frames out of the kept decode the assembly splices, so the
    # relaxation stays head-only. The cut at the trimmed window's committed end
    # (474 = window 1's 241 + window 2's 30 kept latents) resolves to that
    # window end and is refused; one frame earlier is a mid-cut and is accepted.
    sidecar = _ltx2_entry([_ltx2_latents(31), _ltx2_latents(31, seed=1)],
                          head_trims=(0, 0), tail_trims=(0, 8))
    prefix, branch, error = _ltx2_branch(sidecar, 474)
    assert branch is not None and branch["window_no"] == 2
    assert branch["kept_frames"] == 474
    assert prefix is None and "clean window ends" in error


def test_pixel_rule_k2_window_end_geometry_721():
    # The actual 11:22 -> 11:56 shape. A 257-frame window 1 (33 latents) and a
    # 241-frame window 2 that keeps 224 after its 17-frame head (31 latents,
    # head_trim 17) commit 481 frames. A cut at 481 under the pixel rule is the
    # 721-frame job: the 17-frame frozen bridge plus the 241-frame request
    # rounded onto the latent grid (one of which the cut stands on) is a
    # 257-frame head window that commits 240 new frames — no tail trim.
    sidecar = _ltx2_entry([_ltx2_latents(33), _ltx2_latents(31, seed=1)],
                          head_trims=(0, 17))
    assert latent_io.sidecar_total_frames(latent_io.latent_payload(sidecar)) == 481
    prefix, branch, error = _ltx2_branch(sidecar, 481, video_length=241)
    assert error == "" and prefix is not None
    assert branch["window_no"] == 2 and branch["kept_frames"] == 481
    assert prefix["cut_at_window_end"] is True and prefix["shared_cut_frame"] == 1
    assert prefix["assembly_overlap"] == prefix["guide_frames"] == 17
    assert prefix["kept_frames"] == 481
    assert prefix["containing_window_head_trim"] == 17
    assert prefix["head_frame_num"] == 257 and prefix["head_output"] == 240
    assert prefix["head_trim_new"] == 0
    delivered = (int(prefix["head_frame_num"]) - int(prefix["assembly_overlap"])
                 - int(prefix["head_trim_new"]))
    assert delivered == 240 and int(prefix["kept_frames"]) + delivered == 721


def test_pixel_rule_mid_cut(tmp_path):
    # 9.1-3: a mid-window cut at a latent boundary (frame 41 = 6 kept
    # latents, 41 decoded). The head window is [the tail 3 frozen | the
    # new latents]; its geometry is O + the request rounded to the latent
    # grid (17 + 104 = 121), not the kept run's decoded length (41 + 104
    # = 145 under the full freeze).
    sidecar = _ltx2_entry([_ltx2_latents(31)])
    prefix, branch, error = _ltx2_branch(sidecar, 41, video_length=100)
    assert error == "" and prefix is not None and branch["latent_index"] == 6
    plan = _ltx2_plan(sidecar, branch, 100)
    assert plan["prefix_latents"] == 6
    assert plan["head_prefix_latents"] == 3 and plan["overlap_decoded"] == 17
    assert (plan["frame_num"], plan["new_frames"], plan["tail_trim_new"]) == (121, 100, 4)
    assert int(prefix["prefix_latents"].shape[1]) == 3
    assert (prefix["assembly_overlap"], prefix["guide_frames"]) == (17, 17)
    # The prefix tensor is the tail slice of the kept run (the kept 6
    # latents here, so latents 3..5 — not the source's tail).
    source = latent_io.payload_latents(latent_io.latent_payload(sidecar))
    assert torch.equal(prefix["prefix_latents"].float(), source[:, 3:6].float())
    # The planner reshapes the head window around the re-denoised overlap
    info = _ltx2_info(latent_io.latent_payload(sidecar))
    out = plan_case(stub_ctx(tmp_path), branch_latent_info=info, branch_prefix=prefix,
                    branch_carried={}, video_length=100)
    head = out["default_windows_template"][0]
    assert (head["overlap_frames"], head["trim_last_frames"], head["output_frames"],
            head["frame_num"]) == (17, 4, 100, 121)
    assert len(out["default_windows_template"]) == 1
    assert out["branch_prefix"] is not None and out["refusal"] is None


def test_pixel_rule_mid_latent_resolution():
    # 9.1-4: a pick inside a latent (frame 45) resolves to the boundary
    # before it (41) exactly as under the full freeze, and the pixel-rule
    # plan runs on that resolution.
    sidecar = _ltx2_entry([_ltx2_latents(31)])
    prefix, branch, error = _ltx2_branch(sidecar, 45, video_length=100)
    assert branch is not None and branch["mid_latent"] and branch["kept_frames"] == 41
    assert error == "" and prefix is not None
    plan = _ltx2_plan(sidecar, branch, 100)
    assert plan["prefix_latents"] == 6 and plan["head_prefix_latents"] == 3


def test_pixel_rule_n_clamped_to_kept_run():
    # 9.1-5: the kept run (3 latents, 17 frames) is shorter than the
    # requested 33-frame overlap: N clamps to the kept count and O falls
    # to the kept run's own decoded length.
    sidecar = _ltx2_entry([_ltx2_latents(31)])
    prefix, branch, error = _ltx2_branch(sidecar, 17, video_length=100, reuse_frames=33)
    assert error == "" and prefix is not None
    plan = _ltx2_plan(sidecar, branch, 100, reuse_frames=33)
    assert plan["head_prefix_latents"] == 3 and plan["overlap_latents"] == 3
    assert plan["overlap_decoded"] == 17 < 33
    assert int(prefix["prefix_latents"].shape[1]) == 3


def test_pixel_rule_request_clamped_to_cut_tail():
    # 9.1-6: a request past the cut's committed tail clamps the target:
    # the head window completes K exactly (241 = K's own decoded length,
    # no overshoot) and the committed tiling is kept 17 + new 224.
    sidecar = _ltx2_entry([_ltx2_latents(31)])
    prefix, branch, error = _ltx2_branch(sidecar, 17, video_length=500)
    assert error == "" and prefix is not None
    plan = _ltx2_plan(sidecar, branch, 500)
    total = latent_io.sidecar_total_frames(latent_io.latent_payload(sidecar))
    assert plan["new_frames"] == 224 == total - 17
    assert (plan["frame_num"], plan["tail_trim_new"]) == (241, 0)
    assert prefix["head_output"] == 224 and prefix["head_frame_num"] == 241


@pytest.mark.parametrize("reuse, n, o", [(1, 1, 1), (9, 2, 9), (10, 3, 17),
                                         (33, 5, 33), (240, 31, 241)])
def test_pixel_rule_non_default_overlaps(reuse, n, o):
    # 9.1-7: N = 1 + ceil((r - 1) / 8), O = 1 + 8 * (N - 1) — for the 8/1
    # layout O can exceed the requested overlap by up to 7 frames.
    sidecar = _ltx2_entry([_ltx2_latents(31)])
    entry = latent_io.latent_payload(sidecar)
    branch = latent_io.branch_point_for_frame(entry, 241)
    plan = _ltx2_plan(sidecar, branch, 240, reuse_frames=reuse)
    assert plan["head_prefix_latents"] == n
    assert plan["overlap_decoded"] == o
    assert plan["assembly_overlap"] == o and plan["guide_frames"] == o


def test_full_freeze_plan_byte_identical():
    # 9.1-8: the regression guard — the default (and the explicit
    # full_freeze) plan is today's dict: the same keys, the same values.
    sidecar = _ltx2_entry([_ltx2_latents(31)])
    entry = latent_io.latent_payload(sidecar)
    branch = latent_io.branch_point_for_frame(entry, 241)
    default = latent_io.branch_head_window(entry, branch, 240, stride=8, offset=1)
    explicit = latent_io.branch_head_window(entry, branch, 240, stride=8, offset=1,
                                            head_mode="full_freeze")
    assert default == explicit
    assert set(default) == {"window_no", "prefix_latents", "latent_count",
                            "decoded_prefix", "decoded_window", "start_frame",
                            "head_trim", "tail_trim", "first_new_frame",
                            "natural_new", "cut_at_window_end",
                            "requested_new_frames", "new_frames", "frame_num",
                            "tail_trim_new", "committed_start", "committed_end"}
    assert default["prefix_latents"] == 31 and default["cut_at_window_end"]
    assert default["decoded_prefix"] == 241 and default["frame_num"] == 0


def test_full_freeze_prefix_unchanged():
    # The default prefix keeps today's shape and values: no pixel-rule
    # fields, the full kept run, the full-freeze overlap.
    sidecar = _ltx2_entry([_ltx2_latents(31)])
    prefix, branch, error = _ltx2_branch(sidecar, 241, head_mode="full_freeze")
    assert error == "" and prefix is not None
    assert prefix["head_mode"] == "full_freeze"
    assert int(prefix["prefix_latents"].shape[1]) == 31
    assert prefix["assembly_overlap"] == 241 and prefix["guide_frames"] == 241
    assert "head_prefix_latents" not in prefix and "overlap_decoded" not in prefix


def test_conditioning_forces_full_freeze_and_starts_at_the_guide():
    # A branch that carries reference/Control Video conditioning cannot use
    # the pixel-rule overlap: the guide is frozen, and the conditioning grids
    # begin where the new segment begins.
    sidecar = _ltx2_entry([_ltx2_latents(31)])
    prefix, branch, error = _ltx2_branch(
        sidecar, 241, video_length=100, head_mode="pixel_rule", conditioning=True)
    assert error == "" and prefix is not None
    assert prefix["head_mode"] == "full_freeze"
    assert prefix["conditioning_start_frame"] == prefix["guide_frames"]
    assert prefix["conditioning_start_frame"] == 241
    assert prefix["kept_frames"] == branch["kept_frames"] == 241


def test_conditioning_refuses_trimmed_window_end():
    # The trimmed window-end head already has frames cut from this window's
    # start; conditioning cannot be laid over a section the frozen prefix
    # reuses.
    sidecar = _ltx2_entry([_ltx2_latents(31), _ltx2_latents(31)],
                          head_trims=(0, 1))
    prefix, branch, error = _ltx2_branch(
        sidecar, 481, video_length=100, conditioning=True)
    assert error != "" and prefix is None
    assert "conditioning cannot cut at a window end" in error


@pytest.mark.parametrize("frame, window_latents, head_trims", [
    (241, (31,), (0,)),           # the window end of K = 1 (in scope)
    (41, (31,), (0,)),            # a mid-window cut of K = 1 (in scope)
    (282, (31, 31), (0, 0)),      # a mid-window cut of K = 2 (in scope)
    (481, (33, 31), (0, 17)),     # a trimmed window end of K = 2 (in scope)
])
def test_pixel_rule_acceptance_matrix(frame, window_latents, head_trims):
    # 9.1-9 (R3): every in-scope pixel-rule plan shape is accepted with an
    # empty prefix_error — no silent fresh-encode fallback.
    sidecar = _ltx2_entry([_ltx2_latents(t, seed=i) for i, t in enumerate(window_latents)],
                          head_trims=head_trims)
    prefix, branch, error = _ltx2_branch(sidecar, frame, video_length=100)
    assert error == ""
    assert prefix is not None and prefix["head_mode"] == "pixel_rule"


def test_pixel_rule_overlap_overrides():
    # 9.1-10: assembly_overlap and guide_frames are both O (not the kept
    # run's decoded length), and kept_frames stays the full kept count.
    sidecar = _ltx2_entry([_ltx2_latents(31)])
    for frame in (241, 41):
        prefix, branch, error = _ltx2_branch(sidecar, frame, video_length=100)
        assert error == "" and prefix is not None
        assert prefix["assembly_overlap"] == 17 == prefix["guide_frames"]
        assert prefix["kept_frames"] == branch["kept_frames"]


def test_pixel_rule_planner_case_b(tmp_path):
    # 9.1-11: the Case B request the full freeze rejects ("first window
    # too small to hold the carried prefix") plans under the pixel rule;
    # the request-sized shape (kept 241 plus a 240-frame request, one of
    # which the cut stands on) is one window with the re-denoised overlap.
    sidecar = _ltx2_entry([_ltx2_latents(31)])
    entry = latent_io.latent_payload(sidecar)
    prefix, branch, error = _ltx2_branch(sidecar, 241, video_length=240)
    assert error == "" and prefix is not None
    full, _, _ = _ltx2_branch(sidecar, 241, video_length=240, head_mode="full_freeze")
    info = _ltx2_info(entry)
    small = [{"output_frames": 200, "overlap_frames": 16, "discard_last_frames": 0,
              "trim_last_frames": 0, "frame_num": 200}]
    out_full = plan_case(stub_ctx(tmp_path), branch_latent_info=info, branch_prefix=full,
                         branch_carried={}, all_latent_job=True, video_length=240,
                         current_video_length=240, default_windows_template=small)
    assert out_full["branch_prefix"] is None and out_full["refusal"] is not None
    out = plan_case(stub_ctx(tmp_path), branch_latent_info=info, branch_prefix=prefix,
                    branch_carried={}, all_latent_job=True, video_length=240,
                    current_video_length=240, default_windows_template=small)
    assert out["branch_prefix"] is not None and out["refusal"] is None
    head = out["default_windows_template"][0]
    assert head["overlap_frames"] == 17
    # One frame of the request is the frame the cut stands on, so a 240-frame
    # request delivers 239 new frames — the length the same job continued from
    # the video file lands on (241 kept + 239).
    assert prefix["shared_cut_frame"] == 1
    assert sum(int(w["output_frames"]) for w in out["default_windows_template"]) == 239
    sized = [{"output_frames": 240, "overlap_frames": 16, "discard_last_frames": 0,
              "trim_last_frames": 0, "frame_num": 257}]
    out2 = plan_case(stub_ctx(tmp_path), branch_latent_info=info, branch_prefix=prefix,
                     branch_carried={}, video_length=240, current_video_length=240,
                     default_windows_template=sized)
    assert out2["branch_prefix"] is not None
    assert len(out2["default_windows_template"]) == 1
    assert out2["default_windows_template"][0]["overlap_frames"] == 17


def test_branch_log_lines_stay_quiet_and_unnested(tmp_path, capsys):
    # The branch job log announces the cut, the kept region's source and the
    # plan, and nothing else: the normal pixel-rule mechanics (one decode
    # feeding the head window's guide) need no line, and no message nests one
    # parenthesis inside another — these lines are console prose.
    sidecar = _ltx2_entry([_ltx2_latents(31)])
    entry = latent_io.latent_payload(sidecar)
    prefix, branch, error = _ltx2_branch(sidecar, 241, video_length=240)
    assert error == "" and prefix is not None
    sized = [{"output_frames": 240, "overlap_frames": 16, "discard_last_frames": 0,
              "trim_last_frames": 0, "frame_num": 257}]
    capsys.readouterr()
    out = plan_case(stub_ctx(tmp_path), branch_latent_info=_ltx2_info(entry),
                    branch_prefix=prefix, branch_carried={}, video_length=240,
                    current_video_length=240, default_windows_template=sized)
    assert out["branch_prefix"] is not None
    lines = [line for line in capsys.readouterr().out.splitlines()
             if "From latent" in line or "plan:" in line]
    assert not any("copied unchanged" in line for line in lines)
    assert not any("keep the values of the source latent file" in line for line in lines)
    assert any(" in 1 window = " in line for line in lines)
    for line in lines:
        depth = 0
        for ch in line:
            if ch == "(":
                depth += 1
                assert depth < 2, line
            elif ch == ")":
                depth -= 1
                assert depth >= 0, line
        assert depth == 0, line
    # The full freeze does carry the kept latents unchanged, and says so.
    full, _full_branch, full_error = _ltx2_branch(sidecar, 241, video_length=240,
                                                  head_mode="full_freeze")
    assert full_error == "" and full is not None
    capsys.readouterr()
    out_full = plan_case(stub_ctx(tmp_path), branch_latent_info=_ltx2_info(entry),
                         branch_prefix=full, branch_carried={}, video_length=240,
                         current_video_length=240, default_windows_template=sized)
    assert out_full["branch_prefix"] is not None
    logged = capsys.readouterr().out
    assert "keep the values of the source latent file unchanged" in logged


def test_pixel_rule_keeps_an_audio_job_on_the_rule():
    # The audio follow-up: a source soundtrack no longer pulls the job off the
    # pixel rule, and a full-freeze request stays a full freeze with one present
    # (a source audio_entry was never the predicate for either).
    sidecar = _ltx2_entry([_ltx2_latents(31)])
    audio = _ltx2_audio_entry()
    px, _, error = _ltx2_branch(sidecar, 241, head_mode="pixel_rule", audio_entry=audio)
    assert error == "" and px is not None and px["head_mode"] == "pixel_rule"
    assert int(px["prefix_latents"].shape[1]) == 3          # the kept run's tail N
    ff, _, _ = _ltx2_branch(sidecar, 241, head_mode="full_freeze", audio_entry=audio)
    assert ff["head_mode"] == "full_freeze"
    assert int(ff["prefix_latents"].shape[1]) == 31         # the whole kept run
    assert "audio_bridge_latents" not in ff                 # its audio is the whole kept run


def test_pixel_rule_audio_bridge_window_end():
    # The frozen audio is the bridge's share of the source block, not the whole
    # kept run: at a window-end cut it is the block's tail, and its step count
    # rides on the prefix for the writer. The block spans the window's 10.04 s
    # at 25 steps a second, so the 17-frame bridge at 24 fps is 18 steps where
    # the 241-frame kept run is all 251.
    sidecar = _ltx2_entry([_ltx2_latents(31)])
    entry = latent_io.latent_payload(sidecar)
    audio = _ltx2_audio_entry()
    block = latent_io.audio_window_specs(audio)[0]["latents"]
    prefix, branch, error = _ltx2_branch(sidecar, 241, head_mode="pixel_rule", audio_entry=audio)
    assert error == "" and prefix is not None
    bridge = prefix["prefix_audio_latents"]
    assert bridge is not None and bridge.dim() == 4 and tuple(bridge.shape[:2]) == (1, 8)
    steps = int(bridge.shape[2])
    assert steps == prefix["audio_bridge_latents"] == 18
    assert int(latent_io.branch_audio_prefix_latents(audio, entry, branch).shape[2]) == 251
    # The batched [1, c, t, mel] prefix is the block's tail, laid out flat.
    torch.testing.assert_close(bridge[0].reshape(int(block.shape[0]), -1).float(),
                               block[:, -steps * 16:].float(), rtol=0, atol=0)


def test_pixel_rule_audio_bridge_mid_cut():
    # A cut inside the window: the bridge is the steps that end at the cut, not
    # the block's head — and it never reaches past the cut, so a kept run
    # shorter than the requested overlap freezes only its own audio.
    sidecar = _ltx2_entry([_ltx2_latents(31)])
    audio = _ltx2_audio_entry()
    block = latent_io.audio_window_specs(audio)[0]["latents"]
    prefix, branch, error = _ltx2_branch(sidecar, 41, video_length=100, audio_entry=audio)
    assert error == "" and prefix is not None
    # Frame 41 stands on step 43 of the block (41 / 24 s at 25 steps a second),
    # so the 17-frame bridge is the 18 steps before it: steps 25..43, not the
    # block's head.
    bridge = prefix["prefix_audio_latents"]
    assert int(bridge.shape[2]) == prefix["audio_bridge_latents"] == 18
    torch.testing.assert_close(bridge[0].reshape(int(block.shape[0]), -1).float(),
                               block[:, 25 * 16:43 * 16].float(), rtol=0, atol=0)
    # A 2-latent kept run (frame 9): the overlap clamps to the kept run (O = 9
    # frames), so the bridge is the 9 steps before the cut — all the audio there
    # is, and nothing from past the cut.
    short, _, error = _ltx2_branch(sidecar, 9, video_length=100, audio_entry=audio)
    assert error == "" and short is not None and short["overlap_decoded"] == 9
    assert int(short["prefix_audio_latents"].shape[2]) == short["audio_bridge_latents"] == 9
    torch.testing.assert_close(short["prefix_audio_latents"][0].reshape(int(block.shape[0]), -1).float(),
                               block[:, :9 * 16].float(), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Step 5 (the wgp wiring contract, engine level): the job's call-site
# values (the setting's mode, the user's requested overlap, the job's
# explicit audio switch) resolve through the fit check into the head
# window's freeze shape and the console-print data the wgp print
# consumes (wgp.py cannot be imported headless, so the contract is
# asserted on the values it reads).
# ---------------------------------------------------------------------------


def _wgp_covers_kept(prefix, source_video_overlap_frames_count):
    """The wgp seam's test (wgp.py's ``_branch_head_covers_kept``, the
    LTX2 branch-window part): the head window's own decode covers the
    kept region only when that region is exactly the window's frozen
    head (the kept count equals the assembly overlap)."""
    return (prefix is not None
            and int(prefix.get("kept_frames") or 0) > 0
            and int(prefix.get("kept_frames") or 0) == int(source_video_overlap_frames_count))


def test_step5_pixel_rule_wiring_contract():
    # The f241 241 -> 481 job at the 17-frame overlap: the pixel rule
    # fits, the model gets the tail 3, the assembly overlap is O = 17,
    # and the kept region (241) is not the window's frozen head (17) -
    # the seam's test is False, so the job splices a fresh decode of the
    # kept region and the 8309 print takes the pixel-rule wording (the
    # tail-N / O-frame numbers).
    sidecar = _ltx2_entry([_ltx2_latents(31)])
    prefix, branch, error = _ltx2_branch(sidecar, 241, video_length=240,
                                         head_mode="pixel_rule", reuse_frames=17)
    assert error == "" and prefix is not None
    assert prefix["head_mode"] == "pixel_rule"
    assert int(prefix["prefix_latents"].shape[1]) == 3
    assert prefix["assembly_overlap"] == 17 == prefix["overlap_decoded"]
    assert prefix["kept_frames"] == 241
    assert _wgp_covers_kept(prefix, prefix["assembly_overlap"]) is False


def test_step5_audio_job_keeps_the_wiring():
    # The same job with a source soundtrack: the audio bridge changes the
    # prefix's audio and nothing else — the geometry the wgp seam reads is the
    # one the audio-free job gets, because there is no fallback left to announce.
    sidecar = _ltx2_entry([_ltx2_latents(31)])
    silent, _, _ = _ltx2_branch(sidecar, 241, video_length=240, reuse_frames=17)
    loud, _, error = _ltx2_branch(sidecar, 241, video_length=240, reuse_frames=17,
                                  audio_entry=_ltx2_audio_entry())
    assert error == "" and loud is not None
    assert silent["prefix_audio_latents"] is None
    assert loud["prefix_audio_latents"] is not None
    shared = ("head_mode", "assembly_overlap", "guide_frames", "head_prefix_latents",
              "overlap_latents", "overlap_decoded", "kept_frames", "shared_cut_frame",
              "head_frame_num", "head_output", "head_trim_new", "cut_at_window_end")
    assert {key: loud[key] for key in shared} == {key: silent[key] for key in shared}
    assert _wgp_covers_kept(loud, loud["assembly_overlap"]) is False


def test_pixel_rule_window_end_keeps_request_geometry():
    # The window-end continuation keeps the plan's head geometry: the request
    # rides on top of the frozen overlap instead of being charged against the
    # window's own budget. A 241-frame request behind the 17-frame bridge is a
    # 257-frame head window (the overlap plus the new segment rounded up to the
    # latent grid): it commits 240 frames — the request without the frame the
    # cut stands on — and the rounding lands exactly on the latent grid, so
    # nothing is left to trim. The fields used to be dropped here, and the job
    # then shipped the request minus the overlap (224 frames).
    sidecar = _ltx2_entry([_ltx2_latents(31)])
    prefix, branch, error = _ltx2_branch(sidecar, 241, video_length=241,
                                         head_mode="pixel_rule", reuse_frames=17)
    assert error == "" and prefix is not None
    plan = _ltx2_plan(sidecar, branch, 241)
    assert prefix["cut_at_window_end"] is True
    assert prefix["head_frame_num"] == plan["frame_num"] == 257
    assert prefix["head_output"] == plan["new_frames"] == 240
    assert prefix["head_trim_new"] == plan["tail_trim_new"] == 0
    assert prefix["assembly_overlap"] == 17
    assert prefix["shared_cut_frame"] == plan["shared_cut_frame"] == 1
    assert int(prefix["head_frame_num"]) - int(prefix["assembly_overlap"]) \
        - int(prefix["head_trim_new"]) == 240


@pytest.mark.parametrize("wanted, frame_num, trim", [
    (241, 257, 0),   # the request minus one lands on the latent grid
    (100, 121, 5),   # a short request: the rounding overshoot is still trimmed
    (240, 257, 1),
    (2, 25, 7),      # the smallest request that spends the frame
])
def test_pixel_rule_window_end_matches_video_continuation_length(wanted, frame_num, trim):
    # The length parity with a continuation from the video file: the classic
    # path spends one of the requested frames re-generating the kept run's last
    # frame, so a video continuation delivers request - 1 new frames. The
    # branch has no such duplicate (the assembly drops its frozen head instead
    # of re-generating it), so the window-end window gives up its own last
    # frame and lands on the same kept + request - 1. Without it the branch's
    # video runs one frame longer than the video job's, and every post-cut
    # metric taken across the two is shifted by one frame.
    sidecar = _ltx2_entry([_ltx2_latents(31)])
    prefix, branch, error = _ltx2_branch(sidecar, 241, video_length=wanted,
                                         head_mode="pixel_rule", reuse_frames=17)
    assert error == "" and prefix is not None
    assert prefix["shared_cut_frame"] == 1
    assert prefix["head_frame_num"] == frame_num
    assert prefix["head_trim_new"] == trim
    delivered = (int(prefix["head_frame_num"]) - int(prefix["assembly_overlap"])
                 - int(prefix["head_trim_new"]))
    assert delivered == wanted - 1
    # The written video: the kept run plus what the head window commits.
    assert int(prefix["kept_frames"]) + delivered == 241 + wanted - 1


def test_pixel_rule_one_frame_request_spends_nothing():
    # A one-frame request keeps its frame (the window never delivers zero), so
    # no shared frame is recorded and the delivered run is the request itself.
    sidecar = _ltx2_entry([_ltx2_latents(31)])
    prefix, branch, error = _ltx2_branch(sidecar, 241, video_length=1,
                                         head_mode="pixel_rule", reuse_frames=17)
    assert error == "" and prefix is not None
    assert prefix["shared_cut_frame"] == 0
    assert (prefix["head_frame_num"], prefix["head_trim_new"]) == (25, 7)
    assert (int(prefix["head_frame_num"]) - int(prefix["assembly_overlap"])
            - int(prefix["head_trim_new"])) == 1


def test_pixel_rule_mid_cut_spends_no_shared_frame():
    # A cut inside the window cannot spend the frame: its new latents complete
    # the containing window up to that window's committed end, and shortening
    # them would break the committed tiling the sidecar records for it.
    sidecar = _ltx2_entry([_ltx2_latents(31)])
    prefix, branch, error = _ltx2_branch(sidecar, 41, video_length=100)
    assert error == "" and prefix is not None
    plan = _ltx2_plan(sidecar, branch, 100)
    assert plan["shared_cut_frame"] == 0
    assert (plan["frame_num"], plan["new_frames"], plan["tail_trim_new"]) == (121, 100, 4)


def test_full_freeze_window_end_leaves_geometry_to_caller():
    # The full freeze still leaves the head window's geometry to the caller:
    # its frozen head is the whole kept run, so the window's new share is what
    # its budget still leaves and the tail windows make up the rest of the
    # request (the fit check already required the window to hold the kept run).
    sidecar = _ltx2_entry([_ltx2_latents(31)])
    prefix, branch, error = _ltx2_branch(sidecar, 241, video_length=241,
                                         head_mode="full_freeze", reuse_frames=17)
    assert error == "" and prefix is not None
    assert prefix["head_mode"] == "full_freeze"
    assert prefix["head_frame_num"] is None and prefix["head_output"] is None
    assert prefix["head_trim_new"] == 0
    assert prefix["assembly_overlap"] == 241 == prefix["guide_frames"]


def test_step5_pixel_rule_cut_block_shape():
    # The pixel-rule job's sidecar commits the kept region from the
    # cut block (the stand-alone block the writer inserts ahead of the
    # head window), so the job builds it the way the H3 job does: the
    # source sidecar's K capped at the cut. Window-end cut:
    # the block is the full source window; the linear layout's committed
    # range ends at the cut (no tail trim for a cut at the window end).
    sidecar = _ltx2_entry([_ltx2_latents(31)])
    entry = latent_io.latent_payload(sidecar)
    branch = latent_io.branch_point_for_frame(entry, 241)
    block = latent_io.branch_cut_window_block(sidecar, branch)
    assert block is not None
    assert int(block["latents"].shape[1]) == 31
    assert int(block.get("tail_trim") or 0) == 0
    # Mid-cut: the block caps the committed range at the cut's decoded
    # position (the full latents stay; the tail trim extends to the
    # cut, 41 of the 241-frame run).
    branch_mid = latent_io.branch_point_for_frame(entry, 41)
    block_mid = latent_io.branch_cut_window_block(sidecar, branch_mid)
    assert block_mid is not None
    assert int(block_mid["latents"].shape[1]) == 31
    assert int(block_mid.get("tail_trim") or 0) == 241 - 41


# The sidecar writer specs (section 9.2) run against the production writer:
# the pixel-rule mode field, the cut-block insertion for the linear
# families and the mid-cut head trim landed with step 6.


def test_pixel_rule_sidecar_f241(tmp_path):
    # 9.2-12: the 241 -> 481 job writes the source window (cut block)
    # ahead of the head window; the head block commits 240 of its 257
    # (head trim 17 = O); the mode is recorded; the committed total
    # matches the written video (no length flag).
    sidecar = _ltx2_entry([_ltx2_latents(31)])
    prefix, branch, error = _ltx2_branch(sidecar, 241, video_length=240)
    source = latent_io.payload_latents(latent_io.latent_payload(sidecar))
    head_latents = torch.cat([source[:, -3:], _ltx2_latents(30, seed=99)], dim=1)
    rec = latent_jobs.LatentWindowRecorder(
        stub_ctx(tmp_path), model_def={}, model_type=LTX2, base_model_type=LTX2,
        branch_prefix=prefix,
        branch_cut_video_block=latent_io.branch_cut_window_block(sidecar, branch))
    media = str(tmp_path / "f241.mp4")
    rec.on_window(_ltx2_entry_block(head_latents, frames=257), {}, media, 1,
                  **window_args(sliding_window=True, source_video_overlap_frames_count=17,
                                written_frame_count=481))
    entry = latent_io.load_latent_file(latent_io.latent_sidecar_path(media))[latent_io.LATENT_FORMAT_ID]
    specs = latent_io.entry_window_specs(entry)
    assert len(specs) == 2
    assert latent_io.sidecar_total_frames(entry) == 481
    assert entry.get("length_verified") is not False
    assert latent_io.sidecar_length_mismatch(entry) is None
    cut_block, head_block = entry["windows"]
    assert cut_block.get("head_trim") == 0 and cut_block.get("tail_trim") == 0
    assert head_block.get("head_trim") == 17
    assert head_block.get("branch_head_mode") == "pixel_rule"
    assert head_block.get("overlap_latents") == 3
    assert entry.get("branch_head_mode") == "pixel_rule"  # the entry-level mirror
    assert torch.equal(specs[1]["latents"].float(), head_latents.float())


def test_pixel_rule_sidecar_k2_window_end_721(tmp_path):
    # The 721-frame job writes three windows: the fully kept window 1, the
    # containing window 2 as its own cut block (it keeps its 17-frame head
    # trim, so it commits the 224 frames the source committed), and the
    # pixel-rule head window (257 frames, head trim 17 = O, commits 240). The
    # committed total matches the written video (no length flag) and the file
    # still resolves a branch point at its end.
    sidecar = _ltx2_entry([_ltx2_latents(33), _ltx2_latents(31, seed=1)],
                          head_trims=(0, 17))
    prefix, branch, error = _ltx2_branch(sidecar, 481, video_length=241)
    assert error == "" and prefix is not None
    src = latent_io.payload_latents(latent_io.latent_payload(sidecar))
    carried = latent_io.carry_over_latent_windows(sidecar, branch)
    latent_windows = [dict(b) for b in carried["video_windows"]]
    assert [int(b["latents"].shape[1]) for b in latent_windows] == [33]
    head_latents = torch.cat([src[:, -3:], _ltx2_latents(30, seed=99)], dim=1)  # 33 = 257
    rec = latent_jobs.LatentWindowRecorder(
        stub_ctx(tmp_path), model_def={}, model_type=LTX2, base_model_type=LTX2,
        branch_prefix=prefix,
        branch_cut_video_block=latent_io.branch_cut_window_block(sidecar, branch))
    media = str(tmp_path / "k2_721.mp4")
    rec.on_window(_ltx2_entry_block(head_latents, frames=257), {}, media, 1,
                  **window_args(sliding_window=True, source_video_overlap_frames_count=17,
                                written_frame_count=721, latent_windows=latent_windows))
    entry = latent_io.load_latent_file(latent_io.latent_sidecar_path(media))[latent_io.LATENT_FORMAT_ID]
    specs = latent_io.entry_window_specs(entry)
    assert len(specs) == 3
    assert latent_io.sidecar_total_frames(entry) == 721
    assert entry.get("length_verified") is not False
    assert latent_io.sidecar_length_mismatch(entry) is None
    w1, w2, head_block = entry["windows"]
    assert w1.get("head_trim") == 0 and w1.get("tail_trim") == 0     # the kept window 1
    assert w2.get("head_trim") == 17 and w2.get("tail_trim") == 0    # the cut block keeps its own head
    assert "branch_head_mode" not in w2                              # the cut block is a plain window
    assert head_block.get("head_trim") == 17                         # O
    assert head_block.get("branch_head_mode") == "pixel_rule"
    assert entry.get("branch_head_mode") == "pixel_rule"            # the entry-level mirror
    assert torch.equal(specs[2]["latents"].float(), head_latents.float())
    again = latent_io.branch_point_for_frame(entry, 721)
    assert again is not None and again["kept_frames"] == 721


def test_pixel_rule_sidecar_f241_guide_encode(tmp_path):
    # 9.5-24, the live job: the head window is a *fresh encode of the decoded
    # kept region* - the pipeline hands the window guide pixels, never the
    # source latents - so nothing in the run it returns marks it as a
    # pixel-rule window, and the plan is all the writer has to go on. The cut
    # block still goes in ahead of the head window, the head block keeps its
    # O-frame head trim, the recorded length matches the video written, and the
    # file still takes a branch cut. Reading the mode off the latents instead
    # left the kept region out of the file: 240 frames recorded for a 481-frame
    # video, flagged (length_verified: false) and refused by the next branch.
    sidecar = _ltx2_entry([_ltx2_latents(31)])
    prefix, branch, error = _ltx2_branch(sidecar, 241, video_length=241)
    assert error == "" and prefix is not None
    source = latent_io.payload_latents(latent_io.latent_payload(sidecar))
    head_latents = _ltx2_latents(33, seed=424403406)      # 33 latents = 257 frames
    # The head window shares nothing with the kept run's tail: the very
    # comparison the writer used to make reads this window as a full freeze.
    assert not torch.allclose(head_latents[:, :3].float(), source[:, -3:].float(),
                              rtol=2e-2, atol=1e-2)
    rec = latent_jobs.LatentWindowRecorder(
        stub_ctx(tmp_path), model_def={}, model_type=LTX2, base_model_type=LTX2,
        branch_prefix=prefix,
        branch_cut_video_block=latent_io.branch_cut_window_block(sidecar, branch))
    media = str(tmp_path / "f241_guide.mp4")
    rec.on_window(_ltx2_entry_block(head_latents, frames=257), {}, media, 1,
                  **window_args(sliding_window=True, source_video_overlap_frames_count=17,
                                written_frame_count=481))
    entry = latent_io.load_latent_file(latent_io.latent_sidecar_path(media))[latent_io.LATENT_FORMAT_ID]
    specs = latent_io.entry_window_specs(entry)
    assert len(specs) == 2                         # the kept window, then the head window
    assert latent_io.sidecar_total_frames(entry) == 481
    assert entry.get("length_verified") is not False
    assert latent_io.sidecar_length_mismatch(entry) is None
    cut_block, head_block = entry["windows"]
    assert cut_block.get("head_trim") == 0 and cut_block.get("tail_trim") == 0
    assert head_block.get("head_trim") == 17
    assert head_block.get("branch_head_mode") == "pixel_rule"
    assert entry.get("branch_head_mode") == "pixel_rule"   # the entry-level mirror
    assert torch.equal(specs[0]["latents"].float(), source.float())  # the kept run, verbatim
    assert torch.equal(specs[1]["latents"].float(), head_latents.float())
    again = latent_io.branch_point_for_frame(entry, 481)   # and it still branches
    assert again is not None and again["kept_frames"] == 481


def test_pixel_rule_sidecar_mid_cut_guide_encode(tmp_path):
    # The same fresh encode at a cut inside the window: the source window is
    # still capped at the cut and the head block still trims O, so the file
    # commits the kept 41 + the 100 new frames the job wrote.
    sidecar = _ltx2_entry([_ltx2_latents(31)])
    prefix, branch, error = _ltx2_branch(sidecar, 41, video_length=100)
    assert error == "" and prefix is not None
    head_latents = _ltx2_latents(16, seed=20251108)       # 16 latents = 121 frames
    rec = latent_jobs.LatentWindowRecorder(
        stub_ctx(tmp_path), model_def={}, model_type=LTX2, base_model_type=LTX2,
        branch_prefix=prefix,
        branch_cut_video_block=latent_io.branch_cut_window_block(sidecar, branch))
    media = str(tmp_path / "mid_guide.mp4")
    rec.on_window(_ltx2_entry_block(head_latents), {}, media, 1,
                  **window_args(sliding_window=True, source_video_overlap_frames_count=17,
                                automatic_trim_last_frames=4, written_frame_count=141))
    entry = latent_io.load_latent_file(latent_io.latent_sidecar_path(media))[latent_io.LATENT_FORMAT_ID]
    specs = latent_io.entry_window_specs(entry)
    assert len(specs) == 2
    assert latent_io.sidecar_total_frames(entry) == 141
    assert entry.get("length_verified") is not False
    cut_block, head_block = entry["windows"]
    assert cut_block.get("tail_trim") == 200       # 241 - 41: the cut ends the kept region
    assert head_block.get("head_trim") == 17       # O
    assert head_block.get("branch_head_mode") == "pixel_rule"


def test_pixel_rule_sidecar_clamped_guide_encode(tmp_path):
    # The clamped shape (kept == overlap) with a fresh encode: the head
    # window's own head is the kept region, so no cut block is inserted and the
    # containing window's head trim stands - and the mode is still recorded on
    # both the block and the entry.
    sidecar = _ltx2_entry([_ltx2_latents(31)])
    prefix, branch, error = _ltx2_branch(sidecar, 17, video_length=100)
    assert error == "" and prefix is not None
    head_latents = _ltx2_latents(16, seed=31337)          # 16 latents = 121 frames
    rec = latent_jobs.LatentWindowRecorder(
        stub_ctx(tmp_path), model_def={}, model_type=LTX2, base_model_type=LTX2,
        branch_prefix=prefix)
    media = str(tmp_path / "clamped_guide.mp4")
    rec.on_window(_ltx2_entry_block(head_latents), {}, media, 1,
                  **window_args(sliding_window=True, source_video_overlap_frames_count=17,
                                automatic_trim_last_frames=4, written_frame_count=117))
    entry = latent_io.load_latent_file(latent_io.latent_sidecar_path(media))[latent_io.LATENT_FORMAT_ID]
    specs = latent_io.entry_window_specs(entry)
    assert len(specs) == 1
    assert latent_io.sidecar_total_frames(entry) == 117
    assert entry.get("length_verified") is not False
    assert entry["windows"][0].get("head_trim") == 0
    assert entry["windows"][0].get("branch_head_mode") == "pixel_rule"
    assert entry.get("branch_head_mode") == "pixel_rule"


def test_pixel_rule_sidecar_mid_cut(tmp_path):
    # 9.2-13: the mid-cut source window trims to the cut (its tail trim
    # extends to the cut's decoded position) and the head block records
    # O as its head trim (the re-denoised overlap, not the containing
    # window's head trim).
    sidecar = _ltx2_entry([_ltx2_latents(31)])
    prefix, branch, error = _ltx2_branch(sidecar, 41, video_length=100)
    source = latent_io.payload_latents(latent_io.latent_payload(sidecar))
    # The frozen head is the tail-N of the kept run (the latents adjacent
    # to the cut: window 1 keeps latents 0..5 at this cut, N = 3).
    head_latents = torch.cat([source[:, 3:6], _ltx2_latents(13, seed=99)], dim=1)
    rec = latent_jobs.LatentWindowRecorder(
        stub_ctx(tmp_path), model_def={}, model_type=LTX2, base_model_type=LTX2,
        branch_prefix=prefix,
        branch_cut_video_block=latent_io.branch_cut_window_block(sidecar, branch))
    media = str(tmp_path / "mid.mp4")
    rec.on_window(_ltx2_entry_block(head_latents), {}, media, 1,
                  **window_args(sliding_window=True, source_video_overlap_frames_count=17,
                                automatic_trim_last_frames=4, written_frame_count=141))
    entry = latent_io.load_latent_file(latent_io.latent_sidecar_path(media))[latent_io.LATENT_FORMAT_ID]
    specs = latent_io.entry_window_specs(entry)
    assert len(specs) == 2
    assert latent_io.sidecar_total_frames(entry) == 141
    assert entry.get("length_verified") is not False
    cut_block, head_block = entry["windows"]
    assert cut_block.get("tail_trim") == 200   # 241 - 41: the cut ends the kept region
    assert head_block.get("head_trim") == 17   # O (the legacy writer records 0 here)
    assert head_block.get("branch_head_mode") == "pixel_rule"


def test_pixel_rule_sidecar_clamped(tmp_path):
    # 9.2-14: the kept run (17) equals the re-denoised overlap (17): no
    # cut block (the head window's own head is the kept region), the head
    # trim is the containing window's own (0 for a t2v window 1), and the
    # file still verifies its length.
    sidecar = _ltx2_entry([_ltx2_latents(31)])
    prefix, branch, error = _ltx2_branch(sidecar, 17, video_length=100)
    source = latent_io.payload_latents(latent_io.latent_payload(sidecar))
    # The clamped cut keeps N = 3 latents and the frozen head is the whole
    # kept run (window 1's latents 0..2 — the cut at frame 17).
    head_latents = torch.cat([source[:, :3], _ltx2_latents(13, seed=99)], dim=1)
    rec = latent_jobs.LatentWindowRecorder(
        stub_ctx(tmp_path), model_def={}, model_type=LTX2, base_model_type=LTX2,
        branch_prefix=prefix)
    media = str(tmp_path / "clamped.mp4")
    rec.on_window(_ltx2_entry_block(head_latents), {}, media, 1,
                  **window_args(sliding_window=True, source_video_overlap_frames_count=17,
                                automatic_trim_last_frames=4, written_frame_count=117))
    entry = latent_io.load_latent_file(latent_io.latent_sidecar_path(media))[latent_io.LATENT_FORMAT_ID]
    specs = latent_io.entry_window_specs(entry)
    assert len(specs) == 1
    assert latent_io.sidecar_total_frames(entry) == 117
    assert entry.get("length_verified") is not False
    head_block = entry["windows"][0]
    assert head_block.get("head_trim") == 0
    assert head_block.get("branch_head_mode") == "pixel_rule"
    assert entry.get("branch_head_mode") == "pixel_rule"


def test_legacy_sidecar_no_mode_field(tmp_path):
    # 9.2-15: a full-freeze branch sidecar stays a legacy file — no
    # branch_head_mode anywhere, the baseline end-cut head-trim rule, and
    # the file still branches.
    sidecar = _ltx2_entry([_ltx2_latents(31)])
    prefix, branch, error = _ltx2_branch(sidecar, 241, video_length=240,
                                         head_mode="full_freeze")
    source = latent_io.payload_latents(latent_io.latent_payload(sidecar))
    head_latents = torch.cat([source, _ltx2_latents(2, seed=99)], dim=1)  # 33 latents = 257 frames
    tail_latents = _ltx2_latents(31, seed=42)
    rec = latent_jobs.LatentWindowRecorder(
        stub_ctx(tmp_path), model_def={}, model_type=LTX2, base_model_type=LTX2,
        branch_prefix=prefix)
    media = str(tmp_path / "legacy.mp4")
    lw = []
    rec.on_window(_ltx2_entry_block(head_latents, frames=257), {}, media, 1,
                  **window_args(sliding_window=True, source_video_overlap_frames_count=241,
                                written_frame_count=481, latent_windows=lw))
    rec.on_window(_ltx2_entry_block(tail_latents, frames=241), {}, media, 2,
                  **window_args(sliding_window=True, reuse_frames=17,
                                source_video_overlap_frames_count=17,
                                written_frame_count=481, latent_windows=lw))
    entry = latent_io.load_latent_file(latent_io.latent_sidecar_path(media))[latent_io.LATENT_FORMAT_ID]
    specs = latent_io.entry_window_specs(entry)
    assert len(specs) == 2
    assert latent_io.sidecar_total_frames(entry) == 481
    assert entry.get("length_verified") is not False
    for block in entry["windows"]:
        assert "branch_head_mode" not in block
    assert "branch_head_mode" not in entry  # the entry-level mirror stays absent
    assert entry["windows"][0]["head_trim"] == 0   # the baseline rule: kept == overlap
    assert entry["windows"][1]["head_trim"] == 17
    # The file still resolves a branch point at its end
    again = latent_io.branch_point_for_frame(entry, 481)
    assert again is not None and again["kept_frames"] == 481


def test_length_guard_flags_miscount(tmp_path):
    # 9.5-23: a deliberately miscounting write still yields the flagged
    # file (length_verified: false + written_frame_count) that refuses
    # branch cuts.
    rec = _ltx2_recorder(stub_ctx(tmp_path), None)
    media = str(tmp_path / "mis.mp4")
    rec.on_window(_ltx2_block(), {}, media, 1, **window_args(written_frame_count=100))
    entry = latent_io.load_latent_file(latent_io.latent_sidecar_path(media))[latent_io.LATENT_FORMAT_ID]
    assert entry.get("length_verified") is False
    assert entry.get("written_frame_count") == 100
    # 8 latents = 57 committed frames against the claimed 100
    assert latent_io.sidecar_length_mismatch(entry) == (57, 100)


def test_pixel_rule_audio_sidecar(tmp_path):
    # 9.2-16: a pixel-rule branch whose source carries a soundtrack records a
    # self-contained audio entry. The kept region's soundtrack is the carried
    # cut block (stacked into the accumulator the way the job stacks it, as for
    # the other accepted branch shapes); the head window's entry starts after
    # the frozen bridge, so the bridge is not written twice, and its recorded
    # position follows the kept region on the new track.
    # A source sidecar that carries its soundtrack, as the job reads it.
    sidecar = {**_ltx2_entry([_ltx2_latents(31)]), **_ltx2_audio_payload()}
    prefix, branch, error = _ltx2_branch(sidecar, 241, video_length=240,
                                         audio_entry=latent_io.audio_latent_payload(sidecar))
    assert error == "" and prefix is not None
    assert int(prefix["audio_bridge_latents"]) == 18
    # The kept region's audio, as the job stacks it ahead of the head window.
    kept_audio = latent_io.branch_cut_audio_window_block(sidecar, branch)
    assert kept_audio is not None
    carried = latent_io.build_audio_latent_payload(
        [kept_audio["latents"]],
        [{"start_sample": kept_audio["start_sample"], "samples": kept_audio["samples"]}],
        app_version=APP_VERSION, model_type=LTX2, base_model_type=LTX2,
        sample_rate=16000, channels=8, samples=251 * 640, mel_bins=16
    )[latent_io.LATENT_FORMAT_ID_AUDIO]
    # The head window's audio as the pipeline returns it: the bridge the model
    # froze (18 of its steps), then what it generated. Its 257 decoded frames
    # are 268 audio steps.
    head_block = torch.randn(8, 268 * 16, dtype=torch.float16)
    head_audio = latent_io.build_audio_latent_payload(
        [head_block],
        [{"window_no": 1, "start_sample": 0, "samples": 268 * 640}],
        app_version=APP_VERSION, model_type=LTX2, base_model_type=LTX2,
        sample_rate=16000, channels=8, samples=268 * 640, mel_bins=16)
    head_latents = _ltx2_latents(33, seed=7)                     # 33 latents = 257 frames
    rec = latent_jobs.LatentWindowRecorder(
        stub_ctx(tmp_path), model_def={}, model_type=LTX2, base_model_type=LTX2,
        branch_prefix=prefix,
        branch_cut_video_block=latent_io.branch_cut_window_block(sidecar, branch),
        branch_cut_audio_block=kept_audio)
    media = str(tmp_path / "audio.mp4")
    obj = {**_ltx2_entry_block(head_latents, frames=257), **head_audio}
    rec.on_window(obj, {}, media, 1,
                  **window_args(sliding_window=True, source_video_overlap_frames_count=17,
                                latent_audio_windows=[carried], written_frame_count=481))
    saved = latent_io.load_latent_file(latent_io.latent_sidecar_path(media))
    video = saved[latent_io.LATENT_FORMAT_ID]
    assert latent_io.sidecar_total_frames(video) == 481
    assert video.get("length_verified") is not False
    audio = latent_io.audio_latent_payload(saved)
    assert audio is not None
    kept_win, head_win = audio["windows"]
    assert kept_win["latent_len"] == 251 * 16 and kept_win["samples"] == 251 * 640
    # The destination re-stacks the windows, so the head's offset is the kept
    # window's length; what it *keeps* is the window minus the frozen bridge.
    assert head_win["latent_start"] == kept_win["latent_len"]
    assert head_win["latent_len"] == (268 - 18) * 16
    assert head_win["start_sample"] == kept_win["start_sample"] + kept_win["samples"]
    # What the head window commits is its new share: 160000 samples of the 240
    # frames its block commits (17 dropped of 257), the same span in seconds.
    assert head_win["samples"] == 268 * 640 - 11520 == 160000
    assert round(head_win["samples"] / 16000 * 24) == 240
    torch.testing.assert_close(latent_io.audio_window_specs(audio)[0]["latents"].float(),
                               kept_audio["latents"].float(), rtol=0, atol=0)
    # The head window contributes only what the model generated: the bridge's
    # audio is the kept region's and is recorded once, in the cut block.
    torch.testing.assert_close(latent_io.audio_window_specs(audio)[1]["latents"].float(),
                               head_block[:, 18 * 16:].float(), rtol=0, atol=0)


def _branch_join_sidecar(tmp_path, sidecar, prefix, branch, kept_audio, join_seconds):
    """One branch window written with the join of ``join_seconds``, as the job hands
    it to the recorder, returning the audio entry of the sidecar it wrote."""
    carried = latent_io.build_audio_latent_payload(
        [kept_audio["latents"]],
        [{"start_sample": kept_audio["start_sample"], "samples": kept_audio["samples"]}],
        app_version=APP_VERSION, model_type=LTX2, base_model_type=LTX2,
        sample_rate=16000, channels=8, samples=251 * 640, mel_bins=16
    )[latent_io.LATENT_FORMAT_ID_AUDIO]
    head_audio = latent_io.build_audio_latent_payload(
        [torch.randn(8, 268 * 16, dtype=torch.float16)],
        [{"window_no": 1, "start_sample": 0, "samples": 268 * 640}],
        app_version=APP_VERSION, model_type=LTX2, base_model_type=LTX2,
        sample_rate=16000, channels=8, samples=268 * 640, mel_bins=16)
    rec = latent_jobs.LatentWindowRecorder(
        stub_ctx(tmp_path), model_def={}, model_type=LTX2, base_model_type=LTX2,
        branch_prefix=prefix,
        branch_cut_video_block=latent_io.branch_cut_window_block(sidecar, branch),
        branch_cut_audio_block=kept_audio)
    media = str(tmp_path / f"join-{round(join_seconds * 10000)}.mp4")
    rec.on_window({**_ltx2_entry_block(_ltx2_latents(33, seed=7), frames=257), **head_audio}, {}, media, 1,
                  **window_args(sliding_window=True, source_video_overlap_frames_count=17,
                                latent_audio_windows=[carried], written_frame_count=481,
                                branch_audio_join_seconds=join_seconds))
    return latent_io.audio_latent_payload(latent_io.load_latent_file(latent_io.latent_sidecar_path(media)))


def test_branch_audio_join_commits_the_kept_run(tmp_path):
    # O8: the mux force-fits the kept half of a branch's soundtrack to the duration
    # it is handed (apad up to it, atrim at it), so the new scene starts at a
    # video-derived join, not at the length the carried audio latents decode to. The
    # sidecar has to record that join: here the 241-frame kept region decodes 27
    # samples shorter than 241 frames at 16 kHz, and a decode built on the decoded
    # length played the new scene that early (43 ms on the real file).
    sidecar = {**_ltx2_entry([_ltx2_latents(31)]), **_ltx2_audio_payload()}
    prefix, branch, error = _ltx2_branch(sidecar, 241, video_length=240,
                                         audio_entry=latent_io.audio_latent_payload(sidecar))
    assert error == "" and prefix is not None
    kept_audio = latent_io.branch_cut_audio_window_block(sidecar, branch)
    assert kept_audio is not None and kept_audio["samples"] == 251 * 640
    join_samples = round(241 / 24 * 16000)                       # the join the mux was handed
    assert join_samples != kept_audio["samples"]

    audio = _branch_join_sidecar(tmp_path, sidecar, prefix, branch, kept_audio, 241 / 24)
    kept_win, head_win = audio["windows"]
    assert kept_win["samples"] == join_samples                   # the join, not the decoded length
    assert head_win["start_sample"] == join_samples              # the new scene starts on it
    assert audio["samples"] == kept_win["samples"] + head_win["samples"]
    # Only the committed length moved: the kept region still carries its own latents,
    # and merge_audio_window_chunks pads the difference with silence, which is what
    # the written track holds there.
    assert latent_io.audio_window_specs(audio)[0]["latents"].shape[1] == 251 * 16
    torch.testing.assert_close(latent_io.audio_window_specs(audio)[0]["latents"].float(),
                               kept_audio["latents"].float(), rtol=0, atol=0)


def test_branch_audio_join_cuts_a_kept_run_that_runs_past(tmp_path):
    # The other side of the same number: a source soundtrack that runs past its
    # continued frames is trimmed at the join by the mux (atrim), so the kept run's
    # committed length shrinks to the join and the new scene starts there.
    sidecar = {**_ltx2_entry([_ltx2_latents(31)]), **_ltx2_audio_payload()}
    prefix, branch, error = _ltx2_branch(sidecar, 241, video_length=240,
                                         audio_entry=latent_io.audio_latent_payload(sidecar))
    assert error == "" and prefix is not None
    kept_audio = latent_io.branch_cut_audio_window_block(sidecar, branch)
    assert kept_audio is not None
    cut_join = round(239 / 24 * 16000)
    assert cut_join < kept_audio["samples"]
    audio = _branch_join_sidecar(tmp_path, sidecar, prefix, branch, kept_audio, 239 / 24)
    assert audio["windows"][0]["samples"] == cut_join
    assert audio["windows"][1]["start_sample"] == cut_join


def test_branch_audio_join_helpers():
    # The join lands on the kept run's last block and is applied once: a second pass
    # finds it already there, and a job that carries no soundtrack keeps the
    # positions the pipeline recorded.
    carried = [{"sample_rate": 16000, "windows": [
        {"start_sample": 0, "samples": 1000}, {"start_sample": 1000, "samples": 500}]}]
    assert latent_jobs.carried_audio_end_sample(carried) == 1500
    assert latent_jobs._commit_carried_audio_to_join(carried, 1700) == 200
    assert [w["samples"] for w in carried[0]["windows"]] == [1000, 700]
    assert latent_jobs._commit_carried_audio_to_join(carried, 1700) == 0        # idempotent
    assert latent_jobs._commit_carried_audio_to_join(carried, 1200) == -500     # cut at the join
    assert latent_jobs.carried_audio_end_sample(carried) == 1200
    assert latent_jobs._commit_carried_audio_to_join(carried, 0) == 0           # no kept audio
    assert latent_jobs._commit_carried_audio_to_join([], 1700) == 0             # a fresh job

    entry = {"windows": [{"start_sample": 0, "samples": 100}, {"start_sample": 100, "samples": 80}]}
    moved = latent_jobs._rebase_head_audio(entry, 1200)
    assert [w["start_sample"] for w in moved["windows"]] == [1200, 1300]        # gaps preserved
    assert [w["start_sample"] for w in entry["windows"]] == [0, 100]            # the job's entry stands
    assert latent_jobs._rebase_head_audio(entry, 0) is entry                    # nothing carried


def _stub_ltx2_decode_modules(monkeypatch, decode_video):
    """Put dummy ``models.ltx2`` modules in ``sys.modules`` for the decode helpers.

    ``_ltx2_latent_decode_windows`` imports the LTX-2 modules inside the call; the
    real ones need a GPU at import time, so a test hands it dummies instead.
    """
    import sys
    import types

    for name in ("models", "models.ltx2", "models.ltx2.ltx2", "models.ltx2.ltx_core",
                 "models.ltx2.ltx_core.model", "models.ltx2.ltx_core.model.audio_vae",
                 "models.ltx2.ltx_core.model.video_vae"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setattr(sys.modules["models.ltx2.ltx2"], "LTX2_HDR_TRANSFORM", "hlg", raising=False)
    monkeypatch.setattr(sys.modules["models.ltx2.ltx_core.model.audio_vae"], "decode_audio",
                        lambda *a, **k: None, raising=False)
    monkeypatch.setattr(sys.modules["models.ltx2.ltx_core.model.video_vae"], "decode_video_to_tensor",
                        decode_video, raising=False)
    # The decode helper picks "cuda" whenever a card answers, and a card holding a
    # running generation fails these checks on memory instead of on the dtype under
    # test, so the helper is pointed at the CPU here.
    monkeypatch.setattr(latent_jobs.torch.cuda, "is_available", lambda: False)


def test_ltx2_decode_hands_the_decoder_its_own_dtype(monkeypatch):
    """The video latents follow the loaded decoder, not the sidecar's dtype.

    The sidecar records float16; the LTX-2 video VAE only computes in bfloat16,
    and decoding it in float16 yields flat grey (issues.md O6).
    """
    import types as _types

    seen = {}
    decoder = _types.SimpleNamespace()
    decoder.parameters = lambda: iter([torch.nn.Parameter(
        torch.zeros(1, dtype=torch.bfloat16), requires_grad=False)])
    decoder.to = lambda *a, **k: decoder                      # the caller parks the modules on CPU

    def _fake_decode_video(latent, video_decoder, *args, **kwargs):
        seen["latent_dtype"] = latent.dtype
        return torch.zeros(2, 4, 4, 3, dtype=torch.uint8)             # [T, H, W, C]

    _stub_ltx2_decode_modules(monkeypatch, _fake_decode_video)
    entry = {"fps": 24.0}
    window_specs = [{"window_no": 1, "head_trim": 0, "tail_trim": 0,
                     "latents": torch.zeros(128, 2, 4, 4, dtype=torch.float16)}]
    models = _types.SimpleNamespace(video_decoder=decoder)

    frames_list, audio_chunks = latent_jobs._ltx2_latent_decode_windows(
        lambda *a: None, {"abort": False}, {}, entry, window_specs, None, [],
        False, torch.float16, models, None)

    assert seen["latent_dtype"] == torch.bfloat16                     # follows the decoder
    assert frames_list[0].shape == (3, 2, 4, 4)
    assert audio_chunks == []


def test_ltx2_video_vae_dtype_refuses_float16():
    """The helper that fixes O6: float16 (and "no preference") becomes bfloat16."""
    try:
        from models.ltx2.ltx2 import ltx2_video_vae_dtype
    except Exception:                                       # the module probes CUDA at import
        pytest.skip("models.ltx2.ltx2 cannot be imported on this machine")

    assert ltx2_video_vae_dtype(torch.float16) == torch.bfloat16
    assert ltx2_video_vae_dtype(None) == torch.bfloat16
    assert ltx2_video_vae_dtype(torch.bfloat16) == torch.bfloat16
    assert ltx2_video_vae_dtype(torch.float32) == torch.float32


def test_ltx2_audio_decode_dtype_follows_the_model_and_round_trips():
    """The helper that fixes O7: the audio precision a latent file has to record.

    The audio latents are stored in the configured VAE precision, but the audio
    VAE and vocoder decode in the model's compute precision - bfloat16 for the
    usual float16 setup - so a file that records the video VAE's dtype sends the
    decode job to the wrong one and the vocoder returns a waveform about 2% off.
    Recording the rule is only half of it: the string written into the sidecar
    has to be the string the decode reads back, or the fix evaporates on the way
    to disk.
    """
    try:
        from models.ltx2.ltx2 import ltx2_audio_decode_dtype
    except Exception:                                       # the module probes CUDA at import
        pytest.skip("models.ltx2.ltx2 cannot be imported on this machine")

    assert ltx2_audio_decode_dtype(torch.float16) == torch.bfloat16
    assert ltx2_audio_decode_dtype(None) == torch.bfloat16
    assert ltx2_audio_decode_dtype(torch.bfloat16) == torch.bfloat16
    assert ltx2_audio_decode_dtype(torch.float32) == torch.float32

    for model_dtype in (torch.float16, None, torch.bfloat16, torch.float32):
        # What _write_latent_sidecar records, and what _edit_latent_job_ltx2 reads back.
        recorded = str(ltx2_audio_decode_dtype(model_dtype)).replace("torch.", "")
        assert recorded in latent_jobs._DTYPE_MAP
        assert latent_jobs._DTYPE_MAP[recorded] == ltx2_audio_decode_dtype(model_dtype)


def test_ltx2_decode_two_audio_windows_keep_the_frame_tensor(tmp_path, monkeypatch):
    """The LTX-2 decode of a two-audio-window sidecar still writes its video.

    A pixel-rule branch sidecar carries two audio blocks (the kept region's
    and the head window's rebased one), so the decode takes the multi-window
    merge path. That path used to name its per-window audio head drop
    ``frames``, shadowing the decoded frame tensor it is a parameter of, and
    the job died on "'int' object has no attribute 'shape'" before writing
    anything (9.3-17's first live run found it). The rule's head block also
    records a 17-frame video head trim that is the re-denoised bridge, not an
    audio head: the writer already dropped the bridge's columns from that
    block's audio, so the decode must not trim the chunk again.
    """
    import dataclasses

    saved, muxed = {}, {}

    def _fake_save_video(tensor=None, save_file=None, **kwargs):
        saved["tensor"] = tensor
        with open(save_file, "wb") as f:
            f.write(b"video")
        return save_file

    def _fake_write_wav(path, data, sample_rate):
        muxed["data"], muxed["rate"] = data, sample_rate
        with open(path, "wb") as f:
            f.write(b"wav")

    def _fake_combine(out_path, *args, **kwargs):
        muxed["out"] = out_path
        with open(out_path, "wb") as f:
            f.write(b"muxed")
        return out_path

    gen = {"file_list": [], "file_settings_list": []}
    ctx = dataclasses.replace(
        stub_ctx(tmp_path),
        get_available_filename=lambda folder, base, **kwargs: os.path.join(str(folder), base + ".bin"),
        get_gen_info=lambda state: gen)
    monkeypatch.setattr(latent_jobs, "save_video", _fake_save_video)
    monkeypatch.setattr(latent_jobs, "write_wav_file", _fake_write_wav)
    monkeypatch.setattr(latent_jobs, "combine_and_concatenate_video_with_audio_tracks", _fake_combine)
    monkeypatch.setattr(latent_jobs.notifications, "record_generation", lambda *a, **k: None)

    fps, rate = 24.0, 16000
    frames = torch.zeros(3, 9, 4, 4, dtype=torch.uint8)              # [c, f, h, w]
    entry = {"frames": 9, "sliding_window_overlap": 17,
             "windows": [{"window_no": 1, "head_trim": 0},
                         {"window_no": 2, "head_trim": 17, "branch_head_mode": "pixel_rule"}]}
    audio_entry = {"sample_rate": rate, "samples": 6000, "mel_bins": 16,
                   "windows": [{"window_no": 1, "start_sample": 0, "samples": 3000},
                               {"window_no": 2, "start_sample": 3000, "samples": 3000}]}
    chunks = [torch.ones(2, 3000), torch.ones(2, 3000)]

    latent_jobs._save_ltx2_decoded_video(
        ctx, lambda *a: None, {}, os.path.join(str(tmp_path), "source.mp4"), "", {},
        frames, chunks, rate, entry, audio_entry, fps, False, False, False, True)

    assert saved["tensor"].shape[1] == 9                             # the frames survived the merge
    assert muxed["data"].shape == (6000, 2)                          # kept block + head block, no re-trim
    assert muxed["rate"] == rate
    assert gen["file_list"] and gen["file_list"][0].endswith("source_latent_decode.bin")


def test_ltx2_decode_hands_the_audio_latents_the_recorded_dtype(monkeypatch):
    """The audio half follows the precision the audio entry recorded (O7).

    The generation loads every LTX-2 component at the model's precision - bfloat16 -
    and records it on the audio entry, while the video entry records float16.
    Handing the stored audio latents to the vocoder in the video's float16 changes
    the soundtrack it renders.
    """
    import types as _types

    seen = {}

    def _module(dtype):
        module = _types.SimpleNamespace()
        module.parameters = lambda: iter([torch.nn.Parameter(
            torch.zeros(1, dtype=dtype), requires_grad=False)])
        module.to = lambda *a, **k: module
        return module

    def _fake_decode_video(latent, video_decoder, *args, **kwargs):
        return torch.zeros(2, 4, 4, 3, dtype=torch.uint8)             # [T, H, W, C]

    def _fake_decode_audio(latent, audio_decoder, vocoder):
        seen["audio_dtype"] = latent.dtype
        return torch.zeros(1, 640, dtype=torch.float32)

    _stub_ltx2_decode_modules(monkeypatch, _fake_decode_video)
    monkeypatch.setattr(sys.modules["models.ltx2.ltx_core.model.audio_vae"], "decode_audio",
                        _fake_decode_audio, raising=False)

    entry = {"fps": 24.0}
    window_specs = [{"window_no": 1, "head_trim": 0, "tail_trim": 0,
                     "latents": torch.zeros(128, 2, 4, 4, dtype=torch.float16)}]
    audio_entry = {"mel_bins": 16, "sample_rate": 16000, "samples": 640,
                   "windows": [{"window_no": 1, "start_sample": 0, "samples": 640}]}
    audio_windows = [{"window_no": 1, "latents": torch.zeros(8, 16, dtype=torch.float16)}]
    models = _types.SimpleNamespace(video_decoder=_module(torch.bfloat16),
                                    audio_decoder=_module(torch.bfloat16),
                                    vocoder=_module(torch.bfloat16))

    frames_list, audio_chunks = latent_jobs._ltx2_latent_decode_windows(
        lambda *a: None, {"abort": False}, {}, entry, window_specs, audio_entry, audio_windows,
        False, torch.float16, models, None)

    assert seen["audio_dtype"] == torch.bfloat16                      # follows the audio models
    assert len(audio_chunks) == 1
def test_validate_latent_decode_accepts_an_api_latent_source(tmp_path):
    # The API can point at a companion file that is not next to the video;
    # the action must be decode, and a missing file is still refused.
    video = tmp_path / "source.mp4"
    video.write_bytes(b"x")
    api_latent = tmp_path / "elsewhere_latent.pt"
    api_latent.write_bytes(b"x")
    inputs = {"video_source": str(video)}
    assert latent_jobs.validate_latent_decode(
        inputs, api_options={"latent_source": str(api_latent), "latent_action": "decode"}) == ""
    error = latent_jobs.validate_latent_decode(
        inputs, api_options={"latent_action": "encode"})
    assert "Unsupported latent action" in error
    error = latent_jobs.validate_latent_decode(
        inputs, api_options={"latent_source": str(tmp_path / "missing.pt")})
    assert error.startswith("No saved latent companion file")

