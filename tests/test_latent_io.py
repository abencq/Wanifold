"""Tests for the latent save/decode/branching feature.

Run from the repository root:

    venv/bin/python -m pytest tests/test_latent_io.py -v

The pure settings- and latent-format tests always run; the tests that
need the app itself (wgp) skip automatically when the app's runtime
dependencies are not available.
"""

import json
import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from shared.utils import latent_io
from shared.utils import latent_jobs
from shared.utils import setting_names

def _import_wgp():
    """Import the app with a clean command-line argument list.

    The app's CLI parser is strict: the test runner's own arguments
    would be rejected, so they are set aside while the app is imported
    and restored right after.
    """
    saved_argv = list(sys.argv)
    sys.argv = ["wgp.py"]
    try:
        import wgp
    finally:
        sys.argv = saved_argv
    return wgp


try:
    wgp = _import_wgp()
except Exception as exc:  # the app's runtime dependencies may be missing
    wgp = None
    WGP_IMPORT_ERROR = exc
else:
    WGP_IMPORT_ERROR = None

requires_wgp = pytest.mark.skipif(wgp is None, reason=f"wgp unavailable ({WGP_IMPORT_ERROR!r})")


def _latent_ctx():
    """A JobContext for the moved latent engine (shared.utils.latent_jobs),
    built from the imported app module's globals."""
    return latent_jobs.JobContext(
        server_config=wgp.server_config,
        save_path=wgp.save_path,
        WanGP_version=wgp.WanGP_version,
        verbose_level=wgp.verbose_level,
        lock=wgp.lock,
        primary_settings=wgp.primary_settings,
        model_types_handlers=wgp.model_types_handlers,
        transformer_quantization=wgp.transformer_quantization,
        text_encoder_quantization=wgp.text_encoder_quantization,
        get_model_def=wgp.get_model_def,
        get_model_record=wgp.get_model_record,
        get_model_config_groups=wgp.get_model_config_groups,
        get_base_model_type=wgp.get_base_model_type,
        get_available_filename=wgp.get_available_filename,
        clear_status=wgp.clear_status,
        get_gen_info=wgp.get_gen_info,
        resolve_last_video=wgp._resolve_last_video,
    )


# ---------------------------------------------------------------------------
# The per-generation *Save Latents* choice (Misc. tab)
# ---------------------------------------------------------------------------

def test_save_latents_choice_catalog_entry():
    catalog = json.loads(open(os.path.join(REPO_ROOT, "models", "_settings.json"), encoding="utf-8").read())
    assert catalog["save_latents_choice"] == 0


def test_setting_names_accepts_save_latents_choice():
    assert "save_latents_choice" in setting_names.catalog_setting_names()
    assert setting_names.unknown_setting_names({"save_latents_choice": 1, "model_type": "wan_x"}) == []
    settings, rest = setting_names.split_settings({"save_latents_choice": 2, "some_media_note": "x"})
    assert settings == {"save_latents_choice": 2}
    assert rest == {"some_media_note": "x"}


@requires_wgp
@pytest.mark.parametrize("choice, default, expected", [
    (0, True, True),        # Configuration, global on  -> save
    (0, False, False),      # Configuration, global off -> no file
    (1, False, True),       # On                        -> save (one-off)
    (1, True, True),
    (2, True, False),       # Off                       -> no file (one-off)
    (2, False, False),
    (None, True, True),     # no choice (old settings, Deepy, API) -> global
    (None, False, False),
    ("1", False, True),     # a value from a JSON file may be a string
    ("2", True, False),
    ("garbage", True, True),
    ("garbage", False, False),
])
def test_effective_save_latents(choice, default, expected):
    assert latent_jobs.effective_save_latents(choice, default) is expected


@requires_wgp
def test_settings_version():
    # On the local working branch the version tracks main's baseline
    # (2.79, then 2.81 with the v18.21 merge): the one-time bump is carried by the feature
    # commit of the PR, so that the PR is a single bump from main.
    assert wgp.settings_version in (2.79, 2.80, 2.81, 2.82, 2.83)


@requires_wgp
def test_generate_media_takes_optional_choice():
    import inspect
    # Legacy queue items and settings never carry the key: the parameter
    # must default to None so the job falls back to the global option.
    assert inspect.signature(wgp.generate_media).parameters["save_latents_choice"].default is None


@requires_wgp
def test_save_inputs_carries_the_choice():
    import inspect
    # The form plumbing derives the saved names from the signature
    # (inputs_names = parameters[1:-2]); the choice must be among them.
    names = list(inspect.signature(wgp.save_inputs).parameters)[1:-2]
    assert "save_latents_choice" in names


@requires_wgp
def test_prepare_inputs_dict_keeps_the_choice():
    model_types = [t for t in wgp.models_def if str(t).startswith("wan")]
    if not model_types:
        pytest.skip("no wan model in this build")

    def run(target):
        model_type = model_types[0]
        inputs = {
            "state": {"model_type": model_type},
            "activated_loras": [],
            "guidance_phases": 1,
            "video_prompt_type": "",
            "save_latents_choice": 2,
            "output_filename": "",
        }
        for key in wgp.ATTACHMENT_KEYS:
            inputs[key] = None
        return wgp.prepare_inputs_dict(target, inputs)

    for target in ("state", "settings", "metadata"):
        out = run(target)
        assert out["save_latents_choice"] == 2, target
        if target != "metadata":
            assert out["settings_version"] == wgp.settings_version


@requires_wgp
def test_factory_settings_fill_in_the_choice():
    # The task executor fills every generate_media argument missing from
    # an old queue item from the model's factory settings; the factory
    # settings come from the shared catalog, so the new key is covered.
    model_types = [t for t in wgp.models_def if str(t).startswith("wan")]
    if not model_types:
        pytest.skip("no wan model in this build")
    factory = wgp.get_factory_settings(model_types[0])
    assert factory.get("save_latents_choice", 0) == 0


# ---------------------------------------------------------------------------
# The companion latent file format
# ---------------------------------------------------------------------------

def test_window_pixel_frames_layouts():
    latent = torch.zeros(16, 8, 4, 4)  # [c, t, h, w], 8 latent frames
    assert latent_io.window_pixel_frames(latent, stride=4, offset=1) == 1 + 4 * 7   # Wan
    assert latent_io.window_pixel_frames(latent, stride=8, offset=1) == 1 + 8 * 7   # LTX2
    assert latent_io.window_pixel_frames(latent) == 1 + 4 * 7                        # default Wan
    assert latent_io.window_pixel_frames(torch.zeros(16, 0, 4, 4), 4, 1) == 0


def test_h3_video_latent_pixel_frames():
    # the piecewise 5/17 layout: 2 -> 5, 7 -> 22, 12 -> 39, 17 -> 56
    assert latent_io.h3_video_latent_pixel_frames(2) == 5
    assert latent_io.h3_video_latent_pixel_frames(7) == 22
    assert latent_io.h3_video_latent_pixel_frames(12) == 39
    assert latent_io.h3_video_latent_pixel_frames(17) == 56
    assert latent_io.h3_video_latent_pixel_frames(1) == 0
    assert latent_io.h3_video_latent_pixel_frames(0) == 0


def test_is_all_latent_entry():
    good = {"format_id": latent_io.LATENT_FORMAT_ID, "windows": [{}]}
    assert latent_io.is_all_latent_entry(good)
    assert not latent_io.is_all_latent_entry({"format_id": latent_io.LATENT_FORMAT_ID})  # no windows
    assert not latent_io.is_all_latent_entry({"format_id": "other", "windows": [{}]})
    assert not latent_io.is_all_latent_entry(None)
    message = latent_io.non_current_latent_message("foo.pt")
    assert "foo.pt" in message and "Re-run" in message


def test_latent_payload_roundtrip():
    latent = torch.randn(4, 21, 8, 8, dtype=torch.float16)
    payload = latent_io.build_latent_payload(
        latent,
        app_version="17.17",
        model_type="wan_14_14B",
        vae_file="vae.pt",
        vae_sha256="ab",
        vae_z_dim=16,
        vae_dtype="16",
        frames=84,
        fps=16.0,
        seed=7,
        prompt="a prompt",
        steps=30,
        latent_stride=4,
        frame_offset=1,
    )
    entry = latent_io.latent_payload(payload)
    assert entry is not None
    assert entry["format_id"] == latent_io.LATENT_FORMAT_ID
    assert entry["frames"] == 84
    assert entry["vae_sha256"] == "ab"
    out = latent_io.payload_latents(entry)
    assert out.dtype == torch.float16
    assert out.shape == latent.shape
    torch.testing.assert_close(out, latent, rtol=0, atol=0)


def test_sidecar_file_roundtrip(tmp_path):
    latent = torch.randn(16, 34, 64, 36, dtype=torch.float16)
    meta = {
        "prompt": "a prompt",
        "seed": 7,
        "steps": 30,
        "denoising_strength": 1.0,
        "head_trim": 0,
        "tail_trim": 0,
        "frame_count": 133,  # 34 latent frames decode to 1 + 4 * 33 (Wan layout)
    }
    sidecar = latent_io.build_latent_sidecar(
        [latent],
        [meta],
        app_version="17.17",
        model_type="wan_14_14B",
        base_model_type="wan_14_14B",
        vae_file="vae.pt",
        vae_sha256="ab",
        vae_z_dim=16,
        vae_dtype="16",
        width=1024,
        height=576,
        fps=16.0,
        latent_stride=4,
        frame_offset=1,
    )
    path = str(tmp_path / "video_latent.pt")
    torch.save(sidecar, path)
    entry = latent_io.latent_payload(latent_io.load_latent_file(path))
    assert latent_io.is_all_latent_entry(entry)
    assert entry["frames"] == 133
    specs = latent_io.entry_window_specs(entry)
    assert len(specs) == 1
    assert specs[0]["latents"].shape == latent.shape
    torch.testing.assert_close(specs[0]["latents"], latent, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Magi Human (Phase 1: the family predicate and the layout round-trip)
# ---------------------------------------------------------------------------

def test_magi_latent_layout():
    # Magi Human's Turbo VAED video VAE is the linear 4/1 grid (the same
    # layout the 5B Wan 2.2 VAE uses), so the shared helper's default
    # covers it: 26 latent frames decode to 1 + 4 * 25 = 101 pixel frames.
    assert latent_io.latent_layout_for({}, "magi_human") == {"stride": 4, "offset": 1}
    assert latent_io.window_pixel_frames(torch.zeros(48, 26, 16, 28), 4, 1) == 101


def test_magi_sidecar_roundtrip(tmp_path):
    # Two sliding windows (a 101-frame window and an 85-frame window with
    # one head-trimmed frame): 48-channel video latents [48, t, h, w] and
    # 64-channel audio latents [64, t] (one audio latent per pixel frame,
    # 44.1 kHz stereo).
    video = [torch.randn(48, 26, 16, 28, dtype=torch.float16),
             torch.randn(48, 22, 16, 28, dtype=torch.float16)]
    audio = [torch.randn(64, 101, dtype=torch.float16),
             torch.randn(64, 85, dtype=torch.float16)]
    meta = [
        {"prompt": "a prompt", "seed": 1, "steps": 8, "denoising_strength": 1.0,
         "head_trim": 0, "tail_trim": 0, "frame_count": 101},
        {"prompt": "another prompt", "seed": 2, "steps": 8, "denoising_strength": 1.0,
         "head_trim": 1, "tail_trim": 0, "frame_count": 85},
    ]
    sidecar = latent_io.build_latent_sidecar(
        video, meta,
        app_version="17.17", model_type="magi_human", base_model_type="magi_human",
        vae_file="turbo_vae/TurboV3-Wan22-TinyShallow_7_7.safetensors", vae_sha256="ab",
        vae_z_dim=48, vae_dtype="float32",
        width=448, height=256, fps=25.0, latent_stride=4, frame_offset=1,
    )
    # 101 frames at 25 fps -> 177960 samples; 85 -> 149580. The audio latents
    # line up one-to-one with the pixel frames, so the sample spans are exact.
    samples1 = int(101 * 44100 / 25)
    samples2 = int(85 * 44100 / 25)
    audio_payload = latent_io.build_audio_latent_payload(
        audio,
        [{"window_no": 1, "start_sample": 0, "samples": samples1},
         {"window_no": 2, "start_sample": samples1, "samples": samples2}],
        app_version="17.17", model_type="magi_human", base_model_type="magi_human",
        audio_vae_file="stable-audio-open-1.0/model.safetensors", audio_vae_sha256="cd",
        audio_vae_dtype="float32", sample_rate=44100, channels=2,
        samples=samples1 + samples2, seed=1, prompt="a prompt", steps=8, total_windows=2,
    )
    file_obj = dict(sidecar)
    file_obj.update(audio_payload)
    path = str(tmp_path / "magi_latent.pt")
    latent_io.write_latent_file(path, file_obj)
    loaded = latent_io.load_latent_file(path)
    entry = latent_io.latent_payload(loaded)
    assert latent_io.is_all_latent_entry(entry)
    assert entry["frames"] == 101 + (85 - 1)
    specs = latent_io.entry_window_specs(entry)
    assert len(specs) == 2
    assert specs[0]["latents"].shape == video[0].shape
    assert specs[1]["latents"].shape == video[1].shape
    torch.testing.assert_close(specs[0]["latents"], video[0], rtol=0, atol=0)
    torch.testing.assert_close(specs[1]["latents"], video[1], rtol=0, atol=0)
    audio_entry = latent_io.audio_latent_payload(loaded)
    assert audio_entry["format_id"] == latent_io.LATENT_FORMAT_ID_AUDIO
    aspecs = latent_io.audio_window_specs(audio_entry)
    assert len(aspecs) == 2
    assert aspecs[0]["latents"].shape == audio[0].shape
    assert aspecs[1]["latents"].shape == audio[1].shape
    assert aspecs[1]["start_sample"] == samples1
    torch.testing.assert_close(aspecs[0]["latents"], audio[0], rtol=0, atol=0)
    torch.testing.assert_close(aspecs[1]["latents"], audio[1], rtol=0, atol=0)


@requires_wgp
def test_magi_is_all_latent_family():
    # Phase 1: the Magi Human family writes the all-latent sidecar.
    assert latent_jobs._is_magi_human_family("magi_human", {})
    assert latent_jobs._is_magi_human_family("magi_human", {"architecture": "magi_human"})
    assert not latent_jobs._is_magi_human_family("minimax_h3", {"architecture": "minimax_h3"})
    assert latent_jobs._is_all_latent_family(_latent_ctx(), "magi_human", {"architecture": "magi_human"})
    assert not latent_jobs._is_all_latent_family(_latent_ctx(), "hunyuan_video", {})
    if "magi_human_distill" in wgp.models_def:
        # the distill model keeps its own arch in the catalog; the family
        # predicate covers both arches
        base = wgp.get_base_model_type("magi_human_distill")
        assert base in ("magi_human", "magi_human_distill")
        assert latent_jobs._is_all_latent_family(_latent_ctx(), base, wgp.get_model_def("magi_human_distill"))


@requires_wgp
def test_magi_branch_prefix_table():
    # The frozen-prefix support: Magi Human sits on the linear 4/1, 16x,
    # 48-channel grid (the Turbo VAED). full_freeze carries the kept latents
    # verbatim; the default pixel_rule re-encodes the handoff tail as the
    # frozen head (the table entry also enables the kept-region decode check).
    assert latent_jobs._BRANCH_PREFIX_LAYOUTS.get("magi_human") == (4, 1, 16, 48)
    assert latent_jobs._BRANCH_PREFIX_LAYOUTS.get("magi_human_distill") == (4, 1, 16, 48)
    if "magi_human" not in wgp.models_def and "magi_human" not in wgp.model_types_handlers:
        pytest.skip("no Magi Human model in this build")
    assert latent_jobs._branch_prefix_advisory(_latent_ctx(), "magi_human", head_mode="full_freeze") == ""
    note = latent_jobs._branch_prefix_advisory(_latent_ctx(), "magi_human", head_mode="pixel_rule")
    assert "re-encodes" in note and "full_freeze" in note


def test_magi_decode_audio_assembly():
    # Phase 2: the Latent Decode job re-assembles Magi's per-window decoded
    # audio with the writer-mirror model (window 1 in full, every later
    # window with its overlap's samples dropped, placed at the accumulated
    # position). Magi's audio latent is one 64-channel frame per pixel
    # frame, so the sample spans are exact and the merged track's length
    # must equal the committed video's length in samples.
    fps = 25.0
    sample_rate = 44100
    # Two windows: 101 frames, then 85 frames with a 1-frame head trim
    # (the same job the round-trip test writes).
    samples1 = int(round(101 / fps * sample_rate))     # 177960
    samples2 = int(round(85 / fps * sample_rate))      # 149580
    head_trim_samples = int(round(1 * sample_rate / fps))  # 1764
    chunk1 = torch.arange(samples1, dtype=torch.float32).reshape(1, -1).repeat(2, 1)
    chunk2 = (torch.arange(samples2, dtype=torch.float32) + 1000).reshape(1, -1).repeat(2, 1)
    windows = [{"window_no": 1, "start_sample": 0, "samples": samples1},
               {"window_no": 2, "start_sample": samples1, "samples": samples2}]
    head_trims = [0, head_trim_samples]  # what the decode job derives from the video's head trims
    merged = latent_io.merge_audio_window_chunks([chunk1, chunk2], windows, samples1 + samples2, head_trims=head_trims)
    committed_frames = 101 + (85 - 1)
    assert merged.shape == (2, int(round(committed_frames / fps * sample_rate)))
    # Window 1 verbatim, window 2 without its overlap head, no overlap left.
    torch.testing.assert_close(merged[:, :samples1], chunk1, rtol=0, atol=0)
    torch.testing.assert_close(merged[:, samples1:], chunk2[:, head_trim_samples:], rtol=0, atol=0)


def test_merge_audio_window_chunks_crops_a_chunk_to_its_recorded_samples():
    """A decoded chunk longer than the ``samples`` its window recorded must be
    cropped, or it shifts every later window off its recorded start (O7).
    """
    chunk1 = torch.arange(110, dtype=torch.float32).reshape(1, -1)      # 10 more than recorded
    chunk2 = (torch.arange(80, dtype=torch.float32) + 1000).reshape(1, -1)
    windows = [{"window_no": 1, "start_sample": 0, "samples": 100},
               {"window_no": 2, "start_sample": 100, "samples": 80}]
    merged = latent_io.merge_audio_window_chunks([chunk1, chunk2], windows, 180, head_trims=[0, 0])
    assert merged.shape == (1, 180)
    torch.testing.assert_close(merged[:, :100], chunk1[:, :100], rtol=0, atol=0)
    torch.testing.assert_close(merged[:, 100:], chunk2, rtol=0, atol=0)


def test_merge_audio_window_chunks_pads_a_short_run_to_its_recorded_samples():
    """The other side of a committed run (O8): the writer force-fits every run it
    lays down to the length the job recorded for it (``apad`` then ``atrim`` in the
    mux), so a kept soundtrack that decodes shorter than its join really is that
    long on the written track. The decode must pad the shortfall with silence — a
    short run left alone slides every later chunk early by the difference.
    """
    chunk1 = torch.arange(100, dtype=torch.float32).reshape(1, -1)      # 20 short of the join
    chunk2 = (torch.arange(80, dtype=torch.float32) + 1000).reshape(1, -1)
    windows = [{"window_no": 1, "start_sample": 0, "samples": 120},
               {"window_no": 2, "start_sample": 120, "samples": 80}]
    merged = latent_io.merge_audio_window_chunks([chunk1, chunk2], windows, 200, head_trims=[0, 0])
    assert merged.shape == (1, 200)
    torch.testing.assert_close(merged[:, :100], chunk1, rtol=0, atol=0)
    torch.testing.assert_close(merged[:, 100:120], torch.zeros(1, 20), rtol=0, atol=0)
    torch.testing.assert_close(merged[:, 120:], chunk2, rtol=0, atol=0)
    # A head trim shortens the committed run by the trimmed samples, not by the
    # shortfall: the second chunk keeps its own 80 committed samples minus its trim.
    merged = latent_io.merge_audio_window_chunks([chunk1, chunk2], windows, 200, head_trims=[0, 10])
    assert merged.shape == (1, 190)
    torch.testing.assert_close(merged[:, 120:], chunk2[:, 10:], rtol=0, atol=0)


def _magi_branch_sidecar():
    # The same two-window sidecar the round-trip test writes (101-frame
    # window, then an 85-frame window with one head-trimmed frame), as a
    # video entry + audio entry over one in-memory payload (no file
    # needed for the splice).
    fps = 25.0
    sample_rate = 44100
    video = [torch.randn(48, 26, 16, 28, dtype=torch.float16),
             torch.randn(48, 22, 16, 28, dtype=torch.float16)]
    audio = [torch.randn(64, 101, dtype=torch.float16),
             torch.randn(64, 85, dtype=torch.float16)]
    meta = [
        {"prompt": "a prompt", "seed": 1, "steps": 8, "denoising_strength": 1.0,
         "head_trim": 0, "tail_trim": 0, "frame_count": 101},
        {"prompt": "another prompt", "seed": 2, "steps": 8, "denoising_strength": 1.0,
         "head_trim": 1, "tail_trim": 0, "frame_count": 85},
    ]
    samples1 = int(round(101 / fps * sample_rate))     # 177960
    samples2 = int(round(85 / fps * sample_rate))      # 149580
    sidecar = latent_io.build_latent_sidecar(
        video, meta,
        app_version="17.17", model_type="magi_human", base_model_type="magi_human",
        vae_file="turbo_vae/TurboV3-Wan22-TinyShallow_7_7.safetensors", vae_sha256="ab",
        vae_z_dim=48, vae_dtype="float32",
        width=448, height=256, fps=fps, latent_stride=4, frame_offset=1,
    )
    file_obj = dict(sidecar)
    file_obj.update(latent_io.build_audio_latent_payload(
        audio,
        [{"window_no": 1, "start_sample": 0, "samples": samples1},
         {"window_no": 2, "start_sample": samples1, "samples": samples2}],
        app_version="17.17", model_type="magi_human", base_model_type="magi_human",
        audio_vae_file="stable-audio-open-1.0/model.safetensors", audio_vae_sha256="cd",
        audio_vae_dtype="float32", sample_rate=sample_rate, channels=2,
        samples=samples1 + samples2, seed=1, prompt="a prompt", steps=8, total_windows=2,
    ))
    return video, audio, latent_io.latent_payload(file_obj), latent_io.audio_latent_payload(file_obj)


def test_magi_branch_audio_prefix_slice():
    # Phase 3: the branch audio prefix is the containing window's audio
    # block sliced at the resolved cut (the shared time-ratio splice's
    # flat layout): a mid-window cut keeps the block's head, a cut at the
    # window's committed end keeps its committed tail (the shared
    # overlap frame dropped), and the slice length is the kept frames as
    # a fraction of the window's recorded duration.
    video, audio, entry, audio_entry = _magi_branch_sidecar()
    # Window 2's head-trimmed overlap frame is final frame 101, so its
    # decoded frame m is final frame 100 + m. Picking final frame 117
    # (decoded frame 17) lands exactly on the boundary of 5 kept latents
    # (the linear 4/1 arithmetic: 1 + 4 * 4 = 17 decoded frames), so the
    # cut is its own in both modes: latent_index 5, decoded_prefix 17.
    bp = latent_io.branch_point_for_frame(entry, 117)
    assert bp is not None
    assert bp["window_index"] == 1
    assert bp["latent_index"] == 5
    assert bp["decoded_prefix"] == 17
    slice_t = latent_io.magi_branch_audio_prefix_latents(audio_entry, entry, bp)
    assert slice_t is not None and slice_t.shape == (64, 17)
    torch.testing.assert_close(slice_t, audio[1][:, :17], rtol=0, atol=0)
    # A cut inside window 1: final frame 41 is the 11-latent boundary
    # (1 + 4 * 10 = 41) -> latent_index 11, decoded_prefix 41.
    bp1 = latent_io.branch_point_for_frame(entry, 41)
    assert bp1 is not None
    assert bp1["window_index"] == 0
    assert bp1["latent_index"] == 11
    assert bp1["decoded_prefix"] == 41
    slice_1 = latent_io.magi_branch_audio_prefix_latents(audio_entry, entry, bp1)
    assert slice_1 is not None and slice_1.shape == (64, 41)
    torch.testing.assert_close(slice_1, audio[0][:, :41], rtol=0, atol=0)
    # A cut at window 2's committed end (final frame 185): the block's
    # committed tail, the one overlap sample dropped.
    bp_end = latent_io.branch_point_for_frame(entry, 185)
    assert bp_end is not None
    assert bp_end["window_index"] == 1
    assert bp_end["latent_index"] == 22  # the window's full latent count: the cut ends the window
    assert bp_end["decoded_prefix"] == 85
    slice_e = latent_io.magi_branch_audio_prefix_latents(audio_entry, entry, bp_end)
    assert slice_e is not None and slice_e.shape == (64, 84)
    torch.testing.assert_close(slice_e, audio[1][:, 1:], rtol=0, atol=0)


def test_magi_branch_audio_bridge_slice():
    # The shared splice's bridge mode on the flat layout: the columns that end
    # at the cut, not the window's head — the shape every family's pixel-rule
    # port freezes when its turn comes.
    video, audio, entry, audio_entry = _magi_branch_sidecar()
    # Mid cut inside window 2 (decoded frame 17, step 17 of its block): the
    # 9-frame bridge is the columns 8..17 — the nine that end at the cut.
    bp = latent_io.branch_point_for_frame(entry, 117)
    bridge = latent_io._branch_audio_prefix_slice(audio_entry, entry, bp, bridge_frames=9)
    assert bridge is not None and bridge["count"] == 9 and bridge["layout"] == "magi"
    torch.testing.assert_close(bridge["flat"], audio[1][:, 8:17], rtol=0, atol=0)
    # The block's own duration, so a writer can advance the track position by
    # exactly what a slice drops: 9 of window 2's 85 columns of its recorded
    # sample count.
    samples2 = int(round(85 / 25 * 44100))
    assert bridge["samples"] == int(round(samples2 * 9 / 85))
    # A cut at the window's committed end: the bridge is the block's last
    # columns, ending where the block ends.
    bp_end = latent_io.branch_point_for_frame(entry, 185)
    bridge_end = latent_io._branch_audio_prefix_slice(audio_entry, entry, bp_end, bridge_frames=9)
    assert bridge_end is not None and bridge_end["tail"] and bridge_end["count"] == 9
    torch.testing.assert_close(bridge_end["flat"], audio[1][:, 76:85], rtol=0, atol=0)
    # With the bridge longer than the audio before the cut, the slice stops at
    # the cut: the window's head, never its future.
    bridge_long = latent_io._branch_audio_prefix_slice(audio_entry, entry, bp, bridge_frames=40)
    assert bridge_long is not None and bridge_long["count"] == 17
    torch.testing.assert_close(bridge_long["flat"], audio[1][:, :17], rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Ovi (the Wan 2.2 5B-class audio-video model)
# ---------------------------------------------------------------------------

def test_ovi_latent_layout():
    # Ovi's video VAE is the Wan 2.2 5B-class one (the linear 4/1 grid,
    # 16x spatial, 48-channel latent), so the shared helper's default
    # covers it: 31 latent frames decode to 1 + 4 * 30 = 121 pixel frames.
    assert latent_io.latent_layout_for({}, "ovi") == {"stride": 4, "offset": 1}
    assert latent_io.window_pixel_frames(torch.zeros(48, 31, 45, 80), 4, 1) == 121


@requires_wgp
def test_ovi_is_all_latent_family():
    # The Ovi family writes the all-latent sidecar (the handler rides the
    # Wan family, so the family predicate covers it as well).
    assert latent_jobs._is_ovi_family("ovi", {})
    assert latent_jobs._is_ovi_family("ovi", {"architecture": "ovi"})
    assert not latent_jobs._is_ovi_family("ti2v_2_2", {"architecture": "ti2v_2_2"})
    assert latent_jobs._is_all_latent_family(_latent_ctx(), "ovi", {"architecture": "ovi"})


@requires_wgp
def test_ovi_branch_prefix_table():
    # The frozen-prefix support: Ovi sits on the linear 4/1, 16x, 48-channel
    # grid (the 5B Wan 2.2 VAE). full_freeze carries the kept latents verbatim;
    # the default pixel_rule re-encodes the handoff tail as the frozen head.
    assert latent_jobs._BRANCH_PREFIX_LAYOUTS.get("ovi") == (4, 1, 16, 48)
    if "ovi" not in wgp.models_def and "ovi" not in wgp.model_types_handlers:
        pytest.skip("no Ovi model in this build")
    assert latent_jobs._branch_prefix_advisory(_latent_ctx(), "ovi", head_mode="full_freeze") == ""
    note = latent_jobs._branch_prefix_advisory(_latent_ctx(), "ovi", head_mode="pixel_rule")
    assert "re-encodes" in note and "full_freeze" in note


@requires_wgp
def test_ovi_resolves_as_wan_family_handler():
    # The Ovi handler lives in the models.wan package (the ovi_handler
    # module), so the registry's module check resolves it as the Wan
    # family for a base type outside the explicit base-type tuples.
    assert latent_jobs._is_wan_family(_latent_ctx(), "ovi", {})
    assert not latent_jobs._is_ovi_family("t2v_2_2", {"architecture": "t2v_2_2"})


def _ovi_branch_sidecar():
    # Two 121-frame sliding windows at 24 fps (the second head-trimmed by
    # one frame): 48-channel video latents [48, 31, h, w] and 20-channel
    # audio latents [20, 157] (the mmaudio VAE's 16 kHz mono latent), one
    # audio window per video window.
    fps = 24.0
    sample_rate = 16000
    video = [torch.randn(48, 31, 45, 80, dtype=torch.float16),
             torch.randn(48, 31, 45, 80, dtype=torch.float16)]
    audio = [torch.randn(20, 157, dtype=torch.float16),
             torch.randn(20, 157, dtype=torch.float16)]
    meta = [
        {"prompt": "a prompt", "seed": 1, "steps": 30, "denoising_strength": 1.0,
         "head_trim": 0, "tail_trim": 0, "frame_count": 121},
        {"prompt": "another prompt", "seed": 2, "steps": 30, "denoising_strength": 1.0,
         "head_trim": 1, "tail_trim": 0, "frame_count": 121},
    ]
    sidecar = latent_io.build_latent_sidecar(
        video, meta,
        app_version="17.17", model_type="ovi", base_model_type="ovi",
        vae_file="Wan2.2_VAE.safetensors", vae_sha256="ab",
        vae_z_dim=48, vae_dtype="bfloat16",
        width=1280, height=720, fps=fps, latent_stride=4, frame_offset=1,
    )
    # Both windows record their full decoded length (121 frames at 24 fps
    # -> 80667 samples), the way the generation records it.
    samples1 = int(round(121 / fps * sample_rate))
    samples2 = int(round(121 / fps * sample_rate))
    file_obj = dict(sidecar)
    file_obj.update(latent_io.build_audio_latent_payload(
        audio,
        [{"window_no": 1, "start_sample": 0, "samples": samples1},
         {"window_no": 2, "start_sample": samples1, "samples": samples2}],
        app_version="17.17", model_type="ovi", base_model_type="ovi",
        audio_vae_file="v1-16.pth", audio_vae_sha256="cd",
        audio_vae_dtype="bfloat16", sample_rate=sample_rate, channels=1,
        samples=samples1 + samples2, seed=1, prompt="a prompt", steps=30, total_windows=2,
    ))
    return video, audio, file_obj

def test_ovi_full_freeze_prefix_and_pixel_rule_reencode():
    # The same Ovi file and cut: full_freeze carries the whole kept run
    # verbatim; pixel_rule freezes only the overlap tail and leaves the
    # encoded head to the model.
    video, audio, file_obj = _ovi_branch_sidecar()
    entry = latent_io.latent_payload(file_obj)
    point = latent_io.branch_point_for_frame(entry, 121)
    assert point is not None
    carried = latent_io.carry_over_latent_windows(file_obj, point)
    model_def = {"branch_pixel_rule": "encoded_head"}
    full, full_err = latent_jobs._prepare_branch_prefix(
        entry, point, carried, base_model_type="ovi", video_length=40,
        reuse_frames=5, model_def=model_def,
        vae_file="Wan2.2_VAE.safetensors", vae_sha256="ab",
        head_mode="full_freeze")
    assert full_err == "" and full is not None
    assert full["head_mode"] == "full_freeze"
    assert int(full["prefix_latents"].shape[1]) == 31
    assert torch.equal(full["prefix_latents"], video[0])
    rule, rule_err = latent_jobs._prepare_branch_prefix(
        entry, point, carried, base_model_type="ovi", video_length=40,
        reuse_frames=5, model_def=model_def,
        vae_file="Wan2.2_VAE.safetensors", vae_sha256="ab",
        head_mode="pixel_rule")
    assert rule_err == "" and rule is not None
    assert rule["head_mode"] == "pixel_rule"
    assert int(rule["prefix_latents"].shape[1]) == rule["head_prefix_latents"] == 2
    assert rule["assembly_overlap"] == rule["guide_frames"] == 5
    assert rule["kept_frames"] == 121



def test_ovi_sidecar_roundtrip(tmp_path):
    # Ovi's sidecar carries both modalities: the video windows and the
    # aligned audio windows, the audio VAE's 16 kHz mono identity.
    video, audio, file_obj = _ovi_branch_sidecar()
    path = str(tmp_path / "ovi_latent.pt")
    latent_io.write_latent_file(path, file_obj)
    loaded = latent_io.load_latent_file(path)
    entry = latent_io.latent_payload(loaded)
    assert latent_io.is_all_latent_entry(entry)
    assert entry["frames"] == 121 + (121 - 1)
    specs = latent_io.entry_window_specs(entry)
    assert len(specs) == 2
    torch.testing.assert_close(specs[0]["latents"], video[0], rtol=0, atol=0)
    torch.testing.assert_close(specs[1]["latents"], video[1], rtol=0, atol=0)
    audio_entry = latent_io.audio_latent_payload(loaded)
    assert audio_entry["format_id"] == latent_io.LATENT_FORMAT_ID_AUDIO
    assert audio_entry["sample_rate"] == 16000
    assert audio_entry["channels"] == 1
    aspecs = latent_io.audio_window_specs(audio_entry)
    assert len(aspecs) == 2
    torch.testing.assert_close(aspecs[0]["latents"], audio[0], rtol=0, atol=0)
    torch.testing.assert_close(aspecs[1]["latents"], audio[1], rtol=0, atol=0)


def test_ovi_branch_audio_prefix_slice():
    # The branch audio prefix is the containing window's audio block sliced
    # at the resolved cut (the shared time-ratio splice's flat layout): a
    # mid-window cut keeps the block's head, a cut at the window's
    # committed end keeps its committed tail, and the slice length is the
    # kept frames' duration as a fraction of the window's recorded
    # duration (157 audio steps per 121-frame window).
    video, audio, file_obj = _ovi_branch_sidecar()
    entry = latent_io.latent_payload(file_obj)
    audio_entry = latent_io.audio_latent_payload(file_obj)
    # Window 2's head-trimmed overlap frame is final frame 121, so its
    # decoded frame m is final frame 120 + m. Picking final frame 161
    # (decoded frame 41) lands exactly on the boundary of 11 kept latents
    # (the linear 4/1 arithmetic: 1 + 4 * 10 = 41 decoded frames), so the
    # cut is its own in both modes: latent_index 11, decoded_prefix 41.
    bp = latent_io.branch_point_for_frame(entry, 161)
    assert bp is not None
    assert bp["window_index"] == 1
    assert bp["latent_index"] == 11
    assert bp["decoded_prefix"] == 41
    count = int(round(157 * (41 / 24.0) / (80667 / 16000.0)))  # 53
    slice_t = latent_io.ovi_branch_audio_prefix_latents(audio_entry, entry, bp)
    assert slice_t is not None and slice_t.shape == (20, count)
    torch.testing.assert_close(slice_t, audio[1][:, :count], rtol=0, atol=0)
    # A cut inside window 1: final frame 41 is the 11-latent boundary
    # (1 + 4 * 10 = 41) -> latent_index 11, decoded_prefix 41.
    bp1 = latent_io.branch_point_for_frame(entry, 41)
    assert bp1 is not None
    assert bp1["window_index"] == 0
    assert bp1["latent_index"] == 11
    assert bp1["decoded_prefix"] == 41
    slice_1 = latent_io.ovi_branch_audio_prefix_latents(audio_entry, entry, bp1)
    assert slice_1 is not None and slice_1.shape == (20, count)
    torch.testing.assert_close(slice_1, audio[0][:, :count], rtol=0, atol=0)
    # A cut at window 2's committed end (final frame 241): the window's
    # full decoded position (121) — the one overlap frame is dropped by
    # the committed-tail slice below, not by the cut itself.
    bp_end = latent_io.branch_point_for_frame(entry, 241)
    assert bp_end is not None
    assert bp_end["window_index"] == 1
    assert bp_end["latent_index"] == 31  # the window's full latent count: the cut ends the window
    assert bp_end["decoded_prefix"] == 121
    count_end = int(round(157 * (120 / 24.0) / (80667 / 16000.0)))  # 156
    slice_e = latent_io.ovi_branch_audio_prefix_latents(audio_entry, entry, bp_end)
    assert slice_e is not None and slice_e.shape == (20, count_end)
    torch.testing.assert_close(slice_e, audio[1][:, 157 - count_end:], rtol=0, atol=0)


@requires_wgp
def test_generate_media_still_wears_its_own_wrappers():
    # The Phase C acceptance run (latent-revisit.md §8.1) found the branch refusal's
    # message builder inserted *between* generate_media's decorators and its def:
    # @workspace_storage.workspace_generation and @generation_downloads had moved onto
    # the helper, so a branch refusal died in the wrapper with KeyError: 'state'
    # instead of naming its reason, and every generation ran outside its workspace
    # binding and outside download_operation()/MediaControl. A function that only
    # builds a sentence must not wear a job's wrappers, and the job must wear them.
    assert getattr(wgp.generate_media, "__wrapped__", None) is not None
    assert getattr(wgp._branch_decode_refusal, "__wrapped__", None) is None
    refusal = wgp._branch_decode_refusal("the test reason")
    assert refusal.startswith("From latent:") and "the test reason" in refusal

