"""What a latent branch hands the head window as its guide.

A "From WanGP latent file" branch must differ from continuing the same job from
a video file in exactly one way: where the pixel job reads the previous
section's last frames from a file, the branch decodes them from the latents it
kept. From those frames on, the two jobs have to be the same job - the same
conditioning entry, the same latent index, the same strength and the same
encode on each stage.

Handing the window the kept latents themselves was tried, and is wrong however
exact the count looks. N kept latents and this window's first N latents cover
the same number of frames, but they are not the same decomposition of them: the
kept latents come from the middle of the source run, where every latent carries
a full eight frames, while an encode of the same frames on this window's grid
puts one frame in its first latent and eight in each later one. Frozen onto that
head at full strength - unable to fix it - the window generates over-detailed,
boiling texture for its whole length and re-composes the camera: measured
against the same scene continued from a video file, the same crop of the same
trees at the same frame goes from a smooth canopy to a field of white speckles.

So the branch's guide is pixels, and it is the same decode that builds the
output's kept prefix, taken once. These tests pin the three parts of that: the
per-window decode that produces frames in the layout and levels a guide travels
in, the fact that the encode is bypassed nowhere, and the sharing that keeps the
branch to one decode of the kept region.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from models.ltx2 import ltx2
    from models.ltx2.ltx_core.model import video_vae
    from models.ltx2.ltx_pipelines.utils import helpers
except Exception:  # the LTX2 core is a heavy optional dependency
    pytest.skip("the LTX2 pipeline package is not importable", allow_module_level=True)


# --- the tail decode that makes the guide -----------------------------------

class _StubDecoder:
    """Just enough decoder for the tail decode to ask it where it lives."""

    def parameters(self):
        return iter([torch.zeros(1)])


def _guide_latents(frames=3, channels=4, height=8, width=12):
    return torch.zeros(channels, frames, height, width, dtype=torch.float16)


@pytest.fixture
def decoded(monkeypatch):
    """Replaces the VAE decode with a known ramp of pixel levels: the kept
    stretch's first frame black, a middle one at level 127, a last one white."""
    seen = {}

    def _fake_decode(latent, decoder, tiling_config=None, interrupt_check=None, **kwargs):
        seen["latent"] = latent
        frames, height, width = int(latent.shape[2]) * 8 - 7, 4, 6
        out = torch.zeros(frames, height, width, 3, dtype=torch.uint8)
        out[0] = 0
        out[1] = 127
        out[2] = 255
        return out

    monkeypatch.setattr(video_vae, "decode_video_to_tensor", _fake_decode)
    return seen


def test_the_tail_decodes_to_the_layout_a_guide_travels_in(decoded):
    """The decoder answers ``[F, H, W, C]`` of uint8 levels; a guide is handed
    over as ``[C, F, H, W]`` of float - the layout the entry builder permutes
    and the encoder reads - and the causal layout turns 3 latents into 17
    frames."""
    guide = ltx2._decode_latent_tail(_StubDecoder(), _guide_latents())
    assert guide.dim() == 4 and int(guide.shape[0]) == 3
    assert int(guide.shape[1]) == 3 * 8 - 7
    assert guide.dtype == torch.float32


def test_the_tail_decodes_to_the_levels_a_guide_travels_in(decoded):
    guide = ltx2._decode_latent_tail(_StubDecoder(), _guide_latents())
    assert float(guide[:, 0].min()) == pytest.approx(-1.0)              # level 0
    assert float(guide[:, 2].max()) == pytest.approx(1.0)               # level 255
    assert float(guide[:, 1].mean()) == pytest.approx(-0.004, abs=1e-3)  # level 127
    assert float(guide.min()) >= -1.0 and float(guide.max()) <= 1.0


def test_the_decode_is_asked_for_a_batched_latent_on_its_own_dtype(decoded):
    """The decoder wants a batch and the dtype of its own weights, whatever the
    sidecar stored the kept latents in."""
    ltx2._decode_latent_tail(_StubDecoder(), _guide_latents())
    assert decoded["latent"].dim() == 5
    assert decoded["latent"].dtype == torch.float32     # the stub's parameter dtype
@pytest.mark.parametrize("decoder,answer", [
    (None, None),                                                     # nothing loaded
    (_StubDecoder(), None),                                           # interrupted
    (_StubDecoder(), torch.zeros(0, 4, 6, 3, dtype=torch.uint8)),      # came back empty
])
def test_without_frames_there_is_no_guide(monkeypatch, decoder, answer):
    """No frames, no guide - and the caller says so rather than letting the
    head window start unanchored in silence."""
    monkeypatch.setattr(video_vae, "decode_video_to_tensor", lambda *a, **kw: answer)
    assert ltx2._decode_latent_tail(decoder, _guide_latents()) is None


# --- and from there the job is a plain continuation -------------------------

class _StubEncoder:
    """What the conditioning builders read off the encoder: the channel count."""
    latent_channels = 4


def _encoder_kwargs():
    return dict(video_encoder=_StubEncoder(), dtype=torch.float16, device=torch.device("cpu"))


@pytest.fixture
def encode_calls(monkeypatch):
    """Counts the encode instead of running it, so the two routes stay apart."""
    calls = {"n": 0}
    monkeypatch.setattr(helpers, "load_image_conditioning",
                        lambda **kw: torch.zeros(4, 16, 8, 12, dtype=torch.float16))

    def _encode(video, encoder, *args, **kwargs):
        calls["n"] += 1
        return torch.ones(1, 4, 1, 8, 12, dtype=torch.float16)

    monkeypatch.setattr(helpers, "vae_encode_video", _encode)
    return calls


def test_the_guide_goes_through_the_encode(encode_calls):
    """The branch's guide is pixels, so it is encoded exactly like the frames a
    video file hands a continuation, and the conditioning built on it is that
    job's: the same item, at the window's first frame index, at the job's
    strength. ``guiding_images`` is the list the prefix entry lands in."""
    guide = torch.zeros(17, 3, 64, 96, dtype=torch.float32)   # a decoded tail
    item = helpers.image_conditionings_by_adding_guiding_latent(
        images=[(guide, 0, 1.0)], height=64, width=96, **_encoder_kwargs())[0]
    assert isinstance(item, helpers.VideoConditionByKeyframeIndex)
    assert item.frame_idx == 0
    assert item.strength == 1.0
    assert item.num_pixel_frames is None          # a stretch of frames, not one
    assert encode_calls["n"] == 1


def test_the_item_carries_the_jobs_own_strength_unchanged(encode_calls):
    guide = torch.zeros(17, 3, 64, 96, dtype=torch.float32)
    item = helpers.image_conditionings_by_adding_guiding_latent(
        images=[(guide, 0, 0.75)], height=64, width=96, **_encoder_kwargs())[0]
    assert item.strength == 0.75
    assert encode_calls["n"] == 1


def test_no_latent_takes_the_place_of_a_guide_anywhere(encode_calls):
    """The doors that let a latent bypass the encode are gone: a branch reaches
    the conditioning only as pixels, so a stale latent can never be frozen onto
    a window as if it described that window's own frames."""
    assert not hasattr(helpers, "precomputed_guide_latents")
    assert not hasattr(helpers, "prefix_video_latent_conditioning")


# --- the frames come from the decode that builds the kept prefix ------------

def _window_spec(frames=3, channels=4, height=8, width=12):
    """One sidecar window as ``decode_kept_windows`` hands it to the decoder."""
    return {"latents": torch.zeros(channels, frames, height, width, dtype=torch.float16),
            "head_trim": 0, "tail_trim": 0}


def test_a_sidecar_window_decodes_to_frames(decoded):
    """The kept-region decode answers frames, in the channels-first layout the
    assembly tiles and the guide slices, and it keeps the levels the assembly
    writes at: uint8 in, uint8 out - the guide is what turns them into the
    float levels a guide travels in, so the two callers cannot drift apart."""
    frames = ltx2._sidecar_window_decoder(_StubDecoder(), None, lambda: False)(_window_spec())
    assert frames.dim() == 4 and int(frames.shape[0]) == 3
    assert int(frames.shape[1]) == 3 * 8 - 7
    assert frames.dtype == torch.uint8 and frames.device.type == "cpu"


def test_the_sidecar_decode_casts_to_the_decoder_itself(decoded):
    """Sidecar latents are stored in the dtype they were saved in; the decoder
    weights may hold another, and the mismatch is what once made the kept region
    fall back to source pixels in silence."""
    ltx2._sidecar_window_decoder(_StubDecoder(), None, lambda: False)(_window_spec())
    assert decoded["latent"].dtype == torch.float32      # the stub's parameter dtype
    assert decoded["latent"].dim() == 5                  # decoded batched


def test_the_kept_region_is_decoded_once_for_both_uses():
    """The head window's guide and the output's kept prefix are the same frames:
    the guide path decodes them, the assembly takes the cached tensor. Two
    decodes of a kept region at 1088p is work the branch does not need, and one
    decoder - not two near-copies - is what keeps the two uses the same frames."""
    source = (ROOT / "models" / "ltx2" / "ltx2.py").read_text(encoding="utf-8")
    guide_def = source.index("def _kept_region_frames()")
    assembly = source.index('result["decoded_prefix"] = _kept_pixels_cache[0]')
    assert guide_def < assembly                       # the guide decodes first
    guide_block = source[guide_def:source.index("if (window_no == 1 and image_start is None")]
    assert "latent_io.decode_kept_windows" in guide_block
    assert "_sidecar_window_decoder" in guide_block
    assert "_kept_pixels_cache.append" in guide_block
    assembly_block = source[assembly - 700:assembly + 1200]
    before, _, after = assembly_block.partition("else:")
    assert "_kept_pixels_cache[0]" in before            # reused, not re-decoded
    assert "latent_io.decode_kept_windows" in after     # only the fallback decodes
    assert "def _decode_kept_window" not in source     # one shared decoder


def test_the_branch_conditioning_start_frame_reaches_the_ltx2_handler():
    """The branch's conditioning start is one explicit hand-off: wgp sends it,
    and LTX2 slices Control Video/masks to the new segment and starts reference
    conditioning at that same frame rather than at the window's frame 0."""
    import inspect
    signature = inspect.signature(ltx2.LTX2.generate)
    assert "branch_conditioning_start_frame" in signature.parameters
    source = (ROOT / "models" / "ltx2" / "ltx2.py").read_text(encoding="utf-8")
    assert "control_start_frame = branch_start_frame" in source
    assert "input_frames = input_frames[:, :available]" in source
    assert "reference_start_frame = int(branch_start_frame)" in source
    wgp_source = (ROOT / "wgp.py").read_text(encoding="utf-8")
    assert 'latent_gen_kwargs["branch_conditioning_start_frame"]' in wgp_source
