"""Where a reference reaches the model when it is parked partway into a window.

The LTX-2 video VAE writes its first frame with a temporal stride of one instead of
eight, so the positions of every generated frame carry a correction that pulls the
whole timeline back onto that first frame. The window being generated always gets
the correction. A reference - a continuation's kept footage, a control video, an
injected keyframe - used to get it only when it was parked at frame 0, so a guide
handed to the model at frame 48 arrived with the timestamps of frame 55: seven pixel
frames, 292 ms at 24 fps, later than the footage it stands for. A continuation then
has to reconcile its new frames with a reference that is nearly a third of a second
off, and the picture drifts to match.

Upstream pull request 2628 makes the correction a property of the latent grid rather
than of the park position. These tests hold the rule that fixes, in the form a later
change can't satisfy by accident: parking a reference further into the timeline moves
it by exactly that many frames, and by nothing else. Before the fix the same test
fails by 7 frames on every token past the first.

Run with the suite on the RTX 3090:
    CUDA_VISIBLE_DEVICES=1 venv/bin/python -m pytest tests
The checks themselves stay on the card-free side of the code - they only move index
arithmetic - so they run even when the card is busy with a job.
"""

import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from models.ltx2.ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
    from models.ltx2.ltx_core.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex
    from models.ltx2.ltx_core.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
    from models.ltx2.ltx_core.types import LatentState, SpatioTemporalScaleFactors, VideoLatentShape
except Exception:  # the LTX2 core is a heavy optional dependency
    pytest.skip("the LTX2 core package is not importable", allow_module_level=True)

FPS = 24.0
SCALE_FACTORS = SpatioTemporalScaleFactors(time=8, width=32, height=32)
CHANNELS, LATENT_FRAMES, LATENT_H, LATENT_W = 4, 3, 2, 2


def _tools():
    """The four things both conditioning items read off the window's own tools."""
    return types.SimpleNamespace(
        patchifier=VideoLatentPatchifier(patch_size=1),
        scale_factors=SCALE_FACTORS,
        fps=FPS,
        causal_fix=True,
    )


def _reference_latent():
    return torch.zeros(1, CHANNELS, LATENT_FRAMES, LATENT_H, LATENT_W, dtype=torch.float32)


def _empty_state():
    tokens = LATENT_FRAMES * LATENT_H * LATENT_W
    return LatentState(
        latent=torch.zeros(1, 0, CHANNELS),
        denoise_mask=torch.zeros(1, 0, 1),
        positions=torch.zeros(1, 3, 0, 2, dtype=torch.float32),
        clean_latent=torch.zeros(1, 0, CHANNELS),
    )


def _reference_times(make_item, frame_idx):
    """The seconds each reference token claims to cover, as the model reads them."""
    state = make_item(_reference_latent(), frame_idx).apply_to(_empty_state(), _tools())
    return state.positions[0, 0, :, 0]  # per-token start of the temporal span, in seconds


def _make_keyframe_item(latent, frame_idx):
    return VideoConditionByKeyframeIndex(keyframes=latent, frame_idx=frame_idx, strength=1.0)


def _make_reference_item(latent, frame_idx):
    return VideoConditionByReferenceLatent(latent=latent, strength=1.0, frame_idx=frame_idx)


ITEMS = [pytest.param(_make_keyframe_item, id="keyframe"),
         pytest.param(_make_reference_item, id="reference-video")]


@pytest.mark.parametrize("make_item", ITEMS)
def test_parking_a_reference_moves_it_by_exactly_that_many_frames(make_item):
    """A guide parked at frame 48 covers the footage at frame 48, not the footage at 55."""
    at_start = _reference_times(make_item, 0)
    parked = _reference_times(make_item, 48)

    assert torch.allclose(parked, at_start + 48.0 / FPS), (
        f"a reference parked 48 frames in moved by {(parked - at_start) * FPS} frames instead of 48"
    )


def _window_times():
    """The seconds the window's own generated frames claim, the way
    ``VideoLatentTools.create_initial_state`` builds them: the latent grid with the
    causal correction, divided by the frame rate."""
    coords = _tools().patchifier.get_patch_grid_bounds(
        output_shape=VideoLatentShape(batch=1, channels=CHANNELS, frames=LATENT_FRAMES,
                                      height=LATENT_H, width=LATENT_W),
        device=torch.device("cpu"),
    )
    pixel_coords = get_pixel_coords(latent_coords=coords, scale_factors=SCALE_FACTORS, causal_fix=True)
    return pixel_coords.to(dtype=torch.float32)[0, 0, :, 0] / FPS


@pytest.mark.parametrize("make_item", ITEMS)
def test_a_parked_reference_agrees_with_the_frames_it_replaces(make_item):
    """The reference and the window it conditions share one temporal grid."""
    window_times = _window_times()
    parked = _reference_times(make_item, LATENT_FRAMES * SCALE_FACTORS.time)

    assert torch.allclose(parked, window_times + LATENT_FRAMES * SCALE_FACTORS.time / FPS)


@pytest.mark.parametrize("make_item", ITEMS)
def test_a_prepended_reference_carries_the_correction_too(make_item):
    """A guide handed in *before* the window starts is parked with a negative offset.

    Both conditioning items read a negative offset as "prepend": the reference's first
    latent frame is dropped, because the window's own first latent covers it, and the
    rest steps back by one latent stride. That convention is upstream's and is not what
    this file pins - upstream pull request 2628 has a second, separate proposal about
    it. What is pinned here is that the causal correction reaches the surviving tokens
    as well: without it they sit seven pixel frames past where the same reference
    parked at frame 0 sits, shifted by the park distance minus one latent stride.
    """
    prefix = 17
    at_start = _reference_times(make_item, 0)[LATENT_H * LATENT_W:]
    prepended = _reference_times(make_item, -prefix)

    assert torch.allclose(prepended, at_start + (prefix - SCALE_FACTORS.time) / FPS)
    assert prepended.shape == at_start.shape


@pytest.mark.parametrize("make_item", ITEMS)
def test_a_later_token_of_a_parked_reference_names_its_own_frame(make_item):
    """The plain statement of the rule, in milliseconds, for whoever reads this next.

    The first token agrees either way - the causal correction clamps frame 0 to zero -
    so this reads the reference's second latent frame. A causal VAE packs one pixel
    frame into its first latent and eight into each later one, so that frame covers
    pixel frame 1: parked at 48 it must claim 2042 ms. Skipping the correction made it
    claim 2333 ms, the footage four frames on.
    """
    tokens_per_latent_frame = LATENT_H * LATENT_W
    second_latent_frame = float(_reference_times(make_item, 48)[tokens_per_latent_frame])

    assert round(second_latent_frame * 1000) == round((48 + 1) / FPS * 1000)
