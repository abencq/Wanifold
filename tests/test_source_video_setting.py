"""Tests for the Source Video setting as the app reads it.

A task that reaches WanGP outside the interface says "no Source Video" with an empty string, and a media
picker can hand back a list holding that string. Taken for a path, both sent a job looking up the frame rate
of nothing: the read returned 0, and the job divided its frame count by it. The reading now happens once, in
validate_settings, so the window schedule and the job agree on one answer and the person sending the task
gets the message about a missing Source Video.

Run from the repository root:

    venv/bin/python -m pytest tests/test_source_video_setting.py -q

The tests need the app itself (wgp), and the app reads the GPU when it is imported, so they skip
automatically when the app's runtime dependencies or a GPU are not available.
"""
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _import_wgp():
    """Import the app with a clean command-line argument list.

    The app's CLI parser is strict: the test runner's own arguments would be rejected, so they are set
    aside while the app is imported and restored right after.
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
except Exception as exc:  # the app's runtime dependencies, or a GPU, may be missing
    wgp = None
    WGP_IMPORT_ERROR = exc
else:
    WGP_IMPORT_ERROR = None

requires_wgp = pytest.mark.skipif(wgp is None, reason=f"wgp unavailable ({WGP_IMPORT_ERROR!r})")

MISSING_SOURCE = "You must provide a Source Video file to continue"
# Every way a task says it has no Source Video, and every way it names one file.
BLANK_SOURCES = ("", "   ", [], [""], ["", ""])


def _settings(model_type, video_source, image_prompt_type="V"):
    """The settings of a task, filled the way clean_settings fills them."""
    inputs = dict(wgp.get_factory_settings(model_type))
    for key, value in wgp.primary_settings.items():
        inputs.setdefault(key, value)
    inputs["model_type"] = model_type
    inputs["mode"] = ""
    inputs["image_prompt_type"] = image_prompt_type
    inputs["video_source"] = video_source
    return inputs


def _settings_are_valid(model_type, video_source, image_prompt_type="V"):
    """Whether a task carrying only this Source Video is accepted by the app."""
    overrides, _, _, _, _ = wgp.validate_settings(
        {}, model_type, True, _settings(model_type, video_source, image_prompt_type), silent=True)
    return overrides is not None


_VIDEO_MODEL_TYPE = None


def _video_model_type():
    """A model that outputs video and continues one, and asks for nothing else beside it.

    An image model drops the video prompt type and never looks at the Source Video, and models differ on
    what they demand next to it - some want Reference Images - so the answer these tests read comes from the
    first model that accepts a task carrying only a Source Video.
    """
    global _VIDEO_MODEL_TYPE
    if _VIDEO_MODEL_TYPE is None:
        for model_type in wgp.models_def:
            if "V" not in str(wgp.get_model_def(model_type).get("image_prompt_types_allowed", "")):
                continue
            if int(wgp.get_factory_settings(model_type).get("image_mode", 0) or 0) > 0:
                continue
            if _settings_are_valid(model_type, "/videos/run.mp4") and _settings_are_valid(model_type, "", image_prompt_type=""):
                _VIDEO_MODEL_TYPE = model_type
                break
    if _VIDEO_MODEL_TYPE is None:
        pytest.skip("no model accepting a Source Video on its own in this build")
    return _VIDEO_MODEL_TYPE


@requires_wgp
def test_a_blank_source_video_is_read_as_no_file():
    assert wgp.source_video_setting("/videos/run.mp4") == "/videos/run.mp4"
    assert wgp.source_video_setting(["/videos/run.mp4"]) == "/videos/run.mp4"
    assert wgp.source_video_setting(None) is None
    for blank in BLANK_SOURCES:
        assert wgp.source_video_setting(blank) is None, blank


@requires_wgp
def test_validation_asks_for_the_source_video_instead_of_dividing_by_zero():
    model_type = _video_model_type()
    for blank in BLANK_SOURCES:
        inputs = _settings(model_type, blank)
        overrides, _, _, _, error = wgp.validate_settings({}, model_type, True, inputs, silent=True)
        assert overrides is None, f"{blank!r} was accepted as a Source Video"
        assert error == MISSING_SOURCE, f"{blank!r} -> {error!r}"


@requires_wgp
def test_a_task_from_the_api_is_rejected_before_the_job_starts():
    model_type = _video_model_type()
    for blank in BLANK_SOURCES:
        task = {"params": _settings(model_type, blank), "plugin_data": {}}
        validated, error = wgp.validate_task(task, {})
        assert validated is None, f"{blank!r} was accepted for a job"
        assert error == MISSING_SOURCE, f"{blank!r} -> {error!r}"


@requires_wgp
def test_a_source_video_path_still_reaches_the_job():
    model_type = _video_model_type()
    inputs = _settings(model_type, "/videos/run.mp4")
    overrides, _, _, _, error = wgp.validate_settings({}, model_type, True, inputs, silent=True)
    assert overrides is not None, error
    assert overrides["video_source"] == "/videos/run.mp4"

    inputs = _settings(model_type, ["/videos/run.mp4"])
    overrides, _, _, _, error = wgp.validate_settings({}, model_type, True, inputs, silent=True)
    assert overrides is not None, error
    assert overrides["video_source"] == "/videos/run.mp4"


@requires_wgp
def test_a_task_that_continues_nothing_keeps_no_source_video():
    model_type = _video_model_type()
    inputs = _settings(model_type, "", image_prompt_type="")
    overrides, _, _, _, error = wgp.validate_settings({}, model_type, True, inputs, silent=True)
    assert overrides is not None, error
    assert overrides["video_source"] is None
