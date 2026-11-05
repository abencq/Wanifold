# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""The latent save/decode/branch engine, moved out of ``wgp.py`` (the latent
refactor, phase 2).

Everything the app needs to *save latents*, *decode a saved latent*
(the Media Flow *Latent Decode* job) and *branch from a saved latent*: the
model-family detection (which families write the all-latent sidecar), the
branch-point resolution (the keep-frames field to a latent boundary), the
branch-prefix preparation (the kept latents carried over the cut point),
the per-family decode jobs (LTX2, MiniMax H3, Magi Human, Ovi) and the
all-latent sidecar assembly (the ``*_latent.pt`` companion written next to
the media).

A shared module must not import ``wgp``: ``wgp.py``'s module state enters by
injection through :class:`JobContext` (built in ``wgp.py`` once the
settings, the model registry and the job functions are defined).
"""

import io
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable

import gradio as gr
import numpy as np
import torch
from PIL import Image

from mmgp import offload

from shared import config_groups as model_config_groups
from shared import notifications
from shared.api import get_api_output_options, store_api_output_artifact
from shared.attention import get_default_attention_mode
from shared.utils import files_locator as fl
from shared.utils import latent_io
from shared.utils.audio_video import combine_and_concatenate_video_with_audio_tracks, save_hdr_video, save_video, write_wav_file
from shared.utils.utils import convert_image_to_tensor, convert_tensor_to_image, convert_video_tensor_to_uint8_chunked, filter_letters, get_video_info, has_video_file_extension
from shared.utils.virtual_media import media_source_exists


@dataclass(frozen=True)
class JobContext:
    """The ``wgp.py`` module state the latent engine reads (injected).

    The dict/registry fields are shared by reference (the app mutates their
    contents in place); the quantization/verbose fields are read-only once
    ``wgp`` has started; the callables are ``wgp`` module functions (the
    model lookups, the status clear, the generation info and the
    *Continue Last Video* fallback).
    """
    server_config: dict
    save_path: str
    WanGP_version: str
    verbose_level: int
    lock: threading.Lock
    primary_settings: dict
    model_types_handlers: dict
    transformer_quantization: str
    text_encoder_quantization: str
    get_model_def: Callable
    get_model_record: Callable
    get_model_config_groups: Callable
    get_base_model_type: Callable
    get_available_filename: Callable
    clear_status: Callable
    get_gen_info: Callable
    resolve_last_video: Callable


def _prefix_to_frames_space(entry, prefix, frames_space, transform="logc3", device="cpu"):
    """Convert prefix frames into the pixel space of the decoded window so
    the two can be concatenated directly.

    ``frames_space`` is ``"uint8"`` (SDR output), ``"vae"`` (Wan/LongCat HDR:
    VAE range [-1, 1]) or ``"linear"`` (LTX-2 HDR: HDR-linear). The prefix is
    uint8, VAE range or HDR-linear depending on how the original source was
    read (today: the all-latent sidecar's embedded i2v start image, uint8
    PNG bytes).
    """
    if prefix is None:
        return None
    from shared.utils.hdr import hdr_linear_to_vae_range, vae_range_to_hdr_linear
    try:
        if prefix.dtype == torch.float16:
            prefix = prefix.float()
        prefix = prefix.to(device=device)
        prefix_is_linear = bool(entry.get("input_video_is_hdr")) and torch.is_floating_point(prefix)
        if prefix.dtype == torch.uint8:
            if frames_space == "uint8":
                return prefix
            if frames_space == "vae":
                return prefix.float().div_(255.0).sub_(0.5)
            return prefix.float().div_(255.0)
        if prefix_is_linear:
            if frames_space == "linear":
                return prefix
            if frames_space == "vae":
                return hdr_linear_to_vae_range(prefix, transform=transform)
            return prefix.clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)
        # VAE-range float prefix (float SDR source).
        if frames_space == "vae":
            return prefix
        if frames_space == "linear":
            return vae_range_to_hdr_linear(prefix, transform=transform)
        return prefix.add_(1.0).mul_(0.5).clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)
    except Exception as exc:
        print(f"Latent decode: the prefix frames could not be converted to the decoded sample's pixel space: {exc}")
        return None


_LTX2_BASE_TYPES = ("ltx2_19B", "ltx2_22B", "ltx2_25_22B", "ltx2_22B_edit_anything", "ltx2_22B_msr", "joyai_echo")
_H3_BASE_TYPES = ("minimax_h3_fl2va", "minimax_h3_fl2va_pruned", "minimax_h3_ref2va",
                  "minimax_h3_ref2va_pruned", "minimax_h3_tts_ref2va_pruned", "viggle_animate")
# The Wan family's base types (the models.wan handlers' supported types):
# the Wan 2.1 VAE's 8x 16-channel 4/1 layout for every type except the 5B
# class (ti2v_2_2, lucy_edit, kiwi_edit), which loads the Wan 2.2 VAE
# (16x spatial, a different checkpoint).
_WAN_BASE_TYPES = ("multitalk", "infinitetalk", "fantasy", "vace_14B", "vace_14B_2_2", "vace_multitalk_14B",
                   "vace_standin_14B", "vace_lynx_14B", "t2v_1.3B", "standin", "lynx_lite", "lynx", "t2v",
                   "t2v_2_2", "vace_1.3B", "vace_ditto_14B", "phantom_1.3B", "phantom_14B", "recam_1.3B",
                   "animate", "animate2", "alpha", "alpha2", "alpha_lynx", "chrono_edit", "shotplan_t2v",
                   "shotplan_t2v_2_2", "i2v", "i2v_2_2", "i2v_2_2_multitalk", "ti2v_2_2", "lucy_edit",
                   "kiwi_edit", "flf2v_720p", "fun_inp_1.3B", "fun_inp", "mocha", "steadydancer", "wanmove",
                   "scail", "scail2_14B", "scail2_1.3B", "vista4d", "i2v_2_2_svi2pro", "bernini",
                   "bernini_1.3B", "sky_df_1.3", "sky_df_14B")


def _is_ltx2_family(base_model_type, model_def):
    """Whether the latent's model family is LTX-2 (separate video/audio VAEs
    and an optional vocoder; the set mirrors ltx2_handler.family_handler).
    """
    base_model_type = str(base_model_type or "")
    if base_model_type in _LTX2_BASE_TYPES:
        return True
    return str((model_def or {}).get("architecture") or "").startswith("ltx2")


def _is_h3_family(base_model_type, model_def):
    """Whether the latent's model family is MiniMax H3 (separate video/audio
    VAEs, 32 kHz stereo audio, and a piecewise 5/17 latent-to-pixel layout;
    the set mirrors minimax_h3_handler.family_handler).
    """
    base_model_type = str(base_model_type or "")
    if base_model_type in _H3_BASE_TYPES:
        return True
    return str((model_def or {}).get("architecture") or "") in _H3_BASE_TYPES


# The Magi Human family's base types (the models.magi_human handler's
# supported types): the catalog keeps the distill arch as its own
# architecture (the handler's family maps fold it into the base only for
# the model swap), so both arches are listed — the sr1080 sub-models carry
# one of these arches too.
_MAGI_HUMAN_BASE_TYPES = ("magi_human", "magi_human_distill")


def _is_magi_human_family(base_model_type, model_def):
    """Whether the latent's model family is Magi Human (the models.magi_human
    handler: the Turbo VAED video VAE — 48 channels, the 4/1 latent layout,
    16x spatial — and one 64-channel audio latent per pixel frame at
    44.1 kHz; the set mirrors magi_human_handler.family_handler).
    """
    base_model_type = str(base_model_type or "")
    if base_model_type in _MAGI_HUMAN_BASE_TYPES:
        return True
    return str((model_def or {}).get("architecture") or "") in _MAGI_HUMAN_BASE_TYPES


def _is_ovi_family(base_model_type, model_def):
    """Whether the latent's model family is Ovi (the models.wan ovi_handler:
    the Wan 2.2 5B-class VAE for the video — 48 channels, the 4/1 latent
    layout, 16x spatial — and the 16 kHz mmaudio/BigVGAN VAE for the
    soundtrack; the type mirrors ovi_handler.family_handler's supported
    types).
    """
    base_model_type = str(base_model_type or "")
    if base_model_type == "ovi":
        return True
    return str((model_def or {}).get("architecture") or "") == "ovi"


def _is_wan_family(ctx, base_model_type, model_def):
    """Whether the latent's model family is Wan (the models.wan handlers:
    the Wan 2.1 / 2.2 video VAEs, the 4/1 latent layout; the set mirrors
    the handlers' query_supported_types, with the model registry deciding
    for types outside it).
    """
    base_model_type = str(base_model_type or "")
    if base_model_type in _WAN_BASE_TYPES:
        return True
    handler = ctx.model_types_handlers.get(base_model_type)
    if handler is not None and str(getattr(handler, "__module__", "")).startswith("models.wan"):
        return True
    architecture = str((model_def or {}).get("architecture") or "")
    return architecture in _WAN_BASE_TYPES or architecture == "wan"


# The LongCat family's base types (the models.longcat handler's supported
# types): the 1.3B LongCat video / avatar models, all on the Wan 2.1 VAE
# (16 channels, 4/1 temporal, 8x spatial).
_LONGCAT_BASE_TYPES = ("longcat_video", "longcat_avatar", "longcat_avatar_v1_5")


def _is_longcat_family(ctx, base_model_type, model_def):
    """Whether the latent's model family is LongCat (the models.longcat
    handler: the Wan 2.1 VAE, the 4/1 latent layout; the set mirrors
    longcat_handler.family_handler.query_supported_types, with the model
    registry deciding for types outside it).
    """
    base_model_type = str(base_model_type or "")
    if base_model_type in _LONGCAT_BASE_TYPES:
        return True
    handler = ctx.model_types_handlers.get(base_model_type)
    if handler is not None and str(getattr(handler, "__name__", "")).startswith("models.longcat"):
        return True
    return str((model_def or {}).get("architecture") or "") in _LONGCAT_BASE_TYPES


def _is_all_latent_family(ctx, base_model_type, model_def):
    """Whether the model family writes the all-latent sidecar (LTX2, Wan,
    LongCat, MiniMax H3 and Magi Human: the sidecar is the sole content
    source)."""
    return (_is_ltx2_family(base_model_type, model_def)
            or _is_wan_family(ctx, base_model_type, model_def)
            or _is_longcat_family(ctx, base_model_type, model_def)
            or _is_h3_family(base_model_type, model_def)
            or _is_magi_human_family(base_model_type, model_def))


# The *Misc.* tab's *Save Latents* choice (per generation): it overrides
# the *Configuration* option's *Save Latents* for that one job.
SAVE_LATENT_CHOICES = [
    ("Configuration (global default)", 0),
    ("On", 1),
    ("Off", 2),
]


def effective_save_latents(choice, default):
    """The *Save Latents* decision for one generation.

    ``choice`` is the per-generation *Misc.* tab radio (0 follows the
    *Configuration* option, 1 writes the companion file, 2 skips it).
    ``None`` — a setting saved before the choice existed, or a Deepy /
    API job, which has no such control — follows the *Configuration*
    option as well.
    """
    try:
        choice = int(choice)
    except (TypeError, ValueError):
        choice = 0
    return bool(default) if choice <= 0 else choice == 1


# The dtype string a sidecar records for its VAE, mapped to the torch dtype
# the decode jobs use (the app writes "16"/"32" for the VAE precision); the
# server config's vae_precision is the fallback for sidecars that do not
# record the field.
_DTYPE_MAP = {"float16": torch.float16, "16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float, "32": torch.float}

def _h3_window_commit(entry, model_def, window):
    """The pixel frames a window's latents commit to the sidecar's final video.

    For MiniMax H3 the decoded length follows the VAE's piecewise 5/17
    layout (``h3_video_latent_pixel_frames``), the window is truncated to its
    recorded ``h3_target_frames`` and its ``head_trim``/``tail_trim`` are
    dropped; other families fall back to the linear ``latent_stride``/
    ``frame_offset`` layout (``window_pixel_frames``). Returns 0 when the
    window's latents are absent.
    """
    latents = window.get("latents")
    if latents is None or int(latents.shape[1]) <= 0:
        return 0
    if _is_h3_family(str(entry.get("base_model_type") or ""), model_def):
        from shared.utils.latent_io import h3_video_latent_pixel_frames
        decoded = h3_video_latent_pixel_frames(int(latents.shape[1]))
        target = int(window.get("h3_target_frames") or 0)
        if target > 0 and decoded > target:
            decoded = target
    else:
        decoded = latent_io.window_pixel_frames(latents, int(entry.get("latent_stride") or 0), int(entry.get("frame_offset") or 0))
    return max(0, int(decoded) - int(window.get("head_trim") or 0) - int(window.get("tail_trim") or 0))


# ---------------------------------------------------------------------------
# Latent Branch (Phase 0): the *From latent* checkbox makes a *Continue Video*
# / *Continue Last Video* job branch from the source video's saved
# ``*_latent.pt`` sidecar: the branch point (the keep-frames field) resolves
# to a latent boundary of the sidecar (before the latent containing the
# frame by default, after it when *Round branch frame* is checked), the
# source video is truncated at that cut, and the kept latents carry over
# into the job's sidecar. The frame math and the carry-over live in
# shared/utils/latent_io.py.
# ---------------------------------------------------------------------------

KEEP_FRAMES_VIDEO_SOURCE_LABEL = "Truncate Video beyond this number of resampled Frames (empty=Keep All, negative truncates from End)"
BRANCH_FRAMES_VIDEO_SOURCE_LABEL = "Branch after this number of resampled Frames (empty=Branch at the End, negative counts from the End); the latent containing the frame is regenerated, the latents before it are kept unchanged"
BRANCH_FIT_LATENT_LABEL = "Round branch frame up to a latent boundary"
# The pickers' labels switch with the *From latent* mode and with the
# value's *origin*: off (the classic flow) the video is the continuation
# source; on, the two pickers are the same pair seen from either end —
# the latent file is the branch source (the primary input) and the video
# is its companion (a preview of the cut; the branch reads the latent
# file, so it is optional). Whichever picker the app itself filled (the
# companion video of a picked latent file, or the latent file matched to
# a selected video) carries the "Detected..." variant; a value the user
# selected keeps the plain one.
VIDEO_SOURCE_LABEL = "Video to Continue"
COMPANION_VIDEO_LABEL = "Companion video file (used for preview only)"
DETECTED_COMPANION_VIDEO_LABEL = "Detected companion video file (used for preview only)"
# The latent file's picker is the primary input of the *From latent* mode:
# the file names the branch source on its own (its companion video, when
# found, is only a preview aid), so the label says so. The "Detected"
# variant is the mirror of the companion video's: the file the app
# matched to a selected video (the backward direction of the operation).
BRANCH_LATENT_FILE_LABEL = "Select *_latent.pt file"
DETECTED_LATENT_FILE_LABEL = "Detected latent file (matched to the selected video)"
# The *From latent* checkbox's tooltip: which families the mode applies to
# and the one re-compression caveat (the full support table is in
# docs/SETTINGS.md).
BRANCH_LATENT_TOOLTIP = ("Branch from a saved latent file (written by the 'Save Latents' option). "
                         "LTX2 / Wan / LongCat / MiniMax H3 only. Some models keep the pre-cut section "
                         "by re-compression, not bit-identically (see docs/SETTINGS.md). The frame count is "
                         "the new segment added after the kept part, as in the classic continuation; the job "
                         "log prints the planned total before generation.")


def _branch_family_mismatch_error(ctx, entry, model_type, base_model_type):
    """Error message when a sidecar's model family cannot serve a branch job
    (the frozen prefix latents are VAE-specific and never cross families)."""
    sidecar_model_type = str(entry.get("model_type") or "")
    sidecar_base = str(entry.get("base_model_type") or "")
    sidecar_def = ctx.get_model_def(sidecar_model_type) if len(sidecar_model_type) > 0 else None
    job_def = ctx.get_model_def(model_type) if len(str(model_type or "")) > 0 else None
    sidecar_ltx2 = _is_ltx2_family(sidecar_base, sidecar_def)
    job_ltx2 = _is_ltx2_family(base_model_type, job_def)
    sidecar_h3 = _is_h3_family(sidecar_base, sidecar_def)
    job_h3 = _is_h3_family(base_model_type, job_def)
    if sidecar_ltx2 != job_ltx2 or sidecar_h3 != job_h3:
        return f"the latent file was saved by a different model family ({sidecar_model_type or sidecar_base}); a branch job must use the same family as the original job"
    if not sidecar_ltx2 and not sidecar_h3 and len(sidecar_base) > 0 and len(str(base_model_type or "")) > 0 and sidecar_base != str(base_model_type):
        return f"the latent file was saved by model family '{sidecar_base}' but this job runs family '{base_model_type}'"
    return ""


def _branch_latent_search_dirs(ctx):
    """The folders a *From latent* name-based sidecar lookup searches,
    besides the file's own directory: the app's save path, its ``videos``
    subdirectory (kept for configs that use one), and each entry of the
    ``branch_latent_search_dirs`` setting (with its ``videos``
    subdirectory) — for example another instance's save folder, when the
    media was selected from there."""
    search_dirs = [ctx.save_path, os.path.join(ctx.save_path, "videos")]
    for extra in ctx.primary_settings.get("branch_latent_search_dirs") or []:
        extra = os.path.expanduser(str(extra or "").strip())
        if len(extra) == 0:
            continue
        search_dirs += [extra, os.path.join(extra, "videos")]
    return search_dirs


def resolve_latent_branch(ctx, image_prompt_type, video_source, state, model_type, keep_frames="", fit=False, *, base_model_type=None, model_def=None, latent_file=None):
    """Validate and resolve a *From latent* branch job.

    ``fit`` selects the branch-point resolution (see
    :func:`latent_io.branch_point_for_frame`): the mid-latent default
    (unchecked) resolves to the boundary before the latent containing the
    picked frame, the rounded (fit) resolution clamps to the boundary after
    it. ``latent_file`` names the source latent file directly and is the
    primary input: when it is present and readable the branch resolves from
    the file's recorded windows alone, so the companion video (
    ``video_source``) is only a preview / cross-check aid — the branch
    works when the companion video is gone (deleted, moved, on another
    machine). A named file that is no longer on disk (moved, deleted, or
    an auto-detected value persisted from a previous session) is not a
    hard error when a video is present: the classic video-based lookup
    applies instead (the same sidecar, next to the video or by name).
    Without a file the classic flow applies: the sidecar is found next to
    the selected video (or by name across the save directories), and the
    video is then required.

    Returns ``(info, error)``: on success ``info`` carries the latent file
    ``path`` and ``sidecar_obj``, the resolved branch point (see
    :func:`latent_io.branch_point_for_frame`), the ``branch_frame`` the user
    asked for, the effective ``video`` (in *Continue Last Video* the
    auto-resolved last video when the picker is empty),
    ``video_present`` (whether the companion video is resolvable — the
    advisory frame-count cross-check runs only then), the
    ``prefill_prompt``/``prefill_seed`` the form should pick up and the
    identity ``warnings``; on failure ``info`` is None and ``error`` is a
    user-facing message.
    """
    if len(filter_letters(str(image_prompt_type or ""), "VL")) == 0:
        return None, "requires Continue Video / Continue Last Video to be selected"
    video_source = str(video_source or "").strip()
    if len(video_source) == 0 and "L" in str(image_prompt_type or ""):
        # *Continue Last Video* with an empty picker: the app's last video
        # (the same list the form's picker fills from), resolved by the
        # context's resolve_last_video callable (the gallery's last file,
        # then the newest file in the save path).
        video_source = ctx.resolve_last_video(state)
    # The latent file is the primary input: an explicitly named file
    # resolves the branch on its own, so it is looked up before the
    # companion-video gate (a missing video is not an error once the file
    # is resolved). A named file that is no longer on disk (moved,
    # deleted, or an auto-detected value persisted from a previous session)
    # is not a hard error when a video is present: the classic video-based
    # lookup applies (the same sidecar, next to the video or by name).
    explicit_latent = str(latent_file or "").strip()
    sidecar_path = None
    if len(explicit_latent) > 0 and media_source_exists(explicit_latent):
        sidecar_path = latent_io.find_latent_file(explicit_latent)
    if sidecar_path is None:
        if len(explicit_latent) > 0:
            if not media_source_exists(video_source):
                return None, f"the selected latent file could not be found ({os.path.basename(explicit_latent)}) and no source video is present to look its sidecar up"
            print(f"From latent: the latent file {os.path.basename(explicit_latent)} is no longer present; looking up the saved latent file of the selected video instead.", flush=True)
        elif not media_source_exists(video_source):
            return None, "no source video was found for the branch job"
        # No explicit file (or the named one is gone): the classic flow,
        # the sidecar is the video's own companion, found next to it.
        sidecar_path = latent_io.find_latent_file(video_source)
        if sidecar_path is None:
            # The video was selected from outside the save path: its
            # upload temp name was mangled by Gradio (the @ in the fps tag
            # is stripped), so the next-to-file lookup misses. Match the
            # name across the save directories instead.
            sidecar_path = latent_io.find_latent_file_in_dirs(video_source, _branch_latent_search_dirs(ctx))
            if sidecar_path is None:
                search_dirs = _branch_latent_search_dirs(ctx)
                print(f"From latent: no saved latent file matched {os.path.basename(video_source)} in {[os.path.abspath(d) for d in search_dirs if os.path.isdir(d)]}.", flush=True)
                missing_file_clause = f"the selected latent file {os.path.basename(explicit_latent)} is no longer present, and " if len(explicit_latent) > 0 else ""
                return None, (f"{missing_file_clause}no saved latent file was found for {os.path.basename(video_source)}; the original job must have been run with the 'Save Latents' option enabled, "
                              f"and if the video was selected from outside the app save directory its *_latent.pt file can be picked with the 'Latent file' option "
                              f"or its folder added to the 'branch_latent_search_dirs' setting")
    try:
        sidecar_obj = latent_io.load_latent_file(sidecar_path)
        entry = latent_io.latent_payload(sidecar_obj)
    except Exception as exc:
        return None, f"the latent file {os.path.basename(sidecar_path)} could not be read ({exc})"
    if entry is None:
        return None, f"{os.path.basename(sidecar_path)} is not a WanGP latent file"
    # Whether the companion video is resolvable (a virtual-media entry or
    # an existing file): the advisory cross-checks below run only then.
    video_present = media_source_exists(video_source)
    family_error = _branch_family_mismatch_error(ctx, entry, model_type, base_model_type)
    if family_error:
        return None, family_error
    # The branch flow only reads the all-latent layout (every kept frame
    # stored as latents): it carries the sidecar's latents verbatim, so a
    # file in any other layout is refused up front with the one generic
    # message instead of silently dropping a kept region.
    if not latent_io.is_all_latent_entry(entry):
        return None, (latent_io.non_current_latent_message(f"{os.path.basename(sidecar_path)}")
                      # Without a companion video there is no pixel fallback
                      # to suggest (the branch has no video to run from).
                      + (" Run this job without the latent file instead (it will then use the pixels of the source video)." if video_present else ""))
    total_frames = latent_io.sidecar_total_frames(entry)
    if total_frames < 1:
        return None, "the latent file contains no decodable windows"
    # A flagged file (its recorded committed length differs from the written
    # video's length) has bookkeeping that cannot seed a branch cut: the cut
    # is measured against the recorded length, so a mismatched file would
    # place the branch at the wrong frame. Refuse it, naming both lengths.
    _mismatch = latent_io.sidecar_length_mismatch(entry)
    if _mismatch is not None:
        return None, (f"the latent file's recorded length ({_mismatch[0]} frames) does not match the video it was written for "
                      f"({_mismatch[1]} frames); the file was flagged at save time and a branch cut depends on that length, "
                      f"so the branch is refused. Re-run the original job to fix the length, or run this job without the latent file.")
    keep_text = str(keep_frames or "").strip()
    if len(keep_text) == 0:
        branch_frame = total_frames
    else:
        try:
            branch_frame = int(keep_text)
        except ValueError:
            return None, f"the branch point '{keep_text}' is not a valid frame count"
        if branch_frame < 0:
            branch_frame = total_frames + branch_frame
    branch_point = latent_io.branch_point_for_frame(entry, branch_frame, fit=bool(fit))
    if branch_point is None:
        return None, f"the branch point (frame {branch_frame}) is outside the range recorded in the latent file (1..{total_frames})"
    warnings = latent_io.identity_warnings(entry, {"app_version": ctx.WanGP_version, "model_type": "", "base_model_type": "", "vae_sha256": ""})
    if video_present:
        # The latent file may not cover the whole selected video: a picked
        # file can belong to a different job, or record fewer frames than
        # the video has. Either way the branch starts from the latent
        # file's last recorded frame, so flag the mismatch instead of
        # failing the job.
        try:
            video_frame_count = get_video_info(video_source)[3]
        except Exception:
            video_frame_count = None
        if video_frame_count is not None and total_frames > 0 and int(video_frame_count) != int(total_frames):
            _same_job = "; they may not be the same job" if len(explicit_latent) > 0 else ""
            warnings = list(warnings) + [f"the latent file records {int(total_frames)} frames but the selected video has {int(video_frame_count)}, so the branch starts from the last frame recorded in the file{_same_job}"]
    else:
        # No companion video: the file's records are the truth (there is no
        # advisory frame cross-check without a video to check against), so
        # say so.
        warnings = list(warnings) + [f"no companion video was found for the latent file; the branch point is positioned from the {int(total_frames)} frames recorded in the file"]
    info = {
        "path": str(sidecar_path),
        "video": video_source,
        "sidecar_obj": sidecar_obj,
        "entry": entry,
        "branch": branch_point,
        "branch_frame": branch_frame,
        "video_present": video_present,
        "prefill_prompt": latent_io.entry_window_prompt(entry, branch_point["window_no"]),
        "prefill_seed": int(entry.get("seed") or 0),
        "warnings": warnings,
    }
    return info, ""


def validate_latent_branch_task(ctx, state, model_type, inputs):
    """Queue-time validation for a *From latent* branch job.

    Returns an error string to reject the task, or "" when the job is fine
    (the job itself re-resolves the branch point at run time). Outside the
    Continue locations the mode is dormant — the checkbox may sit hidden
    but on, and it neither refuses nor affects such a job (the job
    pipeline is gated on the same condition). The same dormancy applies to
    models whose family does not write latent files: the mode is hidden
    for them (the stored value is preserved), and a job started with it on
    runs as a normal job, never a refusal.
    """
    if not inputs.get("branch_from_latent"):
        return ""
    if int(inputs.get("image_mode", 0) or 0) > 0 or len(filter_letters(str(inputs.get("image_prompt_type") or ""), "VL")) == 0:
        return ""
    if not _is_all_latent_family(ctx, ctx.get_base_model_type(model_type), ctx.get_model_def(model_type)):
        # The family does not write latent files: the mode is dormant
        # (treated as off), never a refusal.
        return ""
    image_prompt_type = str(inputs.get("image_prompt_type") or "")
    _info, error = resolve_latent_branch(ctx, image_prompt_type, inputs.get("video_source"), state, model_type, keep_frames=inputs.get("keep_frames_video_source", ""), fit=bool(inputs.get("branch_fit_latent")), base_model_type=ctx.get_base_model_type(model_type), model_def=ctx.get_model_def(model_type), latent_file=inputs.get("branch_latent_file"))
    if error == "no source video was found for the branch job" \
            and len(str(inputs.get("video_source") or "").strip()) == 0 \
            and len(str(inputs.get("branch_latent_file") or "").strip()) == 0:
        return "From latent: select a source video or a latent file"
    return f"From latent: {error}" if error else ""


# ---------------------------------------------------------------------------
# Latent Branch (Phase 1): the branch job reuses the sidecar's kept latents
# as its head window's frozen prefix instead of re-encoding the source
# pixels: the head window's denoised tensor is [the kept latents (verbatim
# from the sidecar) | new latents], the kept portion stays bit-identical
# through the whole run (the model re-freezes it at every denoise step),
# and the sidecar's kept windows carry over into the branch job's own
# companion file. The model-side substitution is per family (Wan: the i2v
# guide latents or the ti2v source latents; LongCat: the cond latents run).
# The frame math lives in shared/utils/latent_io.branch_head_window.
# ---------------------------------------------------------------------------

_BRANCH_PREFIX_MODEL_TYPES = {
    # The 16-channel 4/1 Wan-VAE families the prefix substitution supports:
    # the 14B i2v class (1.3 / 1.4 / 2.2 share the VAE), the 14B t2v class
    # (1.3B / 14B / 2.2 / shotplan: the standard text-to-video flow, the
    # frozen head enters through the timestep-0 injection) and the ti2v 5B
    # (source-latent path), plus the LongCat family (Wan VAE, 4/1).
    "i2v", "i2v_2_2", "ti2v_2_2",
    "t2v", "t2v_1.3B", "t2v_2_2", "shotplan_t2v", "shotplan_t2v_2_2",
    "longcat_video", "longcat_avatar", "longcat_avatar_v1_5",
    # LTX2 Phase 2: 8/1 causal layout, 32x spatial compression. The latent
    # channel count varies with the VAE version (128 for the 2.3/2.5 VAEs),
    # so the VAE identity is checked through the checkpoint instead.
    "ltx2_19B", "ltx2_22B", "ltx2_25_22B",
    # MiniMax H3 Phase 3: the piecewise 5/17 latent-to-pixel layout (the
    # linear stride/offset check is skipped for it), 16x spatial
    # compression, family-stable 24-channel VAE.
    "minimax_h3_fl2va", "minimax_h3_fl2va_pruned", "minimax_h3_ref2va",
    "minimax_h3_ref2va_pruned", "minimax_h3_tts_ref2va_pruned", "viggle_animate",
}


# The branch-prefix substitution is VAE-specific: the kept latents must
# decode with the job's own VAE, so each family pins the temporal layout,
# the spatial divisor and (when the channel count is family-stable) the
# latent channel count it accepts.
_BRANCH_PREFIX_LAYOUTS = {
    "i2v": (4, 1, 8, 16),
    "i2v_2_2": (4, 1, 8, 16),
    # The 5B class's Wan 2.2 VAE divides space by 16 (not the 14B class's 8)
    # and carries a 48-channel latent (not the 14B class's 16): the two VAE
    # generations are different checkpoints, so the VAE identity check on
    # top decides between them.
    "ti2v_2_2": (4, 1, 16, 48),
    # The t2v class shares the 14B i2v VAE (the Wan 2.1 8x 16-channel 4/1
    # one): same layout, the frozen head enters through the timestep-0
    # injection instead of the i2v guide latents.
    "t2v": (4, 1, 8, 16),
    "t2v_1.3B": (4, 1, 8, 16),
    "t2v_2_2": (4, 1, 8, 16),
    "shotplan_t2v": (4, 1, 8, 16),
    "shotplan_t2v_2_2": (4, 1, 8, 16),
    "longcat_video": (4, 1, 8, 16),
    "longcat_avatar": (4, 1, 8, 16),
    "longcat_avatar_v1_5": (4, 1, 8, 16),
    "ltx2_19B": (8, 1, 32, None),
    "ltx2_22B": (8, 1, 32, None),
    "ltx2_25_22B": (8, 1, 32, None),
    # MiniMax H3 Phase 3: the piecewise 5/17 latent-to-pixel layout cannot
    # be expressed as stride/offset, so (0, 0) is the sentinel that makes
    # _prepare_branch_prefix skip the linear layout check; the canvas
    # rounds up to the 16x grid and the 24 channel count is
    # family-stable (the VAE identity is checked on top of it).
    "minimax_h3_fl2va": (0, 0, 16, 24),
    "minimax_h3_fl2va_pruned": (0, 0, 16, 24),
    "minimax_h3_ref2va": (0, 0, 16, 24),
    "minimax_h3_ref2va_pruned": (0, 0, 16, 24),
    "minimax_h3_tts_ref2va_pruned": (0, 0, 16, 24),
    "viggle_animate": (0, 0, 16, 24),
    # Magi Human Phase 3: the Turbo VAED divides time by 4 (4/1) and space
    # by 16, with a 48-channel latent - the same linear layout the
    # Wan 2.2 5B class uses (the VAE identity check on top keeps the
    # checkpoints apart).
    "magi_human": (4, 1, 16, 48),
    "magi_human_distill": (4, 1, 16, 48),
    # Ovi (the Wan 2.2 5B-class audio-video model): the same 5B VAE the
    # ti2v_2_2 class uses (4/1, 16x spatial, 48-channel latent); the
    # 16 kHz mono soundtrack rides the sidecar's audio entry.
    "ovi": (4, 1, 16, 48),
}


def _branch_prefix_advisory(ctx, model_type):
    """The one-line advisory for the branch info box: the job's family
    writes the all-latent sidecar, but its base type is not in
    ``_BRANCH_PREFIX_LAYOUTS``, so the kept latents cannot be copied
    bit-identically — the pre-cut section is kept by re-compression (a
    fresh encode of a fresh decode of the source file's kept latents).
    Returns "" when the prefix is bit-identical (or the family does not
    write latent files at all).
    """
    base_model_type = ctx.get_base_model_type(model_type)
    if _is_all_latent_family(ctx, base_model_type, ctx.get_model_def(model_type)) \
            and str(base_model_type or "") not in _BRANCH_PREFIX_LAYOUTS:
        return "Note: this model keeps the pre-cut section by re-compression, not bit-identically."
    return ""


def _runtime_model_def(ctx, model_type, model_def, config_id):
    """The model definition as the job's model load resolves it (mirrors
    ``load_models``): the raw definition with the selected config-group
    values merged in, then the family's ``resolve_runtime_model_def``
    applied (the *Auto* choices resolve from the runtime settings — the H3
    Video VAE follows the transformer quantization). The branch VAE
    identity checks must use this form: the sidecar recorded the VAE the
    original job actually loaded, and comparing it against the raw
    definition's default would refuse the branch on a VAE both runs use.
    """
    resolved = dict(model_def) if isinstance(model_def, dict) else {}
    if config_id is not None and len(str(config_id)):
        config_groups = ctx.get_model_config_groups(model_type, resolved)
        for _, _, current_config in model_config_groups.selected_model_configs(config_groups, config_id):
            resolved.update(current_config)
    base_model_type = ctx.get_base_model_type(model_type)
    handler = ctx.model_types_handlers.get(base_model_type) if base_model_type is not None else None
    resolver = getattr(handler, "resolve_runtime_model_def", None) if handler is not None else None
    if resolver is not None:
        resolved = resolver(resolved, dict(ctx.server_config, transformer_quantization=ctx.transformer_quantization,
                                           text_encoder_quantization=ctx.text_encoder_quantization,
                                           mixed_precision=ctx.server_config.get("mixed_precision", "0"),
                                           vae_precision=ctx.server_config.get("vae_precision", "16")))
    return resolved


def _ltx2_video_vae_identity(model_def, base_model_type):
    """The job's LTX2 video VAE identity: (checkpoint filename, sha256).

    The sha is empty when the checkpoint cannot be located (the fit check
    then falls back to the filename comparison); the filename comes from the
    model definition's override or the architecture spec.
    """
    from models.ltx2 import ltx2_handler
    names = ltx2_handler._get_multi_file_names(model_def, base_model_type) or {}
    vae_file = str(names.get("video_vae") or "")
    vae_sha = ""
    if vae_file:
        path = fl.locate_file(vae_file, error_if_none=False)
        if path is not None and os.path.isfile(str(path)):
            try:
                vae_sha = latent_io.sha256_file(str(path))
            except Exception:
                vae_sha = ""
    return vae_file, vae_sha


def _h3_video_vae_identity(model_def, base_model_type, vae_upsampling=None):
    """The job's MiniMax H3 video VAE identity: (checkpoint basename, sha256).

    The file is the one this job's pipeline decodes with: the model
    definition's Video VAE (the caller passes the runtime-resolved
    definition, so the *Auto* choice already follows the transformer
    quantization), or the X2 decoder's ``x2_vae_file`` when the X2 VAE
    upscaling replaces it — the same file the pipeline records in its
    latent companion. The sha is empty when the checkpoint cannot be
    located (the fit check then falls back to the filename comparison);
    the returned name is the basename the sidecar records (the int8 and
    X2 checkpoints live in a subfolder the raw setting carries).
    """
    from models.minimax_h3.minimax_h3_main import VIDEO_VAE_FILE
    from models.minimax_h3.vae_upsampler import X2_VAE_FILE, X2_VAE_VALUE
    model_def = model_def or {}
    if vae_upsampling == X2_VAE_VALUE:
        # the X2 decoder of the selected Video VAE replaces it for the decode
        vae_file = str(model_def.get("x2_vae_file") or X2_VAE_FILE)
    else:
        vae_file = str(model_def.get("video_vae_file") or VIDEO_VAE_FILE)
    vae_sha = ""
    path = fl.locate_file(vae_file, error_if_none=False)
    if path is not None and os.path.isfile(str(path)):
        try:
            vae_sha = latent_io.sha256_file(str(path))
        except Exception:
            vae_sha = ""
    return os.path.basename(vae_file), vae_sha


def _wan_video_vae_identity(model_def, base_model_type):
    """The job's Wan video VAE identity: (checkpoint filename, sha256).

    Mirrors models/wan/any2video.py's VAE selection: a model definition
    VAE_URLs override wins, the 5B class loads the Wan 2.2 VAE (16x
    spatial), the alpha variants their RGB-channel checkpoints, everything
    else the Wan 2.1 VAE. The sha is empty when the checkpoint cannot be
    located (the fit check then falls back to the filename comparison).
    """
    vae_urls = (model_def or {}).get("VAE_URLs", None)
    if isinstance(vae_urls, str) and len(vae_urls) > 0:
        vae_file = vae_urls
    elif isinstance(vae_urls, list) and len(vae_urls) > 0:
        vae_urls = [u for u in vae_urls if str(u).endswith((".pt", ".pth", ".safetensors"))]
        vae_file = vae_urls[0] if len(vae_urls) > 0 else ""
    elif (model_def or {}).get("wan_5B_class", False):
        vae_file = "Wan2.2_VAE.safetensors"
    elif (model_def or {}).get("alpha_class", False):
        vae_file = "wan_alpha_2.1_vae_rgb_channel_v2.safetensors" if str(base_model_type or "") == "alpha2" else "wan_alpha_2.1_vae_rgb_channel.safetensors"
    else:
        vae_file = "Wan2.1_VAE.safetensors"
    vae_file = os.path.basename(str(vae_file or ""))
    vae_sha = ""
    path = fl.locate_file(vae_file, error_if_none=False)
    if path is not None and os.path.isfile(str(path)):
        try:
            vae_sha = latent_io.sha256_file(str(path))
        except Exception:
            vae_sha = ""
    return vae_file, vae_sha


def _ovi_video_vae_identity(model_def, base_model_type):
    """The job's Ovi video VAE identity: (checkpoint filename, sha256).

    Mirrors models/wan/ovi_fusion_engine.py's fixed selection: the Ovi
    engine always loads the Wan 2.2 5B-class VAE from its single fixed
    checkpoint (there is no model-definition override), so this returns
    that filename for the branch fit check's identity comparison. The sha
    is empty when the checkpoint cannot be located (the fit check then
    falls back to the filename comparison).
    """
    vae_file = "Wan2.2_VAE.safetensors"
    vae_sha = ""
    path = fl.locate_file(vae_file, error_if_none=False)
    if path is not None and os.path.isfile(str(path)):
        try:
            vae_sha = latent_io.sha256_file(str(path))
        except Exception:
            vae_sha = ""
    return vae_file, vae_sha


def _longcat_video_vae_identity(model_def, base_model_type):
    """The job's LongCat video VAE identity: (checkpoint filename, sha256).

    Mirrors models/longcat/longcat_main.py's VAE selection: a model
    definition ``vae_URL`` override wins, otherwise the first available
    of the Wan 2.1 VAE family (the LongCat VAE is the Wan 2.1 VAE: 16
    channels, 4/1 temporal, 8x spatial, a different checkpoint than the
    2.2 one the 5B Wan class loads). The sha is empty when the
    checkpoint cannot be located (the fit check then falls back to the
    filename comparison).
    """
    candidates = []
    override = str((model_def or {}).get("vae_URL") or "")
    if len(override) > 0:
        candidates.append(os.path.basename(override))
    candidates += ["Wan2.1_VAE_bf16.safetensors", "Wan2.1_VAE.safetensors",
                  "longcat_vae_bf16.safetensors"]
    vae_file = ""
    for candidate in candidates:
        path = fl.locate_file(candidate, error_if_none=False)
        if path is not None and os.path.isfile(str(path)):
            vae_file = candidate
            break
    vae_sha = ""
    if len(vae_file) > 0:
        path = fl.locate_file(vae_file, error_if_none=False)
        if path is not None and os.path.isfile(str(path)):
            try:
                vae_sha = latent_io.sha256_file(str(path))
            except Exception:
                vae_sha = ""
    return vae_file, vae_sha


def _magi_video_vae_identity(model_def, base_model_type):
    """The job's Magi Human video VAE identity: (relative path, sha256).

    Mirrors models/magi_human/magi_human_model.py's selection: the Turbo
    VAED checkpoint is fixed by the model (there is no model-definition
    VAE override), and the sidecar records its relative path (the
    ``turbo_vae/`` subfolder included), so this returns the same relative
    path for the fit check's filename comparison. The sha is empty when
    the checkpoint cannot be located (the fit check then falls back to
    the filename comparison).
    """
    vae_file = "turbo_vae/TurboV3-Wan22-TinyShallow_7_7.safetensors"
    vae_sha = ""
    path = fl.locate_file(vae_file, error_if_none=False)
    if path is not None and os.path.isfile(str(path)):
        try:
            vae_sha = latent_io.sha256_file(str(path))
        except Exception:
            vae_sha = ""
    return vae_file, vae_sha


def _prepare_branch_prefix(entry, branch_point, carried, *, base_model_type, video_length, height=None, width=None, fit_canvas=None, audio_entry=None, vae_file="", vae_sha256="", model_def=None):
    """Phase 1/2/3 fit check + head-window plan for a sidecar-sourced prefix.

    ``carried`` is the result of ``latent_io.carry_over_latent_windows`` for
    the same sidecar / branch point. Returns ``(prefix, error)``: ``prefix``
    is ``None`` unless the job model can take the sidecar's kept latents
    verbatim (the family's layout and spatial size, a matching VAE identity,
    and a head window that fits the kept run); then it carries the kept
    latents, the head window geometry and the sidecar bookkeeping the new
    job's companion file needs. The error string names the Phase 0 fallback
    the job keeps when the fit fails.

    For MiniMax H3 (Phase 3) the head window geometry comes from the
    sidecar-only committed-totals plan (:func:`latent_io.h3_branch_head_plan`),
    not the shared ``branch_head_window`` helper: every length routes
    through the decode job's own arithmetic (the piecewise 5/17 decode capped
    by the recorded target, the committed ranges), so the plan and the
    decode cannot drift, and the plan is the source of the overlap /
    guide / anchor values the job and the new sidecar use. The other
    families keep the ``branch_head_window`` arithmetic.

    ``audio_entry`` is the sidecar's ``wgp_latent/audio/1`` entry: for LTX2
    jobs (Phase 2) the kept window's audio latents are additionally sliced
    by time ratio and carried as ``prefix_audio_latents`` (a batched
    ``[1, c, t, mel]`` tensor), so the head window's audio prefix does not
    need a waveform re-encode; for MiniMax H3 jobs (Phase 3) the flat
    ``[64, t]`` block is sliced the same way and carried as a
    ``{"batched": [1, 32, 2, N], "flat": [64, N]}`` dict (Case A freezes the
    batched prefix, Case B feeds the flat rows as the audio condition).
    ``vae_file``/``vae_sha256`` are the job's LTX2, Wan, LongCat or
    MiniMax H3 video VAE identity (see
    :func:`_ltx2_video_vae_identity`, :func:`_wan_video_vae_identity`,
    :func:`_longcat_video_vae_identity` and
    :func:`_h3_video_vae_identity`).
    """
    layout = _BRANCH_PREFIX_LAYOUTS.get(str(base_model_type or ""))
    if layout is None:
        return None, f"model family '{base_model_type}' does not (yet) accept prefix latents carried over from the source latent file"
    expected_stride, expected_offset, spatial_divisor, expected_channels = layout
    # Phase 3 (MiniMax H3): (0, 0) is the sentinel for the piecewise 5/17
    # latent-to-pixel layout, which the linear stride/offset check below
    # cannot express (the sidecar carries per-window frame metadata
    # instead); the channel check and the VAE identity check still apply.
    is_h3 = expected_stride == 0
    is_ltx2 = str(base_model_type or "").startswith("ltx2")
    specs = latent_io.entry_window_specs(entry)
    try:
        k = latent_io.branch_window_position(entry, branch_point, specs)
    except ValueError:
        return None, "the source latent file has no readable window for the branch point"
    if not 1 <= k <= len(specs) or carried is None:
        return None, "the source latent file has no readable window for the branch point"
    spec = specs[k - 1]
    stride = int(entry.get("latent_stride") or 4)
    offset = int(entry.get("frame_offset") or 1)
    if not is_h3 and (stride, offset) != (expected_stride, expected_offset):
        return None, (f"the {stride}/{offset} temporal layout recorded in the source latent file does not match the "
                      f"{expected_stride}/{expected_offset} layout the '{base_model_type}' prefix substitution supports")
    prefix_latents = carried["video_prefix"]["latents"]
    if expected_channels is not None:
        if int(prefix_latents.shape[0]) != expected_channels:
            return None, f"the latent channel count recorded in the source latent file ({int(prefix_latents.shape[0])}) does not match the VAE of this job ({expected_channels})"
    if is_h3 or expected_channels is None or (vae_file or vae_sha256):
        # The checkpoint identity decides whether the kept latents belong
        # to this job's VAE (the frozen prefix would decode with the wrong
        # VAE otherwise): the LTX2 VAE's channel count varies with the
        # version, the H3 fp16 and fp8-mixed video VAEs are different
        # checkpoints behind the same 24-channel shape, and the Wan 2.1
        # and 2.2 VAEs are different checkpoints behind an overlapping
        # 16-channel 4/1 shape. The sha comparison wins; the filenames
        # stand in when neither side recorded / has one.
        sidecar_sha = str(entry.get("vae_sha256") or "")
        sidecar_vae_file = str(entry.get("vae_file") or "")
        if str(vae_sha256 or "") and sidecar_sha and sidecar_sha != str(vae_sha256):
            return None, "the video VAE checkpoint of the source latent file does not match the video VAE of this job; the frozen prefix would decode with the wrong VAE"
        if not (str(vae_sha256 or "") and sidecar_sha) and str(vae_file or "") and sidecar_vae_file and sidecar_vae_file != str(vae_file):
            return None, "the video VAE checkpoint of the source latent file does not match the video VAE of this job; the frozen prefix would decode with the wrong VAE"
    # The sizing mode decides whether the job's output canvas is the
    # entered one: Exact output (2) keeps it (so does a direct None, the
    # no-size-check caller); the pixel-budget (0) and max-box (1) modes
    # resize it, so the size check applies only to the stable ones.
    if fit_canvas in (None, 2) and height is not None and width is not None:
        # H3 rounds its canvas up to the 16x grid; the other families sit
        # on the grid, so plain division applies to them.
        _latent_h = (int(height) + spatial_divisor - 1) // spatial_divisor if is_h3 else int(height) // spatial_divisor
        _latent_w = (int(width) + spatial_divisor - 1) // spatial_divisor if is_h3 else int(width) // spatial_divisor
        if int(prefix_latents.shape[2]) != _latent_h or int(prefix_latents.shape[3]) != _latent_w:
            return None, (f"the latent size recorded in the source latent file ({int(prefix_latents.shape[2])}x{int(prefix_latents.shape[3])}) does not match the canvas of this job "
                          f"({_latent_h}x{_latent_w})")
    if is_h3:
        # Phase 3 (MiniMax H3): the sidecar-only committed-totals plan
        # (see the function docstring). It rejects points the shared
        # helper would have planned silently (a cut claimed outside the
        # containing window's committed range, a kept run off the
        # 2+5k grid), and the Phase 0 fresh-encode fallback takes over
        # for them.
        try:
            head = latent_io.h3_branch_head_plan(entry, branch_point, int(video_length or 0))
        except ValueError as err:
            return None, str(err)
    else:
        head = latent_io.branch_head_window(entry, branch_point, int(video_length or 0), stride=stride, offset=offset)
    kept = int(head["prefix_latents"])
    if is_h3 and head["cut_at_window_end"] and int(spec["latents"].shape[1]) != int(prefix_latents.shape[1]):
        # Phase 3 (MiniMax H3): the committed end can sit inside the
        # latent run (a h3_target_frames cap trims the window's tail
        # share), and the Case B substitution freezes the window's full
        # latent run, which decodes to the capped length — not the
        # shorter kept-prefix slice the branch point records.
        prefix_latents = spec["latents"]
        kept = int(spec["latents"].shape[1])
    if kept < 1 or kept != int(prefix_latents.shape[1]):
        return None, "the branch point keeps no latents"
    head_trim_k = int(spec.get("head_trim") or 0)
    tail_trim_k = int(spec.get("tail_trim") or 0)
    if head["cut_at_window_end"]:
        # The whole containing window is kept and the new portion is a fresh
        # standard window. That is clean only when the window commits its
        # full decoded run minus at most a one-frame head share (a larger
        # head share or any tail trim would land uncommitted frames inside
        # the new segment, or drop committed ones from the decode).
        # (For H3 the plan's ``clean_window_end`` is the same test over the
        # same recorded trims.)
        if is_h3 and not head["clean_window_end"] or not is_h3 and (tail_trim_k > 0 or head_trim_k > 1):
            return None, f"the branch point ends a window trimmed by {head_trim_k} head / {tail_trim_k} tail frame(s); the prefix substitution only supports clean window ends"
        decoded_window = int(head["decoded_window"])
        if is_h3:
            # The plan's Case B values: the overlap and the guide are the
            # window's committed kept region (the fresh standard window
            # re-decodes it as its head, and the one-shot decode slices
            # it as the continuation). For a t2v window that equals the
            # decoded run minus the head share the v2 arithmetic used;
            # the i2v image slot counts in the committed region (one
            # more frame).
            assembly_overlap = int(head["assembly_overlap"])
            guide_frames = int(head["guide_frames"])
        else:
            # The window starts at the sidecar window's own start frame (the
            # frozen latents decode at their original absolute positions, no
            # shift): the decoded head that duplicates what already precedes it
            # is the head share (the previous window's tail, or the re-extracted
            # source prefix for the sidecar's own window 1), so the overlap
            # is the decoded run minus that share.
            assembly_overlap = max(1, decoded_window - head_trim_k)
            guide_frames = decoded_window
        head_frame_num = None
        head_output = None
        head_trim_new = 0
    else:
        # The cut falls inside the window: the new latents complete it up to
        # its committed end (clamped to the requested length, the overshoot
        # absorbed as a tail trim so the output ends on the boundary).
        if is_h3:
            # The plan's Case A values: the overlap and the guide are the
            # kept latents' decoded length (the model's conditioning
            # region, capped at the committed end); the head window keeps
            # the containing window's own geometry (its recorded frame
            # count, the request clamped to the cut's committed tail, the
            # overshoot absorbed as the new block's tail trim).
            assembly_overlap = int(head["assembly_overlap"])
            guide_frames = int(head["guide_frames"])
            head_frame_num = int(head["head_frame_num"])
            head_output = int(head["head_output"])
            head_trim_new = int(head["head_trim_new"])
        else:
            decoded_prefix = int(head["decoded_prefix"])
            assembly_overlap = max(1, decoded_prefix)
            guide_frames = decoded_prefix
            head_frame_num = int(head["frame_num"])
            head_output = int(head["new_frames"])
            head_trim_new = int(head["tail_trim_new"])
    # Phase 2 (LTX2): the head window's audio prefix is the sidecar's kept
    # audio latents sliced by time ratio, so the branch job does not
    # re-encode the source waveform for it.
    # Phase 3 (MiniMax H3): the same time-ratio splice on the flat
    # [64, t] block, carried as {"batched": [1, 32, 2, N] frozen prefix,
    # "flat": [64, N] condition rows}.
    # Phase 3 (Magi Human): the same time-ratio splice on the flat
    # [64, t] block (one 64-channel latent per pixel frame), carried in
    # the sidecar's flat layout for the model to freeze at the head of
    # the head window's audio tensor.
    # Ovi: the same time-ratio splice on the flat [20, t] block (the
    # mmaudio VAE's 20-channel 16 kHz mono latent), carried in the
    # sidecar's flat layout for the model to freeze at the head of the
    # head window's audio tensor.
    prefix_audio = None
    if str(base_model_type or "").startswith("ltx2") and audio_entry is not None:
        prefix_audio = latent_io.branch_audio_prefix_latents(audio_entry, entry, branch_point)
    elif is_h3 and audio_entry is not None:
        prefix_audio = latent_io.h3_branch_audio_prefix_latents(audio_entry, entry, branch_point)
    elif _is_magi_human_family(base_model_type, model_def) and audio_entry is not None:
        prefix_audio = latent_io.magi_branch_audio_prefix_latents(audio_entry, entry, branch_point)
    elif _is_ovi_family(base_model_type, model_def) and audio_entry is not None:
        prefix_audio = latent_io.ovi_branch_audio_prefix_latents(audio_entry, entry, branch_point)
    # The all-latent sidecar (LTX2 / Wan / LongCat / H3) carries two
    # head-side values in the prefix, each named for what it is.
    # ``containing_window_head_trim`` is the containing window's own
    # recorded head trim (0 for an all-latent t2v window 1, the i2v image
    # slot for an i2v window 1, the window's own overlap for k > 1) — the
    # shared field every family's writer records for a mid-window cut
    # (H3's Case B anchor head overrides it with the one-frame
    # shared-frame trim; the writer applies it to a frozen-head branch).
    # ``assembly_overlap`` (above) is the prefix of the head window's
    # decoded output that the assembly drops for a cut on a window
    # boundary: the non-H3 writer records it as the branch head block's
    # head trim (so the decode reassembles the written video), while H3
    # never consumes it (its one-shot slice is positional, derived from
    # the plan's committed totals).
    containing_window_head_trim = int(head_trim_k)
    # The LTX2 i2v start image is content, not a pointer: it
    # propagates verbatim through every branch so a descendant's
    # frame 0 stays the exact image (the new sidecar's decode splices
    # it back in; the branch job itself never needs the original
    # image file). A Wan / LongCat / H3 i2v sidecar embeds no image
    # (its frame 0 is a decoded latent), so there is nothing to
    # propagate for them.
    carried_image = carried.get("i2v_image") if is_ltx2 and isinstance(carried, dict) else None
    if not (isinstance(carried_image, dict)
            and str(carried_image.get("kind")) == "image_png"
            and isinstance(carried_image.get("data"), (bytes, bytearray))
            and len(carried_image["data"]) > 0):
        carried_image = None
    else:
        carried_image = {"kind": "image_png", "data": bytes(carried_image["data"])}
    prefix = {
        "prefix_latents": prefix_latents,
        "prefix_audio_latents": prefix_audio,
        "assembly_overlap": int(assembly_overlap),
        "guide_frames": int(guide_frames),
        "head_frame_num": head_frame_num,
        "head_output": head_output,
        "head_trim_new": int(head_trim_new),
        "cut_at_window_end": bool(head["cut_at_window_end"]),
        "kept_frames": int(branch_point.get("kept_frames") or 0),
        "sidecar_window_no": int(k),
        "sidecar_start_frame": int(spec.get("start_frame") or 0),
        "containing_window_head_trim": int(containing_window_head_trim),
        # Phase 3 (MiniMax H3): the plan's GAP-3 marker — the cut sits at
        # a capped window's committed end, so the fresh branch window's
        # first decoded frame re-anchors the carried window's last frame
        # (False for every other family).
        "anchor_head": bool(head.get("anchor_head") or False),
        "carried_image": carried_image,
    }
    return prefix, ""


def _branch_kept_window_specs(source_entry, branch_point):
    """The source sidecar's kept windows (1..K, the branch cut's containing
    window included) as decode-ready specs, plus the cut and the i2v flag —
    the inputs to the branch job's kept-region decode (Step 2, every
    latent-writing family).

    ``source_entry`` is the branch source's ``wgp_latent/1`` entry and
    ``branch_point`` the resolved branch point
    (:func:`latent_io.branch_point_for_frame`: it carries the containing
    window number and the resolved cut). Returns a tuple ``(specs,
    kept_frames, is_i2v)`` where ``specs`` are the first K of the entry's
    window specs (each with its full latents, the decode job's own
    per-window bookkeeping intact) and ``is_i2v`` mirrors the Latent Decode
    job's i2v test (the decode's window-1 image-slot handling follows it).
    ``(None, 0, False)`` when the kept windows cannot be resolved (the job
    keeps the source-pixel path).
    """
    if not isinstance(branch_point, dict) or not isinstance(source_entry, dict):
        return None, 0, False
    specs = latent_io.entry_window_specs(source_entry)
    k = int(branch_point.get("window_no") or 0)
    if not 1 <= k <= len(specs):
        return None, 0, False
    kept_frames = int(branch_point.get("kept_frames") or 0)
    if kept_frames < 1:
        return None, 0, False
    # The same i2v test the Latent Decode job applies (the window-1 head
    # trim exemption follows it).
    is_i2v = bool(source_entry.get("image_source")) and len(str(source_entry.get("video_source") or "")) == 0
    return specs[:k], kept_frames, is_i2v


def _branch_kept_decode_compatible(entry, kept_specs, base_model_type, model_def, *, height, width, fit_canvas, vae_file="", vae_sha256=""):
    """Whether the job's VAE can decode the sidecar's kept windows onto the
    job's canvas — the canvas-compatibility test for the kept-region decode
    (Step 2). It is the fit check's VAE-identity and latent-size checks
    minus the per-family prefix plan: the decode only needs the kept
    latents to belong to the job's VAE and to land on the job's latent
    grid, not the head window's willingness to freeze them, so it decides
    the decode even when the branch prefix itself is rejected. A job that
    fits the canvas (``fit_canvas`` set) would resize the decode off its
    grid, so it is never compatible.

    Returns ``(True, "")`` when compatible, else ``(False, reason)`` with
    a plain sentence naming the setting at fault (the refusal message,
    Gate 1, is built from it).
    """
    layout = _BRANCH_PREFIX_LAYOUTS.get(str(base_model_type or ""))
    if layout is None:
        return False, f"model family '{base_model_type}' cannot decode the kept windows of the latent file"
    stride, _offset, spatial_divisor, _channels = layout
    if not spatial_divisor:
        return False, f"model family '{base_model_type}' cannot decode the kept windows of the latent file"
    if not kept_specs or height is None or width is None:
        return False, "the kept windows could not be resolved from the latent file"
    # The sizing mode decides whether the job's output canvas is the
    # entered one: Exact output (2) keeps it (so does a direct None, the
    # no-size-check caller); the pixel-budget (0) and max-box (1) modes
    # resize it, which would take the kept latents off the job's grid.
    if fit_canvas not in (None, 2):
        return False, "this job resizes the canvas (Fit Canvas), so the kept latents would decode off the canvas grid of this job"
    # The checkpoint identity decides whether the kept latents belong to
    # this job's VAE (the decode would run with the wrong VAE otherwise):
    # the sha comparison wins, the filenames stand in when neither side
    # has one (the same tests the fit check applies to the prefix).
    sidecar_sha = str(entry.get("vae_sha256") or "")
    sidecar_vae_file = str(entry.get("vae_file") or "")
    if str(vae_sha256 or "") and sidecar_sha and sidecar_sha != str(vae_sha256):
        return False, "the video VAE checkpoint of the source latent file does not match the video VAE of this job"
    if not (str(vae_sha256 or "") and sidecar_sha) and str(vae_file or "") and sidecar_vae_file and sidecar_vae_file != str(vae_file):
        return False, "the video VAE checkpoint of the source latent file does not match the video VAE of this job"
    # Every kept window sits on the sidecar's own grid; the first kept
    # window's latent size is that grid. H3 rounds its canvas up to the
    # 16x grid; the other families sit on the grid, so plain division
    # applies to them (the same expressions the fit check uses).
    _latent_h = (int(height) + spatial_divisor - 1) // spatial_divisor if stride == 0 else int(height) // spatial_divisor
    _latent_w = (int(width) + spatial_divisor - 1) // spatial_divisor if stride == 0 else int(width) // spatial_divisor
    lat = kept_specs[0]["latents"]
    if int(lat.shape[-2]) != _latent_h or int(lat.shape[-1]) != _latent_w:
        return False, (f"the resolution recorded in the source latent file does not match the canvas of this job "
                       f"(the latents are {int(lat.shape[-2])}x{int(lat.shape[-1])}, the grid of this job is {_latent_h}x{_latent_w})")
    return True, ""


def _build_cumulative_audio_payload(audio_entries):
    """Combine the per-window audio payloads of a multi-window job into one
    ``wgp_latent/audio/1`` entry with all windows stacked on time (the same
    layout a single-window job records, just with more windows), so the
    decode job can reassemble the full audio track.
    """
    first = audio_entries[0]
    latents = []
    window_metas = []
    for audio_entry in audio_entries:
        entry_latents = latent_io.payload_latents(audio_entry)
        entry_windows = audio_entry.get("windows") or []
        if entry_windows:
            for window in entry_windows:
                start = int(window.get("latent_start") or 0)
                length = int(window.get("latent_len") or 0)
                latents.append(entry_latents[:, start:start + length].clone())
                window_metas.append({k: window[k] for k in ("window_no", "start_sample", "samples") if k in window})
        else:
            latents.append(entry_latents)
            window_metas.append({})
    if not latents:
        return None
    total_samples = sum(int(m.get("samples") or 0) for m in window_metas)
    if total_samples <= 0:
        total_samples = int(first.get("samples") or 0)
    return latent_io.build_audio_latent_payload(
        latents,
        window_metas,
        app_version=str(first.get("app_version") or ""),
        model_type=str(first.get("model_type") or ""),
        base_model_type=str(first.get("base_model_type") or ""),
        model_filename=str(first.get("model_filename") or ""),
        loras=first.get("loras") or (),
        audio_vae_file=str(first.get("audio_vae_file") or ""),
        audio_vae_sha256=str(first.get("audio_vae_sha256") or ""),
        audio_vae_dtype=str(first.get("audio_vae_dtype") or ""),
        sample_rate=int(first.get("sample_rate") or 0),
        channels=int(first.get("channels") or 0),
        samples=total_samples,
        seed=int(first.get("seed") or 0),
        prompt=str(first.get("prompt") or ""),
        denoising_strength=float(first.get("denoising_strength") or 1.0),
        steps=int(first.get("steps") or 0),
        total_windows=len(latents),
        vocoder_file=str(first.get("vocoder_file") or ""),
        mel_bins=int(first.get("mel_bins") or 0),
    )


def _all_latent_sidecar(ctx, latent_entry, latent_audio_entry, window_no, latent_windows, latent_audio_windows, *,
                        branch_prefix, sliding_window, reuse_frames, post_decode_pre_trim, trim_first_frames,
                        source_video_overlap_frames_count, discard_last_frames, automatic_trim_last_frames,
                        window_prefix_is_image, fake_start_image, sample_is_hdr, image_start_tensor,
                        written_frame_count, model_def,
                        model_type="", base_model_type="", width=0, height=0,
                        splices_exact_image=True, branch_carried_image=None,
                        branch_cut_video_block=None, branch_cut_audio_block=None):
    """Build the all-latent sidecar for one LTX2 / Wan / LongCat / MiniMax H3 job window.

    The all-latent sidecar is the sole content source of the video: one
    block per window with its own prompt/seed/steps/denoising strength, the
    pixel-space trims and the decoded frame count, stacked on one time
    dimension. ``splices_exact_image`` distinguishes the families: LTX2
    splices the exact preprocessed start image into frame 0 of the written
    video, so the sidecar embeds that image as lossless PNG bytes (a branch
    job propagates the source sidecar's embedded image verbatim — content,
    not a pointer) and its i2v head trim drops the decoded round-trip frame
    the splice replaces; Wan, LongCat and MiniMax H3 keep the decoded
    latent frame instead (the start image is a model condition, never
    spliced into the video), so their sidecars embed no image block and
    their i2v head trim keeps that frame. ``branch_carried_image`` is the same propagated block
    for a branch whose prefix this run rejected (``branch_prefix`` is None):
    the kept region's fresh decode spliced the source sidecar's image into
    frame 0 of the written video, so the new sidecar carries the same block
    for the decode to splice back in.

    ``latent_windows`` / ``latent_audio_windows`` are the repeat's
    accumulators (they already hold the branch's carried windows); this
    window's block is appended and the whole sidecar is rebuilt. Returns the
    sidecar file bytes, or ``None`` only when the sidecar cannot be written at
    all (an i2v start image that could not be embedded). A committed length
    that does not match the written video is not a drop: the file is written
    flagged (``length_verified: false`` with the actual written length) so the
    decode and branch flows can handle it, instead of destroying the only copy
    of the model's latent view of the video.
    """
    is_h3 = _is_h3_family(str(base_model_type or ""), model_def)
    layout = latent_io.latent_layout_for(model_def, base_model_type)
    branch_sidecar = branch_prefix if (branch_prefix is not None and window_no == 1) else None
    # Pixel-space trims applied to this window's decoded sample in the
    # assembly (they must be recorded exactly so the decode job can
    # re-assemble the original video).
    head_trim = int(post_decode_pre_trim) + int(trim_first_frames)
    # The H3 branch head window's mode (``None`` for non-H3 jobs, non-branch
    # windows, and H3 jobs whose prefix this run rejected — those keep the
    # Phase-0 bookkeeping): ``"frozen"`` — the pipeline froze the kept
    # latents at the head of the window's tensor (a Case A mid-cut);
    # ``"anchor"`` — the pipeline accepted the Case B window-end
    # substitution (``h3_anchor_head``): a pure-new window whose first
    # decoded frame re-anchors the carried window's last frame;
    # ``"rejected-end"`` — the pipeline's grid check rejected the kept run
    # (it fills the whole containing window), so the head window is a
    # fresh encode whose decoded region the assembly replaced with the
    # containing window's own decode.
    _h3_branch_mode = None
    if window_no == 1 and branch_sidecar is not None and is_h3:
        _prefix = branch_sidecar.get("prefix_latents")
        _head_latents = latent_io.payload_latents(latent_entry)
        # The pipeline freezes the kept latents at the head of the tensor in
        # its own working dtype, so the comparison is tolerant to the
        # dtype round trip (a fresh-encode head differs at the ~1.0 scale
        # and never matches).
        _prefix_honored = (
            _prefix is not None and _head_latents is not None
            and int(_head_latents.shape[1]) >= int(_prefix.shape[1])
            and torch.allclose(_head_latents[:, :int(_prefix.shape[1])].float(), _prefix.float(), rtol=2e-2, atol=1e-2))
        if int(latent_entry.get("h3_anchor_head") or 0) == 1:
            _h3_branch_mode = "anchor"
        elif _prefix_honored:
            _h3_branch_mode = "frozen"
        elif branch_sidecar.get("cut_at_window_end"):
            _h3_branch_mode = "rejected-end"
    if window_no == 1 and branch_sidecar is not None:
        # Branch head window: the decoded head is the source window's own
        # decoded head, so the block records a head trim that lets the
        # decode re-assemble the video the job wrote. The frozen prefix
        # (the carried kept latents) is the assembly's kept region itself:
        # step 3 splices a fresh decode of it into the written video (never a
        # re-extracted source segment), and it sits at the head of this
        # block's decoded sample, so it is not an external overlap to trim —
        # only the block's own head (the continuation overlap, the i2v slot,
        # or the anchor frame) is dropped on decode.
        if is_h3:
            if _h3_branch_mode == "frozen":
                # The frozen head decodes the kept latents at its own head
                # (the assembly splices a fresh decode of them into the
                # written video in their place), so the recorded head trim
                # is the window's own rebuilt history plus the kept
                # latents' decoded length (the plan's assembly overlap): the
                # block commits the new latents' decoded run only, and the
                # kept region commits from its own cut block (below).
                head_trim = int(latent_entry.get("h3_history_frames") or 0) + int(branch_sidecar.get("assembly_overlap") or 0)
            elif _h3_branch_mode == "anchor":
                # A Case B anchor head (the pipeline's h3_anchor_head: a
                # branch at a window end whose kept run decodes past the
                # committed region) is a pure-new window whose first decoded
                # frame re-anchors the carried window's last frame (a shared
                # frame), so the block records a one-frame head trim that
                # drops that shared frame and no re-decoded history.
                head_trim = 1
            else:
                # ``rejected-end`` records the containing window's own
                # block instead (below); when that block is unavailable the
                # block keeps the containing window's own recorded head
                # trim (0 for an all-latent t2v window 1, the i2v image slot
                # for an i2v window 1, the window's own overlap for k > 1).
                head_trim = int(branch_sidecar.get("containing_window_head_trim") or 0)
        else:
            # LTX2 / Wan / LongCat: a mid-window cut keeps the containing
            # window's own recorded head trim (0 for an all-latent t2v
            # window 1, the i2v image slot for an LTX2 i2v window 1, the
            # decoded image frame's slot for a Wan i2v window 1, the
            # window's own overlap for k > 1); a cut on a window boundary
            # is a fresh standard window, so the block records the
            # assembly overlap the job trims from its decoded sample.
            if bool(branch_sidecar.get("cut_at_window_end")):
                head_trim = int(branch_sidecar.get("assembly_overlap") or 0)
            else:
                head_trim = int(branch_sidecar.get("containing_window_head_trim") or 0)
    elif window_no == 1:
        # The first window's decoded head overlaps the region the assembly
        # keeps before it: the source prefix's trailing frames
        # (continuation) or the frozen prefix's decoded run (branch: the
        # overlap _prepare_branch_prefix resolved for this job).
        if splices_exact_image:
            # LTX2 i2v: the overlap is the one decoded image frame the
            # spliced start image replaces, so the head trim is exactly the
            # overlap and the decode splices the image back into frame 0.
            head_trim += int(source_video_overlap_frames_count)
        elif not window_prefix_is_image:
            # Video prefix (a continuation): the decoded overlap is dropped
            # in the assembly, as for LTX2. (A fresh Wan t2v has no prefix,
            # so the overlap is 0.)
            head_trim += int(source_video_overlap_frames_count)
        # Wan i2v: frame 0 of the written video is the decoded latent (the
        # start image's VAE round-trip); the assembly drops the image, not
        # the decoded frame, so no overlap is trimmed here.
    head_trim += int(reuse_frames) if (sliding_window and window_no > 1 and reuse_frames > 0) else 0
    tail_trim = int(discard_last_frames) + int(automatic_trim_last_frames)
    window_latents = latent_io.payload_latents(latent_entry)
    i2v_image = None
    if splices_exact_image and branch_prefix is not None:
        # A branch job never owns the original start image: when the source
        # sidecar is i2v-derived the embedded image propagates verbatim
        # (those bytes are exactly the video's frame 0). It must be
        # re-embedded on EVERY per-window rebuild of the sidecar, not only
        # the first: a later window's rebuild would otherwise drop the
        # block — and with it the +1 image slot in the committed total —
        # leaving the sidecar one frame short of the written video. The
        # branch head's recorded head trim keeps the image slot, so the
        # decode job splices the propagated image back into frame 0.
        i2v_image = branch_prefix.get("carried_image")
    elif splices_exact_image and branch_carried_image is not None:
        # The same propagation for a branch whose prefix this run rejected
        # (no branch_prefix): the kept region's fresh decode spliced the
        # source sidecar's image into frame 0 of the written video, so the
        # new sidecar embeds the same block on every rebuild (the decode
        # splices it back into the first window's run, the way it does for
        # the source sidecar).
        i2v_image = branch_carried_image
    elif (splices_exact_image and window_no == 1 and window_prefix_is_image and not fake_start_image
            and not sample_is_hdr and int(source_video_overlap_frames_count) == 1):
        if image_start_tensor is not None:
            try:
                image_pil = convert_tensor_to_image(image_start_tensor)
                image_buffer = io.BytesIO()
                image_pil.save(image_buffer, format="PNG")
                i2v_image = {"kind": "image_png", "data": image_buffer.getvalue()}
            except Exception as exc:
                print(f"Latent save: the i2v start image could not be embedded ({exc}); the latent file is not written.")
                return None
    window_meta = {
        "prompt": str(latent_entry.get("prompt") or ""),
        "seed": int(latent_entry.get("seed") or 0),
        "steps": int(latent_entry.get("steps") or 0),
        "denoising_strength": float(latent_entry.get("denoising_strength") or 1.0),
        "head_trim": int(head_trim),
        "tail_trim": int(tail_trim),
        "frame_count": latent_io.window_pixel_frames(window_latents, layout["stride"], layout["offset"]),
    }
    if is_h3:
        # The H3 VAE's piecewise 5/17 layout cannot be expressed by
        # latent_stride/frame_offset, so the per-window bookkeeping the
        # pipeline recorded travels with the window block. ``frame_count``
        # keeps the pipeline's own output length (the decode's audio
        # bookkeeping measures the window's full output, history included)
        # while the committed arithmetic below applies the piecewise layout
        # to the latents and caps it by the recorded target.
        window_meta["frame_count"] = int(latent_entry.get("frames") or 0)
        window_meta["h3_history_frames"] = int(latent_entry.get("h3_history_frames") or 0)
        window_meta["h3_target_frames"] = int(latent_entry.get("h3_target_frames") or 0)
        if int(latent_entry.get("h3_anchor_head") or 0) == 1:
            window_meta["h3_anchor_head"] = 1
            # The anchor-head window records no re-decoded history: its
            # one-frame head trim drops the shared frame instead.
            window_meta["h3_history_frames"] = 0
    if _h3_branch_mode in ("frozen", "anchor") and branch_cut_video_block is not None:
        # The kept region commits from its own block: the containing
        # window's cut block (its kept latents, the decoded run capped at
        # the cut — the full window itself for a window-end cut), whose
        # stand-alone decode is exactly the kept region the assembly
        # spliced into the written video. It is inserted ahead of this
        # head window's block, which commits the new portion only.
        latent_windows.append(dict(branch_cut_video_block))
    if _h3_branch_mode == "rejected-end" and branch_cut_video_block is not None:
        # The fresh head window's decoded region is not in the written
        # video (the assembly spliced the containing window's own decode
        # in its place), so the sidecar records the containing window's
        # block instead of this fresh window — the sidecar stays
        # self-contained and its committed tiling matches the written
        # video.
        latent_windows.append(dict(branch_cut_video_block))
    elif _h3_branch_mode in (None, "frozen", "anchor"):
        latent_windows.append({"latents": window_latents, **window_meta})
    if latent_audio_entry is not None:
        if _h3_branch_mode == "anchor" and branch_cut_audio_block is not None:
            # The anchor's generated audio starts where the kept region's
            # audio ends (the decoded shared frame's samples are dropped
            # with its one-frame head trim), so the head entry's recorded
            # window starts are rebased onto the cut block's span (the
            # kept region's own audio rides in the preseeded carried entry).
            _rebased = dict(latent_audio_entry)
            _rebased["windows"] = [dict(_w) for _w in (latent_audio_entry.get("windows") or [])]
            _cut_end = (int(branch_cut_audio_block.get("start_sample") or 0)
                        + int(branch_cut_audio_block.get("samples") or 0))
            for _w in _rebased["windows"]:
                _head_start = int(_w.get("start_sample") or 0)
                _w["start_sample"] = _head_start + (_cut_end - _head_start)
            latent_audio_windows.append(_rebased)
        elif _h3_branch_mode != "rejected-end":
            # ``rejected-end``: the fresh head window's generated audio is
            # not in the written video (the containing window's own audio
            # covers the kept region, in the preseeded carried entry), so
            # its audio entry is dropped with its video block.
            latent_audio_windows.append(latent_audio_entry)
    try:
        payload = latent_io.build_latent_sidecar(
            [window["latents"] for window in latent_windows],
            [window for window in latent_windows],
            app_version=ctx.WanGP_version,
            model_type=str(model_type or ""),
            base_model_type=str(base_model_type or ""),
            model_filename=str(latent_entry.get("model_filename") or ""),
            loras=list(latent_entry.get("loras") or []),
            vae_file=str(latent_entry.get("vae_file") or ""),
            vae_sha256=str(latent_entry.get("vae_sha256") or ""),
            vae_z_dim=int(latent_entry.get("vae_z_dim") or 0),
            vae_dtype=str(latent_entry.get("vae_dtype") or ""),
            width=int(width or 0),
            height=int(height or 0),
            fps=float(latent_entry.get("fps") or 0.0),
            latent_stride=int(layout["stride"]),
            frame_offset=int(layout["offset"]),
            is_hdr=bool(sample_is_hdr),
            i2v_start_image=i2v_image,
        )
    except Exception as exc:
        print(f"Latent save: the latent file could not be built ({exc}); it is not written.")
        return None
    file_obj = payload
    if len(latent_audio_windows) > 0:
        # The builder returns the format-tagged payload ({audio_id: entry});
        # unpack it (the same merge the v2 writer uses) so the file object
        # carries the entry itself - a doubly nested tag would be invisible
        # to latent_io.audio_latent_payload and the decode job would drop
        # the soundtrack.
        audio_payload = _build_cumulative_audio_payload(latent_audio_windows)
        if audio_payload is not None:
            file_obj = {**payload, **audio_payload}
    # Verify the committed length against the written video before the file
    # goes out. A mismatch means a miscount by the app (the recorded total is
    # what a decode and a branch cut are measured against), not user error -
    # and the file holds the only unrecoverable copy of the model's latent
    # view of this video. Write it anyway, flagged: the entry records the
    # actual written length beside the recorded total (length_verified:
    # false), a decode proceeds but says the decoded video has the recorded
    # length, and a branch cut is refused on it.
    committed_total = latent_io.sidecar_total_frames(file_obj[latent_io.LATENT_FORMAT_ID])
    if int(written_frame_count) > 0 and committed_total != int(written_frame_count):
        video_entry = file_obj[latent_io.LATENT_FORMAT_ID]
        video_entry["length_verified"] = False
        video_entry["written_frame_count"] = int(written_frame_count)
        print(f"Latent save: the length recorded in the latent file ({committed_total} frames) does not match the written video "
              f"({int(written_frame_count)} frames); the file is written but flagged (length_verified: false) - a decode "
              f"will reproduce the recorded length, not the video's, and a branch cut is refused on it.")
    return latent_io.build_latent_file_bytes(file_obj)

def _ltx2_i2v_prefix_to_frames_space(entry, block, is_hdr, device):
    """The all-latent sidecar's embedded i2v start image in the decoded
    frames' pixel space (uint8 for SDR, the HDR linear mapping for HDR),
    matching how the job's assembly wrote frame 0 from the preprocessed
    image."""
    from models.ltx2.ltx2 import LTX2_HDR_TRANSFORM
    try:
        image = Image.open(io.BytesIO(bytes(block["data"])))
        if image.mode not in ("L", "RGB"):
            image = image.convert("RGB")
        prefix = convert_image_to_tensor(image).unsqueeze(1)
        if torch.is_floating_point(prefix):
            prefix = convert_video_tensor_to_uint8_chunked(prefix)
        return _prefix_to_frames_space(entry, prefix, "linear" if is_hdr else "uint8",
                                       str(entry.get("hdr_transform") or LTX2_HDR_TRANSFORM), device)
    except Exception as exc:
        print(f"Latent decode: the embedded i2v start image could not be read: {exc}")
        return None


def _ltx2_latent_decode_windows(send_cmd, gen, model_def, entry, window_specs, audio_entry, audio_windows, is_hdr, vae_dtype, decode_models, tiling_config):
    """Decode every video (and matching audio) window of a saved LTX-2 latent.

    Returns ``(window_frames_list, audio_waveform_chunks)``; sends the error
    itself and returns ``None`` when the decode fails or is aborted.
    """
    from models.ltx2.ltx2 import LTX2_HDR_TRANSFORM
    from models.ltx2.ltx_core.model.audio_vae import decode_audio
    from models.ltx2.ltx_core.model.video_vae import decode_video_to_tensor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fps = float(entry.get("fps") or 0) or float(model_def.get("fps") or 24)
    interrupt_check = lambda: gen.get("abort", False)
    window_frames_list = []
    audio_waveform_chunks = []
    # The all-latent i2v sidecar splices the embedded start image into frame
    # 0 of the video (content, not a pointer — no source re-extraction).
    is_all_latent = latent_io.is_all_latent_entry(entry)
    i2v_image = latent_io.entry_i2v_image(entry) if is_all_latent else None
    try:
        for idx, spec in enumerate(window_specs):
            # The sidecar stores the canonical unbatched [c, t, h, w] layout;
            # the LTX-2 VAE decoder (both tiled and full) expects the batched
            # 5D [1, c, t, h, w] layout the generation pipelines feed it.
            latent = spec["latents"].to(device=device, dtype=vae_dtype)
            if latent.dim() == 4:
                latent = latent.unsqueeze(0)
            video = decode_video_to_tensor(
                latent,
                decode_models.video_decoder,
                tiling_config,
                interrupt_check=interrupt_check,
                hdr_transform=str(entry.get("hdr_transform") or LTX2_HDR_TRANSFORM) if is_hdr else None,
                output_dtype=torch.float16 if is_hdr else None,
            )
            latent = None
            if video is None:
                if gen.get("abort", False):
                    return None
                raise ValueError(f"window {spec['window_no']} could not be decoded")
            frames = video.permute(3, 0, 1, 2).contiguous()
            video = None
            # Re-apply the exact pixel trims of the original assembly so the
            # decoded windows re-assemble into the original video unchanged.
            head_trim = int(spec["head_trim"])
            tail_trim = int(spec["tail_trim"])
            if head_trim > 0 or tail_trim > 0:
                frames = frames[:, head_trim:max(head_trim, frames.shape[1] - tail_trim)]
            if is_all_latent and i2v_image is not None and idx == 0:
                # The all-latent i2v sidecar splices the embedded start image
                # into frame 0: the recorded head trim already dropped the
                # decoded round-trip the image replaces, so the exact image
                # is placed in front of the trimmed decoded run (the way the
                # job wrote it).
                prefix = _ltx2_i2v_prefix_to_frames_space(entry, i2v_image, is_hdr, device)
                if prefix is None:
                    raise ValueError(
                        f"window {int(spec.get('window_no') or idx + 1)}: the embedded i2v start image could not be read; "
                        f"the decoded output would lose the video's first frame."
                    )
                # The decoded frames come back in the decoder's accumulation
                # buffer (CPU for tiled decode) while the image helper
                # materializes on ``device``: align the prefix to the
                # frames' device before concatenating.
                frames = torch.cat([prefix.to(device=frames.device, dtype=frames.dtype), frames], dim=1)
            window_frames_list.append(frames)
            if idx < len(audio_windows):
                # The 2D [c, t*mel] stored latent goes back to the 4D
                # [1, c, t, mel] layout the audio VAE expects (mel_bins was
                # recorded at save time; 16 is the LTX-2 default).
                mel_bins = int((audio_entry or {}).get("mel_bins") or 16)
                audio_latent = audio_windows[idx]["latents"].to(device=device, dtype=vae_dtype)
                if audio_latent.dim() == 2:
                    audio_latent = audio_latent.reshape(1, audio_latent.shape[0], audio_latent.shape[1] // max(1, mel_bins), max(1, mel_bins))
                waveform = decode_audio(audio_latent, decode_models.audio_decoder, decode_models.vocoder)
                audio_latent = None
                if waveform is not None and waveform.numel() > 0:
                    audio_waveform_chunks.append(waveform.cpu())
    except Exception as exc:
        send_cmd("error", gr.Error(f"The saved latent could not be decoded: {exc}"))
        return None
    finally:
        for module in (decode_models.video_decoder, getattr(decode_models, "audio_decoder", None), getattr(decode_models, "vocoder", None)):
            if module is not None:
                module.to("cpu")
    return window_frames_list, audio_waveform_chunks


def _edit_latent_job_ltx2(ctx, send_cmd, state, video_source, client_id, plugin_data, sidecar_obj, entry, model_def, window_specs,
                          api_return_video_uint8, api_return_audio, api_suppress_metadata_images):
    """Decode a saved LTX-2 latent back to pixels.

    LTX-2 uses a different VAE stack than Wan/LongCat (separate video and
    audio VAEs plus an optional vocoder, with HDR support), so the decode
    job takes a family-specific path: load only the VAE components, decode
    each window and hand the result to ``_save_ltx2_decoded_video``.
    """
    gen = ctx.get_gen_info(state)
    if gen.get("abort", False):
        return
    model_type = str(entry.get("model_type") or "")
    base_model_type = str(entry.get("base_model_type") or "") or str(model_def.get("architecture") or "")
    fps = float(entry.get("fps") or 0) or float(model_def.get("fps") or 24)
    is_hdr = bool(entry.get("is_hdr"))
    # The sidecar records the dtype the VAE ran with at generation; decode
    # with the same dtype (decoding with a different precision than the
    # generation changes the decoded pixels slightly). The server config
    # is the fallback for sidecars that do not record it.
    _recorded_dtype = {"float16": torch.float16, "16": torch.float16,
                       "bfloat16": torch.bfloat16,
                       "float32": torch.float, "32": torch.float}.get(str(entry.get("vae_dtype") or ""))
    vae_dtype = _recorded_dtype if _recorded_dtype is not None else (
        torch.float16 if str(ctx.server_config.get("vae_precision", "16")) == "16" else torch.float)

    from models.ltx2 import ltx2_handler
    from models.ltx2.ltx2 import build_decode_tiling_config, load_ltx2_decode_models

    audio_entry = latent_io.audio_latent_payload(sidecar_obj)
    audio_windows = latent_io.audio_window_specs(audio_entry) if audio_entry is not None else []
    needs_audio = len(audio_windows) > 0

    # The VAE names recorded in the sidecar take precedence (they are the
    # ones that produced this latent); the model definition provides the
    # fallback (and the config files all-in-one checkpoints need).
    resolved = {}
    try:
        resolved = ltx2_handler._resolve_multi_file_paths(model_def, base_model_type, include_spatial_upsampler=False) or {}
    except Exception as exc:
        print(f"Latent decode: the LTX2 model definition could not be fully resolved ({exc}); relying on the VAE names recorded in the latent file.")
    vae_urls = model_def.get("VAE_URLs", None)
    if isinstance(vae_urls, str) and len(vae_urls) > 0:
        default_video_vae = vae_urls
    elif isinstance(vae_urls, list) and len(vae_urls) > 0:
        default_video_vae = vae_urls[0]
    else:
        default_video_vae = os.path.basename(str(resolved.get("video_vae") or ""))
    video_vae_file = os.path.basename(str(entry.get("vae_file") or "")) or default_video_vae
    if not video_vae_file:
        send_cmd("error", gr.Error("Unable to determine which LTX2 video VAE is required to decode this latent."))
        return
    audio_vae_file = os.path.basename(str((audio_entry or {}).get("audio_vae_file") or ""))
    if needs_audio and not audio_vae_file:
        audio_vae_file = os.path.basename(str(resolved.get("audio_vae") or ""))
    vocoder_file = os.path.basename(str((audio_entry or {}).get("vocoder_file") or ""))
    if needs_audio and not vocoder_file:
        vocoder_file = os.path.basename(str(resolved.get("vocoder") or ""))
    config_fallback = str(resolved.get("video_vae_config") or resolved.get("model_config") or "")

    try:
        # Decode on the GPU when available (the latents are decoded there);
        # the VAE/vocoder components are small and get moved back to CPU
        # after the decode in _ltx2_latent_decode_windows.
        decode_device = "cuda" if torch.cuda.is_available() else "cpu"
        decode_models = load_ltx2_decode_models(
            video_vae_file,
            audio_vae_file if needs_audio else "",
            vocoder_file if needs_audio else "",
            VAE_dtype=vae_dtype,
            device=decode_device,
            config_fallback_path=config_fallback,
        )
    except Exception as exc:
        send_cmd("error", gr.Error(f"Unable to load the VAE components required to decode this LTX2 latent: {exc}"))
        return
    if needs_audio and (not hasattr(decode_models, "audio_decoder") or not hasattr(decode_models, "vocoder")):
        send_cmd("error", gr.Error("The LTX2 audio VAE and/or vocoder required to decode the audio part of this latent could not be loaded."))
        return

    device_mem_capacity = int(torch.cuda.get_device_properties(0).total_memory / 1048576) if torch.cuda.is_available() else 0
    # Mirror the generation call exactly (same capacity, same
    # vae_config, and the mixed-precision flag derived from the VAE
    # precision the same way the generation derives it): a different
    # tiling than the generation's decodes the same latents to slightly
    # different pixels (the overlap blending differs), so the decode
    # must use the generation's tiling to reproduce the video.
    tiling_config = build_decode_tiling_config(
        decode_models.video_decoder,
        int(ctx.server_config.get("vae_config", 0) or 0),
        device_mem_capacity,
        str(ctx.server_config.get("vae_precision", "16")) == "32",
        fps,
        output_height=int(entry.get("height") or 0) or None,
        output_width=int(entry.get("width") or 0) or None,
    )

    decoded = _ltx2_latent_decode_windows(send_cmd, gen, model_def, entry, window_specs, audio_entry, audio_windows, is_hdr, vae_dtype, decode_models, tiling_config)
    del decode_models
    if decoded is None:
        return
    window_frames_list, audio_waveform_chunks = decoded
    if not window_frames_list:
        send_cmd("error", gr.Error("The saved latent does not contain any decodable window."))
        return
    frames = torch.cat(window_frames_list, dim=1) if len(window_frames_list) > 1 else window_frames_list[0]
    sample_rate = int((audio_entry or {}).get("sample_rate") or 0)
    _save_ltx2_decoded_video(ctx, send_cmd, state, video_source, client_id, plugin_data, frames, audio_waveform_chunks, sample_rate, entry, audio_entry, fps, is_hdr,
                             api_return_video_uint8, api_return_audio, api_suppress_metadata_images)


def _save_ltx2_decoded_video(ctx, send_cmd, state, video_source, client_id, plugin_data, frames, audio_waveform_chunks, sample_rate, entry, audio_entry, fps, is_hdr,
                             api_return_video_uint8, api_return_audio, api_suppress_metadata_images):
    """Save a decoded LTX-2 latent: SDR goes through ``save_video`` (uint8),
    HDR through ``save_hdr_video`` (HDR-linear float), and the decoded audio
    waveform (when present) is muxed into the output container.
    """
    gen = ctx.get_gen_info(state)
    source_dir = os.path.dirname(str(video_source or ""))
    if len(source_dir) == 0:
        source_dir = str(ctx.save_path)
    source_stem = os.path.splitext(os.path.basename(str(video_source or "")))[0]
    if len(source_stem) == 0:
        source_stem = "latent_decode"
    container = str(ctx.server_config.get("video_container", "mp4")).lstrip(".")
    new_video_path = ctx.get_available_filename(source_dir, f"{source_stem}_latent_decode", force_extension=f".{container}")

    audio_data = None
    if audio_waveform_chunks:
        if sample_rate <= 0:
            sample_rate = 16000
        windows = (audio_entry or {}).get("windows") or []
        if len(windows) == len(audio_waveform_chunks):
            # Multi-window job: mirror the generation-time assembly. The
            # writer commits window 1 in full and, for every later window,
            # drops its first reuse-frames (int(reuse_frames * sr / fps)
            # samples, the same arithmetic as truncate_audio) before
            # concatenating at the accumulated position — the written track
            # has no inter-window overlap at all. Placing each decoded chunk
            # at its recorded start_sample (the window's own timeline
            # origin, one overlap earlier) instead shifts every post-first
            # window by the overlap and re-adds the dropped head. For a
            # post-first window the overlap frame count is its video block's
            # recorded head_trim, capped by the job's sliding overlap (a
            # zero-overlap job trimmed no audio head, and a window's head
            # trim can carry video-only trims the audio never received).
            video_windows = (entry or {}).get("windows") or []
            overlap_frames = int((entry or {}).get("sliding_window_overlap") or 0)
            head_trims = [0] * len(audio_waveform_chunks)
            for idx in range(1, len(audio_waveform_chunks)):
                if idx < len(video_windows) and fps > 0:
                    frames = min(int((video_windows[idx] or {}).get("head_trim") or 0), overlap_frames)
                    if frames > 0:
                        head_trims[idx] = int(frames * sample_rate / fps)
            waveform = latent_io.merge_audio_window_chunks(
                audio_waveform_chunks, windows, int((audio_entry or {}).get("samples") or 0),
                head_trims=head_trims)
        else:
            waveform = torch.cat([c.unsqueeze(0) if c.dim() == 1 else c for c in audio_waveform_chunks], dim=1)
        video_duration = frames.shape[1] / fps if fps > 0 else 0
        max_samples = int(round(video_duration * sample_rate)) if video_duration > 0 else 0
        if max_samples > 0 and waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
        if waveform.numel() > 0:
            # .numpy() needs a CPU tensor (the decode models run on the
            # accelerator and hand back accelerator tensors).
            audio_data = waveform.float().t().cpu().numpy()
    if audio_data is not None:
        # Muxing reads the video from a separate file, so the decoded video
        # is first written to a temp path, then combined with the audio into
        # the final file.
        time_flag = int(time.time())
        temp_video_path = ctx.get_available_filename(ctx.save_path, f"tmp{time_flag}_ltx2_latent_decode", force_extension=f".{container}")
        temp_audio_path = ctx.get_available_filename(ctx.save_path, f"tmp{time_flag}_ltx2_latent_audio", force_extension=".wav")
        if is_hdr:
            save_hdr_video(tensor=frames, save_file=temp_video_path, fps=fps, codec_type=ctx.server_config.get("hdr_video_crf", 8), container=container)
        else:
            save_video(tensor=frames, save_file=temp_video_path, fps=fps, codec_type=ctx.server_config.get("video_output_codec", None), container=container, nrow=1, normalize=False, value_range=(0, 255))
        write_wav_file(temp_audio_path, audio_data, sample_rate)
        combine_and_concatenate_video_with_audio_tracks(
            new_video_path,
            temp_video_path,
            [],
            [temp_audio_path],
            0,
            sample_rate,
            new_audio_from_start=True,
            audio_codec_key=ctx.server_config.get("audio_output_codec", "aac_128"),
            verbose=ctx.verbose_level >= 2,
        )
        os.remove(temp_video_path)
        os.remove(temp_audio_path)
    else:
        if is_hdr:
            save_hdr_video(tensor=frames, save_file=new_video_path, fps=fps, codec_type=ctx.server_config.get("hdr_video_crf", 8), container=container)
        else:
            save_video(tensor=frames, save_file=new_video_path, fps=fps, codec_type=ctx.server_config.get("video_output_codec", None), container=container, nrow=1, normalize=False, value_range=(0, 255))
    print(f"Latent decoded video saved to Path: {new_video_path}")

    configs = {"type": ctx.get_model_record("Latent Decode"), "prompt": str(entry.get("prompt") or "")}
    with ctx.lock:
        gen["file_list"].append(new_video_path)
        gen["file_settings_list"].append(configs)
    if not api_suppress_metadata_images:
        from shared.utils.video_metadata import save_video_metadata
        save_video_metadata(new_video_path, configs, None, allow_inplace_update=True, verbose_level=ctx.verbose_level)
    notifications.record_generation(ctx.server_config, gen, new_video_path, configs)
    if api_return_video_uint8 or api_return_audio:
        # Same layout as the generation flow's sample tensor ([c, f, h, w]).
        store_api_output_artifact(gen, client_id, new_video_path, "video", frames if api_return_video_uint8 else None, audio_data if api_return_audio else None, sample_rate if audio_data is not None else None, fps, hdr=is_hdr)
    gen["last_was_audio"] = False
    send_cmd("output")
    ctx.clear_status(state)


def _h3_sidecar_vae_file(recorded, default):
    """Resolve the VAE name a MiniMax H3 sidecar records back to the loader's
    checkpoint path.

    The sidecar keeps the checkpoint's bare file name (the identity the model
    list shows), while the H3 loaders compare the name they are handed against
    their own constants — several of which carry the family's ``minimax_h3/``
    folder prefix — to decide the model's shape: the light VAE's shallow
    decoder, the X2 VAE's upsampling path. Matching on the basename keeps the
    recorded identity and the loader's expectations in step; an unrecognised
    name is handed through untouched for ``fl.locate_file`` to search.
    """
    from models.minimax_h3.minimax_h3_main import (AUDIO_VAE_FILE, VIDEO_VAE_FILE, VIDEO_VAE_FP8MIX_FILE,
                                                   VIDEO_VAE_INT8_FILE, VIDEO_VAE_LIGHT_FILE,
                                                   VIDEO_VAE_LIGHT_INT8_FILE, X2_VAE_FILE, X2_VAE_INT8_FILE)
    recorded = str(recorded or "").strip()
    if not recorded:
        return str(default or "")
    if os.path.sep in recorded:
        return recorded
    for candidate in (VIDEO_VAE_FILE, VIDEO_VAE_FP8MIX_FILE, VIDEO_VAE_INT8_FILE, VIDEO_VAE_LIGHT_FILE,
                      VIDEO_VAE_LIGHT_INT8_FILE, X2_VAE_FILE, X2_VAE_INT8_FILE, AUDIO_VAE_FILE):
        if os.path.basename(str(candidate)) == recorded:
            return str(candidate)
    return recorded


def _h3_fit_audio_samples(audio, sample_count):
    """Pad or truncate a waveform to exactly ``sample_count`` samples on its
    last dimension (the H3 pipeline's ``_fit_audio_samples``, kept local so
    the decode job never imports the generation pipeline)."""
    count = int(sample_count or 0)
    if count <= 0:
        return audio[..., :0]
    length = int(audio.shape[-1])
    if length >= count:
        return audio[..., :count]
    pad = torch.zeros(*audio.shape[:-1], count - length, dtype=audio.dtype, device=audio.device)
    return torch.cat([audio, pad], dim=-1)


def _edit_latent_job_h3(ctx, send_cmd, state, video_source, client_id, plugin_data, sidecar_obj, entry, model_def, window_specs,
                        api_return_video_uint8, api_return_audio, api_suppress_metadata_images):
    """Decode a saved MiniMax H3 latent back to pixels.

    H3 decodes through two independent VAEs (video ``[1, 24, t, h, w]`` with
    a piecewise 5/17 temporal layout, audio ``[1, 32, 2, t]`` to 32 kHz
    stereo), so the job takes a family-specific path: load only the VAE
    components (no diffusion transformer), decode every window, and re-apply
    the generation pipeline's exact per-window assembly (decoded target
    region plus the re-encoded history prefix and the recorded pixel
    trims) so the decoded output matches the generated video, soundtrack
    included.

    Placement is fully positional (step 6, D5 of the h3branch plan): every
    length comes from the window's own recorded fields (``h3_target_frames``,
    ``h3_history_frames``, ``frame_count``, the head/tail trims) and the
    windows tile in sidecar order — video by concatenation, audio on the
    running cursor — so no derived offset or prefix bookkeeping is read
    here. A branch
    anchor window (``h3_anchor_head``: a Case B window whose first decoded
    frame re-anchors the carried window's last frame) is the first sidecar
    window after the carried ones (window 1 itself when nothing is carried),
    so its one-frame head trim drops the re-decoded shared frame and its
    audio starts exactly where the carried (committed) region's audio ends
    — the prefix's tail, positionally.
    """
    gen = ctx.get_gen_info(state)
    if gen.get("abort", False):
        return
    fps = float(entry.get("fps") or 0) or float(model_def.get("fps") or 24)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from models.minimax_h3.minimax_h3_main import (AUDIO_VAE_FILE, VIDEO_VAE_FILE, X2_VAE_FILE, X2_VAE_INT8_FILE,
                                                   _load_audio_vae, _load_video_vae)
    from models.minimax_h3.pipeline import _video_to_uint8_cpu

    audio_entry = latent_io.audio_latent_payload(sidecar_obj)
    audio_windows = latent_io.audio_window_specs(audio_entry) if audio_entry is not None else []
    needs_audio = len(audio_windows) > 0
    sample_rate = int((audio_entry or {}).get("sample_rate") or 32000)

    # The VAE names recorded in the sidecar take precedence (they produced
    # this latent); the model definition and the family defaults fall back.
    video_vae_file = _h3_sidecar_vae_file(entry.get("vae_file"), model_def.get("video_vae_file") or VIDEO_VAE_FILE)
    audio_vae_file = _h3_sidecar_vae_file((audio_entry or {}).get("audio_vae_file"),
                                          model_def.get("audio_vae_file") or AUDIO_VAE_FILE)
    # The sidecar records the dtype each VAE ran with (the default video
    # checkpoint is fp16, the audio VAE is locked to fp32); loading with the
    # same dtype reproduces the generation decode.
    vae_dtype = {"float16": torch.float16, "16": torch.float16, "bfloat16": torch.bfloat16}.get(str(entry.get("vae_dtype") or "float32"), torch.float32)
    stored_vae_hash = str(entry.get("vae_sha256") or "")
    stored_audio_hash = str((audio_entry or {}).get("audio_vae_sha256") or "")

    # The H3 VAE's decoder blocks dispatch through the shared attention
    # helper, which reads the job's attention mode from the offload state;
    # a decode-only job (no prior generation in this process) must set it.
    offload.shared_state["_attention"] = get_default_attention_mode()
    try:
        # The loader takes the checkpoint name and the offload layout flags —
        # its compute dtype comes from the checkpoint itself (fp16 / bf16
        # native, an fp32 checkpoint lowered to fp16), which is the dtype the
        # generation recorded in the sidecar. qkv_splitting only re-partitions
        # the official VAE's linear layers for offloading; the decode runs one
        # un-offloaded forward pass, so the plain layout is loaded either way.
        # The X2 VAEs decode through their upsampling path, the flag that
        # builds that model shape.
        video_vae = _load_video_vae(
            video_vae_file, qkv_splitting=False,
            upsampling=os.path.basename(video_vae_file) in (os.path.basename(X2_VAE_FILE), os.path.basename(X2_VAE_INT8_FILE)))
        audio_vae = _load_audio_vae(audio_vae_file) if needs_audio else None
    except Exception as exc:
        send_cmd("error", gr.Error(f"The VAE checkpoint(s) required to decode this latent could not be loaded ({exc})."))
        return
    # The latents are cast to the dtype the loaded video VAE computes in (the
    # sidecar's recorded dtype is the same one, the checkpoint being the
    # identity that produced them); the audio VAE is always fp32.
    loaded_vae_dtype = getattr(video_vae, "_model_dtype", None)
    if loaded_vae_dtype is not None and loaded_vae_dtype != vae_dtype:
        print(f"Latent decode identity warning: the video VAE computes in {loaded_vae_dtype}, "
              f"the sidecar recorded {vae_dtype}; the latents follow the loaded VAE.")
        vae_dtype = loaded_vae_dtype
    if len(stored_vae_hash) > 0:
        video_vae_pth = fl.locate_file(video_vae_file, error_if_none=False)
        if os.path.isfile(video_vae_pth) and latent_io.sha256_file(video_vae_pth) != stored_vae_hash:
            print("Latent decode identity warning: the video VAE checkpoint on disk differs from the one used at save time.")
    if needs_audio and len(stored_audio_hash) > 0:
        audio_vae_pth = fl.locate_file(audio_vae_file, error_if_none=False)
        if os.path.isfile(audio_vae_pth) and latent_io.sha256_file(audio_vae_pth) != stored_audio_hash:
            print("Latent decode identity warning: the audio VAE checkpoint on disk differs from the one used at save time.")
    # Same spatial tiling the generation pipeline configures: it lowers the
    # peak memory of one full-size decode pass.
    video_vae.enable_tiling(tile_sample_min_height=256, tile_sample_min_width=256)
    video_vae.to(device)
    if audio_vae is not None:
        audio_vae.to(device)

    window_frames_list = []
    audio_canvas_chunks = []  # (start_sample, [2, samples] waveform) per window
    audio_cursor = 0  # next tiling position on the canvas (the production job
                      # appends each window's trimmed audio in order)
    prev_video = None  # previous window's [H | D] sample after the tail trim (float [-1, 1])
    prev_audio = None  # previous window's decoded audio after the tail trim ([2, samples])
    # i2v jobs never trimmed the audio (the start image carries no samples),
    # unlike their video head trim, so window 1's recorded trim needs one
    # frame back for the audio only.
    image_source = entry.get("image_source")
    is_i2v = bool(image_source) and len(str(entry.get("video_source") or "")) == 0
    aborted = False
    try:
        for idx, spec in enumerate(window_specs):
            if gen.get("abort", False):
                aborted = True
                break
            window_no = int(spec["window_no"])
            h3 = int(spec.get("h3_history_frames") or 0)
            target = int(spec.get("h3_target_frames") or 0)
            frame_count = int(spec.get("frame_count") or 0)
            head_trim = int(spec["head_trim"])
            tail_trim = int(spec["tail_trim"])

            latent = spec["latents"].to(device=device, dtype=vae_dtype)
            if latent.dim() == 4:
                latent = latent.unsqueeze(0)
            decoded = video_vae.decode(latent)
            latent = None
            if decoded is None or decoded.numel() == 0:
                raise ValueError(f"window {window_no} could not be decoded")
            frames = decoded[0].float().clamp_(-1.0, 1.0)  # [3, D, h, w]
            decoded = None
            if target > 0 and frames.shape[1] > target:
                frames = frames[:, :target]
            history_cat = False
            if h3 > 0:
                # The latent covers the target region only; the window's
                # history frames are the continuation the generation job was
                # handed, so they must be rebuilt here for the head trim and
                # for the next window to chain its history from. Window 1
                # has no earlier window to re-extract its history from: a
                # branch head that is the sidecar's own window 1 re-decodes
                # the carried region as its own head, so its block records
                # no external history.
                if window_no > 1 and prev_video is not None:
                    history = prev_video[:, -(h3 + 1):-1].to(frames.device)
                    frames = torch.cat([history, frames], dim=1)
                    history_cat = True
                elif window_no > 1:
                    print(f"Latent decode: window {window_no} records a history prefix but no earlier window decoded; continuing without it.")
                else:
                    print("Latent decode: the history of window 1 came from the source video, which the latent file no longer records; the decoded window starts at its first decoded frame.")
            # Tail trim first: the next window's history (video and audio) is
            # taken from the tail-trimmed sample, before the head trim, like
            # the original assembly.
            if tail_trim > 0:
                frames = frames[:, :max(0, frames.shape[1] - tail_trim)]
            prev_video = frames.cpu()
            # The recorded head trim counts the history frames too; when the
            # history was not rebuilt it has to come off the target region
            # only (the overlap minus what the decoded sample does not hold).
            # The i2v window 1 trim belongs to the start-image prefix (which
            # the original assembly dropped from itself), not to the decoded
            # sample, so it is not applied here.
            if head_trim > 0 and not (window_no == 1 and is_i2v):
                frames = frames[:, head_trim:] if history_cat else frames[:, max(0, head_trim - h3):]
            window_frames_list.append(_video_to_uint8_cpu(frames))

            # ---- Audio for this window (same assembly as the pipeline) ----
            if idx < len(audio_windows):
                a_latent = audio_windows[idx]["latents"]
                if a_latent.dim() == 3:
                    a_latent = a_latent[0]
                # The stored [64, t] flatten is the [32, 2, t] latent
                # (channels, stereo, time) read row-major.
                a_latent = a_latent.reshape(1, 32, 2, int(a_latent.shape[-1]))
                a_decoded = audio_vae.decode(a_latent.to(device=device, dtype=torch.float32))[0]  # [2, samples]
                a_latent = None
                a_decoded = a_decoded.cpu().float()
                target_samples = int(round(target / fps * sample_rate)) if (target > 0 and fps > 0) else int(a_decoded.shape[-1])
                a_decoded = _h3_fit_audio_samples(a_decoded, target_samples)
                if h3 > 0:
                    # The latent covers the target region only, so the
                    # window's [H | D] audio sample is rebuilt like the
                    # pipeline does: the history region from the previous
                    # window (silence for window 1, whose history came from
                    # the source and is dropped by the head trim below).
                    prefix_samples = int(round(h3 / fps * sample_rate)) if fps > 0 else 0
                    overlap_samples = int(round((h3 + 1) / fps * sample_rate)) if fps > 0 else prefix_samples
                    if window_no > 1 and prev_audio is not None:
                        prefix_a = _h3_fit_audio_samples(prev_audio[:, -overlap_samples:][:, :prefix_samples], prefix_samples)
                    else:
                        prefix_a = a_decoded.new_zeros(2, prefix_samples)
                    a_decoded = torch.cat([prefix_a, a_decoded], dim=1)
                total_samples = int(round(frame_count / fps * sample_rate)) if (frame_count > 0 and fps > 0) else int(a_decoded.shape[-1])
                a_decoded = a_decoded[:, :total_samples]
                if tail_trim > 0 and fps > 0:
                    tail_samples = int(round(tail_trim * sample_rate / fps))
                    a_decoded = a_decoded[:, :max(0, a_decoded.shape[-1] - tail_samples)]
                prev_audio = a_decoded
                # The original flow trimmed the audio by the full head trim
                # (history frames plus the anchor frame) except for i2v jobs,
                # where the start image carries no samples.
                audio_head_frames = head_trim - (1 if (window_no == 1 and is_i2v) else 0)
                head_samples = int(round(max(0, audio_head_frames) * sample_rate / fps)) if fps > 0 else 0
                final_audio = a_decoded[:, head_samples:]
                if final_audio.shape[-1] > 0:
                    # The production job appends each window's trimmed audio to
                    # the running track in order, so the chunks tile the
                    # canvas; the sidecar's per-window start_sample tracks the
                    # job's guide position (which drifts for later windows),
                    # not the assembled track, so it is not used here.
                    # Step 6 (D5): an H3 branch anchor window (h3_anchor_head)
                    # is the first sidecar window after the carried ones
                    # (window 1 itself when nothing is carried), so the cursor
                    # places its audio at the tail of the carried (committed)
                    # region; its one-frame head trim (above) drops the
                    # re-decoded shared frame's samples, so the committed
                    # contribution starts at the committed end, positionally.
                    audio_canvas_chunks.append((audio_cursor, final_audio))
                    audio_cursor += int(final_audio.shape[-1])
    except Exception as exc:
        send_cmd("error", gr.Error(f"The saved latent could not be decoded: {exc}"))
        return
    finally:
        if video_vae is not None:
            video_vae.to("cpu")
        if audio_vae is not None:
            audio_vae.to("cpu")
        del video_vae
        del audio_vae
    if aborted:
        return
    if len(window_frames_list) == 0:
        send_cmd("error", gr.Error("The saved latent does not contain any decodable window."))
        return
    frames = torch.cat(window_frames_list, dim=1) if len(window_frames_list) > 1 else window_frames_list[0]

    source_dir = os.path.dirname(str(video_source or ""))
    if len(source_dir) == 0:
        source_dir = str(ctx.save_path)
    source_stem = os.path.splitext(os.path.basename(str(video_source or "")))[0]
    if len(source_stem) == 0:
        source_stem = "latent_decode"
    container = str(ctx.server_config.get("video_container", "mp4")).lstrip(".")
    codec_type = str(ctx.server_config.get("video_output_codec", "libx264_8"))
    new_video_path = ctx.get_available_filename(source_dir, f"{source_stem}_latent_decode", force_extension=f".{container}")

    # Place the decoded chunks on a canvas the way the production job built
    # the track: window 1's trimmed audio starts at sample 0 (silence only
    # where the original output had none, as for a continuation's source
    # prefix) and each following window starts where the previous one ended.
    audio_data = None
    if len(audio_canvas_chunks) > 0:
        canvas_end = max(offset + waveform.shape[1] for offset, waveform in audio_canvas_chunks)
        video_samples = int(round(frames.shape[1] / fps * sample_rate)) if fps > 0 else 0
        canvas_length = max(video_samples, canvas_end) if video_samples > 0 else canvas_end
        canvas = torch.zeros(2, max(0, canvas_length), dtype=torch.float32)
        for offset, waveform in audio_canvas_chunks:
            offset = max(0, min(int(offset), canvas.shape[1]))
            count = min(int(waveform.shape[1]), canvas.shape[1] - offset)
            if count > 0:
                canvas[:, offset:offset + count] = waveform[:, :count]
        if video_samples > 0 and canvas.shape[1] > video_samples:
            canvas = canvas[:, :video_samples]
        if canvas.shape[1] > 0:
            audio_data = canvas.t().cpu().numpy()  # sample-major [samples, 2]
    if audio_data is not None:
        # Muxing reads the video from a separate file, so the decoded video
        # is first written to a temp path, then combined with the audio into
        # the final file.
        time_flag = int(time.time())
        temp_video_path = ctx.get_available_filename(ctx.save_path, f"tmp{time_flag}_h3_latent_decode", force_extension=f".{container}")
        temp_audio_path = ctx.get_available_filename(ctx.save_path, f"tmp{time_flag}_h3_latent_audio", force_extension=".wav")
        save_video(tensor=frames, save_file=temp_video_path, fps=fps, codec_type=codec_type, container=container, nrow=1, normalize=False, value_range=(0, 255))
        write_wav_file(temp_audio_path, audio_data, sample_rate)
        combine_and_concatenate_video_with_audio_tracks(
            new_video_path,
            temp_video_path,
            [],
            [temp_audio_path],
            0,
            sample_rate,
            new_audio_from_start=True,
            audio_codec_key=ctx.server_config.get("audio_output_codec", "aac_128"),
            verbose=ctx.verbose_level >= 2,
        )
        os.remove(temp_video_path)
        os.remove(temp_audio_path)
    else:
        save_video(tensor=frames, save_file=new_video_path, fps=fps, codec_type=codec_type, container=container, nrow=1, normalize=False, value_range=(0, 255))
    print(f"Latent decoded video saved to Path: {new_video_path}")

    configs = {"type": ctx.get_model_record("Latent Decode"), "prompt": str(entry.get("prompt") or "")}
    with ctx.lock:
        gen["file_list"].append(new_video_path)
        gen["file_settings_list"].append(configs)
    if not api_suppress_metadata_images:
        from shared.utils.video_metadata import save_video_metadata
        save_video_metadata(new_video_path, configs, None, allow_inplace_update=True, verbose_level=ctx.verbose_level)
    notifications.record_generation(ctx.server_config, gen, new_video_path, configs)
    if api_return_video_uint8 or api_return_audio:
        # Same layout as the generation flow's sample tensor ([c, f, h, w]).
        store_api_output_artifact(gen, client_id, new_video_path, "video", frames if api_return_video_uint8 else None, audio_data if api_return_audio else None, sample_rate if audio_data is not None else None, fps)
    gen["last_was_audio"] = False
    send_cmd("output")
    ctx.clear_status(state)



def _edit_latent_job_magi(ctx, send_cmd, state, video_source, client_id, plugin_data, sidecar_obj, entry, model_def, window_specs,
                          api_return_video_uint8, api_return_audio, api_suppress_metadata_images):
    """Decode a saved Magi Human latent back to pixels.

    Magi Human uses a different VAE stack than the Wan family (the Turbo
    VAED video VAE — 48 channels, the 4/1 latent layout, 16x spatial — and
    one 64-channel SA audio latent per pixel frame at 44.1 kHz), so the
    decode job takes a family-specific path: load only the two VAEs, decode
    each window the way the model's own decode does and hand the result to
    ``_save_ltx2_decoded_video``.
    """
    gen = ctx.get_gen_info(state)
    if gen.get("abort", False):
        return
    from models.magi_human.magi_human_model import SAAudioFeatureExtractor, get_turbo_vaed, resample_audio_sinc
    from models.minimax_h3.pipeline import _video_to_uint8_cpu

    fps = float(entry.get("fps") or 0) or float(model_def.get("fps") or 25)
    decode_device = "cuda" if torch.cuda.is_available() else "cpu"

    audio_entry = latent_io.audio_latent_payload(sidecar_obj)
    audio_windows = latent_io.audio_window_specs(audio_entry) if audio_entry is not None else []
    needs_audio = len(audio_windows) > 0
    sample_rate = int((audio_entry or {}).get("sample_rate") or 44100)

    # The VAE names recorded in the sidecar take precedence (they produced
    # this latent); the family defaults fall back. The Turbo VAED loads its
    # architecture from the config JSON that sits next to the checkpoint.
    turbo_default = "turbo_vae/TurboV3-Wan22-TinyShallow_7_7.safetensors"
    turbo_vae_file = str(entry.get("vae_file") or "") or turbo_default
    turbo_vae_pth = fl.locate_file(turbo_vae_file, error_if_none=False)
    if not turbo_vae_pth or not os.path.isfile(str(turbo_vae_pth)):
        send_cmd("error", gr.Error(f"The video VAE checkpoint required to decode this latent could not be located: {os.path.basename(str(turbo_vae_file))}"))
        return
    turbo_vae_json = os.path.splitext(str(turbo_vae_pth))[0] + ".json"
    if not os.path.isfile(turbo_vae_json):
        # A custom checkpoint without its config next to it: the default
        # config is the family's, the only one the loader knows.
        default_pth = fl.locate_file(turbo_default, error_if_none=False)
        turbo_vae_json = os.path.splitext(str(default_pth))[0] + ".json" if default_pth is not None else ""
        if not turbo_vae_json or not os.path.isfile(turbo_vae_json):
            send_cmd("error", gr.Error(f"The video VAE config required to decode this latent could not be located next to {os.path.basename(str(turbo_vae_file))}."))
            return

    # The sidecar's recorded dtype is the one the VAE ran with at
    # generation; decode with the same dtype (the server config is the
    # fallback for sidecars that do not record it).
    vae_dtype = _DTYPE_MAP.get(str(entry.get("vae_dtype") or ""))
    if vae_dtype is None:
        vae_dtype = torch.float16 if str(ctx.server_config.get("vae_precision", "16")) == "16" else torch.float
    stored_vae_hash = str(entry.get("vae_sha256") or "")
    if len(stored_vae_hash) > 0 and latent_io.sha256_file(str(turbo_vae_pth)) != stored_vae_hash:
        print("Latent decode identity warning: the video VAE checkpoint on disk differs from the one used at save time.")

    try:
        # Same construction as the model (the VAE loads on the CPU and the
        # decode moves the latents to the accelerator, the way the
        # generation does).
        turbo_vae = get_turbo_vaed(turbo_vae_json, str(turbo_vae_pth), device="cpu", weight_dtype=vae_dtype)
    except Exception as exc:
        send_cmd("error", gr.Error(f"The video VAE required to decode this latent could not be loaded ({exc})."))
        return

    audio_vae = None
    if needs_audio:
        # The extractor loads the whole model folder; the sidecar records
        # the checkpoint inside it, so the folder is its parent path.
        audio_vae_file = str((audio_entry or {}).get("audio_vae_file") or "stable-audio-open-1.0/model.safetensors")
        audio_folder = audio_vae_file.rsplit("/", 1)[0] if "/" in audio_vae_file else "stable-audio-open-1.0"
        audio_model_path = fl.locate_folder(audio_folder, error_if_none=False)
        if audio_model_path is None:
            send_cmd("error", gr.Error(f"The audio VAE model required to decode this latent could not be located: {audio_folder}"))
            return
        stored_audio_hash = str((audio_entry or {}).get("audio_vae_sha256") or "")
        if len(stored_audio_hash) > 0:
            audio_vae_pth = fl.locate_file(audio_vae_file, error_if_none=False)
            if audio_vae_pth is not None and os.path.isfile(str(audio_vae_pth)) and latent_io.sha256_file(str(audio_vae_pth)) != stored_audio_hash:
                print("Latent decode identity warning: the audio VAE checkpoint on disk differs from the one used at save time.")
        try:
            audio_vae = SAAudioFeatureExtractor(device="cpu", model_path=audio_model_path)
        except Exception as exc:
            send_cmd("error", gr.Error(f"The audio VAE required to decode this latent could not be loaded ({exc})."))
            return

    turbo_dtype = next(turbo_vae.parameters()).dtype
    audio_dtype = next(audio_vae.vae_model.parameters()).dtype if audio_vae is not None else None
    window_frames_list = []
    audio_waveform_chunks = []
    try:
        for idx, spec in enumerate(window_specs):
            if gen.get("abort", False):
                return
            window_no = int(spec["window_no"])
            latents = spec["latents"].to(device=decode_device, dtype=turbo_dtype)
            if latents.dim() == 4:
                # The sidecar stores the unbatched [c, t, h, w] layout; the
                # decoder takes the batched layout the generation feeds it.
                latents = latents.unsqueeze(0)
            sample_h = int(latents.shape[-2]) * 16
            sample_w = int(latents.shape[-1]) * 16
            # Tile only when the window's decoded canvas exceeds the tile
            # threshold (spatial tiling is a memory measure and its overlap
            # blending shifts the pixels, so it is used only when needed —
            # the same heuristic the generic decode path applies, scaled to
            # the Turbo VAED's 16x spatial compression).
            tile_size = 256 if (sample_h > 256 or sample_w > 256) else 0
            output_offload = tile_size > 0 or sample_h * sample_w > 832 * 480
            decoded = turbo_vae.decode(latents, output_offload=output_offload, tile_size=tile_size, temporal_chunk_size=0)
            latents = None
            if decoded is None or decoded.numel() == 0:
                if gen.get("abort", False):
                    return
                raise ValueError(f"window {window_no} could not be decoded")
            frames = decoded[0].float().clamp(-1, 1)  # [3, D, h, w]
            decoded = None
            # Re-apply the exact pixel trims of the original assembly so the
            # decoded windows re-assemble into the original video unchanged.
            head_trim = int(spec["head_trim"])
            tail_trim = int(spec["tail_trim"])
            if head_trim > 0 or tail_trim > 0:
                frames = frames[:, head_trim:max(head_trim, frames.shape[1] - tail_trim)]
            window_frames_list.append(_video_to_uint8_cpu(frames))
            frames = None
            # The audio latent is one 64-channel frame per pixel frame, so
            # the window's decoded span needs no frame-to-sample fitting:
            # the 51.2 kHz output resamples to the recorded rate with the
            # per-frame proportion kept exact, like the generation decode.
            if idx < len(audio_windows) and audio_vae is not None:
                a_latent = audio_windows[idx]["latents"]
                if a_latent.dim() == 3:
                    a_latent = a_latent[0]
                a_decoded = audio_vae.decode(a_latent.T.to(dtype=audio_dtype))  # [1, 2, samples] @ 51.2 kHz
                a_latent = None
                if a_decoded is None or a_decoded.numel() == 0:
                    raise ValueError(f"window {window_no} audio could not be decoded")
                a_waveform = a_decoded.float().squeeze(0).T.detach().cpu().numpy()  # [samples, 2]
                a_waveform = resample_audio_sinc(a_waveform, sample_rate / 51200)
                audio_waveform_chunks.append(torch.from_numpy(np.ascontiguousarray(a_waveform.T)).float())
    except Exception as exc:
        send_cmd("error", gr.Error(f"The saved latent could not be decoded: {exc}"))
        return
    finally:
        turbo_vae.to("cpu")
        if audio_vae is not None:
            audio_vae.vae_model.to("cpu")
        del turbo_vae
        if audio_vae is not None:
            del audio_vae
    if len(window_frames_list) == 0:
        send_cmd("error", gr.Error("The saved latent does not contain any decodable window."))
        return
    frames = torch.cat(window_frames_list, dim=1) if len(window_frames_list) > 1 else window_frames_list[0]
    _save_ltx2_decoded_video(ctx, send_cmd, state, video_source, client_id, plugin_data, frames, audio_waveform_chunks, sample_rate, entry, audio_entry, fps, False,
                             api_return_video_uint8, api_return_audio, api_suppress_metadata_images)


def _edit_latent_job_ovi(ctx, send_cmd, state, video_source, client_id, plugin_data, sidecar_obj, entry, model_def, window_specs,
                         api_return_video_uint8, api_return_audio, api_suppress_metadata_images):
    """Decode a saved Ovi latent back to pixels and its soundtrack.

    Ovi decodes through two independent VAEs (the video's Wan 2.2
    5B-class VAE — 48 channels, the 4/1 latent layout, 16x spatial — and
    the 16 kHz mmaudio/BigVGAN audio VAE — a 20-channel latent per audio
    step), so the job takes a family-specific path: load only the VAE
    components (no diffusion transformer), decode every window the way
    the model's own decode does, and hand the result to
    ``_save_ltx2_decoded_video`` (the decoded mono waveforms are
    assembled with the generation's per-window trims and muxed back into
    the video; a window whose audio latent runs short leaves silence at
    the tail instead of shifting the track). A sidecar without an audio
    entry decodes the video only, like the other families.
    """
    gen = ctx.get_gen_info(state)
    if gen.get("abort", False):
        return
    from models.wan.modules.vae2_2 import Wan2_2_VAE
    from models.wan.ovi_fusion_engine import init_mmaudio_vae

    fps = float(entry.get("fps") or 0) or float(model_def.get("fps") or 24)
    decode_device = "cuda" if torch.cuda.is_available() else "cpu"

    audio_entry = latent_io.audio_latent_payload(sidecar_obj)
    audio_windows = latent_io.audio_window_specs(audio_entry) if audio_entry is not None else []
    needs_audio = len(audio_windows) > 0
    sample_rate = int((audio_entry or {}).get("sample_rate") or 16000)

    # The VAE name recorded in the sidecar takes precedence (it produced
    # this latent); the family default is the fallback.
    vae_file = str(entry.get("vae_file") or "") or "Wan2.2_VAE.safetensors"
    vae_pth = fl.locate_file(vae_file, error_if_none=False)
    if not vae_pth or not os.path.isfile(str(vae_pth)):
        send_cmd("error", gr.Error(f"The VAE checkpoint required to decode this latent could not be located: {os.path.basename(str(vae_file))}"))
        return

    # The sidecar's recorded dtype is the one the VAE ran with at
    # generation; decode with the same dtype (the server config is the
    # fallback for sidecars that do not record it).
    vae_dtype = _DTYPE_MAP.get(str(entry.get("vae_dtype") or ""))
    if vae_dtype is None:
        vae_dtype = torch.float16 if str(ctx.server_config.get("vae_precision", "16")) == "16" else torch.float
    stored_vae_hash = str(entry.get("vae_sha256") or "")
    if len(stored_vae_hash) > 0 and latent_io.sha256_file(str(vae_pth)) != stored_vae_hash:
        print("Latent decode identity warning: the video VAE checkpoint on disk differs from the one used at save time.")

    try:
        # Same construction as the model (the VAE loads on the CPU and the
        # decode moves the latents to the accelerator, the way the
        # generation does).
        vae = Wan2_2_VAE(vae_pth=str(vae_pth), dtype=vae_dtype, device="cpu")
        vae.device = decode_device
        vae.model.to(decode_device)
    except Exception as exc:
        send_cmd("error", gr.Error(f"The video VAE required to decode this latent could not be loaded ({exc})."))
        return

    audio_vae = None
    if needs_audio:
        # The mmaudio VAE loads its two checkpoints (the encoder weights
        # and the BigVGAN vocoder) from the mmaudio/ folder; the sidecar
        # records the encoder's file name, and the vocoder travels with
        # it.
        audio_vae_file = str((audio_entry or {}).get("audio_vae_file") or "v1-16.pth")
        try:
            audio_vae = init_mmaudio_vae().to(decode_device).bfloat16()
        except Exception as exc:
            send_cmd("error", gr.Error(f"The audio VAE required to decode this latent's soundtrack could not be loaded ({exc})."))
            return
        stored_audio_hash = str((audio_entry or {}).get("audio_vae_sha256") or "")
        if len(stored_audio_hash) > 0:
            audio_vae_pth = fl.locate_file(os.path.join("mmaudio", audio_vae_file), error_if_none=False)
            if audio_vae_pth and os.path.isfile(str(audio_vae_pth)) and latent_io.sha256_file(str(audio_vae_pth)) != stored_audio_hash:
                print("Latent decode identity warning: the audio VAE checkpoint on disk differs from the one used at save time.")

    # Defensive: Ovi latents are SDR today, but if a sidecar ever records
    # an HDR sample, decode it in VAE range (float) and save it as tagged
    # HDR10 instead of clamping to 8 bits.
    entry_is_hdr = bool(entry.get("is_hdr"))
    window_frames_list = []
    audio_waveform_chunks = []
    try:
        for idx, spec in enumerate(window_specs):
            if gen.get("abort", False):
                return
            window_no = int(spec["window_no"])
            latents = spec["latents"].to(device=decode_device, dtype=vae_dtype)
            # Tile only when the window's decoded canvas exceeds the tile
            # threshold (spatial tiling is a memory measure and its overlap
            # blending shifts the pixels, so it is used only when needed —
            # the same heuristic the generic decode path applies, scaled to
            # the 5B VAE's 16x spatial compression).
            tile_size = 256 if (latents.shape[-1] * 16 > 256 or latents.shape[-2] * 16 > 256) else 0
            videos = vae.decode([latents], tile_size) if entry_is_hdr else vae.decode_to_cpu_uint8([latents], tile_size)
            latents = None
            if len(videos) == 0 or videos[0] is None:
                raise ValueError(f"window {window_no} could not be decoded")
            frames = videos[0]
            videos = None
            # Re-apply the exact pixel trims of the original assembly so the
            # decoded windows re-assemble into the original video unchanged.
            head_trim = int(spec["head_trim"])
            tail_trim = int(spec["tail_trim"])
            if head_trim > 0 or tail_trim > 0:
                frames = frames[:, head_trim:max(head_trim, frames.shape[1] - tail_trim)]
            window_frames_list.append(frames)
            # The audio latent is a 20-channel step per audio position; the
            # vocoder decodes it straight to the entry's 16 kHz mono rate,
            # so the waveform needs no resampling (the generation decode is
            # the same call).
            if needs_audio and idx < len(audio_windows) and audio_vae is not None:
                a_latent = audio_windows[idx]["latents"]
                if a_latent.dim() == 3:
                    a_latent = a_latent[0]
                a_decoded = audio_vae.wrapped_decode(a_latent.to(device=decode_device, dtype=torch.bfloat16).unsqueeze(0))
                a_latent = None
                if a_decoded is None or a_decoded.numel() == 0:
                    raise ValueError(f"window {window_no} audio could not be decoded")
                audio_waveform_chunks.append(a_decoded.reshape(-1).float().cpu())
                a_decoded = None
    except Exception as exc:
        send_cmd("error", gr.Error(f"The saved latent could not be decoded: {exc}"))
        return
    finally:
        vae.model.to("cpu")
        if audio_vae is not None:
            audio_vae.to("cpu")
        del vae
        if audio_vae is not None:
            del audio_vae
    if len(window_frames_list) == 0:
        send_cmd("error", gr.Error("The saved latent does not contain any decodable window."))
        return
    frames = torch.cat(window_frames_list, dim=1) if len(window_frames_list) > 1 else window_frames_list[0]
    _save_ltx2_decoded_video(ctx, send_cmd, state, video_source, client_id, plugin_data, frames, audio_waveform_chunks, sample_rate, entry, audio_entry, fps, entry_is_hdr,
                             api_return_video_uint8, api_return_audio, api_suppress_metadata_images)


def edit_latent_job(ctx, send_cmd, state, video_source, client_id="", plugin_data=None):
    """Decode a saved pre-decode latent (Media Flow 'Latent Decode') back to pixels.

    The latent companion file (``*_latent.pt``) was written next to the media
    file when 'Save Latents' was enabled; this job re-decodes it with the same
    VAE and writes a new video next to the source.
    """
    gen = ctx.get_gen_info(state)
    if gen.get("abort", False):
        return
    api_options = plugin_data.get("api", {}) if isinstance(plugin_data, dict) and isinstance(plugin_data.get("api"), dict) else {}
    api_return_video_uint8, api_return_audio = get_api_output_options(plugin_data)
    api_suppress_metadata_images = bool(api_options.get("suppress_metadata_images"))

    latent_source = str(api_options.get("latent_source") or "")
    if len(latent_source) == 0:
        latent_source = latent_io.find_latent_file(video_source) or ""
    if len(latent_source) == 0 or not os.path.isfile(latent_source):
        send_cmd("error", gr.Error("No saved latent companion file was found for this media. Latent decoding only works on media generated with the 'Save Latents' option enabled."))
        return
    try:
        sidecar_obj = latent_io.load_latent_file(latent_source)
        entry = latent_io.latent_payload(sidecar_obj)
    except Exception as exc:
        send_cmd("error", gr.Error(f"Unable to read the latent companion file {os.path.basename(latent_source)}: {exc}"))
        return
    if entry is None:
        send_cmd("error", gr.Error(f"{os.path.basename(latent_source)} is not a WanGP latent companion file."))
        return
    # The latent decode reads the all-latent layout only (every kept frame
    # stored as latents): a sidecar in any other layout cannot be
    # faithfully decoded, so it is refused up front with the one generic
    # message instead of writing a subtly wrong video.
    if not latent_io.is_all_latent_entry(entry):
        send_cmd("error", gr.Error(latent_io.non_current_latent_message(f"{os.path.basename(latent_source)}")))
        return

    for warning in latent_io.identity_warnings(entry, {"app_version": ctx.WanGP_version, "model_type": "", "base_model_type": "", "vae_sha256": ""}):
        print(f"Latent decode identity warning: {warning}")
    # A flagged sidecar (its recorded length differs from the written
    # video's length) still decodes: say up front that the decoded video
    # will have the file's recorded length, not the neighboring video's.
    _mismatch = latent_io.sidecar_length_mismatch(entry)
    if _mismatch is not None:
        print(f"Latent decode: the file's recorded length ({_mismatch[0]} frames) differs from the video it was written for "
              f"({_mismatch[1]} frames); the decoded video will have the file's recorded length, not the neighboring video's.")

    model_type = str(entry.get("model_type") or "")
    model_def = ctx.get_model_def(model_type) if len(model_type) > 0 else None
    if model_def is None:
        send_cmd("error", gr.Error(f"The model '{model_type}' that produced this latent is not available in this build."))
        return

    window_specs = latent_io.entry_window_specs(entry)
    if not window_specs:
        send_cmd("error", gr.Error(f"{os.path.basename(latent_source)} does not contain any readable latent windows."))
        return
    if len(window_specs) > 1:
        print(f"Latent decode: reconstructing {len(window_specs)} windows from {os.path.basename(latent_source)}.")

    if _is_h3_family(str(entry.get("base_model_type") or ""), model_def):
        _edit_latent_job_h3(ctx, send_cmd, state, video_source, client_id, plugin_data, sidecar_obj, entry, model_def, window_specs,
                            api_return_video_uint8, api_return_audio, api_suppress_metadata_images)
        return
    if _is_ltx2_family(str(entry.get("base_model_type") or ""), model_def):
        _edit_latent_job_ltx2(ctx, send_cmd, state, video_source, client_id, plugin_data, sidecar_obj, entry, model_def, window_specs,
                              api_return_video_uint8, api_return_audio, api_suppress_metadata_images)
        return
    if _is_magi_human_family(str(entry.get("base_model_type") or ""), model_def):
        _edit_latent_job_magi(ctx, send_cmd, state, video_source, client_id, plugin_data, sidecar_obj, entry, model_def, window_specs,
                              api_return_video_uint8, api_return_audio, api_suppress_metadata_images)
        return
    if _is_ovi_family(str(entry.get("base_model_type") or ""), model_def):
        _edit_latent_job_ovi(ctx, send_cmd, state, video_source, client_id, plugin_data, sidecar_obj, entry, model_def, window_specs,
                             api_return_video_uint8, api_return_audio, api_suppress_metadata_images)
        return

    # Resolve the VAE the same way the model loader does (the stored VAE name
    # takes precedence over the current model definition, so a latent keeps
    # decoding with the VAE that produced it).
    from models.wan.modules.vae import WanVAE
    from models.wan.modules.vae2_2 import Wan2_2_VAE
    vae_candidate = None
    vae_urls = model_def.get("VAE_URLs", None)
    if isinstance(vae_urls, str) and len(vae_urls) > 0:
        vae_candidate = vae_urls
    elif isinstance(vae_urls, list) and len(vae_urls) > 0:
        vae_urls = [u for u in vae_urls if str(u).endswith((".pt", ".pth", ".safetensors"))]
        vae_candidate = vae_urls[0] if len(vae_urls) > 0 else None
    if vae_candidate is None:
        vae_candidate = "Wan2.2_VAE.safetensors" if model_def.get("wan_5B_class", False) else "Wan2.1_VAE.safetensors"
    stored_vae_file = str(entry.get("vae_file") or "")
    if len(stored_vae_file) > 0 and os.path.basename(str(vae_candidate)) != stored_vae_file:
        vae_candidate = stored_vae_file
    # The 2.2 (5B) VAE is a 16x-spatial checkpoint and the 2.1 VAE is 8x, and
    # the two classes have different architectures: pick the class from the
    # checkpoint actually loaded (its recorded name) so a latent keeps decoding
    # with the VAE that produced it, even if the current model definition no
    # longer matches it; the model definition is the fallback for custom VAE
    # names that do not identify their generation. LongCat rides this path
    # too: its sidecar stores the loop's normalized latents (the Wan family
    # convention, the same Wan 2.1 VAE), so the wrapper's built-in
    # latents_mean/latents_std rescale is the exact inverse of the writer's
    # denormalize and no separate family branch is needed.

    vae_basename = os.path.basename(str(vae_candidate)).lower()
    if "2.2" in vae_basename:
        vae_class = Wan2_2_VAE
    elif "2.1" in vae_basename:
        vae_class = WanVAE
    else:
        vae_class = Wan2_2_VAE if model_def.get("wan_5B_class", False) else WanVAE
    vae_scale = 16 if vae_class is Wan2_2_VAE else 8
    vae_pth = fl.locate_file(vae_candidate, error_if_none=False)
    if not os.path.isfile(vae_pth):
        send_cmd("error", gr.Error(f"The VAE checkpoint required for this latent is missing: {os.path.basename(str(vae_candidate))}"))
        return
    stored_vae_hash = str(entry.get("vae_sha256") or "")
    if len(stored_vae_hash) > 0:
        current_vae_hash = latent_io.sha256_file(vae_pth)
        if current_vae_hash != stored_vae_hash:
            print("Latent decode identity warning: the VAE checkpoint on disk differs from the one used at save time.")

    # The sidecar records the dtype the VAE ran with at generation; decode
    # with the same dtype (decoding with a different precision than the
    # generation changes the decoded pixels slightly). The server config is
    # the fallback for sidecars that predate the field.
    _recorded_dtype = _DTYPE_MAP.get(str(entry.get("vae_dtype") or ""))
    vae_dtype = _recorded_dtype if _recorded_dtype is not None else (
        torch.float16 if str(ctx.server_config.get("vae_precision", "16")) == "16" else torch.float)
    vae = vae_class(vae_pth=vae_pth, dtype=vae_dtype, device="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae.device = device
    vae.model.to(device)
    # Defensive: Wan/LongCat latents are SDR today, but if a sidecar ever
    # records an HDR sample, decode it in VAE range (float) and save it as
    # tagged HDR10 instead of clamping to 8 bits.
    entry_is_hdr = bool(entry.get("is_hdr"))
    window_frames_list = []
    try:
        for spec in window_specs:
            latents = spec["latents"].to(device=device, dtype=vae_dtype)
            # Tile only when the window's decoded canvas exceeds the tile
            # threshold (spatial tiling is a memory measure and its overlap
            # blending shifts the pixels, so it is used only when needed).
            # The threshold is in pixel space, so it scales with the VAE's
            # spatial compression (8x for the 2.1 VAE, 16x for the 2.2 / 5B VAE).
            tile_size = 256 if (latents.shape[-1] * vae_scale > 256 or latents.shape[-2] * vae_scale > 256) else 0
            videos = vae.decode([latents], tile_size) if entry_is_hdr else vae.decode_to_cpu_uint8([latents], tile_size)
            latents = None
            if len(videos) == 0 or videos[0] is None:
                raise ValueError(f"window {spec['window_no']} could not be decoded")
            frames = videos[0]
            videos = None
            # Re-apply the exact pixel trims of the original assembly so the
            # decoded windows re-assemble into the original video unchanged.
            head_trim = int(spec["head_trim"])
            tail_trim = int(spec["tail_trim"])
            if head_trim > 0 or tail_trim > 0:
                frames = frames[:, head_trim:max(head_trim, frames.shape[1] - tail_trim)]
            window_frames_list.append(frames)
    except Exception as exc:
        send_cmd("error", gr.Error(f"The saved latent could not be decoded: {exc}"))
        return
    finally:
        vae.model.to("cpu")
        del vae
    if len(window_frames_list) == 0:
        send_cmd("error", gr.Error("The saved latent does not contain any decodable window."))
        return
    frames = torch.cat(window_frames_list, dim=1) if len(window_frames_list) > 1 else window_frames_list[0]
    fps = float(entry.get("fps") or 0) or float(model_def.get("fps") or 24)

    source_dir = os.path.dirname(str(video_source or ""))
    if len(source_dir) == 0:
        source_dir = str(ctx.save_path)
    source_stem = os.path.splitext(os.path.basename(str(video_source or "")))[0]
    if len(source_stem) == 0:
        source_stem = "latent_decode"
    container = str(ctx.server_config.get("video_container", "mp4")).lstrip(".")
    codec_type = str(ctx.server_config.get("video_output_codec", "libx264_8"))
    new_video_path = ctx.get_available_filename(source_dir, f"{source_stem}_latent_decode", force_extension=f".{container}")
    if entry_is_hdr:
        # Tagged 10-bit HEVC HDR10 file, like the generation flow: the decoded
        # VAE-range floats go straight into the HDR pipe without clamping.
        save_hdr_video(tensor=frames, save_file=new_video_path, fps=fps, codec_type=ctx.server_config.get("hdr_video_crf", 8), container=container)
    else:
        save_video(tensor=frames, save_file=new_video_path, fps=fps, codec_type=codec_type, container=container, nrow=1, normalize=False, value_range=(0, 255))
    print(f"Latent decoded video saved to Path: {new_video_path}")

    configs = {"type": ctx.get_model_record("Latent Decode"), "prompt": str(entry.get("prompt") or "")}
    with ctx.lock:
        gen["file_list"].append(new_video_path)
        gen["file_settings_list"].append(configs)
    if not api_suppress_metadata_images:
        from shared.utils.video_metadata import save_video_metadata
        save_video_metadata(new_video_path, configs, None, allow_inplace_update=True, verbose_level=ctx.verbose_level)
    notifications.record_generation(ctx.server_config, gen, new_video_path, configs)
    if api_return_video_uint8 or api_return_audio:
        store_api_output_artifact(gen, client_id, new_video_path, "video", frames, None, None, fps)
    gen["last_was_audio"] = False
    send_cmd("output")
    ctx.clear_status(state)


def validate_latent_decode(inputs):
    """Validate a *Latent Decode* job's inputs (moved from ``wgp``'s
    ``validate_task``); returns the error message, or "" when the job is
    runnable."""
    video_source = inputs.get("video_source")
    if not media_source_exists(video_source):
        return "Selected video or image file is missing"
    if not has_video_file_extension(video_source):
        return "Latent decoding is only available with Videos"
    if latent_io.find_latent_file(video_source) is None:
        return "No saved latent companion file was found next to this video. Latent decoding only works on media generated with the 'Save Latents' option enabled."
    return ""


def compute_fit_canvas(model_def, server_config, outpainting_dims):
    """The job's fit_canvas / fit_crop pair (moved from ``wgp``'s
    ``generate_media``): *lock resolutions* models force the canvas fit,
    and an outpainting job never fits (its canvas is the outpainting box)."""
    fit_canvas = 2 if model_def.get("lock_resolutions", False) else server_config.get("fit_canvas", 0)
    fit_crop = fit_canvas == 2
    if fit_crop and outpainting_dims is not None:
        fit_crop = False
        fit_canvas = 0
    return fit_canvas, fit_canvas == 2
