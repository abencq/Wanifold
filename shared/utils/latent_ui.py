# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""The generate-form callbacks of the *From latent* (branch) mode, moved out of
``wgp.py`` (the latent refactor, phase 3).

The branch components' event handlers of the Generate tab: the *Location*
radio handler (which also owns the branch components' visibility and
labels), the *From latent* checkbox / source-video handler, the *Round
branch frame* snap, the latent-file picker handler (the companion video's
auto-select), the source-video label refresh, the label-base helpers and
the shared success tail (the field snap, the prompt / seed pre-fill and the
merged info box).

The module does not import ``wgp``: the app's model lookups and the label
helpers enter by injection through :class:`UiContext`, and the engine calls
(branch resolution, the family detection, the label constants) go through
``shared.utils.latent_jobs`` with the app's job context. ``wgp.py`` builds one :class:`BranchUi`
controller and wires its methods to the components' events (the
``create_form_controller`` shape of ``plugins/media_flow``).
"""

import os
from dataclasses import dataclass
from typing import Callable

import gradio as gr

from shared.utils import latent_io, latent_jobs
from shared.utils.utils import add_to_sequence, del_in_sequence, filter_letters
from shared.utils.latent_jobs import (
    BRANCH_FRAMES_VIDEO_SOURCE_LABEL,
    BRANCH_LATENT_FILE_LABEL,
    COMPANION_VIDEO_LABEL,
    DETECTED_COMPANION_VIDEO_LABEL,
    DETECTED_LATENT_FILE_LABEL,
    KEEP_FRAMES_VIDEO_SOURCE_LABEL,
    VIDEO_SOURCE_LABEL,
)


@dataclass(frozen=True)
class UiContext:
    """The ``wgp.py`` state the branch-form callbacks read (injected).

    The callables are ``wgp`` module functions: the model lookups, the two
    *Location*-dependent visibility helpers and the video picker's label
    info helper (``video_input_label_with_info`` stays in ``wgp.py`` — it
    is shared with the non-latent label refreshes).
    """
    get_model_def: Callable
    get_state_model_type: Callable
    get_base_model_type: Callable
    end_frames_option_visible: Callable
    input_video_strength_visible: Callable
    video_input_label_with_info: Callable


class BranchUi:
    """The form callbacks of the *From latent* (branch) mode.

    ``wgp.py`` builds one controller and wires its methods to the branch
    components' events (``branch_ui.refresh_branch_from_latent`` and the
    like). The controller carries the two contexts the callbacks read:
    the form helpers (:class:`UiContext`) and the app's job context
    (the ``latent_jobs.JobContext`` the engine functions take — the same
    object ``wgp.py`` passes to the job calls, so the form validation
    and the job run resolve the branch identically).
    """

    def __init__(self, ctx: UiContext, jobs_ctx: latent_jobs.JobContext) -> None:
        self._ctx = ctx
        self._jobs = jobs_ctx

    def refresh_image_prompt_type_radio(self, state, image_prompt_type, image_prompt_type_radio, video_prompt_type, video_source, branch_from_latent, branch_fit_latent, branch_latent_file):
        image_prompt_type = del_in_sequence(image_prompt_type, "VLTS")
        image_prompt_type = add_to_sequence(image_prompt_type, image_prompt_type_radio)
        any_video_source = len(filter_letters(image_prompt_type, "VL"))>0
        model_def = self._ctx.get_model_def(self._ctx.get_state_model_type(state))
        end_visible = self._ctx.end_frames_option_visible(model_def, image_prompt_type)
        input_strength_visible = self._ctx.input_video_strength_visible(model_def, image_prompt_type, video_prompt_type)
        # The branch mode persists across every Location change: the latent
        # file works with either Continue location, and the mode / file the
        # user picked are not a Location setting — so the components' values
        # are never written here, only their visibility and labels (re-writing the
        # checkbox's value or clearing the file would silently lose the
        # picked mode, and the next validation would then run the classic
        # video-to-sidecar flow on a different file than the one the user
        # picked). Two conditions hide the components: outside the Continue
        # domain the mode sleeps, and for models whose family does not write
        # latent files it is dormant (hidden, value preserved) — the job
        # pipeline (queue validation, the job-start re-resolution, the job
        # loop, the implicit Save Latents conversion) is gated on the same
        # conditions, so a job started there is neither refused by nor
        # affected by the hidden mode, and the setup is intact when the user
        # comes back. The video label goes back to the legacy one whenever
        # the mode does not apply.
        branch_available = any_video_source and latent_jobs._is_all_latent_family(self._jobs, self._ctx.get_base_model_type(self._ctx.get_state_model_type(state)), model_def)
        branch_on = branch_available and bool(branch_from_latent)
        # The label bases switch with the value's origin (the "Detected..."
        # variant names the value the app itself selected): this handler never
        # writes values, so it only reads the origin markers the branch
        # handlers maintain.
        _vsrc = str(video_source or "").strip()
        _video_auto = isinstance(state, dict) and len(_vsrc) > 0 and state.get("branch_video_origin") == "auto"
        video_base = (DETECTED_COMPANION_VIDEO_LABEL if _video_auto else COMPANION_VIDEO_LABEL) if branch_on else VIDEO_SOURCE_LABEL
        _fsrc = str(branch_latent_file or "").strip()
        _file_auto = isinstance(state, dict) and len(_fsrc) > 0 and state.get("branch_latent_origin") == "auto"
        if branch_available:
            branch_checkbox_update = gr.update(visible = True)
            branch_fit_update = gr.update(visible = branch_on)
            branch_file_update = gr.update(visible = branch_on, label = DETECTED_LATENT_FILE_LABEL if (branch_on and _file_auto) else BRANCH_LATENT_FILE_LABEL)
        else:
            branch_checkbox_update = gr.update(visible = False)
            branch_fit_update = gr.update(visible = False)
            branch_file_update = gr.update(visible = False, label = BRANCH_LATENT_FILE_LABEL)
        return image_prompt_type, gr.update(visible = "S" in image_prompt_type ), gr.update(visible = end_visible and ("E" in image_prompt_type) ), gr.update(visible = "V" in image_prompt_type, label = self._ctx.video_input_label_with_info(video_base, video_source)) , gr.update(visible = input_strength_visible), gr.update(visible = any_video_source), branch_checkbox_update, branch_fit_update, branch_file_update, gr.update(visible = end_visible)

    def _branch_video_label_base(self, state, video_value):
        """The video picker's label base in the *From latent* mode: the
        "Detected companion..." variant when the picker's current value is
        the one the app itself selected (the origin marker the latent-file
        handler sets when it auto-selects the companion), the plain
        "Companion..." base whenever the user selected it (or nothing is
        selected yet — nothing selected is never "detected")."""
        _src = str(video_value or "").strip()
        if len(_src) == 0:
            return COMPANION_VIDEO_LABEL
        auto = isinstance(state, dict) and state.get("branch_video_origin") == "auto"
        return DETECTED_COMPANION_VIDEO_LABEL if auto else COMPANION_VIDEO_LABEL


    def _branch_latent_label_base(self, state, latent_value):
        """The latent file picker's label base: the mirror of
        :func:`_branch_video_label_base` for the backward direction — the
        "Detected latent file..." variant when the picker's value is the one
        the app matched to a selected video, the plain "Select..." base
        whenever the user picked it (or nothing is picked yet)."""
        _src = str(latent_value or "").strip()
        if len(_src) == 0:
            return BRANCH_LATENT_FILE_LABEL
        auto = isinstance(state, dict) and state.get("branch_latent_origin") == "auto"
        return DETECTED_LATENT_FILE_LABEL if auto else BRANCH_LATENT_FILE_LABEL


    def _branch_success_updates(self, info, keep_frames_video_source, branch_fit_latent, prompt, seed, no_companion=False, advisory="", state=None, selected_video=""):
        """The common success tail of the *From latent* handlers (the checkbox
        and the latent-file picker): snaps the field up to the latent boundary
        when *Round branch frame* is on (the empty field is the field's own
        "branch at the end" preset), pre-fills the prompt / seed (only when
        empty / -1), and prints one merged info box: the resolved file with
        its recorded total (which bounds the branch-point field) and the
        branch point, explained in plain terms (the mid-latent cut that
        starts earlier than the typed frame, the rounded-up cut, or the
        on-boundary cut). With ``no_companion`` set (no companion video
        present) the no-companion note is merged into the same box (one box
        instead of two), and the separate no-companion warning is never
        printed on its own. The box (and its console mirror) is re-emitted
        only when the resolved message actually changes: repeated
        re-validations (the companion picker's reapplications included) are
        deduplicated with a per-session memo, cleared when the mode turns
        off. ``selected_video`` (the effective source video in the backward
        direction — the video is the input and no explicit file was picked)
        adds a console-only direction note naming the video the latent file
        was matched from (the merged box already names the file); it is part
        of the memo, so it prints once per resolved state like the box.
        Returns the (possibly snapped) field value and the pre-filled
        prompt / seed."""
        branch = info["branch"]
        total = int(branch["total_frames"])
        kept = int(branch["kept_frames"])
        asked = int(info["branch_frame"])
        new_keep = keep_frames_video_source
        if branch_fit_latent:
            if kept > total:
                # The clamped cut is the end of the video; the empty field is
                # the field's own "branch at the end" preset.
                new_keep = ""
            elif str(keep_frames_video_source or "").strip() != str(kept):
                new_keep = str(kept)
        new_prompt = info["prefill_prompt"] if len(str(prompt or "").strip()) == 0 and len(info["prefill_prompt"]) > 0 else prompt
        try:
            seed_value = int(seed)
        except (TypeError, ValueError):
            seed_value = -1
        new_seed = info["prefill_seed"] if seed_value == -1 and info["prefill_seed"] != 0 else seed
        if branch.get("mid_latent"):
            detail = (f"Branch point: frame {kept} of {total}. You asked for frame {asked}, "
                      f"which falls inside the next latent block, so the branch starts {asked - kept} frames earlier, "
                      f"at frame {kept}; the first new latent is the block that contains frame {asked}.")
        elif branch.get("fit") and kept != asked:
            detail = f"Branch point: frame {kept} of {total}. The frame you typed ({asked}) rounds up to the latent block boundary at frame {kept}."
        else:
            detail = f"Branch point: frame {kept} of {total}, exactly on a latent block boundary."
        if no_companion:
            detail = (f"No companion video was found next to the latent file, so the branch point "
                      f"is taken from the {total} frames recorded in the file. {detail}")
        merged = f"From latent: using saved latent file {info['path']} ({total} frames). {detail}"
        if advisory:
            merged = f"{merged} {advisory}"
        # The tail runs on every event that re-validates the branch (the
        # companion picker's reapplications included); report the boxes only
        # when the resolved state actually changed. The no-companion note
        # stays merged into the box above (one box instead of two) and a
        # stale raw no-companion warning is never printed on its own (several
        # candidates may have been listed after the fact).
        warnings = tuple(f"From latent: {warning}" for warning in info["warnings"]
                         if not str(warning).startswith("no companion video was found"))
        _sv = str(selected_video or "").strip()
        direction_note = f"From latent: source video {_sv}; matched saved latent file {info['path']} ({total} frames)." if len(_sv) > 0 else ""
        memo_key = (merged, warnings, direction_note)
        if isinstance(state, dict) and state.get("branch_success_memo") == memo_key:
            return new_keep, new_prompt, new_seed
        if isinstance(state, dict):
            state["branch_success_memo"] = memo_key
        if len(direction_note) > 0:
            print(direction_note, flush=True)
        gr.Info(merged)
        print(merged, flush=True)
        for note in warnings:
            gr.Info(note)
            print(note, flush=True)
        return new_keep, new_prompt, new_seed


    def refresh_branch_from_latent(self, state, image_prompt_type, image_prompt_type_radio, video_source, branch_from_latent, keep_frames_video_source, branch_fit_latent, prompt, seed, branch_latent_file):
        """The *From latent* checkbox / video-picker change handler: validates
        the branch job (latent file present / same model family / branch point
        in range), pre-fills the prompt and seed from the latent file (only
        when the boxes are empty / -1) and relabels the branch-point field.
        The branch point resolves to the latent boundary before the latent
        containing the entered frame (that latent is the branch's first new
        latent) by default, or to the boundary after it with the *Round
        branch frame* checkbox, which also snaps the field up front so the
        value it shows is the exact cut. With the mode on the two pickers'
        labels switch with the value's origin (the "Detected..." variant
        names the picker the app itself filled; see the label constants). In
        the backward direction (a selected video, no explicit file) the
        resolved sidecar is mirrored into the file picker and a console note
        names the pair, so the detection shows up in the UI as it does in the
        forward direction (a picked file finds its companion video). The soft
        note (validation pending, not an error) is shown only when neither a
        source video nor a latent file is selected yet — the branch point
        resolves from the latent file's recorded frames, so a picked file
        validates the branch on its own. Raises a visible error when the
        combination is invalid."""
        _vsrc = str(video_source or "").strip()
        _fsrc = str(branch_latent_file or "").strip()
        if len(filter_letters(str(image_prompt_type or ""), "VL")) == 0 or not branch_from_latent:
            # Mode off: drop the origin / re-assert markers so re-enabling the
            # mode re-detects and pushes the pickers' values once more (see
            # the success path below), and the success-report memo so a later
            # selection reports again.
            if isinstance(state, dict):
                state.pop("branch_video_asserted", None)
                state.pop("branch_latent_asserted", None)
                state.pop("branch_video_origin", None)
                state.pop("branch_latent_origin", None)
                state.pop("branch_success_memo", None)
            if len(filter_letters(str(image_prompt_type or ""), "VL")) == 0:
                return gr.update(value = False, visible = False), gr.update(label = KEEP_FRAMES_VIDEO_SOURCE_LABEL), gr.update(visible = False), prompt, seed, gr.update(visible = False, label = BRANCH_LATENT_FILE_LABEL), gr.update(label = self._ctx.video_input_label_with_info(VIDEO_SOURCE_LABEL, video_source))
            return gr.update(value = False, visible = True), gr.update(label = KEEP_FRAMES_VIDEO_SOURCE_LABEL), gr.update(visible = False), prompt, seed, gr.update(visible = False, label = BRANCH_LATENT_FILE_LABEL), gr.update(label = self._ctx.video_input_label_with_info(VIDEO_SOURCE_LABEL, video_source))
        model_type = self._ctx.get_state_model_type(state)
        model_def = self._ctx.get_model_def(model_type)
        if not latent_jobs._is_all_latent_family(self._jobs, self._ctx.get_base_model_type(model_type), model_def):
            # The family does not write latent files: the mode is dormant
            # (treated as off) — the components hide, the stored value is
            # preserved (no value write), the classic labels stay, and
            # nothing is resolved (never a refusal).
            if isinstance(state, dict):
                state.pop("branch_video_asserted", None)
                state.pop("branch_latent_asserted", None)
                state.pop("branch_video_origin", None)
                state.pop("branch_latent_origin", None)
                state.pop("branch_success_memo", None)
            return gr.update(visible = False), gr.update(label = KEEP_FRAMES_VIDEO_SOURCE_LABEL), gr.update(visible = False), prompt, seed, gr.update(visible = False, label = BRANCH_LATENT_FILE_LABEL), gr.update(label = self._ctx.video_input_label_with_info(VIDEO_SOURCE_LABEL, video_source))
        # The origin of the incoming video value: a change event carrying the
        # value the server itself pushed (the retrigger of the latent-file
        # handler's companion auto-select) keeps the "auto" origin; any other
        # value is a user selection.
        _was_asserted = isinstance(state, dict) and len(_vsrc) > 0 and state.get("branch_video_asserted") == _vsrc
        if isinstance(state, dict):
            if len(_vsrc) > 0:
                state["branch_video_origin"] = "auto" if _was_asserted else "user"
            else:
                state.pop("branch_video_origin", None)
        info, error = latent_jobs.resolve_latent_branch(self._jobs, image_prompt_type, video_source, state, model_type, keep_frames=keep_frames_video_source, fit=bool(branch_fit_latent), base_model_type=self._ctx.get_base_model_type(model_type), model_def=model_def, latent_file=branch_latent_file)
        if error:
            if error == "no source video was found for the branch job" and len(_vsrc) == 0 and len(_fsrc) == 0:
                note = "From latent: select a source video or a latent file first; the branch point is validated once one is selected"
                gr.Info(note)
                print(note, flush=True)
                return gr.update(value = True, visible = True), gr.update(label = BRANCH_FRAMES_VIDEO_SOURCE_LABEL), gr.update(visible = True), prompt, seed, gr.update(visible = True, label = BRANCH_LATENT_FILE_LABEL), gr.update(label = self._ctx.video_input_label_with_info(COMPANION_VIDEO_LABEL, video_source))
            raise gr.Error(f"From latent: {error}")
        entry = info.get("entry")
        # The backward direction (a selected video, no explicit file): the
        # direction note (console only) names the video the match was
        # determined from. In *Continue Last Video* the video is an implicit
        # queue reference, so the mirror and the note stay off — the
        # job-start log reports the resolved pair instead.
        _v_mode = "V" in str(image_prompt_type or "")
        new_keep, new_prompt, new_seed = self._branch_success_updates(info, keep_frames_video_source, branch_fit_latent, prompt, seed, no_companion=not bool(info.get("video_present")), advisory=latent_jobs._branch_prefix_advisory(self._jobs, model_type), state=state, selected_video=info.get("video", "") if (_v_mode and len(_fsrc) == 0) else "")
        # Re-assert the picker's value alongside the label (once per value per
        # session): the value update may originate in the latent-file handler,
        # and when it is lost in client-side event processing this retrigger
        # (picker change) resends it through the same pipeline. Gradio caches
        # files by content hash, so re-postprocessing the same file converges
        # to the identical payload and cannot loop.
        video_label = self._ctx.video_input_label_with_info(self._branch_video_label_base(state, video_source), video_source)
        video_update = gr.update(label = video_label)
        if len(_vsrc) > 0 and (not isinstance(state, dict) or not _was_asserted):
            if isinstance(state, dict):
                state["branch_video_asserted"] = _vsrc
            video_update = gr.update(value = video_source, label = video_label)
        # The latent file picker: with no explicit file the resolved sidecar
        # is mirrored into it (the backward direction — the video is the
        # input and the file is what the app detected), labelled "Detected"
        # so the user sees which side of the pair the app filled. A picked
        # file keeps its value; its label picks up the origin-aware base and
        # the file's own recorded info.
        if _v_mode and len(_fsrc) == 0:
            if isinstance(state, dict):
                state["branch_latent_asserted"] = info["path"]
                state["branch_latent_origin"] = "auto"
            file_update = gr.update(value = info["path"], label = self.latent_file_label_with_info(DETECTED_LATENT_FILE_LABEL, entry))
        else:
            # The label's info comes from the resolved file, but is only shown
            # when the picker actually holds a file (in *Continue Last Video*
            # the picker stays empty, so the label stays plain there).
            file_update = gr.update(label = self.latent_file_label_with_info(self._branch_latent_label_base(state, branch_latent_file), entry if len(_fsrc) > 0 else None))
        return gr.update(value = True, visible = True), gr.update(label = BRANCH_FRAMES_VIDEO_SOURCE_LABEL, value = new_keep), gr.update(visible = True), new_prompt, new_seed, file_update, video_update


    def refresh_branch_frame_fit(self, state, image_prompt_type, video_source, branch_from_latent, branch_fit_latent, keep_frames_video_source, branch_latent_file):
        """The *Round branch frame* handler of the branch-point field: with
        *From latent* and the fit checkbox enabled, an entered frame is snapped
        up to the latent boundary the cut clamps to, so the value shown in the
        field is the exact cut (a clamped value past the video end becomes the
        empty "branch at the end" preset). Out-of-range values surface the same
        error as the branch validation; the field is left untouched only when
        neither a selected video nor a picked latent file is present yet (the
        snap comes from the branch resolution, which works from the file alone)."""
        if not branch_fit_latent or not branch_from_latent:
            return keep_frames_video_source
        if len(filter_letters(str(image_prompt_type or ""), "VL")) == 0:
            return keep_frames_video_source
        keep_text = str(keep_frames_video_source or "").strip()
        if len(keep_text) == 0:
            return keep_frames_video_source
        model_type = self._ctx.get_state_model_type(state)
        if not latent_jobs._is_all_latent_family(self._jobs, self._ctx.get_base_model_type(model_type), self._ctx.get_model_def(model_type)):
            # Dormant family: the snap is a no-op (never a refusal).
            return keep_frames_video_source
        info, error = latent_jobs.resolve_latent_branch(self._jobs, image_prompt_type, video_source, state, model_type, keep_frames=keep_text, fit=True, base_model_type=self._ctx.get_base_model_type(model_type), model_def=self._ctx.get_model_def(model_type), latent_file=branch_latent_file)
        if error:
            if error == "no source video was found for the branch job" and len(str(video_source or "").strip()) == 0 and len(str(branch_latent_file or "").strip()) == 0:
                return keep_frames_video_source
            raise gr.Error(f"From latent: {error}")
        branch = info["branch"]
        snapped = int(branch["kept_frames"])
        if snapped > int(branch["total_frames"]):
            note = f"From latent: the branch frame {keep_text} rounds up to the last latent of the source latent file, so the cut is the end of the video."
            gr.Info(note)
            print(note, flush=True)
            return ""
        if snapped != int(keep_text):
            note = f"From latent: the branch frame {keep_text} rounds up to the latent boundary at frame {snapped} of {branch['total_frames']}."
            gr.Info(note)
            print(note, flush=True)
            return str(snapped)
        return keep_frames_video_source


    def refresh_branch_from_latent_file(self, state, image_prompt_type, branch_latent_file, video_source, branch_from_latent, keep_frames_video_source, branch_fit_latent, prompt, seed):
        """The latent-file picker's change handler: the file is the primary
        input of the *From latent* mode, so with the mode on and a file picked
        this validates the branch from the file alone (the same validation as
        the checkbox handler — the companion video is optional) and then looks
        the companion video up in the other direction
        (:func:`latent_io.find_video_for_latent`): exactly one candidate that
        is not already selected is auto-selected into the video picker (the
        retrigger that follows is recognised by the re-assert marker and
        skipped, so there is no loop) with the path reported in the note;
        several candidates are listed and the user picks through the normal
        video control (no auto-select); none is merged into the branch-point
        detail (the branch proceeds from the file's records). A change event
        carrying back the file the video handler just mirrored (the retrigger
        of the backward direction) is recognised the same way and does
        nothing — no re-report, no re-resolution, no value re-push. On the
        success path the file's label picks up its origin-aware base and the
        file's own recorded info; the success tail (prompt / seed pre-fill,
        field relabel, fit snap, branch-point details) is the shared helper.
        With the mode off or no file the video and file labels are restored
        and everything else is left unchanged (no error)."""
        # The success-report memo is dropped so a (re)selection reports again;
        # the origin markers survive (the labels depend on them).
        if isinstance(state, dict):
            state.pop("branch_success_memo", None)
        if len(filter_letters(str(image_prompt_type or ""), "VL")) == 0:
            if isinstance(state, dict):
                state.pop("branch_video_asserted", None)
                state.pop("branch_latent_asserted", None)
                state.pop("branch_video_origin", None)
                state.pop("branch_latent_origin", None)
            return video_source, keep_frames_video_source, branch_fit_latent, prompt, seed, gr.update()
        if not branch_from_latent or len(str(branch_latent_file or "").strip()) == 0:
            # Mode off, or the file was cleared: drop the origin / re-assert
            # markers so a later (re)selection re-detects, and the labels go
            # back to the plain ones.
            if isinstance(state, dict):
                state.pop("branch_video_asserted", None)
                state.pop("branch_latent_asserted", None)
                state.pop("branch_video_origin", None)
                state.pop("branch_latent_origin", None)
            # The label switches with the mode (not with the file): the mode
            # is on, so the video picker stays labelled as the (optional)
            # companion video whatever the file picker holds.
            base = COMPANION_VIDEO_LABEL if branch_from_latent else VIDEO_SOURCE_LABEL
            return gr.update(label = self._ctx.video_input_label_with_info(base, video_source)), keep_frames_video_source, branch_fit_latent, prompt, seed, gr.update(label = BRANCH_LATENT_FILE_LABEL)
        model_type = self._ctx.get_state_model_type(state)
        if not latent_jobs._is_all_latent_family(self._jobs, self._ctx.get_base_model_type(model_type), self._ctx.get_model_def(model_type)):
            # Dormant family: no resolution; the video keeps its classic
            # label and the field / fit stay untouched (never a refusal).
            return gr.update(label = self._ctx.video_input_label_with_info(VIDEO_SOURCE_LABEL, video_source)), keep_frames_video_source, branch_fit_latent, prompt, seed, gr.update()
        _fsrc = str(branch_latent_file or "").strip()
        # The origin of the incoming file value: a change event carrying the
        # value the server itself pushed (the retrigger of the video handler's
        # backward auto-fill) is not a user selection — the branch was already
        # reported there, so nothing is re-resolved or re-pushed (the labels
        # stay).
        _was_asserted = isinstance(state, dict) and len(_fsrc) > 0 and state.get("branch_latent_asserted") == _fsrc
        if isinstance(state, dict):
            state["branch_latent_origin"] = "auto" if _was_asserted else "user"
        if _was_asserted:
            # The label switches with the mode (not with the file): the mode
            # is on, so the video picker stays labelled as the (optional)
            # companion video whatever the file picker holds.
            return gr.update(label = self._ctx.video_input_label_with_info(self._branch_video_label_base(state, video_source), video_source)), keep_frames_video_source, branch_fit_latent, prompt, seed, gr.update()
        info, error = latent_jobs.resolve_latent_branch(self._jobs, image_prompt_type, video_source, state, model_type, keep_frames=keep_frames_video_source, fit=bool(branch_fit_latent), base_model_type=self._ctx.get_base_model_type(model_type), model_def=self._ctx.get_model_def(model_type), latent_file=branch_latent_file)
        if error:
            raise gr.Error(f"From latent: {error}")
        # The auto-companion: the file's own directory and the app's save /
        # search directories (the same construction resolve_latent_branch
        # uses) are probed for the video the file was saved from.
        found = latent_io.find_video_for_latent(info["path"], latent_jobs._branch_latent_search_dirs(self._jobs))
        _vsrc = str(video_source or "").strip()
        if len(found) == 1 and _vsrc != found[0]:
            # A new companion: select it and mark it auto-detected (the video
            # handler's retrigger recognises this through the re-assert marker
            # and skips the re-report).
            note = f"From latent: companion video selected: {found[0]}"
            gr.Info(note)
            print(note, flush=True)
            if isinstance(state, dict):
                state["branch_video_asserted"] = found[0]
                state["branch_video_origin"] = "auto"
            video_update = gr.update(value = found[0], label = self._ctx.video_input_label_with_info(DETECTED_COMPANION_VIDEO_LABEL, found[0]))
        else:
            # The companion is already the selected video (the backward flow
            # resolved the same pair) — kept without re-selecting or noting —
            # or several / no candidates (the user picks through the normal
            # video control); the video's origin is then left to the next
            # user selection.
            if len(found) > 1:
                note = f"From latent: several companion videos match the latent file ({', '.join(os.path.basename(p) for p in found)}); select one with the video control."
                gr.Info(note)
                print(note, flush=True)
            if isinstance(state, dict):
                state.pop("branch_video_asserted", None)
                state.pop("branch_video_origin", None)
            # The label switches with the mode (not with the file): the mode
            # is on, so the video picker stays labelled as the (optional)
            # companion video whatever the file picker holds.
            video_update = gr.update(label = self._ctx.video_input_label_with_info(self._branch_video_label_base(state, video_source), video_source))
        new_keep, new_prompt, new_seed = self._branch_success_updates(info, keep_frames_video_source, branch_fit_latent, prompt, seed, no_companion=len(found) == 0, advisory=latent_jobs._branch_prefix_advisory(self._jobs, model_type), state=state)
        return video_update, gr.update(label = BRANCH_FRAMES_VIDEO_SOURCE_LABEL, value = new_keep), gr.update(visible = True), new_prompt, new_seed, gr.update(label = self.latent_file_label_with_info(self._branch_latent_label_base(state, branch_latent_file), info.get("entry")))

    def latent_file_label_with_info(self, base_label, entry):
        """The latent file picker's label info suffix: the same
        (width x height, fps, frame count) shape as
        :func:`video_input_label_with_info`, but taken from the latent file's
        own records (the sidecar stores the video's resolution / fps / frame
        count), so the suffix works with no companion video present."""
        entry = entry or {}
        try:
            width = int(entry.get("width") or 0)
            height = int(entry.get("height") or 0)
            fps = float(entry.get("fps") or 0)
            frames = int(entry.get("frames") or 0)
        except (TypeError, ValueError):
            return base_label
        if width <= 0 or height <= 0:
            return base_label
        info = [f"{width}x{height}"]
        if fps > 0:
            info += [f"{fps:g} fps"]
        if frames > 0:
            info += [f"{frames} {'frame' if frames == 1 else 'frames'}"]
        return f"{base_label} ({', '.join(info)})"

    def refresh_video_source_label(self, video_path, branch_from_latent, image_prompt_type):
        """The source-video picker's label refresh: its base label switches
        with the *From latent* mode (on: the companion of the latent file —
        optional, the branch reads the file; off: the continuation source),
        so the resolution / fps / frame-count suffix is appended to the
        mode-appropriate base instead of the one fixed at layout construction
        time (the shared refresh_video_input_label wiring carries a State that
        would go stale when the mode switches)."""
        base = COMPANION_VIDEO_LABEL if (bool(branch_from_latent) and len(filter_letters(str(image_prompt_type or ""), "VL")) > 0) else VIDEO_SOURCE_LABEL
        return gr.update(label=self._ctx.video_input_label_with_info(base, video_path))
