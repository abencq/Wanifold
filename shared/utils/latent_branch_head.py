"""
Created on 2026-11-11

The latent branch head: one decode of the kept region, one encode of the head.

A branch job reads every pixel of its kept region from its own latent file - the
companion video is a preview - and the head window starts from that region's tail
re-encoded onto the new window's grid. Both steps are callable in / tensor out, so
the grid arithmetic holds without a VAE (see tests/test_branch_parity.py).

@author Richard Therriault
"""
from __future__ import annotations

import torch

from . import latent_io


class BranchRefusal(Exception):
    """A branch job cannot honour its own latents; it stops instead of running
    on the companion video's frames."""


def kept_frames(decode_window, kept_specs, kept_frames, *, is_i2v=False, i2v_image_frames=None):
    """The kept region as a ``[3, kept_frames, h, w]`` run, decoded from the
    sidecar's kept windows (see :func:`latent_io.decode_kept_windows`).

    Raises :class:`BranchRefusal` when the decode comes up short: these frames
    have no other source, so the job is refused rather than run on pixels read
    from the companion video.
    """
    frames = None
    try:
        frames = latent_io.decode_kept_windows(
            decode_window, kept_specs, int(kept_frames),
            is_i2v=bool(is_i2v), i2v_image_frames=i2v_image_frames)
    except Exception as exc:
        raise BranchRefusal(f"the kept region could not be decoded ({exc})") from exc
    if frames is None or int(frames.shape[1]) != int(kept_frames):
        got = 0 if frames is None else int(frames.shape[1])
        raise BranchRefusal(
            f"the kept region decoded to {got} frame(s) instead of {int(kept_frames)}")
    return frames


def tail_frames(frames, overlap_frames):
    """The last ``overlap_frames`` of a ``[3, f, h, w]`` run: the head's pixels."""
    if frames is None:
        raise BranchRefusal("the head has no decoded frames to encode")
    overlap = min(int(overlap_frames), int(frames.shape[1]))
    if overlap <= 0:
        raise BranchRefusal("the head overlap is empty")
    return frames[:, -overlap:, :, :]


def guide_tail(frames, overlap_frames):
    """The tail as a guide: the values a continuation hands over (float [-1, 1])."""
    return tail_frames(frames, overlap_frames).float().div_(127.5).sub_(1.0)


def head_latent_count(overlap_frames, stride=4, offset=1):
    """The latents an encode of ``overlap_frames`` frames holds: one for its first
    frame and one per ``stride`` frames after it. A tail clamped to a short kept run
    counts the same way."""
    frames = int(overlap_frames)
    if frames < 1:
        return 0
    return 1 + max(0, frames - int(offset)) // int(stride)


def head_latents(encode, frames, overlap_frames, frozen_latents, *, stride=4, offset=1):
    """Encode the tail of ``frames`` into the pixel rule's frozen head.

    ``encode`` takes a ``[3, o, h, w]`` run and returns ``[c, t, r, s]``. The head must
    come back holding the ``frozen_latents`` the plan counted - that is where plan and
    engine meet, so anything else stops the job.
    """
    head = encode(tail_frames(frames, overlap_frames))
    if head is None:
        raise BranchRefusal("the head latents could not be encoded")
    if int(head.shape[1]) != int(frozen_latents):
        raise BranchRefusal(
            f"the head encode produced {int(head.shape[1])} latent frame(s) where the plan "
            f"froze {int(frozen_latents)}")
    return head.contiguous() if isinstance(head, torch.Tensor) else head
