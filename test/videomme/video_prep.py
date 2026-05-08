"""Video preparation for the CPP-aligned Video-MME path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from config import DEFAULT_CONFIG, VideoSamplingConfig


@dataclass(frozen=True)
class PreparedVideo:
    frames: list[Image.Image]
    frame_indices: list[int]
    avg_fps: float
    total_frames: int


def uniform_sample(indices: list[int], max_count: int) -> list[int]:
    """Match the midpoint uniform sampling formula used by Torch/CPP evals."""
    if len(indices) <= max_count:
        return indices
    gap = len(indices) / max_count
    return [indices[int(i * gap + gap / 2)] for i in range(max_count)]


def load_video_default(
    video_path: Path,
    config: VideoSamplingConfig = DEFAULT_CONFIG.video,
) -> PreparedVideo:
    """Decode video with decord, sample near 1 FPS, and cap to 64 frames.

    TODO: add an optional diagnostics mode that prints fps/indices for parity
    checks against CPP `eval_cpp_video_prep.py`.
    TODO: decide whether to add ffmpeg/torchvision fallback. The strict default
    path should remain decord-only to stay aligned with the Torch baseline.
    """
    if config.sample_strategy != "default":
        raise ValueError("TODO: only default 1 FPS sampling is implemented.")

    try:
        from decord import VideoReader, cpu
    except ImportError as exc:
        raise RuntimeError("decord is required for the aligned Video-MME video path.") from exc

    vr = VideoReader(str(video_path), ctx=cpu(0))
    total_frames = len(vr)
    avg_fps = float(vr.get_avg_fps())
    sample_fps = max(round(avg_fps / config.max_fps), 1)
    frame_indices = list(range(0, total_frames, sample_fps))
    frame_indices = uniform_sample(frame_indices, config.max_num_frames)

    frames_np = vr.get_batch(frame_indices).asnumpy()
    frames = [
        Image.fromarray(frame.astype(np.uint8)).convert("RGB")
        for frame in frames_np
    ]
    return PreparedVideo(
        frames=frames,
        frame_indices=frame_indices,
        avg_fps=avg_fps,
        total_frames=total_frames,
    )
