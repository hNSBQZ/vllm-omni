"""Daily-Omni 音视频输入准备逻辑。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import DEFAULT_CONFIG, InputMode, MediaConfig


@dataclass(frozen=True)
class PreparedMedia:
    frames: list[Any]
    frame_indices: list[int]
    frame_timestamps: list[float]
    avg_fps: float | None
    total_frames: int | None
    audio: Any | None
    audio_segments: list[Any]
    audio_sample_rate: int | None
    input_mode: InputMode


def uniform_sample(indices: list[int], max_count: int) -> list[int]:
    """对齐 torch 测评路径使用的 midpoint uniform sampling 公式。"""
    if len(indices) <= max_count:
        return indices
    gap = len(indices) / max_count
    return [indices[int(i * gap + gap / 2)] for i in range(max_count)]


def load_media(
    video_path: Path,
    audio_path: Path,
    config: MediaConfig = DEFAULT_CONFIG.media,
) -> PreparedMedia:
    """加载一条 Daily-Omni 样本的媒体。

    `all` 模式会按抽帧时间戳把整段音频切成与帧一一对应的片段，
    供 prompt 层构造成 `<image><audio><image><audio>...` 的交错输入。
    """
    frames: list[Any] = []
    frame_indices: list[int] = []
    frame_timestamps: list[float] = []
    avg_fps: float | None = None
    total_frames: int | None = None
    audio: Any | None = None
    audio_segments: list[Any] = []
    audio_sample_rate: int | None = None

    if config.input_mode in {InputMode.ALL, InputMode.VISUAL}:
        video = load_video_default(video_path, config)
        frames = video.frames
        frame_indices = video.frame_indices
        frame_timestamps = video.frame_timestamps
        avg_fps = video.avg_fps
        total_frames = video.total_frames

    if config.input_mode in {InputMode.ALL, InputMode.AUDIO}:
        audio, audio_sample_rate = load_audio_default(audio_path, config)

    if config.input_mode == InputMode.ALL and audio is not None and audio_sample_rate is not None:
        audio_segments = split_audio_by_frame_timestamps(audio, audio_sample_rate, frame_timestamps)
        num_pairs = min(len(frames), len(audio_segments))
        frames = frames[:num_pairs]
        frame_indices = frame_indices[:num_pairs]
        frame_timestamps = frame_timestamps[:num_pairs]
        audio_segments = audio_segments[:num_pairs]

    return PreparedMedia(
        frames=frames,
        frame_indices=frame_indices,
        frame_timestamps=frame_timestamps,
        avg_fps=avg_fps,
        total_frames=total_frames,
        audio=audio,
        audio_segments=audio_segments,
        audio_sample_rate=audio_sample_rate,
        input_mode=config.input_mode,
    )


@dataclass(frozen=True)
class PreparedVideo:
    frames: list[Any]
    frame_indices: list[int]
    frame_timestamps: list[float]
    avg_fps: float
    total_frames: int


def load_video_default(
    video_path: Path,
    config: MediaConfig = DEFAULT_CONFIG.media,
) -> PreparedVideo:
    """使用 decord 解码视频，按接近 1 FPS 抽帧，并最多保留 64 帧。"""
    if not video_path.exists():
        raise FileNotFoundError(f"找不到 Daily-Omni 视频: {video_path}")

    try:
        from decord import VideoReader, cpu
    except ImportError as exc:
        raise RuntimeError("对齐 Daily-Omni 视频路径需要安装 decord。") from exc
    import numpy as np
    from PIL import Image

    vr = VideoReader(str(video_path), ctx=cpu(0))
    total_frames = len(vr)
    avg_fps = float(vr.get_avg_fps())
    sample_fps = max(round(avg_fps / config.max_fps), 1)
    frame_indices = list(range(0, total_frames, sample_fps))
    frame_indices = uniform_sample(frame_indices, config.max_num_frames)
    frame_timestamps = [
        frame_idx / avg_fps if avg_fps > 0 else 0.0
        for frame_idx in frame_indices
    ]

    frames_np = vr.get_batch(frame_indices).asnumpy()
    frames = [
        Image.fromarray(frame.astype(np.uint8)).convert("RGB")
        for frame in frames_np
    ]
    return PreparedVideo(
        frames=frames,
        frame_indices=frame_indices,
        frame_timestamps=frame_timestamps,
        avg_fps=avg_fps,
        total_frames=total_frames,
    )


def load_audio_default(
    audio_path: Path,
    config: MediaConfig = DEFAULT_CONFIG.media,
) -> tuple[Any, int]:
    """以 16 kHz 单声道 numpy 形式加载 wav，以对齐 MiniCPM-o torch 输入。"""
    if not audio_path.exists():
        raise FileNotFoundError(f"找不到 Daily-Omni 音频: {audio_path}")

    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("加载 Daily-Omni 音频需要安装 librosa。") from exc

    audio, _ = librosa.load(audio_path, sr=config.audio_sample_rate, mono=True)
    return audio, config.audio_sample_rate


def split_audio_by_frame_timestamps(
    audio: Any,
    sample_rate: int,
    frame_timestamps: list[float],
) -> list[Any]:
    """按相邻帧时间戳切分音频，最后一段延伸到音频末尾。"""
    if sample_rate <= 0:
        raise ValueError("sample_rate 必须为正数。")
    if not frame_timestamps:
        return []

    audio_len = len(audio)
    audio_duration = audio_len / float(sample_rate)
    segments: list[Any] = []

    for index, start_time in enumerate(frame_timestamps):
        end_time = frame_timestamps[index + 1] if index + 1 < len(frame_timestamps) else audio_duration
        start_sample = min(max(int(start_time * sample_rate), 0), audio_len)
        end_sample = min(max(int(end_time * sample_rate), start_sample), audio_len)
        segment = audio[start_sample:end_sample]
        if len(segment) == 0:
            continue
        segments.append(segment)

    return segments
