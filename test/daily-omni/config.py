"""vllm-omni Daily-Omni 测评骨架配置。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class InputMode(str, Enum):
    """Daily-Omni 官方本地脚本使用的模态模式。"""

    ALL = "all"
    VISUAL = "visual"
    AUDIO = "audio"


@dataclass(frozen=True)
class DailyOmniPaths:
    dataset_root: Path = Path("/cache/hanqingzhe/daily-omni")
    model_path: Path = Path("/cache/hanqingzhe/o45-pure-py")
    stage_config_path: Path = Path("test/daily-omni/minicpmo45_text_8x4090.yaml")
    output_path: Path = Path("test/daily-omni/outputs/vllm_omni_minicpmo45_daily_omni.jsonl")

    @property
    def jsonl_path(self) -> Path:
        return self.dataset_root / "daily_omni.jsonl"


@dataclass(frozen=True)
class MediaConfig:
    max_num_frames: int = 64
    max_fps: float = 1.0
    audio_sample_rate: int = 16000
    input_mode: InputMode = InputMode.ALL
    require_media: bool = False


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 128
    min_new_tokens: int = 0
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.1
    max_inp_length: int = 40960
    max_slice_nums: int = 1
    use_image_id: bool = False
    invalid_answer_retries: int = 1


@dataclass(frozen=True)
class EvalConfig:
    paths: DailyOmniPaths = DailyOmniPaths()
    media: MediaConfig = MediaConfig()
    generation: GenerationConfig = GenerationConfig()


DEFAULT_CONFIG = EvalConfig()


def parse_input_mode(value: str | InputMode) -> InputMode:
    if isinstance(value, InputMode):
        return value
    try:
        return InputMode(value)
    except ValueError as exc:
        valid = ", ".join(mode.value for mode in InputMode)
        raise ValueError(f"不支持的 input mode: {value!r}。可选值：{valid}") from exc


def validate_config(config: EvalConfig = DEFAULT_CONFIG) -> None:
    """提前检查会阻塞 Daily-Omni 运行的配置。"""
    if not config.paths.jsonl_path.exists():
        raise FileNotFoundError(f"找不到 Daily-Omni JSONL: {config.paths.jsonl_path}")
    if not config.paths.model_path.exists():
        raise FileNotFoundError(f"找不到 MiniCPM-o 4.5 模型路径: {config.paths.model_path}")
    if not config.paths.stage_config_path.exists():
        raise FileNotFoundError(f"找不到 stage 配置: {config.paths.stage_config_path}")
    if config.media.max_num_frames <= 0:
        raise ValueError("max_num_frames 必须为正数。")
    if config.media.max_fps <= 0:
        raise ValueError("max_fps 必须为正数。")
    if config.media.audio_sample_rate != 16000:
        raise ValueError("TODO：目前仅支持 16 kHz 音频，以对齐 MiniCPM-o。")
    if config.generation.temperature != 0.0:
        raise ValueError("Daily-Omni 对比测评应使用 greedy decoding: temperature=0.0。")
    if config.generation.max_new_tokens != 128:
        raise ValueError("Daily-Omni torch evalkit 使用 max_tokens=128。")
    if config.generation.max_slice_nums != 1:
        raise ValueError("max_slice_nums 必须保持为 1，以使用视频帧 no-slice 模式。")
    if config.generation.use_image_id:
        raise ValueError("视频输入时 use_image_id 必须保持 False。")
    if config.generation.invalid_answer_retries < 0:
        raise ValueError("invalid_answer_retries 不能为负数。")
