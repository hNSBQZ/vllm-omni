"""Configuration for the vllm-omni Video-MME evaluation scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoMMEPaths:
    dataset_root: Path = Path("~/Video-MME/Video-MME").expanduser()
    model_path: Path = Path("/cache/hanqingzhe/o45-pure-py")
    stage_config_path: Path = Path("test/videomme/minicpmo45_text_8x4090.yaml")
    output_path: Path = Path("test/videomme/outputs/vllm_omni_minicpmo45_videomme.jsonl")

    @property
    def parquet_path(self) -> Path:
        return self.dataset_root / "videomme" / "test-00000-of-00001.parquet"

    @property
    def video_dir(self) -> Path:
        return self.dataset_root / "data"


@dataclass(frozen=True)
class VideoSamplingConfig:
    max_num_frames: int = 64
    max_fps: float = 1.0
    sample_strategy: str = "default"


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 100
    min_new_tokens: int = 1
    temperature: float = 0.2
    top_p: float = 0.8
    top_k: int = 100
    repetition_penalty: float = 1.02
    max_inp_length: int = 40960
    max_slice_nums: int = 1
    use_image_id: bool = False
    invalid_answer_retries: int = 1


@dataclass(frozen=True)
class EvalConfig:
    paths: VideoMMEPaths = VideoMMEPaths()
    video: VideoSamplingConfig = VideoSamplingConfig()
    generation: GenerationConfig = GenerationConfig()


DEFAULT_CONFIG = EvalConfig()


def validate_config(config: EvalConfig = DEFAULT_CONFIG) -> None:
    """Fail fast on config values that would diverge from the intended default run."""
    if not config.paths.parquet_path.exists():
        raise FileNotFoundError(f"Video-MME parquet not found: {config.paths.parquet_path}")
    if not config.paths.video_dir.exists():
        raise FileNotFoundError(f"Video-MME video directory not found: {config.paths.video_dir}")
    if not config.paths.model_path.exists():
        raise FileNotFoundError(f"MiniCPM-o 4.5 model path not found: {config.paths.model_path}")
    if not config.paths.stage_config_path.exists():
        raise FileNotFoundError(f"Stage config not found: {config.paths.stage_config_path}")
    if config.video.sample_strategy != "default":
        raise ValueError("TODO: only the CPP-aligned default 1 FPS sampling strategy is scaffolded.")
    if config.generation.max_slice_nums != 1:
        raise ValueError("max_slice_nums must stay at 1 to keep MiniCPM-o video frames in no-slice mode.")
    if config.generation.use_image_id:
        raise ValueError("use_image_id must stay False to match the Video-MME comparison setup.")
    if config.generation.invalid_answer_retries < 0:
        raise ValueError("invalid_answer_retries must be non-negative.")
    if config.generation.min_new_tokens < 0:
        raise ValueError("min_new_tokens must be non-negative.")
