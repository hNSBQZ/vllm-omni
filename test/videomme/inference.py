"""Thin MiniCPM-o 4.5 / vllm-omni inference wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import DEFAULT_CONFIG, EvalConfig, GenerationConfig
from dataset import VideoMMEExample
from prompting import build_omni_prompt
from video_prep import PreparedVideo


@dataclass(frozen=True)
class InferenceResult:
    raw_response: str
    stage_id: int | None = None
    final_output_type: str | None = None


def build_sampling_params(config: GenerationConfig = DEFAULT_CONFIG.generation) -> Any:
    """Create vLLM sampling params matching the comparison document.

    TODO: check whether MiniCPM-o 4.5's stage config uses a custom sampling
    params class for non-LLM stages. If so, pass a per-stage list instead of a
    single `SamplingParams` object.
    """
    from vllm import SamplingParams

    return SamplingParams(
        max_tokens=config.max_new_tokens,
        min_tokens=config.min_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        repetition_penalty=config.repetition_penalty,
    )


def create_omni(config: EvalConfig = DEFAULT_CONFIG) -> Any:
    """Initialize the vllm-omni orchestrator."""
    from vllm_omni.entrypoints.omni import Omni

    return Omni(
        model=str(config.paths.model_path),
        stage_configs_path=str(config.paths.stage_config_path),
        trust_remote_code=True,
    )


def run_one(
    omni: Any,
    example: VideoMMEExample,
    prepared_video: PreparedVideo,
    config: EvalConfig = DEFAULT_CONFIG,
) -> InferenceResult:
    """Run one Video-MME question.

    TODO: verify prompt + `multi_modal_data` shape on 1 sample before enabling
    batch execution.
    """
    prompt = build_omni_prompt(example, prepared_video.frames, config.generation)
    outputs = omni.generate(prompt, build_sampling_params(config.generation), use_tqdm=False)
    return _extract_text(outputs)


def _extract_text(outputs: list[Any]) -> InferenceResult:
    """Extract first non-empty text from Omni outputs."""
    for output in outputs:
        for request_output in getattr(output, "request_output", []) or []:
            for completion in getattr(request_output, "outputs", []) or []:
                text = getattr(completion, "text", "")
                if text:
                    return InferenceResult(
                        raw_response=text,
                        stage_id=getattr(output, "stage_id", None),
                        final_output_type=getattr(output, "final_output_type", None),
                    )

    # TODO: capture structured Omni output in the result JSON if no text appears.
    return InferenceResult(raw_response="")
