"""Daily-Omni 使用的 MiniCPM-o 4.5 / vllm-omni 轻量推理封装。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import DEFAULT_CONFIG, EvalConfig, GenerationConfig
from dataset import DailyOmniExample
from media_prep import PreparedMedia
from prompting import build_omni_prompt


@dataclass(frozen=True)
class InferenceResult:
    raw_response: str
    stage_id: int | None = None
    final_output_type: str | None = None


def build_sampling_params(config: GenerationConfig = DEFAULT_CONFIG.generation) -> Any:
    """创建与 evalkit Daily-Omni 对齐的 greedy vLLM 采样参数。"""
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
    """初始化 vllm-omni 编排器。"""
    from vllm_omni.entrypoints.omni import Omni

    return Omni(
        model=str(config.paths.model_path),
        stage_configs_path=str(config.paths.stage_config_path),
        trust_remote_code=True,
    )


def run_one(
    omni: Any,
    example: DailyOmniExample,
    media: PreparedMedia,
    config: EvalConfig = DEFAULT_CONFIG,
) -> InferenceResult:
    """运行一条 Daily-Omni 题目。"""
    prompt = build_omni_prompt(example, media, config.generation)
    # TODO：媒体可用后先跑一条真实样本，确认音频输入确实进入 MiniCPM-o，
    # 而不是被 processor 忽略。
    outputs = omni.generate(prompt, build_sampling_params(config.generation), use_tqdm=False)
    return _extract_text(outputs)


def _extract_text(outputs: list[Any]) -> InferenceResult:
    """从 Omni 输出中提取第一段非空文本。"""
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

    # TODO：无文本输出时，将结构化 Omni 输出持久化，方便调试。
    return InferenceResult(raw_response="")
