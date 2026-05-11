"""Prompt construction for MiniCPM-o 4.5 Video-MME evaluation."""

from __future__ import annotations

from config import DEFAULT_CONFIG, GenerationConfig
from dataset import VideoMMEExample


VIDEO_PLACEHOLDER = "(<video>./</video>)"
NO_THINKING_SUFFIX = "<think>\n\n</think>\n\n"


DEFAULT_USER_PROMPT_TEMPLATE = (
    "Carefully read the following question and select the letter corresponding to the correct answer."
    "Highlight the applicable choices without giving explanations.\n"
    "{question}\n"
    "Options:\n{options}"
)


def build_question_prompt(example: VideoMMEExample) -> str:
    """Build the multiple-choice prompt used by the default baseline."""
    return DEFAULT_USER_PROMPT_TEMPLATE.format(
        question=example.question,
        options="\n".join(example.options),
    )


def build_chat_prompt(question_prompt: str) -> str:
    """Wrap the video placeholder and question in MiniCPM-o chat tokens."""
    return (
        "<|im_start|>user\n"
        f"{VIDEO_PLACEHOLDER}\n"
        f"{question_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{NO_THINKING_SUFFIX}"
    )


def build_omni_prompt(
    example: VideoMMEExample,
    frames: list[object],
    generation_config: GenerationConfig = DEFAULT_CONFIG.generation,
) -> dict[str, object]:
    """Return the prompt dict expected by `Omni.generate`.

    HF `model.chat` appends the empty thinking block when
    `enable_thinking=False`. The MiniCPM-o 4.5 vLLM processor expands the video
    placeholder into per-frame image placeholders with newline separators.
    """
    question_prompt = build_question_prompt(example)
    return {
        "prompt": build_chat_prompt(question_prompt),
        "modalities": ["text"],
        "multi_modal_data": {
            "video": frames,
        },
        "mm_processor_kwargs": {
            "max_slice_nums": generation_config.max_slice_nums,
            "use_image_id": generation_config.use_image_id,
        },
    }
