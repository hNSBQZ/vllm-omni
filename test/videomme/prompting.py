"""Prompt construction for MiniCPM-o 4.5 Video-MME evaluation."""

from __future__ import annotations

from config import DEFAULT_CONFIG, GenerationConfig
from dataset import VideoMMEExample


VIDEO_PLACEHOLDER = "(<video>./</video>)"


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
    """Wrap the video placeholder and question in MiniCPM-o chat tokens.

    TODO: verify this exact offline prompt form with a one-sample MiniCPM-o 4.5
    run. If `Omni.generate` expects pre-rendered chat tokens from the tokenizer,
    replace this string builder with tokenizer.apply_chat_template.
    """
    return (
        "<|im_start|>user\n"
        f"{VIDEO_PLACEHOLDER}\n"
        f"{question_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_omni_prompt(
    example: VideoMMEExample,
    frames: list[object],
    generation_config: GenerationConfig = DEFAULT_CONFIG.generation,
) -> dict[str, object]:
    """Return the prompt dict expected by `Omni.generate`.

    The public vLLM multimodal key is singular `video`. MiniCPM-o's processor
    then expands the `<video>./</video>` placeholder into one no-slice image
    placeholder per frame, which keeps the effective input aligned with CPP's
    frame-by-frame image prefill path.
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
