"""MiniCPM-o 4.5 Daily-Omni 测评 prompt 构造逻辑。"""

from __future__ import annotations

from config import DEFAULT_CONFIG, GenerationConfig, InputMode
from dataset import DailyOmniExample
from media_prep import PreparedMedia

VIDEO_PLACEHOLDER = "(<video>./</video>)"
IMAGE_PLACEHOLDER = "(<image>./</image>)"
AUDIO_PLACEHOLDER = "(<audio>./</audio>)"
NO_THINKING_SUFFIX = "<think>\n\n</think>\n\n"

DAILY_OMNI_USER_PROMPT_TEMPLATE = (
    "Carefully read the following question and select the letter corresponding to the correct answer."
    "Highlight the applicable choices without giving explanations.\n"
    "{question}\n"
    "Options:\n{options}\n"
    "Please select the correct answer from the options above. Only respond with the letter."
)


def build_question_prompt(example: DailyOmniExample) -> str:
    """根据 evalkit 的 nosys interleave 配置构造多选题 prompt。"""
    return DAILY_OMNI_USER_PROMPT_TEMPLATE.format(
        question=example.question,
        options="\n".join(example.options),
    )


def build_chat_prompt(question_prompt: str, media: PreparedMedia) -> str:
    """用 MiniCPM-o chat token 包装媒体占位符和题目。

    `all` 模式使用多张 image 与多段 audio 交错，而不是单个 video
    占位符，以对齐 torch `omni_mode=True` 的内容列表。
    """
    placeholders = _media_placeholders(media)
    return (
        "<|im_start|>user\n"
        f"{placeholders}\n"
        f"{question_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{NO_THINKING_SUFFIX}"
    )


def build_omni_prompt(
    example: DailyOmniExample,
    media: PreparedMedia,
    generation_config: GenerationConfig = DEFAULT_CONFIG.generation,
) -> dict[str, object]:
    """返回 `Omni.generate` 期望的 prompt dict。

    `all` 模式下的 `image` 与 `audio` 数量需要和 prompt 中的占位符
    一一对应，保证 prompt replacement 后仍保持帧/音频交错顺序。
    """
    question_prompt = build_question_prompt(example)
    multi_modal_data: dict[str, object] = {}
    use_tts = media.input_mode in {InputMode.ALL, InputMode.AUDIO}
    if media.input_mode == InputMode.ALL:
        pair_count = min(len(media.frames), len(media.audio_segments))
        if pair_count == 0:
            raise ValueError("all 模式需要至少一对视频帧和音频片段。")
        multi_modal_data["image"] = media.frames[:pair_count]
        multi_modal_data["audio"] = media.audio_segments[:pair_count]
    elif media.input_mode == InputMode.VISUAL:
        multi_modal_data["video"] = media.frames
    elif media.input_mode == InputMode.AUDIO and media.audio is not None:
        multi_modal_data["audio"] = media.audio

    return {
        "prompt": build_chat_prompt(question_prompt, media),
        "modalities": ["text"],
        "multi_modal_data": multi_modal_data,
        "mm_processor_kwargs": {
            "max_slice_nums": generation_config.max_slice_nums,
            "use_image_id": generation_config.use_image_id,
            # vLLM-Omni processor 会根据该开关追加完整 TTS 前缀：
            # <|spk_bos|><|spk|><|spk_eos|><|tts_bos|>。
            "use_tts": use_tts,
        },
    }


def _media_placeholders(media: PreparedMedia) -> str:
    if media.input_mode == InputMode.VISUAL:
        return VIDEO_PLACEHOLDER
    if media.input_mode == InputMode.AUDIO:
        return AUDIO_PLACEHOLDER
    pair_count = min(len(media.frames), len(media.audio_segments))
    return "".join(f"{IMAGE_PLACEHOLDER}{AUDIO_PLACEHOLDER}" for _ in range(pair_count))
