"""Daily-Omni 数据集加载辅助函数。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import DEFAULT_CONFIG, EvalConfig

REQUIRED_FIELDS = {
    "dataset_type",
    "dataset_name",
    "question",
    "choices",
    "gt_answer",
    "VideoPath",
    "WavPath",
    "qa_type",
    "content_parent_category",
    "content_fine_category",
    "video_category",
    "video_duration",
    "video_id",
}


@dataclass(frozen=True)
class DailyOmniExample:
    index: int
    question_id: str
    video_id: str
    question: str
    options: list[str]
    answer: str
    video_path: Path
    audio_path: Path
    qa_type: str | None = None
    content_parent_category: str | None = None
    content_fine_category: str | None = None
    video_category: str | None = None
    video_duration: str | None = None
    dataset_type: str | None = None
    dataset_name: str | None = None
    metadata: dict[str, Any] | None = None


def load_examples(
    config: EvalConfig = DEFAULT_CONFIG,
    *,
    limit: int | None = None,
    start: int = 0,
    end: int | None = None,
) -> list[DailyOmniExample]:
    """从 JSONL 加载并规范化 Daily-Omni 样本。"""
    rows = _read_jsonl(config.paths.jsonl_path)
    if end is not None:
        rows = rows[start:end]
    elif start:
        rows = rows[start:]
    if limit is not None:
        rows = rows[:limit]

    examples = [
        _row_to_example(index=start + offset, row=row, config=config)
        for offset, row in enumerate(rows)
        if row.get("dataset_type") == "mcq"
    ]
    if config.media.require_media:
        _validate_media_exists(examples, config)
    return examples


def _read_jsonl(jsonl_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 格式错误: {jsonl_path}:{line_number}") from exc
            _validate_row(row, jsonl_path, line_number)
            rows.append(row)
    return rows


def _row_to_example(index: int, row: dict[str, Any], config: EvalConfig) -> DailyOmniExample:
    video_id = str(row["video_id"])
    question_id = _question_id(index, row)
    video_path = config.paths.dataset_root / str(row["VideoPath"])
    audio_path = config.paths.dataset_root / str(row["WavPath"])

    return DailyOmniExample(
        index=index,
        question_id=question_id,
        video_id=video_id,
        question=str(row["question"]),
        options=_normalize_options(row["choices"]),
        answer=str(row["gt_answer"]).strip().upper()[:1],
        video_path=video_path,
        audio_path=audio_path,
        qa_type=_clean_optional(row.get("qa_type")),
        content_parent_category=_clean_optional(row.get("content_parent_category")),
        content_fine_category=_clean_optional(row.get("content_fine_category")),
        video_category=_clean_optional(row.get("video_category")),
        video_duration=_clean_optional(row.get("video_duration")),
        dataset_type=_clean_optional(row.get("dataset_type")),
        dataset_name=_clean_optional(row.get("dataset_name")),
        metadata={
            "VideoPath": str(row["VideoPath"]),
            "WavPath": str(row["WavPath"]),
            "video_path": str(video_path),
            "audio_path": str(audio_path),
        },
    )


def _validate_row(row: dict[str, Any], jsonl_path: Path, line_number: int) -> None:
    missing = sorted(REQUIRED_FIELDS - set(row))
    if missing:
        raise KeyError(f"Daily-Omni 字段缺失: {jsonl_path}:{line_number}: {missing}")
    if row.get("dataset_name") != "daily_omni":
        raise ValueError(f"dataset_name 不符合预期: {jsonl_path}:{line_number}: {row.get('dataset_name')!r}")


def _validate_media_exists(examples: list[DailyOmniExample], config: EvalConfig) -> None:
    """仅在调用方准备真实运行时强制要求媒体文件存在。"""
    missing: list[str] = []
    for example in examples:
        if config.media.input_mode.value in {"all", "visual"} and not example.video_path.exists():
            missing.append(str(example.video_path))
        if config.media.input_mode.value in {"all", "audio"} and not example.audio_path.exists():
            missing.append(str(example.audio_path))
        if len(missing) >= 10:
            break
    if missing:
        raise FileNotFoundError(
            "Daily-Omni 媒体文件缺失。请解压 Videos.tar，或设置正确的 dataset_root。"
            f"示例: {missing}"
        )


def _normalize_options(options: Any) -> list[str]:
    """将 Daily-Omni 选项规范化为四个 `A. ...` 字符串。"""
    if isinstance(options, tuple):
        options = list(options)
    if not isinstance(options, list):
        raise TypeError(f"不支持的 Daily-Omni choices 类型: {type(options)!r}")

    normalized = [str(option).strip() for option in options if str(option).strip()]
    if len(normalized) != 4:
        raise ValueError(f"Daily-Omni 应有 4 个选项，实际得到 {len(normalized)} 个。")
    return normalized


def _question_id(index: int, row: dict[str, Any]) -> str:
    """Daily-Omni JSONL 没有显式 question id，因此使用稳定的本地 key。"""
    video_id = str(row["video_id"])
    return f"{video_id}:{index}"


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
