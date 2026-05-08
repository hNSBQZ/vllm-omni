"""Dataset loading helpers for Video-MME."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import DEFAULT_CONFIG, EvalConfig

REQUIRED_COLUMNS = {
    "video_id",
    "duration",
    "domain",
    "sub_category",
    "url",
    "videoID",
    "question_id",
    "task_type",
    "question",
    "options",
    "answer",
}


@dataclass(frozen=True)
class VideoMMEExample:
    index: int
    question_id: str
    video_id: str
    question: str
    options: list[str]
    answer: str
    video_path: Path
    duration: str | None = None
    dataset_video_id: str | None = None
    domain: str | None = None
    sub_category: str | None = None
    task_type: str | None = None
    url: str | None = None
    metadata: dict[str, Any] | None = None


def load_examples(
    config: EvalConfig = DEFAULT_CONFIG,
    *,
    limit: int | None = None,
    start: int = 0,
    end: int | None = None,
) -> list[VideoMMEExample]:
    """Load and normalize Video-MME rows from the official parquet split."""
    import pandas as pd

    df = pd.read_parquet(config.paths.parquet_path)
    _validate_columns(df.columns)
    if end is not None:
        df = df.iloc[start:end]
    elif start:
        df = df.iloc[start:]
    if limit is not None:
        df = df.iloc[:limit]

    return [_row_to_example(i, row.to_dict(), config) for i, row in df.iterrows()]


def _row_to_example(index: int, row: dict[str, Any], config: EvalConfig) -> VideoMMEExample:
    video_id = str(row["videoID"])
    dataset_video_id = str(row["video_id"])
    question_id = str(row["question_id"])
    options = _normalize_options(row["options"])

    return VideoMMEExample(
        index=index,
        question_id=question_id,
        video_id=video_id,
        question=str(row["question"]),
        options=options,
        answer=str(row["answer"]).strip().upper()[:1],
        video_path=config.paths.video_dir / f"{video_id}.mp4",
        duration=str(row["duration"]),
        dataset_video_id=dataset_video_id,
        domain=str(row["domain"]),
        sub_category=str(row["sub_category"]),
        task_type=str(row["task_type"]),
        url=str(row["url"]),
        metadata={
            "dataset_video_id": dataset_video_id,
            "video_path": str(config.paths.video_dir / f"{video_id}.mp4"),
            "domain": str(row["domain"]),
            "sub_category": str(row["sub_category"]),
            "task_type": str(row["task_type"]),
            "url": str(row["url"]),
        },
    )


def _validate_columns(columns: Any) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(columns))
    if missing:
        raise KeyError(f"Missing required Video-MME columns: {missing}")


def _normalize_options(options: Any) -> list[str]:
    """Normalize parquet option values into the `A. ...` text expected by prompts."""
    if hasattr(options, "tolist"):
        options = options.tolist()
    if isinstance(options, list):
        normalized = [str(option) for option in options]
        if len(normalized) != 4:
            raise ValueError(f"Expected 4 Video-MME options, got {len(normalized)}.")
        return normalized
    if isinstance(options, tuple):
        return _normalize_options(list(options))
    if isinstance(options, str):
        lines = [line.strip() for line in options.splitlines() if line.strip()]
        return _normalize_options(lines if lines else [options])
    raise TypeError(f"Unsupported options type: {type(options)!r}")
