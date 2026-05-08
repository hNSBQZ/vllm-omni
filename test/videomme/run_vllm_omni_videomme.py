"""CLI entrypoint for MiniCPM-o 4.5 Video-MME evaluation on vllm-omni.

The file is a reviewable scaffold first. Use `--dry-run` after the next pass to
validate dataset schema and video sampling without launching the model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from config import DEFAULT_CONFIG, EvalConfig, validate_config
from dataset import VideoMMEExample, load_examples
from inference import create_omni, run_one
from scoring import ScoreResult, score_response
from video_prep import PreparedVideo, load_video_default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MiniCPM-o 4.5 on Video-MME with vllm-omni.")
    parser.add_argument("--limit", type=int, default=None, help="Limit examples for smoke tests.")
    parser.add_argument("--start", type=int, default=0, help="Start row offset.")
    parser.add_argument("--end", type=int, default=None, help="End row offset.")
    parser.add_argument("--output", type=Path, default=DEFAULT_CONFIG.paths.output_path)
    parser.add_argument("--stage-config", type=Path, default=DEFAULT_CONFIG.paths.stage_config_path)
    parser.add_argument("--dry-run", action="store_true", help="Only load data and sample frames.")
    parser.add_argument("--overwrite", action="store_true", help="Ignore and replace any existing JSONL output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _config_from_args(args)
    validate_config(config)

    examples = load_examples(config, limit=args.limit, start=args.start, end=args.end)
    if args.dry_run:
        _dry_run(examples, config)
        return

    if args.overwrite and config.paths.output_path.exists():
        config.paths.output_path.unlink()

    completed_question_ids = _load_completed_question_ids(config.paths.output_path)
    if completed_question_ids:
        print(f"Resuming: skipping {len(completed_question_ids)} completed question(s).")
    pending_examples = [
        example for example in examples if example.question_id not in completed_question_ids
    ]
    if not pending_examples:
        _write_summary(config.paths.output_path)
        return

    omni = create_omni(config)
    config.paths.output_path.parent.mkdir(parents=True, exist_ok=True)

    with config.paths.output_path.open("a", encoding="utf-8") as output_file:
        for example in pending_examples:
            prepared_video = load_video_default(example.video_path, config.video)
            raw_response, raw_responses, score = _run_with_invalid_answer_retry(
                omni,
                example,
                prepared_video,
                config,
            )
            output_file.write(
                json.dumps(
                    _record(example, prepared_video, raw_response, raw_responses, score),
                    ensure_ascii=False,
                )
                + "\n"
            )
            output_file.flush()
            completed_question_ids.add(example.question_id)

    _write_summary(config.paths.output_path)


def _config_from_args(args: argparse.Namespace) -> EvalConfig:
    paths = replace(
        DEFAULT_CONFIG.paths,
        output_path=args.output,
        stage_config_path=args.stage_config,
    )
    return replace(DEFAULT_CONFIG, paths=paths)


def _dry_run(examples: list[VideoMMEExample], config: EvalConfig) -> None:
    """Print minimal video sampling diagnostics."""
    for example in examples:
        prepared_video = load_video_default(example.video_path, config.video)
        print(
            json.dumps(
                {
                    "index": example.index,
                    "question_id": example.question_id,
                    "video_id": example.video_id,
                    "dataset_video_id": example.dataset_video_id,
                    "video_path": str(example.video_path),
                    "duration": example.duration,
                    "domain": example.domain,
                    "sub_category": example.sub_category,
                    "task_type": example.task_type,
                    "total_frames": prepared_video.total_frames,
                    "avg_fps": prepared_video.avg_fps,
                    "sampled_frames": len(prepared_video.frame_indices),
                    "frame_indices_head": prepared_video.frame_indices[:8],
                    "frame_indices_tail": prepared_video.frame_indices[-8:],
                },
                ensure_ascii=False,
            )
        )


def _run_with_invalid_answer_retry(
    omni: Any,
    example: VideoMMEExample,
    prepared_video: PreparedVideo,
    config: EvalConfig,
) -> tuple[str, list[str], ScoreResult]:
    """Rerun once when answer extraction cannot find A/B/C/D."""
    raw_responses: list[str] = []
    score: ScoreResult | None = None
    max_attempts = config.generation.invalid_answer_retries + 1

    for _ in range(max_attempts):
        inference_result = run_one(omni, example, prepared_video, config)
        raw_responses.append(inference_result.raw_response)
        score = score_response(inference_result.raw_response, example.answer)
        if score.response is not None:
            break

    if score is None:
        raise RuntimeError("No inference attempt was executed.")
    return raw_responses[-1], raw_responses, score


def _record(
    example: VideoMMEExample,
    prepared_video: PreparedVideo,
    raw_response: str,
    raw_responses: list[str],
    score: ScoreResult,
) -> dict[str, Any]:
    return {
        "index": example.index,
        "question_id": example.question_id,
        "video_id": example.video_id,
        "dataset_video_id": example.dataset_video_id,
        "duration": example.duration,
        "domain": example.domain,
        "sub_category": example.sub_category,
        "task_type": example.task_type,
        "url": example.url,
        "video_path": str(example.video_path),
        "question": example.question,
        "options": example.options,
        "answer": score.answer,
        "response": score.response,
        "raw_response": raw_response,
        "raw_responses": raw_responses,
        "retry_count": max(0, len(raw_responses) - 1),
        "hit": score.hit,
        "frame_indices": prepared_video.frame_indices,
        "avg_fps": prepared_video.avg_fps,
        "total_frames": prepared_video.total_frames,
    }


def _load_completed_question_ids(output_path: Path) -> set[str]:
    """Read existing JSONL output so interrupted runs can resume in-place."""
    if not output_path.exists():
        return set()

    completed: set[str] = set()
    with output_path.open("r", encoding="utf-8") as output_file:
        for line_number, line in enumerate(output_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {output_path}:{line_number}") from exc
            question_id = record.get("question_id")
            if question_id is not None:
                completed.add(str(question_id))
    return completed


def _summary_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.summary.json")


def _write_summary(output_path: Path) -> None:
    summary = _summarize_output(output_path)
    summary_path = _summary_path(output_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _summarize_output(output_path: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as output_file:
            records = [json.loads(line) for line in output_file if line.strip()]

    summary: dict[str, Any] = {
        **_bucket_stats(records),
        "invalid_response": sum(1 for record in records if record.get("response") is None),
        "by_duration": _grouped_stats(records, "duration"),
        "by_domain": _grouped_stats(records, "domain"),
        "by_sub_category": _grouped_stats(records, "sub_category"),
        "by_task_type": _grouped_stats(records, "task_type"),
    }
    return summary


def _grouped_stats(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        grouped.setdefault(str(value), []).append(record)
    return {value: _bucket_stats(items) for value, items in sorted(grouped.items())}


def _bucket_stats(records: list[dict[str, Any]]) -> dict[str, float | int]:
    total = len(records)
    correct = sum(1 for record in records if bool(record.get("hit")))
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
    }


if __name__ == "__main__":
    main()
