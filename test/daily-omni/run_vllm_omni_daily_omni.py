"""MiniCPM-o 4.5 在 vllm-omni 上测评 Daily-Omni 的 CLI 入口。

建议先使用 `--dry-run` 验证 JSONL 加载；媒体解压完成后，再验证媒体准备流程，
整个过程不会启动模型。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from config import DEFAULT_CONFIG, EvalConfig, parse_input_mode, validate_config
from dataset import DailyOmniExample, load_examples
from inference import create_omni, run_one
from media_prep import PreparedMedia, load_media
from scoring import ScoreResult, score_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 vllm-omni 测评 MiniCPM-o 4.5 在 Daily-Omni 上的表现。")
    parser.add_argument("--limit", type=int, default=None, help="限制样本数，用于冒烟测试。")
    parser.add_argument("--start", type=int, default=0, help="起始行偏移。")
    parser.add_argument("--end", type=int, default=None, help="结束行偏移。")
    parser.add_argument("--output", type=Path, default=DEFAULT_CONFIG.paths.output_path)
    parser.add_argument("--stage-config", type=Path, default=DEFAULT_CONFIG.paths.stage_config_path)
    parser.add_argument("--input-mode", choices=["all", "visual", "audio"], default=DEFAULT_CONFIG.media.input_mode.value)
    parser.add_argument("--dry-run", action="store_true", help="只加载数据，并可选准备媒体。")
    parser.add_argument("--overwrite", action="store_true", help="忽略并替换已有 JSONL 输出。")
    parser.add_argument(
        "--require-media",
        action="store_true",
        help="dry-run 时也强制要求引用的 mp4/wav 文件存在；真实运行始终要求媒体存在。",
    )
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
        print(f"续跑：跳过 {len(completed_question_ids)} 条已完成题目。")
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
            media = load_media(example.video_path, example.audio_path, config.media)
            raw_response, raw_responses, score = _run_with_invalid_answer_retry(
                omni,
                example,
                media,
                config,
            )
            output_file.write(
                json.dumps(
                    _record(example, media, raw_response, raw_responses, score, config),
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
    media = replace(
        DEFAULT_CONFIG.media,
        input_mode=parse_input_mode(args.input_mode),
        require_media=args.require_media or not args.dry_run,
    )
    return replace(DEFAULT_CONFIG, paths=paths, media=media)


def _dry_run(examples: list[DailyOmniExample], config: EvalConfig) -> None:
    """打印最小化的数据集与媒体诊断信息。"""
    for example in examples:
        if _should_prepare_media(example, config):
            media = load_media(example.video_path, example.audio_path, config.media)
            payload = _media_diagnostics(example, media, config)
        else:
            # TODO：确认本地媒体解压完整后，移除仅元数据 dry-run 分支。
            payload = _metadata_diagnostics(example, config)
        print(json.dumps(payload, ensure_ascii=False))


def _should_prepare_media(example: DailyOmniExample, config: EvalConfig) -> bool:
    return config.media.require_media


def _metadata_diagnostics(example: DailyOmniExample, config: EvalConfig) -> dict[str, Any]:
    return {
        "index": example.index,
        "question_id": example.question_id,
        "video_id": example.video_id,
        "input_mode": config.media.input_mode.value,
        "video_path": str(example.video_path),
        "video_exists": example.video_path.exists(),
        "audio_path": str(example.audio_path),
        "audio_exists": example.audio_path.exists(),
        "qa_type": example.qa_type,
        "content_parent_category": example.content_parent_category,
        "content_fine_category": example.content_fine_category,
        "video_category": example.video_category,
        "video_duration": example.video_duration,
        "media_prepared": False,
    }


def _media_diagnostics(
    example: DailyOmniExample,
    media: PreparedMedia,
    config: EvalConfig,
) -> dict[str, Any]:
    return {
        **_metadata_diagnostics(example, config),
        "media_prepared": True,
        "total_frames": media.total_frames,
        "avg_fps": media.avg_fps,
        "sampled_frames": len(media.frame_indices),
        "frame_indices_head": media.frame_indices[:8],
        "frame_indices_tail": media.frame_indices[-8:],
        "frame_timestamps_head": media.frame_timestamps[:8],
        "frame_timestamps_tail": media.frame_timestamps[-8:],
        "audio_samples": int(media.audio.shape[0]) if media.audio is not None else None,
        "audio_segment_count": len(media.audio_segments),
        "audio_segment_samples_head": [_num_samples(segment) for segment in media.audio_segments[:8]],
        "audio_sample_rate": media.audio_sample_rate,
    }


def _run_with_invalid_answer_retry(
    omni: Any,
    example: DailyOmniExample,
    media: PreparedMedia,
    config: EvalConfig,
) -> tuple[str, list[str], ScoreResult]:
    """当答案抽取找不到 A/B/C/D 时重试一次。"""
    raw_responses: list[str] = []
    score: ScoreResult | None = None
    max_attempts = config.generation.invalid_answer_retries + 1

    for _ in range(max_attempts):
        inference_result = run_one(omni, example, media, config)
        raw_responses.append(inference_result.raw_response)
        score = score_response(inference_result.raw_response, example.answer)
        if score.response is not None:
            break

    if score is None:
        raise RuntimeError("没有执行任何推理尝试。")
    return raw_responses[-1], raw_responses, score


def _record(
    example: DailyOmniExample,
    media: PreparedMedia,
    raw_response: str,
    raw_responses: list[str],
    score: ScoreResult,
    config: EvalConfig,
) -> dict[str, Any]:
    return {
        "index": example.index,
        "question_id": example.question_id,
        "video_id": example.video_id,
        "dataset_name": example.dataset_name,
        "dataset_type": example.dataset_type,
        "input_mode": config.media.input_mode.value,
        "qa_type": example.qa_type,
        "content_parent_category": example.content_parent_category,
        "content_fine_category": example.content_fine_category,
        "video_category": example.video_category,
        "video_duration": example.video_duration,
        "video_path": str(example.video_path),
        "audio_path": str(example.audio_path),
        "question": example.question,
        "options": example.options,
        "answer": score.answer,
        "response": score.response,
        "raw_response": raw_response,
        "raw_responses": raw_responses,
        "retry_count": max(0, len(raw_responses) - 1),
        "hit": score.hit,
        "frame_indices": media.frame_indices,
        "frame_timestamps": media.frame_timestamps,
        "avg_fps": media.avg_fps,
        "total_frames": media.total_frames,
        "audio_samples": int(media.audio.shape[0]) if media.audio is not None else None,
        "audio_segment_count": len(media.audio_segments),
        "audio_segment_samples": [_num_samples(segment) for segment in media.audio_segments],
        "audio_sample_rate": media.audio_sample_rate,
    }


def _num_samples(audio: Any) -> int:
    if hasattr(audio, "shape"):
        return int(audio.shape[0])
    return len(audio)


def _load_completed_question_ids(output_path: Path) -> set[str]:
    """读取已有 JSONL 输出，让中断后的运行可以原地续跑。"""
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
                raise ValueError(f"JSONL 格式错误: {output_path}:{line_number}") from exc
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

    return {
        **_bucket_stats(records),
        "invalid_response": sum(1 for record in records if record.get("response") is None),
        "by_qa_type": _grouped_stats(records, "qa_type"),
        "by_content_parent_category": _grouped_stats(records, "content_parent_category"),
        "by_content_fine_category": _grouped_stats(records, "content_fine_category"),
        "by_video_category": _grouped_stats(records, "video_category"),
        "by_video_duration": _grouped_stats(records, "video_duration"),
    }


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
