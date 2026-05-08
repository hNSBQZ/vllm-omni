"""Answer extraction and scoring helpers for Video-MME."""

from __future__ import annotations

import re
from dataclasses import dataclass


ANSWER_CHOICES = {"A", "B", "C", "D"}
ANSWER_SPLIT_CHARS = ".()[],:;!*#{}"


@dataclass(frozen=True)
class ScoreResult:
    response: str | None
    answer: str
    hit: bool


def extract_answer(raw_response: str) -> str | None:
    """Extract a single A-D choice from model text.

    This intentionally blends the previous CPP regex approach with the Python
    evaluator's punctuation-token matching, while keeping invalid responses as
    None so JSON output becomes null.
    """
    text = raw_response.strip()
    if not text:
        return None

    upper = text.upper()
    if upper in ANSWER_CHOICES:
        return upper

    normalized = text
    for char in ANSWER_SPLIT_CHARS:
        normalized = normalized.replace(char, " ")
    choices = [
        token.upper()
        for token in normalized.split()
        if token.upper() in ANSWER_CHOICES
    ]
    if len(choices) == 1:
        return choices[0]

    match = re.search(r"(?<![a-zA-Z])([A-D])(?![a-zA-Z])", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def score_response(raw_response: str, answer: str) -> ScoreResult:
    """Score one response using the extracted answer letter."""
    expected = answer.strip().upper()[:1]
    response = extract_answer(raw_response)
    return ScoreResult(
        response=response,
        answer=expected,
        hit=response == expected,
    )


def summarize_accuracy(results: list[ScoreResult]) -> dict[str, float | int]:
    """Return a minimal aggregate for smoke tests.

    TODO: add official Video-MME duration/domain/sub-category/task-type buckets
    once the output schema is finalized.
    """
    total = len(results)
    correct = sum(1 for result in results if result.hit)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
    }
