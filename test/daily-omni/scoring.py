"""Daily-Omni 答案抽取与评分辅助函数。"""

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
    """从模型文本中抽取单个 A-D 选项。"""
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
    """使用抽取出的答案字母为单条回复评分。"""
    expected = answer.strip().upper()[:1]
    response = extract_answer(raw_response)
    return ScoreResult(
        response=response,
        answer=expected,
        hit=response == expected,
    )
