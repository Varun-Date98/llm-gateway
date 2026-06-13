from __future__ import annotations

import re
from dataclasses import dataclass, field

from gateway.schemas import GenerationRequest

REASONING_KEYWORDS = {
    "analyze",
    "architecture",
    "debug",
    "derive",
    "explain",
    "optimize",
    "prove",
    "reason",
    "refactor",
    "tradeoff",
    "why",
}

CODE_KEYWORDS = {
    "api",
    "async",
    "bug",
    "class",
    "code",
    "function",
    "pytest",
    "python",
    "sql",
    "stack trace",
    "typescript",
}


@dataclass(frozen=True)
class DifficultyFeatures:
    prompt_tokens: int
    max_tokens: int
    reasoning_hits: int
    code_hits: int
    message_count: int


@dataclass(frozen=True)
class HeuristicDifficultyClassifier:
    """Cheap, inspectable request difficulty scorer."""

    long_prompt_tokens: int = 2000
    long_completion_tokens: int = 512
    keyword_weight: float = 0.08
    code_weight: float = 0.07
    message_weight: float = 0.02
    reasoning_keywords: set[str] = field(default_factory=lambda: set(REASONING_KEYWORDS))
    code_keywords: set[str] = field(default_factory=lambda: set(CODE_KEYWORDS))

    def score(self, request: GenerationRequest) -> float:
        features = self.features(request)
        prompt_score = min(1.0, features.prompt_tokens / self.long_prompt_tokens)
        completion_score = min(1.0, features.max_tokens / self.long_completion_tokens)
        keyword_score = min(0.35, features.reasoning_hits * self.keyword_weight)
        code_score = min(0.25, features.code_hits * self.code_weight)
        conversation_score = min(0.15, max(0, features.message_count - 2) * self.message_weight)

        score = (
            0.45 * prompt_score
            + 0.25 * completion_score
            + keyword_score
            + code_score
            + conversation_score
        )
        return max(0.0, min(1.0, score))

    def features(self, request: GenerationRequest) -> DifficultyFeatures:
        text = request.prompt_text.lower()
        return DifficultyFeatures(
            prompt_tokens=request.prompt_tokens,
            max_tokens=request.max_tokens,
            reasoning_hits=count_keyword_hits(text, self.reasoning_keywords),
            code_hits=count_keyword_hits(text, self.code_keywords),
            message_count=len(request.messages),
        )


def count_keyword_hits(text: str, keywords: set[str]) -> int:
    hits = 0
    for keyword in keywords:
        pattern = re.escape(keyword)
        if re.search(rf"\b{pattern}\b", text):
            hits += 1
    return hits
