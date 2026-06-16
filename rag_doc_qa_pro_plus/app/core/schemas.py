from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PageText:
    text: str
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TextChunk:
    id: str
    doc_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class SearchResult:
    id: str
    text: str
    score: float
    metadata: dict[str, Any]
    source: str = "unknown"


@dataclass(slots=True)
class AnswerBundle:
    answer: str
    question: str
    contexts: list[SearchResult]
    citations: list[dict[str, Any]]
