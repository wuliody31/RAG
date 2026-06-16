from __future__ import annotations

import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from app.core.schemas import SearchResult

_TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_PATTERN.findall(text)]


class BM25Retriever:
    """In-memory BM25 retriever rebuilt from current vector store chunks."""

    def __init__(self, chunks: list[SearchResult]) -> None:
        self.chunks = chunks
        self.corpus_tokens = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens) if chunks else None

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        if not self.bm25 or not self.chunks:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: float(x[1]), reverse=True)[:top_k]
        results: list[SearchResult] = []
        for idx, score in ranked:
            if float(score) <= 0:
                continue
            c = self.chunks[idx]
            results.append(SearchResult(id=c.id, text=c.text, metadata=c.metadata, score=float(score), source="bm25"))
        return results
