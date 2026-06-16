from __future__ import annotations

import logging
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.core.schemas import SearchResult

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self.model = _load_reranker(model_name, device)

    def rerank(self, query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not candidates:
            return []
        pairs = [(query, c.text) for c in candidates]
        scores = self.model.predict(pairs, batch_size=16, show_progress_bar=False)
        reranked: list[SearchResult] = []
        for c, score in zip(candidates, scores):
            reranked.append(
                SearchResult(id=c.id, text=c.text, metadata=c.metadata, score=float(score), source="rerank")
            )
        return sorted(reranked, key=lambda x: x.score, reverse=True)[:top_k]


@lru_cache(maxsize=2)
def _load_reranker(model_name: str, device: str) -> CrossEncoder:
    logger.info("Loading reranker: %s on %s", model_name, device)
    return CrossEncoder(model_name, device=device)
