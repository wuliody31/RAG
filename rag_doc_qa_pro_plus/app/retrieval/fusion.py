from __future__ import annotations

from collections import defaultdict

from app.core.schemas import SearchResult


def reciprocal_rank_fusion(result_sets: list[list[SearchResult]], k: int = 60) -> list[SearchResult]:
    """Fuse multiple ranked result lists using RRF."""
    scores: dict[str, float] = defaultdict(float)
    best: dict[str, SearchResult] = {}

    for results in result_sets:
        for rank, item in enumerate(results, start=1):
            scores[item.id] += 1.0 / (k + rank)
            if item.id not in best or item.score > best[item.id].score:
                best[item.id] = item

    fused: list[SearchResult] = []
    for chunk_id, score in scores.items():
        item = best[chunk_id]
        fused.append(SearchResult(id=item.id, text=item.text, metadata=item.metadata, score=score, source="rrf"))
    return sorted(fused, key=lambda x: x.score, reverse=True)
