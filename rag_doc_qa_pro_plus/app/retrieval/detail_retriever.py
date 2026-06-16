from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Literal

from app.core.config import Settings, get_settings
from app.core.schemas import SearchResult
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import Reranker
from app.retrieval.vector_store import VectorStore

SearchMode = Literal["hybrid", "semantic", "keyword", "exact"]


class DetailRetriever:
    """Fine-grained document retrieval layer.

    The normal RAG retriever focuses on building a compact LLM context. This
    class focuses on user-facing document inspection: exact phrase search,
    page/section filtering, full chunk return and neighbor expansion.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        vector_store: VectorStore | None = None,
        embedding_model: EmbeddingModel | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.vector_store = vector_store or VectorStore(self.settings)
        self.embedding_model = embedding_model or EmbeddingModel(
            self.settings.embedding_model, self.settings.embedding_device
        )
        self.reranker = reranker or Reranker(self.settings.reranker_model, self.settings.reranker_device)

    def search(
        self,
        query: str,
        mode: SearchMode = "hybrid",
        top_k: int = 10,
        doc_id: str | None = None,
        file_name: str | None = None,
        page_from: int | None = None,
        page_to: int | None = None,
        section: str | None = None,
        include_neighbors: bool = False,
        neighbor_window: int = 1,
        rerank: bool = True,
    ) -> list[SearchResult]:
        query = query.strip()
        if not query:
            return []

        base_chunks = self._filtered_chunks(doc_id, file_name, page_from, page_to, section)
        if not base_chunks:
            return []

        if mode == "exact":
            results = self._exact_search(query, base_chunks, top_k)
        elif mode == "keyword":
            results = BM25Retriever(base_chunks).search(query, top_k)
            for r in results:
                r.source = "keyword-detail"
        elif mode == "semantic":
            results = self._semantic_search(query, top_k, doc_id, file_name, page_from, page_to, section)
        else:
            exact = self._exact_search(query, base_chunks, top_k * 2)
            bm25 = BM25Retriever(base_chunks).search(query, top_k * 3)
            dense = self._semantic_search(query, top_k * 3, doc_id, file_name, page_from, page_to, section)
            results = reciprocal_rank_fusion([dense, bm25, exact])
            for r in results:
                r.source = f"hybrid-detail:{r.source}"
            results = results[: max(top_k * 2, top_k)]

        if rerank and len(results) > 1 and mode in {"hybrid", "semantic", "keyword"}:
            results = self.reranker.rerank(query, results, min(len(results), max(top_k * 2, top_k)))

        results = self._dedupe(results)[:top_k]
        if include_neighbors:
            results = self.expand_with_neighbors(results, neighbor_window)
        return results

    def expand_with_neighbors(self, results: list[SearchResult], window: int = 1) -> list[SearchResult]:
        expanded: list[SearchResult] = []
        matched_ids = {r.id for r in results}
        base_scores = {r.id: r.score for r in results}
        for result in results:
            for neighbor in self.vector_store.get_neighbor_chunks(result.id, window=window):
                if neighbor.id in matched_ids:
                    neighbor.score = base_scores.get(neighbor.id, result.score)
                    neighbor.source = result.source
                else:
                    # Neighbor chunks are added for reading continuity, not because
                    # they matched the query directly.
                    neighbor.score = max(result.score - 0.05, 0.0)
                    neighbor.source = "neighbor-context"
                expanded.append(neighbor)
        return self._sort_document_order(self._dedupe(expanded))

    def get_chunk_with_neighbors(self, chunk_id: str, window: int = 1) -> list[SearchResult]:
        return self.vector_store.get_neighbor_chunks(chunk_id, window=window)

    def _semantic_search(
        self,
        query: str,
        top_k: int,
        doc_id: str | None,
        file_name: str | None,
        page_from: int | None,
        page_to: int | None,
        section: str | None,
    ) -> list[SearchResult]:
        # Push the simplest metadata predicates down to Chroma; apply richer
        # page/section predicates in Python afterwards.
        where: dict[str, Any] | None = None
        if doc_id:
            where = {"doc_id": doc_id}
        elif file_name:
            where = {"file_name": file_name}

        query_emb = self.embedding_model.embed_query(query)
        dense = self.vector_store.query_dense(query_emb, max(top_k * 5, top_k), where)
        return self._apply_python_filters(dense, doc_id, file_name, page_from, page_to, section)[:top_k]

    def _exact_search(self, query: str, chunks: list[SearchResult], top_k: int) -> list[SearchResult]:
        lowered_query = query.casefold()
        terms = [t.casefold() for t in re.findall(r"[\w\u4e00-\u9fff]+", query) if len(t) > 1]
        results: list[SearchResult] = []
        for chunk in chunks:
            text_lower = chunk.text.casefold()
            phrase_hits = text_lower.count(lowered_query)
            term_hits = sum(text_lower.count(t) for t in terms)
            if phrase_hits == 0 and term_hits == 0:
                continue
            # Strongly reward exact phrase hits, then term coverage, then shorter chunks.
            coverage = len({t for t in terms if t in text_lower}) / max(len(set(terms)), 1)
            length_penalty = 1 / math.sqrt(max(len(chunk.text), 1))
            chunk.score = 3.0 * phrase_hits + 0.25 * term_hits + coverage + length_penalty
            chunk.source = "exact-detail"
            results.append(chunk)
        return sorted(results, key=lambda r: r.score, reverse=True)[:top_k]

    def _filtered_chunks(
        self,
        doc_id: str | None,
        file_name: str | None,
        page_from: int | None,
        page_to: int | None,
        section: str | None,
    ) -> list[SearchResult]:
        where: dict[str, Any] | None = None
        if doc_id:
            where = {"doc_id": doc_id}
        elif file_name:
            where = {"file_name": file_name}
        chunks = self.vector_store.get_all_chunks(where)
        return self._apply_python_filters(chunks, doc_id, file_name, page_from, page_to, section)

    @staticmethod
    def _apply_python_filters(
        chunks: list[SearchResult],
        doc_id: str | None,
        file_name: str | None,
        page_from: int | None,
        page_to: int | None,
        section: str | None,
    ) -> list[SearchResult]:
        out: list[SearchResult] = []
        section_lower = section.casefold() if section else None
        for chunk in chunks:
            meta = chunk.metadata
            if doc_id and meta.get("doc_id") != doc_id:
                continue
            if file_name and meta.get("file_name") != file_name:
                continue
            page = meta.get("page")
            try:
                page_int = int(page) if page is not None else None
            except (TypeError, ValueError):
                page_int = None
            if page_from is not None and (page_int is None or page_int < page_from):
                continue
            if page_to is not None and (page_int is None or page_int > page_to):
                continue
            if section_lower:
                section_title = str(meta.get("section_title") or "").casefold()
                if section_lower not in section_title:
                    continue
            out.append(chunk)
        return out

    @staticmethod
    def _dedupe(results: list[SearchResult]) -> list[SearchResult]:
        best: dict[str, SearchResult] = {}
        for result in results:
            old = best.get(result.id)
            if old is None or result.score > old.score:
                best[result.id] = result
        return sorted(best.values(), key=lambda r: r.score, reverse=True)

    @staticmethod
    def _sort_document_order(results: list[SearchResult]) -> list[SearchResult]:
        grouped: dict[str, list[SearchResult]] = defaultdict(list)
        for r in results:
            grouped[str(r.metadata.get("doc_id", "unknown"))].append(r)
        ordered: list[SearchResult] = []
        for group in grouped.values():
            ordered.extend(
                sorted(
                    group,
                    key=lambda r: (
                        int(r.metadata.get("chunk_seq", 0)) if str(r.metadata.get("chunk_seq", "0")).isdigit() else 0,
                        r.id,
                    ),
                )
            )
        return ordered


def result_to_detail_dict(result: SearchResult, full_text: bool = True) -> dict[str, Any]:
    text = result.text if full_text else result.text[:800]
    return {
        "chunk_id": result.id,
        "doc_id": result.metadata.get("doc_id"),
        "file_name": result.metadata.get("file_name"),
        "source_path": result.metadata.get("source_path"),
        "file_type": result.metadata.get("file_type"),
        "page": result.metadata.get("page"),
        "chunk_seq": result.metadata.get("chunk_seq"),
        "section_title": result.metadata.get("section_title"),
        "score": result.score,
        "source": result.source,
        "text": text,
        "metadata": result.metadata,
    }
