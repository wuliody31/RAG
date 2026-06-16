from __future__ import annotations

from app.core.config import Settings, get_settings
from app.core.schemas import SearchResult
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import Reranker
from app.retrieval.vector_store import VectorStore


class HybridRetriever:
    """Dense + BM25 + RRF + reranker retrieval pipeline."""

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

    def retrieve(self, question: str, filters: dict | None = None) -> list[SearchResult]:
        query_emb = self.embedding_model.embed_query(question)
        dense = self.vector_store.query_dense(query_emb, self.settings.dense_top_k, filters)
        all_chunks = self.vector_store.get_all_chunks(filters)
        bm25 = BM25Retriever(all_chunks).search(question, self.settings.bm25_top_k)
        fused = reciprocal_rank_fusion([dense, bm25])
        reranked = self.reranker.rerank(question, fused, self.settings.rerank_top_k)
        return reranked[: self.settings.final_context_k]
