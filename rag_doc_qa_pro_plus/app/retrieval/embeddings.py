from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """SentenceTransformer wrapper. Uses normalized embeddings for cosine-like retrieval."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self.model = _load_embedding_model(model_name, device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
        )
        return np.asarray(vectors, dtype=np.float32).tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        return np.asarray(vector, dtype=np.float32).tolist()


@lru_cache(maxsize=2)
def _load_embedding_model(model_name: str, device: str) -> SentenceTransformer:
    logger.info("Loading embedding model: %s on %s", model_name, device)
    return SentenceTransformer(model_name, device=device)
