from __future__ import annotations

import logging
from pathlib import Path
from time import time

from app.core.config import Settings, get_settings
from app.ingestion.chunker import RecursiveTextChunker
from app.ingestion.loaders import load_document
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import VectorStore
from app.utils.file_utils import iter_supported_files, sha256_file

logger = logging.getLogger(__name__)


class Ingestor:
    def __init__(
        self,
        settings: Settings | None = None,
        vector_store: VectorStore | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedding_model = embedding_model or EmbeddingModel(self.settings.embedding_model, self.settings.embedding_device)
        self.vector_store = vector_store or VectorStore(self.settings)
        self.chunker = RecursiveTextChunker(self.settings.chunk_size, self.settings.chunk_overlap)

    def ingest_path(self, path: str | Path) -> dict:
        path = Path(path)
        files = list(iter_supported_files(path))
        total_chunks = 0
        docs: list[dict] = []
        started = time()

        for file in files:
            result = self.ingest_file(file)
            docs.append(result)
            total_chunks += result.get("chunks", 0)

        return {
            "files_seen": len(files),
            "chunks_added": total_chunks,
            "seconds": round(time() - started, 3),
            "documents": docs,
        }

    def ingest_file(self, file: Path) -> dict:
        doc_id = sha256_file(file)
        logger.info("Loading %s", file)
        pages = load_document(file)
        chunks = self.chunker.split_pages(pages, doc_id=doc_id, source_path=file)
        if not chunks:
            return {"file": str(file), "doc_id": doc_id, "chunks": 0, "status": "empty"}

        embeddings = self.embedding_model.embed_documents([c.text for c in chunks])
        self.vector_store.upsert_chunks(chunks, embeddings)
        return {"file": str(file), "doc_id": doc_id, "chunks": len(chunks), "status": "indexed"}
