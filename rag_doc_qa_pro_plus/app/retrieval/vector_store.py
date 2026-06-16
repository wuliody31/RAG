from __future__ import annotations

import logging
from typing import Any

import chromadb

from app.core.config import Settings, get_settings
from app.core.schemas import SearchResult, TextChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Persistent ChromaDB vector store.

    Besides dense vector search, this wrapper exposes document-level and
    chunk-level lookup helpers used by detail retrieval, citation expansion and
    model-training data mining.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = chromadb.PersistentClient(path=str(self.settings.chroma_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: list[TextChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        ids = [c.id for c in chunks]
        docs = [c.text for c in chunks]
        metadatas = [self._sanitize_metadata(c.metadata) for c in chunks]
        self.collection.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metadatas)
        logger.info("Upserted %s chunks", len(chunks))

    def query_dense(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            kwargs["where"] = filters
        raw = self.collection.query(**kwargs)
        results: list[SearchResult] = []
        for idx, doc in enumerate(raw.get("documents", [[]])[0]):
            chunk_id = raw.get("ids", [[]])[0][idx]
            metadata = raw.get("metadatas", [[]])[0][idx] or {}
            distance = float(raw.get("distances", [[]])[0][idx])
            score = 1.0 - distance
            results.append(SearchResult(id=chunk_id, text=doc, metadata=metadata, score=score, source="dense"))
        return results

    def get_all_chunks(self, filters: dict[str, Any] | None = None) -> list[SearchResult]:
        kwargs: dict[str, Any] = {"include": ["documents", "metadatas"]}
        if filters:
            kwargs["where"] = filters
        raw = self.collection.get(**kwargs)
        return self._raw_get_to_results(raw, source="store")

    def get_chunks_by_ids(self, ids: list[str], source: str = "lookup") -> list[SearchResult]:
        if not ids:
            return []
        raw = self.collection.get(ids=ids, include=["documents", "metadatas"])
        results = self._raw_get_to_results(raw, source=source)
        by_id = {r.id: r for r in results}
        return [by_id[i] for i in ids if i in by_id]

    def get_chunk(self, chunk_id: str) -> SearchResult | None:
        chunks = self.get_chunks_by_ids([chunk_id])
        return chunks[0] if chunks else None

    def get_document_chunks(self, doc_id: str) -> list[SearchResult]:
        chunks = self.get_all_chunks(filters={"doc_id": doc_id})
        return sorted(chunks, key=self._chunk_sort_key)

    def get_neighbor_chunks(self, chunk_id: str, window: int = 1) -> list[SearchResult]:
        """Return surrounding chunks from the same document, including the target chunk."""
        target = self.get_chunk(chunk_id)
        if not target:
            return []
        doc_id = str(target.metadata.get("doc_id", ""))
        seq = int(target.metadata.get("chunk_seq", -1))
        if not doc_id or seq < 0:
            return [target]
        ids = [f"{doc_id}:{i:05d}" for i in range(max(0, seq - window), seq + window + 1)]
        return self.get_chunks_by_ids(ids, source="neighbor")

    def list_documents(self) -> list[dict[str, Any]]:
        chunks = self.get_all_chunks()
        docs: dict[str, dict[str, Any]] = {}
        for chunk in chunks:
            doc_id = chunk.metadata.get("doc_id", "unknown")
            record = docs.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "file_name": chunk.metadata.get("file_name"),
                    "source_path": chunk.metadata.get("source_path"),
                    "file_type": chunk.metadata.get("file_type"),
                    "chunks": 0,
                    "pages": set(),
                    "sections": set(),
                },
            )
            record["chunks"] += 1
            page = chunk.metadata.get("page")
            if page is not None:
                record["pages"].add(page)
            section = chunk.metadata.get("section_title")
            if section:
                record["sections"].add(section)
        normalized = []
        for doc in docs.values():
            pages = sorted(doc.pop("pages"), key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)))
            sections = sorted(doc.pop("sections"))
            doc["page_count"] = len(pages)
            doc["pages"] = pages[:20]
            doc["section_count"] = len(sections)
            doc["sections_preview"] = sections[:10]
            normalized.append(doc)
        return sorted(normalized, key=lambda x: str(x.get("file_name")))

    def document_profile(self, doc_id: str, preview_chunks: int = 5) -> dict[str, Any]:
        chunks = self.get_document_chunks(doc_id)
        if not chunks:
            return {"doc_id": doc_id, "found": False}
        pages = sorted(
            {c.metadata.get("page") for c in chunks if c.metadata.get("page") is not None},
            key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)),
        )
        sections: list[dict[str, Any]] = []
        seen_sections: set[str] = set()
        for c in chunks:
            section = c.metadata.get("section_title")
            if section and section not in seen_sections:
                seen_sections.add(str(section))
                sections.append(
                    {
                        "section_title": section,
                        "chunk_id": c.id,
                        "page": c.metadata.get("page"),
                        "chunk_seq": c.metadata.get("chunk_seq"),
                    }
                )
        return {
            "doc_id": doc_id,
            "found": True,
            "file_name": chunks[0].metadata.get("file_name"),
            "source_path": chunks[0].metadata.get("source_path"),
            "file_type": chunks[0].metadata.get("file_type"),
            "chunks": len(chunks),
            "page_count": len(pages),
            "pages": pages,
            "sections": sections,
            "preview": [self._chunk_to_dict(c, max_chars=400) for c in chunks[:preview_chunks]],
        }

    def delete_document(self, doc_id: str) -> None:
        self.collection.delete(where={"doc_id": doc_id})

    def count(self) -> int:
        return self.collection.count()

    @staticmethod
    def _raw_get_to_results(raw: dict[str, Any], source: str) -> list[SearchResult]:
        results: list[SearchResult] = []
        for idx, doc in enumerate(raw.get("documents", [])):
            metadata = raw.get("metadatas", [])[idx] or {}
            results.append(
                SearchResult(id=raw.get("ids", [])[idx], text=doc or "", metadata=metadata, score=0.0, source=source)
            )
        return results

    @staticmethod
    def _chunk_to_dict(chunk: SearchResult, max_chars: int = 800) -> dict[str, Any]:
        return {
            "chunk_id": chunk.id,
            "doc_id": chunk.metadata.get("doc_id"),
            "chunk_seq": chunk.metadata.get("chunk_seq"),
            "page": chunk.metadata.get("page"),
            "section_title": chunk.metadata.get("section_title"),
            "file_name": chunk.metadata.get("file_name"),
            "score": chunk.score,
            "source": chunk.source,
            "preview": chunk.text[:max_chars],
        }

    @staticmethod
    def _chunk_sort_key(chunk: SearchResult) -> tuple[int, int, str]:
        page = chunk.metadata.get("page")
        seq = chunk.metadata.get("chunk_seq", 0)
        try:
            page_num = int(page) if page is not None else -1
        except (TypeError, ValueError):
            page_num = -1
        try:
            seq_num = int(seq)
        except (TypeError, ValueError):
            seq_num = 0
        return (page_num, seq_num, chunk.id)

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        sanitized: dict[str, Any] = {}
        for k, v in metadata.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                sanitized[k] = v
            else:
                sanitized[k] = str(v)
        return sanitized
