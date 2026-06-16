from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.retrieval.detail_retriever import DetailRetriever
from app.retrieval.vector_store import VectorStore
from app.training.io import read_jsonl, write_jsonl


def build_training_pairs(
    settings: Settings | None = None,
    eval_dataset_path: Path = Path("data/eval/questions.jsonl"),
    output_path: Path = Path("data/training/retrieval_pairs.jsonl"),
    negatives_per_query: int = 3,
    top_k_pool: int = 50,
) -> dict[str, Any]:
    """Mine supervised retrieval examples from an evaluation JSONL file.

    Input rows can contain:
    - question: required
    - expected_doc_ids: optional list[str]
    - answer_keywords: optional list[str]

    Output rows use a simple pairwise format shared by embedder and reranker
    training:
    {"query": str, "positive": str, "negative": str, metadata...}
    """
    settings = settings or get_settings()
    rows = read_jsonl(eval_dataset_path)
    store = VectorStore(settings)
    detail = DetailRetriever(settings=settings, vector_store=store)
    all_chunks = store.get_all_chunks()
    if not all_chunks:
        raise RuntimeError("No indexed chunks found. Ingest documents before building training data.")

    output_rows: list[dict[str, Any]] = []
    total_pos = 0
    total_neg = 0
    rng = random.Random(42)

    for row in rows:
        question = str(row.get("question", "")).strip()
        if not question:
            continue
        expected_doc_ids = {str(x) for x in row.get("expected_doc_ids", [])}
        keywords = [str(x).casefold() for x in row.get("answer_keywords", [])]

        pool = detail.search(question, mode="hybrid", top_k=top_k_pool, rerank=True)
        positives = []
        negatives = []
        for chunk in pool or all_chunks:
            doc_match = bool(expected_doc_ids and str(chunk.metadata.get("doc_id")) in expected_doc_ids)
            keyword_match = bool(keywords and any(k in chunk.text.casefold() for k in keywords))
            if doc_match or keyword_match:
                positives.append(chunk)
            else:
                negatives.append(chunk)

        # Fallback: if labels are sparse, use top retrieved chunk as weak positive.
        if not positives and pool:
            positives = pool[:1]
            negatives = pool[1:] or [c for c in all_chunks if c.id != positives[0].id]

        if not positives:
            continue
        if not negatives:
            negatives = [c for c in all_chunks if c.id not in {p.id for p in positives}]
        rng.shuffle(negatives)

        for pos in positives:
            selected_negatives = negatives[:negatives_per_query]
            for neg in selected_negatives:
                output_rows.append(
                    {
                        "query": question,
                        "positive": pos.text,
                        "negative": neg.text,
                        "positive_chunk_id": pos.id,
                        "negative_chunk_id": neg.id,
                        "positive_doc_id": pos.metadata.get("doc_id"),
                        "negative_doc_id": neg.metadata.get("doc_id"),
                        "source": "mined_from_eval_dataset",
                    }
                )
            total_pos += 1
            total_neg += len(selected_negatives)

    pairs_written = write_jsonl(output_path, output_rows)
    return {
        "output_path": str(output_path),
        "questions_seen": len(rows),
        "pairs_written": pairs_written,
        "positives": total_pos,
        "negatives": total_neg,
    }
