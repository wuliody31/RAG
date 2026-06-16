from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from app.retrieval.hybrid_retriever import HybridRetriever
from app.services.rag_service import RAGService


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def recall_at_k(retrieved_doc_ids: list[str], expected_doc_ids: list[str], k: int) -> float:
    if not expected_doc_ids:
        return 0.0
    retrieved = set(retrieved_doc_ids[:k])
    expected = set(expected_doc_ids)
    return len(retrieved & expected) / len(expected)


def keyword_coverage(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    answer_lower = answer.lower()
    return sum(1 for kw in keywords if kw.lower() in answer_lower) / len(keywords)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answer keyword coverage")
    parser.add_argument("--dataset", required=True, help="JSONL evaluation dataset")
    parser.add_argument("--generate", action="store_true", help="Also call LLM and evaluate answer keywords")
    args = parser.parse_args()

    dataset = load_jsonl(Path(args.dataset))
    retriever = HybridRetriever()
    rag = RAGService(retriever=retriever) if args.generate else None

    rows = []
    for item in dataset:
        question = item["question"]
        expected_doc_ids = item.get("expected_doc_ids", [])
        contexts = retriever.retrieve(question)
        retrieved_doc_ids = [c.metadata.get("doc_id", "") for c in contexts]
        row = {
            "question": question,
            "recall@3": recall_at_k(retrieved_doc_ids, expected_doc_ids, 3),
            "recall@5": recall_at_k(retrieved_doc_ids, expected_doc_ids, 5),
            "retrieved": retrieved_doc_ids,
        }
        if args.generate and rag:
            answer = rag.answer(question).answer
            row["keyword_coverage"] = keyword_coverage(answer, item.get("answer_keywords", []))
            row["answer"] = answer
        rows.append(row)

    summary = {
        "n": len(rows),
        "mean_recall@3": mean([r["recall@3"] for r in rows]) if rows else 0,
        "mean_recall@5": mean([r["recall@5"] for r in rows]) if rows else 0,
    }
    if args.generate and rows:
        summary["mean_keyword_coverage"] = mean([r.get("keyword_coverage", 0) for r in rows])

    print(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
