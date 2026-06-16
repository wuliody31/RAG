from __future__ import annotations

import argparse
from pathlib import Path

from rich import print

from app.core.config import get_settings
from app.training.dataset_builder import build_training_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine RAG training pairs from evaluation questions")
    parser.add_argument("--eval", default="data/eval/questions.jsonl", help="Input eval JSONL")
    parser.add_argument("--output", default="data/training/retrieval_pairs.jsonl", help="Output training JSONL")
    parser.add_argument("--negatives", type=int, default=3, help="Hard negatives per positive")
    parser.add_argument("--top-k-pool", type=int, default=50, help="Candidate pool size for mining")
    args = parser.parse_args()

    report = build_training_pairs(
        settings=get_settings(),
        eval_dataset_path=Path(args.eval),
        output_path=Path(args.output),
        negatives_per_query=args.negatives,
        top_k_pool=args.top_k_pool,
    )
    print(report)


if __name__ == "__main__":
    main()
