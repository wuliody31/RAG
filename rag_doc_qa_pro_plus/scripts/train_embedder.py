from __future__ import annotations

import argparse
from pathlib import Path

from rich import print

from app.core.config import get_settings
from app.training.train_embedder import train_embedder


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the embedding model for retrieval")
    parser.add_argument("--train", default="data/training/retrieval_pairs.jsonl")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--output", default="data/models/embedding_finetuned")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()

    report = train_embedder(
        settings=get_settings(),
        train_path=Path(args.train),
        base_model=args.base_model,
        output_dir=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_examples=args.max_examples,
    )
    print(report)


if __name__ == "__main__":
    main()
