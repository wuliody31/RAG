from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.training.io import read_jsonl


def train_embedder(
    settings: Settings | None = None,
    train_path: Path = Path("data/training/retrieval_pairs.jsonl"),
    base_model: str | None = None,
    output_dir: Path = Path("data/models/embedding_finetuned"),
    epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_examples: int | None = None,
) -> dict[str, Any]:
    """Fine-tune a SentenceTransformer bi-encoder with MultipleNegativesRankingLoss.

    The training data should contain rows with at least `query` and `positive`.
    Negative texts are not directly used by MultipleNegativesRankingLoss; they are
    still useful for reranker training and future hard-negative mining.
    """
    settings = settings or get_settings()
    base_model = base_model or settings.embedding_model
    rows = read_jsonl(train_path)
    if max_examples:
        rows = rows[:max_examples]
    pairs = [(str(r["query"]), str(r["positive"])) for r in rows if r.get("query") and r.get("positive")]
    if not pairs:
        raise ValueError(f"No valid query-positive examples found in {train_path}")

    # Import inside function so normal API/QA usage does not pay training import cost.
    from torch.utils.data import DataLoader
    from sentence_transformers import InputExample, SentenceTransformer, losses

    model = SentenceTransformer(base_model)
    examples = [InputExample(texts=[q, p]) for q, p in pairs]
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)

    output_dir.mkdir(parents=True, exist_ok=True)
    warmup_steps = max(10, int(len(loader) * epochs * 0.1))
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": learning_rate},
        output_path=str(output_dir),
        show_progress_bar=True,
    )
    return {
        "status": "trained",
        "model_output_dir": str(output_dir),
        "train_examples": len(examples),
        "message": "Embedding model fine-tuned. Set EMBEDDING_MODEL to this output_dir and re-ingest documents.",
    }
