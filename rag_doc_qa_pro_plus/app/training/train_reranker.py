from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.training.io import read_jsonl


def train_reranker(
    settings: Settings | None = None,
    train_path: Path = Path("data/training/retrieval_pairs.jsonl"),
    base_model: str | None = None,
    output_dir: Path = Path("data/models/reranker_finetuned"),
    epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_examples: int | None = None,
) -> dict[str, Any]:
    """Fine-tune a CrossEncoder reranker using positive and hard-negative pairs."""
    settings = settings or get_settings()
    base_model = base_model or settings.reranker_model
    rows = read_jsonl(train_path)
    if max_examples:
        rows = rows[:max_examples]

    examples_raw: list[tuple[str, str, float]] = []
    for row in rows:
        q = str(row.get("query", "")).strip()
        pos = str(row.get("positive", "")).strip()
        neg = str(row.get("negative", "")).strip()
        if q and pos:
            examples_raw.append((q, pos, 1.0))
        if q and neg:
            examples_raw.append((q, neg, 0.0))
    if not examples_raw:
        raise ValueError(f"No valid reranker examples found in {train_path}")

    from torch.utils.data import DataLoader
    from sentence_transformers import CrossEncoder, InputExample

    model = CrossEncoder(base_model, num_labels=1)
    examples = [InputExample(texts=[q, d], label=label) for q, d, label in examples_raw]
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    warmup_steps = max(10, int(len(loader) * epochs * 0.1))
    model.fit(
        train_dataloader=loader,
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
        "message": "Reranker fine-tuned. Set RERANKER_MODEL to this output_dir and restart the API.",
    }
