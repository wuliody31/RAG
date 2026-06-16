from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="User question")
    filters: dict[str, Any] | None = Field(default=None, description="Optional Chroma metadata filter")


class Citation(BaseModel):
    source_number: int
    chunk_id: str
    score: float
    file_name: str | None = None
    page: int | None = None
    source_path: str | None = None
    doc_id: str | None = None
    preview: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]


class DetailSearchRequest(BaseModel):
    query: str = Field(min_length=1, description="Phrase, keyword or semantic query")
    mode: Literal["hybrid", "semantic", "keyword", "exact"] = Field(default="hybrid")
    top_k: int = Field(default=10, ge=1, le=100)
    doc_id: str | None = None
    file_name: str | None = None
    page_from: int | None = Field(default=None, ge=1)
    page_to: int | None = Field(default=None, ge=1)
    section: str | None = None
    include_neighbors: bool = Field(default=False)
    neighbor_window: int = Field(default=1, ge=0, le=5)
    rerank: bool = Field(default=True)
    full_text: bool = Field(default=True)


class ChunkLookupRequest(BaseModel):
    chunk_id: str
    neighbor_window: int = Field(default=1, ge=0, le=10)
    full_text: bool = Field(default=True)


class DetailChunk(BaseModel):
    chunk_id: str
    doc_id: str | None = None
    file_name: str | None = None
    source_path: str | None = None
    file_type: str | None = None
    page: int | None = None
    chunk_seq: int | None = None
    section_title: str | None = None
    score: float
    source: str
    text: str
    metadata: dict[str, Any]


class DetailSearchResponse(BaseModel):
    query: str
    mode: str
    total: int
    chunks: list[DetailChunk]


class TrainingDatasetBuildRequest(BaseModel):
    eval_dataset_path: str = Field(default="data/eval/questions.jsonl")
    output_path: str = Field(default="data/training/retrieval_pairs.jsonl")
    negatives_per_query: int = Field(default=3, ge=1, le=20)
    top_k_pool: int = Field(default=50, ge=5, le=200)


class TrainingDatasetBuildResponse(BaseModel):
    output_path: str
    questions_seen: int
    pairs_written: int
    positives: int
    negatives: int


class EmbedderTrainRequest(BaseModel):
    train_path: str = Field(default="data/training/retrieval_pairs.jsonl")
    base_model: str | None = Field(default=None, description="Defaults to EMBEDDING_MODEL")
    output_dir: str = Field(default="data/models/embedding_finetuned")
    epochs: int = Field(default=1, ge=1, le=20)
    batch_size: int = Field(default=8, ge=1, le=128)
    learning_rate: float = Field(default=2e-5, gt=0)
    max_examples: int | None = Field(default=None, ge=1)


class RerankerTrainRequest(BaseModel):
    train_path: str = Field(default="data/training/retrieval_pairs.jsonl")
    base_model: str | None = Field(default=None, description="Defaults to RERANKER_MODEL")
    output_dir: str = Field(default="data/models/reranker_finetuned")
    epochs: int = Field(default=1, ge=1, le=20)
    batch_size: int = Field(default=8, ge=1, le=128)
    learning_rate: float = Field(default=2e-5, gt=0)
    max_examples: int | None = Field(default=None, ge=1)


class TrainResponse(BaseModel):
    status: str
    model_output_dir: str
    train_examples: int
    message: str


class UploadResponse(BaseModel):
    files_uploaded: int
    ingestion: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    chunks: int
    collection: str
    llm_provider: str
