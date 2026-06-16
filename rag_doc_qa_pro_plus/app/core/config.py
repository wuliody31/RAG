from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    chroma_dir: Path = Field(default=Path("./data/chroma"), alias="CHROMA_DIR")
    upload_dir: Path = Field(default=Path("./data/uploads"), alias="UPLOAD_DIR")
    collection_name: str = Field(default="rag_documents", alias="COLLECTION_NAME")
    training_dir: Path = Field(default=Path("./data/training"), alias="TRAINING_DIR")
    models_dir: Path = Field(default=Path("./data/models"), alias="MODELS_DIR")

    embedding_model: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3", alias="RERANKER_MODEL")
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")
    reranker_device: str = Field(default="cpu", alias="RERANKER_DEVICE")

    chunk_size: int = Field(default=900, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")
    dense_top_k: int = Field(default=30, alias="DENSE_TOP_K")
    bm25_top_k: int = Field(default=30, alias="BM25_TOP_K")
    rerank_top_k: int = Field(default=8, alias="RERANK_TOP_K")
    final_context_k: int = Field(default=6, alias="FINAL_CONTEXT_K")

    llm_provider: Literal["ollama", "openai", "mock"] = Field(default="ollama", alias="LLM_PROVIDER")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="qwen2.5:7b-instruct", alias="OLLAMA_MODEL")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")
    temperature: float = Field(default=0.1, alias="TEMPERATURE")
    max_output_tokens: int = Field(default=1200, alias="MAX_OUTPUT_TOKENS")

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "raw").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "processed").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "eval").mkdir(parents=True, exist_ok=True)
        self.training_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
