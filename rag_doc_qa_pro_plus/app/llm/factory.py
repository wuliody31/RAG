from __future__ import annotations

from app.core.config import Settings, get_settings
from app.llm.base import LLMClient
from app.llm.mock_client import MockLLMClient
from app.llm.ollama_client import OllamaClient
from app.llm.openai_client import OpenAIClient


def build_llm(settings: Settings | None = None) -> LLMClient:
    settings = settings or get_settings()
    if settings.llm_provider == "ollama":
        return OllamaClient(settings)
    if settings.llm_provider == "openai":
        return OpenAIClient(settings)
    if settings.llm_provider == "mock":
        return MockLLMClient()
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
