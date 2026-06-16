from __future__ import annotations

from app.core.config import Settings
from app.llm.base import LLMClient


class OpenAIClient(LLMClient):
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install OpenAI support with: pip install -e '.[openai]'") from exc
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.temperature = settings.temperature
        self.max_output_tokens = settings.max_output_tokens

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=user_prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        return response.output_text.strip()
