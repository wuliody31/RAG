from __future__ import annotations

import httpx

from app.core.config import Settings
from app.llm.base import LLMClient


class OllamaClient(LLMClient):
    def __init__(self, settings: Settings) -> None:
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_model
        self.temperature = settings.temperature
        self.max_output_tokens = settings.max_output_tokens

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_output_tokens,
            },
        }
        try:
            with httpx.Client(timeout=120) as client:
                response = client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
                return data.get("message", {}).get("content", "").strip()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                "Ollama request failed. Check that `ollama serve` is running and the model is pulled."
            ) from exc
