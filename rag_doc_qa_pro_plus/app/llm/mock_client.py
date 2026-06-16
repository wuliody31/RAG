from __future__ import annotations

from app.llm.base import LLMClient


class MockLLMClient(LLMClient):
    """A deterministic fallback useful for API tests without an LLM server."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:  # noqa: ARG002
        return (
            "当前使用 mock LLM。系统已完成检索，但没有调用真实大模型。"
            "请把 LLM_PROVIDER 设置为 ollama 或 openai 以生成自然语言答案。"
        )
