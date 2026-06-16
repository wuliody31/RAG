from __future__ import annotations

from app.core.config import Settings, get_settings
from app.core.schemas import AnswerBundle, SearchResult
from app.llm.base import LLMClient
from app.llm.factory import build_llm
from app.retrieval.hybrid_retriever import HybridRetriever
from app.services.prompts import SYSTEM_PROMPT, build_user_prompt


class RAGService:
    def __init__(
        self,
        settings: Settings | None = None,
        retriever: HybridRetriever | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever or HybridRetriever(self.settings)
        self.llm = llm or build_llm(self.settings)

    def answer(self, question: str, filters: dict | None = None) -> AnswerBundle:
        contexts = self.retriever.retrieve(question, filters=filters)
        if not contexts:
            return AnswerBundle(
                question=question,
                answer="我没有在已入库文档中找到足够相关的内容。请先上传/入库文档，或换一个更具体的问题。",
                contexts=[],
                citations=[],
            )

        blocks = [self._format_context(i, c) for i, c in enumerate(contexts, start=1)]
        prompt = build_user_prompt(question, blocks)
        answer = self.llm.generate(SYSTEM_PROMPT, prompt)
        citations = [self._citation(i, c) for i, c in enumerate(contexts, start=1)]
        return AnswerBundle(question=question, answer=answer, contexts=contexts, citations=citations)

    @staticmethod
    def _format_context(i: int, result: SearchResult) -> str:
        file_name = result.metadata.get("file_name", "unknown")
        page = result.metadata.get("page", None)
        page_text = f", page={page}" if page else ""
        return f"[Source {i}: file={file_name}{page_text}, chunk_id={result.id}]\n{result.text}"

    @staticmethod
    def _citation(i: int, result: SearchResult) -> dict:
        return {
            "source_number": i,
            "chunk_id": result.id,
            "score": result.score,
            "file_name": result.metadata.get("file_name"),
            "page": result.metadata.get("page"),
            "source_path": result.metadata.get("source_path"),
            "doc_id": result.metadata.get("doc_id"),
            "preview": result.text[:300],
        }
