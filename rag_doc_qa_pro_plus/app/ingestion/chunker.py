from __future__ import annotations

import re
from pathlib import Path

from app.core.schemas import PageText, TextChunk
from app.utils.file_utils import sha256_text

_SENTENCE_SPLIT = re.compile(r"(?<=[。！？.!?])\s+|\n{2,}")


class RecursiveTextChunker:
    """Simple multilingual recursive chunker using paragraph/sentence/character fallback."""

    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 150) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_pages(self, pages: list[PageText], doc_id: str, source_path: Path) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        seq = 0
        for page in pages:
            for text in self._split_text(page.text):
                text = text.strip()
                if not text:
                    continue
                chunk_id = f"{doc_id}:{seq:05d}"
                chunks.append(
                    TextChunk(
                        id=chunk_id,
                        doc_id=doc_id,
                        text=text,
                        metadata={
                            "doc_id": doc_id,
                            "chunk_seq": seq,
                            "source_path": str(source_path),
                            "file_name": source_path.name,
                            "file_type": source_path.suffix.lower(),
                            "page": page.page,
                            "text_hash": sha256_text(text),
                            "char_count": len(text),
                            "token_estimate": max(1, len(text) // 4),
                            "section_title": self._infer_section_title(text),
                            **page.metadata,
                        },
                    )
                )
                seq += 1
        return chunks

    @staticmethod
    def _infer_section_title(text: str) -> str | None:
        """Infer a lightweight section title from the first few lines of a chunk.

        This is intentionally heuristic: it works for Markdown headings, numbered
        report sections and many PDF-extracted headings, without adding a heavy
        document-layout dependency.
        """
        for raw_line in text.splitlines()[:5]:
            line = raw_line.strip().strip("#").strip()
            if not line:
                continue
            if len(line) <= 90 and (
                raw_line.lstrip().startswith("#")
                or re.match(r"^(\d+(\.\d+)*|[A-Z])\s+[A-Z0-9一-龥]", line)
                or line.isupper()
            ):
                return line
        return None

    def _split_text(self, text: str) -> list[str]:
        text = text.strip()
        if len(text) <= self.chunk_size:
            return [text]

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        units: list[str] = []
        for p in paragraphs:
            if len(p) <= self.chunk_size:
                units.append(p)
            else:
                units.extend([s.strip() for s in _SENTENCE_SPLIT.split(p) if s.strip()])

        chunks: list[str] = []
        current = ""
        for unit in units:
            if len(unit) > self.chunk_size:
                if current:
                    chunks.append(current.strip())
                    current = ""
                chunks.extend(self._char_chunks(unit))
                continue
            if len(current) + len(unit) + 2 <= self.chunk_size:
                current = f"{current}\n{unit}" if current else unit
            else:
                if current:
                    chunks.append(current.strip())
                current = self._with_overlap(chunks[-1] if chunks else "", unit)
        if current:
            chunks.append(current.strip())
        return chunks

    def _char_chunks(self, text: str) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = max(0, end - self.chunk_overlap)
        return chunks

    def _with_overlap(self, previous: str, next_unit: str) -> str:
        if not previous or self.chunk_overlap <= 0:
            return next_unit
        overlap = previous[-self.chunk_overlap :]
        return f"{overlap}\n{next_unit}"
