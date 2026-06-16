from pathlib import Path

from app.core.schemas import PageText
from app.ingestion.chunker import RecursiveTextChunker


def test_chunker_preserves_doc_metadata():
    chunker = RecursiveTextChunker(chunk_size=50, chunk_overlap=10)
    pages = [PageText(text="第一段。第二段。第三段。" * 10, page=1)]
    chunks = chunker.split_pages(pages, doc_id="abc", source_path=Path("demo.pdf"))
    assert chunks
    assert chunks[0].metadata["doc_id"] == "abc"
    assert chunks[0].metadata["file_name"] == "demo.pdf"
    assert chunks[0].metadata["page"] == 1
