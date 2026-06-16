from app.core.schemas import SearchResult
from app.retrieval.detail_retriever import result_to_detail_dict


def test_result_to_detail_dict():
    result = SearchResult(
        id="doc:00001",
        text="hello world",
        score=0.9,
        source="exact-detail",
        metadata={"doc_id": "doc", "chunk_seq": 1, "page": 2, "file_name": "a.pdf"},
    )
    out = result_to_detail_dict(result, full_text=True)
    assert out["chunk_id"] == "doc:00001"
    assert out["text"] == "hello world"
    assert out["page"] == 2
