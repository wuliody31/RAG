from app.core.schemas import SearchResult
from app.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_deduplicates_and_sorts():
    a = SearchResult(id="1", text="a", score=0.9, metadata={})
    b = SearchResult(id="2", text="b", score=0.8, metadata={})
    result = reciprocal_rank_fusion([[a, b], [b]])
    assert [x.id for x in result][0] == "b"
    assert len(result) == 2
