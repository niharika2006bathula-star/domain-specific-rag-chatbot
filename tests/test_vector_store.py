from document_loader import DocumentChunk
from vector_store import VectorStore, build_vector_store


def test_embedding_and_retrieval_from_sample_dataset():
    chunks = [
        DocumentChunk("The office has blue windows and a white desk.", "office_policy.pdf", 1, 0),
        DocumentChunk("The meeting room uses a large projector screen.", "office_policy.pdf", 1, 1),
        DocumentChunk("Employee work hours are from 9AM to 5PM.", "office_policy.pdf", 2, 0),
    ]

    store = build_vector_store(chunks)
    results = store.retrieve("What color are the office windows?", top_k=2)

    assert results
    assert results[0].source == "office_policy.pdf"
    assert "blue" in results[0].text.lower()


def test_returns_no_results_when_index_is_empty():
    store = VectorStore([])
    results = store.retrieve("What is the policy?", top_k=4)
    assert results == []
