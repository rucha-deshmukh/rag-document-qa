"""Tests for the in-memory vector store."""

from app.store import InMemoryStore, cosine_similarity


def test_cosine_similarity_bounds():
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    assert cosine_similarity([1, 0], [0, 1]) == 0.0
    assert cosine_similarity([0, 0], [1, 1]) == 0.0


def test_add_and_search_ranks_by_similarity():
    store = InMemoryStore()
    store.add(
        "doc.txt",
        ["a", "b"],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    matches = store.search([1.0, 0.0], top_k=2)
    assert matches[0].text == "a"
    assert matches[0].score >= matches[1].score
