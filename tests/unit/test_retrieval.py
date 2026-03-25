"""Unit tests for Phase 4 — Elasticsearch candidate retrieval (mocked ES client)."""

import numpy as np
from unittest.mock import MagicMock

from src.retrieval.elastic_store import ElasticsearchItemStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOTAL_DIM = 192
CF_DIM = 128


def _make_hit(item_id: str, title: str = "Test Item", score: float = 0.9) -> dict:
    return {
        "_id": item_id,
        "_score": score,
        "_source": {
            "item_id": item_id,
            "title": title,
            "category": "Electronics",
            "price": 29.99,
            "avg_rating": 4.5,
            "review_count": 100,
        },
    }


def _mock_es_response(n: int) -> dict:
    """Build a fake Elasticsearch search response with n hits."""
    return {"hits": {"hits": [_make_hit(f"item_{i}") for i in range(n)]}}


def _make_store(n_candidates: int = 100) -> tuple[ElasticsearchItemStore, MagicMock]:
    """Return (store, mock_es_client) pre-configured to return n_candidates hits."""
    mock_es = MagicMock()
    mock_es.search.return_value = _mock_es_response(n_candidates)

    # Small synthetic embeddings for cold-start / retrieve_by_items tests
    embeddings = {f"item_{i}": np.random.rand(TOTAL_DIM).astype(np.float32) for i in range(20)}
    proj = np.zeros((TOTAL_DIM, CF_DIM), dtype=np.float32)
    proj[:CF_DIM, :CF_DIM] = np.eye(CF_DIM, dtype=np.float32)

    store = ElasticsearchItemStore(
        client=mock_es, embeddings=embeddings, projection_matrix=proj
    )
    return store, mock_es


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestElasticsearchItemStore:
    def test_retrieve_returns_n_candidates(self):
        store, _ = _make_store(n_candidates=100)
        user_emb = np.random.rand(CF_DIM).astype(np.float32)
        results = store.retrieve_candidates(user_emb, n=100)
        assert len(results) == 100

    def test_retrieved_items_have_required_fields(self):
        store, _ = _make_store(n_candidates=10)
        user_emb = np.random.rand(CF_DIM).astype(np.float32)
        results = store.retrieve_candidates(user_emb, n=10)
        required = {"item_id", "title", "price", "avg_rating"}
        for item in results:
            missing = required - set(item.keys())
            assert not missing, f"Item missing fields: {missing}"

    def test_cold_start_retrieve_by_items(self):
        store, mock_es = _make_store(n_candidates=20)
        mock_es.search.return_value = _mock_es_response(20)
        results = store.retrieve_by_items(["item_0", "item_1", "item_2"], n=20)
        assert len(results) > 0

    def test_text_search_returns_relevant_results(self):
        """search_by_text calls ES with a match query and returns results."""
        store, mock_es = _make_store(n_candidates=5)
        mock_es.search.return_value = {
            "hits": {"hits": [
                _make_hit("h1", "Wireless Headphones Pro"),
                _make_hit("h2", "Bluetooth Headphones"),
            ]}
        }
        results = store.search_by_text("wireless headphones", n=5)
        assert len(results) > 0
        # Verify the ES call used a match query on title
        call_kwargs = mock_es.search.call_args.kwargs
        assert "match" in call_kwargs["query"]

    def test_retrieve_candidates_calls_knn(self):
        """retrieve_candidates must use ES knn query."""
        store, mock_es = _make_store(n_candidates=10)
        user_emb = np.random.rand(CF_DIM).astype(np.float32)
        store.retrieve_candidates(user_emb, n=10)
        call_kwargs = mock_es.search.call_args.kwargs
        assert "knn" in call_kwargs, "Expected knn query in ES search call"
        assert call_kwargs["knn"]["field"] == "embedding"

    def test_retrieve_by_items_empty_ids_returns_empty(self):
        """No matching item_ids → return empty list without calling ES."""
        store, mock_es = _make_store()
        results = store.retrieve_by_items(["nonexistent_item"], n=10)
        assert results == []
        mock_es.search.assert_not_called()
