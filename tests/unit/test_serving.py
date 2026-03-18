"""Unit tests for Phase 5 — serving schemas and recommendation pipeline."""

import pytest
from unittest.mock import MagicMock
from pydantic import ValidationError

from src.serving.schemas import RecommendationRequest, RecommendationResponse, RecommendedItem
from src.serving.pipeline import RecommendationPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_item(item_id: str, score: float = 0.8) -> dict:
    return {
        "item_id": item_id,
        "title": f"Product {item_id}",
        "category": "Electronics",
        "price": 29.99,
        "avg_rating": 4.5,
        "review_count": 100,
        "_score": score,
    }


def _make_pipeline(
    candidates: list[dict] | None = None,
    history: list[str] | None = None,
    has_cf_user: bool = True,
) -> RecommendationPipeline:
    """Build a pipeline with all dependencies mocked."""
    candidates = candidates or [_make_item(f"item_{i}") for i in range(20)]
    history = history or []

    redis = MagicMock()
    redis.get_session_features.return_value = {
        "user_id": "u1",
        "recent_views": ["item_0", "item_1"],
        "session_length": 3,
        "session_age_seconds": 120.0,
        "has_cart_item": False,
    }
    redis.get_user_history.return_value = history

    cf = MagicMock()
    cf.has_user.return_value = has_cf_user
    cf.get_user_embedding.return_value = __import__("numpy").zeros(128, dtype="float32")

    es = MagicMock()
    es.retrieve_candidates.return_value = candidates
    es.retrieve_by_items.return_value = candidates
    es.get_popular_items.return_value = candidates

    ranker = MagicMock()
    ranker.predict_scores.return_value = [float(i) / len(candidates) for i in range(len(candidates))]

    fb = MagicMock()
    fb.build.side_effect = lambda sess, item, ctx: {
        "user_id": sess["user_id"],
        "candidate_item_id": item["item_id"],
        "session_length": 3,
        "session_age_seconds": 120.0,
        "n_recent_views": 2,
        "candidate_in_recent_views": False,
        "candidate_avg_rating": 4.5,
        "candidate_price": 29.99,
        "candidate_review_count": 100,
        "time_of_day": 0.5,
        "device_mobile": False,
    }

    return RecommendationPipeline(
        redis_store=redis,
        elastic_store=es,
        cf_model=cf,
        ranker=ranker,
        feature_builder=fb,
    )


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchemas:
    def test_request_schema_validates_correctly(self):
        req = RecommendationRequest(user_id="user_123", n_recommendations=10)
        assert req.user_id == "user_123"
        assert req.n_recommendations == 10

    def test_n_recommendations_max_enforced(self):
        with pytest.raises(ValidationError):
            RecommendationRequest(user_id="u1", n_recommendations=100)

    def test_n_recommendations_min_enforced(self):
        with pytest.raises(ValidationError):
            RecommendationRequest(user_id="u1", n_recommendations=0)

    def test_context_is_optional(self):
        req = RecommendationRequest(user_id="u1")
        assert req.context is None


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------

class TestRecommendationPipeline:
    def test_response_contains_latency_fields(self):
        pipeline = _make_pipeline()
        resp = pipeline.recommend("u1", n=5)
        assert resp.retrieval_ms >= 0
        assert resp.ranking_ms >= 0
        assert resp.total_ms >= 0

    def test_cold_start_returns_fallback(self):
        """User with no CF embedding falls back to popular items — still returns results."""
        pipeline = _make_pipeline(has_cf_user=False)
        resp = pipeline.recommend("new_user_xyz", n=5)
        assert len(resp.recommendations) > 0

    def test_history_items_excluded_from_results(self):
        """Items in user history must not appear in recommendations."""
        history = [f"item_{i}" for i in range(10)]
        candidates = [_make_item(f"item_{i}") for i in range(15)]
        pipeline = _make_pipeline(candidates=candidates, history=history)
        resp = pipeline.recommend("u1", n=10)
        result_ids = {r.item_id for r in resp.recommendations}
        overlap = result_ids & set(history)
        assert not overlap, f"History items leaked into results: {overlap}"

    def test_returns_at_most_n_recommendations(self):
        pipeline = _make_pipeline(candidates=[_make_item(f"i{i}") for i in range(50)])
        resp = pipeline.recommend("u1", n=5)
        assert len(resp.recommendations) <= 5

    def test_response_model_version_set(self):
        pipeline = _make_pipeline()
        resp = pipeline.recommend("u1", n=3)
        assert resp.model_version != ""
