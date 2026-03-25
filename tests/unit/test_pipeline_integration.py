"""
Phase 7 — Pipeline integration tests.

Verifies that the FeatureBuilder, RedisFeatureStore, and RecommendationPipeline
work correctly together using in-memory fakes (no real Redis or ES needed).
"""

import pytest
import fakeredis
from unittest.mock import MagicMock

from src.features.redis_store import RedisFeatureStore
from src.features.feature_builder import FeatureBuilder
from src.models.ranker import FEATURE_COLS
from src.serving.pipeline import RecommendationPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_item(item_id: str) -> dict:
    return {
        "item_id": item_id,
        "title": f"Product {item_id}",
        "category": "Electronics",
        "price": 29.99,
        "avg_rating": 4.5,
        "review_count": 100,
        "_score": 0.9,
    }


def _make_view_event(user_id: str, item_id: str) -> dict:
    import time
    return {
        "event_id": "evt-001",
        "user_id": user_id,
        "item_id": item_id,
        "event_type": "view",
        "session_id": "sess-001",
        "timestamp": time.time(),
        "metadata": {"page": "product", "referrer": "search", "device": "mobile"},
    }


@pytest.fixture
def redis_store():
    fake = fakeredis.FakeRedis(decode_responses=True)
    return RedisFeatureStore(client=fake)


@pytest.fixture
def feature_builder():
    return FeatureBuilder()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    def test_feature_builder_output_matches_ranker_input(self, redis_store, feature_builder):
        """
        Feature dict from FeatureBuilder must contain all keys that XGBoost expects.
        """
        redis_store.update_session("user_1", _make_view_event("user_1", "item_A"))
        session = redis_store.get_session_features("user_1")
        features = feature_builder.build(session, _make_item("item_B"))

        missing = set(FEATURE_COLS) - set(features.keys())
        assert not missing, f"FeatureBuilder output missing ranker keys: {missing}"

    def test_redis_to_pipeline_roundtrip(self, redis_store, feature_builder):
        """
        Session updated in Redis → get_session_features → FeatureBuilder produces a
        valid, fully-populated feature dict with no None values.
        """
        for item_id in ["item_X", "item_Y", "item_Z"]:
            redis_store.update_session("user_2", _make_view_event("user_2", item_id))

        session = redis_store.get_session_features("user_2")
        features = feature_builder.build(
            session,
            _make_item("item_NEW"),
            context={"device": "mobile"},
        )

        assert features["session_length"] == 3
        assert features["n_recent_views"] == 3
        assert features["device_mobile"] is True
        assert all(v is not None for v in features.values()), "Feature dict has None values"

    def test_cold_start_path_returns_results(self):
        """
        Pipeline for a user with no Redis session and no CF embedding must still
        return recommendations via the popularity fallback.
        """
        candidates = [_make_item(f"pop_{i}") for i in range(20)]

        redis = MagicMock()
        redis.get_session_features.return_value = {
            "user_id": "brand_new_user",
            "recent_views": [],
            "session_length": 0,
            "session_age_seconds": 0.0,
            "has_cart_item": False,
        }
        redis.get_user_history.return_value = []

        cf = MagicMock()
        cf.has_user.return_value = False

        es = MagicMock()
        es.get_popular_items.return_value = candidates

        ranker = MagicMock()
        ranker.predict_scores.return_value = [float(i) for i in range(len(candidates))]

        fb = FeatureBuilder()

        pipeline = RecommendationPipeline(
            redis_store=redis,
            elastic_store=es,
            cf_model=cf,
            ranker=ranker,
            feature_builder=fb,
        )

        resp = pipeline.recommend("brand_new_user", n=5)
        assert len(resp.recommendations) > 0
        es.get_popular_items.assert_called_once()
