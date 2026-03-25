"""Unit tests for Phase 2 — Redis feature store and feature builder."""

import time
import pytest
import fakeredis

from src.features.redis_store import RedisFeatureStore, MAX_RECENT_VIEWS
from src.features.feature_builder import FeatureBuilder

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def redis_store():
    """RedisFeatureStore backed by fakeredis — no real Redis needed."""
    fake = fakeredis.FakeRedis(decode_responses=True)
    return RedisFeatureStore(client=fake)


@pytest.fixture
def view_event():
    def _make(user_id="user_1", item_id="item_A"):
        return {
            "event_id": "evt-001",
            "user_id": user_id,
            "item_id": item_id,
            "event_type": "view",
            "session_id": "sess-001",
            "timestamp": time.time(),
            "metadata": {"page": "product", "referrer": "google", "device": "desktop"},
        }

    return _make


# ---------------------------------------------------------------------------
# RedisFeatureStore tests
# ---------------------------------------------------------------------------


class TestRedisFeatureStore:
    def test_update_session_appends_recent_views(self, redis_store, view_event):
        redis_store.update_session("user_1", view_event(item_id="item_A"))
        redis_store.update_session("user_1", view_event(item_id="item_B"))
        redis_store.update_session("user_1", view_event(item_id="item_C"))

        features = redis_store.get_session_features("user_1")
        assert len(features["recent_views"]) == 3

    def test_recent_views_capped_at_20(self, redis_store, view_event):
        for i in range(25):
            redis_store.update_session("user_1", view_event(item_id=f"item_{i}"))

        features = redis_store.get_session_features("user_1")
        assert len(features["recent_views"]) == MAX_RECENT_VIEWS

    def test_session_ttl_set(self, redis_store, view_event):
        redis_store.update_session("user_1", view_event())
        # Check that the session key has a TTL > 0
        fake_r = redis_store._r
        ttl = fake_r.ttl("user:user_1:session")
        assert ttl > 0, f"Expected TTL > 0, got {ttl}"

    def test_get_session_features_returns_all_keys(self, redis_store, view_event):
        redis_store.update_session("user_1", view_event())
        features = redis_store.get_session_features("user_1")
        required_keys = {
            "user_id",
            "recent_views",
            "session_length",
            "last_seen_timestamp",
            "session_age_seconds",
            "has_cart_item",
        }
        assert required_keys.issubset(set(features.keys()))

    def test_user_history_set_and_get(self, redis_store):
        item_ids = [f"item_{i}" for i in range(10)]
        redis_store.set_user_history("user_1", item_ids)
        retrieved = redis_store.get_user_history("user_1")
        assert set(retrieved) == set(item_ids)


# ---------------------------------------------------------------------------
# FeatureBuilder tests
# ---------------------------------------------------------------------------

REQUIRED_FEATURE_KEYS = {
    "user_id",
    "candidate_item_id",
    "session_length",
    "session_age_seconds",
    "n_recent_views",
    "candidate_in_recent_views",
    "candidate_avg_rating",
    "candidate_price",
    "candidate_review_count",
    "time_of_day",
    "device_mobile",
}


class TestFeatureBuilder:
    def test_feature_builder_output_schema(self):
        builder = FeatureBuilder()
        session = {
            "user_id": "user_1",
            "recent_views": ["item_A", "item_B"],
            "session_length": 2,
            "session_age_seconds": 60.0,
            "has_cart_item": False,
        }
        item = {
            "item_id": "item_C",
            "avg_rating": 4.5,
            "price": 29.99,
            "review_count": 100,
        }
        features = builder.build(session, item)
        missing = REQUIRED_FEATURE_KEYS - set(features.keys())
        assert not missing, f"FeatureBuilder output missing keys: {missing}"

    def test_candidate_in_recent_views_flag(self, redis_store, view_event):
        # Load session with item_A in recent_views
        redis_store.update_session("user_1", view_event(item_id="item_A"))
        session = redis_store.get_session_features("user_1")

        builder = FeatureBuilder()
        item_seen = {
            "item_id": "item_A",
            "avg_rating": 4.0,
            "price": 10.0,
            "review_count": 50,
        }
        item_unseen = {
            "item_id": "item_Z",
            "avg_rating": 3.0,
            "price": 5.0,
            "review_count": 10,
        }

        assert builder.build(session, item_seen)["candidate_in_recent_views"] is True
        assert builder.build(session, item_unseen)["candidate_in_recent_views"] is False
