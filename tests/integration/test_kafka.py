"""Integration tests for Phase 2 — Kafka producer and Redis session update.

These tests require docker-compose services to be running:
    docker-compose up kafka zookeeper redis

Run with:
    pytest tests/integration/test_kafka.py -v -m integration
"""

import time
import pytest

from src.features.redis_store import RedisFeatureStore


@pytest.mark.integration
class TestKafkaProducerIntegration:
    def test_producer_sends_without_error(self):
        """send_event() on a live Kafka broker completes without raising."""
        from src.kafka.producer import ClickstreamProducer

        producer = ClickstreamProducer()
        event = {
            "event_id": "test-evt-001",
            "user_id": "integration_user",
            "item_id": "integration_item",
            "event_type": "view",
            "session_id": "integration-sess",
            "timestamp": time.time(),
            "metadata": {"page": "product", "referrer": "test", "device": "desktop"},
        }
        producer.send_event(event)  # must not raise


@pytest.mark.integration
class TestRedisSessionIntegration:
    def test_redis_session_updated_after_event(self):
        """update_session() on a live Redis instance → get_session_features() returns data."""
        store = RedisFeatureStore()  # connects to localhost:6379
        user_id = f"integration_user_{int(time.time())}"
        event = {
            "event_id": "test-evt-002",
            "user_id": user_id,
            "item_id": "item_X",
            "event_type": "view",
            "session_id": "sess-001",
            "timestamp": time.time(),
            "metadata": {"page": "product", "referrer": "direct", "device": "mobile"},
        }
        store.update_session(user_id, event)
        features = store.get_session_features(user_id)

        assert features["user_id"] == user_id
        assert len(features["recent_views"]) > 0
        assert features["session_length"] > 0
