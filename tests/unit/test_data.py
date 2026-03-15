"""Unit tests for Phase 1 — data loader and clickstream simulator."""

import pytest
import pandas as pd

from src.data.simulator import ClickstreamSimulator, EVENT_TYPES

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_interactions_df():
    """A minimal interactions DataFrame that passes all filters."""
    rows = []
    for user_idx in range(20):          # 20 users
        for item_idx in range(15):      # each interacts with 15 items
            rows.append({
                "user_id": f"user_{user_idx}",
                "item_id": f"item_{item_idx}",
                "rating": 4.0,
                "timestamp": 1_700_000_000.0 + user_idx * 100 + item_idx,
                "verified_purchase": True,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def simulator(small_interactions_df):
    return ClickstreamSimulator(interactions_df=small_interactions_df)


# ---------------------------------------------------------------------------
# Loader tests (use synthetic DataFrames — no disk required)
# ---------------------------------------------------------------------------

class TestInteractionsSchema:
    REQUIRED_COLUMNS = {"user_id", "item_id", "rating", "timestamp", "verified_purchase"}

    def test_interactions_has_required_columns(self, small_interactions_df):
        assert self.REQUIRED_COLUMNS.issubset(set(small_interactions_df.columns))

    def test_min_interactions_per_user_filter(self, small_interactions_df):
        """Verify our fixture satisfies the >=5 interactions-per-user rule."""
        user_counts = small_interactions_df.groupby("user_id").size()
        assert (user_counts >= 5).all(), "Some users have fewer than 5 interactions"

    def test_min_reviews_per_item_filter(self, small_interactions_df):
        """Verify our fixture satisfies the >=10 reviews-per-item rule."""
        item_counts = small_interactions_df.groupby("item_id").size()
        assert (item_counts >= 10).all(), "Some items have fewer than 10 reviews"


# ---------------------------------------------------------------------------
# Simulator tests
# ---------------------------------------------------------------------------

REQUIRED_TOP_LEVEL_KEYS = {"event_id", "user_id", "item_id", "event_type", "session_id", "timestamp", "metadata"}
REQUIRED_METADATA_KEYS = {"page", "referrer", "device"}


class TestSimulator:
    def test_simulator_event_schema(self, simulator):
        events = simulator.generate_batch(n_users=5)
        assert len(events) > 0
        for event in events:
            missing = REQUIRED_TOP_LEVEL_KEYS - set(event.keys())
            assert not missing, f"Event missing keys: {missing}"
            meta_missing = REQUIRED_METADATA_KEYS - set(event["metadata"].keys())
            assert not meta_missing, f"Event metadata missing keys: {meta_missing}"

    def test_simulator_event_types_valid(self, simulator):
        events = simulator.generate_batch(n_users=20)
        for event in events:
            assert event["event_type"] in EVENT_TYPES, (
                f"Invalid event_type: {event['event_type']}"
            )

    def test_simulator_purchase_rate_reasonable(self, simulator):
        """Purchase rate should be between 1% and 10% in a large batch."""
        events = simulator.generate_batch(n_users=200)
        # Keep generating until we have >= 1000 events
        while len(events) < 1000:
            events += simulator.generate_batch(n_users=50)
        events = events[:1000]

        purchase_count = sum(1 for e in events if e["event_type"] == "purchase")
        rate = purchase_count / len(events)
        assert 0.01 <= rate <= 0.10, (
            f"Purchase rate {rate:.2%} outside expected [1%, 10%] range"
        )

    def test_simulator_generate_batch_returns_events(self, simulator):
        """generate_batch(n) returns at least n events (each user gets ≥1 event)."""
        events = simulator.generate_batch(n_users=10)
        assert len(events) >= 10

    def test_simulator_metadata_device_valid(self, simulator):
        from src.data.simulator import DEVICES
        events = simulator.generate_batch(n_users=10)
        for event in events:
            assert event["metadata"]["device"] in DEVICES

    def test_simulator_session_id_groups_user_events(self, simulator):
        """All events in a single user session share the same session_id."""
        events = simulator.generate_batch(n_users=5)
        # Group by session_id and verify all events in each session have same user_id
        from collections import defaultdict
        sessions = defaultdict(set)
        for e in events:
            sessions[e["session_id"]].add(e["user_id"])
        for sid, users in sessions.items():
            assert len(users) == 1, f"Session {sid} has events from multiple users: {users}"
