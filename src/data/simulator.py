"""Clickstream event simulator based on real Amazon interaction data."""

import time
import uuid
import random
import logging
from typing import Generator

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

EVENT_TYPES = ["view", "add_to_cart", "purchase", "search"]
PAGES = ["home", "search", "product", "cart"]
REFERRERS = ["google", "direct", "email", "social", "internal"]
DEVICES = ["mobile", "desktop", "tablet"]

# Realistic transition probabilities for event sequences within a session
# Most sessions are: view → view → ... → (maybe add_to_cart) → (rarely purchase)
_SESSION_FLOW = {
    "start": {"view": 0.70, "search": 0.30},
    "view":  {"view": 0.60, "search": 0.15, "add_to_cart": 0.20, "purchase": 0.03, "end": 0.02},
    "search": {"view": 0.80, "search": 0.10, "end": 0.10},
    "add_to_cart": {"view": 0.50, "purchase": 0.20, "add_to_cart": 0.10, "end": 0.20},
    "purchase": {"end": 1.0},
}

_PAGE_FOR_EVENT = {
    "view": "product",
    "add_to_cart": "cart",
    "purchase": "cart",
    "search": "search",
}


def _next_event_type(current: str) -> str | None:
    transitions = _SESSION_FLOW.get(current, {"end": 1.0})
    choices, weights = zip(*transitions.items())
    chosen = random.choices(choices, weights=weights, k=1)[0]
    return None if chosen == "end" else chosen


class ClickstreamSimulator:
    """Generates realistic synthetic clickstream events from real interaction data."""

    def __init__(self, interactions_df: pd.DataFrame | None = None):
        """
        Args:
            interactions_df: The processed interactions DataFrame. If None, a small
                             synthetic fallback is used (useful for tests without disk data).
        """
        if interactions_df is not None and len(interactions_df) > 0:
            self._user_ids = interactions_df["user_id"].unique().tolist()
            self._item_ids = interactions_df["item_id"].unique().tolist()
            # Weight user sampling by activity so heavy users appear more often
            user_counts = interactions_df["user_id"].value_counts()
            self._user_weights = [user_counts.get(u, 1) for u in self._user_ids]
        else:
            # Fallback for tests
            self._user_ids = [f"user_{i}" for i in range(100)]
            self._item_ids = [f"item_{i}" for i in range(500)]
            self._user_weights = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_users(self, n: int) -> list[str]:
        return random.choices(self._user_ids, weights=self._user_weights, k=n)

    def _generate_session(self, user_id: str, base_ts: float) -> list[dict]:
        session_id = str(uuid.uuid4())
        device = random.choice(DEVICES)
        referrer = random.choice(REFERRERS)
        events: list[dict] = []

        current_event_type = _next_event_type("start")
        ts = base_ts
        max_events = random.randint(3, 15)

        while current_event_type is not None and len(events) < max_events:
            item_id = random.choice(self._item_ids)
            page = _PAGE_FOR_EVENT.get(current_event_type, "home")

            events.append({
                "event_id": str(uuid.uuid4()),
                "user_id": user_id,
                "item_id": item_id,
                "event_type": current_event_type,
                "session_id": session_id,
                "timestamp": ts,
                "metadata": {
                    "page": page,
                    "referrer": referrer,
                    "device": device,
                },
            })
            ts += random.uniform(5, 120)  # 5s–2min between events
            current_event_type = _next_event_type(current_event_type)

        return events

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_batch(self, n_users: int) -> list[dict]:
        """Generate one session per user. Returns a flat list of events."""
        base_ts = time.time()
        all_events: list[dict] = []
        for user_id in self._sample_users(n_users):
            all_events.extend(self._generate_session(user_id, base_ts))
        return all_events

    def stream(self, events_per_second: float = 10.0) -> Generator[dict, None, None]:
        """Infinite generator yielding events at approximately the given rate."""
        delay = 1.0 / events_per_second
        buffer: list[dict] = []
        while True:
            if not buffer:
                buffer = self.generate_batch(n_users=max(1, int(events_per_second)))
            event = buffer.pop(0)
            event["timestamp"] = time.time()
            yield event
            time.sleep(delay)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sim = ClickstreamSimulator()
    batch = sim.generate_batch(5)
    print(f"Generated {len(batch)} events for 5 users")
    print("Sample:", batch[0])
