"""Redis-backed session feature store with TTL-based session expiry."""

import time
import logging
import os
from typing import Any

import redis

logger = logging.getLogger(__name__)

SESSION_TTL_SECONDS = 30 * 60  # 30 minutes
MAX_RECENT_VIEWS = 20


def _session_key(user_id: str) -> str:
    return f"user:{user_id}:session"


def _views_key(user_id: str) -> str:
    return f"user:{user_id}:session:views"


def _history_key(user_id: str) -> str:
    return f"user:{user_id}:history"


class RedisFeatureStore:
    def __init__(self, host: str | None = None, port: int | None = None, client: Any = None):
        """
        Args:
            host: Redis host (defaults to REDIS_HOST env var or localhost).
            port: Redis port (defaults to REDIS_PORT env var or 6379).
            client: Pre-built Redis client (used in tests with fakeredis).
        """
        if client is not None:
            self._r = client
        else:
            self._r = redis.Redis(
                host=host or os.getenv("REDIS_HOST", "localhost"),
                port=port or int(os.getenv("REDIS_PORT", 6379)),
                decode_responses=True,
            )

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def update_session(self, user_id: str, event: dict) -> None:
        """Update the user's session from a single clickstream event."""
        pipe = self._r.pipeline()

        sess_key = _session_key(user_id)
        views_key = _views_key(user_id)
        now = time.time()

        item_id = event.get("item_id", "")
        event_type = event.get("event_type", "")

        # Initialise session_start if not present
        if not self._r.hexists(sess_key, "session_start_timestamp"):
            pipe.hset(sess_key, "session_start_timestamp", now)

        # Increment event count and update last_seen
        pipe.hincrby(sess_key, "session_event_count", 1)
        pipe.hset(sess_key, "last_seen_timestamp", now)

        # Track cart items
        if event_type == "add_to_cart":
            pipe.hset(sess_key, "has_cart_item", "1")

        # Push item to recent views list and cap at MAX_RECENT_VIEWS
        if item_id:
            pipe.lpush(views_key, item_id)
            pipe.ltrim(views_key, 0, MAX_RECENT_VIEWS - 1)
            pipe.expire(views_key, SESSION_TTL_SECONDS)

        # Refresh TTL on session hash
        pipe.expire(sess_key, SESSION_TTL_SECONDS)
        pipe.execute()

    def get_session_features(self, user_id: str) -> dict:
        """Return current session state as a feature dict."""
        sess_key = _session_key(user_id)
        views_key = _views_key(user_id)

        data = self._r.hgetall(sess_key)
        recent_views = self._r.lrange(views_key, 0, -1)

        now = time.time()
        session_start = float(data.get("session_start_timestamp", now))
        last_seen = float(data.get("last_seen_timestamp", now))

        return {
            "user_id": user_id,
            "recent_views": recent_views,
            "session_length": int(data.get("session_event_count", 0)),
            "last_seen_timestamp": last_seen,
            "session_age_seconds": now - session_start,
            "has_cart_item": data.get("has_cart_item", "0") == "1",
        }

    # ------------------------------------------------------------------
    # User history (offline pre-loaded)
    # ------------------------------------------------------------------

    def set_user_history(self, user_id: str, item_ids: list[str]) -> None:
        """Bulk-load a user's historical interactions from the offline dataset."""
        if not item_ids:
            return
        hist_key = _history_key(user_id)
        pipe = self._r.pipeline()
        pipe.delete(hist_key)
        pipe.rpush(hist_key, *item_ids)
        pipe.execute()
        logger.debug("Loaded %d history items for user %s.", len(item_ids), user_id)

    def get_user_history(self, user_id: str) -> list[str]:
        """Return all historical item_ids for a user."""
        return self._r.lrange(_history_key(user_id), 0, -1)
