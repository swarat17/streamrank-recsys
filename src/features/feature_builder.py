"""Assembles the flat feature vector used by the XGBoost ranker."""

import time
import math
import logging

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Combines session context (from Redis) and item metadata into a ranker feature dict."""

    def build(
        self,
        session_features: dict,
        item_metadata: dict,
        context: dict | None = None,
    ) -> dict:
        """
        Build a flat feature dict for a single candidate item.

        Args:
            session_features: Output of RedisFeatureStore.get_session_features().
            item_metadata:    Dict with item fields (item_id, avg_rating, price, review_count, ...).
            context:          Optional request context (device, page, session_id, ...).

        Returns:
            Flat feature dict ready for XGBoost scoring.
        """
        context = context or {}
        candidate_id = item_metadata.get("item_id", "")
        recent_views: list[str] = session_features.get("recent_views", [])

        # Normalise hour to [0, 1]
        hour = time.localtime().tm_hour
        time_of_day = hour / 23.0

        device = context.get("device", "desktop")

        return {
            "user_id": session_features.get("user_id", ""),
            "candidate_item_id": candidate_id,
            "session_length": int(session_features.get("session_length", 0)),
            "session_age_seconds": float(session_features.get("session_age_seconds", 0.0)),
            "n_recent_views": len(recent_views),
            "candidate_in_recent_views": candidate_id in recent_views,
            "candidate_avg_rating": float(item_metadata.get("avg_rating", 0.0)),
            "candidate_price": _safe_float(item_metadata.get("price", 0.0)),
            "candidate_review_count": int(item_metadata.get("review_count", 0)),
            "time_of_day": time_of_day,
            "device_mobile": device == "mobile",
        }


def _safe_float(value) -> float:
    try:
        f = float(value)
        return 0.0 if math.isnan(f) else f
    except (TypeError, ValueError):
        return 0.0
