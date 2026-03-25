"""Two-stage retrieve → rank → return recommendation pipeline."""

import logging
import time
from typing import Any, Optional

from src.serving.schemas import RecommendationResponse, RecommendedItem
from src.monitoring.metrics import observe_request

logger = logging.getLogger(__name__)

MODEL_VERSION = "1.0.0"


class RecommendationPipeline:
    """
    Loaded once at FastAPI startup. All dependencies are injected for testability.

    Flow:
        Redis (session + history)
          → Elasticsearch kNN (or cold-start / popularity fallback)
          → FeatureBuilder (100 feature dicts)
          → XGBoost ranker (scores)
          → filter history → top-n
    """

    def __init__(
        self,
        redis_store,
        elastic_store,
        cf_model,
        ranker,
        feature_builder,
        model_version: str = MODEL_VERSION,
    ):
        self._redis = redis_store
        self._es = elastic_store
        self._cf = cf_model
        self._ranker = ranker
        self._fb = feature_builder
        self.model_version = model_version

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def recommend(
        self,
        user_id: str,
        n: int = 10,
        context: Optional[dict[str, Any]] = None,
    ) -> RecommendationResponse:
        context = context or {}
        t_total_start = time.perf_counter()

        # ── Step 1: Feature retrieval from Redis ──────────────────────────────
        session = self._redis.get_session_features(user_id)
        history = set(self._redis.get_user_history(user_id))

        # ── Step 2: Candidate retrieval from Elasticsearch ────────────────────
        t_ret_start = time.perf_counter()
        candidates = self._retrieve_candidates(user_id, session, history)
        retrieval_ms = (time.perf_counter() - t_ret_start) * 1000

        if not candidates:
            logger.warning("No candidates for user %s — returning empty list.", user_id)
            return RecommendationResponse(
                user_id=user_id,
                recommendations=[],
                retrieval_ms=retrieval_ms,
                ranking_ms=0.0,
                total_ms=(time.perf_counter() - t_total_start) * 1000,
                model_version=self.model_version,
            )

        # ── Step 3 + 4: Feature building + XGBoost ranking ───────────────────
        t_rank_start = time.perf_counter()
        feature_dicts = [self._fb.build(session, item, context) for item in candidates]
        scores = self._ranker.predict_scores(feature_dicts)
        ranking_ms = (time.perf_counter() - t_rank_start) * 1000

        # Sort by score, filter history, take top-n
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        results: list[RecommendedItem] = []
        for item, score in ranked:
            if item["item_id"] in history:
                continue
            results.append(
                RecommendedItem(
                    item_id=item["item_id"],
                    title=item.get("title", ""),
                    category=item.get("category", "Unknown"),
                    price=float(item.get("price") or 0.0),
                    avg_rating=float(item.get("avg_rating") or 0.0),
                    score=float(score),
                )
            )
            if len(results) >= n:
                break

        total_ms = (time.perf_counter() - t_total_start) * 1000
        logger.debug(
            "user=%s candidates=%d results=%d retrieval=%.1fms ranking=%.1fms total=%.1fms",
            user_id,
            len(candidates),
            len(results),
            retrieval_ms,
            ranking_ms,
            total_ms,
        )

        device = context.get("device", "desktop")
        status = "success" if results else "cold_start"
        observe_request(
            recommendations=results,
            retrieval_ms=retrieval_ms,
            ranking_ms=ranking_ms,
            total_ms=total_ms,
            n_candidates=len(candidates),
            status=status,
            device=device,
        )

        return RecommendationResponse(
            user_id=user_id,
            recommendations=results,
            retrieval_ms=retrieval_ms,
            ranking_ms=ranking_ms,
            total_ms=total_ms,
            model_version=self.model_version,
        )

    # ------------------------------------------------------------------
    # Retrieval strategy selection
    # ------------------------------------------------------------------

    def _retrieve_candidates(
        self, user_id: str, session: dict, history: set
    ) -> list[dict]:
        """Choose retrieval path based on what we know about the user."""

        # Known user with CF embedding → kNN
        if self._cf is not None and self._cf.has_user(user_id):
            try:
                user_emb = self._cf.get_user_embedding(user_id)
                return self._es.retrieve_candidates(user_emb, n=100)
            except Exception as e:
                logger.warning("CF retrieval failed for %s: %s", user_id, e)

        # Cold-start with session views → average item embeddings
        recent_views: list[str] = session.get("recent_views", [])
        if recent_views:
            try:
                return self._es.retrieve_by_items(recent_views, n=100)
            except Exception as e:
                logger.warning("Session-based retrieval failed: %s", e)

        # Completely new user → popularity fallback
        try:
            return self._es.get_popular_items(n=100)
        except Exception as e:
            logger.warning("Popularity fallback failed: %s", e)
            return []
