"""Prometheus metrics for the recommendation pipeline."""

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY,
)

# ── Latency histogram per pipeline stage ─────────────────────────────────────
LATENCY_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]

recommendation_latency_seconds = Histogram(
    "recommendation_latency_seconds",
    "End-to-end and per-stage recommendation latency",
    labelnames=["stage"],   # retrieval | ranking | total
    buckets=LATENCY_BUCKETS,
)

# ── Request counter ───────────────────────────────────────────────────────────
recommendation_requests_total = Counter(
    "recommendation_requests_total",
    "Total recommendation requests",
    labelnames=["status", "device"],  # success | cold_start | error, mobile | desktop | tablet
)

# ── Candidate count histogram ─────────────────────────────────────────────────
retrieved_candidates_total = Histogram(
    "retrieved_candidates_total",
    "Number of candidates returned by Elasticsearch per request",
    buckets=[10, 25, 50, 75, 100, 150, 200],
)

# ── Diversity score gauge ─────────────────────────────────────────────────────
recommendation_diversity_score = Gauge(
    "recommendation_diversity_score",
    "Fraction of recommended items from distinct categories (unique_cats / n_recs)",
)

# ── Model version info (static, set at startup) ───────────────────────────────
model_version_info = Info(
    "recsys_model",
    "Model version information",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_diversity_score(recommendations: list) -> float:
    """
    Compute recommendation diversity as fraction of unique categories.

    Args:
        recommendations: List of RecommendedItem objects or dicts with a 'category' field.

    Returns:
        float in [0, 1]. Higher = more diverse.
        Returns 0 if recommendations is empty.
    """
    if not recommendations:
        return 0.0

    def _cat(item):
        if hasattr(item, "category"):
            return item.category
        return item.get("category", "unknown")

    unique_cats = len({_cat(r) for r in recommendations})
    return unique_cats / len(recommendations)


def observe_request(
    recommendations: list,
    retrieval_ms: float,
    ranking_ms: float,
    total_ms: float,
    n_candidates: int,
    status: str,
    device: str,
) -> None:
    """Record all metrics for a single completed request."""
    recommendation_latency_seconds.labels(stage="retrieval").observe(retrieval_ms / 1000)
    recommendation_latency_seconds.labels(stage="ranking").observe(ranking_ms / 1000)
    recommendation_latency_seconds.labels(stage="total").observe(total_ms / 1000)

    recommendation_requests_total.labels(status=status, device=device).inc()
    retrieved_candidates_total.observe(n_candidates)

    diversity = compute_diversity_score(recommendations)
    recommendation_diversity_score.set(diversity)


def set_model_info(cf_version: str = "unknown", ranker_version: str = "unknown") -> None:
    model_version_info.info({
        "cf_model_version": cf_version,
        "ranker_model_version": ranker_version,
    })
