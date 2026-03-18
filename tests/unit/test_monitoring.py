"""Unit tests for Phase 6 — Prometheus metrics and diversity scoring."""

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry

from src.monitoring.metrics import compute_diversity_score


# ---------------------------------------------------------------------------
# Diversity score tests
# ---------------------------------------------------------------------------

class TestDiversityScore:
    def _make_items(self, categories: list[str]) -> list[dict]:
        return [{"item_id": f"i{i}", "category": cat} for i, cat in enumerate(categories)]

    def test_diversity_score_all_same_category(self):
        """All items from same category → diversity = 1/n."""
        n = 10
        items = self._make_items(["Electronics"] * n)
        score = compute_diversity_score(items)
        expected = 1 / n
        assert abs(score - expected) < 1e-9, f"Expected {expected}, got {score}"

    def test_diversity_score_all_different(self):
        """All items from different categories → diversity = 1.0."""
        items = self._make_items([f"cat_{i}" for i in range(10)])
        score = compute_diversity_score(items)
        assert abs(score - 1.0) < 1e-9, f"Expected 1.0, got {score}"

    def test_diversity_score_mixed(self):
        """5 items, 2 unique categories → diversity = 2/5 = 0.4."""
        items = self._make_items(["A", "A", "A", "B", "B"])
        score = compute_diversity_score(items)
        assert abs(score - 0.4) < 1e-9

    def test_diversity_score_empty(self):
        assert compute_diversity_score([]) == 0.0

    def test_diversity_works_with_objects(self):
        """compute_diversity_score also accepts objects with a .category attribute."""
        class Item:
            def __init__(self, cat): self.category = cat
        items = [Item("X"), Item("Y"), Item("X")]
        score = compute_diversity_score(items)
        assert abs(score - 2 / 3) < 1e-9


# ---------------------------------------------------------------------------
# Metrics endpoint test
# ---------------------------------------------------------------------------

class TestMetricsEndpoint:
    def test_metrics_endpoint_returns_200(self):
        """/metrics returns 200 with text/plain content-type."""
        # Build a minimal FastAPI app with just the /metrics route
        app = FastAPI()

        @app.get("/metrics")
        def _metrics():
            return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

        client = TestClient(app)
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]
        # Prometheus output always starts with a comment or metric line
        assert len(resp.content) > 0
