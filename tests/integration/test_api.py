"""Integration tests for Phase 5 — FastAPI endpoints.

Requires all services + trained models:
    docker-compose up redis elasticsearch -d
    python scripts/train.py
    python scripts/index_items.py

Run with:
    pytest tests/integration/test_api.py -v -m integration
"""

import time
import pytest


@pytest.mark.integration
class TestAPIEndpoints:
    BASE_URL = "http://localhost:8000"

    def test_health_endpoint_returns_200(self):
        import httpx
        resp = httpx.get(f"{self.BASE_URL}/health", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_recommend_endpoint_returns_items(self):
        import httpx
        payload = {"user_id": "integration_test_user", "n_recommendations": 5}
        resp = httpx.post(f"{self.BASE_URL}/recommend", json=payload, timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["recommendations"]) >= 1

    def test_latency_under_200ms(self):
        import httpx
        payload = {"user_id": "latency_test_user", "n_recommendations": 10}
        t0 = time.perf_counter()
        resp = httpx.post(f"{self.BASE_URL}/recommend", json=payload, timeout=10)
        wall_ms = (time.perf_counter() - t0) * 1000
        assert resp.status_code == 200
        assert wall_ms < 200, f"Wall-clock latency {wall_ms:.1f}ms exceeded 200ms"
