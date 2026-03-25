"""
End-to-end tests — require the full stack running (docker-compose up + FastAPI).

Run with:
    pytest tests/e2e/ -v -m e2e

Skip in CI unless all services are confirmed healthy.
"""

import time
import statistics

import httpx
import pytest

BASE_URL = "http://localhost:8000"


def _post_recommend(
    user_id: str, n: int = 5, context: dict | None = None
) -> httpx.Response:
    payload = {"user_id": user_id, "n_recommendations": n}
    if context:
        payload["context"] = context
    return httpx.post(f"{BASE_URL}/recommend", json=payload, timeout=10.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestFullPipeline:
    def test_health_endpoint_is_up(self):
        """FastAPI must be reachable and report healthy."""
        resp = httpx.get(f"{BASE_URL}/health", timeout=5.0)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["models_loaded"] is True

    def test_recommend_returns_valid_response(self):
        """
        A known user request must return a well-formed response with at least
        one recommendation and all required fields.
        """
        resp = _post_recommend("INTEGRATION_TEST_USER_001", n=5)
        assert resp.status_code == 200

        body = resp.json()
        assert "recommendations" in body
        assert "retrieval_ms" in body
        assert "ranking_ms" in body
        assert "total_ms" in body

        # Cold-start or known user — either way we expect results
        assert isinstance(body["recommendations"], list)
        assert len(body["recommendations"]) >= 1

        # Check first item has required fields
        item = body["recommendations"][0]
        for field in ("item_id", "title", "category", "price", "avg_rating", "score"):
            assert field in item, f"Missing field '{field}' in recommended item"

    @pytest.mark.e2e
    def test_latency_p99_under_500ms(self):
        """
        50 sequential requests — P99 wall-clock latency must stay under 500ms.
        (Generous ceiling for the e2e test environment; production target is 220ms P99.)
        """
        latencies = []
        for i in range(50):
            user_id = f"LATENCY_TEST_USER_{i % 10}"  # mix of known + cold-start
            t0 = time.perf_counter()
            resp = _post_recommend(user_id, n=10)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            assert resp.status_code == 200, f"Request {i} failed: {resp.text}"
            latencies.append(elapsed_ms)

        latencies.sort()
        p99 = latencies[int(len(latencies) * 0.99)]
        p50 = statistics.median(latencies)

        print(f"\nLatency — P50: {p50:.1f}ms  P99: {p99:.1f}ms")
        assert p99 < 500, f"P99 latency {p99:.1f}ms exceeds 500ms ceiling"
