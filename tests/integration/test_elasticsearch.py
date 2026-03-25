"""Integration tests for Phase 4 — Elasticsearch index and retrieval.

Requires Elasticsearch running:
    docker-compose up elasticsearch -d

Run with:
    pytest tests/integration/test_elasticsearch.py -v -m integration
"""

import numpy as np
import pytest

TOTAL_DIM = 192
CF_DIM = 128


@pytest.mark.integration
class TestElasticsearchIntegration:
    def test_index_and_retrieve_roundtrip(self):
        """Index 10 mock items, run kNN — at least 1 of the 10 appears in results."""
        import pandas as pd
        from src.retrieval.elastic_store import ElasticsearchItemStore
        from infra.elasticsearch.setup_index import create_index

        create_index(recreate=True)

        # Build 10 synthetic items + embeddings
        item_ids = [f"int_item_{i}" for i in range(10)]
        items_df = pd.DataFrame(
            [
                {
                    "item_id": iid,
                    "title": f"Integration Test Item {i}",
                    "category": "Test",
                    "price": float(10 + i),
                    "avg_rating": 4.0,
                    "review_count": 50,
                    "description_snippet": "",
                }
                for i, iid in enumerate(item_ids)
            ]
        )
        embeddings = {
            iid: np.random.rand(TOTAL_DIM).astype(np.float32) for iid in item_ids
        }

        proj = np.zeros((TOTAL_DIM, CF_DIM), dtype=np.float32)
        proj[:CF_DIM, :CF_DIM] = np.eye(CF_DIM, dtype=np.float32)

        store = ElasticsearchItemStore(embeddings=embeddings, projection_matrix=proj)
        store.bulk_index(items_df, embeddings)

        # Give ES a moment to make the index searchable
        import time

        time.sleep(1)

        user_emb = np.random.rand(CF_DIM).astype(np.float32)
        results = store.retrieve_candidates(user_emb, n=10)

        result_ids = {r["item_id"] for r in results}
        overlap = result_ids & set(item_ids)
        assert (
            len(overlap) >= 1
        ), f"Expected at least 1 indexed item in results, got: {result_ids}"
