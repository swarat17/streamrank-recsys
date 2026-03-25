"""Elasticsearch item store: bulk indexing and kNN candidate retrieval."""

import logging
import os
import time
from typing import Any

import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from src.models.embeddings import (
    load_embeddings,
    load_projection_matrix,
    project_user_embedding,
)

logger = logging.getLogger(__name__)

INDEX_NAME = "items"
_ITEM_FIELDS = ["item_id", "title", "category", "price", "avg_rating", "review_count"]


class ElasticsearchItemStore:
    def __init__(
        self,
        es_host: str | None = None,
        embeddings: dict[str, np.ndarray] | None = None,
        projection_matrix: np.ndarray | None = None,
        client: Any = None,
    ):
        """
        Args:
            es_host:           ES URL (defaults to ES_HOST env var or localhost:9200).
            embeddings:        Pre-loaded item embeddings dict (loaded from disk if None).
            projection_matrix: Pre-loaded 192×128 projection matrix (loaded from disk if None).
            client:            Pre-built Elasticsearch client (used in tests).
        """
        self._es = client or Elasticsearch(
            es_host or os.getenv("ES_HOST", "http://localhost:9200")
        )
        self._embeddings = embeddings
        self._proj = projection_matrix

    def _get_embeddings(self) -> dict[str, np.ndarray]:
        if self._embeddings is None:
            self._embeddings = load_embeddings()
        return self._embeddings

    def _get_projection(self) -> np.ndarray:
        if self._proj is None:
            self._proj = load_projection_matrix()
        return self._proj

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def bulk_index(
        self, items_df: pd.DataFrame, embeddings: dict[str, np.ndarray]
    ) -> None:
        """Bulk-index all items with their embeddings into Elasticsearch."""
        t0 = time.time()

        def _actions():
            for _, row in items_df.iterrows():
                item_id = row["item_id"]
                emb = embeddings.get(item_id)
                if emb is None:
                    continue

                def _safe_float(v):
                    try:
                        f = float(v)
                        return 0.0 if (f != f) else f  # NaN check: NaN != NaN
                    except (TypeError, ValueError):
                        return 0.0

                yield {
                    "_index": INDEX_NAME,
                    "_id": item_id,
                    "_source": {
                        "item_id": item_id,
                        "title": str(row.get("title", "")),
                        "category": str(row.get("category", "Unknown")),
                        "price": _safe_float(row.get("price")),
                        "avg_rating": _safe_float(row.get("avg_rating")),
                        "review_count": int(row.get("review_count") or 0),
                        "embedding": emb.tolist(),
                    },
                }

        success, errors = bulk(self._es, _actions(), raise_on_error=False)
        elapsed = time.time() - t0

        if errors:
            logger.warning("Bulk index: %d errors", len(errors))
        logger.info(
            "Indexed %d items in %.1fs (%.0f items/sec).",
            success,
            elapsed,
            success / elapsed if elapsed > 0 else 0,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_candidates(
        self, user_embedding: np.ndarray, n: int = 100
    ) -> list[dict]:
        """
        kNN search using a user's CF embedding (128d → projected to 192d).

        Returns top-n items with metadata + score.
        """
        proj = self._get_projection()
        query_vec = project_user_embedding(user_embedding, proj)
        return self._knn_query(query_vec.tolist(), n)

    def retrieve_by_items(self, item_ids: list[str], n: int = 100) -> list[dict]:
        """
        Cold-start retrieval: average embeddings of given item_ids, then kNN search.
        """
        embeddings = self._get_embeddings()
        vecs = [embeddings[iid] for iid in item_ids if iid in embeddings]
        if not vecs:
            logger.warning(
                "retrieve_by_items: none of %d items found in embeddings.",
                len(item_ids),
            )
            return []

        avg_vec = np.mean(vecs, axis=0).astype(np.float32)
        norm = np.linalg.norm(avg_vec)
        if norm > 0:
            avg_vec = avg_vec / norm

        return self._knn_query(avg_vec.tolist(), n)

    def search_by_text(self, query: str, n: int = 20) -> list[dict]:
        """Full-text search on the title field — fallback for 'search' events."""
        resp = self._es.search(
            index=INDEX_NAME,
            query={"match": {"title": {"query": query, "fuzziness": "AUTO"}}},
            size=n,
            source=_ITEM_FIELDS,
        )
        return [self._hit_to_dict(h) for h in resp["hits"]["hits"]]

    def _knn_query(self, query_vector: list[float], n: int) -> list[dict]:
        resp = self._es.search(
            index=INDEX_NAME,
            knn={
                "field": "embedding",
                "query_vector": query_vector,
                "k": n,
                "num_candidates": max(n * 5, 500),
            },
            size=n,
            source=_ITEM_FIELDS,
        )
        return [self._hit_to_dict(h) for h in resp["hits"]["hits"]]

    @staticmethod
    def _hit_to_dict(hit: dict) -> dict:
        src = hit["_source"]
        return {
            "item_id": src.get("item_id", hit["_id"]),
            "title": src.get("title", ""),
            "category": src.get("category", "Unknown"),
            "price": src.get("price", 0.0),
            "avg_rating": src.get("avg_rating", 0.0),
            "review_count": src.get("review_count", 0),
            "_score": hit.get("_score", 0.0),
        }

    def get_popular_items(self, n: int = 50) -> list[dict]:
        """Return top-n items by review_count — used as cold-start fallback in pipeline."""
        resp = self._es.search(
            index=INDEX_NAME,
            query={"match_all": {}},
            sort=[{"review_count": {"order": "desc"}}],
            size=n,
            source=_ITEM_FIELDS,
        )
        return [self._hit_to_dict(h) for h in resp["hits"]["hits"]]
