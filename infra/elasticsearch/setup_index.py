"""Create the Elasticsearch 'items' index with correct mappings for ANN search."""

import logging
import os

from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)

INDEX_NAME = "items"

MAPPINGS = {
    "properties": {
        "item_id":      {"type": "keyword"},
        "title":        {"type": "text", "analyzer": "standard"},
        "category":     {"type": "keyword"},
        "price":        {"type": "float"},
        "avg_rating":   {"type": "float"},
        "review_count": {"type": "integer"},
        "embedding": {
            "type": "dense_vector",
            "dims": 192,
            "index": True,
            "similarity": "cosine",
        },
    }
}

SETTINGS = {
    "number_of_shards": 1,
    "number_of_replicas": 0,
}


def create_index(es_host: str | None = None, recreate: bool = False) -> None:
    host = es_host or os.getenv("ES_HOST", "http://localhost:9200")
    es = Elasticsearch(host)

    if es.indices.exists(index=INDEX_NAME):
        if recreate:
            es.indices.delete(index=INDEX_NAME)
            logger.info("Deleted existing index '%s'.", INDEX_NAME)
        else:
            logger.info("Index '%s' already exists — skipping creation.", INDEX_NAME)
            return

    es.indices.create(
        index=INDEX_NAME,
        mappings=MAPPINGS,
        settings=SETTINGS,
    )
    logger.info("Created index '%s' with 192-dim dense_vector (HNSW cosine).", INDEX_NAME)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_index()
    print("Elasticsearch index ready.")
