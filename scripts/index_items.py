"""Bulk-index all items and their embeddings into Elasticsearch."""

import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    from src.data.loader import load_items
    from src.models.embeddings import load_embeddings
    from src.retrieval.elastic_store import ElasticsearchItemStore
    from infra.elasticsearch.setup_index import create_index

    # Ensure index exists with correct mappings
    es_host = os.getenv("ES_HOST", "http://localhost:9200")
    create_index(es_host=es_host)

    logger.info("Loading items and embeddings...")
    items_df = load_items()
    embeddings = load_embeddings()

    logger.info("Items: %d  Embeddings: %d", len(items_df), len(embeddings))

    store = ElasticsearchItemStore(es_host=es_host, embeddings=embeddings)
    store.bulk_index(items_df, embeddings)
    logger.info("Done. Run: curl %s/items/_count", es_host)


if __name__ == "__main__":
    main()
