"""Create required Kafka topics on startup."""

import logging
import os

from confluent_kafka.admin import AdminClient, NewTopic

logger = logging.getLogger(__name__)

TOPICS = [
    NewTopic("clickstream-events", num_partitions=6, replication_factor=1),
    NewTopic("recommendation-requests", num_partitions=2, replication_factor=1),
]


def create_topics(bootstrap_servers: str | None = None) -> None:
    servers = bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    admin = AdminClient({"bootstrap.servers": servers})

    existing = set(admin.list_topics(timeout=10).topics.keys())
    to_create = [t for t in TOPICS if t.topic not in existing]

    if not to_create:
        logger.info("All Kafka topics already exist.")
        return

    futures = admin.create_topics(to_create)
    for topic, future in futures.items():
        try:
            future.result()
            logger.info("Created topic: %s", topic)
        except Exception as e:
            logger.error("Failed to create topic %s: %s", topic, e)
            raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_topics()
    print("Kafka topics ready.")
