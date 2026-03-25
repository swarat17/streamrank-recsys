"""PySpark Structured Streaming consumer for the clickstream-events Kafka topic."""

import logging
import os
import sys
import time

# Must be set before PySpark initialises the JVM.
# PYSPARK_PYTHON tells worker subprocesses to use this venv's interpreter
# instead of the bare 'python' shim (which on Windows opens the MS Store).
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
# Fix JAVA_HOME if the env var points to a non-existent path.
_java_home = os.environ.get("JAVA_HOME", "")
if not os.path.exists(os.path.join(_java_home, "bin", "java.exe")):
    _fallback = "C:/Program Files/Java/jre1.8.0_481"
    if os.path.exists(os.path.join(_fallback, "bin", "java.exe")):
        os.environ["JAVA_HOME"] = _fallback

from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql.functions import col, from_json  # noqa: E402
from pyspark.sql.types import (  # noqa: E402
    StructType,
    StructField,
    StringType,
    FloatType,
    MapType,
)

from src.features.redis_store import RedisFeatureStore  # noqa: E402

logger = logging.getLogger(__name__)

CHECKPOINT_LOCATION = "data/checkpoints/kafka_consumer"

EVENT_SCHEMA = StructType(
    [
        StructField("event_id", StringType()),
        StructField("user_id", StringType()),
        StructField("item_id", StringType()),
        StructField("event_type", StringType()),
        StructField("session_id", StringType()),
        StructField("timestamp", FloatType()),
        StructField("metadata", MapType(StringType(), StringType())),
    ]
)


def _process_batch(batch_df, batch_id: int, redis_store: RedisFeatureStore) -> None:
    """foreachBatch handler: write each event's session update to Redis."""
    t0 = time.time()
    rows = batch_df.collect()
    if not rows:
        return

    unique_users = set()
    for row in rows:
        event = {
            "event_id": row.event_id,
            "user_id": row.user_id,
            "item_id": row.item_id,
            "event_type": row.event_type,
            "session_id": row.session_id,
            "timestamp": row.timestamp,
            "metadata": dict(row.metadata) if row.metadata else {},
        }
        redis_store.update_session(row.user_id, event)
        unique_users.add(row.user_id)

    elapsed_ms = (time.time() - t0) * 1000
    logger.info(
        "Batch %d: %d events, %d unique users, %.1fms",
        batch_id,
        len(rows),
        len(unique_users),
        elapsed_ms,
    )


def start_consumer(
    bootstrap_servers: str | None = None,
    redis_host: str | None = None,
    redis_port: int | None = None,
) -> None:
    servers = bootstrap_servers or os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
    )
    redis_store = RedisFeatureStore(host=redis_host, port=redis_port)

    spark = (
        SparkSession.builder.appName("recsys-clickstream-consumer")
        .config(
            "spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1"
        )
        .config("spark.sql.streaming.checkpointLocation", CHECKPOINT_LOCATION)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    raw_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", servers)
        .option("subscribe", "clickstream-events")
        .option("startingOffsets", "latest")
        .load()
    )

    parsed = raw_stream.select(
        from_json(col("value").cast("string"), EVENT_SCHEMA).alias("data")
    ).select("data.*")

    query = (
        parsed.writeStream.foreachBatch(
            lambda df, bid: _process_batch(df, bid, redis_store)
        )
        .trigger(processingTime="5 seconds")
        .start()
    )

    logger.info("Spark consumer started. Waiting for events...")
    query.awaitTermination()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start_consumer()
